from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from src.trajectory import Trajectory
from src.harness import AgentHarness
from src.tools import ToolRegistry
from src.model import ModelInterface
from src.rl.reward import calculate_reward, RewardResult


@dataclass
class RolloutCandidate:
    """
    Single rollout candidate inside a GRPO group.
    """
    trajectory: Trajectory
    reward_result: RewardResult
    advantage: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trajectory": self.trajectory.to_dict(),
            "reward": self.reward_result.to_dict(),
            "advantage": round(self.advantage, 4),
        }


@dataclass
class RolloutResult:
    """
    Group of G rollouts for a single task prompt.
    """
    task_id: str
    category: str
    prompt: str
    candidates: List[RolloutCandidate]
    mean_reward: float
    std_reward: float
    reward_variance: float

    def compute_group_advantages(self) -> None:
        rewards = [c.reward_result.total_reward for c in self.candidates]
        self.mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
        
        var = sum((r - self.mean_reward) ** 2 for r in rewards) / len(rewards) if len(rewards) > 1 else 0.0
        self.std_reward = var ** 0.5
        self.reward_variance = var

        eps = 1e-8
        for c in self.candidates:
            if self.std_reward > 1e-5:
                c.advantage = (c.reward_result.total_reward - self.mean_reward) / (self.std_reward + eps)
            else:
                c.advantage = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "category": self.category,
            "prompt": self.prompt,
            "mean_reward": round(self.mean_reward, 4),
            "std_reward": round(self.std_reward, 4),
            "reward_variance": round(self.reward_variance, 4),
            "candidates": [c.to_dict() for c in self.candidates],
        }


class RolloutGenerator:
    """
    Generates G rollout trajectories per prompt using AgentHarness.
    """

    def __init__(
        self,
        model: ModelInterface,
        tool_registry: ToolRegistry,
        max_steps: int = 10,
    ):
        self.model = model
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.harness = AgentHarness(model=model, tool_registry=tool_registry, max_steps=max_steps)

    def generate_group(
        self,
        task_metadata: Dict[str, Any],
        group_size: int = 4,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> RolloutResult:
        task_id = task_metadata.get("id", task_metadata.get("task_id", "task_001"))
        category = task_metadata.get("category", "general")
        prompt = task_metadata.get("prompt", "")
        
        # If task prompt is embedded in messages
        if not prompt and "messages" in task_metadata:
            msgs = task_metadata["messages"]
            if msgs and msgs[0].get("role") == "user":
                prompt = msgs[0].get("content", "")

        # Set stochastic sampling parameters for candidate generation
        orig_temp = getattr(self.model, "temperature", 0.0)
        orig_top_p = getattr(self.model, "top_p", None)
        if hasattr(self.model, "temperature"):
            self.model.temperature = temperature
        if hasattr(self.model, "top_p"):
            self.model.top_p = top_p

        candidates: List[RolloutCandidate] = []

        try:
            for g_idx in range(group_size):
                # Run environment rollout via harness
                trajectory = self.harness.run(prompt=prompt, task_id=f"{task_id}_g{g_idx}")
                
                # Compute trajectory reward
                r_res = calculate_reward(trajectory, task_metadata)
                candidates.append(RolloutCandidate(trajectory=trajectory, reward_result=r_res))
        finally:
            # Restore model generation parameters
            if hasattr(self.model, "temperature"):
                self.model.temperature = orig_temp
            if hasattr(self.model, "top_p"):
                self.model.top_p = orig_top_p

        result = RolloutResult(
            task_id=task_id,
            category=category,
            prompt=prompt,
            candidates=candidates,
            mean_reward=0.0,
            std_reward=0.0,
            reward_variance=0.0,
        )
        result.compute_group_advantages()
        return result
