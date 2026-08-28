from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path


@dataclass
class GRPOConfig:
    """
    Configuration hyperparameters for GRPO-1 Reinforcement Learning.
    """
    starting_model_path: str = "outputs/qwen3-1.7b-tool-agent-sft-v4-500"
    reference_model_path: str = "outputs/qwen3-1.7b-tool-agent-sft-v4-500"
    train_dataset_path: str = "data/sft/train_v4_500.jsonl"
    eval_benchmark_path: str = "tasks/eval_v1.json"
    output_dir: str = "outputs/grpo1"
    eval_output_dir: str = "evaluation/grpo1"
    
    group_size: int = 4
    learning_rate: float = 5e-6
    temperature: float = 0.8
    top_p: float = 0.95
    beta: float = 0.04
    clip_eps: float = 0.2
    max_env_steps: int = 10
    max_prompt_length: int = 512
    max_completion_length: int = 512
    
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    epochs: int = 3
    pilot_prompts_count: int = 20
    
    save_steps: int = 25
    eval_steps: int = 25
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "starting_model_path": self.starting_model_path,
            "reference_model_path": self.reference_model_path,
            "train_dataset_path": self.train_dataset_path,
            "eval_benchmark_path": self.eval_benchmark_path,
            "output_dir": self.output_dir,
            "eval_output_dir": self.eval_output_dir,
            "group_size": self.group_size,
            "learning_rate": self.learning_rate,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "beta": self.beta,
            "clip_eps": self.clip_eps,
            "max_env_steps": self.max_env_steps,
            "max_prompt_length": self.max_prompt_length,
            "max_completion_length": self.max_completion_length,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "epochs": self.epochs,
            "pilot_prompts_count": self.pilot_prompts_count,
            "save_steps": self.save_steps,
            "eval_steps": self.eval_steps,
            "seed": self.seed,
        }
