import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional


@dataclass
class TrajectoryStep:
    """
    Records a single interaction step in the harness execution trajectory.
    """
    step: int
    model_output: str
    action_type: str  # "tool_call", "final", "parse_error"
    tool_name: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None
    tool_result: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "step": self.step,
            "model_output": self.model_output,
            "action_type": self.action_type,
        }
        if self.tool_name is not None:
            data["tool_name"] = self.tool_name
        if self.arguments is not None:
            data["arguments"] = self.arguments
        if self.tool_result is not None:
            data["tool_result"] = self.tool_result
        if self.error is not None:
            data["error"] = self.error
        return data


@dataclass
class Trajectory:
    """
    Complete interaction trajectory for an agent solving a task.
    """
    task_id: str
    prompt: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    steps: List[TrajectoryStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    termination_reason: str = "running"  # "final_answer", "max_steps_exceeded", "error"
    trajectory_id: Optional[str] = None
    teacher_model: Optional[str] = None
    success: Optional[bool] = None

    def add_step(self, step: TrajectoryStep) -> None:
        self.steps.append(step)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id or f"traj_{self.task_id}",
            "task_id": self.task_id,
            "task": self.prompt,
            "prompt": self.prompt,
            "teacher_model": self.teacher_model or "unknown",
            "messages": self.messages,
            "steps": [step.to_dict() for step in self.steps],
            "final_answer": self.final_answer,
            "termination_reason": self.termination_reason,
            "success": self.success if self.success is not None else (self.termination_reason == "final_answer"),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    def print_trajectory(self) -> None:
        """
        Pretty-print step details to console.
        """
        print(f"\n==================== TASK: {self.task_id} ====================")
        print(f"PROMPT: {self.prompt}\n")
        for s in self.steps:
            print(f"--- STEP {s.step} ---")
            print(f"MODEL OUTPUT:\n{s.model_output}")
            if s.action_type == "tool_call":
                print(f"ACTION: tool_call -> {s.tool_name}")
                print(f"ARGUMENTS: {json.dumps(s.arguments)}")
                print(f"OBSERVATION:\n{s.tool_result}")
            elif s.action_type == "parse_error":
                print(f"PARSE ERROR:\n{s.error}")
            elif s.action_type == "final":
                print(f"FINAL ANSWER:\n{s.model_output}")
            print()
        print(f"TERMINATION: {self.termination_reason}")
        if self.final_answer:
            print(f"FINAL ANSWER RESULT: {self.final_answer}")
        print("============================================================\n")
