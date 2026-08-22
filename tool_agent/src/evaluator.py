import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from src.trajectory import Trajectory
from src.harness import AgentHarness


def normalize_answer(answer: str) -> str:
    """
    Basic string normalization for answer comparison.
    """
    if answer is None:
        return ""
    return str(answer).strip().lower().rstrip(".")


@dataclass
class TaskResult:
    """
    Evaluation metrics for a single task trajectory.
    """
    task_id: str
    category: str
    prompt: str
    expected_answer: Optional[str]
    requires_tool: bool
    allowed_tools: List[str]
    actual_answer: Optional[str]
    task_success: bool
    total_steps: int
    num_tool_calls: int
    tool_call_validity: float
    correct_tool_selected: bool
    unnecessary_tool_used: bool
    termination_reason: str
    trajectory: Trajectory

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "category": self.category,
            "prompt": self.prompt,
            "expected_answer": self.expected_answer,
            "requires_tool": self.requires_tool,
            "allowed_tools": self.allowed_tools,
            "actual_answer": self.actual_answer,
            "task_success": self.task_success,
            "total_steps": self.total_steps,
            "num_tool_calls": self.num_tool_calls,
            "tool_call_validity": self.tool_call_validity,
            "correct_tool_selected": self.correct_tool_selected,
            "unnecessary_tool_used": self.unnecessary_tool_used,
            "termination_reason": self.termination_reason,
        }


@dataclass
class EvaluationSummary:
    """
    Aggregated metrics across a batch of task evaluations.
    """
    total_tasks: int
    successful_tasks: int
    accuracy: float
    tool_required_accuracy: float
    no_tool_accuracy: float
    tool_selection_accuracy: float
    unnecessary_tool_use_rate: float
    multi_step_accuracy: float
    avg_steps_per_task: float
    total_tool_calls: int
    valid_tool_call_rate: float
    category_accuracies: Dict[str, float]
    termination_reasons: Dict[str, int]
    task_results: List[TaskResult] = field(default_factory=list)

    def print_summary(self) -> None:
        print("\n==================== EVALUATION SUMMARY ====================")
        print(f"Total Tasks Evaluated     : {self.total_tasks}")
        print(f"Overall Success Rate     : {self.successful_tasks} / {self.total_tasks} ({self.accuracy * 100:.1f}%)")
        print(f"Tool Tasks Accuracy      : {self.tool_required_accuracy * 100:.1f}%")
        print(f"No-Tool Tasks Accuracy   : {self.no_tool_accuracy * 100:.1f}%")
        print(f"Tool Selection Accuracy  : {self.tool_selection_accuracy * 100:.1f}%")
        print(f"Unnecessary Tool-Use Rate: {self.unnecessary_tool_use_rate * 100:.1f}%")
        print(f"Multi-Step Tasks Accuracy: {self.multi_step_accuracy * 100:.1f}%")
        print(f"Avg Steps / Task         : {self.avg_steps_per_task:.2f}")
        print(f"Valid Tool Call Rate     : {self.valid_tool_call_rate * 100:.1f}%")
        print(f"Category Accuracies      : {json.dumps({k: round(v * 100, 1) for k, v in self.category_accuracies.items()})}")
        print(f"Termination Reasons      : {json.dumps(self.termination_reasons)}")
        print("============================================================\n")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "total_tasks": self.total_tasks,
                "successful_tasks": self.successful_tasks,
                "accuracy": self.accuracy,
                "tool_required_accuracy": self.tool_required_accuracy,
                "no_tool_accuracy": self.no_tool_accuracy,
                "tool_selection_accuracy": self.tool_selection_accuracy,
                "unnecessary_tool_use_rate": self.unnecessary_tool_use_rate,
                "multi_step_accuracy": self.multi_step_accuracy,
                "avg_steps_per_task": self.avg_steps_per_task,
                "total_tool_calls": self.total_tool_calls,
                "valid_tool_call_rate": self.valid_tool_call_rate,
                "category_accuracies": self.category_accuracies,
                "termination_reasons": self.termination_reasons,
            },
            "task_results": [r.to_dict() for r in self.task_results],
        }

    def save_metrics(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict()["summary"], f, indent=2)

    def save_trajectories_jsonl(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            for r in self.task_results:
                f.write(json.dumps(r.trajectory.to_dict()) + "\n")


class Evaluator:
    """
    Modular evaluator for assessing harness trajectories against task benchmarks.
    """

    def evaluate_task(self, task: Dict[str, Any], trajectory: Trajectory) -> TaskResult:
        task_id = task.get("id", "unknown_task")
        category = task.get("category", "general")
        prompt = task.get("prompt", "")
        expected_answer = task.get("expected_answer")
        requires_tool = task.get("requires_tool", False)
        
        expected_behavior = task.get("expected_behavior", {})
        allowed_tools = expected_behavior.get("allowed_tools", [])

        actual_answer = trajectory.final_answer

        # Exact match or normalized substring match check
        task_success = False
        if expected_answer is not None and actual_answer is not None:
            norm_exp = normalize_answer(expected_answer)
            norm_act = normalize_answer(actual_answer)
            task_success = norm_exp in norm_act or norm_act in norm_exp

        # Step metrics
        tool_calls = [step for step in trajectory.steps if step.action_type == "tool_call"]
        num_tool_calls = len(tool_calls)
        valid_tool_calls = sum(1 for step in tool_calls if step.error is None)
        tool_call_validity = (valid_tool_calls / num_tool_calls) if num_tool_calls > 0 else 1.0

        # Tool selection check
        correct_tool_selected = True
        if requires_tool and allowed_tools and tool_calls:
            used_tools = set(step.tool_name for step in tool_calls if step.tool_name)
            correct_tool_selected = any(t in allowed_tools for t in used_tools)
        elif requires_tool and not tool_calls:
            correct_tool_selected = False

        # Unnecessary tool use check
        unnecessary_tool_used = (not requires_tool) and (num_tool_calls > 0)

        return TaskResult(
            task_id=task_id,
            category=category,
            prompt=prompt,
            expected_answer=expected_answer,
            requires_tool=requires_tool,
            allowed_tools=allowed_tools,
            actual_answer=actual_answer,
            task_success=task_success,
            total_steps=len(trajectory.steps),
            num_tool_calls=num_tool_calls,
            tool_call_validity=tool_call_validity,
            correct_tool_selected=correct_tool_selected,
            unnecessary_tool_used=unnecessary_tool_used,
            termination_reason=trajectory.termination_reason,
            trajectory=trajectory,
        )

    def evaluate_batch(self, tasks: List[Dict[str, Any]], harness: AgentHarness) -> EvaluationSummary:
        results: List[TaskResult] = []
        term_reasons: Dict[str, int] = {}

        for task in tasks:
            task_id = task.get("id", "task_0")
            prompt = task.get("prompt", "")
            trajectory = harness.run(prompt=prompt, task_id=task_id)

            res = self.evaluate_task(task, trajectory)
            results.append(res)

            term = trajectory.termination_reason
            term_reasons[term] = term_reasons.get(term, 0) + 1

        total_tasks = len(results)
        if total_tasks == 0:
            return EvaluationSummary(
                total_tasks=0,
                successful_tasks=0,
                accuracy=0.0,
                tool_required_accuracy=0.0,
                no_tool_accuracy=0.0,
                tool_selection_accuracy=0.0,
                unnecessary_tool_use_rate=0.0,
                multi_step_accuracy=0.0,
                avg_steps_per_task=0.0,
                total_tool_calls=0,
                valid_tool_call_rate=0.0,
                category_accuracies={},
                termination_reasons={},
                task_results=[],
            )

        successful_tasks = sum(1 for r in results if r.task_success)
        accuracy = successful_tasks / total_tasks

        tool_tasks = [r for r in results if r.requires_tool]
        no_tool_tasks = [r for r in results if not r.requires_tool]
        multi_step_tasks = [r for r in results if r.category in ("two_step", "multi_step")]

        tool_req_acc = (sum(1 for r in tool_tasks if r.task_success) / len(tool_tasks)) if tool_tasks else 0.0
        no_tool_acc = (sum(1 for r in no_tool_tasks if r.task_success) / len(no_tool_tasks)) if no_tool_tasks else 0.0
        multi_step_acc = (sum(1 for r in multi_step_tasks if r.task_success) / len(multi_step_tasks)) if multi_step_tasks else 0.0

        tool_sel_acc = (sum(1 for r in tool_tasks if r.correct_tool_selected) / len(tool_tasks)) if tool_tasks else 1.0
        unnecessary_rate = (sum(1 for r in no_tool_tasks if r.unnecessary_tool_used) / len(no_tool_tasks)) if no_tool_tasks else 0.0

        avg_steps = sum(r.total_steps for r in results) / total_tasks
        total_calls = sum(r.num_tool_calls for r in results)
        valid_calls = sum(
            sum(1 for s in r.trajectory.steps if s.action_type == "tool_call" and s.error is None)
            for r in results
        )
        valid_rate = (valid_calls / total_calls) if total_calls > 0 else 1.0

        # Category breakdown
        cat_accuracies: Dict[str, float] = {}
        cats = set(r.category for r in results)
        for cat in cats:
            cat_items = [r for r in results if r.category == cat]
            cat_accuracies[cat] = (sum(1 for r in cat_items if r.task_success) / len(cat_items)) if cat_items else 0.0

        return EvaluationSummary(
            total_tasks=total_tasks,
            successful_tasks=successful_tasks,
            accuracy=accuracy,
            tool_required_accuracy=tool_req_acc,
            no_tool_accuracy=no_tool_acc,
            tool_selection_accuracy=tool_sel_acc,
            unnecessary_tool_use_rate=unnecessary_rate,
            multi_step_accuracy=multi_step_acc,
            avg_steps_per_task=avg_steps,
            total_tool_calls=total_calls,
            valid_tool_call_rate=valid_rate,
            category_accuracies=cat_accuracies,
            termination_reasons=term_reasons,
            task_results=results,
        )

