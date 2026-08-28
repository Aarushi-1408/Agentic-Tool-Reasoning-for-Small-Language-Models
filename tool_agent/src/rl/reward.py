from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from src.trajectory import Trajectory
from src.evaluator import normalize_answer


@dataclass
class RewardResult:
    """
    Structured trajectory reward assessment.
    """
    total_reward: float
    correctness_reward: float
    policy_reward: float
    selection_reward: float
    completion_reward: float
    unnecessary_tool_penalty: float
    repetition_penalty: float
    premature_penalty: float
    parse_penalty: float
    execution_error_penalty: float
    excess_step_penalty: float
    breakdown_string: str
    is_correct: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_reward": round(self.total_reward, 4),
            "correctness_reward": round(self.correctness_reward, 4),
            "policy_reward": round(self.policy_reward, 4),
            "selection_reward": round(self.selection_reward, 4),
            "completion_reward": round(self.completion_reward, 4),
            "unnecessary_tool_penalty": round(self.unnecessary_tool_penalty, 4),
            "repetition_penalty": round(self.repetition_penalty, 4),
            "premature_penalty": round(self.premature_penalty, 4),
            "parse_penalty": round(self.parse_penalty, 4),
            "execution_error_penalty": round(self.execution_error_penalty, 4),
            "excess_step_penalty": round(self.excess_step_penalty, 4),
            "breakdown_string": self.breakdown_string,
            "is_correct": self.is_correct,
        }


class RewardCalculator:
    """
    Trajectory-level reward calculator for GRPO.
    """

    def evaluate(self, trajectory: Trajectory, task_metadata: Dict[str, Any]) -> RewardResult:
        category = task_metadata.get("category", "general")
        expected_answer = task_metadata.get("expected_answer")
        requires_tool = task_metadata.get("requires_tool", False)
        
        expected_behavior = task_metadata.get("expected_behavior", {})
        allowed_tools = expected_behavior.get("allowed_tools", task_metadata.get("allowed_tools", []))
        expected_steps = expected_behavior.get("expected_steps", 1)

        actual_answer = trajectory.final_answer

        # 1. Base Correctness
        is_correct = False
        if expected_answer is not None and actual_answer is not None:
            norm_exp = normalize_answer(expected_answer)
            norm_act = normalize_answer(actual_answer)
            is_correct = norm_exp in norm_act or norm_act in norm_exp

        r_correctness = 1.0 if is_correct else 0.0

        # Step analysis
        tool_calls = [s for s in trajectory.steps if s.action_type == "tool_call"]
        parse_errors = [s for s in trajectory.steps if s.action_type == "parse_error"]
        exec_errors = [s for s in trajectory.steps if s.action_type == "tool_call" and s.error is not None]
        num_tool_calls = len(tool_calls)

        # 2. No-Tool Policy & Unnecessary Tool Penalty
        r_policy = 0.0
        p_unnecessary = 0.0

        if not requires_tool:
            if num_tool_calls == 0 and is_correct:
                r_policy = 0.5
            elif num_tool_calls > 0:
                p_unnecessary = 0.75
        else:
            if num_tool_calls > 0:
                r_policy = 0.25

        # 3. Tool Selection
        r_selection = 0.0
        if requires_tool and allowed_tools and tool_calls:
            used_tools = set(s.tool_name for s in tool_calls if s.tool_name)
            if any(t in allowed_tools for t in used_tools):
                r_selection = 0.25

        # 4. Multi-Step Completion & Premature Termination
        r_completion = 0.0
        p_premature = 0.0

        if category in ("two_step", "multi_step") or requires_tool:
            required_count = 2 if category == "two_step" else (3 if category == "multi_step" else 1)
            if num_tool_calls >= required_count:
                r_completion = 0.50
            elif trajectory.termination_reason == "final_answer" and num_tool_calls < required_count:
                p_premature = 0.50

        # 5. Repetition Penalty
        p_repetition = 0.0
        if num_tool_calls > 1:
            for i in range(len(tool_calls) - 1):
                t1, t2 = tool_calls[i], tool_calls[i+1]
                if t1.tool_name == t2.tool_name and t1.arguments == t2.arguments:
                    p_repetition += 0.50

        # 6. Parse Errors & Tool Execution Errors
        p_parse = 0.25 * len(parse_errors)
        p_exec = 0.25 * len(exec_errors)

        # 7. Excess Steps Penalty
        total_steps = len(trajectory.steps)
        excess_steps = max(0, total_steps - expected_steps)
        p_excess = 0.05 * excess_steps

        # Total Calculation
        total_reward = (
            r_correctness
            + r_policy
            + r_selection
            + r_completion
            - p_unnecessary
            - p_repetition
            - p_premature
            - p_parse
            - p_exec
            - p_excess
        )

        breakdown = (
            f"R_total={total_reward:+.2f} | R_corr={r_correctness:+.2f}, R_pol={r_policy:+.2f}, "
            f"R_sel={r_selection:+.2f}, R_comp={r_completion:+.2f} | P_unnec={-p_unnecessary:-.2f}, "
            f"P_rep={-p_repetition:-.2f}, P_prem={-p_premature:-.2f}, P_parse={-p_parse:-.2f}, "
            f"P_exec={-p_exec:-.2f}, P_excess={-p_excess:-.2f}"
        )

        return RewardResult(
            total_reward=total_reward,
            correctness_reward=r_correctness,
            policy_reward=r_policy,
            selection_reward=r_selection,
            completion_reward=r_completion,
            unnecessary_tool_penalty=p_unnecessary,
            repetition_penalty=p_repetition,
            premature_penalty=p_premature,
            parse_penalty=p_parse,
            execution_error_penalty=p_exec,
            excess_step_penalty=p_excess,
            breakdown_string=breakdown,
            is_correct=is_correct,
        )


def calculate_reward(trajectory: Trajectory, task_metadata: Dict[str, Any]) -> RewardResult:
    """
    Functional wrapper for reward computation.
    """
    calc = RewardCalculator()
    return calc.evaluate(trajectory, task_metadata)
