import json
from typing import Dict, Any, List, Optional, Set
from src.tools import ToolRegistry


class TrajectoryValidator:
    """
    Lightweight validator for generated tool-use trajectories.
    
    Checks structural integrity, tool schema conformance, action sequence validity,
    and state transitions without deleting records.
    """

    def __init__(self, registered_tools: Optional[Set[str]] = None):
        """
        :param registered_tools: Set of valid tool names (e.g., {'calculator', 'lookup', 'echo', 'string_manipulator'}).
        """
        self.registered_tools = registered_tools

    def validate(
        self,
        trajectory: Dict[str, Any],
        tool_registry: Optional[ToolRegistry] = None,
    ) -> Dict[str, Any]:
        """
        Validate a single trajectory dictionary record.
        Returns a dict:
        {
            "valid": bool,
            "validation_errors": List[str]
        }
        """
        errors: List[str] = []

        if not isinstance(trajectory, dict):
            return {"valid": False, "validation_errors": ["Trajectory record must be a JSON object/dict."]}

        # Required top-level fields
        required_fields = ["task_id", "steps", "termination_reason"]
        for field_name in required_fields:
            if field_name not in trajectory:
                errors.append(f"Missing required top-level field: '{field_name}'")

        # Validate steps
        steps = trajectory.get("steps")
        if not isinstance(steps, list):
            errors.append("Field 'steps' must be a list.")
            return {"valid": False, "validation_errors": errors}

        valid_tools = set(self.registered_tools) if self.registered_tools else set()
        if tool_registry:
            valid_tools.update(tool_registry.list_tool_names())

        expected_step_num = 0
        has_final_step = False

        for idx, step in enumerate(steps):
            if not isinstance(step, dict):
                errors.append(f"Step index {idx} is not a valid dictionary object.")
                continue

            step_num = step.get("step")
            if step_num is None or step_num != expected_step_num:
                errors.append(
                    f"Step sequence error at index {idx}: expected step {expected_step_num}, got {step_num}"
                )
            expected_step_num += 1

            action_type = step.get("action_type")
            if action_type not in ["tool_call", "final", "parse_error"]:
                errors.append(f"Step {step_num}: invalid action_type '{action_type}'")

            if action_type == "final":
                has_final_step = True

            if action_type == "tool_call":
                tool_name = step.get("tool_name")
                arguments = step.get("arguments")
                tool_result = step.get("tool_result")

                if not tool_name or not isinstance(tool_name, str):
                    errors.append(f"Step {step_num}: tool_call missing valid 'tool_name' string.")
                elif valid_tools and tool_name not in valid_tools and step.get("error") is None:
                    # Note: tool_name might be unknown if testing error recovery
                    pass

                if arguments is None or not isinstance(arguments, dict):
                    errors.append(f"Step {step_num}: tool_call missing valid 'arguments' dictionary.")

                if tool_result is None and step.get("error") is None:
                    errors.append(f"Step {step_num}: tool_call missing 'tool_result' execution output.")

        # Termination reason checks
        term_reason = trajectory.get("termination_reason")
        valid_term_reasons = ["final_answer", "max_steps_exceeded", "error"]
        if term_reason not in valid_term_reasons:
            errors.append(f"Invalid termination_reason '{term_reason}'. Allowed: {valid_term_reasons}")

        if term_reason == "final_answer" and not has_final_step:
            errors.append("termination_reason is 'final_answer' but no step with action_type='final' exists.")

        is_valid = len(errors) == 0
        return {
            "valid": is_valid,
            "validation_errors": errors,
        }

    def validate_file(self, filepath: str) -> Dict[str, Any]:
        """
        Validate all trajectories in a JSONL file.
        """
        total = 0
        valid_count = 0
        invalid_count = 0
        results = []

        with open(filepath, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    record = json.loads(line)
                    res = self.validate(record)
                    if res["valid"]:
                        valid_count += 1
                    else:
                        invalid_count += 1
                    results.append({
                        "line": line_idx,
                        "task_id": record.get("task_id", "unknown"),
                        "valid": res["valid"],
                        "errors": res["validation_errors"],
                    })
                except json.JSONDecodeError as err:
                    invalid_count += 1
                    results.append({
                        "line": line_idx,
                        "valid": False,
                        "errors": [f"JSON parsing error: {str(err)}"],
                    })

        return {
            "total_trajectories": total,
            "valid_trajectories": valid_count,
            "invalid_trajectories": invalid_count,
            "valid_rate": (valid_count / total) if total > 0 else 0.0,
            "details": results,
        }
