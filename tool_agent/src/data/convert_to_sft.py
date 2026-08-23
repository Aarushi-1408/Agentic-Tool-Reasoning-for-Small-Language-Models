import os
import re
import json
from typing import Dict, Any, List, Optional, Tuple, Set
from src.tools import ToolRegistry
from src.tool_schema import validate_arguments
from tools.example_tools import CalculatorTool, EchoTool, LookupTool, StringManipulatorTool


def clean_think_tags(text: str) -> str:
    """
    Remove <think>...</think> reasoning blocks from assistant text.
    Preserves clean observable assistant outputs (tool calls or final answers).
    """
    if not text or not isinstance(text, str):
        return ""
    # Strip <think>...</think> blocks including multi-line content
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Strip leading/trailing empty lines left behind
    return cleaned.strip()


class SFTConverter:
    """
    Converter that transforms raw teacher trajectories into standard SFT dataset records.
    
    Performs:
    1. Trajectory validation and filtering.
    2. Removal of hidden reasoning tags (<think>...</think>).
    3. Categorization into no_tool, single_tool, two_step, multi_step.
    4. Serialization to Hugging Face TRL-compatible SFT format.
    5. Rejection logging and detailed dataset statistics export.
    """

    def __init__(self, tool_registry: Optional[ToolRegistry] = None):
        if tool_registry is None:
            tool_registry = ToolRegistry()
            tool_registry.register(CalculatorTool())
            tool_registry.register(EchoTool())
            tool_registry.register(LookupTool())
            tool_registry.register(StringManipulatorTool())
        self.tool_registry = tool_registry
        self.registered_tool_names: Set[str] = set(tool_registry.list_tool_names())

    def validate_trajectory(self, record: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate trajectory structure and execution consistency.
        Returns (is_valid, rejection_reason).
        """
        if not isinstance(record, dict):
            return False, "malformed_json"

        # Check required fields
        required_fields = ["task_id", "steps", "termination_reason"]
        for key in required_fields:
            if key not in record:
                return False, "malformed_json"

        termination_reason = record.get("termination_reason")
        if termination_reason == "max_steps_exceeded":
            return False, "max_steps_reached"
        if termination_reason != "final_answer":
            return False, "incomplete_trajectory"

        final_answer = record.get("final_answer")
        if final_answer is None or str(final_answer).strip() == "":
            return False, "missing_final_answer"

        steps = record.get("steps")
        if not isinstance(steps, list) or len(steps) == 0:
            return False, "incomplete_trajectory"

        has_final_step = False
        expected_step = 0

        for step in steps:
            if not isinstance(step, dict):
                return False, "malformed_json"

            step_idx = step.get("step")
            if step_idx is None or step_idx != expected_step:
                return False, "invalid_message_sequence"
            expected_step += 1

            action_type = step.get("action_type")
            if action_type not in ["tool_call", "final", "parse_error"]:
                return False, "invalid_message_sequence"

            if action_type == "parse_error":
                return False, "malformed_json"

            if action_type == "final":
                has_final_step = True

            if action_type == "tool_call":
                tool_name = step.get("tool_name")
                arguments = step.get("arguments")
                tool_result = step.get("tool_result")

                if not tool_name or not isinstance(tool_name, str) or tool_name not in self.registered_tool_names:
                    return False, "invalid_tool"

                if arguments is None or not isinstance(arguments, dict):
                    return False, "invalid_arguments"

                # Validate argument schema
                tool_obj = self.tool_registry.get(tool_name)
                if tool_obj:
                    is_valid_arg, _ = validate_arguments(tool_obj.parameters, arguments)
                    if not is_valid_arg:
                        return False, "invalid_arguments"

                if tool_result is None or str(tool_result).strip() == "":
                    return False, "missing_tool_result"

                if step.get("error") is not None:
                    # Trajectory contains an unrecovered tool error
                    return False, "missing_tool_result"

        if not has_final_step:
            return False, "missing_final_answer"

        # Validate messages structure
        messages = record.get("messages")
        if not isinstance(messages, list) or len(messages) == 0:
            return False, "invalid_message_sequence"

        if messages[0].get("role") != "user":
            return False, "invalid_message_sequence"

        return True, None

    def convert_trajectory(self, record: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str], str]:
        """
        Convert a raw trajectory record into a clean SFT dataset record.
        Returns (sft_record, rejection_reason, category).
        """
        is_valid, rejection_reason = self.validate_trajectory(record)
        if not is_valid:
            return None, rejection_reason, "rejected"

        trajectory_id = record.get("trajectory_id", f"traj_{record.get('task_id', 'unknown')}")
        task_id = record.get("task_id", "unknown")
        raw_messages = record.get("messages", [])

        sft_messages = []
        tool_calls_count = 0

        for msg in raw_messages:
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "user":
                sft_messages.append({
                    "role": "user",
                    "content": content
                })
            elif role == "assistant":
                cleaned_content = clean_think_tags(content)
                sft_messages.append({
                    "role": "assistant",
                    "content": cleaned_content
                })
            else:
                # Other custom roles
                sft_messages.append({
                    "role": role,
                    "content": content
                })

        # Count tool calls based on trajectory steps
        steps = record.get("steps", [])
        tool_calls_count = sum(1 for s in steps if s.get("action_type") == "tool_call")

        if tool_calls_count == 0:
            category = "no_tool"
        elif tool_calls_count == 1:
            category = "single_tool"
        elif tool_calls_count == 2:
            category = "two_step"
        else:
            category = "multi_step"

        sft_record = {
            "trajectory_id": trajectory_id,
            "task_id": task_id,
            "category": category,
            "messages": sft_messages,
        }

        return sft_record, None, category

    def process_file(
        self,
        input_file: str,
        output_sft_file: str,
        output_rejected_file: Optional[str] = None,
        stats_file: Optional[str] = None,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """
        Process a file of raw teacher trajectories into SFT datasets and dataset statistics.
        """
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input trajectories file not found at: '{input_file}'")

        if os.path.exists(output_sft_file) and not overwrite:
            raise FileExistsError(
                f"Output SFT dataset already exists at '{output_sft_file}'. "
                "Use --overwrite to allow overwriting existing datasets."
            )

        output_rejected_file = output_rejected_file or os.path.join(
            os.path.dirname(output_sft_file), "rejected.jsonl"
        )
        stats_file = stats_file or os.path.join(
            os.path.dirname(output_sft_file), "dataset_stats.json"
        )

        total_raw = 0
        accepted_records = []
        rejected_records = []

        category_counts = {
            "no_tool": 0,
            "single_tool": 0,
            "two_step": 0,
            "multi_step": 0,
        }
        rejection_reasons: Dict[str, int] = {}
        total_messages = 0
        total_tool_calls = 0

        with open(input_file, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                total_raw += 1

                try:
                    record = json.loads(line)
                    sft_record, rejection_reason, category = self.convert_trajectory(record)

                    if sft_record is not None:
                        accepted_records.append(sft_record)
                        category_counts[category] += 1
                        total_messages += len(sft_record["messages"])

                        # Count tool calls
                        steps = record.get("steps", [])
                        t_calls = sum(1 for s in steps if s.get("action_type") == "tool_call")
                        total_tool_calls += t_calls
                    else:
                        traj_id = record.get("trajectory_id", f"line_{line_idx}")
                        reason = rejection_reason or "unknown"
                        rejected_records.append({
                            "trajectory_id": traj_id,
                            "task_id": record.get("task_id", "unknown"),
                            "reason": reason,
                        })
                        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

                except json.JSONDecodeError:
                    rejected_records.append({
                        "trajectory_id": f"line_{line_idx}",
                        "reason": "malformed_json",
                    })
                    rejection_reasons["malformed_json"] = rejection_reasons.get("malformed_json", 0) + 1

        # Save output files
        os.makedirs(os.path.dirname(os.path.abspath(output_sft_file)), exist_ok=True)
        with open(output_sft_file, "w", encoding="utf-8") as out_f:
            for item in accepted_records:
                out_f.write(json.dumps(item) + "\n")

        os.makedirs(os.path.dirname(os.path.abspath(output_rejected_file)), exist_ok=True)
        with open(output_rejected_file, "w", encoding="utf-8") as rej_f:
            for item in rejected_records:
                rej_f.write(json.dumps(item) + "\n")

        valid_count = len(accepted_records)
        rejected_count = len(rejected_records)
        avg_messages = (total_messages / valid_count) if valid_count > 0 else 0.0
        avg_tool_calls = (total_tool_calls / valid_count) if valid_count > 0 else 0.0

        stats = {
            "raw_trajectories": total_raw,
            "valid_trajectories": valid_count,
            "rejected_trajectories": rejected_count,
            "acceptance_rate": (valid_count / total_raw) if total_raw > 0 else 0.0,
            "categories": category_counts,
            "average_messages_per_example": round(avg_messages, 2),
            "average_tool_calls_per_example": round(avg_tool_calls, 2),
            "rejection_reasons": rejection_reasons,
        }

        os.makedirs(os.path.dirname(os.path.abspath(stats_file)), exist_ok=True)
        with open(stats_file, "w", encoding="utf-8") as s_f:
            json.dump(stats, s_f, indent=2)

        return stats
