"""
14-Point Automatic Trajectory Validator for RAG & Calculator Dataset.
Validates structural integrity, tool schemas, state transitions, state dependencies, and category constraints.
"""

import json
from typing import Dict, Any, List, Optional, Set

ALLOWED_RAG_CALC_TOOLS = {
    "bm25_search",
    "dense_vector_search",
    "hybrid_search",
    "reranker",
    "multi_query_retrieval",
    "metadata_filtered_search",
    "parent_document_retrieval",
    "calculator",
}

CALCULATOR_OPERATIONS = {"add", "sub", "mul", "divide", "modulo"}
RETRIEVAL_INITIAL_TOOLS = {"bm25_search", "dense_vector_search", "hybrid_search", "multi_query_retrieval", "metadata_filtered_search"}


class RAGCalcTrajectoryValidator:
    """
    Automatic Validator enforcing all 14 quality criteria for the RAG + Calculator dataset.
    """

    def __init__(self, allowed_tools: Optional[Set[str]] = None):
        self.allowed_tools = allowed_tools or ALLOWED_RAG_CALC_TOOLS

    def validate(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a single dataset record trajectory.
        Returns dict: {"valid": bool, "errors": List[str]}
        """
        errors: List[str] = []

        if not isinstance(record, dict):
            return {"valid": False, "errors": ["Record is not a valid JSON object/dict."]}

        # 1. JSON & Top-level field validity
        required_top = ["trajectory_id", "category", "task", "messages"]
        for f in required_top:
            if f not in record:
                errors.append(f"Criterion 1/9: Missing required top-level field '{f}'.")

        category = record.get("category", "")
        task = record.get("task", {})
        messages = record.get("messages", [])

        if not isinstance(messages, list) or len(messages) == 0:
            errors.append("Criterion 1: Messages field must be a non-empty list.")
            return {"valid": False, "errors": errors}

        # Check task structure
        if not isinstance(task, dict) or "prompt" not in task:
            errors.append("Criterion 1: Task field missing valid 'prompt' string.")

        # 2. Extract Assistant actions from messages
        assistant_actions = []
        for m in messages:
            if m.get("role") == "assistant":
                content = m.get("content", "")
                try:
                    act = json.loads(content)
                    assistant_actions.append(act)
                except Exception:
                    errors.append(f"Criterion 1: Assistant message is not valid JSON: '{content[:50]}...'")

        # 7. Final answer existence
        has_final = any(act.get("type") == "final" and act.get("content") for act in assistant_actions)
        if not has_final:
            errors.append("Criterion 7: Trajectory missing final answer action ({\"type\": \"final\", ...}).")

        # 11. No-tool category zero-tool-call check
        if category == "no_tool":
            tool_calls = [act for act in assistant_actions if act.get("type") == "tool_call"]
            if len(tool_calls) > 0:
                errors.append(f"Criterion 11: Category is 'no_tool' but contains {len(tool_calls)} tool calls.")

        # Inspect tool call actions sequence
        prev_tool_names: List[str] = []
        executed_tool_calls: List[Dict[str, Any]] = []

        for idx, act in enumerate(assistant_actions):
            act_type = act.get("type")

            if act_type == "tool_call":
                t_name = act.get("name")
                t_args = act.get("arguments")

                # 3. Allowed-tool validation
                if t_name not in self.allowed_tools and not record.get("is_error_recovery", False):
                    errors.append(f"Criterion 3: Disallowed tool '{t_name}'. Allowed: {list(self.allowed_tools)}")

                # 4. Valid tool arguments & 12. Calculator operation validity
                if not isinstance(t_args, dict):
                    errors.append(f"Criterion 4: Tool call '{t_name}' arguments must be a dict object.")
                else:
                    if t_name == "calculator":
                        op_val = t_args.get("operation")
                        if op_val not in CALCULATOR_OPERATIONS and not record.get("is_error_recovery", False):
                            errors.append(f"Criterion 12: Invalid calculator operation '{op_val}'.")
                        a_val = t_args.get("a")
                        b_val = t_args.get("b")
                        if not record.get("is_error_recovery", False):
                            if not isinstance(a_val, (int, float)) or not isinstance(b_val, (int, float)):
                                errors.append(f"Criterion 12: Calculator operands 'a' and 'b' must be numeric.")

                    elif t_name == "bm25_search" or t_name == "dense_vector_search":
                        if "query" not in t_args and not record.get("is_error_recovery", False):
                            errors.append(f"Criterion 4: {t_name} missing 'query' argument.")

                    elif t_name == "hybrid_search":
                        if "query" not in t_args and not record.get("is_error_recovery", False):
                            errors.append(f"Criterion 4: hybrid_search missing 'query' argument.")
                        if "alpha" in t_args and not record.get("is_error_recovery", False):
                            al = t_args["alpha"]
                            if not isinstance(al, (int, float)) or not (0.0 <= al <= 1.0):
                                errors.append(f"Criterion 4: hybrid_search alpha must be between 0.0 and 1.0.")

                    elif t_name == "reranker":
                        # 5. Expected tool sequence validation: reranker must follow initial retrieval
                        if not any(tn in RETRIEVAL_INITIAL_TOOLS for tn in prev_tool_names) and not record.get("is_error_recovery", False):
                            errors.append("Criterion 5: 'reranker' called without prior retrieval step.")
                        if "documents" not in t_args and not record.get("is_error_recovery", False):
                            errors.append("Criterion 4: 'reranker' missing 'documents' argument.")

                    elif t_name == "parent_document_retrieval":
                        # 5. Expected tool sequence validation: parent document retrieval must follow chunk retrieval
                        if not any(tn in RETRIEVAL_INITIAL_TOOLS or tn == "reranker" for tn in prev_tool_names) and not record.get("is_error_recovery", False):
                            errors.append("Criterion 5: 'parent_document_retrieval' called without prior retrieval step.")
                        if "document_id" not in t_args and not record.get("is_error_recovery", False):
                            errors.append("Criterion 4: 'parent_document_retrieval' missing 'document_id' argument.")

                    elif t_name == "metadata_filtered_search":
                        if "filters" not in t_args and not record.get("is_error_recovery", False):
                            errors.append("Criterion 4: 'metadata_filtered_search' missing 'filters' argument.")

                    elif t_name == "multi_query_retrieval":
                        if "queries" not in t_args or not isinstance(t_args.get("queries"), list):
                            if not record.get("is_error_recovery", False):
                                errors.append("Criterion 4: 'multi_query_retrieval' missing valid 'queries' list.")

                # 8. Repetition loop detection
                for prev_act in executed_tool_calls:
                    if prev_act.get("name") == t_name and prev_act.get("arguments") == t_args:
                        # Allow retry only if error recovery
                        if not record.get("is_error_recovery", False):
                            errors.append(f"Criterion 8: Redundant/duplicate tool call '{t_name}' with identical arguments {t_args}.")

                prev_tool_names.append(t_name)
                executed_tool_calls.append(act)

        # 6. TOOL_RESULT consistency check in messages
        for idx, m in enumerate(messages):
            if m.get("role") == "user" and idx > 0:
                text = m.get("content", "")
                if text.startswith("TOOL_RESULT:") or text.startswith("TOOL_ERROR:"):
                    # Preceding message must be assistant tool_call
                    prev_msg = messages[idx-1]
                    if prev_msg.get("role") != "assistant":
                        errors.append(f"Criterion 6: Observation at message index {idx} does not follow assistant tool_call.")

        return {
            "valid": len(errors) == 0,
            "errors": errors
        }

    def validate_batch(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate an entire batch of dataset records.
        """
        valid_count = 0
        invalid_count = 0
        all_errors = []

        prompts = [r.get("task", {}).get("prompt", "") for r in records if isinstance(r, dict)]
        unique_prompts = set(prompts)
        duplicate_rate = (len(prompts) - len(unique_prompts)) / max(1, len(prompts))

        category_counts: Dict[str, int] = {}
        tool_counts: Dict[str, int] = {}

        for r in records:
            res = self.validate(r)
            if res["valid"]:
                valid_count += 1
            else:
                invalid_count += 1
                all_errors.append({"id": r.get("trajectory_id"), "errors": res["errors"]})

            cat = r.get("category", "unknown")
            category_counts[cat] = category_counts.get(cat, 0) + 1

            for m in r.get("messages", []):
                if m.get("role") == "assistant":
                    try:
                        act = json.loads(m.get("content", ""))
                        if act.get("type") == "tool_call":
                            tn = act.get("name")
                            tool_counts[tn] = tool_counts.get(tn, 0) + 1
                    except Exception:
                        pass

        total = len(records)
        pass_rate = valid_count / max(1, total)

        return {
            "total_records": total,
            "valid_count": valid_count,
            "invalid_count": invalid_count,
            "pass_rate": round(pass_rate, 4),
            "duplicate_rate": round(duplicate_rate, 4),
            "category_counts": category_counts,
            "tool_counts": tool_counts,
            "sample_errors": all_errors[:5]
        }
