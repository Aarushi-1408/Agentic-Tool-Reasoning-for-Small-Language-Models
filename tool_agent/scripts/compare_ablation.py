#!/usr/bin/env python3
"""
CLI script to analyze and compare two evaluation runs (Ablation Comparison):
Run A: evaluation/sft_v2_100/ (SFT Model without system prompt)
Run B: evaluation/sft_v2_100_system_prompt/ (SFT Model with system prompt)

Usage:
  python scripts/compare_ablation.py \
    --run-a evaluation/sft_v2_100/ \
    --run-b evaluation/sft_v2_100_system_prompt/
"""

import sys
import os
import json
import argparse
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_json(path_str: str):
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"File not found at '{path_str}'")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path_str: str):
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"File not found at '{path_str}'")
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def format_val(val, is_percentage=True):
    if val is None:
        return "N/A"
    if isinstance(val, (int, float)):
        if is_percentage and 0.0 <= val <= 1.0:
            return f"{val * 100:.2f}%"
        elif isinstance(val, float):
            return f"{val:.2f}"
        return str(val)
    return str(val)


def analyze_trajectories(trajectories, eval_tasks):
    categorized = {"A": [], "B": [], "C": [], "D": [], "E": [], "F": [], "G": [], "PASSED": []}
    
    repeated_identical_calls = 0
    tc_to_result_to_final = 0
    tc_to_result_to_same_tc = 0
    max_steps_cases = 0

    for t in trajectories:
        task_id = t["task_id"]
        spec = eval_tasks.get(task_id, {})
        success = t.get("success", False)
        category = spec.get("category", t.get("category", "unknown"))
        prompt = spec.get("prompt", t.get("prompt", ""))
        expected_answer = spec.get("expected_answer")
        expected_tools = spec.get("expected_tools", [])
        steps = t.get("steps", [])
        term_reason = t.get("termination_reason", "")
        final_answer = t.get("final_answer")

        if term_reason == "max_steps_exceeded":
            max_steps_cases += 1

        # Analyze transitions & repetitions
        tool_calls = [s.get("tool_name") for s in steps if s.get("action_type") == "tool_call"]
        args_list = [json.dumps(s.get("arguments", {}), sort_keys=True) for s in steps if s.get("action_type") == "tool_call"]

        has_repetition = False
        for i in range(len(args_list) - 1):
            if tool_calls[i] == tool_calls[i+1] and args_list[i] == args_list[i+1]:
                has_repetition = True
                repeated_identical_calls += 1

        # Check transitions after tool result
        for i in range(len(steps) - 1):
            curr_step = steps[i]
            next_step = steps[i+1]
            if curr_step.get("action_type") == "tool_call" and curr_step.get("tool_result") is not None:
                if next_step.get("action_type") == "final":
                    tc_to_result_to_final += 1
                elif next_step.get("action_type") == "tool_call":
                    if curr_step.get("tool_name") == next_step.get("tool_name") and curr_step.get("arguments") == next_step.get("arguments"):
                        tc_to_result_to_same_tc += 1

        if success:
            categorized["PASSED"].append(task_id)
            continue

        first_step = steps[0] if steps else None
        first_action = first_step.get("action_type") if first_step else None
        first_tool = first_step.get("tool_name") if first_step else None

        # Failure classification
        if category == "no_tool" and first_action == "tool_call":
            categorized["A"].append((task_id, f"Invoked tool '{first_tool}' on no_tool task"))
            continue

        if first_action == "parse_error":
            categorized["E"].append((task_id, "First step parse error"))
            continue

        if first_action == "tool_call" and expected_tools and first_tool not in expected_tools:
            categorized["D"].append((task_id, f"Used '{first_tool}', expected {expected_tools}"))
            continue

        if has_repetition and term_reason == "max_steps_exceeded":
            categorized["B"].append((task_id, f"Repeated tool '{first_tool}' until max_steps"))
            continue

        if term_reason == "max_steps_exceeded":
            categorized["C"].append((task_id, f"Executed tools {tool_calls[:3]} but hit max_steps without final answer"))
            continue

        has_parse_or_exec_error = any(s.get("action_type") == "parse_error" or (s.get("tool_result") and "TOOL_ERROR" in str(s.get("tool_result"))) for s in steps)
        if has_parse_or_exec_error:
            categorized["E"].append((task_id, "Parse/execution error in trajectory"))
            continue

        if term_reason == "final_answer":
            categorized["G"].append((task_id, f"Final answer '{final_answer}' != expected '{expected_answer}'"))
            continue

        categorized["F"].append((task_id, "Tool result misunderstood"))

    return {
        "categorized": categorized,
        "repeated_identical_calls": repeated_identical_calls,
        "tc_to_result_to_final": tc_to_result_to_final,
        "tc_to_result_to_same_tc": tc_to_result_to_same_tc,
        "max_steps_cases": max_steps_cases,
    }


def main():
    parser = argparse.ArgumentParser(description="System Prompt Ablation Comparison Tool")
    parser.add_argument("--run-a", type=str, default="evaluation/sft_v2_100/", help="Directory path to Run A (No System Prompt)")
    parser.add_argument("--run-b", type=str, default="evaluation/sft_v2_100_system_prompt/", help="Directory path to Run B (With System Prompt)")
    parser.add_argument("--tasks", type=str, default="tasks/eval_v1.json", help="Path to tasks JSON file")
    args = parser.parse_args()

    dir_a = Path(args.run_a)
    dir_b = Path(args.run_b)
    eval_tasks_path = Path(args.tasks)

    m_a = load_json(dir_a / "metrics.json")
    t_a = load_jsonl(dir_a / "trajectories.jsonl")

    m_b = load_json(dir_b / "metrics.json")
    t_b = load_jsonl(dir_b / "trajectories.jsonl")

    eval_tasks = {}
    if eval_tasks_path.exists():
        with open(eval_tasks_path, "r", encoding="utf-8") as f:
            eval_tasks = {t["id"]: t for t in json.load(f)}

    analysis_a = analyze_trajectories(t_a, eval_tasks)
    analysis_b = analyze_trajectories(t_b, eval_tasks)

    print("\n" + "=" * 85)
    print("      SFT MODEL SYSTEM PROMPT ABLATION COMPARISON (RUN A vs RUN B)")
    print("=" * 85)
    print(f"Run A (No System Prompt)   : {dir_a}")
    print(f"Run B (With System Prompt) : {dir_b}")
    print("-" * 85)

    metrics_list = [
        ("Overall Accuracy", "accuracy", True),
        ("Tool Tasks Accuracy", "tool_required_accuracy", True),
        ("No-Tool Tasks Accuracy", "no_tool_accuracy", True),
        ("Tool Selection Accuracy", "tool_selection_accuracy", True),
        ("Unnecessary Tool-Use Rate", "unnecessary_tool_use_rate", True),
        ("Multi-Step Accuracy", "multi_step_accuracy", True),
        ("Valid Tool-Call Rate", "valid_tool_call_rate", True),
        ("Average Steps / Task", "avg_steps_per_task", False),
    ]

    print(f"{'Metric':<30} | {'Run A (No System Prompt)':<26} | {'Run B (With System Prompt)':<26} | {'Delta':<10}")
    print("-" * 85)

    for label, key, is_pct in metrics_list:
        v_a = m_a.get(key, 0.0)
        v_b = m_b.get(key, 0.0)
        str_a = format_val(v_a, is_pct)
        str_b = format_val(v_b, is_pct)
        diff = v_b - v_a
        diff_str = f"{diff * 100:+.2f}%" if is_pct else f"{diff:+.2f}"
        print(f"{label:<30} | {str_a:<26} | {str_b:<26} | {diff_str:<10}")

    print("-" * 85)
    print("Category Breakdown Accuracy:")
    cat_a = m_a.get("category_accuracies", {})
    cat_b = m_b.get("category_accuracies", {})
    all_cats = sorted(list(set(cat_a.keys()) | set(cat_b.keys())))

    for cat in all_cats:
        va = cat_a.get(cat, 0.0)
        vb = cat_b.get(cat, 0.0)
        str_a = format_val(va, True)
        str_b = format_val(vb, True)
        diff_str = f"{(vb - va) * 100:+.2f}%"
        print(f"  - {cat:<26} | {str_a:<26} | {str_b:<26} | {diff_str:<10}")

    print("-" * 85)
    print("Failure Taxonomy Breakdown (A through G):")
    taxonomy_labels = {
        "A": "A. Unnecessary first tool",
        "B": "B. Correct first tool but repeats it",
        "C": "C. Correct tool sequence but doesn't terminate",
        "D": "D. Wrong tool",
        "E": "E. Malformed tool call",
        "F": "F. Tool result misunderstood",
        "G": "G. Final answer incorrect",
    }
    cat_tax_a = analysis_a["categorized"]
    cat_tax_b = analysis_b["categorized"]

    for code in ["A", "B", "C", "D", "E", "F", "G"]:
        count_a = len(cat_tax_a[code])
        count_b = len(cat_tax_b[code])
        diff_count = count_b - count_a
        print(f"  - {taxonomy_labels[code]:<46} | Run A: {count_a:>2} tasks | Run B: {count_b:>2} tasks | Delta: {diff_count:+d}")

    print("-" * 85)
    print("Specific Trajectory Transition & Repetition Statistics:")
    stat_rows = [
        ("Repeated Identical Tool Calls", analysis_a["repeated_identical_calls"], analysis_b["repeated_identical_calls"]),
        ("tool_call -> TOOL_RESULT -> final", analysis_a["tc_to_result_to_final"], analysis_b["tc_to_result_to_final"]),
        ("tool_call -> TOOL_RESULT -> repeat same tool", analysis_a["tc_to_result_to_same_tc"], analysis_b["tc_to_result_to_same_tc"]),
        ("Max Steps Exceeded Cases", analysis_a["max_steps_cases"], analysis_b["max_steps_cases"]),
    ]

    for label, va, vb in stat_rows:
        diff_val = vb - va
        print(f"  - {label:<46} | Run A: {va:>2}       | Run B: {vb:>2}       | Delta: {diff_val:+d}")

    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()
