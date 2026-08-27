#!/usr/bin/env python3
"""
CLI script to automatically compare any evaluation run against the baseline and previous runs.

Usage:
  python scripts/check_regression.py --current evaluation/sft_v2_100_system_prompt/metrics.json --baseline evaluation/baseline/metrics.json
"""

import sys
import os
import json
import argparse
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_json(file_path: str) -> dict:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found at '{file_path}'")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_val(val, is_pct=True) -> str:
    if val is None:
        return "N/A"
    if isinstance(val, (int, float)):
        if is_pct and 0.0 <= val <= 1.0:
            return f"{val * 100:.2f}%"
        elif isinstance(val, float):
            return f"{val:.2f}"
        return str(val)
    return str(val)


def main():
    parser = argparse.ArgumentParser(description="Automated Regression Check against Baseline.")
    parser.add_argument("--current", type=str, required=True, help="Path to current run metrics.json")
    parser.add_argument("--baseline", type=str, default="evaluation/baseline/metrics.json", help="Path to baseline metrics.json")
    args = parser.parse_args()

    try:
        b_metrics = load_json(args.baseline)
    except FileNotFoundError:
        print(f"[ERROR] Baseline metrics not found at '{args.baseline}'")
        sys.exit(1)

    try:
        c_metrics = load_json(args.current)
    except FileNotFoundError:
        print(f"[ERROR] Current metrics not found at '{args.current}'")
        sys.exit(1)

    print("\n" + "=" * 85)
    print("                     AUTOMATED REGRESSION CHECK REPORT")
    print("=" * 85)
    print(f"Baseline Path : {args.baseline}")
    print(f"Current Run   : {args.current}")
    print("-" * 85)

    metrics_map = [
        ("Overall Accuracy", "accuracy", True),
        ("Tool Tasks Accuracy", "tool_required_accuracy", True),
        ("No-Tool Tasks Accuracy", "no_tool_accuracy", True),
        ("Tool Selection Accuracy", "tool_selection_accuracy", True),
        ("Unnecessary Tool-Use Rate", "unnecessary_tool_use_rate", True),
        ("Multi-Step Accuracy", "multi_step_accuracy", True),
        ("Valid Tool-Call Rate", "valid_tool_call_rate", True),
        ("Average Steps / Task", "avg_steps_per_task", False),
    ]

    print(f"{'Metric':<30} | {'Baseline':<12} | {'Current Run':<12} | {'Delta':<12}")
    print("-" * 85)

    regressions = []

    for label, key, is_pct in metrics_map:
        b_val = b_metrics.get(key, 0.0)
        c_val = c_metrics.get(key, 0.0)

        b_str = format_val(b_val, is_pct)
        c_str = format_val(c_val, is_pct)

        diff = c_val - b_val
        diff_str = f"{diff * 100:+.2f}%" if is_pct else f"{diff:+.2f}"

        # Flag regressions (accuracy drops or unnecessary tool rate spikes)
        if key == "unnecessary_tool_use_rate" and diff > 0.15:
            regressions.append(f"Unnecessary Tool Use rate increased by {diff * 100:+.2f}%")
        elif is_pct and key != "unnecessary_tool_use_rate" and diff < -0.05:
            regressions.append(f"{label} dropped by {diff * 100:+.2f}%")

        print(f"{label:<30} | {b_str:<12} | {c_str:<12} | {diff_str:<12}")

    print("-" * 85)
    print("Category Accuracies Breakdown:")
    b_cats = b_metrics.get("category_accuracies", {})
    c_cats = c_metrics.get("category_accuracies", {})
    all_cats = sorted(list(set(b_cats.keys()) | set(c_cats.keys())))

    for cat in all_cats:
        bv = b_cats.get(cat, 0.0)
        cv = c_cats.get(cat, 0.0)
        bv_str = format_val(bv, True)
        cv_str = format_val(cv, True)
        diff_str = f"{(cv - bv) * 100:+.2f}%"
        print(f"  - {cat:<26} | {bv_str:<12} | {cv_str:<12} | {diff_str:<12}")

    print("-" * 85)
    if regressions:
        print("[WARNING] REGRESSION WARNINGS DETECTED:")
        for reg in regressions:
            print(f"  - {reg}")
    else:
        print("[OK] NO CRITICAL REGRESSIONS DETECTED VS BASELINE.")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()
