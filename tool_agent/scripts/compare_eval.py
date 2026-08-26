#!/usr/bin/env python3
"""
CLI script to compare baseline evaluation metrics with SFT evaluation metrics.

Usage:
  python scripts/compare_eval.py --baseline evaluation/baseline/metrics.json --sft evaluation/sft_v2_100/metrics.json
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


def load_json(file_path: str):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found at '{file_path}'")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def main():
    parser = argparse.ArgumentParser(description="Compare Baseline vs SFT metrics.")
    parser.add_argument("--baseline", type=str, default="evaluation/baseline/metrics.json", help="Path to baseline metrics JSON")
    parser.add_argument("--sft", type=str, default="evaluation/sft_v2_100/metrics.json", help="Path to SFT metrics JSON")
    args = parser.parse_args()

    try:
        b_metrics = load_json(args.baseline)
    except FileNotFoundError:
        print(f"[ERROR] Baseline metrics not found at '{args.baseline}'")
        sys.exit(1)

    try:
        s_metrics = load_json(args.sft)
    except FileNotFoundError:
        print(f"[ERROR] SFT metrics not found at '{args.sft}'. Please run evaluate.py first.")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("           BASELINE vs SFT EVALUATION BENCHMARK COMPARISON")
    print("=" * 80)
    print(f"Baseline Metrics Path : {args.baseline}")
    print(f"SFT Metrics Path      : {args.sft}")
    print("-" * 80)

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

    print(f"{'Metric':<30} | {'Baseline':<12} | {'SFT Model':<12} | {'Delta':<12}")
    print("-" * 80)

    for label, key, is_pct in metrics_list:
        b_val = b_metrics.get(key, 0.0)
        s_val = s_metrics.get(key, 0.0)

        b_str = format_val(b_val, is_pct)
        s_str = format_val(s_val, is_pct)

        diff = s_val - b_val
        if is_pct:
            diff_str = f"{diff * 100:+.2f}%"
        else:
            diff_str = f"{diff:+.2f}"

        print(f"{label:<30} | {b_str:<12} | {s_str:<12} | {diff_str:<12}")

    print("-" * 80)
    print("Category Breakdown Accuracy:")
    b_cats = b_metrics.get("category_accuracies", {})
    s_cats = s_metrics.get("category_accuracies", {})
    all_cats = sorted(list(set(b_cats.keys()) | set(s_cats.keys())))

    for cat in all_cats:
        bv = b_cats.get(cat, 0.0)
        sv = s_cats.get(cat, 0.0)
        bv_str = format_val(bv, True)
        sv_str = format_val(sv, True)
        diff_str = f"{(sv - bv) * 100:+.2f}%"
        print(f"  - {cat:<26} | {bv_str:<12} | {sv_str:<12} | {diff_str:<12}")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
