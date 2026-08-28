#!/usr/bin/env python3
"""
Evaluation CLI script for GRPO-1.
Evaluates baseline SFT-v4 model and GRPO-1 checkpoint against frozen benchmark tasks/eval_v1.json.
"""

import sys
import os
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import HuggingFaceModel, MockModel
from src.tools import ToolRegistry
from src.harness import AgentHarness
from src.evaluator import Evaluator
from tools.example_tools import CalculatorTool, EchoTool, LookupTool, StringManipulatorTool


def run_eval(model_path: str, tasks: list, registry: ToolRegistry, use_mock: bool = False):
    if use_mock:
        model = MockModel()
    else:
        model = HuggingFaceModel(model_name=model_path, use_system_prompt=True)
    
    harness = AgentHarness(model=model, tool_registry=registry, max_steps=10)
    evaluator = Evaluator()
    summary = evaluator.evaluate_batch(tasks, harness)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate GRPO-1 Checkpoint against Baseline SFT-v4")
    parser.add_argument("--tasks", type=str, default="tasks/eval_v1.json", help="Path to frozen benchmark JSON")
    parser.add_argument("--baseline", type=str, default="outputs/qwen3-1.7b-tool-agent-sft-v4-500", help="Baseline SFT-v4 checkpoint path")
    parser.add_argument("--grpo-model", type=str, default="outputs/grpo1/checkpoint-0020", help="GRPO-1 checkpoint path")
    parser.add_argument("--output-dir", type=str, default="evaluation/grpo1", help="Output directory for metrics")
    parser.add_argument("--mock", action="store_true", help="Use MockModel backend for fast testing")
    args = parser.parse_args()

    tasks_path = Path(args.tasks)
    if not tasks_path.exists():
        print(f"[ERROR] Tasks benchmark not found at '{args.tasks}'")
        sys.exit(1)

    with open(tasks_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(EchoTool())
    registry.register(LookupTool())
    registry.register(StringManipulatorTool())

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n==================== EVALUATING BASELINE (SFT-v4) ====================")
    baseline_summary = run_eval(args.baseline, tasks, registry, use_mock=args.mock)
    baseline_summary.print_summary()
    
    base_metrics_path = os.path.join(args.output_dir, "baseline_metrics.json")
    baseline_summary.save_metrics(base_metrics_path)

    print(f"\n==================== EVALUATING GRPO-1 CHECKPOINT ====================")
    grpo_summary = run_eval(args.grpo_model, tasks, registry, use_mock=args.mock)
    grpo_summary.print_summary()

    grpo_metrics_path = os.path.join(args.output_dir, "grpo1_metrics.json")
    grpo_traj_path = os.path.join(args.output_dir, "grpo1_trajectories.jsonl")

    grpo_summary.save_metrics(grpo_metrics_path)
    grpo_summary.save_trajectories_jsonl(grpo_traj_path)

    print(f"\n==================== SFT-v4 vs GRPO-1 BENCHMARK COMPARISON ====================")
    print(f"{'Metric':<32} | {'SFT-v4 Baseline':<20} | {'GRPO-1 Model':<20} | {'Delta':<10}")
    print("-" * 90)

    metrics_list = [
        ("Overall Accuracy", "accuracy"),
        ("No-Tool Accuracy", "no_tool_accuracy"),
        ("Tool-Required Accuracy", "tool_required_accuracy"),
        ("Tool Selection Accuracy", "tool_selection_accuracy"),
        ("Unnecessary Tool-Use Rate", "unnecessary_tool_use_rate"),
        ("Multi-Step Accuracy", "multi_step_accuracy"),
        ("Valid Tool-Call Rate", "valid_tool_call_rate"),
        ("Average Steps / Task", "avg_steps_per_task"),
    ]

    for label, key in metrics_list:
        v_base = baseline_summary.to_dict()["summary"].get(key, 0.0)
        v_grpo = grpo_summary.to_dict()["summary"].get(key, 0.0)
        delta = v_grpo - v_base

        if key == "avg_steps_per_task":
            print(f"{label:<32} | {v_base:<20.2f} | {v_grpo:<20.2f} | {delta:+.2f}")
        else:
            print(f"{label:<32} | {v_base*100:<19.1f}% | {v_grpo*100:<19.1f}% | {delta*100:+.1f}%")


if __name__ == "__main__":
    main()
