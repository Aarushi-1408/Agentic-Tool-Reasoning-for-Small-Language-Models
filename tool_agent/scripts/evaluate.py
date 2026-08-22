#!/usr/bin/env python3
"""
CLI script to evaluate a benchmark suite of tasks using the harness.

Usage:
  python scripts/evaluate.py --tasks tasks/example_tasks.json --mock
  python scripts/evaluate.py --tasks tasks/example_tasks.json --config configs/default.yaml
"""

import sys
import os
import json
import argparse
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import HuggingFaceModel, MockModel
from src.tools import ToolRegistry
from src.harness import AgentHarness
from src.evaluator import Evaluator
from tools.example_tools import CalculatorTool, EchoTool, LookupTool, StringManipulatorTool


def main():
    parser = argparse.ArgumentParser(description="Evaluate Agent Harness on a JSON dataset of tasks.")
    parser.add_argument("--tasks", type=str, default="tasks/eval_v1.json", help="Path to tasks JSON file")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--mock", action="store_true", help="Use MockModel backend for fast offline evaluation")
    parser.add_argument("--output", type=str, default=None, help="Optional output path for evaluation report JSON")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory path to save metrics.json and trajectories.jsonl")
    args = parser.parse_args()

    if not os.path.exists(args.tasks):
        print(f"[ERROR] Tasks file not found at '{args.tasks}'")
        sys.exit(1)

    with open(args.tasks, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    # Load config
    config = {}
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    model_cfg = config.get("model", {})
    agent_cfg = config.get("agent", {})

    # Register tools
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(EchoTool())
    registry.register(LookupTool())
    registry.register(StringManipulatorTool())

    # Initialize model
    if args.mock:
        print("[INFO] Evaluating with MockModel backend.")
        model = MockModel()
    else:
        model_name = model_cfg.get("name", "Qwen/Qwen3-1.7B")
        device = model_cfg.get("device", "auto")
        dtype = model_cfg.get("dtype", "auto")
        max_new_tokens = model_cfg.get("max_new_tokens", 1024)
        print(f"[INFO] Loading HuggingFace model '{model_name}'...")
        model = HuggingFaceModel(
            model_name=model_name,
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
        )

    max_steps = agent_cfg.get("max_steps", 10)
    harness = AgentHarness(model=model, tool_registry=registry, max_steps=max_steps)

    evaluator = Evaluator()
    summary = evaluator.evaluate_batch(tasks, harness)

    summary.print_summary()

    if args.output_dir:
        metrics_path = os.path.join(args.output_dir, "metrics.json")
        traj_path = os.path.join(args.output_dir, "trajectories.jsonl")
        summary.save_metrics(metrics_path)
        summary.save_trajectories_jsonl(traj_path)
        print(f"[INFO] Summary metrics saved to '{metrics_path}'")
        print(f"[INFO] Trajectories saved to '{traj_path}'")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2)
        print(f"[INFO] Evaluation report saved to '{args.output}'")


if __name__ == "__main__":
    main()
