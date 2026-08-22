#!/usr/bin/env python3
"""
CLI script to run a single task prompt through the harness.

Usage:
  python scripts/run_agent.py --task "Calculate 25 * 37."
  python scripts/run_agent.py --task "What is the capital of France?" --mock
  python scripts/run_agent.py --task "Calculate 100 / 4" --config configs/default.yaml
"""

import sys
import os
import argparse
import yaml

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import HuggingFaceModel, MockModel
from src.tools import ToolRegistry
from src.harness import AgentHarness
from tools.example_tools import CalculatorTool, EchoTool, LookupTool, StringManipulatorTool


def main():
    parser = argparse.ArgumentParser(description="Run Agent Harness on a single task prompt.")
    parser.add_argument("--task", type=str, required=True, help="Task prompt string")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--mock", action="store_true", help="Use MockModel backend for offline execution/testing")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save trajectory JSON")
    args = parser.parse_args()

    # Load configuration
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

    # Initialize model backend
    if args.mock:
        print("[INFO] Using MockModel backend.")
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

    print(f"\nRunning harness on task: '{args.task}'")
    trajectory = harness.run(prompt=args.task)

    # Print output formatted as specified
    trajectory.print_trajectory()

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        trajectory.save(args.output)
        print(f"Trajectory saved to {args.output}")


if __name__ == "__main__":
    main()
