#!/usr/bin/env python3
"""
Interactive REPL script to load the model ONCE and execute multiple prompts continuously.

Usage:
  python scripts/interactive.py
  python scripts/interactive.py --mock
  python scripts/interactive.py --config configs/default.yaml
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
    parser = argparse.ArgumentParser(description="Interactive Agent Harness Shell.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--mock", action="store_true", help="Use MockModel backend")
    args = parser.parse_args()

    config = {}
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    model_cfg = config.get("model", {})
    agent_cfg = config.get("agent", {})

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(EchoTool())
    registry.register(LookupTool())
    registry.register(StringManipulatorTool())

    if args.mock:
        print("[INFO] Using MockModel backend.")
        model = MockModel()
    else:
        model_name = model_cfg.get("name", "Qwen/Qwen3-1.7B")
        device = model_cfg.get("device", "auto")
        dtype = model_cfg.get("dtype", "auto")
        max_new_tokens = model_cfg.get("max_new_tokens", 1024)
        print(f"[INFO] Loading HuggingFace model '{model_name}' (loads ONLY ONCE)...")
        model = HuggingFaceModel(
            model_name=model_name,
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
        )

    max_steps = agent_cfg.get("max_steps", 10)
    harness = AgentHarness(model=model, tool_registry=registry, max_steps=max_steps)

    print("\n" + "=" * 60)
    print(" Interactive Harness Session Ready!")
    print(" Type your task prompt and press Enter.")
    print(" Type 'exit' or 'quit' to stop.")
    print("=" * 60 + "\n")

    while True:
        try:
            task = input("Task > ").strip()
            if not task:
                continue
            if task.lower() in ("exit", "quit"):
                print("Exiting interactive session.")
                break

            print(f"\n--- Running Harness on: '{task}' ---")
            trajectory = harness.run(prompt=task)
            trajectory.print_trajectory()
            print("-" * 60 + "\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive session.")
            break


if __name__ == "__main__":
    main()
