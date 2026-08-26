#!/usr/bin/env python3
"""
Smoke test script to evaluate an SFT model checkpoint or base model
on 3 representative tasks through the harness before running full benchmark evaluation.

Usage:
  python scripts/smoke_test.py --mock
  python scripts/smoke_test.py --model-path outputs/sft_qwen3_1.7b_v2_100/final/
  python scripts/smoke_test.py --model-path Qwen/Qwen3-1.7B
  python scripts/smoke_test.py --model-path models/qwen3-1.7b-tool-agent-sft-v2/
"""

import sys
import os
import argparse
import json
import yaml
from pathlib import Path
from typing import Any

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import HuggingFaceModel, MockModel
from src.tools import ToolRegistry
from src.harness import AgentHarness
from src.parser import ModelOutputParser
from tools.example_tools import CalculatorTool, EchoTool, LookupTool, StringManipulatorTool


SMOKE_TASKS = [
    {
        "id": "smoke_no_tool_01",
        "category": "no_tool",
        "prompt": "What does CPU stand for in computer hardware?",
    },
    {
        "id": "smoke_single_tool_02",
        "category": "single_tool",
        "prompt": "Calculate 25 * 37.",
    },
    {
        "id": "smoke_multi_step_03",
        "category": "multi_step",
        "prompt": "Fetch 'apple_price' and 'orange_price' from database, then calculate apple_price + orange_price + 5.",
    },
]


def run_smoke_test(model: Any, registry: ToolRegistry) -> bool:
    harness = AgentHarness(model=model, tool_registry=registry, max_steps=10)
    parser = ModelOutputParser()

    print("\n" + "=" * 70)
    print("                      SMOKE TEST RUN")
    print("=" * 70)

    all_passed = True

    for item in SMOKE_TASKS:
        task_id = item["id"]
        category = item["category"]
        prompt = item["prompt"]

        print(f"\n----------------------------------------------------------------------")
        print(f"Task ID     : {task_id} [{category.upper()}]")
        print(f"Prompt      : '{prompt}'")
        print(f"----------------------------------------------------------------------")

        trajectory = harness.run(prompt)

        print("\nTrajectory Steps:")
        for idx, step in enumerate(trajectory.steps):
            print(f"  Step {idx} [{step.action_type}]:")
            print(f"    Raw Output  : {step.model_output}")
            parsed = parser.parse(step.model_output)
            print(f"    Parsed      : action_type='{parsed.action_type}'")
            if step.action_type == "tool_call":
                print(f"    Tool Call   : {step.tool_name}({step.arguments})")
                print(f"    Tool Result : {step.tool_result}")
            elif step.action_type == "final":
                print(f"    Final Answer: {trajectory.final_answer}")
            elif step.action_type == "parse_error":
                print(f"    Parse Error : {step.error}")

        print(f"\n  Final Answer      : {trajectory.final_answer}")
        print(f"  Total Steps       : {len(trajectory.steps)}")
        print(f"  Termination Reason: {trajectory.termination_reason}")

        if trajectory.termination_reason == "final_answer":
            print(f"  Task Verification : PASSED")
        else:
            print(f"  Task Verification : FAILED (Did not reach final answer)")
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("SUMMARY: ALL SMOKE TEST TASKS PASSED SUCCESSFULLY!")
    else:
        print("SUMMARY: SMOKE TEST COMPLETED WITH FAILURES")
    print("=" * 70 + "\n")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Run 3-task smoke test through AgentHarness.")
    parser.add_argument("--model-path", type=str, default=None, help="HuggingFace model ID or local directory path")
    parser.add_argument("--model", type=str, default=None, help="Alias for --model-path")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--mock", action="store_true", help="Use MockModel backend for fast test verification")

    args = parser.parse_args()

    # Register tools
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(EchoTool())
    registry.register(LookupTool())
    registry.register(StringManipulatorTool())

    # Load config if present
    config = {}
    config_path = PROJECT_ROOT / args.config
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    model_cfg = config.get("model", {})

    if args.mock:
        print("[INFO] Running smoke test using MockModel backend.")
        model = MockModel()
    else:
        target_model = args.model_path or args.model or model_cfg.get("name", "Qwen/Qwen3-1.7B")
        device = model_cfg.get("device", "auto")
        dtype = model_cfg.get("dtype", "auto")
        max_new_tokens = model_cfg.get("max_new_tokens", 1024)

        print(f"[INFO] Initializing HuggingFaceModel with path: '{target_model}'...")
        model = HuggingFaceModel(
            model_path=target_model,
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
        )

    passed = run_smoke_test(model, registry)
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
