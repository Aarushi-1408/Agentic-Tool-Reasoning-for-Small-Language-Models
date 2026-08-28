#!/usr/bin/env python3
"""
CLI script to execute System Prompt Ablation Experiment across 3 prompt conditions:
- Condition A: Current / Control Prompt (evaluation/ablation_v4/current_prompt/)
- Condition B: Basic Tool-Protocol Prompt (evaluation/ablation_v4/basic_prompt/)
- Condition C: Explicit Decision-Boundary Prompt (evaluation/ablation_v4/decision_boundary_prompt/)

Usage:
  python scripts/run_v4_ablation_experiment.py --model outputs/qwen3-1.7b-tool-agent-sft-v3-500
  python scripts/run_v4_ablation_experiment.py --mock
"""

import sys
import os
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import HuggingFaceModel, MockModel, DEFAULT_SYSTEM_PROMPT
from src.tools import ToolRegistry
from src.harness import AgentHarness
from src.evaluator import Evaluator
from tools.example_tools import CalculatorTool, EchoTool, LookupTool, StringManipulatorTool

PROMPT_CONDITION_A = DEFAULT_SYSTEM_PROMPT

PROMPT_CONDITION_B = """You are a tool-use agent.

Decide whether the user's task requires a tool.

If a tool is required, output ONLY:
{"type":"tool_call","name":"<tool_name>","arguments":{...}}

After receiving TOOL_RESULT, use the result to continue solving the task.

When you have enough information to answer, output ONLY:
{"type":"final","content":"<answer>"}

Do not repeat an identical tool call unnecessarily."""

PROMPT_CONDITION_C = """You are an autonomous tool-use agent.

Before responding, decide whether the task actually requires a tool.

USE A TOOL when:
- the requested operation requires a provided tool
- the required information must be obtained from a tool
- a calculation should be performed using the calculator
- a string operation requires the string manipulation tool
- another provided tool is genuinely necessary to complete the task

DO NOT USE A TOOL when:
- you can answer the question correctly using your existing knowledge
- the task is a simple factual question that does not require external information
- using a tool would not provide information necessary to answer the user
- the requested answer can be produced directly

If no tool is required, immediately output ONLY:
{"type":"final","content":"<answer>"}

If a tool is required, output ONLY:
{"type":"tool_call","name":"<tool_name>","arguments":{...}}

After receiving TOOL_RESULT:
1. Examine the result.
2. Decide whether another tool is genuinely necessary.
3. If all required information has been obtained, output the final answer.
4. Do not repeat an identical tool call.
5. Do not call additional tools unnecessarily.

After all required tool operations are complete, output ONLY:
{"type":"final","content":"<answer>"}"""


def run_condition(
    condition_name: str,
    output_dir: str,
    system_prompt_text: str,
    model_name: str,
    tasks: list,
    registry: ToolRegistry,
    use_mock: bool = False,
):
    print(f"\n==================== RUNNING ABLATION: {condition_name.upper()} ====================")
    print(f"Output Directory : '{output_dir}'")
    print(f"System Prompt    :\n{system_prompt_text}\n")

    if use_mock:
        print("[INFO] Running with MockModel backend...")
        model = MockModel()
    else:
        print(f"[INFO] Loading HuggingFace model '{model_name}'...")
        model = HuggingFaceModel(
            model_name=model_name,
            use_system_prompt=True,
            custom_system_prompt=system_prompt_text,
        )

    harness = AgentHarness(model=model, tool_registry=registry, max_steps=10)
    evaluator = Evaluator()
    summary = evaluator.evaluate_batch(tasks, harness)
    summary.print_summary()

    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "metrics.json")
    traj_path = os.path.join(output_dir, "trajectories.jsonl")
    prompt_path = os.path.join(output_dir, "system_prompt.txt")

    summary.save_metrics(metrics_path)
    summary.save_trajectories_jsonl(traj_path)
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(system_prompt_text)

    print(f"[INFO] Condition '{condition_name}' saved to '{output_dir}'")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run 3-condition System Prompt Ablation Experiment.")
    parser.add_argument("--tasks", type=str, default="tasks/eval_v1.json", help="Tasks benchmark JSON file")
    parser.add_argument("--model", type=str, default="outputs/qwen3-1.7b-tool-agent-sft-v3-500", help="Model name or checkpoint directory")
    parser.add_argument("--mock", action="store_true", help="Use MockModel for testing")
    parser.add_argument("--base-output-dir", type=str, default="evaluation/ablation_v4", help="Base directory for ablation outputs")
    args = parser.parse_args()

    tasks_path = Path(args.tasks)
    if not tasks_path.exists():
        print(f"[ERROR] Tasks file not found at '{args.tasks}'")
        sys.exit(1)

    with open(tasks_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(EchoTool())
    registry.register(LookupTool())
    registry.register(StringManipulatorTool())

    conditions = [
        ("current_prompt", os.path.join(args.base_output_dir, "current_prompt"), PROMPT_CONDITION_A),
        ("basic_prompt", os.path.join(args.base_output_dir, "basic_prompt"), PROMPT_CONDITION_B),
        ("decision_boundary_prompt", os.path.join(args.base_output_dir, "decision_boundary_prompt"), PROMPT_CONDITION_C),
    ]

    for name, out_dir, prompt in conditions:
        run_condition(
            condition_name=name,
            output_dir=out_dir,
            system_prompt_text=prompt,
            model_name=args.model,
            tasks=tasks,
            registry=registry,
            use_mock=args.mock,
        )

    print("\n==================== ALL 3 ABLATION CONDITIONS COMPLETED ====================")


if __name__ == "__main__":
    main()
