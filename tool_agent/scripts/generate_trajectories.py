#!/usr/bin/env python3
"""
CLI script to generate synthetic tool-use trajectories using a Teacher Model.

Usage:
  python scripts/generate_trajectories.py --backend mock \
      --num-no-tool 20 --num-single-tool 25 --num-two-step 25 --num-multi-step 20 --num-error-recovery 10 \
      --seed 42 --output data/raw/teacher_trajectories_v2_100.jsonl
"""

import sys
import os
import time
import json
import argparse
import yaml
from typing import List, Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tools import ToolRegistry
from src.harness import AgentHarness
from src.generation.teacher import create_teacher, TeacherInterface
from src.generation.validator import TrajectoryValidator
from src.generation.task_generator import TaskGenerator
from tools.example_tools import CalculatorTool, EchoTool, LookupTool, StringManipulatorTool


def load_tasks(filepath: str) -> List[Dict[str, Any]]:
    """
    Load tasks from a JSON or JSONL file.
    """
    tasks = []
    resolved_path = filepath
    if not os.path.exists(resolved_path):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        alt_path = os.path.join(project_root, filepath)
        if os.path.exists(alt_path):
            resolved_path = alt_path
        else:
            raise FileNotFoundError(f"Tasks file not found at: '{filepath}' (also checked: '{alt_path}')")

    with open(resolved_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            return tasks

        if resolved_path.endswith(".jsonl") or not content.startswith("["):
            for line in content.splitlines():
                line = line.strip()
                if line:
                    tasks.append(json.loads(line))
        else:
            tasks = json.loads(content)

    return tasks


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic trajectories using Teacher Model.")
    parser.add_argument("--teacher-model", type=str, default=None, help="Teacher model name or path (e.g. Qwen/Qwen3-8B)")
    parser.add_argument("--backend", type=str, default=None, help="Teacher backend: transformers, api, or mock")
    parser.add_argument("--config", type=str, default="configs/teacher.yaml", help="Path to teacher config YAML")
    parser.add_argument("--tasks", type=str, default=None, help="Optional existing tasks file (overrides generation)")
    parser.add_argument("--tasks-output", type=str, default="data/tasks/train_tasks_v2_100.jsonl", help="Output path for generated tasks JSONL")
    parser.add_argument("--output", type=str, default="data/raw/teacher_trajectories_v2_100.jsonl", help="Output path for teacher trajectories JSONL")
    parser.add_argument("--rejected-output", type=str, default="data/raw/rejected_v2_100.jsonl", help="Output path for rejected trajectories JSONL")
    parser.add_argument("--metrics-output", type=str, default="data/raw/generation_stats_v2_100.json", help="Output path for generation metrics JSON")
    parser.add_argument("--max-steps", type=int, default=None, help="Max execution steps per task")

    # Category distribution arguments
    parser.add_argument("--num-no-tool", type=int, default=20, help="Number of no-tool tasks")
    parser.add_argument("--num-single-tool", type=int, default=25, help="Number of single-tool tasks")
    parser.add_argument("--num-two-step", type=int, default=25, help="Number of two-step tasks")
    parser.add_argument("--num-multi-step", type=int, default=20, help="Number of multi-step tasks")
    parser.add_argument("--num-error-recovery", type=int, default=10, help="Number of error recovery tasks")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for task generation")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output files")

    args = parser.parse_args()

    # 0. Safety Checks - Do NOT overwrite existing output files unless --overwrite is set
    if os.path.exists(args.output) and not args.overwrite:
        print(f"[ERROR] Output file '{args.output}' already exists. Use --overwrite to allow overwriting.")
        sys.exit(1)

    # Load YAML configuration if exists
    config = {}
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    teacher_cfg = config.get("teacher", {})
    agent_cfg = config.get("agent", {})

    if args.teacher_model:
        teacher_cfg["model_name"] = args.teacher_model
    if args.backend:
        teacher_cfg["backend"] = args.backend

    if "backend" not in teacher_cfg:
        if args.teacher_model and "mock" in args.teacher_model.lower():
            teacher_cfg["backend"] = "mock"
        else:
            teacher_cfg["backend"] = "transformers"

    max_steps = args.max_steps if args.max_steps is not None else agent_cfg.get("max_steps", 8)

    # 1. Load or Generate tasks
    print(f"\n==================== GENERATION INITIALIZATION ====================")
    if args.tasks:
        print(f"Loading custom training tasks from: '{args.tasks}'")
        tasks = load_tasks(args.tasks)
    else:
        print(f"Generating tasks with category distribution (Seed: {args.seed}):")
        print(f"  No-tool       : {args.num_no_tool}")
        print(f"  Single-tool   : {args.num_single_tool}")
        print(f"  Two-step      : {args.num_two_step}")
        print(f"  Multi-step    : {args.num_multi_step}")
        print(f"  Error-recovery: {args.num_error_recovery}")

        generator = TaskGenerator(seed=args.seed)
        tasks = generator.generate_tasks(
            num_no_tool=args.num_no_tool,
            num_single_tool=args.num_single_tool,
            num_two_step=args.num_two_step,
            num_multi_step=args.num_multi_step,
            num_error_recovery=args.num_error_recovery,
            id_prefix="train_v2_",
        )

        # Save tasks to task output file
        if os.path.exists(args.tasks_output) and not args.overwrite:
            print(f"[ERROR] Tasks output file '{args.tasks_output}' already exists. Use --overwrite to allow overwriting.")
            sys.exit(1)

        os.makedirs(os.path.dirname(os.path.abspath(args.tasks_output)), exist_ok=True)
        with open(args.tasks_output, "w", encoding="utf-8") as t_out:
            for task in tasks:
                t_out.write(json.dumps(task) + "\n")
        print(f"Generated tasks saved to: '{args.tasks_output}'")

    print(f"Total tasks to execute: {len(tasks)}")

    # 2. Instantiate tools ONCE
    print(f"Initializing registered tools ONCE...")
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(EchoTool())
    registry.register(LookupTool())
    registry.register(StringManipulatorTool())
    print(f"Registered tools: {', '.join(registry.list_tool_names())}")

    # 3. Instantiate teacher model ONCE
    config["teacher"] = teacher_cfg
    print(f"Instantiating Teacher Model backend ONCE...")
    teacher: TeacherInterface = create_teacher(config)
    teacher_info = teacher.get_info()
    print(f"Teacher Model Info       : {json.dumps(teacher_info)}")
    print(f"Maximum Steps per Task   : {max_steps}")
    print("===================================================================\n")

    # 4. Prepare harness and output files
    harness = AgentHarness(model=teacher, tool_registry=registry, max_steps=max_steps)
    validator = TrajectoryValidator(registered_tools=set(registry.list_tool_names()))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    if args.rejected_output:
        os.makedirs(os.path.dirname(os.path.abspath(args.rejected_output)), exist_ok=True)

    total_tasks = len(tasks)
    valid_trajectories = []
    rejected_trajectories = []
    category_counts = {
        "no_tool": 0,
        "single_tool": 0,
        "two_step": 0,
        "multi_step": 0,
        "error_recovery": 0,
    }
    rejection_reasons: Dict[str, int] = {}
    total_steps = 0
    total_tool_calls = 0

    start_time = time.time()

    for idx, task_info in enumerate(tasks, 1):
        raw_task_id = task_info.get("task_id", f"train_v2_{idx:03d}")
        trajectory_id = f"traj_v2_{raw_task_id}"
        prompt = task_info.get("prompt", task_info.get("task", ""))
        category = task_info.get("category", "general")

        print(f"[{idx}/{total_tasks}] Running task '{raw_task_id}' ({category}): \"{prompt[:50]}...\"")

        try:
            trajectory = harness.run(prompt=prompt, task_id=raw_task_id)
            trajectory.teacher_model = teacher_info.get("teacher_model", "unknown")
            trajectory_dict = trajectory.to_dict()
            trajectory_dict["trajectory_id"] = trajectory_id
            trajectory_dict["category"] = category

            # Validate trajectory
            val_res = validator.validate(trajectory_dict, tool_registry=registry)
            trajectory_dict["valid"] = val_res["valid"]
            trajectory_dict["validation_errors"] = val_res["validation_errors"]

            steps_count = len(trajectory.steps)
            tool_calls_count = sum(1 for s in trajectory.steps if s.action_type == "tool_call")

            if val_res["valid"]:
                valid_trajectories.append(trajectory_dict)
                category_counts[category] = category_counts.get(category, 0) + 1
                total_steps += steps_count
                total_tool_calls += tool_calls_count
                print(f"    -> [VALID] Steps: {steps_count} | Tool calls: {tool_calls_count}")
            else:
                reason = val_res["validation_errors"][0] if val_res["validation_errors"] else "invalid_structure"
                rejected_trajectories.append({
                    "trajectory_id": trajectory_id,
                    "task_id": raw_task_id,
                    "reason": reason,
                    "validation_errors": val_res["validation_errors"],
                })
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                print(f"    -> [REJECTED] Reason: {reason}")

        except Exception as e:
            reason = f"exception: {str(e)}"
            rejected_trajectories.append({
                "trajectory_id": trajectory_id,
                "task_id": raw_task_id,
                "reason": reason,
            })
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            print(f"    -> [ERROR] Exception: {str(e)}")

    end_time = time.time()
    elapsed_seconds = end_time - start_time
    valid_count = len(valid_trajectories)
    rejected_count = len(rejected_trajectories)
    avg_steps = (total_steps / valid_count) if valid_count > 0 else 0.0
    avg_tool_calls = (total_tool_calls / valid_count) if valid_count > 0 else 0.0

    # Write output JSONL files
    with open(args.output, "w", encoding="utf-8") as out_f:
        for traj in valid_trajectories:
            out_f.write(json.dumps(traj) + "\n")

    if args.rejected_output:
        with open(args.rejected_output, "w", encoding="utf-8") as rej_f:
            for rej in rejected_trajectories:
                rej_f.write(json.dumps(rej) + "\n")

    metrics = {
        "requested": total_tasks,
        "generated": total_tasks,
        "valid": valid_count,
        "rejected": rejected_count,
        "categories": category_counts,
        "average_steps": round(avg_steps, 2),
        "average_tool_calls": round(avg_tool_calls, 2),
        "rejection_reasons": rejection_reasons,
        "generation_time": round(elapsed_seconds, 2),
        "seed": args.seed,
        "teacher_model": teacher_info.get("teacher_model", "unknown"),
        "backend": teacher_info.get("backend", "unknown"),
    }

    if args.metrics_output:
        os.makedirs(os.path.dirname(os.path.abspath(args.metrics_output)), exist_ok=True)
        with open(args.metrics_output, "w", encoding="utf-8") as m_f:
            json.dump(metrics, m_f, indent=2)

    print(f"\n==================== GENERATION STATS ====================")
    print(json.dumps(metrics, indent=2))
    print(f"Valid trajectories saved to : '{args.output}'")
    print(f"Rejected trajectories saved to: '{args.rejected_output}'")
    print(f"Generation stats saved to    : '{args.metrics_output}'")
    print("=========================================================\n")


if __name__ == "__main__":
    main()
