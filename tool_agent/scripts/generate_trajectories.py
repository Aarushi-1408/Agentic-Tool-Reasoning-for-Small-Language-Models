#!/usr/bin/env python3
"""
CLI script to generate synthetic tool-use trajectories using a Teacher Model.

Usage:
  python scripts/generate_trajectories.py --backend mock --tasks data/tasks/train_tasks.jsonl --output data/raw/teacher_trajectories.jsonl
  python scripts/generate_trajectories.py --teacher-model Qwen/Qwen3-8B --config configs/teacher.yaml
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
from src.generation.teacher import create_teacher, TeacherInterface, MockTeacher
from src.generation.validator import TrajectoryValidator
from tools.example_tools import CalculatorTool, EchoTool, LookupTool, StringManipulatorTool


def load_tasks(filepath: str) -> List[Dict[str, Any]]:
    """
    Load tasks from a JSON or JSONL file.
    """
    tasks = []
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Tasks file not found at: '{filepath}'")

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            return tasks

        if filepath.endswith(".jsonl") or not content.startswith("["):
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
    parser.add_argument("--tasks", type=str, default="data/tasks/train_tasks.jsonl", help="Path to training tasks JSONL")
    parser.add_argument("--output", type=str, default="data/raw/teacher_trajectories.jsonl", help="Output path for teacher trajectories JSONL")
    parser.add_argument("--metrics-output", type=str, default="data/raw/generation_metrics.json", help="Output path for generation metrics JSON")
    parser.add_argument("--max-steps", type=int, default=None, help="Max execution steps per task")
    parser.add_argument("--num-tasks", type=int, default=None, help="Number of tasks to generate")
    args = parser.parse_args()

    # Load YAML configuration if exists
    config = {}
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    teacher_cfg = config.get("teacher", {})
    agent_cfg = config.get("agent", {})

    # Override config with CLI arguments if specified
    if args.teacher_model:
        teacher_cfg["model_name"] = args.teacher_model
    if args.backend:
        teacher_cfg["backend"] = args.backend

    # Default backend fallback if not specified
    if "backend" not in teacher_cfg:
        if args.teacher_model and "mock" in args.teacher_model.lower():
            teacher_cfg["backend"] = "mock"
        else:
            teacher_cfg["backend"] = "transformers"

    max_steps = args.max_steps if args.max_steps is not None else agent_cfg.get("max_steps", 8)

    # 1. Load tasks
    print(f"\n==================== GENERATION INITIALIZATION ====================")
    print(f"Loading training tasks from: '{args.tasks}'")
    tasks = load_tasks(args.tasks)
    if args.num_tasks and args.num_tasks > 0:
        tasks = tasks[:args.num_tasks]
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

    total_tasks = len(tasks)
    successful_count = 0
    failed_count = 0
    no_tool_count = 0
    single_tool_count = 0
    multi_step_count = 0
    valid_count = 0
    invalid_count = 0
    total_steps = 0
    total_tool_calls = 0
    tool_failures = 0
    max_steps_exceeded_count = 0

    start_time = time.time()

    # Clear/open output file
    with open(args.output, "w", encoding="utf-8") as out_f:
        for idx, task_info in enumerate(tasks, 1):
            task_id = task_info.get("task_id", f"train_{idx:03d}")
            prompt = task_info.get("prompt", task_info.get("task", ""))

            print(f"[{idx}/{total_tasks}] Running task '{task_id}': \"{prompt[:60]}...\"")
            
            try:
                trajectory = harness.run(prompt=prompt, task_id=task_id)
                trajectory.teacher_model = teacher_info.get("teacher_model", "unknown")
                trajectory_dict = trajectory.to_dict()

                # Add extra fields if task has category metadata
                if "category" in task_info:
                    trajectory_dict["category"] = task_info["category"]

                # Validate trajectory
                val_res = validator.validate(trajectory_dict, tool_registry=registry)
                trajectory_dict["valid"] = val_res["valid"]
                trajectory_dict["validation_errors"] = val_res["validation_errors"]

                if val_res["valid"]:
                    valid_count += 1
                else:
                    invalid_count += 1

                # Track metrics
                steps_count = len(trajectory.steps)
                total_steps += steps_count
                
                tool_calls_in_traj = sum(1 for s in trajectory.steps if s.action_type == "tool_call")
                total_tool_calls += tool_calls_in_traj
                tool_failures += sum(1 for s in trajectory.steps if s.error is not None)

                if trajectory.termination_reason == "final_answer":
                    successful_count += 1
                else:
                    failed_count += 1

                if trajectory.termination_reason == "max_steps_exceeded":
                    max_steps_exceeded_count += 1

                if tool_calls_in_traj == 0:
                    no_tool_count += 1
                elif tool_calls_in_traj == 1:
                    single_tool_count += 1
                else:
                    multi_step_count += 1

                # Save raw trajectory record
                out_f.write(json.dumps(trajectory_dict) + "\n")
                out_f.flush()

                print(f"    -> Status: {trajectory.termination_reason} | Steps: {steps_count} | Tool calls: {tool_calls_in_traj} | Valid: {val_res['valid']}")

            except Exception as e:
                failed_count += 1
                print(f"    [ERROR] Task '{task_id}' failed with exception: {str(e)}")
                continue

    end_time = time.time()
    elapsed_seconds = end_time - start_time
    avg_time_per_task = (elapsed_seconds / total_tasks) if total_tasks > 0 else 0.0
    avg_steps = (total_steps / total_tasks) if total_tasks > 0 else 0.0

    metrics = {
        "teacher_model": teacher_info.get("teacher_model", "unknown"),
        "backend": teacher_info.get("backend", "unknown"),
        "total_tasks": total_tasks,
        "successful_trajectories": successful_count,
        "failed_trajectories": failed_count,
        "valid_trajectories": valid_count,
        "invalid_trajectories": invalid_count,
        "no_tool_trajectories": no_tool_count,
        "single_tool_trajectories": single_tool_count,
        "multi_step_trajectories": multi_step_count,
        "average_steps_per_task": round(avg_steps, 2),
        "total_tool_calls": total_tool_calls,
        "tool_call_failures": tool_failures,
        "max_steps_exceeded_terminations": max_steps_exceeded_count,
        "total_generation_time_seconds": round(elapsed_seconds, 2),
        "average_time_per_task_seconds": round(avg_time_per_task, 2),
    }

    # Save metrics JSON
    if args.metrics_output:
        os.makedirs(os.path.dirname(os.path.abspath(args.metrics_output)), exist_ok=True)
        with open(args.metrics_output, "w", encoding="utf-8") as m_f:
            json.dump(metrics, m_f, indent=2)

    # Print summary
    print(f"\n==================== GENERATION SUMMARY METRICS ====================")
    print(json.dumps(metrics, indent=2))
    print(f"Raw trajectories saved to : '{args.output}'")
    print(f"Metrics report saved to   : '{args.metrics_output}'")
    print("===================================================================\n")


if __name__ == "__main__":
    main()
