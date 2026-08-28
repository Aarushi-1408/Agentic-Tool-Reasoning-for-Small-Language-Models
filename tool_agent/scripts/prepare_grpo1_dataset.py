#!/usr/bin/env python3
"""
Preprocessing script to create GRPO training prompt artifacts under data/rl/grpo1/
from data/raw/teacher_trajectories_v4_500.jsonl without modifying original dataset.
"""

import os
import json
from pathlib import Path

INPUT_RAW = Path("data/raw/teacher_trajectories_v4_500.jsonl")
OUTPUT_RL_DIR = Path("data/rl/grpo1")
OUTPUT_PROMPTS = OUTPUT_RL_DIR / "train_prompts_v4_500.jsonl"


def prepare_grpo_dataset():
    if not INPUT_RAW.exists():
        raise FileNotFoundError(f"Input file not found at '{INPUT_RAW}'")

    os.makedirs(OUTPUT_RL_DIR, exist_ok=True)

    records = []
    with open(INPUT_RAW, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            
            category = obj.get("category", "general")
            requires_tool = (category != "no_tool")
            steps = obj.get("steps", [])
            allowed_tools = list(set([s.get("tool_name") for s in steps if s.get("tool_name")]))

            rec = {
                "id": obj.get("task_id", obj.get("trajectory_id")),
                "task_id": obj.get("task_id", obj.get("trajectory_id")),
                "prompt": obj.get("prompt", obj.get("task")),
                "category": category,
                "requires_tool": requires_tool,
                "expected_answer": obj.get("final_answer"),
                "expected_behavior": {
                    "allowed_tools": allowed_tools,
                    "expected_steps": len(steps) if steps else 1,
                }
            }
            records.append(rec)

    with open(OUTPUT_PROMPTS, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"==================== GRPO-1 DATASET PREPARATION ====================")
    print(f"Total training prompts extracted : {len(records)}")
    print(f"Output RL prompt dataset         : '{OUTPUT_PROMPTS}'")
    
    cat_counts = {}
    for r in records:
        c = r["category"]
        cat_counts[c] = cat_counts.get(c, 0) + 1
    print("Category Breakdown:")
    for cat, cnt in cat_counts.items():
        print(f"  {cat:<18}: {cnt} ({cnt/len(records)*100:.1f}%)")
    print("====================================================================\n")


if __name__ == "__main__":
    prepare_grpo_dataset()
