#!/usr/bin/env python3
"""
Full-Parameter GRPO-1 Training Script for Qwen/Qwen3-1.7B starting from SFT-v4.

Usage:
  python scripts/train_grpo.py --pilot --mock
  python scripts/train_grpo.py --model outputs/qwen3-1.7b-tool-agent-sft-v4-500 --pilot
  python scripts/train_grpo.py --model outputs/qwen3-1.7b-tool-agent-sft-v4-500 --full
"""

import sys
import os
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.config import GRPOConfig
from src.rl.reward import RewardCalculator, calculate_reward
from src.rl.rollout import RolloutGenerator, RolloutResult
from src.rl.grpo_trainer import GRPOTrainer
from src.model import HuggingFaceModel, MockModel
from src.tools import ToolRegistry
from tools.example_tools import CalculatorTool, EchoTool, LookupTool, StringManipulatorTool

try:
    from tests.test_grpo_reward import test_reward_safety_cases
except ImportError:
    sys.path.insert(0, str(PROJECT_ROOT / "tests"))
    from test_grpo_reward import test_reward_safety_cases


def main():
    parser = argparse.ArgumentParser(description="Full-Parameter GRPO-1 Agent Training")
    parser.add_argument("--model", type=str, default="outputs/qwen3-1.7b-tool-agent-sft-v4-500", help="Starting policy model path")
    parser.add_argument("--prompts", type=str, default="data/rl/grpo1/train_prompts_v4_500.jsonl", help="Path to RL prompts file")
    parser.add_argument("--pilot", action="store_true", help="Run small diagnostic pilot run (20 prompts)")
    parser.add_argument("--full", action="store_true", help="Run full GRPO training")
    parser.add_argument("--mock", action="store_true", help="Use MockModel backend for offline testing")
    parser.add_argument("--output-dir", type=str, default="outputs/grpo1", help="Checkpoint output directory")
    parser.add_argument("--eval-dir", type=str, default="evaluation/grpo1", help="Metrics output directory")
    args = parser.parse_args()

    # 1. ALWAYS RUN REWARD SAFETY TEST FIRST
    print("[INFO] Running mandatory reward safety test...")
    test_reward_safety_cases()

    config = GRPOConfig(
        starting_model_path=args.model,
        train_dataset_path=args.prompts,
        output_dir=args.output_dir,
        eval_output_dir=args.eval_dir,
    )

    # 2. Load Prompts
    prompts_path = Path(args.prompts)
    if not prompts_path.exists():
        print(f"[INFO] Prompts file not found at '{args.prompts}'. Generating from raw trajectories...")
        from scripts.prepare_grpo1_dataset import prepare_grpo_dataset
        prepare_grpo_dataset()

    prompts = []
    with open(prompts_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))

    if args.pilot or (not args.full):
        print(f"[INFO] Executing GRPO-1 PILOT run on {config.pilot_prompts_count} prompts...")
        prompts = prompts[:config.pilot_prompts_count]

    # 3. Setup Environment Tools & Model
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(EchoTool())
    registry.register(LookupTool())
    registry.register(StringManipulatorTool())

    if args.mock:
        print("[INFO] Loading MockModel backend...")
        model = MockModel()
        ref_model = MockModel()
        tokenizer = None
    else:
        print(f"[INFO] Loading HuggingFace model policy '{args.model}'...")
        model = HuggingFaceModel(model_name=args.model, use_system_prompt=True)
        ref_model = HuggingFaceModel(model_name=args.model, use_system_prompt=True)
        tokenizer = model.tokenizer

    generator = RolloutGenerator(model=model, tool_registry=registry, max_steps=config.max_env_steps)
    trainer = GRPOTrainer(config=config, model=model if not args.mock else None, ref_model=ref_model if not args.mock else None, tokenizer=tokenizer)

    os.makedirs(args.eval_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_log_path = os.path.join(args.eval_dir, "training_metrics.jsonl")

    # 4. GRPO Training Loop
    print("\n==================== STARTING GRPO-1 TRAINING LOOP ====================")
    step = 0
    zero_variance_count = 0

    with open(metrics_log_path, "w", encoding="utf-8") as f_log:
        for epoch in range(config.epochs if args.full else 1):
            print(f"\n--- Epoch {epoch + 1} ---")
            for i, p_meta in enumerate(prompts):
                step += 1
                
                # Rollout G candidates
                group_result = generator.generate_group(task_metadata=p_meta, group_size=config.group_size)
                
                if group_result.reward_variance < 1e-6:
                    zero_variance_count += 1

                # Optimization step
                metrics = trainer.train_step([group_result])
                
                # Extract detailed trajectory metrics
                all_rewards = [c.reward_result.total_reward for c in group_result.candidates]
                correct_count = sum(1 for c in group_result.candidates if c.reward_result.is_correct)
                
                log_entry = {
                    "step": step,
                    "task_id": group_result.task_id,
                    "category": group_result.category,
                    "mean_reward": round(group_result.mean_reward, 4),
                    "std_reward": round(group_result.std_reward, 4),
                    "reward_variance": round(group_result.reward_variance, 4),
                    "correctness_rate": round(correct_count / len(group_result.candidates), 4),
                    "total_loss": metrics.get("total_loss", 0.0),
                }
                
                f_log.write(json.dumps(log_entry) + "\n")
                f_log.flush()

                if step % 5 == 0 or step == len(prompts):
                    print(f"Step {step:03d} | Task: {group_result.task_id:<15} ({group_result.category:<12}) | Mean R: {group_result.mean_reward:+.2f} | Std R: {group_result.std_reward:.2f} | Loss: {metrics.get('total_loss', 0.0):.4f}")

                if step % config.save_steps == 0:
                    ckpt_path = trainer.save_checkpoint(args.output_dir, step)
                    print(f"[INFO] Saved checkpoint at '{ckpt_path}'")

    ckpt_final = trainer.save_checkpoint(args.output_dir, step)
    print(f"\n[INFO] GRPO-1 Training completed. Final checkpoint saved to '{ckpt_final}'")
    
    if zero_variance_count > (step * 0.8):
        print("\n[WARNING] GRPO reward signal may be insufficient! Over 80% of groups had identical rewards.")
    else:
        print(f"\n[SUCCESS] GRPO rollouts produced active group reward variance across {step - zero_variance_count}/{step} steps.")


if __name__ == "__main__":
    main()
