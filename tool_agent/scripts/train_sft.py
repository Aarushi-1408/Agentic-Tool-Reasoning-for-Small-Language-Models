#!/usr/bin/env python3
"""
Full-Parameter SFT Training Script for Qwen/Qwen3-1.7B on train_v2_100.jsonl.

IMPORTANT: PEFT, LoRA, QLoRA, adapters, and parameter-efficient fine-tuning methods
are strictly DISABLED. All model parameters remain 100% trainable.

Usage:
  python scripts/train_sft.py --model Qwen/Qwen3-1.7B --dataset data/sft/train_v2_100.jsonl
  python scripts/train_sft.py --smoke_test_only --force_cpu
"""

import sys
import os
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.sft_dataset import (
    load_sft_dataset,
    create_train_val_split,
    SFTDataset,
    DataCollatorForSFT,
)
from src.parser import ModelOutputParser, Action


def get_git_commit_hash() -> str:
    """Retrieve current git commit hash if available."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL)
        return commit.decode("utf-8").strip()
    except Exception:
        return "unknown"


def print_gpu_diagnostics(force_cpu: bool = False) -> Dict[str, Any]:
    """Inspect and report GPU environment, CUDA status, and VRAM."""
    import torch

    info = {
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": None,
        "total_vram_gb": 0.0,
        "allocated_vram_gb": 0.0,
        "device": "cpu",
    }

    print("\n==================== GPU / ENVIRONMENT DIAGNOSTICS ====================")
    if torch.cuda.is_available() and not force_cpu:
        device_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        alloc_vram = torch.cuda.memory_allocated(0) / (1024 ** 3)
        info["gpu_name"] = device_name
        info["total_vram_gb"] = round(total_vram, 2)
        info["allocated_vram_gb"] = round(alloc_vram, 2)
        info["device"] = "cuda"

        print(f"CUDA Status        : Available (CUDA {torch.version.cuda})")
        print(f"GPU Model          : {device_name}")
        print(f"Total VRAM         : {total_vram:.2f} GB")
        print(f"Allocated VRAM     : {alloc_vram:.2f} GB")
    else:
        print(f"CUDA Status        : {'Disabled (--force_cpu)' if force_cpu else 'Not Available (Running on CPU)'}")
        if not force_cpu:
            print("WARNING: Training Qwen3-1.7B full-parameter SFT on CPU is extremely slow!")
    print("=======================================================================\n")
    return info


def verify_full_parameter_sft(model: Any) -> Dict[str, Any]:
    """
    Explicitly verify that PEFT/LoRA adapters are NOT present
    and all parameters are set as trainable.
    """
    # 1. PEFT module check
    for module_name, module in model.named_modules():
        mod_type = type(module).__name__.lower()
        if "lora" in mod_type or "adapter" in mod_type or "peft" in mod_type:
            raise RuntimeError(
                f"PEFT/LoRA adapter detected in module '{module_name}' ({type(module).__name__})! "
                "PEFT is strictly forbidden for full-parameter SFT."
            )

    # 2. Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_percentage = (trainable_params / total_params * 100) if total_params > 0 else 0.0

    print("==================== FULL PARAMETER SFT GUARANTEE ====================")
    print("FULL PARAMETER SFT: PEFT disabled")
    print(f"Total Parameters     : {total_params:,}")
    print(f"Trainable Parameters : {trainable_params:,}")
    print(f"Trainable Percentage : {trainable_percentage:.2f}%")
    print("=======================================================================\n")

    return {
        "peft_disabled": True,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_percentage": round(trainable_percentage, 2),
    }


def run_inference_smoke_test(model: Any, tokenizer: Any) -> bool:
    """
    Run post-training inference smoke test on a no-tool task and single-tool task.
    Verifies that ModelOutputParser parses outputs properly.
    """
    import torch

    print("\n==================== INFERENCE SMOKE TEST ====================")
    parser = ModelOutputParser()
    test_tasks = [
        ("no_tool", "What does CPU stand for in computer hardware?"),
        ("single_tool", "Calculate 25 * 37."),
    ]

    all_passed = True
    device = next(model.parameters()).device

    model.eval()
    for cat, prompt in test_tasks:
        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, "apply_chat_template"):
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                pad_token_id=tokenizer.eos_token_id,
            )

        input_len = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_len:]
        completion = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        action = parser.parse(completion)

        print(f"[{cat.upper()}] Prompt: '{prompt}'")
        print(f"  Completion : {completion}")
        print(f"  Parsed Action: action_type='{action.action_type}' -> {action}")

        if action.is_final_answer() or action.is_tool_call():
            print(f"  Result: PASSED (Valid protocol output)\n")
        else:
            print(f"  Result: FAILED (Unrecognized protocol output)\n")
            all_passed = False

    print("===============================================================\n")
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Full-parameter SFT training script for Qwen/Qwen3-1.7B.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B", help="Model name or local directory path")
    parser.add_argument("--dataset", type=str, default="data/sft/train_v2_100.jsonl", help="SFT JSONL dataset path")
    parser.add_argument("--output_dir", type=str, default="outputs/sft_qwen3_1.7b_v2_100", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device train batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X steps")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log metrics every X steps")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path or flag to resume checkpoint")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU mode (for testing)")
    parser.add_argument("--smoke_test_only", action="store_true", help="Run setup, data split & verification without full training")

    args = parser.parse_args()

    # 1. GPU Diagnostics
    import torch
    gpu_info = print_gpu_diagnostics(force_cpu=args.force_cpu)

    # 2. Check dataset availability (Do NOT modify dataset file)
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        fallback_path = PROJECT_ROOT / args.dataset
        if fallback_path.exists():
            dataset_path = fallback_path
        else:
            print(f"[ERROR] Dataset file not found at '{args.dataset}' or '{fallback_path}'")
            print("Please ensure you have committed and pushed 'data/sft/train_v2_100.jsonl' to git.")
            sys.exit(1)

    print(f"[INFO] Reading dataset from '{dataset_path}' (Read-Only)...")
    raw_examples = load_sft_dataset(str(dataset_path))
    print(f"[INFO] Total dataset size: {len(raw_examples)} examples")

    # 3. Create Train / Validation Split (90/10)
    train_data, val_data, split_info = create_train_val_split(raw_examples, val_ratio=0.10, seed=args.seed)
    print(f"[INFO] Train split      : {len(train_data)} examples")
    print(f"[INFO] Validation split : {len(val_data)} examples (Seed {args.seed})")

    # 4. Import Transformers & Torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        set_seed,
    )

    set_seed(args.seed)

    # 5. Load Model & Tokenizer
    print(f"[INFO] Loading tokenizer from '{args.model}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    use_bf16 = gpu_info["cuda_available"] and torch.cuda.is_bf16_supported() and not args.force_cpu
    use_fp16 = gpu_info["cuda_available"] and not use_bf16 and not args.force_cpu

    if use_bf16:
        dtype = torch.bfloat16
    elif use_fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    print(f"[INFO] Loading model from '{args.model}' (dtype: {dtype}, bf16: {use_bf16}, fp16: {use_fp16})...")
    if gpu_info["cuda_available"] and not args.force_cpu:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        model.to("cpu")

    # Ensure all parameters have requires_grad = True (Full-parameter SFT)
    for p in model.parameters():
        p.requires_grad = True

    # 6. Verify NO PEFT
    param_info = verify_full_parameter_sft(model)

    # 7. Prepare Datasets & Data Collator
    train_dataset = SFTDataset(train_data, tokenizer, max_seq_length=args.max_seq_length)
    val_dataset = SFTDataset(val_data, tokenizer, max_seq_length=args.max_seq_length)
    collator = DataCollatorForSFT(tokenizer)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    final_model_path = output_path / "final"

    # Save split info
    split_file = output_path / "split_info.json"
    with open(split_file, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    # 8. Handle Smoke Test Only mode
    if args.smoke_test_only:
        print("[INFO] --smoke_test_only enabled. Running setup & inference smoke test...")
        smoke_passed = run_inference_smoke_test(model, tokenizer)
        
        # Save dry run metrics
        metrics = {
            "model_name": args.model,
            "dataset": args.dataset,
            "train_examples": len(train_data),
            "val_examples": len(val_data),
            "seed": args.seed,
            "param_info": param_info,
            "gpu_info": gpu_info,
            "git_commit": get_git_commit_hash(),
            "smoke_test_passed": smoke_passed,
            "status": "dry_run_complete",
        }
        metrics_file = output_path / "metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[SUCCESS] Setup and dry-run completed successfully! Metrics saved to '{metrics_file}'")
        return

    # 9. Full Training setup with HuggingFace Trainer
    import inspect

    valid_arg_names = set(inspect.signature(TrainingArguments.__init__).parameters.keys())

    training_kwargs = {
        "output_dir": str(output_path),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": 3,
        "seed": args.seed,
        "bf16": use_bf16,
        "fp16": use_fp16,
        "report_to": "none",
    }

    if "warmup_ratio" in valid_arg_names:
        training_kwargs["warmup_ratio"] = args.warmup_ratio
    elif "warmup_steps" in valid_arg_names:
        total_steps = (len(train_data) // max(1, args.batch_size * args.gradient_accumulation_steps)) * args.epochs
        training_kwargs["warmup_steps"] = max(1, int(total_steps * args.warmup_ratio))

    if len(val_data) > 0:
        if "eval_strategy" in valid_arg_names:
            training_kwargs["eval_strategy"] = "steps"
            training_kwargs["eval_steps"] = args.save_steps
        elif "evaluation_strategy" in valid_arg_names:
            training_kwargs["evaluation_strategy"] = "steps"
            training_kwargs["eval_steps"] = args.save_steps

    filtered_kwargs = {k: v for k, v in training_kwargs.items() if k in valid_arg_names}
    training_args = TrainingArguments(**filtered_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    print("\n==================== STARTING FULL SFT TRAINING ====================")
    resume_arg = args.resume_from_checkpoint
    if resume_arg and resume_arg.lower() in ("true", "1"):
        resume_arg = True

    train_result = trainer.train(resume_from_checkpoint=resume_arg)
    print("====================================================================\n")

    # 10. Save Final Model & Tokenizer
    print(f"[INFO] Saving final model checkpoint to '{final_model_path}' and '{output_path}'...")
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    # 11. Record Metrics & Training State
    metrics = train_result.metrics
    metrics["dataset"] = args.dataset
    metrics["train_examples"] = len(train_data)
    metrics["val_examples"] = len(val_data)
    metrics["param_info"] = param_info
    metrics["gpu_info"] = gpu_info
    metrics["git_commit"] = get_git_commit_hash()

    metrics_file = output_path / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 12. Run Post-Training Inference Smoke Test
    run_inference_smoke_test(model, tokenizer)
    print(f"[SUCCESS] Full-parameter SFT training completed. Model ready at '{final_model_path}'!")


if __name__ == "__main__":
    main()
