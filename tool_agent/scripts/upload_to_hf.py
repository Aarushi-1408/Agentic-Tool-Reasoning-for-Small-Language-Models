#!/usr/bin/env python3
"""
CLI script to upload trained SFT model checkpoint to Hugging Face Hub.

Usage:
  # First login to Hugging Face (or set HF_TOKEN environment variable):
  huggingface-cli login

  # Push trained SFT v3 model:
  python scripts/upload_to_hf.py \
    --model-dir outputs/qwen3-1.7b-tool-agent-sft-v3/final \
    --repo-id Bitsbrains/aarushi-qwen3-1.7b-tool-agent-sft-v3
"""

import os
import sys
import argparse
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Upload trained model checkpoint to Hugging Face Hub.")
    parser.add_argument("--model-dir", type=str, default="outputs/qwen3-1.7b-tool-agent-sft-v3/final", help="Path to local trained model directory")
    parser.add_argument("--repo-id", type=str, default="Bitsbrains/aarushi-qwen3-1.7b-tool-agent-sft-v3", help="Target HF Hub repo ID (e.g. username/model_name)")
    parser.add_argument("--private", action="store_true", help="Set repository as private on Hugging Face")
    args = parser.parse_args()

    model_path = Path(args.model_dir)
    if not model_path.exists():
        alt_path = model_path.parent
        if alt_path.exists() and (alt_path / "config.json").exists():
            model_path = alt_path
        else:
            print(f"[ERROR] Model directory not found at '{args.model_dir}'")
            sys.exit(1)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("[ERROR] transformers or torch package missing. Install via `pip install transformers torch`")
        sys.exit(1)

    print("\n==================== UPLOADING TO HUGGING FACE HUB ====================")
    print(f"Local Model Directory : {model_path}")
    print(f"Target HF Repo ID     : {args.repo_id}")
    print(f"Private Repository    : {args.private}")
    print("=======================================================================\n")

    print("[INFO] Loading local model and tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    except Exception as e:
        print(f"[WARNING] Failed to load tokenizer from '{model_path}' ({e}). Loading base tokenizer 'Qwen/Qwen3-1.7B'...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )

    print(f"[INFO] Pushing tokenizer to '{args.repo_id}'...")
    tokenizer.push_to_hub(repo_id=args.repo_id, private=args.private)

    print(f"[INFO] Pushing model weights to '{args.repo_id}'...")
    model.push_to_hub(repo_id=args.repo_id, private=args.private)

    print("\n=======================================================================")
    print(f"[SUCCESS] Model successfully uploaded to Hugging Face Hub!")
    print(f"Repository URL: https://huggingface.co/{args.repo_id}")
    print("=======================================================================\n")


if __name__ == "__main__":
    main()
