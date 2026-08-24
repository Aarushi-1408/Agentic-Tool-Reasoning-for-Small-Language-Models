#!/usr/bin/env python3
"""
CLI script to convert raw teacher trajectories into a clean, validated SFT dataset.

Usage:
  python scripts/convert_to_sft.py --input data/raw/teacher_trajectories.jsonl --output data/sft/train.jsonl --overwrite
"""

import sys
import os
import json
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.convert_to_sft import SFTConverter


def main():
    parser = argparse.ArgumentParser(description="Convert raw teacher trajectories to SFT dataset format.")
    parser.add_argument("--input", type=str, default="data/raw/teacher_trajectories.jsonl", help="Input raw teacher trajectories JSONL file")
    parser.add_argument("--output", type=str, default="data/sft/train.jsonl", help="Output SFT dataset JSONL file")
    parser.add_argument("--rejected", type=str, default="data/sft/rejected.jsonl", help="Output path for rejected trajectories JSONL")
    parser.add_argument("--stats", type=str, default="data/sft/dataset_stats.json", help="Output path for dataset statistics JSON")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output dataset")
    parser.add_argument("--print-samples", type=int, default=5, help="Number of converted SFT sample records to print to console")
    args = parser.parse_args()

    # Resolve input path with project root fallback
    input_path = args.input
    if not os.path.exists(input_path):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        alt_path = os.path.join(project_root, args.input)
        if os.path.exists(alt_path):
            input_path = alt_path
        else:
            print(f"[ERROR] Input file not found at '{args.input}' (also checked '{alt_path}')")
            sys.exit(1)

    print(f"\n==================== SFT DATASET CONVERSION ====================")
    print(f"Input Raw Trajectories : '{input_path}'")
    print(f"Output SFT Dataset     : '{args.output}'")
    print(f"Output Rejected File   : '{args.rejected}'")
    print(f"Dataset Stats File     : '{args.stats}'")
    print(f"Overwrite Enabled      : {args.overwrite}")
    print("=================================================================\n")

    converter = SFTConverter()

    try:
        stats = converter.process_file(
            input_file=input_path,
            output_sft_file=args.output,
            output_rejected_file=args.rejected,
            stats_file=args.stats,
            overwrite=args.overwrite,
        )
    except FileExistsError as err:
        print(f"[ERROR] {str(err)}")
        sys.exit(1)
    except Exception as err:
        print(f"[ERROR] Conversion failed: {str(err)}")
        sys.exit(1)

    print(f"==================== DATASET CONVERSION METRICS ====================")
    print(f"Raw trajectories       : {stats['raw_trajectories']}")
    print(f"Valid trajectories     : {stats['valid_trajectories']}")
    print(f"Rejected trajectories  : {stats['rejected_trajectories']}")
    print(f"Acceptance Rate        : {stats['acceptance_rate'] * 100:.2f}%")
    print(f"\nCategory Breakdown:")
    for cat_name, cat_cnt in stats['categories'].items():
        print(f"  {cat_name:<20}: {cat_cnt}")
    print(f"\nAverages:")
    print(f"  Avg messages/example : {stats['average_messages_per_example']}")
    print(f"  Avg tool calls/ex.   : {stats['average_tool_calls_per_example']}")
    if stats['rejection_reasons']:
        print(f"\nRejection Reasons      : {json.dumps(stats['rejection_reasons'], indent=2)}")
    print("===================================================================\n")

    # Print sample converted records (Requirement 13)
    if args.print_samples > 0 and os.path.exists(args.output):
        print(f"--- SAMPLE {args.print_samples} CONVERTED SFT RECORDS ---")
        with open(args.output, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= args.print_samples:
                    break
                sample_data = json.loads(line)
                print(f"\n[Sample {idx + 1}] ID: {sample_data.get('trajectory_id')} | Category: {sample_data.get('category')}")
                print(json.dumps(sample_data, indent=2))
        print("-----------------------------------------------------------\n")


if __name__ == "__main__":
    main()
