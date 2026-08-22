#!/usr/bin/env python3
"""
CLI script to validate a trajectory JSONL file using TrajectoryValidator.

Usage:
  python scripts/validate_trajectories.py --trajectories data/raw/teacher_trajectories.jsonl
"""

import sys
import os
import json
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.generation.validator import TrajectoryValidator
from tools.example_tools import CalculatorTool, EchoTool, LookupTool, StringManipulatorTool


def main():
    parser = argparse.ArgumentParser(description="Validate trajectory JSONL file structure and tool calls.")
    parser.add_argument("--trajectories", type=str, default="data/raw/teacher_trajectories.jsonl", help="Path to trajectories JSONL file")
    args = parser.parse_args()

    if not os.path.exists(args.trajectories):
        print(f"[ERROR] Trajectory file not found at '{args.trajectories}'")
        sys.exit(1)

    tools = {"calculator", "echo", "lookup", "string_manipulator"}
    validator = TrajectoryValidator(registered_tools=tools)

    report = validator.validate_file(args.trajectories)

    print(f"\n==================== TRAJECTORY VALIDATION REPORT ====================")
    print(f"File Path            : {args.trajectories}")
    print(f"Total Trajectories   : {report['total_trajectories']}")
    print(f"Valid Trajectories   : {report['valid_trajectories']}")
    print(f"Invalid Trajectories : {report['invalid_trajectories']}")
    print(f"Validation Rate      : {report['valid_rate'] * 100:.2f}%")
    print("======================================================================\n")

    if report['invalid_trajectories'] > 0:
        print("[WARNING] Details of invalid trajectories:")
        for detail in report['details']:
            if not detail['valid']:
                print(f"  Line {detail['line']} (Task ID: {detail['task_id']}): {', '.join(detail['errors'])}")
        sys.exit(1)
    else:
        print("[SUCCESS] All trajectory records passed validation successfully!")


if __name__ == "__main__":
    main()
