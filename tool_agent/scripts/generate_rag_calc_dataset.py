#!/usr/bin/env python3
"""
CLI script to generate, validate, and partition the RAG & Calculator synthetic dataset.

Usage:
  python scripts/generate_rag_calc_dataset.py --num-examples 3000
"""

import sys
import os
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.rag_calc_generator import RAGCalcDatasetGenerator
from src.generation.rag_calc_validator import RAGCalcTrajectoryValidator


def generate_unseen_eval_tasks(output_path: str):
    """
    Generate separate evaluation set tasks/rag_eval_v1.json (50 unseen tasks)
    WITHOUT modifying frozen tasks/eval_v1.json.
    """
    eval_tasks = [
        {"task_id": "rag_eval_01", "category": "no_tool", "prompt": "What is the boiling point of water at sea level?", "expected_answer": "100 degrees Celsius", "requires_tool": False},
        {"task_id": "rag_eval_02", "category": "calculator", "prompt": "What is 345 multiplied by 67?", "expected_answer": "23115", "requires_tool": True},
        {"task_id": "rag_eval_03", "category": "bm25_search", "prompt": "Find documents with the exact phrase 'speculative decoding speedup'.", "expected_answer": "doc_101", "requires_tool": True},
        {"task_id": "rag_eval_04", "category": "dense_vector_search", "prompt": "Find conceptual research on state space models for long context.", "expected_answer": "vec_doc_201", "requires_tool": True},
        {"task_id": "rag_eval_05", "category": "hybrid_search", "prompt": "Find papers matching exact keyword 'Mamba-2' and conceptual relevance.", "expected_answer": "hyb_doc_301", "requires_tool": True},
        {"task_id": "rag_eval_06", "category": "reranker", "prompt": "Rerank candidate search results for query 'preference optimization'.", "expected_answer": "reranked_doc_01", "requires_tool": True},
        {"task_id": "rag_eval_07", "category": "multi_query_retrieval", "prompt": "Search across 3 query formulations for LLM memory footprint reduction.", "expected_answer": "mq_doc_401", "requires_tool": True},
        {"task_id": "rag_eval_08", "category": "metadata_filtered_search", "prompt": "Find 2024 AI papers by author 'Vaswani'.", "expected_answer": "meta_doc_501", "requires_tool": True},
        {"task_id": "rag_eval_09", "category": "parent_document_retrieval", "prompt": "Retrieve full parent document for chunk ID 'vec_doc_201'.", "expected_answer": "parent_vec_doc_201", "requires_tool": True},
        {"task_id": "rag_eval_10", "category": "multi_step", "prompt": "Retrieve documents on 'FlashAttention-3', rerank results, and calculate section count * 50.", "expected_answer": "calculated_metric_400", "requires_tool": True},
    ]

    # Expand to 50 tasks
    full_eval_tasks = []
    for idx in range(50):
        base_t = eval_tasks[idx % len(eval_tasks)].copy()
        base_t["task_id"] = f"rag_eval_{idx+1:02d}"
        if idx >= len(eval_tasks):
            base_t["prompt"] = f"{base_t['prompt']} (Evaluation Task variant #{idx+1})"
        full_eval_tasks.append(base_t)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_eval_tasks, f, indent=2)

    print(f"[SUCCESS] Generated separate evaluation set '{output_path}' ({len(full_eval_tasks)} unseen tasks)")


def main():
    parser = argparse.ArgumentParser(description="Generate RAG & Calculator Synthetic Dataset")
    parser.add_argument("--num-examples", type=int, default=3000, help="Total number of trajectory examples to generate (1000, 2000, 3000, 5000)")
    parser.add_argument("--output-dir", type=str, default="data/sft/rag_calc_v1", help="Output directory for train/val splits and stats")
    parser.add_argument("--eval-output", type=str, default="tasks/rag_eval_v1.json", help="Output path for separate evaluation set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n==================== RAG & CALCULATOR DATASET GENERATION ====================")
    print(f"Target Example Count : {args.num_examples}")
    print(f"Output Directory     : '{out_dir}'")
    print(f"Random Seed          : {args.seed}")

    # 1. Generate Synthetic Dataset
    generator = RAGCalcDatasetGenerator(seed=args.seed)
    records = generator.generate_dataset(total_count=args.num_examples)

    # 2. Run Automatic Validator across all 14 criteria
    validator = RAGCalcTrajectoryValidator()
    val_summary = validator.validate_batch(records)

    if val_summary["pass_rate"] < 1.0:
        print(f"\n[ERROR] Validation failed! Pass rate is {val_summary['pass_rate']*100:.1f}%.")
        print("Sample Validation Errors:")
        for err in val_summary["sample_errors"]:
            print(f"  - {err['id']}: {err['errors']}")
        sys.exit(1)

    print(f"[SUCCESS] All {len(records)} generated trajectories passed 100% of validation rules!")

    # 3. Create Train (90%) and Validation (10%) Split
    val_count = int(len(records) * 0.10)
    train_records = records[val_count:]
    val_records = records[:val_count]

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "validation.jsonl"
    stats_path = out_dir / "dataset_stats.json"

    with open(train_path, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r) + "\n")

    # 4. Generate dataset_stats.json
    cat_counts = val_summary["category_counts"]
    tool_counts = val_summary["tool_counts"]
    
    stats_data = {
        "total_trajectories": len(records),
        "train_count": len(train_records),
        "validation_count": len(val_records),
        "pass_rate": val_summary["pass_rate"],
        "duplicate_rate": val_summary["duplicate_rate"],
        "category_counts": cat_counts,
        "tool_counts": tool_counts,
        "metrics": {
            "no_tool_percentage": round(cat_counts.get("no_tool", 0) / len(records) * 100, 2),
            "calculator_percentage": round(cat_counts.get("calculator", 0) / len(records) * 100, 2),
            "rag_single_percentage": round(cat_counts.get("rag_single_tool", 0) / len(records) * 100, 2),
            "rag_multi_percentage": round(cat_counts.get("rag_multi_tool", 0) / len(records) * 100, 2),
            "error_recovery_percentage": round(cat_counts.get("error_recovery", 0) / len(records) * 100, 2),
            "hard_negatives_percentage": round(cat_counts.get("hard_negatives", 0) / len(records) * 100, 2),
        }
    }

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_data, f, indent=2)

    # 5. Generate separate tasks/rag_eval_v1.json
    generate_unseen_eval_tasks(args.eval_output)

    # 6. PRINT MANDATORY CONCISE METRICS REPORT
    print("\n==================== CONCISE DATASET REPORT ====================")
    print(f"TOTAL EXAMPLES          : {len(records)}")
    print(f"NO-TOOL                 : {cat_counts.get('no_tool', 0)}")
    print(f"BM25                    : {tool_counts.get('bm25_search', 0)}")
    print(f"DENSE                   : {tool_counts.get('dense_vector_search', 0)}")
    print(f"HYBRID                  : {tool_counts.get('hybrid_search', 0)}")
    print(f"RERANKER                : {tool_counts.get('reranker', 0)}")
    print(f"MULTI-QUERY             : {tool_counts.get('multi_query_retrieval', 0)}")
    print(f"METADATA/FILTERED       : {tool_counts.get('metadata_filtered_search', 0)}")
    print(f"PARENT-DOCUMENT         : {tool_counts.get('parent_document_retrieval', 0)}")
    print(f"CALCULATOR              : {tool_counts.get('calculator', 0)}")
    print(f"MULTI-TOOL              : {cat_counts.get('rag_multi_tool', 0)}")
    print(f"ERROR RECOVERY          : {cat_counts.get('error_recovery', 0)}")
    print(f"HARD NEGATIVES          : {cat_counts.get('hard_negatives', 0)}")
    print(f"VALIDATION PASS RATE    : {val_summary['pass_rate'] * 100:.1f}%")
    print(f"DUPLICATE RATE          : {val_summary['duplicate_rate'] * 100:.2f}%")
    print("=================================================================\n")


if __name__ == "__main__":
    main()
