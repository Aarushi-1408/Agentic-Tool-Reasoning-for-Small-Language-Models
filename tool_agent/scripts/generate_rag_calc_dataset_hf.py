#!/usr/bin/env python3
"""
Synthetic Trajectory Dataset Generator using HuggingFace Transformers directly on GPU.

Supports:
- Qwen/Qwen2.5-7B-Instruct (Fits on ~15 GB GPU VRAM)
- Qwen/Qwen2.5-14B-Instruct (with load_in_4bit=True)
- Qwen/Qwen2.5-72B-Instruct (requires multi-GPU / >40 GB VRAM with load_in_4bit=True)

Usage:
  python scripts/generate_rag_calc_dataset_hf.py --num-examples 3000 --model-name "Qwen/Qwen2.5-7B-Instruct"
"""

import sys
import os
import json
import torch
import argparse
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tools.rag_calc_tools import (
    BM25SearchTool,
    DenseVectorSearchTool,
    HybridSearchTool,
    RerankerTool,
    MultiQueryRetrievalTool,
    MetadataFilteredSearchTool,
    ParentDocumentRetrievalTool,
    CalculatorToolV2,
)
from src.generation.rag_calc_validator import RAGCalcTrajectoryValidator


SYSTEM_PROMPT = """You are an expert AI dataset generator producing high-quality trajectory examples for a Qwen3-1.7B tool-using agent.
The agent has access to 8 explicit tools:
1. bm25_search(query: str, top_k: int)
2. dense_vector_search(query: str, top_k: int)
3. hybrid_search(query: str, top_k: int, alpha: float)
4. reranker(query: str, documents: list)
5. multi_query_retrieval(queries: list, top_k: int)
6. metadata_filtered_search(query: str, filters: dict, top_k: int)
7. parent_document_retrieval(document_id: str)
8. calculator(operation: str ['add','sub','mul','divide','modulo'], a: float, b: float)

Output ONLY a single valid JSON object representing the trajectory specification:
{
  "category": "<no_tool|calculator|rag_single_tool|rag_multi_tool|error_recovery|hard_negatives>",
  "task": {
    "prompt": "<user question>",
    "requires_tool": true/false,
    "expected_tools": ["<tool_name>"],
    "expected_answer": "<final answer string>"
  },
  "initial_action": {
    "type": "tool_call",
    "name": "<tool_name>",
    "arguments": { ... }
  }
}
"""


class HuggingFaceTeacherGenerator:
    """
    Direct GPU Trajectory Generator using HuggingFace Transformers.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", load_4bit: bool = False):
        self.model_name = model_name
        print(f"[HF GENERATOR] Loading '{model_name}' on CUDA GPU (4-bit={load_4bit})...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        if load_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )

        self.tools = {
            "bm25_search": BM25SearchTool(),
            "dense_vector_search": DenseVectorSearchTool(),
            "hybrid_search": HybridSearchTool(),
            "reranker": RerankerTool(),
            "multi_query_retrieval": MultiQueryRetrievalTool(),
            "metadata_filtered_search": MetadataFilteredSearchTool(),
            "parent_document_retrieval": ParentDocumentRetrievalTool(),
            "calculator": CalculatorToolV2(),
        }

    def generate_single_trajectory(self, category: str, index: int) -> Dict[str, Any]:
        """
        Generate trajectory specification using local HuggingFace GPU model.
        """
        user_msg = f"Generate a high-quality '{category}' trajectory specification for example #{index+1}."
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        text_input = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text_input], return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
            )

        gen_ids = outputs[0][inputs.input_ids.shape[1]:]
        response_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # Parse JSON
        try:
            # Extract JSON block
            if "{" in response_text and "}" in response_text:
                json_str = response_text[response_text.find("{"):response_text.rfind("}")+1]
                spec = json.loads(json_str)
            else:
                spec = json.loads(response_text)
        except Exception:
            # Fallback spec if parsing fails
            spec = {
                "category": category,
                "task": {
                    "prompt": f"Task example #{index+1} for {category}",
                    "requires_tool": category != "no_tool",
                    "expected_tools": ["bm25_search"] if category != "no_tool" else [],
                    "expected_answer": f"Answer #{index+1}"
                },
                "initial_action": {"type": "final", "content": f"Answer #{index+1}"} if category == "no_tool" else {
                    "type": "tool_call", "name": "bm25_search", "arguments": {"query": f"query #{index+1}", "top_k": 5}
                }
            }

        # Build trajectory messages and execute live tool
        traj_msgs = [{"role": "user", "content": spec["task"]["prompt"]}]
        init_act = spec.get("initial_action", {"type": "final", "content": spec["task"]["expected_answer"]})

        if init_act.get("type") == "tool_call":
            t_name = init_act.get("name")
            t_args = init_act.get("arguments", {})
            traj_msgs.append({"role": "assistant", "content": json.dumps(init_act)})

            if t_name in self.tools:
                try:
                    tool_res = self.tools[t_name].execute(t_args)
                    traj_msgs.append({"role": "user", "content": f"TOOL_RESULT: {json.dumps(tool_res)}"})
                except Exception as e:
                    traj_msgs.append({"role": "user", "content": f"TOOL_ERROR: {str(e)}"})

            final_ans = spec["task"]["expected_answer"]
            traj_msgs.append({"role": "assistant", "content": json.dumps({"type": "final", "content": final_ans})})
        else:
            traj_msgs.append({"role": "assistant", "content": json.dumps(init_act)})

        return {
            "trajectory_id": f"hf_rag_calc_{index+1:05d}",
            "category": category,
            "task": spec["task"],
            "messages": traj_msgs,
            "metadata": {
                "teacher_model": self.model_name,
                "tool_count": 1 if init_act.get("type") == "tool_call" else 0,
            }
        }


def main():
    parser = argparse.ArgumentParser(description="Generate Dataset directly on GPU using HuggingFace")
    parser.add_argument("--num-examples", type=int, default=3000, help="Total examples to generate")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="HuggingFace model repo")
    parser.add_argument("--load-4bit", action="store_true", help="Load model in 4-bit precision to save VRAM")
    parser.add_argument("--output-dir", type=str, default="data/sft/rag_calc_v1", help="Output directory")
    args = parser.parse_args()

    print(f"\n==================== HUGGINGFACE GPU DATASET GENERATION ====================")
    print(f"Model Name   : {args.model_name}")
    print(f"4-bit Quant  : {args.load_4bit}")
    print(f"Num Examples : {args.num_examples}")
    print(f"Output Dir   : '{args.output_dir}'")
    print(f"============================================================================\n")

    teacher = HuggingFaceTeacherGenerator(model_name=args.model_name, load_4bit=args.load_4bit)
    categories = ["no_tool", "calculator", "rag_single_tool", "rag_multi_tool", "error_recovery", "hard_negatives"]
    records = []

    for i in range(args.num_examples):
        cat = categories[i % len(categories)]
        print(f"[{i+1}/{args.num_examples}] Generating '{cat}' trajectory on GPU via HF...", end="\r")
        rec = teacher.generate_single_trajectory(category=cat, index=i)
        records.append(rec)

    print(f"\n[SUCCESS] Completed generation of {len(records)} trajectories on GPU!")

    validator = RAGCalcTrajectoryValidator()
    val_res = validator.validate_batch(records)
    print(f"Validation Pass Rate : {val_res['pass_rate']*100:.1f}%")
    print(f"Duplicate Rate       : {val_res['duplicate_rate']*100:.2f}%")


if __name__ == "__main__":
    main()
