#!/usr/bin/env python3
"""
Synthetic Trajectory Dataset Generator using Qwen/Qwen2.5-72B-Instruct as Teacher LLM.

Supports OpenAI-compatible API endpoints (OpenRouter, DeepInfra, Together AI, vLLM local endpoint).

Usage:
  # Using OpenRouter / DeepInfra / Together API:
  export OPENAI_API_KEY="your-api-key"
  python scripts/generate_rag_calc_dataset_qwen72b.py --num-examples 3000 --api-base "https://api.together.xyz/v1"

  # Or using local vLLM server:
  python scripts/generate_rag_calc_dataset_qwen72b.py --num-examples 3000 --api-base "http://localhost:8000/v1"
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

try:
    import openai
except ImportError:
    openai = None


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

Rules for output JSON format:
Output a single JSON object for the trajectory matching this exact structure:
{
  "category": "<no_tool|calculator|rag_single_tool|rag_multi_tool|error_recovery|hard_negatives>",
  "task": {
    "prompt": "<user question>",
    "requires_tool": <true|false>,
    "expected_tools": ["<tool_name>"],
    "expected_answer": "<final answer string>"
  },
  "initial_action": {
    "type": "tool_call",
    "name": "<tool_name>",
    "arguments": { ... }
  } or {"type": "final", "content": "<answer>"}
}
"""


class Qwen72BTeacherGenerator:
    """
    Trajectory dataset generator powered by Qwen2.5-72B-Instruct.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: str = "https://openrouter.ai/api/v1",
        model_name: str = "Qwen/Qwen2.5-72B-Instruct",
    ):
        if openai is None:
            raise ImportError("Please install the 'openai' python package: pip install openai")

        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "dummy_key")
        self.api_base = api_base
        self.model_name = model_name

        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
        )

        # Live Python tools for grounded observation execution
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
        Query Qwen2.5-72B-Instruct to generate a synthetic trajectory specification.
        """
        user_prompt = f"Generate a high-quality '{category}' trajectory specification for example #{index+1}."
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
        )

        spec = json.loads(response.choices[0].message.content)
        
        # Execute tool calls against live python tool instances for grounded results
        messages = [{"role": "user", "content": spec["task"]["prompt"]}]
        initial_action = spec.get("initial_action", {"type": "final", "content": spec["task"]["expected_answer"]})

        if initial_action.get("type") == "tool_call":
            t_name = initial_action.get("name")
            t_args = initial_action.get("arguments", {})
            messages.append({"role": "assistant", "content": json.dumps(initial_action)})

            # Execute tool if valid
            if t_name in self.tools:
                try:
                    tool_res = self.tools[t_name].execute(t_args)
                    messages.append({"role": "user", "content": f"TOOL_RESULT: {json.dumps(tool_res)}"})
                except Exception as e:
                    messages.append({"role": "user", "content": f"TOOL_ERROR: {str(e)}"})

            # Append final action
            final_ans = spec["task"]["expected_answer"]
            messages.append({"role": "assistant", "content": json.dumps({"type": "final", "content": final_ans})})
        else:
            messages.append({"role": "assistant", "content": json.dumps(initial_action)})

        return {
            "trajectory_id": f"qwen72b_rag_calc_{index+1:05d}",
            "category": category,
            "task": spec["task"],
            "messages": messages,
            "metadata": {
                "teacher_model": self.model_name,
                "tool_count": 1 if initial_action.get("type") == "tool_call" else 0,
            }
        }


def main():
    parser = argparse.ArgumentParser(description="Generate RAG + Calculator Dataset using Qwen2.5-72B-Instruct")
    parser.add_argument("--num-examples", type=int, default=3000, help="Total trajectories to generate")
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI / OpenRouter / Together API key")
    parser.add_argument("--api-base", type=str, default="https://openrouter.ai/api/v1", help="API base URL")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-72B-Instruct", help="Teacher model identifier")
    parser.add_argument("--output-dir", type=str, default="data/sft/rag_calc_v1", help="Output directory")
    args = parser.parse_args()

    print(f"\n==================== QWEN2.5-72B TEACHER DATASET GENERATION ====================")
    print(f"Teacher Model : {args.model_name}")
    print(f"API Base      : {args.api_base}")
    print(f"Num Examples  : {args.num_examples}")
    print(f"Output Dir    : '{args.output_dir}'")
    print(f"=================================================================================\n")

    # Instantiate Teacher Generator
    teacher = Qwen72BTeacherGenerator(
        api_key=args.api_key,
        api_base=args.api_base,
        model_name=args.model_name,
    )

    categories = ["no_tool", "calculator", "rag_single_tool", "rag_multi_tool", "error_recovery", "hard_negatives"]
    records = []

    for i in range(args.num_examples):
        cat = categories[i % len(categories)]
        print(f"[{i+1}/{args.num_examples}] Generating '{cat}' trajectory via Qwen2.5-72B...", end="\r")
        rec = teacher.generate_single_trajectory(category=cat, index=i)
        records.append(rec)

    print(f"\n[SUCCESS] Completed generation of {len(records)} trajectories via {args.model_name}!")

    # Validate generated records
    validator = RAGCalcTrajectoryValidator()
    val_res = validator.validate_batch(records)
    print(f"Validation Pass Rate : {val_res['pass_rate']*100:.1f}%")
    print(f"Duplicate Rate       : {val_res['duplicate_rate']*100:.2f}%")


if __name__ == "__main__":
    main()
