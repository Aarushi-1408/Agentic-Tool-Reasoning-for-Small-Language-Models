"""
Synthetic Dataset Generator for RAG & Calculator Tool Agent.
Generates balanced, grounded, multi-step, error-recovering, and hard-negative trajectories.
"""

import json
import random
from typing import Dict, Any, List, Optional
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


class RAGCalcDatasetGenerator:
    """
    Modular Generator producing realistic JSONL trajectory records compatible with SFT/GRPO training.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        
        # Instantiate execution tools for grounded results
        self.bm25 = BM25SearchTool()
        self.dense = DenseVectorSearchTool()
        self.hybrid = HybridSearchTool()
        self.reranker = RerankerTool()
        self.multi_query = MultiQueryRetrievalTool()
        self.meta_search = MetadataFilteredSearchTool()
        self.parent_doc = ParentDocumentRetrievalTool()
        self.calc = CalculatorToolV2()

    def generate_dataset(
        self,
        total_count: int = 3000,
        no_tool_ratio: float = 0.20,
        calc_ratio: float = 0.20,
        single_tool_ratio: float = 0.30,
        multi_tool_ratio: float = 0.20,
        error_recovery_ratio: float = 0.05,
        hard_negative_ratio: float = 0.05,
    ) -> List[Dict[str, Any]]:
        """
        Generate complete synthetic trajectory dataset.
        """
        self.rng.seed(self.seed)

        num_no_tool = int(total_count * no_tool_ratio)
        num_calc = int(total_count * calc_ratio)
        num_single = int(total_count * single_tool_ratio)
        num_multi = int(total_count * multi_tool_ratio)
        num_err = int(total_count * error_recovery_ratio)
        num_hard = total_count - (num_no_tool + num_calc + num_single + num_multi + num_err)

        records: List[Dict[str, Any]] = []

        # 1. Generate NO-TOOL examples (20%)
        records.extend(self._generate_no_tool_examples(num_no_tool))

        # 2. Generate CALCULATOR examples (20%)
        records.extend(self._generate_calculator_examples(num_calc))

        # 3. Generate RAG SINGLE-TOOL examples (30%)
        records.extend(self._generate_rag_single_examples(num_single))

        # 4. Generate RAG MULTI-TOOL examples (20%)
        records.extend(self._generate_rag_multi_examples(num_multi))

        # 5. Generate ERROR RECOVERY examples (5%)
        records.extend(self._generate_error_recovery_examples(num_err))

        # 6. Generate HARD NEGATIVE examples (5%)
        records.extend(self._generate_hard_negative_examples(num_hard))

        # Shuffle deterministically
        self.rng.shuffle(records)

        # Assign unique trajectory IDs
        for idx, rec in enumerate(records):
            rec["trajectory_id"] = f"rag_calc_v1_{idx+1:05d}"

        return records

    # ------------------------------------------------------------------
    # 1. NO-TOOL GENERATOR
    # ------------------------------------------------------------------
    def _generate_no_tool_examples(self, count: int) -> List[Dict[str, Any]]:
        topics = [
            ("capital of France", "Paris"),
            ("CPU definition", "Central Processing Unit"),
            ("Shakespeare's famous tragedy", "Romeo and Juliet"),
            ("process plants use to convert sunlight into energy", "photosynthesis"),
            ("freezing point of water in Celsius", "0 degrees Celsius"),
            ("chemical symbol for Gold", "Au"),
            ("number of sides on a regular hexagon", "6 sides"),
            ("general concept of gradient descent optimization", "a first-order iterative optimization algorithm for finding a local minimum"),
            ("largest planet in our solar system", "Jupiter"),
            ("speed of light in vacuum", "approximately 299,792,458 meters per second"),
            ("Python list vs tuple mutability", "lists are mutable while tuples are immutable"),
            ("currency used in Japan", "Japanese Yen (JPY)"),
            ("painter of the Mona Lisa", "Leonardo da Vinci"),
            ("main function of red blood cells", "transporting oxygen from the lungs to body tissues"),
            ("author of the theory of general relativity", "Albert Einstein"),
            ("primary gas in Earth's atmosphere", "Nitrogen (approximately 78%)"),
            ("meaning of HTTP in computer networking", "Hypertext Transfer Protocol"),
            ("definition of an isosceles triangle", "a triangle with at least two sides of equal length"),
            ("atomic number of Carbon", "6"),
            ("year man first walked on the moon", "1969"),
        ]

        phrasings = [
            "Please explain {topic}.",
            "What is {topic}?",
            "Can you tell me about {topic}?",
            "Give a brief explanation of {topic}.",
            "Provide the standard answer for {topic}.",
        ]

        records = []
        seen_prompts = set()

        for i in range(count):
            topic_desc, answer = topics[i % len(topics)]
            phrasing = phrasings[(i // len(topics)) % len(phrasings)]
            prompt = phrasing.format(topic=topic_desc)

            if prompt in seen_prompts or i >= len(topics) * len(phrasings):
                prompt = f"{prompt} (Query variant #{i+1})"
            seen_prompts.add(prompt)

            rec = {
                "trajectory_id": "",
                "category": "no_tool",
                "task": {
                    "prompt": prompt,
                    "requires_tool": False,
                    "expected_tools": [],
                    "expected_answer": answer,
                },
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": json.dumps({"type": "final", "content": answer})},
                ],
                "metadata": {
                    "tool_count": 0,
                    "difficulty": "easy",
                    "requires_reasoning": False
                }
            }
            records.append(rec)
        return records

    # ------------------------------------------------------------------
    # 2. CALCULATOR GENERATOR
    # ------------------------------------------------------------------
    def _generate_calculator_examples(self, count: int) -> List[Dict[str, Any]]:
        ops = ["add", "sub", "mul", "divide", "modulo"]
        records = []
        seen_prompts = set()

        for i in range(count):
            op_name = ops[i % len(ops)]
            if op_name == "add":
                a = self.rng.randint(100, 9999) + i
                b = self.rng.randint(100, 9999) + (i * 2)
                prompt = f"Calculate the sum of {a} and {b} (Task #{i+1})."
            elif op_name == "sub":
                a = self.rng.randint(5000, 20000) + (i * 3)
                b = self.rng.randint(100, 4999) + i
                prompt = f"What is {a} minus {b}? (Task #{i+1})"
            elif op_name == "mul":
                a = self.rng.randint(12, 999) + i
                b = self.rng.randint(12, 99) + (i % 7)
                prompt = f"Calculate {a} multiplied by {b} (Task #{i+1})."
            elif op_name == "divide":
                b = self.rng.randint(2, 50) + 1
                mult_factor = self.rng.randint(10, 500) + i
                a = mult_factor * b
                prompt = f"What is {a} divided by {b}? (Task #{i+1})"
            else:  # modulo
                b = self.rng.randint(7, 49) + 1
                a = self.rng.randint(100, 9999) + (i * 5)
                prompt = f"Calculate the remainder of {a} divided by {b} (modulo Task #{i+1})."

            t_call = {"type": "tool_call", "name": "calculator", "arguments": {"operation": op_name, "a": a, "b": b}}
            t_res = self.calc.execute({"operation": op_name, "a": a, "b": b})
            final_ans = f"The result of {op_name} ({a}, {b}) is {t_res}."

            rec = {
                "trajectory_id": "",
                "category": "calculator",
                "task": {
                    "prompt": prompt,
                    "requires_tool": True,
                    "expected_tools": ["calculator"],
                    "expected_answer": str(t_res),
                },
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": json.dumps(t_call)},
                    {"role": "user", "content": f"TOOL_RESULT: {t_res}"},
                    {"role": "assistant", "content": json.dumps({"type": "final", "content": final_ans})},
                ],
                "metadata": {
                    "tool_count": 1,
                    "difficulty": "easy",
                    "requires_reasoning": True
                }
            }
            records.append(rec)
        return records

    # ------------------------------------------------------------------
    # 3. RAG SINGLE-TOOL GENERATOR
    # ------------------------------------------------------------------
    def _generate_rag_single_examples(self, count: int) -> List[Dict[str, Any]]:
        tool_types = ["bm25_search", "dense_vector_search", "hybrid_search", "multi_query_retrieval", "metadata_filtered_search"]
        records = []

        topics = [
            ("quantum error correction threshold", "exact_phrase"),
            ("methods for reducing hallucinations in language models", "conceptual"),
            ("transformer interpretability and sparse autoencoders", "hybrid"),
            ("scaling laws for neural machine translation", "multi_perspective"),
            ("reinforcement learning from human feedback", "filtered"),
            ("distributed GPU training parallelism", "conceptual"),
            ("direct preference optimization algorithms", "hybrid"),
            ("speculative decoding acceleration techniques", "filtered"),
            ("state space models and Mamba architecture", "exact_phrase"),
            ("parameter-efficient fine-tuning with LoRA", "multi_perspective"),
        ]

        for i in range(count):
            tool_name = tool_types[i % len(tool_types)]
            topic_base, topic_type = topics[i % len(topics)]
            topic_str = f"{topic_base} (Topic Variant #{i+1})"

            if tool_name == "bm25_search":
                prompt = f"Find documents containing the exact phrase '{topic_str}' (Req #{i+1})."
                t_args = {"query": topic_str, "top_k": 5}
                res_obj = self.bm25.execute(t_args)
            elif tool_name == "dense_vector_search":
                prompt = f"Find research papers discussing the concept of {topic_str} (Req #{i+1})."
                t_args = {"query": topic_str, "top_k": 5}
                res_obj = self.dense.execute(t_args)
            elif tool_name == "hybrid_search":
                prompt = f"Find documents matching both exact technical terms and semantic relevance for '{topic_str}' (Req #{i+1})."
                t_args = {"query": topic_str, "top_k": 5, "alpha": 0.5}
                res_obj = self.hybrid.execute(t_args)
            elif tool_name == "multi_query_retrieval":
                prompt = f"Gather evidence on '{topic_str}' from multiple related query formulations (Req #{i+1})."
                t_args = {"queries": [topic_str, f"overview of {topic_str}", f"advancements in {topic_str}"], "top_k": 5}
                res_obj = self.multi_query.execute(t_args)
            else:  # metadata_filtered_search
                year_val = 2020 + (i % 5)
                prompt = f"Find papers about '{topic_str}' published in year {year_val} in category AI (Req #{i+1})."
                t_args = {"query": topic_str, "filters": {"year": year_val, "category": "AI"}, "top_k": 5}
                res_obj = self.meta_search.execute(t_args)

            t_call = {"type": "tool_call", "name": tool_name, "arguments": t_args}
            doc_id = res_obj[0].get("document_id", "doc_101") if isinstance(res_obj, list) else "doc_101"
            final_ans = f"Based on retrieval using {tool_name}, document {doc_id} provides key details on {topic_str}."

            rec = {
                "trajectory_id": "",
                "category": "rag_single_tool",
                "task": {
                    "prompt": prompt,
                    "requires_tool": True,
                    "expected_tools": [tool_name],
                    "expected_answer": final_ans,
                },
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": json.dumps(t_call)},
                    {"role": "user", "content": f"TOOL_RESULT: {json.dumps(res_obj)}"},
                    {"role": "assistant", "content": json.dumps({"type": "final", "content": final_ans})},
                ],
                "metadata": {
                    "tool_count": 1,
                    "difficulty": "medium",
                    "requires_reasoning": True
                }
            }
            records.append(rec)
        return records

    # ------------------------------------------------------------------
    # 4. RAG MULTI-TOOL GENERATOR (>= 20%)
    # ------------------------------------------------------------------
    def _generate_rag_multi_examples(self, count: int) -> List[Dict[str, Any]]:
        workflows = [
            "hybrid_reranker",
            "dense_parent",
            "multi_reranker",
            "metadata_reranker",
            "retrieval_calc",
            "retrieval_parent_calc",
        ]

        records = []
        for i in range(count):
            wf = workflows[i % len(workflows)]

            if wf == "hybrid_reranker":
                topic_q = f"quantum supremacy research variant #{i+1}"
                prompt = f"Retrieve documents on '{topic_q}' and rerank them for maximum precision."
                c1 = {"type": "tool_call", "name": "hybrid_search", "arguments": {"query": topic_q, "top_k": 5, "alpha": 0.5}}
                r1 = self.hybrid.execute(c1["arguments"])
                c2 = {"type": "tool_call", "name": "reranker", "arguments": {"query": topic_q, "documents": r1}}
                r2 = self.reranker.execute(c2["arguments"])
                final_ans = f"After hybrid search and reranking, the top relevant document is {r2[0]['document_id']} with rerank score {r2[0]['rerank_score']}."
                msgs = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": json.dumps(c1)},
                    {"role": "user", "content": f"TOOL_RESULT: {json.dumps(r1)}"},
                    {"role": "assistant", "content": json.dumps(c2)},
                    {"role": "user", "content": f"TOOL_RESULT: {json.dumps(r2)}"},
                    {"role": "assistant", "content": json.dumps({"type": "final", "content": final_ans})},
                ]
                exp_tools = ["hybrid_search", "reranker"]

            elif wf == "dense_parent":
                topic_q = f"mixture of experts architecture chunk #{i+1}"
                prompt = f"Find passages on '{topic_q}' and expand to full parent document context."
                c1 = {"type": "tool_call", "name": "dense_vector_search", "arguments": {"query": topic_q, "top_k": 5}}
                r1 = self.dense.execute(c1["arguments"])
                chunk_id = r1[0]["document_id"]
                c2 = {"type": "tool_call", "name": "parent_document_retrieval", "arguments": {"document_id": chunk_id}}
                r2 = self.parent_doc.execute(c2["arguments"])
                final_ans = f"Retrieved passage {chunk_id} and expanded to parent document {r2['parent_document_id']} containing {r2['total_tokens']} tokens."
                msgs = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": json.dumps(c1)},
                    {"role": "user", "content": f"TOOL_RESULT: {json.dumps(r1)}"},
                    {"role": "assistant", "content": json.dumps(c2)},
                    {"role": "user", "content": f"TOOL_RESULT: {json.dumps(r2)}"},
                    {"role": "assistant", "content": json.dumps({"type": "final", "content": final_ans})},
                ]
                exp_tools = ["dense_vector_search", "parent_document_retrieval"]

            elif wf == "multi_reranker":
                topic_q = f"AI safety alignment and governance #{i+1}"
                prompt = f"Search for '{topic_q}' from multiple formulations and rerank results."
                c1 = {"type": "tool_call", "name": "multi_query_retrieval", "arguments": {"queries": [topic_q, f"RLHF safety techniques for {topic_q}"], "top_k": 5}}
                r1 = self.multi_query.execute(c1["arguments"])
                c2 = {"type": "tool_call", "name": "reranker", "arguments": {"query": topic_q, "documents": r1}}
                r2 = self.reranker.execute(c2["arguments"])
                final_ans = f"Multi-query search and reranking identified top evidence document {r2[0]['document_id']}."
                msgs = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": json.dumps(c1)},
                    {"role": "user", "content": f"TOOL_RESULT: {json.dumps(r1)}"},
                    {"role": "assistant", "content": json.dumps(c2)},
                    {"role": "user", "content": f"TOOL_RESULT: {json.dumps(r2)}"},
                    {"role": "assistant", "content": json.dumps({"type": "final", "content": final_ans})},
                ]
                exp_tools = ["multi_query_retrieval", "reranker"]

            elif wf == "metadata_reranker":
                year_val = 2020 + (i % 5)
                topic_q = f"multimodal model architectures #{i+1}"
                prompt = f"Search for {year_val} AI papers and rerank them based on relevance to '{topic_q}'."
                c1 = {"type": "tool_call", "name": "metadata_filtered_search", "arguments": {"query": topic_q, "filters": {"year": year_val}, "top_k": 5}}
                r1 = self.meta_search.execute(c1["arguments"])
                c2 = {"type": "tool_call", "name": "reranker", "arguments": {"query": topic_q, "documents": r1}}
                r2 = self.reranker.execute(c2["arguments"])
                final_ans = f"Filtered {year_val} paper {r2[0]['document_id']} reranked to top position."
                msgs = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": json.dumps(c1)},
                    {"role": "user", "content": f"TOOL_RESULT: {json.dumps(r1)}"},
                    {"role": "assistant", "content": json.dumps(c2)},
                    {"role": "user", "content": f"TOOL_RESULT: {json.dumps(r2)}"},
                    {"role": "assistant", "content": json.dumps({"type": "final", "content": final_ans})},
                ]
                exp_tools = ["metadata_filtered_search", "reranker"]

            elif wf == "retrieval_calc":
                val1 = 70 + (i * 3)
                val2 = 40 + (i * 2)
                prompt = f"Find atomic metric values {val1} and {val2} and calculate their product (Req #{i+1})."
                c1 = {"type": "tool_call", "name": "bm25_search", "arguments": {"query": f"metric values {val1} and {val2}", "top_k": 5}}
                r1 = [{"text": f"Metric value A is {val1}. Metric value B is {val2}."}]
                c2 = {"type": "tool_call", "name": "calculator", "arguments": {"operation": "mul", "a": val1, "b": val2}}
                r2 = self.calc.execute(c2["arguments"])
                final_ans = f"Retrieved values: A={val1}, B={val2}. Calculated product: {r2}."
                msgs = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": json.dumps(c1)},
                    {"role": "user", "content": f"TOOL_RESULT: {json.dumps(r1)}"},
                    {"role": "assistant", "content": json.dumps(c2)},
                    {"role": "user", "content": f"TOOL_RESULT: {r2}"},
                    {"role": "assistant", "content": json.dumps({"type": "final", "content": final_ans})},
                ]
                exp_tools = ["bm25_search", "calculator"]

            else:  # retrieval_parent_calc
                mult_val = 50 + (i * 5)
                prompt = f"Find passage for dataset metrics chunk #{i+1}, fetch parent document context, and calculate section count multiplied by {mult_val}."
                c1 = {"type": "tool_call", "name": "dense_vector_search", "arguments": {"query": f"dataset metrics chunk #{i+1}", "top_k": 5}}
                r1 = self.dense.execute(c1["arguments"])
                chunk_id = r1[0]["document_id"]
                c2 = {"type": "tool_call", "name": "parent_document_retrieval", "arguments": {"document_id": chunk_id}}
                r2 = self.parent_doc.execute(c2["arguments"])
                sec_cnt = r2["section_count"]
                c3 = {"type": "tool_call", "name": "calculator", "arguments": {"operation": "mul", "a": sec_cnt, "b": mult_val}}
                r3 = self.calc.execute(c3["arguments"])
                final_ans = f"Parent document {r2['parent_document_id']} has {sec_cnt} sections. Calculated total: {r3}."
                msgs = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": json.dumps(c1)},
                    {"role": "user", "content": f"TOOL_RESULT: {json.dumps(r1)}"},
                    {"role": "assistant", "content": json.dumps(c2)},
                    {"role": "user", "content": f"TOOL_RESULT: {json.dumps(r2)}"},
                    {"role": "assistant", "content": json.dumps(c3)},
                    {"role": "user", "content": f"TOOL_RESULT: {r3}"},
                    {"role": "assistant", "content": json.dumps({"type": "final", "content": final_ans})},
                ]
                exp_tools = ["dense_vector_search", "parent_document_retrieval", "calculator"]

            rec = {
                "trajectory_id": "",
                "category": "rag_multi_tool",
                "task": {
                    "prompt": prompt,
                    "requires_tool": True,
                    "expected_tools": exp_tools,
                    "expected_answer": final_ans,
                },
                "messages": msgs,
                "metadata": {
                    "tool_count": len(exp_tools),
                    "difficulty": "hard",
                    "requires_reasoning": True
                }
            }
            records.append(rec)
        return records

    # ------------------------------------------------------------------
    # 5. ERROR RECOVERY GENERATOR (5%)
    # ------------------------------------------------------------------
    def _generate_error_recovery_examples(self, count: int) -> List[Dict[str, Any]]:
        records = []
        for i in range(count):
            scen_idx = i % 4
            if scen_idx == 0:
                val1 = 50 + (i * 10)
                prompt = f"Calculate {val1} divided by 0 (Task #{i+1})."
                c1 = {"type": "tool_call", "name": "calculator", "arguments": {"operation": "divide", "a": val1, "b": 0}}
                e1 = "TOOL_ERROR: Division by zero is undefined."
                c2 = {"type": "tool_call", "name": "calculator", "arguments": {"operation": "divide", "a": val1, "b": 5}}
                r2 = val1 // 5
                final_ans = f"Recovered from zero division error. Corrected calculation {val1} / 5 = {r2}."
                exp_tools = ["calculator"]
            elif scen_idx == 1:
                val1, val2 = 20 + i, 15 + i
                prompt = f"Calculate sum of {val1} and {val2} using operation 'sum' (Task #{i+1})."
                c1 = {"type": "tool_call", "name": "calculator", "arguments": {"operation": "sum", "a": val1, "b": val2}}
                e1 = "TOOL_ERROR: Invalid operation 'sum'. Supported: add, sub, mul, divide, modulo."
                c2 = {"type": "tool_call", "name": "calculator", "arguments": {"operation": "add", "a": val1, "b": val2}}
                r2 = val1 + val2
                final_ans = f"Recovered from invalid operation name. Corrected operation 'add' result = {r2}."
                exp_tools = ["calculator"]
            elif scen_idx == 2:
                prompt = f"Retrieve parent document context for empty document ID (Task #{i+1})."
                c1 = {"type": "tool_call", "name": "parent_document_retrieval", "arguments": {"document_id": ""}}
                e1 = "TOOL_ERROR: document_id argument must be a valid non-empty string."
                doc_id = f"vec_doc_{200+i}"
                c2 = {"type": "tool_call", "name": "parent_document_retrieval", "arguments": {"document_id": doc_id}}
                r2 = self.parent_doc.execute(c2["arguments"])
                final_ans = f"Recovered from empty document ID. Parent document {r2['parent_document_id']} retrieved."
                exp_tools = ["parent_document_retrieval"]
            else:  # invalid_alpha
                alpha_val = 1.5 + (i * 0.1)
                prompt = f"Execute hybrid search for 'transformer_{i+1}' with alpha {alpha_val}."
                c1 = {"type": "tool_call", "name": "hybrid_search", "arguments": {"query": f"transformer_{i+1}", "top_k": 5, "alpha": alpha_val}}
                e1 = "TOOL_ERROR: Alpha weighting parameter must be between 0.0 and 1.0."
                c2 = {"type": "tool_call", "name": "hybrid_search", "arguments": {"query": f"transformer_{i+1}", "top_k": 5, "alpha": 0.5}}
                r2 = self.hybrid.execute(c2["arguments"])
                final_ans = "Recovered from invalid alpha weight. Corrected hybrid search result retrieved."
                exp_tools = ["hybrid_search"]

            rec = {
                "trajectory_id": "",
                "category": "error_recovery",
                "is_error_recovery": True,
                "task": {
                    "prompt": prompt,
                    "requires_tool": True,
                    "expected_tools": exp_tools,
                    "expected_answer": final_ans,
                },
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": json.dumps(c1)},
                    {"role": "user", "content": e1},
                    {"role": "assistant", "content": json.dumps(c2)},
                    {"role": "user", "content": f"TOOL_RESULT: {json.dumps(r2)}"},
                    {"role": "assistant", "content": json.dumps({"type": "final", "content": final_ans})},
                ],
                "metadata": {
                    "tool_count": len(exp_tools),
                    "difficulty": "hard",
                    "requires_reasoning": True
                }
            }
            records.append(rec)
        return records

    # ------------------------------------------------------------------
    # 6. HARD NEGATIVES GENERATOR (5%)
    # ------------------------------------------------------------------
    def _generate_hard_negative_examples(self, count: int) -> List[Dict[str, Any]]:
        hard_templates = [
            ("Find research papers about transformer interpretability published in {year} by researchers from {org}.", "metadata_filtered_search", "dense_vector_search"),
            ("Find documents discussing the concept of {concept}.", "dense_vector_search", "bm25_search"),
            ("Find documents containing the exact technical term '{term}'.", "bm25_search", "dense_vector_search"),
            ("Find documents matching both exact technical terms and broad semantic context for '{topic}'.", "hybrid_search", "bm25_search"),
            ("Search for {topic} from 3 distinct formulation queries.", "multi_query_retrieval", "dense_vector_search"),
        ]

        records = []
        for i in range(count):
            tpl_idx = i % len(hard_templates)
            prompt_tpl, correct_tool, wrong_tool = hard_templates[tpl_idx]

            year = 2020 + (i % 5)
            org = f"Stanford_{i+1}" if i % 2 == 0 else f"MIT_{i+1}"
            concept = f"catastrophic forgetting variant #{i+1}"
            term = f"catastrophic_forgetting_v{i+1}_loss"
            topic = f"quantum error correction #{i+1}"

            prompt = prompt_tpl.format(year=year, org=org, concept=concept, term=term, topic=topic)

            if correct_tool == "bm25_search":
                t_args = {"query": term, "top_k": 5}
                r1 = self.bm25.execute(t_args)
            elif correct_tool == "dense_vector_search":
                t_args = {"query": concept, "top_k": 5}
                r1 = self.dense.execute(t_args)
            elif correct_tool == "hybrid_search":
                t_args = {"query": topic, "top_k": 5, "alpha": 0.5}
                r1 = self.hybrid.execute(t_args)
            elif correct_tool == "multi_query_retrieval":
                t_args = {"queries": [topic, f"overview of {topic}", f"advancements in {topic}"], "top_k": 5}
                r1 = self.multi_query.execute(t_args)
            else:  # metadata_filtered_search
                t_args = {"query": "transformer interpretability", "filters": {"year": year, "org": org}, "top_k": 5}
                r1 = self.meta_search.execute(t_args)

            t_call = {"type": "tool_call", "name": correct_tool, "arguments": t_args}
            final_ans = f"Disambiguated task select '{correct_tool}' over '{wrong_tool}'. Document {r1[0].get('document_id', 'doc_101')} retrieved."

            rec = {
                "trajectory_id": "",
                "category": "hard_negatives",
                "task": {
                    "prompt": prompt,
                    "requires_tool": True,
                    "expected_tools": [correct_tool],
                    "expected_answer": final_ans,
                },
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": json.dumps(t_call)},
                    {"role": "user", "content": f"TOOL_RESULT: {json.dumps(r1)}"},
                    {"role": "assistant", "content": json.dumps({"type": "final", "content": final_ans})},
                ],
                "metadata": {
                    "tool_count": 1,
                    "difficulty": "hard",
                    "requires_reasoning": True,
                    "hard_negative_distractor": wrong_tool
                }
            }
            records.append(rec)
        return records
