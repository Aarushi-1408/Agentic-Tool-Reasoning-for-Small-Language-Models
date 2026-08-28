"""
RAG & Calculator Tool implementations for Synthetic Trajectory Generation and Agent Execution.
Defines explicit schemas and execution logic for 8 tools:
1. BM25 Search (bm25_search)
2. Dense Vector Search (dense_vector_search)
3. Hybrid Search (hybrid_search)
4. Reranker (reranker)
5. Multi-Query Retrieval (multi_query_retrieval)
6. Metadata/Filtered Search (metadata_filtered_search)
7. Parent-Document Retrieval (parent_document_retrieval)
8. Calculator (calculator)
"""

import ast
import operator as op
from typing import Dict, Any, List, Optional
from src.tools import Tool


# ----------------------------------------------------------------------
# TOOL 1: BM25 SEARCH (bm25_search)
# ----------------------------------------------------------------------
class BM25SearchTool(Tool):
    """
    Lexical keyword-based retrieval. Best for exact terms, names, IDs, rare keywords.
    """

    def __init__(self):
        super().__init__(
            name="bm25_search",
            description="Lexical keyword-based retrieval. Use when exact terms, proper names, identifiers, or specific technical phrases matter.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Exact text or keyword query string to search."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top lexical documents to retrieve (default: 5)."
                    }
                },
                "required": ["query"]
            }
        )

    def execute(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        query = arguments.get("query", "")
        top_k = int(arguments.get("top_k", 5))
        if not query:
            raise ValueError("Query string cannot be empty.")
        
        # Deterministic mock lexical search response
        return [
            {
                "document_id": f"bm25_doc_{i+101}",
                "title": f"Lexical Record for '{query[:30]}'",
                "text": f"Exact lexical match snippet #{i+1} for query: {query}. Contains key terms and references.",
                "score": round(0.95 - (i * 0.08), 2)
            }
            for i in range(min(top_k, 5))
        ]


# ----------------------------------------------------------------------
# TOOL 2: DENSE VECTOR SEARCH (dense_vector_search)
# ----------------------------------------------------------------------
class DenseVectorSearchTool(Tool):
    """
    Semantic similarity retrieval. Best for conceptual questions, paraphrased descriptions.
    """

    def __init__(self):
        super().__init__(
            name="dense_vector_search",
            description="Semantic similarity retrieval using vector embeddings. Use for conceptual queries, natural language descriptions, or topic searches where exact keywords may vary.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Conceptual or natural language search query."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top semantic documents to retrieve (default: 5)."
                    }
                },
                "required": ["query"]
            }
        )

    def execute(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        query = arguments.get("query", "")
        top_k = int(arguments.get("top_k", 5))
        if not query:
            raise ValueError("Query string cannot be empty.")

        return [
            {
                "document_id": f"vec_doc_{i+201}",
                "title": f"Semantic Analysis of '{query[:30]}'",
                "text": f"Conceptual research paper chunk #{i+1} related to {query}. Discusses core underlying principles.",
                "score": round(0.92 - (i * 0.06), 2)
            }
            for i in range(min(top_k, 5))
        ]


# ----------------------------------------------------------------------
# TOOL 3: HYBRID SEARCH (hybrid_search)
# ----------------------------------------------------------------------
class HybridSearchTool(Tool):
    """
    Combines lexical BM25 and semantic vector retrieval with weighting alpha.
    """

    def __init__(self):
        super().__init__(
            name="hybrid_search",
            description="Hybrid search combining lexical keyword matching (BM25) and semantic vector similarity. Use when both exact technical terms and broad conceptual relevance are required.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query text."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of documents to retrieve (default: 5)."
                    },
                    "alpha": {
                        "type": "number",
                        "description": "Weight parameter between 0.0 (lexical BM25) and 1.0 (dense vector). Default: 0.5."
                    }
                },
                "required": ["query"]
            }
        )

    def execute(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        query = arguments.get("query", "")
        top_k = int(arguments.get("top_k", 5))
        alpha = float(arguments.get("alpha", 0.5))
        if not query:
            raise ValueError("Query string cannot be empty.")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("Alpha weighting parameter must be between 0.0 and 1.0.")

        return [
            {
                "document_id": f"hyb_doc_{i+301}",
                "title": f"Hybrid Result for '{query[:30]}'",
                "text": f"Hybrid retrieved passage #{i+1} balancing keywords and semantic context for query: {query}. (alpha={alpha})",
                "score": round(0.94 - (i * 0.05), 2)
            }
            for i in range(min(top_k, 5))
        ]


# ----------------------------------------------------------------------
# TOOL 4: RERANKER (reranker)
# ----------------------------------------------------------------------
class RerankerTool(Tool):
    """
    Re-ranks an existing set of retrieved documents according to relevance. Should occur AFTER retrieval.
    """

    def __init__(self):
        super().__init__(
            name="reranker",
            description="Re-ranks an existing list of retrieved candidate documents based on deep semantic relevance to the query. Must be called AFTER initial retrieval.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Target query string for scoring candidate documents."
                    },
                    "documents": {
                        "type": "array",
                        "items": {
                            "type": "object"
                        },
                        "description": "List of candidate document objects to rerank."
                    }
                },
                "required": ["query", "documents"]
            }
        )

    def execute(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        query = arguments.get("query", "")
        documents = arguments.get("documents", [])
        if not isinstance(documents, list) or len(documents) == 0:
            raise ValueError("Documents argument must be a non-empty list of candidate documents.")

        # Re-sort documents deterministically and update reranker score
        reranked = []
        for idx, doc in enumerate(documents):
            doc_copy = dict(doc) if isinstance(doc, dict) else {"text": str(doc)}
            doc_copy["rerank_score"] = round(0.98 - (idx * 0.04), 2)
            reranked.append(doc_copy)

        return reranked


# ----------------------------------------------------------------------
# TOOL 5: MULTI-QUERY RETRIEVAL (multi_query_retrieval)
# ----------------------------------------------------------------------
class MultiQueryRetrievalTool(Tool):
    """
    Executes retrieval over multiple query formulations for broad recall.
    """

    def __init__(self):
        super().__init__(
            name="multi_query_retrieval",
            description="Executes search across multiple query formulations to gather complementary evidence for complex or multi-faceted questions.",
            parameters={
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "List of distinct query formulations to retrieve evidence for."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top results per query (default: 5)."
                    }
                },
                "required": ["queries"]
            }
        )

    def execute(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        queries = arguments.get("queries", [])
        top_k = int(arguments.get("top_k", 5))
        if not isinstance(queries, list) or len(queries) == 0:
            raise ValueError("Queries argument must be a non-empty list of strings.")

        results = []
        for q_idx, q in enumerate(queries):
            results.append({
                "document_id": f"mq_doc_{q_idx+401}",
                "query_used": q,
                "title": f"Perspective #{q_idx+1} for '{q[:25]}'",
                "text": f"Retrieved context addressing perspective formulation: {q}.",
                "score": 0.90
            })
        return results[:top_k]


# ----------------------------------------------------------------------
# TOOL 6: METADATA/FILTERED SEARCH (metadata_filtered_search)
# ----------------------------------------------------------------------
class MetadataFilteredSearchTool(Tool):
    """
    Search documents with explicit metadata constraints (year, author, category, etc.).
    """

    def __init__(self):
        super().__init__(
            name="metadata_filtered_search",
            description="Search documents constrained by explicit metadata filters such as publication year, author name, document category, or department.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query text."
                    },
                    "filters": {
                        "type": "object",
                        "description": "Dictionary of metadata constraints (e.g. {'year': 2024, 'author': 'Smith', 'category': 'AI'})."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top filtered documents to retrieve (default: 5)."
                    }
                },
                "required": ["query", "filters"]
            }
        )

    def execute(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        query = arguments.get("query", "")
        filters = arguments.get("filters", {})
        top_k = int(arguments.get("top_k", 5))
        if not query:
            raise ValueError("Query string cannot be empty.")
        if not isinstance(filters, dict) or len(filters) == 0:
            raise ValueError("Filters argument must be a non-empty metadata key-value object.")

        return [
            {
                "document_id": f"meta_doc_{i+501}",
                "title": f"Filtered Paper ({filters.get('year', '2024')})",
                "text": f"Document snippet matching query '{query}' and metadata constraints {filters}.",
                "metadata": filters,
                "score": round(0.96 - (i * 0.05), 2)
            }
            for i in range(min(top_k, 5))
        ]


# ----------------------------------------------------------------------
# TOOL 7: PARENT-DOCUMENT RETRIEVAL (parent_document_retrieval)
# ----------------------------------------------------------------------
class ParentDocumentRetrievalTool(Tool):
    """
    Retrieve the full parent document/context given a specific document_id chunk.
    """

    def __init__(self):
        super().__init__(
            name="parent_document_retrieval",
            description="Retrieve the complete parent document context associated with a previously retrieved passage or document chunk ID. Must be called as a follow-up operation.",
            parameters={
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "The target document chunk ID (e.g., 'doc_184' or 'hyb_doc_301')."
                    }
                },
                "required": ["document_id"]
            }
        )

    def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        document_id = arguments.get("document_id", "")
        if not document_id or not isinstance(document_id, str):
            raise ValueError("document_id argument must be a valid non-empty string.")

        return {
            "parent_document_id": f"parent_{document_id}",
            "original_chunk_id": document_id,
            "full_title": f"Full Parent Document Specification for {document_id}",
            "full_text": f"Complete multi-page parent document context associated with chunk {document_id}. Contains full background details, figures, methodology, and numerical metrics.",
            "section_count": 8,
            "total_tokens": 3450
        }


# ----------------------------------------------------------------------
# TOOL 8: CALCULATOR (calculator)
# ----------------------------------------------------------------------
class CalculatorToolV2(Tool):
    """
    Structured arithmetic calculator supporting: add, sub, mul, divide, modulo.
    """

    def __init__(self):
        super().__init__(
            name="calculator",
            description="Perform structured arithmetic calculations. Supported operations: 'add', 'sub', 'mul', 'divide', 'modulo'.",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "sub", "mul", "divide", "modulo"],
                        "description": "Arithmetic operation to perform."
                    },
                    "a": {
                        "type": "number",
                        "description": "First numeric operand."
                    },
                    "b": {
                        "type": "number",
                        "description": "Second numeric operand."
                    }
                },
                "required": ["operation", "a", "b"]
            }
        )

    def execute(self, arguments: Dict[str, Any]) -> float:
        op_name = arguments.get("operation")
        a = arguments.get("a")
        b = arguments.get("b")

        if op_name not in ["add", "sub", "mul", "divide", "modulo"]:
            raise ValueError(f"Invalid operation '{op_name}'. Supported: add, sub, mul, divide, modulo.")
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            raise TypeError(f"Operands 'a' and 'b' must be numeric (got {type(a).__name__}, {type(b).__name__}).")

        if op_name == "add":
            res = a + b
        elif op_name == "sub":
            res = a - b
        elif op_name == "mul":
            res = a * b
        elif op_name == "divide":
            if b == 0:
                raise ValueError("Division by zero is undefined.")
            res = a / b
        elif op_name == "modulo":
            if b == 0:
                raise ValueError("Modulo by zero is undefined.")
            res = a % b
        else:
            raise ValueError(f"Unknown operation '{op_name}'.")

        if isinstance(res, float) and res.is_integer():
            return int(res)
        return round(res, 4)
