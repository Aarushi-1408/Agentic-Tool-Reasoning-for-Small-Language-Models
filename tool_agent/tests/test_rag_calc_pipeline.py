import os
import sys
import json
import pytest
from pathlib import Path

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
from src.generation.rag_calc_generator import RAGCalcDatasetGenerator
from src.generation.rag_calc_validator import RAGCalcTrajectoryValidator
from src.training.sft_dataset import load_sft_dataset


def test_tool_schemas_and_execution():
    bm25 = BM25SearchTool()
    dense = DenseVectorSearchTool()
    hybrid = HybridSearchTool()
    reranker = RerankerTool()
    mq = MultiQueryRetrievalTool()
    meta = MetadataFilteredSearchTool()
    parent = ParentDocumentRetrievalTool()
    calc = CalculatorToolV2()

    # Schemas check
    assert bm25.name == "bm25_search"
    assert dense.name == "dense_vector_search"
    assert hybrid.name == "hybrid_search"
    assert reranker.name == "reranker"
    assert mq.name == "multi_query_retrieval"
    assert meta.name == "metadata_filtered_search"
    assert parent.name == "parent_document_retrieval"
    assert calc.name == "calculator"

    # Execution check
    res_bm25 = bm25.execute({"query": "quantum error correction", "top_k": 3})
    assert len(res_bm25) == 3

    res_calc = calc.execute({"operation": "add", "a": 10, "b": 20})
    assert res_calc == 30

    res_mod = calc.execute({"operation": "modulo", "a": 17, "b": 5})
    assert res_mod == 2


def test_dataset_generation_and_validation():
    generator = RAGCalcDatasetGenerator(seed=42)
    records = generator.generate_dataset(total_count=100)

    assert len(records) == 100
    validator = RAGCalcTrajectoryValidator()
    val_res = validator.validate_batch(records)

    assert val_res["pass_rate"] == 1.0
    assert val_res["duplicate_rate"] == 0.0


def test_sft_loader_compatibility():
    generator = RAGCalcDatasetGenerator(seed=42)
    records = generator.generate_dataset(total_count=10)

    # Verify messages format matches SFTDataset format requirements
    for r in records:
        assert "messages" in r
        assert isinstance(r["messages"], list)
        for msg in r["messages"]:
            assert "role" in msg
            assert "content" in msg
