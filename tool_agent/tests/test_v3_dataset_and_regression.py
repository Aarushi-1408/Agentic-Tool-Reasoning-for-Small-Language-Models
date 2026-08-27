import os
import json
import pytest
from src.generation.task_generator import TaskGenerator
from src.data.convert_to_sft import SFTConverter
from scripts.check_regression import load_json, format_val


def test_v3_task_generator_composition():
    generator = TaskGenerator(seed=42)
    tasks = generator.generate_tasks(
        num_no_tool=175,
        num_single_tool=115,
        num_two_step=90,
        num_multi_step=90,
        num_error_recovery=30,
        id_prefix="test_v3_",
    )

    assert len(tasks) == 500

    categories = {}
    for t in tasks:
        cat = t["category"]
        categories[cat] = categories.get(cat, 0) + 1

    assert categories["no_tool"] == 175
    assert categories["single_tool"] == 115
    assert categories["two_step"] == 90
    assert categories["multi_step"] == 90
    assert categories["error_recovery"] == 30


def test_versioned_files_exist_and_protected():
    """
    Ensure historical versioned datasets and baseline evaluations are preserved.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    eval_v1_path = os.path.join(project_root, "tasks", "eval_v1.json")
    baseline_metrics_path = os.path.join(project_root, "evaluation", "baseline", "metrics.json")
    sft_v2_metrics_path = os.path.join(project_root, "evaluation", "sft_v2_100", "metrics.json")
    sft_v2_dataset_path = os.path.join(project_root, "data", "sft", "train_v2_100.jsonl")
    sft_v3_dataset_path = os.path.join(project_root, "data", "sft", "train_v3_500.jsonl")

    assert os.path.exists(eval_v1_path)
    assert os.path.exists(baseline_metrics_path)
    assert os.path.exists(sft_v2_metrics_path)
    assert os.path.exists(sft_v2_dataset_path)
    assert os.path.exists(sft_v3_dataset_path)


def test_sft_v3_dataset_format():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sft_v3_path = os.path.join(project_root, "data", "sft", "train_v3_500.jsonl")

    assert os.path.exists(sft_v3_path)

    with open(sft_v3_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    assert len(lines) == 500

    first_example = json.loads(lines[0])
    assert "trajectory_id" in first_example
    assert "messages" in first_example
    assert isinstance(first_example["messages"], list)
    assert len(first_example["messages"]) >= 2
    assert first_example["messages"][0]["role"] == "user"
    assert first_example["messages"][1]["role"] == "assistant"


def test_regression_formatting_helpers():
    assert format_val(0.75, True) == "75.00%"
    assert format_val(2.5, False) == "2.50"
    assert format_val(None) == "N/A"
