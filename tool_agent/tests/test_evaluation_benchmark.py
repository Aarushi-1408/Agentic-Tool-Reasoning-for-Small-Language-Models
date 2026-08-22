import os
import json
import tempfile
import pytest
from src.tools import ToolRegistry
from src.model import MockModel
from src.harness import AgentHarness
from src.evaluator import Evaluator, EvaluationSummary
from tools.example_tools import CalculatorTool, EchoTool, LookupTool, StringManipulatorTool


def test_eval_v1_dataset_schema():
    dataset_path = os.path.join("tasks", "eval_v1.json")
    assert os.path.exists(dataset_path), f"Dataset file not found at {dataset_path}"

    with open(dataset_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    assert len(tasks) >= 50, f"Expected at least 50 tasks in benchmark, got {len(tasks)}"

    categories = set()
    for task in tasks:
        assert "id" in task
        assert "category" in task
        assert "prompt" in task
        assert "requires_tool" in task
        assert isinstance(task["requires_tool"], bool)
        categories.add(task["category"])

    expected_categories = {"no_tool", "single_tool", "two_step", "multi_step", "tool_selection", "error_recovery"}
    assert expected_categories.issubset(categories), f"Missing categories: {expected_categories - categories}"


def test_evaluator_metrics_calculation():
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(EchoTool())
    registry.register(LookupTool())
    registry.register(StringManipulatorTool())

    mock_model = MockModel()
    harness = AgentHarness(model=mock_model, tool_registry=registry)
    evaluator = Evaluator()

    test_tasks = [
        {
            "id": "test_notool",
            "category": "no_tool",
            "prompt": "What is 2 + 2?",
            "expected_answer": "4",
            "requires_tool": False,
        },
        {
            "id": "test_single",
            "category": "single_tool",
            "prompt": "Calculate 25 * 37.",
            "expected_answer": "925",
            "requires_tool": True,
            "expected_behavior": {"allowed_tools": ["calculator"]},
        },
        {
            "id": "test_twostep",
            "category": "two_step",
            "prompt": "Calculate 50 * 4, then calculate the result divided by 5.",
            "expected_answer": "40",
            "requires_tool": True,
            "expected_behavior": {"allowed_tools": ["calculator"]},
        },
    ]

    summary: EvaluationSummary = evaluator.evaluate_batch(test_tasks, harness)
    assert summary.total_tasks == 3
    assert isinstance(summary.accuracy, float)
    assert summary.avg_steps_per_task > 0
    assert "no_tool" in summary.category_accuracies


def test_trajectory_and_metrics_file_export():
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    mock_model = MockModel()
    harness = AgentHarness(model=mock_model, tool_registry=registry)
    evaluator = Evaluator()

    test_tasks = [
        {
            "id": "t1",
            "category": "single_tool",
            "prompt": "Calculate 10 + 20.",
            "expected_answer": "30",
            "requires_tool": True,
            "expected_behavior": {"allowed_tools": ["calculator"]},
        }
    ]

    summary = evaluator.evaluate_batch(test_tasks, harness)

    with tempfile.TemporaryDirectory() as tmpdir:
        metrics_file = os.path.join(tmpdir, "metrics.json")
        traj_file = os.path.join(tmpdir, "trajectories.jsonl")

        summary.save_metrics(metrics_file)
        summary.save_trajectories_jsonl(traj_file)

        assert os.path.exists(metrics_file)
        assert os.path.exists(traj_file)

        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
            assert "total_tasks" in metrics_data
            assert metrics_data["total_tasks"] == 1

        with open(traj_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 1
            traj_data = json.loads(lines[0])
            assert traj_data["task_id"] == "t1"
