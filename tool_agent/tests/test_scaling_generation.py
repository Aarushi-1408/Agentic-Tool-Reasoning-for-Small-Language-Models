import os
import json
import pytest
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.generation.task_generator import TaskGenerator
from src.generation.teacher import MockTeacher
from src.tools import ToolRegistry
from src.harness import AgentHarness
from tools.example_tools import CalculatorTool, EchoTool, LookupTool, StringManipulatorTool
from src.data.convert_to_sft import SFTConverter


def test_v1_files_immutability():
    """
    Ensure original 25 raw trajectories, 21 SFT dataset records, and 60 eval tasks remain untouched.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    raw_v1_path = os.path.join(project_root, "data", "raw", "teacher_trajectories.jsonl")
    assert os.path.exists(raw_v1_path), "data/raw/teacher_trajectories.jsonl must exist"
    with open(raw_v1_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) == 25, f"Expected exactly 25 raw v1 trajectories, found {len(lines)}"

    sft_v1_path = os.path.join(project_root, "data", "sft", "train.jsonl")
    assert os.path.exists(sft_v1_path), "data/sft/train.jsonl must exist"
    with open(sft_v1_path, "r", encoding="utf-8") as f:
        sft_lines = [line.strip() for line in f if line.strip()]
    assert len(sft_lines) == 21, f"Expected exactly 21 SFT v1 examples, found {len(sft_lines)}"

    eval_tasks_path = os.path.join(project_root, "tasks", "eval_v1.json")
    assert os.path.exists(eval_tasks_path), "tasks/eval_v1.json must exist"
    with open(eval_tasks_path, "r", encoding="utf-8") as f:
        eval_tasks = json.load(f)
    assert len(eval_tasks) == 60, f"Expected exactly 60 eval benchmark tasks, found {len(eval_tasks)}"


def test_category_distribution():
    """
    Verify TaskGenerator respects category target counts.
    """
    generator = TaskGenerator(seed=42)
    tasks = generator.generate_tasks(
        num_no_tool=20,
        num_single_tool=25,
        num_two_step=25,
        num_multi_step=20,
        num_error_recovery=10,
    )
    assert len(tasks) == 100

    cats = {}
    for t in tasks:
        c = t["category"]
        cats[c] = cats.get(c, 0) + 1

    assert cats["no_tool"] == 20
    assert cats["single_tool"] == 25
    assert cats["two_step"] == 25
    assert cats["multi_step"] == 20
    assert cats["error_recovery"] == 10


def test_unique_task_and_trajectory_ids():
    """
    Verify task IDs and trajectory IDs are unique.
    """
    generator = TaskGenerator(seed=42)
    tasks = generator.generate_tasks(
        num_no_tool=10,
        num_single_tool=10,
        num_two_step=10,
        num_multi_step=10,
        num_error_recovery=10,
        id_prefix="train_v2_",
    )
    task_ids = [t["task_id"] for t in tasks]
    assert len(task_ids) == len(set(task_ids)), "Task IDs must be unique"
    assert all(tid.startswith("train_v2_") for tid in task_ids)

    traj_ids = [f"traj_v2_{tid}" for tid in task_ids]
    assert len(traj_ids) == len(set(traj_ids)), "Trajectory IDs must be unique"
    assert all(trid.startswith("traj_v2_") for trid in traj_ids)


def test_no_eval_benchmark_overlap():
    """
    Verify generated training task prompts do not overlap with frozen eval benchmark questions.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    eval_tasks_path = os.path.join(project_root, "tasks", "eval_v1.json")
    with open(eval_tasks_path, "r", encoding="utf-8") as f:
        eval_tasks = json.load(f)
    eval_prompts = set(t["prompt"].strip().lower() for t in eval_tasks)

    generator = TaskGenerator(seed=42)
    train_tasks = generator.generate_tasks(20, 25, 25, 20, 10)
    train_prompts = set(t["prompt"].strip().lower() for t in train_tasks)

    # Check no exact match overlap
    overlap = eval_prompts.intersection(train_prompts)
    assert len(overlap) == 0, f"Training tasks must not overlap with eval questions: {overlap}"


def test_single_teacher_instantiation():
    """
    Verify teacher model is loaded once across multi-task harness execution.
    """
    initial_count = MockTeacher.instantiation_count
    teacher = MockTeacher(model_name="test-single-load")
    assert MockTeacher.instantiation_count == initial_count + 1

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    harness = AgentHarness(model=teacher, tool_registry=registry)

    # Run multiple tasks
    for prompt in ["Calculate 25 * 37.", "What is the capital of France?"]:
        harness.run(prompt)

    # Teacher should NOT have been re-instantiated
    assert MockTeacher.instantiation_count == initial_count + 1


def test_overwrite_safety(tmp_path):
    """
    Verify SFTConverter raises FileExistsError if output file exists and overwrite is False.
    """
    input_file = tmp_path / "dummy_raw.jsonl"
    output_file = tmp_path / "dummy_sft.jsonl"
    
    input_file.write_text(json.dumps({
        "task_id": "t1",
        "steps": [{"step": 0, "action_type": "final", "content": "ans"}],
        "termination_reason": "final_answer",
        "final_answer": "ans",
        "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ans"}]
    }) + "\n")
    output_file.write_text("existing content\n")

    converter = SFTConverter()
    with pytest.raises(FileExistsError):
        converter.process_file(
            input_file=str(input_file),
            output_sft_file=str(output_file),
            overwrite=False
        )
