import json
import os
import pytest
from src.tools import ToolRegistry
from src.harness import AgentHarness
from src.trajectory import Trajectory, TrajectoryStep
from src.generation.teacher import (
    TeacherInterface,
    HuggingFaceTeacher,
    APITeacher,
    MockTeacher,
    create_teacher,
)
from src.generation.validator import TrajectoryValidator
from tools.example_tools import CalculatorTool, EchoTool, LookupTool, StringManipulatorTool


def test_1_teacher_loads_once():
    """Verify that teacher instantiation lifecycle occurs once per generation session."""
    initial_count = MockTeacher.instantiation_count
    teacher = MockTeacher(model_name="test-teacher-v1")
    assert MockTeacher.instantiation_count == initial_count + 1

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    harness = AgentHarness(model=teacher, tool_registry=registry, max_steps=5)

    # Execute multiple tasks with the SAME single teacher instance
    for i in range(3):
        harness.run(f"Calculate {i} + 1.")

    # Model was instantiated only ONCE
    assert MockTeacher.instantiation_count == initial_count + 1
    assert teacher.call_count >= 3


def test_2_no_tool_trajectory():
    """Verify trajectory generated when model provides direct final answer without calling tools."""
    responses = ["The capital of Japan is Tokyo."]
    teacher = MockTeacher(responses=responses)

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    harness = AgentHarness(model=teacher, tool_registry=registry, max_steps=5)

    trajectory = harness.run("What is the capital of Japan?", task_id="test_no_tool")

    assert trajectory.termination_reason == "final_answer"
    assert trajectory.final_answer == "The capital of Japan is Tokyo."
    assert len(trajectory.steps) == 1
    assert trajectory.steps[0].action_type == "final"

    # Validate
    validator = TrajectoryValidator(registered_tools={"calculator"})
    val_res = validator.validate(trajectory.to_dict())
    assert val_res["valid"] is True


def test_3_single_tool_trajectory():
    """Verify single-tool call trajectory generation and actual execution through harness."""
    responses = [
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "144 / 12"}}),
        json.dumps({"type": "final", "content": "144 divided by 12 is 12."}),
    ]
    teacher = MockTeacher(responses=responses)

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    harness = AgentHarness(model=teacher, tool_registry=registry, max_steps=5)

    trajectory = harness.run("Calculate 144 / 12.", task_id="test_single_tool")

    assert trajectory.termination_reason == "final_answer"
    assert len(trajectory.steps) == 2
    assert trajectory.steps[0].action_type == "tool_call"
    assert trajectory.steps[0].tool_name == "calculator"
    assert trajectory.steps[0].arguments == {"expression": "144 / 12"}
    assert "12" in trajectory.steps[0].tool_result
    assert trajectory.steps[1].action_type == "final"

    validator = TrajectoryValidator(registered_tools={"calculator"})
    assert validator.validate(trajectory.to_dict())["valid"] is True


def test_4_two_step_trajectory():
    """Verify multi-step tool call chaining (Lookup -> Calculator -> Final Answer)."""
    responses = [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "apple_price"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "5 * 3"}}),
        json.dumps({"type": "final", "content": "The total price for 3 apples is 15."}),
    ]
    teacher = MockTeacher(responses=responses)

    registry = ToolRegistry()
    registry.register(LookupTool())
    registry.register(CalculatorTool())
    harness = AgentHarness(model=teacher, tool_registry=registry, max_steps=5)

    trajectory = harness.run("Look up apple price and calculate 3 apples.", task_id="test_two_step")

    assert trajectory.termination_reason == "final_answer"
    assert len(trajectory.steps) == 3
    assert trajectory.steps[0].tool_name == "lookup"
    assert trajectory.steps[1].tool_name == "calculator"
    assert trajectory.steps[2].action_type == "final"


def test_5_tool_error_and_recovery():
    """Verify trajectory when model makes an error (e.g. unknown tool), receives observation feedback, and recovers."""
    responses = [
        json.dumps({"type": "tool_call", "name": "non_existent_tool", "arguments": {"foo": "bar"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "20 + 30"}}),
        json.dumps({"type": "final", "content": "Recovered and calculated 50."}),
    ]
    teacher = MockTeacher(responses=responses)

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    harness = AgentHarness(model=teacher, tool_registry=registry, max_steps=5)

    trajectory = harness.run("Task with tool recovery.", task_id="test_recovery")

    assert len(trajectory.steps) == 3
    # Step 0 was invalid tool error
    assert trajectory.steps[0].action_type == "tool_call"
    assert trajectory.steps[0].tool_name == "non_existent_tool"
    assert trajectory.steps[0].error is not None
    assert "Unknown tool" in trajectory.steps[0].error
    # Step 1 was successful tool call
    assert trajectory.steps[1].tool_name == "calculator"
    # Step 2 was final answer
    assert trajectory.steps[2].action_type == "final"


def test_6_invalid_tool_call_handling():
    """Verify trajectory when invalid tool arguments schema error occurs."""
    responses = [
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": 9999}}),
        json.dumps({"type": "final", "content": "Handled invalid arg."}),
    ]
    teacher = MockTeacher(responses=responses)

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    harness = AgentHarness(model=teacher, tool_registry=registry, max_steps=5)

    trajectory = harness.run("Invalid arg test.", task_id="test_invalid_arg")

    assert len(trajectory.steps) == 2
    assert trajectory.steps[0].error is not None
    assert "Invalid arguments for tool 'calculator'" in trajectory.steps[0].error


def test_7_maximum_step_termination():
    """Verify safe harness termination when max_steps limit is reached."""
    responses = [
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "1 + 1"}})
    ] * 10
    teacher = MockTeacher(responses=responses)

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    harness = AgentHarness(model=teacher, tool_registry=registry, max_steps=4)

    trajectory = harness.run("Infinite loop task.", task_id="test_max_steps")

    assert trajectory.termination_reason == "max_steps_exceeded"
    assert len(trajectory.steps) == 4


def test_8_trajectory_serialization_deserialization():
    """Verify that trajectories serialize to exact requested dictionary structure and JSON format."""
    traj = Trajectory(
        task_id="train_001",
        prompt="What is 2 + 2?",
        teacher_model="Qwen/Qwen3-8B",
        termination_reason="final_answer",
        final_answer="4",
    )
    step0 = TrajectoryStep(
        step=0,
        model_output='{"type": "tool_call", "name": "calculator", "arguments": {"expression": "2 + 2"}}',
        action_type="tool_call",
        tool_name="calculator",
        arguments={"expression": "2 + 2"},
        tool_result="TOOL_RESULT: 4",
    )
    traj.add_step(step0)

    data = traj.to_dict()
    assert data["task_id"] == "train_001"
    assert data["task"] == "What is 2 + 2?"
    assert data["teacher_model"] == "Qwen/Qwen3-8B"
    assert data["termination_reason"] == "final_answer"
    assert data["success"] is True
    assert len(data["steps"]) == 1
    assert data["steps"][0]["tool_name"] == "calculator"
    assert data["steps"][0]["tool_result"] == "TOOL_RESULT: 4"

    json_str = traj.to_json()
    reconstructed = json.loads(json_str)
    assert reconstructed["task_id"] == "train_001"
    assert reconstructed["final_answer"] == "4"


def test_9_teacher_backend_abstraction():
    """Verify factory creation of different teacher backends."""
    # Mock backend creation
    cfg_mock = {"teacher": {"backend": "mock", "model_name": "mock-8b"}}
    teacher_mock = create_teacher(cfg_mock)
    assert isinstance(teacher_mock, MockTeacher)
    assert teacher_mock.get_info()["teacher_model"] == "mock-8b"

    # API backend creation
    cfg_api = {"teacher": {"backend": "api", "model_name": "gpt-4o", "api_key": "test_key"}}
    teacher_api = create_teacher(cfg_api)
    assert isinstance(teacher_api, APITeacher)
    assert teacher_api.get_info()["teacher_model"] == "gpt-4o"


def test_10_existing_evaluation_benchmark_unchanged():
    """Verify that existing evaluation baseline files and 60-task benchmark are intact and unchanged."""
    baseline_metrics_path = os.path.join("evaluation", "baseline", "metrics.json")
    eval_dataset_path = os.path.join("tasks", "eval_v1.json")

    assert os.path.exists(baseline_metrics_path), "Baseline metrics file must exist"
    assert os.path.exists(eval_dataset_path), "60-task evaluation benchmark dataset must exist"

    with open(baseline_metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert metrics["total_tasks"] == 60
    assert metrics["successful_tasks"] == 48
    assert abs(metrics["accuracy"] - 0.8) < 1e-4

    with open(eval_dataset_path, "r", encoding="utf-8") as f:
        eval_tasks = json.load(f)

    assert len(eval_tasks) == 60
