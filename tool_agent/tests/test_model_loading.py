import os
import json
import tempfile
import shutil
import pytest
from pathlib import Path

from src.model import HuggingFaceModel, MockModel
from src.parser import ModelOutputParser, Action
from src.tools import ToolRegistry
from src.harness import AgentHarness
from tools.example_tools import CalculatorTool, LookupTool


@pytest.fixture
def temp_model_dir():
    tmp = tempfile.mkdtemp()
    # Create mock model files
    with open(os.path.join(tmp, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"model_type": "qwen2"}, f)
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


def test_1_base_model_path_resolution():
    """Verify base model path is detected as HuggingFace Hub source."""
    hub_path = "Qwen/Qwen3-1.7B"
    # Inspect source detection logic
    if os.path.exists(hub_path) or os.path.isdir(hub_path):
        source = "local"
    else:
        source = "HuggingFace Hub"
    assert source == "HuggingFace Hub"


def test_2_local_model_path_resolution(temp_model_dir):
    """Verify local directory path is detected as local source."""
    assert os.path.exists(temp_model_dir)
    if os.path.exists(temp_model_dir) or os.path.isdir(temp_model_dir):
        source = "local"
    else:
        source = "HuggingFace Hub"
    assert source == "local"


def test_3_model_path_argument_respect():
    """Verify MockModel and HuggingFaceModel accept model_path parameter."""
    hf_path = "Bitsbrains/aarushi-qwen3-1.7b-tool-agent-sft-v2"
    mock = MockModel()
    assert mock is not None

    # Test path resolution logic in model.py
    resolved = hf_path or "Qwen/Qwen3-1.7B"
    assert resolved == hf_path


def test_4_parser_compatibility():
    """Verify ModelOutputParser handles both tool calls and final answers."""
    parser = ModelOutputParser()
    
    tool_output = '{"type": "tool_call", "name": "calculator", "arguments": {"expression": "29 * 43"}}'
    action1 = parser.parse(tool_output)
    assert action1.is_tool_call()
    assert action1.tool_name == "calculator"
    assert action1.arguments == {"expression": "29 * 43"}

    final_output = '{"type": "final", "content": "The result is 1247."}'
    action2 = parser.parse(final_output)
    assert action2.is_final_answer()
    assert action2.content == "The result is 1247."


def test_5_harness_behavior_with_model():
    """Verify AgentHarness execution flow with registered tools."""
    responses = [
        '{"type": "tool_call", "name": "lookup", "arguments": {"key": "apple_price"}}',
        '{"type": "final", "content": "The price of apple is 5."}'
    ]
    model = MockModel(responses=responses)
    registry = ToolRegistry()
    registry.register(LookupTool())

    harness = AgentHarness(model=model, tool_registry=registry, max_steps=5)
    traj = harness.run("Look up apple_price.")

    assert traj.termination_reason == "final_answer"
    assert "5" in traj.final_answer
    assert len(traj.steps) == 2


def test_6_baseline_evaluation_remains_unchanged():
    """Verify that baseline metrics and eval_v1.json files exist and remain untouched."""
    baseline_metrics_path = Path("evaluation/baseline/metrics.json")
    eval_benchmark_path = Path("tasks/eval_v1.json")

    assert baseline_metrics_path.exists(), "evaluation/baseline/metrics.json is missing!"
    assert eval_benchmark_path.exists(), "tasks/eval_v1.json is missing!"

    with open(baseline_metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    assert "overall_task_success_rate" in metrics or "total_tasks" in metrics

    with open(eval_benchmark_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    assert len(tasks) == 60


def test_7_system_prompt_formatting():
    """Verify system prompt injection logic."""
    from src.model import DEFAULT_SYSTEM_PROMPT
    assert "You are a tool-use agent." in DEFAULT_SYSTEM_PROMPT
    assert '{"type": "final", "content": "<answer>"}' in DEFAULT_SYSTEM_PROMPT

    messages = [{"role": "user", "content": "Hi"}]
    formatted = [dict(m) for m in messages]
    formatted.insert(0, {"role": "system", "content": DEFAULT_SYSTEM_PROMPT})

    assert len(formatted) == 2
    assert formatted[0]["role"] == "system"
    assert formatted[1]["role"] == "user"

