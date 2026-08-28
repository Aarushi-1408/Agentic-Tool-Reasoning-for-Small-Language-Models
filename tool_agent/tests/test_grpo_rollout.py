import pytest
from src.model import MockModel
from src.tools import ToolRegistry
from tools.example_tools import CalculatorTool, LookupTool
from src.rl.rollout import RolloutGenerator, RolloutResult


def test_rollout_generator_mock():
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(LookupTool())

    model = MockModel()
    generator = RolloutGenerator(model=model, tool_registry=registry, max_steps=5)

    task_meta = {
        "id": "test_rollout_001",
        "category": "no_tool",
        "prompt": "What is the capital of France?",
        "requires_tool": False,
        "expected_answer": "Paris",
    }

    result = generator.generate_group(task_meta, group_size=4)
    assert isinstance(result, RolloutResult)
    assert len(result.candidates) == 4
    assert result.task_id == "test_rollout_001"
    
    d = result.to_dict()
    assert "mean_reward" in d
    assert "candidates" in d
