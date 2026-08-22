import pytest
from src.model import MockModel
from src.tools import ToolRegistry
from src.harness import AgentHarness
from src.evaluator import Evaluator
from tools.example_tools import CalculatorTool, EchoTool


def test_model_instantiation_once_per_evaluation_run():
    """
    Diagnostic test confirming the model backend is instantiated exactly once
    and reused across all evaluation tasks.
    """
    initial_count = MockModel.instantiation_count

    # 1. Instantiate single model backend
    model_instance = MockModel()
    assert MockModel.instantiation_count == initial_count + 1

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(EchoTool())

    # 2. Instantiate harness with single model instance
    harness = AgentHarness(model=model_instance, tool_registry=registry)

    # 3. Create batch of 10 tasks
    tasks = [
        {"id": f"t_{i}", "category": "single_tool", "prompt": f"Calculate {i} + 1", "expected_answer": str(i + 1), "requires_tool": True}
        for i in range(10)
    ]

    evaluator = Evaluator()
    summary = evaluator.evaluate_batch(tasks, harness)

    # 4. Assert model instantiation count did NOT increase during batch run
    assert MockModel.instantiation_count == initial_count + 1, "Model was re-instantiated during evaluation batch!"

    # 5. Assert harness model instance reference remains identical
    assert harness.model is model_instance
    assert id(harness.model) == id(model_instance)

    # 6. Verify model generate was called multiple times on the SAME instance
    assert model_instance.call_count >= 10
