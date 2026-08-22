import json
import pytest
from src.model import MockModel
from src.tools import ToolRegistry
from src.harness import AgentHarness
from tools.example_tools import CalculatorTool, EchoTool


def test_harness_single_tool_call():
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    # Scripted model responses: Step 0 tool call, Step 1 final answer
    responses = [
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "25 * 37"}}),
        json.dumps({"type": "final", "content": "The answer is 925."}),
    ]
    model = MockModel(responses=responses)
    harness = AgentHarness(model=model, tool_registry=registry, max_steps=10)

    trajectory = harness.run("Calculate 25 * 37.")

    assert trajectory.termination_reason == "final_answer"
    assert trajectory.final_answer == "The answer is 925."
    assert len(trajectory.steps) == 2

    # Step 0 details
    step0 = trajectory.steps[0]
    assert step0.action_type == "tool_call"
    assert step0.tool_name == "calculator"
    assert step0.arguments == {"expression": "25 * 37"}
    assert "925" in step0.tool_result

    # Step 1 details
    step1 = trajectory.steps[1]
    assert step1.action_type == "final"


def test_harness_multi_step_tool_calls():
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(EchoTool())

    responses = [
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "10 + 20"}}),
        json.dumps({"type": "tool_call", "name": "echo", "arguments": {"message": "Result is 30"}}),
        json.dumps({"type": "final", "content": "Completed multi-step task."}),
    ]
    model = MockModel(responses=responses)
    harness = AgentHarness(model=model, tool_registry=registry, max_steps=10)

    trajectory = harness.run("Calculate 10 + 20 then echo result.")

    assert trajectory.termination_reason == "final_answer"
    assert len(trajectory.steps) == 3
    assert trajectory.steps[0].tool_name == "calculator"
    assert trajectory.steps[1].tool_name == "echo"
    assert trajectory.steps[2].action_type == "final"


def test_harness_no_tool_direct_final_answer():
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    responses = [
        "The capital of France is Paris."
    ]
    model = MockModel(responses=responses)
    harness = AgentHarness(model=model, tool_registry=registry, max_steps=10)

    trajectory = harness.run("What is the capital of France?")

    assert trajectory.termination_reason == "final_answer"
    assert trajectory.final_answer == "The capital of France is Paris."
    assert len(trajectory.steps) == 1
    assert trajectory.steps[0].action_type == "final"


def test_harness_unknown_tool_handling():
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    # Model requests unknown tool 'fake_tool', harness returns TOOL_ERROR, then model self-corrects
    responses = [
        json.dumps({"type": "tool_call", "name": "fake_tool", "arguments": {"foo": "bar"}}),
        json.dumps({"type": "final", "content": "Recovered from unknown tool error."}),
    ]
    model = MockModel(responses=responses)
    harness = AgentHarness(model=model, tool_registry=registry, max_steps=10)

    trajectory = harness.run("Do something with fake tool.")

    assert len(trajectory.steps) == 2
    step0 = trajectory.steps[0]
    assert step0.action_type == "tool_call"
    assert step0.tool_name == "fake_tool"
    assert step0.error is not None
    assert "Unknown tool: 'fake_tool'" in step0.error
    assert "TOOL_ERROR" in step0.tool_result


def test_harness_invalid_argument_handling():
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    # Model supplies invalid integer arg for expression
    responses = [
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": 12345}}),
        json.dumps({"type": "final", "content": "Stopped after invalid arg observation."}),
    ]
    model = MockModel(responses=responses)
    harness = AgentHarness(model=model, tool_registry=registry, max_steps=10)

    trajectory = harness.run("Calculate invalid expression.")

    step0 = trajectory.steps[0]
    assert step0.error is not None
    assert "Invalid arguments for tool 'calculator'" in step0.error


def test_harness_max_steps_exceeded():
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    # Infinite tool calling loop
    responses = [
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "1 + 1"}})
    ] * 5

    model = MockModel(responses=responses)
    harness = AgentHarness(model=model, tool_registry=registry, max_steps=3)

    trajectory = harness.run("Loop forever.")

    assert trajectory.termination_reason == "max_steps_exceeded"
    assert len(trajectory.steps) == 3
