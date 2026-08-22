import pytest
from src.tools import Tool, ToolRegistry, ToolResult
from src.tool_schema import generate_tool_schema, validate_arguments
from tools.example_tools import CalculatorTool, EchoTool


def test_tool_schema_generation():
    tool = CalculatorTool()
    schema = tool.to_schema()
    assert schema["name"] == "calculator"
    assert "description" in schema
    assert "parameters" in schema
    assert schema["parameters"]["properties"]["expression"]["type"] == "string"


def test_tool_registry_registration():
    registry = ToolRegistry()
    calc = CalculatorTool()
    registry.register(calc)

    assert registry.has("calculator")
    assert registry.get("calculator") is calc
    assert len(registry.schemas()) == 1
    assert registry.schemas()[0]["name"] == "calculator"


def test_tool_registry_unknown_tool():
    registry = ToolRegistry()
    result = registry.execute("nonexistent_tool", {"arg": "val"})

    assert isinstance(result, ToolResult)
    assert not result.success
    assert "Unknown tool: 'nonexistent_tool'" in result.error
    assert "TOOL_ERROR:" in result.to_observation_string()


def test_tool_argument_validation():
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    # Valid arguments
    valid_res = registry.execute("calculator", {"expression": "10 + 20"})
    assert valid_res.success
    assert valid_res.result == 30

    # Invalid arguments: expression as integer instead of string
    invalid_res = registry.execute("calculator", {"expression": 12345})
    assert not invalid_res.success
    assert "Invalid arguments for tool 'calculator'" in invalid_res.error

    # Missing required parameter
    missing_res = registry.execute("calculator", {})
    assert not missing_res.success
    assert "Invalid arguments for tool 'calculator'" in missing_res.error
