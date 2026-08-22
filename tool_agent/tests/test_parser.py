import pytest
from src.parser import ModelOutputParser, Action


def test_parse_valid_tool_call_json():
    parser = ModelOutputParser()
    raw = '{"type": "tool_call", "name": "calculator", "arguments": {"expression": "25 * 37"}}'
    action = parser.parse(raw)

    assert action.is_tool_call()
    assert action.tool_name == "calculator"
    assert action.arguments == {"expression": "25 * 37"}


def test_parse_valid_tool_call_tag_format():
    parser = ModelOutputParser()
    raw = '<tool_call>{"name": "calculator", "arguments": {"expression": "100 / 4"}}</tool_call>'
    action = parser.parse(raw)

    assert action.is_tool_call()
    assert action.tool_name == "calculator"
    assert action.arguments == {"expression": "100 / 4"}


def test_parse_valid_tool_call_codeblock():
    parser = ModelOutputParser()
    raw = """```json
{
  "type": "tool_call",
  "name": "echo",
  "arguments": {"message": "hello"}
}
```"""
    action = parser.parse(raw)

    assert action.is_tool_call()
    assert action.tool_name == "echo"
    assert action.arguments == {"message": "hello"}


def test_parse_valid_final_answer_json():
    parser = ModelOutputParser()
    raw = '{"type": "final", "content": "The answer is 925."}'
    action = parser.parse(raw)

    assert action.is_final_answer()
    assert action.content == "The answer is 925."


def test_parse_plain_text_final_answer():
    parser = ModelOutputParser()
    raw = "The capital of France is Paris."
    action = parser.parse(raw)

    assert action.is_final_answer()
    assert action.content == "The capital of France is Paris."


def test_parse_malformed_tool_call():
    parser = ModelOutputParser()
    # Missing required 'arguments' key
    raw = '{"type": "tool_call", "name": "calculator"}'
    action = parser.parse(raw)

    assert action.is_parse_error()
    assert "arguments" in action.error_message.lower()


def test_parse_malformed_json_inside_tag():
    parser = ModelOutputParser()
    raw = '<tool_call>{"name": "calculator", "arguments": invalid_json}</tool_call>'
    action = parser.parse(raw)

    assert action.is_parse_error()
    assert "Malformed JSON" in action.error_message
