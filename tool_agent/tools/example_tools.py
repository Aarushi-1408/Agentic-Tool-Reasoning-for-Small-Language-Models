"""
DEMO TOOLS ONLY.
These example tools are provided purely for testing the harness infrastructure.
They are completely isolated from real tool implementations and can be easily
unregistered or replaced with fixed target tools for your research project.
"""

import ast
import operator as op
from typing import Dict, Any
from src.tools import Tool

# Safe AST-based mathematical evaluator
ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):  # Python >= 3.8
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    elif isinstance(node, ast.BinOp):
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        op_type = type(node.op)
        if op_type in ALLOWED_OPERATORS:
            return ALLOWED_OPERATORS[op_type](left, right)
        raise ValueError(f"Unsupported operator: {op_type.__name__}")
    elif isinstance(node, ast.UnaryOp):
        operand = _safe_eval(node.operand)
        op_type = type(node.op)
        if op_type in ALLOWED_OPERATORS:
            return ALLOWED_OPERATORS[op_type](operand)
        raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
    else:
        raise ValueError(f"Unsupported AST node: {type(node).__name__}")


class CalculatorTool(Tool):
    """
    DEMO TOOL ONLY: Evaluate a simple mathematical expression.
    """

    def __init__(self):
        super().__init__(
            name="calculator",
            description="Evaluate a mathematical expression. Input should be a valid string expression like '25 * 37' or '100 / 4'.",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate, e.g. '25 * 37'",
                    }
                },
                "required": ["expression"],
            },
        )

    def execute(self, arguments: Dict[str, Any]) -> Any:
        expression = arguments.get("expression")
        if not isinstance(expression, str):
            raise TypeError(f"Expression must be a string, got {type(expression).__name__}")

        # Clean string
        expression_str = expression.strip()
        try:
            parsed = ast.parse(expression_str, mode="eval")
            res = _safe_eval(parsed.body)
            # Return formatted string or int/float
            if isinstance(res, float) and res.is_integer():
                return int(res)
            return res
        except Exception as e:
            raise ValueError(f"Could not evaluate expression '{expression_str}': {str(e)}")


class EchoTool(Tool):
    """
    DEMO TOOL ONLY: Echoes the input text back. Useful for testing multi-step workflows.
    """

    def __init__(self):
        super().__init__(
            name="echo",
            description="Echo back the provided message string.",
            parameters={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The text string to echo back.",
                    }
                },
                "required": ["message"],
            },
        )

    def execute(self, arguments: Dict[str, Any]) -> Any:
        message = arguments.get("message")
        if not isinstance(message, str):
            raise TypeError(f"Message must be a string, got {type(message).__name__}")
        return f"Echo: {message}"


class LookupTool(Tool):
    """
    DEMO TOOL ONLY: Look up structured reference data (e.g. key-value facts).
    """

    MOCK_DB = {
        "tokyo_population": "14000000",
        "paris_berlin_distance_km": "1050",
        "gold_atomic_number": "79",
        "usd_to_eur": "0.92",
        "apple_price": "5",
        "orange_price": "8",
        "speed_of_light_km_s": "300000",
        "moon_distance_km": "384400",
    }

    def __init__(self):
        super().__init__(
            name="lookup",
            description="Look up reference information by key. Available keys: 'tokyo_population', 'paris_berlin_distance_km', 'gold_atomic_number', 'usd_to_eur', 'apple_price', 'orange_price', 'speed_of_light_km_s', 'moon_distance_km'.",
            parameters={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The lookup key identifier.",
                    }
                },
                "required": ["key"],
            },
        )

    def execute(self, arguments: Dict[str, Any]) -> Any:
        key = arguments.get("key")
        if not isinstance(key, str):
            raise TypeError(f"Key must be a string, got {type(key).__name__}")
        key_clean = key.strip().lower()
        if key_clean not in self.MOCK_DB:
            keys_avail = ", ".join(self.MOCK_DB.keys())
            raise ValueError(f"Key '{key_clean}' not found in lookup database. Available keys: {keys_avail}")
        return self.MOCK_DB[key_clean]


class StringManipulatorTool(Tool):
    """
    DEMO TOOL ONLY: Perform string transformations (reverse, uppercase, length).
    """

    def __init__(self):
        super().__init__(
            name="string_manipulator",
            description="Perform string operations: 'reverse', 'uppercase', 'lowercase', or 'length'.",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "The operation to perform: 'reverse', 'uppercase', 'lowercase', or 'length'.",
                    },
                    "text": {
                        "type": "string",
                        "description": "The input text string.",
                    },
                },
                "required": ["operation", "text"],
            },
        )

    def execute(self, arguments: Dict[str, Any]) -> Any:
        op = arguments.get("operation")
        text = arguments.get("text")
        if not isinstance(op, str) or not isinstance(text, str):
            raise TypeError("Both 'operation' and 'text' must be strings.")
        op = op.strip().lower()
        if op == "reverse":
            return text[::-1]
        elif op == "uppercase":
            return text.upper()
        elif op == "lowercase":
            return text.lower()
        elif op == "length":
            return len(text)
        else:
            raise ValueError(f"Unknown operation '{op}'. Supported: reverse, uppercase, lowercase, length.")

