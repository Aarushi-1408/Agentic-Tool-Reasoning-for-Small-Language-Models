from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from src.tool_schema import generate_tool_schema, validate_arguments


@dataclass
class ToolResult:
    """
    Standardized observation container for tool executions.
    """
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None

    def to_observation_string(self) -> str:
        """
        Format the result or error as a clean observation string for the model.
        """
        if self.success:
            return f"TOOL_RESULT: {self.result}"
        else:
            return f"TOOL_ERROR: {self.error}"


class Tool(ABC):
    """
    Generic tool abstraction.
    Each tool defines its metadata (name, description, parameters schema) and execution logic.
    """

    def __init__(self, name: str, description: str, parameters: Dict[str, Any]):
        self.name = name
        self.description = description
        self.parameters = parameters

    @abstractmethod
    def execute(self, arguments: Dict[str, Any]) -> Any:
        """
        Execute the tool action given validated argument dictionary.
        """
        pass

    def to_schema(self) -> Dict[str, Any]:
        """
        Return standard JSON schema for model prompts / function definitions.
        """
        return generate_tool_schema(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )


class ToolRegistry:
    """
    Central registry for available tools.
    
    IMPORTANT:
    The registry does NOT select tools or infer intent.
    It purely exposes schemas and handles validated execution calls.
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """
        Register a tool instance.
        """
        if not isinstance(tool, Tool):
            raise TypeError(f"Expected instance of Tool, got {type(tool).__name__}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        """
        Retrieve a registered tool by name.
        """
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """
        Check if a tool is registered.
        """
        return name in self._tools

    def schemas(self) -> List[Dict[str, Any]]:
        """
        Return standard JSON schemas for all registered tools.
        """
        return [tool.to_schema() for tool in self._tools.values()]

    def list_tool_names(self) -> List[str]:
        """
        Return list of all registered tool names.
        """
        return list(self._tools.keys())

    def execute(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        """
        Execute requested tool by name with arguments.
        
        Performs:
        1. Tool existence lookup (returns unknown tool error if missing).
        2. Parameter schema validation (returns argument error if invalid).
        3. Tool execution (catches exceptions and returns execution error).
        """
        if not self.has(name):
            available = ", ".join(self.list_tool_names()) if self._tools else "none"
            return ToolResult(
                success=False,
                error=f"Unknown tool: '{name}'. Available tools: [{available}]"
            )

        tool = self.get(name)
        
        # Validate argument types & required keys against parameter schema
        is_valid, val_error = validate_arguments(tool.parameters, arguments)
        if not is_valid:
            return ToolResult(
                success=False,
                error=f"Invalid arguments for tool '{name}': {val_error}"
            )

        try:
            raw_result = tool.execute(arguments)
            return ToolResult(success=True, result=raw_result)
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Execution error in tool '{name}': {str(e)}"
            )
