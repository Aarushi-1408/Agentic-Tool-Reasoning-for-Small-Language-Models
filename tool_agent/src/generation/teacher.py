import json
import os
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable

from src.model import ModelInterface, HuggingFaceModel, MockModel


class TeacherInterface(ModelInterface, ABC):
    """
    Model-agnostic Teacher Interface.
    
    The teacher model receives conversation history and tool definitions,
    and returns a raw text model completion. The teacher does NOT execute tools.
    """

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Generate model completion given messages and available tool schemas.
        """
        raise NotImplementedError

    def get_info(self) -> Dict[str, Any]:
        """
        Return metadata about the teacher instance (name, backend, device, dtype, etc.).
        """
        return {
            "teacher_model": getattr(self, "model_name", "unknown"),
            "backend": self.__class__.__name__,
        }


class HuggingFaceTeacher(HuggingFaceModel, TeacherInterface):
    """
    Hugging Face / Transformers backend implementation for Teacher models.
    Supports local execution of models such as Qwen/Qwen3-8B, Qwen/Qwen3-1.7B, etc.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
    ):
        super().__init__(
            model_name=model_name,
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    def get_info(self) -> Dict[str, Any]:
        return {
            "teacher_model": self.model_name,
            "backend": "transformers",
            "device": self.device,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }


class APITeacher(TeacherInterface):
    """
    API-based Teacher interface for calling external model endpoints (e.g. OpenAI / vLLM / Ollama REST API).
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        self.model_name = model_name
        self.api_base = api_base or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        # Build payload for OpenAI-compatible chat completion endpoint
        url = f"{self.api_base.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            # Format tools as OpenAI function call definitions
            payload["tools"] = [
                {"type": "function", "function": tool} for tool in tools
            ]

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                choice = result["choices"][0]["message"]
                if choice.get("tool_calls"):
                    # Convert first tool call to harness JSON format
                    tc = choice["tool_calls"][0]["function"]
                    args = json.loads(tc.get("arguments", "{}"))
                    return json.dumps({
                        "type": "tool_call",
                        "name": tc["name"],
                        "arguments": args,
                    })
                return choice.get("content", "")
        except urllib.error.URLError as e:
            raise RuntimeError(f"APITeacher HTTP request failed to '{url}': {str(e)}")

    def get_info(self) -> Dict[str, Any]:
        return {
            "teacher_model": self.model_name,
            "backend": "api",
            "api_base": self.api_base,
            "temperature": self.temperature,
        }


DEFAULT_MOCK_TASK_BEHAVIORS: Dict[str, List[str]] = {
    "capital of japan": [json.dumps({"type": "final", "content": "The capital of Japan is Tokyo."})],
    "list and a tuple": [json.dumps({"type": "final", "content": "In Python, lists are mutable while tuples are immutable."})],
    "hamlet": [json.dumps({"type": "final", "content": "William Shakespeare wrote the play Hamlet."})],
    "144 / 12": [
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "144 / 12"}}),
        json.dumps({"type": "final", "content": "144 divided by 12 is 12."}),
    ],
    "gold_atomic_number": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "gold_atomic_number"}}),
        json.dumps({"type": "final", "content": "The atomic number of gold is 79."}),
    ],
    "antigravity": [
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "reverse", "text": "Antigravity"}}),
        json.dumps({"type": "final", "content": "The reversed string is ytivargitnA."}),
    ],
    "hello world": [
        json.dumps({"type": "tool_call", "name": "echo", "arguments": {"message": "Hello World"}}),
        json.dumps({"type": "final", "content": "Echo: Hello World"}),
    ],
    "population of tokyo": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "tokyo_population"}}),
        json.dumps({"type": "final", "content": "The population of Tokyo is 14,000,000."}),
    ],
    "supercalifragilisticexpialidocious": [
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "length", "text": "Supercalifragilisticexpialidocious"}}),
        json.dumps({"type": "final", "content": "The length of the string is 34."}),
    ],
    "paris to berlin": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "paris_berlin_distance_km"}}),
        json.dumps({"type": "final", "content": "The distance between Paris and Berlin is 1050 km."}),
    ],
    "price of an apple": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "apple_price"}}),
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "orange_price"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "5 + 8"}}),
        json.dumps({"type": "final", "content": "The total cost of an apple and an orange is 13."}),
    ],
    "speed_of_light_km_s": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "moon_distance_km"}}),
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "speed_of_light_km_s"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "384400 / 300000"}}),
        json.dumps({"type": "final", "content": "Light takes approximately 1.28 seconds to travel from the Moon to Earth."}),
    ],
    "machine learning": [
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "uppercase", "text": "machine learning"}}),
        json.dumps({"type": "tool_call", "name": "echo", "arguments": {"message": "MACHINE LEARNING"}}),
        json.dumps({"type": "final", "content": "Echo: MACHINE LEARNING"}),
    ],
    "calculate their sum": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "apple_price"}}),
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "orange_price"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "5 + 8"}}),
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "reverse", "text": "13"}}),
        json.dumps({"type": "final", "content": "The reversed sum string is 31."}),
    ],
    "multiply it by 10": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "gold_atomic_number"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "79 * 10"}}),
        json.dumps({"type": "tool_call", "name": "echo", "arguments": {"message": "790"}}),
        json.dumps({"type": "final", "content": "Final value is 790."}),
    ],
    "paris_berlin_distance_km', divide": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "paris_berlin_distance_km"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "1050 / 10"}}),
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "reverse", "text": "105"}}),
        json.dumps({"type": "final", "content": "Reversed quotient string is 501."}),
    ],
    "greater than 10,000,000": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "tokyo_population"}}),
        json.dumps({"type": "final", "content": "Tokyo population is 14,000,000, which is greater than 10,000,000."}),
    ],
    "(50 * 4) - 25": [
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "(50 * 4) - 25"}}),
        json.dumps({"type": "final", "content": "The result is 175, which is an odd number."}),
    ],
    "artificial intelligence": [
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "length", "text": "Artificial Intelligence"}}),
        json.dumps({"type": "final", "content": "Length is 23, which is greater than 15 characters."}),
    ],
    "mars_distance_km": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "mars_distance_km"}}),
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "moon_distance_km"}}),
        json.dumps({"type": "final", "content": "Recovered by looking up moon_distance_km: 384400."}),
    ],
    "bad_var": [
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "25 * 37 + bad_var"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "25 * 37"}}),
        json.dumps({"type": "final", "content": "Corrected evaluation result is 925."}),
    ],
    "db_search": [
        json.dumps({"type": "tool_call", "name": "db_search", "arguments": {"key": "apple_price"}}),
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "apple_price"}}),
        json.dumps({"type": "final", "content": "Recovered using lookup: 5."}),
    ],
    "expression=500": [
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": 500}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "500 / 20"}}),
        json.dumps({"type": "final", "content": "Corrected evaluation result is 25."}),
    ],
    "usd_to_eur": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "apple_price"}}),
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "orange_price"}}),
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "usd_to_eur"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "(5 + 8) * 0.92"}}),
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "reverse", "text": "11.96"}}),
        json.dumps({"type": "final", "content": "Final reversed string is 69.11."}),
    ],
    "speed of sound": [json.dumps({"type": "final", "content": "The speed of sound in dry air at 20°C is approximately 343 m/s."})],
}


class MockTeacher(MockModel, TeacherInterface):
    """
    Mock Teacher backend for testing offline generation pipelines, multi-step scenarios,
    error recovery, and validator routines without requiring GPU hardware.
    """

    def __init__(
        self,
        model_name: str = "mock-teacher-v1",
        responses: Optional[List[str]] = None,
        generator_fn: Optional[Callable] = None,
        task_behavior_map: Optional[Dict[str, List[str]]] = None,
    ):
        super().__init__(responses=responses, generator_fn=generator_fn)
        self.model_name = model_name
        self.task_behavior_map = task_behavior_map or DEFAULT_MOCK_TASK_BEHAVIORS

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        # Calculate step index within the current task based on assistant messages
        step_idx = len([m for m in messages if m.get("role") == "assistant"])

        # Check task behavior map if provided
        initial_user_msg = messages[0]["content"] if messages else ""
        for task_key, canned_responses in self.task_behavior_map.items():
            if task_key.lower() in initial_user_msg.lower():
                if step_idx < len(canned_responses):
                    self.call_count += 1
                    self.history.append({"messages": messages, "tools": tools})
                    return canned_responses[step_idx]

        # Use base MockModel generate logic
        return super().generate(messages, tools)

    def get_info(self) -> Dict[str, Any]:
        return {
            "teacher_model": self.model_name,
            "backend": "mock",
        }


def create_teacher(config: Dict[str, Any]) -> TeacherInterface:
    """
    Factory function to create a Teacher model instance based on configuration dictionary.
    
    Example config:
        teacher:
            backend: transformers  # or 'api', 'mock'
            model_name: Qwen/Qwen3-8B
            device: auto
            dtype: bfloat16
    """
    teacher_cfg = config.get("teacher", config)
    backend = teacher_cfg.get("backend", "transformers").lower()
    model_name = teacher_cfg.get("model_name", "Qwen/Qwen3-8B")
    device = teacher_cfg.get("device", "auto")
    dtype = teacher_cfg.get("dtype", "auto")
    max_new_tokens = teacher_cfg.get("max_new_tokens", 1024)
    temperature = teacher_cfg.get("temperature", 0.0)

    if backend in ["transformers", "huggingface", "hf"]:
        return HuggingFaceTeacher(
            model_name=model_name,
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    elif backend in ["api", "openai", "rest"]:
        return APITeacher(
            model_name=model_name,
            api_base=teacher_cfg.get("api_base"),
            api_key=teacher_cfg.get("api_key"),
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
    elif backend in ["mock", "testing"]:
        return MockTeacher(
            model_name=model_name,
            task_behavior_map=teacher_cfg.get("task_behavior_map"),
        )
    else:
        raise ValueError(f"Unsupported teacher backend: '{backend}'. Supported: 'transformers', 'api', 'mock'.")
