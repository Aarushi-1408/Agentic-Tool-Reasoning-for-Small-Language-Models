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
    "capital of germany": [json.dumps({"type": "final", "content": "The capital of Germany is Berlin."})],
    "starry night": [json.dumps({"type": "final", "content": "Vincent van Gogh painted The Starry Night."})],
    "29 * 43": [
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "29 * 43"}}),
        json.dumps({"type": "final", "content": "The result of 29 * 43 is 1247."}),
    ],
    "(180 + 320) / 10": [
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "(180 + 320) / 10"}}),
        json.dumps({"type": "final", "content": "The result is 50."}),
    ],
    "169 / 13": [
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "169 / 13"}}),
        json.dumps({"type": "final", "content": "The result is 13."}),
    ],
    "888 - 321": [
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "888 - 321"}}),
        json.dumps({"type": "final", "content": "The result is 567."}),
    ],
    "hello synthetic dataset": [
        json.dumps({"type": "tool_call", "name": "echo", "arguments": {"message": "Hello Synthetic Dataset"}}),
        json.dumps({"type": "final", "content": "Echo: Hello Synthetic Dataset"}),
    ],
    "silver_atomic_number": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "silver_atomic_number"}}),
        json.dumps({"type": "final", "content": "The atomic number of silver is 47."}),
    ],
    "banana_price": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "banana_price"}}),
        json.dumps({"type": "final", "content": "The price of a banana is 4."}),
    ],
    "atomic number of gold with key": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "gold_atomic_number"}}),
        json.dumps({"type": "final", "content": "The atomic number of gold is 79."}),
    ],
    "value for 'tokyo_population' from database": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "tokyo_population"}}),
        json.dumps({"type": "final", "content": "The population of Tokyo is 14,000,000."}),
    ],
    "value 'paris_berlin_distance_km'": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "paris_berlin_distance_km"}}),
        json.dumps({"type": "final", "content": "The distance between Paris and Berlin is 1050 km."}),
    ],
    "key 'orange_price' in the store inventory": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "orange_price"}}),
        json.dumps({"type": "final", "content": "The price of an orange is 8."}),
    ],
    "key 'moon_distance_km'": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "moon_distance_km"}}),
        json.dumps({"type": "final", "content": "The moon distance is 384400 km."}),
    ],
    "key 'usd_to_eur'": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "usd_to_eur"}}),
        json.dumps({"type": "final", "content": "The exchange rate for usd_to_eur is 0.92."}),
    ],
    "apple_price + orange_price + 5": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "apple_price"}}),
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "orange_price"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "5 + 8 + 5"}}),
        json.dumps({"type": "final", "content": "The total is 18."}),
    ],
    "gold_atomic_number * 7": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "gold_atomic_number"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "79 * 7"}}),
        json.dumps({"type": "final", "content": "The result is 553."}),
    ],
    "tokyo_population', then divide it by 2000": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "tokyo_population"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "14000000 / 2000"}}),
        json.dumps({"type": "final", "content": "The result is 7000."}),
    ],
    "paris_berlin_distance_km' and add 150 km": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "paris_berlin_distance_km"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "1050 + 150"}}),
        json.dumps({"type": "final", "content": "The total distance is 1200 km."}),
    ],
    "orange_price * 25": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "orange_price"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "8 * 25"}}),
        json.dumps({"type": "final", "content": "The cost of 25 oranges is 200."}),
    ],
    "5 apples and 4 oranges": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "apple_price"}}),
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "orange_price"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "(5 * 5) + (4 * 8)"}}),
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "reverse", "text": "57"}}),
        json.dumps({"type": "final", "content": "The reversed total cost string is 75."}),
    ],
    "gold_atomic_number', multiply by 15": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "gold_atomic_number"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "79 * 15"}}),
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "reverse", "text": "1185"}}),
        json.dumps({"type": "final", "content": "The reversed digits string is 5811."}),
    ],
    "paris_berlin_distance_km', add 300": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "paris_berlin_distance_km"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "1050 + 300"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "1350 / 5"}}),
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "reverse", "text": "270"}}),
        json.dumps({"type": "final", "content": "The reversed digits string is 072."}),
    ],
    "(apple_price + orange_price) * usd_to_eur * 10": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "apple_price"}}),
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "orange_price"}}),
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "usd_to_eur"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "(5 + 8) * 0.92 * 10"}}),
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "reverse", "text": "119.6"}}),
        json.dumps({"type": "final", "content": "The reversed digits string is 6.911."}),
    ],
    "antigravitysystem": [
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "reverse", "text": "AntigravitySystem"}}),
        json.dumps({"type": "final", "content": "The reversed string is metsySytivargitnA."}),
    ],
    "antidisestablishmentarianism": [
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "length", "text": "Antidisestablishmentarianism"}}),
        json.dumps({"type": "final", "content": "The length of the string is 28."}),
    ],
    "deep learning system": [
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "length", "text": "Deep Learning System"}}),
        json.dumps({"type": "final", "content": "The length of the string is 20."}),
    ],
    "saturn_distance_km": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "saturn_distance_km"}}),
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "moon_distance_km"}}),
        json.dumps({"type": "final", "content": "Recovered by looking up moon_distance_km: 384400."}),
    ],
    "30 * 40": [
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "30 * 40 + bad_var"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "30 * 40"}}),
        json.dumps({"type": "final", "content": "Corrected result is 1200."}),
    ],
    "cpu stand for": [json.dumps({"type": "final", "content": "CPU stands for Central Processing Unit."})],
    "freezing point of water": [json.dumps({"type": "final", "content": "The freezing point of water is 0°C."})],
    "currency is used in the united kingdom": [json.dumps({"type": "final", "content": "The currency used in the UK is the Pound Sterling."})],
    "chemical symbol for gold": [json.dumps({"type": "final", "content": "The chemical symbol for Gold is Au."})],
    "sides does a hexagon have": [json.dumps({"type": "final", "content": "A hexagon has 6 sides."})],
    "largest ocean on earth": [json.dumps({"type": "final", "content": "The Pacific Ocean is the largest ocean on Earth."})],
    "ram and rom": [json.dumps({"type": "final", "content": "RAM is volatile while ROM is non-volatile."})],
    "red planet": [json.dumps({"type": "final", "content": "Mars is known as the Red Planet."})],
    "speed of light in vacuum": [json.dumps({"type": "final", "content": "The speed of light in a vacuum is approximately 299,792 km/s."})],
    "general relativity": [json.dumps({"type": "final", "content": "Albert Einstein proposed general relativity."})],
    "earth's atmosphere": [json.dumps({"type": "final", "content": "Nitrogen is the primary gas in Earth's atmosphere."})],
    "http stand for": [json.dumps({"type": "final", "content": "HTTP stands for Hypertext Transfer Protocol."})],
    "bits are in a single byte": [json.dumps({"type": "final", "content": "There are 8 bits in a single byte."})],
    "square root of 64": [json.dumps({"type": "final", "content": "The square root of 64 is 8."})],
    "plants convert sunlight into energy": [json.dumps({"type": "final", "content": "Photosynthesis is the process."})],
    "smallest prime number": [json.dumps({"type": "final", "content": "2 is the smallest prime number."})],
    "gpu stand for": [json.dumps({"type": "final", "content": "GPU stands for Graphics Processing Unit."})],
    "egypt primarily located on": [json.dumps({"type": "final", "content": "Egypt is located in Africa."})],
    "capital of australia": [json.dumps({"type": "final", "content": "The capital of Australia is Canberra."})],
    "chemical formula for carbon dioxide": [json.dumps({"type": "final", "content": "The chemical formula is CO2."})],
    "colors are in a standard rainbow": [json.dumps({"type": "final", "content": "There are 7 colors in a rainbow."})],
    "function of red blood cells": [json.dumps({"type": "final", "content": "Red blood cells carry oxygen."})],
    "(150 + 250) / 5": [
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "(150 + 250) / 5"}}),
        json.dumps({"type": "final", "content": "The result is 80."}),
    ],
    "999 - 432": [
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "999 - 432"}}),
        json.dumps({"type": "final", "content": "The result is 567."}),
    ],
    "18 * 14": [
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "18 * 14"}}),
        json.dumps({"type": "final", "content": "The result is 252."}),
    ],
    "2048 / 64": [
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "2048 / 64"}}),
        json.dumps({"type": "final", "content": "The result is 32."}),
    ],
    "315 + 485": [
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "315 + 485"}}),
        json.dumps({"type": "final", "content": "The result is 800."}),
    ],
    "85 * 9": [
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "85 * 9"}}),
        json.dumps({"type": "final", "content": "The result is 765."}),
    ],
    "1024 - 512": [
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "1024 - 512"}}),
        json.dumps({"type": "final", "content": "The result is 512."}),
    ],
    "hello synthetic world": [
        json.dumps({"type": "tool_call", "name": "echo", "arguments": {"message": "Hello Synthetic World"}}),
        json.dumps({"type": "final", "content": "Echo: Hello Synthetic World"}),
    ],
    "agentic data scaling": [
        json.dumps({"type": "tool_call", "name": "echo", "arguments": {"message": "Agentic Data Scaling"}}),
        json.dumps({"type": "final", "content": "Echo: Agentic Data Scaling"}),
    ],
    "qwen3 model training": [
        json.dumps({"type": "tool_call", "name": "echo", "arguments": {"message": "Qwen3 Model Training"}}),
        json.dumps({"type": "final", "content": "Echo: Qwen3 Model Training"}),
    ],
    "system verification ok": [
        json.dumps({"type": "tool_call", "name": "echo", "arguments": {"message": "System Verification OK"}}),
        json.dumps({"type": "final", "content": "Echo: System Verification OK"}),
    ],
    "pipeline generation complete": [
        json.dumps({"type": "tool_call", "name": "echo", "arguments": {"message": "Pipeline Generation Complete"}}),
        json.dumps({"type": "final", "content": "Echo: Pipeline Generation Complete"}),
    ],
    "tool execution test": [
        json.dumps({"type": "tool_call", "name": "echo", "arguments": {"message": "Tool Execution Test"}}),
        json.dumps({"type": "final", "content": "Echo: Tool Execution Test"}),
    ],
    "deepmind": [
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "reverse", "text": "DeepMind"}}),
        json.dumps({"type": "final", "content": "The reversed string is dniMpeeD."}),
    ],
    "transformers": [
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "reverse", "text": "Transformers"}}),
        json.dumps({"type": "final", "content": "The reversed string is sremrofsnarT."}),
    ],
    "syntheticdata": [
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "reverse", "text": "SyntheticData"}}),
        json.dumps({"type": "final", "content": "The reversed string is ataDcitehtnyS."}),
    ],
    "cynaptics": [
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "uppercase", "text": "cynaptics"}}),
        json.dumps({"type": "final", "content": "The uppercase string is CYNAPTICS."}),
    ],
    "hellomodel": [
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "lowercase", "text": "HELLOMODEL"}}),
        json.dumps({"type": "final", "content": "The lowercase string is hellomodel."}),
    ],
    "opensourceai": [
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "length", "text": "OpenSourceAI"}}),
        json.dumps({"type": "final", "content": "The length of OpenSourceAI is 12."}),
    ],
    "length of 'supercalifragilisticexpialidocious' using string_manipulator, then multiply": [
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "length", "text": "Supercalifragilisticexpialidocious"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "34 * 10"}}),
        json.dumps({"type": "final", "content": "The final result is 340."}),
    ],
    "gold_atomic_number * 5": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "gold_atomic_number"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "79 * 5"}}),
        json.dumps({"type": "final", "content": "The result is 395."}),
    ],
    "tokyo_population', then divide it by 1000": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "tokyo_population"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "14000000 / 1000"}}),
        json.dumps({"type": "final", "content": "Tokyo population in thousands is 14000."}),
    ],
    "autonomous agent": [
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "uppercase", "text": "autonomous agent"}}),
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "reverse", "text": "AUTONOMOUS AGENT"}}),
        json.dumps({"type": "final", "content": "The reversed uppercase string is TNEGA SUOMONOTUA."}),
    ],
    "agentictoolreasoning": [
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "length", "text": "AgenticToolReasoning"}}),
        json.dumps({"type": "tool_call", "name": "echo", "arguments": {"message": "20"}}),
        json.dumps({"type": "final", "content": "Echo: 20"}),
    ],
    "cost of a dozen apples": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "apple_price"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "5 * 12"}}),
        json.dumps({"type": "final", "content": "The cost of a dozen apples is 60."}),
    ],
    "add 10 to it using calculator": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "orange_price"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "8 + 10"}}),
        json.dumps({"type": "final", "content": "The result is 18."}),
    ],
    "pythoncode": [
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "reverse", "text": "PythonCode"}}),
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "uppercase", "text": "edoCnohtyP"}}),
        json.dumps({"type": "final", "content": "The result is EDOCNOHTYP."}),
    ],
    "15 * 8 using calculator, then echo": [
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "15 * 8"}}),
        json.dumps({"type": "tool_call", "name": "echo", "arguments": {"message": "120"}}),
        json.dumps({"type": "final", "content": "Echo: 120"}),
    ],
    "divide it by 100 using calculator": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "moon_distance_km"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "384400 / 100"}}),
        json.dumps({"type": "final", "content": "The result is 3844."}),
    ],
    "benchmarkevaluation": [
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "reverse", "text": "BenchmarkEvaluation"}}),
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "length", "text": "noitaulevAlramhcneB"}}),
        json.dumps({"type": "final", "content": "The length is 19."}),
    ],
    "neuralnetwork": [
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "lowercase", "text": "NeuralNetwork"}}),
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "reverse", "text": "neuralnetwork"}}),
        json.dumps({"type": "final", "content": "The reversed string is krowtenlaruen."}),
    ],
    "add 100 to it": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "gold_atomic_number"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "79 + 100"}}),
        json.dumps({"type": "final", "content": "The result is 179."}),
    ],
    "500 / 25, then echo": [
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "500 / 25"}}),
        json.dumps({"type": "tool_call", "name": "echo", "arguments": {"message": "20"}}),
        json.dumps({"type": "final", "content": "Echo: 20"}),
    ],
    "add 50 km to it": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "paris_berlin_distance_km"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "1050 + 50"}}),
        json.dumps({"type": "final", "content": "The result is 1100."}),
    ],
    "datascaling": [
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "reverse", "text": "DataScaling"}}),
        json.dumps({"type": "tool_call", "name": "echo", "arguments": {"message": "gnilacSataD"}}),
        json.dumps({"type": "final", "content": "Echo: gnilacSataD"}),
    ],
    "multiply it by 20": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "orange_price"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "8 * 20"}}),
        json.dumps({"type": "final", "content": "The result is 160."}),
    ],
    "largelanguagemodel": [
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "lowercase", "text": "LargeLanguageModel"}}),
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "length", "text": "largelanguagemodel"}}),
        json.dumps({"type": "final", "content": "The length is 18."}),
    ],
    "120 + 380, then divide": [
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "120 + 380"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "500 / 10"}}),
        json.dumps({"type": "final", "content": "The result is 50."}),
    ],
    "apple_price' and add 3": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "apple_price"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "5 + 3"}}),
        json.dumps({"type": "final", "content": "The result is 8."}),
    ],
    "finetuning": [
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "reverse", "text": "FineTuning"}}),
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "length", "text": "gninuTeniF"}}),
        json.dumps({"type": "final", "content": "The length is 10."}),
    ],
    "tokyo_population', divide by 1000": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "tokyo_population"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "14000000 / 1000"}}),
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "uppercase", "text": "14000"}}),
        json.dumps({"type": "final", "content": "Final value is 14000."}),
    ],
    "add 21 to it using calculator, reverse": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "gold_atomic_number"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "79 + 21"}}),
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "reverse", "text": "100"}}),
        json.dumps({"type": "tool_call", "name": "echo", "arguments": {"message": "001"}}),
        json.dumps({"type": "final", "content": "Echo: 001"}),
    ],
    "3 apples and 2 oranges": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "apple_price"}}),
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "orange_price"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "(3 * 5) + (2 * 8)"}}),
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "reverse", "text": "31"}}),
        json.dumps({"type": "final", "content": "The reversed total cost string is 13."}),
    ],
    "antigravityagent": [
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "length", "text": "AntigravityAgent"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "16 * 5"}}),
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "reverse", "text": "80"}}),
        json.dumps({"type": "final", "content": "The reversed string is 08."}),
    ],
    "moon_distance_km', divide by 100 using calculator, reverse": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "moon_distance_km"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "384400 / 100"}}),
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "reverse", "text": "3844"}}),
        json.dumps({"type": "tool_call", "name": "echo", "arguments": {"message": "4483"}}),
        json.dumps({"type": "final", "content": "Echo: 4483"}),
    ],
    "syntheticdataset": [
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "uppercase", "text": "SyntheticDataset"}}),
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "length", "text": "SYNTHETICDATASET"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "16 * 3"}}),
        json.dumps({"type": "tool_call", "name": "echo", "arguments": {"message": "48"}}),
        json.dumps({"type": "final", "content": "Final result is 48."}),
    ],
    "multiply by 15 using calculator, reverse": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "orange_price"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "8 * 15"}}),
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "reverse", "text": "120"}}),
        json.dumps({"type": "tool_call", "name": "echo", "arguments": {"message": "021"}}),
        json.dumps({"type": "final", "content": "Echo: 021"}),
    ],
    "add 200 using calculator, divide by 5, then reverse": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "paris_berlin_distance_km"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "1050 + 200"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "1250 / 5"}}),
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "reverse", "text": "250"}}),
        json.dumps({"type": "final", "content": "Reversed string is 052."}),
    ],
    "multistepreasoning": [
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "reverse", "text": "MultiStepReasoning"}}),
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "uppercase", "text": "gninosaeRPetSitluM"}}),
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "length", "text": "GNINOSAEPRPETSITLUM"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "19 * 2"}}),
        json.dumps({"type": "final", "content": "The result is 38."}),
    ],
    "multiply by 10, then echo": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "apple_price"}}),
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "orange_price"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "5 + 8"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "13 * 10"}}),
        json.dumps({"type": "tool_call", "name": "echo", "arguments": {"message": "130"}}),
        json.dumps({"type": "final", "content": "Echo: 130"}),
    ],
    "multiply by 2, add 42, then reverse": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "gold_atomic_number"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "79 * 2"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "158 + 42"}}),
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "reverse", "text": "200"}}),
        json.dumps({"type": "final", "content": "Final reversed string is 002."}),
    ],
    "add them together, divide by 10": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "tokyo_population"}}),
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "paris_berlin_distance_km"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "14000000 + 1050"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "14001050 / 10"}}),
        json.dumps({"type": "tool_call", "name": "echo", "arguments": {"message": "1400105"}}),
        json.dumps({"type": "final", "content": "Echo: 1400105"}),
    ],
    "supercalifragilistic' to lowercase": [
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "lowercase", "text": "Supercalifragilistic"}}),
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "length", "text": "supercalifragilistic"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "20 * 4"}}),
        json.dumps({"type": "tool_call", "name": "echo", "arguments": {"message": "80"}}),
        json.dumps({"type": "final", "content": "Echo: 80"}),
    ],
    "divide distance by speed, multiply result by 100": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "moon_distance_km"}}),
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "speed_of_light_km_s"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "384400 / 300000"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "1.28133 * 100"}}),
        json.dumps({"type": "tool_call", "name": "echo", "arguments": {"message": "128.133"}}),
        json.dumps({"type": "final", "content": "Echo: 128.133"}),
    ],
    "trajectoryvalidation": [
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "reverse", "text": "TrajectoryValidation"}}),
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "length", "text": "noitadilaVyrotcejarT"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "20 * 7"}}),
        json.dumps({"type": "tool_call", "name": "echo", "arguments": {"message": "140"}}),
        json.dumps({"type": "final", "content": "Echo: 140"}),
    ],
    "add 5, multiply by 8, then reverse": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "orange_price"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "8 + 5"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "13 * 8"}}),
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "reverse", "text": "104"}}),
        json.dumps({"type": "final", "content": "Reversed digits: 401"}),
    ],
    "toolregistryexecution": [
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "uppercase", "text": "ToolRegistryExecution"}}),
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "length", "text": "TOOLREGISTRYEXECUTION"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "21 / 2"}}),
        json.dumps({"type": "tool_call", "name": "echo", "arguments": {"message": "10.5"}}),
        json.dumps({"type": "final", "content": "Echo: 10.5"}),
    ],
    "subtract 9, multiply by 10": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "gold_atomic_number"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "79 - 9"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "70 * 10"}}),
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "reverse", "text": "700"}}),
        json.dumps({"type": "final", "content": "Reversed digits: 007"}),
    ],
    "jupiter_radius_km": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "jupiter_radius_km"}}),
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "gold_atomic_number"}}),
        json.dumps({"type": "final", "content": "Recovered by looking up gold_atomic_number: 79."}),
    ],
    "100 / 0": [
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "100 / 0"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "100 / 5"}}),
        json.dumps({"type": "final", "content": "Corrected division result is 20."}),
    ],
    "translate": [
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "translate", "text": "hello"}}),
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "uppercase", "text": "hello"}}),
        json.dumps({"type": "final", "content": "Uppercase result is HELLO."}),
    ],
    "invalid_key_999": [
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "invalid_key_999"}}),
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "tokyo_population"}}),
        json.dumps({"type": "final", "content": "Tokyo population is 14,000,000."}),
    ],
    "50 + unknown_variable": [
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "50 + unknown_variable"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "50 + 250"}}),
        json.dumps({"type": "final", "content": "Calculated result is 300."}),
    ],
    "web_search": [
        json.dumps({"type": "tool_call", "name": "web_search", "arguments": {"query": "moon_distance_km"}}),
        json.dumps({"type": "tool_call", "name": "lookup", "arguments": {"key": "moon_distance_km"}}),
        json.dumps({"type": "final", "content": "Moon distance is 384400 km."}),
    ],
    "reversing text with missing argument": [
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "reverse"}}),
        json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": {"operation": "reverse", "text": "Antigravity"}}),
        json.dumps({"type": "final", "content": "Reversed text is ytivargitnA."}),
    ],
    "(10 + 20": [
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "(10 + 20"}}),
        json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "(10 + 20) * 2"}}),
        json.dumps({"type": "final", "content": "Result is 60."}),
    ],

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
