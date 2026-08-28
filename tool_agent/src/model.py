import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable


class ModelInterface(ABC):
    """
    Abstract interface for models used by the harness.
    
    The harness communicates with the model strictly through messages and tool definitions.
    The harness does not decide or guide model responses.
    """

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Generate a text response given conversation messages and available tool schemas.
        """
        raise NotImplementedError


DEFAULT_SYSTEM_PROMPT = (
    "You are a tool-use agent.\n\n"
    "For each user request:\n"
    "1. Decide whether you can answer directly or whether a tool is required.\n"
    "2. If a tool is required, output ONLY:\n"
    '   {"type": "tool_call", "name": "<tool_name>", "arguments": {...}}\n'
    "3. After receiving TOOL_RESULT, inspect the result and decide what to do next.\n"
    "4. If another tool is necessary, output the next tool_call.\n"
    "5. If you have enough information to answer the user's request, output ONLY:\n"
    '   {"type": "final", "content": "<answer>"}\n'
    "6. Do not repeat an identical tool call unless there is a genuine reason to do so.\n"
    "7. Do not use a tool when the question can be answered directly.\n"
    "8. The harness executes tool calls and provides their results. You only decide what action to take next."
)


class HuggingFaceModel(ModelInterface):
    """
    Hugging Face Transformers implementation supporting Qwen series models
    (e.g., Qwen/Qwen3-1.7B, Qwen/Qwen3-4B).
    """

    instantiation_count: int = 0

    def __init__(
        self,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        use_system_prompt: bool = False,
        custom_system_prompt: Optional[str] = None,
    ):
        try:
            import os
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "huggingface transformers or torch is missing. "
                "Install them via `pip install torch transformers accelerate`"
            )

        HuggingFaceModel.instantiation_count += 1
        
        # Resolve target model path or HF hub ID
        resolved_path = (model_path or model_name or "Qwen/Qwen3-1.7B").rstrip("~")
        self.model_name = resolved_path
        self.model_path = resolved_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.use_system_prompt = use_system_prompt
        self.custom_system_prompt = custom_system_prompt

        # Detect source type (local filesystem vs HF Hub)
        if os.path.exists(resolved_path) or os.path.isdir(resolved_path):
            self.model_source = "local"
            # Auto-resolve final/ or checkpoint-*/ subfolder if config.json is inside subfolder
            if not os.path.exists(os.path.join(resolved_path, "config.json")):
                final_sub = os.path.join(resolved_path, "final")
                if os.path.exists(os.path.join(final_sub, "config.json")):
                    resolved_path = final_sub
                else:
                    # Find latest checkpoint-* subfolder
                    import glob
                    ckpts = sorted(glob.glob(os.path.join(resolved_path, "checkpoint-*")), key=os.path.getmtime, reverse=True)
                    for ckpt in ckpts:
                        if os.path.exists(os.path.join(ckpt, "config.json")):
                            resolved_path = ckpt
                            break
        else:
            self.model_source = "HuggingFace Hub"

        # Resolve device & dtype
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "float32":
            torch_dtype = torch.float32
        else:
            if torch.cuda.is_available() and device == "cuda" and torch.cuda.is_bf16_supported():
                torch_dtype = torch.bfloat16
            elif torch.cuda.is_available() and device == "cuda":
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32

        print(f"\n==================== MODEL CONFIGURATION ====================")
        print(f"Model source  : {self.model_source}")
        print(f"Model path    : {resolved_path}")
        print(f"Model type    : HuggingFace (Instance #{HuggingFaceModel.instantiation_count})")
        print(f"Device        : {device}")
        print(f"Dtype         : {torch_dtype}")
        print(f"System Prompt : {'ENABLED' if self.use_system_prompt else 'DISABLED'}")

        if torch.cuda.is_available() and device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            alloc_vram = torch.cuda.memory_allocated(0) / (1024 ** 3)
            print(f"GPU Model     : {gpu_name}")
            print(f"GPU VRAM      : Total {total_vram:.2f} GB | Allocated {alloc_vram:.2f} GB")
        else:
            print(f"CUDA Status   : Not available / Disabled (Running on CPU)")
        print("============================================================\n")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(resolved_path, trust_remote_code=True)
        except Exception as e:
            print(f"[WARNING] Failed to load tokenizer from '{resolved_path}' ({e}). Falling back to 'Qwen/Qwen3-1.7B' tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)
        
        if device == "cpu":
            self.model = AutoModelForCausalLM.from_pretrained(
                resolved_path,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
            self.model.to("cpu")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                resolved_path,
                device_map="auto",
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )

    def parameters(self):
        return self.model.parameters() if hasattr(self, "model") and self.model is not None else []

    def named_modules(self):
        return self.model.named_modules() if hasattr(self, "model") and self.model is not None else []

    def train(self, mode: bool = True):
        if hasattr(self, "model") and self.model is not None:
            self.model.train(mode)
        return self

    def eval(self):
        if hasattr(self, "model") and self.model is not None:
            self.model.eval()
        return self

    def _build_prompt(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Format prompt using tokenizer chat template or fallback system prompt.
        """
        formatted_messages = [dict(m) for m in messages]

        # Inject system prompt if enabled
        if self.use_system_prompt:
            sys_text = self.custom_system_prompt if self.custom_system_prompt else DEFAULT_SYSTEM_PROMPT
            if not formatted_messages or formatted_messages[0].get("role") != "system":
                formatted_messages.insert(0, {"role": "system", "content": sys_text})

        # Try native chat template first
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                formatted = self.tokenizer.apply_chat_template(
                    formatted_messages,
                    tools=tools if tools else None,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return formatted
            except Exception:
                pass  # Fallback to custom formatting if template fails

        # Fallback system prompt formatting
        formatted_messages = list(messages)
        if tools:
            tools_str = json.dumps(tools, indent=2)
            tool_system_prompt = (
                f"You have access to the following tools:\n{tools_str}\n\n"
                "To call a tool, respond ONLY with a JSON object:\n"
                "{\n"
                '  "type": "tool_call",\n'
                '  "name": "<tool_name>",\n'
                '  "arguments": { ... }\n'
                "}\n\n"
                "When you have the final answer, respond ONLY with a JSON object:\n"
                "{\n"
                '  "type": "final",\n'
                '  "content": "<final_answer>"\n'
                "}"
            )
            # Prepend or insert tool prompt in system message
            if formatted_messages and formatted_messages[0]["role"] == "system":
                formatted_messages[0] = {
                    "role": "system",
                    "content": formatted_messages[0]["content"] + "\n\n" + tool_system_prompt,
                }
            else:
                formatted_messages.insert(0, {"role": "system", "content": tool_system_prompt})

        try:
            return self.tokenizer.apply_chat_template(
                formatted_messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Plain string formatting fallback
            prompt_parts = []
            for msg in formatted_messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
            prompt_parts.append("<|im_start|>assistant\n")
            return "\n".join(prompt_parts)

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        import torch

        prompt = self._build_prompt(messages, tools)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Slice off input tokens to get generated completion
        input_len = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_len:]
        response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        return response_text


class MockModel(ModelInterface):
    """
    Mock model backend for fast unit testing and offline E2E harness validation.
    
    Can accept a custom generation callback function or a predefined sequence of response strings.
    """

    instantiation_count: int = 0

    def __init__(self, responses: Optional[List[str]] = None, generator_fn: Optional[Callable] = None):
        MockModel.instantiation_count += 1
        self.responses = responses or []
        self.generator_fn = generator_fn
        self.call_count = 0
        self.history: List[Dict[str, Any]] = []

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        self.call_count += 1
        self.history.append({"messages": messages, "tools": tools})

        if self.generator_fn is not None:
            return self.generator_fn(messages, tools, self.call_count)

        if self.responses and self.call_count <= len(self.responses):
            return self.responses[self.call_count - 1]

        # Default fallback behavior for general mock queries
        last_message = messages[-1]["content"] if messages else ""
        
        # If observation from tool result is in message, parse and return final answer
        if "TOOL_RESULT:" in last_message:
            # Extract result content
            res_val = last_message.split("TOOL_RESULT:")[1].strip()
            return json.dumps({
                "type": "final",
                "content": f"The answer is {res_val}."
            })
        
        # If user task has a calculation format like 25 * 37
        if "25 * 37" in last_message:
            return json.dumps({
                "type": "tool_call",
                "name": "calculator",
                "arguments": {"expression": "25 * 37"}
            })

        # Default direct answer
        return json.dumps({
            "type": "final",
            "content": f"Mock answer for task: '{last_message}'"
        })
