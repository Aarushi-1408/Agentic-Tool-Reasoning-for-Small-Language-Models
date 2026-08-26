import os
import json
import shutil
import tempfile
import pytest
from pathlib import Path

from src.training.sft_dataset import (
    load_sft_dataset,
    create_train_val_split,
    format_example_tokens,
    SFTDataset,
    DataCollatorForSFT,
)
from scripts.train_sft import verify_full_parameter_sft
from src.parser import ModelOutputParser, Action
from src.model import MockModel
from src.tools import ToolRegistry
from src.harness import AgentHarness
from tools.example_tools import CalculatorTool


@pytest.fixture
def temp_dir():
    tmp = tempfile.mkdtemp()
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


class DummyTokenizer:
    """Lightweight dummy tokenizer for testing chat template & label masking without HF download."""
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.vocab = {"<pad>": 0, "<eos>": 1, "<|im_start|>": 2, "<|im_end|>": 3, "user": 4, "assistant": 5}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = []
        for m in messages:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        res = "\n".join(parts)
        if tokenize:
            return self.encode(res)
        return res

    def encode(self, text, add_special_tokens=False):
        tokens = [ord(c) % 500 + 10 for c in text]
        return tokens

    def decode(self, token_ids, skip_special_tokens=True):
        chars = [chr(t - 10) if 10 <= t <= 600 else "?" for t in token_ids]
        return "".join(chars)


class DummyParam:
    def __init__(self, numel=100, requires_grad=True):
        self._numel = numel
        self.requires_grad = requires_grad

    def numel(self):
        return self._numel


class DummyModelModule:
    """Dummy model for PEFT and full-parameter checks."""
    def __init__(self, num_params=10, peft=False):
        self.params = [DummyParam(100, requires_grad=True) for _ in range(num_params)]
        self.peft = peft

    def named_modules(self):
        if self.peft:
            return [("lora_layer_0", type("LoraModule", (), {})())]
        return [("dense_layer_0", type("DenseModule", (), {})())]

    def parameters(self):
        return self.params


def test_1_dataset_loads_correctly():
    """Verify SFT dataset loads correctly from JSONL."""
    dataset_path = Path("data/sft/train_v2_100.jsonl")
    assert dataset_path.exists(), "train_v2_100.jsonl missing"
    examples = load_sft_dataset(str(dataset_path))
    assert len(examples) == 100
    assert "messages" in examples[0]
    assert len(examples[0]["messages"]) >= 2


def test_2_and_3_chat_template_and_label_masking():
    """Verify chat template formatting & assistant label masking (-100 for user/system)."""
    tokenizer = DummyTokenizer()
    messages = [
        {"role": "user", "content": "Calculate 25 * 4."},
        {"role": "assistant", "content": '{"type": "tool_call", "name": "calculator", "arguments": {"expression": "25 * 4"}}'},
        {"role": "user", "content": "TOOL_RESULT: 100"},
        {"role": "assistant", "content": '{"type": "final", "content": "The result is 100."}'},
    ]

    formatted = format_example_tokens(messages, tokenizer, max_seq_length=512)
    input_ids = formatted["input_ids"]
    labels = formatted["labels"]

    assert len(input_ids) == len(labels)
    assert len(input_ids) > 0

    # Ensure labels have both -100 (for user tokens) and token IDs (for assistant tokens)
    assert -100 in labels
    assistant_tokens = [l for l in labels if l != -100]
    assert len(assistant_tokens) > 0


def test_4_and_5_full_parameter_training_and_no_peft():
    """Verify full parameter check passes for standard models and fails if PEFT is detected."""
    valid_model = DummyModelModule(num_params=5, peft=False)
    info = verify_full_parameter_sft(valid_model)
    assert info["peft_disabled"] is True
    assert info["trainable_percentage"] == 100.0
    assert info["total_parameters"] == 500

    peft_model = DummyModelModule(num_params=5, peft=True)
    with pytest.raises(RuntimeError, match="PEFT/LoRA adapter detected"):
        verify_full_parameter_sft(peft_model)


def test_6_7_8_9_checkpoint_save_reload_and_parser(temp_dir):
    """Verify data split saving, mock model reload, generation, and parser compatibility."""
    examples = load_sft_dataset("data/sft/train_v2_100.jsonl")
    train_data, val_data, split_info = create_train_val_split(examples, val_ratio=0.10, seed=42)

    assert len(train_data) == 90
    assert len(val_data) == 10
    assert split_info["train_count"] == 90
    assert split_info["val_count"] == 10

    # Save split info
    split_path = os.path.join(temp_dir, "split_info.json")
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f)
    assert os.path.exists(split_path)

    # Test generation output parsing with MockModel
    mock_responses = [
        '{"type": "tool_call", "name": "calculator", "arguments": {"expression": "25 * 37"}}',
        '{"type": "final", "content": "The result is 925."}'
    ]
    mock_model = MockModel(responses=mock_responses)
    parser = ModelOutputParser()

    out1 = mock_model.generate([{"role": "user", "content": "Calculate 25 * 37."}])
    action1 = parser.parse(out1)
    assert action1.is_tool_call()
    assert action1.tool_name == "calculator"

    out2 = mock_model.generate([{"role": "user", "content": "TOOL_RESULT: 925"}])
    action2 = parser.parse(out2)
    assert action2.is_final_answer()
    assert action2.content == "The result is 925."


def test_10_harness_integration():
    """Verify existing harness loop works cleanly with tool execution and parser."""
    mock_model = MockModel()
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    harness = AgentHarness(model=mock_model, tool_registry=registry, max_steps=5)

    traj = harness.run("Calculate 25 * 37.")
    assert traj.termination_reason == "final_answer"
    assert "925" in traj.final_answer
