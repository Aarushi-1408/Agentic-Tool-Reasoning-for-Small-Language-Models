import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    Dataset = object


def load_sft_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """
    Load SFT dataset from JSONL file without modifying original file.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at '{dataset_path}'")
    
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def create_train_val_split(
    examples: List[Dict[str, Any]],
    val_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Create deterministic train and validation split from dataset.
    """
    rng = random.Random(seed)
    indexed_examples = list(enumerate(examples))
    rng.shuffle(indexed_examples)

    val_size = int(len(examples) * val_ratio)
    val_indexed = indexed_examples[:val_size]
    train_indexed = indexed_examples[val_size:]

    # Sort back by original index to keep relative order
    train_indexed.sort(key=lambda x: x[0])
    val_indexed.sort(key=lambda x: x[0])

    train_data = [item[1] for item in train_indexed]
    val_data = [item[1] for item in val_indexed]

    split_info = {
        "seed": seed,
        "val_ratio": val_ratio,
        "total_examples": len(examples),
        "train_count": len(train_data),
        "val_count": len(val_data),
        "train_indices": [item[0] for item in train_indexed],
        "val_indices": [item[0] for item in val_indexed],
        "train_trajectory_ids": [item[1].get("trajectory_id", f"idx_{item[0]}") for item in train_indexed],
        "val_trajectory_ids": [item[1].get("trajectory_id", f"idx_{item[0]}") for item in val_indexed],
    }

    return train_data, val_data, split_info


def format_example_tokens(
    messages: List[Dict[str, Any]],
    tokenizer: Any,
    max_seq_length: int = 1024,
) -> Dict[str, List[int]]:
    """
    Apply Qwen chat template turn-by-turn to build input_ids and labels.
    Tokens corresponding to assistant roles are target labels for loss,
    while system and user prompt tokens are masked with -100.
    """
    input_ids: List[int] = []
    labels: List[int] = []

    accumulated_messages: List[Dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "user")
        accumulated_messages.append(msg)

        if hasattr(tokenizer, "apply_chat_template"):
            try:
                formatted_str = tokenizer.apply_chat_template(
                    accumulated_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception:
                formatted_str = "\n".join(
                    [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in accumulated_messages]
                )
        else:
            formatted_str = "\n".join(
                [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in accumulated_messages]
            )

        turn_ids = tokenizer.encode(formatted_str, add_special_tokens=False)

        new_token_count = len(turn_ids) - len(input_ids)
        if new_token_count > 0:
            turn_new_tokens = turn_ids[len(input_ids):]
            input_ids.extend(turn_new_tokens)

            if role == "assistant":
                labels.extend(turn_new_tokens)
            else:
                labels.extend([-100] * new_token_count)

    # Truncate if exceeds max_seq_length
    if len(input_ids) > max_seq_length:
        input_ids = input_ids[:max_seq_length]
        labels = labels[:max_seq_length]

    # Ensure at least one label token is non--100 to prevent CausalLM loss calculation failure
    if labels and all(lbl == -100 for lbl in labels):
        labels[-1] = input_ids[-1]

    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class SFTDataset(Dataset):
    """
    PyTorch Dataset wrapper for tool-use SFT examples.
    """

    def __init__(
        self,
        examples: List[Dict[str, Any]],
        tokenizer: Any,
        max_seq_length: int = 1024,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.examples[idx]
        messages = item.get("messages", [])
        formatted = format_example_tokens(messages, self.tokenizer, self.max_seq_length)
        return formatted


class DataCollatorForSFT:
    """
    Dynamic padding collator for SFT batching with loss masking (-100).
    """

    def __init__(self, tokenizer: Any, pad_to_multiple_of: Optional[int] = None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_token_id = (
            tokenizer.pad_token_id
            if getattr(tokenizer, "pad_token_id", None) is not None
            else getattr(tokenizer, "eos_token_id", 0)
        )

    def __call__(self, batch: List[Dict[str, List[int]]]) -> Dict[str, Any]:
        import torch

        max_len = max(len(feature["input_ids"]) for feature in batch)
        if self.pad_to_multiple_of is not None and self.pad_to_multiple_of > 0:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for feature in batch:
            input_ids = feature["input_ids"]
            attention_mask = feature["attention_mask"]
            labels = feature["labels"]

            remainder = max_len - len(input_ids)
            padded_input_ids = input_ids + [self.pad_token_id] * remainder
            padded_attention_mask = attention_mask + [0] * remainder
            padded_labels = labels + [-100] * remainder

            batch_input_ids.append(padded_input_ids)
            batch_attention_mask.append(padded_attention_mask)
            batch_labels.append(padded_labels)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }
