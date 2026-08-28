import pytest
from src.rl.config import GRPOConfig


def test_grpo_config_defaults():
    config = GRPOConfig()
    assert config.group_size == 4
    assert config.learning_rate == 5e-6
    assert config.beta == 0.04
    assert config.starting_model_path == "outputs/qwen3-1.7b-tool-agent-sft-v4-500"
    
    d = config.to_dict()
    assert isinstance(d, dict)
    assert d["group_size"] == 4
