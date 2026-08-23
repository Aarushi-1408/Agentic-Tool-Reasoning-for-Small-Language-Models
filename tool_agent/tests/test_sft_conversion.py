import json
import os
import tempfile
import pytest
from src.data.convert_to_sft import SFTConverter, clean_think_tags


def test_1_no_tool_trajectory_conversion():
    """Verify conversion of direct no-tool user/assistant conversation."""
    record = {
        "trajectory_id": "traj_001",
        "task_id": "t1",
        "task": "What is the capital of Japan?",
        "messages": [
            {"role": "user", "content": "What is the capital of Japan?"},
            {"role": "assistant", "content": "The capital of Japan is Tokyo."}
        ],
        "steps": [
            {"step": 0, "action_type": "final", "model_output": "The capital of Japan is Tokyo."}
        ],
        "final_answer": "The capital of Japan is Tokyo.",
        "termination_reason": "final_answer"
    }

    converter = SFTConverter()
    sft_rec, rej_reason, category = converter.convert_trajectory(record)

    assert rej_reason is None
    assert category == "no_tool"
    assert sft_rec["trajectory_id"] == "traj_001"
    assert len(sft_rec["messages"]) == 2
    assert sft_rec["messages"][0] == {"role": "user", "content": "What is the capital of Japan?"}
    assert sft_rec["messages"][1] == {"role": "assistant", "content": "The capital of Japan is Tokyo."}


def test_2_single_tool_trajectory_conversion():
    """Verify single-tool trajectory conversion preserving assistant tool call and tool result."""
    record = {
        "trajectory_id": "traj_002",
        "task_id": "t2",
        "task": "Calculate 144 / 12.",
        "messages": [
            {"role": "user", "content": "Calculate 144 / 12."},
            {"role": "assistant", "content": '{"type": "tool_call", "name": "calculator", "arguments": {"expression": "144 / 12"}}'},
            {"role": "user", "content": "TOOL_RESULT: 12"},
            {"role": "assistant", "content": "144 divided by 12 is 12."}
        ],
        "steps": [
            {
                "step": 0,
                "action_type": "tool_call",
                "tool_name": "calculator",
                "arguments": {"expression": "144 / 12"},
                "tool_result": "TOOL_RESULT: 12",
                "model_output": '{"type": "tool_call", "name": "calculator", "arguments": {"expression": "144 / 12"}}'
            },
            {
                "step": 1,
                "action_type": "final",
                "model_output": "144 divided by 12 is 12."
            }
        ],
        "final_answer": "144 divided by 12 is 12.",
        "termination_reason": "final_answer"
    }

    converter = SFTConverter()
    sft_rec, rej_reason, category = converter.convert_trajectory(record)

    assert rej_reason is None
    assert category == "single_tool"
    assert len(sft_rec["messages"]) == 4
    assert "calculator" in sft_rec["messages"][1]["content"]
    assert sft_rec["messages"][2]["content"] == "TOOL_RESULT: 12"


def test_3_multi_step_trajectory_conversion():
    """Verify multi-step tool trajectory conversion preserving the full interaction sequence."""
    record = {
        "trajectory_id": "traj_003",
        "task_id": "t3",
        "task": "Lookup and calculate.",
        "messages": [
            {"role": "user", "content": "Lookup and calculate."},
            {"role": "assistant", "content": '{"type": "tool_call", "name": "lookup", "arguments": {"key": "apple_price"}}'},
            {"role": "user", "content": "TOOL_RESULT: 5"},
            {"role": "assistant", "content": '{"type": "tool_call", "name": "calculator", "arguments": {"expression": "5 * 3"}}'},
            {"role": "user", "content": "TOOL_RESULT: 15"},
            {"role": "assistant", "content": "Total is 15."}
        ],
        "steps": [
            {
                "step": 0, "action_type": "tool_call", "tool_name": "lookup",
                "arguments": {"key": "apple_price"}, "tool_result": "TOOL_RESULT: 5",
                "model_output": '{"type": "tool_call", "name": "lookup", "arguments": {"key": "apple_price"}}'
            },
            {
                "step": 1, "action_type": "tool_call", "tool_name": "calculator",
                "arguments": {"expression": "5 * 3"}, "tool_result": "TOOL_RESULT: 15",
                "model_output": '{"type": "tool_call", "name": "calculator", "arguments": {"expression": "5 * 3"}}'
            },
            {
                "step": 2, "action_type": "final", "model_output": "Total is 15."
            }
        ],
        "final_answer": "Total is 15.",
        "termination_reason": "final_answer"
    }

    converter = SFTConverter()
    sft_rec, rej_reason, category = converter.convert_trajectory(record)

    assert rej_reason is None
    assert category == "two_step"
    assert len(sft_rec["messages"]) == 6


def test_4_think_removal():
    """Verify that <think>...</think> blocks are stripped from assistant messages."""
    raw_text = "<think>\nLet me calculate 25 * 37.\nFirst 25 * 30 = 750...\n</think>\n{\"type\": \"tool_call\", \"name\": \"calculator\", \"arguments\": {\"expression\": \"25 * 37\"}}"
    cleaned = clean_think_tags(raw_text)

    assert "<think>" not in cleaned
    assert "Let me calculate" not in cleaned
    assert cleaned == '{"type": "tool_call", "name": "calculator", "arguments": {"expression": "25 * 37"}}'


def test_5_invalid_tool_rejection():
    """Verify rejection when trajectory uses an unregistered tool name."""
    record = {
        "trajectory_id": "traj_bad_tool",
        "task_id": "t_bad",
        "task": "Run unknown tool.",
        "messages": [{"role": "user", "content": "Run unknown tool."}],
        "steps": [
            {
                "step": 0, "action_type": "tool_call", "tool_name": "unknown_tool",
                "arguments": {"x": 1}, "tool_result": "TOOL_RESULT: ok", "model_output": "..."
            },
            {"step": 1, "action_type": "final", "model_output": "done"}
        ],
        "final_answer": "done",
        "termination_reason": "final_answer"
    }

    converter = SFTConverter()
    sft_rec, rej_reason, category = converter.convert_trajectory(record)

    assert sft_rec is None
    assert rej_reason == "invalid_tool"


def test_6_missing_tool_result_rejection():
    """Verify rejection when tool call step is missing execution result."""
    record = {
        "trajectory_id": "traj_missing_res",
        "task_id": "t_no_res",
        "task": "Calc test.",
        "messages": [{"role": "user", "content": "Calc test."}],
        "steps": [
            {
                "step": 0, "action_type": "tool_call", "tool_name": "calculator",
                "arguments": {"expression": "1 + 1"}, "tool_result": None, "model_output": "..."
            },
            {"step": 1, "action_type": "final", "model_output": "done"}
        ],
        "final_answer": "done",
        "termination_reason": "final_answer"
    }

    converter = SFTConverter()
    sft_rec, rej_reason, category = converter.convert_trajectory(record)

    assert sft_rec is None
    assert rej_reason == "missing_tool_result"


def test_7_incomplete_trajectory_rejection():
    """Verify rejection when trajectory reached max steps or lacks final answer."""
    record = {
        "trajectory_id": "traj_max_steps",
        "task_id": "t_max",
        "task": "Loop task.",
        "messages": [{"role": "user", "content": "Loop task."}],
        "steps": [
            {
                "step": 0, "action_type": "tool_call", "tool_name": "calculator",
                "arguments": {"expression": "1 + 1"}, "tool_result": "TOOL_RESULT: 2", "model_output": "..."
            }
        ],
        "final_answer": None,
        "termination_reason": "max_steps_exceeded"
    }

    converter = SFTConverter()
    sft_rec, rej_reason, category = converter.convert_trajectory(record)

    assert sft_rec is None
    assert rej_reason == "max_steps_reached"


def test_8_correct_jsonl_serialization():
    """Verify process_file generates train.jsonl, rejected.jsonl, and dataset_stats.json."""
    record1 = {
        "trajectory_id": "traj_ok",
        "task_id": "t_ok",
        "task": "Hi",
        "messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}],
        "steps": [{"step": 0, "action_type": "final", "model_output": "Hello!"}],
        "final_answer": "Hello!",
        "termination_reason": "final_answer"
    }
    record2 = {
        "trajectory_id": "traj_bad",
        "task_id": "t_bad",
        "task": "Bad",
        "messages": [{"role": "user", "content": "Bad"}],
        "steps": [{"step": 0, "action_type": "tool_call", "tool_name": "fake_tool", "arguments": {}, "tool_result": "res", "model_output": "..."}],
        "final_answer": None,
        "termination_reason": "max_steps_exceeded"
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "raw.jsonl")
        output_path = os.path.join(tmpdir, "train.jsonl")
        rejected_path = os.path.join(tmpdir, "rejected.jsonl")
        stats_path = os.path.join(tmpdir, "stats.json")

        with open(input_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(record1) + "\n")
            f.write(json.dumps(record2) + "\n")

        converter = SFTConverter()
        stats = converter.process_file(
            input_file=input_path,
            output_sft_file=output_path,
            output_rejected_file=rejected_path,
            stats_file=stats_path,
            overwrite=True
        )

        assert stats["raw_trajectories"] == 2
        assert stats["valid_trajectories"] == 1
        assert stats["rejected_trajectories"] == 1
        assert os.path.exists(output_path)
        assert os.path.exists(rejected_path)
        assert os.path.exists(stats_path)


def test_9_preservation_of_tool_arguments():
    """Verify tool arguments are preserved without modification."""
    args_dict = {"operation": "reverse", "text": "Testing 123"}
    record = {
        "trajectory_id": "traj_args",
        "task_id": "t_args",
        "task": "Reverse",
        "messages": [
            {"role": "user", "content": "Reverse"},
            {"role": "assistant", "content": json.dumps({"type": "tool_call", "name": "string_manipulator", "arguments": args_dict})},
            {"role": "user", "content": "TOOL_RESULT: 321 gnitseT"},
            {"role": "assistant", "content": "321 gnitseT"}
        ],
        "steps": [
            {
                "step": 0, "action_type": "tool_call", "tool_name": "string_manipulator",
                "arguments": args_dict, "tool_result": "TOOL_RESULT: 321 gnitseT", "model_output": "..."
            },
            {"step": 1, "action_type": "final", "model_output": "321 gnitseT"}
        ],
        "final_answer": "321 gnitseT",
        "termination_reason": "final_answer"
    }

    converter = SFTConverter()
    sft_rec, rej_reason, category = converter.convert_trajectory(record)

    assert rej_reason is None
    assistant_msg = sft_rec["messages"][1]["content"]
    parsed_msg = json.loads(assistant_msg)
    assert parsed_msg["arguments"] == args_dict


def test_10_preservation_of_tool_results():
    """Verify tool observations/results are preserved in the user tool response message."""
    tool_res_str = "TOOL_RESULT: 925"
    record = {
        "trajectory_id": "traj_res",
        "task_id": "t_res",
        "task": "Calc",
        "messages": [
            {"role": "user", "content": "Calc"},
            {"role": "assistant", "content": '{"type": "tool_call", "name": "calculator", "arguments": {"expression": "25 * 37"}}'},
            {"role": "user", "content": tool_res_str},
            {"role": "assistant", "content": "925"}
        ],
        "steps": [
            {
                "step": 0, "action_type": "tool_call", "tool_name": "calculator",
                "arguments": {"expression": "25 * 37"}, "tool_result": tool_res_str, "model_output": "..."
            },
            {"step": 1, "action_type": "final", "model_output": "925"}
        ],
        "final_answer": "925",
        "termination_reason": "final_answer"
    }

    converter = SFTConverter()
    sft_rec, rej_reason, category = converter.convert_trajectory(record)

    assert rej_reason is None
    user_obs_msg = sft_rec["messages"][2]["content"]
    assert user_obs_msg == tool_res_str
