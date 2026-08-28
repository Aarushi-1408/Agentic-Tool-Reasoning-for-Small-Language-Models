import os
import sys
import json
import pytest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.trajectory import Trajectory, TrajectoryStep
from src.rl.reward import calculate_reward, RewardResult


def build_trajectory(task_id: str, prompt: str, steps_data: list, final_ans: str = None, term_reason: str = "final_answer") -> Trajectory:
    traj = Trajectory(task_id=task_id, prompt=prompt)
    for idx, s in enumerate(steps_data):
        action_type = s.get("action_type", "tool_call")
        step_obj = TrajectoryStep(
            step=idx,
            model_output=s.get("model_output", ""),
            action_type=action_type,
            tool_name=s.get("tool_name"),
            arguments=s.get("arguments"),
            tool_result=s.get("tool_result"),
            error=s.get("error"),
        )
        traj.add_step(step_obj)
    traj.final_answer = final_ans
    traj.termination_reason = term_reason
    return traj


def test_reward_safety_cases():
    print("\n==================== TESTING GRPO REWARD SAFETY CASES ====================")
    
    # Task metadata definitions
    meta_no_tool = {
        "id": "notool_test",
        "category": "no_tool",
        "requires_tool": False,
        "expected_answer": "Tokyo",
        "expected_behavior": {"allowed_tools": [], "expected_steps": 1}
    }

    meta_single = {
        "id": "single_test",
        "category": "single_tool",
        "requires_tool": True,
        "expected_answer": "925",
        "expected_behavior": {"allowed_tools": ["calculator"], "expected_steps": 2}
    }

    meta_multi = {
        "id": "multi_test",
        "category": "multi_step",
        "requires_tool": True,
        "expected_answer": "100",
        "expected_behavior": {"allowed_tools": ["lookup", "calculator"], "expected_steps": 3}
    }

    # Case 1: no_tool -> direct correct final answer
    t1 = build_trajectory("c1", "Capital of Japan?", [{"action_type": "final"}], final_ans="Tokyo")
    r1 = calculate_reward(t1, meta_no_tool)

    # Case 2: no_tool -> unnecessary calculator -> correct final answer
    t2 = build_trajectory("c2", "Capital of Japan?", [
        {"action_type": "tool_call", "tool_name": "calculator", "arguments": {"expression": "1+1"}},
        {"action_type": "final"}
    ], final_ans="Tokyo")
    r2 = calculate_reward(t2, meta_no_tool)

    # Case 2b: no_tool -> unnecessary tool -> incorrect final answer
    t2b = build_trajectory("c2b", "Capital of Japan?", [
        {"action_type": "tool_call", "tool_name": "calculator", "arguments": {"expression": "1+1"}},
        {"action_type": "final"}
    ], final_ans="Paris")
    r2b = calculate_reward(t2b, meta_no_tool)

    # Case 3: single_tool -> correct tool -> correct final answer
    t3 = build_trajectory("c3", "25 * 37", [
        {"action_type": "tool_call", "tool_name": "calculator", "arguments": {"expression": "25 * 37"}},
        {"action_type": "final"}
    ], final_ans="925")
    r3 = calculate_reward(t3, meta_single)

    # Case 4: single_tool -> correct tool repeated 5 times
    t4_steps = [{"action_type": "tool_call", "tool_name": "calculator", "arguments": {"expression": "25 * 37"}} for _ in range(5)]
    t4_steps.append({"action_type": "final"})
    t4 = build_trajectory("c4", "25 * 37", t4_steps, final_ans="925")
    r4 = calculate_reward(t4, meta_single)

    # Case 5: multi_step -> all tools completed -> correct final
    t5 = build_trajectory("c5", "Complex task", [
        {"action_type": "tool_call", "tool_name": "lookup", "arguments": {"key": "a"}},
        {"action_type": "tool_call", "tool_name": "lookup", "arguments": {"key": "b"}},
        {"action_type": "tool_call", "tool_name": "calculator", "arguments": {"expression": "1+1"}},
        {"action_type": "final"}
    ], final_ans="100")
    r5 = calculate_reward(t5, meta_multi)

    # Case 6: multi_step -> only first tool -> premature final answer
    t6 = build_trajectory("c6", "Complex task", [
        {"action_type": "tool_call", "tool_name": "lookup", "arguments": {"key": "a"}},
        {"action_type": "final"}
    ], final_ans="100")
    r6 = calculate_reward(t6, meta_multi)

    # Case 7: malformed tool call (parse error)
    t7 = build_trajectory("c7", "25 * 37", [
        {"action_type": "parse_error", "error": "Malformed JSON"},
        {"action_type": "final"}
    ], final_ans="925")
    r7 = calculate_reward(t7, meta_single)

    # Case 8: wrong tool
    t8 = build_trajectory("c8", "25 * 37", [
        {"action_type": "tool_call", "tool_name": "string_manipulator", "arguments": {"text": "hello"}},
        {"action_type": "final"}
    ], final_ans="925")
    r8 = calculate_reward(t8, meta_single)

    print(f"CASE 1 (no_tool direct correct)             : {r1.total_reward:+.2f} | {r1.breakdown_string}")
    print(f"CASE 2 (no_tool unnec tool + correct)        : {r2.total_reward:+.2f} | {r2.breakdown_string}")
    print(f"CASE 2b(no_tool unnec tool + wrong)          : {r2b.total_reward:+.2f} | {r2b.breakdown_string}")
    print(f"CASE 3 (single_tool correct)                 : {r3.total_reward:+.2f} | {r3.breakdown_string}")
    print(f"CASE 4 (single_tool repeated 5x)             : {r4.total_reward:+.2f} | {r4.breakdown_string}")
    print(f"CASE 5 (multi_step full chain correct)       : {r5.total_reward:+.2f} | {r5.breakdown_string}")
    print(f"CASE 6 (multi_step premature stop)           : {r6.total_reward:+.2f} | {r6.breakdown_string}")
    print(f"CASE 7 (malformed parse error)               : {r7.total_reward:+.2f} | {r7.breakdown_string}")
    print(f"CASE 8 (wrong tool)                          : {r8.total_reward:+.2f} | {r8.breakdown_string}")

    # VERIFY REQUIRED ORDER RELATIONSHIPS
    # 1. Correct trajectories > incorrect trajectories
    assert r1.total_reward > r2b.total_reward, "Correct answer must score higher than incorrect answer."
    
    # 2. correct no-tool answer > unnecessary tool + correct answer > unnecessary tool + incorrect answer
    assert r1.total_reward > r2.total_reward > r2b.total_reward, "No-tool ordering violated!"

    # 3. complete multi-step answer > premature multi-step answer
    assert r5.total_reward > r6.total_reward, "Multi-step completeness ordering violated!"

    # 4. correct single tool > repeated tool loop
    assert r3.total_reward > r4.total_reward, "Repetition penalty ordering violated!"

    print("==================== ALL REWARD SAFETY RELATIONSHIPS VERIFIED ====================\n")

    # Save reward sanity check JSON
    sanity_data = {
        "case_1_no_tool_direct_correct": r1.to_dict(),
        "case_2_no_tool_unnecessary_tool_correct": r2.to_dict(),
        "case_2b_no_tool_unnecessary_tool_wrong": r2b.to_dict(),
        "case_3_single_tool_correct": r3.to_dict(),
        "case_4_single_tool_repeated_loop": r4.to_dict(),
        "case_5_multi_step_complete_correct": r5.to_dict(),
        "case_6_multi_step_premature_stop": r6.to_dict(),
        "case_7_parse_error": r7.to_dict(),
        "case_8_wrong_tool": r8.to_dict(),
        "verification_status": "PASSED"
    }

    out_dir = Path("evaluation/grpo1")
    os.makedirs(out_dir, exist_ok=True)
    with open(out_dir / "reward_sanity_check.json", "w", encoding="utf-8") as f:
        json.dump(sanity_data, f, indent=2)


if __name__ == "__main__":
    test_reward_safety_cases()
