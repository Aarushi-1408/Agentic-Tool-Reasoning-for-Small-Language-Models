# Research Overview: Agentic Tool Reasoning for Small Language Models (SLMs)

## Research Question
*"How do training-data composition and inference-protocol alignment affect tool-use reliability in small language models?"*

---

## Executive Summary

Small Language Models (e.g. `Qwen/Qwen3-1.7B` and `Qwen/Qwen3-4B`) present unique challenges when deployed as autonomous agents in multi-step tool execution loops:
1. **Tool Over-Activation Bias:** Fine-tuning on tool-heavy trajectories causes SLMs to hallucinate tool calls on direct answer / `no_tool` tasks (100% unnecessary tool use in SFT v2 without prompt).
2. **Infinite Tool Loops:** Without strict inference protocol enforcement, SLMs repeatedly invoke identical tool calls without transitioning to final answer synthesis.
3. **Inference-Protocol Alignment:** Enforcing system prompts acts as a state anchor, eliminating 100% of max-step timeouts and boosting benchmark accuracy from **16.67% to 70.00%** (+53.33%).
4. **Data Composition Balancing:** SFT v3 balances dataset composition (35% no-tool, 23% single-tool, 18% two-step, 18% multi-step, 6% error recovery/selection) to address remaining failure modes.

---

## Core Hypotheses

- **H1 (Data Composition):** Rebalancing `no_tool` trajectories (35%) suppresses tool over-activation on direct answer tasks.
- **H2 (Long Horizon Chains):** Increasing 3-step and 4-step tool chains (18%) resolves premature multi-step workflow termination.
- **H3 (State Transition Supervision):** Explicit supervision of `tool_call -> TOOL_RESULT -> final_answer` transitions prevents repetitive loops.
- **H4 (Error Recovery):** Trajectories demonstrating fallback behavior after `TOOL_ERROR` observations improve robustness.
- **H5 (Formatting Alignment):** Standardized, unadorned target answers improve exact-match evaluation reliability.

---

## Directory Structure

- `experiment_log.md`: Detailed chronological log of all research runs.
- `results/results_table.md`: Immutable persistent metric comparison table.
- `experiments/`: Individual experiment documentation files (`001_baseline` through `005_sft_v3`).
- `dataset_versions.md`: Registry of versioned training and evaluation datasets.
