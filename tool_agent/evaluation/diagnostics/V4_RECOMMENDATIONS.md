# SFT v4 Dataset Design Proposal & Implementation Roadmap

**Project:** Agentic Tool Reasoning for Small Language Models (Qwen3-1.7B)  
**Target:** SFT v4 Dataset (`data/sft/train_v4_500.jsonl`)  
**Objective:** Surpass Baseline Accuracy (>85.0%) by eliminating Tool Over-Activation and Premature Multi-Step Stopping.  

---

## 1. Executive Summary & Recommended Proportions

Based on empirical evidence from SFT v2 (70.0%) and SFT v3 (65.0%), naive 65% tool-biased datasets cause an **83.33% Unnecessary Tool-Use Rate** on direct knowledge tasks.

To achieve **>85% Overall Benchmark Accuracy**, SFT v4 must enforce a strict **50% No-Tool / 50% Tool-Required** balanced design.

### Recommended Category Distribution (500 Trajectories Total)

| Category | Recommended Count | Percentage | Target Function / Role |
| :--- | :---: | :---: | :--- |
| **no_tool** | **250** | **50.0%** | Suppress tool over-activation; teach direct parametric response |
| **single_tool** | **100** | **20.0%** | Single tool execution (`calculator`, `lookup`, `string_manipulator`, `echo`) |
| **two_step** | **75** | **15.0%** | Two-tool reasoning chains (`lookup -> calculator -> final`) |
| **multi_step** | **50** | **10.0%** | 3-step & 4-step continuation chains (`tool -> result -> tool -> result -> final`) |
| **error_recovery** | **25** | **5.0%** | Invalid argument / missing key recovery (`invalid -> error -> correction`) |

---

## 2. Detailed Trajectory Breakdown & Target Specs

### 1. Direct-Answer Trajectories (`no_tool`): 250 Examples
- **Structure:** `user -> assistant (final answer)`
- **Goal:** Protect parametric knowledge and eliminate Category A unnecessary tool calls.
- **Diversity Target:** 250 unique prompts covering general knowledge, science, literature, history, and basic reasoning.

### 2. Multi-Step Continuation Trajectories: 125 Examples (`two_step` + `multi_step`)
- **Structure:** `user -> assistant (tool_call) -> user (TOOL_RESULT) -> assistant (tool_call) -> user (TOOL_RESULT) -> assistant (final)`
- **Goal:** Fix Category F premature termination by explicitly training the model to emit a *second tool call* after receiving the first `TOOL_RESULT`.

### 3. Error-Recovery Trajectories: 25 Examples
- **Structure:** `user -> assistant (invalid tool_call) -> user (TOOL_RESULT error) -> assistant (corrected tool_call) -> user (TOOL_RESULT) -> assistant (final)`
- **Goal:** Maintain 100% error recovery accuracy.

---

## 3. Dataset Generation Specifications for `task_generator.py`

```python
# SFT v4 Task Generation Distribution Configuration
SFT_V4_CONFIG = {
    "num_no_tool": 250,        # 50%
    "num_single_tool": 100,    # 20%
    "num_two_step": 75,        # 15%
    "num_multi_step": 50,      # 10%
    "num_error_recovery": 25,  # 5%
    "total_examples": 500,
    "min_unique_prompts": 400  # Strict template expansion to prevent overfitting
}
```

---

## 4. Expected Performance Metrics for SFT v4

| Metric | Baseline | SFT v3 (Current) | SFT v4 Target | Validation Criteria |
| :--- | :---: | :---: | :---: | :--- |
| **Overall Accuracy** | 80.00% | 65.00% | **>85.00%** | Pass >51 of 60 eval tasks |
| **No-Tool Accuracy** | 91.67% | 33.33% | **>83.33%** | Pass >10 of 12 no-tool tasks |
| **Unnecessary Tool Rate** | 8.33% | 83.33% | **<16.67%** | <2 unnecessary tool calls |
| **Tool Selection Accuracy**| 79.17% | 100.00% | **100.00%** | Maintain 100% selection |
| **Multi-Step Accuracy** | 55.56% | 50.00% | **>75.00%** | Pass >18 of 24 multi-step tasks |
| **Timeout Failures** | 0 | 0 | **0** | 0 max-step timeouts |
