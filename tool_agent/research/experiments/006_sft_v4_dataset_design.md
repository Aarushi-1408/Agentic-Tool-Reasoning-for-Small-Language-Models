# Experiment Design Record: 006_sft_v4_dataset_design

- **Document ID:** `006_sft_v4_dataset_design`
- **Date:** 2026-08-28
- **Target Model:** `Qwen/Qwen3-1.7B`
- **Generated Dataset Files:**
  - Raw Trajectories: [`data/raw/teacher_trajectories_v4_500.jsonl`](file:///c:/Users/bpraj/OneDrive/Desktop/cynaptics_project/tool_agent/data/raw/teacher_trajectories_v4_500.jsonl)
  - SFT Dataset: [`data/sft/train_v4_500.jsonl`](file:///c:/Users/bpraj/OneDrive/Desktop/cynaptics_project/tool_agent/data/sft/train_v4_500.jsonl)
  - Dataset Statistics: [`data/sft/dataset_stats_v4.json`](file:///c:/Users/bpraj/OneDrive/Desktop/cynaptics_project/tool_agent/data/sft/dataset_stats_v4.json)
- **Status:** Dataset Created & Audited (Training NOT Started)

---

## 1. Why SFT V3 Failed

Supervised Fine-Tuning on the 500-trajectory `v3` dataset (`sft_v3_500`) yielded an overall benchmark accuracy of **65.00%**, representing a **-15.00% regression** compared to the pre-trained zero-shot base model (80.00%) and a **-5.00% regression** compared to SFT-100+Prompt (70.00%).

While `v3` successfully eliminated infinite repetition loops (0 timeout failures), it failed due to two fundamental policy flaws:
1. **Tool Over-Activation:** Severe tool bias caused the model to invoke tools on queries that requested direct parametric answers.
2. **Premature Termination:** On multi-step reasoning chains requiring 3 to 4 sequential tool invocations, the model frequently stopped early after step 1 or step 2.

---

## 2. Evidence for Failure Modes in V3

### A. Evidence for Tool Over-Activation
- **No-Tool Task Accuracy:** Dropped from **91.67% (Base)** down to **33.33% (SFT-500)** (-58.33% regression).
- **Unnecessary Tool-Use Rate:** Spiked from **8.33% (Base)** up to **83.33% (SFT-500)** (+75.00% increase).
- **Diagnosis:** In SFT `v3`, 65% of all trajectories required tool calls (325 / 500), while only 35% (175 / 500) were direct knowledge queries. This unbalanced 65:35 ratio trained the model to view tool invocation as the default action for almost every user prompt.

### B. Evidence for Premature Termination
- **Multi-Step Accuracy:** Dropped from **55.56% (Base)** down to **22.22% (SFT-500)** (-33.33% regression).
- **Behavioral Trajectory Pattern:** On tasks like `multistep_001` or `multistep_004`, after completing the 1st or 2nd tool call, the model emitted a final answer output without performing the remaining required tool steps.
- **Diagnosis:** SFT `v3` lacked sufficient long-horizon 3-step and 4-step continuation trajectories, causing the model to learn a strong bias toward emitting a final answer after 1 or 2 tool turns.

---

## 3. SFT V4 Design Rationale

The primary objective of SFT V4 is to train a balanced decision boundary policy:

> *"Use a tool only when necessary, and continue using tools until the task is actually solved."*

To achieve this, SFT V4 enforces:
1. **Strict 50:50 Decision Boundary Balance:** 250 `no_tool` trajectories (50.0%) vs 250 tool-required trajectories (50.0%) to suppress tool over-activation.
2. **Long-Horizon Multi-Step Chains:** 75 long multi-step trajectories (3-step and 4-step chains) where later tool arguments depend explicitly on earlier tool observations.
3. **Explicit Stop vs Continue Transition Signals:** Providing equal density of `tool_call -> TOOL_RESULT -> final` (Stop when done) and `tool_call -> TOOL_RESULT -> different tool_call` (Continue when more information is required).

---

## 4. Dataset Composition: SFT V3 vs SFT V4 Comparison

| Category | SFT V3 Count | SFT V3 Share | SFT V4 Target | SFT V4 Share | Primary Policy Function |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`no_tool`** | 175 | 35.0% | **250** | **50.0%** | Teach decision boundary to respond directly without tools |
| **`single_tool`** | 115 | 23.0% | **75** | **15.0%** | Teach model to stop after 1 sufficient tool turn |
| **`two_step`** | 90 | 18.0% | **75** | **15.0%** | Teach 2-turn dependent tool invocation chains |
| **`multi_step`** | 90 | 18.0% | **75** | **15.0%** | Teach long-horizon 3-step and 4-step continuation |
| **`error_recovery`** | 30 | 6.0% | **25** | **5.0%** | Teach error recovery (`TOOL_ERROR -> correction -> final`) |
| **Total** | **500** | **100.0%** | **500** | **100.0%** | **Balanced 50% No-Tool / 50% Tool-Required Dataset** |

---

## 5. Expected Behavioral Improvements

1. **Suppression of Tool Over-Activation:** Increasing `no_tool` trajectories to 50% of the dataset will reduce the unnecessary tool-use rate from 83.33% down to **<16.67%**, restoring direct knowledge accuracy to **>83.33%**.
2. **Resolution of Premature Termination:** Adding 75 dedicated multi-step continuation chains will increase multi-step accuracy from 22.22% to **>75.00%**.
3. **Target Benchmark Accuracy:** Surpassing the **>85.00%** overall benchmark accuracy threshold on `tasks/eval_v1.json`.

---

## 6. Dataset Quality Audit & Validation Statistics (`train_v4_500.jsonl`)

The generated V4 dataset was processed through `SFTConverter` and passed 100% of validation checks:

- **Total Valid Trajectories:** **500 / 500** (100.0% acceptance rate)
- **Category Breakdown:**
  - `no_tool`: **250** (50.0%)
  - `single_tool`: **75** (15.0%)
  - `two_step`: **75** (15.0%)
  - `multi_step`: **75** (15.0%)
  - `error_recovery`: **25** (5.0%)
- **Quality Metrics:**
  - Final answer completion rate: **100.0%** (500 / 500)
  - Malformed tool calls: **0**
  - Repeated identical tool calls: **0**
  - Non-terminating timeouts / loops: **0**
  - Average tool calls per trajectory: **1.02**
  - Maximum tool calls in a trajectory: **5**
- **Transition Signal Density:**
  - `tool_call -> TOOL_RESULT -> final` (Stop signal): **250**
  - `tool_call -> TOOL_RESULT -> different tool_call` (Continue signal): **258**
  - `tool_call -> TOOL_RESULT -> SAME tool_call` (Repetition signal): **0**
- **Chain Length Distribution:**
  - 1-step (`tool -> result -> final`): **112**
  - 2-step (`2 tools -> final`): **73**
  - 3-step (`3 tools -> final`): **18**
  - 4-step (`4 tools -> final`): **39**
  - 5-step (`5 tools -> final`): **8**

---

## 7. Next Step: Exact Training & Evaluation Commands

When ready to train full-parameter SFT on V4:

### 1. Training Command (Full SFT - No PEFT / LoRA)
```bash
python scripts/train_sft.py \
  --model Qwen/Qwen3-1.7B \
  --dataset data/sft/train_v4_500.jsonl \
  --output_dir outputs/qwen3-1.7b-tool-agent-sft-v4-500 \
  --epochs 3 \
  --learning_rate 2e-5 \
  --batch_size 2 \
  --gradient_accumulation_steps 4
```

### 2. Evaluation Command (Frozen Benchmark with System Prompt Protocol)
```bash
python scripts/evaluate.py \
  --tasks tasks/eval_v1.json \
  --model outputs/qwen3-1.7b-tool-agent-sft-v4-500/final \
  --system-prompt \
  --output-dir evaluation/sft_v4_500
```
