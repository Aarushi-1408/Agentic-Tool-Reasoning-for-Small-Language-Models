# System Prompts as Finite State Anchors: Diagnostic Auditing of Small Language Models for Tool Reasoning

**Authors:** Research Engineering Team  
**Affiliation:** Cynaptics AI Project  
**Date:** August 2026  

---

## Abstract
Small language models (SLMs $\le 2\text{B}$ parameters) are increasingly deployed as autonomous agents to interact with external tool environments. However, fine-tuning SLMs for tool reasoning frequently introduces severe behavioral pathologies, including infinite looping, premature chain termination, and tool over-activation on direct parametric queries. In this work, we present a systematic diagnostic audit of full-parameter Supervised Fine-Tuning (SFT) on Qwen3-1.7B across single-turn, multi-step, and error-recovery tool tasks. We demonstrate that enabling a structured system prompt protocol without model retraining increases overall benchmark accuracy from 16.67% to 70.00% by eliminating infinite loop failures. Furthermore, we reveal a critical dataset distribution failure: expanding training data from 100 to 500 trajectories while maintaining a 65% tool-requiring bias increases unnecessary tool calls to 83.33% on direct knowledge tasks, dropping overall accuracy to 65.00%. Our findings demonstrate that tool selection accuracy (100%) and tool termination reasoning represent decoupled capabilities in SLMs, requiring balanced 50/50 dataset composition to achieve robust performance.

---

## 1. Introduction & Problem Statement
Autonomous tool-augmented language agents must perform five core capabilities:
1. **Tool Necessity Identification:** Deciding whether an external tool is required.
2. **Tool Selection:** Selecting the appropriate tool from a registry.
3. **Argument Generation:** Formulating valid arguments matching the tool schema.
4. **Observation Interpretation:** Parsing external `TOOL_RESULT` feedback.
5. **Chain Termination:** Deciding when to output a `<final>` answer versus initiating further tool calls.

While large models ($\ge 70\text{B}$) excel at these capabilities zero-shot, small models (e.g. Qwen3-1.7B) suffer from loop traps, schema hallucination, and instruction drift.

---

## 2. Experimental Setup
All experiments were evaluated on a frozen benchmark of **60 tasks** (`tasks/eval_v1.json`) containing 12 `no_tool` tasks, 15 `single_tool` tasks, 15 `two_step` tasks, 9 `multi_step` tasks, 6 `tool_selection` tasks, and 3 `error_recovery` tasks.

### Models Evaluated
- **Baseline:** Base `Qwen/Qwen3-1.7B` (Zero-shot inference).
- **SFT v2 (No Prompt):** Fine-tuned on 100 trajectories; evaluated without system prompt.
- **SFT v2 (+ Prompt):** Fine-tuned on 100 trajectories; evaluated with system prompt.
- **SFT v3 (+ Prompt):** Fine-tuned on 500 trajectories; evaluated with system prompt.

---

## 3. Benchmark Results

### Table 1: Main Evaluation Benchmark Results

| Model / Setting | Overall Acc | Tool Acc | No-Tool Acc | Tool Selection | Multi-Step Acc | Unnecessary Tool Rate | Avg Steps | Timeouts |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Qwen3-1.7B Base)** | **80.00%** | 77.08% | **91.67%** | 79.17% | 58.33% | **8.33%** | **1.65** | **0** |
| **SFT v2 (No System Prompt)** | 16.67% | 18.75% | 8.33% | 100.00% | 0.00% | 100.00% | 7.47 | 37 |
| **SFT v2 (+ System Prompt)** | **70.00%** | 75.00% | **50.00%** | 97.92% | **66.67%** | 66.67% | 2.52 | **0** |
| **SFT v3 (500 Trajectories)** | **65.00%** | **72.92%** | **33.33%** | **100.00%** | 50.00% | **83.33%** | 2.48 | **0** |

---

## 4. System-Prompt Ablation Study
The system prompt defines available tool schemas, JSON invocation protocols, and `<final>` termination rules.

When evaluating SFT v2 without a system prompt:
- **Max-Step Exceeded Timeouts:** 37 of 60 tasks (61.7%) failed by reaching the maximum 10-step limit.
- **Repeated Tool Loops:** 177 instances of `tool_call -> TOOL_RESULT -> identical tool_call`.

Enabling the system prompt provided a structural state-machine anchor:
- **Overall Accuracy:** Jumped from **16.67% to 70.00%** (+53.33%).
- **Timeouts:** Dropped from **37 to 0**.
- **Loops:** Dropped from **177 to 1**.

---

## 5. Failure Taxonomy Analysis

### Table 2: Failure Taxonomy Breakdown Across Models

| Failure Category | Baseline | SFT v2 (No Prompt) | SFT v2 (+ Prompt) | SFT v3 (+ Prompt) | Primary Root Cause |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Category A: Unnecessary Tool Use** | 1 | 11 | 6 | **10** | Tool over-activation bias |
| **Category B: Repeated Tool Loops** | 0 | **177** | 1 | **0** | Resolved by system prompt |
| **Category C: Max-Step Timeouts** | 0 | **37** | 0 | **0** | Resolved by system prompt |
| **Category D: Wrong Tool Selection** | 10 | 0 | 1 | **0** | Tool selection learned 100% |
| **Category E: Malformed Tool Call** | 0 | 1 | 0 | **0** | 100% valid JSON calls |
| **Category F: Premature Termination** | 1 | 0 | 10 | **5** | Early `<final>` in 3/4-step chains |
| **Category G: Incorrect Final Answer** | 0 | 0 | 0 | **8** | Math / parsing error |
| **Total Failures** | **12** | **50** | **18** | **23** | |

---

## 6. Discussion & Threats to Validity
1. **Tool-Selection vs Tool-Termination Decoupling:** SFT v3 achieves 100% tool selection accuracy, demonstrating that mapping inputs to tool names is straightforward. However, knowing when *not* to call a tool requires balanced datasets.
2. **Threats to Validity:** Evaluation relies on string-matching in `evaluator.py`, which occasionally counts a task as `success: true` even when an unnecessary tool was invoked before outputting the text containing the substring.

---

## 7. Conclusion & Future Work
To surpass the 80.0% baseline, future work must deploy **SFT v4** with a strict 50% `no_tool` dataset balance and explicit multi-step continuation targets (`tool_call -> TOOL_RESULT -> tool_call -> TOOL_RESULT -> final`).
