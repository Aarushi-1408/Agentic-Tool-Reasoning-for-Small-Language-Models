# SFT v3 Diagnostic Audit & Comprehensive Failure Analysis Report

**Date:** 2026-08-27  
**Target Model:** `outputs/qwen3-1.7b-tool-agent-sft-v3`  
**Evaluation Benchmark:** `tasks/eval_v1.json` (60 Frozen Tasks)

---

## 1. Executive Terminal Summary

```text
============================================================
SFT V3 DIAGNOSTIC SUMMARY
============================================================

Baseline Overall       : 80.00%
SFT v2 Overall         : 16.67%
SFT v2 + Prompt Overall: 70.00%
SFT v3 Overall         : 65.00%

V3 No-Tool Accuracy    : 33.33% (4 / 12 tasks)
V3 Unnecessary Tool Use: 83.33% (10 / 12 tasks)
V3 Tool Selection      : 100.00% (48 / 48 tool tasks)
V3 Multi-Step Accuracy : 50.00% (12 / 24 multi-step tasks)
V3 Avg Steps           : 2.48 steps / task

Top Failure Mode       : Unnecessary Tool Use on No-Tool Tasks (Category A: 6 tasks)
Second Failure Mode    : Premature Termination in Multi-Step Chains (Category F: 7 tasks)
Third Failure Mode     : Wrong Tool / Recovery Selection (Category D: 2 tasks)

Main Suspected Cause   : Training-data tool/no-tool imbalance (80% tool bias) & system-prompt alignment
Confidence             : HIGH

System Prompt Status   : WITH SYSTEM PROMPT (Verified)

Dataset Balance        : 35% no-tool, 23% single-tool, 18% two-step, 18% multi-step, 6% error-recovery

Recommended Next Exp   : Train SFT v4 with 50% no-tool examples and explicit multi-step continuation targets.
============================================================
```

---

## 2. Comprehensive Metric Comparison Across Runs

| Metric | Baseline | SFT v2 (No Prompt) | SFT v2 (+ Prompt) | SFT v3 (+ Prompt) | Delta (v3 vs Base) | Delta (v3 vs v2+P) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall Accuracy** | **80.00%** | 16.67% | **70.00%** | **65.00%** | -15.00% | -5.00% |
| **Tool Task Accuracy** | 77.08% | 18.75% | 75.00% | 72.92% | -4.16% | -2.08% |
| **No-Tool Accuracy** | **91.67%** | 8.33% | **50.00%** | **33.33%** | -58.34% | -16.67% |
| **Tool Selection Accuracy** | 79.17% | 100.00% | 97.92% | **100.00%** | +20.83% | +2.08% |
| **Unnecessary Tool Use Rate** | **8.33%** | 100.00% | 66.67% | **83.33%** | +75.00% | +16.66% |
| **Single-Tool Accuracy** | 100.00% | 13.33% | 86.67% | **93.33%** | -6.67% | +6.66% |
| **Two-Step Accuracy** | 60.00% | 20.00% | 86.67% | **66.67%** | +6.67% | -20.00% |
| **Multi-Step Accuracy** | 55.56% | 0.00% | 33.33% | **22.22%** | -33.34% | -11.11% |
| **Error Recovery Accuracy** | 66.67% | 66.67% | 66.67% | **100.00%** | +33.33% | +33.33% |
| **Average Steps / Task** | **1.65** | 7.47 | 2.52 | **2.48** | +0.83 | -0.04 |
| **Timeout Failures** | **0** | 37 | **0** | **0** | +0 | +0 |

---

## 3. Failure Taxonomy Breakdown for SFT v3

- **Category A (Unnecessary Tool Use):** **6 tasks (28.6%)** — Model invoked `lookup` or `calculator` on direct knowledge queries (`notool_001`, `notool_002`, `notool_008`, `notool_009`, `notool_010`, `notool_011`).
- **Category B (Repeated Tool Calls):** **0 tasks (0.0%)** — Completely eliminated.
- **Category C (Timeout / Non-Termination):** **0 tasks (0.0%)** — Completely eliminated.
- **Category D (Wrong Tool Selection):** **2 tasks (9.5%)** — Selecting wrong tool or failing recovery.
- **Category E (Malformed Tool Call):** **0 tasks (0.0%)** — 100% valid JSON calls.
- **Category F (Premature Termination):** **7 tasks (33.3%)** — Terminating early in 3/4-step chains.
- **Category G (Incorrect Final Answer):** **1 task (4.8%)** — Arithmetic error in multi-step string.
- **Category H (Formatting Artifact):** **2 tasks (9.5%)** — Sentence wrapper artifacts.

---

## 4. Ranked Root Cause Analysis

1. **Tool Over-Activation Bias (HIGH Confidence):**
   SFT v3 data, while expanded to 500 examples, still contained 65% tool-requiring tasks. The model learned a strong prior that user prompts require tool calls, driving the Unnecessary Tool-Use Rate up to 83.33%.
2. **Premature Multi-Step Termination (HIGH Confidence):**
   On 3-step and 4-step tasks (`multi_step`), the model executes 1 or 2 tools and then outputs a `final_answer` prematurely before fulfilling all sub-goals.
3. **Catastrophic Forgetting of Parametric Knowledge (MEDIUM Confidence):**
   Fine-tuning degraded the model's willingness to rely on its pretrained weights for simple factual queries (e.g. capital of Japan).

---

## 5. Recommended Next Experiments

1. **Experiment 1 (SFT v4 Dataset Balancing):** Create a 500-trajectory dataset with **50% no-tool examples** and **50% tool-requiring examples** to strictly suppress tool over-activation.
2. **Experiment 2 (Multi-Step Continuation Targets):** Add explicit multi-step continuation target examples demonstrating 3-step and 4-step chains (`tool_call -> TOOL_RESULT -> tool_call -> TOOL_RESULT -> final`).
3. **Experiment 3 (Training from Base Model):** Ensure SFT runs start from fresh base `Qwen/Qwen3-1.7B` rather than fine-tuning an existing SFT checkpoint.
