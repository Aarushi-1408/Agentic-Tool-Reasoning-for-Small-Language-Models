# Research Experiment Log

Chronological record of research iterations, hypotheses, and findings.

---

## [2026-08-26] Experiment `001_baseline`
- **Goal:** Establish baseline zero-shot tool reasoning performance on frozen 60-task benchmark (`tasks/eval_v1.json`).
- **Model:** Pre-trained `Qwen/Qwen3-1.7B` without fine-tuning.
- **Results:** Overall Accuracy = **80.00%**, No-tool = 91.67%, Single-tool = 100.00%, Multi-step = 55.56%.
- **Finding:** Base model has strong zero-shot tool selection, but lacks standardized completion format.

---

## [2026-08-26] Experiment `002_sft_v2_100`
- **Goal:** Fine-tune `Qwen/Qwen3-1.7B` on 100 synthetic tool trajectories (`train_v2_100.jsonl`).
- **Inference Condition:** Executed without system prompt protocol.
- **Results:** Overall Accuracy = **16.67%**, Avg Steps = 7.47, Timeout Failures = 37.
- **Finding:** Fine-tuning without system prompt causes tool over-activation (100% unnecessary tool rate) and infinite loop collapse.

---

## [2026-08-27] Experiment `003_system_prompt_ablation`
- **Goal:** Test hypothesis H5: Enforcing inference system prompt protocol restores state transition discipline.
- **Results:** Overall Accuracy = **70.00%** (+53.33%), Timeout Failures = **0** (down from 37), Avg Steps = 2.52.
- **Finding:** System prompt acts as essential protocol enforcer, eliminating 100% of timeouts.

---

## [2026-08-27] Experiment `004_failure_analysis`
- **Goal:** Classify remaining 18 failures in Run B.
- **Findings:**
  - 6 failures (A): Unnecessary tool use on direct answer tasks due to 80% tool bias in SFT v2.
  - 7 failures (F): Premature workflow termination on multi-step tasks due to under-representation of 3-step and 4-step chains.
  - 2 failures (H): String formatting exact-match artifacts.

---

## [2026-08-27] Experiment `005_sft_v3` (Planned)
- **Goal:** Test hypotheses H1, H2, H3, H4 with balanced 500-trajectory dataset (`train_v3_500.jsonl`).
- **Target Distribution:** 35% no-tool, 23% single-tool, 18% two-step, 18% multi-step, 6% error recovery & tool selection.
