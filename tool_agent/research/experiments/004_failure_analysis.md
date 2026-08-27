# Experiment Record: 004_failure_analysis

- **Experiment ID:** `004_failure_analysis`
- **Date:** 2026-08-27
- **Target Run:** Run B (SFT v2 100 with System Prompt)
- **Evaluation Benchmark:** `tasks/eval_v1.json` (60 frozen tasks)
- **Total Tasks:** 60 | **Passed:** 42 (70.0%) | **Failed:** 18 (30.0%)

---

## Failure Taxonomy Breakdown (A through H)

- **A. Unnecessary Tool Use:** 6 tasks (33.3%) — Invoking tools on direct knowledge queries (`notool_001`, `notool_002`, `notool_008`, `notool_009`, `notool_010`, `notool_011`).
- **B. Repeated Tool Call:** 0 tasks (0.0%) — Completely resolved.
- **C. Timeout / Non-Termination:** 0 tasks (0.0%) — Completely resolved.
- **D. Wrong Tool Selection:** 2 tasks (11.1%) — Selecting `echo` instead of `string_manipulator` (`select_001`) or failing error recovery fallback (`errrec_003`).
- **E. Malformed Tool Call:** 0 tasks (0.0%).
- **F. Misunderstood Tool Result / Premature Termination:** 7 tasks (38.9%) — Terminating early in 3-step/4-step chains (`twostep_004`, `twostep_005`, `multistep_001`, `multistep_003`, `multistep_004`, `multistep_005`, `multistep_008`).
- **G. Incorrect Final Answer:** 1 task (5.6%) — Multi-step arithmetic calculation mistake (`multistep_007`).
- **H. Exact-Match Formatting Artifact:** 2 tasks (11.1%) — Explanatory string wrapping (`single_003`, `single_008`).

---

## Primary Diagnostic Insights

1. **`no_tool` Accuracy (50.0%):** Caused by tool over-activation bias in SFT v2 dataset (80% tool-use ratio).
2. **`multi_step` Accuracy (33.33%):** Caused by lack of long-horizon 3/4-step training trajectories, leading to premature chain termination.

---

## Recommendations for SFT v3

1. Rebalance `no_tool` to 35% of training trajectories.
2. Rebalance long `multi_step` chains (3–5 tools) to 18–35% of training trajectories.
3. Clean target final answer formatting.
