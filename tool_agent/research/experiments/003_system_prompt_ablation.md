# Experiment Record: 003_system_prompt_ablation

- **Experiment ID:** `003_system_prompt_ablation`
- **Date:** 2026-08-27
- **Model:** `Bitsbrains/aarushi-qwen3-1.7b-tool-agent-sft-v2` (`Qwen/Qwen3-1.7B`)
- **Dataset:** `data/sft/train_v2_100.jsonl` (100 synthetic trajectories)
- **Number of Trajectories:** 100
- **Training Method:** Full fine-tuning
- **System Prompt Condition:** Enabled (Inference protocol system prompt injected)
- **Inference Configuration:** `scripts/evaluate.py --system-prompt`
- **Evaluation Benchmark:** `tasks/eval_v1.json` (60 frozen tasks)
- **Output Directory:** `evaluation/sft_v2_100_system_prompt/`

---

## Evaluation Results

- **Overall Accuracy:** **70.00%** (42 / 60 tasks) — **+53.33% gain over Run A**
- **Tool Tasks Accuracy:** **75.00%** (+56.25%)
- **No-Tool Tasks Accuracy:** **50.00%** (+41.67%)
- **Single-Tool Accuracy:** **86.67%** (+73.33%)
- **Two-Step Accuracy:** **86.67%** (+66.67%)
- **Multi-Step Accuracy:** **33.33%** (+33.33%)
- **Tool Selection Accuracy:** **83.33%** (+50.00%)
- **Unnecessary Tool-Use Rate:** **66.67%** (-33.33%)
- **Average Steps / Task:** **2.52** (-4.95 steps)
- **Timeout Failures (`max_steps_exceeded`):** **0** (-37 failures)

---

## Key Observations

1. **Complete Timeout Resolution:** `max_steps_exceeded` failures dropped from 37 tasks down to **0**.
2. **Loop Elimination:** Identical repeated tool calls dropped from 177 loops down to **1**.
3. **Execution Speed:** Average steps per task dropped from 7.47 to 2.52.

---

## Conclusion & Next Steps

System prompt alignment acts as an essential protocol enforcer, restoring state transition discipline. Perform detailed failure analysis of the remaining 18 failed tasks to inform SFT v3 dataset design.
