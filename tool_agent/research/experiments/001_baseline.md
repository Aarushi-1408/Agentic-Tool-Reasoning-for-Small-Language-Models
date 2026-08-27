# Experiment Record: 001_baseline

- **Experiment ID:** `001_baseline`
- **Date:** 2026-08-26
- **Model:** `Qwen/Qwen3-1.7B`
- **Dataset:** None (Pre-trained base model zero-shot execution)
- **Number of Trajectories:** 0
- **Training Method:** None
- **System Prompt Condition:** Disabled
- **Inference Configuration:** `configs/default.yaml` (`max_new_tokens=512`, `temperature=0.0`, `max_steps=10`)
- **Evaluation Benchmark:** `tasks/eval_v1.json` (60 frozen tasks)
- **Output Directory:** `evaluation/baseline/`

---

## Evaluation Results

- **Overall Accuracy:** **80.00%** (48 / 60 tasks)
- **Tool Tasks Accuracy:** 77.08%
- **No-Tool Tasks Accuracy:** 91.67%
- **Single-Tool Accuracy:** 100.00%
- **Two-Step Accuracy:** 60.00%
- **Multi-Step Accuracy:** 55.56%
- **Tool Selection Accuracy:** 100.00%
- **Unnecessary Tool-Use Rate:** 8.33%
- **Average Steps / Task:** 1.65
- **Timeout Failures (`max_steps_exceeded`):** 0

---

## Observations & Findings

Base pre-trained `Qwen3-1.7B` possesses strong zero-shot tool reasoning capabilities out of the box. It answers direct-knowledge questions correctly without calling tools 91.67% of the time, and executes single-tool calls perfectly. However, accuracy drops on two-step (60%) and multi-step (55.56%) tasks due to raw completion format variations.

---

## Conclusion & Next Steps

Base zero-shot capability provides a solid 80% baseline performance. Proceed to SFT fine-tuning on synthetic agent trajectories to evaluate supervised alignment.
