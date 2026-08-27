# Experiment Results Comparison Table

Persistent comparison table tracking performance metrics across all research iterations on the frozen 60-task evaluation benchmark (`tasks/eval_v1.json`).

| Experiment ID | Model | Dataset | System Prompt | Accuracy | No-Tool Acc | Single-Tool Acc | Two-Step Acc | Multi-Step Acc | Tool Sel Acc | Unnecessary Tool Rate | Avg Steps | Timeout Failures |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `001_baseline` | Qwen3-1.7B | Pre-trained (Zero-shot) | Disabled | 80.00% | 91.67% | 100.00% | 60.00% | 55.56% | 100.00% | 8.33% | 1.65 | 0 |
| `002_sft_v2_100` | Qwen3-1.7B | `train_v2_100.jsonl` (100 ex) | Disabled | 16.67% | 8.33% | 13.33% | 20.00% | 0.00% | 33.33% | 100.00% | 7.47 | 37 |
| `003_system_prompt_ablation` | Qwen3-1.7B | `train_v2_100.jsonl` (100 ex) | Enabled | 70.00% | 50.00% | 86.67% | 86.67% | 33.33% | 83.33% | 66.67% | 2.52 | 0 |
| `005_sft_v3` | Qwen3-1.7B | `train_v3_500.jsonl` (500 ex) | Enabled | *Pending SFT v3 run* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |

---

## Metric Definitions

- **Accuracy:** Overall task success rate across all 60 benchmark tasks.
- **No-Tool Acc:** Accuracy on 12 direct-answer tasks (`notool_001` - `notool_012`).
- **Single-Tool Acc:** Accuracy on 15 single tool call tasks (`single_001` - `single_015`).
- **Two-Step Acc:** Accuracy on 15 two-step tool chain tasks (`twostep_001` - `twostep_015`).
- **Multi-Step Acc:** Accuracy on 9 multi-step tool chain tasks (`multistep_001` - `multistep_009`).
- **Tool Sel Acc:** Accuracy on 6 tool selection tasks (`select_001` - `select_006`).
- **Unnecessary Tool Rate:** Percentage of `no_tool` tasks where tools were unnecessarily invoked.
- **Avg Steps:** Mean steps per task trajectory.
- **Timeout Failures:** Count of tasks failing due to `max_steps_exceeded`.
