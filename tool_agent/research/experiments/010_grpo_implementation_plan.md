# Experiment Implementation Plan: 010_grpo_implementation_plan

- **Document ID:** `010_grpo_implementation_plan`
- **Date:** 2026-08-28
- **Target Model:** `Qwen/Qwen3-1.7B` (Full-Parameter GRPO-1 Phase)
- **Starting Checkpoint:** `outputs/qwen3-1.7b-tool-agent-sft-v4-500`
- **Frozen Benchmark:** `tasks/eval_v1.json` (60 tasks)
- **Status:** Implementation Architecture & Plan

---

## 1. Existing Architecture & Interface Inspection

The repository currently implements a modular, model-agnostic agent framework:

1. **`src/model.py` (`ModelInterface` & `HuggingFaceModel`)**:
   - `ModelInterface.generate(messages, tools)` defines the core abstraction.
   - `HuggingFaceModel` loads CausalLM models (e.g., `Qwen/Qwen3-1.7B`), formats prompts via tokenizer chat templates, and generates text completions.
2. **`src/harness.py` (`AgentHarness`)**:
   - Manages the multi-turn interaction loop up to `max_steps` (default 10).
   - Sends messages and tool schemas to the model.
   - Receives model output, parses actions via `ModelOutputParser`, executes tools via `ToolRegistry`, and appends `TOOL_RESULT` or `TOOL_ERROR` observations back into message history.
   - Returns a structured `Trajectory` object containing all turns and the final answer.
3. **`src/parser.py` (`ModelOutputParser`)**:
   - Parses raw model text into `Action` dataclasses (`tool_call`, `final`, `parse_error`).
   - Expects `{"type": "tool_call", "name": "...", "arguments": {...}}` or `{"type": "final", "content": "..."}`.
4. **`src/evaluator.py` (`Evaluator` & `normalize_answer`)**:
   - Assesses task completion via `normalize_answer(expected) in normalize_answer(actual)`.
   - Computes task success, tool selection accuracy, unnecessary tool use rate, multi-step completion, and valid tool call rate.

---

## 2. Proposed GRPO Integration Architecture

```text
                  data/sft/train_v4_500.jsonl
                              │
                              ▼
            data/rl/grpo1/train_prompts_v4_500.jsonl
                              │
                              ▼
                       RolloutGenerator
               (G=4 candidate trajectories / prompt)
                              │
                   ┌──────────┴──────────┐
                   ▼                     ▼
          AgentHarness Loop      ToolRegistry Execution
                   │                     │
                   └──────────┬──────────┘
                              ▼
                   Trajectory Collection
                              │
                              ▼
                   RewardCalculator (src/rl/reward.py)
                              │
                              ▼
                   Group Advantage Computation (A_i)
                              │
                              ▼
                   GRPOTrainer (src/rl/grpo_trainer.py)
                              │
                              ▼
                   Full-Parameter Model Update (θ)
                              │
                              ▼
                   outputs/grpo1/checkpoint-XX/
                              │
                              ▼
                 scripts/evaluate_grpo.py
               (tasks/eval_v1.json Benchmark)
```

---

## 3. Component Design & New Modules

1. **`src/rl/config.py` (`GRPOConfig`)**:
   - Dataclass defining hyperparameters: `group_size=4`, `learning_rate=5e-6`, `temperature=0.7`, `top_p=0.9`, `beta=0.04` (KL penalty), `max_prompt_length=512`, `max_completion_length=512`, `max_env_steps=10`.
2. **`src/rl/reward.py` (`RewardCalculator` & `calculate_reward`)**:
   - Trajectory-level reward calculation:
     $$R = R_{\text{correctness}} + R_{\text{policy}} + R_{\text{selection}} + R_{\text{completion}} - P_{\text{unnecessary}} - P_{\text{repetition}} - P_{\text{premature}} - P_{\text{parse}} - P_{\text{exec\_err}} - P_{\text{excess\_steps}}$$
   - Returns structured `RewardResult` with explicit breakdown and human-readable diagnostics.
3. **`src/rl/rollout.py` (`RolloutGenerator`)**:
   - Takes a model, `ToolRegistry`, and prompt item.
   - Samples $G=4$ distinct rollout trajectories per prompt using stochastic generation ($\tau=0.7, \text{top\_p}=0.9$).
   - Logs trajectory steps, token log-probabilities, actions, observations, and rewards.
4. **`src/rl/grpo_trainer.py` (`GRPOTrainer`)**:
   - Manages policy model $\pi_\theta$ and frozen reference model $\pi_{\text{ref}}$.
   - Calculates normalized group advantages:
     $$A_i = \frac{R(\tau_i) - \text{mean}(R)}{\text{std}(R) + 1\text{e}-8}$$
   - Computes per-token GRPO loss with PPO clipping ($\epsilon=0.2$) and KL penalty ($\beta=0.04$).
   - Performs full-parameter backward gradient updates with gradient checkpointing.
5. **`scripts/train_grpo.py`**:
   - CLI script running reward sanity checks, pilot execution (20 prompts), full training loop, logging to `evaluation/grpo1/training_metrics.jsonl`, and saving checkpoints to `outputs/grpo1/`.
6. **`scripts/evaluate_grpo.py`**:
   - Evaluates SFT V4 baseline vs GRPO-1 checkpoint on frozen `tasks/eval_v1.json` benchmark without changing evaluation logic.

---

## 4. Dependencies & Prerequisites

- PyTorch with CUDA support.
- Transformers & Accelerate.
- Full-Parameter training (PEFT/LoRA disabled).
- Gradient Checkpointing enabled to fit 1.7B GRPO backward pass in VRAM.

---

## 5. Key Assumptions & Safeguards

1. **Frozen Benchmark Protection:** `tasks/eval_v1.json` will NEVER be used for training or modified.
2. **Non-Trivial Group Advantage Guarantee:** Every training batch checks for non-zero reward variance within groups. If all rollouts in a group receive identical rewards, advantage is set to 0.
3. **No Model Weight Commits:** All checkpoints saved under `outputs/grpo1/` are excluded by `.gitignore`.
