# Dataset Versioning Registry

Registry of all historical and active task definitions, teacher trajectory files, and SFT datasets in the repository.

---

## Evaluation Benchmark Datasets (Frozen)

### `tasks/eval_v1.json`
- **Description:** Static, frozen 60-task evaluation benchmark dataset.
- **Status:** **FROZEN & IMMUTABLE**. Never modify, train on, or add to training datasets.
- **Category Breakdown:**
  - `no_tool`: 12 tasks (20%)
  - `single_tool`: 15 tasks (25%)
  - `two_step`: 15 tasks (25%)
  - `multi_step`: 9 tasks (15%)
  - `tool_selection`: 6 tasks (10%)
  - `error_recovery`: 3 tasks (5%)

---

## Training Task & Trajectory Datasets

### Version 1 (`v1_25`)
- **Tasks File:** `data/tasks/train_tasks.jsonl`
- **Raw Trajectories:** `data/raw/teacher_trajectories_v1_25.jsonl`
- **SFT Dataset:** `data/sft/train.jsonl`
- **Count:** 25 trajectories (initial pilot data).

### Version 2 (`v2_100`)
- **Tasks File:** `data/tasks/train_tasks_v2_100.jsonl`
- **Raw Trajectories:** `data/raw/teacher_trajectories_v2_100.jsonl`
- **SFT Dataset:** `data/sft/train_v2_100.jsonl`
- **Stats File:** `data/sft/dataset_stats_v2_100.json`
- **Count:** 100 trajectories.
- **Category Composition:** 20% no-tool, 25% single-tool, 25% two-step, 20% multi-step, 10% error-recovery.

### Version 3 (`v3_500`)
- **Tasks File:** `data/tasks/train_tasks_v3_500.jsonl`
- **Raw Trajectories:** `data/raw/teacher_trajectories_v3_500.jsonl`
- **SFT Dataset:** `data/sft/train_v3_500.jsonl`
- **Stats File:** `data/sft/dataset_stats_v3.json`
- **Count:** 500 trajectories.
- **Target Category Composition:**
  - `no_tool`: 35% (175 trajectories) — Rebalances tool over-activation bias.
  - `single_tool`: 23% (115 trajectories) — Covers all active tools.
  - `two_step`: 18% (90 trajectories) — Dual tool chains.
  - `multi_step`: 18% (90 trajectories) — 3-step and 4-step long horizon chains.
  - `error_recovery` / `tool_selection`: 6% (30 trajectories) — Fallback recovery & selection.
