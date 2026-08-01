# DriftSQL-RL

Failure-driven Agentic Reinforcement Learning for SQL enterprise analytics under
schema and metric drift.

## Project boundary

DriftSQL-RL trains a small, privately deployable SQL agent to:

1. detect stale schema or business-metric context;
2. inspect version changes and retrieve the current definition;
3. clarify genuine ambiguity;
4. execute, validate, and repair SQL;
5. learn from execution-verified failures without regressing on clean tasks.

The training runtime is VERL with a DriftSQL-owned agent loop and executable
reward. BIRD-RL provides a useful SQL-RL reference recipe, while BIRD-Interact
supplies clean evaluation tasks. This repository owns the drift data factory,
environment adapter, tools, rewards, failure replay, and regression gate.

## Repository layout

```text
driftsql/               Core package
  contracts.py          Shared task, step, trajectory, and failure schemas
  drift/                Schema and metric drift generation
  environment/          Versioned database environments
  tools/                Agent tools and safety boundaries
  rewards/              Verifiable reward components
configs/                Data, reward, and training configuration
scripts/                Bootstrap and experiment entry points
tests/                  Unit and integration tests
frontend/               DriftSQL Studio React/TypeScript dashboard (P3)
third_party/            Pinned upstream frameworks (gitignored)
```

## Framework responsibilities

- **BIRD-RL**: reference SQL-specific SFT/RL recipe and baseline.
- **VERL**: GRPO, LoRA/FSDP, distributed workers, checkpoints, and vLLM
  rollouts.
- **BIRD-Interact**: user simulator, HKB/metadata, databases, and executable
  evaluation.
- **DriftSQL-RL**: versioned schema/metric mutations, drift-aware tools,
  failure mining, replay, regression gating, and serving.

Agent Lightning is intentionally not part of the main runtime. Keeping one
agent loop prevents train/serve skew.

## Local MVP

The first smoke test uses only Python's standard library. It demonstrates:

```text
valid v1 query
  -> rename a column in database v2
  -> stale query fails
  -> inspect the schema diff
  -> rewrite the identifier
  -> repaired query passes with the same result
```

Run it with Python 3.10 or newer:

```bash
python -m driftsql.smoke
python -m unittest discover -s tests -v
```

For the full development environment:

```bash
export DRIFTSQL_BASE_PYTHON=/data/yjz/anaconda_tmp/envs/lcpy311/bin/python
scripts/bootstrap_env.sh
.venv/bin/python scripts/check_environment.py
```

`DRIFTSQL_BASE_PYTHON` must already provide a CUDA-compatible PyTorch, vLLM,
Ray, Transformers, PEFT, and Datasets. The bootstrap script creates a local
venv with system packages visible, then installs only the pinned training
overlay and editable DriftSQL-RL/VERL sources. It never modifies the base
environment. On this machine, `.venv/bin/python` resolves to the `lcpy311`
Python 3.11 interpreter; the local overlay is necessary because `lcpy311`
retains Pydantic 1 for AppWorld while current VERL needs Pydantic 2.

The upstream framework revisions are recorded in `frameworks.lock`.

The default base model is pinned in `models.lock.json` and downloaded locally:

```bash
/data/yjz/anaconda_tmp/envs/lcpy311/bin/python scripts/bootstrap_model.py
export DRIFTSQL_MODEL_PATH="$PWD/models/Qwen2.5-Coder-7B-Instruct"
```

The resource-scaled GRPO smoke model is pinned separately and does not replace
the 7B comparison model:

```bash
.venv/bin/python scripts/bootstrap_model.py --model-key smoke_model_3b
```

## Data

The core pipeline uses fully open, execution-verifiable data:

- **BIRD23 Train Filtered**: 6,601 clean text-to-SQL tasks with Gold SQL over
  69 SQLite databases; this is the main schema/metric drift seed.
- **SIX-GYM-SQLite**: 5,000 SQL issue-repair tasks with buggy SQL, Gold SQL,
  executable test cases, and 13 template databases; this drives debugging SFT,
  agentic RL, and failure replay.
- **BIRD Mini-Dev SQLite**: 500 Gold-SQL tasks over 11 databases that never
  occur in the training sources; this is the clean and generated-drift test
  split.
- **Mini-Interact**: 300 public interactive tasks retained only for qualitative
  demos because its public release omits Gold SQL and test cases.

All Hugging Face revisions and external archive sizes are recorded in
`datasets.lock.json`; roles and paths are defined in `configs/data/sources.yaml`.

```bash
.venv/bin/python scripts/bootstrap_data.py
.venv/bin/python scripts/bootstrap_database_archives.py
.venv/bin/python scripts/audit_data.py \
  --dataset all --quick-check --summary-only
```

Splits are enforced by `db_id`, not random row. The current training set uses
73 unique databases and Mini-Dev uses 11 disjoint databases, preventing schema
leakage. DriftSQL-RL derives versioned schema/metric mutations and
failure-replay trajectories from these clean sources; generated data is not
downloaded benchmark data.

## Stage 1 clean baselines

The frozen 500-task Mini-Dev evaluation now has three comparable baselines
under one evaluator and one maximum tool/token budget:

| Baseline | EX | Executable rate |
|---|---:|---:|
| Qwen2.5-Coder-7B direct SQL | 49.2% | 87.4% |
| Same base model, fixed untrained ReAct | 23.2% | 41.2% |
| BIRD-Zeno-7B, fixed ReAct | **58.4%** | **91.6%** |

The untrained ReAct result verifies that adding tools alone is harmful, while
BIRD-Zeno shows a +9.2-point trained-agent gain over direct SQL at roughly 4.1x
the token cost. The paired comparison, per-difficulty results, termination
failures, exact revisions, and reproduction commands are recorded in
[`docs/experiments/stage1_baselines_20260727.md`](docs/experiments/stage1_baselines_20260727.md).
Machine-readable results are under `reports/stage1/full`.

## Stage 2 interactive environment

The VERL agent loop now has one persistent environment Session per trajectory
and five Mini-Interact actions: guarded `ask_user`, `get_schema`,
`get_knowledge_definition`, isolated `execute_sql`, and terminal
`submit_solution`. SQL actions run against a per-trajectory SQLite copy with a
read-only authorizer, deadline, savepoint rollback, and guaranteed cleanup.
Each request writes an atomic JSON trace containing prompts, messages, tool
arguments, observations, rewards, metrics, and latency.

The public 300-task/26-database Mini-Interact adapter, real-asset environment
smoke, four-rollout VERL lifecycle smoke, security checks, and full test suite
all pass. This completes the Stage 2 engineering environment; it does not
create an official Mini-Interact score because all public labels and test cases
are empty. See
[`docs/experiments/stage2_interactive_environment_20260727.md`](docs/experiments/stage2_interactive_environment_20260727.md).

```bash
.venv/bin/python scripts/prepare_interactive_eval.py
env TMPDIR="$PWD/data/tmp" DRIFTSQL_TMPDIR="$PWD/data/tmp" \
  .venv/bin/python scripts/smoke_interactive_environment.py
```

## Stage 3 two-stage SFT

The formal Reasoning SFT builder retains 6,596 of 6,601 BIRD23 training rows
after real read-only SQLite execution. The database-grouped split contains
5,143 training rows over 55 databases and 1,453 validation rows over 14
disjoint databases. Targets are deterministic AST-derived relational plans
plus Gold SQL, rather than unverified free-form reasoning.

Formal Qwen2.5-Coder-3B LoRA/FSDP training saved step 40 and step 80. On the
fixed 128-task/14-database gate, selected step 40 improves direct SQL EX from
`39.8%` to `53.1%` and executable rate from `74.2%` to `78.1%` (31 paired
gains, 14 losses, exact McNemar `p=0.0161`).

The second SFT stage contains 396 execution-verified six-action trajectories,
expanded into 1,908 train and 468 validation next-action examples with zero
database overlap. Its selected step-80 adapter uses structured tool history
and one parser-compatible JSON target per turn. On the unified 78-task
interactive evaluation it reaches `24/78 = 30.8%` task success and `43/78 =
55.1%` executable submissions, versus `1/78 = 1.3%` for Base. Invalid tool
output falls from 72 tasks to 2. See the full data, failure analysis, and
reproduction record in
[`docs/experiments/stage3_complete_20260727.md`](docs/experiments/stage3_complete_20260727.md).

```bash
env TMPDIR="$PWD/data/tmp" .venv/bin/python scripts/prepare_reasoning_sft.py
CUDA_VISIBLE_DEVICES=0,3 bash scripts/train_3b_reasoning_sft_formal.sh
env TMPDIR="$PWD/data/tmp" .venv/bin/python \
  scripts/prepare_five_tool_sft.py \
  --output-dir data/processed/five_tool_sft_native_v2
CUDA_VISIBLE_DEVICES=0,3 bash scripts/train_3b_five_tool_sft_semantic.sh
env TMPDIR="$PWD/data/tmp" .venv/bin/python \
  scripts/expand_five_tool_sft_next_actions.py \
  --input-dir data/processed/five_tool_sft_native_v2 \
  --output-dir data/processed/five_tool_sft_native_v4_json \
  --plain-json-targets
CUDA_VISIBLE_DEVICES=0,3 bash scripts/train_3b_five_tool_sft.sh
```

## Drift trajectory factory

The public trajectory survey is recorded in
`docs/trajectory_data_survey.md`. Existing corpora cover SQL self-correction,
tool-use conversations, or static schema perturbations, but none provides
replayable schema-version transitions with execution-verified recovery.

The factory builds deterministic, execution-verified episodes for four drift
operations:

- column rename;
- table rename;
- column replacement (copy into a new field, then drop the stale field);
- added columns that silently break the result contract of `SELECT *`.

Build the balanced multi-source corpus:

```bash
env TMPDIR="$PWD/data/tmp" .venv/bin/python scripts/build_schema_drift_data.py \
  --per-type 100 --max-scan 2000 \
  --output data/generated/schema_drift/train.jsonl
```

The legacy column-rename-only builder remains available for focused ablations:

```bash
env TMPDIR=/tmp .venv/bin/python scripts/build_drift_data.py \
  --limit 100 --seed 42
```

For every retained row the factory:

1. resolves a Gold-SQL column unambiguously against the real SQLite schema;
2. materializes a temporary changed database and applies the migration;
3. requires the stale SQL to fail;
4. rewrites and executes the repaired SQL;
5. requires its result fingerprint to equal the pre-drift result;
6. emits a five-step oracle tool trajectory and removes the temporary database.

The current balanced corpus contains 400 validated trajectories (100 per
operation) from 57 databases. The hard failures come from BIRD23 Train
Filtered; silent `SELECT *` contract drift comes from SIX-GYM because the
filtered BIRD SQL contains no usable single-table star queries. No synthetic
SQL is fabricated.
Only compact JSONL manifests are retained under `data/generated`; databases
are materialized per episode to avoid multiplying the 33 GB BIRD database
corpus. Mutation rollout is configured in
`configs/data/drift_factory.yaml`. EvoSchema is pinned as a reference taxonomy,
not used as the runtime or treated as verified trajectory data.

Dataset V2 scales this corpus to 1,102 execution-verified tasks: 802 atomic
drifts, 200 compound drifts, and 100 clean negative controls. Its
variable 2/4/5/6-action trajectories distinguish genuine clarification needs
from schema-only, knowledge-only, and direct-clean cases. Train/dev/test are
database-disjoint, while the existing 78-task evaluation remains frozen inside
the new test split. Composition, leakage policy, and exact build commands are
documented in [`docs/dataset_v2.md`](docs/dataset_v2.md).

## Tool SFT and GRPO pipeline

Convert the validated manifests into VERL-native datasets:

```bash
.venv/bin/python scripts/prepare_training_data.py
.venv/bin/python scripts/smoke_agentic_pipeline.py
```

The column-rename smoke split contains 74 train and 26 validation episodes.
The formal multi-drift split contains 322 train and 78 validation episodes
over 46 and 11 databases respectively, with zero database overlap. Each RL row carries tool initialization parameters,
schema-diff metadata, and the expected result fingerprint. The smoke pipeline
loads VERL's real tool registry, materializes v2, observes the stale query
failure, verifies the repair, and gives the successful trajectory reward
`1.10` versus `0.00` for the stale solution.

Run the short oracle SFT:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_sft_smoke.sh
```

The script now stores BF16 LoRA-only shards by default. To export a standard
PEFT adapter from an older HF checkpoint:

```bash
.venv/bin/python scripts/export_lora_adapter.py \
  --checkpoint-hf checkpoints/EXPERIMENT/global_step_N/huggingface \
  --output checkpoints/EXPERIMENT/global_step_N/lora_adapter \
  --base-model models/Qwen2.5-Coder-7B-Instruct
```

Warm-start the one-step GRPO integration run from SFT:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
LORA_ADAPTER_PATH="$PWD/checkpoints/EXPERIMENT/global_step_N/lora_adapter" \
bash scripts/train_grpo_smoke.sh
```

The GRPO script uses the local base model, async vLLM multi-turn rollout,
DriftSQL's terminal submit agent loop, live version/schema/SQL tools,
execution-based reward, and LoRA-only checkpoints. The first completed SFT
smoke and its resource measurements are recorded in
`docs/experiments/sft_smoke_20260726.md`.

The formal 3B Agentic GRPO tuning is now complete.  On the locked 78-task
execution set, the deployable GRPO policy plus conservative terminal controller
scores 28/78 versus Tool-SFT's 24/78, reduces turn limits from 23 to 16
(`30.4%`), and keeps unsafe tasks at zero.  The controller only submits SQL
that already passed the read-only `execute_sql` sandbox.  Pure-policy and fresh
generation results, including failed tuning runs and vLLM reproducibility
caveats, are separated in
`docs/experiments/stage4_complete_20260729.md`.

The formal two-GPU 7B run is reproducible with:

```bash
CUDA_VISIBLE_DEVICES=0,3 bash scripts/train_7b_schema_sft.sh
.venv/bin/python -m verl.model_merger merge --backend fsdp \
  --local_dir checkpoints/sft_schema_drift_7b/global_step_80 \
  --target_dir checkpoints/sft_schema_drift_7b/global_step_80/merged
CUDA_VISIBLE_DEVICES=0,3 bash scripts/train_7b_schema_grpo.sh
```

See `docs/experiments/schema_drift_7b_20260726.md` for measured results and
the distinction between training-rollout reward and held-out evaluation.

## Milestone sequence

1. **Complete:** create the project skeleton and pin dependencies.
2. **Complete:** reproduce resource-scaled BIRD-RL single- and multi-turn
   training.
3. **Public path complete:** run BIRD-Interact Mini's public environment;
   official SR is unavailable because the release omits Gold SQL and tests.
4. **Complete:** generate deterministic, execution-verified column renames.
5. **Complete:** connect `inspect_schema_diff`, SQL execution, and reward.
6. **Complete:** run the 3B SFT warm-start and one-step multi-turn GRPO smoke.
   See `docs/experiments/grpo_3b_smoke_20260726.md`.
7. **Complete:** formal 3B Reasoning SFT and five-tool next-action SFT with
   unified held-out execution evaluation.
8. **Complete:** formal 3B five-tool Agentic GRPO tuning, conservative terminal
   control, and the locked correctness/efficiency/safety promotion gate.
9. **Complete for Dataset V2:** 1,102 execution-verified atomic, compound, and
   clean-control trajectories; database-disjoint Train/Dev/Test; variable
   interaction profiles; and 5,060 next-action SFT examples. Metric-definition
   drift now has an execution-verified factory and uses the same isolated
   reward path; expanding it into a formal dataset split remains P5 follow-up.
10. Add failure mining and replay.
11. Add clean/drift/cost/safety regression gates.
12. **Product P0-P3 complete:** FastAPI contracts, persistent frozen SFT20
    vLLM backend, bounded Session queue, event streaming, isolated read-only
    SQLite sandboxes, durable trajectory storage, and the DriftSQL Studio Web
    Dashboard. See `docs/product_service_p0_p2.md` and
    `docs/product_service_p3.md`.
13. **Product P4 complete:** persisted run KPIs, drift-stratified metrics,
    daily trends, failure classification and trajectory replay, deployment
    provenance, and optional server-side W&B run discovery. See
    `docs/product_service_p4.md`.
14. **P5 completed as a negative promotion result:** human-reviewed replay,
    database-isolated Train/Tune/Gate, 10-step 7B GRPO, deterministic Tune
    selection, and the permanently sealed one-shot Gate all ran. GRPO did not
    beat SFT20, and the frozen SFT20 candidate failed the precommitted Gate;
    no Gate result is reused for tuning. See
    `docs/experiments/p5_reviewed_replay_20260801.md`.
