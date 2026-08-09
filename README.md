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
driftsql/cli/            Full-screen database TUI, classic CLI, and SSE client
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

## Interactive terminal workbench

The product surface is a Hermes-style full-screen terminal workbench backed by
the existing FastAPI/SSE service. It has searchable model/database/session
pickers, a persistent Agent transcript, live tool cards, and always-visible
budget, Reward, and SQL-safety state. It keeps the trained DriftSQL Agent loop,
result-contract controller, isolated read-only SQLite sessions, replay, and W&B
observability instead of delegating database decisions to a generic outer
agent.

Start the persistent vLLM service on GPUs 0 and 2:

```bash
CUDA_VISIBLE_DEVICES=0,2 \
DRIFTSQL_SERVICE_TENSOR_PARALLEL_SIZE=2 \
DRIFTSQL_SERVICE_PORT=8001 \
bash scripts/serve_service.sh
```

Open the TUI from another terminal:

```bash
bash scripts/run_cli.sh
```

The TUI uses the terminal alternate screen and supports mouse interaction.
Useful shortcuts are `Ctrl+M` for models, `Ctrl+D` for databases, `Ctrl+K` for
sessions, `Ctrl+P`/`F1` for the command panel, and `Ctrl+C` to cancel the active
run. Use `Up`/`Down` to move through persistent input history and return to the
current unsent draft. Type `@` anywhere in the composer to search safe logical database paths at
database, table, or column granularity; `Tab` or `Enter` inserts the selected
path and switches to its database. The previous scrolling interface remains available for logs, pipes, or
very small terminals:

```bash
bash scripts/run_cli.sh --classic
```

Plain text is treated as a free-form instruction for the selected database.
Free queries are execution- and safety-verified, but are explicitly marked as
having no hidden semantic oracle. Verified drift-recovery evaluation remains
available through `/recover <scenario_id>`.

Chinese instructions are translated server-side by the pinned
`Qwen2.5-0.5B-Instruct` language adapter before they enter the English-trained
Agent. The adapter runs lazily on CPU and preserves `@schema` paths, SQL, code
spans, identifiers, numbers, and DriftSQL domain terms through protected
placeholders. The original Chinese and generated English are both retained in
the Session trajectory; the CLI displays the English Agent input before the
first model turn. Install and download it once with:

```bash
.venv/bin/pip install -e '.[service,translation]'
.venv/bin/python scripts/download_translation_model.py
```

Set `DRIFTSQL_SERVICE_TRANSLATION_ENABLED=false` only when an English-only
deployment is desired. If translation is enabled but the pinned local model is
missing or cannot preserve protected tokens, the API rejects the request
instead of silently passing Chinese into the Agent.

```text
@database/table/column      search and insert a logical Schema reference
/db                         list databases
/db <db_id>                 select a database
/models                     list Base/SFT/GRPO checkpoints
/models info <model_id>     inspect provenance and Tune metrics
/models use <model_id>      activate a registered model for new Sessions
/recover <scenario_id>      run a result-contract recovery task
/budget key=value           configure turn/tool/token/timeout budgets
/sessions                   list durable Session history
/trace [session_id]         replay model/tool events
/reward [session_id]        inspect Reward decomposition
/experiments                compare frozen evaluation results
/ops                        inspect persisted operational metrics
/failures [type]            inspect classified failures
/wandb [run_id]             inspect W&B runs and metric series
/replay                     list or review replay candidates
/help                       open the searchable command reference
```

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
export DRIFTSQL_BASE_PYTHON=python3.11
scripts/bootstrap_env.sh
.venv/bin/python scripts/check_environment.py
```

`DRIFTSQL_BASE_PYTHON` must already provide a CUDA-compatible PyTorch, vLLM,
Ray, Transformers, PEFT, and Datasets. The bootstrap script creates a local
venv with system packages visible, then installs only the pinned training
overlay and editable DriftSQL-RL/VERL sources. It never modifies the base
environment. Set it to an absolute interpreter path when `python3.11` is not
the desired CUDA environment. The local overlay keeps DriftSQL's pinned Python
packages separate from other projects that may require incompatible versions.

The upstream framework revisions are recorded in `frameworks.lock`.

The default base model is pinned in `models.lock.json` and downloaded locally:

```bash
.venv/bin/python scripts/bootstrap_model.py
export DRIFTSQL_MODEL_PATH="$PWD/models/Qwen2.5-Coder-7B-Instruct"
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
are empty. The retained environment conclusions are summarized in
[`docs/experiments/p6_agentic_rl_iteration_retrospective_20260808.md`](docs/experiments/p6_agentic_rl_iteration_retrospective_20260808.md).

```bash
.venv/bin/python scripts/prepare_interactive_eval.py
env TMPDIR="$PWD/data/tmp" DRIFTSQL_TMPDIR="$PWD/data/tmp" \
  .venv/bin/python scripts/smoke_interactive_environment.py
```

## Historical SFT exploration

The early 3B Reasoning-SFT and Tool-SFT experiments established that
execution-verified targets and structured tool history were necessary, but
their checkpoints and one-off launchers have been superseded by the current
7B Recovery SFT pipeline. Their measured conclusions and failure analysis are
preserved in
[`docs/experiments/p6_agentic_rl_iteration_retrospective_20260808.md`](docs/experiments/p6_agentic_rl_iteration_retrospective_20260808.md),
while the active scripts directory contains only the current training path.

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

## Current 7B Agentic SFT and GRPO pipeline

The maintained path builds the Scale-up protocol, mines real on-policy
failures, creates Recovery SFT and Hard Replay data, trains SFT160, and then
optimizes the corrected-observation policy with episode-level GRPO. The
canonical entry points and their dependencies are indexed in
[`scripts/README.md`](scripts/README.md).

The generic SFT and GRPO launchers store BF16 LoRA-only shards. Export a
portable PEFT adapter with:

```bash
.venv/bin/python scripts/export_lora_adapter.py \
  --checkpoint-hf checkpoints/EXPERIMENT/global_step_N/huggingface \
  --output checkpoints/EXPERIMENT/global_step_N/lora_adapter \
  --base-model models/Qwen2.5-Coder-7B-Instruct
```

Train the maintained SFT160 and corrected-observation GRPO stages on four
GPUs with:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_7b_p6_scaleup_sft.sh
ARM=B SEED=20260810 \
OUTPUT_DIR="$PWD/checkpoints/p6_contract_observation_grpo_arm_c_7b" \
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_7b_p6_first_action_grpo.sh
```

Both stages use the local Qwen2.5-Coder-7B base model, VERL multi-turn rollout,
the seven-tool DriftSQL environment, execution-based Reward V3, dynamic tool
masks and LoRA-only checkpoints. See the experiment retrospective for the
measured Base/SFT/GRPO comparison and the distinction between training reward
and held-out Tune432 behavior.

## Milestone sequence

1. **Complete:** create the project skeleton and pin dependencies.
2. **Historical framework-validation milestone:** resource-scaled BIRD-RL
   single- and multi-turn training was reproduced during bring-up. Its
   one-off Smoke launchers were removed after the maintained seven-tool P6
   training path replaced them; the pinned upstream framework and DriftSQL
   integration remain.
3. **Public path complete:** run BIRD-Interact Mini's public environment;
   official SR is unavailable because the release omits Gold SQL and tests.
4. **Complete:** generate deterministic, execution-verified column renames.
5. **Complete:** connect `inspect_schema_diff`, SQL execution, and reward.
6. **Historical:** the 3B SFT/GRPO smoke established the initial integration;
   its conclusions remain in the retrospective, while its superseded local
   launchers and weights have been removed.
7. **Complete:** formal 3B Reasoning SFT and five-tool next-action SFT with
   unified held-out execution evaluation.
8. **Complete:** formal 3B five-tool Agentic GRPO tuning, conservative terminal
   control, and the locked correctness/efficiency/safety promotion gate.
9. **Complete for Dataset V2:** 1,102 execution-verified atomic, compound, and
   clean-control trajectories; database-disjoint Train/Dev/Test; variable
   interaction profiles; and 5,060 next-action SFT examples. Metric-definition
   drift now has an execution-verified factory and uses the same isolated
   reward path; expanding it into a formal dataset split remains P5 follow-up.
10. **Complete:** mine real on-policy failures and build Recovery SFT plus Hard
    Replay data.
11. **Complete:** enforce clean/drift/cost/safety regression gates on Tune432.
12. **Product P0-P3 complete:** FastAPI contracts, persistent vLLM backend,
    bounded Session queue, event streaming, isolated read-only SQLite
    sandboxes, durable trajectory storage, and the DriftSQL interactive CLI.
    See `docs/product_service_p0_p2.md`.
13. **Product P4 complete:** persisted run KPIs, drift-stratified metrics,
    daily trends, failure classification and trajectory replay, deployment
    provenance, and optional server-side W&B run discovery.
14. **Historical P5 negative result:** human-reviewed replay,
    database-isolated Train/Tune/Gate, 10-step 7B GRPO, deterministic Tune
    selection, and the permanently sealed one-shot Gate all ran. GRPO did not
    beat SFT20, and the frozen SFT20 candidate failed the precommitted Gate;
    no Gate result is reused for tuning. Its one-off scripts and reports have
    been removed after consolidation. See
    `docs/experiments/p6_agentic_rl_iteration_retrospective_20260808.md`.
