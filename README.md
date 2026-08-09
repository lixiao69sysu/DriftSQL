<p align="center">
  <img src="docs/assets/driftsql-logo.svg" alt="DriftSQL" width="100%">
</p>

<p align="center">
  <a href="https://github.com/lixiao69sysu/DriftSQL/actions/workflows/ci.yml"><img src="https://github.com/lixiao69sysu/DriftSQL/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/Code%20License-MIT-yellow.svg" alt="MIT License"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&amp;logoColor=white" alt="Python 3.10+"></a>
  <a href="https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct"><img src="https://img.shields.io/badge/Base%20Model-Qwen2.5--Coder--7B-6C5CE7" alt="Qwen2.5-Coder-7B"></a>
  <img src="https://img.shields.io/badge/Agentic%20RL-SFT%20%2B%20GRPO-8A63D2" alt="SFT and GRPO">
</p>

<p align="center">
  <a href="https://huggingface.co/datasets/lxSYSU/DriftSQL-Recovery"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-DriftSQL--Recovery-FFD21E" alt="DriftSQL-Recovery on Hugging Face"></a>
  <img src="https://img.shields.io/badge/Tasks-3%2C152-4C8BF5" alt="3152 execution-verified tasks">
  <img src="https://img.shields.io/badge/Databases-98-00A6A6" alt="98 databases">
  <a href="https://creativecommons.org/licenses/by-sa/4.0/"><img src="https://img.shields.io/badge/Data%20License-CC%20BY--SA%204.0-2EA44F" alt="CC BY-SA 4.0"></a>
</p>

# DriftSQL-RL

Failure-driven Agentic Reinforcement Learning for execution-verified SQL
recovery under schema and business-metric drift.

DriftSQL-RL trains a privately deployable database agent to detect stale
context, inspect database changes, clarify genuine ambiguity, repair SQL in a
read-only sandbox, validate its result, and safely submit the answer. The
project combines on-policy failure recovery data, supervised fine-tuning, and
full-episode GRPO instead of treating SQL generation as a single-turn task.

## Highlights

- **Execution-verified Agent loop:** every SQL action runs against a real,
  isolated SQLite database rather than a string-matching simulator.
- **Failure-driven training:** real on-policy failures are mined into Recovery
  SFT, Hard Replay, and GRPO datasets.
- **Open training corpus:** DriftSQL-Recovery contains 3,152 execution-verified
  tasks over 98 databases, 2,400 rollout outcomes, and 1,066 complete failure
  trajectories.
- **Drift-aware interaction:** dynamic tools cover schema versions, schema
  diffs, knowledge retrieval, clarification, execution, and final submission.
- **Safety by construction:** read-only authorization, deadlines, rollback,
  result-contract validation, dynamic action masks, and safe auto-submit.
- **Usable product surface:** a bilingual terminal agent streams reasoning,
  tool calls, SQL results, reward decomposition, and durable trajectories.

## Results

All current model comparisons use the same deterministic seven-tool evaluator,
seven-turn budget, safety policy, and Train-derived **Tune432** split. Tune432
contains 72 clean tasks and 360 drift tasks across five drift categories. The
held-out **Fresh Blind320** split remains sealed and is not used for model
selection.

| Model | Task success | Drift recovery | Safe submission | Avg. tools | Unsafe / timeout |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-Coder-7B-Instruct | 59/432 (13.66%) | 55/360 (15.28%) | 59/432 | 3.23 | 0 / 0 |
| Recovery + Hard Replay SFT160 | 341/432 (78.94%) | 269/360 (74.72%) | 341/432 | 4.93 | 0 / 0 |
| SFT160 + GRPO Step25 | **348/432 (80.56%)** | **276/360 (76.67%)** | **348/432** | **4.88** | **0 / 0** |

The largest gain comes from execution-verified Recovery SFT: +65.28 percentage
points over the base model. The current GRPO checkpoint adds a smaller but
measurable +1.62 points over the strong SFT initialization. This result is not
presented as a solved RL problem: reducing paired regressions and improving
cross-seed stability remain active work.

## How it works

```text
Database instruction
        |
        v
detect drift -> inspect version/diff -> retrieve schema or knowledge
        |                    |
        |                    +-> clarify only when ambiguity is genuine
        v
execute SQL in an isolated read-only Session
        |
        v
validate result contract -> repair if needed -> safe submit
        |
        v
execution reward + persisted trajectory -> Failure Miner -> SFT / GRPO
```

The training runtime uses [VERL](https://github.com/volcengine/verl) for GRPO,
FSDP/LoRA, distributed workers, and vLLM rollouts. DriftSQL owns the agent loop,
versioned database environment, dynamic tool policy, executable Reward V3,
failure mining, replay, regression gates, and serving layer. BIRD-RL provides
the SQL-RL reference recipe, while BIRD-Interact supplies interactive task and
environment assets. Exact upstream revisions are pinned in `frameworks.lock`.

## Terminal agent

The product interface is a full-screen database TUI backed by a persistent
FastAPI/SSE service and vLLM. It supports searchable databases and model
checkpoints, streamed tool events, SQL result tables, Session replay, input
history, configurable budgets, and detailed reward inspection.

- Type `@` to search a database, table, or column path.
- Use `/models` to inspect or switch Base/SFT/GRPO checkpoints.
- Use `/recover <scenario_id>` for verified drift-recovery evaluation.
- Use `/trace` and `/reward` to inspect the current trajectory.
- Press `Ctrl+P` or `F1` for the complete command reference.

Chinese instructions are translated server-side by a pinned
`Qwen2.5-0.5B-Instruct` adapter before entering the English-trained Agent. SQL,
identifiers, numbers, code spans, and `@schema` references are protected during
translation; both the original and translated instructions are retained in
the Session trajectory.

Start the service with the GPU IDs appropriate for the local machine:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
DRIFTSQL_SERVICE_TENSOR_PARALLEL_SIZE=2 \
DRIFTSQL_SERVICE_PORT=8001 \
bash scripts/serve_service.sh
```

Open the TUI from another terminal:

```bash
bash scripts/run_cli.sh
```

The scrolling fallback remains available for logs, pipes, and small terminals:

```bash
bash scripts/run_cli.sh --classic
```

Install the optional translation dependencies and download the pinned adapter
once when Chinese input is required:

```bash
.venv/bin/pip install -e '.[service,translation]'
.venv/bin/python scripts/download_translation_model.py
```

## Quick start

The dependency-light smoke test demonstrates an execution-verified recovery:
a valid query is made stale by a column rename, the Agent inspects the diff,
rewrites the identifier, and verifies that the repaired result matches the
pre-drift result.

Create the Python 3.11 development environment, run the smoke test, and then
run the complete test suite:

```bash
export DRIFTSQL_BASE_PYTHON=python3.11
scripts/bootstrap_env.sh
.venv/bin/python scripts/check_environment.py
.venv/bin/python -m driftsql.smoke
.venv/bin/pytest -q
```

`DRIFTSQL_BASE_PYTHON` must provide a CUDA-compatible PyTorch, vLLM, Ray,
Transformers, PEFT, and Datasets installation. The bootstrap script creates a
project-local overlay and does not modify the base environment. Use an
absolute interpreter path when `python3.11` is not the intended CUDA
environment.

Download the pinned base model:

```bash
.venv/bin/python scripts/bootstrap_model.py
export DRIFTSQL_MODEL_PATH="$PWD/models/Qwen2.5-Coder-7B-Instruct"
```

Package versions, model revisions, framework commits, and dataset revisions
are recorded in `requirements/training-overlay.txt`, `models.lock.json`,
`frameworks.lock`, and `datasets.lock.json`.

## Data

The pipeline uses open, execution-verifiable sources:

| Source | Role |
|---|---|
| BIRD23 Train Filtered | 6,601 clean text-to-SQL seeds over 69 SQLite databases |
| SIX-GYM-SQLite | 5,000 SQL repair tasks with Gold SQL, tests, and 13 template databases |
| BIRD-Critic-SQLite | 500 tasks and database assets used for isolated Tune/Blind generation |
| Mini-Interact | 300 public interactive tasks and database assets |
| BIRD Mini-Dev SQLite | 500 clean tasks over 11 databases, retained for the Stage 1 baseline |

DriftSQL materializes deterministic schema and metric changes from these clean
sources. The resulting
[DriftSQL-Recovery](https://huggingface.co/datasets/lxSYSU/DriftSQL-Recovery)
dataset uses one unified Scale-up V1 protocol:

| Dataset scale | Count |
|---|---:|
| Independent execution-verified tasks | 3,152 |
| Database-isolated SQLite environments | 98 |
| Multi-seed on-policy rollout outcomes | 2,400 |
| Complete unique failure trajectories | 1,066 |
| Seven-tool next-action examples | 26,539 |
| Recovery SFT examples | 1,714 |
| Hard Replay examples | 1,600 |
| Full-episode GRPO records | 3,632 |

The dataset covers clean controls, column addition/rename/replacement, table
rename, compound recovery, and schema-only, knowledge-only, must-ask, and
direct-clean interaction profiles. All splits are assigned by `db_id`, with
zero database overlap. Final blind labels remain sealed and are not exposed in
the public files.

The Hugging Face release is a 13 MB, CC BY-SA 4.0 collection of viewer-ready
Parquet configurations for tasks, canonical trajectories, rollout outcomes,
failures, SFT, replay, and GRPO.

Load any configuration directly:

```python
from datasets import load_dataset

tasks = load_dataset("lxSYSU/DriftSQL-Recovery", "tasks")
failures = load_dataset("lxSYSU/DriftSQL-Recovery", "failure_trajectories")
grpo = load_dataset("lxSYSU/DriftSQL-Recovery", "grpo")
```

Raw SQLite files are not redistributed; the pinned bootstrap scripts retrieve
upstream assets, and databases are materialized per episode instead of
duplicating the full source corpus.

Bootstrap and audit the public data:

```bash
.venv/bin/python scripts/bootstrap_data.py
.venv/bin/python scripts/bootstrap_database_archives.py
.venv/bin/python scripts/audit_data.py \
  --dataset all --quick-check --summary-only
```

See [Dataset V2](docs/dataset_v2.md) for the earlier factory design and leakage
policy, the [P6 retrospective](docs/experiments/p6_agentic_rl_iteration_retrospective_20260808.md)
for the current Scale-up pipeline, and the
[trajectory data survey](docs/trajectory_data_survey.md) for the motivation
behind the executable trajectory factory.

## Training and evaluation

The maintained pipeline proceeds in three stages:

1. sample the current policy in the Train environments and mine real failure
   states;
2. build Recovery SFT and Hard Replay examples whose target actions and SQL are
   linked to execution-verified canonical trajectories;
3. initialize full-episode GRPO from the strong SFT checkpoint, using complete
   task coverage, episode-level advantage, Reward V3, and dynamic action masks.

Train the current SFT160 and corrected-observation GRPO stages on four GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/train_7b_p6_scaleup_sft.sh

ARM=B SEED=20260810 \
OUTPUT_DIR="$PWD/checkpoints/p6_contract_observation_grpo_arm_c_7b" \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/train_7b_p6_first_action_grpo.sh
```

Export a portable PEFT LoRA adapter:

```bash
.venv/bin/python scripts/export_lora_adapter.py \
  --checkpoint-hf checkpoints/EXPERIMENT/global_step_N/huggingface \
  --output checkpoints/EXPERIMENT/global_step_N/lora_adapter \
  --base-model models/Qwen2.5-Coder-7B-Instruct
```

The active builders, launchers, evaluation scripts, and required execution
order are indexed in [`scripts/README.md`](scripts/README.md). Fresh Blind320
must remain unread until a candidate has been selected on Tune432.

## Safety and observability

Each Session receives an independent SQLite copy. SQL execution uses a
read-only authorizer, timeout, savepoint rollback, and guaranteed cleanup.
Every run persists its model and adapter hashes, database version, inference
parameters, tool events, SQL observations, termination reason, and decomposed
reward. The service exposes trajectory replay, failure categories, operational
metrics, and optional W&B run discovery.

Deployment and CI controls are documented in
[`docs/deployment_security_ci.md`](docs/deployment_security_ci.md); service
contracts and sandbox design are documented in
[`docs/product_service_p0_p2.md`](docs/product_service_p0_p2.md).

## Repository layout

```text
driftsql/               Core package, Agent loop, tools, rewards, service, CLI
configs/                Data, reward, and training configuration
scripts/                Maintained build, training, evaluation, and serving entry points
tests/                  Unit, integration, security, and regression tests
docs/                   Dataset, experiment, deployment, and product documentation
third_party/            Pinned upstream frameworks, populated locally and gitignored
```

## Documentation

- [DriftSQL-Recovery dataset](https://huggingface.co/datasets/lxSYSU/DriftSQL-Recovery)
- [Agentic RL iteration retrospective](docs/experiments/p6_agentic_rl_iteration_retrospective_20260808.md)
- [Stage 1 baseline report](docs/experiments/stage1_baselines_20260727.md)
- [Dataset V2](docs/dataset_v2.md)
- [Trajectory dataset survey](docs/trajectory_data_survey.md)
- [Product service and sandbox](docs/product_service_p0_p2.md)
- [Deployment, security, and CI](docs/deployment_security_ci.md)

## Acknowledgements

DriftSQL-RL builds on the work of the following open-source projects and
research communities:

- [BIRD-RL](https://github.com/bird-bench/BIRD-RL) and
  [BIRD-Interact](https://github.com/bird-bench/BIRD-Interact) for SQL Agent
  training and interactive evaluation foundations;
- [VERL](https://github.com/volcengine/verl) for the distributed Agentic RL
  training runtime;
- [Qwen2.5-Coder](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) for
  the base language model;
- [vLLM](https://github.com/vllm-project/vllm) for high-throughput inference;
- [SIX-GYM-SQLite](https://huggingface.co/datasets/birdsql/six-gym-sqlite)
  and the BIRD benchmark community for execution-verifiable SQL data;
- [EvoSchema](https://github.com/zhangtianshu/EvoSchema) for its schema-evolution
  taxonomy, used as a reference rather than the runtime environment;
- [FastAPI](https://github.com/fastapi/fastapi),
  [Textual](https://github.com/Textualize/textual), and
  [Weights & Biases](https://wandb.ai/) for the service, terminal UI, and
  experiment-observability ecosystem.

Their licenses and terms remain applicable to their respective code, models,
and datasets. DriftSQL-specific data generation, environment logic, rewards,
failure replay, safety controller, and product integration are implemented in
this repository.
