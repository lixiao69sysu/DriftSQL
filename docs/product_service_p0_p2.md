# Product service P0-P2

## Delivered boundary

P0-P2 provides the backend contract and runtime needed by the later Web
Dashboard. It deliberately does not implement the dashboard itself.

- P0: typed FastAPI contracts for Session, Event and Trajectory, plus catalog,
  execution and cancellation endpoints. `/openapi.json` describes every API,
  including the `text/event-stream` response.
- P1: one process-lifetime vLLM engine, the frozen Stage-8 SFT20 LoRA loaded and
  pinned once during application lifespan, a two-session semaphore, the same
  DriftSQL VERL tools and state policy used in evaluation, and a replayable SSE
  wrapper around the turn-based agent loop.
- P2: one materialized SQLite copy per Session, URI read-only mode, SQLite
  authorizer, query deadline, savepoint rollback, durable metadata in SQLite,
  and complete append-only event JSON.

The public scenario catalog is the 55-task Tune split at
`data/processed/stage8_fresh_sft/tune_agent_eval.jsonl`. Gold SQL and expected
fingerprints remain server-internal; the API exposes only the problem,
previously valid SQL and audited drift description.

## Runtime flow

```text
POST /api/sessions
  -> load verified Tune scenario
  -> materialize a private v2 SQLite copy
  -> persist model/adapter/db metadata

POST /api/sessions/{id}/run
  -> bounded queue (two Sessions by default)
  -> persistent vLLM + pinned SFT20 adapter
  -> dynamic tool schemas from state_policy
  -> real DriftSQL tool execution
  -> append model/tool/budget/reward events
  -> execution-verified Agentic reward
  -> terminal status + sandbox cleanup
```

The database copy is created before a run is accepted, so a Session can never
fall back to the source database. A service restart marks unfinished persisted
Sessions as failed with `service_restart`; they are replayable but are not
silently resumed against a new model process.

## Start the production service

Use the project-local `lcpy311` overlay and two visible GPUs. Tensor parallel
size must match the number of visible GPUs.

```bash
cd /path/to/driftsql-rl
export CUDA_VISIBLE_DEVICES=0,2
export DRIFTSQL_SERVICE_TENSOR_PARALLEL_SIZE=2
export DRIFTSQL_SERVICE_MAX_CONCURRENT_SESSIONS=2
bash scripts/serve_service.sh
```

The default paths are:

- base model: `models/Qwen2.5-Coder-7B-Instruct`;
- adapter: `checkpoints/stage8_fresh_sft_7b/global_step_20/merged/lora_adapter`;
- frozen manifest: `reports/stage8/final_candidate/frozen_candidate.json`;
- metadata/event store: `data/service/driftsql_service.sqlite`;
- per-Session databases: `data/tmp/service` (removed at terminal state).

Startup verifies every adapter file hash in the frozen manifest before engine
construction. Tensor-parallel workers use the CUDA-safe `spawn` start method,
and async scheduling/prefix caching remain disabled to match the frozen
candidate. `/health` changes to ready only after the adapter has been added and
pinned.

Tool observations use the frozen evaluator's serving limits by default:
`execute_sql` returns at most 5 rows, Schema retrieval returns at most 3,500
characters, and knowledge retrieval returns at most one definition. These
limits prevent observation bloat and train/serve skew; they remain configurable
through the corresponding `DRIFTSQL_SERVICE_*` environment variables.

## API smoke

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/api/scenarios

curl -s -X POST http://127.0.0.1:8000/api/sessions \
  -H 'content-type: application/json' \
  -d '{"scenario_id":"drift_coladd_336a8e6d4010d75e"}'

curl -s -X POST http://127.0.0.1:8000/api/sessions/SESSION_ID/run \
  -H 'content-type: application/json' -d '{}'

curl -N http://127.0.0.1:8000/api/sessions/SESSION_ID/events
curl -s http://127.0.0.1:8000/api/sessions/SESSION_ID/trajectory
```

Interactive documentation is available at `/docs`; the raw contract is at
`/openapi.json`.

## Verification

The service tests use a deterministic model double but execute the real
scenario catalog, schema materializer, state policy, tools, SQLite sandbox and
reward. This lets CI verify safety without reserving a GPU.

```bash
cd /path/to/driftsql-rl
TMPDIR=/tmp .venv/bin/python -m pytest tests/service -q
TMPDIR=/tmp .venv/bin/python -m pytest -q
```

A real two-GPU frozen-adapter smoke (engine startup plus one API trajectory)
is available separately from the CPU-safe regression suite:

```bash
CUDA_VISIBLE_DEVICES=0,2 .venv/bin/python scripts/smoke_product_service.py
```

The real smoke completed on 2026-07-31 with the frozen SFT20 adapter: ready
health, five model calls, the expected five-tool recovery sequence, 15 stored
events, a submitted terminal state and `task_success=true`. The machine-readable
record is `reports/service/p0_p2_real_smoke.json`.

Acceptance coverage includes:

- complete add-column trajectory through the HTTP API and SSE replay;
- model, adapter hash, database version, budget and event persistence;
- three concurrent Sessions with two active generation slots and no state
  crossover;
- rejected `UPDATE` with rollback and an unchanged source database hash;
- max-turn, timeout and explicit cancellation terminal states.
