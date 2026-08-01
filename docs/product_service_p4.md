# DriftSQL Studio P4

## Delivered boundary

P4 adds experiment management and operational observability without changing
the frozen SFT20 inference path. The source of truth for online metrics is the
P2 SQLite Session/Event store; restarting the Studio does not reset charts or
failure records.

The Chinese `运行监控` workspace provides:

- terminal success rate, average latency, average model/tool calls and total
  prompt/response Token usage;
- persisted tool-call failure rate and explicit unsafe/timeout counters;
- success rate stratified by schema-drift type;
- a 30-day success/failure run trend;
- deployed base-model, Adapter path/hash, Session count and success rate;
- failure classification into task failure, unsafe operation, timeout, budget
  exhaustion, cancellation and service error;
- failure filtering and one-click replay of the original full trajectory;
- immutable P4 failure candidates with hash-bound, append-only human
  approve/reject decisions in the Chinese Replay review panel;
- optional W&B run discovery for reward, KL, loss, learning-rate, throughput
  and validation/test summary metrics, plus sampled in-Studio training curves.

The aggregate APIs never expose gold SQL, expected fingerprints, database
paths or W&B credentials. W&B summaries whitelist only numeric training metric
names; arbitrary run configuration and host metadata are not returned.

## API

```text
GET /api/observability/summary
GET /api/observability/failures?failure_type=task_failure
GET /api/replay/candidates
POST /api/replay/candidates/{candidate_id}/reviews
GET /api/observability/wandb/runs
GET /api/observability/wandb/runs/{run_id}/history
```

The product OpenAPI describes all JSON/SSE and review interfaces. Failure records carry
the Session ID required by the existing trajectory endpoint, so replay uses the
same durable evidence as the Agent debugger rather than a copied summary.

## Enable W&B

W&B is deliberately disabled by default. Log in with `wandb login` or export
`WANDB_API_KEY`, then start the service with the entity and project configured:

```bash
cd /path/to/driftsql-rl
export WANDB_API_KEY='your-server-side-key'
export DRIFTSQL_SERVICE_WANDB_ENABLED=true
export DRIFTSQL_SERVICE_WANDB_ENTITY='your-entity'
export DRIFTSQL_SERVICE_WANDB_PROJECT='driftsql-rl'
export DRIFTSQL_SERVICE_PORT=8001
export CUDA_VISIBLE_DEVICES=0,2
export DRIFTSQL_SERVICE_TENSOR_PARALLEL_SIZE=2
bash scripts/serve_studio.sh
```

The API key remains a Pydantic secret and is passed only to the server-side W&B
client. A missing key, unavailable network or W&B error produces a typed error
state in the monitoring card and does not stop inference or application
startup.

## Verification

```bash
bash scripts/build_frontend.sh
TMPDIR=/tmp .venv/bin/python -m pytest tests/service -q
TMPDIR=/tmp .venv/bin/python -m pytest -q
```

The real-model P4 acceptance used Qwen2.5-Coder-7B-Instruct with the frozen
Stage-8 SFT20 Adapter, persisted its complete trajectory, and exercised the
Chinese Studio and W&B curve view. Machine-readable evidence is stored at
`reports/service/p4_real_acceptance.json`; browser recordings and screenshots
are under `artifacts/p4_real_acceptance/`. Replay-review and authentication
browser acceptances are separate so they never create fake human decisions.
