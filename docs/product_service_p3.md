# DriftSQL Studio P3

## Delivered surface

P3 turns the P0-P2 inference service into a same-origin Chinese operations dashboard.
It is a React 19, TypeScript and Vite single-page application served directly
by FastAPI; no second production web server or CORS configuration is required.

The Studio provides:

- searchable access to all 55 Tune scenarios and persisted Session history;
- editable turn, tool, token and timeout budgets before a run starts;
- Session creation, queueing, cancellation and terminal-state handling;
- live, replayable SSE rendering of model decisions, tool calls, observations,
  SQL results and recovery steps;
- final SQL, execution verdict and decomposed reward/safety flags;
- model, SFT20 adapter hash, sandbox identity, database hash, usage and budget
  metadata;
- a sanitized comparison of the frozen Stage 7, Stage 8 and GRPO Tune runs.

Gold SQL, expected fingerprints, database filesystem paths and raw frozen
manifests never enter the browser contract. Experiment comparison exposes only
aggregate Tune metrics from the frozen candidate manifest.

## Runtime boundary

```text
Browser
  -> FastAPI catalog/session endpoints
  -> POST Session + Run
  -> replayable EventSource stream
  -> persistent vLLM + pinned SFT20 adapter
  -> isolated read-only SQLite copy
  -> durable Session/Event/Reward records
```

FastAPI serves `/` without caching and fingerprinted `/assets/*` with immutable
caching. If `frontend/dist` is absent, `/` returns an actionable 503 instead of
silently serving a stale or incomplete interface. API documentation remains at
`/docs`; all 11 JSON/SSE interfaces are described by `/openapi.json`.

## Start the integrated Studio

```bash
cd /path/to/driftsql-rl
export CUDA_VISIBLE_DEVICES=0,2
export DRIFTSQL_SERVICE_TENSOR_PARALLEL_SIZE=2
bash scripts/serve_studio.sh
```

Open `http://127.0.0.1:8000`. The script restores locked npm dependencies when
needed, type-checks, tests and builds the frontend, then starts the same P0-P2
service with the persistent vLLM engine. To reuse an already verified bundle:

```bash
DRIFTSQL_SKIP_FRONTEND_BUILD=1 CUDA_VISIBLE_DEVICES=0,2 \
  bash scripts/serve_studio.sh
```

For frontend-only development, start `bash scripts/serve_service.sh` in one
terminal and run `npm run dev` from `frontend/` in another. Vite proxies API
and SSE traffic to port 8000.

## Verification

```bash
bash scripts/build_frontend.sh
TMPDIR=/tmp .venv/bin/python -m pytest tests/service -q
TMPDIR=/tmp .venv/bin/python -m pytest -q
```

P3 acceptance on 2026-08-01 passed TypeScript compilation, 5 frontend tests,
the production Vite build, 7 product-service tests and 98 total Python tests.
The built JavaScript is 217.72 kB (67.56 kB gzip) and CSS is 21.08 kB (5.65 kB
gzip). The machine-readable record is `reports/service/p3_acceptance.json`.

The product tests cover the same-origin index and fingerprinted asset host,
complete add-column API/SSE recovery, aggregate experiment data, sandbox
isolation, two-slot concurrency, write rejection, cancellation, timeout,
budget termination and durable trajectory replay. The existing real two-GPU
SFT20 service smoke remains recorded in
`reports/service/p0_p2_real_smoke.json`; P3 does not alter that inference path.
