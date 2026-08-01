# Stage 2 interactive environment acceptance — 2026-07-27

## Outcome

The Stage 2 engineering environment is complete. DriftSQL's VERL agent loop
now exposes a trajectory-stateful BIRD-Interact adapter with bounded user
clarification, schema and HKB retrieval, an isolated read-only SQLite session,
per-action timeout/rollback, and atomic full-trajectory logs.

This is an environment acceptance result, not a BIRD-Interact success-rate
claim. The public Mini-Interact release has no Gold SQL or executable test
cases in any of its 300 rows.

## Acceptance matrix

| Requirement | Implementation | Validation | Status |
|---|---|---|---|
| BIRD-Interact user simulator | `AskUserTool`, limited to documented ambiguity points and three questions | Real `alien_1` ambiguity is answered; duplicate and unrelated questions are tested | Complete |
| Schema retrieval | `GetSchemaTool`, optional term ranking and response truncation | Real schema asset retrieved | Complete |
| HKB retrieval | `GetKnowledgeDefinitionTool`, exact/overlap retrieval over per-database JSONL | Real SNQI definition retrieved | Complete |
| Isolated database Session | SQLite backup per trajectory, reused across actions | Active path differs from source; source SHA is unchanged | Complete |
| Timeout and rollback | read-only URI, `query_only`, authorizer, progress deadline, savepoint rollback | writes are denied, recursive query times out, successful action reports rollback | Complete |
| Complete trajectory logs | one atomic JSON file per VERL request with prompts, messages, actions, observations, metrics and latency | four real GRPO rollouts produced four `completed` traces | Complete |
| VERL lifecycle | stable request ID, tool state retained until trajectory termination, release in `finally` | real actor update and checkpoint save completed | Complete |

## Data adapter

`scripts/prepare_interactive_eval.py` converts the public Mini-Interact package
into 300 VERL-compatible records over 26 databases. Every row receives the
same five-tool contract:

1. `ask_user`
2. `get_schema`
3. `get_knowledge_definition`
4. `execute_sql`
5. `submit_solution`

The record carries only public task metadata and local asset paths. Its reward
ground truth is deliberately empty and
`public_ground_truth_available=false`; no generated SQL is substituted for a
withheld official label.

Output: `data/processed/mini_interact/interactive_eval.jsonl`.

## Real-asset environment smoke

The smoke test used Mini-Interact task `alien_1` and one stable trajectory ID.
All seven checks passed:

- ambiguity clarification matched;
- schema retrieved;
- HKB definition retrieved;
- `SELECT 1` executed successfully;
- the database session was isolated;
- the SQL action was rolled back;
- final submission was accepted by the environment.

The source database hash was identical before and after the trajectory. The
machine-readable evidence is in
`reports/interactive_environment_smoke.json`.

## Real VERL/GRPO lifecycle smoke

A two-GPU Qwen2.5-Coder-3B one-step run exercised the modified agent loop on
four model-generated drift rollouts:

- all 4 environment traces ended with `status=completed`;
- 13 stateful tool actions were logged;
- score mean/min/max: `0.2625 / 0.0 / 1.05`;
- advantage min/max: `-0.7071 / 0.7071`;
- actor loss: `0.115158`;
- actor gradient norm: `0.075195`;
- aborted ratio: `0.0`;
- checkpoint and rollout dump were saved.

Artifacts:

- `checkpoints/stage2_stateful_loop_smoke/environment_traces/`
- `checkpoints/stage2_stateful_loop_smoke/rollouts/1.jsonl`
- `checkpoints/stage2_stateful_loop_smoke/global_step_1/`

The vLLM engine-shutdown message occurred after checkpointing during service
teardown; the training command exited with code 0.

## Test evidence

The full suite passes: `32 passed`. It covers state reuse, cleanup, guarded
clarification, schema/HKB lookup, write denial, timeout, rollback, source
immutability, data format, trace persistence, and the existing drift pipeline.
No `driftsql-session-*` or `driftsql-rollout-*` temporary directory remains
after the suite.

Reproduce the checks with:

```bash
.venv/bin/python scripts/prepare_interactive_eval.py
env TMPDIR="$PWD/data/tmp" DRIFTSQL_TMPDIR="$PWD/data/tmp" \
  .venv/bin/python scripts/smoke_interactive_environment.py
env TMPDIR="$PWD/data/tmp" .venv/bin/python -m pytest -q
CUDA_VISIBLE_DEVICES=0,3 \
  OUTPUT_DIR="$PWD/checkpoints/stage2_stateful_loop_smoke" \
  bash scripts/train_3b_grpo_smoke.sh
```

## Boundary for Stage 3

The deterministic user simulator is intentional: it answers only the
published ambiguity metadata, so evaluation is reproducible and cannot leak
an invented solution. It is not a free-form LLM user.

Stage 2 proves that the environment and training lifecycle work. It does not
prove that the current model has learned when to clarify or retrieve HKB.
That behavior belongs to Stage 3: execution-verified SQL-reasoning SFT and
five-tool trajectory SFT, followed by held-out behavioral evaluation.
