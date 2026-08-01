# P5 Unseen-Database Protocol

P5 targets the two residual production failures that matter most after Stage 8:
add-column projection recovery and trajectories that exhaust the turn budget.
It does not reuse the opened Stage-8 Gate55 for model selection.

## Data boundary

The source is BIRD-Critic-SQLite, whose 15 database IDs are disjoint from all
36 Stage-7 and 30 Stage-8 database IDs. Every candidate database is subjected
to real SQLite v1/v2 execution and result-fingerprint validation. Two databases
that did not preserve the projection contract were rejected. Twelve verified
databases were frozen at the `db_id` level:

- Train: 6 databases, 36 add-column tasks, including 24 turn-limit hard tasks.
- Tune: 3 databases, 18 tasks, including 12 turn-limit hard tasks.
- Gate: 3 databases, 18 tasks, including 12 turn-limit hard tasks.

Single-table projection tasks isolate add-column correctness. Multi-table
plain and qualified wildcard tasks are the turn-limit hard slice. Each task
has a five-model-call/five-tool-call efficiency target.

## Gate policy

`sealed_gate.jsonl` is generated and hashed once. It must not be read for
training, tuning, failure mining, replay generation, prompt editing or reward
design. It may be opened exactly once after the P5 candidate, inference budget,
and acceptance thresholds have been frozen.

The already-opened Stage-8 Gate55 is permanently sealed for P5. Only hashes of
its frozen candidate, result, audit summary and trajectory file are used; rows
and metrics are not parsed for P5 decisions.

## Commands

```bash
TMPDIR=/tmp .venv/bin/python scripts/build_p5_isolated_protocol.py
TMPDIR=/tmp .venv/bin/python scripts/verify_p5_protocol.py
```

The builder refuses to overwrite an existing protocol or seal. `--seal-only`
exists solely to recover a missing seal after validating the already-frozen P5
Gate hash; it does not rebuild or parse either Gate.
