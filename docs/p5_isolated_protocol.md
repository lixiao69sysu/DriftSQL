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

## Human-reviewed replay and training

P4 failures are immutable candidates, not automatically trusted labels. Review
them in the Chinese Studio or through `POST /api/replay/candidates/{id}/reviews`.
Every append-only decision is bound to the trajectory SHA-256. Only approved
failure strata may sample new rows, and those rows always come from P5 Train;
the original P4 Tune row is never copied into optimization data.

```bash
TMPDIR=/tmp .venv/bin/python scripts/build_p5_reviewed_replay.py
TMPDIR=/tmp .venv/bin/python scripts/prepare_p5_grpo.py
TMPDIR=/tmp .venv/bin/python scripts/verify_p5_training_inputs.py
CUDA_VISIBLE_DEVICES=0,2 bash scripts/train_7b_p5_reviewed_grpo.sh
```

The preparation step fails closed when there are no real approvals, when a
review hash is stale, or when any Train/Tune/Gate database overlap is found.
W&B receives the P5 GRPO training metrics; the sealed Gate is not read.

## Tune selection and one-shot Gate

```bash
CUDA_VISIBLE_DEVICES=0,2,3 bash scripts/run_p5_tune_matrix.sh
TMPDIR=/tmp .venv/bin/python scripts/freeze_p5_candidate.py
TMPDIR=/tmp .venv/bin/python scripts/prepare_p5_gate_eval.py
TMPDIR=/tmp .venv/bin/python scripts/eval_p5_frozen_gate.py --gpus 0,2,3
TMPDIR=/tmp .venv/bin/python scripts/finalize_p5_gate.py
```

Tune compares frozen SFT20 with P5 GRPO steps 5 and 10 using the same budget.
Selection prioritizes overall success, then the 12-row turn-limit hard slice,
then lower interaction cost; SFT20 wins an exact tie. Candidate files, input
data, code, reports, thresholds and inference budget are hashed before the Gate
can be opened. The first Gate-open attempt creates an exclusive lifecycle file
before reading any row, so even a failed attempt cannot be repeated as a new
selection opportunity. Finalization records the one-shot result and permanent
seal; it never feeds failures back into training.
