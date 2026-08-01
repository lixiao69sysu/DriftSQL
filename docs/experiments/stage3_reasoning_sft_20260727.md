# Stage 3 SQL Reasoning SFT smoke — 2026-07-27

> Historical smoke record. The formal step-40/80 run and completed five-tool
> stage are documented in `stage3_complete_20260727.md`.

## Outcome

The first half of Stage 3 now has a formal data pipeline and a completed 3B
training smoke. The dataset is database-disjoint, every retained Gold SQL is
parsed and executed against its real BIRD SQLite database, and the output is a
concise AST-derived relational plan plus executable SQL.

This run proves data and training correctness. It is not yet a held-out
execution-accuracy claim or the final Reasoning SFT training run.

## Dataset

Source: BIRD23 Train Filtered.

| Item | Value |
|---|---:|
| Source rows | 6,601 |
| Accepted, executable rows | 6,596 |
| Rejected timeouts | 5 |
| Train split | 5,143 rows / 55 databases |
| Validation split | 1,453 rows / 14 databases |
| Train/validation database overlap | 0 |
| Rows with full database schema | 5,846 |
| Rows using bounded schema context | 750 |
| Token length median / p95 / max | 803 / 2,266 / 2,954 |
| Token budget | 4,096 |

The five rejected queries exceeded the strict 10-second SQLite progress
deadline. They are recorded in `data/processed/reasoning_sft/rejected.jsonl`
rather than silently retained.

For databases whose complete DDL fits the context budget, the prompt contains
the entire schema. For oversized schemas, the offline training builder keeps
all Gold-referenced physical tables and fills the remaining budget with
question/Evidence-ranked distractor tables. This operation is explicitly
tagged as `gold_tables_plus_question_ranked_distractors` in the manifest; it is
not presented as the production retrieval policy.

The target uses an auditable logical plan:

```text
<plan>
1. Read the required tables.
2. Join and filter using the parsed SQL conditions.
3. Group, project, sort, and limit as required.
</plan>
<sql>
SELECT ...
</sql>
```

It deliberately avoids teacher-generated free-form chain-of-thought that
cannot be checked against the database.

## 3B LoRA smoke

- base model: Qwen2.5-Coder-3B-Instruct
- method: BF16 LoRA, rank/alpha 32/32, two-GPU FSDP
- sampled train/validation rows: 512 / 128
- optimizer steps: 10
- initial/final training loss: `0.8299 / 0.5900`
- validation loss: `0.6294`
- peak allocated/reserved CUDA memory: `8.89 / 11.32 GiB` per GPU
- checkpoint: `checkpoints/stage3_reasoning_sft_3b_smoke/global_step_10`
- portable adapter: `global_step_10/merged/lora_adapter`
- adapter tensors: 504/504 nonzero
- adapter size: 119,802,000 bytes
- adapter SHA-256: `aaaaa229d554bd52f33037b079299e6e56b35fcb8847ff66c7ecbb98e00c6959`

The full repository suite passes: `34 passed`.

## Reproduction

```bash
env TMPDIR="$PWD/data/tmp" .venv/bin/python scripts/prepare_reasoning_sft.py
CUDA_VISIBLE_DEVICES=0,3 bash scripts/train_3b_reasoning_sft_smoke.sh
.venv/bin/python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir checkpoints/stage3_reasoning_sft_3b_smoke/global_step_10 \
  --target_dir checkpoints/stage3_reasoning_sft_3b_smoke/global_step_10/merged
```

## Remaining gate

The first fixed executable comparison has now been run on 128 validation rows,
stratified across all 14 held-out training databases with identical greedy
decoding:

| Model | EX | Executable | Exact output wrapper |
|---|---:|---:|---:|
| Base Qwen2.5-Coder-3B | 39.8% | 74.2% | 100.0% |
| 10-step Reasoning LoRA | 44.5% | 71.9% | 100.0% |

The LoRA gained 17 tasks and lost 11, for a net +6/128 (+4.7 points). The
exact paired McNemar p-value is `0.3449`, so this small tuning set does not yet
establish a statistically stable improvement.

The structural slices are informative: joins improved by 11 correct tasks,
grouping by 2, and ordering by 3, while the seven simple queries lost 2. Wrong
executable results fell from 44 to 35 and syntax failures fell from 5 to 3,
but missing-column errors increased from 28 to 33. The current adapter is
learning relational structure faster than schema precision.

Accordingly, the gate is **promising but not yet passed for model promotion**.
The next tuning run should checkpoint at 40 and 80 steps, retain simple-query
examples in every batch, and select on both EX and executable rate. Use this
same 128-task set for checkpoint selection, then run the untouched full 1,453
validation rows once on the selected model.

The paired artifacts are under `reports/stage3/reasoning_eval/`. The second
half of Stage 3 still requires five-tool SFT trajectories with `ask_user`,
schema/HKB retrieval, SQL execution/repair, and final submission.
