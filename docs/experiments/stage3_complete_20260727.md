# Stage 3 two-stage SFT completion — 2026-07-27

## Outcome

Both Stage 3 supervised stages are complete and reproducible on
Qwen2.5-Coder-3B-Instruct:

1. SQL Reasoning SFT improves held-out direct SQL execution accuracy from
   39.8% to 53.1% on the fixed 128-task gate.
2. Five-tool SFT improves held-out interactive task success from 1.3% to
   30.8% on all 78 validation tasks under a fixed seven-turn budget.

The selected adapters are:

- Reasoning: `checkpoints/stage3_reasoning_sft_3b_formal/global_step_40/merged/lora_adapter`
- Tool use: `checkpoints/stage3_five_tool_sft_3b_native_v4_json/global_step_80/merged/lora_adapter`

The two evaluations measure different capabilities. The Reasoning adapter is
promoted by direct SQL execution accuracy. It is not expected to follow the
multi-turn tool protocol without the second SFT stage.

## SQL Reasoning SFT

The BIRD23 builder accepted 6,596 of 6,601 rows after AST parsing and real
read-only SQLite execution. The database-disjoint split contains 5,143 train
rows over 55 databases and 1,453 validation rows over 14 databases; five
queries exceeded the ten-second execution deadline.

Formal LoRA/FSDP training saved step 40 and step 80. The same fixed 128 tasks,
greedy decoding, schemas, and execution evaluator were used for checkpoint
selection.

| Model | Correct | EX | Executable | Paired gains/losses | McNemar p |
|---|---:|---:|---:|---:|---:|
| Base 3B | 51/128 | 39.8% | 74.2% | — | — |
| Reasoning step 40 | 68/128 | 53.1% | 78.1% | 31 / 14 | 0.0161 |
| Reasoning step 80 | 60/128 | 46.9% | 82.0% | 23 / 14 | 0.1877 |

Step 40 wins on the primary EX metric. It gains 17 net correct tasks and the
paired improvement is significant on this fixed gate. Its largest structural
gain is joins: 56/104 correct versus 36/104 for Base.

Artifacts:

- `reports/stage3/reasoning_eval/comparison_step40/reasoning_comparison.json`
- `reports/stage3/reasoning_eval/comparison_step80/reasoning_comparison.json`

## Execution-verified five-tool data

Each accepted trajectory is replayed against a private active SQLite session
and contains this target sequence:

1. `get_schema`
2. `ask_user`
3. `get_knowledge_definition`
4. `execute_sql` on the stale query
5. `execute_sql` on the repaired query
6. `submit_solution`

The repaired execution must match the stored row-count/value fingerprint. SQL
is read-only, deadline-bounded, rolled back, and released after each episode.

| Item | Value |
|---|---:|
| Source trajectories | 400 |
| Accepted / rejected | 396 / 4 |
| Train | 318 trajectories, 46 databases |
| Validation | 78 trajectories, 11 databases |
| Database overlap | 0 |
| Next-action train examples | 1,908 |
| Next-action validation examples | 468 |
| Total supervised actions | 2,376 |
| Full-trajectory token median / max | 1,699.5 / 3,307 |

All four rejects are from the `address` database and exceeded the 4,096-token
budget. They remain in `rejected.jsonl`; none was silently truncated.

The Parquet `tools` field is stored as a JSON string. This is intentional:
Arrow's list-of-struct union otherwise injected unrelated nullable parameters
into every function schema. `NestedToolsSFTDataset` decodes it after loading.

## Protocol tuning and failure analysis

The useful engineering result was not obtained by accepting the first low
validation loss:

| Iteration | Change | Fixed-32 result | Diagnosis |
|---|---|---:|---|
| v1 | text labels with user-role observations | 9/78 on the earlier full run | training/runtime message protocol mismatch |
| v2 | native structured calls, full trajectory per row | 0/32 | model emitted the whole scripted trajectory in one response |
| v3 | one next-action prefix, final-assistant loss | 0/32 | model ended after `<think>` because the native `<tool_call>` token stayed poorly calibrated |
| v4 | structured history, parser-compatible plain-JSON target, final JSON+EOS loss | 8/32 | stable one-action generations; step 80 selected |

The final data keeps previous assistant calls in Qwen's native structured
format, exactly as the live agent loop records them. Only the next target is
rendered as one bare JSON object. `find_tool_calls` already supports that form,
and the evaluator converts it back to structured history before the next turn.

This avoids changing the agent state machine merely to accommodate a model
format. It also avoids forcing a frozen output embedding for a special token
that had only 0.0048% probability after the v3 thought text, versus 5.8% for
EOS in a token-level diagnostic.

## Unified held-out interactive evaluation

All three variants use the same 78 tasks, real versioned SQLite sessions,
tool schemas, greedy decoding, maximum seven model/tool calls, and final result
fingerprint evaluator.

| Model | Success | Executable | Submitted | All five tools | Invalid output | Avg. calls |
|---|---:|---:|---:|---:|---:|---:|
| Base 3B | 1/78 (1.3%) | 4/78 (5.1%) | 4 | 0 | 72 | 2.83 |
| Reasoning step 40 | 0/78 (0.0%) | 2/78 (2.6%) | 2 | 0 | 73 | 2.59 |
| Tool v4 step 80 | 24/78 (30.8%) | 43/78 (55.1%) | 49 | 48 | 2 | 6.51 |

Against Base, Tool v4 has 24 paired gains, one paired loss, and zero tasks
jointly correct; exact McNemar p is `1.55e-6`. Success by drift type is:

| Drift | Tasks | Base success | Tool v4 success |
|---|---:|---:|---:|
| Added column / result-contract drift | 31 | 0 | 4 |
| Replaced column | 17 | 1 | 4 |
| Renamed column | 16 | 0 | 6 |
| Renamed table | 14 | 0 | 10 |

The remaining bottleneck is behavioral efficiency, not protocol parsing: 27
Tool v4 tasks hit the seven-turn limit, while only two end in invalid output.
This gives Stage 4 GRPO a concrete target: reduce repeated clarification and
retrieval, execute the repaired SQL earlier, and submit within budget without
losing result correctness.

Artifacts:

- `reports/stage3/five_tool_eval_native_v4_json_selection/summary.json`
- `reports/stage3/five_tool_eval_native_v4_json_final/summary.json`
- per-task trajectories in the same final report directory

## Reproduction

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

CUDA_VISIBLE_DEVICES=0,3 .venv/bin/python scripts/run_five_tool_eval.py \
  --data data/processed/five_tool_sft_native_v4_json/val_agent_eval.jsonl \
  --adapter-spec reasoning-step40=checkpoints/stage3_reasoning_sft_3b_formal/global_step_40/merged/lora_adapter \
  --adapter-spec tool-json-v4-step80=checkpoints/stage3_five_tool_sft_3b_native_v4_json/global_step_80/merged/lora_adapter \
  --batch-size 32 --max-turns 7 \
  --output-dir reports/stage3/five_tool_eval_native_v4_json_final
```

The final repository regression is `37 passed` in the `lcpy311` environment.
