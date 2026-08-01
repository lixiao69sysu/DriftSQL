# Multi-drift 7B SFT and GRPO — 2026-07-26

## Scope

This run moves beyond the single column-rename smoke to four executable drift
types: `rename_column`, `rename_table`, `replace_column`, and `add_column`.
The last case is a silent contract failure: stale `SELECT *` still executes,
but returns an unexpected extra field, so error-only recovery is insufficient.

The generated corpus has 400 validated trajectories, 100 per drift type,
covering 57 source databases. Database-grouped splitting produced 322 training
rows over 46 databases and 78 validation rows over 11 disjoint databases.

## 7B SFT

- model: Qwen2.5-Coder-7B-Instruct
- method: BF16 LoRA, rank/alpha 32/32, two-GPU FSDP
- steps: 80
- initial training loss: 1.4239
- final training loss: 0.3098
- final validation loss: 0.3064
- active CUDA memory peak: 14.05 GiB per GPU
- adapter: 392 tensors, 154.1 MiB

Artifacts:

- log: `logs/sft_schema_drift_7b.log`
- checkpoint: `checkpoints/sft_schema_drift_7b/global_step_80`
- PEFT adapter: `checkpoints/sft_schema_drift_7b/global_step_80/merged/lora_adapter`

## 7B GRPO

The GRPO run warm-started from the SFT adapter and used live multi-turn tools:
schema version lookup, schema-diff inspection, read-only SQL execution, and
terminal solution submission. It ran 10 optimizer steps with two prompts per
step and two sampled trajectories per prompt.

The first launch exposed a real co-location tuning issue. A vLLM utilization
target of 0.8 requires 18.95 GiB free per 24 GiB card at startup, while the
resident FSDP actor left only 13.14–15.57 GiB. Reducing vLLM's budget to 0.5
passed initialization and retained the 5,120-token model context under TP=2.

Measured results:

- completed: 10/10 steps, exit code 0
- sampled trajectories: 40
- mean execution reward: 0.2183
- successful trajectories (reward >= 1.0): 8/40 (20.0%)
- maximum reward: 1.10
- final-step reward mean/range: 0.260 / 0–1.01
- final-step policy-gradient loss: -0.3858
- observed non-zero GRPO advantages: approximately +/-0.707
- actor CUDA memory peak: 13.43 GiB per GPU

Training-rollout breakdown:

| Drift operation | Rollouts | Mean reward | Success rate |
| --- | ---: | ---: | ---: |
| add column / silent contract drift | 4 | 0.0075 | 0.0% |
| rename column | 12 | 0.2750 | 25.0% |
| rename table | 8 | 0.4113 | 37.5% |
| replace column | 16 | 0.1319 | 12.5% |

These are on-policy training samples, not a held-out quality claim. Their main
purpose is to prove that reward varies within GRPO groups and drives real
updates. The low silent-drift result is also actionable: the current policy is
much better at recovering from explicit SQLite errors than detecting a query
that executes with the wrong output schema.

Artifacts:

- log: `logs/grpo_schema_drift_7b.log`
- rollouts: `checkpoints/grpo_schema_drift_7b/rollouts/{1..10}.jsonl`
- summary: `checkpoints/grpo_schema_drift_7b/rollout_summary.json`
- checkpoint: `checkpoints/grpo_schema_drift_7b/global_step_10/actor`
- PEFT adapter: `checkpoints/grpo_schema_drift_7b/global_step_10/merged/lora_adapter`

## Verification and next experiment

All 23 non-Ray unit/integration tests pass. The Ray tool test is excluded from
the sandboxed test command because sandbox event-loop wakeups hang; the same
four tool/reward paths were executed successfully outside the sandbox during
the formal data smoke.

The next statistically valid comparison is a fixed, database-disjoint
validation subset evaluated with identical decoding for: base 7B, SFT 7B, and
SFT+GRPO 7B. Report execution success, result equivalence, tool cost, and the
four drift-type slices separately. After that comparison, extend the factory
to split/merge and metric-definition drift rather than treating the current
training-rollout success rate as the final project result.
