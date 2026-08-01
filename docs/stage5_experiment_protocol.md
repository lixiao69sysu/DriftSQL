# Stage 5: actual tuning protocol

Stage 5 uses database-disjoint Dataset V2 throughout. Model selection and all
ablation decisions use **Dev 169 only**. `Test 181` and `frozen 78` remain sealed
until the complete experiment matrix has been selected.

## Fixed conditions

- Base: `Qwen2.5-Coder-7B-Instruct`.
- Actor adaptation: rank-32 LoRA, BF16.
- SFT warm start: previous 7B five-tool adapter.
- Main RL algorithm: VERL GRPO, `n=2`, learning rate `1e-6`, KL coefficient
  `0.01`, maximum 7 assistant turns.
- Main shaped reward: success `+1.0`, matched clarification `+0.2`, executable
  final SQL `+0.1`, efficiency `+0.1`, plus explicit tool/token/repetition/
  invalid/timeout/turn-limit/missing-submit costs and unsafe penalty `-1.0`.
- Evaluation: greedy decoding, identical model/tool/turn/token budget, real
  SQLite execution and result-fingerprint match, with terminal fallback off.

## Required experiment matrix

| ID | Training condition | Dev comparison | Purpose |
|---|---|---|---|
| S0 | Base 7B, no tuning | unified Dev 169 | raw model floor |
| S1 | previous 7B Tool-SFT | unified Dev 169 | pre-V2 baseline |
| S2 | Dataset-V2 7B Tool-SFT | unified Dev 169 | new SFT baseline |
| R0 | S2 + shaped GRPO | unified Dev 169 | main model |
| A1 | R0 environment without `ask_user` | R0 under same restricted environment | inference-only tool value |
| A2 | S2 + GRPO trained without `ask_user` | A1 | adaptation to missing clarification |
| H1 | R0 environment without HKB retrieval | R0 under same restricted environment | inference-only HKB value |
| H2 | S2 + GRPO trained without HKB retrieval | H1 | adaptation to missing HKB |
| T3/T5/T7 | GRPO with max turns 3/5/7 | same respective Dev budget | interaction budget curve |
| RW0/RW1 | sparse success reward / shaped reward | full Dev environment | credit-assignment effect |
| RP0/RP1 | uniform continuation / 50% hard-failure replay | same R0 checkpoint and update count | replay effect |

Tool ablation parquet files retain the same instance IDs, labels, result
fingerprints, DB split, and row counts. Only `tool_selection` and its matching
system instruction change.

Hard replay is mined from R0 training rollouts by exact `instance_id`. The two
replay arms start from the same R0 adapter, reset optimizer state, run the same
number of continuation updates, and differ only in sampling distribution.

## Gates

1. Choose the V2 SFT checkpoint only by Dev loss, then verify that selected
   checkpoint against S0/S1 using Dev execution success.
2. Establish S0/S1/S2 before GRPO.
3. Run and evaluate R0 before mining failures.
4. Complete A/H/T/RW/RP comparisons on Dev.
5. Freeze the selected policy and config.
6. Run Test 181 and frozen 78 exactly once; do not use either result for tuning.

Primary metric is execution-grounded task success. Secondary metrics are
executable rate, submission rate, average model/tool calls, turn-limit rate,
duplicate questions/executions, timeouts, and unsafe actions.
