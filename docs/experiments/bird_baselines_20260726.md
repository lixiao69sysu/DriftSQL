# BIRD baseline reproduction — 2026-07-26

This note records the resource-scaled baseline runs completed before adding
DriftSQL mutations. All training used the local
`Qwen2.5-Coder-7B-Instruct` checkpoint, LoRA rank 32, VERL, and two GPUs.

## BIRD-RL single-turn GRPO

The single-turn reasoning recipe completed one optimizer step with four
rollouts and saved a checkpoint.

- reward: mean `0.25`, min `0.0`, max `1.0`
- advantage: min `-0.7071`, max `0.7071`
- policy-gradient loss: `0.021885`
- actor gradient norm: `0.055908`
- log: `logs/bird_rl_reasoning_grpo_smoke_retry3.log`
- checkpoint: `checkpoints/bird_rl_reasoning_grpo_smoke_retry3/global_step_1`

This is a meaningful GRPO smoke rather than a launch-only check: the rollout
group contains both passing and failing executions, produces non-zero relative
advantages, updates the actor, and persists its state.

## BIRD-RL multi-turn SFT and GRPO

The Stage-2-style tool SFT uses 64 training and 8 validation trajectories built
from SIX-GYM. Each oracle trajectory executes the buggy SQL and then submits
the Gold SQL. The 32-step run reached:

- final training loss: `0.179226`
- validation loss: `0.321932`
- peak reserved GPU memory: `16.30 GiB`
- log: `logs/bird_rl_multiturn_sft_smoke_retry2.log`
- checkpoint: `checkpoints/bird_rl_multiturn_sft_smoke_retry2/global_step_32`
- merged PEFT adapter:
  `checkpoints/bird_rl_multiturn_sft_smoke_retry2/global_step_32/merged/lora_adapter`

The subsequent stateful GRPO smoke completed a real five-turn interaction:
the model called `execute_sql`, observed the database response, repaired the
query, called `submit_solution`, received the official executable reward, ran
the actor update, and saved a checkpoint.

- two trajectories, both officially scored `1.0`
- actor loss: `0.019707`; gradient norm: `0.002548`
- average turns: `5.0`; aborted ratio: `0.0`
- log: `logs/bird_rl_agentic_grpo_smoke_retry5.log`
- rollout: `checkpoints/bird_rl_agentic_grpo_smoke_retry5/rollouts/1.jsonl`
- checkpoint: `checkpoints/bird_rl_agentic_grpo_smoke_retry5/global_step_1`

Both easy-sample rollouts passed, so within-group GRPO advantages are zero in
this particular multi-turn smoke. The harder retry logs contain genuine failed
executions, while the single-turn run above demonstrates non-zero GRPO
advantages. A larger mixed-difficulty batch is therefore required before
claiming learning quality; this run only establishes that the full multi-turn
training plumbing works.

### Compatibility boundary

The upstream BIRD-RL loop expects Hermes `<tool_call>` tags. Qwen2.5-Coder
reliably emitted equivalent bare or fenced JSON. DriftSQL registers the
`bird-json-compat` parser and normalizes those calls before delegating to the
upstream BIRD executable reward. Prompts, tools, database cleanup, tests,
reward semantics, GRPO, and checkpointing remain upstream-compatible; only
JSON surface parsing and resource sizes differ from the original recipe.

## BIRD-Interact Mini public evaluation

`scripts/run_bird_interact_public_smoke.py` ran the upstream SQLite action
handler against the public Mini-Interact package. The audit found 300 tasks
over 26 databases, with all database and external-knowledge assets present.
The official actions `get_schema`, `get_knowledge_definition`, `execute`, and
`submit` all ran successfully. The machine-readable report is
`reports/bird_interact_public_smoke.json`.

An official success rate cannot be computed from the public package: all
300 rows have empty `sol_sql` and `test_cases` fields. Accordingly, the public
`SELECT 1` probe receives reward zero and does not finish the task. This is a
ground-truth availability limitation, not an environment failure. We do not
substitute generated labels and call them an official BIRD-Interact score.

## Acceptance status

- Checklist item 2, BIRD-RL original single-/multi-turn training: **complete
  as a resource-scaled reproduction**, with the JSON compatibility boundary
  documented above.
- Checklist item 3, BIRD-Interact Mini original evaluation: **public
  environment path complete; official SR blocked by withheld ground truth**.
  It can only become a fully scored official evaluation after obtaining the
  private test cases from the benchmark maintainers.

Reproduce the public checks with:

```bash
bash scripts/prepare_bird_rl_baseline.sh
CUDA_VISIBLE_DEVICES=0,3 bash scripts/train_bird_rl_reasoning_smoke.sh
CUDA_VISIBLE_DEVICES=0,3 bash scripts/train_bird_rl_multiturn_sft_smoke.sh
CUDA_VISIBLE_DEVICES=0,3 bash scripts/train_bird_rl_agentic_smoke.sh
.venv/bin/python scripts/run_bird_interact_public_smoke.py
```
