# GRPO tool-loop smoke test — 2026-07-26

## Result

The two-GPU, one-step GRPO smoke test completed successfully on physical GPUs
0 and 3 with Qwen2.5-Coder-7B-Instruct and the five-step SFT LoRA adapter.

- Output: `checkpoints/grpo_column_rename_smoke_2gpu_tp2_retry6`
- Log: `logs/grpo_column_rename_smoke_2gpu_tp2_retry6.log`
- Rollouts: `rollouts/1.jsonl`
- Step time: 79.50 seconds
- Throughput: 34.06 tokens/second
- Mean agent turns: 7.25 (min 3, max 10)
- Mean reward: 0.8175 (min 0.0, max 1.1)
- Advantage range: -0.7071 to 0.7071
- Policy-gradient loss: -0.286607
- Actor gradient norm: 0.06934
- Peak allocated GPU memory: 12.82 GiB

All four sampled trajectories reached `submit_solution`. Three recovered the
correct result and scored 1.07–1.10; one submitted an incorrect stale query and
scored 0.0. This within-group reward variation produced a non-zero GRPO update.

## Tool-call parser change

The SFT adapter emitted a mixture of tagged JSON, bare JSON, and fenced Markdown
JSON. The stock Hermes parser only accepted `<tool_call>...</tool_call>`, which
terminated those trajectories after their first assistant response. The
`driftsql-json` parser now normalizes all three forms. The same shared scanner is
used by the execution-grounded reward so rollout execution and scoring cannot
disagree about the recognized call sequence.

The four exact outputs captured by the preceding retry are regression fixtures
in `tests/test_tool_call_parsing.py`.

## Memory adjustment

Once tool execution worked, longer trajectories made VERL's redundant
old-log-prob entropy pass exceed 24 GiB. The smoke configuration now enables
rollout-correction bypass mode. Because rollout weights are synchronized before
sampling and vLLM already returns rollout log probabilities, this removes the
extra entropy forward while retaining the PPO clipped objective. Old-log-prob
time fell to 0.05 seconds and the actor update completed within memory.

## Verification

The full local test suite passes: 21 tests. The production smoke run completed
without a traceback, saved both FSDP LoRA shards, wrote the rollout JSONL, and
released GPUs 0 and 3.
