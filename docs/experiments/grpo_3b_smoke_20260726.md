# 3B column-rename GRPO smoke — 2026-07-26

## Result

The Qwen2.5-Coder-3B-Instruct multi-turn GRPO smoke completed on physical GPUs
0 and 3. The model used a five-step oracle SFT LoRA warm-start and interacted
with the real versioned SQLite tools and execution-grounded reward.

- model revision: `488639f1ff808d1d3d0ba301aef8c11461451ec5`
- SFT: 5 steps, validation loss `1.15913`
- SFT adapter: rank/alpha `32/32`, 504 tensors, 114.25 MiB
- GRPO: 2 prompts x 2 rollouts, one optimizer step
- reward: mean `0.2775`, min `0.0`, max `1.1`
- advantage: min `-0.7071`, max `0.7071`
- policy-gradient loss: `0.051274`
- actor gradient norm: `0.040771`
- peak reserved GPU memory: `7.81 GiB` per GPU
- step time: `67.26 s`; throughput: `42.51 tokens/s`
- mean turns: `9.5`; aborted ratio: `0.0`

Artifacts:

- log: `logs/grpo_column_rename_3b_smoke.log`
- rollout: `checkpoints/grpo_column_rename_3b_smoke/rollouts/1.jsonl`
- checkpoint: `checkpoints/grpo_column_rename_3b_smoke/global_step_1`
- warm-start adapter:
  `checkpoints/sft_column_rename_3b_smoke/global_step_5/merged/lora_adapter`

## Rollout diagnosis

All four trajectories entered the stateful tool loop. One trajectory correctly
called `get_schema_version`, `inspect_schema_diff`, tested the repaired SQL,
submitted it, and scored `1.10`. A sibling rollout made a subtle wrong repair
and scored `0.01`. The two harder airline-schema trajectories repeatedly tested
invalid table/column rewrites; one never submitted and the other submitted an
invalid query, so both scored `0.0`.

This is useful GRPO signal: there is within-prompt success/failure variation,
non-zero relative advantage, and a completed actor update. It also exposes the
actual 3B bottleneck: the model recognizes that drift inspection is required,
but its schema-grounded identifier repair is not yet reliable after only five
SFT steps. The next formal training run should increase SFT coverage and mix
easy and hard drift groups rather than simply increasing GRPO steps.

## Reproduction

```bash
.venv/bin/python scripts/bootstrap_model.py --model-key smoke_model_3b
CUDA_VISIBLE_DEVICES=0,3 bash scripts/train_3b_sft_smoke.sh
.venv/bin/python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir checkpoints/sft_column_rename_3b_smoke/global_step_5 \
  --target_dir checkpoints/sft_column_rename_3b_smoke/global_step_5/merged
CUDA_VISIBLE_DEVICES=0,3 bash scripts/train_3b_grpo_smoke.sh
```
