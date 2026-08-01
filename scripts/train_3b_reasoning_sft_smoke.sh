#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/Qwen2.5-Coder-3B-Instruct}"
export TRAIN_DATA="${TRAIN_DATA:-${PROJECT_ROOT}/data/processed/reasoning_sft/train.parquet}"
export VAL_DATA="${VAL_DATA:-${PROJECT_ROOT}/data/processed/reasoning_sft/val.parquet}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/stage3_reasoning_sft_3b_smoke}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-10}"

exec bash "${PROJECT_ROOT}/scripts/train_sft_smoke.sh" \
  trainer.experiment_name=stage3-reasoning-sft-3b-smoke \
  data.train_batch_size=8 \
  data.train_max_samples=512 \
  data.val_max_samples=128 \
  +data.shuffle=true \
  +data.seed=42 \
  optim.lr=2e-5 \
  "$@"
