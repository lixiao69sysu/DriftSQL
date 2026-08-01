#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INITIAL_ADAPTER_PATH="${INITIAL_ADAPTER_PATH:-${PROJECT_ROOT}/checkpoints/stage7_failure_balanced_grpo_7b/global_step_10/merged/lora_adapter}"

if [[ ! -f "${INITIAL_ADAPTER_PATH}/adapter_model.safetensors" ]]; then
  echo "Frozen Stage 7 GRPO10 adapter not found: ${INITIAL_ADAPTER_PATH}" >&2
  exit 1
fi
if [[ ! -f "${PROJECT_ROOT}/data/processed/stage8_fresh_sft/train.parquet" ]]; then
  echo "Build the Stage 8 fresh SFT data first." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2}"
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/Qwen2.5-Coder-7B-Instruct}"
export TRAIN_DATA="${TRAIN_DATA:-${PROJECT_ROOT}/data/processed/stage8_fresh_sft/train.parquet}"
export VAL_DATA="${VAL_DATA:-${PROJECT_ROOT}/data/processed/stage8_fresh_sft/tune.parquet}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/stage8_fresh_sft_7b}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-20}"
export SAVE_FREQ="${SAVE_FREQ:-5}"
export TEST_FREQ="${TEST_FREQ:-5}"
export LEARNING_RATE="${LEARNING_RATE:-2e-6}"
export MAX_TOKEN_LENGTH="${MAX_TOKEN_LENGTH:-4096}"

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${WANDB_DIR:-${PROJECT_ROOT}/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${PROJECT_ROOT}/.cache/wandb}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${PROJECT_ROOT}/.config/wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-driftsql-rl}"
mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}"

exec bash "${PROJECT_ROOT}/scripts/train_sft_smoke.sh" \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="${WANDB_PROJECT}" \
  trainer.experiment_name=stage8-fresh-db-submit-sft-7b \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  data.train_batch_size=8 \
  data.num_workers=0 \
  data.max_token_len_per_gpu="${MAX_TOKEN_LENGTH}" \
  data.max_length="${MAX_TOKEN_LENGTH}" \
  data.custom_cls.path="${PROJECT_ROOT}/driftsql/data/verl_sft_dataset.py" \
  data.custom_cls.name=JsonActionSFTDataset \
  +data.shuffle=true \
  +data.seed=82028 \
  model.lora_adapter_path="${INITIAL_ADAPTER_PATH}" \
  optim.lr="${LEARNING_RATE}" \
  "$@"
