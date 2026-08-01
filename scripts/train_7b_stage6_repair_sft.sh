#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INITIAL_ADAPTER_PATH="${INITIAL_ADAPTER_PATH:-${PROJECT_ROOT}/checkpoints/stage5_ablation_no_ask_user_7b/global_step_40/merged/lora_adapter}"

if [[ ! -f "${INITIAL_ADAPTER_PATH}/adapter_model.safetensors" ]]; then
  echo "Stage 5 selected adapter not found: ${INITIAL_ADAPTER_PATH}" >&2
  exit 1
fi

export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/Qwen2.5-Coder-7B-Instruct}"
export TRAIN_DATA="${TRAIN_DATA:-${PROJECT_ROOT}/data/processed/stage6_repair_next_action_v2/train.parquet}"
export VAL_DATA="${VAL_DATA:-${PROJECT_ROOT}/data/processed/stage6_repair_next_action_v2/tune.parquet}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/stage6_repair_sft_7b}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-40}"
export SAVE_FREQ="${SAVE_FREQ:-20}"
export TEST_FREQ="${TEST_FREQ:-20}"
export LEARNING_RATE="${LEARNING_RATE:-5e-6}"

exec bash "${PROJECT_ROOT}/scripts/train_sft_smoke.sh" \
  trainer.experiment_name=stage6-repair-next-action-sft-7b \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  data.train_batch_size=8 \
  data.num_workers=0 \
  data.max_token_len_per_gpu=3072 \
  data.max_length=3072 \
  data.custom_cls.path="${PROJECT_ROOT}/driftsql/data/verl_sft_dataset.py" \
  data.custom_cls.name=JsonActionSFTDataset \
  +data.shuffle=true \
  +data.seed=62026 \
  model.lora_adapter_path="${INITIAL_ADAPTER_PATH}" \
  optim.lr="${LEARNING_RATE}" \
  "$@"
