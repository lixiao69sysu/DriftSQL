#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REASONING_ADAPTER_PATH="${REASONING_ADAPTER_PATH:-${PROJECT_ROOT}/checkpoints/stage3_five_tool_sft_3b_native_v2/global_step_160/merged/lora_adapter}"

if [[ ! -f "${REASONING_ADAPTER_PATH}/adapter_model.safetensors" ]]; then
  echo "Reasoning adapter not found: ${REASONING_ADAPTER_PATH}" >&2
  exit 1
fi

export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/Qwen2.5-Coder-3B-Instruct}"
export TRAIN_DATA="${TRAIN_DATA:-${PROJECT_ROOT}/data/processed/five_tool_sft_native_v4_json/train.parquet}"
export VAL_DATA="${VAL_DATA:-${PROJECT_ROOT}/data/processed/five_tool_sft_native_v4_json/val.parquet}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/stage3_five_tool_sft_3b_native_v4_json}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-80}"
export DATASET_CLASS="${DATASET_CLASS:-JsonActionSFTDataset}"
export SAVE_FREQ="${SAVE_FREQ:-40}"
export TEST_FREQ="${TEST_FREQ:-40}"
export LEARNING_RATE="${LEARNING_RATE:-1e-5}"

exec bash "${PROJECT_ROOT}/scripts/train_sft_smoke.sh" \
  trainer.experiment_name=stage3-five-tool-sft-3b \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  data.train_batch_size=4 \
  data.num_workers=0 \
  data.custom_cls.path="${PROJECT_ROOT}/driftsql/data/verl_sft_dataset.py" \
  data.custom_cls.name="${DATASET_CLASS}" \
  +data.shuffle=true \
  +data.seed=42 \
  model.lora_adapter_path="${REASONING_ADAPTER_PATH}" \
  optim.lr="${LEARNING_RATE}" \
  "$@"
