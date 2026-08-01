#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INITIAL_ADAPTER_PATH="${INITIAL_ADAPTER_PATH:-${PROJECT_ROOT}/checkpoints/sft_schema_drift_7b/global_step_80/merged/lora_adapter}"

if [[ ! -f "${INITIAL_ADAPTER_PATH}/adapter_model.safetensors" ]]; then
  echo "Initial 7B Tool-SFT adapter not found: ${INITIAL_ADAPTER_PATH}" >&2
  exit 1
fi

export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/Qwen2.5-Coder-7B-Instruct}"
export TRAIN_DATA="${TRAIN_DATA:-${PROJECT_ROOT}/data/processed/stratified_five_tool_next_action_v2/train.parquet}"
export VAL_DATA="${VAL_DATA:-${PROJECT_ROOT}/data/processed/stratified_five_tool_next_action_v2/dev.parquet}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/stage5_stratified_five_tool_sft_7b}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-3}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-160}"
export DATASET_CLASS="${DATASET_CLASS:-JsonActionSFTDataset}"
export SAVE_FREQ="${SAVE_FREQ:-40}"
export TEST_FREQ="${TEST_FREQ:-40}"
export LEARNING_RATE="${LEARNING_RATE:-1e-5}"

exec bash "${PROJECT_ROOT}/scripts/train_sft_smoke.sh" \
  trainer.experiment_name=stage5-stratified-five-tool-sft-7b \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  data.train_batch_size=6 \
  data.num_workers=0 \
  data.max_token_len_per_gpu=6144 \
  data.max_length=6144 \
  data.custom_cls.path="${PROJECT_ROOT}/driftsql/data/verl_sft_dataset.py" \
  data.custom_cls.name="${DATASET_CLASS}" \
  +data.shuffle=true \
  +data.seed=42 \
  model.lora_adapter_path="${INITIAL_ADAPTER_PATH}" \
  optim.lr="${LEARNING_RATE}" \
  "$@"
