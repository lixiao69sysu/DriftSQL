#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export REASONING_ADAPTER_PATH="${REASONING_ADAPTER_PATH:-${PROJECT_ROOT}/checkpoints/stage3_reasoning_sft_3b_formal/global_step_40/merged/lora_adapter}"
export TRAIN_DATA="${TRAIN_DATA:-${PROJECT_ROOT}/data/processed/five_tool_sft_native_v2/train.parquet}"
export VAL_DATA="${VAL_DATA:-${PROJECT_ROOT}/data/processed/five_tool_sft_native_v2/val.parquet}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/stage3_five_tool_sft_3b_native_v2}"
export DATASET_CLASS="${DATASET_CLASS:-NestedToolsSFTDataset}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-160}"
export SAVE_FREQ="${SAVE_FREQ:-80}"
export TEST_FREQ="${TEST_FREQ:-80}"
export LEARNING_RATE="${LEARNING_RATE:-1e-5}"

exec bash "${PROJECT_ROOT}/scripts/train_3b_five_tool_sft.sh" "$@"
