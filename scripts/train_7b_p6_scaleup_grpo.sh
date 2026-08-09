#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SFT_ADAPTER_PATH="${SFT_ADAPTER_PATH:-${PROJECT_ROOT}/checkpoints/p6_scaleup_recovery_hard_sft_7b/global_step_160/merged/lora_adapter}"

if [[ ! -f "${SFT_ADAPTER_PATH}/adapter_model.safetensors" ]]; then
  echo "Scale-up SFT adapter not found: ${SFT_ADAPTER_PATH}" >&2
  exit 1
fi

export DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/processed/p6_scaleup_v1_grpo}"
export SFT_ADAPTER_PATH
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/p6_scaleup_full_episode_grpo_7b}"
export ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-${OUTPUT_DIR}/rollouts}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-p6-scaleup-full-episode-grpo-7b}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-12}"
export SAVE_FREQ="${SAVE_FREQ:-2}"
export TEST_FREQ="${TEST_FREQ:--1}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
export GRPO_LEARNING_RATE="${GRPO_LEARNING_RATE:-1e-7}"
export KL_COEF="${KL_COEF:-0.03}"

exec bash "${PROJECT_ROOT}/scripts/train_7b_p6_targeted_grpo.sh" "$@"
