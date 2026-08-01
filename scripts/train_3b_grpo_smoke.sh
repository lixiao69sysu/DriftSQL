#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/Qwen2.5-Coder-3B-Instruct}"
export LORA_ADAPTER_PATH="${LORA_ADAPTER_PATH:-${PROJECT_ROOT}/checkpoints/sft_column_rename_3b_smoke/global_step_5/merged/lora_adapter}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/grpo_column_rename_3b_smoke}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-2}"
export ROLLOUT_N="${ROLLOUT_N:-2}"
export ROLLOUT_TP="${ROLLOUT_TP:-1}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-6144}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-5120}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.65}"
export AGENT_WORKERS="${AGENT_WORKERS:-2}"

exec bash "${PROJECT_ROOT}/scripts/train_grpo_smoke.sh" \
  trainer.experiment_name=column-rename-grpo-3b-smoke \
  "$@"
