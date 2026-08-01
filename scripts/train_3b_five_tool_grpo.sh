#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/Qwen2.5-Coder-3B-Instruct}"
export TRAIN_DATA="${TRAIN_DATA:-${PROJECT_ROOT}/data/processed/five_tool_sft_native_v2/rl_train.parquet}"
export VAL_DATA="${VAL_DATA:-${PROJECT_ROOT}/data/processed/five_tool_sft_native_v2/rl_val.parquet}"
export LORA_ADAPTER_PATH="${LORA_ADAPTER_PATH:-${PROJECT_ROOT}/checkpoints/stage3_five_tool_sft_3b_native_v4_json/global_step_80/merged/lora_adapter}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/stage4_five_tool_grpo_3b_v2}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-3}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-40}"
export SAVE_FREQ="${SAVE_FREQ:-10}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-3}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-12}"
export ROLLOUT_N="${ROLLOUT_N:-4}"
export ROLLOUT_TP="${ROLLOUT_TP:-1}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-12}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.50}"
export AGENT_WORKERS="${AGENT_WORKERS:-3}"
export MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-7}"
export MAX_USER_TURNS="${MAX_USER_TURNS:-7}"

for required in \
  "${MODEL_PATH}/config.json" \
  "${LORA_ADAPTER_PATH}/adapter_model.safetensors" \
  "${TRAIN_DATA}" \
  "${VAL_DATA}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Required Stage 4 artifact not found: ${required}" >&2
    exit 1
  fi
done

exec bash "${PROJECT_ROOT}/scripts/train_grpo_smoke.sh" \
  trainer.experiment_name=stage4-five-tool-grpo-3b \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  "$@"
