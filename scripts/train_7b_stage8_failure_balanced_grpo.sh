#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SFT_ADAPTER_PATH="${SFT_ADAPTER_PATH:-${PROJECT_ROOT}/checkpoints/stage8_fresh_sft_7b/global_step_20/merged/lora_adapter}"

if [[ ! -f "${SFT_ADAPTER_PATH}/adapter_model.safetensors" ]]; then
  echo "Stage 8 SFT20 adapter not found: ${SFT_ADAPTER_PATH}" >&2
  exit 1
fi
if [[ ! -f "${PROJECT_ROOT}/data/processed/stage8_failure_balanced_grpo/train.parquet" ]]; then
  echo "Build Stage 8 failure-balanced GRPO data first." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2}"
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/Qwen2.5-Coder-7B-Instruct}"
export TRAIN_DATA="${TRAIN_DATA:-${PROJECT_ROOT}/data/processed/stage8_failure_balanced_grpo/train.parquet}"
export VAL_DATA="${VAL_DATA:-${PROJECT_ROOT}/data/processed/stage8_fresh_sft/rl_tune.parquet}"
export LORA_ADAPTER_PATH="${SFT_ADAPTER_PATH}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/stage8_failure_balanced_grpo_7b}"
export ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-${OUTPUT_DIR}/rollouts}"

export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-10}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-2}"
export ROLLOUT_N="${ROLLOUT_N:-4}"
export ROLLOUT_TP="${ROLLOUT_TP:-2}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1280}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-6144}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-6144}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.50}"
export AGENT_WORKERS="${AGENT_WORKERS:-4}"
export MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-7}"
export MAX_USER_TURNS="${MAX_USER_TURNS:-7}"
export SAVE_FREQ="${SAVE_FREQ:-5}"
export DRIFTSQL_STATE_GUARDS="${DRIFTSQL_STATE_GUARDS:-1}"
export DRIFTSQL_DYNAMIC_TOOL_MASK="${DRIFTSQL_DYNAMIC_TOOL_MASK:-1}"
export PARAM_OFFLOAD="${PARAM_OFFLOAD:-false}"
export OPTIMIZER_OFFLOAD="${OPTIMIZER_OFFLOAD:-false}"

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${WANDB_DIR:-${PROJECT_ROOT}/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${PROJECT_ROOT}/.cache/wandb}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${PROJECT_ROOT}/.config/wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-driftsql-rl}"
mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}"

exec bash "${PROJECT_ROOT}/scripts/train_grpo_smoke.sh" \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="${WANDB_PROJECT}" \
  trainer.experiment_name=stage8-fresh-db-failure-balanced-grpo-7b \
  trainer.val_before_train=false \
  trainer.test_freq=-1 \
  actor_rollout_ref.model.lora.merge=false \
  actor_rollout_ref.model.use_fused_kernels=true \
  actor_rollout_ref.model.fused_kernel_options.impl_backend=triton \
  actor_rollout_ref.actor.optim.lr="${GRPO_LEARNING_RATE:-5e-7}" \
  actor_rollout_ref.actor.kl_loss_coef="${KL_COEF:-0.01}" \
  +reward.reward_kwargs.success_weight="${SUCCESS_WEIGHT:-1.0}" \
  +reward.reward_kwargs.clarify_weight="${CLARIFY_WEIGHT:-0.2}" \
  +reward.reward_kwargs.valid_weight="${VALID_WEIGHT:-0.1}" \
  +reward.reward_kwargs.efficient_weight="${EFFICIENT_WEIGHT:-0.1}" \
  +reward.reward_kwargs.tool_call_cost="${TOOL_CALL_COST:-0.01}" \
  +reward.reward_kwargs.token_cost="${TOKEN_COST:-0.00001}" \
  +reward.reward_kwargs.duplicate_penalty="${DUPLICATE_PENALTY:-0.08}" \
  +reward.reward_kwargs.repeated_tool_penalty="${REPEATED_TOOL_PENALTY:-0.08}" \
  +reward.reward_kwargs.invalid_penalty="${INVALID_PENALTY:-0.1}" \
  +reward.reward_kwargs.timeout_penalty="${TIMEOUT_PENALTY:-0.2}" \
  +reward.reward_kwargs.turn_limit_penalty="${TURN_LIMIT_PENALTY:-0.5}" \
  +reward.reward_kwargs.missing_submit_penalty="${MISSING_SUBMIT_PENALTY:-0.5}" \
  +reward.reward_kwargs.unsafe_penalty="${UNSAFE_PENALTY:-1.0}" \
  +reward.reward_kwargs.efficient_tool_calls="${EFFICIENT_TOOL_CALLS:-5}" \
  "$@"
