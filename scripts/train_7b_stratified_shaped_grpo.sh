#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SFT_ADAPTER_PATH="${SFT_ADAPTER_PATH:-}"

if [[ -z "${SFT_ADAPTER_PATH}" || ! -f "${SFT_ADAPTER_PATH}/adapter_model.safetensors" ]]; then
  echo "Set SFT_ADAPTER_PATH to the selected Dataset V2 7B PEFT adapter." >&2
  exit 1
fi

export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/Qwen2.5-Coder-7B-Instruct}"
export TRAIN_DATA="${TRAIN_DATA:-${PROJECT_ROOT}/data/processed/stratified_five_tool_v2/rl_train.parquet}"
export VAL_DATA="${VAL_DATA:-${PROJECT_ROOT}/data/processed/stratified_five_tool_v2/rl_dev.parquet}"
export LORA_ADAPTER_PATH="${SFT_ADAPTER_PATH}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/stage5_stratified_shaped_grpo_7b}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-40}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-2}"
export ROLLOUT_N="${ROLLOUT_N:-2}"
export ROLLOUT_TP="${ROLLOUT_TP:-2}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-6144}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.5}"
export AGENT_WORKERS="${AGENT_WORKERS:-2}"
export MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-7}"
export MAX_USER_TURNS="${MAX_USER_TURNS:-7}"
export SAVE_FREQ="${SAVE_FREQ:-10}"
export USE_FUSED_KERNELS="${USE_FUSED_KERNELS:-true}"
export FUSED_KERNEL_BACKEND="${FUSED_KERNEL_BACKEND:-triton}"

exec bash "${PROJECT_ROOT}/scripts/train_grpo_smoke.sh" \
  trainer.experiment_name=stage5-stratified-shaped-grpo-7b \
  actor_rollout_ref.model.lora.merge="${LORA_MERGE:-false}" \
  actor_rollout_ref.model.use_fused_kernels="${USE_FUSED_KERNELS}" \
  actor_rollout_ref.model.fused_kernel_options.impl_backend="${FUSED_KERNEL_BACKEND}" \
  actor_rollout_ref.actor.optim.lr="${GRPO_LEARNING_RATE:-1e-6}" \
  actor_rollout_ref.actor.kl_loss_coef="${KL_COEF:-0.01}" \
  +reward.reward_kwargs.success_weight="${SUCCESS_WEIGHT:-1.0}" \
  +reward.reward_kwargs.clarify_weight="${CLARIFY_WEIGHT:-0.2}" \
  +reward.reward_kwargs.valid_weight="${VALID_WEIGHT:-0.1}" \
  +reward.reward_kwargs.efficient_weight="${EFFICIENT_WEIGHT:-0.1}" \
  +reward.reward_kwargs.tool_call_cost="${TOOL_CALL_COST:-0.01}" \
  +reward.reward_kwargs.token_cost="${TOKEN_COST:-0.00001}" \
  +reward.reward_kwargs.duplicate_penalty="${DUPLICATE_PENALTY:-0.05}" \
  +reward.reward_kwargs.repeated_tool_penalty="${REPEATED_TOOL_PENALTY:-0.05}" \
  +reward.reward_kwargs.invalid_penalty="${INVALID_PENALTY:-0.05}" \
  +reward.reward_kwargs.timeout_penalty="${TIMEOUT_PENALTY:-0.2}" \
  +reward.reward_kwargs.turn_limit_penalty="${TURN_LIMIT_PENALTY:-0.3}" \
  +reward.reward_kwargs.missing_submit_penalty="${MISSING_SUBMIT_PENALTY:-0.2}" \
  +reward.reward_kwargs.unsafe_penalty="${UNSAFE_PENALTY:-1.0}" \
  +reward.reward_kwargs.efficient_tool_calls="${EFFICIENT_TOOL_CALLS:-6}" \
  "$@"
