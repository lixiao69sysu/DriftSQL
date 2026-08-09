#!/usr/bin/env bash
set -euo pipefail

# Conservative GRPO from the strongest current P6 Recovery-SFT checkpoint.
# Dev169/Test181 are intentionally absent: checkpoints are first selected on
# the Train-derived tune split, then gated on the fixed Fast42 evaluator.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SFT_ADAPTER_PATH="${SFT_ADAPTER_PATH:-${PROJECT_ROOT}/checkpoints/p6_on_policy_recovery_sft_round2_mixed_7b/global_step_10/merged/lora_adapter}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/processed/p6_targeted_grpo}"

if [[ ! -f "${SFT_ADAPTER_PATH}/adapter_model.safetensors" ]]; then
  echo "Strong P6 Recovery-SFT adapter not found: ${SFT_ADAPTER_PATH}" >&2
  exit 1
fi
if [[ ! -f "${DATA_DIR}/train.parquet" || ! -f "${DATA_DIR}/tune.parquet" ]]; then
  echo "Build the P6 targeted GRPO curriculum first." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/Qwen2.5-Coder-7B-Instruct}"
export TRAIN_DATA="${TRAIN_DATA:-${DATA_DIR}/train.parquet}"
export VAL_DATA="${VAL_DATA:-${DATA_DIR}/tune.parquet}"
export LORA_ADAPTER_PATH="${SFT_ADAPTER_PATH}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/p6_targeted_grpo_7b}"
export ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-${OUTPUT_DIR}/rollouts}"

export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-6}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
export ROLLOUT_TP="${ROLLOUT_TP:-2}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1280}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-6144}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-6144}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.48}"
export AGENT_WORKERS="${AGENT_WORKERS:-8}"
export MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-7}"
export MAX_USER_TURNS="${MAX_USER_TURNS:-7}"
export SAVE_FREQ="${SAVE_FREQ:-2}"
export DRIFTSQL_STATE_GUARDS=1
export DRIFTSQL_DYNAMIC_TOOL_MASK=1
export ADVANTAGE_SCOPE="${ADVANTAGE_SCOPE:-key_action}"
case "${ADVANTAGE_SCOPE}" in
  episode)
    export DRIFTSQL_KEY_ACTION_MASK=0
    export LOSS_AGG_MODE="${LOSS_AGG_MODE:-seq-mean-token-mean}"
    ;;
  key_action)
    export DRIFTSQL_KEY_ACTION_MASK=1
    export LOSS_AGG_MODE="${LOSS_AGG_MODE:-token-mean}"
    ;;
  *)
    echo "ADVANTAGE_SCOPE must be episode or key_action, got: ${ADVANTAGE_SCOPE}" >&2
    exit 2
    ;;
esac
export DRIFTSQL_KEY_ACTION_TOKENS="${DRIFTSQL_KEY_ACTION_TOKENS:-96}"
export PARAM_OFFLOAD="${PARAM_OFFLOAD:-false}"
export OPTIMIZER_OFFLOAD="${OPTIMIZER_OFFLOAD:-false}"

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${WANDB_DIR:-${PROJECT_ROOT}/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${PROJECT_ROOT}/.cache/wandb}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${PROJECT_ROOT}/.config/wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-driftsql-rl}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-p6-strong-sft-targeted-grpo-7b}"
mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}"

exec bash "${PROJECT_ROOT}/scripts/train_grpo_smoke.sh" \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="${WANDB_PROJECT}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.val_before_train=false \
  trainer.test_freq="${TEST_FREQ:--1}" \
  actor_rollout_ref.model.lora.merge=false \
  actor_rollout_ref.model.use_fused_kernels=true \
  actor_rollout_ref.model.fused_kernel_options.impl_backend=triton \
  actor_rollout_ref.actor.optim.lr="${GRPO_LEARNING_RATE:-1e-7}" \
  actor_rollout_ref.actor.kl_loss_coef="${KL_COEF:-0.03}" \
  actor_rollout_ref.actor.loss_agg_mode="${LOSS_AGG_MODE}" \
  algorithm.norm_adv_by_std_in_grpo=true \
  +reward.custom_reward_function.reward_kwargs.reward_version="${REWARD_VERSION:-v1}" \
  +reward.custom_reward_function.reward_kwargs.success_weight="${SUCCESS_WEIGHT:-1.0}" \
  +reward.custom_reward_function.reward_kwargs.clarify_weight="${CLARIFY_WEIGHT:-0.1}" \
  +reward.custom_reward_function.reward_kwargs.required_clarification_weight="${REQUIRED_CLARIFICATION_WEIGHT:-0.25}" \
  +reward.custom_reward_function.reward_kwargs.clarification_attempt_weight="${CLARIFICATION_ATTEMPT_WEIGHT:-0.2}" \
  +reward.custom_reward_function.reward_kwargs.post_clarification_weight="${POST_CLARIFICATION_WEIGHT:-0.1}" \
  +reward.custom_reward_function.reward_kwargs.terminal_weight="${TERMINAL_WEIGHT:-0.2}" \
  +reward.custom_reward_function.reward_kwargs.add_column_weight="${ADD_COLUMN_WEIGHT:-0.2}" \
  +reward.custom_reward_function.reward_kwargs.add_column_inspect_weight="${ADD_COLUMN_INSPECT_WEIGHT:-0.0}" \
  +reward.custom_reward_function.reward_kwargs.semantic_candidate_weight="${SEMANTIC_CANDIDATE_WEIGHT:-0.0}" \
  +reward.custom_reward_function.reward_kwargs.decision_action_weight="${DECISION_ACTION_WEIGHT:-0.0}" \
  +reward.custom_reward_function.reward_kwargs.decision_action_mismatch_penalty="${DECISION_ACTION_MISMATCH_PENALTY:-0.0}" \
  +reward.custom_reward_function.reward_kwargs.premature_stale_execute_penalty="${PREMATURE_STALE_EXECUTE_PENALTY:-0.0}" \
  +reward.custom_reward_function.reward_kwargs.valid_weight="${VALID_WEIGHT:-0.1}" \
  +reward.custom_reward_function.reward_kwargs.efficient_weight="${EFFICIENT_WEIGHT:-0.1}" \
  +reward.custom_reward_function.reward_kwargs.tool_call_cost="${TOOL_CALL_COST:-0.01}" \
  +reward.custom_reward_function.reward_kwargs.token_cost="${TOKEN_COST:-0.00001}" \
  +reward.custom_reward_function.reward_kwargs.duplicate_penalty="${DUPLICATE_PENALTY:-0.08}" \
  +reward.custom_reward_function.reward_kwargs.repeated_tool_penalty="${REPEATED_TOOL_PENALTY:-0.08}" \
  +reward.custom_reward_function.reward_kwargs.invalid_penalty="${INVALID_PENALTY:-0.1}" \
  +reward.custom_reward_function.reward_kwargs.timeout_penalty="${TIMEOUT_PENALTY:-0.25}" \
  +reward.custom_reward_function.reward_kwargs.turn_limit_penalty="${TURN_LIMIT_PENALTY:-0.5}" \
  +reward.custom_reward_function.reward_kwargs.missing_submit_penalty="${MISSING_SUBMIT_PENALTY:-0.5}" \
  +reward.custom_reward_function.reward_kwargs.unsafe_penalty="${UNSAFE_PENALTY:-1.0}" \
  +reward.custom_reward_function.reward_kwargs.missing_required_clarification_penalty="${MISSING_REQUIRED_CLARIFICATION_PENALTY:-0.25}" \
  +reward.custom_reward_function.reward_kwargs.unmatched_clarification_penalty="${UNMATCHED_CLARIFICATION_PENALTY:-0.15}" \
  +reward.custom_reward_function.reward_kwargs.invalid_post_clarification_penalty="${INVALID_POST_CLARIFICATION_PENALTY:-0.15}" \
  +reward.custom_reward_function.reward_kwargs.add_column_protocol_penalty="${ADD_COLUMN_PROTOCOL_PENALTY:-0.15}" \
  +reward.custom_reward_function.reward_kwargs.efficient_tool_calls="${EFFICIENT_TOOL_CALLS:-7}" \
  "$@"
