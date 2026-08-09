#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARM="${ARM:-${1:-}}"
case "${ARM}" in
  A|a)
    ARM=A
    GRPO_LEARNING_RATE="${GRPO_LEARNING_RATE:-1e-7}"
    KL_COEF="${KL_COEF:-0.03}"
    ;;
  B|b)
    ARM=B
    # Arm A remained at roughly 1e-4 PPO KL and 1e-3 clip fraction and did
    # not move greedy AddColumn behavior. Arm B deliberately tests a stronger
    # but still bounded update while keeping data and Reward identical.
    GRPO_LEARNING_RATE="${GRPO_LEARNING_RATE:-5e-7}"
    KL_COEF="${KL_COEF:-0.01}"
    ;;
  *)
    echo "Usage: ARM=A|B $0 (or pass A/B as the first argument)" >&2
    exit 2
    ;;
esac

export DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/processed/p6_first_action_focus200_v2}"
export SFT_ADAPTER_PATH="${SFT_ADAPTER_PATH:-${PROJECT_ROOT}/checkpoints/p6_scaleup_recovery_hard_sft_7b/global_step_160/merged/lora_adapter}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/p6_first_action_grpo_arm_${ARM,,}_v2_7b}"
export ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-${OUTPUT_DIR}/rollouts}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-p6-first-action-grpo-arm-${ARM,,}-v2-7b}"
export WANDB_PROJECT="${WANDB_PROJECT:-driftsql-rl}"
export WANDB_MODE="${WANDB_MODE:-online}"
RESUME_FROM="${RESUME_FROM:-}"

if [[ ! -f "${DATA_DIR}/train.parquet" || ! -f "${DATA_DIR}/tune.parquet" ]]; then
  echo "Build the P6 first-action Focus200 curriculum first: ${DATA_DIR}" >&2
  exit 1
fi
if [[ ! -f "${SFT_ADAPTER_PATH}/adapter_model.safetensors" ]]; then
  echo "SFT160 adapter is missing: ${SFT_ADAPTER_PATH}" >&2
  exit 1
fi
if [[ -e "${OUTPUT_DIR}" && -z "${RESUME_FROM}" ]]; then
  echo "Refusing to overwrite existing output: ${OUTPUT_DIR}" >&2
  exit 1
fi
if [[ -n "${RESUME_FROM}" && ! -d "${RESUME_FROM}" ]]; then
  echo "Resume checkpoint is missing: ${RESUME_FROM}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-25}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
export SAVE_FREQ="${SAVE_FREQ:-5}"
export TEST_FREQ="${TEST_FREQ:--1}"
export ADVANTAGE_SCOPE=episode
export LOSS_AGG_MODE=seq-mean-token-mean
export GRPO_LEARNING_RATE
export KL_COEF

# Reward V3 plus explicit AddColumn policy shaping.  The target action exists
# only in extra_info and is absent from the model-visible prompt.
export REWARD_VERSION=v3
export SUCCESS_WEIGHT=1.0
export CLARIFY_WEIGHT=0.0
export REQUIRED_CLARIFICATION_WEIGHT=0.2
export CLARIFICATION_ATTEMPT_WEIGHT=0.0
export POST_CLARIFICATION_WEIGHT=0.1
export TERMINAL_WEIGHT=0.15
export ADD_COLUMN_WEIGHT=0.15
export ADD_COLUMN_INSPECT_WEIGHT=0.3
export SEMANTIC_CANDIDATE_WEIGHT=0.35
export DECISION_ACTION_WEIGHT=0.3
export DECISION_ACTION_MISMATCH_PENALTY=0.0
export PREMATURE_STALE_EXECUTE_PENALTY=0.5
export VALID_WEIGHT=0.0
export EFFICIENT_WEIGHT=0.05
export TOOL_CALL_COST=0.01
export TOKEN_COST=0.00001
export DUPLICATE_PENALTY=0.08
export REPEATED_TOOL_PENALTY=0.08
export INVALID_PENALTY=0.1
export TIMEOUT_PENALTY=0.25
export TURN_LIMIT_PENALTY=0.5
export MISSING_SUBMIT_PENALTY=0.3
export UNSAFE_PENALTY=1.0
export MISSING_REQUIRED_CLARIFICATION_PENALTY=0.2
export UNMATCHED_CLARIFICATION_PENALTY=0.1
export INVALID_POST_CLARIFICATION_PENALTY=0.1
export ADD_COLUMN_PROTOCOL_PENALTY=0.05
export EFFICIENT_TOOL_CALLS=7

SEED="${SEED:-20260809}"
command=(
  bash "${PROJECT_ROOT}/scripts/train_7b_p6_targeted_grpo.sh"
  "data.shuffle=false"
  "data.seed=${SEED}"
  "actor_rollout_ref.actor.data_loader_seed=${SEED}"
  "actor_rollout_ref.rollout.seed=${SEED}"
)
if [[ -n "${RESUME_FROM}" ]]; then
  command+=(
    "trainer.resume_mode=resume_path"
    "trainer.resume_from_path=${RESUME_FROM}"
  )
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'arm=%s lr=%s kl=%s steps=%s batch=%s rollout_n=%s seed=%s\n' \
    "${ARM}" "${GRPO_LEARNING_RATE}" "${KL_COEF}" "${TOTAL_TRAINING_STEPS}" \
    "${TRAIN_BATCH_SIZE}" "${ROLLOUT_N}" "${SEED}"
  printf 'resume_from=%s\n' "${RESUME_FROM:-none}"
  printf 'decision_reward=%s add_inspect=%s stale_penalty=%s advantage=%s\n' \
    "${DECISION_ACTION_WEIGHT}" "${ADD_COLUMN_INSPECT_WEIGHT}" \
    "${PREMATURE_STALE_EXECUTE_PENALTY}" "${ADVANTAGE_SCOPE}"
  printf 'DRY-RUN '
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

exec "${command[@]}"
