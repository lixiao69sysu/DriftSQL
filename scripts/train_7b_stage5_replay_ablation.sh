#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${REPLAY_MODE:-${1:-}}"
if [[ "${MODE}" != "hard_replay" && "${MODE}" != "uniform_control" ]]; then
  echo "Usage: REPLAY_MODE={hard_replay|uniform_control} $0" >&2
  exit 2
fi
if [[ $# -gt 0 ]]; then
  shift
fi

MAIN_GRPO_ADAPTER_PATH="${MAIN_GRPO_ADAPTER_PATH:-}"
if [[ -z "${MAIN_GRPO_ADAPTER_PATH}" || ! -f "${MAIN_GRPO_ADAPTER_PATH}/adapter_model.safetensors" ]]; then
  echo "Set MAIN_GRPO_ADAPTER_PATH to the selected shaped-GRPO PEFT adapter." >&2
  exit 1
fi

if [[ "${MODE}" == "hard_replay" ]]; then
  export TRAIN_DATA="${PROJECT_ROOT}/data/processed/stage5_failure_replay/mixed_replay.parquet"
else
  export TRAIN_DATA="${PROJECT_ROOT}/data/processed/stage5_failure_replay/uniform_control.parquet"
fi
export VAL_DATA="${PROJECT_ROOT}/data/processed/stratified_five_tool_v2/rl_dev.parquet"
export SFT_ADAPTER_PATH="${MAIN_GRPO_ADAPTER_PATH}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/stage5_replay_${MODE}_7b}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-20}"
export SAVE_FREQ="${SAVE_FREQ:-10}"

# Both arms reset optimizer state and continue for the same number of updates
# from the exact same main-GRPO policy; only sampling distribution differs.
exec bash "${PROJECT_ROOT}/scripts/train_7b_stratified_shaped_grpo.sh" \
  "trainer.experiment_name=stage5-replay-${MODE}-7b" "$@"
