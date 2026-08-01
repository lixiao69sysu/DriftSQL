#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ABLATION="${ABLATION:-${1:-}}"
if [[ -z "${ABLATION}" ]]; then
  echo "Usage: ABLATION={no_ask_user|no_hkb|turn_3|turn_5|sparse_reward} $0" >&2
  exit 2
fi
if [[ $# -gt 0 ]]; then
  shift
fi

case "${ABLATION}" in
  no_ask_user)
    export TRAIN_DATA="${PROJECT_ROOT}/data/processed/stage5_tool_ablations/no_ask_user/rl_train.parquet"
    export VAL_DATA="${PROJECT_ROOT}/data/processed/stage5_tool_ablations/no_ask_user/rl_dev.parquet"
    ;;
  no_hkb)
    export TRAIN_DATA="${PROJECT_ROOT}/data/processed/stage5_tool_ablations/no_hkb/rl_train.parquet"
    export VAL_DATA="${PROJECT_ROOT}/data/processed/stage5_tool_ablations/no_hkb/rl_dev.parquet"
    ;;
  turn_3)
    export MAX_ASSISTANT_TURNS=3
    export MAX_USER_TURNS=3
    ;;
  turn_5)
    export MAX_ASSISTANT_TURNS=5
    export MAX_USER_TURNS=5
    ;;
  sparse_reward)
    export CLARIFY_WEIGHT=0.0
    export VALID_WEIGHT=0.0
    export EFFICIENT_WEIGHT=0.0
    export TOOL_CALL_COST=0.0
    export TOKEN_COST=0.0
    export DUPLICATE_PENALTY=0.0
    export REPEATED_TOOL_PENALTY=0.0
    export INVALID_PENALTY=0.0
    export TIMEOUT_PENALTY=0.0
    export TURN_LIMIT_PENALTY=0.0
    export MISSING_SUBMIT_PENALTY=0.0
    ;;
  *)
    echo "Unknown ABLATION=${ABLATION}" >&2
    exit 2
    ;;
esac

export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/stage5_ablation_${ABLATION}_7b}"

OVERRIDES=("trainer.experiment_name=stage5-ablation-${ABLATION}-7b")
# ``sparse_reward`` keeps task success (+1) and the unsafe-action invariant
# (-1), while every shaping and efficiency term is zeroed above.

exec bash "${PROJECT_ROOT}/scripts/train_7b_stratified_shaped_grpo.sh" \
  "${OVERRIDES[@]}" "$@"
