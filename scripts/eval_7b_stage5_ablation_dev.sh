#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ABLATION="${ABLATION:-${1:-}}"
if [[ -z "${ABLATION}" ]]; then
  echo "Usage: ABLATION={no_ask_user|no_hkb|turn_3|turn_5|sparse_reward|replay} $0" >&2
  exit 2
fi
if [[ $# -gt 0 ]]; then
  shift
fi

MAIN_GRPO_ADAPTER_PATH="${MAIN_GRPO_ADAPTER_PATH:-}"
ABLATION_ADAPTER_PATH="${ABLATION_ADAPTER_PATH:-}"
for variable in MAIN_GRPO_ADAPTER_PATH ABLATION_ADAPTER_PATH; do
  path="${!variable:-}"
  if [[ -z "${path}" || ! -f "${path}/adapter_model.safetensors" ]]; then
    echo "Set ${variable} to an exported 7B PEFT adapter." >&2
    exit 1
  fi
done

DATA="${PROJECT_ROOT}/data/processed/stratified_five_tool_v2/dev_agent_eval.jsonl"
MAX_TURNS="${MAX_TURNS:-7}"
EXTRA_ARGS=()
case "${ABLATION}" in
  no_ask_user)
    DATA="${PROJECT_ROOT}/data/processed/stage5_tool_ablations/no_ask_user/dev_agent_eval.jsonl"
    EXTRA_ARGS+=(--disable-tool ask_user)
    ;;
  no_hkb)
    DATA="${PROJECT_ROOT}/data/processed/stage5_tool_ablations/no_hkb/dev_agent_eval.jsonl"
    EXTRA_ARGS+=(--disable-tool get_knowledge_definition)
    ;;
  turn_3)
    MAX_TURNS=3
    ;;
  turn_5)
    MAX_TURNS=5
    ;;
  sparse_reward|replay)
    ;;
  *)
    echo "Unknown ABLATION=${ABLATION}" >&2
    exit 2
    ;;
esac

exec "${PROJECT_ROOT}/.venv/bin/python" "${PROJECT_ROOT}/scripts/run_five_tool_eval.py" \
  --data "${DATA}" \
  --model "${PROJECT_ROOT}/models/Qwen2.5-Coder-7B-Instruct" \
  --output-dir "${OUTPUT_DIR:-${PROJECT_ROOT}/reports/stage5/dev_ablation_${ABLATION}_7b}" \
  --skip-base \
  --adapter-spec "main-shaped-grpo=${MAIN_GRPO_ADAPTER_PATH}" \
  --adapter-spec "ablation-${ABLATION}=${ABLATION_ADAPTER_PATH}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-2}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.65}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --max-turns "${MAX_TURNS}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-512}" \
  --max-model-len "${MAX_MODEL_LEN:-8192}" \
  "${EXTRA_ARGS[@]}" "$@"
