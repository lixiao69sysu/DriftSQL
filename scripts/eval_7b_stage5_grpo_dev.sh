#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
V2_SFT_ADAPTER_PATH="${V2_SFT_ADAPTER_PATH:-}"
GRPO_ADAPTER_PATH="${GRPO_ADAPTER_PATH:-}"
for variable in V2_SFT_ADAPTER_PATH GRPO_ADAPTER_PATH; do
  path="${!variable:-}"
  if [[ -z "${path}" || ! -f "${path}/adapter_model.safetensors" ]]; then
    echo "Set ${variable} to an exported 7B PEFT adapter." >&2
    exit 1
  fi
done

exec "${PROJECT_ROOT}/.venv/bin/python" "${PROJECT_ROOT}/scripts/run_five_tool_eval.py" \
  --data "${PROJECT_ROOT}/data/processed/stratified_five_tool_v2/dev_agent_eval.jsonl" \
  --model "${PROJECT_ROOT}/models/Qwen2.5-Coder-7B-Instruct" \
  --output-dir "${OUTPUT_DIR:-${PROJECT_ROOT}/reports/stage5/dev_shaped_grpo_7b}" \
  --adapter-spec "v2-7b-sft=${V2_SFT_ADAPTER_PATH}" \
  --adapter-spec "shaped-grpo-7b=${GRPO_ADAPTER_PATH}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-2}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.65}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --max-turns "${MAX_TURNS:-7}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-512}" \
  --max-model-len "${MAX_MODEL_LEN:-8192}"
