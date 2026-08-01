#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OLD_ADAPTER_PATH="${OLD_ADAPTER_PATH:-${PROJECT_ROOT}/checkpoints/sft_schema_drift_7b/global_step_80/merged/lora_adapter}"
V2_ADAPTER_PATH="${V2_ADAPTER_PATH:-}"

if [[ ! -f "${OLD_ADAPTER_PATH}/adapter_model.safetensors" ]]; then
  echo "Old 7B SFT adapter not found: ${OLD_ADAPTER_PATH}" >&2
  exit 1
fi
if [[ -z "${V2_ADAPTER_PATH}" || ! -f "${V2_ADAPTER_PATH}/adapter_model.safetensors" ]]; then
  echo "Set V2_ADAPTER_PATH to an exported Dataset V2 7B PEFT adapter." >&2
  exit 1
fi

exec "${PROJECT_ROOT}/.venv/bin/python" "${PROJECT_ROOT}/scripts/run_five_tool_eval.py" \
  --data "${PROJECT_ROOT}/data/processed/stratified_five_tool_v2/dev_agent_eval.jsonl" \
  --model "${PROJECT_ROOT}/models/Qwen2.5-Coder-7B-Instruct" \
  --output-dir "${OUTPUT_DIR:-${PROJECT_ROOT}/reports/stage5/dev_sft_7b}" \
  --adapter-spec "old-7b-sft=${OLD_ADAPTER_PATH}" \
  --adapter-spec "v2-7b-sft=${V2_ADAPTER_PATH}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-2}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.65}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --max-turns "${MAX_TURNS:-7}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-512}" \
  --max-model-len "${MAX_MODEL_LEN:-8192}"
