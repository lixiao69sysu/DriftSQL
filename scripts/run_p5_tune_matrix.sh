#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${P5_TUNE_DATA:-${PROJECT_ROOT}/data/processed/p5_grpo/tune_agent_eval.jsonl}"
REPORT_ROOT="${P5_TUNE_REPORT_ROOT:-${PROJECT_ROOT}/reports/p5/tune}"
SFT20="${SFT20_ADAPTER_PATH:-${PROJECT_ROOT}/checkpoints/stage8_fresh_sft_7b/global_step_20/merged/lora_adapter}"
GRPO_ROOT="${P5_GRPO_ROOT:-${PROJECT_ROOT}/checkpoints/p5_reviewed_grpo_7b}"
GPUS="${CUDA_VISIBLE_DEVICES:-0,2,3}"

if [[ ! -f "${DATA}" ]]; then
  echo "P5 Tune evaluator data not found: ${DATA}" >&2
  exit 1
fi

ALIASES=(p5-sft20 p5-grpo-step5 p5-grpo-step10)
ADAPTERS=(
  "${SFT20}"
  "${GRPO_ROOT}/global_step_5/merged/lora_adapter"
  "${GRPO_ROOT}/global_step_10/merged/lora_adapter"
)

for index in "${!ALIASES[@]}"; do
  alias="${ALIASES[index]}"
  adapter="${ADAPTERS[index]}"
  if [[ ! -f "${adapter}/adapter_model.safetensors" ]]; then
    echo "Portable adapter missing for ${alias}: ${adapter}" >&2
    echo "Export the checkpoint with scripts/export_stage5_lora_checkpoint.sh first." >&2
    exit 1
  fi
  "${PROJECT_ROOT}/.venv/bin/python" "${PROJECT_ROOT}/scripts/run_stage7_process_isolated_eval.py" \
    --data "${DATA}" \
    --output-dir "${REPORT_ROOT}/${alias}" \
    --adapter-alias "${alias}" \
    --adapter-path "${adapter}" \
    --drift-type add_column \
    --gpus "${GPUS}" \
    --max-turns "${MAX_TURNS:-7}" \
    --max-new-tokens "${MAX_NEW_TOKENS:-512}" \
    --max-model-len "${MAX_MODEL_LEN:-8192}" \
    --resume
done

