#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${PROJECT_ROOT}/reports/stage5/final_selection/manifest.json"
OUTPUT_DIR="${PROJECT_ROOT}/reports/stage5/final_sealed_eval"
RAW_DIR="${OUTPUT_DIR}/raw"
DATA_DIR="${PROJECT_ROOT}/data/processed/stage5_final_sealed/no_ask_user"
ADAPTER="${PROJECT_ROOT}/checkpoints/stage5_ablation_no_ask_user_7b/global_step_40/merged/lora_adapter"
EXPECTED_ADAPTER_SHA256="99e66c93298654fc7424a5c2d1636b0b1915e495af5f9706c21635e4ed18ca28"
EXPECTED_EVALUATOR_SHA256="836a6a4bc38c14118a11656ee7eec5d4237cc24e001d17757cb9e783079ad38b"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Final selection manifest is missing: ${MANIFEST}" >&2
  exit 1
fi
if [[ -e "${OUTPUT_DIR}/RUN_STARTED" || -e "${OUTPUT_DIR}/RUN_COMPLETE" ]]; then
  echo "Sealed evaluation was already started; refusing a second pass." >&2
  exit 1
fi
if [[ ! -f "${ADAPTER}/adapter_model.safetensors" ]]; then
  echo "Frozen adapter is missing: ${ADAPTER}" >&2
  exit 1
fi
if [[ "$(sha256sum "${ADAPTER}/adapter_model.safetensors" | cut -d' ' -f1)" != "${EXPECTED_ADAPTER_SHA256}" ]]; then
  echo "Frozen adapter hash mismatch" >&2
  exit 1
fi
if [[ "$(sha256sum "${PROJECT_ROOT}/scripts/run_five_tool_eval.py" | cut -d' ' -f1)" != "${EXPECTED_EVALUATOR_SHA256}" ]]; then
  echo "Frozen evaluator hash mismatch" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
touch "${OUTPUT_DIR}/RUN_STARTED"

"${PROJECT_ROOT}/.venv/bin/python" "${PROJECT_ROOT}/scripts/prepare_stage5_final_eval.py"

"${PROJECT_ROOT}/.venv/bin/python" "${PROJECT_ROOT}/scripts/run_five_tool_eval.py" \
  --data "${DATA_DIR}/test_agent_eval.jsonl" \
  --model "${PROJECT_ROOT}/models/Qwen2.5-Coder-7B-Instruct" \
  --output-dir "${RAW_DIR}" \
  --skip-base \
  --adapter-spec "final-no-ask-user=${ADAPTER}" \
  --disable-tool ask_user \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.65 \
  --batch-size 16 \
  --max-turns 7 \
  --max-new-tokens 512 \
  --max-model-len 8192

"${PROJECT_ROOT}/.venv/bin/python" "${PROJECT_ROOT}/scripts/summarize_stage5_final_eval.py" \
  --test-results "${RAW_DIR}/final-no-ask-user.jsonl" \
  --frozen-data "${DATA_DIR}/frozen_regression_78_agent_eval.jsonl" \
  --output-dir "${OUTPUT_DIR}"

touch "${OUTPUT_DIR}/RUN_COMPLETE"
echo "Final sealed evaluation complete: ${OUTPUT_DIR}/sealed_summary.json"
