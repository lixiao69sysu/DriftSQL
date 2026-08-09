#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_PATH="${PROJECT_ROOT}/data/processed/p6_scaleup_v1_targeted_supplement100/train_agent_eval.jsonl"
OUTPUT_DIR="${PROJECT_ROOT}/reports/p6_scaleup/on_policy_seed503_targeted100"
LOG_PATH="${PROJECT_ROOT}/reports/p6_scaleup/on_policy_seed503_targeted100.log"
RESULT_PATH="${OUTPUT_DIR}/strong-sft.jsonl"
PARTIAL_PATH="${OUTPUT_DIR}/.strong-sft.partial.jsonl"
EXPECTED_ROWS=100
RESUME_ARGS=()

if [[ ! -f "${DATA_PATH}" ]]; then
  echo "Missing targeted supplement dataset: ${DATA_PATH}" >&2
  exit 1
fi

input_lines="$(wc -l <"${DATA_PATH}")"
if [[ "${input_lines}" != "${EXPECTED_ROWS}" ]]; then
  echo "Unexpected targeted dataset length: ${input_lines}/${EXPECTED_ROWS}" >&2
  exit 1
fi

if [[ -f "${RESULT_PATH}" ]]; then
  lines="$(wc -l <"${RESULT_PATH}")"
  if [[ "${lines}" == "${EXPECTED_ROWS}" ]]; then
    echo "Seed503 targeted supplement already complete: ${EXPECTED_ROWS} rollouts"
    exit 0
  fi
  echo "Unexpected completed result length: ${lines}/${EXPECTED_ROWS}" >&2
  exit 1
fi

if [[ -f "${PARTIAL_PATH}" ]]; then
  partial_lines="$(wc -l <"${PARTIAL_PATH}")"
  echo "Resuming seed503 targeted supplement from ${partial_lines}/${EXPECTED_ROWS}"
  RESUME_ARGS+=(--resume-partial)
elif [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Output directory exists without a resumable checkpoint: ${OUTPUT_DIR}" >&2
  exit 1
fi

mkdir -p "$(dirname "${LOG_PATH}")"
env \
  CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/third_party/verl${PYTHONPATH:+:${PYTHONPATH}}" \
  TOKENIZERS_PARALLELISM=false \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  DRIFTSQL_REWARD_TIMEOUT=20 \
  "${PROJECT_ROOT}/.venv/bin/python" "${PROJECT_ROOT}/scripts/run_p6_generalized_eval.py" \
    --data "${DATA_PATH}" \
    --model "${PROJECT_ROOT}/models/Qwen2.5-Coder-7B-Instruct" \
    --output-dir "${OUTPUT_DIR}" \
    --adapter-spec "strong-sft=${PROJECT_ROOT}/checkpoints/p6_on_policy_recovery_sft_round2_mixed_7b/global_step_10/merged/lora_adapter" \
    --skip-base \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.82 \
    --batch-size 1 \
    --max-turns 7 \
    --max-new-tokens 512 \
    --max-model-len 8192 \
    --disable-async-scheduling \
    --disable-prefix-caching \
    --episode-major \
    --state-guards \
    --dynamic-tool-mask \
    --constrained-tool-names \
    --knowledge-first-after-ask \
    --temperature 0.6 \
    --top-p 0.95 \
    --seed 503 \
    --limit "${EXPECTED_ROWS}" \
    "${RESUME_ARGS[@]}" \
    >>"${LOG_PATH}" 2>&1

lines="$(wc -l <"${RESULT_PATH}")"
if [[ "${lines}" != "${EXPECTED_ROWS}" ]]; then
  echo "Seed503 targeted supplement ended with ${lines}/${EXPECTED_ROWS} rows" >&2
  exit 1
fi
echo "Seed503 targeted supplement completed: ${EXPECTED_ROWS} rollouts"
