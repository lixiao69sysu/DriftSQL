#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/reports/p6_scaleup/on_policy_seed101_supplement500"
LOG_PATH="${PROJECT_ROOT}/reports/p6_scaleup/on_policy_seed101_supplement500.log"
RESULT_PATH="${OUTPUT_DIR}/strong-sft.jsonl"
PARTIAL_PATH="${OUTPUT_DIR}/.strong-sft.partial.jsonl"
RESUME_ARGS=()

if [[ -f "${RESULT_PATH}" ]]; then
  lines="$(wc -l <"${RESULT_PATH}")"
  if [[ "${lines}" == "500" ]]; then
    echo "Seed101 supplement already complete: 500 rollouts"
    exit 0
  fi
  echo "Unexpected completed result length: ${lines}/500" >&2
  exit 1
fi

if [[ -f "${PARTIAL_PATH}" ]]; then
  partial_lines="$(wc -l <"${PARTIAL_PATH}")"
  echo "Resuming seed101 supplement from ${partial_lines}/500"
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
    --data "${PROJECT_ROOT}/data/processed/p6_scaleup_v1_rollout_pool600/train_agent_eval.jsonl" \
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
    --seed 101 \
    --limit 500 \
    "${RESUME_ARGS[@]}" \
    >>"${LOG_PATH}" 2>&1

lines="$(wc -l <"${RESULT_PATH}")"
if [[ "${lines}" != "500" ]]; then
  echo "Seed101 supplement ended with ${lines}/500 rows" >&2
  exit 1
fi
echo "Seed101 supplement completed: 500 rollouts"
