#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/Qwen2.5-Coder-7B-Instruct}"
ADAPTER_PATH="${ADAPTER_PATH:-${PROJECT_ROOT}/checkpoints/stage5_ablation_no_ask_user_7b/global_step_40/merged/lora_adapter}"

export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/third_party/verl${PYTHONPATH:+:${PYTHONPATH}}"

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" scripts/run_stage6_eval.py \
  --data data/processed/stage6_ablation/b0/tune_agent_eval.jsonl \
  --model "${MODEL_PATH}" \
  --output-dir reports/stage6/b0_tune \
  --adapter-spec "b0=${ADAPTER_PATH}" \
  --skip-base \
  --disable-tool get_schema_version \
  --disable-tool inspect_schema_diff \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.65 \
  --batch-size 32 \
  --max-turns 7

"${PYTHON_BIN}" scripts/run_stage6_eval.py \
  --data data/processed/stage6_ablation/b1/tune_agent_eval.jsonl \
  --model "${MODEL_PATH}" \
  --output-dir reports/stage6/b1_tune \
  --adapter-spec "b1=${ADAPTER_PATH}" \
  --skip-base \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.65 \
  --batch-size 32 \
  --max-turns 7
