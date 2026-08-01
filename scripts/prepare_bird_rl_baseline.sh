#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
RAW_DIR="${PROJECT_ROOT}/data/processed/bird_rl_baseline/raw"
SINGLE_DIR="${PROJECT_ROOT}/data/processed/bird_rl_baseline/reasoning_rl"
AGENTIC_DIR="${PROJECT_ROOT}/data/processed/bird_rl_baseline/agentic_rl"
SFT_DIR="${PROJECT_ROOT}/data/processed/bird_rl_baseline/multiturn_sft"

export PYTHONPATH="${PROJECT_ROOT}/third_party/BIRD-RL:${PROJECT_ROOT}:${PYTHONPATH:-}"

"${PYTHON}" "${PROJECT_ROOT}/scripts/prepare_bird_rl_baseline_data.py"

"${PYTHON}" -m bird_rl.data.prepare_reasoning_rl_data \
  --data "${RAW_DIR}/train.jsonl" \
  --schema "${RAW_DIR}/train_schema.jsonl" \
  --output "${SINGLE_DIR}/train.parquet" \
  --data-source bird_critic/sqlite_train \
  --split train

"${PYTHON}" -m bird_rl.data.prepare_reasoning_rl_data \
  --data "${RAW_DIR}/val.jsonl" \
  --schema "${RAW_DIR}/val_schema.jsonl" \
  --output "${SINGLE_DIR}/val.parquet" \
  --data-source bird_critic/sqlite_dev \
  --split dev

"${PYTHON}" -m bird_rl.data.prepare_agentic_rl_data \
  --data "${RAW_DIR}/train.jsonl" \
  --schema "${RAW_DIR}/train_schema.jsonl" \
  --output "${AGENTIC_DIR}/train.parquet" \
  --data-source bird_critic/sqlite_train \
  --split train \
  --max-turns 5

"${PYTHON}" -m bird_rl.data.prepare_agentic_rl_data \
  --data "${RAW_DIR}/val.jsonl" \
  --schema "${RAW_DIR}/val_schema.jsonl" \
  --output "${AGENTIC_DIR}/val.parquet" \
  --data-source bird_critic/sqlite_dev \
  --split dev \
  --max-turns 5

for SPLIT in train val; do
  "${PYTHON}" "${PROJECT_ROOT}/scripts/prepare_bird_rl_oracle_sft.py" \
    --data "${RAW_DIR}/${SPLIT}.jsonl" \
    --status-output "${SFT_DIR}/${SPLIT}_status.jsonl" \
    --trajectory-output "${SFT_DIR}/${SPLIT}_trajectory.jsonl"

  "${PYTHON}" -m bird_rl.data.prepare_multi_turn_sft_data \
    --status-file "${SFT_DIR}/${SPLIT}_status.jsonl" \
    --trajectory-file "${SFT_DIR}/${SPLIT}_trajectory.jsonl" \
    --train-data "${RAW_DIR}/${SPLIT}.jsonl" \
    --schema-data "${RAW_DIR}/${SPLIT}_schema.jsonl" \
    --output-path "${SFT_DIR}/${SPLIT}.parquet" \
    --max-turns 5 \
    --use-think-tags
done

"${PYTHON}" "${PROJECT_ROOT}/scripts/select_bird_rl_smoke_subset.py" \
  --input "${AGENTIC_DIR}/train.parquet" \
  --output "${AGENTIC_DIR}/smoke_train.parquet" \
  --instance-id TRAIN_100 \
  --instance-id TRAIN_1014
