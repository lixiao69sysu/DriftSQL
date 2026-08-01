#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_ENV="${PROJECT_ROOT}/.venv"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/Qwen2.5-Coder-7B-Instruct}"
TRAIN_DATA="${TRAIN_DATA:-${PROJECT_ROOT}/data/processed/column_rename/sft_train.parquet}"
VAL_DATA="${VAL_DATA:-${PROJECT_ROOT}/data/processed/column_rename/sft_val.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/sft_column_rename_smoke}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-5}"
SAVE_CONTENTS="${SAVE_CONTENTS:-[\"model\",\"extra\"]}"
SAVE_LORA_ONLY="${SAVE_LORA_ONLY:-true}"

export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TOKENIZERS_PARALLELISM=false

"${PYTHON_ENV}/bin/python" -m torch.distributed.run \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  -m verl.trainer.sft_trainer \
  data.train_files="${TRAIN_DATA}" \
  data.val_files="${VAL_DATA}" \
  data.messages_key=messages \
  data.tools_key=tools \
  data.enable_thinking_key=enable_thinking \
  data.train_batch_size=8 \
  data.micro_batch_size_per_gpu=1 \
  data.max_token_len_per_gpu=4096 \
  data.max_length=4096 \
  data.use_dynamic_bsz=true \
  data.truncation=error \
  model.path="${MODEL_PATH}" \
  model.use_remove_padding=true \
  model.enable_gradient_checkpointing=true \
  +model.override_config.attn_implementation=sdpa \
  model.lora_rank=32 \
  model.lora_alpha=32 \
  model.target_modules=all-linear \
  optim.lr=2e-5 \
  engine=fsdp \
  engine.model_dtype=bfloat16 \
  engine.use_torch_compile=false \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.project_name=driftsql-rl \
  trainer.experiment_name=column-rename-oracle-sft-smoke \
  trainer.logger='["console"]' \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  trainer.save_freq="${TOTAL_TRAINING_STEPS}" \
  trainer.test_freq="${TOTAL_TRAINING_STEPS}" \
  trainer.resume_mode=disable \
  checkpoint.save_contents="${SAVE_CONTENTS}" \
  +checkpoint.save_lora_only="${SAVE_LORA_ONLY}" \
  "$@"
