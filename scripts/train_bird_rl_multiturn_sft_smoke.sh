#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/Qwen2.5-Coder-7B-Instruct}"
TRAIN_DATA="${TRAIN_DATA:-${PROJECT_ROOT}/data/processed/bird_rl_baseline/multiturn_sft/train.parquet}"
VAL_DATA="${VAL_DATA:-${PROJECT_ROOT}/data/processed/bird_rl_baseline/multiturn_sft/val.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/bird_rl_multiturn_sft_smoke}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-32}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"

export PYTHONPATH="${PROJECT_ROOT}/third_party/BIRD-RL:${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TOKENIZERS_PARALLELISM=false

"${PYTHON}" -m torch.distributed.run \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  -m verl.trainer.sft_trainer \
  data.train_files="${TRAIN_DATA}" \
  data.val_files="${VAL_DATA}" \
  data.messages_key=messages \
  data.train_batch_size=4 \
  data.micro_batch_size_per_gpu=1 \
  data.max_token_len_per_gpu=3072 \
  data.max_length=3072 \
  data.use_dynamic_bsz=true \
  data.truncation=error \
  model.path="${MODEL_PATH}" \
  model.use_remove_padding=true \
  model.enable_gradient_checkpointing=true \
  +model.override_config.attn_implementation=sdpa \
  model.lora_rank=32 \
  model.lora_alpha=32 \
  model.target_modules=all-linear \
  optim.lr="${LEARNING_RATE}" \
  engine=fsdp \
  engine.model_dtype=bfloat16 \
  engine.use_torch_compile=false \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.project_name=driftsql-rl-baselines \
  trainer.experiment_name=bird-rl-multiturn-oracle-sft-smoke \
  trainer.logger='["console"]' \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  trainer.save_freq="${TOTAL_TRAINING_STEPS}" \
  trainer.test_freq="${TOTAL_TRAINING_STEPS}" \
  trainer.resume_mode=disable \
  checkpoint.save_contents='["model","extra"]' \
  +checkpoint.save_lora_only=true \
  "$@"
