#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${1:-}}"
if [[ -z "${CHECKPOINT_DIR}" || ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "Usage: CHECKPOINT_DIR=checkpoints/.../global_step_N $0" >&2
  exit 2
fi
CHECKPOINT_DIR="$(cd "${CHECKPOINT_DIR}" && pwd)"
LOCAL_DIR="${CHECKPOINT_DIR}"
if [[ -d "${CHECKPOINT_DIR}/actor" ]]; then
  LOCAL_DIR="${CHECKPOINT_DIR}/actor"
fi
if [[ ! -f "${LOCAL_DIR}/lora_train_meta.json" ]]; then
  echo "Not a VERL LoRA checkpoint: ${LOCAL_DIR}" >&2
  exit 1
fi

TARGET_DIR="${TARGET_DIR:-${CHECKPOINT_DIR}/merged}"
ADAPTER_DIR="${TARGET_DIR}/lora_adapter"

"${PROJECT_ROOT}/.venv/bin/python" -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "${LOCAL_DIR}" \
  --target_dir "${TARGET_DIR}"

if [[ ! -f "${ADAPTER_DIR}/adapter_model.safetensors" ]]; then
  # Older VERL exports left LoRA tensors in the HF shard index; current
  # LoRA-only checkpoints are exported directly by model_merger.
  if [[ ! -f "${TARGET_DIR}/model.safetensors.index.json" ]]; then
    echo "model_merger produced neither a PEFT adapter nor an HF shard index" >&2
    exit 1
  fi
  "${PROJECT_ROOT}/.venv/bin/python" "${PROJECT_ROOT}/scripts/export_lora_adapter.py" \
    --checkpoint-hf "${TARGET_DIR}" \
    --output "${ADAPTER_DIR}" \
    --base-model "${PROJECT_ROOT}/models/Qwen2.5-Coder-7B-Instruct" \
    --lora-meta "${LOCAL_DIR}/lora_train_meta.json"
fi

"${PROJECT_ROOT}/.venv/bin/python" -c \
  'import math,sys; from safetensors.torch import load_file; p=sys.argv[1]; x=load_file(p); assert x and all(t.numel() and t.isfinite().all().item() for t in x.values()); print(f"validated_adapter_tensors={len(x)}")' \
  "${ADAPTER_DIR}/adapter_model.safetensors"

echo "Portable adapter: ${ADAPTER_DIR}"
