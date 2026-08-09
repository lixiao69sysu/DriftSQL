#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
DATA_PATH="${PROJECT_ROOT}/data/processed/p6_scaleup_v1_rollout_pool600/train_agent_eval.jsonl"
MODEL_PATH="${PROJECT_ROOT}/models/Qwen2.5-Coder-7B-Instruct"
ADAPTER_PATH="${PROJECT_ROOT}/checkpoints/p6_on_policy_recovery_sft_round2_mixed_7b/global_step_10/merged/lora_adapter"
REPORT_ROOT="${PROJECT_ROOT}/reports/p6_scaleup"

GPUS=(0 1 2 3)
SEEDS=(101 211 307 401)
PIDS=()
RUN_GPUS=()
RUN_SEEDS=()
COMPLETED_SEEDS=()

for required in \
  "${PYTHON_BIN}" \
  "${DATA_PATH}" \
  "${MODEL_PATH}/config.json" \
  "${ADAPTER_PATH}/adapter_model.safetensors"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required input: ${required}" >&2
    exit 1
  fi
done

mkdir -p "${REPORT_ROOT}"

for index in "${!GPUS[@]}"; do
  gpu="${GPUS[$index]}"
  seed="${SEEDS[$index]}"
  output_dir="${REPORT_ROOT}/on_policy_seed${seed}"
  log_path="${REPORT_ROOT}/on_policy_seed${seed}.log"
  result_path="${output_dir}/strong-sft.jsonl"
  summary_path="${output_dir}/summary.json"
  if [[ -f "${result_path}" && -f "${summary_path}" ]]; then
    lines="$(wc -l <"${result_path}")"
    if [[ "${lines}" == "600" ]]; then
      echo "Reusing completed seed ${seed}: ${lines} rollouts"
      COMPLETED_SEEDS+=("${seed}")
      continue
    fi
    echo "Incomplete existing result for seed ${seed}: ${lines}/600" >&2
    exit 1
  fi
  if [[ -e "${output_dir}" || -e "${log_path}" ]]; then
    echo "Refusing to overwrite incomplete output: ${output_dir} or ${log_path}" >&2
    exit 1
  fi
done

cleanup() {
  if ((${#PIDS[@]})); then
    kill "${PIDS[@]}" 2>/dev/null || true
  fi
}
trap cleanup INT TERM

for index in "${!GPUS[@]}"; do
  gpu="${GPUS[$index]}"
  seed="${SEEDS[$index]}"
  output_dir="${REPORT_ROOT}/on_policy_seed${seed}"
  log_path="${REPORT_ROOT}/on_policy_seed${seed}.log"
  if [[ " ${COMPLETED_SEEDS[*]} " == *" ${seed} "* ]]; then
    continue
  fi

  echo "Launching GPU ${gpu}, seed ${seed}, output ${output_dir}"
  env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/third_party/verl${PYTHONPATH:+:${PYTHONPATH}}" \
    TOKENIZERS_PARALLELISM=false \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    DRIFTSQL_REWARD_TIMEOUT=20 \
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/run_p6_generalized_eval.py" \
      --data "${DATA_PATH}" \
      --model "${MODEL_PATH}" \
      --output-dir "${output_dir}" \
      --adapter-spec "strong-sft=${ADAPTER_PATH}" \
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
      --seed "${seed}" \
      >"${log_path}" 2>&1 &
  PIDS+=("$!")
  RUN_GPUS+=("${gpu}")
  RUN_SEEDS+=("${seed}")
done

echo "Started PIDs: ${PIDS[*]}"
failed=0
for index in "${!PIDS[@]}"; do
  if wait "${PIDS[$index]}"; then
    echo "GPU ${RUN_GPUS[$index]} seed ${RUN_SEEDS[$index]} completed"
  else
    status=$?
    echo "GPU ${RUN_GPUS[$index]} seed ${RUN_SEEDS[$index]} failed with status ${status}" >&2
    failed=1
  fi
done

trap - INT TERM
if ((failed)); then
  exit 1
fi
echo "All four seeds completed: 2400 on-policy rollouts."
