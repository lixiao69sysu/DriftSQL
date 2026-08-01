#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/Qwen2.5-Coder-7B-Instruct}"
TRAIN_DATA="${TRAIN_DATA:-${PROJECT_ROOT}/data/processed/column_rename/rl_train.parquet}"
VAL_DATA="${VAL_DATA:-${PROJECT_ROOT}/data/processed/column_rename/rl_val.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/grpo_column_rename_smoke}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-${OUTPUT_DIR}/rollouts}"
TOOL_CONFIG="${PROJECT_ROOT}/configs/tools/drift_tools.yaml"
AGENT_LOOP_CONFIG="${PROJECT_ROOT}/configs/agent_loop.yaml"
REWARD_PATH="${PROJECT_ROOT}/driftsql/rewards/agentic.py"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1}"
LORA_ADAPTER_PATH="${LORA_ADAPTER_PATH:-}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
ROLLOUT_N="${ROLLOUT_N:-4}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-10240}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-10240}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
AGENT_WORKERS="${AGENT_WORKERS:-4}"
MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-5}"
MAX_USER_TURNS="${MAX_USER_TURNS:-5}"
SAVE_FREQ="${SAVE_FREQ:-${TOTAL_TRAINING_STEPS}}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
PARAM_OFFLOAD="${PARAM_OFFLOAD:-false}"
OPTIMIZER_OFFLOAD="${OPTIMIZER_OFFLOAD:-false}"
MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"
BYPASS_OLD_LOG_PROB="${BYPASS_OLD_LOG_PROB:-true}"

export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export DRIFTSQL_TMPDIR="${DRIFTSQL_TMPDIR:-${PROJECT_ROOT}/tmp}"
mkdir -p "${DRIFTSQL_TMPDIR}"
export DRIFTSQL_TRAJECTORY_LOG_DIR="${DRIFTSQL_TRAJECTORY_LOG_DIR:-${OUTPUT_DIR}/environment_traces}"
export DRIFTSQL_TRAJECTORY_TIMEOUT="${DRIFTSQL_TRAJECTORY_TIMEOUT:-300}"
export DRIFTSQL_STATE_GUARDS="${DRIFTSQL_STATE_GUARDS:-0}"
export DRIFTSQL_DYNAMIC_TOOL_MASK="${DRIFTSQL_DYNAMIC_TOOL_MASK:-0}"
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=1
# FSDP and the colocated vLLM engine repeatedly hand GPU memory back and forth.
# Keep blocks moderately sized to limit fragmentation. Do not use expandable
# segments here: vLLM's sleep-mode memory pool explicitly rejects them.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-max_split_size_mb:128}"

MODEL_ARGS=()
if [[ -n "${LORA_ADAPTER_PATH}" ]]; then
  MODEL_ARGS+=("actor_rollout_ref.model.lora_adapter_path=${LORA_ADAPTER_PATH}")
fi

"${PYTHON}" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=false \
  algorithm.rollout_correction.bypass_mode="${BYPASS_OLD_LOG_PROB}" \
  data.train_files="${TRAIN_DATA}" \
  data.val_files="${VAL_DATA}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=true \
  data.filter_overlong_prompts_workers=1 \
  data.dataloader_num_workers="${DATALOADER_NUM_WORKERS}" \
  data.truncation=error \
  data.return_raw_chat=true \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.lora_rank=32 \
  actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=true \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload="${PARAM_OFFLOAD}" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${OPTIMIZER_OFFLOAD}" \
  actor_rollout_ref.actor.fsdp_config.model_dtype="${MODEL_DTYPE}" \
  actor_rollout_ref.actor.fsdp_config.use_torch_compile=false \
  actor_rollout_ref.actor.checkpoint.save_contents='["model","optimizer","extra"]' \
  +actor_rollout_ref.actor.checkpoint.save_lora_only=true \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.free_cache_engine=true \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=true \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS}" \
  actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS}" \
  actor_rollout_ref.rollout.multi_turn.enable=true \
  actor_rollout_ref.rollout.multi_turn.format=driftsql-json \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${MAX_ASSISTANT_TURNS}" \
  actor_rollout_ref.rollout.multi_turn.max_user_turns="${MAX_USER_TURNS}" \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1024 \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG}" \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_LOOP_CONFIG}" \
  actor_rollout_ref.rollout.agent.default_agent_loop=driftsql_tool_agent \
  actor_rollout_ref.rollout.agent.num_workers="${AGENT_WORKERS}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  actor_rollout_ref.ref.fsdp_config.model_dtype="${MODEL_DTYPE}" \
  reward.custom_reward_function.path="${REWARD_PATH}" \
  reward.custom_reward_function.name=compute_score \
  reward.reward_manager.name=naive \
  trainer.project_name=driftsql-rl \
  trainer.experiment_name=column-rename-grpo-smoke \
  trainer.logger='["console"]' \
  trainer.n_gpus_per_node="${NPROC_PER_NODE}" \
  trainer.nnodes=1 \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  trainer.total_epochs=1 \
  trainer.val_before_train=false \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq=-1 \
  trainer.resume_mode=disable \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
  "${MODEL_ARGS[@]}" \
  "$@"
