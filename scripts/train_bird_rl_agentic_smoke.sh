#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/Qwen2.5-Coder-7B-Instruct}"
TRAIN_DATA="${TRAIN_DATA:-${PROJECT_ROOT}/data/processed/bird_rl_baseline/agentic_rl/train.parquet}"
VAL_DATA="${VAL_DATA:-${PROJECT_ROOT}/data/processed/bird_rl_baseline/agentic_rl/val.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/bird_rl_agentic_grpo_smoke}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.5}"
LORA_ADAPTER_PATH="${LORA_ADAPTER_PATH:-}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-2}"
ROLLOUT_N="${ROLLOUT_N:-2}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-768}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-3072}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-3072}"

export PYTHONPATH="${PROJECT_ROOT}/third_party/BIRD-RL:${PROJECT_ROOT}:${PYTHONPATH:-}"
export SQL_REWARD_DB_DIR="${SQL_REWARD_DB_DIR:-${PROJECT_ROOT}/data/raw/six-gym-sqlite/database}"
export SQL_EXECUTION_TIMEOUT="${SQL_EXECUTION_TIMEOUT:-30}"
export TRAJECTORY_TIMEOUT="${TRAJECTORY_TIMEOUT:-180}"
export BIRD_RL_POOL_DB_IDS="${BIRD_RL_POOL_DB_IDS:-book_publishing_company}"
export BIRD_RL_NUM_ACTORS="${BIRD_RL_NUM_ACTORS:-2}"
export BIRD_RL_POOL_COPIES_PER_ACTOR="${BIRD_RL_POOL_COPIES_PER_ACTOR:-2}"
export BIRD_RL_REWARD_POOL_START="${BIRD_RL_REWARD_POOL_START:-4}"
export BIRD_RL_REWARD_POOL_SIZE="${BIRD_RL_REWARD_POOL_SIZE:-2}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=1

MODEL_ARGS=()
if [[ -n "${LORA_ADAPTER_PATH}" ]]; then
  MODEL_ARGS+=("actor_rollout_ref.model.lora_adapter_path=${LORA_ADAPTER_PATH}")
fi

"${PYTHON}" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=false \
  algorithm.rollout_correction.bypass_mode=true \
  data.train_files="${TRAIN_DATA}" \
  data.val_files="${VAL_DATA}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=true \
  data.filter_overlong_prompts_workers=1 \
  data.dataloader_num_workers=0 \
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
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.actor.fsdp_config.use_torch_compile=false \
  actor_rollout_ref.actor.checkpoint.save_contents='["model","extra"]' \
  +actor_rollout_ref.actor.checkpoint.save_lora_only=true \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.free_cache_engine=true \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=true \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
  actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.max_num_seqs=16 \
  actor_rollout_ref.rollout.multi_turn.enable=true \
  actor_rollout_ref.rollout.multi_turn.format=bird-json-compat \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5 \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=5 \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1024 \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="${PROJECT_ROOT}/third_party/BIRD-RL/configs/tools/critic_tools.yaml" \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${PROJECT_ROOT}/configs/bird_rl/agent_loop.yaml" \
  actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent_with_db_cleanup \
  actor_rollout_ref.rollout.agent.num_workers=2 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
  reward.custom_reward_function.path="${PROJECT_ROOT}/driftsql/rewards/bird_compat.py" \
  reward.custom_reward_function.name=compute_score \
  reward.reward_manager.name=naive \
  trainer.project_name=driftsql-rl-baselines \
  trainer.experiment_name=bird-rl-agentic-grpo-smoke \
  trainer.logger='["console"]' \
  trainer.n_gpus_per_node="${NPROC_PER_NODE}" \
  trainer.nnodes=1 \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  trainer.total_epochs=1 \
  trainer.val_before_train=false \
  trainer.save_freq="${TOTAL_TRAINING_STEPS}" \
  trainer.test_freq=-1 \
  trainer.resume_mode=disable \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.rollout_data_dir="${OUTPUT_DIR}/rollouts" \
  "${MODEL_ARGS[@]}" \
  "$@"
