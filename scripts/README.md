# Maintained scripts

The scripts directory contains the active DriftSQL build, training, evaluation
and product paths. Historical P5/Stage5-8, 3B smoke, Terminal-action and failed
Reward A/B launchers were removed after their conclusions were consolidated in
the P6 experiment retrospective.

## Product entry points

- `run_cli.sh`: start the interactive terminal client.
- `serve_service.sh`: start the persistent FastAPI/vLLM service.
- `smoke_product_service.py`: run the real product-service acceptance smoke.
- `download_translation_model.py`: fetch the optional Chinese-to-English model.

## Dataset and on-policy failure pipeline

Run these in order. Most defaults point at the retained canonical locations;
the Recovery SFT builder deliberately requires its Train-only inputs:

1. `build_p6_scaleup_protocol.py`
2. `prepare_p6_scaleup_low_write.py`
3. `build_p6_rollout_pool.py`
4. `run_p6_scaleup_on_policy_4gpu.sh`, followed by the retained supplement
   launchers when reproducing the exact 2,400-rollout pool
5. `mine_p6_scaleup_failures.py`
6. `build_p6_on_policy_recovery_sft.py`
7. `build_p6_scaleup_hard_replay.py`
8. `build_p6_scaleup_sft_mix.py` and `build_p6_scaleup_grpo.py`
9. `validate_p6_scaleup_training_data.py`

The exact Recovery SFT invocation is:

```bash
.venv/bin/python scripts/build_p6_on_policy_recovery_sft.py \
  --rollouts data/processed/p6_scaleup_v1_on_policy_failures/failures.jsonl \
  --agent-records data/processed/p6_scaleup_v1_rollout_pool600/train_agent_eval.jsonl \
  --canonical-trajectories data/processed/p6_scaleup_v1_low_write_protocol/train_trajectories.parquet \
  --output-dir data/processed/p6_scaleup_v1_recovery_sft \
  --curriculum-stage mixed --max-tokens 6144
```

`build_stage7_add_column_protocol.py`, `split_stage6_train_tune_gate.py` and
`prepare_p6_generalized_protocol.py` remain because the current Scale-up
builders import their database-splitting and verified-trajectory helpers.

## Maintained training path

- `train_sft_smoke.sh` and `train_grpo_smoke.sh`: shared VERL launchers; their
  names are historical, but current 7B wrappers depend on them.
- `train_7b_p6_scaleup_sft.sh`: Recovery + Hard Replay SFT160.
- `build_p6_focus1000.py`, `build_p6_focus1000_coverage_order.py` and
  `build_p6_first_action_focus1000.py`: audited Focus200 GRPO curriculum.
- `train_7b_p6_first_action_grpo.sh`: corrected-observation episode GRPO.
- `train_7b_p6_targeted_grpo.sh`: shared 7B GRPO implementation.

The retained best run used `ARM=B`, seed `20260810`, and output directory
`checkpoints/p6_contract_observation_grpo_arm_c_7b`.

## Evaluation and audit

- `run_p6_generalized_eval.py`: canonical seven-tool evaluator.
- `run_p6_process_isolated_eval.py`: strict process-isolated evaluator.
- `build_p6_scaleup_tune432_comparison.py`: unified Base/SFT/GRPO comparison.
- `summarize_p6_addcolumn72_checkpoints.py` and
  `summarize_p6_eval_matrix.py`: checkpoint and stratified metrics.
- `audit_p6_grpo_coverage.py`, `replay_p6_reward_versions.py`,
  `replay_p6_first_action_reward.py` and `check_p6_candidate_gate.py`: sampling,
  Reward and promotion audits.

Fresh Blind320 must remain unread until a candidate has been selected on
Tune432.
