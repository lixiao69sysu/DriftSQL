#!/usr/bin/env python3
"""Freeze the Tune-selected Stage 8 candidate before one-shot Gate55."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADAPTER = (
    PROJECT_ROOT
    / "checkpoints/stage8_fresh_sft_7b/global_step_20/merged/lora_adapter"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "reports/stage8/final_candidate/frozen_candidate.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def subset(rows: list[dict[str, Any]], drift_type: str | None) -> dict[str, Any]:
    selected = rows if drift_type is None else [row for row in rows if row["drift_type"] == drift_type]
    success = sum(bool(row["task_success"]) for row in selected)
    return {
        "tasks": len(selected),
        "task_success": success,
        "task_success_rate": success / len(selected),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Stage 8 candidate already frozen: {args.output}")

    report_paths = {
        "stage7_frozen_tune55": PROJECT_ROOT / "reports/stage8/tune55_stage7_frozen_process_isolated/summary.json",
        "stage8_sft20_tune55": PROJECT_ROOT / "reports/stage8/tune55_stage8_sft20_process_isolated/summary.json",
        "grpo_trial1_step5_add30": PROJECT_ROOT / "reports/stage8/tune30_add_grpo_step5_process_isolated/summary.json",
        "grpo_trial1_step10_add30": PROJECT_ROOT / "reports/stage8/tune30_add_grpo_step10_process_isolated/summary.json",
        "grpo_conservative_step2_add30": PROJECT_ROOT / "reports/stage8/tune30_add_conservative_step2_process_isolated/summary.json",
        "grpo_conservative_step4_add30": PROJECT_ROOT / "reports/stage8/tune30_add_conservative_step4_process_isolated/summary.json",
        "grpo_conservative_step6_add30": PROJECT_ROOT / "reports/stage8/tune30_add_conservative_step6_process_isolated/summary.json",
    }
    results = {name: load(path)["result"] for name, path in report_paths.items()}
    sft_rows_path = (
        PROJECT_ROOT
        / "reports/stage8/tune55_stage8_sft20_process_isolated/stage8-sft20-fresh-tune.jsonl"
    )
    baseline_rows_path = (
        PROJECT_ROOT
        / "reports/stage8/tune55_stage7_frozen_process_isolated/stage7-frozen-fresh-tune.jsonl"
    )
    sft_rows = load_jsonl(sft_rows_path)
    baseline_rows = load_jsonl(baseline_rows_path)
    stratified = {
        "stage7_frozen": {
            "all": subset(baseline_rows, None),
            "add_column": subset(baseline_rows, "add_column"),
            "general": subset(
                [row for row in baseline_rows if row["drift_type"] != "add_column"], None
            ),
        },
        "stage8_sft20": {
            "all": subset(sft_rows, None),
            "add_column": subset(sft_rows, "add_column"),
            "general": subset(
                [row for row in sft_rows if row["drift_type"] != "add_column"], None
            ),
        },
    }
    sft = results["stage8_sft20_tune55"]
    baseline = results["stage7_frozen_tune55"]
    grpo_results = [result for name, result in results.items() if name.startswith("grpo_")]
    acceptance = {
        "baseline_tasks_eq_55": baseline["tasks"] == 55,
        "sft_tasks_eq_55": sft["tasks"] == 55,
        "sft_overall_success_gt_stage7": sft["task_success"] > baseline["task_success"],
        "sft_add_success_gt_stage7": (
            stratified["stage8_sft20"]["add_column"]["task_success"]
            > stratified["stage7_frozen"]["add_column"]["task_success"]
        ),
        "sft_general_success_gt_stage7": (
            stratified["stage8_sft20"]["general"]["task_success"]
            > stratified["stage7_frozen"]["general"]["task_success"]
        ),
        "no_grpo_add_candidate_beats_sft": all(
            result["task_success"]
            < stratified["stage8_sft20"]["add_column"]["task_success"]
            for result in grpo_results
        ),
        "sft_unsafe_eq_0": sft["unsafe_tasks"] == 0,
        "sft_timeout_eq_0": sft["timeout_tasks"] == 0,
    }
    if not all(acceptance.values()):
        raise RuntimeError(f"Stage 8 Tune acceptance failed: {acceptance}")

    locked = [
        PROJECT_ROOT / "models.lock.json",
        PROJECT_ROOT / "configs/tools/drift_tools.yaml",
        PROJECT_ROOT / "driftsql/rewards/agentic.py",
        PROJECT_ROOT / "driftsql/drift/factory.py",
        PROJECT_ROOT / "driftsql/integrations/verl_tools.py",
        PROJECT_ROOT / "driftsql/integrations/state_policy.py",
        PROJECT_ROOT / "driftsql/planning/projection_contract.py",
        PROJECT_ROOT / "scripts/run_five_tool_eval.py",
        PROJECT_ROOT / "scripts/run_stage6_eval.py",
        PROJECT_ROOT / "scripts/run_stage7_process_isolated_eval.py",
        PROJECT_ROOT / "scripts/prepare_stage8_fresh_sft.py",
        PROJECT_ROOT / "data/processed/stage8_fresh_protocol/summary.json",
        PROJECT_ROOT / "data/processed/stage8_fresh_protocol/gate_add_column.jsonl",
        PROJECT_ROOT / "data/processed/stage8_fresh_protocol/gate_general_replay.jsonl",
    ]
    adapter_files = sorted(path for path in args.adapter.iterdir() if path.is_file())
    payload = {
        "protocol": "driftsql_stage8_frozen_candidate_v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": {
            "name": "stage8-fresh-db-submit-sft20",
            "base_model": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "adapter_path": str(args.adapter.resolve().relative_to(PROJECT_ROOT)),
            "adapter_files_sha256": {
                str(path.relative_to(PROJECT_ROOT)): sha256(path) for path in adapter_files
            },
            "wandb_run": "https://wandb.ai/lixiao69-/driftsql-rl/runs/lb10hmn4",
            "inference": {
                "max_turns": 7,
                "max_new_tokens": 512,
                "max_model_len": 8192,
                "dynamic_tool_mask": True,
                "state_guards": True,
                "context_bounded_generation": True,
                "async_scheduling": False,
                "prefix_caching": False,
                "isolation": "one OS process and one vLLM engine per episode",
            },
        },
        "tune_selection": {
            "reports_sha256": {
                str(path.relative_to(PROJECT_ROOT)): sha256(path)
                for path in [*report_paths.values(), sft_rows_path, baseline_rows_path]
            },
            "results": results,
            "stratified_results": stratified,
            "acceptance": acceptance,
            "decision": "SFT20 selected; two GRPO trials failed the add-column Tune funnel.",
        },
        "locked_files_sha256": {
            str(path.relative_to(PROJECT_ROOT)): sha256(path) for path in locked
        },
        "one_shot_gate55": {
            "tasks": 55,
            "add_column_tasks": 30,
            "general_tasks": 25,
            "allowed_candidate_runs": 1,
            "acceptance_precommitted": {
                "overall_success_gte_0_55": 0.55,
                "add_column_success_gte_0_20": 0.20,
                "general_success_gte_0_80": 0.80,
                "unsafe_eq_0": 0,
                "timeout_eq_0": 0,
            },
        },
        "prior_gates": "Stage 6 Gate112 and Stage 7 Gate106 remain permanently sealed.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "acceptance": acceptance}, indent=2))


if __name__ == "__main__":
    main()
