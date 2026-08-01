#!/usr/bin/env python3
"""Read-only requirement audit for the completed Stage 8 objective."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from driftsql.integrations.verl_tools import _active_schema_for_projection
from driftsql.planning import plan_projection_contract


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def load_jsonl(relative: str) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (ROOT / relative).read_text(encoding="utf-8").splitlines()
        if line
    ]


def verify_hashes(mapping: dict[str, str]) -> bool:
    return all((ROOT / relative).is_file() and sha256(ROOT / relative) == expected for relative, expected in mapping.items())


def wandb_summary(run_id: str) -> dict[str, Any]:
    candidates = sorted((ROOT / "wandb/wandb").glob(f"run-*-{run_id}"))
    if not candidates:
        return {}
    path = candidates[-1] / "files/wandb-summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def main() -> None:
    protocol = load("data/processed/stage8_fresh_protocol/summary.json")
    sft = load("data/processed/stage8_fresh_sft/summary.json")
    replay = load("data/processed/stage8_failure_balanced_grpo/summary.json")
    frozen = load("reports/stage8/final_candidate/frozen_candidate.json")
    gate = load("reports/stage8/final_candidate/gate55_result.json")
    gate112 = load("reports/stage7/stage6_gate112_seal.json")
    gate106 = load("reports/stage8/stage7_gate106_seal.json")
    tune = frozen["tune_selection"]["stratified_results"]
    baseline = tune["stage7_frozen"]
    candidate = tune["stage8_sft20"]

    split_dbs = {
        split: set(protocol["splits"][split]["database_ids"])
        for split in ("train", "tune", "gate")
    }
    sft_wandb = wandb_summary("lb10hmn4")
    grpo_wandb = wandb_summary("i57aenm4")
    conservative_wandb = wandb_summary("ozk0pw95")
    launchers = "\n".join(
        (ROOT / path).read_text(encoding="utf-8")
        for path in (
            "scripts/train_7b_stage8_fresh_sft.sh",
            "scripts/train_7b_stage8_failure_balanced_grpo.sh",
            "scripts/train_7b_stage8_conservative_grpo.sh",
        )
    )

    expected_profiles = {
        "train": {
            "single_table_plain": 20,
            "single_table_qualified": 20,
            "multi_table_plain": 40,
            "multi_table_qualified": 40,
        },
        "tune": {
            "single_table_plain": 5,
            "single_table_qualified": 5,
            "multi_table_plain": 10,
            "multi_table_qualified": 10,
        },
        "gate": {
            "single_table_plain": 5,
            "single_table_qualified": 5,
            "multi_table_plain": 10,
            "multi_table_qualified": 10,
        },
    }
    profile_checks: dict[str, bool] = {}
    planner_checks: dict[str, bool] = {}
    audit_column_checks: dict[str, bool] = {}
    expected_audit_columns = {
        "source_sync_flag",
        "pipeline_batch_id",
        "ingestion_audit_id",
        "quality_review_state",
        "compliance_trace_tag",
        "record_lineage_tag",
    }
    for split in ("train", "tune", "gate"):
        rows = load_jsonl(f"data/processed/stage8_fresh_protocol/{split}_add_column.jsonl")
        profile_checks[split] = Counter(row["wildcard_profile"] for row in rows) == expected_profiles[split]
        added_names = {
            str(operation["new_name"])
            for row in rows
            for operation in row["schema_diff"]["operations"]
            if operation["type"] == "add_column"
        }
        audit_column_checks[split] = added_names == expected_audit_columns
        planner_ok = True
        for row in rows:
            active_schema = _active_schema_for_projection(Path(row["source_db"]), row["schema_diff"])
            plan = plan_projection_contract(row["stale_sql"], row["schema_diff"], active_schema)
            added = {
                str(operation["new_name"]).casefold()
                for operation in row["schema_diff"]["operations"]
                if operation["type"] == "add_column"
            }
            planner_ok = planner_ok and (
                plan.repaired_sql == row["repaired_sql"]
                and all(name not in plan.repaired_sql.casefold() for name in added)
            )
        planner_checks[split] = planner_ok

    checks = {
        "database_split_counts_20_5_5": tuple(len(split_dbs[x]) for x in ("train", "tune", "gate")) == (20, 5, 5),
        "database_splits_pairwise_disjoint": not (
            split_dbs["train"] & split_dbs["tune"]
            or split_dbs["train"] & split_dbs["gate"]
            or split_dbs["tune"] & split_dbs["gate"]
        ),
        "stage7_database_overlap_empty": protocol["stage7_database_overlap"] == [],
        "wildcard_profiles_cover_select_alias_and_multitable": all(profile_checks.values()),
        "audit_column_scenarios_cover_all_splits": all(audit_column_checks.values()),
        "projection_planner_exactly_reproduces_oracle_and_excludes_additions": all(planner_checks.values()),
        "gate112_hashes_intact": verify_hashes(gate112["files_sha256"]),
        "gate106_hashes_intact": verify_hashes(gate106["files_sha256"]),
        "sft_train_tune_only": (
            sft["stage8_gate_read"] is False
            and sft["stage7_gate106_read"] is False
            and sft["train_tune_database_overlap"] == []
        ),
        "sft_has_general_replay": (
            sft["splits"]["train"]["general_replay_examples"] == 800
            and sft["splits"]["train"]["general_replay_ratio_actual"] >= 0.37
        ),
        "failure_balanced_grpo_uses_real_failures": (
            replay["real_tune_trajectories"] == 55
            and replay["output_rows"] == 440
            and replay["add_column_rows"] == 308
            and replay["general_replay_rows"] == 132
        ),
        "failure_balanced_grpo_has_no_tune_or_gate_rows": (
            replay["train_tune_task_overlap"] == []
            and replay["train_tune_database_overlap"] == []
            and replay["stage6_gate112_read"] is False
            and replay["stage7_gate106_read"] is False
            and replay["stage8_gate_read"] is False
        ),
        "verl_launchers_enable_online_wandb": (
            launchers.count("WANDB_MODE=\"${WANDB_MODE:-online}\"") == 3
            and launchers.count("trainer.logger='[\"console\",\"wandb\"]'") == 3
        ),
        "sft_wandb_completed_step20": sft_wandb.get("_step") == 20 and "val/loss" in sft_wandb,
        "grpo_wandb_completed_step10_with_rl_metrics": (
            grpo_wandb.get("_step") == 10
            and "critic/score/mean" in grpo_wandb
            and "actor/ppo_kl" in grpo_wandb
        ),
        "conservative_grpo_wandb_completed_step6": (
            conservative_wandb.get("_step") == 6
            and "critic/score/mean" in conservative_wandb
            and "actor/ppo_kl" in conservative_wandb
        ),
        "add_column_improves_on_same_tune": (
            candidate["add_column"]["task_success"] > baseline["add_column"]["task_success"]
        ),
        "overall_regression_within_2pp": (
            candidate["all"]["task_success_rate"] >= baseline["all"]["task_success_rate"] - 0.02
        ),
        "frozen_candidate_hashes_intact": (
            verify_hashes(frozen["candidate"]["adapter_files_sha256"])
            and verify_hashes(frozen["locked_files_sha256"])
            and verify_hashes(frozen["tune_selection"]["reports_sha256"])
        ),
        "one_shot_gate55_passed": gate["one_shot_runs"] == 1 and gate["passed"] is True,
        "one_shot_gate55_hashes_intact": verify_hashes(gate["artifacts_sha256"]),
        "gate55_safe_and_timeout_free": (
            gate["results"]["overall"]["unsafe"] == 0
            and gate["results"]["overall"]["timeout"] == 0
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    result = {
        "protocol": "driftsql_stage8_completion_audit_v1",
        "passed": not failed,
        "checks": checks,
        "failed": failed,
        "metrics": {
            "baseline_tune": baseline,
            "candidate_tune": candidate,
            "candidate_gate": gate["results"],
            "wandb_runs": {
                "sft": "https://wandb.ai/lixiao69-/driftsql-rl/runs/lb10hmn4",
                "grpo_trial1": "https://wandb.ai/lixiao69-/driftsql-rl/runs/i57aenm4",
                "grpo_conservative": "https://wandb.ai/lixiao69-/driftsql-rl/runs/ozk0pw95",
            },
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
