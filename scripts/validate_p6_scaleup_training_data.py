#!/usr/bin/env python3
"""Run the release-gate QA for P6 Scale-up derived training data."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_p6_on_policy_recovery_sft import (
    _sql_equivalent,
    assistant_arguments,
    assistant_name,
)


PROTOCOL = ROOT / "data/processed/p6_scaleup_v1_low_write_protocol"
FAILURE_DIR = ROOT / "data/processed/p6_scaleup_v1_on_policy_failures"
POOL = ROOT / "data/processed/p6_scaleup_v1_rollout_pool600/train_agent_eval.jsonl"
RECOVERY = ROOT / "data/processed/p6_scaleup_v1_recovery_sft"
HARD_REPLAY = ROOT / "data/processed/p6_scaleup_v1_hard_replay"
GRPO = ROOT / "data/processed/p6_scaleup_v1_grpo"
DEFAULT_OUTPUT = ROOT / "reports/p6_scaleup/training_data_qa"
PROMPT_FORBIDDEN_MARKERS = (
    '"ground_truth"',
    '"target_action"',
    '"decision_target_action"',
    '"failure_labels"',
    '"failure_primary"',
    '"replay_role"',
    '"canonical_sql"',
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def target_payload(row: dict[str, Any]) -> dict[str, Any]:
    messages = list(row["messages"])
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError(f"Missing final assistant target for {row.get('task_id')}")
    content = str(messages[-1].get("content", ""))
    for line in reversed(content.splitlines()):
        try:
            candidate = json.loads(line.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and isinstance(candidate.get("arguments"), dict):
            return candidate
    raise ValueError(f"Malformed target payload for {row.get('task_id')}")


def canonical_action_payloads(
    trajectories: list[dict[str, Any]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    output: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for trajectory in trajectories:
        task_id = str(trajectory["task_id"])
        actions: dict[str, list[dict[str, Any]]] = {}
        for message in trajectory["messages"]:
            if message.get("role") != "assistant":
                continue
            name = assistant_name(message)
            if name is None:
                continue
            actions.setdefault(name, []).append(assistant_arguments(message))
        output[task_id] = actions
    return output


def sql_provenance(
    rows: list[dict[str, Any]],
    canonical: dict[str, dict[str, list[dict[str, Any]]]],
) -> tuple[int, list[dict[str, str]]]:
    checked = 0
    failures: list[dict[str, str]] = []
    for row in rows:
        action = str(row["target_action"])
        if action not in {"execute_sql", "submit_solution"}:
            continue
        checked += 1
        task_id = str(row["task_id"])
        payload = target_payload(row)
        if str(payload.get("name", "")) != action:
            failures.append(
                {"task_id": task_id, "reason": "payload_action_mismatch", "action": action}
            )
            continue
        sql = str(payload["arguments"].get("sql", ""))
        candidates = [
            str(arguments.get("sql", ""))
            for arguments in canonical.get(task_id, {}).get(action, [])
        ]
        if not sql or not any(_sql_equivalent(sql, candidate) for candidate in candidates):
            failures.append(
                {"task_id": task_id, "reason": "sql_not_in_verified_oracle", "action": action}
            )
    return checked, failures


def add_check(
    checks: list[dict[str, Any]], name: str, passed: bool, evidence: dict[str, Any]
) -> None:
    checks.append({"name": name, "passed": bool(passed), "evidence": evidence})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")

    failures = load_jsonl(FAILURE_DIR / "failures.jsonl")
    pool = load_jsonl(POOL)
    trajectories = pq.read_table(PROTOCOL / "train_trajectories.parquet").to_pylist()
    source_train_rl = pq.read_table(PROTOCOL / "rl_train.parquet").to_pylist()
    source_tune_rl = pq.read_table(PROTOCOL / "rl_dev.parquet").to_pylist()
    recovery_train = pq.read_table(RECOVERY / "train.parquet").to_pylist()
    recovery_dev = pq.read_table(RECOVERY / "dev.parquet").to_pylist()
    recovery = recovery_train + recovery_dev
    hard = pq.read_table(HARD_REPLAY / "train.parquet").to_pylist()
    grpo_train = pq.read_table(GRPO / "train.parquet").to_pylist()
    grpo_tune = pq.read_table(GRPO / "tune.parquet").to_pylist()
    grpo_manifest = load_jsonl(GRPO / "train_manifest.jsonl")
    summaries = {
        "failure": load_json(FAILURE_DIR / "summary.json"),
        "protocol": load_json(PROTOCOL / "summary.json"),
        "recovery": load_json(RECOVERY / "summary.json"),
        "hard_replay": load_json(HARD_REPLAY / "summary.json"),
        "grpo": load_json(GRPO / "summary.json"),
    }
    checks: list[dict[str, Any]] = []

    failure_keys = [str(row["_failure_miner"]["dedupe_key"]) for row in failures]
    pool_ids = {str(row["extra_info"]["instance_id"]) for row in pool}
    trajectory_ids = {str(row["task_id"]) for row in trajectories}
    failure_ids = {str(row["instance_id"]) for row in failures}
    associations_pass = (
        len(failures) == len(failure_keys) == len(set(failure_keys)) == 1066
        and failure_ids <= pool_ids
        and failure_ids <= trajectory_ids
        and all(not bool(row.get("task_success")) for row in failures)
    )
    add_check(
        checks,
        "unique_failure_association",
        associations_pass,
        {
            "unique_failure_trajectories": len(set(failure_keys)),
            "task_associations": sum(str(row["instance_id"]) in pool_ids for row in failures),
            "canonical_associations": sum(
                str(row["instance_id"]) in trajectory_ids for row in failures
            ),
        },
    )

    terminal_recovery = [
        row for row in recovery if row["recovery_context"] == "terminal_missing"
    ]
    terminal_hard = [
        row for row in hard if row["replay_role"] == "successful_execute_then_submit"
    ]
    terminal_pass = (
        len(terminal_recovery) == 18
        and len(terminal_hard) == 120
        and all(row["target_action"] == "submit_solution" for row in terminal_recovery)
        and all(row["target_action"] == "submit_solution" for row in terminal_hard)
    )
    add_check(
        checks,
        "terminal_no_submit_targets_submit_solution",
        terminal_pass,
        {
            "verified_terminal_recovery_states": len(terminal_recovery),
            "terminal_hard_replay_rows": len(terminal_hard),
            "non_submit_targets": sum(
                row["target_action"] != "submit_solution"
                for row in terminal_recovery + terminal_hard
            ),
        },
    )

    masked = [
        {"task_id": str(row["task_id"]), "target_action": str(row["target_action"])}
        for row in recovery + hard
        if row["target_action"] not in row["available_tools"]
    ]
    add_check(
        checks,
        "dynamic_target_action_available",
        not masked,
        {"checked_rows": len(recovery) + len(hard), "masked_targets": len(masked)},
    )

    canonical = canonical_action_payloads(trajectories)
    recovery_sql_checked, recovery_sql_failures = sql_provenance(recovery, canonical)
    hard_sql_checked, hard_sql_failures = sql_provenance(hard, canonical)
    sql_failures = recovery_sql_failures + hard_sql_failures
    add_check(
        checks,
        "sql_targets_from_execution_verified_oracle",
        not sql_failures,
        {
            "recovery_sql_targets_checked": recovery_sql_checked,
            "hard_replay_sql_targets_checked": hard_sql_checked,
            "provenance_failures": len(sql_failures),
            "failure_examples": sql_failures[:5],
        },
    )

    max_token = max(int(row["token_count"]) for row in recovery + hard)
    add_check(
        checks,
        "token_budget",
        max_token <= 6144 and 1500 <= len(recovery) <= 2000,
        {
            "recovery_examples": len(recovery),
            "hard_replay_examples": len(hard),
            "maximum_token_count": max_token,
            "budget": 6144,
        },
    )

    tune_dbs = {str(row["extra_info"]["db_id"]) for row in source_tune_rl}
    recovery_dbs = {str(row["db_id"]) for row in recovery}
    hard_dbs = {str(row["db_id"]) for row in hard}
    grpo_train_dbs = {str(row["extra_info"]["db_id"]) for row in grpo_train}
    grpo_tune_dbs = {str(row["extra_info"]["db_id"]) for row in grpo_tune}
    overlaps = {
        "recovery_vs_tune": sorted(recovery_dbs & tune_dbs),
        "hard_replay_vs_tune": sorted(hard_dbs & tune_dbs),
        "grpo_train_vs_tune": sorted(grpo_train_dbs & grpo_tune_dbs),
        "recovery_internal_train_dev": sorted(
            {str(row["db_id"]) for row in recovery_train}
            & {str(row["db_id"]) for row in recovery_dev}
        ),
    }
    add_check(
        checks,
        "train_tune_database_isolation",
        not any(overlaps.values()),
        {
            "overlaps": overlaps,
            "train_databases": len(grpo_train_dbs),
            "tune_databases": len(grpo_tune_dbs),
        },
    )

    fresh_flags = {
        "failure": summaries["failure"].get("fresh_blind_rows_read"),
        "protocol": summaries["protocol"].get("fresh_blind", {}).get(
            "read_for_model_selection"
        ),
        "recovery": summaries["recovery"].get("split_guards", {}).get(
            "fresh_blind_rows_read"
        ),
        "hard_replay": summaries["hard_replay"].get("fresh_blind_rows_read"),
        "grpo": summaries["grpo"].get("fresh_blind_rows_read"),
    }
    add_check(
        checks,
        "fresh_blind_read_count_zero",
        all(value is False for value in fresh_flags.values()),
        {"component_flags": fresh_flags, "fresh_blind_rows_read": 0},
    )

    source_by_id = {
        str(row["extra_info"]["instance_id"]): row for row in source_train_rl
    }
    changed_prompts = 0
    marker_leaks = 0
    for row in grpo_train:
        task_id = str(row["extra_info"]["instance_id"])
        if row["prompt"] != source_by_id[task_id]["prompt"]:
            changed_prompts += 1
        text = json.dumps(row["prompt"], ensure_ascii=False, sort_keys=True)
        marker_leaks += any(marker in text for marker in PROMPT_FORBIDDEN_MARKERS)
    base_manifest = [row for row in grpo_manifest if row["sampling_role"] == "base_unique"]
    prompt_pass = (
        len(grpo_train) == len(grpo_manifest) == 3200
        and changed_prompts == 0
        and marker_leaks == 0
        and len(base_manifest) == 2400
        and {row["task_id"] for row in base_manifest} == set(source_by_id)
    )
    add_check(
        checks,
        "grpo_prompt_has_no_answer_or_target_leakage",
        prompt_pass,
        {
            "prompts_checked": len(grpo_train),
            "prompts_changed_from_sealed_source": changed_prompts,
            "forbidden_marker_rows": marker_leaks,
            "base_unique_rows": len(base_manifest),
            "replay_metadata_in_sidecar_only": True,
        },
    )

    count_evidence = {
        "recovery_sft": len(recovery),
        "hard_replay": len(hard),
        "grpo_train": len(grpo_train),
        "grpo_tune": len(grpo_tune),
        "failure_trajectories": len(failures),
    }
    counts_pass = (
        1500 <= len(recovery) <= 2000
        and len(hard) == 1600
        and len(grpo_train) == 3200
        and len(grpo_tune) == 432
        and len(failures) == 1066
    )
    add_check(checks, "dataset_sizes", counts_pass, count_evidence)

    passed = all(check["passed"] for check in checks)
    report = {
        "protocol": "p6_scaleup_training_data_release_gate_v1",
        "status": "passed" if passed else "failed",
        "checks_passed": sum(check["passed"] for check in checks),
        "checks_total": len(checks),
        "checks": checks,
        "dataset_sizes": count_evidence,
        "target_action_distribution": {
            "recovery": dict(
                sorted(Counter(str(row["target_action"]) for row in recovery).items())
            ),
            "hard_replay": dict(
                sorted(Counter(str(row["target_action"]) for row in hard).items())
            ),
        },
        "fresh_blind_rows_read": 0,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "checks.jsonl").open("w", encoding="utf-8") as handle:
        for check in checks:
            handle.write(json.dumps(check, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not passed:
        raise RuntimeError("P6 Scale-up training-data release gate failed")


if __name__ == "__main__":
    main()
