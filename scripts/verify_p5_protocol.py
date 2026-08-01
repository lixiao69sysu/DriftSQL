#!/usr/bin/env python3
"""Read-only P5 protocol audit that never parses either sealed Gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "data/processed/p5_isolated_protocol"
SEAL = ROOT / "reports/p5/stage8_gate55_permanent_seal.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    summary = load_json(PROTOCOL / "summary.json")
    seal = load_json(SEAL)
    train = load_jsonl(PROTOCOL / "train.jsonl")
    tune = load_jsonl(PROTOCOL / "tune.jsonl")
    split_dbs = {
        split: set(summary["splits"][split]["database_ids"])
        for split in ("train", "tune", "gate")
    }
    development_rows = train + tune
    checks = {
        "database_counts_6_3_3": tuple(len(split_dbs[x]) for x in ("train", "tune", "gate")) == (6, 3, 3),
        "database_splits_pairwise_disjoint": not (
            split_dbs["train"] & split_dbs["tune"]
            or split_dbs["train"] & split_dbs["gate"]
            or split_dbs["tune"] & split_dbs["gate"]
        ),
        "stage7_and_stage8_overlap_empty": (
            summary["stage7_database_overlap"] == []
            and summary["stage8_database_overlap"] == []
        ),
        "gate55_never_read": (
            summary["stage8_gate55_rows_read"] is False
            and seal["gate55_rows_read"] is False
        ),
        "development_rows_54": len(train) == 36 and len(tune) == 18,
        "development_never_contains_gate_database": not (
            {row["db_id"] for row in development_rows} & split_dbs["gate"]
        ),
        "all_development_rows_are_add_column": all(
            row["p5"]["failure_focus"][0] == "add_column" for row in development_rows
        ),
        "turn_limit_hard_slice_36": sum(
            bool(row["p5"]["turn_limit_focus"]) for row in development_rows
        ) == 36,
        "all_sources_are_unseen_bird_critic": all(
            row["p5"]["source_cohort"] == "bird_critic" for row in development_rows
        ),
        "new_gate_hash_intact_without_parsing": (
            sha256(PROTOCOL / "sealed_gate.jsonl") == summary["gate"]["sha256"]
        ),
        "gate_policy_forbids_development_reads": set(summary["gate"]["forbidden_uses"]) == {
            "training", "tuning", "failure_mining", "replay_generation"
        },
        "gate55_seal_hashes_intact": all(
            (ROOT / relative).is_file() and sha256(ROOT / relative) == expected
            for relative, expected in seal["files_sha256"].items()
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    result = {
        "protocol": "driftsql_p5_protocol_audit_v1",
        "passed": not failed,
        "checks": checks,
        "failed": failed,
        "metrics": {
            "database_counts": summary["database_counts"],
            "task_counts": {
                split: summary["splits"][split]["tasks"]["rows"]
                for split in ("train", "tune", "gate")
            },
            "turn_limit_focus": {
                split: summary["splits"][split]["tasks"]["failure_focus"]["turn_limit"]
                for split in ("train", "tune", "gate")
            },
            "rejected_databases": summary["skipped_databases"],
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
