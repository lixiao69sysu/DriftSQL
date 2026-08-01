#!/usr/bin/env python3
"""Fail closed when P5 GRPO inputs violate review or database isolation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/p5_grpo")
    args = parser.parse_args()
    summary = json.loads((args.data_dir / "summary.json").read_text(encoding="utf-8"))
    train = pq.read_table(args.data_dir / "train.parquet").to_pylist()
    tune = pq.read_table(args.data_dir / "tune.parquet").to_pylist()
    train_extra = [row["extra_info"] for row in train]
    tune_extra = [row["extra_info"] for row in tune]
    replay = [row for row in train_extra if row["p5_reviewed_replay"]]
    train_dbs = {str(row["db_id"]) for row in train_extra}
    tune_dbs = {str(row["db_id"]) for row in tune_extra}
    checks = {
        "protocol": summary.get("protocol") == "driftsql_p5_reviewed_replay_grpo_data_v1",
        "human_approved_candidate_present": summary.get("approved_candidates", 0) > 0,
        "human_reviewed_replay_present": len(replay) == summary["rows"]["reviewed_replay"] > 0,
        "replay_has_candidate_and_reviewer": all(
            row["p5_replay_candidate_id"] and row["p5_replay_reviewer"] for row in replay
        ),
        "row_counts_match": (
            len(train) == summary["rows"]["train"]
            and len(tune) == summary["rows"]["tune"]
        ),
        "train_tune_database_isolated": not (train_dbs & tune_dbs),
        "train_gate_database_isolated": summary["train_gate_database_overlap"] == [],
        "tune_gate_database_isolated": summary["tune_gate_database_overlap"] == [],
        "all_train_rows_marked_train": all(row["p5_split"] == "train" for row in train_extra),
        "all_tune_rows_marked_tune": all(row["p5_split"] == "tune" for row in tune_extra),
        "p5_gate_never_read": summary["p5_gate_read"] is False,
        "gate55_never_read": summary["stage8_gate55_read"] is False,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    result = {
        "protocol": "driftsql_p5_training_input_audit_v1",
        "passed": not failed,
        "checks": checks,
        "failed": failed,
        "metrics": {
            "train_rows": len(train),
            "tune_rows": len(tune),
            "reviewed_replay_rows": len(replay),
            "train_databases": len(train_dbs),
            "tune_databases": len(tune_dbs),
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

