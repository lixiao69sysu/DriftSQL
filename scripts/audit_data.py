#!/usr/bin/env python3
"""Print public-data integrity and training-readiness status."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from driftsql.data import (
    audit_bird23_train,
    audit_bird_mini_dev,
    audit_mini_interact,
    audit_six_gym_sqlite,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=("all", "bird23", "six-gym", "mini-dev", "mini-interact"),
        default="all",
    )
    parser.add_argument(
        "--quick-check",
        action="store_true",
        help="Run PRAGMA quick_check on every referenced SQLite database.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print compact counts instead of per-database details.",
    )
    args = parser.parse_args()

    roots = {
        "bird23": PROJECT_ROOT / "data" / "raw" / "bird23-train-filtered",
        "six-gym": PROJECT_ROOT / "data" / "raw" / "six-gym-sqlite",
        "mini-dev": PROJECT_ROOT / "data" / "raw" / "bird-mini-dev",
        "mini-interact": PROJECT_ROOT / "data" / "raw" / "mini-interact",
    }
    auditors = {
        "bird23": lambda: audit_bird23_train(
            roots["bird23"], quick_check=args.quick_check
        ),
        "six-gym": lambda: audit_six_gym_sqlite(
            roots["six-gym"], quick_check=args.quick_check
        ),
        "mini-dev": lambda: audit_bird_mini_dev(
            roots["mini-dev"], quick_check=args.quick_check
        ),
        "mini-interact": lambda: audit_mini_interact(roots["mini-interact"]),
    }
    selected = list(auditors) if args.dataset == "all" else [args.dataset]
    reports = {name: auditors[name]() for name in selected}

    train_databases = set()
    for name in ("bird23", "six-gym"):
        if name in reports:
            train_databases.update(reports[name]["database_checks"])
    eval_databases = set(reports.get("mini-dev", {}).get("database_checks", {}))
    summary = {
        "reports": reports,
        "split_policy": {
            "unit": "db_id",
            "train_database_count": len(train_databases),
            "eval_database_count": len(eval_databases),
            "train_eval_overlap": sorted(train_databases & eval_databases),
        },
    }
    if args.summary_only:
        compact = {
            name: {
                key: report[key]
                for key in (
                    "rows",
                    "databases",
                    "ground_truth_rows",
                    "test_case_rows",
                    "missing_databases",
                    "invalid_databases",
                    "ready",
                )
                if key in report
            }
            for name, report in reports.items()
        }
        compact["split_policy"] = summary["split_policy"]
        print(json.dumps(compact, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    required_reports = [
        report
        for name, report in reports.items()
        if name in {"bird23", "six-gym", "mini-dev"}
    ]
    if any(not report["ready"] for report in required_reports):
        raise SystemExit("One or more core datasets failed integrity checks")
    if summary["split_policy"]["train_eval_overlap"]:
        raise SystemExit(
            "Database leakage detected between training and Mini-Dev evaluation"
        )


if __name__ == "__main__":
    main()
