#!/usr/bin/env python3
"""Fail closed unless a P6 candidate satisfies the declared evaluation gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FAST_EXPECTED_TASKS = 42
FAST_EXPECTED_DRIFT_TASKS = 35
FAST_MIN_SUCCESS = 15
FAST_MIN_DRIFT_RECOVERY = 10

DEV_EXPECTED_TASKS = 169
DEV_EXPECTED_DRIFT_TASKS = 154
DEV_MIN_SUCCESS_RATE = 0.35
BASE_DEV_DRIFT_RECOVERY = 40


def check_gate(
    summary: dict[str, Any],
    stage: str,
    baseline_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = summary["requested_metrics"]
    if stage == "fast":
        checks = {
            "tasks_exact": int(metrics["tasks"]) == FAST_EXPECTED_TASKS,
            "drift_tasks_exact": int(metrics["drift_tasks"]) == FAST_EXPECTED_DRIFT_TASKS,
            "overall_success": int(metrics["execution_success"]) >= FAST_MIN_SUCCESS,
            "non_clean_recovery": int(metrics["drift_recovery"]) >= FAST_MIN_DRIFT_RECOVERY,
            "zero_unsafe": int(metrics["unsafe_tasks"]) == 0,
            "zero_timeout": int(metrics["timeout_tasks"]) == 0,
        }
    elif stage == "dev":
        checks = {
            "tasks_exact": int(metrics["tasks"]) == DEV_EXPECTED_TASKS,
            "drift_tasks_exact": int(metrics["drift_tasks"]) == DEV_EXPECTED_DRIFT_TASKS,
            "overall_success_rate": float(metrics["execution_success_rate"])
            >= DEV_MIN_SUCCESS_RATE,
            "beats_base_non_clean": int(metrics["drift_recovery"])
            > BASE_DEV_DRIFT_RECOVERY,
            "zero_unsafe": int(metrics["unsafe_tasks"]) == 0,
            "zero_timeout": int(metrics["timeout_tasks"]) == 0,
        }
    else:
        raise ValueError(stage)
    baseline_metrics = None
    if baseline_summary is not None:
        baseline_metrics = baseline_summary["requested_metrics"]
        checks.update(
            {
                "beats_same_protocol_baseline_overall": int(metrics["execution_success"])
                > int(baseline_metrics["execution_success"]),
                "beats_same_protocol_baseline_non_clean": int(metrics["drift_recovery"])
                > int(baseline_metrics["drift_recovery"]),
            }
        )
    return {
        "stage": stage,
        "alias": summary.get("alias"),
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": metrics,
        "baseline_alias": baseline_summary.get("alias") if baseline_summary else None,
        "baseline_metrics": baseline_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path)
    parser.add_argument("--stage", choices=("fast", "dev"), required=True)
    args = parser.parse_args()
    baseline = (
        json.loads(args.baseline_summary.read_text(encoding="utf-8"))
        if args.baseline_summary is not None
        else None
    )
    result = check_gate(
        json.loads(args.summary.read_text(encoding="utf-8")),
        args.stage,
        baseline,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
