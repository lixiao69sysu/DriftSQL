#!/usr/bin/env python3
"""Create the paired Stage 6 B0/B1 tune report and choose the next bottleneck."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def metrics(rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    selected = [row for row in rows if predicate(row)]
    tasks = len(selected)
    success = sum(bool(row["task_success"]) for row in selected)
    submitted = sum(row["termination_reason"] in {"submitted", "fallback_submitted"} for row in selected)
    turn_limit = sum(row["termination_reason"] == "turn_limit" for row in selected)
    return {
        "tasks": tasks,
        "success": success,
        "success_rate": success / tasks,
        "submitted": submitted,
        "submission_rate": submitted / tasks,
        "turn_limit": turn_limit,
        "turn_limit_rate": turn_limit / tasks,
        "unsafe": sum(bool(row["safety"]["unsafe"]) for row in selected),
        "timeout": sum(bool(row["safety"]["timed_out"]) for row in selected),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b0", type=Path, default=PROJECT_ROOT / "reports/stage6/b0_tune/b0.jsonl")
    parser.add_argument("--b1", type=Path, default=PROJECT_ROOT / "reports/stage6/b1_tune/b1.jsonl")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports/stage6/b0_b1_comparison.json")
    args = parser.parse_args()

    variants = {"b0": load_jsonl(args.b0), "b1": load_jsonl(args.b1)}
    indexed = {
        name: {str(row["instance_id"]): row for row in rows}
        for name, rows in variants.items()
    }
    if set(indexed["b0"]) != set(indexed["b1"]):
        raise RuntimeError("B0/B1 task IDs are not paired")

    slices: dict[str, Callable[[dict[str, Any]], bool]] = {
        "overall": lambda _row: True,
        "non_clean": lambda row: row["scenario_type"] != "clean",
        "schema_only": lambda row: row["interaction_profile"] == "schema_only",
        "add_column": lambda row: row["drift_type"] == "add_column",
        "compound": lambda row: row["drift_type"] == "compound",
    }
    result: dict[str, Any] = {
        "protocol": "stage6_paired_b0_b1_tune_v1",
        "tasks": len(indexed["b0"]),
        "variants": {
            name: {slice_name: metrics(rows, predicate) for slice_name, predicate in slices.items()}
            for name, rows in variants.items()
        },
    }
    transitions = Counter(
        (
            bool(indexed["b0"][instance_id]["task_success"]),
            bool(indexed["b1"][instance_id]["task_success"]),
        )
        for instance_id in indexed["b0"]
    )
    result["paired_transitions"] = {
        "both_fail": transitions[(False, False)],
        "b1_gain": transitions[(False, True)],
        "b1_regression": transitions[(True, False)],
        "both_success": transitions[(True, True)],
    }
    result["b1_tool_repetition"] = {}
    for tool in ("get_schema_version", "inspect_schema_diff", "get_schema", "execute_sql"):
        counts = Counter(row["called_tools"].count(tool) for row in variants["b1"])
        result["b1_tool_repetition"][tool] = {
            "total_calls": sum(count * tasks for count, tasks in counts.items()),
            "tasks_with_repetition": sum(tasks for count, tasks in counts.items() if count > 1),
        }
    b0_non_clean = result["variants"]["b0"]["non_clean"]["success_rate"]
    b1_non_clean = result["variants"]["b1"]["non_clean"]["success_rate"]
    b1_overall = result["variants"]["b1"]["overall"]
    result["acceptance_diagnostics"] = {
        "overall_success_at_least_20pct": b1_overall["success_rate"] >= 0.20,
        "non_clean_success_at_least_2x": b1_non_clean >= 2.0 * b0_non_clean,
        "schema_only_success_at_least_10pct": result["variants"]["b1"]["schema_only"]["success_rate"] >= 0.10,
        "submission_at_least_40pct": b1_overall["submission_rate"] >= 0.40,
        "turn_limit_at_most_55pct": b1_overall["turn_limit_rate"] <= 0.55,
        "unsafe_and_timeout_zero": b1_overall["unsafe"] == 0 and b1_overall["timeout"] == 0,
    }
    result["next_action"] = (
        "B2 repair/next-action SFT focused on submit-after-success, repeated retrieval loops, "
        "add-column result contracts, and compound drift; do not start broad GRPO yet."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
