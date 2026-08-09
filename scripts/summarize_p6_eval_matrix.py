#!/usr/bin/env python3
"""Summarize comparable P6 evaluations overall and by protocol stratum."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


SUBMITTED_REASONS = {"submitted", "fallback_submitted", "contract_validated_auto_submit"}
ATOMIC_DRIFTS = {"add_column", "rename_column", "rename_table", "replace_column"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = len(rows)
    if not tasks:
        return {"tasks": 0, "success": 0, "success_rate": 0.0}
    submitted = [row for row in rows if row.get("termination_reason") in SUBMITTED_REASONS]
    safe_submitted = [
        row
        for row in submitted
        if bool(row.get("executable"))
        and not bool(row.get("safety", {}).get("unsafe"))
        and not bool(row.get("safety", {}).get("timed_out"))
    ]
    masked_actions = [
        event
        for row in rows
        for event in row.get("trajectory", [])
        if bool(event.get("metrics", {}).get("action_masked"))
    ]
    masked_tasks = sum(
        any(bool(event.get("metrics", {}).get("action_masked")) for event in row.get("trajectory", []))
        for row in rows
    )
    success = sum(bool(row.get("task_success")) for row in rows)
    return {
        "tasks": tasks,
        "success": success,
        "success_rate": success / tasks,
        "submitted": len(submitted),
        "safe_submitted": len(safe_submitted),
        "safe_submission_precision": len(safe_submitted) / len(submitted) if submitted else 0.0,
        "turn_limit": sum(row.get("termination_reason") == "turn_limit" for row in rows),
        "invalid_output": sum(row.get("termination_reason") == "invalid_output" for row in rows),
        "unsafe_tasks": sum(bool(row.get("safety", {}).get("unsafe")) for row in rows),
        "timeout_tasks": sum(bool(row.get("safety", {}).get("timed_out")) for row in rows),
        "average_tool_calls": sum(int(row.get("usage", {}).get("tool_calls", 0)) for row in rows)
        / tasks,
        "average_model_calls": sum(int(row.get("usage", {}).get("model_calls", 0)) for row in rows)
        / tasks,
        "masked_actions": len(masked_actions),
        "masked_tasks": masked_tasks,
        "termination_reasons": dict(
            sorted(Counter(str(row.get("termination_reason")) for row in rows).items())
        ),
    }


def grouped(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    values = sorted({str(row.get(key, "")) for row in rows})
    return {value: metrics([row for row in rows if str(row.get(key, "")) == value]) for value in values}


def summarize_variant(alias: str, path: Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    non_clean = [row for row in rows if str(row.get("drift_type")) != "clean"]
    atomic = [row for row in rows if str(row.get("drift_type")) in ATOMIC_DRIFTS]
    compound = [row for row in rows if str(row.get("drift_type")) == "compound"]
    return {
        "alias": alias,
        "path": str(path.resolve()),
        "instance_ids": [str(row["instance_id"]) for row in rows],
        "overall": metrics(rows),
        "non_clean": metrics(non_clean),
        "atomic": metrics(atomic),
        "compound": metrics(compound),
        "by_drift_type": grouped(rows, "drift_type"),
        "by_interaction_profile": grouped(rows, "interaction_profile"),
        "by_difficulty": grouped(rows, "difficulty"),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# P6 evaluation matrix",
        "",
        "| Variant | Overall | Non-clean | Atomic | Compound | Safe precision | Avg tools | Masked actions | Turn limit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for alias in summary["ranking"]:
        row = summary["variants"][alias]
        lines.append(
            f"| {alias} | {row['overall']['success']}/{row['overall']['tasks']} | "
            f"{row['non_clean']['success']}/{row['non_clean']['tasks']} | "
            f"{row['atomic']['success']}/{row['atomic']['tasks']} | "
            f"{row['compound']['success']}/{row['compound']['tasks']} | "
            f"{row['overall']['safe_submission_precision']:.2%} | "
            f"{row['overall']['average_tool_calls']:.2f} | "
            f"{row['overall']['masked_actions']} | {row['overall']['turn_limit']} |"
        )
    for group_key, title in (
        ("by_drift_type", "Drift type"),
        ("by_interaction_profile", "Interaction profile"),
        ("by_difficulty", "Difficulty"),
    ):
        lines.extend(["", f"## {title}", ""])
        labels = sorted(
            {label for row in summary["variants"].values() for label in row[group_key]}
        )
        lines.append("| Variant | " + " | ".join(labels) + " |")
        lines.append("|---|" + "---:|" * len(labels))
        for alias in summary["ranking"]:
            values = []
            for label in labels:
                value = summary["variants"][alias][group_key].get(label, {"success": 0, "tasks": 0})
                values.append(f"{value['success']}/{value['tasks']}")
            lines.append(f"| {alias} | " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", action="append", required=True, help="alias=/path/to/results.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    variants: dict[str, dict[str, Any]] = {}
    for spec in args.variant:
        alias, separator, value = spec.partition("=")
        if not separator or not alias or alias in variants:
            raise ValueError(f"Invalid or duplicate variant: {spec}")
        variants[alias] = summarize_variant(alias, Path(value))
    identity_sets = {tuple(row["instance_ids"]) for row in variants.values()}
    if len(identity_sets) != 1:
        raise RuntimeError("Variants do not contain the same ordered task IDs")
    ranking = sorted(
        variants,
        key=lambda alias: (
            -variants[alias]["overall"]["success"],
            -variants[alias]["non_clean"]["success"],
            -variants[alias]["atomic"]["success"],
            variants[alias]["overall"]["unsafe_tasks"],
            variants[alias]["overall"]["timeout_tasks"],
            variants[alias]["overall"]["average_tool_calls"],
        ),
    )
    for row in variants.values():
        row.pop("instance_ids")
    summary = {"protocol": "p6_eval_matrix_v1", "ranking": ranking, "variants": variants}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "matrix.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "matrix.md").write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps({"ranking": ranking}, ensure_ascii=False))


if __name__ == "__main__":
    main()
