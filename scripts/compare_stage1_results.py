#!/usr/bin/env python3
"""Build a compact, paired comparison from unified Stage-1 eval reports."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def wilson_interval(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    rate = correct / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return center - margin, center + margin


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def task_key(item: dict[str, Any]) -> str:
    return str(item.get("question_id", item.get("instance_idx")))


def aggregate(report: dict[str, Any]) -> dict[str, Any]:
    results = report["results"]
    summary = report["summary"]["overall"]
    total = int(summary["total"])
    correct = int(summary["correct"])
    low, high = wilson_interval(correct, total)
    usage_keys = ("model_calls", "tool_calls", "sql_executions", "new_tokens", "total_tokens")
    averages = {
        key: sum(float(item.get("usage", {}).get(key, 0)) for item in results) / total
        if total
        else 0.0
        for key in usage_keys
    }
    return {
        "baseline": report["baseline"],
        "total": total,
        "correct": correct,
        "execution_accuracy": float(summary["execution_accuracy"]),
        "executable_rate": float(summary["executable_rate"]),
        "accuracy_ci95": [low, high],
        "average_usage": averages,
        "budget_violations": int(report["summary"]["budget_violations"]),
        "termination_reasons": report["summary"]["termination_reasons"],
    }


def paired_comparison(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    reference_by_id = {task_key(item): bool(item["correct"]) for item in reference["results"]}
    candidate_by_id = {task_key(item): bool(item["correct"]) for item in candidate["results"]}
    shared = sorted(reference_by_id.keys() & candidate_by_id.keys())
    gains = sum(not reference_by_id[key] and candidate_by_id[key] for key in shared)
    losses = sum(reference_by_id[key] and not candidate_by_id[key] for key in shared)
    both_correct = sum(reference_by_id[key] and candidate_by_id[key] for key in shared)
    return {
        "reference": reference["baseline"],
        "candidate": candidate["baseline"],
        "shared_tasks": len(shared),
        "net_correct_delta": gains - losses,
        "gains": gains,
        "losses": losses,
        "both_correct": both_correct,
    }


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage 1 baseline comparison",
        "",
        "| Baseline | EX | 95% CI | Executable | Avg model calls | Avg SQL exec | Avg total tokens | Budget violations |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["baselines"]:
        low, high = row["accuracy_ci95"]
        usage = row["average_usage"]
        lines.append(
            f"| {row['baseline']} | {row['execution_accuracy']:.1%} | "
            f"[{low:.1%}, {high:.1%}] | {row['executable_rate']:.1%} | "
            f"{usage['model_calls']:.2f} | {usage['sql_executions']:.2f} | "
            f"{usage['total_tokens']:.0f} | {row['budget_violations']} |"
        )
    lines.extend(["", f"Paired against `{payload['reference']}`:", ""])
    for pair in payload["paired"]:
        lines.append(
            f"- `{pair['candidate']}`: +{pair['gains']} / -{pair['losses']} task flips; "
            f"net {pair['net_correct_delta']:+d} correct over {pair['shared_tasks']} shared tasks."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    reports = [load_report(path) for path in args.reports]
    by_name = {str(report["baseline"]): report for report in reports}
    if args.reference not in by_name:
        raise ValueError(f"Reference baseline not found: {args.reference}")
    reference = by_name[args.reference]
    payload = {
        "reference": args.reference,
        "baselines": [aggregate(report) for report in reports],
        "paired": [
            paired_comparison(reference, report)
            for report in reports
            if report["baseline"] != args.reference
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "comparison.md").write_text(markdown(payload), encoding="utf-8")
    print(markdown(payload))


if __name__ == "__main__":
    main()
