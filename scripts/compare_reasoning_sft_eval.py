#!/usr/bin/env python3
"""Paired analysis for the Stage-3 Base-vs-Reasoning-LoRA gate."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sqlglot import exp, parse_one


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def key(row: dict[str, Any]) -> str:
    return str(row["instance_idx"])


def mcnemar_exact_p(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    smaller = min(gains, losses)
    tail = sum(math.comb(discordant, value) for value in range(smaller + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def failure_type(row: dict[str, Any]) -> str:
    if row.get("correct"):
        return "correct"
    if row.get("pred_executable"):
        return "wrong_result"
    error = str(row.get("error", "")).casefold()
    if "no such column" in error:
        return "no_such_column"
    if "no such table" in error:
        return "no_such_table"
    if "ambiguous column" in error:
        return "ambiguous_column"
    if "syntax" in error:
        return "syntax"
    return "other_execution"


def sql_features(sql: str) -> list[str]:
    tree = parse_one(sql, read="sqlite")
    features: list[str] = []
    if list(tree.find_all(exp.Join)):
        features.append("join")
    if tree.find(exp.Group):
        features.append("group")
    if any(isinstance(node, exp.AggFunc) for node in tree.walk()):
        features.append("aggregate")
    if tree.find(exp.Order):
        features.append("order")
    if tree.find(exp.Subquery):
        features.append("subquery")
    return features or ["simple"]


def summary(report: dict[str, Any]) -> dict[str, Any]:
    rows = report["results"]
    total = len(rows)
    return {
        "baseline": report["baseline"],
        "tasks": total,
        "correct": sum(bool(row["correct"]) for row in rows),
        "execution_accuracy": sum(bool(row["correct"]) for row in rows) / total,
        "executable": sum(bool(row["pred_executable"]) for row in rows),
        "executable_rate": sum(bool(row["pred_executable"]) for row in rows) / total,
        "exact_wrapper": sum(bool(row.get("format", {}).get("exact_wrapper")) for row in rows) / total,
        "average_new_tokens": sum(int(row["usage"]["new_tokens"]) for row in rows) / total,
        "failure_types": dict(sorted(Counter(failure_type(row) for row in rows).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    reference = load(args.reference)
    candidate = load(args.candidate)
    ref = {key(row): row for row in reference["results"]}
    cand = {key(row): row for row in candidate["results"]}
    shared = sorted(ref.keys() & cand.keys(), key=int)
    if len(shared) != len(ref) or len(shared) != len(cand):
        raise ValueError("Reports do not contain the same paired task set")
    gains = [task for task in shared if not ref[task]["correct"] and cand[task]["correct"]]
    losses = [task for task in shared if ref[task]["correct"] and not cand[task]["correct"]]

    databases: dict[str, list[str]] = defaultdict(list)
    for task in shared:
        databases[str(ref[task]["db_id"])].append(task)
    per_database = []
    for db_id, tasks in sorted(databases.items()):
        ref_correct = sum(bool(ref[task]["correct"]) for task in tasks)
        cand_correct = sum(bool(cand[task]["correct"]) for task in tasks)
        per_database.append(
            {
                "db_id": db_id,
                "tasks": len(tasks),
                "reference_correct": ref_correct,
                "candidate_correct": cand_correct,
                "delta": cand_correct - ref_correct,
            }
        )

    feature_tasks: dict[str, list[str]] = defaultdict(list)
    for task in shared:
        for feature in sql_features(str(ref[task]["gold_sql"])):
            feature_tasks[feature].append(task)
    feature_slices = []
    for feature, tasks in sorted(feature_tasks.items()):
        ref_correct = sum(bool(ref[task]["correct"]) for task in tasks)
        cand_correct = sum(bool(cand[task]["correct"]) for task in tasks)
        feature_slices.append(
            {
                "feature": feature,
                "tasks": len(tasks),
                "reference_correct": ref_correct,
                "candidate_correct": cand_correct,
                "delta": cand_correct - ref_correct,
            }
        )

    payload = {
        "protocol": "stage3_reasoning_base_vs_lora_v1",
        "reference": summary(reference),
        "candidate": summary(candidate),
        "paired": {
            "shared_tasks": len(shared),
            "gains": len(gains),
            "losses": len(losses),
            "net_correct_delta": len(gains) - len(losses),
            "mcnemar_exact_p": mcnemar_exact_p(len(gains), len(losses)),
            "gain_task_ids": gains,
            "loss_task_ids": losses,
        },
        "per_database": per_database,
        "feature_slices": feature_slices,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "reasoning_comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    reference_summary = payload["reference"]
    candidate_summary = payload["candidate"]
    lines = [
        "# Stage 3 Reasoning SFT paired comparison",
        "",
        "| Model | EX | Executable | Exact wrapper | Avg new tokens |",
        "|---|---:|---:|---:|---:|",
        f"| {reference_summary['baseline']} | {reference_summary['execution_accuracy']:.1%} | "
        f"{reference_summary['executable_rate']:.1%} | {reference_summary['exact_wrapper']:.1%} | "
        f"{reference_summary['average_new_tokens']:.1f} |",
        f"| {candidate_summary['baseline']} | {candidate_summary['execution_accuracy']:.1%} | "
        f"{candidate_summary['executable_rate']:.1%} | {candidate_summary['exact_wrapper']:.1%} | "
        f"{candidate_summary['average_new_tokens']:.1f} |",
        "",
        f"Paired flips: +{len(gains)} / -{len(losses)}; net {len(gains) - len(losses):+d}; "
        f"exact McNemar p={payload['paired']['mcnemar_exact_p']:.4f}.",
        "",
        "## SQL feature slices",
        "",
        "| Feature | Tasks | Base correct | SFT correct | Delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in feature_slices:
        lines.append(
            f"| {row['feature']} | {row['tasks']} | {row['reference_correct']} | "
            f"{row['candidate_correct']} | {row['delta']:+d} |"
        )
    (args.output_dir / "reasoning_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
