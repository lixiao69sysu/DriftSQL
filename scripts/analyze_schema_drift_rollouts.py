#!/usr/bin/env python3
"""Aggregate VERL rollout rewards by schema-drift operation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--success-threshold", type=float, default=1.0)
    return parser.parse_args()


def operation_type(row: dict) -> str:
    operations = row["extra_info"]["schema_diff"]["operations"]
    return operations[0]["type"] if operations else "unknown"


def main() -> None:
    args = parse_args()
    data_rows = pq.read_table(args.data).to_pylist()
    prompt_index = [
        (
            row["prompt"][-1]["content"],
            row["reward_model"]["ground_truth"],
            operation_type(row),
            row["extra_info"]["instance_id"],
        )
        for row in data_rows
    ]

    records: list[dict] = []
    unmatched = 0
    for rollout_path in sorted(args.rollout_dir.glob("*.jsonl"), key=lambda p: int(p.stem)):
        for line in rollout_path.read_text().splitlines():
            rollout = json.loads(line)
            matches = [
                entry
                for entry in prompt_index
                if entry[0] in rollout["input"] and entry[1] == rollout["gts"]
            ]
            if not matches:
                unmatched += 1
                continue
            _, _, operation, instance_id = max(matches, key=lambda entry: len(entry[0]))
            score = float(rollout["score"])
            records.append(
                {
                    "step": int(rollout["step"]),
                    "uid": rollout["uid"],
                    "instance_id": instance_id,
                    "operation": operation,
                    "score": score,
                    "success": score >= args.success_threshold,
                }
            )

    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["operation"]].append(record)

    def summarize(items: list[dict]) -> dict:
        return {
            "rollouts": len(items),
            "mean_score": sum(item["score"] for item in items) / len(items) if items else 0.0,
            "successes": sum(item["success"] for item in items),
            "success_rate": sum(item["success"] for item in items) / len(items) if items else 0.0,
        }

    report = {
        "overall": summarize(records),
        "by_operation": {key: summarize(value) for key, value in sorted(grouped.items())},
        "unmatched_rollouts": unmatched,
        "records": records,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
