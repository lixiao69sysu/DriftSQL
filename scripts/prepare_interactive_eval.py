#!/usr/bin/env python3
"""Prepare VERL-compatible public Mini-Interact inference records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from driftsql.data import build_mini_interact_eval_record, load_mini_interact_rows


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data/raw/mini-interact",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/processed/mini_interact/interactive_eval.jsonl",
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rows = load_mini_interact_rows(args.data_root)
    if args.limit is not None:
        rows = rows[: args.limit]
    records = [
        build_mini_interact_eval_record(row, args.data_root, index=index)
        for index, row in enumerate(rows)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(args.output)
    summary = {
        "rows": len(records),
        "databases": len({record["extra_info"]["db_id"] for record in records}),
        "public_ground_truth_rows": sum(
            bool(record["extra_info"]["public_ground_truth_available"])
            for record in records
        ),
        "output": str(args.output.resolve()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
