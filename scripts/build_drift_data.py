#!/usr/bin/env python3
"""Build execution-verified column-rename trajectories from BIRD23."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from driftsql.drift.factory import build_column_rename_example

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "bird23-train-filtered"
    / "data"
    / "train-00000-of-00001.jsonl"
)
DEFAULT_DATABASES = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "bird23-train-filtered"
    / "full"
    / "train"
    / "train_databases"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "column_rename"
    / "train.jsonl"
)


def _records(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--database-root", type=Path, default=DEFAULT_DATABASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministically shuffle source rows to improve database diversity.",
    )
    parser.add_argument(
        "--max-scan",
        type=int,
        default=2_000,
        help="Maximum source rows to inspect while looking for valid examples.",
    )
    args = parser.parse_args()
    if args.limit <= 0 or args.max_scan <= 0:
        parser.error("--limit and --max-scan must be positive")

    indexed_tasks = list(enumerate(_records(args.tasks)))
    random.Random(args.seed).shuffle(indexed_tasks)
    examples: list[dict] = []
    rejected: dict[str, int] = {}
    scanned = 0
    for source_index, row in indexed_tasks[: args.max_scan]:
        scanned += 1
        database = (
            args.database_root
            / str(row["db_id"])
            / f"{row['db_id']}.sqlite"
        )
        try:
            example = build_column_rename_example(
                source="bird23_train_filtered",
                source_index=source_index,
                db_id=str(row["db_id"]),
                question=str(row["question"]),
                evidence=str(row.get("evidence", "")),
                sql=str(row["SQL"]),
                database=database,
            )
        except Exception as error:  # noqa: BLE001 - rejection reasons are data.
            reason = f"{type(error).__name__}: {error}"
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        examples.append(example.to_dict())
        print(
            f"[{len(examples):04d}/{args.limit:04d}] "
            f"{example.db_id} {example.task_id}"
        )
        if len(examples) >= args.limit:
            break

    if len(examples) < args.limit:
        top_rejections = sorted(
            rejected.items(),
            key=lambda item: (-item[1], item[0]),
        )[:10]
        raise RuntimeError(
            f"Built only {len(examples)} of {args.limit} requested examples. "
            f"Top rejection reasons: {top_rejections}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples),
        encoding="utf-8",
    )
    temporary.replace(args.output)
    summary = {
        "output": str(args.output.resolve()),
        "examples": len(examples),
        "databases": len({row["db_id"] for row in examples}),
        "drift_type": "rename_column",
        "validated": True,
        "materialized_databases_retained": 0,
        "seed": args.seed,
        "source_rows_scanned": scanned,
        "rejected_rows": scanned - len(examples),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
