#!/usr/bin/env python3
"""Build a balanced, execution-verified schema-drift dataset from BIRD23."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Callable

from driftsql.drift import (
    DriftExample,
    build_add_column_star_example,
    build_column_rename_example,
    build_column_replacement_example,
    build_table_rename_example,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = (
    PROJECT_ROOT
    / "data/raw/bird23-train-filtered/data/train-00000-of-00001.jsonl"
)
DEFAULT_DATABASES = (
    PROJECT_ROOT
    / "data/raw/bird23-train-filtered/full/train/train_databases"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data/generated/schema_drift/train.jsonl"
DEFAULT_SIX_TASKS = PROJECT_ROOT / "data/raw/six-gym-sqlite/train.jsonl"
DEFAULT_SIX_DATABASES = PROJECT_ROOT / "data/raw/six-gym-sqlite/database"

BUILDERS: dict[str, Callable[..., DriftExample]] = {
    "add_column": build_add_column_star_example,
    "rename_column": build_column_rename_example,
    "rename_table": build_table_rename_example,
    "replace_column": build_column_replacement_example,
}


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
    parser.add_argument("--six-gym-tasks", type=Path, default=DEFAULT_SIX_TASKS)
    parser.add_argument(
        "--six-gym-database-root",
        type=Path,
        default=DEFAULT_SIX_DATABASES,
    )
    parser.add_argument("--per-type", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-scan", type=int, default=4_000)
    parser.add_argument(
        "--types",
        nargs="+",
        choices=sorted(BUILDERS),
        default=sorted(BUILDERS),
    )
    args = parser.parse_args()
    if args.per_type <= 0 or args.max_scan <= 0:
        parser.error("--per-type and --max-scan must be positive")

    indexed_tasks = list(enumerate(_records(args.tasks)))
    random.Random(args.seed).shuffle(indexed_tasks)
    examples: dict[str, list[dict]] = {drift_type: [] for drift_type in args.types}
    rejected: dict[str, Counter[str]] = {
        drift_type: Counter() for drift_type in args.types
    }
    scanned = 0

    # BIRD's filtered text-to-SQL set intentionally contains no usable
    # single-table SELECT-star query. SIX-GYM provides executable Gold SQL over
    # real template databases, so it supplies this silent-drift slice.
    if "add_column" in args.types:
        for source_index, row in enumerate(_records(args.six_gym_tasks)):
            if len(examples["add_column"]) >= args.per_type:
                break
            solution_sql = row.get("sol_sql", [])
            if (
                str(row.get("dialect", "")).casefold() != "sqlite"
                or not isinstance(solution_sql, list)
                or len(solution_sql) != 1
                or row.get("preprocess_sql")
                or row.get("clean_up_sql")
            ):
                continue
            db_id = str(row["db_id"])
            database = (
                args.six_gym_database_root
                / db_id
                / f"{db_id}_template.sqlite"
            )
            try:
                example = build_add_column_star_example(
                    source="six_gym_sqlite",
                    source_index=source_index,
                    db_id=db_id,
                    question=str(row["query"]),
                    evidence="",
                    sql=str(solution_sql[0]),
                    database=database,
                )
            except Exception as error:  # noqa: BLE001 - rejections are data.
                rejected["add_column"][f"{type(error).__name__}: {error}"] += 1
                continue
            examples["add_column"].append(example.to_dict())
            print(
                f"[add_column {len(examples['add_column']):04d}/{args.per_type:04d}] "
                f"{example.db_id} {example.task_id}"
            )

    for source_index, row in indexed_tasks[: args.max_scan]:
        scanned += 1
        database = (
            args.database_root
            / str(row["db_id"])
            / f"{row['db_id']}.sqlite"
        )
        for drift_type in args.types:
            if drift_type == "add_column":
                continue
            if len(examples[drift_type]) >= args.per_type:
                continue
            try:
                example = BUILDERS[drift_type](
                    source="bird23_train_filtered",
                    source_index=source_index,
                    db_id=str(row["db_id"]),
                    question=str(row["question"]),
                    evidence=str(row.get("evidence", "")),
                    sql=str(row["SQL"]),
                    database=database,
                )
            except Exception as error:  # noqa: BLE001 - rejections are data.
                rejected[drift_type][f"{type(error).__name__}: {error}"] += 1
                continue
            examples[drift_type].append(example.to_dict())
            print(
                f"[{drift_type} {len(examples[drift_type]):04d}/{args.per_type:04d}] "
                f"{example.db_id} {example.task_id}"
            )
        if all(len(rows) >= args.per_type for rows in examples.values()):
            break

    missing = {
        drift_type: args.per_type - len(rows)
        for drift_type, rows in examples.items()
        if len(rows) < args.per_type
    }
    if missing:
        top_rejections = {
            drift_type: counts.most_common(5)
            for drift_type, counts in rejected.items()
        }
        raise RuntimeError(
            f"Could not fill balanced targets; missing={missing}; "
            f"top_rejections={top_rejections}"
        )

    # Interleave types so small prefixes remain balanced.
    combined = [
        row
        for index in range(args.per_type)
        for drift_type in args.types
        for row in (examples[drift_type][index],)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in combined),
        encoding="utf-8",
    )
    temporary.replace(args.output)

    summary = {
        "output": str(args.output.resolve()),
        "examples": len(combined),
        "examples_by_type": {
            drift_type: len(rows) for drift_type, rows in examples.items()
        },
        "databases": len({row["db_id"] for row in combined}),
        "databases_by_type": {
            drift_type: len({row["db_id"] for row in rows})
            for drift_type, rows in examples.items()
        },
        "validated": True,
        "materialized_databases_retained": 0,
        "seed": args.seed,
        "source_rows_scanned": scanned,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
