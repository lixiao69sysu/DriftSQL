#!/usr/bin/env python3
"""Build a deterministic, executable SIX-GYM slice for BIRD-RL smoke runs."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def schema_after_preprocess(row: dict, database_root: Path, compact: bool = True) -> str:
    db_id = row["db_id"]
    template = database_root / db_id / f"{db_id}_template.sqlite"
    if not template.is_file():
        raise FileNotFoundError(f"Missing template database: {template}")

    source = sqlite3.connect(f"file:{template}?mode=ro", uri=True)
    working = sqlite3.connect(":memory:")
    try:
        source.backup(working)
        for statement in row.get("preprocess_sql", []):
            working.execute(statement)
        working.commit()
        definitions = working.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type IN ('table', 'view') AND sql IS NOT NULL "
            "ORDER BY type, name"
        ).fetchall()
        if compact:
            sql_context = "\n".join(
                str(sql)
                for field in ("issue_sql", "sol_sql", "preprocess_sql")
                for sql in (row.get(field) or [])
            )
            selected = [
                (name, definition)
                for name, definition in definitions
                if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", sql_context, re.I)
            ]
            if selected:
                definitions = selected
        return "\n\n".join(definition.rstrip(";") + ";" for _, definition in definitions)
    finally:
        working.close()
        source.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data/raw/six-gym-sqlite/train.jsonl",
    )
    parser.add_argument(
        "--database-root",
        type=Path,
        default=PROJECT_ROOT / "data/raw/six-gym-sqlite/database",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed/bird_rl_baseline/raw",
    )
    parser.add_argument("--db-id", default="book_publishing_company")
    parser.add_argument("--train-samples", type=int, default=64)
    parser.add_argument("--val-samples", type=int, default=8)
    parser.add_argument(
        "--full-schema",
        action="store_true",
        help="Keep every table. The smoke default keeps only SQL-referenced tables.",
    )
    args = parser.parse_args()

    required = args.train_samples + args.val_samples
    candidates = [
        row
        for row in load_jsonl(args.input)
        if row.get("db_id") == args.db_id
        and row.get("query")
        and row.get("issue_sql")
        and row.get("sol_sql")
        and row.get("test_cases")
    ]
    if len(candidates) < required:
        raise SystemExit(f"Need {required} valid {args.db_id} rows, found {len(candidates)}")

    # Stable ordering makes the exact smoke dataset reproducible across machines.
    selected: list[dict] = []
    schema_texts: list[str] = []
    for row in sorted(candidates, key=lambda item: item["instance_id"]):
        try:
            schema_text = schema_after_preprocess(row, args.database_root, compact=not args.full_schema)
        except sqlite3.Error:
            # SIX-GYM contains a few cross-dialect setup statements even in its
            # SQLite release. They cannot form an executable SQLite smoke case.
            continue
        selected.append(row)
        schema_texts.append(schema_text)
        if len(selected) == required:
            break
    if len(selected) < required:
        raise SystemExit(f"Need {required} executable {args.db_id} rows, found {len(selected)}")

    schemas = [
        {
            "instance_id": row["instance_id"],
            "instance_idx": index,
            "after_preprocess_schema": schema_texts[index],
        }
        for index, row in enumerate(selected)
    ]

    train_rows = selected[: args.train_samples]
    val_rows = selected[args.train_samples :]
    train_schemas = schemas[: args.train_samples]
    val_schemas = schemas[args.train_samples :]

    write_jsonl(args.output_dir / "train.jsonl", train_rows)
    write_jsonl(args.output_dir / "train_schema.jsonl", train_schemas)
    write_jsonl(args.output_dir / "val.jsonl", val_rows)
    write_jsonl(args.output_dir / "val_schema.jsonl", val_schemas)

    report = {
        "database": args.db_id,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_instances": [row["instance_id"] for row in train_rows],
        "val_instances": [row["instance_id"] for row in val_rows],
        "schema_mode": "full" if args.full_schema else "sql_referenced_tables",
        "schema_characters": {
            "min": min(map(len, schema_texts)),
            "max": max(map(len, schema_texts)),
        },
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
