#!/usr/bin/env python3
"""Create database-disjoint VERL SFT and RL parquet files."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from driftsql.data import build_rl_record, build_sft_record

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "column_rename"
    / "train.jsonl"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "column_rename"


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_parquet(rows: list[dict], path: Path) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty parquet file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--val-db-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-turns", type=int, default=5)
    args = parser.parse_args()
    if not 0 < args.val_db_fraction < 1:
        parser.error("--val-db-fraction must be between 0 and 1")

    manifests = _read_jsonl(args.manifest)
    database_ids = sorted({str(row["db_id"]) for row in manifests})
    random.Random(args.seed).shuffle(database_ids)
    val_count = max(1, round(len(database_ids) * args.val_db_fraction))
    validation_databases = set(database_ids[:val_count])

    partitions: dict[str, list[dict]] = {"train": [], "val": []}
    for row in manifests:
        split = (
            "val"
            if str(row["db_id"]) in validation_databases
            else "train"
        )
        partitions[split].append(row)

    split_summary: dict[str, dict[str, Any]] = {}
    for split, rows in partitions.items():
        rl_rows = [
            build_rl_record(
                row,
                index=index,
                split=split,
                max_turns=args.max_turns,
            )
            for index, row in enumerate(rows)
        ]
        sft_rows = [
            build_sft_record(row, max_turns=args.max_turns) for row in rows
        ]
        _write_parquet(rl_rows, args.output_dir / f"rl_{split}.parquet")
        _write_parquet(sft_rows, args.output_dir / f"sft_{split}.parquet")
        split_summary[split] = {
            "rows": len(rows),
            "databases": sorted({str(row["db_id"]) for row in rows}),
        }

    train_databases = set(split_summary["train"]["databases"])
    val_databases = set(split_summary["val"]["databases"])
    overlap = sorted(train_databases & val_databases)
    if overlap:
        raise RuntimeError(f"Database leakage between train and val: {overlap}")

    summary = {
        "manifest": str(args.manifest.resolve()),
        "seed": args.seed,
        "split_unit": "db_id",
        "max_turns": args.max_turns,
        "splits": split_summary,
        "database_overlap": overlap,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
