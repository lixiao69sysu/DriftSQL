#!/usr/bin/env python3
"""Build database-disjoint, execution-verified BIRD reasoning SFT data."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from driftsql.data.reasoning import (
    build_reasoning_messages,
    read_schema_objects,
    select_schema_context,
    validate_gold_sql,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "data/raw/bird23-train-filtered/data/train-00000-of-00001.jsonl"
DEFAULT_DB_ROOT = PROJECT_ROOT / "data/raw/bird23-train-filtered/full/train/train_databases"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/processed/reasoning_sft"
DEFAULT_TOKENIZER = PROJECT_ROOT / "models/Qwen2.5-Coder-3B-Instruct"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No accepted rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary)
    temporary.replace(path)


def stable_row_score(row: dict[str, Any]) -> str:
    identity = f"{row['db_id']}|{row['question']}|{row['SQL']}"
    return hashlib.sha256(identity.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--database-root", type=Path, default=DEFAULT_DB_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--val-db-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--max-result-rows", type=int, default=100)
    parser.add_argument("--max-schema-chars", type=int, default=10_000)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if not 0 < args.val_db_fraction < 1:
        parser.error("--val-db-fraction must be between 0 and 1")

    source_rows = read_jsonl(args.source)
    if args.limit > 0:
        source_rows = sorted(source_rows, key=stable_row_score)[: args.limit]
    database_ids = sorted({str(row["db_id"]) for row in source_rows})
    random.Random(args.seed).shuffle(database_ids)
    val_count = max(1, round(len(database_ids) * args.val_db_fraction))
    val_databases = set(database_ids[:val_count])

    schema_cache: dict[str, list[tuple[str, str]]] = {}
    database_paths: dict[str, Path] = {}
    for db_id in sorted({str(row["db_id"]) for row in source_rows}):
        database = args.database_root / db_id / f"{db_id}.sqlite"
        if not database.is_file():
            raise FileNotFoundError(database)
        database_paths[db_id] = database.resolve()
        schema_cache[db_id] = read_schema_objects(database)

    validations: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                validate_gold_sql,
                database_paths[str(row["db_id"])],
                str(row["SQL"]),
                timeout_seconds=args.timeout_seconds,
                max_rows=args.max_result_rows,
            ): index
            for index, row in enumerate(source_rows)
        }
        for future in as_completed(futures):
            validations[futures[future]] = future.result()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    accepted: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    manifests: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    rejected: list[dict[str, Any]] = []
    schema_modes: Counter[str] = Counter()
    token_lengths: list[int] = []

    for index, row in enumerate(source_rows):
        db_id = str(row["db_id"])
        validation = validations[index]
        if not validation["success"]:
            rejected.append({"source_index": index, "db_id": db_id, "validation": validation})
            continue
        try:
            schema, schema_meta = select_schema_context(
                schema_cache[db_id],
                question=str(row["question"]),
                evidence=str(row.get("evidence", "")),
                gold_sql=str(row["SQL"]),
                max_chars=args.max_schema_chars,
            )
            messages = build_reasoning_messages(
                question=str(row["question"]),
                evidence=str(row.get("evidence", "")),
                schema=schema,
                gold_sql=str(row["SQL"]),
            )
        except Exception as error:
            rejected.append(
                {"source_index": index, "db_id": db_id, "validation": {"success": False, "stage": "build", "error": str(error)}}
            )
            continue

        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        token_count = len(token_ids)
        if token_count > args.max_tokens:
            rejected.append(
                {
                    "source_index": index,
                    "db_id": db_id,
                    "validation": {
                        "success": False,
                        "stage": "token_budget",
                        "error": f"{token_count} > {args.max_tokens}",
                    },
                }
            )
            continue

        split = "val" if db_id in val_databases else "train"
        accepted[split].append({"messages": messages, "enable_thinking": False})
        schema_modes[str(schema_meta["mode"])] += 1
        token_lengths.append(token_count)
        manifests[split].append(
            {
                "source_index": index,
                "db_id": db_id,
                "db_path": str(database_paths[db_id]),
                "question": str(row["question"]),
                "evidence": str(row.get("evidence", "")),
                "gold_sql": str(row["SQL"]),
                "plan": messages[-1]["content"],
                "token_count": token_count,
                "schema": schema_meta,
                "validation": validation,
            }
        )

    train_dbs = {row["db_id"] for row in manifests["train"]}
    final_val_dbs = {row["db_id"] for row in manifests["val"]}
    overlap = sorted(train_dbs & final_val_dbs)
    if overlap:
        raise RuntimeError(f"Database leakage: {overlap}")

    for split in ("train", "val"):
        write_parquet(args.output_dir / f"{split}.parquet", accepted[split])
        write_jsonl(args.output_dir / f"{split}_manifest.jsonl", manifests[split])
    write_jsonl(args.output_dir / "rejected.jsonl", rejected)

    rejection_stages = Counter(item["validation"].get("stage", "unknown") for item in rejected)
    summary = {
        "name": "bird23_execution_verified_reasoning_sft_v1",
        "source": str(args.source.resolve()),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "split_unit": "db_id",
        "seed": args.seed,
        "source_rows": len(source_rows),
        "accepted_rows": sum(len(rows) for rows in accepted.values()),
        "rejected_rows": len(rejected),
        "rejection_stages": dict(sorted(rejection_stages.items())),
        "splits": {
            split: {
                "rows": len(accepted[split]),
                "databases": len({row["db_id"] for row in manifests[split]}),
            }
            for split in ("train", "val")
        },
        "database_overlap": overlap,
        "schema_modes": dict(sorted(schema_modes.items())),
        "token_length": {
            "min": min(token_lengths),
            "median": statistics.median(token_lengths),
            "p95": sorted(token_lengths)[int(0.95 * (len(token_lengths) - 1))],
            "max": max(token_lengths),
            "budget": args.max_tokens,
        },
        "execution": {
            "timeout_seconds": args.timeout_seconds,
            "max_observed_rows": args.max_result_rows,
            "read_only": True,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
