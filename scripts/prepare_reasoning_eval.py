#!/usr/bin/env python3
"""Freeze a paired, database-stratified Reasoning SFT validation set."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from driftsql.evaluation.reasoning import stratified_database_sample


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARQUET = PROJECT_ROOT / "data/processed/reasoning_sft/val.parquet"
DEFAULT_MANIFEST = PROJECT_ROOT / "data/processed/reasoning_sft/val_manifest.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/evaluation/stage3_reasoning"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--size", type=int, default=128)
    args = parser.parse_args()

    records = pq.read_table(args.parquet).to_pylist()
    metadata = load_jsonl(args.manifest)
    if len(records) != len(metadata):
        raise ValueError(f"Parquet/manifest length mismatch: {len(records)} != {len(metadata)}")

    rows: list[dict[str, Any]] = []
    for record, meta in zip(records, metadata, strict=True):
        messages = list(record["messages"])
        if len(messages) != 3 or messages[-1]["role"] != "assistant":
            raise ValueError("Expected system, user, assistant Reasoning SFT messages")
        rows.append(
            {
                "instance_idx": int(meta["source_index"]),
                "source_index": int(meta["source_index"]),
                "db_id": str(meta["db_id"]),
                "db_path": str(meta["db_path"]),
                "question": str(meta["question"]),
                "evidence": str(meta.get("evidence", "")),
                "gold_sql": str(meta["gold_sql"]),
                "messages": messages[:2],
                "training_token_count": int(meta["token_count"]),
            }
        )
    selected = stratified_database_sample(rows, args.size)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"reasoning_val_{len(selected)}.jsonl"
    write_jsonl(output, selected)
    distribution = Counter(row["db_id"] for row in selected)
    protocol = {
        "name": "stage3_reasoning_base_vs_lora_v1",
        "tasks": len(selected),
        "databases": len(distribution),
        "database_distribution": dict(sorted(distribution.items())),
        "selection": "stable-hash round-robin by db_id",
        "source_parquet_sha256": hashlib.sha256(args.parquet.read_bytes()).hexdigest(),
        "source_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "task_file": str(output.resolve()),
        "decoding": {
            "temperature": 0.0,
            "max_new_tokens": 768,
            "max_model_length": 4096,
        },
        "metrics": [
            "set-normalized execution accuracy",
            "SQL executable rate",
            "plan/sql wrapper compliance",
            "paired task flips",
        ],
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
