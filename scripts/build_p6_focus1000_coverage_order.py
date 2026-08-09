#!/usr/bin/env python3
"""Build a unique-task-first order for one complete Focus1000 GRPO epoch."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from driftsql.training.grpo_coverage import coverage_plan, unique_first_order


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "data/processed/p6_focus1000_reward_ab"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def prompt_hash(row: dict[str, Any]) -> str:
    payload = json.dumps(
        row["prompt"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    source_path = data_dir / "train.parquet"
    manifest_path = data_dir / "train_manifest.jsonl"
    output_path = data_dir / "train_coverage.parquet"
    output_manifest_path = data_dir / "train_coverage_manifest.jsonl"
    summary_path = data_dir / "coverage_summary.json"
    for path in (source_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    existing = [path for path in (output_path, output_manifest_path, summary_path) if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite coverage artifacts: {existing}")

    table = pq.read_table(source_path)
    rows = table.to_pylist()
    manifest = load_jsonl(manifest_path)
    if len(rows) != 1000 or len(manifest) != len(rows):
        raise RuntimeError(f"Expected aligned Focus1000 rows, got {len(rows)}/{len(manifest)}")
    instance_ids = [str(row["extra_info"]["instance_id"]) for row in rows]
    if any(instance_id != str(meta["task_id"]) for instance_id, meta in zip(instance_ids, manifest, strict=True)):
        raise RuntimeError("Focus1000 parquet and manifest are not row-aligned")

    order = unique_first_order(instance_ids, seed=args.seed)
    reordered_rows: list[dict[str, Any]] = []
    for coverage_index, source_index in enumerate(order):
        row = dict(rows[source_index])
        extra_info = dict(row["extra_info"])
        # RLHFDataset promotes extra_info.index to a batch field and also
        # forwards the full extra_info to the reward function. This gives the
        # post-run auditor an exact row identity even when hard replay repeats
        # the same instance_id and model-visible prompt.
        extra_info["index"] = coverage_index
        row["extra_info"] = extra_info
        reordered_rows.append(row)
    reordered_table = pa.Table.from_pylist(reordered_rows)
    reordered_manifest: list[dict[str, Any]] = []
    for coverage_index, source_index in enumerate(order):
        meta = dict(manifest[source_index])
        meta.update(
            {
                "coverage_index": coverage_index,
                "coverage_source_index": source_index,
                "coverage_seed": args.seed,
            }
        )
        reordered_manifest.append(meta)

    unique_tasks = len(set(instance_ids))
    prefix_ids = [str(row["extra_info"]["instance_id"]) for row in reordered_rows[:unique_tasks]]
    if len(set(prefix_ids)) != unique_tasks:
        raise RuntimeError("Unique-first prefix contains a repeated task")
    if Counter(instance_ids) != Counter(str(row["extra_info"]["instance_id"]) for row in reordered_rows):
        raise RuntimeError("Coverage ordering changed the task multiset")
    if sorted(prompt_hash(row) for row in rows) != sorted(prompt_hash(row) for row in reordered_rows):
        raise RuntimeError("Coverage ordering changed model-visible prompts")

    plan = coverage_plan(len(rows), args.train_batch_size, len(rows) // args.train_batch_size)
    if not plan.full_row_coverage:
        raise RuntimeError(f"Batch size cannot cover Focus1000 exactly: {plan.as_dict()}")
    summary = {
        "protocol": "p6_focus1000_unique_first_full_epoch_v1",
        "source": str(source_path),
        "output": str(output_path),
        "seed": args.seed,
        "train_rows": len(rows),
        "unique_tasks": unique_tasks,
        "duplicate_rows": len(rows) - unique_tasks,
        "unique_prompts": len({prompt_hash(row) for row in rows}),
        "first_duplicate_position": unique_tasks,
        "coverage_indices": [0, len(rows) - 1],
        "full_epoch": plan.as_dict(),
        "data_shuffle": False,
        "prompt_multiset_unchanged": True,
        "fresh_blind_rows_read": 0,
    }
    pq.write_table(reordered_table, output_path, compression="zstd")
    write_jsonl(output_manifest_path, reordered_manifest)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
