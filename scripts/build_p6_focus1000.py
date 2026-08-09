#!/usr/bin/env python3
"""Build the deterministic Focus1000 curriculum for Reward V1/V2 A/B."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data/processed/p6_scaleup_v1_grpo"
DEFAULT_OUTPUT = ROOT / "data/processed/p6_focus1000_reward_ab"
FORBIDDEN_PROMPT_MARKERS = (
    '"ground_truth"',
    '"target_action"',
    '"decision_target_action"',
    '"failure_labels"',
    '"canonical_sql"',
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def prompt_bytes(row: dict[str, Any]) -> bytes:
    return json.dumps(
        row["prompt"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def choose(
    rng: random.Random,
    population: list[int],
    count: int,
    label: str,
) -> list[int]:
    if len(population) < count:
        raise RuntimeError(f"Focus1000 pool {label} has {len(population)} rows, needs {count}")
    return sorted(rng.sample(population, count))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()
    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    train_table = pq.read_table(source / "train.parquet")
    tune_table = pq.read_table(source / "tune.parquet")
    rows = train_table.to_pylist()
    tune_rows = tune_table.to_pylist()
    manifest = load_jsonl(source / "train_manifest.jsonl")
    if len(rows) != 3200 or len(manifest) != 3200 or len(tune_rows) != 432:
        raise RuntimeError(
            f"Expected source Train3200/Manifest3200/Tune432, got "
            f"{len(rows)}/{len(manifest)}/{len(tune_rows)}"
        )
    if any(
        str(row["extra_info"]["instance_id"]) != str(meta["task_id"])
        for row, meta in zip(rows, manifest, strict=True)
    ):
        raise RuntimeError("Source train parquet and manifest are not row-aligned")

    rng = random.Random(args.seed)
    selected: list[tuple[int, str]] = []

    def pool(predicate: Callable[[dict[str, Any], dict[str, Any]], bool]) -> list[int]:
        return [
            index
            for index, (row, meta) in enumerate(zip(rows, manifest, strict=True))
            if predicate(row, meta)
        ]

    add_roles = {
        "base_unique": 420,
        "successful_execute_no_submit": 80,
        "must_ask_error": 60,
        "compound_recovery": 21,
        "post_diff_wrong_retrieval": 19,
    }
    for role, quota in add_roles.items():
        indices = pool(
            lambda row, meta, role=role: row["extra_info"]["drift_type"]
            == "add_column"
            and meta["sampling_role"] == role
        )
        for index in choose(rng, indices, quota, f"add_column/{role}"):
            selected.append((index, f"add_column/{role}"))

    for role, quota in (("base_unique", 120), ("compound_recovery", 40)):
        indices = pool(
            lambda row, meta, role=role: row["extra_info"]["drift_type"] == "compound"
            and meta["sampling_role"] == role
        )
        for index in choose(rng, indices, quota, f"compound/{role}"):
            selected.append((index, f"compound/{role}"))

    for role, quota in (
        ("base_unique", 100),
        ("must_ask_error", 46),
        ("post_diff_wrong_retrieval", 14),
    ):
        indices = pool(
            lambda row, meta, role=role: row["extra_info"]["drift_type"]
            not in {"add_column", "compound"}
            and row["extra_info"]["interaction_profile"] == "must_ask"
            and meta["sampling_role"] == role
        )
        for index in choose(rng, indices, quota, f"must_ask/{role}"):
            selected.append((index, f"must_ask/{role}"))

    for drift_type in ("clean", "rename_table", "rename_column", "replace_column"):
        indices = pool(
            lambda row, meta, drift_type=drift_type: row["extra_info"]["drift_type"]
            == drift_type
            and row["extra_info"]["interaction_profile"] != "must_ask"
            and meta["sampling_role"] == "base_unique"
        )
        for index in choose(rng, indices, 20, f"other/{drift_type}"):
            selected.append((index, f"other/{drift_type}"))

    if len(selected) != 1000:
        raise RuntimeError(f"Expected Focus1000, selected {len(selected)}")
    source_indices = [index for index, _category in selected]
    if len(set(source_indices)) != 1000:
        duplicates = len(source_indices) - len(set(source_indices))
        raise RuntimeError(f"Focus categories overlap by {duplicates} source rows")
    rng.shuffle(selected)

    output_indices = [index for index, _category in selected]
    output_table = train_table.take(pa.array(output_indices, type=pa.int64()))
    output_rows = output_table.to_pylist()
    output_manifest = []
    for sampling_index, ((source_index, category), row) in enumerate(
        zip(selected, output_rows, strict=True)
    ):
        source_meta = dict(manifest[source_index])
        source_meta.update(
            {
                "focus_category": category,
                "focus_source_index": source_index,
                "focus_sampling_index": sampling_index,
                "prompt_sha256": hashlib.sha256(prompt_bytes(row)).hexdigest(),
            }
        )
        output_manifest.append(source_meta)

    if any(prompt_bytes(row) != prompt_bytes(rows[index]) for row, index in zip(output_rows, output_indices, strict=True)):
        raise RuntimeError("Focus1000 changed a model-visible prompt")
    if any(
        marker in prompt_bytes(row).decode("utf-8")
        for row in output_rows
        for marker in FORBIDDEN_PROMPT_MARKERS
    ):
        raise RuntimeError("Focus1000 prompt leakage marker detected")
    train_dbs = {str(row["extra_info"]["db_id"]) for row in output_rows}
    tune_dbs = {str(row["extra_info"]["db_id"]) for row in tune_rows}
    overlap = sorted(train_dbs & tune_dbs)
    if overlap:
        raise RuntimeError(f"Focus1000 Train/Tune database overlap: {overlap}")

    category_counts = Counter(category.split("/", 1)[0] for _, category in selected)
    expected_categories = {
        "add_column": 600,
        "compound": 160,
        "must_ask": 160,
        "other": 80,
    }
    if dict(category_counts) != expected_categories:
        raise RuntimeError(f"Focus1000 category mismatch: {dict(category_counts)}")

    output.mkdir(parents=True, exist_ok=False)
    pq.write_table(output_table, output / "train.parquet", compression="zstd")
    pq.write_table(tune_table, output / "tune.parquet", compression="zstd")
    write_jsonl(output / "train_manifest.jsonl", output_manifest)
    summary = {
        "protocol": "p6_focus1000_reward_ab_v1",
        "source": str(source),
        "seed": args.seed,
        "train_rows": len(output_rows),
        "tune_rows": len(tune_rows),
        "focus_categories": expected_categories,
        "focus_subcategories": dict(sorted(Counter(category for _, category in selected).items())),
        "drift_distribution": dict(
            sorted(Counter(str(row["extra_info"]["drift_type"]) for row in output_rows).items())
        ),
        "interaction_distribution": dict(
            sorted(
                Counter(
                    str(row["extra_info"]["interaction_profile"]) for row in output_rows
                ).items()
            )
        ),
        "unique_tasks": len(
            {str(row["extra_info"]["instance_id"]) for row in output_rows}
        ),
        "train_databases": len(train_dbs),
        "tune_databases": len(tune_dbs),
        "train_tune_database_overlap": overlap,
        "schema_matches_source": output_table.schema == train_table.schema,
        "prompt_bytes_unchanged": True,
        "prompt_target_leakage": False,
        "fresh_blind_rows_read": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
