#!/usr/bin/env python3
"""Create database-disjoint Dataset V2 splits with multi-dimensional balance."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data/generated/stratified_v2/tasks.jsonl"
DEFAULT_FROZEN = PROJECT_ROOT / "data/processed/five_tool_sft_native_v2/val_manifest.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/processed/stratified_v2"
SPLITS = ("train", "dev", "test")
DIMENSIONS = (
    "scenario_type",
    "drift_type",
    "interaction_profile",
    "difficulty",
    "failure_mode",
    "source",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def counts(rows: list[dict[str, Any]]) -> dict[str, Counter[str]]:
    return {
        key: Counter(str(row[key]) for row in rows)
        for key in DIMENSIONS
    }


def objective(
    split_rows: dict[str, list[dict[str, Any]]],
    fractions: dict[str, float],
    global_counts: dict[str, Counter[str]],
    total: int,
) -> float:
    score = 0.0
    for split in SPLITS:
        expected_rows = total * fractions[split]
        score += 4.0 * ((len(split_rows[split]) - expected_rows) / max(expected_rows, 1)) ** 2
        observed = counts(split_rows[split])
        for dimension in DIMENSIONS:
            for label, global_count in global_counts[dimension].items():
                expected = global_count * fractions[split]
                actual = observed[dimension][label]
                score += ((actual - expected) / max(expected, 3)) ** 2
    return score


def assign_databases(
    rows: list[dict[str, Any]],
    *,
    frozen_test_databases: set[str],
    fractions: dict[str, float],
    seed: int,
    trials: int,
) -> tuple[dict[str, list[dict[str, Any]]], float]:
    by_database: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_database[str(row["db_id"])].append(row)
    global_counts = counts(rows)
    best: dict[str, list[dict[str, Any]]] | None = None
    best_score = float("inf")
    movable = [db_id for db_id in by_database if db_id not in frozen_test_databases]

    for trial in range(trials):
        rng = random.Random(seed + trial)
        ordered = list(movable)
        rng.shuffle(ordered)
        ordered.sort(key=lambda db_id: len(by_database[db_id]), reverse=True)
        split_rows = {split: [] for split in SPLITS}
        for db_id in sorted(frozen_test_databases):
            split_rows["test"].extend(by_database.get(db_id, []))
        for db_id in ordered:
            group = by_database[db_id]
            candidate_scores = []
            for split in ("train", "dev", "test"):
                candidate = {key: list(value) for key, value in split_rows.items()}
                candidate[split].extend(group)
                candidate_scores.append(
                    (
                        objective(candidate, fractions, global_counts, len(rows)),
                        rng.random(),
                        split,
                    )
                )
            selected = min(candidate_scores)[2]
            split_rows[selected].extend(group)
        score = objective(split_rows, fractions, global_counts, len(rows))
        if score < best_score:
            best = split_rows
            best_score = score
    assert best is not None
    return best, best_score


def describe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "databases": len({str(row["db_id"]) for row in rows}),
        "dimensions": {
            key: dict(sorted(Counter(str(row[key]) for row in rows).items()))
            for key in DIMENSIONS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--frozen-manifest", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--dev-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=200)
    args = parser.parse_args()
    test_fraction = 1.0 - args.train_fraction - args.dev_fraction
    if min(args.train_fraction, args.dev_fraction, test_fraction) <= 0:
        parser.error("train/dev/test fractions must all be positive")

    rows = load_jsonl(args.input)
    frozen = load_jsonl(args.frozen_manifest)
    frozen_ids = {str(row["task_id"]) for row in frozen}
    frozen_databases = {str(row["db_id"]) for row in frozen}
    split_rows, score = assign_databases(
        rows,
        frozen_test_databases=frozen_databases,
        fractions={"train": args.train_fraction, "dev": args.dev_fraction, "test": test_fraction},
        seed=args.seed,
        trials=args.trials,
    )

    split_databases = {
        split: {str(row["db_id"]) for row in values}
        for split, values in split_rows.items()
    }
    overlaps = {
        "train_dev": sorted(split_databases["train"] & split_databases["dev"]),
        "train_test": sorted(split_databases["train"] & split_databases["test"]),
        "dev_test": sorted(split_databases["dev"] & split_databases["test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"database leakage: {overlaps}")
    test_ids = {str(row["task_id"]) for row in split_rows["test"]}
    missing_frozen = sorted(frozen_ids - test_ids)
    if missing_frozen:
        raise RuntimeError(f"frozen regression tasks missing from test: {missing_frozen[:5]}")

    for split, values in split_rows.items():
        values.sort(key=lambda row: str(row["task_id"]))
        write_jsonl(args.output_dir / f"{split}.jsonl", values)
    frozen_rows = [row for row in split_rows["test"] if str(row["task_id"]) in frozen_ids]
    write_jsonl(args.output_dir / "frozen_regression_78.jsonl", frozen_rows)
    result = {
        "name": "driftsql_stratified_dataset_v2_split",
        "seed": args.seed,
        "split_unit": "db_id",
        "target_fractions": {
            "train": args.train_fraction,
            "dev": args.dev_fraction,
            "test": test_fraction,
        },
        "balance_objective": score,
        "splits": {split: describe(values) for split, values in split_rows.items()},
        "frozen_regression": {
            "rows": len(frozen_rows),
            "databases": len(frozen_databases),
        },
        "database_overlap": overlaps,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
