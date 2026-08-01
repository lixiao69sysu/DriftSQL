#!/usr/bin/env python3
"""Create the Stage 6 DB-disjoint train/tune/gate protocol.

Only the historical Stage 5 *training* partition is eligible as input.  The
script also reads the historical dev/test partitions to prove that no held-out
database leaked into the new optimization protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "data/processed/stratified_v2/train.jsonl"
DEFAULT_HISTORICAL_DEV = PROJECT_ROOT / "data/processed/stratified_v2/dev.jsonl"
DEFAULT_HISTORICAL_TEST = PROJECT_ROOT / "data/processed/stratified_v2/test.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/processed/stage6_protocol"
SPLITS = ("train", "tune", "gate")
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


def task_digest(rows: list[dict[str, Any]]) -> str:
    task_ids = sorted(str(row["task_id"]) for row in rows)
    return hashlib.sha256("\n".join(task_ids).encode("utf-8")).hexdigest()


def dimension_counts(rows: list[dict[str, Any]]) -> dict[str, Counter[str]]:
    return {
        dimension: Counter(str(row[dimension]) for row in rows)
        for dimension in DIMENSIONS
    }


def split_score(
    split_rows: dict[str, list[dict[str, Any]]],
    fractions: dict[str, float],
    global_counts: dict[str, Counter[str]],
    total_rows: int,
) -> float:
    score = 0.0
    for split in SPLITS:
        expected_rows = total_rows * fractions[split]
        score += 8.0 * ((len(split_rows[split]) - expected_rows) / expected_rows) ** 2
        observed = dimension_counts(split_rows[split])
        for dimension in DIMENSIONS:
            for label, global_count in global_counts[dimension].items():
                expected = global_count * fractions[split]
                score += ((observed[dimension][label] - expected) / max(expected, 3.0)) ** 2
    return score


def assign_databases(
    rows: list[dict[str, Any]],
    *,
    fractions: dict[str, float],
    seed: int,
    trials: int,
) -> tuple[dict[str, list[dict[str, Any]]], float]:
    by_database: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_database[str(row["db_id"])].append(row)
    databases = sorted(by_database)
    if len(databases) < len(SPLITS):
        raise ValueError("At least three databases are required")

    # Exact DB counts make every split useful even when database sizes are
    # highly uneven.  Row and stratum balance is optimized across random
    # database assignments without ever splitting one database.
    train_count = round(len(databases) * fractions["train"])
    tune_count = round(len(databases) * fractions["tune"])
    train_count = min(max(train_count, 1), len(databases) - 2)
    tune_count = min(max(tune_count, 1), len(databases) - train_count - 1)
    db_counts = {
        "train": train_count,
        "tune": tune_count,
        "gate": len(databases) - train_count - tune_count,
    }

    global_counts = dimension_counts(rows)
    best_rows: dict[str, list[dict[str, Any]]] | None = None
    best_score = float("inf")
    for trial in range(trials):
        ordered = list(databases)
        random.Random(seed + trial).shuffle(ordered)
        train_end = db_counts["train"]
        tune_end = train_end + db_counts["tune"]
        assignment = {
            "train": ordered[:train_end],
            "tune": ordered[train_end:tune_end],
            "gate": ordered[tune_end:],
        }
        candidate = {
            split: [row for db_id in assignment[split] for row in by_database[db_id]]
            for split in SPLITS
        }
        score = split_score(candidate, fractions, global_counts, len(rows))
        if score < best_score:
            best_rows = candidate
            best_score = score
    assert best_rows is not None
    return best_rows, best_score


def describe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "databases": len({str(row["db_id"]) for row in rows}),
        "task_id_sha256": task_digest(rows),
        "dimensions": {
            dimension: dict(sorted(values.items()))
            for dimension, values in dimension_counts(rows).items()
        },
    }


def validate_protocol(
    source_rows: list[dict[str, Any]],
    split_rows: dict[str, list[dict[str, Any]]],
    historical_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    source_ids = {str(row["task_id"]) for row in source_rows}
    output_ids = {
        str(row["task_id"])
        for split in SPLITS
        for row in split_rows[split]
    }
    if output_ids != source_ids:
        raise RuntimeError("Stage 6 outputs do not exactly partition Stage 5 train task IDs")

    split_databases = {
        split: {str(row["db_id"]) for row in split_rows[split]}
        for split in SPLITS
    }
    overlaps = {
        "train_tune": sorted(split_databases["train"] & split_databases["tune"]),
        "train_gate": sorted(split_databases["train"] & split_databases["gate"]),
        "tune_gate": sorted(split_databases["tune"] & split_databases["gate"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Stage 6 database leakage: {overlaps}")

    historical_databases = {str(row["db_id"]) for row in historical_rows}
    leaked = sorted(set().union(*split_databases.values()) & historical_databases)
    if leaked:
        raise RuntimeError(f"Historical dev/test databases leaked into Stage 6: {leaked}")
    return {
        "database_overlap": overlaps,
        "historical_dev_test_database_overlap": leaked,
        "source_task_ids_preserved": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--historical-dev", type=Path, default=DEFAULT_HISTORICAL_DEV)
    parser.add_argument("--historical-test", type=Path, default=DEFAULT_HISTORICAL_TEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-fraction", type=float, default=0.72)
    parser.add_argument("--tune-fraction", type=float, default=0.14)
    parser.add_argument("--seed", type=int, default=62026)
    parser.add_argument("--trials", type=int, default=5000)
    args = parser.parse_args()

    gate_fraction = 1.0 - args.train_fraction - args.tune_fraction
    fractions = {
        "train": args.train_fraction,
        "tune": args.tune_fraction,
        "gate": gate_fraction,
    }
    if min(fractions.values()) <= 0:
        parser.error("train/tune/gate fractions must all be positive")
    if args.trials < 1:
        parser.error("trials must be positive")

    source_rows = load_jsonl(args.source)
    historical_rows = load_jsonl(args.historical_dev) + load_jsonl(args.historical_test)
    split_rows, score = assign_databases(
        source_rows,
        fractions=fractions,
        seed=args.seed,
        trials=args.trials,
    )
    validation = validate_protocol(source_rows, split_rows, historical_rows)
    for split in SPLITS:
        split_rows[split].sort(key=lambda row: str(row["task_id"]))
        write_jsonl(args.output_dir / f"{split}.jsonl", split_rows[split])

    summary = {
        "name": "driftsql_stage6_train_tune_gate_v1",
        "source": str(args.source.resolve()),
        "source_rows": len(source_rows),
        "source_task_id_sha256": task_digest(source_rows),
        "seed": args.seed,
        "trials": args.trials,
        "split_unit": "db_id",
        "target_fractions": fractions,
        "balance_objective": score,
        "splits": {split: describe(split_rows[split]) for split in SPLITS},
        **validation,
        "policy": {
            "tune": "May be used for Stage 6 model and hyperparameter selection.",
            "gate": "One final Stage 6 acceptance pass after candidate freeze.",
            "historical_dev_test": "Audit-only; prohibited for Stage 6 tuning.",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
