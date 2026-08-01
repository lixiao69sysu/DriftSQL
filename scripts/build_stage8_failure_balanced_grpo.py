#!/usr/bin/env python3
"""Build Stage 8 GRPO replay from real Tune failure strata and Train rows only."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_stage7_failure_balanced_grpo import (  # noqa: E402
    allocate_counts,
    classify_failure,
    load_jsonl,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL = (
    ROOT
    / "reports/stage8/tune55_stage8_sft20_process_isolated/stage8-sft20-fresh-tune.jsonl"
)
DEFAULT_TUNE = ROOT / "data/processed/stage8_fresh_sft/rl_tune.parquet"
DEFAULT_TRAIN = ROOT / "data/processed/stage8_fresh_sft/rl_train.parquet"
DEFAULT_OUTPUT = ROOT / "data/processed/stage8_failure_balanced_grpo"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--tune", type=Path, default=DEFAULT_TUNE)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rows", type=int, default=440)
    parser.add_argument("--add-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=82029)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if args.rows <= 0:
        parser.error("--rows must be positive")
    if not 0.0 < args.add_ratio < 1.0:
        parser.error("--add-ratio must be in (0, 1)")

    tune_table = pq.read_table(args.tune)
    tune_rows = tune_table.to_pylist()
    tune_by_id = {str(row["extra_info"]["instance_id"]): row for row in tune_rows}
    evaluations = load_jsonl(args.eval)
    if {str(row.get("instance_id", "")) for row in evaluations} != set(tune_by_id):
        raise RuntimeError("Tune evaluation IDs and Tune metadata IDs differ")

    diagnostics: list[dict[str, Any]] = []
    add_failure_strata: Counter[tuple[str, int, str]] = Counter()
    general_failure_types: Counter[str] = Counter()
    for evaluation in evaluations:
        instance_id = str(evaluation["instance_id"])
        tune = tune_by_id[instance_id]
        extra = tune["extra_info"]
        drift_type = str(extra["drift_type"])
        label, evidence = classify_failure(
            evaluation, str(tune["reward_model"]["ground_truth"])
        )
        diagnostic = {
            "instance_id": instance_id,
            "db_id": str(extra["db_id"]),
            "drift_type": drift_type,
            "wildcard_profile": extra.get("wildcard_profile"),
            "added_column_count": extra.get("added_column_count"),
            "failure_type": label,
            "task_success": bool(evaluation.get("task_success")),
            **evidence,
        }
        diagnostics.append(diagnostic)
        if label == "success":
            continue
        if drift_type == "add_column":
            add_failure_strata[
                (
                    str(extra["wildcard_profile"]),
                    int(extra["added_column_count"]),
                    label,
                )
            ] += 1
        else:
            general_failure_types[drift_type] += 1
    if not add_failure_strata:
        raise RuntimeError("No real Stage 8 add-column Tune failures were observed")

    train_table = pq.read_table(args.train)
    train_rows = train_table.to_pylist()
    train_ids = {str(row["extra_info"]["instance_id"]) for row in train_rows}
    tune_ids = set(tune_by_id)
    leaked_ids = sorted(train_ids & tune_ids)
    train_dbs = {str(row["extra_info"]["db_id"]) for row in train_rows}
    tune_dbs = {str(row["extra_info"]["db_id"]) for row in tune_rows}
    leaked_dbs = sorted(train_dbs & tune_dbs)
    if leaked_ids or leaked_dbs:
        raise RuntimeError(f"Train/Tune leakage ids={leaked_ids[:3]} dbs={leaked_dbs}")

    add_pool: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    general_pool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        extra = row["extra_info"]
        drift_type = str(extra["drift_type"])
        if drift_type == "add_column":
            add_pool[
                (str(extra["wildcard_profile"]), int(extra["added_column_count"]))
            ].append(row)
        else:
            general_pool[drift_type].append(row)
    if set(general_pool) != {
        "clean", "compound", "rename_column", "rename_table", "replace_column"
    }:
        raise RuntimeError(f"Incomplete general replay pool: {sorted(general_pool)}")

    rng = random.Random(args.seed)
    sampled: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    add_total = round(args.rows * args.add_ratio)
    strata = sorted(add_failure_strata)
    allocations = allocate_counts(
        [add_failure_strata[stratum] for stratum in strata], add_total
    )
    for stratum, count in zip(strata, allocations, strict=True):
        profile, added_count, failure_type = stratum
        pool = add_pool.get((profile, added_count), [])
        if not pool:
            raise RuntimeError(f"No Train rows for real failure stratum {stratum}")
        for _ in range(count):
            row = rng.choice(pool)
            sampled.append(row)
            manifest.append(
                {
                    "source_instance_id": str(row["extra_info"]["instance_id"]),
                    "replay_source": "real_tune_failure_matched_train",
                    "failure_type": failure_type,
                    "wildcard_profile": profile,
                    "added_column_count": added_count,
                }
            )

    general_total = args.rows - add_total
    general_types = sorted(general_pool)
    # Every family gets a replay floor. Real Tune failures add two extra units
    # so the lone compound wrong-submit is emphasized without removing broad
    # regression coverage.
    general_weights = [1 + 2 * general_failure_types[name] for name in general_types]
    general_allocations = allocate_counts(general_weights, general_total)
    for drift_type, count in zip(general_types, general_allocations, strict=True):
        pool = general_pool[drift_type]
        for _ in range(count):
            row = rng.choice(pool)
            sampled.append(row)
            manifest.append(
                {
                    "source_instance_id": str(row["extra_info"]["instance_id"]),
                    "replay_source": "general_drift_replay",
                    "drift_type": drift_type,
                    "tune_failure_weight": general_failure_types[drift_type],
                }
            )

    order = list(range(len(sampled)))
    rng.shuffle(order)
    sampled = [sampled[index] for index in order]
    manifest = [manifest[index] | {"replay_index": replay_index} for replay_index, index in enumerate(order)]
    args.output_dir.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(sampled, schema=train_table.schema),
        args.output_dir / "train.parquet",
        compression="zstd",
    )
    write_jsonl(args.output_dir / "failure_diagnostics.jsonl", diagnostics)
    write_jsonl(args.output_dir / "sampling_manifest.jsonl", manifest)
    summary = {
        "protocol": "stage8_real_failure_balanced_grpo_v1",
        "policy": "Tune failures set weights; optimization rows come only from DB-disjoint Train",
        "candidate_eval": str(args.eval.resolve()),
        "source_train": str(args.train.resolve()),
        "tune_metadata": str(args.tune.resolve()),
        "real_tune_trajectories": len(evaluations),
        "real_tune_outcomes": dict(sorted(Counter(row["failure_type"] for row in diagnostics).items())),
        "add_failure_strata": {
            "|".join(map(str, key)): value for key, value in sorted(add_failure_strata.items())
        },
        "general_failure_types": dict(sorted(general_failure_types.items())),
        "output_rows": len(sampled),
        "add_column_rows": add_total,
        "general_replay_rows": general_total,
        "add_ratio": add_total / len(sampled),
        "sampled_failure_types": dict(
            sorted(Counter(row.get("failure_type", "") for row in manifest if row["replay_source"] == "real_tune_failure_matched_train").items())
        ),
        "sampled_wildcard_profiles": dict(
            sorted(Counter(row.get("wildcard_profile", "") for row in manifest if row["replay_source"] == "real_tune_failure_matched_train").items())
        ),
        "sampled_general_types": dict(
            sorted(Counter(row.get("drift_type", "") for row in manifest if row["replay_source"] == "general_drift_replay").items())
        ),
        "train_tune_task_overlap": leaked_ids,
        "train_tune_database_overlap": leaked_dbs,
        "stage6_gate112_read": False,
        "stage7_gate106_read": False,
        "stage8_gate_read": False,
        "seed": args.seed,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
