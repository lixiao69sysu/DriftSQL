#!/usr/bin/env python3
"""Build Stage 7 GRPO data from real Tune failures without training on Tune rows.

The failed Tune trajectories only define replay strata.  The actual GRPO rows
are sampled from the database-disjoint Stage 7 Train partition.  Stage 7 Gate
and the permanently sealed Stage 6 Gate112 are never read by this program.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL = (
    PROJECT_ROOT
    / "reports/stage7/tune24_add_sft20_process_isolated/stage7-sft20.jsonl"
)
DEFAULT_TUNE = PROJECT_ROOT / "data/processed/stage7_add_column_sft/rl_tune.parquet"
DEFAULT_TRAIN = PROJECT_ROOT / "data/processed/stage7_add_column_sft/rl_train.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/processed/stage7_failure_balanced_grpo"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def normalise_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", str(sql).strip().rstrip(";")).casefold()


def classify_failure(
    evaluation: dict[str, Any], ground_truth: str
) -> tuple[str, dict[str, Any]]:
    """Classify one real agent outcome into an actionable replay category."""

    calls = [
        (
            str(event.get("tool_name", "")),
            str(event.get("arguments", {}).get("sql", "")),
        )
        for event in evaluation.get("trajectory", [])
    ]
    expected = normalise_sql(ground_truth)
    repaired_executed = any(
        name == "execute_sql" and normalise_sql(sql) == expected
        for name, sql in calls
    )
    submitted = evaluation.get("termination_reason") in {
        "submitted",
        "fallback_submitted",
    }
    final_sql = str(evaluation.get("final_sql", ""))
    stale_wildcard_submitted = submitted and "*" in final_sql

    if bool(evaluation.get("task_success")):
        label = "success"
    elif stale_wildcard_submitted:
        label = "premature_stale_submit"
    elif repaired_executed and not submitted:
        label = "repaired_not_submitted"
    elif submitted:
        label = "wrong_submit"
    else:
        label = "repair_not_reached"
    evidence = {
        "repaired_executed": repaired_executed,
        "submitted": submitted,
        "stale_wildcard_submitted": stale_wildcard_submitted,
        "called_tools": [name for name, _ in calls],
        "termination_reason": str(evaluation.get("termination_reason", "")),
    }
    return label, evidence


def allocate_counts(weights: list[int], total: int) -> list[int]:
    if not weights or sum(weights) <= 0:
        raise ValueError("positive failure weights are required")
    raw = [total * weight / sum(weights) for weight in weights]
    allocated = [int(value) for value in raw]
    for index in sorted(
        range(len(weights)), key=lambda item: raw[item] - allocated[item], reverse=True
    )[: total - sum(allocated)]:
        allocated[index] += 1
    return allocated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--tune", type=Path, default=DEFAULT_TUNE)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rows", type=int, default=403)
    parser.add_argument("--add-ratio", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=72028)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if args.rows <= 0:
        parser.error("--rows must be positive")
    if not 0.0 < args.add_ratio < 1.0:
        parser.error("--add-ratio must be in (0, 1)")

    tune_rows = pq.read_table(args.tune).to_pylist()
    tune_by_id = {
        str(row["extra_info"]["instance_id"]): row for row in tune_rows
    }
    evaluations = load_jsonl(args.eval)
    diagnostics: list[dict[str, Any]] = []
    failure_strata: Counter[tuple[str, int, str]] = Counter()
    for evaluation in evaluations:
        instance_id = str(evaluation.get("instance_id", ""))
        if instance_id not in tune_by_id:
            raise ValueError(f"Tune evaluation ID is absent from Tune parquet: {instance_id}")
        tune = tune_by_id[instance_id]
        extra = tune["extra_info"]
        if str(extra.get("drift_type")) != "add_column":
            raise ValueError(f"Failure miner only accepts add_column Tune rows: {instance_id}")
        label, evidence = classify_failure(
            evaluation, str(tune["reward_model"]["ground_truth"])
        )
        diagnostic = {
            "instance_id": instance_id,
            "db_id": str(extra["db_id"]),
            "wildcard_profile": str(extra["wildcard_profile"]),
            "added_column_count": int(extra["added_column_count"]),
            "failure_type": label,
            "task_success": bool(evaluation.get("task_success")),
            **evidence,
        }
        diagnostics.append(diagnostic)
        if label != "success":
            failure_strata[
                (
                    diagnostic["wildcard_profile"],
                    diagnostic["added_column_count"],
                    label,
                )
            ] += 1
    if not failure_strata:
        raise RuntimeError("No real Tune failures were observed")

    train_table = pq.read_table(args.train)
    train_rows = train_table.to_pylist()
    train_ids = {str(row["extra_info"]["instance_id"]) for row in train_rows}
    leaked_ids = sorted(train_ids & set(tune_by_id))
    if leaked_ids:
        raise RuntimeError(f"Train/Tune task leakage: {leaked_ids[:3]}")
    train_dbs = {str(row["extra_info"]["db_id"]) for row in train_rows}
    tune_dbs = {str(row["extra_info"]["db_id"]) for row in tune_rows}
    leaked_dbs = sorted(train_dbs & tune_dbs)
    if leaked_dbs:
        raise RuntimeError(f"Train/Tune database leakage: {leaked_dbs}")

    add_pool: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    general_pool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        extra = row["extra_info"]
        drift_type = str(extra.get("drift_type", ""))
        if drift_type == "add_column":
            add_pool[
                (str(extra.get("wildcard_profile", "")), int(extra["added_column_count"]))
            ].append(row)
        else:
            general_pool[drift_type].append(row)
    if not general_pool:
        raise RuntimeError("General drift replay pool is empty")

    rng = random.Random(args.seed)
    strata = sorted(failure_strata)
    add_total = round(args.rows * args.add_ratio)
    allocations = allocate_counts([failure_strata[key] for key in strata], add_total)
    sampled: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for stratum, count in zip(strata, allocations, strict=True):
        profile, added_column_count, failure_type = stratum
        pool = add_pool.get((profile, added_column_count), [])
        if not pool:
            raise RuntimeError(f"No Train rows match real failure stratum {stratum}")
        for _ in range(count):
            row = rng.choice(pool)
            sampled.append(row)
            manifest.append(
                {
                    "source_instance_id": str(row["extra_info"]["instance_id"]),
                    "replay_source": "real_tune_failure_matched_train",
                    "failure_type": failure_type,
                    "wildcard_profile": profile,
                    "added_column_count": added_column_count,
                }
            )

    general_total = args.rows - add_total
    general_types = sorted(general_pool)
    general_allocations = allocate_counts(
        [1 for _ in general_types], general_total
    )
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
                }
            )

    order = list(range(len(sampled)))
    rng.shuffle(order)
    sampled = [sampled[index] for index in order]
    manifest = [manifest[index] for index in order]
    for replay_index, item in enumerate(manifest):
        item["replay_index"] = replay_index

    args.output_dir.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(sampled, schema=train_table.schema),
        args.output_dir / "train.parquet",
        compression="zstd",
    )
    write_jsonl(args.output_dir / "failure_diagnostics.jsonl", diagnostics)
    write_jsonl(args.output_dir / "sampling_manifest.jsonl", manifest)
    summary = {
        "protocol": "stage7_real_failure_balanced_grpo_v1",
        "policy": (
            "Tune failures define structural weights; only database-disjoint Train rows "
            "are optimized"
        ),
        "candidate_eval": str(args.eval.resolve()),
        "source_train": str(args.train.resolve()),
        "tune_metadata": str(args.tune.resolve()),
        "real_tune_trajectories": len(evaluations),
        "real_tune_outcomes": dict(
            sorted(Counter(item["failure_type"] for item in diagnostics).items())
        ),
        "failure_strata": {
            "|".join(map(str, key)): value for key, value in sorted(failure_strata.items())
        },
        "output_rows": len(sampled),
        "add_column_rows": add_total,
        "general_replay_rows": general_total,
        "add_ratio": add_total / len(sampled),
        "sampled_replay_sources": dict(
            sorted(Counter(item["replay_source"] for item in manifest).items())
        ),
        "sampled_general_types": dict(
            sorted(
                Counter(
                    item.get("drift_type", "")
                    for item in manifest
                    if item["replay_source"] == "general_drift_replay"
                ).items()
            )
        ),
        "train_tune_task_overlap": leaked_ids,
        "train_tune_database_overlap": leaked_dbs,
        "stage6_gate112_read": False,
        "stage7_gate_read": False,
        "seed": args.seed,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
