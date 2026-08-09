#!/usr/bin/env python3
"""Select a deterministic 600-task hard on-policy rollout pool.

Only Train records are read.  Tasks are restricted to small source databases
to keep repeated stochastic rollouts I/O bounded, then allocated to four
disjoint failure-oriented buckets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/processed/p6_scaleup_v1_low_write_protocol/train_agent_eval.jsonl"
DEFAULT_OUTPUT = ROOT / "data/processed/p6_scaleup_v1_rollout_pool600"
SEED = 20260805


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def stable_key(row: dict[str, Any], bucket: str) -> str:
    task_id = str(row["extra_info"]["instance_id"])
    return hashlib.sha256(f"{SEED}:{bucket}:{task_id}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-db-mb", type=int, default=16)
    parser.add_argument("--max-per-db", type=int, default=30)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")

    rows = load_jsonl(args.input)
    maximum_bytes = args.max_db_mb * 1024 * 1024
    candidates = [
        row
        for row in rows
        if Path(str(row["extra_info"]["source_db"])).stat().st_size <= maximum_bytes
        and str(row["extra_info"]["drift_type"]) != "clean"
    ]
    if len(candidates) < 600:
        raise RuntimeError(f"Only {len(candidates)} bounded-I/O drift candidates")

    def compound(extra: dict[str, Any]) -> bool:
        return str(extra["drift_type"]) == "compound"

    def must_ask_atomic(extra: dict[str, Any]) -> bool:
        return (
            str(extra["interaction_profile"]) == "must_ask"
            and str(extra["drift_type"]) != "compound"
        )

    def silent_add(extra: dict[str, Any]) -> bool:
        return (
            str(extra["drift_type"]) == "add_column"
            and str(extra["failure_mode"]) == "silent_result_mismatch"
        )

    def post_diff_retrieval(extra: dict[str, Any]) -> bool:
        return (
            str(extra["interaction_profile"]) in {"knowledge_only", "schema_only"}
            and str(extra["drift_type"])
            in {"rename_column", "rename_table", "replace_column"}
        )

    specifications: list[tuple[str, int, Callable[[dict[str, Any]], bool]]] = [
        ("compound_recovery", 200, compound),
        ("must_ask_decision", 180, must_ask_atomic),
        ("silent_add_contract", 100, silent_add),
        ("post_diff_retrieval", 120, post_diff_retrieval),
    ]
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    per_db: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    audit: list[dict[str, Any]] = []

    for bucket, target, predicate in specifications:
        eligible = sorted(
            (
                row
                for row in candidates
                if str(row["extra_info"]["instance_id"]) not in selected_ids
                and predicate(row["extra_info"])
            ),
            key=lambda row: stable_key(row, bucket),
        )
        for row in eligible:
            if bucket_counts[bucket] >= target:
                break
            extra = row["extra_info"]
            db_id = str(extra["db_id"])
            if per_db[db_id] >= args.max_per_db:
                continue
            task_id = str(extra["instance_id"])
            output = json.loads(json.dumps(row))
            output["extra_info"]["rollout_sampling"] = {
                "pool": "p6_scaleup_v1_hard600",
                "selection_bucket": bucket,
                "source_db_bytes": Path(str(extra["source_db"])).stat().st_size,
                "fresh_blind": False,
            }
            selected.append(output)
            selected_ids.add(task_id)
            per_db[db_id] += 1
            bucket_counts[bucket] += 1
            audit.append(
                {
                    "task_id": task_id,
                    "db_id": db_id,
                    "bucket": bucket,
                    "drift_type": extra["drift_type"],
                    "interaction_profile": extra["interaction_profile"],
                    "difficulty": extra["difficulty"],
                    "source_db_bytes": Path(str(extra["source_db"])).stat().st_size,
                }
            )
        if bucket_counts[bucket] != target:
            raise RuntimeError(
                f"Could not fill {bucket}: {bucket_counts[bucket]}/{target}; "
                f"increase --max-db-mb or --max-per-db"
            )

    if len(selected) != 600 or len(selected_ids) != 600:
        raise RuntimeError(f"Rollout pool invariant failed: {len(selected)}/{len(selected_ids)}")
    selected.sort(key=lambda row: str(row["extra_info"]["instance_id"]))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.output_dir / "train_agent_eval.jsonl", selected)
    write_jsonl(args.output_dir / "selection_audit.jsonl", audit)
    summary = {
        "protocol": "p6_scaleup_v1_hard_rollout_pool600",
        "source_split": "train_only",
        "fresh_blind_rows_read": False,
        "tasks": len(selected),
        "databases": len(per_db),
        "max_source_db_mb": args.max_db_mb,
        "max_tasks_per_db": max(per_db.values()),
        "selection_buckets": dict(sorted(bucket_counts.items())),
        "drift_types": dict(
            sorted(Counter(str(row["extra_info"]["drift_type"]) for row in selected).items())
        ),
        "profiles": dict(
            sorted(
                Counter(str(row["extra_info"]["interaction_profile"]) for row in selected).items()
            )
        ),
        "difficulty": dict(
            sorted(Counter(str(row["extra_info"]["difficulty"]) for row in selected).items())
        ),
        "database_distribution": dict(sorted(per_db.items())),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
