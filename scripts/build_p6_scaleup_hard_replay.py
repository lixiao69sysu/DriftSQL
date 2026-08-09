#!/usr/bin/env python3
"""Build the 1,600-row P6 Scale-up hard-replay SFT mixture."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECOVERY = ROOT / "data/processed/p6_scaleup_v1_recovery_sft"
DEFAULT_CANONICAL = ROOT / "data/processed/p6_scaleup_v1_low_write_protocol/train.parquet"
DEFAULT_TUNE = ROOT / "data/processed/p6_scaleup_v1_low_write_protocol/rl_dev.parquet"
DEFAULT_OUTPUT = ROOT / "data/processed/p6_scaleup_v1_hard_replay"
QUOTAS = {
    "post_diff_recovery": 300,
    "must_ask_recovery": 220,
    "compound_recovery": 160,
    "successful_execute_then_submit": 120,
    "standard_submit": 300,
    "standard_execute": 200,
    "standard_diff": 150,
    "standard_ask": 150,
}
OUTPUT_FIELDS = (
    "messages",
    "tools",
    "enable_thinking",
    "target_action",
    "task_id",
    "db_id",
    "scenario_type",
    "drift_type",
    "interaction_profile",
    "difficulty",
    "failure_mode",
    "available_tools",
    "token_count",
    "replay_role",
    "replay_index",
    "replay_source",
    "source_example_id",
    "failure_dedupe_key",
    "failure_primary",
    "failure_labels",
    "recovery_context",
    "target_source",
)


def stable_id(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def sample_rows(
    pool: list[dict[str, Any]], count: int, *, rng: random.Random
) -> tuple[list[dict[str, Any]], bool]:
    if not pool:
        raise RuntimeError("Hard-replay category has an empty source pool")
    if len(pool) >= count:
        return rng.sample(pool, count), False
    return [rng.choice(pool) for _ in range(count)], True


def normalize(
    row: dict[str, Any], *, role: str, index: int, source: str
) -> dict[str, Any]:
    output = copy.deepcopy(row)
    source_id = stable_id(
        {
            "task_id": output["task_id"],
            "messages": output["messages"],
            "target_action": output["target_action"],
            "recovery_context": output.get("recovery_context"),
        }
    )
    output.update(
        {
            "replay_role": role,
            "replay_index": index,
            "replay_source": source,
            "source_example_id": source_id,
            "failure_dedupe_key": str(output.get("failure_dedupe_key") or ""),
            "failure_primary": str(output.get("failure_primary") or ""),
            "failure_labels": list(output.get("failure_labels") or []),
            "recovery_context": str(output.get("recovery_context") or "canonical"),
            "target_source": "execution_verified_train_oracle",
        }
    )
    return {field: output.get(field) for field in OUTPUT_FIELDS}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recovery-dir", type=Path, default=DEFAULT_RECOVERY)
    parser.add_argument("--canonical-sft", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--tune-rl", type=Path, default=DEFAULT_TUNE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--max-tokens", type=int, default=6144)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")

    recovery_summary = json.loads(
        (args.recovery_dir / "summary.json").read_text(encoding="utf-8")
    )
    if (
        recovery_summary.get("unique_failure_trajectories") != 1066
        or recovery_summary.get("task_associations") != 1066
        or recovery_summary.get("canonical_associations") != 1066
    ):
        raise RuntimeError("Recovery provenance does not cover all 1,066 failures")
    recovery = pq.read_table(args.recovery_dir / "train.parquet").to_pylist()
    canonical = pq.read_table(args.canonical_sft).to_pylist()
    tune = pq.read_table(args.tune_rl).to_pylist()
    if len(canonical) != 24285 or len(tune) != 432:
        raise RuntimeError(
            f"Expected canonical Train24285/Tune432, got {len(canonical)}/{len(tune)}"
        )

    recovery_pools = {
        "post_diff_recovery": [
            row for row in recovery if "post_diff_wrong_retrieval" in row["failure_labels"]
        ],
        "must_ask_recovery": [
            row for row in recovery if "must_ask_error" in row["failure_labels"]
        ],
        "compound_recovery": [
            row for row in recovery if "compound_recovery" in row["failure_labels"]
        ],
        "successful_execute_then_submit": [
            row
            for row in recovery
            if row["recovery_context"] == "terminal_missing"
            and row["target_action"] == "submit_solution"
        ],
    }
    canonical_pools = {
        "standard_submit": [row for row in canonical if row["target_action"] == "submit_solution"],
        "standard_execute": [row for row in canonical if row["target_action"] == "execute_sql"],
        "standard_diff": [row for row in canonical if row["target_action"] == "inspect_schema_diff"],
        "standard_ask": [row for row in canonical if row["target_action"] == "ask_user"],
    }
    pools = recovery_pools | canonical_pools
    rng = random.Random(args.seed)
    selected: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    replacement: dict[str, bool] = {}
    for role, quota in QUOTAS.items():
        chosen, used_replacement = sample_rows(pools[role], quota, rng=rng)
        replacement[role] = used_replacement
        source = "real_on_policy_recovery" if role in recovery_pools else "canonical_replay"
        for row in chosen:
            output = normalize(row, role=role, index=len(selected), source=source)
            selected.append(output)
            manifest.append(
                {
                    "replay_index": output["replay_index"],
                    "replay_role": role,
                    "replay_source": source,
                    "source_example_id": output["source_example_id"],
                    "task_id": output["task_id"],
                    "db_id": output["db_id"],
                    "target_action": output["target_action"],
                    "failure_dedupe_key": output["failure_dedupe_key"],
                }
            )

    if len(selected) != 1600:
        raise RuntimeError(f"Expected 1,600 hard-replay rows, got {len(selected)}")
    rng.shuffle(selected)
    for index, row in enumerate(selected):
        row["replay_index"] = index
    manifest = [
        {
            "replay_index": row["replay_index"],
            "replay_role": row["replay_role"],
            "replay_source": row["replay_source"],
            "source_example_id": row["source_example_id"],
            "task_id": row["task_id"],
            "db_id": row["db_id"],
            "target_action": row["target_action"],
            "failure_dedupe_key": row["failure_dedupe_key"],
        }
        for row in selected
    ]
    if any(row["target_action"] not in row["available_tools"] for row in selected):
        raise RuntimeError("Hard Replay contains a dynamically masked target action")
    if max(int(row["token_count"]) for row in selected) > args.max_tokens:
        raise RuntimeError("Hard Replay exceeds the token budget")
    terminal_rows = [
        row for row in selected if row["replay_role"] == "successful_execute_then_submit"
    ]
    if len(terminal_rows) != 120 or any(
        row["target_action"] != "submit_solution" for row in terminal_rows
    ):
        raise RuntimeError("Terminal replay must contain 120 submit_solution targets")

    train_dbs = {str(row["db_id"]) for row in selected}
    tune_dbs = {str(row["extra_info"]["db_id"]) for row in tune}
    if train_dbs & tune_dbs:
        raise RuntimeError("Hard Replay Train/Tune database leakage")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    pq.write_table(
        pa.Table.from_pylist(selected),
        args.output_dir / "train.parquet",
        compression="zstd",
    )
    write_jsonl(args.output_dir / "manifest.jsonl", manifest)
    summary = {
        "protocol": "p6_scaleup_hard_replay_sft_v1",
        "rows": len(selected),
        "real_recovery_rows": sum(
            row["replay_source"] == "real_on_policy_recovery" for row in selected
        ),
        "canonical_replay_rows": sum(
            row["replay_source"] == "canonical_replay" for row in selected
        ),
        "quotas": QUOTAS,
        "source_pool_sizes": {name: len(pool) for name, pool in pools.items()},
        "sampling_with_replacement": replacement,
        "replay_roles": dict(sorted(Counter(row["replay_role"] for row in selected).items())),
        "target_actions": dict(
            sorted(Counter(row["target_action"] for row in selected).items())
        ),
        "train_databases": len(train_dbs),
        "tune_databases": len(tune_dbs),
        "train_tune_database_overlap": sorted(train_dbs & tune_dbs),
        "target_action_available": True,
        "terminal_submit_targets": len(terminal_rows),
        "max_token_count": max(int(row["token_count"]) for row in selected),
        "target_sources": dict(
            sorted(Counter(row["target_source"] for row in selected).items())
        ),
        "fresh_blind_rows_read": False,
        "seed": args.seed,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
