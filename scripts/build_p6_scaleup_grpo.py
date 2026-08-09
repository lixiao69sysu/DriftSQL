#!/usr/bin/env python3
"""Build full-episode P6 GRPO data: Train2400 + 800 hard replay rows."""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = ROOT / "data/processed/p6_scaleup_v1_low_write_protocol/rl_train.parquet"
DEFAULT_TUNE = ROOT / "data/processed/p6_scaleup_v1_low_write_protocol/rl_dev.parquet"
DEFAULT_FAILURES = ROOT / "data/processed/p6_scaleup_v1_on_policy_failures/failures.jsonl"
DEFAULT_OUTPUT = ROOT / "data/processed/p6_scaleup_v1_grpo"
REPLAY_QUOTAS = {
    "post_diff_wrong_retrieval": 300,
    "must_ask_error": 220,
    "compound_recovery": 160,
    "successful_execute_no_submit": 120,
}
PROMPT_FORBIDDEN_MARKERS = (
    '"ground_truth"',
    '"target_action"',
    '"decision_target_action"',
    '"failure_labels"',
    '"failure_primary"',
    '"replay_role"',
    '"canonical_sql"',
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def visible_prompt(row: dict[str, Any]) -> str:
    return json.dumps(row["prompt"], ensure_ascii=False, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--tune", type=Path, default=DEFAULT_TUNE)
    parser.add_argument("--failures", type=Path, default=DEFAULT_FAILURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")

    train_table = pq.read_table(args.train)
    tune_table = pq.read_table(args.tune)
    train_rows = train_table.to_pylist()
    tune_rows = tune_table.to_pylist()
    failures = load_jsonl(args.failures)
    if len(train_rows) != 2400 or len(tune_rows) != 432 or len(failures) != 1066:
        raise RuntimeError(
            f"Expected Train2400/Tune432/Failure1066, got "
            f"{len(train_rows)}/{len(tune_rows)}/{len(failures)}"
        )

    train_by_id = {
        str(row["extra_info"]["instance_id"]): row for row in train_rows
    }
    if len(train_by_id) != 2400:
        raise RuntimeError("Train2400 instance IDs are not unique")
    failure_keys = [
        str(row["_failure_miner"]["dedupe_key"]) for row in failures
    ]
    if len(set(failure_keys)) != 1066:
        raise RuntimeError("GRPO requires 1,066 unique Failure Miner trajectories")
    if any(str(row["instance_id"]) not in train_by_id for row in failures):
        raise RuntimeError("Failure trajectory is not associated with Train2400")

    for row in train_rows + tune_rows:
        prompt_text = visible_prompt(row)
        if any(marker in prompt_text for marker in PROMPT_FORBIDDEN_MARKERS):
            raise RuntimeError("Source GRPO prompt already contains a target/leakage marker")

    pools = {
        label: [
            row
            for row in failures
            if label in row["_failure_miner"]["classification"]["labels"]
        ]
        for label in REPLAY_QUOTAS
    }
    rng = random.Random(args.seed)
    output_rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for row in train_rows:
        output_rows.append(copy.deepcopy(row))
        manifest.append(
            {
                "sampling_role": "base_unique",
                "task_id": str(row["extra_info"]["instance_id"]),
                "db_id": str(row["extra_info"]["db_id"]),
                "source_failure_dedupe_key": "",
                "source_failure_labels": [],
            }
        )

    replacement: dict[str, bool] = {}
    for label, quota in REPLAY_QUOTAS.items():
        pool = pools[label]
        if not pool:
            raise RuntimeError(f"Empty GRPO failure pool for {label}")
        use_replacement = len(pool) < quota
        replacement[label] = use_replacement
        chosen = (
            [rng.choice(pool) for _ in range(quota)]
            if use_replacement
            else rng.sample(pool, quota)
        )
        for failure in chosen:
            task_id = str(failure["instance_id"])
            output_rows.append(copy.deepcopy(train_by_id[task_id]))
            classification = failure["_failure_miner"]["classification"]
            manifest.append(
                {
                    "sampling_role": label,
                    "task_id": task_id,
                    "db_id": str(failure["db_id"]),
                    "source_failure_dedupe_key": str(
                        failure["_failure_miner"]["dedupe_key"]
                    ),
                    "source_failure_labels": list(classification["labels"]),
                }
            )

    if len(output_rows) != 3200 or len(manifest) != 3200:
        raise RuntimeError(f"Expected 3,200 GRPO rows, got {len(output_rows)}")
    order = list(range(len(output_rows)))
    rng.shuffle(order)
    output_rows = [output_rows[index] for index in order]
    manifest = [manifest[index] for index in order]
    for index, row in enumerate(manifest):
        row["sampling_index"] = index

    # The model-visible prompt must remain byte-equivalent to the sealed source
    # prompt.  Replay annotations live only in the sidecar manifest.
    for row in output_rows:
        task_id = str(row["extra_info"]["instance_id"])
        if row["prompt"] != train_by_id[task_id]["prompt"]:
            raise RuntimeError(f"GRPO replay changed the visible prompt: {task_id}")
        prompt_text = visible_prompt(row)
        if any(marker in prompt_text for marker in PROMPT_FORBIDDEN_MARKERS):
            raise RuntimeError(f"GRPO target leakage detected: {task_id}")

    train_dbs = {str(row["extra_info"]["db_id"]) for row in output_rows}
    tune_dbs = {str(row["extra_info"]["db_id"]) for row in tune_rows}
    if train_dbs & tune_dbs:
        raise RuntimeError("GRPO Train/Tune database leakage")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    pq.write_table(
        pa.Table.from_pylist(output_rows, schema=train_table.schema),
        args.output_dir / "train.parquet",
        compression="zstd",
    )
    pq.write_table(tune_table, args.output_dir / "tune.parquet", compression="zstd")
    write_jsonl(args.output_dir / "train_manifest.jsonl", manifest)
    summary = {
        "protocol": "p6_scaleup_full_episode_grpo_v1",
        "train_rows": len(output_rows),
        "base_unique_rows": 2400,
        "failure_replay_rows": 800,
        "tune_rows": len(tune_rows),
        "replay_quotas": REPLAY_QUOTAS,
        "failure_pool_sizes": {name: len(pool) for name, pool in pools.items()},
        "sampling_with_replacement": replacement,
        "sampling_roles": dict(
            sorted(Counter(row["sampling_role"] for row in manifest).items())
        ),
        "train_databases": len(train_dbs),
        "tune_databases": len(tune_dbs),
        "train_tune_database_overlap": sorted(train_dbs & tune_dbs),
        "full_episode_agent_loop": True,
        "model_visible_prompt_source": "byte-identical sealed Train2400 prompt",
        "replay_metadata_location": "sidecar train_manifest.jsonl only",
        "ground_truth_location": "reward_model/environment only; never copied into prompt",
        "prompt_target_leakage": False,
        "prompt_rows_changed": 0,
        "fresh_blind_rows_read": False,
        "seed": args.seed,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
