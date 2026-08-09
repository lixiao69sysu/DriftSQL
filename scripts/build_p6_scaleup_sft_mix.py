#!/usr/bin/env python3
"""Combine all Scale-up Recovery SFT and Hard Replay for continuation SFT."""

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
RECOVERY = ROOT / "data/processed/p6_scaleup_v1_recovery_sft"
HARD = ROOT / "data/processed/p6_scaleup_v1_hard_replay/train.parquet"
TUNE = ROOT / "data/processed/p6_scaleup_v1_low_write_protocol/dev.parquet"
OUTPUT = ROOT / "data/processed/p6_scaleup_v1_sft_mix"
FIELDS = (
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
    "mixture_source",
)


def normalize(row: dict[str, Any], source: str) -> dict[str, Any]:
    value = copy.deepcopy(row)
    value["mixture_source"] = source
    return {field: value.get(field) for field in FIELDS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recovery-dir", type=Path, default=RECOVERY)
    parser.add_argument("--hard-replay", type=Path, default=HARD)
    parser.add_argument("--tune", type=Path, default=TUNE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")

    recovery_train = pq.read_table(args.recovery_dir / "train.parquet").to_pylist()
    recovery_dev = pq.read_table(args.recovery_dir / "dev.parquet").to_pylist()
    hard = pq.read_table(args.hard_replay).to_pylist()
    tune = pq.read_table(args.tune).to_pylist()
    recovery = recovery_train + recovery_dev
    if len(recovery) != 1714 or len(hard) != 1600 or len(tune) != 2254:
        raise RuntimeError(
            f"Expected Recovery1714/Hard1600/Tune2254, got "
            f"{len(recovery)}/{len(hard)}/{len(tune)}"
        )

    train = [normalize(row, "recovery_sft") for row in recovery]
    train.extend(normalize(row, "hard_replay") for row in hard)
    validation = [normalize(row, "tune432_canonical") for row in tune]
    rng = random.Random(args.seed)
    rng.shuffle(train)

    train_dbs = {str(row["db_id"]) for row in train}
    tune_dbs = {str(row["db_id"]) for row in validation}
    if train_dbs & tune_dbs:
        raise RuntimeError("Scale-up SFT Train/Tune database leakage")
    if any(row["target_action"] not in row["available_tools"] for row in train + validation):
        raise RuntimeError("Scale-up SFT mix contains a masked target")
    max_tokens = max(int(row["token_count"]) for row in train + validation)
    if max_tokens > 6144:
        raise RuntimeError(f"Scale-up SFT mix exceeds token budget: {max_tokens}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    pq.write_table(
        pa.Table.from_pylist(train), args.output_dir / "train.parquet", compression="zstd"
    )
    pq.write_table(
        pa.Table.from_pylist(validation),
        args.output_dir / "dev.parquet",
        compression="zstd",
    )
    summary = {
        "protocol": "p6_scaleup_recovery_hard_replay_sft_mix_v1",
        "train_rows": len(train),
        "validation_rows": len(validation),
        "mixture_sources": dict(
            sorted(Counter(str(row["mixture_source"]) for row in train).items())
        ),
        "target_actions": dict(
            sorted(Counter(str(row["target_action"]) for row in train).items())
        ),
        "train_databases": len(train_dbs),
        "tune_databases": len(tune_dbs),
        "train_tune_database_overlap": sorted(train_dbs & tune_dbs),
        "target_action_available": True,
        "max_token_count": max_tokens,
        "fresh_blind_rows_read": False,
        "seed": args.seed,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
