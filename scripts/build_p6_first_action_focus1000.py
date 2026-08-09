#!/usr/bin/env python3
"""Build a Train-only 25-step first-action/shortcut GRPO curriculum.

The target action is reward-only metadata.  Model-visible prompts remain byte
identical to the sealed Focus1000 source and Fresh Blind320 is never opened.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data/processed/p6_focus1000_reward_ab"
DEFAULT_OUTPUT = ROOT / "data/processed/p6_first_action_focus200_v2"
FORBIDDEN_PROMPT_MARKERS = (
    '"ground_truth"',
    '"target_action"',
    '"decision_target_action"',
    '"canonical_sql"',
)


def prompt_bytes(row: dict[str, Any]) -> bytes:
    return json.dumps(
        row["prompt"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def annotate(row: dict[str, Any], *, coverage_index: int | None = None) -> dict[str, Any]:
    output = copy.deepcopy(row)
    extra = dict(output["extra_info"])
    # Every execution-verified P6 canonical trajectory begins by probing the
    # cached read-only SQL exactly once. This target is reward-only metadata;
    # the shortcut penalty separately teaches the model not to submit that
    # stale query when an AddColumn result-contract drift is detected.
    extra["decision_target_action"] = "execute_sql"
    extra["first_action_objective"] = "probe_then_recover_without_stale_submit_v2"
    if coverage_index is not None:
        extra["index"] = coverage_index
    output["extra_info"] = extra
    return output


def select_unique(
    rows: list[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
    count: int,
    rng: random.Random,
    excluded: set[str],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen = set(excluded)
    for row in rows:
        task_id = str(row["extra_info"]["instance_id"])
        if task_id in seen or not predicate(row):
            continue
        seen.add(task_id)
        candidates.append(row)
    rng.shuffle(candidates)
    if len(candidates) < count:
        raise RuntimeError(f"Unique curriculum pool has {len(candidates)} rows, needs {count}")
    chosen = candidates[:count]
    excluded.update(str(row["extra_info"]["instance_id"]) for row in chosen)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260809)
    args = parser.parse_args()

    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    train_path = source / "train_coverage.parquet"
    tune_path = source / "tune.parquet"
    if not train_path.is_file() or not tune_path.is_file():
        raise FileNotFoundError("Focus1000 coverage train or Tune432 is missing")

    source_train = pq.read_table(train_path).to_pylist()
    source_tune = pq.read_table(tune_path).to_pylist()
    if len(source_train) != 1000 or len(source_tune) != 432:
        raise RuntimeError(
            f"Expected Focus1000/Tune432, got {len(source_train)}/{len(source_tune)}"
        )

    rng = random.Random(args.seed)
    excluded: set[str] = set()
    selected: list[dict[str, Any]] = []
    selected.extend(
        select_unique(
            source_train,
            lambda row: row["extra_info"]["drift_type"] == "add_column",
            160,
            rng,
            excluded,
        )
    )
    selected.extend(
        select_unique(
            source_train,
            lambda row: row["extra_info"]["drift_type"] != "add_column"
            and row["extra_info"]["interaction_profile"] == "must_ask",
            20,
            rng,
            excluded,
        )
    )
    selected.extend(
        select_unique(
            source_train,
            lambda row: row["extra_info"]["drift_type"] != "add_column"
            and row["extra_info"]["interaction_profile"] != "must_ask",
            20,
            rng,
            excluded,
        )
    )
    rng.shuffle(selected)
    train_rows = [annotate(row, coverage_index=index) for index, row in enumerate(selected)]
    tune_rows = [annotate(row) for row in source_tune]

    if len(train_rows) != 200 or len(excluded) != 200:
        raise RuntimeError("First-action diagnostic must contain 200 unique Train tasks")
    if not all(
        row["extra_info"]["decision_target_action"] == "execute_sql"
        for row in train_rows
    ):
        raise RuntimeError("Every P6 first-action target must match the canonical execute probe")
    if any(
        marker in prompt_bytes(row).decode("utf-8")
        for row in train_rows + tune_rows
        for marker in FORBIDDEN_PROMPT_MARKERS
    ):
        raise RuntimeError("Reward-only first-action target leaked into a prompt")
    selected_hashes = Counter(hashlib.sha256(prompt_bytes(row)).hexdigest() for row in train_rows)
    source_hashes = Counter(hashlib.sha256(prompt_bytes(row)).hexdigest() for row in source_train)
    if any(selected_hashes[key] > source_hashes[key] for key in selected_hashes):
        raise RuntimeError("First-action curriculum changed a model-visible prompt")

    train_dbs = {str(row["extra_info"]["db_id"]) for row in train_rows}
    tune_dbs = {str(row["extra_info"]["db_id"]) for row in tune_rows}
    train_tasks = {str(row["extra_info"]["instance_id"]) for row in train_rows}
    tune_tasks = {str(row["extra_info"]["instance_id"]) for row in tune_rows}
    if train_dbs & tune_dbs or train_tasks & tune_tasks:
        raise RuntimeError("First-action Train/Tune isolation failed")

    output.mkdir(parents=True, exist_ok=False)
    pq.write_table(pa.Table.from_pylist(train_rows), output / "train.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist(tune_rows), output / "tune.parquet", compression="zstd")
    summary = {
        "protocol": "p6_first_action_focus200_v2",
        "source": str(source),
        "seed": args.seed,
        "train_rows": len(train_rows),
        "tune_rows": len(tune_rows),
        "training_steps_at_batch8": len(train_rows) // 8,
        "unique_train_tasks": len(train_tasks),
        "train_databases": len(train_dbs),
        "tune_databases": len(tune_dbs),
        "train_tune_database_overlap": sorted(train_dbs & tune_dbs),
        "train_tune_task_overlap": sorted(train_tasks & tune_tasks),
        "drift_distribution": dict(
            sorted(Counter(row["extra_info"]["drift_type"] for row in train_rows).items())
        ),
        "interaction_distribution": dict(
            sorted(
                Counter(
                    row["extra_info"]["interaction_profile"] for row in train_rows
                ).items()
            )
        ),
        "target_actions": dict(
            sorted(
                Counter(
                    row["extra_info"]["decision_target_action"] for row in train_rows
                ).items()
            )
        ),
        "prompt_bytes_unchanged": True,
        "prompt_target_leakage": False,
        "full_episode_agent_loop": True,
        "fresh_blind_rows_read": 0,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
