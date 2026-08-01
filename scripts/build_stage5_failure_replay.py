#!/usr/bin/env python3
"""Mine failed Stage-5 rollouts and build a controlled hard-replay dataset."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def load_rollout_files(
    sources: list[tuple[Path, int | None, int | None]],
) -> list[Path]:
    """Resolve one rollout file per training step across resumed run segments."""

    by_step: dict[int, Path] = {}
    for directory, first_step, last_step in sources:
        files = sorted(directory.glob("*.jsonl"), key=lambda path: int(path.stem))
        if not files:
            raise FileNotFoundError(f"No rollout JSONL files in {directory}")
        for path in files:
            step = int(path.stem)
            if first_step is not None and step < first_step:
                continue
            if last_step is not None and step > last_step:
                continue
            previous = by_step.get(step)
            if previous is not None and previous.resolve() != path.resolve():
                raise ValueError(
                    f"Training step {step} is present in both {previous} and {path}; "
                    "use non-overlapping --rollout-segment ranges"
                )
            by_step[step] = path
    if not by_step:
        raise FileNotFoundError("No rollout JSONL files matched the configured sources")
    return [by_step[step] for step in sorted(by_step)]


def load_rollouts(directory: Path) -> list[dict[str, Any]]:
    files = load_rollout_files([(directory, None, None)])
    return load_rollout_rows(files)


def load_rollout_rows(files: list[Path]) -> list[dict[str, Any]]:
    if not files:
        raise FileNotFoundError("No rollout JSONL files were selected")
    rows = []
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                row["_rollout_file"] = path.name
                rows.append(row)
    return rows


def parse_rollout_segment(value: str) -> tuple[Path, int, int]:
    """Parse ``DIR@FIRST-LAST`` without confusing ':' in absolute paths."""

    raw_directory, separator, raw_range = value.rpartition("@")
    match = re.fullmatch(r"([0-9]+)-([0-9]+)", raw_range)
    if not separator or not raw_directory or match is None:
        raise argparse.ArgumentTypeError(
            "rollout segments must use DIR@FIRST-LAST, for example run/rollouts@1-10"
        )
    first_step, last_step = (int(part) for part in match.groups())
    if first_step < 1 or last_step < first_step:
        raise argparse.ArgumentTypeError("rollout segment step range is invalid")
    return Path(raw_directory), first_step, last_step


def mine_failures(
    rollouts: list[dict[str, Any]],
    *,
    success_rate_threshold: float,
) -> tuple[list[str], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rollouts:
        instance_id = str(row.get("instance_id", "")).strip()
        if not instance_id:
            raise ValueError(
                "Rollout is missing instance_id. Run the shaped GRPO experiment "
                "with the Stage-5 reward artifact identity patch enabled."
            )
        grouped[instance_id].append(row)

    diagnostics = []
    for instance_id, attempts in grouped.items():
        successes = sum(bool(row.get("task_success", False)) for row in attempts)
        scores = [float(row.get("score", 0.0)) for row in attempts]
        rate = successes / len(attempts)
        diagnostics.append(
            {
                "instance_id": instance_id,
                "attempts": len(attempts),
                "successes": successes,
                "success_rate": rate,
                "mean_score": mean(scores),
                "min_score": min(scores),
                "turn_limit": sum(bool(row.get("turn_limit", False)) for row in attempts),
                "missing_submit": sum(bool(row.get("missing_submit", False)) for row in attempts),
                "invalid_sql": sum(int(row.get("invalid_sql", 0)) for row in attempts),
                "unsafe": sum(bool(row.get("unsafe", False)) for row in attempts),
            }
        )
    diagnostics.sort(key=lambda item: (item["success_rate"], item["mean_score"], -item["attempts"]))
    hard_ids = [
        item["instance_id"]
        for item in diagnostics
        if item["success_rate"] < success_rate_threshold
    ]
    if not hard_ids:
        raise RuntimeError("No hard failures satisfy the configured threshold")
    return hard_ids, diagnostics


def sample_replay_rows(
    source_rows: list[dict[str, Any]],
    hard_ids: list[str],
    *,
    hard_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_id = {str(row["extra_info"]["instance_id"]): row for row in source_rows}
    missing = sorted(set(hard_ids) - set(by_id))
    if missing:
        raise ValueError(f"{len(missing)} mined instance IDs are absent from source parquet")
    hard_unique = [by_id[instance_id] for instance_id in hard_ids]
    rng = random.Random(seed)
    size = len(source_rows)
    hard_count = round(size * hard_fraction)
    base_count = size - hard_count
    replay = [rng.choice(hard_unique) for _ in range(hard_count)]
    replay.extend(rng.sample(source_rows, k=base_count))
    rng.shuffle(replay)

    # Same number of continuation updates, but without failure prioritisation.
    control = list(source_rows)
    rng.shuffle(control)
    return hard_unique, replay, control


def write_parquet(rows: list[dict[str, Any]], path: Path, schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression="zstd")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rollout-dir",
        type=Path,
        action="append",
        default=[],
        help="Whole rollout directory. Repeat only when step numbers do not overlap.",
    )
    parser.add_argument(
        "--rollout-segment",
        type=parse_rollout_segment,
        action="append",
        default=[],
        metavar="DIR@FIRST-LAST",
        help="Select a non-overlapping step range from a resumed run directory.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/processed/stratified_five_tool_v2/rl_train.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/stage5_failure_replay"),
    )
    parser.add_argument("--success-rate-threshold", type=float, default=0.5)
    parser.add_argument("--hard-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()
    sources = [(path, None, None) for path in args.rollout_dir]
    sources.extend(args.rollout_segment)
    if not sources:
        parser.error("provide --rollout-dir or --rollout-segment")
    if not 0.0 < args.success_rate_threshold <= 1.0:
        parser.error("--success-rate-threshold must be in (0, 1]")
    if not 0.0 < args.hard_fraction < 1.0:
        parser.error("--hard-fraction must be in (0, 1)")

    source_table = pq.read_table(args.source)
    source_rows = source_table.to_pylist()
    rollout_files = load_rollout_files(sources)
    rollout_rows = load_rollout_rows(rollout_files)
    hard_ids, diagnostics = mine_failures(
        rollout_rows,
        success_rate_threshold=args.success_rate_threshold,
    )
    hard, replay, control = sample_replay_rows(
        source_rows,
        hard_ids,
        hard_fraction=args.hard_fraction,
        seed=args.seed,
    )
    write_parquet(hard, args.output_dir / "hard_failures.parquet", source_table.schema)
    write_parquet(replay, args.output_dir / "mixed_replay.parquet", source_table.schema)
    write_parquet(control, args.output_dir / "uniform_control.parquet", source_table.schema)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "failure_diagnostics.jsonl").open("w", encoding="utf-8") as handle:
        for item in diagnostics:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary = {
        "protocol": "stage5_failure_replay_v2",
        "source": str(args.source.resolve()),
        "rollout_sources": [
            {
                "directory": str(directory.resolve()),
                "first_step": first_step,
                "last_step": last_step,
            }
            for directory, first_step, last_step in sources
        ],
        "rollout_steps": [int(path.stem) for path in rollout_files],
        "source_rows": len(source_rows),
        "rollout_attempts": len(rollout_rows),
        "observed_instances": len(diagnostics),
        "hard_instances": len(hard),
        "hard_fraction": args.hard_fraction,
        "replay_rows": len(replay),
        "control_rows": len(control),
        "success_rate_threshold": args.success_rate_threshold,
        "seed": args.seed,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
