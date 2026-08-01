#!/usr/bin/env python3
"""Expand execution-verified trajectories into next-action SFT examples."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from verl.utils.py_functional import convert_nested_value_to_list_recursive

from driftsql.data.tool_sft import expand_next_action_messages, use_plain_json_for_last_action


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data/processed/five_tool_sft_native_v2"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/processed/five_tool_sft_native_v3"


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary)
    temporary.replace(path)


def expand_split(
    input_dir: Path,
    output_dir: Path,
    split: str,
    *,
    plain_json_targets: bool,
    stage6_balance: bool,
) -> tuple[int, Counter[str]]:
    frame = pd.read_parquet(input_dir / f"{split}.parquet")
    expanded: list[dict[str, Any]] = []
    actions: Counter[str] = Counter()
    for trajectory_index, row in frame.iterrows():
        messages = convert_nested_value_to_list_recursive(row["messages"])
        for prefix in expand_next_action_messages(messages):
            action = str(prefix[-1]["tool_calls"][0]["function"]["name"])
            if plain_json_targets:
                prefix = use_plain_json_for_last_action(prefix)
            repeat = 1
            if stage6_balance:
                # B1's two measured deficits were missing submission and
                # add-column/compound failures.  Duplicate only supervised
                # next-action examples; task IDs and DB boundaries do not
                # change, and each copy retains the same verified context.
                repeat += int(action == "submit_solution")
                repeat += int(str(row.get("drift_type", "")) in {"add_column", "compound"})
            actions[action] += repeat
            payload = {
                    "messages": prefix,
                    "tools": str(row["tools"]),
                    "enable_thinking": bool(row["enable_thinking"]),
                    "target_action": action,
                    "trajectory_index": int(trajectory_index),
                }
            for name in (
                "task_id",
                "db_id",
                "drift_type",
                "interaction_profile",
                "difficulty",
                "failure_mode",
            ):
                if name in row:
                    payload[name] = row[name]
            expanded.extend(dict(payload) for _ in range(repeat))
    write_parquet(output_dir / f"{split}.parquet", expanded)
    return len(expanded), actions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plain-json-targets", action="store_true")
    parser.add_argument(
        "--stage6-balance",
        action="store_true",
        help="Upweight submit actions and add-column/compound verified trajectories.",
    )
    args = parser.parse_args()

    source_summary = json.loads((args.input_dir / "summary.json").read_text(encoding="utf-8"))
    splits = tuple(
        split
        for split in ("train", "tune", "dev", "val", "test")
        if split in source_summary.get("splits", {})
        and (args.input_dir / f"{split}.parquet").is_file()
    )
    if not splits:
        raise RuntimeError(f"No SFT parquet splits found in {args.input_dir}")
    split_rows: dict[str, int] = {}
    action_counts: Counter[str] = Counter()
    for split in splits:
        rows, counts = expand_split(
            args.input_dir,
            args.output_dir,
            split,
            plain_json_targets=args.plain_json_targets,
            stage6_balance=args.stage6_balance,
        )
        split_rows[split] = rows
        action_counts.update(counts)
        shutil.copy2(
            args.input_dir / f"{split}_manifest.jsonl",
            args.output_dir / f"{split}_manifest.jsonl",
        )
    for name in ("rejected.jsonl", "dev_agent_eval.jsonl", "val_agent_eval.jsonl", "test_agent_eval.jsonl"):
        source = args.input_dir / name
        if source.is_file():
            shutil.copy2(source, args.output_dir / name)

    summary = dict(source_summary)
    summary.update(
        {
            "name": "driftsql_execution_verified_five_tool_next_action_sft_v2",
            "derived_from": str(args.input_dir.resolve()),
            "supervision": "conversation prefix -> exactly one next assistant tool action",
            "loss_scope": "final assistant message only",
            "target_format": "plain_json" if args.plain_json_targets else "native_tool_call",
            "stage6_failure_balance": bool(args.stage6_balance),
            "supervision_examples": sum(split_rows.values()),
            "target_actions": dict(sorted(action_counts.items())),
        }
    )
    for split in splits:
        trajectories = int(source_summary["splits"][split]["rows"])
        summary["splits"][split] = dict(source_summary["splits"][split]) | {
            "rows": split_rows[split],
            "trajectories": trajectories,
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
