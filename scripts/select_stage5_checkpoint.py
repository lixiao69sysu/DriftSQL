#!/usr/bin/env python3
"""Select a Stage-5 GRPO checkpoint with a fixed Dev-only ordering."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def checkpoint_step(alias: str) -> int:
    match = re.search(r"(?:step|global_step_)([0-9]+)", alias)
    return int(match.group(1)) if match else 2**31 - 1


def selection_key(summary: dict[str, Any]) -> tuple[Any, ...]:
    """Higher is better; the final tie-breaker prefers the earlier checkpoint."""

    return (
        int(summary["task_success"]),
        int(summary["executable"]),
        int(summary["submitted"]),
        -int(summary.get("unsafe_actions", summary.get("unsafe_tasks", 0))),
        -int(summary.get("timeout_tasks", 0)),
        -int(summary.get("turn_limit", 0)),
        -int(summary.get("duplicate_question_tasks", 0)),
        -int(summary.get("duplicate_execution_tasks", 0)),
        -float(summary.get("average_tool_calls", 0.0)),
        -checkpoint_step(str(summary["variant"])),
    )


def select_checkpoint(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not summaries:
        raise ValueError("No checkpoint summaries were supplied")
    task_counts = {int(item["tasks"]) for item in summaries}
    if len(task_counts) != 1:
        raise ValueError(f"Checkpoint task counts differ: {sorted(task_counts)}")
    aliases = [str(item["variant"]) for item in summaries]
    if len(set(aliases)) != len(aliases):
        raise ValueError("Checkpoint aliases must be unique")
    return max(summaries, key=selection_key)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--adapter-spec",
        action="append",
        default=[],
        help="Repeat alias=path to record the portable adapter selected for the next stage.",
    )
    args = parser.parse_args()

    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    summaries = list(payload.get("variants", []))
    selected = select_checkpoint(summaries)
    adapters: dict[str, str] = {}
    for spec in args.adapter_spec:
        if "=" not in spec:
            parser.error("--adapter-spec must use alias=path")
        alias, raw_path = spec.split("=", 1)
        adapters[alias] = str(Path(raw_path).resolve())
    selected_alias = str(selected["variant"])
    if adapters and selected_alias not in adapters:
        parser.error(f"No adapter path supplied for selected alias {selected_alias!r}")

    result = {
        "protocol": "stage5_dev_checkpoint_selection_v1",
        "selection_data": str(args.summary.resolve()),
        "selection_split": "dev",
        "selection_order": [
            "task_success_desc",
            "executable_desc",
            "submitted_desc",
            "unsafe_actions_asc",
            "timeout_tasks_asc",
            "turn_limit_asc",
            "duplicate_question_tasks_asc",
            "duplicate_execution_tasks_asc",
            "average_tool_calls_asc",
            "checkpoint_step_asc",
        ],
        "selected_variant": selected_alias,
        "selected_adapter": adapters.get(selected_alias, ""),
        "selected_metrics": selected,
        "ranked_variants": [
            item["variant"] for item in sorted(summaries, key=selection_key, reverse=True)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
