#!/usr/bin/env python3
"""Audit prompt coverage and episode-level advantage fields in GRPO rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def load_rollouts(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    files = sorted(root.glob("*.jsonl"), key=lambda path: int(path.stem))
    for path in files:
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--rollout-n", type=int, default=8)
    parser.add_argument("--require-full-coverage", action="store_true")
    args = parser.parse_args()

    train_rows = pq.read_table(args.train.resolve()).to_pylist()
    rollout_rows = load_rollouts(args.rollout_dir.resolve())
    prompt_groups = {
        str(row["uid"]).rsplit("_", 2)[0]
        for row in rollout_rows
        if str(row.get("uid", ""))
    }
    seen_tasks = {str(row.get("instance_id", "")) for row in rollout_rows if row.get("instance_id")}
    expected_tasks = {str(row["extra_info"]["instance_id"]) for row in train_rows}
    seen_coverage_indices = {
        int(row["coverage_index"])
        for row in rollout_rows
        if int(row.get("coverage_index", -1)) >= 0
    }
    expected_coverage_indices = {
        int(row["extra_info"]["index"])
        for row in train_rows
    }
    scopes = {str(row.get("advantage_scope", "")) for row in rollout_rows}
    mask_mismatches = sum(
        int(row.get("advantage_mask_tokens", -1))
        != int(row.get("episode_response_mask_tokens", -2))
        for row in rollout_rows
    )
    expected_groups = len(train_rows)
    result = {
        "protocol": "p6_grpo_sampling_episode_advantage_audit_v1",
        "train_rows": len(train_rows),
        "expected_unique_tasks": len(expected_tasks),
        "rollout_rows": len(rollout_rows),
        "rollout_n": args.rollout_n,
        "prompt_groups": len(prompt_groups),
        "seen_coverage_indices": len(seen_coverage_indices),
        "seen_unique_tasks": len(seen_tasks),
        "task_coverage_rate": len(seen_tasks & expected_tasks) / len(expected_tasks),
        "advantage_scopes": sorted(scopes),
        "episode_mask_mismatches": mask_mismatches,
        "full_prompt_coverage": bool(
            len(prompt_groups) >= expected_groups
            and seen_coverage_indices == expected_coverage_indices
        ),
        "full_task_coverage": expected_tasks <= seen_tasks,
        "episode_level_advantage": scopes == {"episode"} and mask_mismatches == 0,
        "fresh_blind_rows_read": 0,
    }
    if args.require_full_coverage and not all(
        (result["full_prompt_coverage"], result["full_task_coverage"], result["episode_level_advantage"])
    ):
        raise RuntimeError(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
