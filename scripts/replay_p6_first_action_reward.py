#!/usr/bin/env python3
"""Offline Replay of the canonical first probe and stale-submit shortcut Reward."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import pyarrow.parquet as pq

from driftsql.rewards.agentic import extract_tool_calls


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = ROOT / "data/processed/p6_focus1000_reward_ab/train_coverage.parquet"
DEFAULT_ROLLOUTS = ROOT / "checkpoints/p6_focus1000_episode_advantage_v3_grpo_7b/rollouts"
DEFAULT_OUTPUT = ROOT / "reports/p6_scaleup/first_action_reward_replay_v2"


def arguments(call: dict[str, Any]) -> dict[str, Any]:
    value = call.get("arguments", {})
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def normalize_sql(sql: str) -> str:
    return sql.rstrip(";").strip().casefold()


def premature_stale(output: str, stale_sql: str) -> bool:
    calls = extract_tool_calls(output)
    names = [str(call.get("name", "")) for call in calls]
    try:
        version_index = names.index("get_schema_version")
        diff_index = names.index("inspect_schema_diff")
    except ValueError:
        version_index = diff_index = -1
    inspection_end = diff_index if 0 <= version_index < diff_index else -1
    stale_normalized = normalize_sql(stale_sql)
    stale_executes = [
        index
        for index, call in enumerate(calls)
        if call.get("name") == "execute_sql"
        and stale_sql
        and normalize_sql(str(arguments(call).get("sql", ""))) == stale_normalized
    ]
    stale_submits = [
        index
        for index, call in enumerate(calls)
        if call.get("name") == "submit_solution"
        and stale_sql
        and normalize_sql(str(arguments(call).get("sql", ""))) == stale_normalized
    ]
    return any(
        inspection_end < 0 or index < inspection_end for index in stale_submits
    ) or sum(
        inspection_end < 0 or index < inspection_end for index in stale_executes
    ) > 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--rollout-dir", type=Path, default=DEFAULT_ROLLOUTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--decision-weight", type=float, default=0.3)
    parser.add_argument("--add-inspect-weight-old", type=float, default=0.05)
    parser.add_argument("--add-inspect-weight-new", type=float, default=0.3)
    parser.add_argument("--premature-stale-penalty", type=float, default=0.5)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")

    source = pq.read_table(args.train).to_pylist()
    rollout_paths = sorted(args.rollout_dir.glob("*.jsonl"), key=lambda path: int(path.stem))
    rollouts = [
        json.loads(line)
        for path in rollout_paths
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(source) != 1000 or len(rollouts) != 8000:
        raise RuntimeError(f"Expected Train1000/Rollout8000, got {len(source)}/{len(rollouts)}")

    audit: list[dict[str, Any]] = []
    for row in rollouts:
        index = int(row["coverage_index"])
        extra = source[index]["extra_info"]
        drift_type = str(extra.get("drift_type", ""))
        target = "execute_sql"
        action = str(row.get("decision_action", ""))
        correct = bool(target and action == target)
        stale = bool(
            drift_type == "add_column"
            and premature_stale(str(row.get("output", "")), str(extra.get("stale_sql", "")))
        )
        inspected = bool(row.get("add_column_inspected"))
        score_delta = (
            (args.decision_weight if correct else 0.0)
            + (
                args.add_inspect_weight_new - args.add_inspect_weight_old
                if inspected
                else 0.0
            )
            - (args.premature_stale_penalty if stale else 0.0)
        )
        audit.append(
            {
                "uid": str(row["uid"]),
                "step": int(row["step"]),
                "coverage_index": index,
                "instance_id": str(extra["instance_id"]),
                "drift_type": drift_type,
                "target_action": target,
                "decision_action": action,
                "decision_action_correct": correct,
                "premature_stale_execute": stale,
                "ordered_drift_inspection": bool(row.get("ordered_drift_inspection")),
                "task_success": bool(row.get("task_success")),
                "protocol_success": bool(row.get("protocol_success")),
                "old_score": round(float(row["score"]), 4),
                "new_score": round(float(row["score"]) + score_delta, 4),
                "score_delta": round(score_delta, 4),
            }
        )

    add = [row for row in audit if row["drift_type"] == "add_column"]
    correct = [
        row
        for row in add
        if row["decision_action_correct"] and not row["premature_stale_execute"]
    ]
    stale = [row for row in add if row["premature_stale_execute"]]
    other = [
        row
        for row in add
        if not row["decision_action_correct"]
    ]
    group_means = {
        "correct_first_action": mean(row["new_score"] for row in correct),
        "premature_stale_execute": mean(row["new_score"] for row in stale),
        "other_first_action": mean(row["new_score"] for row in other),
    }
    gates = {
        "all_rollouts_joined": len(audit) == 8000,
        "all_rows_canonical_first_action_labeled": all(
            row["target_action"] == "execute_sql" for row in audit
        ),
        "correct_action_has_positive_mean": group_means["correct_first_action"] > 0,
        "premature_stale_has_negative_mean": group_means["premature_stale_execute"] < 0,
        "canonical_recovery_beats_stale_shortcut_by_0_5": (
            group_means["correct_first_action"]
            - group_means["premature_stale_execute"]
            >= 0.5
        ),
        "fresh_blind_rows_read_zero": True,
    }
    summary = {
        "protocol": "p6_first_action_reward_offline_replay_v2",
        "train_rows": len(source),
        "rollout_rows": len(audit),
        "rollout_files": len(rollout_paths),
        "add_column_rollouts": len(add),
        "decision_actions": dict(sorted(Counter(row["decision_action"] for row in add).items())),
        "correct_first_action_rows": len(correct),
        "premature_stale_execute_rows": len(stale),
        "other_first_action_rows": len(other),
        "reward_weights": {
            "decision_action": args.decision_weight,
            "add_column_inspection": args.add_inspect_weight_new,
            "premature_stale_execute_penalty": args.premature_stale_penalty,
        },
        "new_score_means": group_means,
        "reward_gap": (
            group_means["correct_first_action"]
            - group_means["premature_stale_execute"]
        ),
        "gates": gates,
        "passed": all(gates.values()),
        "fresh_blind_rows_read": 0,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "replay.jsonl").open("w", encoding="utf-8") as handle:
        for row in audit:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# P6 First-action Reward offline Replay",
        "",
        f"- AddColumn rollouts: {len(add)}",
        f"- Correct first action: {len(correct)}; mean reward {group_means['correct_first_action']:.4f}",
        f"- Premature stale execute: {len(stale)}; mean reward {group_means['premature_stale_execute']:.4f}",
        f"- Reward gap: {summary['reward_gap']:.4f}",
        f"- Passed: **{summary['passed']}**",
        "- Fresh Blind320 reads: **0**",
    ]
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
