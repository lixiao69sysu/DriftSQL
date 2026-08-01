#!/usr/bin/env python3
"""Summarize real (non-padding) Stage 4 VERL rollout artifacts."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from sqlglot import exp, parse_one

from driftsql.tool_calls import extract_tool_call_dicts


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "checkpoints/stage4_five_tool_grpo_3b_v2/rollouts"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports/stage4/training_rollouts.json"


def unsafe_sql(sql: str) -> bool:
    if not sql.strip():
        return False
    if sql.lstrip().upper().startswith("EXPLAIN"):
        return False
    try:
        expression = parse_one(sql, read="sqlite")
    except Exception:
        return False
    return not isinstance(expression, (exp.Query, exp.Subquery))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-tool-calls", type=int, default=7)
    args = parser.parse_args()

    paths = sorted(args.input_dir.glob("*.jsonl"), key=lambda path: int(path.stem))
    step_rows: list[dict[str, Any]] = []
    all_scores: list[float] = []
    all_tool_counts: list[int] = []
    total_success = 0
    total_submitted = 0
    total_unsafe = 0
    total_turn_limit_proxy = 0
    for path in paths:
        raw = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        rows = [row for row in raw if str(row.get("input", "")).strip()]
        scores = [float(row["score"]) for row in rows]
        calls = [extract_tool_call_dicts(str(row.get("output", ""))) for row in rows]
        successes = sum(score > 0.5 for score in scores)
        submitted = sum(any(call.get("name") == "submit_solution" for call in trace) for trace in calls)
        unsafe = sum(
            any(
                call.get("name") in {"execute_sql", "submit_solution"}
                and unsafe_sql(str((call.get("arguments") or {}).get("sql", "")))
                for call in trace
            )
            for trace in calls
        )
        turn_limit_proxy = sum(
            len(trace) >= args.max_tool_calls
            and not any(call.get("name") == "submit_solution" for call in trace)
            for trace in calls
        )
        step_rows.append(
            {
                "step": int(path.stem),
                "rollouts": len(rows),
                "mean_reward": statistics.mean(scores),
                "min_reward": min(scores),
                "max_reward": max(scores),
                "successful": successes,
                "submitted": submitted,
                "unsafe": unsafe,
                "turn_limit_proxy": turn_limit_proxy,
                "mean_tool_calls": statistics.mean(len(trace) for trace in calls),
            }
        )
        all_scores.extend(scores)
        all_tool_counts.extend(len(trace) for trace in calls)
        total_success += successes
        total_submitted += submitted
        total_unsafe += unsafe
        total_turn_limit_proxy += turn_limit_proxy

    result = {
        "steps": len(step_rows),
        "rollouts": len(all_scores),
        "mean_reward": statistics.mean(all_scores),
        "min_reward": min(all_scores),
        "max_reward": max(all_scores),
        "successful": total_success,
        "success_rate": total_success / len(all_scores),
        "submitted": total_submitted,
        "unsafe": total_unsafe,
        "turn_limit_proxy": total_turn_limit_proxy,
        "mean_tool_calls": statistics.mean(all_tool_counts),
        "by_step": step_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
