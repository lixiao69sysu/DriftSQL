#!/usr/bin/env python3
"""Build a deterministic 100-task supplement from observed on-policy failures.

The selector reads only the Train hard-pool records and completed Train rollout
files.  Difficulty is defined by repeated empirical policy failure rather than
the factory's static difficulty label.  Four disjoint, failure-oriented quotas
are filled in priority order so one task cannot occupy multiple slots.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POOL = ROOT / "data/processed/p6_scaleup_v1_rollout_pool600/train_agent_eval.jsonl"
DEFAULT_OUTPUT = ROOT / "data/processed/p6_scaleup_v1_targeted_supplement100"
DEFAULT_ROLLOUTS = [
    ROOT / "reports/p6_scaleup/on_policy_seed211/strong-sft.jsonl",
    ROOT / "reports/p6_scaleup/on_policy_seed307/strong-sft.jsonl",
    ROOT / "reports/p6_scaleup/on_policy_seed401/strong-sft.jsonl",
    ROOT / "reports/p6_scaleup/on_policy_seed101_supplement500/strong-sft.jsonl",
]
EXPECTED_NEXT_TOOL = {
    "must_ask": "ask_user",
    "knowledge_only": "get_knowledge_definition",
    "schema_only": "execute_sql",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def observation_succeeded(value: Any) -> bool:
    if isinstance(value, dict):
        return value.get("success") is True
    if not isinstance(value, str):
        return False
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return '"success": true' in value.casefold()
    return isinstance(parsed, dict) and parsed.get("success") is True


def failure_flags(row: dict[str, Any]) -> dict[str, bool]:
    failed = not bool(row.get("task_success"))
    trajectory = row.get("trajectory") or []
    tools = [turn.get("tool_name") for turn in trajectory]
    successful_execution = any(
        turn.get("tool_name") == "execute_sql"
        and observation_succeeded(turn.get("observation"))
        for turn in trajectory
    )

    post_diff_tool = None
    for index, turn in enumerate(trajectory):
        if turn.get("tool_name") == "inspect_schema_diff":
            if index + 1 < len(trajectory):
                post_diff_tool = trajectory[index + 1].get("tool_name")
            break
    expected = EXPECTED_NEXT_TOOL.get(str(row.get("interaction_profile")))

    return {
        "terminal_no_submit": failed
        and successful_execution
        and "submit_solution" not in tools,
        "compound_recovery": failed and row.get("scenario_type") == "compound",
        "must_ask_decision": failed and row.get("interaction_profile") == "must_ask",
        "post_diff_wrong_retrieval": failed
        and post_diff_tool is not None
        and expected is not None
        and post_diff_tool != expected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--rollout", type=Path, action="append", dest="rollouts")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-failure-rate", type=float, default=0.5)
    parser.add_argument("--terminal-no-submit", type=int, default=15)
    parser.add_argument("--compound-recovery", type=int, default=30)
    parser.add_argument("--must-ask-decision", type=int, default=25)
    parser.add_argument("--post-diff-wrong-retrieval", type=int, default=30)
    args = parser.parse_args()

    rollout_paths = args.rollouts or DEFAULT_ROLLOUTS
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if not 0 < args.min_failure_rate <= 1:
        parser.error("--min-failure-rate must be in (0, 1]")

    pool_rows = load_jsonl(args.pool)
    pool_by_id = {
        str(row["extra_info"]["instance_id"]): row for row in pool_rows
    }
    if len(pool_rows) != 600 or len(pool_by_id) != 600:
        raise RuntimeError(
            f"Expected a unique 600-task Train pool, got {len(pool_rows)}/{len(pool_by_id)}"
        )

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    input_counts: dict[str, int] = {}
    for path in rollout_paths:
        lowered = str(path).casefold()
        if any(token in lowered for token in ("fresh", "blind", "tune", "test", "dev")):
            raise RuntimeError(f"Non-Train rollout path is forbidden: {path}")
        rows = load_jsonl(path)
        input_counts[str(path)] = len(rows)
        seen_in_file: set[str] = set()
        for row in rows:
            task_id = str(row["instance_id"])
            if task_id not in pool_by_id:
                raise RuntimeError(f"Rollout task is outside Train pool: {task_id}")
            if task_id in seen_in_file:
                raise RuntimeError(f"Duplicate task in {path}: {task_id}")
            seen_in_file.add(task_id)
            by_task[task_id].append(row)

    if sum(input_counts.values()) != 2300 or len(by_task) != 600:
        raise RuntimeError(
            f"Expected 2,300 Train rollouts over 600 tasks, got "
            f"{sum(input_counts.values())} over {len(by_task)}"
        )

    candidates: list[dict[str, Any]] = []
    for task_id, rows in by_task.items():
        failures = sum(not bool(row.get("task_success")) for row in rows)
        failure_rate = failures / len(rows)
        flag_counts = {
            name: sum(failure_flags(row)[name] for row in rows)
            for name in failure_flags(rows[0])
        }
        if failure_rate < args.min_failure_rate:
            continue
        extra = pool_by_id[task_id]["extra_info"]
        candidates.append(
            {
                "task_id": task_id,
                "exposures": len(rows),
                "failures": failures,
                "failure_rate": failure_rate,
                "flag_counts": flag_counts,
                "difficulty": str(extra["difficulty"]),
            }
        )

    quotas = [
        ("terminal_no_submit", args.terminal_no_submit),
        ("compound_recovery", args.compound_recovery),
        ("must_ask_decision", args.must_ask_decision),
        ("post_diff_wrong_retrieval", args.post_diff_wrong_retrieval),
    ]
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for category, quota in quotas:
        eligible = [
            candidate
            for candidate in candidates
            if candidate["task_id"] not in used
            and candidate["flag_counts"][category] > 0
        ]
        eligible.sort(
            key=lambda candidate: (
                -candidate["flag_counts"][category],
                -candidate["failure_rate"],
                -candidate["failures"],
                -(candidate["difficulty"] == "hard"),
                candidate["task_id"],
            )
        )
        if len(eligible) < quota:
            raise RuntimeError(
                f"Insufficient disjoint {category} candidates: {len(eligible)}/{quota}"
            )
        for candidate in eligible[:quota]:
            candidate = dict(candidate)
            candidate["selection_category"] = category
            selected.append(candidate)
            used.add(candidate["task_id"])

    expected_total = sum(quota for _, quota in quotas)
    if len(selected) != expected_total or len(used) != expected_total:
        raise RuntimeError(f"Selection invariant failed: {len(selected)}/{len(used)}")

    output_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for candidate in selected:
        source = json.loads(json.dumps(pool_by_id[candidate["task_id"]]))
        extra = source["extra_info"]
        previous_sampling = dict(extra.get("rollout_sampling") or {})
        extra["rollout_sampling"] = {
            **previous_sampling,
            "targeted_supplement": "p6_scaleup_v1_targeted_supplement100",
            "targeted_category": candidate["selection_category"],
            "historical_exposures": candidate["exposures"],
            "historical_failures": candidate["failures"],
            "historical_failure_rate": candidate["failure_rate"],
            "fresh_blind": False,
        }
        output_rows.append(source)
        audit_rows.append(
            {
                **candidate,
                "db_id": str(extra["db_id"]),
                "drift_type": str(extra["drift_type"]),
                "interaction_profile": str(extra["interaction_profile"]),
                "scenario_type": str(extra["scenario_type"]),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.output_dir / "train_agent_eval.jsonl", output_rows)
    write_jsonl(args.output_dir / "selection_audit.jsonl", audit_rows)

    summary = {
        "protocol": "p6_scaleup_v1_targeted_supplement100",
        "source_split": "train_only",
        "fresh_blind_rows_read": False,
        "input_rollouts": input_counts,
        "input_rollout_total": sum(input_counts.values()),
        "input_failures": sum(
            not bool(row.get("task_success"))
            for rows in by_task.values()
            for row in rows
        ),
        "tasks": len(output_rows),
        "databases": len({row["db_id"] for row in audit_rows}),
        "minimum_historical_failure_rate": args.min_failure_rate,
        "mean_historical_failure_rate": sum(
            row["failure_rate"] for row in audit_rows
        )
        / len(audit_rows),
        "selection_categories": dict(
            sorted(Counter(row["selection_category"] for row in audit_rows).items())
        ),
        "historical_failure_counts": dict(
            sorted(Counter(str(row["failures"]) for row in audit_rows).items())
        ),
        "historical_exposures": dict(
            sorted(Counter(str(row["exposures"]) for row in audit_rows).items())
        ),
        "drift_types": dict(
            sorted(Counter(row["drift_type"] for row in audit_rows).items())
        ),
        "profiles": dict(
            sorted(Counter(row["interaction_profile"] for row in audit_rows).items())
        ),
        "scenario_types": dict(
            sorted(Counter(row["scenario_type"] for row in audit_rows).items())
        ),
        "difficulty_labels": dict(
            sorted(Counter(row["difficulty"] for row in audit_rows).items())
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
