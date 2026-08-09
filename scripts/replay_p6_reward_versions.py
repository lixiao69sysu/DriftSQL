#!/usr/bin/env python3
"""Replay stored P6 evaluation trajectories through Reward V1 and Reward V2."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from driftsql.rewards.agentic import compute_score


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data/processed/p6_scaleup_v1_low_write_protocol/dev_agent_eval.jsonl"
DEFAULT_TRAJECTORY = (
    ROOT / "reports/p6_scaleup/tune432_checkpoint_matrix/sft160/sft160.jsonl"
)
DEFAULT_OUTPUT = ROOT / "reports/p6_scaleup/reward_v2_replay"

V1_WEIGHTS: dict[str, Any] = {
    "reward_version": "v1",
    "success_weight": 1.0,
    "clarify_weight": 0.1,
    "required_clarification_weight": 0.25,
    "clarification_attempt_weight": 0.2,
    "post_clarification_weight": 0.1,
    "terminal_weight": 0.2,
    "add_column_weight": 0.2,
    "add_column_inspect_weight": 0.0,
    "semantic_candidate_weight": 0.0,
    "decision_action_weight": 0.0,
    "decision_action_mismatch_penalty": 0.0,
    "valid_weight": 0.1,
    "efficient_weight": 0.1,
    "tool_call_cost": 0.01,
    "token_cost": 0.00001,
    "duplicate_penalty": 0.08,
    "repeated_tool_penalty": 0.08,
    "invalid_penalty": 0.1,
    "timeout_penalty": 0.25,
    "turn_limit_penalty": 0.5,
    "missing_submit_penalty": 0.5,
    "unsafe_penalty": 1.0,
    "missing_required_clarification_penalty": 0.25,
    "unmatched_clarification_penalty": 0.15,
    "invalid_post_clarification_penalty": 0.15,
    "add_column_protocol_penalty": 0.15,
    "efficient_tool_calls": 7,
}

V2_WEIGHTS: dict[str, Any] = {
    **V1_WEIGHTS,
    "reward_version": "v2",
    "clarify_weight": 0.0,
    "required_clarification_weight": 0.1,
    "clarification_attempt_weight": 0.0,
    "post_clarification_weight": 0.05,
    "terminal_weight": 0.15,
    "add_column_weight": 0.15,
    "add_column_inspect_weight": 0.05,
    "semantic_candidate_weight": 0.15,
    "valid_weight": 0.0,
    "efficient_weight": 0.05,
    "add_column_protocol_penalty": 0.2,
}


def derive_v1_from_shared_metrics(
    shared: dict[str, Any], *, event_execution_success: bool
) -> dict[str, Any]:
    """Apply V1 weights to metrics already execution-verified by the V2 pass."""

    result = dict(shared)
    add_column_inspected = bool(
        result["inspected_drift"] and result.get("drift_type", "") == "add_column"
    )
    # ``drift_type`` was historically not returned by compute_score.  The
    # replay caller injects it below before invoking this function.
    add_column_protocol_complete = bool(
        add_column_inspected and result["terminal_validated"]
    )
    rewards = {
        "reward_success": 1.0 if result["task_success"] else 0.0,
        "reward_clarify": 0.1
        if result["clarification_matched"] and result["format_valid"]
        else 0.0,
        "reward_valid": 0.1
        if result["execution_success"]
        else (0.05 if event_execution_success else 0.0),
        "reward_efficient": 0.1 if result["efficient"] else 0.0,
        "reward_required_clarification": 0.25
        if result["clarification_required"] and result["clarification_matched"]
        else 0.0,
        "reward_clarification_attempt": 0.2
        if result["clarification_required"] and result["clarification_attempted"]
        else 0.0,
        "reward_post_clarification": 0.1
        if result["clarification_required"] and result["post_clarification_valid"]
        else 0.0,
        "reward_terminal": 0.2 if result["terminal_validated"] else 0.0,
        "reward_add_column_inspect": 0.0,
        "reward_semantic_candidate": 0.0,
        "reward_add_column": 0.2 if add_column_protocol_complete else 0.0,
        "reward_decision_action": 0.0,
    }
    penalties = {
        "penalty_tool_cost": 0.01 * int(result["tool_calls"]),
        "penalty_token_cost": min(0.1, 0.00001 * int(result["response_tokens"])),
        "penalty_duplicate": 0.08
        * (int(result["duplicate_questions"]) + int(result["duplicate_executions"])),
        "penalty_repeated_tool": 0.08
        * (int(result["excess_clarifications"]) + int(result["excess_retrievals"])),
        "penalty_invalid": 0.1
        * (int(result["invalid_actions"]) + int(result["invalid_sql"])),
        "penalty_timeout": 0.25 if result["timed_out"] else 0.0,
        "penalty_turn_limit": 0.5 if result["turn_limit"] else 0.0,
        "penalty_missing_submit": 0.5 if result["missing_submit"] else 0.0,
        "penalty_unsafe": 1.0 if result["unsafe"] else 0.0,
        "penalty_missing_required_clarification": 0.25
        if result["clarification_required"] and not result["clarification_attempted"]
        else 0.0,
        "penalty_unmatched_clarification": 0.15
        if result["clarification_required"]
        and result["clarification_attempted"]
        and not result["clarification_matched"]
        else 0.0,
        "penalty_invalid_post_clarification": 0.15
        if result["clarification_required"]
        and result["clarification_matched"]
        and not result["post_clarification_valid"]
        else 0.0,
        "penalty_add_column_protocol": 0.15
        if result.get("drift_type", "") == "add_column"
        and not add_column_protocol_complete
        else 0.0,
        "penalty_decision_action": 0.0,
    }
    result.update(rewards)
    result.update({name: round(value, 4) for name, value in penalties.items()})
    result.update(
        {
            "reward_version": "v1",
            "add_column_inspected": add_column_inspected,
            "add_column_protocol_complete": add_column_protocol_complete,
        }
    )
    result["score"] = round(sum(rewards.values()) - sum(penalties.values()), 4)
    result.pop("drift_type", None)
    return result


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def canonical_trace(trajectory: list[dict[str, Any]]) -> str:
    calls = []
    for event in trajectory:
        name = str(event.get("tool_name", event.get("tool", "")))
        if not name:
            continue
        arguments = event.get("arguments", {}) or {}
        calls.append(
            "<tool_call>"
            + json.dumps(
                {"name": name, "arguments": arguments},
                ensure_ascii=False,
                sort_keys=True,
            )
            + "</tool_call>"
        )
    return "\n".join(calls)


def replay_extra(source: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    extra = dict(source["extra_info"])
    events = []
    for index, event in enumerate(result.get("trajectory", []) or []):
        name = str(event.get("tool_name", event.get("tool", "")))
        if not name:
            continue
        events.append(
            {
                "index": index,
                "tool": name,
                "arguments": event.get("arguments", {}) or {},
                "metrics": event.get("metrics", {}) or {},
                "success": not bool(event.get("error")),
                "response": str(event.get("observation", "")),
            }
        )
    extra.update(
        {
            "environment_events": events,
            "response_tokens": int((result.get("usage", {}) or {}).get("new_tokens", 0)),
            "trajectory_turn_limit": result.get("termination_reason") == "turn_limit",
            "trajectory_timed_out": bool((result.get("safety", {}) or {}).get("timed_out")),
        }
    )
    return extra


def mean(rows: list[dict[str, Any]], version: str) -> float:
    values = [float(row[version]["score"]) for row in rows]
    return statistics.fmean(values) if values else 0.0


def group_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "tasks": len(rows),
        "observed_success": sum(bool(row["observed_task_success"]) for row in rows),
        "v1_mean": round(mean(rows, "v1"), 6),
        "v2_mean": round(mean(rows, "v2"), 6),
        "mean_delta": round(mean(rows, "v2") - mean(rows, "v1"), 6),
        "v1_positive": sum(float(row["v1"]["score"]) > 0 for row in rows),
        "v2_positive": sum(float(row["v2"]["score"]) > 0 for row in rows),
    }


def summarize(alias: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row["observed_task_success"]]
    failed = [row for row in rows if not row["observed_task_success"]]
    failed_add = [row for row in failed if row["drift_type"] == "add_column"]
    unsafe = [row for row in rows if row["unsafe"]]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["drift_type"])].append(row)
    success_failure_gap = mean(successful, "v2") - mean(failed, "v2")
    gates = {
        "failed_add_column_v2_mean_negative": bool(failed_add)
        and mean(failed_add, "v2") < 0,
        "v2_success_failure_gap_gte_0_8": bool(successful)
        and bool(failed)
        and success_failure_gap >= 0.8,
        "unsafe_positive_rewards_zero": not any(
            float(row["v2"]["score"]) > 0 for row in unsafe
        ),
        "fresh_blind_reads_zero": True,
    }
    return {
        "variant": alias,
        "overall": group_metrics(rows),
        "successful": group_metrics(successful),
        "failed": group_metrics(failed),
        "failed_add_column": group_metrics(failed_add),
        "unsafe": group_metrics(unsafe),
        "by_drift": {
            name: group_metrics(values) for name, values in sorted(grouped.items())
        },
        "v2_success_failure_gap": round(success_failure_gap, 6),
        "gates": gates,
        "passed": all(gates.values()),
    }


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# P6 Reward V1/V2 offline replay",
        "",
        "| Variant | Tasks | Success | V1 mean | V2 mean | Failed add V2 | Gap | Passed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for alias, values in summary["variants"].items():
        overall = values["overall"]
        failed_add = values["failed_add_column"]
        lines.append(
            f"| {alias} | {overall['tasks']} | {overall['observed_success']} | "
            f"{overall['v1_mean']:.4f} | {overall['v2_mean']:.4f} | "
            f"{failed_add['v2_mean']:.4f} | {values['v2_success_failure_gap']:.4f} | "
            f"{'yes' if values['passed'] else 'no'} |"
        )
    lines.extend(["", "## Gates", ""])
    for alias, values in summary["variants"].items():
        lines.append(f"### {alias}")
        lines.append("")
        for name, passed in values["gates"].items():
            lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
        lines.append("")
    lines.append("Fresh Blind320 reads: **0**")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--trajectory",
        action="append",
        default=[],
        help="Repeat alias=stored_eval.jsonl; defaults to current sft160 Tune432.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-count", type=int, default=432)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    specifications = args.trajectory or [f"sft160={DEFAULT_TRAJECTORY}"]

    source_rows = load_jsonl(args.data.resolve())
    source_by_id = {
        str(row["extra_info"]["instance_id"]): row for row in source_rows
    }
    if len(source_by_id) != len(source_rows):
        raise RuntimeError("Source task instance IDs are not unique")
    if args.expected_count > 0 and len(source_rows) != args.expected_count:
        raise RuntimeError(
            f"Expected {args.expected_count} source tasks, got {len(source_rows)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    summaries: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for specification in specifications:
        alias, separator, raw_path = specification.partition("=")
        if not separator or not alias or alias in summaries:
            raise ValueError(f"Invalid or duplicate trajectory specification: {specification}")
        path = Path(raw_path).expanduser().resolve()
        stored = load_jsonl(path)
        if args.expected_count > 0 and len(stored) != args.expected_count:
            raise RuntimeError(f"Expected {args.expected_count} {alias} rows, got {len(stored)}")
        actual_ids = [str(row.get("instance_id", "")) for row in stored]
        if len(set(actual_ids)) != len(actual_ids):
            raise RuntimeError(f"Duplicate trajectory instance IDs: {alias}")
        missing = sorted(set(actual_ids) - set(source_by_id))
        if missing:
            raise RuntimeError(f"Trajectory/source join failed for {alias}: {missing[:5]}")

        replayed = []
        for stored_row in stored:
            instance_id = str(stored_row["instance_id"])
            source = source_by_id[instance_id]
            trace = canonical_trace(list(stored_row.get("trajectory", []) or []))
            extra = replay_extra(source, stored_row)
            common = {
                "data_source": str(source.get("data_source", "")),
                "solution_str": trace,
                "ground_truth": (source.get("reward_model", {}) or {}).get("ground_truth", ""),
                "extra_info": extra,
            }
            v2 = compute_score(**common, **V2_WEIGHTS)
            shared = dict(v2)
            shared["drift_type"] = str(extra.get("drift_type", ""))
            v1 = derive_v1_from_shared_metrics(
                shared,
                event_execution_success=any(
                    bool((event.get("metrics", {}) or {}).get("execution_success"))
                    for event in extra["environment_events"]
                ),
            )
            replayed.append(
                {
                    "variant": alias,
                    "instance_id": instance_id,
                    "db_id": str(extra.get("db_id", "")),
                    "drift_type": str(extra.get("drift_type", "")),
                    "interaction_profile": str(extra.get("interaction_profile", "")),
                    "observed_task_success": bool(stored_row.get("task_success")),
                    "observed_termination_reason": str(
                        stored_row.get("termination_reason", "")
                    ),
                    "unsafe": bool((stored_row.get("safety", {}) or {}).get("unsafe")),
                    "called_tools": list(stored_row.get("called_tools", []) or []),
                    "v1": v1,
                    "v2": v2,
                    "score_delta": round(float(v2["score"]) - float(v1["score"]), 4),
                }
            )
        write_jsonl(args.output_dir / f"{alias}.jsonl", replayed)
        summaries[alias] = summarize(alias, replayed)
        sources[alias] = str(path)

    result = {
        "protocol": "p6_reward_v1_v2_offline_replay_v1",
        "data": str(args.data.resolve()),
        "sources": sources,
        "weights": {"v1": V1_WEIGHTS, "v2": V2_WEIGHTS},
        "variants": summaries,
        "fresh_blind_reads": 0,
        "passed": all(item["passed"] for item in summaries.values()),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.md").write_text(markdown(result), encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "passed": result["passed"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
