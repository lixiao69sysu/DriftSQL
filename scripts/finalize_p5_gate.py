#!/usr/bin/env python3
"""Finalize and permanently seal the one-shot P5 Gate result."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    success = sum(bool(row["task_success"]) for row in rows)
    return {
        "tasks": len(rows),
        "task_success": success,
        "task_success_rate": success / len(rows),
        "turn_limit": sum(row["termination_reason"] == "turn_limit" for row in rows),
        "timeout": sum(bool(row["safety"]["timed_out"]) for row in rows),
        "unsafe": sum(bool(row["safety"]["unsafe"]) for row in rows),
        "average_model_calls": sum(int(row["usage"]["model_calls"]) for row in rows) / len(rows),
        "average_tool_calls": sum(int(row["usage"]["tool_calls"]) for row in rows) / len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, default=ROOT / "reports/p5/final_candidate/frozen_candidate.json")
    parser.add_argument("--gate-dir", type=Path, default=ROOT / "data/processed/p5_gate_eval")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports/p5/gate_one_shot")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/p5/final_candidate/gate_result.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"P5 Gate is already permanently sealed: {args.output}")
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    for relative, expected in freeze["candidate"]["adapter_files_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"Frozen adapter changed before Gate finalization: {relative}")
    for relative, expected in freeze["locked_files_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"Frozen P5 input changed before Gate finalization: {relative}")
    gate_input = json.loads((args.gate_dir / "summary.json").read_text(encoding="utf-8"))
    eval_state_path = args.freeze.parent / "gate_eval_state.json"
    eval_state = json.loads(eval_state_path.read_text(encoding="utf-8"))
    if eval_state.get("protocol") != "driftsql_p5_one_shot_eval_state_v1":
        raise RuntimeError("Invalid P5 Gate evaluation state")
    if eval_state.get("status") != "completed":
        raise RuntimeError("P5 Gate evaluation is not completed")
    if eval_state.get("candidate_freeze_sha256") != sha256(args.freeze):
        raise RuntimeError("P5 Gate evaluation used a different candidate freeze")
    if eval_state.get("gate_input_summary_sha256") != sha256(args.gate_dir / "summary.json"):
        raise RuntimeError("P5 Gate evaluation input changed")
    rows_path = args.report_dir / "p5-frozen-gate.jsonl"
    summary_path = args.report_dir / "summary.json"
    rows = load_jsonl(rows_path)
    evaluator_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_results = eval_state.get("result_files_sha256", {})
    if expected_results.get(summary_path.name) != sha256(summary_path):
        raise RuntimeError("P5 Gate evaluator summary changed after the one-shot run")
    if expected_results.get(rows_path.name) != sha256(rows_path):
        raise RuntimeError("P5 Gate evaluator rows changed after the one-shot run")
    metadata = load_jsonl(args.gate_dir / "gate_agent_eval.jsonl")
    hard_ids = {str(row["extra_info"]["instance_id"]) for row in metadata if row["extra_info"]["p5_turn_limit_focus"]}
    ids = [str(row["instance_id"]) for row in rows]
    if len(rows) != 18 or len(set(ids)) != 18 or ids != [str(row["extra_info"]["instance_id"]) for row in metadata]:
        raise RuntimeError("P5 Gate result identity/cardinality failed")
    if evaluator_summary["result"]["tasks"] != 18:
        raise RuntimeError("P5 Gate evaluator summary cardinality failed")
    if sha256(args.freeze) != gate_input["candidate_freeze_sha256"]:
        raise RuntimeError("P5 candidate freeze changed after Gate preparation")
    overall = summarize(rows)
    hard = summarize([row for row in rows if str(row["instance_id"]) in hard_ids])
    thresholds = freeze["one_shot_gate"]["acceptance_precommitted"]
    acceptance = {
        "overall_success_gte_0_50": overall["task_success_rate"] >= thresholds["overall_success_gte_0_50"],
        "turn_limit_focus_success_gte_0_33": (
            hard["task_success_rate"] >= thresholds["turn_limit_focus_success_gte_0_33"]
        ),
        "turn_limit_terminations_lte_6": overall["turn_limit"] <= thresholds["turn_limit_terminations_lte_6"],
        "unsafe_eq_0": overall["unsafe"] == thresholds["unsafe_eq_0"],
        "timeout_eq_0": overall["timeout"] == thresholds["timeout_eq_0"],
    }
    payload = {
        "protocol": "driftsql_p5_gate_permanent_seal_v1",
        "sealed_at_utc": datetime.now(UTC).isoformat(),
        "one_shot_runs": 1,
        "candidate": freeze["candidate"],
        "results": {
            "overall": overall,
            "turn_limit_focus": hard,
            "termination_reasons": dict(sorted(Counter(str(row["termination_reason"]) for row in rows).items())),
        },
        "acceptance_precommitted": thresholds,
        "acceptance": acceptance,
        "passed": all(acceptance.values()),
        "artifacts_sha256": {
            str(path.resolve().relative_to(ROOT)): sha256(path)
            for path in [
                args.freeze,
                eval_state_path,
                args.gate_dir / "summary.json",
                args.gate_dir / "gate_agent_eval.jsonl",
                summary_path,
                rows_path,
            ]
        },
        "policy": "P5 Gate is permanently sealed; no tuning, replay mining, model selection or rerun is allowed.",
        "stage8_gate55_read": False,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lifecycle = args.freeze.parent / "gate_lifecycle.jsonl"
    with lifecycle.open("a", encoding="utf-8") as handle:
        lifecycle_event = {
            "event": "gate_permanently_sealed",
            "at": datetime.now(UTC).isoformat(),
            "result_sha256": sha256(args.output),
            "passed": payload["passed"],
        }
        handle.write(json.dumps(lifecycle_event, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(
        json.dumps(
            {
                "output": str(args.output),
                "passed": payload["passed"],
                "results": payload["results"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
