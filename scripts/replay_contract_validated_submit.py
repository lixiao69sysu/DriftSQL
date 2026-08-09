#!/usr/bin/env python3
"""Replay Tune trajectories with a conservative contract-validated submit."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from driftsql.controllers import find_contract_validated_submission


ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/p5_grpo/tune_agent_eval.jsonl")
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temporary-root", type=Path, default=ROOT / "data/tmp")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    args = parser.parse_args()

    for path in (args.data, args.rows):
        if "gate" in {part.casefold() for part in path.resolve().parts}:
            raise RuntimeError(f"Gate input is forbidden for controller Replay: {path}")
    metadata = load_jsonl(args.data)
    if any(str(row.get("extra_info", {}).get("p5_split", "")) != "tune" for row in metadata):
        raise RuntimeError("Controller Replay accepts P5 Tune rows only")
    by_id = {str(row["extra_info"]["instance_id"]): row["extra_info"] for row in metadata}
    input_rows = load_jsonl(args.rows)
    expected_ids = [str(row["extra_info"]["instance_id"]) for row in metadata]
    if [str(row["instance_id"]) for row in input_rows] != expected_ids:
        raise RuntimeError("Trajectory identity/order does not match P5 Tune")

    output_rows = copy.deepcopy(input_rows)
    decisions: list[dict[str, Any]] = []
    applied = 0
    for row in output_rows:
        decision = find_contract_validated_submission(
            list(row.get("trajectory", [])),
            by_id[str(row["instance_id"])],
            temporary_root=args.temporary_root,
            timeout_seconds=args.timeout_seconds,
        )
        decision_record = {"instance_id": row["instance_id"], **decision.to_dict()}
        decisions.append(decision_record)
        if bool(row.get("task_success")) or not decision.accepted:
            continue
        applied += 1
        row["final_sql"] = decision.sql
        row["termination_reason"] = "contract_validated_auto_submit"
        row["executable"] = True
        row["task_success"] = True
        row["error"] = ""
        row.setdefault("called_tools", []).append("submit_solution")
        row.setdefault("usage", {})["tool_calls"] = int(row.get("usage", {}).get("tool_calls", 0)) + 1
        row["controller"] = {
            "name": "contract_validated_submit_v1",
            "model_calls_added": 0,
            "tool_calls_added": 1,
            "decision": decision.to_dict(),
        }
        row.setdefault("trajectory", []).append(
            {
                "turn": len(row.get("trajectory", [])),
                "tool_name": "submit_solution",
                "arguments": {"sql": decision.sql},
                "observation": json.dumps(
                    {
                        "accepted": True,
                        "controller": "contract_validated_submit_v1",
                        "result_contract_match": True,
                    },
                    ensure_ascii=False,
                ),
                "metrics": {
                    "submitted": True,
                    "controller_applied": True,
                    "result_contract_match": True,
                },
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.output_dir / "replayed.jsonl", output_rows)
    write_jsonl(args.output_dir / "decisions.jsonl", decisions)
    original_success = sum(bool(row.get("task_success")) for row in input_rows)
    replayed_success = sum(bool(row.get("task_success")) for row in output_rows)
    summary = {
        "protocol": "driftsql_contract_validated_submit_replay_v1",
        "scope": "P5 Tune only; Gate rows are forbidden",
        "rows": len(input_rows),
        "original_success": original_success,
        "replayed_success": replayed_success,
        "success_delta": replayed_success - original_success,
        "controller_applied": applied,
        "accepted_contracts": sum(bool(item["accepted"]) for item in decisions),
        "decision_reasons": dict(sorted(Counter(str(item["reason"]) for item in decisions).items())),
        "unsafe_auto_submissions": sum(
            bool(row.get("controller")) and bool(row.get("safety", {}).get("unsafe"))
            for row in output_rows
        ),
        "timeouts": sum(bool(row.get("safety", {}).get("timed_out")) for row in output_rows),
        "gate_rows_read": False,
        "model_calls_added": 0,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
