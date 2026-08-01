#!/usr/bin/env python3
"""Verify the frozen candidate and write the one-shot Gate112 audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FREEZE = PROJECT_ROOT / "reports/stage6/final_candidate/frozen_candidate.json"
DEFAULT_ROWS = PROJECT_ROOT / "reports/stage6/final_gate112/stage6-frozen-candidate.jsonl"
DEFAULT_SUMMARY = PROJECT_ROOT / "reports/stage6/final_gate112/summary.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports/stage6/final_gate112/audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def grouped(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    for row in rows:
        values = counts[str(row[field])]
        values[0] += 1
        values[1] += int(bool(row["task_success"]))
        values[2] += int(row["termination_reason"] == "submitted")
        values[3] += int(row["termination_reason"] == "turn_limit")
    return {
        name: {
            "tasks": value[0],
            "success": value[1],
            "success_rate": value[1] / value[0],
            "submitted": value[2],
            "turn_limit": value[3],
        }
        for name, value in sorted(counts.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Gate audit already exists: {args.output}")

    frozen = json.loads(args.freeze.read_text(encoding="utf-8"))
    mismatches: dict[str, dict[str, str]] = {}
    expected_hashes = {
        **frozen["locked_files_sha256"],
        **frozen["candidate"]["adapter_files_sha256"],
    }
    for relative, expected in expected_hashes.items():
        actual = sha256(PROJECT_ROOT / relative)
        if actual != expected:
            mismatches[relative] = {"expected": expected, "actual": actual}
    if mismatches:
        raise RuntimeError(f"Frozen candidate changed before Gate audit: {mismatches}")

    rows = [json.loads(line) for line in args.rows.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if len(rows) != 112 or summary.get("dynamic_tool_mask") is not True or summary.get("state_guards") is not True:
        raise RuntimeError("Gate output does not match the frozen 112-task protocol")

    total = len(rows)
    non_clean = [row for row in rows if row["scenario_type"] != "clean"]
    profiles = grouped(rows, "interaction_profile")
    drift_types = grouped(rows, "drift_type")
    difficulties = grouped(rows, "difficulty")
    overall = {
        "tasks": total,
        "success": sum(bool(row["task_success"]) for row in rows),
        "success_rate": sum(bool(row["task_success"]) for row in rows) / total,
        "submitted": sum(row["termination_reason"] == "submitted" for row in rows),
        "submission_rate": sum(row["termination_reason"] == "submitted" for row in rows) / total,
        "turn_limit": sum(row["termination_reason"] == "turn_limit" for row in rows),
        "turn_limit_rate": sum(row["termination_reason"] == "turn_limit" for row in rows) / total,
        "unsafe": sum(bool(row["safety"]["unsafe"]) for row in rows),
        "timeout": sum(bool(row["safety"]["timed_out"]) for row in rows),
    }
    non_clean_result = {
        "tasks": len(non_clean),
        "success": sum(bool(row["task_success"]) for row in non_clean),
        "success_rate": sum(bool(row["task_success"]) for row in non_clean) / len(non_clean),
    }
    acceptance = {
        "overall_success_gte_0_20": overall["success_rate"] >= 0.20,
        "non_clean_success_gte_2x_b0": non_clean_result["success_rate"] >= 0.14,
        "schema_only_success_gte_0_10": profiles["schema_only"]["success_rate"] >= 0.10,
        "submission_gte_0_40": overall["submission_rate"] >= 0.40,
        "turn_limit_lte_0_55": overall["turn_limit_rate"] <= 0.55,
        "unsafe_eq_0": overall["unsafe"] == 0,
        "timeout_eq_0": overall["timeout"] == 0,
    }
    audit = {
        "protocol": "driftsql_stage6_one_shot_gate_audit_v1",
        "candidate_freeze": str(args.freeze.resolve().relative_to(PROJECT_ROOT)),
        "frozen_candidate_hashes_verified": True,
        "gate_runs": 1,
        "gate_rows_sha256": sha256(args.rows),
        "gate_summary_sha256": sha256(args.summary),
        "results": {
            "overall": overall,
            "non_clean": non_clean_result,
            "by_interaction_profile": profiles,
            "by_drift_type": drift_types,
            "by_difficulty": difficulties,
        },
        "acceptance": acceptance,
        "accepted": all(acceptance.values()),
        "residual_risk": {
            "add_column": drift_types.get("add_column", {}),
            "note": (
                "Aggregate acceptance passed, but add-column remains a zero-success held-out slice. "
                "It must be addressed with a fresh Stage 7 train/tune/gate protocol, never by tuning on Gate112."
            ),
        },
    }
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"accepted": audit["accepted"], "acceptance": acceptance}, indent=2))


if __name__ == "__main__":
    main()
