#!/usr/bin/env python3
"""Audit the single frozen-candidate Stage 7 Gate106 run without rerunning it."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FREEZE = PROJECT_ROOT / "reports/stage7/final_candidate/frozen_candidate.json"
DEFAULT_DATA_SUMMARY = PROJECT_ROOT / "data/processed/stage7_gate106/summary.json"
DEFAULT_ROWS = PROJECT_ROOT / "reports/stage7/final_gate106/stage7-frozen-candidate.jsonl"
DEFAULT_SUMMARY = PROJECT_ROOT / "reports/stage7/final_gate106/summary.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports/stage7/final_gate106/audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    success = sum(bool(row["task_success"]) for row in rows)
    return {
        "tasks": total,
        "success": success,
        "success_rate": success / total,
        "executable": sum(bool(row["executable"]) for row in rows),
        "submitted": sum(row["termination_reason"] == "submitted" for row in rows),
        "turn_limit": sum(row["termination_reason"] == "turn_limit" for row in rows),
        "unsafe": sum(bool(row["safety"]["unsafe"]) for row in rows),
        "timeout": sum(bool(row["safety"]["timed_out"]) for row in rows),
    }


def grouped(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = row.get(field, row.get("extra_info", {}).get(field, ""))
        groups[str(value)].append(row)
    return {name: metrics(values) for name, values in sorted(groups.items())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--data-summary", type=Path, default=DEFAULT_DATA_SUMMARY)
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Stage 7 Gate audit already exists: {args.output}")

    frozen = json.loads(args.freeze.read_text(encoding="utf-8"))
    mismatches = {}
    expected = {
        **frozen["locked_files_sha256"],
        **frozen["candidate"]["adapter_files_sha256"],
    }
    for relative, digest in expected.items():
        actual = sha256(PROJECT_ROOT / relative)
        if actual != digest:
            mismatches[relative] = {"expected": digest, "actual": actual}
    if mismatches:
        raise RuntimeError(f"Frozen Stage 7 artifacts changed: {mismatches}")

    data_summary = json.loads(args.data_summary.read_text(encoding="utf-8"))
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in args.rows.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 106 or summary.get("episodes") != 106 or data_summary.get("tasks") != 106:
        raise RuntimeError("Gate106 output/data count mismatch")

    add_rows = [row for row in rows if row["drift_type"] == "add_column"]
    general_rows = [row for row in rows if row["drift_type"] != "add_column"]
    if len(add_rows) != 24 or len(general_rows) != 82:
        raise RuntimeError("Gate106 add/general slice mismatch")
    result = {
        "overall": metrics(rows),
        "add_column": metrics(add_rows),
        "general": metrics(general_rows),
        "by_drift_type": grouped(rows, "drift_type"),
        "add_by_wildcard_profile": grouped(add_rows, "wildcard_profile"),
        "termination_reasons": dict(sorted(Counter(row["termination_reason"] for row in rows).items())),
    }
    thresholds = frozen["one_shot_gate106"]["acceptance_precommitted"]
    acceptance = {
        "overall_success_gte_0_70": result["overall"]["success_rate"] >= thresholds["overall_success_gte_0_70"],
        "add_column_success_gte_0_125": result["add_column"]["success_rate"] >= thresholds["add_column_success_gte_0_125"],
        "general_success_gte_0_80": result["general"]["success_rate"] >= thresholds["general_success_gte_0_80"],
        "unsafe_eq_0": result["overall"]["unsafe"] == thresholds["unsafe_eq_0"],
        "timeout_eq_0": result["overall"]["timeout"] == thresholds["timeout_eq_0"],
    }
    audit = {
        "protocol": "driftsql_stage7_one_shot_gate106_audit_v1",
        "candidate_freeze": str(args.freeze.relative_to(PROJECT_ROOT)),
        "frozen_hashes_verified": True,
        "gate_candidate_runs": 1,
        "reruns_allowed": 0,
        "hashes": {
            "candidate_freeze": sha256(args.freeze),
            "gate_data_summary": sha256(args.data_summary),
            "gate_rows": sha256(args.rows),
            "gate_summary": sha256(args.summary),
        },
        "results": result,
        "acceptance": acceptance,
        "accepted": all(acceptance.values()),
        "closure_policy": (
            "Do not rerun or tune on Gate106. Any further model changes require a fresh "
            "database-disjoint Stage 8 train/tune/gate protocol."
        ),
    }
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"accepted": audit["accepted"], "results": result, "acceptance": acceptance}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
