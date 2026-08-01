#!/usr/bin/env python3
"""Verify, summarize, and permanently seal the one-shot Stage 8 Gate55."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "reports/stage8/final_candidate/frozen_candidate.json"
GATE_INPUT_DIR = ROOT / "data/processed/stage8_gate_eval"
GATE_REPORT_DIR = ROOT / "reports/stage8/gate55_sft20_one_shot_process_isolated"
OUTPUT = ROOT / "reports/stage8/final_candidate/gate55_result.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    success = sum(bool(row["task_success"]) for row in rows)
    return {
        "tasks": len(rows),
        "task_success": success,
        "task_success_rate": success / len(rows),
        "submitted": sum(row["termination_reason"] == "submitted" for row in rows),
        "turn_limit": sum(row["termination_reason"] == "turn_limit" for row in rows),
        "timeout": sum(row["termination_reason"] == "timeout" for row in rows),
        "unsafe": sum(bool(row["safety"]["unsafe"]) for row in rows),
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Stage 8 Gate55 already permanently sealed: {OUTPUT}")
    freeze = load(FREEZE)
    gate_input = load(GATE_INPUT_DIR / "summary.json")
    gate_summary_path = GATE_REPORT_DIR / "summary.json"
    gate_rows_path = GATE_REPORT_DIR / "stage8-sft20-gate55.jsonl"
    gate_summary = load(gate_summary_path)
    gate_rows = load_jsonl(gate_rows_path)

    if sha256(FREEZE) != gate_input["candidate_freeze_sha256"]:
        raise RuntimeError("Candidate freeze changed after Gate input creation")
    eval_path = GATE_INPUT_DIR / "gate55_agent_eval.jsonl"
    audit_path = GATE_INPUT_DIR / "gate55_audit.jsonl"
    if sha256(eval_path) != gate_input["eval_jsonl_sha256"]:
        raise RuntimeError("Gate evaluator input changed")
    if sha256(audit_path) != gate_input["audit_jsonl_sha256"]:
        raise RuntimeError("Gate evaluator audit changed")
    for relative, expected in freeze["candidate"]["adapter_files_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"Frozen candidate changed: {relative}")
    if len(gate_rows) != 55 or len({row["instance_id"] for row in gate_rows}) != 55:
        raise RuntimeError("Gate55 result identity/cardinality failed")
    if gate_summary["result"]["tasks"] != 55:
        raise RuntimeError("Gate55 aggregate cardinality failed")

    add = [row for row in gate_rows if row["drift_type"] == "add_column"]
    general = [row for row in gate_rows if row["drift_type"] != "add_column"]
    overall_result = summarize(gate_rows)
    add_result = summarize(add)
    general_result = summarize(general)
    per_drift = {
        drift_type: summarize([row for row in gate_rows if row["drift_type"] == drift_type])
        for drift_type in sorted({row["drift_type"] for row in gate_rows})
    }
    acceptance = {
        "overall_success_gte_0_55": overall_result["task_success_rate"] >= 0.55,
        "add_column_success_gte_0_20": add_result["task_success_rate"] >= 0.20,
        "general_success_gte_0_80": general_result["task_success_rate"] >= 0.80,
        "unsafe_eq_0": overall_result["unsafe"] == 0,
        "timeout_eq_0": overall_result["timeout"] == 0,
    }
    payload = {
        "protocol": "driftsql_stage8_gate55_permanent_seal_v1",
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "one_shot_runs": 1,
        "candidate": freeze["candidate"],
        "results": {
            "overall": overall_result,
            "add_column": add_result,
            "general": general_result,
            "per_drift": per_drift,
            "termination_reasons": dict(
                sorted(Counter(row["termination_reason"] for row in gate_rows).items())
            ),
        },
        "acceptance_precommitted": freeze["one_shot_gate55"]["acceptance_precommitted"],
        "acceptance": acceptance,
        "passed": all(acceptance.values()),
        "artifacts_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [FREEZE, eval_path, audit_path, gate_summary_path, gate_rows_path]
        },
        "policy": "Gate55 is permanently sealed; no model selection, training, or rerun is allowed.",
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "passed": payload["passed"], "results": payload["results"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
