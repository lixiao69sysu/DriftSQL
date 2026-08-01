#!/usr/bin/env python3
"""Write a non-mutating profile addendum for the completed Gate106 audit."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/stage7_gate106/agent_eval.jsonl"
RESULTS = ROOT / "reports/stage7/final_gate106/stage7-frozen-candidate.jsonl"
AUDIT = ROOT / "reports/stage7/final_gate106/audit.json"
OUTPUT = ROOT / "reports/stage7/final_gate106/profile_addendum.json"


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if not AUDIT.is_file():
        raise FileNotFoundError("Main Gate audit must exist first")
    profile_by_id = {
        str(row["extra_info"]["instance_id"]): str(row["extra_info"].get("wildcard_profile", ""))
        for row in rows(DATA)
        if row["extra_info"].get("drift_type") == "add_column"
    }
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows(RESULTS):
        if row["drift_type"] == "add_column":
            groups[profile_by_id[str(row["instance_id"])]].append(row)
    payload = {
        "protocol": "driftsql_stage7_gate106_profile_addendum_v1",
        "main_audit": str(AUDIT.relative_to(ROOT)),
        "changes_acceptance": False,
        "method": "join frozen Gate data and completed results by instance_id",
        "add_by_wildcard_profile": {
            name: {
                "tasks": len(values),
                "success": sum(bool(row["task_success"]) for row in values),
                "success_rate": sum(bool(row["task_success"]) for row in values) / len(values),
                "executable": sum(bool(row["executable"]) for row in values),
                "submitted": sum(row["termination_reason"] == "submitted" for row in values),
                "turn_limit": sum(row["termination_reason"] == "turn_limit" for row in values),
            }
            for name, values in sorted(groups.items())
        },
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
