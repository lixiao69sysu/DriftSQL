#!/usr/bin/env python3
"""Mine immutable human-review candidates from persisted P4 failures."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driftsql.service.catalog import ScenarioCatalog
from driftsql.service.observability import _failure_type
from driftsql.service.repository import SQLiteSessionRepository


ROOT = Path(__file__).resolve().parents[1]


def digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=ROOT / "data/service/driftsql_service.sqlite")
    parser.add_argument("--scenario-path", type=Path, default=ROOT / "data/processed/stage8_fresh_sft/tune_agent_eval.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/p4_replay_review")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")

    repository = SQLiteSessionRepository(args.repository)
    repository.initialize()
    catalog = ScenarioCatalog(args.scenario_path)
    catalog.load()
    scenario_ids = set(catalog.scenario_ids())
    candidates: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for session in repository.list_sessions(limit=10000):
        failure_type = _failure_type(session)
        if failure_type is None:
            continue
        if session.scenario_id not in scenario_ids:
            skipped.append({"session_id": session.session_id, "reason": "not_in_tune_catalog"})
            continue
        trajectory = repository.get_trajectory(session.session_id).model_dump(mode="json")
        tool_events = [event for event in trajectory["events"] if event["event_type"] == "tool"]
        tool_sequence = [str(event["payload"].get("tool", "")) for event in tool_events]
        raw = catalog.raw_record(session.scenario_id)
        extra = raw["extra_info"]
        classification = {
            "max_turns": "turn_limit",
            "max_tool_calls": "tool_budget",
            "submitted": "wrong_submit",
        }.get(session.termination_reason or "", failure_type)
        trajectory_hash = digest(trajectory)
        candidate_id = "p4r_" + hashlib.sha256(
            f"{session.session_id}:{trajectory_hash}".encode()
        ).hexdigest()[:16]
        candidates.append(
            {
                "candidate_id": candidate_id,
                "source": "p4_persisted_real_sft20_failure",
                "source_split": "stage8_tune",
                "session_id": session.session_id,
                "scenario_id": session.scenario_id,
                "db_id": session.db_id,
                "drift_type": session.drift_type,
                "wildcard_profile": extra.get("wildcard_profile"),
                "added_column_count": extra.get("added_column_count"),
                "failure_type": failure_type,
                "failure_class": classification,
                "status": session.status.value,
                "termination_reason": session.termination_reason,
                "success": session.success,
                "model_calls": session.usage.model_calls,
                "tool_calls": session.usage.tool_calls,
                "tool_sequence": tool_sequence,
                "final_sql": session.final_sql,
                "reward": trajectory["reward"],
                "trajectory_sha256": trajectory_hash,
                "trajectory": {
                    "events": trajectory["events"],
                    "messages": trajectory["messages"],
                },
                "review": {
                    "status": "pending",
                    "required_checks": [
                        "failure_is_actionable",
                        "failure_label_is_correct",
                        "no_sensitive_content",
                        "safe_to_match_against_p5_train",
                    ],
                },
            }
        )
    candidates.sort(key=lambda row: row["candidate_id"])
    args.output_dir.mkdir(parents=True)
    write_jsonl(args.output_dir / "candidates.jsonl", candidates)
    (args.output_dir / "reviews.jsonl").touch()
    summary = {
        "protocol": "driftsql_p4_failure_replay_review_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "repository": str(args.repository.resolve()),
        "scenario_source": "Stage-8 Tune only; defines failure strata, never copied as optimization rows",
        "candidates": len(candidates),
        "failure_types": dict(sorted(Counter(row["failure_type"] for row in candidates).items())),
        "failure_classes": dict(sorted(Counter(row["failure_class"] for row in candidates).items())),
        "review_status": {"pending": len(candidates), "approved": 0, "rejected": 0},
        "skipped": skipped,
        "p5_gate_read": False,
        "stage8_gate55_read": False,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
