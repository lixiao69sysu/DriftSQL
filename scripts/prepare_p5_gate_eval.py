#!/usr/bin/env python3
"""Open the sealed P5 Gate exactly once, only after candidate freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def verify_freeze(path: Path) -> dict[str, Any]:
    freeze = json.loads(path.read_text(encoding="utf-8"))
    if freeze.get("protocol") != "driftsql_p5_tune_frozen_candidate_v1":
        raise RuntimeError("P5 candidate is not frozen")
    for relative, expected in freeze["candidate"]["adapter_files_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"Frozen adapter changed: {relative}")
    for relative, expected in freeze["locked_files_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"Frozen P5 input changed: {relative}")
    return freeze


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, default=ROOT / "reports/p5/final_candidate/frozen_candidate.json")
    parser.add_argument("--protocol-dir", type=Path, default=ROOT / "data/processed/p5_isolated_protocol")
    parser.add_argument("--tools", type=Path, default=ROOT / "configs/tools/drift_tools.yaml")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "models/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/p5_gate_eval")
    parser.add_argument("--max-tokens", type=int, default=6144)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    freeze = verify_freeze(args.freeze)

    lifecycle = args.freeze.parent / "gate_lifecycle.jsonl"
    args.freeze.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lifecycle, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"P5 Gate has already been opened or attempted: {lifecycle}") from error
    opened = {
        "event": "gate_open_started",
        "at": datetime.now(UTC).isoformat(),
        "freeze_sha256": sha256(args.freeze),
    }
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(opened, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    gate_path = args.protocol_dir / "sealed_gate.jsonl"
    if sha256(gate_path) != freeze["one_shot_gate"]["sealed_input_sha256"]:
        raise RuntimeError("Sealed P5 Gate hash changed after candidate freeze")
    gate_rows = load_jsonl(gate_path)
    if len(gate_rows) != 18 or len({str(row["task_id"]) for row in gate_rows}) != 18:
        raise RuntimeError("P5 Gate identity/cardinality failed")

    # Import training-only dependencies only after the one-shot Gate has been
    # opened, hash-checked, and parsed. This keeps fail-closed lifecycle checks
    # runnable in lightweight CI and avoids doing expensive model imports
    # before the irreversible Gate transition.
    from transformers import AutoTokenizer

    from scripts.prepare_stage7_add_column_sft import (
        build_add_trajectory,
        load_tool_schemas,
        write_jsonl,
    )

    schemas_json = json.dumps(load_tool_schemas(args.tools), ensure_ascii=False)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    agents: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for row in gate_rows:
        _trajectory, agent, audit = build_add_trajectory(
            row,
            schemas_json=schemas_json,
            tokenizer=tokenizer,
            max_tokens=args.max_tokens,
            stage_name="p5",
        )
        extra = agent["extra_info"]
        extra.update(
            {
                "p5_split": "gate",
                "p5_source_task_id": str(row["task_id"]),
                "p5_source_cohort": str(row["p5"]["source_cohort"]),
                "p5_turn_limit_focus": bool(row["p5"]["turn_limit_focus"]),
                "p5_reviewed_replay": False,
                "p5_replay_candidate_id": "",
                "p5_replay_failure_class": "",
                "p5_replay_reviewer": "",
                "p5_replay_reviewed_at": "",
            }
        )
        agent["data_source"] = f"driftsql/p5/gate/add_column/{row['wildcard_profile']}"
        audit["validations"].update(
            {
                "p5_gate_read_after_freeze": True,
                "stage8_gate55_read": False,
            }
        )
        agents.append(agent)
        audits.append(audit)
    args.output_dir.mkdir(parents=True)
    eval_path = args.output_dir / "gate_agent_eval.jsonl"
    audit_path = args.output_dir / "gate_audit.jsonl"
    write_jsonl(eval_path, agents)
    write_jsonl(audit_path, audits)
    summary = {
        "protocol": "driftsql_p5_one_shot_gate_eval_input_v1",
        "candidate_freeze_sha256": sha256(args.freeze),
        "candidate": freeze["candidate"]["name"],
        "rows": len(agents),
        "databases": len({str(row["extra_info"]["db_id"]) for row in agents}),
        "turn_limit_focus_rows": sum(bool(row["extra_info"]["p5_turn_limit_focus"]) for row in agents),
        "wildcard_profiles": dict(
            sorted(
                Counter(
                    str(row["extra_info"]["wildcard_profile"]) for row in agents
                ).items()
            )
        ),
        "eval_jsonl_sha256": sha256(eval_path),
        "audit_jsonl_sha256": sha256(audit_path),
        "training_outputs_written": False,
        "gate_opened_after_candidate_freeze": True,
        "gate55_read": False,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with lifecycle.open("a", encoding="utf-8") as handle:
        lifecycle_event = {
            "event": "gate_input_prepared",
            "at": datetime.now(UTC).isoformat(),
            "summary_sha256": sha256(summary_path),
        }
        handle.write(json.dumps(lifecycle_event, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
