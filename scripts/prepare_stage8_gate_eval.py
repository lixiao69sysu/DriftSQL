#!/usr/bin/env python3
"""Materialize the one-shot Stage 8 Gate55 evaluator input after freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_stage7_add_column_sft import (
    build_add_trajectory,
    load_jsonl,
    load_tool_schemas,
    write_jsonl,
)
from scripts.prepare_stage8_fresh_sft import build_general_trajectory


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FREEZE = ROOT / "reports/stage8/final_candidate/frozen_candidate.json"
DEFAULT_PROTOCOL = ROOT / "data/processed/stage8_fresh_protocol"
DEFAULT_TOOLS = ROOT / "configs/tools/drift_tools.yaml"
DEFAULT_TOKENIZER = ROOT / "models/Qwen2.5-Coder-7B-Instruct"
DEFAULT_OUTPUT = ROOT / "data/processed/stage8_gate_eval"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_freeze(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != "driftsql_stage8_frozen_candidate_v1":
        raise RuntimeError("Stage 8 candidate is not frozen")
    for relative, expected in payload["candidate"]["adapter_files_sha256"].items():
        actual = sha256(ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"Frozen adapter changed: {relative}")
    for relative, expected in payload["locked_files_sha256"].items():
        actual = sha256(ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"Frozen evaluation input changed: {relative}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--protocol-dir", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--tools", type=Path, default=DEFAULT_TOOLS)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-tokens", type=int, default=6144)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    freeze = verify_freeze(args.freeze)

    # Gate rows are deliberately opened only after every frozen hash verifies.
    add_rows = load_jsonl(args.protocol_dir / "gate_add_column.jsonl")
    general_rows = load_jsonl(args.protocol_dir / "gate_general_replay.jsonl")
    if len(add_rows) != 30 or len(general_rows) != 25:
        raise RuntimeError("Stage 8 Gate cardinality changed")

    schemas = load_tool_schemas(args.tools)
    schemas_json = json.dumps(schemas, ensure_ascii=False)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    agents: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for row in add_rows:
        _, agent, audit = build_add_trajectory(
            row,
            schemas_json=schemas_json,
            tokenizer=tokenizer,
            max_tokens=args.max_tokens,
            stage_name="stage8",
        )
        audit["validations"]["stage8_gate_read_after_freeze"] = True
        agents.append(agent)
        audits.append(audit | {"family": "add_column"})
    for row in general_rows:
        _, agent, audit = build_general_trajectory(
            row,
            schemas=schemas,
            schemas_json=schemas_json,
            tokenizer=tokenizer,
            max_tokens=args.max_tokens,
        )
        audit["validations"]["stage8_gate_read_after_freeze"] = True
        agents.append(agent)
        audits.append(audit | {"family": "general_replay"})

    ids = [str(row["extra_info"]["instance_id"]) for row in agents]
    db_ids = {str(row["extra_info"]["db_id"]) for row in agents}
    if len(ids) != len(set(ids)) or len(db_ids) != 5:
        raise RuntimeError("Stage 8 Gate identity invariant failed")
    args.output_dir.mkdir(parents=True)
    eval_path = args.output_dir / "gate55_agent_eval.jsonl"
    audit_path = args.output_dir / "gate55_audit.jsonl"
    write_jsonl(eval_path, agents)
    write_jsonl(audit_path, audits)
    summary = {
        "protocol": "driftsql_stage8_gate55_eval_only_v1",
        "candidate_freeze_sha256": sha256(args.freeze),
        "candidate": freeze["candidate"]["name"],
        "rows": len(agents),
        "databases": len(db_ids),
        "families": dict(sorted(Counter(row["family"] for row in audits).items())),
        "drift_types": dict(
            sorted(Counter(row["extra_info"]["drift_type"] for row in agents).items())
        ),
        "eval_jsonl_sha256": sha256(eval_path),
        "audit_jsonl_sha256": sha256(audit_path),
        "training_outputs_written": False,
        "gate_opened_after_candidate_freeze": True,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
