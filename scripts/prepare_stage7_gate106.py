#!/usr/bin/env python3
"""Materialize the one-shot Stage 7 Gate106 agent records after model freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from prepare_stage7_add_column_sft import (
    build_add_trajectory,
    load_jsonl,
    load_tool_schemas,
    write_jsonl,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FREEZE = PROJECT_ROOT / "reports/stage7/final_candidate/frozen_candidate.json"
DEFAULT_PROTOCOL = PROJECT_ROOT / "data/processed/stage7_add_column_protocol"
DEFAULT_STAGE6_AGENT = PROJECT_ROOT / "data/processed/stage6_ablation/b1/train_agent_eval.jsonl"
DEFAULT_TOOLS = PROJECT_ROOT / "configs/tools/drift_tools.yaml"
DEFAULT_TOKENIZER = PROJECT_ROOT / "models/Qwen2.5-Coder-7B-Instruct"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/processed/stage7_gate106"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--protocol-dir", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--stage6-agent", type=Path, default=DEFAULT_STAGE6_AGENT)
    parser.add_argument("--tools", type=Path, default=DEFAULT_TOOLS)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-tokens", type=int, default=6144)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite one-shot Gate assets: {args.output_dir}")

    frozen = json.loads(args.freeze.read_text(encoding="utf-8"))
    expected = {
        **frozen["locked_files_sha256"],
        **frozen["candidate"]["adapter_files_sha256"],
    }
    mismatches = {
        relative: {"expected": digest, "actual": sha256(PROJECT_ROOT / relative)}
        for relative, digest in expected.items()
        if sha256(PROJECT_ROOT / relative) != digest
    }
    if mismatches:
        raise RuntimeError(f"Frozen Stage 7 candidate/protocol changed: {mismatches}")

    schemas_json = json.dumps(load_tool_schemas(args.tools), ensure_ascii=False)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    add_rows = load_jsonl(args.protocol_dir / "gate_add_column.jsonl")
    general_manifest = load_jsonl(args.protocol_dir / "gate_general_replay.jsonl")
    stage6_rows = load_jsonl(args.stage6_agent)
    stage6_by_id = {
        str(row["extra_info"]["instance_id"]): row for row in stage6_rows
    }

    add_agent = []
    for index, row in enumerate(add_rows, 1):
        _, agent, _ = build_add_trajectory(
            row,
            schemas_json=schemas_json,
            tokenizer=tokenizer,
            max_tokens=args.max_tokens,
        )
        add_agent.append(agent)
        if index % 6 == 0:
            print(f"prepared Gate add-column {index}/{len(add_rows)}", flush=True)

    general_ids = [str(row["task_id"]) for row in general_manifest]
    missing = [task_id for task_id in general_ids if task_id not in stage6_by_id]
    if missing:
        raise RuntimeError(f"Missing Stage 6 Train parent rows for Gate: {missing[:5]}")
    general_agent = [stage6_by_id[task_id] for task_id in general_ids]
    records = add_agent + general_agent
    ids = [str(row["extra_info"]["instance_id"]) for row in records]
    if len(add_agent) != 24 or len(general_agent) != 82 or len(records) != 106:
        raise RuntimeError(
            f"Gate106 count mismatch: add={len(add_agent)} general={len(general_agent)}"
        )
    if len(set(ids)) != len(ids):
        raise RuntimeError("Gate106 contains duplicate instance IDs")

    protocol_summary = json.loads(
        (args.protocol_dir / "summary.json").read_text(encoding="utf-8")
    )
    expected_dbs = set(protocol_summary["splits"]["gate"]["database_ids"])
    actual_dbs = {str(row["extra_info"]["db_id"]) for row in records}
    if actual_dbs != expected_dbs:
        raise RuntimeError(
            f"Gate database mismatch: expected={sorted(expected_dbs)} actual={sorted(actual_dbs)}"
        )

    args.output_dir.mkdir(parents=True)
    rows_path = args.output_dir / "agent_eval.jsonl"
    write_jsonl(rows_path, records)
    summary = {
        "protocol": "driftsql_stage7_one_shot_gate106_data_v1",
        "candidate_freeze": str(args.freeze.relative_to(PROJECT_ROOT)),
        "candidate_freeze_sha256": sha256(args.freeze),
        "tasks": len(records),
        "add_column_tasks": len(add_agent),
        "general_tasks": len(general_agent),
        "database_ids": sorted(actual_dbs),
        "rows_sha256": sha256(rows_path),
        "training_assets_created": False,
        "allowed_candidate_runs": 1,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
