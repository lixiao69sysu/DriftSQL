#!/usr/bin/env python3
"""Prepare P5 VERL Agentic-GRPO data from isolated Train/Tune and reviewed replay."""

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

from scripts.prepare_stage7_add_column_sft import (  # noqa: E402
    build_add_trajectory,
    load_jsonl,
    load_tool_schemas,
    write_jsonl,
    write_parquet,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "data/processed/p5_isolated_protocol"
DEFAULT_REPLAY = ROOT / "data/processed/p5_reviewed_replay"
DEFAULT_TOOLS = ROOT / "configs/tools/drift_tools.yaml"
DEFAULT_TOKENIZER = ROOT / "models/Qwen2.5-Coder-7B-Instruct"
DEFAULT_OUTPUT = ROOT / "data/processed/p5_grpo"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def annotate_agent(
    agent: dict[str, Any],
    audit: dict[str, Any],
    raw: dict[str, Any],
    *,
    split: str,
    replay_index: int | None,
) -> None:
    replay = raw.get("p5_replay") if isinstance(raw.get("p5_replay"), dict) else None
    p5 = raw["p5"]
    extra = agent["extra_info"]
    source_task_id = str(raw["task_id"])
    candidate_id = str(replay["candidate_id"]) if replay else ""
    if replay:
        extra["instance_id"] = f"{source_task_id}__p5r__{candidate_id}__{replay_index:04d}"
        agent["data_source"] = f"driftsql/p5/reviewed_replay/{replay['failure_class']}"
    else:
        agent["data_source"] = f"driftsql/p5/{split}/add_column/{raw['wildcard_profile']}"
    extra.update(
        {
            "p5_split": split,
            "p5_source_task_id": source_task_id,
            "p5_source_cohort": str(p5["source_cohort"]),
            "p5_turn_limit_focus": bool(p5["turn_limit_focus"]),
            "p5_reviewed_replay": replay is not None,
            "p5_replay_candidate_id": candidate_id,
            "p5_replay_failure_class": str(replay["failure_class"]) if replay else "",
            "p5_replay_reviewer": str(replay["reviewer"]) if replay else "",
            "p5_replay_reviewed_at": str(replay["reviewed_at"]) if replay else "",
        }
    )
    audit.update(
        {
            "p5_split": split,
            "source_task_id": source_task_id,
            "reviewed_replay": replay is not None,
            "replay_candidate_id": candidate_id,
            "failure_class": str(replay["failure_class"]) if replay else "",
        }
    )
    audit["validations"].update(
        {
            "p5_tune_read_for_training": False,
            "p5_gate_read": False,
            "stage8_gate55_read": False,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-dir", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--reviewed-replay-dir", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--tools", type=Path, default=DEFAULT_TOOLS)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-tokens", type=int, default=6144)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")

    protocol_summary = json.loads((args.protocol_dir / "summary.json").read_text(encoding="utf-8"))
    replay_summary = json.loads((args.reviewed_replay_dir / "summary.json").read_text(encoding="utf-8"))
    if replay_summary.get("approved_candidates", 0) <= 0:
        raise RuntimeError("P5 training requires at least one human-approved replay candidate")
    if replay_summary.get("p5_tune_database_overlap") or replay_summary.get("p5_gate_database_overlap"):
        raise RuntimeError("Reviewed replay overlaps P5 Tune or sealed Gate")

    base_train = load_jsonl(args.protocol_dir / "train.jsonl")
    tune = load_jsonl(args.protocol_dir / "tune.jsonl")
    reviewed_replay = load_jsonl(args.reviewed_replay_dir / "train.jsonl")
    train = base_train + reviewed_replay
    train_dbs = {str(row["db_id"]) for row in train}
    tune_dbs = {str(row["db_id"]) for row in tune}
    gate_dbs = set(protocol_summary["splits"]["gate"]["database_ids"])
    if train_dbs & tune_dbs or train_dbs & gate_dbs or tune_dbs & gate_dbs:
        raise RuntimeError("P5 database-isolation invariant failed")
    if any(row.get("p5", {}).get("split") != "train" for row in train):
        raise RuntimeError("Non-Train row entered P5 optimization data")
    if any(row.get("p5", {}).get("split") != "tune" for row in tune):
        raise RuntimeError("Non-Tune row entered P5 selection data")

    schemas = load_tool_schemas(args.tools)
    schemas_json = json.dumps(schemas, ensure_ascii=False)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    outputs: dict[str, list[dict[str, Any]]] = {"train": [], "tune": []}
    audits: dict[str, list[dict[str, Any]]] = {"train": [], "tune": []}
    for split, rows in (("train", train), ("tune", tune)):
        replay_counter = 0
        for row in rows:
            _trajectory, agent, audit = build_add_trajectory(
                row,
                schemas_json=schemas_json,
                tokenizer=tokenizer,
                max_tokens=args.max_tokens,
                stage_name="p5",
            )
            is_replay = isinstance(row.get("p5_replay"), dict)
            annotate_agent(
                agent,
                audit,
                row,
                split=split,
                replay_index=replay_counter if is_replay else None,
            )
            replay_counter += int(is_replay)
            outputs[split].append(agent)
            audits[split].append(audit)

    args.output_dir.mkdir(parents=True)
    write_parquet(args.output_dir / "train.parquet", outputs["train"])
    write_parquet(args.output_dir / "tune.parquet", outputs["tune"])
    write_jsonl(args.output_dir / "tune_agent_eval.jsonl", outputs["tune"])
    write_jsonl(args.output_dir / "train_audit.jsonl", audits["train"])
    write_jsonl(args.output_dir / "tune_audit.jsonl", audits["tune"])
    summary = {
        "protocol": "driftsql_p5_reviewed_replay_grpo_data_v1",
        "policy": "Optimization rows are P5 Train only; P4 failures define human-approved replay strata",
        "source_sha256": {
            "p5_train": sha256(args.protocol_dir / "train.jsonl"),
            "p5_tune": sha256(args.protocol_dir / "tune.jsonl"),
            "reviewed_replay": sha256(args.reviewed_replay_dir / "train.jsonl"),
        },
        "approved_candidates": int(replay_summary["approved_candidates"]),
        "approved_candidate_ids": replay_summary.get("approved_candidate_ids", []),
        "rows": {
            "base_train": len(base_train),
            "reviewed_replay": len(reviewed_replay),
            "train": len(outputs["train"]),
            "tune": len(outputs["tune"]),
        },
        "train_database_ids": sorted(train_dbs),
        "tune_database_ids": sorted(tune_dbs),
        "train_tune_database_overlap": sorted(train_dbs & tune_dbs),
        "train_gate_database_overlap": sorted(train_dbs & gate_dbs),
        "tune_gate_database_overlap": sorted(tune_dbs & gate_dbs),
        "reviewed_failure_classes": dict(
            sorted(Counter(str(row["p5_replay"]["failure_class"]) for row in reviewed_replay).items())
        ),
        "turn_limit_focus_rows": {
            split: sum(bool(row["extra_info"]["p5_turn_limit_focus"]) for row in rows)
            for split, rows in outputs.items()
        },
        "p5_gate_read": False,
        "stage8_gate55_read": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

