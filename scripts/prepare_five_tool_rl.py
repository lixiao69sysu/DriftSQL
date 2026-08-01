#!/usr/bin/env python3
"""Build GRPO rows from the already execution-verified five-tool SFT corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from verl.utils.py_functional import convert_nested_value_to_list_recursive

from driftsql.data.tool_sft import clarification_spec


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data/processed/five_tool_sft_native_v2"
DEFAULT_MANIFEST = PROJECT_ROOT / "data/generated/schema_drift/train.jsonl"
TOOL_NAMES = ("get_schema", "ask_user", "get_knowledge_definition", "execute_sql", "submit_solution")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary)
    temporary.replace(path)


def schema_from_messages(messages: list[dict[str, Any]]) -> str:
    for index, message in enumerate(messages[:-1]):
        calls = message.get("tool_calls", []) or []
        if message.get("role") != "assistant" or not calls:
            continue
        if calls[0].get("function", {}).get("name") != "get_schema":
            continue
        observation = messages[index + 1]
        if observation.get("role") != "tool":
            break
        payload = json.loads(str(observation.get("content", "{}")))
        schema = str(payload.get("schema", ""))
        if schema:
            return schema
    raise ValueError("Verified trajectory does not contain a get_schema observation")


def build_record(raw: dict[str, Any], messages: list[dict[str, Any]]) -> dict[str, Any]:
    operation = dict(raw["schema_diff"]["operations"][0])
    spec = clarification_spec(raw)
    state = {
        "db_id": str(raw["db_id"]),
        "db_version": "v2",
        "metric_version": "task-governed-v1",
        "source_db": str(Path(raw["source_db"]).resolve()),
        "schema_diff": raw["schema_diff"],
        "query": str(raw["question"]),
        "stale_sql": str(raw["stale_sql"]),
        "ground_truth": str(raw["repaired_sql"]),
        "result_fingerprint": raw["result_fingerprint"],
        "user_query_ambiguity": {
            "critical_ambiguity": [
                {
                    "term": spec["term"],
                    "sql_snippet": spec["definition"],
                    "type": spec["ambiguity_type"],
                }
            ],
            "non_critical_ambiguity": [],
        },
        "knowledge_entries": [spec["knowledge_entry"]],
        "schema": schema_from_messages(messages),
    }
    return {
        "data_source": f"driftsql/five_tool/{operation['type']}",
        "prompt": [
            {"role": str(message["role"]), "content": str(message["content"])}
            for message in messages[:2]
        ],
        "ability": "interactive_sql_drift_recovery",
        "reward_model": {"ground_truth": str(raw["repaired_sql"])},
        "extra_info": {
            "instance_id": str(raw["task_id"]),
            "db_id": str(raw["db_id"]),
            "source_db": str(Path(raw["source_db"]).resolve()),
            "schema_diff": raw["schema_diff"],
            "result_fingerprint": raw["result_fingerprint"],
            "stale_sql": str(raw["stale_sql"]),
            "need_tools_kwargs": True,
            "tools_kwargs": {name: {"create_kwargs": dict(state)} for name in TOOL_NAMES},
            "tool_selection": list(TOOL_NAMES),
        },
        "return_raw_chat": True,
        "agent_name": "driftsql_tool_agent",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    source_summary = json.loads((args.input_dir / "summary.json").read_text(encoding="utf-8"))
    if source_summary.get("validation") != "real versioned SQLite + native VERL tools + result fingerprint":
        raise RuntimeError("Input corpus is not marked execution-verified")
    raw_by_id = {str(row["task_id"]): row for row in load_jsonl(args.manifest)}
    split_rows: dict[str, list[dict[str, Any]]] = {}
    split_databases: dict[str, set[str]] = {}
    for split in ("train", "val"):
        frame = pd.read_parquet(args.input_dir / f"{split}.parquet")
        verified = load_jsonl(args.input_dir / f"{split}_manifest.jsonl")
        if len(frame) != len(verified):
            raise RuntimeError(f"{split}: SFT rows and verified manifests differ")
        rows: list[dict[str, Any]] = []
        for (_, sft_row), item in zip(frame.iterrows(), verified, strict=True):
            task_id = str(item["task_id"])
            if task_id not in raw_by_id:
                raise KeyError(task_id)
            messages = convert_nested_value_to_list_recursive(sft_row["messages"])
            rows.append(build_record(raw_by_id[task_id], messages))
        write_parquet(args.input_dir / f"rl_{split}.parquet", rows)
        split_rows[split] = rows
        split_databases[split] = {str(row["extra_info"]["db_id"]) for row in rows}

    overlap = sorted(split_databases["train"] & split_databases["val"])
    if overlap:
        raise RuntimeError(f"database leakage: {overlap}")
    summary = {
        "name": "driftsql_execution_verified_five_tool_grpo_v1",
        "derived_from": str(args.input_dir.resolve()),
        "validation": source_summary["validation"],
        "splits": {
            split: {"rows": len(split_rows[split]), "databases": len(split_databases[split])}
            for split in ("train", "val")
        },
        "database_overlap": overlap,
        "tools": list(TOOL_NAMES),
    }
    (args.input_dir / "rl_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
