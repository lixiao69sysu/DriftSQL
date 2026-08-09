#!/usr/bin/env python3
"""Export the scale-up corpus to P6 without copying SQLite databases.

The scale-up factory has already executed the original, stale-active, and
repaired-active queries before accepting a task.  This adapter consumes that
immutable audit and reconstructs the active *schema metadata* from SQLite
PRAGMA calls plus the audited operations.  It therefore avoids the old
per-task multi-GB database materialization path.

Fresh Blind is excluded by default.  It can only be packaged with the explicit
``--include-sealed-blind`` gate used after Tune model selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from sqlglot import exp, parse_one
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_p6_generalized_protocol import (
    SUPERVISED_SPLITS,
    TOOL_NAMES,
    build_trajectory,
    expand_supervision,
    load_jsonl,
    load_tool_schemas,
    write_jsonl,
    write_parquet,
)


DEFAULT_INPUT = ROOT / "data/processed/p6_scaleup_v1_final_raw"
DEFAULT_OUTPUT = ROOT / "data/processed/p6_scaleup_v1_low_write_protocol"
DEFAULT_MODEL = ROOT / "models/Qwen2.5-Coder-7B-Instruct"
DEFAULT_TOOLS = ROOT / "configs/tools/drift_tools.yaml"


def quote(identifier: str) -> str:
    return f'"{identifier.replace(chr(34), chr(34) * 2)}"'


def source_schema(database: Path) -> dict[str, list[tuple[str, str]]]:
    uri = database.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        names = [
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        result: dict[str, list[tuple[str, str]]] = {}
        for table in names:
            escaped = table.replace('"', '""')
            result[table] = [
                (str(row[1]), str(row[2]).strip() or "TEXT")
                for row in connection.execute(f'PRAGMA table_info("{escaped}")')
            ]
    return result


def find_name(values: list[str] | set[str], target: str) -> str:
    lookup = {value.casefold(): value for value in values}
    if target.casefold() not in lookup:
        raise KeyError(f"Schema identifier is unavailable: {target}")
    return lookup[target.casefold()]


def apply_schema_diff(
    schema: dict[str, list[tuple[str, str]]], schema_diff: dict[str, Any]
) -> dict[str, list[tuple[str, str]]]:
    schema = {table: list(columns) for table, columns in schema.items()}
    for operation in schema_diff.get("operations", []) or []:
        kind = str(operation.get("type", ""))
        if kind == "metric_definition_change":
            continue
        if kind == "rename_table":
            old = find_name(set(schema), str(operation["old_name"]))
            new = str(operation["new_name"])
            schema[new] = schema.pop(old)
            continue
        table = find_name(set(schema), str(operation["table"]))
        if kind == "add_column":
            name = str(operation["new_name"])
            if name.casefold() not in {column.casefold() for column, _ in schema[table]}:
                schema[table].append((name, str(operation.get("declared_type", "TEXT"))))
            continue
        if kind in {"rename_column", "replace_column"}:
            old = find_name([column for column, _ in schema[table]], str(operation["old_name"]))
            new = str(operation["new_name"])
            declared = str(operation.get("declared_type", "")).strip()
            schema[table] = [
                (new, declared or column_type) if column == old else (column, column_type)
                for column, column_type in schema[table]
            ]
            continue
        raise NotImplementedError(f"Unsupported schema-only operation: {kind}")
    return schema


def referenced_tables(sql: str) -> set[str]:
    tree = parse_one(sql, read="sqlite")
    return {table.name.casefold() for table in tree.find_all(exp.Table) if table.name}


def active_schema_ddl(row: dict[str, Any]) -> str:
    schema = apply_schema_diff(
        source_schema(Path(str(row["source_db"]))), dict(row["schema_diff"])
    )
    references = referenced_tables(str(row["repaired_sql"]))
    selected = [table for table in schema if table.casefold() in references]
    if not selected:
        raise RuntimeError(f"No active table found for {row['task_id']}")
    statements = []
    for table in selected:
        columns = ", ".join(
            f"{quote(name)} {column_type}" for name, column_type in schema[table]
        )
        statements.append(f"CREATE TABLE {quote(table)} ({columns});")
    return "\n".join(statements)


def trusted_verification(row: dict[str, Any], schema: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if not row.get("oracle_steps") or not row.get("result_fingerprint"):
        raise RuntimeError(f"Missing immutable execution audit: {row['task_id']}")
    scaleup = row.get("scaleup") or {}
    if not bool(row.get("legacy_task")) and not bool(
        scaleup.get("execution_verified_at_generation")
    ):
        raise RuntimeError(f"Synthetic row lacks generation verification: {row['task_id']}")
    verification = {
        "validations": {
            "repaired_execution_success": True,
            "submitted": True,
            "fingerprint_matches": True,
            "stale_behavior_verified": True,
            "rolled_back": True,
        }
    }
    agent = {
        "extra_info": {
            "tools_kwargs": {
                "execute_sql": {"create_kwargs": {"schema": schema}}
            }
        }
    }
    return verification, agent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tools", type=Path, default=DEFAULT_TOOLS)
    parser.add_argument("--max-tokens", type=int, default=6144)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--limit-per-split", type=int, default=0)
    parser.add_argument("--include-sealed-blind", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")

    splits = ("train", "dev", "test") if args.include_sealed_blind else ("train", "dev")
    schemas = load_tool_schemas(args.tools)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    manifests_by_split: dict[str, list[dict[str, Any]]] = {}
    agents_by_split: dict[str, list[dict[str, Any]]] = {}
    trajectories_by_split: dict[str, list[dict[str, Any]]] = {}
    contracts_by_split: dict[str, list[dict[str, Any]]] = {}
    examples_by_split: dict[str, list[dict[str, Any]]] = {}
    stats: dict[str, Any] = {}

    for split_index, split in enumerate(splits):
        rows = load_jsonl(args.input_dir / f"{split}.jsonl")
        if args.limit_per_split > 0:
            rows = rows[: args.limit_per_split]
        manifests: list[dict[str, Any]] = []
        agents: list[dict[str, Any]] = []
        trajectories: list[dict[str, Any]] = []
        contracts: list[dict[str, Any]] = []
        examples: list[dict[str, Any]] = []
        for index, row in enumerate(rows, 1):
            schema = active_schema_ddl(row)
            verification, prior_agent = trusted_verification(row, schema)
            trajectory, agent, manifest, contract = build_trajectory(
                row,
                verification,
                prior_agent,
                schemas=schemas,
                temporary_root=args.output_dir / ".unused-live-replay",
                contract_root=args.output_dir / ".unused-contract-replay",
                live_replay=False,
            )
            trajectory["validation_mode"] = "generation_factory_execution_audit+schema_metadata"
            manifest["validation_mode"] = "generation_factory_execution_audit+schema_metadata"
            trajectories.append(trajectory)
            agents.append(agent)
            manifests.append(manifest)
            contracts.append(contract)
            if split in SUPERVISED_SPLITS:
                examples.extend(
                    expand_supervision(
                        trajectory,
                        schemas=schemas,
                        tokenizer=tokenizer,
                        max_tokens=args.max_tokens,
                        train=split == "train",
                    )
                )
            if index % 100 == 0 or index == len(rows):
                print(f"{split}: adapted {index}/{len(rows)}", flush=True)
        random.Random(args.seed + split_index).shuffle(examples)
        manifests_by_split[split] = manifests
        agents_by_split[split] = agents
        trajectories_by_split[split] = trajectories
        contracts_by_split[split] = contracts
        examples_by_split[split] = examples
        stats[split] = {
            "tasks": len(rows),
            "databases": len({row["db_id"] for row in manifests}),
            "supervision_examples": len(examples),
            "contract_validated": sum(bool(row["accepted"]) for row in contracts),
            "profiles": dict(sorted(Counter(row["interaction_profile"] for row in manifests).items())),
            "drift_types": dict(sorted(Counter(row["drift_type"] for row in manifests).items())),
            "target_actions": dict(sorted(Counter(row["target_action"] for row in examples).items())),
        }

    databases = {
        split: {row["db_id"] for row in manifests_by_split[split]} for split in splits
    }
    overlap = sorted(databases["train"] & databases["dev"])
    if overlap:
        raise RuntimeError(f"Train/Tune database leakage: {overlap}")
    task_ids = [
        row["task_id"] for split in splits for row in manifests_by_split[split]
    ]
    if len(task_ids) != len(set(task_ids)):
        raise RuntimeError("Cross-split task ID leakage")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    for split in splits:
        write_parquet(args.output_dir / f"rl_{split}.parquet", agents_by_split[split])
        write_jsonl(args.output_dir / f"{split}_agent_eval.jsonl", agents_by_split[split])
        write_jsonl(args.output_dir / f"{split}_manifest.jsonl", manifests_by_split[split])
        write_jsonl(args.output_dir / f"{split}_contract_audit.jsonl", contracts_by_split[split])
        write_parquet(
            args.output_dir / f"{split}_trajectories.parquet",
            trajectories_by_split[split],
        )
        if split in SUPERVISED_SPLITS:
            write_parquet(args.output_dir / f"{split}.parquet", examples_by_split[split])

    token_counts = [
        int(row["token_count"])
        for split in splits
        if split in SUPERVISED_SPLITS
        for row in examples_by_split[split]
    ]
    summary = {
        "protocol": "driftsql_p6_scaleup_low_write_v1",
        "validation_mode": "immutable factory execution audit + schema-only metadata reconstruction",
        "database_copies": 0,
        "splits": stats,
        "train_tune_database_overlap": overlap,
        "fresh_blind": {
            "packaged": bool(args.include_sealed_blind),
            "read_for_model_selection": False,
            "policy": "sealed until Tune432 model selection completes",
        },
        "token_length": {
            "min": min(token_counts),
            "median": statistics.median(token_counts),
            "max": max(token_counts),
            "budget": args.max_tokens,
        },
        "source_sha256": {
            split: sha256(args.input_dir / f"{split}.jsonl") for split in splits
        },
        "tool_names": list(TOOL_NAMES),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
