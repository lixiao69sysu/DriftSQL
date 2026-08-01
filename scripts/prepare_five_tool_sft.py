#!/usr/bin/env python3
"""Build five-tool SFT trajectories by replaying real versioned databases."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from verl.tools.tool_registry import load_all_tools

from driftsql.data.tool_sft import build_five_tool_messages, clarification_spec
from driftsql.data.trajectory import relevant_schema_ddl
from driftsql.drift import fingerprint_query


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data/generated/schema_drift/train.jsonl"
DEFAULT_SPLIT = PROJECT_ROOT / "data/processed/schema_drift/summary.json"
DEFAULT_TOOLS = PROJECT_ROOT / "configs/tools/drift_tools.yaml"
DEFAULT_TOKENIZER = PROJECT_ROOT / "models/Qwen2.5-Coder-3B-Instruct"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/processed/five_tool_sft"
TOOL_NAMES = ("get_schema", "ask_user", "get_knowledge_definition", "execute_sql", "submit_solution")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No accepted rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary)
    temporary.replace(path)


async def build(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_jsonl(args.manifest)
    if args.limit > 0:
        rows = rows[: args.limit]
    split_summary = json.loads(args.split_summary.read_text(encoding="utf-8"))
    val_databases = set(split_summary["splits"]["val"]["databases"])
    loaded_tools = load_all_tools(tool_config_path=str(args.tools), function_tool_path=None)
    tools = {tool.name: tool for tool in loaded_tools if tool.name in TOOL_NAMES}
    if set(tools) != set(TOOL_NAMES):
        raise RuntimeError(f"Missing tools: {sorted(set(TOOL_NAMES) - set(tools))}")
    tools["get_schema"].config["max_chars"] = args.max_schema_chars
    tools["get_knowledge_definition"].config["max_results"] = 1
    tools["execute_sql"].config["max_rows"] = args.max_result_rows
    schemas = [tools[name].tool_schema.model_dump(mode="json") for name in TOOL_NAMES]
    schemas_json = json.dumps(schemas, ensure_ascii=False)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)

    records: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    rl_records: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    manifests: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    eval_records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    token_lengths: list[int] = []
    tool_counts: Counter[str] = Counter()

    for index, row in enumerate(rows):
        instance_id = f"five-tool-{row['task_id']}"
        spec = clarification_spec(row)
        operation = dict(row["schema_diff"]["operations"][0])
        search_terms = " ".join(
            str(operation.get(key, ""))
            for key in ("table", "old_name", "new_name")
            if operation.get(key)
        )
        base_state = {
            "db_id": str(row["db_id"]),
            "db_version": "v2",
            "metric_version": "task-governed-v1",
            "source_db": str(Path(row["source_db"]).resolve()),
            "schema_diff": row["schema_diff"],
            "query": str(row["question"]),
            "stale_sql": str(row["stale_sql"]),
            "ground_truth": str(row["repaired_sql"]),
            "result_fingerprint": row["result_fingerprint"],
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
        }
        created: set[str] = set()
        try:
            await tools["execute_sql"].create(instance_id=instance_id, create_kwargs=base_state)
            created.add("execute_sql")
            active_db = Path(tools["execute_sql"]._state(instance_id)["db_path"])
            active_schema = relevant_schema_ddl(active_db, str(row["repaired_sql"]))
            state = base_state | {"schema": active_schema}
            for name in TOOL_NAMES:
                if name != "execute_sql":
                    await tools[name].create(instance_id=instance_id, create_kwargs=state)
                    created.add(name)

            async def call(name: str, parameters: dict[str, Any]) -> tuple[str, dict[str, Any]]:
                response, _, metrics = await tools[name].execute(instance_id, parameters)
                tool_counts[name] += 1
                return str(response.text), dict(metrics)

            schema_text, schema_metrics = await call("get_schema", {"query": search_terms})
            ask_text, ask_metrics = await call("ask_user", {"question": spec["question"]})
            knowledge_text, knowledge_metrics = await call(
                "get_knowledge_definition", {"name": spec["term"]}
            )
            stale_text, stale_metrics = await call("execute_sql", {"sql": str(row["stale_sql"])})
            repaired_text, repaired_metrics = await call(
                "execute_sql", {"sql": str(row["repaired_sql"])}
            )
            submit_text, submit_metrics = await call(
                "submit_solution", {"sql": str(row["repaired_sql"])}
            )

            repaired_fingerprint = await asyncio.to_thread(
                fingerprint_query, active_db, str(row["repaired_sql"])
            )
            expected = row["result_fingerprint"]
            fingerprint_matches = (
                repaired_fingerprint.row_count == int(expected["row_count"])
                and repaired_fingerprint.value_hash == str(expected["value_hash"])
            )
            silent_mismatch_verified = True
            if str(row["stale_error"]) == "silent_result_schema_mismatch":
                stale_fingerprint = await asyncio.to_thread(
                    fingerprint_query, active_db, str(row["stale_sql"])
                )
                silent_mismatch_verified = stale_fingerprint != repaired_fingerprint

            validations = {
                "schema_retrieved": bool(schema_metrics.get("schema_retrieved")),
                "clarification_matched": bool(ask_metrics.get("clarification_matched")),
                "knowledge_retrieved": bool(knowledge_metrics.get("knowledge_retrieved")),
                "repaired_execution_success": bool(repaired_metrics.get("execution_success")),
                "session_isolated": bool(repaired_metrics.get("session_isolated")),
                "rolled_back": bool(repaired_metrics.get("rolled_back")),
                "submitted": bool(submit_metrics.get("submitted")),
                "fingerprint_matches": fingerprint_matches,
                "silent_mismatch_verified": silent_mismatch_verified,
            }
            if not all(validations.values()):
                raise RuntimeError(f"validation failed: {validations}")
            if str(row["stale_error"]) != "silent_result_schema_mismatch" and stale_metrics.get(
                "execution_success"
            ):
                raise RuntimeError("stale SQL unexpectedly executed successfully")

            steps = [
                {"action": "get_schema", "arguments": {"query": search_terms}, "observation": schema_text},
                {"action": "ask_user", "arguments": {"question": spec["question"]}, "observation": ask_text},
                {
                    "action": "get_knowledge_definition",
                    "arguments": {"name": spec["term"]},
                    "observation": knowledge_text,
                },
                {
                    "action": "execute_sql",
                    "thought_key": "execute_sql_stale",
                    "arguments": {"sql": str(row["stale_sql"])},
                    "observation": stale_text,
                },
                {
                    "action": "execute_sql",
                    "thought_key": "execute_sql_repaired",
                    "arguments": {"sql": str(row["repaired_sql"])},
                    "observation": repaired_text,
                },
                {
                    "action": "submit_solution",
                    "arguments": {"sql": str(row["repaired_sql"])},
                    "observation": submit_text,
                },
            ]
            messages = build_five_tool_messages(
                question=str(row["question"]), stale_sql=str(row["stale_sql"]), steps=steps
            )
            token_ids = tokenizer.apply_chat_template(
                messages,
                tools=schemas,
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            token_count = len(token_ids)
            if token_count > args.max_tokens:
                raise RuntimeError(f"token budget: {token_count} > {args.max_tokens}")
            split = "val" if str(row["db_id"]) in val_databases else "train"
            records[split].append(
                {"messages": messages, "tools": schemas_json, "enable_thinking": False}
            )
            manifests[split].append(
                {
                    "task_id": str(row["task_id"]),
                    "db_id": str(row["db_id"]),
                    "source": str(row["source"]),
                    "drift_type": str(operation["type"]),
                    "tool_sequence": [step["action"] for step in steps],
                    "token_count": token_count,
                    "clarification": spec,
                    "validations": validations,
                }
            )
            tools_kwargs = {name: {"create_kwargs": dict(state)} for name in TOOL_NAMES}
            rl_record = {
                "data_source": f"driftsql/five_tool/{operation['type']}",
                "prompt": messages[:2],
                "ability": "interactive_sql_drift_recovery",
                "reward_model": {"ground_truth": str(row["repaired_sql"])},
                "extra_info": {
                    "instance_id": str(row["task_id"]),
                    "db_id": str(row["db_id"]),
                    "source_db": str(Path(row["source_db"]).resolve()),
                    "schema_diff": row["schema_diff"],
                    "result_fingerprint": row["result_fingerprint"],
                    "stale_sql": str(row["stale_sql"]),
                    "need_tools_kwargs": True,
                    "tools_kwargs": tools_kwargs,
                    "tool_selection": list(TOOL_NAMES),
                },
                "return_raw_chat": True,
                "agent_name": "driftsql_tool_agent",
            }
            rl_records[split].append(rl_record)
            if split == "val":
                eval_records.append(rl_record)
            token_lengths.append(token_count)
        except Exception as error:
            rejected.append(
                {
                    "task_id": str(row.get("task_id", index)),
                    "db_id": str(row.get("db_id", "")),
                    "error": str(error),
                }
            )
        finally:
            for name in created:
                try:
                    await tools[name].release(instance_id)
                except KeyError:
                    pass
        if (index + 1) % 25 == 0 or index + 1 == len(rows):
            print(
                f"validated {index + 1}/{len(rows)}; accepted="
                f"{len(records['train']) + len(records['val'])}; rejected={len(rejected)}",
                flush=True,
            )

    train_dbs = {row["db_id"] for row in manifests["train"]}
    val_dbs = {row["db_id"] for row in manifests["val"]}
    overlap = sorted(train_dbs & val_dbs)
    if overlap:
        raise RuntimeError(f"database leakage: {overlap}")
    for split in ("train", "val"):
        write_parquet(args.output_dir / f"{split}.parquet", records[split])
        write_parquet(args.output_dir / f"rl_{split}.parquet", rl_records[split])
        write_jsonl(args.output_dir / f"{split}_manifest.jsonl", manifests[split])
    write_jsonl(args.output_dir / "rejected.jsonl", rejected)
    write_jsonl(args.output_dir / "val_agent_eval.jsonl", eval_records)
    summary = {
        "name": "driftsql_execution_verified_five_tool_sft_v1",
        "source_rows": len(rows),
        "accepted_rows": len(records["train"]) + len(records["val"]),
        "rejected_rows": len(rejected),
        "splits": {
            split: {
                "rows": len(records[split]),
                "databases": len({row["db_id"] for row in manifests[split]}),
            }
            for split in ("train", "val")
        },
        "database_overlap": overlap,
        "rl_rows": {split: len(rl_records[split]) for split in ("train", "val")},
        "tool_calls": dict(sorted(tool_counts.items())),
        "token_length": {
            "min": min(token_lengths),
            "median": statistics.median(token_lengths),
            "max": max(token_lengths),
            "budget": args.max_tokens,
        },
        "validation": "real versioned SQLite + native VERL tools + result fingerprint",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split-summary", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--tools", type=Path, default=DEFAULT_TOOLS)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-schema-chars", type=int, default=3500)
    parser.add_argument("--max-result-rows", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    asyncio.run(build(args))


if __name__ == "__main__":
    main()
