#!/usr/bin/env python3
"""Replay Dataset V2 with native tools and emit SFT/RL-ready splits.

Unlike the original fixed six-action trajectory builder, this script respects
the interaction profile assigned by the V2 data factory.  Every accepted row
is replayed against an isolated, materialized SQLite database before it is
written to parquet.
"""

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
DEFAULT_INPUT = PROJECT_ROOT / "data/processed/stratified_v2"
DEFAULT_TOOLS = PROJECT_ROOT / "configs/tools/drift_tools.yaml"
DEFAULT_TOKENIZER = PROJECT_ROOT / "models/Qwen2.5-Coder-3B-Instruct"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/processed/stratified_five_tool_v2"
SPLITS = ("train", "dev", "test")
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


def search_terms(row: dict[str, Any]) -> str:
    values: list[str] = []
    for operation in row["schema_diff"]["operations"]:
        for key in ("table", "old_name", "new_name"):
            value = str(operation.get(key, "")).strip()
            if value and value not in values:
                values.append(value)
    return " ".join(values)


def expected_tool_sequence(profile: str) -> list[str]:
    sequences = {
        "must_ask": [
            "get_schema", "ask_user", "get_knowledge_definition",
            "execute_sql", "execute_sql", "submit_solution",
        ],
        "knowledge_only": [
            "get_schema", "get_knowledge_definition",
            "execute_sql", "execute_sql", "submit_solution",
        ],
        "schema_only": [
            "get_schema", "execute_sql", "execute_sql", "submit_solution",
        ],
        "direct_clean": ["execute_sql", "submit_solution"],
    }
    if profile not in sequences:
        raise ValueError(f"Unknown interaction profile: {profile}")
    return sequences[profile]


async def replay_row(
    row: dict[str, Any],
    *,
    tools: dict[str, Any],
    schemas: list[dict[str, Any]],
    schemas_json: str,
    tokenizer: Any,
    max_tokens: int,
    tool_counts: Counter[str],
    debug: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], int]:
    task_id = str(row["task_id"])
    instance_id = f"stratified-v2-{task_id}"
    profile = str(row["interaction_profile"])
    clean = profile == "direct_clean"
    spec = None if clean else clarification_spec(row)
    ambiguity = []
    knowledge_entries = []
    if profile == "must_ask":
        assert spec is not None
        ambiguity = [{
            "term": spec["term"],
            "sql_snippet": spec["definition"],
            "type": spec["ambiguity_type"],
        }]
    if profile in {"must_ask", "knowledge_only"}:
        assert spec is not None
        knowledge_entries = [spec["knowledge_entry"]]

    base_state = {
        "db_id": str(row["db_id"]),
        "db_version": str(row["schema_diff"].get("to_version", "v2")),
        "metric_version": "stratified-v2",
        "source_db": str(Path(row["source_db"]).resolve()),
        "schema_diff": row["schema_diff"],
        "query": str(row["question"]),
        "stale_sql": str(row["stale_sql"]),
        "ground_truth": str(row["repaired_sql"]),
        "result_fingerprint": row["result_fingerprint"],
        "user_query_ambiguity": {
            "critical_ambiguity": ambiguity,
            "non_critical_ambiguity": [],
        },
        "knowledge_entries": knowledge_entries,
        # Local batch preparation runs inside a process that has already
        # initialized Torch/Ray thread pools. On this machine, scheduling the
        # SQLite copy through asyncio.to_thread can leave its Future asleep
        # even while the worker is idle. Synchronous I/O is deterministic for
        # offline replay; live agent loops retain the default async path.
        "sync_io": True,
    }
    created: set[str] = set()
    def log(stage: str) -> None:
        if debug:
            print(f"debug {task_id}: {stage}", flush=True)

    try:
        log("create execute_sql")
        await tools["execute_sql"].create(instance_id=instance_id, create_kwargs=base_state)
        created.add("execute_sql")
        log("execute_sql created")
        active_db = Path(tools["execute_sql"]._state(instance_id)["db_path"])
        active_schema = relevant_schema_ddl(active_db, str(row["repaired_sql"]))
        log("schema derived")
        state = base_state | {"schema": active_schema}
        for name in TOOL_NAMES:
            if name != "execute_sql":
                await tools[name].create(instance_id=instance_id, create_kwargs=state)
                created.add(name)
        log("all tools created")

        async def call(name: str, parameters: dict[str, Any]) -> tuple[str, dict[str, Any]]:
            response, _, metrics = await tools[name].execute(instance_id, parameters)
            tool_counts[name] += 1
            log(f"called {name}")
            return str(response.text), dict(metrics)

        steps: list[dict[str, Any]] = []
        validations: dict[str, bool] = {}
        if profile != "direct_clean":
            schema_text, metrics = await call("get_schema", {"query": search_terms(row)})
            validations["schema_retrieved"] = bool(metrics.get("schema_retrieved"))
            steps.append({
                "action": "get_schema",
                "arguments": {"query": search_terms(row)},
                "observation": schema_text,
            })
        if profile == "must_ask":
            assert spec is not None
            text, metrics = await call("ask_user", {"question": spec["question"]})
            validations["clarification_matched"] = bool(metrics.get("clarification_matched"))
            steps.append({
                "action": "ask_user",
                "arguments": {"question": spec["question"]},
                "observation": text,
            })
        if profile in {"must_ask", "knowledge_only"}:
            assert spec is not None
            text, metrics = await call("get_knowledge_definition", {"name": spec["term"]})
            validations["knowledge_retrieved"] = bool(metrics.get("knowledge_retrieved"))
            steps.append({
                "action": "get_knowledge_definition",
                "arguments": {"name": spec["term"]},
                "observation": text,
            })

        if not clean:
            stale_text, stale_metrics = await call("execute_sql", {"sql": str(row["stale_sql"])})
            steps.append({
                "action": "execute_sql",
                "thought_key": "execute_sql_stale",
                "arguments": {"sql": str(row["stale_sql"])},
                "observation": stale_text,
            })
        else:
            stale_metrics = {}

        repaired_text, repaired_metrics = await call("execute_sql", {"sql": str(row["repaired_sql"])})
        steps.append({
            "action": "execute_sql",
            "thought_key": "execute_sql_clean" if clean else "execute_sql_repaired",
            "arguments": {"sql": str(row["repaired_sql"])},
            "observation": repaired_text,
        })
        submit_text, submit_metrics = await call("submit_solution", {"sql": str(row["repaired_sql"])})
        steps.append({
            "action": "submit_solution",
            "thought_key": "submit_clean" if clean else "submit_solution",
            "arguments": {"sql": str(row["repaired_sql"])},
            "observation": submit_text,
        })

        repaired_fingerprint = fingerprint_query(active_db, str(row["repaired_sql"]))
        log("repaired fingerprinted")
        expected = row["result_fingerprint"]
        validations.update({
            "repaired_execution_success": bool(repaired_metrics.get("execution_success")),
            "session_isolated": bool(repaired_metrics.get("session_isolated")),
            "rolled_back": bool(repaired_metrics.get("rolled_back")),
            "submitted": bool(submit_metrics.get("submitted")),
            "fingerprint_matches": (
                repaired_fingerprint.row_count == int(expected["row_count"])
                and repaired_fingerprint.value_hash == str(expected["value_hash"])
            ),
        })
        failure_mode = str(row["failure_mode"])
        if failure_mode == "silent_result_mismatch":
            stale_fingerprint = fingerprint_query(active_db, str(row["stale_sql"]))
            validations["stale_behavior_verified"] = stale_fingerprint != repaired_fingerprint
        elif failure_mode == "clean_no_drift":
            # Generation already guarantees stale_sql == repaired_sql for a
            # clean control. Re-fingerprinting the identical query would
            # double the cost of the slowest clean episodes without adding a
            # new validation signal.
            validations["stale_behavior_verified"] = (
                str(row["stale_sql"]) == str(row["repaired_sql"])
            )
        else:
            validations["stale_behavior_verified"] = not bool(stale_metrics.get("execution_success"))
        if not all(validations.values()):
            raise RuntimeError(f"validation failed: {validations}")

        actions = [step["action"] for step in steps]
        if actions != expected_tool_sequence(profile):
            raise RuntimeError(f"profile sequence mismatch: {actions}")
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
        log(f"tokenized {token_count}")
        if token_count > max_tokens:
            raise RuntimeError(f"token budget: {token_count} > {max_tokens}")

        tools_kwargs = {name: {"create_kwargs": dict(state)} for name in TOOL_NAMES}
        sft_record = {"messages": messages, "tools": schemas_json, "enable_thinking": False}
        rl_record = {
            "data_source": (
                f"driftsql/v2/{row['scenario_type']}/{row['drift_type']}/{profile}"
            ),
            "prompt": messages[:2],
            "ability": "interactive_sql_drift_recovery",
            "reward_model": {"ground_truth": str(row["repaired_sql"])},
            "extra_info": {
                "instance_id": task_id,
                "db_id": str(row["db_id"]),
                "source_db": str(Path(row["source_db"]).resolve()),
                "schema_diff": row["schema_diff"],
                "result_fingerprint": row["result_fingerprint"],
                "stale_sql": str(row["stale_sql"]),
                "scenario_type": str(row["scenario_type"]),
                "drift_type": str(row["drift_type"]),
                "interaction_profile": profile,
                "difficulty": str(row["difficulty"]),
                "failure_mode": failure_mode,
                "need_tools_kwargs": True,
                "tools_kwargs": tools_kwargs,
                "tool_selection": list(TOOL_NAMES),
            },
            "return_raw_chat": True,
            "agent_name": "driftsql_tool_agent",
        }
        manifest = {
            "task_id": task_id,
            "db_id": str(row["db_id"]),
            "source": str(row["source"]),
            "scenario_type": str(row["scenario_type"]),
            "drift_type": str(row["drift_type"]),
            "interaction_profile": profile,
            "difficulty": str(row["difficulty"]),
            "failure_mode": failure_mode,
            "legacy_task": bool(row["legacy_task"]),
            "tool_sequence": actions,
            "token_count": token_count,
            "clarification": spec,
            "validations": validations,
        }
        return sft_record, rl_record, manifest, token_count
    finally:
        for name in created:
            log(f"release {name}")
            try:
                await tools[name].release(instance_id)
            except KeyError:
                pass
        log("released")


async def build(args: argparse.Namespace) -> dict[str, Any]:
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

    records: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    rl_records: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    manifests: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    rejected: list[dict[str, Any]] = []
    token_lengths: list[int] = []
    tool_counts: Counter[str] = Counter()

    for split in SPLITS:
        rows = load_jsonl(args.input_dir / f"{split}.jsonl")
        if args.limit_per_split > 0:
            rows = rows[: args.limit_per_split]
        for start in range(0, len(rows), args.concurrency):
            batch = rows[start : start + args.concurrency]

            async def guarded(row: dict[str, Any]) -> tuple[Any, Exception | None]:
                try:
                    return (
                        await replay_row(
                            row,
                            tools=tools,
                            schemas=schemas,
                            schemas_json=schemas_json,
                            tokenizer=tokenizer,
                            max_tokens=args.max_tokens,
                            tool_counts=tool_counts,
                            debug=args.debug,
                        ),
                        None,
                    )
                except Exception as error:  # noqa: BLE001 - rejection is a data artifact
                    return None, error

            results = await asyncio.gather(*(guarded(row) for row in batch))
            for offset, (row, (result, error)) in enumerate(zip(batch, results, strict=True)):
                index = start + offset
                if error is not None:
                    rejected.append({
                        "split": split,
                        "task_id": str(row.get("task_id", index)),
                        "db_id": str(row.get("db_id", "")),
                        "error": f"{type(error).__name__}: {error}",
                    })
                    continue
                assert result is not None
                sft, rl, manifest, token_count = result
                records[split].append(sft)
                rl_records[split].append(rl)
                manifests[split].append(manifest)
                token_lengths.append(token_count)
            completed = min(start + len(batch), len(rows))
            if completed % 25 < len(batch) or completed == len(rows):
                print(
                    f"{split}: replayed {completed}/{len(rows)}; "
                    f"accepted={len(manifests[split])}; rejected_total={len(rejected)}",
                    flush=True,
                )

    split_databases = {
        split: {row["db_id"] for row in manifests[split]}
        for split in SPLITS
    }
    overlaps = {
        "train_dev": sorted(split_databases["train"] & split_databases["dev"]),
        "train_test": sorted(split_databases["train"] & split_databases["test"]),
        "dev_test": sorted(split_databases["dev"] & split_databases["test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"database leakage after replay: {overlaps}")
    if rejected and not args.allow_rejections:
        write_jsonl(args.output_dir / "rejected.jsonl", rejected)
        raise RuntimeError(f"Dataset replay rejected {len(rejected)} rows; inspect rejected.jsonl")

    for split in SPLITS:
        write_parquet(args.output_dir / f"{split}.parquet", records[split])
        write_parquet(args.output_dir / f"rl_{split}.parquet", rl_records[split])
        write_jsonl(args.output_dir / f"{split}_manifest.jsonl", manifests[split])
        if split in {"dev", "test"}:
            write_jsonl(args.output_dir / f"{split}_agent_eval.jsonl", rl_records[split])
    write_jsonl(args.output_dir / "rejected.jsonl", rejected)

    summary = {
        "name": "driftsql_stratified_execution_verified_five_tool_v2",
        "accepted_rows": sum(len(values) for values in manifests.values()),
        "rejected_rows": len(rejected),
        "splits": {
            split: {
                "rows": len(manifests[split]),
                "databases": len(split_databases[split]),
                "profiles": dict(sorted(Counter(row["interaction_profile"] for row in manifests[split]).items())),
                "tool_steps": dict(sorted(Counter(len(row["tool_sequence"]) for row in manifests[split]).items())),
            }
            for split in SPLITS
        },
        "database_overlap": overlaps,
        "tool_calls": dict(sorted(tool_counts.items())),
        "token_length": {
            "min": min(token_lengths),
            "median": statistics.median(token_lengths),
            "max": max(token_lengths),
            "budget": args.max_tokens,
        },
        "validation": "real versioned SQLite + native VERL tools + result fingerprint",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--tools", type=Path, default=DEFAULT_TOOLS)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-schema-chars", type=int, default=3500)
    parser.add_argument("--max-result-rows", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=6144)
    parser.add_argument("--limit-per-split", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--allow-rejections", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    asyncio.run(build(args))


if __name__ == "__main__":
    main()
