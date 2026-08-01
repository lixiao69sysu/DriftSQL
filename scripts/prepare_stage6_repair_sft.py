#!/usr/bin/env python3
"""Build execution-verified Stage 6 repair/next-action SFT trajectories.

The curriculum is intentionally short: observe the cached SQL, inspect the
audited drift once, repair once, and submit.  It does not include repeated
schema/version calls, so the supervision directly targets the B1 failure mode.
Only Stage 6 train and tune are consumed; the sealed gate is never read.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data/processed/stage6_ablation/b1"
DEFAULT_TOOLS = PROJECT_ROOT / "configs/tools/drift_tools.yaml"
DEFAULT_TOKENIZER = PROJECT_ROOT / "models/Qwen2.5-Coder-7B-Instruct"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/processed/stage6_repair_sft"
DEFAULT_VERIFIED_MANIFEST = (
    PROJECT_ROOT / "data/processed/stratified_five_tool_v2/train_manifest.jsonl"
)
SPLITS = ("train", "tune")
TOOL_NAMES = (
    "get_schema_version",
    "inspect_schema_diff",
    "get_schema",
    "get_knowledge_definition",
    "execute_sql",
    "submit_solution",
)

THOUGHTS = {
    "execute_stale": "I will run the cached read-only SQL once to observe its active behavior.",
    "execute_clean": "The cached SQL may still be valid, so I will validate it once before submission.",
    "version": "The cached SQL failed or changed behavior, so I will check the active schema version once.",
    "diff": "The schema changed; I will inspect the audited diff once instead of guessing identifiers.",
    "knowledge": "I will retrieve the governed definition once before repairing the business query.",
    "execute_repaired": "I will apply the audited change and validate the repaired SQL once.",
    "submit": "The SQL executed successfully, so I will submit that exact validated query now.",
}


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
        raise ValueError(f"No records for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
    temporary.replace(path)


def assistant_message(thought: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": f"<think>{THOUGHTS[thought]}</think>",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        ],
    }


async def build_trajectory(
    record: dict[str, Any],
    *,
    tools: dict[str, Any],
    schemas: list[dict[str, Any]],
    schemas_json: str,
    tokenizer: Any,
    max_tokens: int,
    verified_task_ids: set[str],
) -> tuple[dict[str, Any], dict[str, Any], int]:
    extra = record["extra_info"]
    instance_id = f"stage6-repair-{extra['instance_id']}"
    profile = str(extra["interaction_profile"])
    clean = str(extra["scenario_type"]) == "clean"
    stale_sql = str(extra["stale_sql"])
    repaired_sql = str(record["reward_model"]["ground_truth"])
    state = dict(extra["tools_kwargs"]["execute_sql"]["create_kwargs"])
    # This standalone CPU data job has no initialized vLLM/Ray thread pools,
    # so native tools can safely move database copy/query work to asyncio's
    # executor.  That makes the per-instance isolation genuinely concurrent.
    state["sync_io"] = False
    created: set[str] = set()
    messages = [dict(message) for message in record["prompt"]]
    actions: list[str] = []
    validations: dict[str, bool] = {}
    validations["source_fingerprint_verified"] = str(extra["instance_id"]) in verified_task_ids

    async def call(name: str, arguments: dict[str, Any], thought: str) -> tuple[str, dict[str, Any]]:
        if name not in created:
            await tools[name].create(instance_id=instance_id, create_kwargs=state)
            created.add(name)
        response, _, metrics = await tools[name].execute(instance_id, arguments)
        messages.append(assistant_message(thought, name, arguments))
        actions.append(name)
        if name != "submit_solution":
            messages.append({"role": "tool", "content": str(response.text)})
        return str(response.text), dict(metrics)

    try:
        if clean:
            _, repaired_metrics = await call(
                "execute_sql", {"sql": repaired_sql}, "execute_clean"
            )
        else:
            _, stale_metrics = await call(
                "execute_sql", {"sql": stale_sql}, "execute_stale"
            )
            _, version_metrics = await call("get_schema_version", {}, "version")
            _, diff_metrics = await call("inspect_schema_diff", {}, "diff")
            validations["schema_version_checked"] = bool(
                version_metrics.get("schema_version_checked")
            )
            validations["schema_diff_inspected"] = bool(
                diff_metrics.get("schema_diff_inspected")
            )

            knowledge_entries = list(state.get("knowledge_entries", []) or [])
            if profile in {"knowledge_only", "must_ask"} and knowledge_entries:
                knowledge_name = str(knowledge_entries[0].get("knowledge", "")).strip()
                _, knowledge_metrics = await call(
                    "get_knowledge_definition", {"name": knowledge_name}, "knowledge"
                )
                validations["knowledge_retrieved"] = bool(
                    knowledge_metrics.get("knowledge_retrieved")
                )

            _, repaired_metrics = await call(
                "execute_sql", {"sql": repaired_sql}, "execute_repaired"
            )
            failure_mode = str(extra["failure_mode"])
            if failure_mode == "explicit_schema_error":
                validations["stale_behavior_verified"] = not bool(
                    stale_metrics.get("execution_success")
                )
            else:
                # The immutable source manifest already verified the stale and
                # repaired full-result fingerprints.  Re-executing here proves
                # that the new tool path still reaches a live isolated DB;
                # rescanning the full result would only duplicate that work.
                validations["stale_behavior_verified"] = bool(
                    stale_metrics.get("execution_success")
                )

        _, submit_metrics = await call(
            "submit_solution", {"sql": repaired_sql}, "submit"
        )
        validations.update(
            {
                "repaired_execution_success": bool(repaired_metrics.get("execution_success")),
                "session_isolated": bool(repaired_metrics.get("session_isolated")),
                "rolled_back": bool(repaired_metrics.get("rolled_back")),
                "submitted": bool(submit_metrics.get("submitted")),
                "fingerprint_matches": validations["source_fingerprint_verified"],
                "no_repeated_tools": all(count == 1 for count in Counter(actions).values() if count),
            }
        )
        # execute_sql legitimately appears twice on drift trajectories: stale
        # observation followed by one repaired candidate.
        validations["no_repeated_tools"] = all(
            count <= (2 if name == "execute_sql" and not clean else 1)
            for name, count in Counter(actions).items()
        )
        if not all(validations.values()):
            raise RuntimeError(f"validation failed: {validations}")

        token_count = len(
            tokenizer.apply_chat_template(
                messages,
                tools=schemas,
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        )
        if token_count > max_tokens:
            raise RuntimeError(f"token budget: {token_count} > {max_tokens}")
        sft = {
            "messages": messages,
            "tools": schemas_json,
            "enable_thinking": False,
        }
        manifest = {
            "task_id": str(extra["instance_id"]),
            "db_id": str(extra["db_id"]),
            "scenario_type": str(extra["scenario_type"]),
            "drift_type": str(extra["drift_type"]),
            "interaction_profile": profile,
            "difficulty": str(extra["difficulty"]),
            "failure_mode": str(extra["failure_mode"]),
            "tool_sequence": actions,
            "token_count": token_count,
            "validations": validations,
        }
        return sft, manifest, token_count
    finally:
        for name in created:
            try:
                await tools[name].release(instance_id)
            except KeyError:
                pass


async def build(args: argparse.Namespace) -> dict[str, Any]:
    loaded = load_all_tools(tool_config_path=str(args.tools), function_tool_path=None)
    tools = {tool.name: tool for tool in loaded if tool.name in TOOL_NAMES}
    if set(tools) != set(TOOL_NAMES):
        raise RuntimeError(f"Missing tools: {sorted(set(TOOL_NAMES) - set(tools))}")
    tools["execute_sql"].config["max_rows"] = args.max_result_rows
    tools["get_knowledge_definition"].config["max_results"] = 1
    schemas = [tools[name].tool_schema.model_dump(mode="json") for name in TOOL_NAMES]
    schemas_json = json.dumps(schemas, ensure_ascii=False)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    verified_manifest = load_jsonl(args.verified_manifest)
    verified_task_ids = {
        str(row["task_id"])
        for row in verified_manifest
        if bool(row.get("validations", {}).get("fingerprint_matches"))
        and bool(row.get("validations", {}).get("repaired_execution_success"))
    }

    all_manifests: dict[str, list[dict[str, Any]]] = {}
    rejected: list[dict[str, Any]] = []
    token_counts: list[int] = []
    for split in SPLITS:
        records = load_jsonl(args.input_dir / f"{split}_agent_eval.jsonl")
        if args.limit_per_split > 0:
            records = records[: args.limit_per_split]
        accepted: list[dict[str, Any]] = []
        manifests: list[dict[str, Any]] = []
        for start in range(0, len(records), args.concurrency):
            batch = records[start : start + args.concurrency]

            async def guarded(record: dict[str, Any]) -> tuple[Any, Exception | None]:
                try:
                    return (
                        await build_trajectory(
                            record,
                            tools=tools,
                            schemas=schemas,
                            schemas_json=schemas_json,
                            tokenizer=tokenizer,
                            max_tokens=args.max_tokens,
                            verified_task_ids=verified_task_ids,
                        ),
                        None,
                    )
                except Exception as error:  # noqa: BLE001 - auditable rejection
                    return None, error

            results = await asyncio.gather(*(guarded(record) for record in batch))
            for offset, (record, (result, error)) in enumerate(
                zip(batch, results, strict=True)
            ):
                if error is not None:
                    rejected.append(
                        {
                            "split": split,
                            "task_id": record.get("extra_info", {}).get(
                                "instance_id", str(start + offset + 1)
                            ),
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                    continue
                assert result is not None
                sft, manifest, token_count = result
                accepted.append(sft)
                manifests.append(manifest)
                token_counts.append(token_count)
            completed = min(start + len(batch), len(records))
            if completed % 25 < len(batch) or completed == len(records):
                print(
                    f"{split}: replayed {completed}/{len(records)}; accepted={len(accepted)}; "
                    f"rejected_total={len(rejected)}",
                    flush=True,
                )
        write_parquet(args.output_dir / f"{split}.parquet", accepted)
        write_jsonl(args.output_dir / f"{split}_manifest.jsonl", manifests)
        all_manifests[split] = manifests

    write_jsonl(args.output_dir / "rejected.jsonl", rejected)
    if rejected and not args.allow_rejections:
        raise RuntimeError(f"Repair SFT rejected {len(rejected)} rows; inspect rejected.jsonl")
    split_databases = {
        split: {row["db_id"] for row in manifests}
        for split, manifests in all_manifests.items()
    }
    overlap = sorted(split_databases["train"] & split_databases["tune"])
    if overlap:
        raise RuntimeError(f"Stage 6 train/tune database leakage: {overlap}")
    summary = {
        "name": "driftsql_stage6_execution_verified_repair_sft_v1",
        "curriculum": "observe cached SQL -> inspect drift once -> repair once -> submit",
        "splits": {
            split: {
                "rows": len(manifests),
                "databases": len(split_databases[split]),
                "profiles": dict(sorted(Counter(row["interaction_profile"] for row in manifests).items())),
                "target_sequences": dict(
                    sorted(Counter(" -> ".join(row["tool_sequence"]) for row in manifests).items())
                ),
            }
            for split, manifests in all_manifests.items()
        },
        "train_tune_database_overlap": overlap,
        "rejected_rows": len(rejected),
        "token_length": {
            "min": min(token_counts),
            "median": statistics.median(token_counts),
            "max": max(token_counts),
            "budget": args.max_tokens,
        },
        "validation": "real isolated versioned SQLite + native tools + result fingerprint",
        "verified_source_manifest": str(args.verified_manifest.resolve()),
        "sealed_gate_read": False,
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
    parser.add_argument(
        "--verified-manifest", type=Path, default=DEFAULT_VERIFIED_MANIFEST
    )
    parser.add_argument("--max-result-rows", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=6144)
    parser.add_argument("--limit-per-split", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--allow-rejections", action="store_true")
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    asyncio.run(build(args))


if __name__ == "__main__":
    main()
