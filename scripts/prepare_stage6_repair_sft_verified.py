#!/usr/bin/env python3
"""Recompose verified Dataset V2 episodes into concise Stage 6 repair SFT.

No SQL or expected result is synthesized here.  The script joins the immutable
Stage 6 protocol rows with the prior execution-verified manifest, then reuses
their verified oracle observations in a shorter version/diff/repair/submit
policy.  Gate112 is deliberately absent from ``SPLITS``.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORDS = PROJECT_ROOT / "data/processed/stage6_ablation/b1"
DEFAULT_PROTOCOL = PROJECT_ROOT / "data/processed/stage6_protocol"
DEFAULT_VERIFIED = PROJECT_ROOT / "data/processed/stratified_five_tool_v2/train_manifest.jsonl"
DEFAULT_TOOLS = PROJECT_ROOT / "configs/tools/drift_tools.yaml"
DEFAULT_TOKENIZER = PROJECT_ROOT / "models/Qwen2.5-Coder-7B-Instruct"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/processed/stage6_repair_sft"
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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
    temporary.replace(path)


def load_schemas(path: Path) -> list[dict[str, Any]]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    by_name = {
        item["tool_schema"]["function"]["name"]: item["tool_schema"]
        for item in config["tools"]
    }
    missing = sorted(set(TOOL_NAMES) - set(by_name))
    if missing:
        raise RuntimeError(f"Missing tool schemas: {missing}")
    return [by_name[name] for name in TOOL_NAMES]


def assistant(thought: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
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


def append_step(
    messages: list[dict[str, Any]],
    actions: list[str],
    *,
    thought: str,
    name: str,
    arguments: dict[str, Any],
    observation: Any | None,
) -> None:
    messages.append(assistant(thought, name, arguments))
    actions.append(name)
    if observation is not None:
        text = observation if isinstance(observation, str) else json.dumps(observation, ensure_ascii=False)
        messages.append({"role": "tool", "content": text})


def oracle_execution(row: dict[str, Any], sql: str, *, last: bool) -> dict[str, Any]:
    matches = [
        step["observation"]
        for step in row["oracle_steps"]
        if step.get("action") == "execute_sql"
        and str(step.get("arguments", {}).get("sql", "")).strip() == sql.strip()
    ]
    if not matches:
        raise RuntimeError("Verified oracle execution observation is missing")
    observation = dict(matches[-1] if last else matches[0])
    # Normalize the generator's compact audit fields toward the live executor
    # contract without inventing rows or success outcomes.
    observation["success"] = bool(observation.pop("ok", observation.get("success", False)))
    observation.setdefault("error", None)
    observation["rolled_back"] = True
    observation["source"] = "verified_dataset_v2_oracle"
    return observation


def build_trajectory(
    record: dict[str, Any],
    source: dict[str, Any],
    verified: dict[str, Any],
    *,
    schemas: list[dict[str, Any]],
    schemas_json: str,
    tokenizer: Any,
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    extra = record["extra_info"]
    task_id = str(extra["instance_id"])
    if task_id != str(source["task_id"]) or task_id != str(verified["task_id"]):
        raise RuntimeError("Task join changed task identity")
    source_validations = dict(verified.get("validations", {}))
    required = {
        "repaired_execution_success",
        "submitted",
        "fingerprint_matches",
        "stale_behavior_verified",
    }
    if not all(bool(source_validations.get(name)) for name in required):
        raise RuntimeError(f"Source verification is incomplete: {source_validations}")

    stale_sql = str(extra["stale_sql"])
    repaired_sql = str(record["reward_model"]["ground_truth"])
    if repaired_sql != str(source["repaired_sql"]):
        raise RuntimeError("Ground-truth SQL changed during Stage 6 join")
    clean = str(extra["scenario_type"]) == "clean"
    profile = str(extra["interaction_profile"])
    messages = [dict(message) for message in record["prompt"]]
    actions: list[str] = []

    if clean:
        append_step(
            messages,
            actions,
            thought="execute_clean",
            name="execute_sql",
            arguments={"sql": repaired_sql},
            observation=oracle_execution(source, repaired_sql, last=True),
        )
    else:
        append_step(
            messages,
            actions,
            thought="execute_stale",
            name="execute_sql",
            arguments={"sql": stale_sql},
            observation=oracle_execution(source, stale_sql, last=False),
        )
        state = extra["tools_kwargs"]["execute_sql"]["create_kwargs"]
        append_step(
            messages,
            actions,
            thought="version",
            name="get_schema_version",
            arguments={},
            observation={
                "db_id": str(extra["db_id"]),
                "db_version": state.get("db_version"),
                "metric_version": state.get("metric_version"),
            },
        )
        append_step(
            messages,
            actions,
            thought="diff",
            name="inspect_schema_diff",
            arguments={},
            observation=extra["schema_diff"],
        )
        knowledge_entries = list(state.get("knowledge_entries", []) or [])
        if profile in {"knowledge_only", "must_ask"} and knowledge_entries:
            name = str(knowledge_entries[0].get("knowledge", "")).strip()
            append_step(
                messages,
                actions,
                thought="knowledge",
                name="get_knowledge_definition",
                arguments={"name": name},
                observation={"query": name, "matches": [knowledge_entries[0]]},
            )
        append_step(
            messages,
            actions,
            thought="execute_repaired",
            name="execute_sql",
            arguments={"sql": repaired_sql},
            observation=oracle_execution(source, repaired_sql, last=True),
        )
    append_step(
        messages,
        actions,
        thought="submit",
        name="submit_solution",
        arguments={"sql": repaired_sql},
        observation=None,
    )

    counts = Counter(actions)
    if counts["execute_sql"] != (1 if clean else 2):
        raise RuntimeError(f"Unexpected execute count: {actions}")
    if any(count > 1 for name, count in counts.items() if name != "execute_sql"):
        raise RuntimeError(f"Repeated retrieval/submit action: {actions}")
    if len(actions) > 6 or actions[-1] != "submit_solution":
        raise RuntimeError(f"Trajectory violates six-turn submit contract: {actions}")

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
        "task_id": task_id,
        "db_id": str(extra["db_id"]),
        "drift_type": str(extra["drift_type"]),
        "interaction_profile": profile,
        "difficulty": str(extra["difficulty"]),
        "failure_mode": str(extra["failure_mode"]),
    }
    manifest = {
        "task_id": task_id,
        "db_id": str(extra["db_id"]),
        "scenario_type": str(extra["scenario_type"]),
        "drift_type": str(extra["drift_type"]),
        "interaction_profile": profile,
        "difficulty": str(extra["difficulty"]),
        "failure_mode": str(extra["failure_mode"]),
        "tool_sequence": actions,
        "token_count": token_count,
        "source_validations": source_validations,
    }
    return sft, manifest, token_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-dir", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--protocol-dir", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--verified-manifest", type=Path, default=DEFAULT_VERIFIED)
    parser.add_argument("--tools", type=Path, default=DEFAULT_TOOLS)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-tokens", type=int, default=6144)
    args = parser.parse_args()

    schemas = load_schemas(args.tools)
    schemas_json = json.dumps(schemas, ensure_ascii=False)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    verified_rows = load_jsonl(args.verified_manifest)
    verified_by_id = {str(row["task_id"]): row for row in verified_rows}
    all_manifests: dict[str, list[dict[str, Any]]] = {}
    token_counts: list[int] = []

    for split in SPLITS:
        records = load_jsonl(args.records_dir / f"{split}_agent_eval.jsonl")
        source_rows = load_jsonl(args.protocol_dir / f"{split}.jsonl")
        source_by_id = {str(row["task_id"]): row for row in source_rows}
        if {str(row["extra_info"]["instance_id"]) for row in records} != set(source_by_id):
            raise RuntimeError(f"{split}: B1 records and protocol task IDs differ")
        sft_rows: list[dict[str, Any]] = []
        manifests: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for record in records:
            task_id = str(record["extra_info"]["instance_id"])
            try:
                sft, manifest, token_count = build_trajectory(
                    record,
                    source_by_id[task_id],
                    verified_by_id[task_id],
                    schemas=schemas,
                    schemas_json=schemas_json,
                    tokenizer=tokenizer,
                    max_tokens=args.max_tokens,
                )
                sft_rows.append(sft)
                manifests.append(manifest)
                token_counts.append(token_count)
            except Exception as error:  # noqa: BLE001 - auditable data rejection
                rejected.append({"split": split, "task_id": task_id, "error": f"{type(error).__name__}: {error}"})
        if rejected:
            write_jsonl(args.output_dir / f"{split}_rejected.jsonl", rejected)
            raise RuntimeError(f"{split}: rejected {len(rejected)} verified trajectories")
        write_parquet(args.output_dir / f"{split}.parquet", sft_rows)
        write_jsonl(args.output_dir / f"{split}_manifest.jsonl", manifests)
        all_manifests[split] = manifests

    databases = {
        split: {row["db_id"] for row in manifests}
        for split, manifests in all_manifests.items()
    }
    overlap = sorted(databases["train"] & databases["tune"])
    if overlap:
        raise RuntimeError(f"train/tune database leakage: {overlap}")
    summary = {
        "name": "driftsql_stage6_verified_repair_sft_v2",
        "source_policy": "immutable Dataset V2 execution/fingerprint audit; no SQL synthesized",
        "curriculum": "execute cached -> version -> diff -> optional HKB -> execute repair -> submit",
        "splits": {
            split: {
                "rows": len(manifests),
                "databases": len(databases[split]),
                "profiles": dict(sorted(Counter(row["interaction_profile"] for row in manifests).items())),
                "target_sequences": dict(sorted(Counter(" -> ".join(row["tool_sequence"]) for row in manifests).items())),
            }
            for split, manifests in all_manifests.items()
        },
        "train_tune_database_overlap": overlap,
        "token_length": {
            "min": min(token_counts),
            "median": statistics.median(token_counts),
            "max": max(token_counts),
            "budget": args.max_tokens,
        },
        "sealed_gate_read": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
