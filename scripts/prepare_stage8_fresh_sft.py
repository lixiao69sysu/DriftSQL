#!/usr/bin/env python3
"""Prepare Stage 8 SFT/RL data from fresh Train/Tune databases only.

The Stage 8 Gate and all prior gates are intentionally absent from every
input/default.  Add-column examples use the deterministic projection planner;
general replay preserves execution-verified factory observations.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driftsql.data.tool_sft import expand_next_action_messages, use_plain_json_for_last_action
from driftsql.integrations.state_policy import schema_diff_recovery_guidance
from scripts.prepare_stage7_add_column_sft import (
    B1_SYSTEM_PROMPT,
    TOOL_NAMES,
    active_schema_ddl,
    balanced_sample,
    build_add_trajectory,
    load_jsonl,
    load_tool_schemas,
    write_jsonl,
    write_parquet,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "data/processed/stage8_fresh_protocol"
DEFAULT_TOOLS = ROOT / "configs/tools/drift_tools.yaml"
DEFAULT_TOKENIZER = ROOT / "models/Qwen2.5-Coder-7B-Instruct"
DEFAULT_OUTPUT = ROOT / "data/processed/stage8_fresh_sft"
SPLITS = ("train", "tune")

GENERAL_USER = """## Analytics request
{question}

## Previously valid cached SQL
{stale_sql}

The active database schema may differ from the cached schema. Use the audited interactive tools,
validate the correct read-only SQL, and submit that exact validated query."""

GENERAL_THOUGHTS = {
    "execute_stale": "I will execute the cached SQL once to observe its active behavior.",
    "execute_clean": "The cached SQL may still be valid, so I will validate it before submission.",
    "version": "The cached behavior is stale, so I will check the active schema version.",
    "diff": "I will inspect the audited schema diff instead of guessing identifiers.",
    "execute_repaired": "I will validate the repaired SQL against the active database.",
    "submit": "The SQL executed successfully, so I will submit that exact validated query.",
}


def assistant(thought: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": f"<think>{GENERAL_THOUGHTS[thought]}</think>",
        "tool_calls": [{
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
        }],
    }


def execution_observation(row: dict[str, Any], sql: str, *, last: bool) -> dict[str, Any]:
    matches = [
        dict(step["observation"])
        for step in row["oracle_steps"]
        if step.get("action") == "execute_sql"
        and str(step.get("arguments", {}).get("sql", "")).strip() == sql.strip()
    ]
    if not matches:
        raise RuntimeError(f"Missing execution audit for {row['task_id']}")
    result = matches[-1 if last else 0]
    result["success"] = bool(result.pop("ok", result.get("success", False)))
    result.setdefault("error", None)
    result["rolled_back"] = True
    result["source"] = "stage8_execution_verified_factory"
    return result


def build_general_trajectory(
    row: dict[str, Any],
    *,
    schemas: list[dict[str, Any]],
    schemas_json: str,
    tokenizer: Any,
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    stale_sql = str(row["stale_sql"])
    repaired_sql = str(row["repaired_sql"])
    clean = str(row["drift_type"]) == "clean"
    prompt = [
        {"role": "system", "content": B1_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": GENERAL_USER.format(question=row["question"], stale_sql=stale_sql),
        },
    ]
    messages = list(prompt)
    actions: list[str] = []

    def append(
        thought: str,
        name: str,
        arguments: dict[str, Any],
        observation: dict[str, Any] | None,
    ) -> None:
        messages.append(assistant(thought, name, arguments))
        actions.append(name)
        if observation is not None:
            messages.append({"role": "tool", "content": json.dumps(observation, ensure_ascii=False)})

    if clean:
        append(
            "execute_clean",
            "execute_sql",
            {"sql": repaired_sql},
            execution_observation(row, repaired_sql, last=True),
        )
    else:
        append(
            "execute_stale",
            "execute_sql",
            {"sql": stale_sql},
            execution_observation(row, stale_sql, last=False),
        )
        append(
            "version",
            "get_schema_version",
            {},
            {"db_id": row["db_id"], "db_version": "v2", "metric_version": "stage8-v1"},
        )
        diff = dict(row["schema_diff"])
        guidance = schema_diff_recovery_guidance(diff)
        if guidance:
            diff["recovery_guidance"] = guidance
        append("diff", "inspect_schema_diff", {}, diff)
        append(
            "execute_repaired",
            "execute_sql",
            {"sql": repaired_sql},
            execution_observation(row, repaired_sql, last=True),
        )
    append("submit", "submit_solution", {"sql": repaired_sql}, None)
    expected = (
        ["execute_sql", "submit_solution"]
        if clean
        else [
            "execute_sql",
            "get_schema_version",
            "inspect_schema_diff",
            "execute_sql",
            "submit_solution",
        ]
    )
    if actions != expected:
        raise RuntimeError(f"Unexpected Stage 8 action sequence: {actions}")
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
        raise RuntimeError(f"{row['task_id']}: {token_count} > {max_tokens}")

    source_db = str(Path(row["source_db"]).resolve())
    state = {
        "db_id": str(row["db_id"]),
        "db_version": "v1" if clean else "v2",
        "metric_version": "stage8-v1",
        "source_db": source_db,
        "schema_diff": row["schema_diff"],
        "query": str(row["question"]),
        "stale_sql": stale_sql,
        "ground_truth": repaired_sql,
        "result_fingerprint": row["result_fingerprint"],
        "schema": active_schema_ddl(row),
        "knowledge_entries": [],
        "user_query_ambiguity": {"critical_ambiguity": [], "non_critical_ambiguity": []},
    }
    extra = {
        "instance_id": str(row["task_id"]),
        "db_id": str(row["db_id"]),
        "source_db": source_db,
        "schema_diff": row["schema_diff"],
        "result_fingerprint": row["result_fingerprint"],
        "stale_sql": stale_sql,
        "scenario_type": str(row["scenario_type"]),
        "drift_type": str(row["drift_type"]),
        "interaction_profile": str(row["interaction_profile"]),
        "difficulty": str(row["difficulty"]),
        "failure_mode": str(row["failure_mode"]),
        "need_tools_kwargs": True,
        "tools_kwargs": {name: {"create_kwargs": dict(state)} for name in TOOL_NAMES},
        "tool_selection": list(TOOL_NAMES),
        "stage8_variant": "fresh_db_general_replay",
    }
    agent = {
        "data_source": f"driftsql/stage8/general/{row['drift_type']}",
        "prompt": prompt,
        "ability": "interactive_sql_drift_recovery",
        "reward_model": {"ground_truth": repaired_sql},
        "extra_info": extra,
        "return_raw_chat": True,
        "agent_name": "driftsql_tool_agent",
    }
    trajectory = {
        "messages": messages,
        "tools": schemas_json,
        "enable_thinking": False,
        "task_id": str(row["task_id"]),
        "db_id": str(row["db_id"]),
        "drift_type": str(row["drift_type"]),
        "interaction_profile": str(row["interaction_profile"]),
        "difficulty": str(row["difficulty"]),
        "failure_mode": str(row["failure_mode"]),
    }
    audit = {
        "task_id": str(row["task_id"]),
        "db_id": str(row["db_id"]),
        "tool_sequence": actions,
        "token_count": token_count,
        "validations": {
            "factory_execution_verified": True,
            "repaired_fingerprint_present": True,
            "stage7_gate106_read": False,
            "stage8_gate_read": False,
        },
    }
    return trajectory, agent, audit


def expand_trajectory(
    trajectory: dict[str, Any],
    *,
    train: bool,
    replay_source: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prefix in expand_next_action_messages(trajectory["messages"]):
        action = str(prefix[-1]["tool_calls"][0]["function"]["name"])
        prior = [
            str(message["tool_calls"][0]["function"]["name"])
            for message in prefix[:-1]
            if message.get("role") == "assistant" and message.get("tool_calls")
        ]
        repeats = 1
        if train:
            repeats += int(action == "inspect_schema_diff")
            repeats += int(action == "execute_sql" and "inspect_schema_diff" in prior)
            repeats += 4 * int(action == "submit_solution" and replay_source == "stage8_add_column")
            repeats += 2 * int(action == "submit_solution" and replay_source == "stage8_general_replay")
        payload = {key: value for key, value in trajectory.items() if key != "messages"}
        payload.update(
            {
                "messages": use_plain_json_for_last_action(prefix),
                "target_action": action,
                "replay_source": replay_source,
            }
        )
        rows.extend(dict(payload) for _ in range(repeats))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-dir", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--tools", type=Path, default=DEFAULT_TOOLS)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-tokens", type=int, default=6144)
    parser.add_argument("--general-replay-ratio", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=82028)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if not 0.0 < args.general_replay_ratio < 1.0:
        parser.error("--general-replay-ratio must be between zero and one")

    schemas = load_tool_schemas(args.tools)
    schemas_json = json.dumps(schemas, ensure_ascii=False)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    summary: dict[str, Any] = {
        "name": "driftsql_stage8_fresh_db_submit_sft_v1",
        "policy": "projection-contract focus + submit-decision weighting + fresh general replay",
        "general_replay_ratio_requested": args.general_replay_ratio,
        "stage7_gate106_read": False,
        "stage8_gate_read": False,
        "splits": {},
    }
    split_dbs: dict[str, set[str]] = {}
    for split in SPLITS:
        add_rows = load_jsonl(args.protocol_dir / f"{split}_add_column.jsonl")
        general_rows = load_jsonl(args.protocol_dir / f"{split}_general_replay.jsonl")
        add_trajectories = []
        general_trajectories = []
        agents = []
        audits = []
        for index, row in enumerate(add_rows, 1):
            trajectory, agent, audit = build_add_trajectory(
                row,
                schemas_json=schemas_json,
                tokenizer=tokenizer,
                max_tokens=args.max_tokens,
                stage_name="stage8",
            )
            audit["validations"]["stage8_gate_read"] = False
            add_trajectories.append(trajectory)
            agents.append(agent)
            audits.append(audit | {"family": "add_column"})
            if index % 30 == 0:
                print(f"{split}: prepared {index}/{len(add_rows)} add trajectories", flush=True)
        for row in general_rows:
            trajectory, agent, audit = build_general_trajectory(
                row,
                schemas=schemas,
                schemas_json=schemas_json,
                tokenizer=tokenizer,
                max_tokens=args.max_tokens,
            )
            general_trajectories.append(trajectory)
            agents.append(agent)
            audits.append(audit | {"family": "general_replay"})

        add_examples = [
            example
            for trajectory in add_trajectories
            for example in expand_trajectory(
                trajectory, train=split == "train", replay_source="stage8_add_column"
            )
        ]
        general_pool = [
            example
            for trajectory in general_trajectories
            for example in expand_trajectory(
                trajectory, train=split == "train", replay_source="stage8_general_replay"
            )
        ]
        replay_count = round(
            len(add_examples) * args.general_replay_ratio / (1.0 - args.general_replay_ratio)
        )
        replay = balanced_sample(general_pool, replay_count, seed=args.seed + (split == "tune"))
        for item in replay:
            item["replay_source"] = "stage8_general_replay"
        combined = add_examples + replay
        random.Random(args.seed + 10 + (split == "tune")).shuffle(combined)
        split_dbs[split] = {str(row["extra_info"]["db_id"]) for row in agents}

        write_parquet(args.output_dir / f"{split}.parquet", combined)
        write_parquet(args.output_dir / f"rl_{split}.parquet", agents)
        write_jsonl(args.output_dir / f"{split}_agent_eval.jsonl", agents)
        write_jsonl(args.output_dir / f"{split}_audit.jsonl", audits)
        write_parquet(args.output_dir / f"{split}_add_trajectories.parquet", add_trajectories)
        write_parquet(args.output_dir / f"{split}_general_trajectories.parquet", general_trajectories)
        summary["splits"][split] = {
            "databases": len(split_dbs[split]),
            "add_trajectories": len(add_trajectories),
            "general_trajectories": len(general_trajectories),
            "agent_eval_rows": len(agents),
            "add_supervision_examples": len(add_examples),
            "general_replay_pool": len(general_pool),
            "general_replay_examples": len(replay),
            "supervision_examples": len(combined),
            "general_replay_ratio_actual": len(replay) / len(combined),
            "target_actions": dict(sorted(Counter(row["target_action"] for row in combined).items())),
            "replay_sources": dict(sorted(Counter(row["replay_source"] for row in combined).items())),
        }

    overlap = sorted(split_dbs["train"] & split_dbs["tune"])
    if overlap:
        raise RuntimeError(f"Stage 8 train/tune database leakage: {overlap}")
    summary["train_tune_database_overlap"] = overlap
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
