#!/usr/bin/env python3
"""Prepare focused Stage 7 add-column SFT plus general-drift replay.

Only the Stage 7 train/tune partitions are consumed.  The new Stage 7 Gate
and permanently sealed Stage 6 Gate112 are deliberately absent from every
default and code path in this program.
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from transformers import AutoTokenizer

from driftsql.data.tool_sft import expand_next_action_messages, use_plain_json_for_last_action
from driftsql.data.trajectory import relevant_schema_ddl
from driftsql.drift import materialize_schema_diff
from driftsql.integrations.state_policy import schema_diff_recovery_guidance
from driftsql.integrations.verl_tools import _active_schema_for_projection
from driftsql.planning import plan_projection_contract


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = PROJECT_ROOT / "data/processed/stage7_add_column_protocol"
DEFAULT_STAGE6_RECORDS = PROJECT_ROOT / "data/processed/stage6_ablation/b1"
DEFAULT_STAGE6_SFT = PROJECT_ROOT / "data/processed/stage6_repair_next_action_v2"
DEFAULT_TOOLS = PROJECT_ROOT / "configs/tools/drift_tools.yaml"
DEFAULT_TOKENIZER = PROJECT_ROOT / "models/Qwen2.5-Coder-7B-Instruct"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/processed/stage7_add_column_sft"
SPLITS = ("train", "tune")
TOOL_NAMES = (
    "get_schema_version",
    "inspect_schema_diff",
    "get_schema",
    "get_knowledge_definition",
    "execute_sql",
    "submit_solution",
)

B1_SYSTEM_PROMPT = """You are a production SQL recovery agent operating on a versioned database.
Previously valid SQL may contain stale identifiers or a stale result-column contract. You cannot ask
the user follow-up questions. First check the active schema version. When the database has changed,
inspect the audited schema diff before editing SQL. Retrieve the active schema and governed business
knowledge only when they are needed, execute a read-only candidate for validation, then submit it.

For additive schema drift, preserve the original result-column contract: expand projection wildcards
to the old ordered columns and exclude newly added audit columns. Use the deterministic
projection_contract_plan returned by inspect_schema_diff when it is available.

Never guess an identifier that is available in inspect_schema_diff. Do not repeat an identical tool
call or SQL execution. After a successful execute_sql, either repair the query using new evidence or
submit the validated SQL. Every assistant turn must contain one concise <think> block followed by
exactly one tool call. Finish with submit_solution within the tool budget."""

USER_TEMPLATE = """## Analytics request
{question}

## Previously valid cached SQL
{stale_sql}

The active database schema may have added operational or audit columns.
Use the interactive tools, preserve the original projection contract, validate the query, and submit it."""

THOUGHTS = {
    "execute_stale": "I will execute the cached SQL once to detect a silent result-column contract change.",
    "version": "The cached result contract may be stale, so I will check the active schema version.",
    "diff": "I will inspect the audited diff and use its deterministic projection-contract plan.",
    "execute_repaired": "I will validate the planned explicit projection on the active database.",
    "submit": "The repaired projection executed successfully, so I will submit that exact SQL.",
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
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
    temporary.replace(path)


def load_tool_schemas(path: Path) -> list[dict[str, Any]]:
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
                "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
            }
        ],
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
    result["source"] = "stage7_execution_verified_factory"
    return result


def diff_observation(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row["schema_diff"])
    guidance = schema_diff_recovery_guidance(result)
    if guidance:
        result["recovery_guidance"] = guidance
    active_schema = _active_schema_for_projection(Path(row["source_db"]), result)
    plan = plan_projection_contract(row["stale_sql"], result, active_schema)
    if plan.repaired_sql != row["repaired_sql"]:
        raise RuntimeError(f"Projection planner changed for {row['task_id']}")
    result["projection_contract_plan"] = plan.to_dict()
    return result


def active_schema_ddl(row: dict[str, Any]) -> str:
    temporary_root = PROJECT_ROOT / "tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="stage7-schema-", dir=temporary_root) as directory:
        active = Path(directory) / f"{row['db_id']}__v2.sqlite"
        materialize_schema_diff(Path(row["source_db"]), active, row["schema_diff"])
        return relevant_schema_ddl(active, row["repaired_sql"])


def build_add_trajectory(
    row: dict[str, Any],
    *,
    schemas_json: str,
    tokenizer: Any,
    max_tokens: int,
    stage_name: str = "stage7",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    stale_sql = str(row["stale_sql"])
    repaired_sql = str(row["repaired_sql"])
    schema = active_schema_ddl(row)
    prompt = [
        {"role": "system", "content": B1_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(question=row["question"], stale_sql=stale_sql),
        },
    ]
    messages = list(prompt)
    steps = [
        ("execute_stale", "execute_sql", {"sql": stale_sql}, execution_observation(row, stale_sql, last=False)),
        (
            "version",
            "get_schema_version",
            {},
            {
                "db_id": row["db_id"],
                "db_version": "v2",
                "metric_version": f"{stage_name}-v1",
            },
        ),
        ("diff", "inspect_schema_diff", {}, diff_observation(row)),
        (
            "execute_repaired",
            "execute_sql",
            {"sql": repaired_sql},
            execution_observation(row, repaired_sql, last=True),
        ),
        ("submit", "submit_solution", {"sql": repaired_sql}, None),
    ]
    for thought, name, arguments, observation in steps:
        messages.append(assistant(thought, name, arguments))
        if observation is not None:
            messages.append(
                {
                    "role": "tool",
                    "content": json.dumps(observation, ensure_ascii=False),
                }
            )
    token_count = len(
        tokenizer.apply_chat_template(
            messages,
            tools=json.loads(schemas_json),
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    )
    if token_count > max_tokens:
        raise RuntimeError(f"{row['task_id']}: {token_count} > {max_tokens}")
    state = {
        "db_id": str(row["db_id"]),
        "db_version": "v2",
        "metric_version": f"{stage_name}-v1",
        "source_db": str(Path(row["source_db"]).resolve()),
        "schema_diff": row["schema_diff"],
        "query": str(row["question"]),
        "stale_sql": stale_sql,
        "ground_truth": repaired_sql,
        "result_fingerprint": row["result_fingerprint"],
        "schema": schema,
        "knowledge_entries": [],
        "user_query_ambiguity": {"critical_ambiguity": [], "non_critical_ambiguity": []},
    }
    profile = str(row["wildcard_profile"])
    difficulty = "hard" if profile.startswith("multi_table") else (
        "medium" if profile.endswith("qualified") else "easy"
    )
    extra = {
        "instance_id": str(row["task_id"]),
        "db_id": str(row["db_id"]),
        "source_db": state["source_db"],
        "schema_diff": row["schema_diff"],
        "result_fingerprint": row["result_fingerprint"],
        "stale_sql": stale_sql,
        "scenario_type": "schema_drift",
        "drift_type": "add_column",
        "interaction_profile": "schema_only",
        "difficulty": difficulty,
        "failure_mode": "silent_result_mismatch",
        "wildcard_profile": profile,
        "added_column_count": int(row["added_column_count"]),
        "need_tools_kwargs": True,
        "tools_kwargs": {name: {"create_kwargs": dict(state)} for name in TOOL_NAMES},
        "tool_selection": list(TOOL_NAMES),
        f"{stage_name}_variant": "projection_contract_planner",
    }
    agent_record = {
        "data_source": f"driftsql/{stage_name}/add_column/{profile}",
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
        "drift_type": "add_column",
        "interaction_profile": "schema_only",
        "difficulty": difficulty,
        "failure_mode": "silent_result_mismatch",
        "wildcard_profile": profile,
        "added_column_count": int(row["added_column_count"]),
    }
    audit = {
        "task_id": str(row["task_id"]),
        "db_id": str(row["db_id"]),
        "token_count": token_count,
        "wildcard_profile": profile,
        "validations": {
            "factory_execution_verified": True,
            "planner_matches_repaired_sql": True,
            "projection_plan_in_observation": True,
            "stage6_gate112_read": False,
            "stage7_gate_read": False,
        },
    }
    return trajectory, agent_record, audit


def expand_add(trajectories: list[dict[str, Any]], *, train: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trajectory_index, trajectory in enumerate(trajectories):
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
                repeats += 2 * int(action == "execute_sql" and "inspect_schema_diff" in prior)
                repeats += 2 * int(action == "submit_solution")
            payload = {
                key: value
                for key, value in trajectory.items()
                if key != "messages"
            }
            payload.update(
                {
                    "messages": use_plain_json_for_last_action(prefix),
                    "target_action": action,
                    "trajectory_index": trajectory_index,
                    "replay_source": "stage7_add_column",
                }
            )
            rows.extend(dict(payload) for _ in range(repeats))
    return rows


def balanced_sample(rows: list[dict[str, Any]], count: int, *, seed: int) -> list[dict[str, Any]]:
    if count >= len(rows):
        return list(rows)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("drift_type", "")), str(row.get("target_action", "")))].append(row)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    selected: list[dict[str, Any]] = []
    keys = sorted(groups)
    while len(selected) < count:
        progressed = False
        for key in keys:
            if groups[key] and len(selected) < count:
                item = dict(groups[key].pop())
                item["replay_source"] = "stage6_general_drift"
                selected.append(item)
                progressed = True
        if not progressed:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-dir", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--stage6-record-dir", type=Path, default=DEFAULT_STAGE6_RECORDS)
    parser.add_argument("--stage6-sft-dir", type=Path, default=DEFAULT_STAGE6_SFT)
    parser.add_argument("--tools", type=Path, default=DEFAULT_TOOLS)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-tokens", type=int, default=6144)
    parser.add_argument("--general-replay-ratio", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=72027)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")

    schemas = load_tool_schemas(args.tools)
    schemas_json = json.dumps(schemas, ensure_ascii=False)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    stage6_sft = pq.read_table(args.stage6_sft_dir / "train.parquet").to_pylist()
    stage6_agent = load_jsonl(args.stage6_record_dir / "train_agent_eval.jsonl")
    sft_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in stage6_sft:
        sft_by_task[str(row.get("task_id", ""))].append(row)
    agent_by_task = {str(row["extra_info"]["instance_id"]): row for row in stage6_agent}

    summary: dict[str, Any] = {
        "name": "driftsql_stage7_projection_contract_sft_v1",
        "policy": "focused add-column supervision + database-isolated general-drift replay",
        "general_replay_ratio": args.general_replay_ratio,
        "splits": {},
        "stage6_gate112_read": False,
        "stage7_gate_read": False,
    }
    for split in SPLITS:
        add_rows = load_jsonl(args.protocol_dir / f"{split}_add_column.jsonl")
        general_manifest = load_jsonl(args.protocol_dir / f"{split}_general_replay.jsonl")
        trajectories: list[dict[str, Any]] = []
        add_agent: list[dict[str, Any]] = []
        audits: list[dict[str, Any]] = []
        for index, row in enumerate(add_rows, 1):
            trajectory, agent, audit = build_add_trajectory(
                row,
                schemas_json=schemas_json,
                tokenizer=tokenizer,
                max_tokens=args.max_tokens,
            )
            trajectories.append(trajectory)
            add_agent.append(agent)
            audits.append(audit)
            if index % 12 == 0:
                print(f"{split}: prepared {index}/{len(add_rows)} add-column trajectories", flush=True)

        add_examples = expand_add(trajectories, train=split == "train")
        general_ids = {str(row["task_id"]) for row in general_manifest}
        general_pool = [
            dict(item)
            for task_id in sorted(general_ids)
            for item in sft_by_task.get(task_id, [])
        ]
        missing_sft = sorted(task_id for task_id in general_ids if task_id not in sft_by_task)
        missing_agent = sorted(task_id for task_id in general_ids if task_id not in agent_by_task)
        if missing_sft or missing_agent:
            raise RuntimeError(
                f"{split}: missing Stage6 replay rows sft={missing_sft[:3]} agent={missing_agent[:3]}"
            )
        replay_count = round(
            len(add_examples) * args.general_replay_ratio / (1.0 - args.general_replay_ratio)
        )
        replay = balanced_sample(general_pool, replay_count, seed=args.seed + (split == "tune"))
        combined = add_examples + replay
        random.Random(args.seed + 10 + (split == "tune")).shuffle(combined)
        general_agent = [agent_by_task[task_id] for task_id in sorted(general_ids)]
        agent_records = add_agent + general_agent

        write_parquet(args.output_dir / f"{split}.parquet", combined)
        write_parquet(args.output_dir / f"rl_{split}.parquet", agent_records)
        write_jsonl(args.output_dir / f"{split}_agent_eval.jsonl", agent_records)
        write_jsonl(args.output_dir / f"{split}_add_audit.jsonl", audits)
        write_parquet(args.output_dir / f"{split}_add_trajectories.parquet", trajectories)
        summary["splits"][split] = {
            "add_trajectories": len(trajectories),
            "add_supervision_examples": len(add_examples),
            "general_replay_examples": len(replay),
            "supervision_examples": len(combined),
            "agent_eval_rows": len(agent_records),
            "agent_eval_add_rows": len(add_agent),
            "agent_eval_general_rows": len(general_agent),
            "databases": len({row["extra_info"]["db_id"] for row in agent_records}),
            "wildcard_profiles": dict(sorted(Counter(row["wildcard_profile"] for row in audits).items())),
            "target_actions": dict(sorted(Counter(row["target_action"] for row in combined).items())),
            "replay_sources": dict(sorted(Counter(row["replay_source"] for row in combined).items())),
        }

    train_dbs = {
        row["extra_info"]["db_id"] for row in load_jsonl(args.output_dir / "train_agent_eval.jsonl")
    }
    tune_dbs = {
        row["extra_info"]["db_id"] for row in load_jsonl(args.output_dir / "tune_agent_eval.jsonl")
    }
    overlap = sorted(train_dbs & tune_dbs)
    if overlap:
        raise RuntimeError(f"Stage 7 train/tune database leakage: {overlap}")
    summary["train_tune_database_overlap"] = overlap
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
