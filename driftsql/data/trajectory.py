"""Convert drift manifests into VERL RL and multi-turn SFT records."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from sqlglot import exp, parse_one

DRIFT_SYSTEM_PROMPT = """You are a production SQL recovery agent.

The user provides a cached schema snapshot and a SQL query that used to be
valid. The active database may have changed since that snapshot.

Use the available tools to diagnose the active schema version, inspect audited
schema changes, test SQL against the active database, and submit one final
read-only SQL query. Do not guess a renamed identifier from naming patterns.
Validate the repaired query before submission when possible.

Every assistant turn must contain one concise <think> block followed by exactly
one <tool_call> JSON object. The final action must be submit_solution. You have
at most {max_turns} assistant turns."""

DRIFT_USER_TEMPLATE = """## Cached schema snapshot
Version: v1

{schema}

## Question
{question}

## Evidence
{evidence}

## Previously valid SQL
{stale_sql}

Recover and submit a correct query for the active database."""


def drift_tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_schema_version",
                "description": "Return active schema and metric versions.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "inspect_schema_diff",
                "description": "Inspect audited changes from cached to active schema.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "execute_sql",
                "description": "Execute one read-only SQL query on the active database.",
                "parameters": {
                    "type": "object",
                    "properties": {"sql": {"type": "string"}},
                    "required": ["sql"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "submit_solution",
                "description": "Submit the final SQL and end the trajectory.",
                "parameters": {
                    "type": "object",
                    "properties": {"sql": {"type": "string"}},
                    "required": ["sql"],
                },
            },
        },
    ]


def relevant_schema_ddl(database: Path, sql: str) -> str:
    tree = parse_one(sql, read="sqlite")
    referenced = {
        table.name.casefold()
        for table in tree.find_all(exp.Table)
        if table.name
    }
    uri = Path(database).resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        rows = connection.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
    selected = [
        str(ddl)
        for name, ddl in rows
        if ddl and (not referenced or str(name).casefold() in referenced)
    ]
    if not selected:
        raise ValueError("No physical table DDL found for query")
    return "\n\n".join(f"{ddl};" for ddl in selected)


def _prompt(
    manifest: dict[str, Any],
    schema: str,
    max_turns: int,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": DRIFT_SYSTEM_PROMPT.format(max_turns=max_turns),
        },
        {
            "role": "user",
            "content": DRIFT_USER_TEMPLATE.format(
                schema=schema,
                question=str(manifest["question"]),
                evidence=str(manifest.get("evidence", "")) or "(none)",
                stale_sql=str(manifest["stale_sql"]),
            ),
        },
    ]


def build_rl_record(
    manifest: dict[str, Any],
    *,
    index: int,
    split: str,
    max_turns: int = 5,
) -> dict[str, Any]:
    source_db = Path(str(manifest["source_db"])).resolve()
    schema = relevant_schema_ddl(source_db, str(manifest["stale_sql"]))
    metric_version = next(
        (
            str(operation.get("to_version"))
            for operation in reversed(manifest["schema_diff"].get("operations", []))
            if operation.get("type") == "metric_definition_change"
            and operation.get("to_version")
        ),
        "v1",
    )
    create_kwargs = {
        "db_id": str(manifest["db_id"]),
        "db_version": "v2",
        "metric_version": metric_version,
        "source_db": str(source_db),
        "schema_diff": manifest["schema_diff"],
        "query": str(manifest["question"]),
        "stale_sql": str(manifest["stale_sql"]),
        "ground_truth": str(manifest["repaired_sql"]),
        "result_fingerprint": manifest["result_fingerprint"],
    }
    tool_names = (
        "get_schema_version",
        "inspect_schema_diff",
        "execute_sql",
        "submit_solution",
    )
    tools_kwargs = {
        name: {"create_kwargs": dict(create_kwargs)} for name in tool_names
    }
    drift_types = sorted(
        {
            str(operation.get("type", "unknown"))
            for operation in manifest["schema_diff"].get("operations", [])
        }
    )
    data_source = "driftsql/" + "+".join(drift_types or ["unknown"])
    return {
        "data_source": data_source,
        "prompt": _prompt(manifest, schema, max_turns),
        "ability": "sql_drift_recovery",
        "reward_model": {"ground_truth": str(manifest["repaired_sql"])},
        "extra_info": {
            "instance_id": str(manifest["task_id"]),
            "index": index,
            "split": split,
            "source": str(manifest["source"]),
            "source_index": int(manifest["source_index"]),
            "db_id": str(manifest["db_id"]),
            "db_version": "v2",
            "metric_version": metric_version,
            "source_db": str(source_db),
            "schema": schema,
            "stale_sql": str(manifest["stale_sql"]),
            "schema_diff": manifest["schema_diff"],
            "result_fingerprint": manifest["result_fingerprint"],
            "need_tools_kwargs": True,
            "tools_kwargs": tools_kwargs,
            "tool_selection": list(tool_names),
        },
        "return_raw_chat": True,
        "agent_name": "driftsql_tool_agent",
    }


def _tool_call(action: str, arguments: dict[str, Any]) -> str:
    payload = json.dumps(
        {"name": action, "arguments": arguments},
        ensure_ascii=False,
    )
    return f"<tool_call>{payload}</tool_call>"


def _oracle_thought(action: str) -> str:
    return {
        "execute_sql": (
            "I will execute the candidate query to reproduce or verify its "
            "behavior on the active database."
        ),
        "get_schema_version": (
            "The execution error may come from stale metadata, so I will "
            "check the active schema version."
        ),
        "inspect_schema_diff": (
            "The cached and active versions differ, so I will inspect the "
            "audited schema changes."
        ),
        "submit_solution": (
            "The repaired query has been validated, so I will submit it."
        ),
    }[action]


def build_sft_record(
    manifest: dict[str, Any],
    *,
    max_turns: int = 5,
) -> dict[str, Any]:
    source_db = Path(str(manifest["source_db"])).resolve()
    schema = relevant_schema_ddl(source_db, str(manifest["stale_sql"]))
    messages: list[dict[str, str]] = _prompt(manifest, schema, max_turns)
    for step_index, step in enumerate(manifest["oracle_steps"]):
        action = str(step["action"])
        assistant = (
            f"<think>{_oracle_thought(action)}</think>\n"
            f"{_tool_call(action, dict(step.get('arguments', {})))}"
        )
        messages.append({"role": "assistant", "content": assistant})
        if (
            action != "submit_solution"
            and step_index < len(manifest["oracle_steps"]) - 1
        ):
            observation = json.dumps(
                step.get("observation", {}),
                ensure_ascii=False,
            )
            messages.append(
                {
                    "role": "user",
                    "content": f"Tool observation ({action}):\n{observation}",
                }
            )
    return {
        "messages": messages,
        "tools": drift_tool_schemas(),
        "enable_thinking": False,
    }
