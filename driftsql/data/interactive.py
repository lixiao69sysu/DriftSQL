"""BIRD Mini-Interact records for the trajectory-stateful VERL environment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


INTERACTIVE_SYSTEM_PROMPT = """You are a production analytics SQL agent.
The user's request can contain genuine ambiguity and organization-specific
business concepts. Ask one focused clarification question when required,
retrieve schema and knowledge definitions, validate SQL in the isolated
database session, and submit one final read-only query.

Do not invent user requirements or metric definitions. Use ask_user only for
ambiguities in the request and get_knowledge_definition for business knowledge.
Every assistant turn must contain one concise <think> block followed by exactly
one tool call. Finish with submit_solution."""


INTERACTIVE_TOOL_NAMES = (
    "ask_user",
    "get_schema",
    "get_knowledge_definition",
    "execute_sql",
    "submit_solution",
)


def load_mini_interact_rows(root: Path) -> list[dict[str, Any]]:
    task_file = Path(root) / "mini_interact.jsonl"
    if not task_file.is_file():
        raise FileNotFoundError(task_file)
    return [
        json.loads(line)
        for line in task_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def mini_interact_tool_state(row: dict[str, Any], root: Path) -> dict[str, Any]:
    """Resolve all public assets needed by one guarded interactive session."""

    root = Path(root).resolve()
    db_id = str(row["selected_database"])
    db_dir = root / db_id
    assets = {
        "db_path": db_dir / f"{db_id}.sqlite",
        "schema_path": db_dir / f"{db_id}_schema.txt",
        "knowledge_base_path": db_dir / f"{db_id}_kb.jsonl",
        "column_meaning_path": db_dir / f"{db_id}_column_meaning_base.json",
    }
    missing = [str(path) for path in assets.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Mini-Interact assets: " + ", ".join(missing))
    return {
        "db_id": db_id,
        "db_version": "mini-interact-public",
        "metric_version": "hkb-public",
        "instance_id": str(row["instance_id"]),
        "query": str(row["amb_user_query"]),
        "user_query_ambiguity": row.get("user_query_ambiguity", {}),
        "knowledge_ambiguity": row.get("knowledge_ambiguity", []),
        "isolate_db": True,
        **{name: str(path.resolve()) for name, path in assets.items()},
    }


def build_mini_interact_eval_record(
    row: dict[str, Any],
    root: Path,
    *,
    index: int,
) -> dict[str, Any]:
    """Build an inference/evaluation record; public labels remain intentionally empty."""

    state = mini_interact_tool_state(row, root)
    tools_kwargs = {
        name: {"create_kwargs": dict(state)} for name in INTERACTIVE_TOOL_NAMES
    }
    return {
        "data_source": "bird_interact_public",
        "prompt": [
            {"role": "system", "content": INTERACTIVE_SYSTEM_PROMPT},
            {"role": "user", "content": str(row["amb_user_query"])},
        ],
        "ability": "interactive_text_to_sql",
        "reward_model": {"ground_truth": ""},
        "extra_info": {
            "instance_id": str(row["instance_id"]),
            "index": index,
            "db_id": state["db_id"],
            "public_ground_truth_available": bool(row.get("sol_sql") and row.get("test_cases")),
            "need_tools_kwargs": True,
            "tools_kwargs": tools_kwargs,
            "tool_selection": list(INTERACTIVE_TOOL_NAMES),
        },
        "return_raw_chat": True,
        "agent_name": "driftsql_tool_agent",
    }
