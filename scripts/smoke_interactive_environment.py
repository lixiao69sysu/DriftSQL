#!/usr/bin/env python3
"""Run a real Mini-Interact task through every DriftSQL interactive tool."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from verl.tools.tool_registry import load_all_tools

from driftsql.data import build_mini_interact_eval_record, load_mini_interact_rows


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


async def run_smoke(data_root: Path, tool_config: Path, row_index: int) -> dict[str, Any]:
    rows = load_mini_interact_rows(data_root)
    row = rows[row_index]
    record = build_mini_interact_eval_record(row, data_root, index=row_index)
    tools = {
        tool.name: tool
        for tool in load_all_tools(tool_config_path=str(tool_config), function_tool_path=None)
    }
    selected = record["extra_info"]["tool_selection"]
    tool_kwargs = record["extra_info"]["tools_kwargs"]
    instance_id = f"interactive-smoke-{row['instance_id']}"
    db_path = Path(tool_kwargs["execute_sql"]["create_kwargs"]["db_path"])
    before = sha256(db_path)
    events: list[dict[str, Any]] = []

    async def call(name: str, parameters: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        tool = tools[name]
        started = time.monotonic()
        await tool.create(
            instance_id=instance_id,
            create_kwargs=tool_kwargs[name]["create_kwargs"],
        )
        response, reward, metrics = await tool.execute(instance_id, parameters)
        events.append(
            {
                "tool": name,
                "arguments": parameters,
                "observation": response.text,
                "reward": reward,
                "metrics": metrics,
                "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
            }
        )
        return str(response.text), metrics

    critical = row["user_query_ambiguity"]["critical_ambiguity"][0]
    knowledge_items = row.get("knowledge_ambiguity", [])
    knowledge_name = (
        str(knowledge_items[0]["term"])
        if knowledge_items
        else str(critical["term"])
    )
    try:
        user_text, user_metrics = await call(
            "ask_user", {"question": f"What does {critical['term']} mean?"}
        )
        schema_text, schema_metrics = await call("get_schema", {"query": ""})
        knowledge_text, knowledge_metrics = await call(
            "get_knowledge_definition", {"name": knowledge_name}
        )
        execution_text, execution_metrics = await call("execute_sql", {"sql": "SELECT 1 AS ok"})
        submit_text, submit_metrics = await call("submit_solution", {"sql": "SELECT 1 AS ok"})
        isolated_path = Path(tools["execute_sql"]._state(instance_id)["db_path"])
        validations = {
            "clarification_matched": bool(user_metrics.get("clarification_matched")),
            "schema_retrieved": bool(schema_metrics.get("schema_retrieved")),
            "knowledge_retrieved": bool(knowledge_metrics.get("knowledge_retrieved")),
            "execution_success": bool(execution_metrics.get("execution_success")),
            "session_isolated": bool(execution_metrics.get("session_isolated"))
            and isolated_path.resolve() != db_path.resolve(),
            "action_rolled_back": bool(execution_metrics.get("rolled_back")),
            "submitted": bool(submit_metrics.get("submitted")),
        }
        if not all(validations.values()):
            raise RuntimeError(f"Interactive smoke validation failed: {validations}")
        result = {
            "instance_id": row["instance_id"],
            "database": row["selected_database"],
            "selected_tools": selected,
            "validations": validations,
            "source_db_unchanged": False,
            "events": events,
            "observations": {
                "ask_user": user_text,
                "get_schema": json.loads(schema_text),
                "get_knowledge_definition": json.loads(knowledge_text),
                "execute_sql": json.loads(execution_text),
                "submit_solution": submit_text,
            },
        }
    finally:
        for name in selected:
            try:
                await tools[name].release(instance_id)
            except KeyError:
                pass
    result["source_db_unchanged"] = sha256(db_path) == before
    if not result["source_db_unchanged"]:
        raise RuntimeError("Source Mini-Interact database changed during isolated smoke")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/raw/mini-interact")
    parser.add_argument("--tools", type=Path, default=PROJECT_ROOT / "configs/tools/drift_tools.yaml")
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports/interactive_environment_smoke.json",
    )
    args = parser.parse_args()

    result = asyncio.run(run_smoke(args.data_root, args.tools, args.row))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
