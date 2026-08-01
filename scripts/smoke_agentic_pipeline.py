#!/usr/bin/env python3
"""Run one real drift record through VERL tools and the custom reward."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from verl.tools.tool_registry import load_all_tools

from driftsql.rewards.agentic import compute_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "column_rename"
    / "rl_train.parquet"
)
DEFAULT_TOOLS = PROJECT_ROOT / "configs" / "tools" / "drift_tools.yaml"


def _trace(calls: list[tuple[str, dict[str, Any]]]) -> str:
    return "\n".join(
        "<tool_call>"
        + json.dumps(
            {"name": name, "arguments": arguments},
            ensure_ascii=False,
        )
        + "</tool_call>"
        for name, arguments in calls
    )


async def _execute_tool(
    tool: Any,
    *,
    instance_id: str,
    create_kwargs: dict[str, Any],
    parameters: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    created_id, _ = await tool.create(
        instance_id=instance_id,
        create_kwargs=create_kwargs,
    )
    try:
        response, _, metrics = await tool.execute(created_id, parameters)
        return str(response.text), metrics
    finally:
        await tool.release(created_id)


async def _run(row: dict[str, Any], tool_config: Path) -> dict[str, Any]:
    tools = {
        tool.name: tool
        for tool in load_all_tools(
            tool_config_path=str(tool_config),
            function_tool_path=None,
        )
    }
    extra = row["extra_info"]
    tool_kwargs = extra["tools_kwargs"]
    stale_sql = str(extra["stale_sql"])
    repaired_sql = str(row["reward_model"]["ground_truth"])

    version_text, _ = await _execute_tool(
        tools["get_schema_version"],
        instance_id="smoke-version",
        create_kwargs=tool_kwargs["get_schema_version"]["create_kwargs"],
        parameters={},
    )
    diff_text, _ = await _execute_tool(
        tools["inspect_schema_diff"],
        instance_id="smoke-diff",
        create_kwargs=tool_kwargs["inspect_schema_diff"]["create_kwargs"],
        parameters={},
    )
    stale_text, stale_metrics = await _execute_tool(
        tools["execute_sql"],
        instance_id="smoke-stale",
        create_kwargs=tool_kwargs["execute_sql"]["create_kwargs"],
        parameters={"sql": stale_sql},
    )
    repaired_text, repaired_metrics = await _execute_tool(
        tools["execute_sql"],
        instance_id="smoke-repaired",
        create_kwargs=tool_kwargs["execute_sql"]["create_kwargs"],
        parameters={"sql": repaired_sql},
    )
    submit_text, submit_metrics = await _execute_tool(
        tools["submit_solution"],
        instance_id="smoke-submit",
        create_kwargs=tool_kwargs["submit_solution"]["create_kwargs"],
        parameters={"sql": repaired_sql},
    )

    calls = [
        ("execute_sql", {"sql": stale_sql}),
        ("get_schema_version", {}),
        ("inspect_schema_diff", {}),
        ("execute_sql", {"sql": repaired_sql}),
        ("submit_solution", {"sql": repaired_sql}),
    ]
    correct_reward = compute_score(
        data_source=str(row["data_source"]),
        solution_str=_trace(calls),
        ground_truth=row["reward_model"]["ground_truth"],
        extra_info=extra,
    )
    stale_reward = compute_score(
        data_source=str(row["data_source"]),
        solution_str=_trace(
            [("submit_solution", {"sql": stale_sql})]
        ),
        ground_truth=row["reward_model"]["ground_truth"],
        extra_info=extra,
    )

    drift_types = {
        str(operation.get("type"))
        for operation in extra["schema_diff"].get("operations", [])
    }
    expects_silent_mismatch = "add_column" in drift_types
    if expects_silent_mismatch and not stale_metrics.get("execution_success"):
        raise RuntimeError("Silent-drift SQL unexpectedly failed to execute")
    if not expects_silent_mismatch and stale_metrics.get("execution_success"):
        raise RuntimeError("Hard-drift SQL unexpectedly executed successfully")
    if not repaired_metrics.get("execution_success"):
        raise RuntimeError("Repaired SQL failed in the rollout tool")
    if not submit_metrics.get("submitted"):
        raise RuntimeError("Submission tool rejected a non-empty SQL query")
    if not correct_reward["task_success"] or stale_reward["task_success"]:
        raise RuntimeError("Reward did not separate repaired and stale SQL")

    return {
        "instance_id": extra["instance_id"],
        "db_id": extra["db_id"],
        "version_observation": json.loads(version_text),
        "diff_observation": json.loads(diff_text),
        "stale_observation": json.loads(stale_text),
        "repaired_observation": json.loads(repaired_text),
        "submit_observation": submit_text,
        "correct_reward": correct_reward,
        "stale_reward": stale_reward,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--tools", type=Path, default=DEFAULT_TOOLS)
    parser.add_argument("--row", type=int, default=0)
    args = parser.parse_args()

    table = pq.read_table(args.data)
    if not 0 <= args.row < table.num_rows:
        parser.error(f"--row must be between 0 and {table.num_rows - 1}")
    row = table.slice(args.row, 1).to_pylist()[0]
    result = asyncio.run(_run(row, args.tools))
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
