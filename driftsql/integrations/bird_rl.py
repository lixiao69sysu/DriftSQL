"""Build VERL/BIRD-RL compatible records without importing either framework."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from driftsql.contracts import Task
from driftsql.drift import SchemaDiff


def build_agentic_rl_record(
    task: Task,
    prompt_messages: Sequence[dict[str, str]],
    db_path: Path,
    schema: str,
    schema_diff: SchemaDiff,
    ground_truth_sql: str,
    test_cases: Sequence[str],
    index: int,
    split: str,
) -> dict[str, Any]:
    """Create one row matching BIRD-RL's agentic parquet contract.

    The adapter keeps DriftSQL metadata in ``extra_info`` and forwards the
    same state to every trajectory-scoped tool through ``tools_kwargs``.
    """

    resolved_db_path = Path(db_path).resolve()
    if not resolved_db_path.exists():
        raise FileNotFoundError(resolved_db_path)
    if task.db_id != schema_diff.db_id:
        raise ValueError("Task and schema diff refer to different databases")
    if task.db_version != schema_diff.to_version:
        raise ValueError("Task version must match the schema diff target version")

    shared_create_kwargs = {
        "db_id": task.db_id,
        "db_path": str(resolved_db_path),
        "db_version": task.db_version,
        "metric_version": task.metric_version,
        "schema": schema,
        "schema_diff": schema_diff.to_observation(),
        "query": task.question,
        "ground_truth": ground_truth_sql,
        "test_cases": list(test_cases),
    }
    tool_names = [
        "get_schema_version",
        "inspect_schema_diff",
        "execute_sql",
        "submit_solution",
    ]
    tools_kwargs = {name: {"create_kwargs": shared_create_kwargs} for name in tool_names}

    return {
        "data_source": "driftsql_agentic",
        "prompt": list(prompt_messages),
        "ability": "sql_drift_recovery",
        "reward_model": {
            "ground_truth": ground_truth_sql,
            "test_cases": list(test_cases),
        },
        "extra_info": {
            "instance_id": task.task_id,
            "query": task.question,
            "schema": schema,
            "index": index,
            "split": split,
            "db_id": task.db_id,
            "db_version": task.db_version,
            "metric_version": task.metric_version,
            "schema_diff": schema_diff.to_observation(),
            "need_tools_kwargs": True,
            "tools_kwargs": tools_kwargs,
        },
        "return_raw_chat": True,
        "agent_name": "tool_agent_with_db_cleanup",
    }
