#!/usr/bin/env python3
"""Build focused recovery supervision for measured Stage 6 loop failures."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from verl.utils.py_functional import convert_nested_value_to_list_recursive

from driftsql.data.tool_sft import expand_next_action_messages, use_plain_json_for_last_action


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAJECTORIES = PROJECT_ROOT / "data/processed/stage6_repair_sft"
DEFAULT_STATE = PROJECT_ROOT / "data/processed/stage6_ablation/b1"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/processed/stage6_recovery_next_action"
RECOVERY_TOOLS = ("get_schema_version", "inspect_schema_diff", "get_schema")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def action_name(message: dict[str, Any]) -> str:
    calls = message.get("tool_calls", [])
    if len(calls) != 1:
        raise ValueError("Expected exactly one tool call")
    return str(calls[0]["function"]["name"])


def recovery_observation(tool: str, state: dict[str, Any]) -> str:
    if tool == "get_schema_version":
        payload = {
            "db_id": state["db_id"],
            "db_version": state.get("db_version"),
            "metric_version": state.get("metric_version"),
        }
    elif tool == "inspect_schema_diff":
        payload = state["schema_diff"]
    elif tool == "get_schema":
        schema = str(state.get("schema", ""))
        payload = {"query": "", "schema": schema[:3500], "truncated": len(schema) > 3500}
    else:
        raise ValueError(tool)
    return json.dumps(payload, ensure_ascii=False)


def injected_prefix(
    prefix: list[dict[str, Any]],
    *,
    tool: str,
    observation: str,
) -> list[dict[str, Any]]:
    result = copy.deepcopy(prefix[:-1])
    arguments = {"query": ""} if tool == "get_schema" else {}
    result.extend(
        [
            {
                "role": "assistant",
                "content": "<think>I will verify this retrieval once more before acting.</think>",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": tool,
                            "arguments": json.dumps(arguments, ensure_ascii=False),
                        },
                    }
                ],
            },
            {"role": "tool", "content": observation},
            copy.deepcopy(prefix[-1]),
        ]
    )
    return result


def payload(
    messages: list[dict[str, Any]],
    row: pd.Series,
    *,
    recovery_from: str,
) -> dict[str, Any]:
    converted = use_plain_json_for_last_action(messages)
    return {
        "messages": converted,
        "tools": str(row["tools"]),
        "enable_thinking": bool(row["enable_thinking"]),
        "target_action": action_name(messages[-1]),
        "task_id": str(row["task_id"]),
        "db_id": str(row["db_id"]),
        "drift_type": str(row["drift_type"]),
        "interaction_profile": str(row["interaction_profile"]),
        "difficulty": str(row["difficulty"]),
        "failure_mode": str(row["failure_mode"]),
        "recovery_from": recovery_from,
    }


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-dir", type=Path, default=DEFAULT_TRAJECTORIES)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    frame = pd.read_parquet(args.trajectory_dir / "train.parquet")
    state_rows = load_jsonl(args.state_dir / "train_agent_eval.jsonl")
    state_by_id = {
        str(row["extra_info"]["instance_id"]): row["extra_info"]["tools_kwargs"]["execute_sql"]["create_kwargs"]
        for row in state_rows
    }
    output: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        task_id = str(row["task_id"])
        state = state_by_id[task_id]
        messages = convert_nested_value_to_list_recursive(row["messages"])
        for prefix in expand_next_action_messages(messages):
            target = action_name(prefix[-1])
            prior_actions = [
                action_name(message)
                for message in prefix[:-1]
                if message.get("role") == "assistant"
            ]
            repaired_execute = target == "execute_sql" and "inspect_schema_diff" in prior_actions
            governed_next = target == "get_knowledge_definition"
            submit_next = target == "submit_solution"
            if not (repaired_execute or governed_next or submit_next):
                continue

            base_repeats = 2 if target == "submit_solution" else 1
            hard_repeats = 2 if str(row["drift_type"]) == "add_column" else (
                1 if str(row["drift_type"]) == "compound" else 0
            )
            for _ in range(base_repeats + hard_repeats):
                output.append(payload(prefix, row, recovery_from="canonical"))
            for tool in RECOVERY_TOOLS:
                recovery = injected_prefix(
                    prefix,
                    tool=tool,
                    observation=recovery_observation(tool, state),
                )
                repeats = 1 + hard_repeats
                for _ in range(repeats):
                    output.append(payload(recovery, row, recovery_from=tool))

    write_parquet(args.output_dir / "train.parquet", output)
    # Keep validation independent of the synthetic recovery contexts.  It is
    # a small health signal only; policy selection remains Tune109 agent eval.
    tune = pq.read_table(
        PROJECT_ROOT / "data/processed/stage6_repair_next_action_v2/tune.parquet"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(tune, args.output_dir / "tune.parquet", compression="zstd")
    summary = {
        "name": "driftsql_stage6_failure_recovery_sft_v1",
        "source_trajectories": len(frame),
        "train_examples": len(output),
        "tune_examples": tune.num_rows,
        "target_actions": dict(sorted(Counter(row["target_action"] for row in output).items())),
        "recovery_contexts": dict(sorted(Counter(row["recovery_from"] for row in output).items())),
        "drift_types": dict(sorted(Counter(row["drift_type"] for row in output).items())),
        "sealed_gate_read": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
