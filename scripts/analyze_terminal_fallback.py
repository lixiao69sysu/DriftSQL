#!/usr/bin/env python3
"""Estimate the value of submitting the final executed SQL at a turn limit.

This is an analysis-only oracle: it never changes evaluation results.  It
replays the last SQL from each turn-limited trajectory against the task's
materialized drift database and compares its result fingerprint with the
locked answer.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from driftsql.drift import fingerprint_query, materialize_schema_diff


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def matches_expected(database: Path, sql: str, expected: dict[str, Any]) -> bool:
    try:
        actual = fingerprint_query(database, sql, timeout_seconds=30)
    except Exception:
        return False
    return (
        actual.row_count == int(expected.get("row_count", -1))
        and actual.value_hash == str(expected.get("value_hash", ""))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-rows", type=Path)
    parser.add_argument("--output-summary", type=Path)
    args = parser.parse_args()

    records = {
        str(row["extra_info"]["instance_id"]): row["extra_info"]
        for row in load_jsonl(args.data)
    }
    input_rows = load_jsonl(args.rows)
    output_rows = copy.deepcopy(input_rows)
    turn_limited = [
        row
        for row in output_rows
        if row.get("termination_reason") == "turn_limit"
    ]
    with_execution = 0
    last_execution_correct = 0
    any_execution_correct = 0
    fallback_applied = 0
    examples: list[dict[str, Any]] = []

    temporary_root = Path(os.environ.get("DRIFTSQL_TMPDIR", PROJECT_ROOT / "tmp"))
    temporary_root.mkdir(parents=True, exist_ok=True)
    for row in turn_limited:
        executions = [
            (
                str(event.get("arguments", {}).get("sql", "")).strip(),
                bool(event.get("metrics", {}).get("execution_success")),
            )
            for event in row.get("trajectory", [])
            if event.get("tool_name") == "execute_sql"
            and str(event.get("arguments", {}).get("sql", "")).strip()
        ]
        sqls = [sql for sql, succeeded in executions if succeeded]
        if not sqls:
            continue
        with_execution += 1
        extra = records[str(row["instance_id"])]
        with tempfile.TemporaryDirectory(
            prefix="driftsql-fallback-analysis-",
            dir=temporary_root,
            ignore_cleanup_errors=True,
        ) as directory:
            database = Path(directory) / f"{extra['db_id']}__v2.sqlite"
            materialize_schema_diff(
                Path(extra["source_db"]), database, extra["schema_diff"]
            )
            correctness = []
            executable = []
            for sql in sqls:
                try:
                    actual = fingerprint_query(database, sql, timeout_seconds=30)
                    executable.append(True)
                    correctness.append(
                        actual.row_count
                        == int(extra["result_fingerprint"].get("row_count", -1))
                        and actual.value_hash
                        == str(extra["result_fingerprint"].get("value_hash", ""))
                    )
                except Exception:
                    executable.append(False)
                    correctness.append(False)
        last_execution_correct += int(correctness[-1])
        any_execution_correct += int(any(correctness))
        fallback_applied += 1
        row["termination_reason"] = "fallback_submitted"
        row["final_sql"] = sqls[-1]
        row["executable"] = executable[-1]
        row["task_success"] = correctness[-1]
        row["error"] = "" if executable[-1] else "offline_fallback_replay_failed"
        if correctness[-1] and len(examples) < 5:
            examples.append(
                {
                    "instance_id": row["instance_id"],
                    "tool_calls": row["usage"]["tool_calls"],
                    "last_sql": sqls[-1],
                }
            )

    result = {
        "rows": len(input_rows),
        "turn_limit": len(turn_limited),
        "turn_limit_with_execution": with_execution,
        "fallback_applied": fallback_applied,
        "remaining_turn_limit": len(turn_limited) - fallback_applied,
        "last_execution_correct": last_execution_correct,
        "any_execution_correct": any_execution_correct,
        "original_success": sum(bool(row.get("task_success")) for row in input_rows),
        "replayed_success": sum(bool(row.get("task_success")) for row in output_rows),
        "examples": examples,
    }
    if args.output_rows:
        write_jsonl(args.output_rows, output_rows)
    if args.output_summary:
        args.output_summary.parent.mkdir(parents=True, exist_ok=True)
        args.output_summary.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
