#!/usr/bin/env python3
"""Create executable two-turn trajectories for BIRD-RL Stage 2 smoke SFT.

The resulting JSONL files intentionally use BIRD-RL's own trajectory schema;
``bird_rl.data.prepare_multi_turn_sft_data`` performs the final conversion.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def as_sql_list(value: str | list[str]) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [sql for sql in value if sql and sql.strip()]


def execute_issue(row: dict, database_root: Path) -> tuple[bool, str]:
    db_id = row["db_id"]
    template = database_root / db_id / f"{db_id}_template.sqlite"
    source = sqlite3.connect(f"file:{template}?mode=ro", uri=True)
    working = sqlite3.connect(":memory:")
    try:
        source.backup(working)
        for sql in as_sql_list(row.get("preprocess_sql", [])):
            working.execute(sql)
        issue_sql = as_sql_list(row["issue_sql"])[0]
        cursor = working.execute(issue_sql)
        rows = cursor.fetchmany(5) if cursor.description else []
        columns = [description[0] for description in cursor.description] if cursor.description else []
        result = {
            "status": "success",
            "columns": columns,
            "rows": [list(values) for values in rows],
            "truncated_to": 5,
        }
        return True, json.dumps(result, ensure_ascii=False)
    except sqlite3.Error as exc:
        return False, json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False)
    finally:
        working.close()
        source.close()


def build_rows(data_path: Path, database_root: Path) -> tuple[list[dict], list[dict]]:
    statuses: list[dict] = []
    trajectories: list[dict] = []
    for index, row in enumerate(load_jsonl(data_path)):
        issue_sql = as_sql_list(row["issue_sql"])[0]
        solution_sql = as_sql_list(row["sol_sql"])
        ok, observation = execute_issue(row, database_root)
        status_text = "returned rows" if ok else "raised an execution error"
        trajectory = [
            {
                "thought": "<think>I will run the problematic query first to observe its current behavior.</think>",
                "action": "<tool_call>"
                + json.dumps({"name": "execute_sql", "arguments": {"sql": issue_sql}}, ensure_ascii=False)
                + "</tool_call>",
                "observation": observation,
                "end_flag": False,
            },
            {
                "thought": (
                    "<think>The problematic query " + status_text
                    + "; I will submit the corrected SQL that satisfies the requested behavior.</think>"
                ),
                "action": "<tool_call>"
                + json.dumps(
                    {"name": "submit_solution", "arguments": {"sql_list": solution_sql}},
                    ensure_ascii=False,
                )
                + "</tool_call>",
                "observation": "",
                "end_flag": True,
            },
        ]
        statuses.append({"instance_id": row["instance_id"], "status": "success"})
        trajectories.append(
            {
                "idx": index,
                "instance_idx": index,
                "instance_id": row["instance_id"],
                "db_id": row["db_id"],
                "trajectory": trajectory,
            }
        )
    return statuses, trajectories


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--database-root",
        type=Path,
        default=PROJECT_ROOT / "data/raw/six-gym-sqlite/database",
    )
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument("--trajectory-output", type=Path, required=True)
    args = parser.parse_args()

    statuses, trajectories = build_rows(args.data, args.database_root)
    write_jsonl(args.status_output, statuses)
    write_jsonl(args.trajectory_output, trajectories)
    print(
        json.dumps(
            {
                "examples": len(trajectories),
                "status_output": str(args.status_output),
                "trajectory_output": str(args.trajectory_output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
