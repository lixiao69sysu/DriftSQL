"""One execution-accuracy evaluator shared by all Stage-1 BIRD baselines."""

from __future__ import annotations

import csv
import json
import re
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from driftsql.tool_calls import find_tool_calls


@dataclass(frozen=True)
class BirdEvalBudget:
    max_model_calls: int = 5
    max_tool_calls: int = 5
    max_sql_executions: int = 3
    max_new_tokens: int = 3072
    max_new_tokens_per_call: int = 1024
    max_prompt_tokens: int = 16384
    max_total_tokens: int = 32768
    sql_timeout_seconds: float = 30.0
    max_result_rows_for_agent: int = 100

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


_SQL_FENCE = re.compile(r"```sql\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_SOLUTION_TAG = re.compile(r"<solution>\s*(.*?)\s*</solution>", re.IGNORECASE | re.DOTALL)
_LEADING_SQL = re.compile(r"\b(SELECT|WITH)\b", re.IGNORECASE)


def extract_candidate_sql(text: str, *, allow_execute_fallback: bool = True) -> str:
    """Extract final SQL from tool calls, solution tags, fences, or raw SQL."""

    calls = find_tool_calls(text or "")
    preferred = [call for call in calls if call.name == "submit_solution"]
    if allow_execute_fallback and not preferred:
        preferred = [call for call in calls if call.name == "execute_sql"]
    for call in reversed(preferred):
        sql = call.arguments.get("sql")
        if isinstance(sql, str) and sql.strip():
            return sql.strip()
        sql_list = call.arguments.get("sql_list")
        if isinstance(sql_list, list) and sql_list and str(sql_list[0]).strip():
            return str(sql_list[0]).strip()

    for pattern in (_SOLUTION_TAG, _SQL_FENCE):
        matches = pattern.findall(text or "")
        if matches:
            return matches[-1].strip()

    stripped = (text or "").strip()
    match = _LEADING_SQL.search(stripped)
    if match:
        candidate = stripped[match.start() :]
        candidate = re.split(r"\n```|</?(?:final|answer)>", candidate, maxsplit=1)[0]
        return candidate.strip().strip("`")
    return ""


def get_schema_from_db(db_path: Path, *, sample_rows: int = 3) -> str:
    """Return the BIRD-RL schema representation: DDL plus small sample rows."""

    parts: list[str] = []
    with sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        definitions = connection.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        for table_name, ddl in definitions:
            if not ddl:
                continue
            parts.append(str(ddl).rstrip(";") + ";")
            escaped = str(table_name).replace('"', '""')
            columns = [row[1] for row in connection.execute(f'PRAGMA table_info("{escaped}")')]
            rows = connection.execute(f'SELECT * FROM "{escaped}" LIMIT ?', (sample_rows,)).fetchall()
            if rows:
                parts.append(f"First {len(rows)} rows:")
                parts.append("  " + "\t".join(map(str, columns)))
                parts.extend(
                    "  " + "\t".join("NULL" if value is None else str(value) for value in row)
                    for row in rows
                )
            parts.append("")
    return "\n".join(parts).strip()


def build_column_meanings(database_root: Path) -> dict[str, str]:
    """Build BIRD-RL's db|table|column lookup from official description CSVs."""

    meanings: dict[str, str] = {}
    for csv_path in sorted(database_root.glob("*/database_description/*.csv")):
        db_id = csv_path.parents[1].name
        table = csv_path.stem
        try:
            with csv_path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
                for row in csv.DictReader(handle):
                    column = str(row.get("original_column_name", "")).strip()
                    description = str(row.get("column_description", "")).strip()
                    value_description = str(row.get("value_description", "")).strip()
                    if column and (description or value_description):
                        meanings[f"{db_id}|{table}|{column}"] = " | ".join(
                            value for value in (description, value_description) if value
                        )
        except (OSError, csv.Error):
            continue
    return meanings


def column_descriptions(db_id: str, meanings: dict[str, str]) -> str:
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    prefix = f"{db_id}|"
    for key, description in meanings.items():
        if key.startswith(prefix):
            _, table, column = key.split("|", 2)
            grouped[table].append((column, description))
    if not grouped:
        return "(No column descriptions available)"
    lines: list[str] = []
    for table in sorted(grouped):
        lines.append(f"Table: {table}")
        lines.extend(f"  - {column}: {description}" for column, description in grouped[table])
    return "\n".join(lines)


def _read_only_result(
    db_path: Path,
    sql: str,
    *,
    timeout_seconds: float,
    max_rows: int | None = None,
) -> tuple[tuple[str, ...], list[tuple[Any, ...]]]:
    deadline = time.monotonic() + timeout_seconds
    with sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        connection.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 1000)
        cursor = connection.execute(sql)
        columns = tuple(item[0] for item in (cursor.description or ()))
        rows = cursor.fetchmany(max_rows + 1) if max_rows is not None else cursor.fetchall()
        if max_rows is not None:
            rows = rows[:max_rows]
        return columns, [tuple(row) for row in rows]


def execute_for_agent(
    db_path: Path,
    sql: str,
    *,
    timeout_seconds: float,
    max_rows: int,
) -> dict[str, Any]:
    try:
        columns, rows = _read_only_result(
            db_path,
            sql,
            timeout_seconds=timeout_seconds,
            max_rows=max_rows,
        )
        return {"success": True, "columns": columns, "rows": rows, "truncated_at": max_rows}
    except (sqlite3.Error, ValueError) as error:
        return {"success": False, "error": str(error), "columns": (), "rows": ()}


def _normalize(rows: Iterable[tuple[Any, ...]]) -> set[tuple[Any, ...]]:
    normalized: set[tuple[Any, ...]] = set()
    for row in rows:
        normalized.add(
            tuple(
                round(value, 10)
                if isinstance(value, float)
                else value.strip().casefold()
                if isinstance(value, str)
                else value
                for value in row
            )
        )
    return normalized


def evaluate_prediction(
    *,
    predicted_sql: str,
    gold_sql: str,
    db_path: Path,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Apply the same set-normalized EX semantics as BIRD-RL's evaluator."""

    if not predicted_sql.strip():
        return {"correct": False, "pred_executable": False, "error": "missing_sql"}
    try:
        _, predicted_rows = _read_only_result(
            db_path, predicted_sql, timeout_seconds=timeout_seconds
        )
    except (sqlite3.Error, ValueError) as error:
        return {"correct": False, "pred_executable": False, "error": str(error)}
    try:
        _, gold_rows = _read_only_result(db_path, gold_sql, timeout_seconds=timeout_seconds)
    except (sqlite3.Error, ValueError) as error:
        return {
            "correct": False,
            "pred_executable": True,
            "error": f"gold_execution_error: {error}",
        }
    return {
        "correct": _normalize(predicted_rows) == _normalize(gold_rows),
        "pred_executable": True,
        "error": "",
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    def metrics(items: list[dict[str, Any]]) -> dict[str, float | int]:
        total = len(items)
        correct = sum(bool(item["correct"]) for item in items)
        executable = sum(bool(item["pred_executable"]) for item in items)
        return {
            "total": total,
            "correct": correct,
            "execution_accuracy": correct / total if total else 0.0,
            "executable": executable,
            "executable_rate": executable / total if total else 0.0,
        }

    by_difficulty: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in results:
        by_difficulty[str(item.get("difficulty", "unknown"))].append(item)
    return {
        "overall": metrics(results),
        "by_difficulty": {key: metrics(value) for key, value in sorted(by_difficulty.items())},
        "termination_reasons": dict(Counter(str(item.get("termination_reason", "unknown")) for item in results)),
        "budget_violations": sum(not bool(item.get("within_budget", True)) for item in results),
    }


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
