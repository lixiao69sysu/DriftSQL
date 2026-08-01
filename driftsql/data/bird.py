"""Audits for the open BIRD training and evaluation datasets."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any


def _read_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array: {path}")
        return data
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _inspect_database(path: Path, quick_check: bool) -> dict[str, Any]:
    if not path.is_file():
        return {"present": False, "healthy": False, "table_count": 0, "detail": "missing"}

    try:
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
            table_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM sqlite_master "
                    "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
                ).fetchone()[0]
            )
            detail = str(connection.execute("PRAGMA quick_check").fetchone()[0]) if quick_check else "opened"
        return {
            "present": True,
            "healthy": detail in {"ok", "opened"} and table_count > 0,
            "table_count": table_count,
            "detail": detail,
        }
    except sqlite3.Error as error:
        return {
            "present": True,
            "healthy": False,
            "table_count": 0,
            "detail": str(error),
        }


def _audit_records(
    *,
    name: str,
    task_file: Path,
    required_fields: tuple[str, ...],
    nonempty_fields: tuple[str, ...],
    database_path: Callable[[str], Path],
    quick_check: bool,
) -> dict[str, Any]:
    rows = _read_records(task_file)
    malformed_rows: list[dict[str, Any]] = []
    database_ids: set[str] = set()

    for index, row in enumerate(rows):
        missing = sorted(set(required_fields) - set(row))
        empty = sorted(field for field in nonempty_fields if not row.get(field))
        if missing or empty:
            malformed_rows.append({"index": index, "missing": missing, "empty": empty})
            continue
        database_ids.add(str(row["db_id"]))

    database_checks = {
        db_id: _inspect_database(database_path(db_id), quick_check)
        for db_id in sorted(database_ids)
    }
    missing_databases = sorted(
        db_id for db_id, state in database_checks.items() if not state["present"]
    )
    invalid_databases = sorted(
        db_id
        for db_id, state in database_checks.items()
        if state["present"] and not state["healthy"]
    )
    return {
        "dataset": name,
        "task_file": str(task_file.resolve()),
        "rows": len(rows),
        "databases": len(database_ids),
        "ground_truth_rows": sum(
            bool(row.get("SQL") or row.get("sol_sql")) for row in rows
        ),
        "test_case_rows": sum(bool(row.get("test_cases")) for row in rows),
        "malformed_rows": malformed_rows,
        "missing_databases": missing_databases,
        "invalid_databases": invalid_databases,
        "database_checks": database_checks,
        "ready": not malformed_rows and not missing_databases and not invalid_databases,
    }


def audit_bird23_train(root: Path, *, quick_check: bool = False) -> dict[str, Any]:
    root = Path(root)
    database_root = root / "full" / "train" / "train_databases"
    return _audit_records(
        name="bird23_train_filtered",
        task_file=root / "data" / "train-00000-of-00001.jsonl",
        required_fields=("db_id", "question", "evidence", "SQL"),
        nonempty_fields=("db_id", "question", "SQL"),
        database_path=lambda db_id: database_root / db_id / f"{db_id}.sqlite",
        quick_check=quick_check,
    )


def audit_six_gym_sqlite(root: Path, *, quick_check: bool = False) -> dict[str, Any]:
    root = Path(root)
    database_root = root / "database"
    return _audit_records(
        name="six_gym_sqlite",
        task_file=root / "train.jsonl",
        required_fields=(
            "instance_id",
            "db_id",
            "query",
            "issue_sql",
            "sol_sql",
            "test_cases",
            "preprocess_sql",
            "clean_up_sql",
            "category",
        ),
        nonempty_fields=(
            "instance_id",
            "db_id",
            "query",
            "issue_sql",
            "sol_sql",
            "test_cases",
        ),
        database_path=lambda db_id: database_root / db_id / f"{db_id}_template.sqlite",
        quick_check=quick_check,
    )


def audit_bird_mini_dev(root: Path, *, quick_check: bool = False) -> dict[str, Any]:
    root = Path(root)
    database_root = root / "full" / "dev_20240627" / "dev_databases"
    return _audit_records(
        name="bird_mini_dev_sqlite",
        task_file=root / "data" / "mini_dev_sqlite-00000-of-00001.json",
        required_fields=("question_id", "db_id", "question", "evidence", "SQL", "difficulty"),
        nonempty_fields=("db_id", "question", "SQL", "difficulty"),
        database_path=lambda db_id: database_root / db_id / f"{db_id}.sqlite",
        quick_check=quick_check,
    )
