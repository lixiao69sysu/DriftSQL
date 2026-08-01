"""Integrity and readiness checks for the public Mini-Interact release."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = {
    "instance_id",
    "selected_database",
    "amb_user_query",
    "sol_sql",
    "test_cases",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quick_check(database: Path) -> tuple[bool, int, str]:
    uri = database.resolve().as_uri() + "?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True) as connection:
            integrity = str(connection.execute("PRAGMA quick_check").fetchone()[0])
            table_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM sqlite_master "
                    "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
                ).fetchone()[0]
            )
        return integrity == "ok", table_count, integrity
    except sqlite3.Error as error:
        return False, 0, str(error)


def audit_mini_interact(root: Path) -> dict[str, Any]:
    """Audit public files and distinguish evaluation data from RL-ready data."""

    root = Path(root).resolve()
    task_file = root / "mini_interact.jsonl"
    if not task_file.is_file():
        raise FileNotFoundError(task_file)

    rows: list[dict[str, Any]] = []
    malformed_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    duplicate_ids: list[str] = []

    with task_file.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                malformed_rows.append({"line": line_number, "error": str(error)})
                continue
            missing = sorted(REQUIRED_FIELDS - set(row))
            if missing:
                malformed_rows.append({"line": line_number, "missing_fields": missing})
                continue
            instance_id = str(row["instance_id"])
            if instance_id in seen_ids:
                duplicate_ids.append(instance_id)
            seen_ids.add(instance_id)
            rows.append(row)

    selected_databases = sorted({str(row["selected_database"]) for row in rows})
    missing_assets: list[str] = []
    database_checks: dict[str, dict[str, Any]] = {}

    for db_id in selected_databases:
        db_dir = root / db_id
        expected = {
            "database": db_dir / f"{db_id}.sqlite",
            "template_database": db_dir / f"{db_id}_template.sqlite",
            "schema": db_dir / f"{db_id}_schema.txt",
            "knowledge_base": db_dir / f"{db_id}_kb.jsonl",
            "column_meanings": db_dir / f"{db_id}_column_meaning_base.json",
        }
        for asset_name, asset_path in expected.items():
            if not asset_path.is_file():
                missing_assets.append(f"{db_id}:{asset_name}")

        database = expected["database"]
        if database.is_file():
            healthy, table_count, detail = _quick_check(database)
            database_checks[db_id] = {
                "healthy": healthy,
                "table_count": table_count,
                "detail": detail,
            }

    rows_with_sql = sum(bool(row.get("sol_sql")) for row in rows)
    rows_with_test_cases = sum(bool(row.get("test_cases")) for row in rows)
    invalid_databases = sorted(
        db_id for db_id, state in database_checks.items() if not state["healthy"]
    )
    files_ok = not malformed_rows and not duplicate_ids and not missing_assets and not invalid_databases
    ground_truth_complete = bool(rows) and rows_with_sql == len(rows) and rows_with_test_cases == len(rows)

    return {
        "root": str(root),
        "task_file_sha256": _sha256(task_file),
        "rows": len(rows),
        "selected_databases": len(selected_databases),
        "rows_with_solution_sql": rows_with_sql,
        "rows_with_test_cases": rows_with_test_cases,
        "ground_truth_complete": ground_truth_complete,
        "ready_for_rl": files_ok and ground_truth_complete,
        "malformed_rows": malformed_rows,
        "duplicate_ids": duplicate_ids,
        "missing_assets": missing_assets,
        "invalid_databases": invalid_databases,
        "database_checks": database_checks,
    }
