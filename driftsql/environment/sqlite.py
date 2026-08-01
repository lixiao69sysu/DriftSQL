"""A deterministic, file-backed SQLite environment for drift episodes."""

from __future__ import annotations

import re
import shutil
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

_SAFE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class QueryResult:
    columns: tuple[str, ...]
    rows: tuple[tuple[object, ...], ...]


class VersionedSQLite:
    def __init__(self, root: Path, db_id: str) -> None:
        if not _SAFE_NAME.fullmatch(db_id):
            raise ValueError(f"Unsafe database id: {db_id!r}")
        self.root = Path(root)
        self.db_id = db_id
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, version: str) -> Path:
        if not _SAFE_NAME.fullmatch(version):
            raise ValueError(f"Unsafe database version: {version!r}")
        return self.root / f"{self.db_id}__{version}.sqlite"

    def create(self, version: str, ddl: Sequence[str], seed_sql: Sequence[str]) -> Path:
        path = self.path_for(version)
        if path.exists():
            raise FileExistsError(f"Database version already exists: {path}")
        with sqlite3.connect(str(path)) as connection:
            for statement in ddl:
                connection.execute(statement)
            for statement in seed_sql:
                connection.execute(statement)
            connection.commit()
        return path

    def clone(self, from_version: str, to_version: str) -> Path:
        source = self.path_for(from_version)
        target = self.path_for(to_version)
        if not source.exists():
            raise FileNotFoundError(source)
        if target.exists():
            raise FileExistsError(target)
        shutil.copy2(str(source), str(target))
        return target

    def connect(self, version: str) -> sqlite3.Connection:
        path = self.path_for(version)
        if not path.exists():
            raise FileNotFoundError(path)
        return sqlite3.connect(str(path))

    def execute_read_only(self, version: str, sql: str, parameters: Iterable[object] | None = None) -> QueryResult:
        first_token = sql.lstrip().split(None, 1)[0].upper() if sql.strip() else ""
        if first_token not in {"SELECT", "WITH", "PRAGMA"}:
            raise PermissionError("Only read-only SQL is allowed in this environment")
        with self.connect(version) as connection:
            cursor = connection.execute(sql, tuple(parameters or ()))
            columns = tuple(item[0] for item in (cursor.description or ()))
            rows = tuple(tuple(row) for row in cursor.fetchall())
        return QueryResult(columns=columns, rows=rows)

    def schema(self, version: str) -> list[dict]:
        with self.connect(version) as connection:
            tables = connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
            result = []
            for (table_name,) in tables:
                escaped = table_name.replace('"', '""')
                columns = connection.execute(f'PRAGMA table_info("{escaped}")').fetchall()
                result.append(
                    {
                        "table": table_name,
                        "columns": [
                            {
                                "name": row[1],
                                "type": row[2],
                                "not_null": bool(row[3]),
                                "primary_key": bool(row[5]),
                            }
                            for row in columns
                        ],
                    }
                )
        return result
