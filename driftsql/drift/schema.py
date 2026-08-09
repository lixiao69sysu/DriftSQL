"""Schema drift mutations with executable migrations and auditable diffs."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import Any

from driftsql.sql_rewrite import rewrite_sql_identifier

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_identifier(value: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"Unsafe SQL identifier: {value!r}")
    return value


def _quote_identifier(value: str) -> str:
    return f'"{_validate_identifier(value)}"'


@dataclass(frozen=True)
class SchemaDiff:
    db_id: str
    from_version: str
    to_version: str
    operations: tuple[dict[str, Any], ...]

    def to_observation(self) -> dict[str, object]:
        return {
            "db_id": self.db_id,
            "from_version": self.from_version,
            "to_version": self.to_version,
            "operations": list(self.operations),
        }


@dataclass(frozen=True)
class ColumnRename:
    table: str
    old_name: str
    new_name: str

    def __post_init__(self) -> None:
        _validate_identifier(self.table)
        _validate_identifier(self.old_name)
        _validate_identifier(self.new_name)

    def apply(self, connection: sqlite3.Connection) -> None:
        statement = (
            f"ALTER TABLE {_quote_identifier(self.table)} "
            f"RENAME COLUMN {_quote_identifier(self.old_name)} "
            f"TO {_quote_identifier(self.new_name)}"
        )
        connection.execute(statement)
        connection.commit()

    def as_operation(self) -> dict[str, str]:
        return {
            "type": "rename_column",
            "table": self.table,
            "old_name": self.old_name,
            "new_name": self.new_name,
        }

    def rewrite(self, sql: str) -> str:
        return rewrite_sql_identifier(sql, self.old_name, self.new_name)
