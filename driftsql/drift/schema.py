"""Schema drift mutations with executable migrations and auditable diffs."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

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
    operations: tuple[dict[str, str], ...]

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


def rewrite_sql_identifier(sql: str, old_name: str, new_name: str) -> str:
    """Rewrite one SQL identifier without touching string literals.

    This small lexer is intentionally conservative and supports the MVP. The
    production data factory will use SQLGlot AST transforms and validate every
    rewritten query through execution.
    """

    _validate_identifier(old_name)
    _validate_identifier(new_name)

    output = []
    index = 0
    length = len(sql)

    while index < length:
        char = sql[index]

        if char == "'":
            start = index
            index += 1
            while index < length:
                if sql[index] == "'" and index + 1 < length and sql[index + 1] == "'":
                    index += 2
                    continue
                if sql[index] == "'":
                    index += 1
                    break
                index += 1
            output.append(sql[start:index])
            continue

        if char in ('"', "`", "["):
            closing = "]" if char == "[" else char
            start = index
            index += 1
            while index < length and sql[index] != closing:
                index += 1
            index = min(index + 1, length)
            quoted = sql[start:index]
            content = quoted[1:-1]
            if content.lower() == old_name.lower():
                output.append(char + new_name + closing)
            else:
                output.append(quoted)
            continue

        if char.isalpha() or char == "_":
            start = index
            index += 1
            while index < length and (sql[index].isalnum() or sql[index] == "_"):
                index += 1
            token = sql[start:index]
            output.append(new_name if token.lower() == old_name.lower() else token)
            continue

        output.append(char)
        index += 1

    return "".join(output)
