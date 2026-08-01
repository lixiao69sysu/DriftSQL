"""Execution-verified schema-drift example generation.

The factory stores compact mutation manifests and materializes changed SQLite
databases only while validating or running an episode. This avoids creating a
full database copy for every training row.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sqlglot import exp, parse_one

from driftsql.planning import analyze_wildcard_projection, plan_projection_contract

from .schema import SchemaDiff

_TOKEN_SYNONYMS = {
    "address": ("location",),
    "amount": ("value", "total"),
    "city": ("municipality",),
    "code": ("key",),
    "count": ("quantity", "total"),
    "country": ("nation",),
    "customer": ("client",),
    "date": ("timestamp", "day"),
    "description": ("details",),
    "director": ("filmmaker",),
    "employee": ("staff",),
    "id": ("key", "identifier"),
    "movie": ("film",),
    "name": ("label", "title"),
    "number": ("num",),
    "order": ("purchase",),
    "phone": ("telephone",),
    "price": ("cost",),
    "revenue": ("sales",),
    "state": ("region",),
    "status": ("state",),
    "user": ("account",),
    "year": ("yr",),
}
_FALLBACK_PREFIXES = ("canonical", "current", "reported", "source")


@dataclass(frozen=True)
class QueryFingerprint:
    row_count: int
    value_hash: str


@dataclass(frozen=True)
class DriftExample:
    task_id: str
    source: str
    source_index: int
    db_id: str
    question: str
    evidence: str
    source_db: str
    stale_sql: str
    repaired_sql: str
    schema_diff: SchemaDiff
    stale_error: str
    result_fingerprint: QueryFingerprint
    oracle_steps: tuple[dict[str, Any], ...]
    wildcard_profile: str = ""
    added_column_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Backward-compatible public name used by the first factory stage.
ColumnRenameExample = DriftExample


def _quote_identifier(identifier: str) -> str:
    if "\x00" in identifier:
        raise ValueError("SQLite identifiers cannot contain NUL")
    return f'"{identifier.replace(chr(34), chr(34) * 2)}"'


def _schema_columns(database: Path) -> dict[str, set[str]]:
    uri = database.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        tables = connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        return {
            str(table): {
                str(row[1])
                for row in connection.execute(
                    f"PRAGMA table_info({_quote_identifier(str(table))})"
                )
            }
            for (table,) in tables
        }


def _column_type(database: Path, table: str, column: str) -> str:
    uri = database.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        rows = connection.execute(
            f"PRAGMA table_info({_quote_identifier(table)})"
        ).fetchall()
    for row in rows:
        if str(row[1]).casefold() == column.casefold():
            return str(row[2]).strip() or "TEXT"
    raise ValueError(f"Column {table}.{column} does not exist")


def _column_order(database: Path, table: str) -> list[str]:
    uri = database.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        return [
            str(row[1])
            for row in connection.execute(
                f"PRAGMA table_info({_quote_identifier(table)})"
            )
        ]


def _case_lookup(values: set[str]) -> dict[str, str]:
    return {value.casefold(): value for value in values}


def _table_aliases(tree: exp.Expression, schemas: dict[str, set[str]]) -> dict[str, str]:
    table_lookup = _case_lookup(set(schemas))
    aliases: dict[str, str] = {}
    for table in tree.find_all(exp.Table):
        physical = table_lookup.get(table.name.casefold())
        if not physical:
            continue
        aliases[physical.casefold()] = physical
        if table.alias:
            aliases[table.alias.casefold()] = physical
    return aliases


def _resolve_column(
    column: exp.Column,
    schemas: dict[str, set[str]],
    aliases: dict[str, str],
) -> tuple[str, str] | None:
    name = column.name
    if not name:
        return None
    if column.table:
        table = aliases.get(column.table.casefold())
        if not table:
            return None
        actual = _case_lookup(schemas[table]).get(name.casefold())
        return (table, actual) if actual else None

    matches = [
        (table, actual)
        for table, columns in schemas.items()
        if (actual := _case_lookup(columns).get(name.casefold()))
    ]
    return matches[0] if len(matches) == 1 else None


def _candidate_columns(
    tree: exp.Expression,
    schemas: dict[str, set[str]],
    aliases: dict[str, str],
) -> list[tuple[str, str]]:
    candidates = {
        resolved
        for column in tree.find_all(exp.Column)
        if (resolved := _resolve_column(column, schemas, aliases))
    }
    return sorted(candidates, key=lambda item: (item[0].casefold(), item[1].casefold()))


def _match_case(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source.istitle():
        return replacement.title()
    return replacement


def _stable_choice(values: tuple[str, ...], key: str) -> str:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return values[int.from_bytes(digest[:4], "big") % len(values)]


def _new_column_name(table: str, old_name: str, existing: set[str]) -> str:
    existing_folded = {name.casefold() for name in existing}
    parts = re.split(r"([_\s]+)", old_name)
    replacement_index = next(
        (
            index
            for index, part in enumerate(parts)
            if part.casefold() in _TOKEN_SYNONYMS
        ),
        None,
    )
    if replacement_index is not None:
        token = parts[replacement_index]
        choices = _TOKEN_SYNONYMS[token.casefold()]
        replacement = _stable_choice(choices, f"{table}:{old_name}:{token}")
        parts[replacement_index] = _match_case(token, replacement)
        base = "".join(parts)
    else:
        prefix = _stable_choice(_FALLBACK_PREFIXES, f"{table}:{old_name}")
        separator = " " if " " in old_name and "_" not in old_name else "_"
        base = f"{prefix}{separator}{old_name}"

    candidate = base
    suffix = 2
    while candidate.casefold() in existing_folded:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def _rewrite_column(
    tree: exp.Expression,
    schemas: dict[str, set[str]],
    aliases: dict[str, str],
    table: str,
    old_name: str,
    new_name: str,
) -> str:
    rewritten = tree.copy()
    rewritten_aliases = _table_aliases(rewritten, schemas)
    for column in rewritten.find_all(exp.Column):
        if _resolve_column(column, schemas, rewritten_aliases) == (table, old_name):
            quoted = bool(column.this.args.get("quoted"))
            column.set("this", exp.to_identifier(new_name, quoted=quoted))
    return rewritten.sql(dialect="sqlite")


def _rewrite_table(
    tree: exp.Expression,
    old_name: str,
    new_name: str,
) -> str:
    rewritten = tree.copy()
    matched = False
    update_qualifier = False
    for table in rewritten.find_all(exp.Table):
        if table.name.casefold() != old_name.casefold():
            continue
        matched = True
        if not table.alias:
            update_qualifier = True
        quoted = bool(table.this.args.get("quoted")) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*",
            new_name,
        )
        table.set("this", exp.to_identifier(new_name, quoted=quoted))

    if update_qualifier:
        for column in rewritten.find_all(exp.Column):
            if column.table.casefold() == old_name.casefold():
                quoted = bool(column.args.get("table").args.get("quoted")) or not re.fullmatch(
                    r"[A-Za-z_][A-Za-z0-9_]*",
                    new_name,
                )
                column.set("table", exp.to_identifier(new_name, quoted=quoted))

    if not matched:
        raise ValueError(f"Table {old_name!r} is not referenced by the query")
    return rewritten.sql(dialect="sqlite")


def _expand_single_table_star(
    tree: exp.Expression,
    table: str,
    columns: list[str],
) -> str:
    rewritten = tree.copy()
    selects = list(rewritten.find_all(exp.Select))
    if len(selects) != 1:
        raise ValueError("Only a single SELECT scope is supported for star expansion")
    select = selects[0]
    expanded: list[exp.Expression] = []
    replaced = False
    for expression in select.expressions:
        is_plain_star = isinstance(expression, exp.Star)
        is_qualified_star = (
            isinstance(expression, exp.Column)
            and isinstance(expression.this, exp.Star)
            and expression.table.casefold() == table.casefold()
        )
        if not (is_plain_star or is_qualified_star):
            expanded.append(expression)
            continue
        replaced = True
        qualifier = table if is_qualified_star else None
        expanded.extend(
            exp.column(column, table=qualifier, quoted=not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", column))
            for column in columns
        )
    if not replaced:
        raise ValueError("Query does not contain a supported SELECT star")
    select.set("expressions", expanded)
    return rewritten.sql(dialect="sqlite")


def fingerprint_query(
    database: Path,
    sql: str,
    *,
    timeout_seconds: float = 30,
) -> QueryFingerprint:
    """Fingerprint an ordered query result without retaining all rows."""

    digest = hashlib.sha256()
    row_count = 0
    deadline = time.monotonic() + timeout_seconds
    uri = database.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.execute("PRAGMA query_only = ON")
        connection.set_progress_handler(
            lambda: 1 if time.monotonic() > deadline else 0,
            1_000,
        )
        cursor = connection.execute(sql)
        while rows := cursor.fetchmany(1_000):
            for row in rows:
                digest.update(repr(tuple(row)).encode("utf-8"))
                digest.update(b"\n")
            row_count += len(rows)
    return QueryFingerprint(row_count=row_count, value_hash=digest.hexdigest())


def _copy_database(source: Path, target: Path) -> None:
    """Prefer a copy-on-write reflink and fall back to a regular copy."""

    target.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["cp", "--reflink=always", "--sparse=always", str(source), str(target)],
        check=False,
        capture_output=True,
    )
    if result.returncode:
        shutil.copy2(source, target)


def materialize_column_rename(
    source: Path,
    target: Path,
    *,
    table: str,
    old_name: str,
    new_name: str,
) -> None:
    if target.exists():
        raise FileExistsError(target)
    _copy_database(source, target)
    with sqlite3.connect(target) as connection:
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute(
            f"ALTER TABLE {_quote_identifier(table)} "
            f"RENAME COLUMN {_quote_identifier(old_name)} "
            f"TO {_quote_identifier(new_name)}"
        )
        connection.commit()


def materialize_table_rename(
    source: Path,
    target: Path,
    *,
    old_name: str,
    new_name: str,
) -> None:
    if target.exists():
        raise FileExistsError(target)
    _copy_database(source, target)
    with sqlite3.connect(target) as connection:
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute(
            f"ALTER TABLE {_quote_identifier(old_name)} "
            f"RENAME TO {_quote_identifier(new_name)}"
        )
        connection.commit()


def materialize_column_replacement(
    source: Path,
    target: Path,
    *,
    table: str,
    old_name: str,
    new_name: str,
    declared_type: str,
) -> None:
    if target.exists():
        raise FileExistsError(target)
    _copy_database(source, target)
    with sqlite3.connect(target) as connection:
        connection.execute("PRAGMA journal_mode = DELETE")
        escaped_table = table.replace('"', '""')
        source_type = next(
            (
                str(row[2]).strip().casefold()
                for row in connection.execute(
                    f'PRAGMA table_info("{escaped_table}")'
                )
                if str(row[1]).casefold() == old_name.casefold()
            ),
            None,
        )
        # Replacement examples preserve the source values.  When the declared
        # type is unchanged, SQLite's metadata-only rename is equivalent for
        # all explicit-column queries and avoids rewriting multi-GB tables.
        # Keep the original add/copy/drop path for genuine type changes.
        if source_type == declared_type.strip().casefold():
            connection.execute(
                f"ALTER TABLE {_quote_identifier(table)} "
                f"RENAME COLUMN {_quote_identifier(old_name)} "
                f"TO {_quote_identifier(new_name)}"
            )
            connection.commit()
            return
        connection.execute(
            f"ALTER TABLE {_quote_identifier(table)} "
            f"ADD COLUMN {_quote_identifier(new_name)} {declared_type}"
        )
        connection.execute(
            f"UPDATE {_quote_identifier(table)} "
            f"SET {_quote_identifier(new_name)} = {_quote_identifier(old_name)}"
        )
        connection.execute(
            f"ALTER TABLE {_quote_identifier(table)} "
            f"DROP COLUMN {_quote_identifier(old_name)}"
        )
        connection.commit()


def materialize_column_addition(
    source: Path,
    target: Path,
    *,
    table: str,
    new_name: str,
    declared_type: str = "INTEGER",
    default_sql: str = "0",
) -> None:
    if target.exists():
        raise FileExistsError(target)
    _copy_database(source, target)
    with sqlite3.connect(target) as connection:
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute(
            f"ALTER TABLE {_quote_identifier(table)} "
            f"ADD COLUMN {_quote_identifier(new_name)} {declared_type} "
            f"NOT NULL DEFAULT {default_sql}"
        )
        connection.commit()


def materialize_schema_diff(
    source: Path,
    target: Path,
    schema_diff: SchemaDiff | dict[str, Any],
) -> None:
    """Materialize an ordered schema diff, including clean/no-op episodes."""

    if isinstance(schema_diff, SchemaDiff):
        operations = schema_diff.operations
    else:
        operations = tuple(schema_diff.get("operations", ()))
    source = Path(source).resolve()
    target = Path(target).resolve()
    if not operations:
        if target.exists():
            raise FileExistsError(target)
        _copy_database(source, target)
        return
    if len(operations) == 1:
        _materialize_schema_operation(source, target, operations[0])
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="driftsql-schema-chain-",
        dir=target.parent,
        ignore_cleanup_errors=True,
    ) as directory:
        current = source
        for index, operation in enumerate(operations):
            destination = (
                target
                if index == len(operations) - 1
                else Path(directory) / f"operation-{index}.sqlite"
            )
            _materialize_schema_operation(current, destination, operation)
            current = destination


def _materialize_schema_operation(
    source: Path,
    target: Path,
    operation: dict[str, Any],
) -> None:
    operation_type = operation.get("type")
    if operation_type == "rename_column":
        materialize_column_rename(
            source,
            target,
            table=str(operation["table"]),
            old_name=str(operation["old_name"]),
            new_name=str(operation["new_name"]),
        )
        return
    if operation_type == "rename_table":
        materialize_table_rename(
            source,
            target,
            old_name=str(operation["old_name"]),
            new_name=str(operation["new_name"]),
        )
        return
    if operation_type == "replace_column":
        materialize_column_replacement(
            source,
            target,
            table=str(operation["table"]),
            old_name=str(operation["old_name"]),
            new_name=str(operation["new_name"]),
            declared_type=str(operation.get("declared_type", "TEXT")),
        )
        return
    if operation_type == "add_column":
        materialize_column_addition(
            source,
            target,
            table=str(operation["table"]),
            new_name=str(operation["new_name"]),
            declared_type=str(operation.get("declared_type", "INTEGER")),
            default_sql=str(operation.get("default_sql", "0")),
        )
        return
    else:
        raise NotImplementedError(
            f"Unsupported schema operation: {operation_type}"
        )


def build_column_rename_example(
    *,
    source: str,
    source_index: int,
    db_id: str,
    question: str,
    evidence: str,
    sql: str,
    database: Path,
) -> DriftExample:
    """Build one deterministic oracle trajectory and verify it by execution."""

    database = Path(database).resolve()
    if not database.is_file():
        raise FileNotFoundError(database)
    tree = parse_one(sql, read="sqlite")
    if not isinstance(tree, (exp.Query, exp.Subquery)):
        raise ValueError("Only read queries can seed drift examples")

    schemas = _schema_columns(database)
    aliases = _table_aliases(tree, schemas)
    candidates = _candidate_columns(tree, schemas, aliases)
    if not candidates:
        raise ValueError("No unambiguous referenced column can be renamed")

    original_fingerprint = fingerprint_query(database, sql)
    failures: list[str] = []
    for table, old_name in candidates:
        new_name = _new_column_name(table, old_name, schemas[table])
        repaired_sql = _rewrite_column(
            tree,
            schemas,
            aliases,
            table,
            old_name,
            new_name,
        )
        if repaired_sql == sql:
            continue

        try:
            with tempfile.TemporaryDirectory(
                prefix="driftsql-materialized-",
                dir=os.environ.get("DRIFTSQL_TMPDIR"),
                ignore_cleanup_errors=True,
            ) as temp_dir:
                changed_database = Path(temp_dir) / f"{db_id}__v2.sqlite"
                materialize_column_rename(
                    database,
                    changed_database,
                    table=table,
                    old_name=old_name,
                    new_name=new_name,
                )
                try:
                    fingerprint_query(changed_database, sql)
                except sqlite3.Error as error:
                    stale_error = str(error)
                else:
                    failures.append(f"{table}.{old_name}: stale SQL still succeeds")
                    continue
                repaired_fingerprint = fingerprint_query(
                    changed_database,
                    repaired_sql,
                )
        except (OSError, sqlite3.Error) as error:
            failures.append(f"{table}.{old_name}: {error}")
            continue

        if repaired_fingerprint != original_fingerprint:
            failures.append(f"{table}.{old_name}: result fingerprint changed")
            continue

        operation = {
            "type": "rename_column",
            "table": table,
            "old_name": old_name,
            "new_name": new_name,
        }
        schema_diff = SchemaDiff(
            db_id=db_id,
            from_version="v1",
            to_version="v2",
            operations=(operation,),
        )
        oracle_steps = (
            {
                "action": "execute_sql",
                "arguments": {"sql": sql},
                "observation": {"ok": False, "error": stale_error},
            },
            {
                "action": "get_schema_version",
                "arguments": {},
                "observation": {"db_id": db_id, "version": "v2"},
            },
            {
                "action": "inspect_schema_diff",
                "arguments": {"from_version": "v1", "to_version": "v2"},
                "observation": schema_diff.to_observation(),
            },
            {
                "action": "execute_sql",
                "arguments": {"sql": repaired_sql},
                "observation": {
                    "ok": True,
                    "row_count": repaired_fingerprint.row_count,
                    "value_hash": repaired_fingerprint.value_hash,
                },
            },
            {
                "action": "submit_solution",
                "arguments": {"sql": repaired_sql},
                "observation": {"accepted": True},
            },
        )
        task_key = json.dumps(
            [source, source_index, db_id, table, old_name, new_name],
            ensure_ascii=True,
        )
        task_hash = hashlib.sha256(task_key.encode("utf-8")).hexdigest()[:16]
        return DriftExample(
            task_id=f"drift_colrename_{task_hash}",
            source=source,
            source_index=source_index,
            db_id=db_id,
            question=question,
            evidence=evidence,
            source_db=str(database),
            stale_sql=sql,
            repaired_sql=repaired_sql,
            schema_diff=schema_diff,
            stale_error=stale_error,
            result_fingerprint=repaired_fingerprint,
            oracle_steps=oracle_steps,
        )

    detail = "; ".join(failures[:3]) or "all candidates were no-ops"
    raise ValueError(f"No executable column rename found: {detail}")


def build_table_rename_example(
    *,
    source: str,
    source_index: int,
    db_id: str,
    question: str,
    evidence: str,
    sql: str,
    database: Path,
) -> DriftExample:
    """Build one deterministic, execution-verified table-rename trajectory."""

    database = Path(database).resolve()
    if not database.is_file():
        raise FileNotFoundError(database)
    tree = parse_one(sql, read="sqlite")
    if not isinstance(tree, (exp.Query, exp.Subquery)):
        raise ValueError("Only read queries can seed drift examples")

    schemas = _schema_columns(database)
    table_lookup = _case_lookup(set(schemas))
    candidates = sorted(
        {
            actual
            for table in tree.find_all(exp.Table)
            if table.name and (actual := table_lookup.get(table.name.casefold()))
        },
        key=str.casefold,
    )
    if not candidates:
        raise ValueError("No referenced physical table can be renamed")

    original_fingerprint = fingerprint_query(database, sql)
    failures: list[str] = []
    for old_name in candidates:
        new_name = _new_column_name("schema", old_name, set(schemas))
        try:
            repaired_sql = _rewrite_table(tree, old_name, new_name)
            with tempfile.TemporaryDirectory(
                prefix="driftsql-materialized-",
                dir=os.environ.get("DRIFTSQL_TMPDIR"),
                ignore_cleanup_errors=True,
            ) as temp_dir:
                changed_database = Path(temp_dir) / f"{db_id}__v2.sqlite"
                materialize_table_rename(
                    database,
                    changed_database,
                    old_name=old_name,
                    new_name=new_name,
                )
                try:
                    fingerprint_query(changed_database, sql)
                except sqlite3.Error as error:
                    stale_error = str(error)
                else:
                    failures.append(f"{old_name}: stale SQL still succeeds")
                    continue
                repaired_fingerprint = fingerprint_query(changed_database, repaired_sql)
        except (OSError, sqlite3.Error, ValueError) as error:
            failures.append(f"{old_name}: {error}")
            continue

        if repaired_fingerprint != original_fingerprint:
            failures.append(f"{old_name}: result fingerprint changed")
            continue

        operation = {
            "type": "rename_table",
            "old_name": old_name,
            "new_name": new_name,
        }
        schema_diff = SchemaDiff(
            db_id=db_id,
            from_version="v1",
            to_version="v2",
            operations=(operation,),
        )
        oracle_steps = (
            {
                "action": "execute_sql",
                "arguments": {"sql": sql},
                "observation": {"ok": False, "error": stale_error},
            },
            {
                "action": "get_schema_version",
                "arguments": {},
                "observation": {"db_id": db_id, "version": "v2"},
            },
            {
                "action": "inspect_schema_diff",
                "arguments": {"from_version": "v1", "to_version": "v2"},
                "observation": schema_diff.to_observation(),
            },
            {
                "action": "execute_sql",
                "arguments": {"sql": repaired_sql},
                "observation": {
                    "ok": True,
                    "row_count": repaired_fingerprint.row_count,
                    "value_hash": repaired_fingerprint.value_hash,
                },
            },
            {
                "action": "submit_solution",
                "arguments": {"sql": repaired_sql},
                "observation": {"accepted": True},
            },
        )
        task_key = json.dumps(
            [source, source_index, db_id, "rename_table", old_name, new_name],
            ensure_ascii=True,
        )
        task_hash = hashlib.sha256(task_key.encode("utf-8")).hexdigest()[:16]
        return DriftExample(
            task_id=f"drift_tabrename_{task_hash}",
            source=source,
            source_index=source_index,
            db_id=db_id,
            question=question,
            evidence=evidence,
            source_db=str(database),
            stale_sql=sql,
            repaired_sql=repaired_sql,
            schema_diff=schema_diff,
            stale_error=stale_error,
            result_fingerprint=repaired_fingerprint,
            oracle_steps=oracle_steps,
        )

    detail = "; ".join(failures[:3]) or "all candidates were no-ops"
    raise ValueError(f"No executable table rename found: {detail}")


def build_column_replacement_example(
    *,
    source: str,
    source_index: int,
    db_id: str,
    question: str,
    evidence: str,
    sql: str,
    database: Path,
) -> DriftExample:
    """Replace a referenced column through copy-and-drop data migration."""

    database = Path(database).resolve()
    tree = parse_one(sql, read="sqlite")
    if not isinstance(tree, (exp.Query, exp.Subquery)):
        raise ValueError("Only read queries can seed drift examples")
    if any(isinstance(node, exp.Star) for node in tree.walk()):
        raise ValueError("SELECT star is handled by the add-column drift builder")

    schemas = _schema_columns(database)
    aliases = _table_aliases(tree, schemas)
    candidates = _candidate_columns(tree, schemas, aliases)
    if not candidates:
        raise ValueError("No unambiguous referenced column can be replaced")

    original_fingerprint = fingerprint_query(database, sql)
    failures: list[str] = []
    for table, old_name in candidates:
        new_name = _new_column_name(table, old_name, schemas[table])
        declared_type = _column_type(database, table, old_name)
        repaired_sql = _rewrite_column(tree, schemas, aliases, table, old_name, new_name)
        try:
            with tempfile.TemporaryDirectory(
                prefix="driftsql-materialized-",
                dir=os.environ.get("DRIFTSQL_TMPDIR"),
                ignore_cleanup_errors=True,
            ) as temp_dir:
                changed_database = Path(temp_dir) / f"{db_id}__v2.sqlite"
                materialize_column_replacement(
                    database,
                    changed_database,
                    table=table,
                    old_name=old_name,
                    new_name=new_name,
                    declared_type=declared_type,
                )
                try:
                    fingerprint_query(changed_database, sql)
                except sqlite3.Error as error:
                    stale_error = str(error)
                else:
                    failures.append(f"{table}.{old_name}: stale SQL still succeeds")
                    continue
                repaired_fingerprint = fingerprint_query(changed_database, repaired_sql)
        except (OSError, sqlite3.Error, ValueError) as error:
            failures.append(f"{table}.{old_name}: {error}")
            continue

        if repaired_fingerprint != original_fingerprint:
            failures.append(f"{table}.{old_name}: result fingerprint changed")
            continue

        operation = {
            "type": "replace_column",
            "table": table,
            "old_name": old_name,
            "new_name": new_name,
            "declared_type": declared_type,
        }
        schema_diff = SchemaDiff(db_id, "v1", "v2", (operation,))
        oracle_steps = _recovery_oracle_steps(
            db_id=db_id,
            stale_sql=sql,
            stale_observation={"ok": False, "error": stale_error},
            repaired_sql=repaired_sql,
            repaired_fingerprint=repaired_fingerprint,
            schema_diff=schema_diff,
        )
        task_hash = _task_hash(
            source, source_index, db_id, "replace_column", table, old_name, new_name
        )
        return DriftExample(
            task_id=f"drift_colreplace_{task_hash}",
            source=source,
            source_index=source_index,
            db_id=db_id,
            question=question,
            evidence=evidence,
            source_db=str(database),
            stale_sql=sql,
            repaired_sql=repaired_sql,
            schema_diff=schema_diff,
            stale_error=stale_error,
            result_fingerprint=repaired_fingerprint,
            oracle_steps=oracle_steps,
        )

    detail = "; ".join(failures[:3]) or "all candidates were no-ops"
    raise ValueError(f"No executable column replacement found: {detail}")


def build_add_column_projection_example(
    *,
    source: str,
    source_index: int,
    db_id: str,
    question: str,
    evidence: str,
    sql: str,
    database: Path,
    added_column_specs: list[dict[str, str]] | None = None,
) -> DriftExample:
    """Build an execution-verified additive projection-contract drift."""

    database = Path(database).resolve()
    if not database.is_file():
        raise FileNotFoundError(database)
    schemas = {table: _column_order(database, table) for table in _schema_columns(database)}
    analysis = analyze_wildcard_projection(sql, schemas)
    wildcard_tables = list(analysis["wildcard_tables"])
    if added_column_specs is None:
        table = wildcard_tables[0]
        added_column_specs = [
            {
                "table": table,
                "new_name": _new_column_name(table, "audit_flag", set(schemas[table])),
                "declared_type": "INTEGER",
                "default_sql": "0",
            }
        ]
    operations: list[dict[str, str]] = []
    seen_targets: set[tuple[str, str]] = set()
    table_lookup = _case_lookup(set(schemas))
    for spec in added_column_specs:
        table = table_lookup.get(str(spec.get("table", "")).casefold())
        if not table or table not in wildcard_tables:
            raise ValueError(f"Added column must target a wildcard table: {spec}")
        new_name = str(spec.get("new_name", "")).strip()
        if not new_name or new_name.casefold() in {name.casefold() for name in schemas[table]}:
            raise ValueError(f"Added column is empty or already exists: {spec}")
        target = (table.casefold(), new_name.casefold())
        if target in seen_targets:
            raise ValueError(f"Duplicate add-column target: {spec}")
        seen_targets.add(target)
        operations.append(
            {
                "type": "add_column",
                "table": table,
                "new_name": new_name,
                "declared_type": str(spec.get("declared_type", "INTEGER")),
                "default_sql": str(spec.get("default_sql", "0")),
            }
        )
    schema_diff = SchemaDiff(db_id, "v1", "v2", tuple(operations))
    plan = plan_projection_contract(sql, schema_diff.to_observation(), schemas)
    repaired_sql = plan.repaired_sql
    original_fingerprint = fingerprint_query(database, sql)

    with tempfile.TemporaryDirectory(
        prefix="driftsql-materialized-",
        dir=os.environ.get("DRIFTSQL_TMPDIR"),
        ignore_cleanup_errors=True,
    ) as temp_dir:
        changed_database = Path(temp_dir) / f"{db_id}__v2.sqlite"
        materialize_schema_diff(database, changed_database, schema_diff)
        stale_fingerprint = fingerprint_query(changed_database, sql)
        if stale_fingerprint == original_fingerprint:
            raise ValueError("Added column did not change SELECT-star result")
        repaired_fingerprint = fingerprint_query(changed_database, repaired_sql)
    if repaired_fingerprint != original_fingerprint:
        raise ValueError("Expanded SELECT-star result differs from the original")

    stale_error = "silent_result_schema_mismatch"
    oracle_steps = _recovery_oracle_steps(
        db_id=db_id,
        stale_sql=sql,
        stale_observation={
            "ok": True,
            "error": None,
            "semantic_mismatch": True,
            "row_count": stale_fingerprint.row_count,
            "value_hash": stale_fingerprint.value_hash,
        },
        repaired_sql=repaired_sql,
        repaired_fingerprint=repaired_fingerprint,
        schema_diff=schema_diff,
    )
    task_hash = _task_hash(
        source,
        source_index,
        db_id,
        "add_column",
        plan.wildcard_profile,
        *(f"{item['table']}.{item['new_name']}" for item in operations),
    )
    return DriftExample(
        task_id=f"drift_coladd_{task_hash}",
        source=source,
        source_index=source_index,
        db_id=db_id,
        question=question,
        evidence=evidence,
        source_db=str(database),
        stale_sql=sql,
        repaired_sql=repaired_sql,
        schema_diff=schema_diff,
        stale_error=stale_error,
        result_fingerprint=repaired_fingerprint,
        oracle_steps=oracle_steps,
        wildcard_profile=plan.wildcard_profile,
        added_column_count=len(operations),
    )


def build_add_column_star_example(**kwargs: Any) -> DriftExample:
    """Backward-compatible wrapper for additive wildcard drift generation."""

    return build_add_column_projection_example(**kwargs)


def build_clean_example(
    *,
    source: str,
    source_index: int,
    db_id: str,
    question: str,
    evidence: str,
    sql: str,
    database: Path,
) -> DriftExample:
    """Build an execution-verified no-drift negative-control episode."""

    database = Path(database).resolve()
    if not database.is_file():
        raise FileNotFoundError(database)
    tree = parse_one(sql, read="sqlite")
    if not isinstance(tree, (exp.Query, exp.Subquery)):
        raise ValueError("Only read queries can seed clean examples")
    fingerprint = fingerprint_query(database, sql)
    schema_diff = SchemaDiff(db_id, "v1", "v1", ())
    oracle_steps = (
        {
            "action": "execute_sql",
            "arguments": {"sql": sql},
            "observation": {
                "ok": True,
                "row_count": fingerprint.row_count,
                "value_hash": fingerprint.value_hash,
            },
        },
        {
            "action": "submit_solution",
            "arguments": {"sql": sql},
            "observation": {"accepted": True},
        },
    )
    task_hash = _task_hash(source, source_index, db_id, "clean")
    return DriftExample(
        task_id=f"drift_clean_{task_hash}",
        source=source,
        source_index=source_index,
        db_id=db_id,
        question=question,
        evidence=evidence,
        source_db=str(database),
        stale_sql=sql,
        repaired_sql=sql,
        schema_diff=schema_diff,
        stale_error="none",
        result_fingerprint=fingerprint,
        oracle_steps=oracle_steps,
    )


def build_compound_drift_example(
    *,
    source: str,
    source_index: int,
    db_id: str,
    question: str,
    evidence: str,
    sql: str,
    database: Path,
) -> DriftExample:
    """Compose two execution-preserving schema mutations in a fixed order."""

    database = Path(database).resolve()
    original_fingerprint = fingerprint_query(database, sql)
    builders = {
        "rename_column": build_column_rename_example,
        "rename_table": build_table_rename_example,
        "replace_column": build_column_replacement_example,
    }
    pairs = (
        ("rename_table", "rename_column"),
        ("rename_column", "replace_column"),
        ("replace_column", "rename_table"),
        ("rename_column", "rename_column"),
    )
    failures: list[str] = []
    for first_name, second_name in pairs:
        try:
            first = builders[first_name](
                source=source,
                source_index=source_index,
                db_id=db_id,
                question=question,
                evidence=evidence,
                sql=sql,
                database=database,
            )
            with tempfile.TemporaryDirectory(
                prefix="driftsql-compound-",
                dir=os.environ.get("DRIFTSQL_TMPDIR"),
                ignore_cleanup_errors=True,
            ) as directory:
                first_database = Path(directory) / f"{db_id}__first.sqlite"
                materialize_schema_diff(database, first_database, first.schema_diff)
                second = builders[second_name](
                    source=source,
                    source_index=source_index,
                    db_id=db_id,
                    question=question,
                    evidence=evidence,
                    sql=first.repaired_sql,
                    database=first_database,
                )
                operations = first.schema_diff.operations + second.schema_diff.operations
                schema_diff = SchemaDiff(db_id, "v1", "v2", operations)
                final_database = Path(directory) / f"{db_id}__compound.sqlite"
                materialize_schema_diff(database, final_database, schema_diff)
                repaired_fingerprint = fingerprint_query(
                    final_database, second.repaired_sql
                )
                try:
                    stale_fingerprint = fingerprint_query(final_database, sql)
                except sqlite3.Error as error:
                    stale_error = str(error)
                else:
                    if stale_fingerprint == original_fingerprint:
                        raise ValueError("compound drift leaves stale SQL equivalent")
                    stale_error = "silent_result_schema_mismatch"
            if repaired_fingerprint != original_fingerprint:
                raise ValueError("compound repair changed the result fingerprint")
        except (OSError, sqlite3.Error, ValueError) as error:
            failures.append(f"{first_name}+{second_name}: {error}")
            continue

        oracle_steps = _recovery_oracle_steps(
            db_id=db_id,
            stale_sql=sql,
            stale_observation={"ok": False, "error": stale_error},
            repaired_sql=second.repaired_sql,
            repaired_fingerprint=repaired_fingerprint,
            schema_diff=schema_diff,
        )
        task_hash = _task_hash(
            source,
            source_index,
            db_id,
            "compound",
            first_name,
            second_name,
            operations,
        )
        return DriftExample(
            task_id=f"drift_compound_{task_hash}",
            source=source,
            source_index=source_index,
            db_id=db_id,
            question=question,
            evidence=evidence,
            source_db=str(database),
            stale_sql=sql,
            repaired_sql=second.repaired_sql,
            schema_diff=schema_diff,
            stale_error=stale_error,
            result_fingerprint=repaired_fingerprint,
            oracle_steps=oracle_steps,
        )

    detail = "; ".join(failures[:4]) or "no compound pair was applicable"
    raise ValueError(f"No executable compound drift found: {detail}")


def _task_hash(*parts: object) -> str:
    payload = json.dumps(parts, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _recovery_oracle_steps(
    *,
    db_id: str,
    stale_sql: str,
    stale_observation: dict[str, Any],
    repaired_sql: str,
    repaired_fingerprint: QueryFingerprint,
    schema_diff: SchemaDiff,
) -> tuple[dict[str, Any], ...]:
    return (
        {
            "action": "execute_sql",
            "arguments": {"sql": stale_sql},
            "observation": stale_observation,
        },
        {
            "action": "get_schema_version",
            "arguments": {},
            "observation": {"db_id": db_id, "version": "v2"},
        },
        {
            "action": "inspect_schema_diff",
            "arguments": {"from_version": "v1", "to_version": "v2"},
            "observation": schema_diff.to_observation(),
        },
        {
            "action": "execute_sql",
            "arguments": {"sql": repaired_sql},
            "observation": {
                "ok": True,
                "row_count": repaired_fingerprint.row_count,
                "value_hash": repaired_fingerprint.value_hash,
            },
        },
        {
            "action": "submit_solution",
            "arguments": {"sql": repaired_sql},
            "observation": {"accepted": True},
        },
    )
