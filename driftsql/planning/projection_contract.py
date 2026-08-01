"""Projection-contract recovery for additive schema drift.

An added column leaves ``SELECT *`` executable while silently changing the
result tuple contract.  This planner reconstructs the pre-change projection
from the active ordered schema and the audited add-column operations.  It does
not invent predicates, joins, or business logic.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from sqlglot import exp, parse_one


@dataclass(frozen=True)
class ProjectionContractPlan:
    original_sql: str
    repaired_sql: str
    wildcard_profile: str
    expanded_columns: int
    referenced_tables: tuple[str, ...]
    wildcard_tables: tuple[str, ...]
    excluded_added_columns: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _lookup(values: list[str] | tuple[str, ...] | set[str]) -> dict[str, str]:
    return {str(value).casefold(): str(value) for value in values}


def _one_select(sql: str) -> tuple[exp.Expression, exp.Select]:
    tree = parse_one(sql, read="sqlite")
    if not isinstance(tree, (exp.Query, exp.Subquery)):
        raise ValueError("Projection planner supports read queries only")
    selects = list(tree.find_all(exp.Select))
    if len(selects) != 1:
        raise ValueError("Projection planner requires exactly one SELECT scope")
    return tree, selects[0]


def _table_bindings(
    tree: exp.Expression,
    ordered_schema: dict[str, list[str]],
) -> tuple[list[tuple[str, str]], dict[str, tuple[str, str]]]:
    table_names = _lookup(list(ordered_schema))
    ordered: list[tuple[str, str]] = []
    by_qualifier: dict[str, tuple[str, str]] = {}
    for table in tree.find_all(exp.Table):
        physical = table_names.get(table.name.casefold())
        if not physical:
            raise ValueError(f"Referenced table is absent from active schema: {table.name}")
        qualifier = table.alias or table.name
        binding = (physical, qualifier)
        ordered.append(binding)
        by_qualifier[qualifier.casefold()] = binding
        by_qualifier[physical.casefold()] = binding
    if not ordered:
        raise ValueError("Projection planner requires at least one physical table")
    return ordered, by_qualifier


def analyze_wildcard_projection(
    sql: str,
    ordered_schema: dict[str, list[str]],
) -> dict[str, Any]:
    tree, select = _one_select(sql)
    ordered, by_qualifier = _table_bindings(tree, ordered_schema)
    wildcard_tables: list[str] = []
    plain_count = 0
    qualified_count = 0
    for expression in select.expressions:
        if isinstance(expression, exp.Star):
            plain_count += 1
            wildcard_tables.extend(physical for physical, _ in ordered)
        elif isinstance(expression, exp.Column) and isinstance(expression.this, exp.Star):
            binding = by_qualifier.get(expression.table.casefold())
            if not binding:
                raise ValueError(f"Unknown wildcard qualifier: {expression.table}")
            qualified_count += 1
            wildcard_tables.append(binding[0])
    if not wildcard_tables:
        raise ValueError("Top-level projection contains no supported wildcard")
    unique_tables = tuple(dict.fromkeys(wildcard_tables))
    if len(ordered) > 1:
        profile = "multi_table_plain" if plain_count else "multi_table_qualified"
    else:
        profile = "single_table_plain" if plain_count else "single_table_qualified"
    return {
        "wildcard_profile": profile,
        "referenced_tables": tuple(physical for physical, _ in ordered),
        "wildcard_tables": unique_tables,
        "plain_wildcards": plain_count,
        "qualified_wildcards": qualified_count,
    }


def _column(identifier: str, qualifier: str | None) -> exp.Column:
    # Always quote physical column names. SQLite permits names such as
    # ``group`` that sqlglot's generic keyword table does not reliably mark
    # as reserved, and an unquoted expansion would turn valid ``*`` SQL into
    # a syntax error.
    quoted_column = True
    quoted_table = bool(qualifier and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", qualifier))
    return exp.Column(
        this=exp.to_identifier(identifier, quoted=quoted_column),
        table=exp.to_identifier(qualifier, quoted=quoted_table) if qualifier else None,
    )


def plan_projection_contract(
    sql: str,
    schema_diff: dict[str, Any],
    ordered_active_schema: dict[str, list[str]],
) -> ProjectionContractPlan:
    """Expand top-level wildcards to the pre-addition ordered column contract."""

    analysis = analyze_wildcard_projection(sql, ordered_active_schema)
    additions: dict[str, set[str]] = {}
    table_names = _lookup(list(ordered_active_schema))
    excluded: list[str] = []
    for operation in schema_diff.get("operations", []) or []:
        if not isinstance(operation, dict) or operation.get("type") != "add_column":
            continue
        raw_table = str(operation.get("table") or "")
        raw_name = str(operation.get("new_name") or "")
        table = table_names.get(raw_table.casefold())
        if not table or not raw_name:
            raise ValueError(f"Invalid add-column operation for projection planner: {operation}")
        additions.setdefault(table, set()).add(raw_name.casefold())
        excluded.append(f"{table}.{raw_name}")
    if not additions:
        raise ValueError("Projection planner requires at least one add-column operation")

    tree, select = _one_select(sql)
    ordered, by_qualifier = _table_bindings(tree, ordered_active_schema)

    def old_columns(physical: str) -> list[str]:
        removed = additions.get(physical, set())
        columns = [name for name in ordered_active_schema[physical] if name.casefold() not in removed]
        if not columns:
            raise ValueError(f"No pre-change columns remain for table {physical}")
        return columns

    expanded: list[exp.Expression] = []
    expanded_count = 0
    for expression in select.expressions:
        if isinstance(expression, exp.Star):
            multi_table = len(ordered) > 1
            for physical, qualifier in ordered:
                columns = old_columns(physical)
                expanded.extend(_column(name, qualifier if multi_table else None) for name in columns)
                expanded_count += len(columns)
            continue
        if isinstance(expression, exp.Column) and isinstance(expression.this, exp.Star):
            binding = by_qualifier.get(expression.table.casefold())
            if not binding:
                raise ValueError(f"Unknown wildcard qualifier: {expression.table}")
            physical, _ = binding
            columns = old_columns(physical)
            expanded.extend(_column(name, expression.table) for name in columns)
            expanded_count += len(columns)
            continue
        expanded.append(expression)
    select.set("expressions", expanded)
    return ProjectionContractPlan(
        original_sql=sql,
        repaired_sql=tree.sql(dialect="sqlite"),
        wildcard_profile=str(analysis["wildcard_profile"]),
        expanded_columns=expanded_count,
        referenced_tables=tuple(analysis["referenced_tables"]),
        wildcard_tables=tuple(analysis["wildcard_tables"]),
        excluded_added_columns=tuple(excluded),
    )
