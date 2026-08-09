"""Deterministic SQL repair candidates derived only from audited schema diffs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from sqlglot import exp, parse_one

from driftsql.sql_rewrite import rewrite_sql_identifier

from .projection_contract import plan_projection_contract


@dataclass(frozen=True)
class AuditedSchemaRepairPlan:
    original_sql: str
    repaired_sql: str
    operations_applied: tuple[str, ...]
    changed: bool
    source: str = "stale_sql+audited_schema_diff"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_only(sql: str) -> bool:
    try:
        expression = parse_one(sql, read="sqlite")
    except Exception:
        return False
    return isinstance(expression, (exp.Query, exp.Subquery))


def plan_audited_schema_repair(
    stale_sql: str,
    schema_diff: dict[str, Any],
    *,
    ordered_active_schema: dict[str, list[str]] | None = None,
) -> AuditedSchemaRepairPlan:
    """Apply explicit old/new identifier mappings without inventing semantics.

    Rename and replacement operations use the conservative identifier lexer,
    which never rewrites string literals. Add-column drift delegates to the
    projection-contract planner so a top-level wildcard preserves the cached
    result tuple. Every transformation is fully determined by the audited diff.
    """

    if not _read_only(stale_sql):
        raise ValueError("Audited repair requires one read-only SQL query")
    operations = [
        operation
        for operation in schema_diff.get("operations", []) or []
        if isinstance(operation, dict)
    ]
    repaired = stale_sql
    applied: list[str] = []
    for operation in operations:
        operation_type = str(operation.get("type", ""))
        if operation_type not in {"rename_table", "rename_column", "replace_column"}:
            continue
        old_name = str(operation.get("old_name", "")).strip()
        new_name = str(operation.get("new_name", "")).strip()
        if not old_name or not new_name:
            raise ValueError(f"Audited identifier mapping is incomplete: {operation}")
        updated = rewrite_sql_identifier(repaired, old_name, new_name)
        if updated != repaired:
            repaired = updated
            applied.append(operation_type)

    additions = [operation for operation in operations if operation.get("type") == "add_column"]
    if additions:
        if ordered_active_schema is None:
            raise ValueError("Add-column repair requires the ordered active schema")
        try:
            projection = plan_projection_contract(
                repaired,
                {**schema_diff, "operations": additions},
                ordered_active_schema,
            )
        except ValueError as error:
            # An explicit projection is unaffected by an added column. It is a
            # valid no-op repair candidate; only wildcard planner failures are
            # considered actionable.
            if "no supported wildcard" not in str(error):
                raise
        else:
            repaired = projection.repaired_sql
            applied.append("add_column_projection_contract")

    if not _read_only(repaired):
        raise ValueError("Audited repair produced a non-query expression")
    return AuditedSchemaRepairPlan(
        original_sql=stale_sql,
        repaired_sql=repaired,
        operations_applied=tuple(applied),
        changed=repaired.rstrip(";").strip() != stale_sql.rstrip(";").strip(),
    )
