"""Business-metric definition drift with auditable SQL expression rewrites."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlglot import Expression, parse_one


def _normalized(expression: Expression, *, dialect: str) -> str:
    return "".join(expression.sql(dialect=dialect).casefold().split())


def rewrite_metric_expression(
    sql: str,
    old_expression: str,
    new_expression: str,
    *,
    dialect: str = "sqlite",
) -> str:
    """Replace every structural occurrence of an obsolete metric expression."""

    tree = parse_one(sql, read=dialect)
    old = parse_one(old_expression, read=dialect)
    new = parse_one(new_expression, read=dialect)
    target = _normalized(old, dialect=dialect)
    matches = [node for node in tree.walk() if _normalized(node, dialect=dialect) == target]
    if not matches:
        raise ValueError(f"Metric expression is absent from SQL: {old_expression}")
    for node in matches:
        node.replace(new.copy())
    return tree.sql(dialect=dialect)


@dataclass(frozen=True)
class MetricDefinitionChange:
    metric_name: str
    old_expression: str
    new_expression: str
    from_version: str
    to_version: str
    reason: str
    requires_clarification: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("metric_name", self.metric_name),
            ("old_expression", self.old_expression),
            ("new_expression", self.new_expression),
            ("from_version", self.from_version),
            ("to_version", self.to_version),
            ("reason", self.reason),
        ):
            if not value.strip():
                raise ValueError(f"{name} must not be empty")
        parse_one(self.old_expression, read="sqlite")
        parse_one(self.new_expression, read="sqlite")
        if _normalized(parse_one(self.old_expression), dialect="sqlite") == _normalized(
            parse_one(self.new_expression), dialect="sqlite"
        ):
            raise ValueError("Metric expressions must differ")

    def rewrite(self, sql: str, *, dialect: str = "sqlite") -> str:
        return rewrite_metric_expression(
            sql,
            self.old_expression,
            self.new_expression,
            dialect=dialect,
        )

    def as_operation(self) -> dict[str, Any]:
        return {
            "type": "metric_definition_change",
            "metric_name": self.metric_name,
            "old_expression": self.old_expression,
            "new_expression": self.new_expression,
            "from_version": self.from_version,
            "to_version": self.to_version,
            "reason": self.reason,
            "requires_clarification": self.requires_clarification,
        }

    def as_knowledge_entry(self) -> dict[str, Any]:
        return {
            "knowledge": self.metric_name,
            "type": "metric_definition",
            "description": self.reason,
            "definition": self.new_expression,
            "version": self.to_version,
            "previous_definition": self.old_expression,
            "requires_clarification": self.requires_clarification,
        }
