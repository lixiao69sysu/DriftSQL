"""Execution-verified SQL-reasoning SFT records.

The target is a concise, inspectable logical plan rather than an unverifiable
free-form chain of thought.  Plans are derived deterministically from the Gold
SQL AST, and every retained SQL statement is executed against its source DB.
"""

from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path
from typing import Any

from sqlglot import exp, parse_one


REASONING_SYSTEM_PROMPT = """You are a production text-to-SQL engineer.
Given a question, database schema, and optional business evidence, produce a
short auditable query plan and one executable SQLite query.

Return exactly:
<plan>
1. concise relational operation
2. ...
</plan>
<sql>
SELECT ...
</sql>

Do not use tools, invent schema fields, or add prose outside these tags."""

REASONING_USER_TEMPLATE = """## Database schema
{schema}

## Business evidence
{evidence}

## Question
{question}"""


def _compact_sql(value: exp.Expression | None, maximum: int = 320) -> str:
    if value is None:
        return ""
    rendered = re.sub(r"\s+", " ", value.sql(dialect="sqlite")).strip()
    return rendered if len(rendered) <= maximum else rendered[: maximum - 3] + "..."


def _terms(text: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_]*", text)
        if len(token) > 1
    }


def read_schema_objects(database: Path) -> list[tuple[str, str]]:
    """Read physical table/view DDL without sampling data rows."""

    uri = Path(database).resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        rows = connection.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%' "
            "AND sql IS NOT NULL ORDER BY name"
        ).fetchall()
    return [(str(name), str(ddl).rstrip(";") + ";") for name, ddl in rows]


def referenced_physical_tables(sql: str, available: set[str]) -> list[str]:
    tree = parse_one(sql, read="sqlite")
    cte_names = {
        cte.alias_or_name.casefold()
        for cte in tree.find_all(exp.CTE)
        if cte.alias_or_name
    }
    references = {
        table.name.casefold()
        for table in tree.find_all(exp.Table)
        if table.name and table.name.casefold() not in cte_names
    }
    return sorted(references & {name.casefold() for name in available})


def select_schema_context(
    objects: list[tuple[str, str]],
    *,
    question: str,
    evidence: str,
    gold_sql: str,
    max_chars: int = 10_000,
) -> tuple[str, dict[str, Any]]:
    """Keep full DDL when possible; otherwise retain targets plus distractors.

    Gold table retention is an explicitly recorded offline-training operation,
    not the retrieval policy used during evaluation.
    """

    if max_chars < 1000:
        raise ValueError("max_chars must be at least 1000")
    rendered_all = "\n\n".join(ddl for _, ddl in objects)
    names = {name for name, _ in objects}
    referenced = referenced_physical_tables(gold_sql, names)
    if len(rendered_all) <= max_chars:
        return rendered_all, {
            "mode": "full_schema",
            "total_objects": len(objects),
            "selected_objects": len(objects),
            "referenced_tables": referenced,
        }

    by_folded_name = {name.casefold(): (name, ddl) for name, ddl in objects}
    selected: list[tuple[str, str]] = [
        by_folded_name[name] for name in referenced if name in by_folded_name
    ]
    required_chars = sum(len(ddl) + 2 for _, ddl in selected)
    if required_chars > max_chars:
        raise ValueError("Gold-referenced schema exceeds the context budget")

    context_terms = _terms(question + " " + evidence)
    remaining = [item for item in objects if item[0].casefold() not in set(referenced)]
    remaining.sort(
        key=lambda item: (
            len(context_terms & _terms(item[0] + " " + item[1])),
            -len(item[1]),
            item[0].casefold(),
        ),
        reverse=True,
    )
    current = required_chars
    for item in remaining:
        addition = len(item[1]) + 2
        if current + addition <= max_chars:
            selected.append(item)
            current += addition
    selected_names = {name for name, _ in selected}
    selected.sort(key=lambda item: item[0].casefold())
    return "\n\n".join(ddl for _, ddl in selected), {
        "mode": "gold_tables_plus_question_ranked_distractors",
        "total_objects": len(objects),
        "selected_objects": len(selected),
        "referenced_tables": referenced,
        "selected_names": sorted(selected_names),
    }


def build_logical_plan(sql: str) -> list[str]:
    """Convert SQL structure into a deterministic high-level relational plan."""

    tree = parse_one(sql, read="sqlite")
    cte_names = {
        cte.alias_or_name.casefold()
        for cte in tree.find_all(exp.CTE)
        if cte.alias_or_name
    }
    tables: list[str] = []
    for table in tree.find_all(exp.Table):
        if table.name and table.name.casefold() not in cte_names and table.name not in tables:
            tables.append(table.name)

    plan: list[str] = []
    if tree.find(exp.Union) or tree.find(exp.Intersect) or tree.find(exp.Except):
        plan.append("Evaluate each query branch and combine them with the requested set operation.")
    if tables:
        quoted = ", ".join(f"`{name}`" for name in tables)
        plan.append(f"Read the required data from {quoted}.")

    main_select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    if main_select is not None:
        joins = list(main_select.find_all(exp.Join))
        if joins:
            conditions = [_compact_sql(join.args.get("on"), 180) for join in joins]
            conditions = [value for value in conditions if value]
            suffix = f" using {'; '.join(conditions)}" if conditions else ""
            plan.append(f"Join the related rows{suffix}.")

        where = main_select.args.get("where")
        if where is not None:
            plan.append(f"Filter rows by `{_compact_sql(where.this)}`.")

        group = main_select.args.get("group")
        if group is not None and group.expressions:
            plan.append(
                "Group rows by "
                + ", ".join(f"`{_compact_sql(item, 160)}`" for item in group.expressions)
                + "."
            )

        having = main_select.args.get("having")
        if having is not None:
            plan.append(f"Filter grouped results by `{_compact_sql(having.this)}`.")

        projections = main_select.expressions
        if projections:
            plan.append(
                "Return "
                + ", ".join(f"`{_compact_sql(item, 180)}`" for item in projections)
                + "."
            )

        order = main_select.args.get("order")
        if order is not None and order.expressions:
            plan.append(
                "Sort by "
                + ", ".join(f"`{_compact_sql(item, 160)}`" for item in order.expressions)
                + "."
            )
        limit = main_select.args.get("limit")
        if limit is not None:
            plan.append(f"Limit the result to `{_compact_sql(limit.expression, 80)}` rows.")

    return plan or ["Execute the requested read-only query and return its projected result."]


def validate_gold_sql(
    database: Path,
    sql: str,
    *,
    timeout_seconds: float = 10.0,
    max_rows: int = 100,
) -> dict[str, Any]:
    """Parse and execute a Gold SELECT against the real read-only database."""

    started = time.monotonic()
    try:
        tree = parse_one(sql, read="sqlite")
    except Exception as error:
        return {"success": False, "stage": "parse", "error": str(error)}
    forbidden = (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop, exp.Alter)
    if isinstance(tree, forbidden) or any(tree.find(kind) for kind in forbidden):
        return {"success": False, "stage": "safety", "error": "non-read-only Gold SQL"}

    deadline = time.monotonic() + timeout_seconds
    uri = Path(database).resolve().as_uri() + "?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True) as connection:
            connection.execute("PRAGMA query_only = ON")
            connection.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 1000)
            cursor = connection.execute(sql)
            columns = [item[0] for item in (cursor.description or ())]
            rows = cursor.fetchmany(max_rows + 1)
        return {
            "success": True,
            "stage": "execute",
            "error": "",
            "columns": columns,
            "observed_rows": min(len(rows), max_rows),
            "truncated": len(rows) > max_rows,
            "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
        }
    except (sqlite3.Error, ValueError) as error:
        return {
            "success": False,
            "stage": "execute",
            "error": str(error),
            "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
        }


def build_reasoning_messages(
    *,
    question: str,
    evidence: str,
    schema: str,
    gold_sql: str,
) -> list[dict[str, str]]:
    plan = build_logical_plan(gold_sql)
    numbered = "\n".join(f"{index}. {step}" for index, step in enumerate(plan, 1))
    assistant = f"<plan>\n{numbered}\n</plan>\n<sql>\n{gold_sql.strip()}\n</sql>"
    return [
        {"role": "system", "content": REASONING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": REASONING_USER_TEMPLATE.format(
                schema=schema,
                evidence=evidence.strip() or "(none)",
                question=question.strip(),
            ),
        },
        {"role": "assistant", "content": assistant},
    ]
