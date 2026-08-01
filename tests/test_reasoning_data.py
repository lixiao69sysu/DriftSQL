from __future__ import annotations

import sqlite3
from pathlib import Path

from driftsql.data.reasoning import (
    build_logical_plan,
    build_reasoning_messages,
    read_schema_objects,
    select_schema_context,
    validate_gold_sql,
)


def test_reasoning_record_is_structured_and_execution_verified(tmp_path: Path) -> None:
    database = tmp_path / "retail.sqlite"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT);"
            "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, amount REAL);"
            "CREATE TABLE audit_log (id INTEGER PRIMARY KEY, message TEXT);"
            "INSERT INTO customers VALUES (1, 'Ada');"
            "INSERT INTO orders VALUES (1, 1, 12.5);"
        )
    sql = (
        "SELECT c.name, SUM(o.amount) AS total FROM customers c "
        "JOIN orders o ON o.customer_id = c.id WHERE o.amount > 0 "
        "GROUP BY c.name ORDER BY total DESC LIMIT 5"
    )
    objects = read_schema_objects(database)
    schema, metadata = select_schema_context(
        objects,
        question="Show total order amount by customer.",
        evidence="total means SUM(amount)",
        gold_sql=sql,
        max_chars=10_000,
    )
    plan = build_logical_plan(sql)
    messages = build_reasoning_messages(
        question="Show total order amount by customer.",
        evidence="total means SUM(amount)",
        schema=schema,
        gold_sql=sql,
    )
    validation = validate_gold_sql(database, sql)

    assert metadata["mode"] == "full_schema"
    assert "CREATE TABLE customers" in schema
    assert any("Join" in step for step in plan)
    assert any("Group" in step for step in plan)
    assert messages[-1]["content"].startswith("<plan>")
    assert messages[-1]["content"].endswith("</sql>")
    assert validation["success"]
    assert validation["columns"] == ["name", "total"]


def test_reasoning_validator_rejects_writes(tmp_path: Path) -> None:
    database = tmp_path / "safe.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE values_table (value INTEGER)")
    result = validate_gold_sql(database, "UPDATE values_table SET value = 1")
    assert not result["success"]
    assert result["stage"] == "safety"
