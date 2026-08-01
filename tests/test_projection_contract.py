from __future__ import annotations

from driftsql.planning import analyze_wildcard_projection, plan_projection_contract


def test_plain_star_excludes_added_column_in_active_order() -> None:
    schema = {"orders": ["order_id", "amount", "audit_flag"]}
    diff = {
        "operations": [
            {"type": "add_column", "table": "orders", "new_name": "audit_flag"}
        ]
    }
    plan = plan_projection_contract("SELECT * FROM orders", diff, schema)
    assert plan.repaired_sql == 'SELECT "order_id", "amount" FROM orders'
    assert plan.excluded_added_columns == ("orders.audit_flag",)
    assert plan.wildcard_profile == "single_table_plain"


def test_qualified_alias_star_preserves_alias() -> None:
    schema = {"orders": ["order_id", "amount", "audit_flag"]}
    diff = {
        "operations": [
            {"type": "add_column", "table": "orders", "new_name": "audit_flag"}
        ]
    }
    plan = plan_projection_contract("SELECT src.* FROM orders AS src", diff, schema)
    assert plan.repaired_sql == 'SELECT src."order_id", src."amount" FROM orders AS src'
    assert plan.wildcard_profile == "single_table_qualified"


def test_multi_table_wildcards_exclude_each_tables_addition() -> None:
    schema = {
        "customers": ["customer_id", "name", "lineage_tag"],
        "orders": ["order_id", "customer_id", "audit_flag"],
    }
    diff = {
        "operations": [
            {"type": "add_column", "table": "customers", "new_name": "lineage_tag"},
            {"type": "add_column", "table": "orders", "new_name": "audit_flag"},
        ]
    }
    sql = (
        "SELECT c.*, o.* FROM customers AS c JOIN orders AS o "
        "ON c.customer_id = o.customer_id"
    )
    plan = plan_projection_contract(sql, diff, schema)
    assert plan.repaired_sql == (
        'SELECT c."customer_id", c."name", o."order_id", o."customer_id" '
        "FROM customers AS c JOIN orders AS o ON c.customer_id = o.customer_id"
    )
    assert plan.wildcard_profile == "multi_table_qualified"
    assert plan.expanded_columns == 4


def test_count_star_is_not_a_projection_wildcard() -> None:
    schema = {"orders": ["order_id", "audit_flag"]}
    try:
        analyze_wildcard_projection("SELECT COUNT(*) FROM orders", schema)
    except ValueError as error:
        assert "no supported wildcard" in str(error)
    else:
        raise AssertionError("COUNT(*) must not be expanded")
