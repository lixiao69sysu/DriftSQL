from __future__ import annotations

from driftsql.planning import plan_audited_schema_repair


def test_audited_repair_changes_identifier_not_string_literal() -> None:
    plan = plan_audited_schema_repair(
        "SELECT T2.city FROM stores AS T2 WHERE T2.city = 'city'",
        {
            "operations": [
                {
                    "type": "replace_column",
                    "table": "stores",
                    "old_name": "city",
                    "new_name": "municipality",
                }
            ]
        },
    )
    assert "T2.municipality" in plan.repaired_sql
    assert "'city'" in plan.repaired_sql
    assert plan.operations_applied == ("replace_column",)
    assert plan.changed is True


def test_audited_repair_applies_compound_operations_in_order() -> None:
    plan = plan_audited_schema_repair(
        "SELECT old_table.old_col FROM old_table",
        {
            "operations": [
                {"type": "rename_table", "old_name": "old_table", "new_name": "new_table"},
                {
                    "type": "rename_column",
                    "table": "new_table",
                    "old_name": "old_col",
                    "new_name": "new_col",
                },
            ]
        },
    )
    assert plan.repaired_sql == "SELECT new_table.new_col FROM new_table"
    assert plan.operations_applied == ("rename_table", "rename_column")


def test_audited_repair_expands_add_column_wildcard_contract() -> None:
    plan = plan_audited_schema_repair(
        "SELECT * FROM orders",
        {
            "operations": [
                {"type": "add_column", "table": "orders", "new_name": "audit_flag"}
            ]
        },
        ordered_active_schema={"orders": ["id", "amount", "audit_flag"]},
    )
    assert plan.repaired_sql == 'SELECT "id", "amount" FROM orders'
    assert plan.operations_applied == ("add_column_projection_contract",)
