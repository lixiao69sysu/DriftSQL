from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from driftsql.drift.factory import (
    build_add_column_star_example,
    build_add_column_projection_example,
    build_clean_example,
    build_column_rename_example,
    build_column_replacement_example,
    build_compound_drift_example,
    build_table_rename_example,
    fingerprint_query,
    materialize_column_rename,
    materialize_schema_diff,
)


class DriftFactoryTest(unittest.TestCase):
    def test_builds_execution_verified_oracle_trajectory(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="driftsql-factory-",
            dir="/tmp",
        ) as temp_dir:
            root = Path(temp_dir)
            source = root / "retail.sqlite"
            with sqlite3.connect(source) as connection:
                connection.execute(
                    "CREATE TABLE orders "
                    "(order_id INTEGER PRIMARY KEY, total_amount REAL)"
                )
                connection.executemany(
                    "INSERT INTO orders VALUES (?, ?)",
                    [(1, 12.5), (2, 7.0)],
                )

            example = build_column_rename_example(
                source="unit_test",
                source_index=0,
                db_id="retail",
                question="Show order totals.",
                evidence="",
                sql="SELECT total_amount FROM orders ORDER BY order_id",
                database=source,
            )

            operation = example.schema_diff.operations[0]
            self.assertEqual(operation["type"], "rename_column")
            self.assertIn(operation["new_name"], example.repaired_sql)
            self.assertIn("no such column", example.stale_error)
            self.assertEqual(example.result_fingerprint.row_count, 2)
            self.assertEqual(
                [step["action"] for step in example.oracle_steps],
                [
                    "execute_sql",
                    "get_schema_version",
                    "inspect_schema_diff",
                    "execute_sql",
                    "submit_solution",
                ],
            )

            changed = root / "changed.sqlite"
            materialize_column_rename(
                source,
                changed,
                table=str(operation["table"]),
                old_name=str(operation["old_name"]),
                new_name=str(operation["new_name"]),
            )
            with sqlite3.connect(changed) as connection:
                rows = connection.execute(example.repaired_sql).fetchall()
            self.assertEqual(rows, [(12.5,), (7.0,)])

    def test_builds_value_preserving_column_replacement(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="driftsql-replace-factory-",
            dir="/tmp",
        ) as temp_dir:
            root = Path(temp_dir)
            source = root / "retail.sqlite"
            with sqlite3.connect(source) as connection:
                connection.execute(
                    "CREATE TABLE orders "
                    "(order_id INTEGER PRIMARY KEY, total_amount REAL)"
                )
                connection.executemany(
                    "INSERT INTO orders VALUES (?, ?)",
                    [(1, 12.5), (2, 7.0)],
                )

            example = build_column_replacement_example(
                source="unit_test",
                source_index=2,
                db_id="retail",
                question="Show order totals.",
                evidence="",
                sql="SELECT total_amount FROM orders ORDER BY order_id",
                database=source,
            )
            operation = example.schema_diff.operations[0]
            self.assertEqual(operation["type"], "replace_column")
            self.assertIn("no such column", example.stale_error)
            with sqlite3.connect(source) as connection:
                original_columns = [
                    row[1]
                    for row in connection.execute("PRAGMA table_info(orders)")
                ]
            original_index = original_columns.index(operation["old_name"])

            changed = root / "changed.sqlite"
            materialize_schema_diff(source, changed, example.schema_diff)
            with sqlite3.connect(changed) as connection:
                rows = connection.execute(example.repaired_sql).fetchall()
                columns = [
                    row[1]
                    for row in connection.execute("PRAGMA table_info(orders)")
                ]
            self.assertEqual(rows, [(12.5,), (7.0,)])
            self.assertNotIn(operation["old_name"], columns)
            self.assertIn(operation["new_name"], columns)
            self.assertEqual(columns.index(operation["new_name"]), original_index)

    def test_replacement_keeps_copy_drop_fallback_for_type_change(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="driftsql-replace-type-change-",
            dir="/tmp",
        ) as temp_dir:
            root = Path(temp_dir)
            source = root / "retail.sqlite"
            changed = root / "changed.sqlite"
            with sqlite3.connect(source) as connection:
                connection.execute("CREATE TABLE orders (old_value INTEGER, marker TEXT)")
                connection.execute("INSERT INTO orders VALUES (7, 'kept')")

            materialize_schema_diff(
                source,
                changed,
                {
                    "operations": [
                        {
                            "type": "replace_column",
                            "table": "orders",
                            "old_name": "old_value",
                            "new_name": "new_value",
                            "declared_type": "TEXT",
                        }
                    ]
                },
            )

            with sqlite3.connect(changed) as connection:
                columns = list(connection.execute("PRAGMA table_info(orders)"))
                row = connection.execute(
                    "SELECT new_value, marker FROM orders"
                ).fetchone()
            self.assertEqual([column[1] for column in columns], ["marker", "new_value"])
            self.assertEqual(str(row[0]), "7")
            self.assertEqual(row[1], "kept")

    def test_add_column_detects_select_star_semantic_drift(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="driftsql-add-factory-",
            dir="/tmp",
        ) as temp_dir:
            root = Path(temp_dir)
            source = root / "retail.sqlite"
            with sqlite3.connect(source) as connection:
                connection.execute(
                    "CREATE TABLE orders "
                    "(order_id INTEGER PRIMARY KEY, total_amount REAL)"
                )
                connection.executemany(
                    "INSERT INTO orders VALUES (?, ?)",
                    [(1, 12.5), (2, 7.0)],
                )

            example = build_add_column_star_example(
                source="unit_test",
                source_index=3,
                db_id="retail",
                question="Show all order data.",
                evidence="",
                sql="SELECT * FROM orders ORDER BY order_id",
                database=source,
            )
            operation = example.schema_diff.operations[0]
            self.assertEqual(operation["type"], "add_column")
            self.assertEqual(example.stale_error, "silent_result_schema_mismatch")
            self.assertNotIn("*", example.repaired_sql)

            changed = root / "changed.sqlite"
            materialize_schema_diff(source, changed, example.schema_diff)
            with sqlite3.connect(changed) as connection:
                stale_rows = connection.execute(example.stale_sql).fetchall()
                repaired_rows = connection.execute(example.repaired_sql).fetchall()
            self.assertEqual(repaired_rows, [(1, 12.5), (2, 7.0)])
            self.assertEqual(stale_rows, [(1, 12.5, 0), (2, 7.0, 0)])

            with sqlite3.connect(source) as connection:
                columns = [
                    row[1]
                    for row in connection.execute("PRAGMA table_info(orders)")
                ]
            self.assertIn("total_amount", columns)

    def test_add_column_supports_multi_table_alias_wildcards(self) -> None:
        with tempfile.TemporaryDirectory(prefix="driftsql-add-multi-", dir="/tmp") as temp_dir:
            source = Path(temp_dir) / "retail.sqlite"
            with sqlite3.connect(source) as connection:
                connection.execute("CREATE TABLE customers (customer_id INTEGER, name TEXT)")
                connection.execute("CREATE TABLE orders (order_id INTEGER, customer_id INTEGER)")
                connection.execute("INSERT INTO customers VALUES (1, 'Ada')")
                connection.execute("INSERT INTO orders VALUES (10, 1)")
            example = build_add_column_projection_example(
                source="unit_test",
                source_index=9,
                db_id="retail",
                question="Return the complete customer and order contract.",
                evidence="",
                sql=(
                    "SELECT c.*, o.* FROM customers c JOIN orders o "
                    "ON c.customer_id = o.customer_id"
                ),
                database=source,
                added_column_specs=[
                    {"table": "customers", "new_name": "lineage_tag", "declared_type": "TEXT", "default_sql": "'new'"},
                    {"table": "orders", "new_name": "audit_flag", "declared_type": "INTEGER", "default_sql": "0"},
                ],
            )
            self.assertEqual(example.wildcard_profile, "multi_table_qualified")
            self.assertEqual(example.added_column_count, 2)
            self.assertNotIn("*", example.repaired_sql)
            changed = Path(temp_dir) / "changed.sqlite"
            materialize_schema_diff(source, changed, example.schema_diff)
            self.assertEqual(
                fingerprint_query(source, example.stale_sql),
                fingerprint_query(changed, example.repaired_sql),
            )

    def test_builds_table_rename_with_qualified_columns(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="driftsql-table-factory-",
            dir="/tmp",
        ) as temp_dir:
            root = Path(temp_dir)
            source = root / "retail.sqlite"
            with sqlite3.connect(source) as connection:
                connection.execute(
                    'CREATE TABLE "order items" '
                    '(item_id INTEGER PRIMARY KEY, amount REAL)'
                )
                connection.executemany(
                    'INSERT INTO "order items" VALUES (?, ?)',
                    [(1, 12.5), (2, 7.0)],
                )

            example = build_table_rename_example(
                source="unit_test",
                source_index=1,
                db_id="retail",
                question="Show item amounts.",
                evidence="",
                sql=(
                    'SELECT "order items".amount FROM "order items" '
                    'ORDER BY "order items".item_id'
                ),
                database=source,
            )

            operation = example.schema_diff.operations[0]
            self.assertEqual(operation["type"], "rename_table")
            self.assertNotIn('"order items"', example.repaired_sql)
            self.assertIn("no such table", example.stale_error)

            changed = root / "changed.sqlite"
            materialize_schema_diff(source, changed, example.schema_diff)
            with sqlite3.connect(changed) as connection:
                rows = connection.execute(example.repaired_sql).fetchall()
            self.assertEqual(rows, [(12.5,), (7.0,)])

    def test_builds_clean_negative_control(self) -> None:
        with tempfile.TemporaryDirectory(prefix="driftsql-clean-", dir="/tmp") as temp_dir:
            root = Path(temp_dir)
            source = root / "retail.sqlite"
            with sqlite3.connect(source) as connection:
                connection.execute("CREATE TABLE orders (order_id INTEGER PRIMARY KEY)")
                connection.executemany("INSERT INTO orders VALUES (?)", [(1,), (2,)])
            example = build_clean_example(
                source="unit_test",
                source_index=4,
                db_id="retail",
                question="Show orders.",
                evidence="",
                sql="SELECT order_id FROM orders ORDER BY order_id",
                database=source,
            )
            self.assertEqual(example.schema_diff.operations, ())
            self.assertEqual(example.stale_sql, example.repaired_sql)
            changed = root / "clean.sqlite"
            materialize_schema_diff(source, changed, example.schema_diff)
            with sqlite3.connect(changed) as connection:
                self.assertEqual(connection.execute(example.repaired_sql).fetchall(), [(1,), (2,)])

    def test_builds_and_materializes_compound_drift(self) -> None:
        with tempfile.TemporaryDirectory(prefix="driftsql-compound-test-", dir="/tmp") as temp_dir:
            root = Path(temp_dir)
            source = root / "retail.sqlite"
            with sqlite3.connect(source) as connection:
                connection.execute(
                    "CREATE TABLE orders (order_id INTEGER PRIMARY KEY, total_amount REAL)"
                )
                connection.executemany(
                    "INSERT INTO orders VALUES (?, ?)", [(1, 12.5), (2, 7.0)]
                )
            example = build_compound_drift_example(
                source="unit_test",
                source_index=5,
                db_id="retail",
                question="Show order totals.",
                evidence="",
                sql="SELECT total_amount FROM orders ORDER BY order_id",
                database=source,
            )
            self.assertEqual(len(example.schema_diff.operations), 2)
            changed = root / "compound.sqlite"
            materialize_schema_diff(source, changed, example.schema_diff)
            with sqlite3.connect(changed) as connection:
                rows = connection.execute(example.repaired_sql).fetchall()
            self.assertEqual(rows, [(12.5,), (7.0,)])


if __name__ == "__main__":
    unittest.main()
