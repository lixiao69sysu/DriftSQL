from __future__ import annotations

import unittest
from pathlib import Path

from driftsql.drift import rewrite_sql_identifier
from driftsql.environment import VersionedSQLite
from driftsql.smoke import run_smoke_episode


class IdentifierRewriteTest(unittest.TestCase):
    def test_rewrites_identifiers_but_not_string_literals(self) -> None:
        sql = "SELECT customer_id FROM customers WHERE note = 'customer_id' AND \"customer_id\" > 0"
        rewritten = rewrite_sql_identifier(sql, "customer_id", "client_id")
        self.assertEqual(
            rewritten,
            "SELECT client_id FROM customers WHERE note = 'customer_id' AND \"client_id\" > 0",
        )

    def test_rejects_unsafe_identifiers(self) -> None:
        with self.assertRaises(ValueError):
            rewrite_sql_identifier("SELECT id FROM users", "id", 'id"; DROP TABLE users; --')


class DriftRecoverySmokeTest(unittest.TestCase):
    def test_column_rename_recovery(self) -> None:
        result = run_smoke_episode()
        self.assertTrue(result["success"])
        self.assertIn("no such column", result["stale_error"])
        self.assertEqual(
            result["repaired_sql"],
            "SELECT client_id, name FROM customers WHERE annual_revenue > 10000",
        )
        self.assertEqual(result["result_rows"], ((1, "Ada"),))
        self.assertGreater(result["reward"]["total"], 0.0)

    def test_rejects_path_traversal_database_id(self) -> None:
        with self.assertRaises(ValueError):
            VersionedSQLite(Path("/tmp"), "../outside")


if __name__ == "__main__":
    unittest.main()
