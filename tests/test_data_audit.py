from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from driftsql.data import audit_mini_interact
from driftsql.environment import VersionedSQLite


class MiniInteractAuditTest(unittest.TestCase):
    def test_reports_public_data_as_valid_but_not_rl_ready(self) -> None:
        with tempfile.TemporaryDirectory(prefix="driftsql-data-", dir="/tmp") as temp_dir:
            root = Path(temp_dir)
            db_id = "retail"
            db_dir = root / db_id
            db_dir.mkdir()
            environment = VersionedSQLite(db_dir, db_id)
            database = environment.create(
                "source",
                ddl=["CREATE TABLE customers (customer_id INTEGER PRIMARY KEY)"],
                seed_sql=["INSERT INTO customers VALUES (1)"],
            )
            target = db_dir / f"{db_id}.sqlite"
            database.rename(target)
            (db_dir / f"{db_id}_template.sqlite").write_bytes(target.read_bytes())
            (db_dir / f"{db_id}_schema.txt").write_text("customers(customer_id)", encoding="utf-8")
            (db_dir / f"{db_id}_kb.jsonl").write_text("{}\n", encoding="utf-8")
            (db_dir / f"{db_id}_column_meaning_base.json").write_text("{}", encoding="utf-8")
            row = {
                "instance_id": "retail_1",
                "selected_database": db_id,
                "amb_user_query": "List customers.",
                "sol_sql": [],
                "test_cases": [],
            }
            (root / "mini_interact.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            report = audit_mini_interact(root)

            self.assertEqual(report["rows"], 1)
            self.assertEqual(report["selected_databases"], 1)
            self.assertFalse(report["ground_truth_complete"])
            self.assertFalse(report["ready_for_rl"])
            self.assertEqual(report["missing_assets"], [])
            self.assertEqual(report["invalid_databases"], [])


if __name__ == "__main__":
    unittest.main()
