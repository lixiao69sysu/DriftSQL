from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from driftsql.data import (
    audit_bird23_train,
    audit_bird_mini_dev,
    audit_six_gym_sqlite,
)


def _database(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE records (record_id INTEGER PRIMARY KEY)")
        connection.execute("INSERT INTO records VALUES (1)")


class BirdDataAuditTest(unittest.TestCase):
    def test_audits_open_training_and_evaluation_contracts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="driftsql-bird-data-", dir="/tmp") as temp_dir:
            root = Path(temp_dir)

            bird = root / "bird"
            bird_task = bird / "data" / "train-00000-of-00001.jsonl"
            bird_task.parent.mkdir(parents=True)
            bird_task.write_text(
                json.dumps(
                    {
                        "db_id": "retail",
                        "question": "List records.",
                        "evidence": "",
                        "SQL": "SELECT * FROM records",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            _database(
                bird
                / "full"
                / "train"
                / "train_databases"
                / "retail"
                / "retail.sqlite"
            )

            gym = root / "gym"
            gym.mkdir()
            (gym / "train.jsonl").write_text(
                json.dumps(
                    {
                        "instance_id": "TRAIN_1",
                        "db_id": "retail",
                        "query": "Fix this query.",
                        "issue_sql": ["SELECT missing FROM records"],
                        "sol_sql": ["SELECT record_id FROM records"],
                        "test_cases": ["def test_case(): return 1"],
                        "preprocess_sql": [],
                        "clean_up_sql": [],
                        "category": "Query",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            _database(gym / "database" / "retail" / "retail_template.sqlite")

            mini = root / "mini"
            mini_task = mini / "data" / "mini_dev_sqlite-00000-of-00001.json"
            mini_task.parent.mkdir(parents=True)
            mini_task.write_text(
                json.dumps(
                    [
                        {
                            "question_id": 1,
                            "db_id": "finance",
                            "question": "List records.",
                            "evidence": "",
                            "SQL": "SELECT * FROM records",
                            "difficulty": "simple",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            _database(
                mini
                / "full"
                / "dev_20240627"
                / "dev_databases"
                / "finance"
                / "finance.sqlite"
            )

            reports = [
                audit_bird23_train(bird, quick_check=True),
                audit_six_gym_sqlite(gym, quick_check=True),
                audit_bird_mini_dev(mini, quick_check=True),
            ]

            self.assertTrue(all(report["ready"] for report in reports))
            self.assertEqual([report["ground_truth_rows"] for report in reports], [1, 1, 1])
            self.assertEqual(reports[1]["test_case_rows"], 1)


if __name__ == "__main__":
    unittest.main()
