from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from driftsql.contracts import Task
from driftsql.drift import ColumnRename, SchemaDiff
from driftsql.environment import VersionedSQLite
from driftsql.integrations import build_agentic_rl_record


class BirdRLRecordTest(unittest.TestCase):
    def test_builds_stateful_tool_record(self) -> None:
        with tempfile.TemporaryDirectory(prefix="driftsql-adapter-", dir="/tmp") as temp_dir:
            environment = VersionedSQLite(Path(temp_dir), "retail")
            db_path = environment.create(
                "v2",
                ddl=["CREATE TABLE customers (client_id INTEGER PRIMARY KEY, name TEXT)"],
                seed_sql=["INSERT INTO customers VALUES (1, 'Ada')"],
            )
            mutation = ColumnRename("customers", "customer_id", "client_id")
            diff = SchemaDiff("retail", "v1", "v2", (mutation.as_operation(),))
            task = Task(
                task_id="adapter-001",
                db_id="retail",
                db_version="v2",
                question="List customers.",
                metric_version="metrics-v3",
            )

            record = build_agentic_rl_record(
                task=task,
                prompt_messages=[{"role": "user", "content": task.question}],
                db_path=db_path,
                schema="customers(client_id, name)",
                schema_diff=diff,
                ground_truth_sql="SELECT client_id, name FROM customers",
                test_cases=["execution_match"],
                index=0,
                split="train",
            )

            self.assertEqual(record["data_source"], "driftsql_agentic")
            self.assertEqual(record["agent_name"], "tool_agent_with_db_cleanup")
            tools = record["extra_info"]["tools_kwargs"]
            self.assertEqual(
                set(tools),
                {
                    "get_schema_version",
                    "inspect_schema_diff",
                    "execute_sql",
                    "submit_solution",
                },
            )
            for tool in tools.values():
                create_kwargs = tool["create_kwargs"]
                self.assertEqual(create_kwargs["db_version"], "v2")
                self.assertEqual(create_kwargs["metric_version"], "metrics-v3")
                self.assertEqual(create_kwargs["db_path"], str(db_path.resolve()))


if __name__ == "__main__":
    unittest.main()
