from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from driftsql.environment import VersionedSQLite
from driftsql.drift import fingerprint_query

try:
    from verl.tools.schemas import OpenAIFunctionToolSchema

    from driftsql.integrations.verl_tools import InspectSchemaDiffTool, VersionedSqlExecutorTool

    VERL_AVAILABLE = True
except ImportError:
    VERL_AVAILABLE = False


@unittest.skipUnless(VERL_AVAILABLE, "VERL runtime dependencies are not installed")
class VersionedSqlExecutorToolTest(unittest.TestCase):
    def test_inspect_diff_returns_projection_contract_plan(self) -> None:
        with tempfile.TemporaryDirectory(prefix="driftsql-plan-tool-", dir="/tmp") as temp_dir:
            environment = VersionedSQLite(Path(temp_dir), "retail")
            db_path = environment.create(
                "v1",
                ddl=["CREATE TABLE orders (order_id INTEGER PRIMARY KEY, amount REAL)"],
                seed_sql=["INSERT INTO orders VALUES (1, 9.5)"],
            )
            schema = OpenAIFunctionToolSchema.model_validate(
                {
                    "type": "function",
                    "function": {
                        "name": "inspect_schema_diff",
                        "description": "Inspect an audited schema diff.",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            )
            tool = InspectSchemaDiffTool({}, schema)

            async def scenario() -> tuple:
                instance_id, _ = await tool.create(
                    instance_id="projection-1",
                    create_kwargs={
                        "db_id": "retail",
                        "source_db": str(db_path),
                        "stale_sql": "SELECT src.* FROM orders AS src ORDER BY src.rowid",
                        "schema_diff": {
                            "db_id": "retail",
                            "from_version": "v1",
                            "to_version": "v2",
                            "operations": [
                                {
                                    "type": "add_column",
                                    "table": "orders",
                                    "new_name": "audit_flag",
                                }
                            ],
                        },
                    },
                )
                response, _, metrics = await tool.execute(instance_id, {})
                await tool.release(instance_id)
                return response, metrics

            response, metrics = asyncio.run(scenario())
            payload = json.loads(response.text)
            plan = payload["projection_contract_plan"]
            repair = payload["repair_candidate"]
            self.assertEqual(plan["wildcard_profile"], "single_table_qualified")
            self.assertEqual(plan["excluded_added_columns"], ["orders.audit_flag"])
            self.assertIn('src."order_id"', plan["repaired_sql"])
            self.assertNotIn("audit_flag", plan["repaired_sql"])
            self.assertEqual(repair["repaired_sql"], plan["repaired_sql"])
            self.assertTrue(repair["changed"])
            self.assertTrue(metrics["projection_contract_planned"])
            self.assertTrue(metrics["repair_candidate_planned"])
            self.assertIsNone(metrics["projection_contract_error"])
            self.assertIsNone(metrics["repair_candidate_error"])

    def test_executes_select_and_denies_update(self) -> None:
        with tempfile.TemporaryDirectory(prefix="driftsql-verl-tool-", dir="/tmp") as temp_dir:
            environment = VersionedSQLite(Path(temp_dir), "retail")
            db_path = environment.create(
                "v1",
                ddl=["CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT)"],
                seed_sql=["INSERT INTO customers VALUES (1, 'Ada')"],
            )
            schema = OpenAIFunctionToolSchema.model_validate(
                {
                    "type": "function",
                    "function": {
                        "name": "execute_sql",
                        "description": "Execute read-only SQL.",
                        "parameters": {
                            "type": "object",
                            "properties": {"sql": {"type": "string"}},
                            "required": ["sql"],
                        },
                    },
                }
            )
            tool = VersionedSqlExecutorTool(
                {"timeout_seconds": 2, "max_rows": 10},
                schema,
            )

            async def scenario() -> tuple:
                instance_id, _ = await tool.create(
                    instance_id="trajectory-1",
                    create_kwargs={
                        "db_id": "retail",
                        "db_path": str(db_path),
                        "sync_io": True,
                    },
                )
                select_response, _, select_metrics = await asyncio.wait_for(
                    tool.execute(
                        instance_id,
                        {"sql": "SELECT customer_id, name FROM customers"},
                    ),
                    timeout=5,
                )
                update_response, _, update_metrics = await asyncio.wait_for(
                    tool.execute(
                        instance_id,
                        {"sql": "UPDATE customers SET name = 'Eve'"},
                    ),
                    timeout=5,
                )
                await tool.release(instance_id)
                return select_response, select_metrics, update_response, update_metrics

            select_response, select_metrics, update_response, update_metrics = asyncio.run(scenario())
            select_result = json.loads(select_response.text)
            update_result = json.loads(update_response.text)

            self.assertTrue(select_result["success"])
            self.assertEqual(select_result["rows"], [[1, "Ada"]])
            self.assertTrue(select_metrics["execution_success"])
            self.assertFalse(update_result["success"])
            self.assertFalse(update_metrics["execution_success"])

    def test_executor_exposes_boolean_contract_status_without_hashes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="driftsql-contract-tool-", dir="/tmp") as temp_dir:
            environment = VersionedSQLite(Path(temp_dir), "retail")
            db_path = environment.create(
                "v2",
                ddl=[
                    "CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT)"
                ],
                seed_sql=["INSERT INTO customers VALUES (1, 'Ada')"],
            )
            expected = fingerprint_query(
                db_path, "SELECT customer_id FROM customers ORDER BY customer_id"
            )
            schema = OpenAIFunctionToolSchema.model_validate(
                {
                    "type": "function",
                    "function": {
                        "name": "execute_sql",
                        "description": "Execute read-only SQL.",
                        "parameters": {
                            "type": "object",
                            "properties": {"sql": {"type": "string"}},
                            "required": ["sql"],
                        },
                    },
                }
            )
            tool = VersionedSqlExecutorTool(
                {"timeout_seconds": 2, "max_rows": 10}, schema
            )

            async def scenario() -> tuple:
                instance_id, _ = await tool.create(
                    instance_id="contract-1",
                    create_kwargs={
                        "db_id": "retail",
                        "db_path": str(db_path),
                        "sync_io": True,
                        "result_fingerprint": {
                            "row_count": expected.row_count,
                            "value_hash": expected.value_hash,
                        },
                    },
                )
                matching, _, matching_metrics = await tool.execute(
                    instance_id,
                    {"sql": "SELECT customer_id FROM customers ORDER BY customer_id"},
                )
                changed, _, changed_metrics = await tool.execute(
                    instance_id,
                    {"sql": "SELECT * FROM customers ORDER BY customer_id"},
                )
                await tool.release(instance_id)
                return matching, matching_metrics, changed, changed_metrics

            matching, matching_metrics, changed, changed_metrics = asyncio.run(scenario())
            matching_payload = json.loads(matching.text)
            changed_payload = json.loads(changed.text)

            self.assertTrue(matching_payload["result_contract_match"])
            self.assertTrue(matching_payload["validated_for_submit"])
            self.assertFalse(matching_payload["requires_schema_recovery"])
            self.assertFalse(changed_payload["result_contract_match"])
            self.assertFalse(changed_payload["validated_for_submit"])
            self.assertTrue(changed_payload["requires_schema_recovery"])
            self.assertTrue(matching_metrics["result_contract_checked"])
            self.assertFalse(changed_metrics["result_contract_match"])
            self.assertNotIn(expected.value_hash, matching.text)
            self.assertNotIn(expected.value_hash, changed.text)


if __name__ == "__main__":
    unittest.main()
