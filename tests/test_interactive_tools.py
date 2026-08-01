from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
from pathlib import Path

from verl.tools.schemas import OpenAIFunctionToolSchema

from driftsql.integrations.verl_tools import (
    AskUserTool,
    GetKnowledgeDefinitionTool,
    GetSchemaTool,
    VersionedSqlExecutorTool,
)


def schema(name: str, parameters: dict) -> OpenAIFunctionToolSchema:
    return OpenAIFunctionToolSchema.model_validate(
        {
            "type": "function",
            "function": {
                "name": name,
                "description": name,
                "parameters": parameters,
            },
        }
    )


def test_guarded_user_schema_and_hkb_tools_preserve_trajectory_state() -> None:
    async def scenario(root: Path) -> None:
        schema_path = root / "db_schema.txt"
        schema_path.write_text(
            "CREATE TABLE customers (id INTEGER, name TEXT);\n\n"
            "CREATE TABLE orders (id INTEGER, customer_id INTEGER, amount REAL);\n",
            encoding="utf-8",
        )
        knowledge_path = root / "db_kb.jsonl"
        knowledge_path.write_text(
            json.dumps(
                {
                    "knowledge": "Net Revenue",
                    "description": "Revenue after discounts",
                    "definition": "SUM(amount * (1 - discount))",
                    "type": "calculation_knowledge",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        state = {
            "db_id": "retail",
            "schema_path": str(schema_path),
            "knowledge_base_path": str(knowledge_path),
            "user_query_ambiguity": {
                "critical_ambiguity": [
                    {
                        "term": "active customer",
                        "sql_snippet": "last_order_date >= date('now', '-90 day')",
                        "type": "condition_ambiguity",
                    }
                ],
                "non_critical_ambiguity": [],
            },
        }

        ask = AskUserTool(
            {"max_questions": 3},
            schema(
                "ask_user",
                {
                    "type": "object",
                    "properties": {"question": {"type": "string"}},
                    "required": ["question"],
                },
            ),
        )
        get_schema = GetSchemaTool(
            {"max_chars": 2000},
            schema(
                "get_schema",
                {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            ),
        )
        get_hkb = GetKnowledgeDefinitionTool(
            {"max_results": 3},
            schema(
                "get_knowledge_definition",
                {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            ),
        )
        for tool in (ask, get_schema, get_hkb):
            await tool.create(instance_id="trajectory-1", create_kwargs=state)

        answer, _, answer_metrics = await ask.execute(
            "trajectory-1", {"question": "What does active customer mean?"}
        )
        duplicate, _, duplicate_metrics = await ask.execute(
            "trajectory-1", {"question": "Clarify active customer again"}
        )
        unknown, _, unknown_metrics = await ask.execute(
            "trajectory-1", {"question": "What is the CEO's phone number?"}
        )
        schema_response, _, schema_metrics = await get_schema.execute(
            "trajectory-1", {"query": "orders amount"}
        )
        hkb_response, _, hkb_metrics = await get_hkb.execute(
            "trajectory-1", {"name": "net revenue"}
        )

        assert answer_metrics["clarification_matched"]
        assert "90 day" in answer.text
        assert duplicate_metrics["duplicate_question"]
        assert unknown_metrics["unanswerable_question"]
        assert "orders" in json.loads(schema_response.text)["schema"]
        assert schema_metrics["schema_query_used"]
        assert json.loads(hkb_response.text)["matches"][0]["knowledge"] == "Net Revenue"
        assert hkb_metrics["knowledge_retrieved"]

        inline = GetKnowledgeDefinitionTool(
            {"max_results": 1},
            schema(
                "get_knowledge_definition",
                {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            ),
        )
        await inline.create(
            instance_id="trajectory-inline",
            create_kwargs={
                "db_id": "retail",
                "knowledge_entries": [
                    {"knowledge": "Gross Margin", "definition": "revenue - cost"}
                ],
            },
        )
        inline_response, _, inline_metrics = await inline.execute(
            "trajectory-inline", {"name": "gross margin"}
        )
        assert inline_metrics["knowledge_retrieved"]
        assert json.loads(inline_response.text)["matches"][0]["knowledge"] == "Gross Margin"
        await inline.release("trajectory-inline")

        for tool in (ask, get_schema, get_hkb):
            await tool.release("trajectory-1")

    with tempfile.TemporaryDirectory(prefix="driftsql-interactive-", dir="/tmp") as directory:
        asyncio.run(scenario(Path(directory)))


def test_sql_session_is_isolated_read_only_timed_and_rolled_back() -> None:
    async def scenario(root: Path) -> None:
        source = root / "retail.sqlite"
        with sqlite3.connect(source) as connection:
            connection.execute("CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT)")
            connection.execute("INSERT INTO customers VALUES (1, 'Ada')")

        tool = VersionedSqlExecutorTool(
            {"timeout_seconds": 0.01, "max_rows": 10, "isolate_existing_db": True},
            schema(
                "execute_sql",
                {
                    "type": "object",
                    "properties": {"sql": {"type": "string"}},
                    "required": ["sql"],
                },
            ),
        )
        instance_id, _ = await tool.create(
            instance_id="trajectory-2",
            create_kwargs={
                "db_id": "retail",
                "db_path": str(source),
                "isolate_db": True,
                "sync_io": True,
            },
        )
        active = Path(tool._state(instance_id)["db_path"])
        assert active != source.resolve()
        assert active.is_file()

        reused_id, _ = await tool.create(
            instance_id="trajectory-2",
            create_kwargs={
                "db_id": "retail",
                "db_path": str(source),
                "isolate_db": True,
                "sync_io": True,
            },
        )
        assert reused_id == instance_id
        assert Path(tool._state(instance_id)["db_path"]) == active

        selected, _, selected_metrics = await tool.execute(
            instance_id, {"sql": "SELECT id, name FROM customers"}
        )
        denied, _, denied_metrics = await tool.execute(
            instance_id, {"sql": "UPDATE customers SET name = 'Eve' WHERE id = 1"}
        )
        empty, _, empty_metrics = await tool.execute(instance_id, {"sql": ""})
        timed, _, timed_metrics = await tool.execute(
            instance_id,
            {
                "sql": (
                    "WITH RECURSIVE counter(x) AS (SELECT 1 UNION ALL "
                    "SELECT x + 1 FROM counter) SELECT SUM(x) FROM counter"
                )
            },
        )
        selected_payload = json.loads(selected.text)
        assert selected_payload["rows"] == [[1, "Ada"]]
        assert selected_payload["rolled_back"]
        assert selected_metrics["session_isolated"]
        assert selected_metrics["rolled_back"]
        assert not json.loads(denied.text)["success"]
        assert not denied_metrics["execution_success"]
        empty_payload = json.loads(empty.text)
        assert empty_payload["error"] == "No SQL provided"
        assert empty_payload["elapsed_ms"] == 0.0
        assert not empty_metrics["execution_success"]
        assert not json.loads(timed.text)["success"]
        assert not timed_metrics["execution_success"]

        with sqlite3.connect(source) as connection:
            assert connection.execute("SELECT name FROM customers WHERE id = 1").fetchone()[0] == "Ada"
        await tool.release(instance_id)
        assert not active.exists()

    with tempfile.TemporaryDirectory(prefix="driftsql-session-", dir="/tmp") as directory:
        asyncio.run(scenario(Path(directory)))
