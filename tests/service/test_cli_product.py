from __future__ import annotations

import asyncio
from pathlib import Path

from driftsql.service.inference.backend import ScriptedModelBackend

from .test_product_service import service_client, wait_terminal


def test_model_registry_lists_and_activates_registered_checkpoint(tmp_path: Path) -> None:
    async def run() -> None:
        backend = ScriptedModelBackend()
        async with service_client(tmp_path, backend) as (_app, client):
            listed = (await client.get("/api/models")).json()
            assert {item["model_id"] for item in listed["models"]} >= {
                "base-qwen25-coder-7b",
                "strong-sft160",
                "grpo-step25-seed20260810",
            }
            response = await client.post(
                "/api/models/activate",
                json={"model_id": "grpo-step25-seed20260810"},
            )
            assert response.status_code == 200
            activated = response.json()
            assert activated["active_model_id"] == "grpo-step25-seed20260810"
            assert sum(item["active"] for item in activated["models"]) == 1
            health = (await client.get("/health")).json()
            assert health["model"]["model_id"] == "grpo-step25-seed20260810"

    asyncio.run(run())


def test_free_query_uses_isolated_database_and_execution_only_contract(tmp_path: Path) -> None:
    async def run() -> None:
        backend = ScriptedModelBackend()
        async with service_client(tmp_path, backend) as (_app, client):
            databases = (await client.get("/api/databases")).json()
            paths = (await client.get("/api/database-paths")).json()
            assert any(item["kind"] == "table" for item in paths)
            assert any(item["kind"] == "column" for item in paths)
            assert all(item["path"].startswith("@") for item in paths)
            assert all("/data/" not in item["path"] for item in paths)
            response = await client.post(
                "/api/query-sessions",
                json={
                    "db_id": databases[0]["db_id"],
                    "question": "返回常量 1。",
                    "locale": "zh-CN",
                },
            )
            assert response.status_code == 201
            session = response.json()
            assert session["mode"] == "query"
            assert session["sandbox_isolated"] is True
            assert session["result"]["verification_scope"] == "execution_only"
            await client.post(f"/api/sessions/{session['session_id']}/run", json={})
            final = await wait_terminal(client, session["session_id"])
            assert final["status"] == "completed"
            assert final["success"] is True
            assert final["result"]["query_completed"] is True
            assert final["result"]["semantic_verified"] is False
            execution_result = final["result"]["execution_result"]
            assert execution_result["columns"] == ["result"]
            assert execution_result["rows"] == [[1]]
            assert execution_result["returned_rows"] == 1
            assert execution_result["truncated"] is False
            assert execution_result["elapsed_ms"] >= 0
            trajectory = (await client.get(f"/api/sessions/{session['session_id']}/trajectory")).json()
            assert trajectory["reward"]["verification_scope"] == "execution_only"
            assert trajectory["reward"]["task_success"] is False
            tools = [event["payload"]["tool"] for event in trajectory["events"] if event["event_type"] == "tool"]
            assert tools[-2:] == ["execute_sql", "submit_solution"]
            assert tools[0] == "get_schema"
            assert "get_schema_version" not in tools

    asyncio.run(run())
