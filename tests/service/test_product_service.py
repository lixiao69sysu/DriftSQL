from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx

from driftsql.service import create_app
from driftsql.service.inference.backend import ScriptedModelBackend
from driftsql.service.settings import ServiceSettings

TERMINAL = {"completed", "failed", "cancelled", "timed_out", "budget_exhausted"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@asynccontextmanager
async def service_client(
    tmp_path: Path,
    backend: ScriptedModelBackend,
    **settings_overrides: Any,
) -> AsyncIterator[tuple[Any, httpx.AsyncClient]]:
    settings = ServiceSettings(
        environment="test",
        model_backend="scripted",
        repository_path=tmp_path / "repository.sqlite",
        temporary_root=tmp_path / "sandboxes",
        **settings_overrides,
    )
    app = create_app(settings, backend=backend)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            yield app, client


async def first_scenario(client: httpx.AsyncClient) -> dict[str, Any]:
    response = await client.get("/api/scenarios")
    response.raise_for_status()
    return response.json()[0]


async def create(client: httpx.AsyncClient, scenario_id: str) -> dict[str, Any]:
    response = await client.post("/api/sessions", json={"scenario_id": scenario_id})
    response.raise_for_status()
    return response.json()


async def wait_terminal(client: httpx.AsyncClient, session_id: str) -> dict[str, Any]:
    for _ in range(500):
        response = await client.get(f"/api/sessions/{session_id}")
        response.raise_for_status()
        session = response.json()
        if session["status"] in TERMINAL:
            return session
        await asyncio.sleep(0.01)
    raise AssertionError(f"Session did not terminate: {session_id}")


def test_add_column_api_executes_complete_real_tool_trajectory(tmp_path: Path) -> None:
    async def run() -> None:
        backend = ScriptedModelBackend()
        async with service_client(tmp_path, backend) as (app, client):
            health = (await client.get("/health")).json()
            assert health["status"] == "ready"
            assert health["max_concurrent_sessions"] == 2
            experiments = (await client.get("/api/experiments")).json()
            assert experiments["selected_experiment_id"] == "stage8_sft20_tune55"
            assert sum(item["selected"] for item in experiments["experiments"]) == 1

            scenario = await first_scenario(client)
            assert scenario["drift_type"] == "add_column"
            assert "ground_truth" not in json.dumps(scenario)
            session = await create(client, scenario["scenario_id"])
            assert session["sandbox_isolated"] is True
            assert session["model"]["loaded"] is True
            assert session["model"]["persistent"] is True
            assert app.state.tools._tools["execute_sql"].config["max_rows"] == 5

            run_response = await client.post(f"/api/sessions/{session['session_id']}/run", json={})
            assert run_response.status_code == 202
            final = await wait_terminal(client, session["session_id"])
            assert final["status"] == "completed"
            assert final["success"] is True
            assert final["result"]["task_success"] is True
            assert final["budget"]["max_turns"] == 7

            trajectory = (await client.get(f"/api/sessions/{session['session_id']}/trajectory")).json()
            tools = [event["payload"]["tool"] for event in trajectory["events"] if event["event_type"] == "tool"]
            assert tools == ["get_schema_version", "inspect_schema_diff", "execute_sql", "submit_solution"]
            assert trajectory["reward"]["task_success"] is True

            summary = (await client.get("/api/observability/summary")).json()
            assert summary["total_sessions"] == 1
            assert summary["terminal_sessions"] == 1
            assert summary["successful_sessions"] == 1
            assert summary["success_rate"] == 1
            assert summary["drift_metrics"] == [
                {
                    "drift_type": "add_column",
                    "sessions": 1,
                    "successful": 1,
                    "success_rate": 1,
                }
            ]
            assert summary["deployments"][0]["adapter_sha256"] == session["model"]["adapter_sha256"]
            wandb = (await client.get("/api/observability/wandb/runs")).json()
            assert wandb["configured"] is False
            assert wandb["status"] == "disabled"
            assert "api_key" not in json.dumps(wandb).lower()

            stream = await client.get(f"/api/sessions/{session['session_id']}/events")
            assert stream.status_code == 200
            assert "event: reward" in stream.text
            assert "event: status" in stream.text

            with sqlite3.connect(app.state.settings.repository_path) as connection:
                metadata = connection.execute(
                    "SELECT payload_json FROM sessions WHERE session_id=?", (session["session_id"],)
                ).fetchone()[0]
                event_count = connection.execute(
                    "SELECT COUNT(*) FROM events WHERE session_id=?", (session["session_id"],)
                ).fetchone()[0]
            persisted = json.loads(metadata)
            assert persisted["model"]["adapter_sha256"]
            assert persisted["db_version"] == "v2"
            assert event_count == len(trajectory["events"])

    asyncio.run(run())


def test_sessions_are_isolated_and_gpu_queue_is_bounded(tmp_path: Path) -> None:
    async def run() -> None:
        backend = ScriptedModelBackend(delay_seconds=0.03)
        async with service_client(tmp_path, backend, max_concurrent_sessions=2) as (app, client):
            scenario = await first_scenario(client)
            sessions = [await create(client, scenario["scenario_id"]) for _ in range(3)]
            assert len({session["sandbox_ref"] for session in sessions}) == 3
            executor = app.state.tools._tools["execute_sql"]
            database_paths = [executor._state(session["session_id"])["db_path"] for session in sessions]
            assert len(set(database_paths)) == 3
            assert all(Path(path).is_file() for path in database_paths)

            await asyncio.gather(
                *(client.post(f"/api/sessions/{session['session_id']}/run", json={}) for session in sessions)
            )
            finals = await asyncio.gather(*(wait_terminal(client, session["session_id"]) for session in sessions))
            assert all(session["success"] is True for session in finals)
            assert backend.max_active_generations == 2
            for session in sessions:
                trajectory = (await client.get(f"/api/sessions/{session['session_id']}/trajectory")).json()
                assert {event["session_id"] for event in trajectory["events"]} == {session["session_id"]}

    asyncio.run(run())


def test_write_sql_is_rejected_and_source_database_unchanged(tmp_path: Path) -> None:
    async def run() -> None:
        backend = ScriptedModelBackend()
        async with service_client(tmp_path, backend) as (app, client):
            scenario = await first_scenario(client)
            source_db = Path(app.state.catalog.create_kwargs(scenario["scenario_id"])["source_db"])
            before = sha256(source_db)
            writes = [
                "INSERT INTO TeamsSC(year, tmID) VALUES (9999, 'X')",
                "UPDATE TeamsSC SET W = 0",
                "DELETE FROM TeamsSC",
                "CREATE TABLE forbidden(id INTEGER)",
                "DROP TABLE TeamsSC",
                "ALTER TABLE TeamsSC ADD COLUMN forbidden INTEGER",
            ]
            backend.scripts[scenario["scenario_id"]] = [
                *({"name": "execute_sql", "arguments": {"sql": sql}} for sql in writes),
                {"name": "submit_solution", "arguments": {"sql": writes[1]}},
            ]
            session = await create(client, scenario["scenario_id"])
            await client.post(
                f"/api/sessions/{session['session_id']}/run",
                json={"max_total_tokens": 65536},
            )
            final = await wait_terminal(client, session["session_id"])
            trajectory = (await client.get(f"/api/sessions/{session['session_id']}/trajectory")).json()
            execute_events = [
                event
                for event in trajectory["events"]
                if event["event_type"] == "tool" and event["payload"]["tool"] == "execute_sql"
            ]
            assert len(execute_events) == len(writes)
            for execute in execute_events:
                observation = json.loads(execute["payload"]["observation"])
                assert observation["success"] is False
                assert observation["rolled_back"] is True
                assert "authoriz" in observation["error"].lower() or "readonly" in observation["error"].lower()
            assert final["success"] is False
            assert trajectory["reward"]["unsafe"] is True
            assert sha256(source_db) == before
            failures = (await client.get("/api/observability/failures?failure_type=unsafe")).json()
            assert failures["total"] == 1
            assert failures["counts"] == {"unsafe": 1}
            assert failures["failures"][0]["session_id"] == session["session_id"]
            assert failures["failures"][0]["failure_type"] == "unsafe"

    asyncio.run(run())


def test_budget_timeout_and_cancellation_terminate_sessions(tmp_path: Path) -> None:
    async def run_budget() -> None:
        async with service_client(tmp_path / "budget", ScriptedModelBackend()) as (_app, client):
            scenario = await first_scenario(client)
            session = await create(client, scenario["scenario_id"])
            await client.post(f"/api/sessions/{session['session_id']}/run", json={"max_turns": 1})
            final = await wait_terminal(client, session["session_id"])
            assert final["status"] == "budget_exhausted"
            assert final["termination_reason"] == "max_turns"

    async def run_timeout() -> None:
        backend = ScriptedModelBackend(delay_seconds=0.2)
        async with service_client(tmp_path / "timeout", backend) as (_app, client):
            scenario = await first_scenario(client)
            session = await create(client, scenario["scenario_id"])
            await client.post(f"/api/sessions/{session['session_id']}/run", json={"timeout_seconds": 0.03})
            final = await wait_terminal(client, session["session_id"])
            assert final["status"] == "timed_out"

    async def run_cancel() -> None:
        backend = ScriptedModelBackend(delay_seconds=0.2)
        async with service_client(tmp_path / "cancel", backend) as (_app, client):
            scenario = await first_scenario(client)
            session = await create(client, scenario["scenario_id"])
            await client.post(f"/api/sessions/{session['session_id']}/run", json={})
            response = await client.post(f"/api/sessions/{session['session_id']}/cancel")
            assert response.status_code == 200
            final = await wait_terminal(client, session["session_id"])
            assert final["status"] == "cancelled"
            assert final["cancellation_requested"] is True

    asyncio.run(run_budget())
    asyncio.run(run_timeout())
    asyncio.run(run_cancel())
