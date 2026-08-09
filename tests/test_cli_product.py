from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console
from textual import events
from textual.containers import VerticalScroll
from textual.widgets import Input, RichLog, Static

from driftsql.cli.client import DriftSQLClient
from driftsql.cli.shell import DriftSQLShell
from driftsql.cli.tui import PROJECT_ROOT, DriftSQLTUI, PickerScreen


def render_static(app: DriftSQLTUI, selector: str, width: int) -> str:
    output = io.StringIO()
    console = Console(file=output, width=width, force_terminal=False)
    console.print(app.query_one(selector).content)
    return output.getvalue()


def test_welcome_card_and_transcript_share_one_scrollable_page() -> None:
    async def exercise() -> None:
        app = DriftSQLTUI(FakeClient(), close_client=False)
        async with app.run_test(size=(90, 24)) as pilot:
            await pilot.pause()
            welcome = app.query_one("#welcome", Static)
            transcript = app.query_one("#transcript", RichLog)
            page = app.query_one("#page-scroll", VerticalScroll)
            initial_screen_y = welcome.region.y - page.scroll_y

            for index in range(80):
                app._write_transcript(f"trajectory event {index}")
            await pilot.pause()

            assert page.scroll_y > 0
            assert transcript.scroll_y == 0
            assert welcome.region.y - page.scroll_y < initial_screen_y
            assert welcome.styles.dock == "none"

    import asyncio

    asyncio.run(exercise())


def test_mouse_wheel_scrolls_entire_page_from_anywhere() -> None:
    async def exercise() -> None:
        app = DriftSQLTUI(FakeClient(), close_client=False)
        async with app.run_test(size=(90, 24)) as pilot:
            await pilot.pause()
            welcome = app.query_one("#welcome", Static)
            transcript = app.query_one("#transcript", RichLog)
            page = app.query_one("#page-scroll", VerticalScroll)
            for index in range(80):
                transcript.write(f"trajectory event {index}")
            page.scroll_home(animate=False, immediate=True)
            await pilot.pause()
            assert page.scroll_y == 0

            welcome.post_message(
                events.MouseScrollDown(
                    welcome,
                    x=1,
                    y=1,
                    delta_x=0,
                    delta_y=1,
                    button=0,
                    shift=False,
                    meta=False,
                    ctrl=False,
                )
            )
            await pilot.pause()
            assert page.scroll_y > 0
            assert transcript.scroll_y == 0

    import asyncio

    asyncio.run(exercise())


def test_reward_breakdown_is_details_only_and_status_uses_elapsed_time() -> None:
    async def exercise() -> None:
        app = DriftSQLTUI(FakeClient(), close_client=False)
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            reward = {
                "score": 1.03,
                "reward_success": 1.0,
                "reward_valid": 0.1,
                "reward_efficient": 0.1,
                "penalty_tool_cost": 0.12,
                "penalty_unsafe": 0.05,
                "unsafe": True,
            }
            app._render_inspector(session={"usage": {"tool_calls": 4, "elapsed_ms": 2350}}, reward=reward)
            status = render_static(app, "#status-rule", 118)
            assert "2.4s" in status
            assert "reward" not in status.casefold()

            output = io.StringIO()
            console = Console(file=output, width=100, force_terminal=False)
            console.print(app._reward_breakdown(reward))
            details = output.getvalue()
            for component in ("success", "valid", "efficient", "cost", "unsafe"):
                assert component in details
            assert "+1.000" in details
            assert "-0.120" in details
            assert "-0.050" in details

    import asyncio

    asyncio.run(exercise())


def test_http_client_parses_models_and_sse_events() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/models":
            return httpx.Response(
                200,
                json={
                    "active_model_id": "grpo",
                    "models": [
                        {
                            "model_id": "grpo",
                            "display_name": "GRPO",
                            "category": "GRPO",
                            "base_model": "/models/base",
                            "adapter": "/models/grpo",
                            "adapter_sha256": "abc",
                            "available": True,
                            "active": True,
                            "metrics": {},
                            "notes": "",
                        }
                    ],
                },
            )
        if request.url.path.endswith("/events"):
            event = {
                "session_id": "s1",
                "sequence": 1,
                "event_type": "status",
                "created_at": "2026-08-09T00:00:00Z",
                "payload": {"status": "completed"},
            }
            return httpx.Response(200, text=f"id: 1\nevent: status\ndata: {json.dumps(event)}\n\n")
        if request.url.path == "/api/database-paths":
            return httpx.Response(
                200,
                json=[
                    {
                        "path": "@fake/orders/order_id",
                        "kind": "column",
                        "db_id": "fake",
                        "table": "orders",
                        "column": "order_id",
                    }
                ],
            )
        raise AssertionError(request.url)

    client = DriftSQLClient("http://test", transport=httpx.MockTransport(handler))
    try:
        assert client.models()["active_model_id"] == "grpo"
        assert client.database_paths()[0]["path"] == "@fake/orders/order_id"
        assert list(client.stream_events("s1"))[0]["event_type"] == "status"
    finally:
        client.close()


class FakeClient:
    def __init__(self) -> None:
        self.activated: str | None = None
        self.queries: list[tuple[str, str, str]] = []

    def models(self) -> dict[str, Any]:
        return {
            "active_model_id": self.activated or "base",
            "models": [
                {
                    "model_id": "base",
                    "display_name": "Base",
                    "category": "BASE",
                    "base_model": "/models/Qwen2.5-Coder-7B-Instruct",
                    "available": True,
                    "active": self.activated in {None, "base"},
                    "metrics": {"tune432_success_rate": 0.1},
                },
                {
                    "model_id": "grpo",
                    "display_name": "GRPO",
                    "category": "GRPO",
                    "base_model": "/models/Qwen2.5-Coder-7B-Instruct",
                    "available": True,
                    "active": self.activated == "grpo",
                    "metrics": {"tune432_success_rate": 0.8},
                },
            ],
        }

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "model": {"model_id": self.activated or "base", "backend": "scripted"}}

    def databases(self) -> list[dict[str, Any]]:
        return [
            {
                "db_id": "fake",
                "scenario_count": 2,
                "drift_types": ["add_column", "rename_column"],
            }
        ]

    def database_paths(self) -> list[dict[str, Any]]:
        return [
            {"path": "@fake", "kind": "database", "db_id": "fake"},
            {"path": "@fake/orders", "kind": "table", "db_id": "fake", "table": "orders"},
            {
                "path": "@fake/orders/order_id",
                "kind": "column",
                "db_id": "fake",
                "table": "orders",
                "column": "order_id",
                "data_type": "INTEGER",
            },
        ]

    def sessions(self, limit: int = 20) -> dict[str, Any]:
        return {
            "total": 1,
            "sessions": [
                {
                    "session_id": "previous-session",
                    "db_id": "fake",
                    "mode": "query",
                    "status": "completed",
                    "model": {"model_id": "grpo"},
                }
            ][:limit],
        }

    def activate_model(self, model_id: str) -> dict[str, Any]:
        self.activated = model_id
        return self.models()

    def experiments(self) -> dict[str, Any]:
        return {
            "selected_experiment_id": "grpo",
            "experiments": [
                {
                    "display_name": "GRPO",
                    "category": "GRPO",
                    "task_success_rate": 0.8,
                    "executable_rate": 0.9,
                    "average_model_calls": 3.0,
                    "average_tool_calls": 4.0,
                    "unsafe_tasks": 0,
                    "selected": True,
                }
            ],
        }

    def operations(self) -> dict[str, Any]:
        return {"total_sessions": 1, "terminal_sessions": 1, "success_rate": 1.0}

    def failures(self, failure_type: str | None = None) -> dict[str, Any]:
        return {"total": 0, "failures": [], "counts": {}}

    def wandb_runs(self) -> dict[str, Any]:
        return {"status": "disabled", "error": None, "runs": []}

    def replay_candidates(self, review_status: str | None = None) -> dict[str, Any]:
        return {"total": 0, "candidates": [], "counts": {}, "available": True}

    def create_query(self, db_id: str, question: str, locale: str) -> dict[str, Any]:
        self.queries.append((db_id, question, locale))
        return {"session_id": "s1"}

    def run_session(self, session_id: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"session_id": session_id, "status": "queued"}

    def stream_events(self, session_id: str):
        yield {
            "event_type": "tool",
            "payload": {"tool": "execute_sql", "arguments": {"sql": "SELECT 1"}, "success": True},
        }

    def trajectory(self, session_id: str) -> dict[str, Any]:
        return {
            "session": self.session(session_id),
            "events": [],
            "reward": {"total_reward": 1.0, "success_reward": 1.0},
        }

    def session(self, session_id: str) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "mode": "query",
            "status": "completed",
            "success": True,
            "termination_reason": "submitted",
            "final_sql": "SELECT 1",
            "usage": {"tool_calls": 2},
            "result": {
                "query_completed": True,
                "execution_result": {
                    "columns": ["result"],
                    "rows": [[1]],
                    "returned_rows": 1,
                    "truncated": False,
                    "elapsed_ms": 0.2,
                },
            },
        }


def test_shell_switches_models_and_sends_plain_text_as_database_query() -> None:
    output = io.StringIO()
    client = FakeClient()
    shell = DriftSQLShell(client, console=Console(file=output, force_terminal=False))  # type: ignore[arg-type]
    shell.current_db = "fake"
    assert shell.execute_line("/models use grpo") is True
    assert client.activated == "grpo"
    assert shell.execute_line("返回常量 1") is True
    assert client.queries == [("fake", "返回常量 1", "zh-CN")]
    rendered = output.getvalue()
    assert "已激活模型" in rendered
    assert "仅执行与安全提交" in rendered
    assert "查询结果" in rendered


def test_shell_exposes_budget_experiment_operations_wandb_and_replay_surfaces() -> None:
    output = io.StringIO()
    shell = DriftSQLShell(FakeClient(), console=Console(file=output, force_terminal=False))  # type: ignore[arg-type]
    for command in (
        "/budget max_turns=5 timeout_seconds=30",
        "/experiments",
        "/ops",
        "/failures",
        "/wandb",
        "/replay",
    ):
        assert shell.execute_line(command) is True
    assert shell.run_options["max_turns"] == 5
    rendered = output.getvalue()
    assert "统一实验对比" in rendered
    assert "运行指标" in rendered
    assert "W&B 状态" in rendered
    assert "Replay 候选" in rendered


def test_tui_bootstraps_full_screen_pickers_and_runs_a_query() -> None:
    async def exercise() -> None:
        client = FakeClient()
        app = DriftSQLTUI(client, close_client=False)  # type: ignore[arg-type]
        async with app.run_test(size=(150, 46)) as pilot:
            await pilot.pause(0.2)
            assert app.current_db == "fake"
            assert app.active_model_id == "base"
            assert "base" in str(app.query_one("#status-rule").render()).casefold()
            welcome = render_static(app, "#welcome", 140)
            assert "Core Workflow" in welcome
            assert "Qwen2.5-Coder-7B-Instruct · Base" in welcome
            assert str(PROJECT_ROOT) in welcome
            assert "DATABASE STORE" not in welcome
            assert "⣿" in welcome and "⣾" in welcome
            assert "/fake" not in welcome
            assert "DriftSQL Agent v0.1.0 · Qwen2.5-7B" not in welcome
            assert "DriftSQL Agent v1.0.1" in welcome

            composer = app.query_one("#composer", Input)
            composer.focus()
            composer.value = "/"
            await pilot.pause()
            assert app.command_matches
            assert app.query_one("#prompt-zone").has_class("visible")
            composer.value = ""

            composer.value = "统计 @ord"
            composer.cursor_position = len(composer.value)
            await pilot.pause()
            assert app.path_matches[0]["path"] == "@fake/orders"
            await pilot.press("tab")
            assert composer.value == "统计 @fake/orders "
            assert not app.query_one("#prompt-zone").has_class("visible")
            composer.value = ""

            app.database_payload.append({"db_id": "archive", "scenario_count": 1, "drift_types": []})
            archive_path = {"path": "@archive/logs", "kind": "table", "db_id": "archive", "table": "logs"}
            app.database_path_payload.append(archive_path)
            composer.value = "查看 @archive/log"
            composer.cursor_position = len(composer.value)
            app._insert_path_match(archive_path)
            assert composer.value == "查看 @archive/logs "
            assert app.current_db == "archive"
            app._select_database("fake")
            composer.value = ""

            await pilot.press("ctrl+m")
            await pilot.pause()
            assert isinstance(app.screen, PickerScreen)
            await pilot.press("escape")

            composer.focus()
            await pilot.click("#composer")
            await pilot.press(*"返回常量 1")
            await pilot.press("enter")
            await pilot.pause(0.3)
            assert client.queries == [("fake", "返回常量 1", "zh-CN")]
            assert app.busy is False
            assert app.current_session_id == "s1"
            assert len(app.query_one("#transcript", RichLog).lines) > 0

            composer.focus()
            composer.value = "第二条输入"
            await pilot.press("enter")
            await pilot.pause(0.3)
            composer.value = "尚未提交的草稿"
            composer.cursor_position = len(composer.value)
            await pilot.press("up")
            assert composer.value == "第二条输入"
            await pilot.press("up")
            assert composer.value == "返回常量 1"
            await pilot.press("down")
            assert composer.value == "第二条输入"
            await pilot.press("down")
            assert composer.value == "尚未提交的草稿"

            app.busy = True
            app._start_query("下一条排队任务")
            assert app.message_queue == ["下一条排队任务"]

    import asyncio

    asyncio.run(exercise())


def test_tui_medium_welcome_keeps_database_identity_without_full_catalog() -> None:
    async def exercise() -> None:
        app = DriftSQLTUI(FakeClient(), close_client=False)  # type: ignore[arg-type]
        async with app.run_test(size=(90, 24)) as pilot:
            await pilot.pause(0.2)
            welcome = render_static(app, "#welcome", 86)
            assert "Core Agent" in welcome
            assert "Qwen2.5-Coder-7B-Instruct · Base" in welcome
            assert str(PROJECT_ROOT) in welcome
            assert "DriftSQL Agent · Qwen2.5-7B" not in welcome
            assert "██████" in welcome
            assert "⣿" in welcome and "⣾" in welcome
            assert "workflow" in welcome
            assert "Available Tools" not in welcome
            assert "Data & Evaluation" not in welcome

    import asyncio

    asyncio.run(exercise())


def test_tui_welcome_reflows_when_terminal_resizes() -> None:
    async def exercise() -> None:
        app = DriftSQLTUI(FakeClient(), close_client=False)  # type: ignore[arg-type]
        async with app.run_test(size=(150, 46)) as pilot:
            await pilot.pause(0.2)
            assert "Core Workflow" in render_static(app, "#welcome", 140)
            await pilot.resize_terminal(82, 24)
            await pilot.pause(0.2)
            resized = render_static(app, "#welcome", 76)
            assert "Core Agent" in resized
            assert "Core Workflow" not in resized
            assert max(len(line) for line in resized.splitlines()) <= 76

    import asyncio

    asyncio.run(exercise())


def test_tui_uses_full_width_beyond_previous_156_column_cap() -> None:
    async def exercise() -> None:
        app = DriftSQLTUI(FakeClient(), close_client=False)  # type: ignore[arg-type]
        async with app.run_test(size=(190, 46)) as pilot:
            await pilot.pause(0.2)
            shell = app.query_one("#hermes-shell")
            page = app.query_one("#page-scroll")
            welcome = app.query_one("#welcome")
            assert shell.region.width == 190
            assert page.region.width >= 184
            assert welcome.region.width >= 183

    import asyncio

    asyncio.run(exercise())


def test_tui_input_history_persists_between_processes(tmp_path: Path) -> None:
    history_file = str(tmp_path / "history")
    first = DriftSQLTUI(FakeClient(), close_client=False, history_file=history_file)  # type: ignore[arg-type]
    first._remember_input("第一条历史输入")
    first._remember_input("第二条历史输入")

    second = DriftSQLTUI(FakeClient(), close_client=False, history_file=history_file)  # type: ignore[arg-type]
    assert second.input_history == ["第一条历史输入", "第二条历史输入"]
