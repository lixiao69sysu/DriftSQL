"""Hermes-inspired terminal workbench for DriftSQL.

The UI keeps the terminal-native structure that makes Hermes pleasant to use:
an immutable transcript, a separate live activity lane, a bottom composer,
fuzzy slash completions, and compact overlays. DriftSQL-specific state is
mapped into that structure instead of being presented as a dashboard.
"""

from __future__ import annotations

import json
import os
import re
import shlex
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from prompt_toolkit.history import FileHistory
from rich.console import Group
from rich.json import JSON
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Input, Label, ListItem, ListView, RichLog, Static

from .client import DriftSQLApiError, DriftSQLClient

TERMINAL_STATES = {"completed", "failed", "cancelled", "timed_out", "budget_exhausted"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DRIFTSQL_LOGO = (
    "██████╗ ██████╗ ██╗███████╗████████╗███████╗ ██████╗ ██╗     ",
    "██╔══██╗██╔══██╗██║██╔════╝╚══██╔══╝██╔════╝██╔═══██╗██║     ",
    "██║  ██║██████╔╝██║█████╗     ██║   ███████╗██║   ██║██║     ",
    "██║  ██║██╔══██╗██║██╔══╝     ██║   ╚════██║██║▄▄ ██║██║     ",
    "██████╔╝██║  ██║██║██║        ██║   ███████║╚██████╔╝███████╗",
    "╚═════╝ ╚═╝  ╚═╝╚═╝╚═╝        ╚═╝   ╚══════╝ ╚══▀▀═╝ ╚══════╝",
)

DATABASE_ART = (
    " ⢷⣖⠦⢤⣀⣀     ⣀⣠⣤⣤⣤⣤⣤⣀⡀      ⣀⣠⣤⡾⠂",
    "⣄⣀⣙⣷⣄⡀⠙⢷⣄ ⢰⣟⠉⠁     ⠉⢙⣷⡄⣠⡾⠋⠉⣡⡾⠋ ⣠",
    "⠙⠳⣦⣀⠙⠿⠷⣦⣌⡻⣾⡏⠛⠳⢶⢶⢶⢶⠶⠛⢋⣽⣿⢭⣴⠶⠚⠋⣩⡽⠟⠉",
    "⢠⠤⠴⠛⠻⣶⣶⣤⣤⣽⣿⣇⢤⣪⡷⠛⠙⠳⣯⣢⣽⣿⣿⣿⣶⡶⠶⠿⠭⣄⣀⡄",
    "⠈⠉⠙⠒⢦⣤⡤⠤⠼⢿⣿⣿⣿⡑⠒⢒⣒⠒⠒⣹⣿⣽⣿⣓⣒⣲⣒⡚⠉⠁ ⠁",
    " ⢀⣤⠾⢋⣁⣀⣤⠴⠚⣿⡟⣿⡇ ⢿⣿⠇ ⣿⣿⣾⡯⣗⣦⡤⠬⣟⣶⣄⡀",
    " ⠘⠉⠉⠉⣠⡾⢁⣠⠾⢻⡇⢱⢿⣄  ⢀⣼⢿ ⢹⡗⠦⣌⢻⣄  ⠈⠋",
    "    ⣴⡯⠖⠉  ⢸⡇ ⠑⢝⢷⢴⢟⠕⠁ ⢸⡇  ⠙⠻⣧⡀",
    "    ⠁     ⠘⢷⣄⣀ ⠑⠔⠁⢀⣀⣴⠟⠁    ⠈",
    "            ⠈⠉⠛⠛⠛⠛⠋⠉",
)

MINI_DATABASE_ART = (
    "⠘⣿⡶⢦⣤⡀ ⢀⣤⡴⠶⠶⠶⣤⣀ ⢀⣀⣠⣴⣾⠃",
    "⣶⣾⠿⣦⣌⠳⣤⣿⣥⣄⣀⣀⣀⣤⢿⣧⠞⣁⣴⢿⣥⡴",
    "⣀⣙⡷⣮⣭⣛⣺⣿⣀⣽⡿⠿⣿⣄⣾⣿⣿⣭⣴⣾⡋⣀",
    "⠙⠛⠶⣬⣭⣭⢿⣿⣿⡻⢶⡶⠾⣻⣿⣿⣿⣷⣶⠛⠋⠛",
    " ⣴⣟⣭⣤⡶⢚⣿⣿⡇⠻⠿⢀⣿⢿⡿⢷⣶⠿⡷⣤⡀",
    " ⠁ ⣰⣟⠴⠛⣿⠸⣿⣄⣠⣾⡟⢸⡏⠳⢽⣦ ⠈",
    "  ⠘⠛⠁  ⢿⡄⠈⠻⡽⠋⢀⣼⠇  ⠙⠃",
    "        ⠙⠻⠶⠶⠞⠛⠁",
)

COMMANDS: tuple[tuple[str, str], ...] = (
    ("/new", "开始新的转录"),
    ("/db", "选择数据库"),
    ("/models", "选择或比较 Base / SFT / GRPO 模型"),
    ("/sessions", "恢复并查看历史会话"),
    ("/recover <场景ID>", "运行标准漂移恢复任务"),
    ("/scenarios", "查看当前数据库的漂移场景"),
    ("/trace [会话ID]", "回放完整 Agent 轨迹"),
    ("/reward [会话ID]", "查看 Reward 分解"),
    ("/budget [key=value]", "查看或修改运行预算"),
    ("/experiments", "查看统一实验对比"),
    ("/ops", "查看服务运行指标"),
    ("/failures [类型]", "查看失败样本"),
    ("/wandb [run_id]", "查看训练指标"),
    ("/replay", "查看 Hard Replay 候选"),
    ("/queue", "查看等待发送的指令"),
    ("/details", "切换推理与工具详情"),
    ("/status", "显示当前运行状态"),
    ("/cancel", "取消当前会话"),
    ("/clear", "清空当前转录"),
    ("/help", "打开命令面板"),
    ("/quit", "退出 DriftSQL"),
)


def _pct(value: Any) -> str:
    return f"{float(value) * 100:.1f}%" if value is not None else "—"


def _compact(value: Any, limit: int = 640) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, indent=2)
    return text if len(text) <= limit else f"{text[:limit]}\n…"


def _intro_listing(title: str, rows: Iterable[tuple[str, str]]) -> Group:
    lines: list[Text] = [Text(title, style="bold #58a6ff")]
    for label, value in rows:
        lines.append(Text.assemble((f"{label}: ", "#2878bd"), (value, "#e6f1ff")))
    return Group(*lines)


def _database_art(lines: Iterable[str]) -> Group:
    """Render the Braille dot-matrix emblem with cyan pixel highlights."""
    rendered: list[Text] = []
    for line in lines:
        text = Text(line, style="#2f81c7")
        for marker in ("⣿", "⣾", "⣷", "⡿", "⢿"):
            start = 0
            while (index := line.find(marker, start)) >= 0:
                text.stylize("bold #22d3ee", index, index + len(marker))
                start = index + len(marker)
        rendered.append(text)
    return Group(*rendered)


class PickerItem(ListItem):
    """A list row carrying the original domain payload."""

    def __init__(self, key: str, title: str, meta: str = "", payload: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.key = key
        self.title = title
        self.meta = meta
        self.payload = payload or {}

    def compose(self) -> ComposeResult:
        yield Label(f"{self.title}\n[dim]{self.meta}[/dim]", markup=True)


class PickerScreen(ModalScreen[str | None]):
    """Searchable Hermes-style overlay used for models, DBs and sessions."""

    BINDINGS = [
        Binding("escape", "cancel", "关闭", show=False, priority=True),
    ]

    DEFAULT_CSS = """
    PickerScreen { align: center middle; background: #0b1220 72%; }
    PickerScreen > Vertical {
        width: 78; max-width: 92%; height: 76%;
        border: round #2f81c7; background: #0b1220; padding: 1 2;
    }
    PickerScreen .picker-title { color: #58a6ff; text-style: bold; height: 2; }
    PickerScreen .picker-hint { color: #7f93a8; height: 2; }
    PickerScreen Input { border: tall #234a73; background: #0b1220; margin-bottom: 1; color: #e6f1ff; }
    PickerScreen Input:focus { border: tall #4aa8ff; }
    PickerScreen ListView { height: 1fr; background: #0b1220; }
    PickerScreen ListItem { padding: 0 1; height: 3; }
    PickerScreen ListItem.--highlight { background: #132a44; color: #e6f1ff; }
    """

    def __init__(self, title: str, hint: str, items: Iterable[PickerItem]) -> None:
        super().__init__()
        self.picker_title = title
        self.hint = hint
        self.all_items = list(items)

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(self.picker_title, classes="picker-title")
            yield Static(self.hint, classes="picker-hint")
            yield Input(placeholder="输入关键词筛选…", id="picker-filter")
            yield ListView(*self.all_items, id="picker-list")

    def on_mount(self) -> None:
        self.query_one("#picker-filter", Input).focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_input_changed(self, event: Input.Changed) -> None:
        needle = event.value.casefold().strip()
        matches = [
            item
            for item in self.all_items
            if not needle or needle in f"{item.key} {item.title} {item.meta}".casefold()
        ]
        view = self.query_one("#picker-list", ListView)
        view.clear()
        view.extend(matches)

    def on_input_submitted(self, _: Input.Submitted) -> None:
        view = self.query_one("#picker-list", ListView)
        if view.highlighted_child is not None:
            view.action_select_cursor()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item = event.item
        if isinstance(item, PickerItem):
            self.dismiss(item.key)


class HelpScreen(ModalScreen[None]):
    BINDINGS = [Binding("escape", "close", "关闭", show=False), Binding("q", "close", "关闭", show=False)]
    DEFAULT_CSS = """
    HelpScreen { align: center middle; background: #0b1220 72%; }
    HelpScreen > Vertical {
        width: 86; max-width: 94%; height: 84%; border: round #2f81c7;
        background: #0b1220; padding: 1 2;
    }
    HelpScreen .help-title { color: #58a6ff; text-style: bold; height: 2; }
    HelpScreen RichLog { background: #0b1220; scrollbar-color: #234a73; }
    """

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("DRIFTSQL · 命令面板", classes="help-title")
            yield RichLog(id="help-log", wrap=True, markup=True)

    def on_mount(self) -> None:
        table = Table.grid(padding=(0, 3))
        table.add_column(style="bold #58a6ff", no_wrap=True)
        table.add_column(style="#c4d7ea")
        for command, description in COMMANDS:
            table.add_row(command, description)
        table.add_row("@数据库/表/字段", "搜索并插入只读数据库 Schema 路径")
        table.add_section()
        table.add_row("Enter", "发送 · 方向键浏览选择器 · Esc 关闭浮层")
        table.add_row("Ctrl+P", "命令面板 · Ctrl+M 模型 · Ctrl+D 数据库 · Ctrl+K 会话")
        table.add_row("Ctrl+C", "运行中取消；空闲时退出")
        self.query_one("#help-log", RichLog).write(table)

    def action_close(self) -> None:
        self.dismiss(None)


class DriftSQLTUI(App[None]):
    """A full-screen database-agent console backed by the service API."""

    TITLE = "DriftSQL"
    SUB_TITLE = "Agentic SQL Workbench"
    ENABLE_COMMAND_PALETTE = False

    BINDINGS = [
        Binding("ctrl+p", "help", "命令", priority=True),
        Binding("ctrl+m", "models", "模型", priority=True),
        Binding("alt+p", "models", "模型", priority=True),
        Binding("ctrl+d", "databases", "数据库", priority=True),
        Binding("ctrl+k", "sessions", "会话", priority=True),
        Binding("ctrl+l", "redraw", "重绘", priority=True),
        Binding("ctrl+o", "toggle_details", "详情", priority=True),
        Binding("ctrl+c", "cancel_or_quit", "取消", priority=True),
        Binding("escape", "interrupt", "中断"),
        Binding("f1", "help", "帮助", priority=True),
    ]

    CSS = """
    Screen { background: #0b1220; color: #e6f1ff; }
    #hermes-shell {
        width: 100%; height: 100%;
        align-horizontal: center; padding: 0 2;
        background: #0b1220;
    }
    #page-scroll {
        width: 100%;
        height: 1fr;
        overflow-x: hidden;
        overflow-y: auto;
        background: #0b1220;
        scrollbar-color: #234a73;
        scrollbar-background: #0b1220;
        scrollbar-size-horizontal: 0;
        scrollbar-size-vertical: 1;
    }
    #welcome {
        width: 100%;
        height: auto;
        padding: 0 1;
        background: #0b1220;
    }
    #transcript {
        width: 100%;
        height: auto;
        min-height: 1;
        padding: 0 1;
        overflow: hidden;
        background: #0b1220;
        scrollbar-size-horizontal: 0;
        scrollbar-size-vertical: 0;
    }
    #prompt-zone {
        display: none; height: auto; max-height: 8; margin: 0 1;
        border: round #234a73; background: #0f1b2d; padding: 0 1;
    }
    #prompt-zone.visible { display: block; }
    #status-rule {
        height: 1; padding: 0 1; color: #7f93a8;
        content-align: left middle;
    }
    #composer-row { height: 1; margin: 0 1; }
    #prompt-glyph { width: 3; color: #58a6ff; text-style: bold; content-align: left middle; }
    #composer { border: none; background: #0b1220; height: 1; padding: 0; color: #e6f1ff; }
    #composer:focus { border: none; }
    #help-hint {
        height: 1; padding: 0 4; color: #64748b;
        content-align: left middle;
    }
    #runtime-data { display: none; }
    """

    def __init__(
        self,
        client: DriftSQLClient,
        *,
        locale: str = "zh-CN",
        close_client: bool = True,
        history_file: str | None = None,
    ) -> None:
        super().__init__()
        self.client = client
        self.locale = locale
        self.close_client = close_client
        self.current_db: str | None = None
        self.current_session_id: str | None = None
        self.active_model_id = "—"
        self.root_path = str(PROJECT_ROOT)
        self.model_payload: dict[str, Any] = {"models": []}
        self.database_payload: list[dict[str, Any]] = []
        self.database_path_payload: list[dict[str, Any]] = []
        self.session_payload: list[dict[str, Any]] = []
        self.busy = False
        self.details_expanded = False
        self.last_reward: dict[str, Any] = {}
        self.last_session: dict[str, Any] = {}
        self.message_queue: list[str] = []
        self.command_matches: list[tuple[str, str]] = []
        self.path_matches: list[dict[str, Any]] = []
        self.command_index = 0
        self._history_store = FileHistory(history_file) if history_file else None
        self.input_history = self._load_input_history()
        self.history_index: int | None = None
        self.history_draft = ""
        self._suppress_completion_once = False
        self._viewport_width = 0
        self._viewport_height = 0
        self.activity_status = "正在连接服务"
        self.activity_tone = "warn"
        self._idle_interrupt_armed = False
        self.run_options: dict[str, Any] = {
            "max_turns": 7,
            "timeout_seconds": 120,
            "max_tool_calls": 7,
            "max_new_tokens": 512,
            "max_total_tokens": 32768,
        }

    def _load_input_history(self) -> list[str]:
        if self._history_store is None:
            return []
        try:
            return list(reversed(list(self._history_store.load_history_strings())))[-200:]
        except OSError:
            return []

    def compose(self) -> ComposeResult:
        with Vertical(id="hermes-shell"):
            with VerticalScroll(id="page-scroll"):
                yield Static(id="welcome")
                yield RichLog(
                    id="transcript",
                    min_width=1,
                    wrap=True,
                    highlight=True,
                    markup=False,
                    auto_scroll=False,
                )
            yield Static(id="prompt-zone")
            yield Static(id="status-rule")
            with Horizontal(id="composer-row"):
                yield Static("❯", id="prompt-glyph")
                yield Input(placeholder='试试“检查字段漂移并返回每个地区的活跃客户数”', id="composer")
            yield Static("↑↓ 历史  ·  @ 数据库路径  ·  / 命令  ·  Ctrl+O 详情  ·  Ctrl+C 中断", id="help-hint")
            yield Static(id="runtime-data")

    def on_mount(self) -> None:
        self._set_status("正在连接服务", tone="warn")
        self._render_context()
        self._render_inspector()
        self.bootstrap()
        self.query_one("#composer", Input).focus()

    def on_unmount(self) -> None:
        if self.close_client:
            self.client.close()

    def on_resize(self, event: events.Resize) -> None:
        self._viewport_width = event.size.width
        self._viewport_height = event.size.height
        self._update_responsive_classes(event.size.width)
        if self.model_payload.get("models"):
            self._append_welcome()
        self._render_context()

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        """Let the wheel scroll the entire page from anywhere in the shell."""
        if event.ctrl or event.shift:
            return
        self.query_one("#page-scroll", VerticalScroll).scroll_relative(
            y=self.scroll_sensitivity_y,
            animate=False,
            immediate=True,
        )
        event.stop()

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        """Let the wheel scroll the entire page from anywhere in the shell."""
        if event.ctrl or event.shift:
            return
        self.query_one("#page-scroll", VerticalScroll).scroll_relative(
            y=-self.scroll_sensitivity_y,
            animate=False,
            immediate=True,
        )
        event.stop()

    def _update_responsive_classes(self, width: int) -> None:
        del width

    @work(thread=True, exclusive=True, group="bootstrap")
    def bootstrap(self) -> None:
        try:
            health = self.client.health()
            models = self.client.models()
            databases = self.client.databases()
            sessions = self.client.sessions(limit=20)
        except Exception as error:  # the UI must remain usable enough to show connection failures
            self.call_from_thread(self._show_error, "服务连接失败", error)
            return
        try:
            database_paths = self.client.database_paths()
        except Exception:
            database_paths = [
                {"path": f"@{item['db_id']}", "kind": "database", "db_id": item["db_id"]}
                for item in databases
            ]
        self.call_from_thread(self._apply_bootstrap, health, models, databases, database_paths, sessions)

    def _apply_bootstrap(
        self,
        health: dict[str, Any],
        models: dict[str, Any],
        databases: list[dict[str, Any]],
        database_paths: list[dict[str, Any]],
        sessions: dict[str, Any],
    ) -> None:
        self.model_payload = models
        self.database_payload = databases
        self.database_path_payload = database_paths
        self.session_payload = sessions.get("sessions", [])
        self.active_model_id = models.get("active_model_id") or health.get("model", {}).get("model_id") or "—"
        if not self.current_db and databases:
            self.current_db = databases[0]["db_id"]
        self._append_welcome()
        self._populate_sidebar()
        self._render_context()
        self._render_inspector()
        self._set_status("就绪", tone="ok")

    def _populate_sidebar(self) -> None:
        """The Claude-style shell exposes these collections through overlays."""

    def _append_welcome(self) -> None:
        width = self._viewport_width or self.size.width
        height = self._viewport_height or self.size.height
        if width >= 100 and height >= 28:
            self._append_wide_welcome()
            return
        if width >= 78 and height >= 20:
            self._append_medium_welcome()
            return

        # Reserve room for the RichLog's vertical scrollbar. Rich otherwise
        # treats the right rule as one word and wraps the whole segment.
        width = max(34, min(94, width - 12))
        name = "DriftSQL Agent"
        name_width = len(name) + 2
        remaining = max(2, width - name_width)
        left = remaining // 2
        right = remaining - left
        header = f"{'─' * left} {name} {'─' * right}"
        welcome = Group(
            Text(header, style="#58a6ff"),
            Text("Schema Drift Recovery · Execution-verified SQL".center(width), style="#7f93a8"),
            Text("─" * width, style="#234a73"),
            Text(""),
            Text.assemble(
                ("  ◈  model      ", "#4aa8ff"),
                (self._clip_label(self._active_model_name(), max(16, width - 18)), "#c4d7ea"),
            ),
            Text.assemble(
                ("  ◈  root       ", "#4aa8ff"),
                (self._clip_label(self.root_path, max(16, width - 18)), "#c4d7ea"),
            ),
            Text.assemble(("  ◈  database   ", "#4aa8ff"), (self.current_db or "loading…", "#c4d7ea")),
            Text.assemble(("  ◈  sandbox    ", "#4aa8ff"), ("isolated SQLite · read-only", "#78c6a3")),
            Text.assemble(("  ◈  tools      ", "#4aa8ff"), ("7-tool dynamic policy · safe auto-submit", "#c4d7ea")),
            Text(""),
            Text("  输入数据库任务；键入 / 可搜索所有命令。", style="#91a7bd"),
        )
        self.query_one("#welcome", Static).update(welcome)

    def _append_medium_welcome(self) -> None:
        width = self._viewport_width or self.size.width
        content_width = max(58, width - 10)
        art_width = max(20, min(26, content_width // 3))
        left = _database_art(MINI_DATABASE_ART)
        right = _intro_listing(
            "Core Agent",
            (
                ("model", self._active_model_name()),
                ("root", self.root_path),
                ("database", self._clip_label(self.current_db or "select-database", 28)),
                ("workflow", "diff → schema → SQL → submit"),
                ("recovery", "rename · add · drop · type · compound"),
                ("safety", "read-only · timeout · rollback"),
            ),
        )
        body = Table.grid(expand=True, padding=(0, 2))
        body.add_column(width=art_width, no_wrap=True)
        body.add_column(ratio=1)
        body.add_row(left, right)
        panel = Panel(
            body,
            title="DriftSQL Agent",
            title_align="right",
            border_style="#2878bd",
            padding=(0, 1),
            expand=True,
        )
        welcome = Group(
            Group(*(Text(line, style="bold #79c0ff") for line in DRIFTSQL_LOGO)),
            Text("Schema Drift Recovery · Execution-verified SQL", style="#7f93a8"),
            panel,
            Text("输入数据库任务；键入 @ 引用数据库路径，/ 查看更多命令。", style="#91a7bd"),
        )
        self.query_one("#welcome", Static).update(welcome)

    def _append_wide_welcome(self) -> None:
        width = self._viewport_width or self.size.width
        content_width = max(80, width - 10)
        left_width = max(34, min(46, content_width * 2 // 5))
        recent_session = self.session_payload[0].get("session_id") if self.session_payload else None
        session_label = str(self.current_session_id or recent_session or "ready")[:20]
        logo = Group(*(Text(line, style="bold #79c0ff") for line in DRIFTSQL_LOGO))
        left = Group(
            _database_art(DATABASE_ART),
            Text(""),
            Text(self._active_model_name(), style="bold #58a6ff"),
            Text.assemble(("root: ", "#2878bd"), (self.root_path, "#7f93a8")),
            Text.assemble(("Session: ", "#64748b"), (session_label, "#7f93a8")),
        )
        workflow = _intro_listing(
            "Core Workflow",
            (
                ("agent", "inspect diff → schema → clarify if needed → execute → submit"),
                ("policy", "7-tool dynamic action mask · safe auto-submit"),
            ),
        )
        recovery = _intro_listing(
            "Recovery & Safety",
            (
                ("drift", "rename · add · drop · type-change · compound"),
                ("interaction", "must-ask detection · clarification recovery"),
                ("sandbox", "read-only authorizer · timeout · rollback"),
            ),
        )
        evaluation = _intro_listing(
            "Model & Evaluation",
            (
                ("training", "Recovery SFT → Full-episode GRPO"),
                ("data", "2,400 on-policy rollouts · 1,066 mined failures"),
                ("gate", "AddColumn72 → Tune432 → Fresh Blind320"),
            ),
        )
        right = Group(
            workflow,
            Text(""),
            recovery,
            Text(""),
            evaluation,
            Text("7 tools · /help for commands", style="#3d78a8"),
        )
        body = Table.grid(expand=True, padding=(0, 2))
        body.add_column(width=left_width)
        body.add_column(ratio=1)
        body.add_row(left, right)
        panel = Panel(
            body,
            title="DriftSQL Agent v1.0.1",
            title_align="right",
            border_style="#2878bd",
            padding=(0, 1),
            expand=True,
        )
        welcome = Group(
            logo,
            panel,
            Text("Welcome to DriftSQL Agent! Type a database instruction or /help for commands.", style="#e6f1ff"),
        )
        self.query_one("#welcome", Static).update(welcome)

    def _render_context(self) -> None:
        busy = "running" if self.busy else "ready"
        model_label = self._active_model_name()
        usage = self.last_session.get("usage", {})
        elapsed_ms = float(usage.get("elapsed_ms") or 0.0)
        elapsed = f" │ {elapsed_ms / 1000:.1f}s" if elapsed_ms > 0 else ""
        database_label = self.current_db or "未选择数据库"
        width = max(24, (self._viewport_width or self.size.width) - 6)
        if width < 76:
            database_label = self._clip_label(database_label, 16)
            model_label = self._clip_label(model_label, 17)
            left = f"─ {busy} │ {model_label} │ {usage.get('tool_calls', 0)}/{self.run_options['max_tool_calls']} tools"
            right = database_label
        elif width < 112:
            database_label = self._clip_label(database_label, 22)
            model_label = self._clip_label(model_label, 26)
            queue = f" │ q{len(self.message_queue)}" if self.message_queue else ""
            left = (
                f"─ {busy} │ {model_label} │ {usage.get('tool_calls', 0)}/{self.run_options['max_tool_calls']} tools"
                f"{elapsed}{queue}"
            )
            right = f"ro · {database_label}"
        else:
            queue = f" │ queue {len(self.message_queue)}" if self.message_queue else ""
            left = (
                f"─ {busy} │ {model_label} │ tools {usage.get('tool_calls', 0)}/{self.run_options['max_tool_calls']}"
                f"{elapsed}{queue}"
            )
            session = f" │ {self.current_session_id[:8]}" if self.current_session_id else ""
            right = f"readonly · {database_label}{session}"
        gap = max(1, width - len(left) - len(right) - 3)
        text = Text()
        text.append("─ ", style="#234a73")
        text.append(left[2:].split(" │ ", 1)[0], style="#78c6a3" if not self.busy else "#58a6ff")
        tail = left[2 + len(left[2:].split(" │ ", 1)[0]) :]
        text.append(tail, style="#7f93a8")
        text.append(" " * gap, style="#234a73")
        text.append("─ ", style="#234a73")
        text.append(right, style="#7f93a8")
        self.query_one("#status-rule", Static).update(text)

    @staticmethod
    def _clip_label(value: str, width: int) -> str:
        return value if len(value) <= width else f"{value[: max(1, width - 1)]}…"

    def _active_model_name(self) -> str:
        model = next(
            (
                item
                for item in self.model_payload.get("models", [])
                if item.get("model_id") == self.active_model_id or item.get("active")
            ),
            None,
        )
        if not model:
            return self.active_model_id
        base_name = Path(str(model.get("base_model") or "")).name
        display_name = str(model.get("display_name") or model.get("model_id") or self.active_model_id)
        if base_name and base_name.casefold() not in display_name.casefold():
            return f"{base_name} · {display_name}"
        return display_name

    def _render_inspector(self, session: dict[str, Any] | None = None, reward: dict[str, Any] | None = None) -> None:
        if session is not None:
            self.last_session = session
        if reward is not None:
            self.last_reward = reward
        self.query_one("#runtime-data", Static).update(
            json.dumps({"session": self.last_session, "reward": self.last_reward}, ensure_ascii=False)
        )
        self._render_context()

    @staticmethod
    def _reward_breakdown(reward: dict[str, Any]) -> Table:
        """Build the compact experiment-only reward view used by Ctrl+O."""

        def number(*keys: str) -> float | None:
            for key in keys:
                value = reward.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    return float(value)
            return None

        def contribution(value: float | None, fallback: str = "—") -> Text:
            if value is None:
                return Text(fallback, style="#7f93a8")
            style = "#78c6a3" if value >= 0 else "#ff6b6b"
            return Text(f"{value:+.3f}", style=style)

        success = number("reward_success", "success_reward")
        if success is None and "query_completed" in reward:
            success = number("score")
        valid = number("reward_valid", "valid_reward")
        efficient = number("reward_efficient", "efficient_reward")

        cost_keys = [
            key
            for key in reward
            if key.startswith("penalty_") and key != "penalty_unsafe"
        ]
        cost_values = [number(key) for key in cost_keys]
        cost = -sum(abs(value) for value in cost_values if value is not None) if cost_keys else number("cost")
        unsafe = number("penalty_unsafe", "unsafe_penalty")
        if unsafe is not None:
            unsafe = -abs(unsafe)

        total = number("score", "total_reward", "reward")
        total_label = f"{total:+.3f}" if total is not None else "—"
        table = Table(
            title=f"Reward 详情 · total {total_label}",
            title_style="bold #79c0ff",
            header_style="bold #58a6ff",
            border_style="#234a73",
            show_lines=False,
            expand=True,
        )
        for label in ("success", "valid", "efficient", "cost", "unsafe"):
            table.add_column(label, justify="center", ratio=1)
        valid_fallback = "✓" if reward.get("execution_success") else "×" if "execution_success" in reward else "—"
        unsafe_fallback = "unsafe" if reward.get("unsafe") else "safe" if "unsafe" in reward else "—"
        table.add_row(
            contribution(success),
            contribution(valid, valid_fallback),
            contribution(efficient),
            contribution(cost),
            contribution(unsafe, unsafe_fallback),
        )
        return table

    def _set_status(self, status: str, *, tone: str = "normal") -> None:
        self.activity_status = status
        self.activity_tone = tone
        hints = " · ctrl+o 查看轨迹" if self.busy and not self.details_expanded else ""
        queue = f" · {len(self.message_queue)} 条等待" if self.message_queue else ""
        self.query_one("#help-hint", Static).update(
            f"{status}{hints}{queue}  ·  ↑↓ 历史  ·  @ 数据库路径  ·  / 命令  ·  Ctrl+C 中断"
        )
        self._render_context()

    @staticmethod
    def _status_glyph(status: str | None) -> str:
        return {"completed": "✓", "failed": "×", "running": "◐", "queued": "◌", "cancelled": "–"}.get(
            status or "", "·"
        )

    def on_input_changed(self, event: Input.Changed) -> None:
        self._idle_interrupt_armed = False
        value = event.value
        target = self.query_one("#prompt-zone", Static)
        if self._suppress_completion_once:
            self._suppress_completion_once = False
            self.command_matches = []
            self.path_matches = []
            self.command_index = 0
            target.update("")
            target.remove_class("visible")
            return
        path_query = self._path_query(value, event.input.cursor_position)
        if path_query is not None:
            self.command_matches = []
            prefix = [
                item
                for item in self.database_path_payload
                if item.get("path", "")[1:].casefold().startswith(path_query)
            ]
            fuzzy = [
                item
                for item in self.database_path_payload
                if path_query in item.get("path", "")[1:].casefold() and item not in prefix
            ]
            self.path_matches = (prefix + fuzzy)[:8]
            self.command_index = min(self.command_index, max(0, len(self.path_matches) - 1))
            self._render_path_completions()
            return
        self.path_matches = []
        stripped = value.lstrip()
        if not stripped.startswith("/"):
            self.command_matches = []
            self.command_index = 0
            target.update("")
            target.remove_class("visible")
            return
        needle = stripped.casefold()
        token = needle.split(maxsplit=1)[0]
        prefix = [(cmd, help_text) for cmd, help_text in COMMANDS if cmd.casefold().startswith(token)]
        fuzzy = [
            (cmd, help_text)
            for cmd, help_text in COMMANDS
            if token in cmd.casefold() and (cmd, help_text) not in prefix
        ]
        self.command_matches = (prefix + fuzzy)[:8]
        self.command_index = min(self.command_index, max(0, len(self.command_matches) - 1))
        self._render_command_completions()

    @staticmethod
    def _path_query(value: str, cursor_position: int) -> str | None:
        match = re.search(r"(?<!\S)@[^\s]*$", value[:cursor_position])
        if match is None:
            return None
        return match.group(0)[1:].casefold()

    def _render_path_completions(self) -> None:
        target = self.query_one("#prompt-zone", Static)
        if not self.path_matches:
            target.update(Text("  没有匹配的数据库路径", style="#7f93a8"))
            target.add_class("visible")
            return
        kind_labels = {"database": "数据库", "table": "表", "column": "字段"}
        rows: list[Text] = []
        for index, item in enumerate(self.path_matches):
            selected = index == self.command_index
            kind = str(item.get("kind", "path"))
            data_type = f" · {item['data_type']}" if item.get("data_type") else ""
            row = Text()
            row.append("› " if selected else "  ", style="#58a6ff" if selected else "#234a73")
            row.append(
                f"{self._clip_label(str(item.get('path', '')), 58):<60}",
                style="bold #e6f1ff" if selected else "#4aa8ff",
            )
            row.append(f"{kind_labels.get(kind, kind)}{data_type}", style="#c4d7ea" if selected else "#7f93a8")
            if selected:
                row.stylize("on #132a44")
            rows.append(row)
        target.update(Group(*rows))
        target.add_class("visible")

    def _render_command_completions(self) -> None:
        target = self.query_one("#prompt-zone", Static)
        if not self.command_matches:
            target.update("")
            target.remove_class("visible")
            return
        rows: list[Text] = []
        for index, (command, description) in enumerate(self.command_matches):
            selected = index == self.command_index
            row = Text()
            row.append("› " if selected else "  ", style="#58a6ff" if selected else "#234a73")
            row.append(f"{command:<24}", style="bold #e6f1ff" if selected else "#4aa8ff")
            row.append(description, style="#c4d7ea" if selected else "#7f93a8")
            if selected:
                row.stylize("on #132a44")
            rows.append(row)
        target.update(Group(*rows))
        target.add_class("visible")

    def on_key(self, event: events.Key) -> None:
        composer = self.query_one("#composer", Input)
        if not composer.has_focus:
            return
        matches: list[Any] = self.path_matches or self.command_matches
        if matches:
            if event.key == "down":
                self.command_index = (self.command_index + 1) % len(matches)
            elif event.key == "up":
                self.command_index = (self.command_index - 1) % len(matches)
            elif event.key in {"tab", "enter"} and self.path_matches:
                self._insert_path_match(self.path_matches[self.command_index])
            elif event.key == "tab" and self.command_matches:
                command = self.command_matches[self.command_index][0].split(" ", 1)[0]
                composer.value = f"{command} "
                composer.cursor_position = len(composer.value)
            else:
                return
            event.prevent_default()
            event.stop()
            if self.path_matches:
                self._render_path_completions()
            elif self.command_matches:
                self._render_command_completions()
            return
        if event.key == "up":
            handled = self._navigate_input_history(-1)
        elif event.key == "down":
            handled = self._navigate_input_history(1)
        else:
            return
        if not handled:
            return
        event.prevent_default()
        event.stop()

    def _navigate_input_history(self, direction: int) -> bool:
        if not self.input_history:
            return False
        composer = self.query_one("#composer", Input)
        if direction < 0:
            if self.history_index is None:
                self.history_draft = composer.value
                self.history_index = len(self.input_history) - 1
            elif self.history_index > 0:
                self.history_index -= 1
        else:
            if self.history_index is None:
                return False
            if self.history_index < len(self.input_history) - 1:
                self.history_index += 1
            else:
                self.history_index = None
        value = self.history_draft if self.history_index is None else self.input_history[self.history_index]
        self._suppress_completion_once = True
        composer.value = value
        composer.cursor_position = len(value)
        return True

    def _remember_input(self, value: str) -> None:
        if not self.input_history or self.input_history[-1] != value:
            self.input_history.append(value)
            del self.input_history[:-200]
            if self._history_store is not None:
                try:
                    self._history_store.append_string(value)
                except OSError:
                    pass
        self.history_index = None
        self.history_draft = ""

    def _insert_path_match(self, item: dict[str, Any]) -> None:
        composer = self.query_one("#composer", Input)
        cursor = composer.cursor_position
        match = re.search(r"(?<!\S)@[^\s]*$", composer.value[:cursor])
        if match is None:
            return
        path = str(item.get("path", ""))
        suffix = composer.value[cursor:]
        composer.value = f"{composer.value[: match.start()]}{path} {suffix}"
        composer.cursor_position = match.start() + len(path) + 1
        db_id = str(item.get("db_id", ""))
        if db_id and db_id != self.current_db:
            self._select_database(db_id)
        self.path_matches = []
        self.command_index = 0
        target = self.query_one("#prompt-zone", Static)
        target.update("")
        target.remove_class("visible")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        line = event.value.strip()
        if not line:
            return
        self._remember_input(line)
        event.input.value = ""
        self.command_matches = []
        self.path_matches = []
        self.query_one("#prompt-zone", Static).update("")
        self.query_one("#prompt-zone", Static).remove_class("visible")
        if line.startswith("/"):
            self._dispatch_command(line)
        else:
            self._start_query(line)

    def _start_query(self, question: str) -> None:
        if self.busy:
            self.message_queue.append(question)
            self._append_notice(f"已加入队列 #{len(self.message_queue)}；本轮结束后自动发送。", "normal")
            self._set_status(self.activity_status, tone=self.activity_tone)
            return
        if not self.current_db:
            self._append_notice("请先按 Ctrl+D 选择数据库。", "warn")
            return
        referenced_paths = set(re.findall(r"(?<!\S)@[\w./-]+", question))
        referenced_databases = {
            str(item.get("db_id", ""))
            for item in self.database_path_payload
            if item.get("path") in referenced_paths
        }
        referenced_databases.discard("")
        if len(referenced_databases) > 1:
            self._append_notice("单个任务不能同时引用多个数据库；请拆分为多个查询。", "warn")
            return
        if referenced_databases and self.current_db not in referenced_databases:
            self._select_database(next(iter(referenced_databases)))
        self._append_user(question)
        self.busy = True
        self._render_context()
        self._set_status("Agent 正在思考…", tone="warn")
        self.run_query(question)

    @work(thread=True, group="agent-run")
    def run_query(self, question: str) -> None:
        try:
            session = self.client.create_query(self.current_db or "", question, self.locale)
            self._execute_session(session)
        except Exception as error:
            self.call_from_thread(self._finish_with_error, error)

    @work(thread=True, group="agent-run")
    def run_recovery(self, scenario_id: str) -> None:
        try:
            session = self.client.create_recovery(scenario_id)
            self._execute_session(session)
        except Exception as error:
            self.call_from_thread(self._finish_with_error, error)

    def _execute_session(self, session: dict[str, Any]) -> None:
        session_id = session["session_id"]
        self.call_from_thread(self._session_started, session_id)
        self.client.run_session(session_id, self.run_options)
        for event in self.client.stream_events(session_id):
            self.call_from_thread(self._render_event, event)
        final = self.client.session(session_id)
        reward: dict[str, Any] = {}
        try:
            reward = self.client.trajectory(session_id).get("reward", {})
        except DriftSQLApiError:
            pass
        self.call_from_thread(self._finish_session, final, reward)

    def _session_started(self, session_id: str) -> None:
        self.current_session_id = session_id
        self._render_context()
        self._append_notice(f"Session {session_id[:12]} 已进入 GPU 队列", "normal")

    def _write_transcript(self, renderable: Any) -> None:
        self.query_one("#transcript", RichLog).write(renderable, scroll_end=False)
        self.call_after_refresh(self._scroll_page_end)

    def _scroll_page_end(self) -> None:
        self.query_one("#page-scroll", VerticalScroll).scroll_end(animate=False, immediate=True)

    def _append_user(self, text: str) -> None:
        self._write_transcript(Text.assemble(("❯ ", "bold #58a6ff"), (text, "bold #e6f1ff")))

    def _append_notice(self, text: str, tone: str = "normal") -> None:
        colors = {"normal": "#7f93a8", "warn": "#ffb454", "error": "#ff6b6b", "ok": "#78c6a3"}
        self._write_transcript(Text(f"  ◈ {text}", style=colors[tone]))

    def _render_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("event_type")
        payload = event.get("payload", {})
        if event_type == "session":
            translation = payload.get("input_translation", {})
            if translation.get("applied"):
                translated = str(translation.get("agent_input") or translation.get("translated", ""))
                model = str(translation.get("model_id", "zh-en"))
                elapsed = float(translation.get("elapsed_ms", 0.0))
                self._write_transcript(
                    Text(f"  ↳ 中文 → English · {model} · {elapsed:.0f} ms", style="#4aa8ff")
                )
                self._write_transcript(Text(f"    {_compact(translated, 700)}", style="#91a7bd"))
        elif event_type == "model":
            turn = payload.get("turn", "—")
            tool = payload.get("tool_name") or "生成响应"
            reasoning = payload.get("reasoning") or payload.get("content") or ""
            tokens = payload.get("response_tokens", 0)
            if self.details_expanded and reasoning:
                self._write_transcript(Text(f"  ┊ Thinking · turn {turn}", style="#4aa8ff"))
                self._write_transcript(Text(f"  ┊ {_compact(reasoning, 700)}", style="#7f93a8"))
            elif reasoning:
                self._write_transcript(
                    Text(f"  ┊ Thinking · turn {turn}  (ctrl+o 展开)", style="#64748b")
                )
            self._set_status(f"{tool}…  ·  ↓ {tokens} tokens", tone="warn")
        elif event_type == "tool":
            success = bool(payload.get("success"))
            tool_name = payload.get("tool", "unknown_tool")
            args = payload.get("arguments", {})
            observation = payload.get("observation", "")
            color = "#8fbc8f" if success else "#ff6b6b"
            glyph = "●"
            compact_args = ", ".join(f"{key}={_compact(value, 80)!r}" for key, value in args.items())
            self._write_transcript(
                Text.assemble((f"  {glyph} ", color), (tool_name, f"bold {color}"), (f"({compact_args})", "#7f93a8"))
            )
            if observation:
                if self.details_expanded:
                    self._write_transcript(Text(f"    └ {_compact(observation, 1200)}", style="#91a7bd"))
                else:
                    first_line = _compact(observation, 180).splitlines()[0]
                    self._write_transcript(Text(f"    └ {first_line}  (ctrl+o 展开)", style="#64748b"))
        elif event_type == "reward":
            self._render_inspector(reward=payload)
            if self.details_expanded:
                self._write_transcript(self._reward_breakdown(payload))
        elif event_type in {"budget", "error", "cancelled"}:
            tone = "error" if event_type == "error" else "warn"
            self._append_notice(f"{event_type.upper()}  {_compact(payload, 300)}", tone)
        elif event_type == "status":
            status = payload.get("status")
            if status and status not in TERMINAL_STATES:
                self._set_status(str(status), tone="warn")

    def _finish_session(self, session: dict[str, Any], reward: dict[str, Any]) -> None:
        self.busy = False
        result = session.get("result", {})
        success = bool(session.get("success"))
        sql = session.get("final_sql") or ""
        summary_rows = [
            ("状态", str(session.get("status", "—"))),
            ("终止原因", str(session.get("termination_reason") or "—")),
            ("工具调用", str(session.get("usage", {}).get("tool_calls", 0))),
            ("执行闭环", "通过" if result.get("query_completed") or result.get("task_success") else "未通过"),
        ]
        parts: list[Any] = []
        result_line = " · ".join(f"{key} {value}" for key, value in summary_rows)
        parts.append(Text("  └─ Response", style="#234a73"))
        parts.append(Text(f"⚕ {result_line}", style="#8fbc8f" if success else "#ff6b6b"))
        if sql:
            parts.extend(
                [Text("\n  最终 SQL", style="bold #e6f1ff"), Syntax(sql, "sql", theme="ansi_dark", word_wrap=True)]
            )
        execution_result = result.get("execution_result", {})
        columns = execution_result.get("columns", [])
        data_rows = execution_result.get("rows", [])
        if columns and isinstance(data_rows, list):
            table = Table(
                title="SQL 查询结果",
                title_style="bold #e6f1ff",
                header_style="bold #58a6ff",
                border_style="#234a73",
                show_lines=False,
                expand=True,
            )
            for column in columns:
                table.add_column(str(column), overflow="fold")
            for row in data_rows:
                values = row if isinstance(row, (list, tuple)) else [row]
                table.add_row(*(self._format_result_value(value) for value in values))
            truncation = " · 仅显示前几行" if execution_result.get("truncated") else ""
            parts.extend(
                [
                    Text(
                        f"  {len(data_rows)} 行 · {float(execution_result.get('elapsed_ms', 0.0)):.1f} ms{truncation}",
                        style="#7f93a8",
                    ),
                    table,
                ]
            )
        if session.get("mode") == "query":
            parts.append(Text("  仅验证只读执行与安全提交；业务语义没有隐藏答案自动证明。", style="#7aa7d9"))
        self._write_transcript(Group(*parts))
        self._render_inspector(session, reward)
        self._render_context()
        self._set_status("就绪" if success else "本轮失败", tone="ok" if success else "error")
        self._refresh_sessions()
        if self.message_queue:
            self.set_timer(0.05, self._drain_message_queue)

    @staticmethod
    def _format_result_value(value: Any) -> str:
        if value is None:
            return "NULL"
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, default=str)
        return str(value)

    def _drain_message_queue(self) -> None:
        if self.busy or not self.message_queue:
            return
        question = self.message_queue.pop(0)
        self._append_notice(f"正在发送队列中的下一条指令；剩余 {len(self.message_queue)} 条。", "normal")
        self._start_query(question)

    def _finish_with_error(self, error: Exception) -> None:
        self.busy = False
        self._show_error("Agent 运行失败", error)
        self._render_context()

    def _show_error(self, title: str, error: Exception) -> None:
        message = str(error)
        if isinstance(error, DriftSQLApiError) and error.status_code:
            message = f"HTTP {error.status_code} · {message}"
        self._write_transcript(Panel(message, title=title, border_style="#ff6b6b"))
        self._set_status(title, tone="error")

    def _dispatch_command(self, line: str) -> None:
        try:
            parts = shlex.split(line)
        except ValueError as error:
            self._show_error("命令格式错误", error)
            return
        command = parts[0].casefold()
        args = parts[1:]
        if command in {"/quit", "/exit"}:
            self.exit()
        elif command == "/help":
            self.action_help()
        elif command in {"/db", "/databases"}:
            if args:
                self._select_database(args[0])
            else:
                self.action_databases()
        elif command in {"/model", "/models"}:
            if len(args) == 2 and args[0] == "use":
                self.activate_model(args[1])
            elif len(args) == 2 and args[0] == "info":
                self._show_model_info(args[1])
            elif args and args[0] == "compare":
                self.command_worker("/experiments")
            else:
                self.action_models()
        elif command == "/sessions":
            self.action_sessions()
        elif command in {"/clear", "/new"}:
            self.action_clear()
        elif command in {"/details", "/detail"}:
            self.action_toggle_details()
        elif command in {"/queue", "/q"}:
            self._render_queue()
        elif command == "/status":
            self._render_status_panel()
        elif command == "/cancel":
            self.action_cancel_or_quit()
        elif command == "/recover":
            if not args:
                self._append_notice("用法：/recover <scenario_id>", "warn")
            elif self.busy:
                self._append_notice("当前会话仍在运行。", "warn")
            else:
                self._append_user(f"运行标准恢复场景：{args[0]}")
                self.busy = True
                self.run_recovery(args[0])
        elif command == "/budget":
            self._update_budget(args)
        elif command in {
            "/scenarios",
            "/trace",
            "/resume",
            "/reward",
            "/experiments",
            "/compare",
            "/ops",
            "/operations",
            "/failures",
            "/wandb",
            "/replay",
        }:
            self.command_worker(line)
        else:
            self._append_notice(f"未知命令：{command}，按 Ctrl+P 查看命令面板。", "warn")

    def _render_queue(self) -> None:
        if not self.message_queue:
            self._append_notice("消息队列为空。", "normal")
            return
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="#4aa8ff", justify="right")
        table.add_column(style="#c4d7ea")
        for index, question in enumerate(self.message_queue, start=1):
            table.add_row(str(index), self._clip_label(question, 100))
        self._write_command_result(f"等待发送 · {len(self.message_queue)}", table)

    def _render_status_panel(self) -> None:
        status = {
            "state": "running" if self.busy else "ready",
            "model": self.active_model_id,
            "database": self.current_db,
            "session": self.current_session_id,
            "readonly": True,
            "queued_messages": len(self.message_queue),
            "budget": self.run_options,
            "usage": self.last_session.get("usage", {}),
            "reward": self.last_reward,
        }
        self._write_command_result("DriftSQL 状态", JSON.from_data(status))

    def _update_budget(self, args: list[str]) -> None:
        if not args:
            self._render_budget_table()
            return
        integer_keys = {"max_turns", "max_tool_calls", "max_new_tokens", "max_total_tokens"}
        try:
            for assignment in args:
                key, raw = assignment.split("=", 1)
                if key not in self.run_options:
                    raise ValueError(f"未知预算字段：{key}")
                self.run_options[key] = int(raw) if key in integer_keys else float(raw)
        except (ValueError, TypeError) as error:
            self._show_error("预算设置失败", error)
            return
        self._render_budget_table()
        self._render_inspector()

    def _render_budget_table(self) -> None:
        table = Table(title="运行预算", border_style="#2f81c7")
        table.add_column("字段")
        table.add_column("值", justify="right")
        for key, value in self.run_options.items():
            table.add_row(key, str(value))
        self._write_transcript(table)

    @work(thread=True, group="commands")
    def command_worker(self, line: str) -> None:
        try:
            parts = shlex.split(line)
            command, args = parts[0].casefold(), parts[1:]
            title, renderable = self._command_result(command, args)
            self.call_from_thread(self._write_command_result, title, renderable)
        except Exception as error:
            self.call_from_thread(self._show_error, "命令执行失败", error)

    def _command_result(self, command: str, args: list[str]) -> tuple[str, Any]:
        if command == "/scenarios":
            rows = self.client.scenarios()
            db_id = args[0] if args else self.current_db
            rows = [row for row in rows if not db_id or row.get("db_id") == db_id]
            table = Table()
            for label in ("场景", "漂移", "难度", "问题"):
                table.add_column(label)
            for row in rows:
                table.add_row(
                    row["scenario_id"],
                    row["drift_type"],
                    str(row.get("difficulty") or "—"),
                    str(row.get("question", ""))[:70],
                )
            return f"恢复场景 · {db_id or '全部'}", table
        if command in {"/trace", "/resume"}:
            session_id = args[0] if args else self.current_session_id
            if not session_id:
                raise ValueError("没有可查看的 Session")
            trajectory = self.client.trajectory(session_id)
            return f"轨迹 · {session_id[:12]}", JSON.from_data(trajectory)
        if command == "/reward":
            session_id = args[0] if args else self.current_session_id
            if not session_id:
                raise ValueError("没有可查看的 Session")
            reward = self.client.trajectory(session_id).get("reward", {})
            return f"Reward · {session_id[:12]}", JSON.from_data(reward)
        if command in {"/experiments", "/compare"}:
            rows = self.client.experiments().get("experiments", [])
            table = Table()
            for label in ("选择", "实验", "类型", "任务成功", "可执行", "工具", "Unsafe"):
                table.add_column(label)
            for row in rows:
                table.add_row(
                    "●" if row.get("selected") else "○",
                    row["display_name"],
                    row["category"],
                    _pct(row["task_success_rate"]),
                    _pct(row["executable_rate"]),
                    f"{row['average_tool_calls']:.2f}",
                    str(row["unsafe_tasks"]),
                )
            return "统一实验对比", table
        if command in {"/ops", "/operations"}:
            return "运行指标", JSON.from_data(self.client.operations())
        if command == "/failures":
            payload = self.client.failures(args[0] if args else None)
            table = Table()
            for label in ("Session", "类型", "数据库", "漂移", "终止原因"):
                table.add_column(label)
            for row in payload.get("failures", []):
                table.add_row(
                    row["session_id"][:12],
                    row["failure_type"],
                    row["db_id"],
                    row["drift_type"],
                    str(row.get("termination_reason") or "—"),
                )
            return f"失败记录 · {payload.get('total', 0)}", table
        if command == "/wandb":
            if args:
                return f"W&B · {args[0]}", JSON.from_data(self.client.wandb_history(args[0]))
            return "W&B Runs", JSON.from_data(self.client.wandb_runs())
        if command == "/replay":
            if args and args[0] in {"approve", "reject"}:
                if len(args) < 4:
                    raise ValueError("用法：/replay approve|reject <candidate_id> <reviewer> <reason>")
                reviewed = self.client.review_replay(args[1], args[0], args[2], " ".join(args[3:]))
                return f"Replay 审核 · {reviewed['candidate_id']}", JSON.from_data(reviewed)
            return "Hard Replay 候选", JSON.from_data(self.client.replay_candidates(args[0] if args else None))
        raise ValueError(f"暂不支持：{command}")

    def _write_command_result(self, title: str, renderable: Any) -> None:
        self._write_transcript(Panel(renderable, title=title, border_style="#234a73"))

    def _show_model_info(self, model_id: str) -> None:
        model = next(
            (item for item in self.model_payload.get("models", []) if item.get("model_id") == model_id),
            None,
        )
        if model is None:
            self._append_notice(f"未知模型：{model_id}", "warn")
            return
        self._write_command_result(model.get("display_name", model_id), JSON.from_data(model))

    def action_help(self) -> None:
        self.push_screen(HelpScreen())

    def action_models(self) -> None:
        items = []
        for model in self.model_payload.get("models", []):
            metrics = model.get("metrics", {})
            marker = "●" if model.get("model_id") == self.active_model_id else "○"
            available = "可用" if model.get("available") else "未导出"
            items.append(
                PickerItem(
                    model["model_id"],
                    f"{marker} {model.get('display_name', model['model_id'])}",
                    f"{model.get('category', '—')} · Tune {_pct(metrics.get('tune432_success_rate'))} · {available}",
                    model,
                )
            )
        self.push_screen(PickerScreen("选择模型", "Enter 激活 · 输入筛选 · Esc 关闭", items), self._on_model_picked)

    def _on_model_picked(self, model_id: str | None) -> None:
        if model_id and model_id != self.active_model_id:
            self.activate_model(model_id)

    @work(thread=True, exclusive=True, group="model-switch")
    def activate_model(self, model_id: str) -> None:
        try:
            self.call_from_thread(self._set_status, "正在切换模型…", tone="warn")
            payload = self.client.activate_model(model_id)
            self.call_from_thread(self._model_activated, model_id, payload)
        except Exception as error:
            self.call_from_thread(self._show_error, "模型切换失败", error)

    def _model_activated(self, model_id: str, payload: dict[str, Any]) -> None:
        self.model_payload = payload
        self.active_model_id = payload.get("active_model_id", model_id)
        self._append_notice(f"模型已切换为 {self.active_model_id}", "ok")
        self._render_context()
        self._render_inspector()
        self._set_status("就绪", tone="ok")

    def action_databases(self) -> None:
        items = [
            PickerItem(
                item["db_id"],
                f"{'●' if item['db_id'] == self.current_db else '○'} {item['db_id']}",
                f"{item.get('scenario_count', 0)} 场景 · {', '.join(item.get('drift_types', [])) or '无漂移'}",
                item,
            )
            for item in self.database_payload
        ]
        self.push_screen(PickerScreen("选择数据库", "每个 Session 会复制独立只读沙箱", items), self._on_database_picked)

    def _on_database_picked(self, db_id: str | None) -> None:
        if db_id:
            self._select_database(db_id)

    def _select_database(self, db_id: str) -> None:
        if db_id not in {item["db_id"] for item in self.database_payload}:
            self._append_notice(f"未知数据库：{db_id}", "warn")
            return
        self.current_db = db_id
        self._append_notice(f"当前数据库已切换为 {db_id}", "ok")
        self._populate_sidebar()
        self._render_context()

    def action_sessions(self) -> None:
        items = [
            PickerItem(
                item["session_id"],
                f"{self._status_glyph(item.get('status'))} {item.get('db_id', '—')} · {item.get('status', '—')}",
                f"{item.get('mode', 'recovery')} · {item['session_id']} · "
                f"{item.get('model', {}).get('model_id', 'legacy')}",
                item,
            )
            for item in self.session_payload
        ]
        self.push_screen(
            PickerScreen("会话历史", "Enter 回放完整轨迹 · 输入 Session / DB 筛选", items),
            self._on_session_picked,
        )

    def _on_session_picked(self, session_id: str | None) -> None:
        if session_id:
            self.current_session_id = session_id
            self.command_worker(f"/trace {shlex.quote(session_id)}")
            self._render_context()

    def _refresh_sessions(self) -> None:
        self.refresh_sessions()

    @work(thread=True, exclusive=True, group="sessions-refresh")
    def refresh_sessions(self) -> None:
        try:
            payload = self.client.sessions(limit=20)
            self.call_from_thread(self._apply_sessions, payload.get("sessions", []))
        except Exception:
            return

    def _apply_sessions(self, sessions: list[dict[str, Any]]) -> None:
        self.session_payload = sessions
        self._populate_sidebar()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if not isinstance(event.item, PickerItem):
            return
        if event.list_view.id == "db-list":
            self._select_database(event.item.key)
        elif event.list_view.id == "session-list":
            self._on_session_picked(event.item.key)

    def action_clear(self) -> None:
        self.query_one("#transcript", RichLog).clear()
        self._append_welcome()
        self.current_session_id = None
        self.last_session = {}
        self.last_reward = {}
        self._render_context()

    def action_redraw(self) -> None:
        self.refresh(layout=True)
        self._append_notice("界面已重绘；使用 /new 可开始新转录。", "normal")

    def action_toggle_details(self) -> None:
        self.details_expanded = not self.details_expanded
        state = "展开" if self.details_expanded else "折叠"
        self._append_notice(f"工具、推理与 Reward 分项已{state}。", "normal")
        self._set_status("Agent 正在运行…" if self.busy else "就绪", tone="warn" if self.busy else "ok")
        if self.details_expanded and self.current_session_id:
            self.expanded_trace(self.current_session_id)

    @work(thread=True, exclusive=True, group="expanded-trace")
    def expanded_trace(self, session_id: str) -> None:
        try:
            trajectory = self.client.trajectory(session_id)
            events = [
                event
                for event in trajectory.get("events", [])
                if event.get("event_type") in {"model", "tool", "reward", "error"}
            ]
            self.call_from_thread(self._render_expanded_trace, session_id, events)
        except Exception as error:
            self.call_from_thread(self._show_error, "详情加载失败", error)

    def _render_expanded_trace(self, session_id: str, events: list[dict[str, Any]]) -> None:
        self._append_notice(f"详细轨迹 · {session_id[:12]}", "normal")
        for event in events:
            self._render_event(event)

    def action_interrupt(self) -> None:
        if self.busy and self.current_session_id:
            self.cancel_session(self.current_session_id)
            return
        composer = self.query_one("#composer", Input)
        if composer.value:
            composer.value = ""
            self._append_notice("已清空当前输入。", "normal")

    def action_cancel_or_quit(self) -> None:
        if self.busy and self.current_session_id:
            self.cancel_session(self.current_session_id)
            return
        composer = self.query_one("#composer", Input)
        if composer.value:
            composer.value = ""
            self._idle_interrupt_armed = False
            return
        if self._idle_interrupt_armed:
            self.exit()
            return
        self._idle_interrupt_armed = True
        self._append_notice("再次按 Ctrl+C 退出 DriftSQL。", "normal")
        self.set_timer(1.5, self._disarm_idle_interrupt)

    def _disarm_idle_interrupt(self) -> None:
        self._idle_interrupt_armed = False

    @work(thread=True, exclusive=True, group="cancel")
    def cancel_session(self, session_id: str) -> None:
        try:
            self.client.cancel(session_id)
            self.call_from_thread(self._append_notice, f"已请求取消 {session_id[:12]}", "warn")
        except Exception as error:
            self.call_from_thread(self._show_error, "取消失败", error)


def run_tui(client: DriftSQLClient, *, locale: str = "zh-CN", history_file: str | None = None) -> None:
    """Run the full-screen interface and own the client lifecycle."""

    if os.getenv("DRIFTSQL_NO_COLOR") != "1":
        os.environ.pop("NO_COLOR", None)
        os.environ.setdefault("COLORTERM", "truecolor")
    DriftSQLTUI(client, locale=locale, history_file=history_file).run()
