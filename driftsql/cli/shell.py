"""Hermes/OpenClaw-style terminal shell backed by the DriftSQL service."""

from __future__ import annotations

import json
import shlex
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .client import DriftSQLApiError, DriftSQLClient

TERMINAL = {"completed", "failed", "cancelled", "timed_out", "budget_exhausted"}


class DriftSQLShell:
    def __init__(self, client: DriftSQLClient, *, console: Console | None = None, locale: str = "zh-CN") -> None:
        self.client = client
        self.console = console or Console()
        self.locale = locale
        self.current_db: str | None = None
        self.current_session_id: str | None = None
        self.run_options: dict[str, Any] = {
            "max_turns": 7,
            "timeout_seconds": 120,
            "max_tool_calls": 7,
            "max_new_tokens": 512,
            "max_total_tokens": 32768,
        }

    def banner(self) -> None:
        health = self.client.health()
        model = health.get("model", {})
        self.console.print(
            Panel.fit(
                f"[bold cyan]DriftSQL CLI[/bold cyan]\n"
                f"模型: {model.get('model_id') or model.get('base_model', 'unknown')}\n"
                f"后端: {model.get('backend', 'unknown')} · 沙箱: 只读 SQLite\n"
                "输入数据库需求，或使用 /help 查看命令。",
                border_style="cyan",
            )
        )

    def run(self, history_file: str) -> None:
        self.banner()
        databases = self.client.databases()
        if databases:
            self.current_db = databases[0]["db_id"]
            self.console.print(f"当前数据库：[bold]{self.current_db}[/bold]（用 /db 切换）")
        prompt = PromptSession(history=FileHistory(history_file))
        while True:
            try:
                line = prompt.prompt(self._prompt()).strip()
            except (EOFError, KeyboardInterrupt):
                self.console.print("\n已退出。")
                return
            if not line:
                continue
            try:
                if self.execute_line(line) is False:
                    return
            except DriftSQLApiError as error:
                self.console.print(f"[bold red]API 错误[/bold red] ({error.status_code}): {error}")
            except (ValueError, IndexError) as error:
                self.console.print(f"[bold red]输入错误：[/bold red]{error}")

    def _prompt(self) -> str:
        return f"driftsql[{self.current_db or '-'}]> "

    def execute_line(self, line: str) -> bool:
        if not line.startswith("/"):
            self.run_query(line)
            return True
        parts = shlex.split(line)
        command = parts[0].casefold()
        args = parts[1:]
        if command in {"/exit", "/quit"}:
            return False
        if command == "/help":
            self.show_help()
        elif command in {"/db", "/databases"}:
            self.database_command(args)
        elif command in {"/model", "/models"}:
            self.models_command(args)
        elif command == "/recover":
            if not args:
                raise ValueError("用法：/recover <scenario_id>")
            self.run_recovery(args[0])
        elif command == "/sessions":
            self.show_sessions()
        elif command == "/scenarios":
            self.show_scenarios(args[0] if args else self.current_db)
        elif command == "/budget":
            self.budget_command(args)
        elif command in {"/experiments", "/compare"}:
            self.show_experiments()
        elif command in {"/ops", "/operations"}:
            self.show_operations()
        elif command == "/failures":
            self.show_failures(args[0] if args else None)
        elif command == "/wandb":
            self.show_wandb(args)
        elif command == "/replay":
            self.replay_command(args)
        elif command in {"/resume", "/trace"}:
            session_id = args[0] if args else self.current_session_id
            if not session_id:
                raise ValueError("没有可查看的 Session")
            self.show_trajectory(session_id)
        elif command == "/reward":
            self.show_reward(args[0] if args else self.current_session_id)
        elif command == "/cancel":
            session_id = args[0] if args else self.current_session_id
            if not session_id:
                raise ValueError("没有可取消的 Session")
            self.client.cancel(session_id)
            self.console.print(f"已请求取消 {session_id}")
        else:
            raise ValueError(f"未知命令：{parts[0]}")
        return True

    def show_help(self) -> None:
        self.console.print(
            "[bold]/db[/bold] [db_id]  列出或切换数据库\n"
            "[bold]/models[/bold]  查看 Base/SFT/GRPO 模型\n"
            "[bold]/models info <id>[/bold]  查看模型详情\n"
            "[bold]/models use <id>[/bold]  激活模型（无活跃 Session 时）\n"
            "[bold]/models compare[/bold]  对比统一 Tune 指标\n"
            "[bold]/recover <scenario_id>[/bold]  运行标准漂移恢复任务\n"
            "[bold]/scenarios[/bold] [db_id]  查看标准恢复场景\n"
            "[bold]/budget[/bold] [key=value ...]  查看或修改运行预算\n"
            "[bold]/sessions[/bold]  查看历史 Session\n"
            "[bold]/trace[/bold] [session_id]  回放完整轨迹\n"
            "[bold]/reward[/bold] [session_id]  查看 Reward\n"
            "[bold]/cancel[/bold] [session_id]  取消运行\n"
            "[bold]/experiments[/bold]  查看统一实验对比\n"
            "[bold]/ops[/bold]  查看持久化运行指标\n"
            "[bold]/failures[/bold] [type]  查看失败记录\n"
            "[bold]/wandb[/bold] [run_id]  查看 W&B 运行或指标\n"
            "[bold]/replay[/bold]  查看 Replay 候选\n"
            "[bold]/replay approve|reject <id> <reviewer> <reason>[/bold]\n"
            "[bold]/exit[/bold]  退出\n\n"
            "普通文本会作为当前数据库的自然语言查询执行。"
        )

    def database_command(self, args: list[str]) -> None:
        databases = self.client.databases()
        ids = {item["db_id"] for item in databases}
        if args:
            if args[0] not in ids:
                raise ValueError(f"未知数据库：{args[0]}")
            self.current_db = args[0]
            self.console.print(f"当前数据库已切换为 [bold]{self.current_db}[/bold]")
            return
        table = Table(title="可用数据库", show_lines=False)
        table.add_column("数据库")
        table.add_column("场景数", justify="right")
        table.add_column("漂移类型")
        for item in databases:
            marker = "* " if item["db_id"] == self.current_db else ""
            table.add_row(marker + item["db_id"], str(item["scenario_count"]), ", ".join(item["drift_types"]))
        self.console.print(table)

    def models_command(self, args: list[str]) -> None:
        payload = self.client.models()
        if args and args[0] == "use":
            if len(args) != 2:
                raise ValueError("用法：/models use <model_id>")
            payload = self.client.activate_model(args[1])
            self.console.print(f"[green]已激活模型：{payload.get('active_model_id')}[/green]")
        if args and args[0] == "info":
            if len(args) != 2:
                raise ValueError("用法：/models info <model_id>")
            model = next((item for item in payload["models"] if item["model_id"] == args[1]), None)
            if model is None:
                raise ValueError(f"未知模型：{args[1]}")
            self.console.print(Panel(json.dumps(model, ensure_ascii=False, indent=2), title=model["display_name"]))
            return
        self._print_models(payload)

    def _print_models(self, payload: dict[str, Any]) -> None:
        table = Table(title="模型注册表")
        table.add_column("状态")
        table.add_column("模型 ID")
        table.add_column("类型")
        table.add_column("Tune432")
        table.add_column("漂移恢复")
        table.add_column("平均工具")

        def pct(value: Any) -> str:
            return f"{float(value) * 100:.2f}%" if value is not None else "-"

        for model in payload["models"]:
            metrics = model.get("metrics", {})
            table.add_row(
                "●" if model.get("active") else ("○" if model.get("available") else "×"),
                model["model_id"],
                model["category"],
                pct(metrics.get("tune432_success_rate")),
                pct(metrics.get("drift_recovery_rate")),
                str(metrics.get("average_tool_calls", "-")),
            )
        self.console.print(table)

    def run_query(self, question: str) -> None:
        if not self.current_db:
            raise ValueError("请先使用 /db 选择数据库")
        session = self.client.create_query(self.current_db, question, self.locale)
        self._run_and_stream(session)

    def run_recovery(self, scenario_id: str) -> None:
        session = self.client.create_recovery(scenario_id)
        self._run_and_stream(session)

    def _run_and_stream(self, session: dict[str, Any]) -> None:
        session_id = session["session_id"]
        self.current_session_id = session_id
        self.client.run_session(session_id, self.run_options)
        self.console.print(f"Session [dim]{session_id}[/dim] 已进入队列。")
        for event in self.client.stream_events(session_id):
            self.render_event(event)
        final = self.client.session(session_id)
        self.render_final(final)

    def render_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("event_type")
        payload = event.get("payload", {})
        if event_type == "session":
            translation = payload.get("input_translation", {})
            if translation.get("applied"):
                self.console.print(
                    f"[blue]中文 → English[/blue] [dim]({translation.get('model_id')}, "
                    f"{float(translation.get('elapsed_ms', 0.0)):.0f} ms)[/dim]"
                )
                self.console.print(f"  {translation.get('agent_input') or translation.get('translated', '')}")
        elif event_type == "model":
            self.console.print(
                f"[cyan][模型 {payload.get('turn')}][/cyan] 动作={payload.get('tool_name') or '-'} "
                f"tokens={payload.get('response_tokens', 0)}"
            )
        elif event_type == "tool":
            success = "[green]✓[/green]" if payload.get("success") else "[red]×[/red]"
            self.console.print(f"{success} [bold]{payload.get('tool')}[/bold] {payload.get('arguments', {})}")
            observation = str(payload.get("observation", ""))
            if observation:
                self.console.print(f"  [dim]{observation[:800]}[/dim]")
        elif event_type == "reward":
            self.console.print(f"[magenta]Reward[/magenta] {json.dumps(payload, ensure_ascii=False)}")
        elif event_type in {"budget", "error", "cancelled"}:
            self.console.print(f"[yellow]{event_type}[/yellow] {payload}")

    def render_final(self, session: dict[str, Any]) -> None:
        result = session.get("result", {})
        color = "green" if session.get("success") else "red"
        lines = [
            f"状态: [{color}]{session.get('status')}[/{color}]",
            f"终止原因: {session.get('termination_reason')}",
            f"工具调用: {session.get('usage', {}).get('tool_calls', 0)}",
            f"最终 SQL: {session.get('final_sql') or '-'}",
        ]
        if session.get("mode") == "query":
            lines.append("验证范围: 仅执行与安全提交；业务语义没有隐藏标准答案自动证明")
            lines.append(f"执行闭环: {'通过' if result.get('query_completed') else '未通过'}")
        else:
            lines.append(f"任务成功: {bool(result.get('task_success'))}")
        self.console.print(Panel("\n".join(lines), title="运行结果", border_style=color))
        execution_result = result.get("execution_result", {})
        columns = execution_result.get("columns", [])
        rows = execution_result.get("rows", [])
        if columns and isinstance(rows, list):
            table = Table(title="SQL 查询结果", expand=True)
            for column in columns:
                table.add_column(str(column))
            for row in rows:
                values = row if isinstance(row, (list, tuple)) else [row]
                table.add_row(*(self._format_result_value(value) for value in values))
            self.console.print(table)
            truncated = "，结果已截断" if execution_result.get("truncated") else ""
            self.console.print(
                f"[dim]{len(rows)} 行 · {float(execution_result.get('elapsed_ms', 0.0)):.1f} ms{truncated}[/dim]"
            )

    @staticmethod
    def _format_result_value(value: Any) -> str:
        if value is None:
            return "NULL"
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, default=str)
        return str(value)

    def show_sessions(self) -> None:
        payload = self.client.sessions()
        table = Table(title=f"Session 历史（共 {payload['total']}）")
        for title in ("ID", "模式", "数据库", "状态", "成功", "模型"):
            table.add_column(title)
        for item in payload["sessions"]:
            table.add_row(
                item["session_id"][:12],
                item.get("mode", "recovery"),
                item["db_id"],
                item["status"],
                str(item.get("success")),
                item.get("model", {}).get("model_id") or "legacy",
            )
        self.console.print(table)

    def show_scenarios(self, db_id: str | None) -> None:
        scenarios = self.client.scenarios()
        if db_id:
            scenarios = [item for item in scenarios if item["db_id"] == db_id]
        table = Table(title=f"恢复场景{f' · {db_id}' if db_id else ''}")
        for title in ("场景 ID", "数据库", "漂移", "难度", "问题"):
            table.add_column(title)
        for item in scenarios:
            table.add_row(
                item["scenario_id"],
                item["db_id"],
                item["drift_type"],
                str(item.get("difficulty") or "-"),
                str(item.get("question", ""))[:80],
            )
        self.console.print(table)

    def budget_command(self, args: list[str]) -> None:
        integer_keys = {"max_turns", "max_tool_calls", "max_new_tokens", "max_total_tokens"}
        for assignment in args:
            if "=" not in assignment:
                raise ValueError("预算使用 key=value 格式")
            key, raw = assignment.split("=", 1)
            if key not in self.run_options:
                raise ValueError(f"未知预算字段：{key}")
            self.run_options[key] = int(raw) if key in integer_keys else float(raw)
        table = Table(title="运行预算")
        table.add_column("字段")
        table.add_column("值", justify="right")
        for key, value in self.run_options.items():
            table.add_row(key, str(value))
        self.console.print(table)

    def show_experiments(self) -> None:
        payload = self.client.experiments()
        table = Table(title="统一实验对比")
        for title in ("选择", "实验", "类型", "任务成功", "可执行", "模型调用", "工具调用", "Unsafe"):
            table.add_column(title)
        for item in payload["experiments"]:
            table.add_row(
                "●" if item.get("selected") else "○",
                item["display_name"],
                item["category"],
                f"{item['task_success_rate'] * 100:.2f}%",
                f"{item['executable_rate'] * 100:.2f}%",
                f"{item['average_model_calls']:.2f}",
                f"{item['average_tool_calls']:.2f}",
                str(item["unsafe_tasks"]),
            )
        self.console.print(table)

    def show_operations(self) -> None:
        item = self.client.operations()
        summary = {
            "总 Session": item.get("total_sessions"),
            "终态 Session": item.get("terminal_sessions"),
            "成功率": item.get("success_rate"),
            "平均耗时(ms)": item.get("average_latency_ms"),
            "平均模型调用": item.get("average_model_calls"),
            "平均工具调用": item.get("average_tool_calls"),
            "Unsafe": item.get("unsafe_sessions"),
            "Timeout": item.get("timed_out_sessions"),
        }
        self.console.print(Panel(json.dumps(summary, ensure_ascii=False, indent=2), title="运行指标"))

    def show_failures(self, failure_type: str | None) -> None:
        payload = self.client.failures(failure_type)
        table = Table(title=f"失败记录（{payload.get('total', 0)}）")
        for title in ("Session", "类型", "数据库", "漂移", "终止原因"):
            table.add_column(title)
        for item in payload.get("failures", []):
            table.add_row(
                item["session_id"][:12],
                item["failure_type"],
                item["db_id"],
                item["drift_type"],
                str(item.get("termination_reason") or "-"),
            )
        self.console.print(table)

    def show_wandb(self, args: list[str]) -> None:
        if args:
            payload = self.client.wandb_history(args[0])
            table = Table(title=f"W&B · {args[0]}")
            table.add_column("指标")
            table.add_column("点数", justify="right")
            table.add_column("最新值", justify="right")
            for series in payload.get("series", []):
                points = series.get("points", [])
                latest = points[-1].get("value") if points else None
                table.add_row(series.get("name", "-"), str(len(points)), str(latest))
            self.console.print(table)
            return
        payload = self.client.wandb_runs()
        if payload.get("status") != "ready":
            self.console.print(f"W&B 状态：{payload.get('status')} · {payload.get('error') or ''}")
            return
        table = Table(title="W&B Runs")
        for title in ("Run ID", "名称", "状态", "创建时间"):
            table.add_column(title)
        for item in payload.get("runs", []):
            table.add_row(item["run_id"], item["name"], item["state"], str(item.get("created_at") or "-"))
        self.console.print(table)

    def replay_command(self, args: list[str]) -> None:
        if args and args[0] in {"approve", "reject"}:
            if len(args) < 4:
                raise ValueError("用法：/replay approve|reject <candidate_id> <reviewer> <reason>")
            reviewed = self.client.review_replay(args[1], args[0], args[2], " ".join(args[3:]))
            self.console.print(
                f"[green]Replay {reviewed['candidate_id']} 已记录决策：{reviewed['review_status']}[/green]"
            )
            return
        status = args[0] if args else None
        payload = self.client.replay_candidates(status)
        table = Table(title=f"Replay 候选（{payload.get('total', 0)}）")
        for title in ("Candidate", "Session", "失败类型", "审核状态", "轨迹 Hash"):
            table.add_column(title)
        for item in payload.get("candidates", []):
            table.add_row(
                item["candidate_id"],
                item["session_id"][:12],
                item["failure_type"],
                item["review_status"],
                item["trajectory_sha256"][:12],
            )
        self.console.print(table)

    def show_trajectory(self, session_id: str) -> None:
        trajectory = self.client.trajectory(session_id)
        self.current_session_id = trajectory["session"]["session_id"]
        for event in trajectory["events"]:
            self.render_event(event)
        self.render_final(trajectory["session"])

    def show_reward(self, session_id: str | None) -> None:
        if not session_id:
            raise ValueError("没有可查看的 Session")
        reward = self.client.trajectory(session_id).get("reward", {})
        self.console.print(Panel(json.dumps(reward, ensure_ascii=False, indent=2), title="Reward"))
