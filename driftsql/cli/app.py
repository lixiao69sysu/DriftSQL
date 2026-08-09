"""Command-line entry point for DriftSQL."""

from __future__ import annotations

import os
from pathlib import Path

import typer
from rich.console import Console

from .client import DriftSQLApiError, DriftSQLClient
from .shell import DriftSQLShell

app = typer.Typer(add_completion=False, no_args_is_help=True, help="DriftSQL Agentic SQL CLI")
console = Console()


def _client(url: str, api_key: str | None) -> DriftSQLClient:
    return DriftSQLClient(url, api_key=api_key or os.getenv("DRIFTSQL_API_KEY"))


@app.command()
def chat(
    url: str = typer.Option("http://127.0.0.1:8001", help="DriftSQL service URL"),
    api_key: str | None = typer.Option(None, help="Service API key; defaults to DRIFTSQL_API_KEY"),
    locale: str = typer.Option("zh-CN", help="zh-CN or en-US"),
    classic: bool = typer.Option(
        False,
        "--classic",
        help="Use the legacy scrolling REPL instead of the full-screen TUI",
    ),
) -> None:
    """Open the full-screen interactive database Agent workbench."""

    client = _client(url, api_key)
    history = str(Path.home() / ".driftsql_history")
    try:
        if classic:
            DriftSQLShell(client, console=console, locale=locale).run(history)
        else:
            from .tui import run_tui

            run_tui(client, locale=locale, history_file=history)
            client = None
    except DriftSQLApiError as error:
        console.print(f"[red]无法连接 DriftSQL 服务：{error}[/red]")
        raise typer.Exit(1) from error
    finally:
        if client is not None:
            client.close()


@app.command("models")
def list_models(
    url: str = typer.Option("http://127.0.0.1:8001"),
    api_key: str | None = typer.Option(None),
) -> None:
    """List registered Base, SFT and GRPO models."""

    client = _client(url, api_key)
    try:
        DriftSQLShell(client, console=console).models_command([])
    finally:
        client.close()


@app.command()
def query(
    instruction: str = typer.Argument(..., help="Natural-language database instruction"),
    db: str = typer.Option(..., "--db", help="Registered database ID"),
    url: str = typer.Option("http://127.0.0.1:8001"),
    api_key: str | None = typer.Option(None),
) -> None:
    """Run one free-form, execution-verified database query."""

    client = _client(url, api_key)
    try:
        shell = DriftSQLShell(client, console=console)
        shell.current_db = db
        shell.run_query(instruction)
    finally:
        client.close()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
