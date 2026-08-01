from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


comparison = _load("compare_stage4_eval")
evaluator = _load("run_five_tool_eval")
exact_mcnemar = comparison.exact_mcnemar
row_metrics = comparison.row_metrics
summarize = evaluator.summarize
unsafe_sql = evaluator.unsafe_sql
apply_terminal_submit_fallback = evaluator.apply_terminal_submit_fallback


def _row(*, success: bool, reason: str, sql: str) -> dict:
    return {
        "task_success": success,
        "executable": sql.upper().startswith("SELECT"),
        "termination_reason": reason,
        "all_five_tools": False,
        "called_tools": ["execute_sql"],
        "usage": {"model_calls": 1, "tool_calls": 1},
        "trajectory": [
            {
                "tool_name": "execute_sql",
                "arguments": {"sql": sql},
                "metrics": {"execution_error": "not authorized" if unsafe_sql(sql) else None},
            }
        ],
        "safety": {
            "unsafe": unsafe_sql(sql),
            "unsafe_actions": int(unsafe_sql(sql)),
            "timed_out": False,
            "duplicate_questions": 0,
            "duplicate_executions": 0,
        },
    }


def test_stage4_summary_surfaces_turn_limit_and_safety() -> None:
    rows = [
        _row(success=True, reason="submitted", sql="SELECT * FROM orders"),
        _row(success=False, reason="turn_limit", sql="DROP TABLE orders"),
    ]
    summary = summarize("candidate", rows)
    assert summary["task_success"] == 1
    assert summary["turn_limit"] == 1
    assert summary["unsafe_tasks"] == 1
    assert summary["unsafe_actions"] == 1


def test_comparison_metrics_and_exact_mcnemar() -> None:
    rows = [
        _row(success=True, reason="submitted", sql="SELECT * FROM orders"),
        _row(success=False, reason="turn_limit", sql="DROP TABLE orders"),
    ]
    metrics = row_metrics(rows)
    assert metrics["task_success"] == 1
    assert metrics["turn_limit"] == 1
    assert metrics["unsafe_tasks"] == 1
    assert exact_mcnemar(3, 0) == 0.25


def test_read_only_explain_is_not_reported_as_unsafe() -> None:
    sql = "EXPLAIN ANALYZE SELECT * FROM orders"
    assert not unsafe_sql(sql)
    assert not comparison.unsafe_sql(sql)


def test_terminal_fallback_uses_only_successfully_executed_sql() -> None:
    state = evaluator.EvalState(
        record={},
        variant="candidate",
        conversation=[],
        instance_id="fallback-test",
        trajectory=[
            {
                "tool_name": "execute_sql",
                "arguments": {"sql": "SELECT * FROM orders"},
                "metrics": {"execution_success": True},
            }
        ],
    )
    assert apply_terminal_submit_fallback(state)
    assert state.final_sql == "SELECT * FROM orders"
    assert state.termination_reason == "fallback_submitted"


def test_terminal_fallback_rejects_failed_or_unsafe_execution() -> None:
    state = evaluator.EvalState(
        record={},
        variant="candidate",
        conversation=[],
        instance_id="fallback-test",
        trajectory=[
            {
                "tool_name": "execute_sql",
                "arguments": {"sql": "DROP TABLE orders"},
                "metrics": {"execution_success": False},
            }
        ],
    )
    assert not apply_terminal_submit_fallback(state)
    assert state.termination_reason == "running"
