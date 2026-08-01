from __future__ import annotations

import json
import sqlite3

import pytest

from driftsql.data.trajectory import build_rl_record
from driftsql.drift import (
    MetricDefinitionChange,
    build_metric_definition_example,
    fingerprint_query,
    materialize_schema_diff,
    rewrite_metric_expression,
)
from driftsql.integrations.state_policy import schema_diff_recovery_guidance
from driftsql.rewards.agentic import compute_score


def tool_trace(calls: list[tuple[str, dict]]) -> str:
    return "\n".join(
        "<tool_call>"
        + json.dumps({"name": name, "arguments": arguments})
        + "</tool_call>"
        for name, arguments in calls
    )


def test_metric_definition_rewrites_aggregate_without_changing_alias() -> None:
    drift = MetricDefinitionChange(
        metric_name="net_revenue",
        old_expression="SUM(gross_amount)",
        new_expression="SUM(gross_amount - COALESCE(refund_amount, 0))",
        from_version="metric-v1",
        to_version="metric-v2",
        reason="refunds are now excluded from booked revenue",
    )

    repaired = drift.rewrite(
        "SELECT region, SUM(gross_amount) AS revenue FROM sales GROUP BY region"
    )

    assert "SUM(gross_amount - COALESCE(refund_amount, 0)) AS revenue" in repaired
    assert drift.as_knowledge_entry()["definition"].startswith("SUM(gross_amount")


def test_metric_rewrite_refuses_silent_noop() -> None:
    with pytest.raises(ValueError, match="absent"):
        rewrite_metric_expression("SELECT COUNT(*) FROM sales", "SUM(amount)", "SUM(net_amount)")


def test_metric_change_is_exposed_as_agent_recovery_guidance() -> None:
    operation = MetricDefinitionChange(
        metric_name="active_customer",
        old_expression="COUNT(DISTINCT customer_id)",
        new_expression="COUNT(DISTINCT CASE WHEN status = 'active' THEN customer_id END)",
        from_version="metric-v1",
        to_version="metric-v2",
        reason="inactive accounts no longer count",
        requires_clarification=True,
    ).as_operation()

    guidance = schema_diff_recovery_guidance({"operations": [operation]})

    assert len(guidance) == 1
    assert "active_customer" in guidance[0]
    assert "confirming the requested metric version" in guidance[0]


def test_metric_drift_factory_executes_and_feeds_standard_reward_environment(tmp_path) -> None:
    database = tmp_path / "sales.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE sales(region TEXT, gross_amount REAL, refund_amount REAL)"
        )
        connection.executemany(
            "INSERT INTO sales VALUES (?, ?, ?)",
            [("east", 100.0, 10.0), ("east", 50.0, None), ("west", 80.0, 20.0)],
        )

    change = MetricDefinitionChange(
        metric_name="net_revenue",
        old_expression="SUM(gross_amount)",
        new_expression="SUM(gross_amount - COALESCE(refund_amount, 0))",
        from_version="metric-v1",
        to_version="metric-v2",
        reason="refunds are excluded from booked revenue",
    )
    example = build_metric_definition_example(
        source="unit-test",
        source_index=0,
        db_id="sales",
        question="按区域统计净收入",
        evidence="净收入按 metric-v2 口径计算",
        sql="SELECT region, SUM(gross_amount) AS revenue FROM sales GROUP BY region ORDER BY region",
        database=database,
        change=change,
    )

    active_database = tmp_path / "sales-v2.sqlite"
    materialize_schema_diff(database, active_database, example.schema_diff)
    assert fingerprint_query(active_database, example.repaired_sql) == example.result_fingerprint
    assert fingerprint_query(active_database, example.stale_sql) != example.result_fingerprint
    assert example.oracle_steps[0]["observation"]["semantic_mismatch"] is True
    assert example.schema_diff.operations[0]["type"] == "metric_definition_change"

    record = build_rl_record(example.to_dict(), index=0, split="train")
    assert record["extra_info"]["metric_version"] == "metric-v2"
    assert record["extra_info"]["result_fingerprint"] == {
        "row_count": example.result_fingerprint.row_count,
        "value_hash": example.result_fingerprint.value_hash,
    }

    correct = compute_score(
        data_source=record["data_source"],
        solution_str=tool_trace(
            [
                ("execute_sql", {"sql": example.stale_sql}),
                ("get_schema_version", {}),
                ("inspect_schema_diff", {}),
                ("execute_sql", {"sql": example.repaired_sql}),
                ("submit_solution", {"sql": example.repaired_sql}),
            ]
        ),
        ground_truth=record["reward_model"]["ground_truth"],
        extra_info=record["extra_info"],
    )
    stale = compute_score(
        data_source=record["data_source"],
        solution_str=tool_trace([("submit_solution", {"sql": example.stale_sql})]),
        ground_truth=record["reward_model"]["ground_truth"],
        extra_info=record["extra_info"],
    )
    assert correct["task_success"] is True
    assert correct["inspected_drift"] is True
    assert stale["task_success"] is False
