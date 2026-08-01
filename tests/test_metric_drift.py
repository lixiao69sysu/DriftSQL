from __future__ import annotations

import pytest

from driftsql.drift import MetricDefinitionChange, rewrite_metric_expression
from driftsql.integrations.state_policy import schema_diff_recovery_guidance


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
