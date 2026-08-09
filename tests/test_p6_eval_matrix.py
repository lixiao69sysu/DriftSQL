from __future__ import annotations

from scripts.summarize_p6_eval_matrix import metrics


def row(*, success: bool, reason: str, drift: str, masked: int = 0) -> dict:
    return {
        "task_success": success,
        "termination_reason": reason,
        "drift_type": drift,
        "executable": reason == "submitted",
        "safety": {"unsafe": False, "timed_out": False},
        "usage": {"tool_calls": 2 + masked, "model_calls": 2 + masked},
        "trajectory": [{"metrics": {"action_masked": True}} for _ in range(masked)],
    }


def test_metrics_counts_success_safety_cost_and_masked_actions() -> None:
    result = metrics(
        [
            row(success=True, reason="submitted", drift="clean"),
            row(success=False, reason="turn_limit", drift="rename_table", masked=2),
        ]
    )
    assert result["tasks"] == 2
    assert result["success"] == 1
    assert result["success_rate"] == 0.5
    assert result["safe_submission_precision"] == 1.0
    assert result["turn_limit"] == 1
    assert result["masked_actions"] == 2
    assert result["masked_tasks"] == 1
    assert result["average_tool_calls"] == 3.0
