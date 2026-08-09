from __future__ import annotations

from scripts.check_p6_candidate_gate import check_gate


def summary(*, tasks: int, success: int, drift_tasks: int, drift_success: int) -> dict:
    return {
        "alias": "candidate",
        "requested_metrics": {
            "tasks": tasks,
            "execution_success": success,
            "execution_success_rate": success / tasks,
            "drift_tasks": drift_tasks,
            "drift_recovery": drift_success,
            "drift_recovery_rate": drift_success / drift_tasks,
            "unsafe_tasks": 0,
            "timeout_tasks": 0,
        },
    }


def test_fast_gate_uses_stratified_absolute_thresholds() -> None:
    assert check_gate(summary(tasks=42, success=15, drift_tasks=35, drift_success=10), "fast")[
        "passed"
    ]
    assert not check_gate(
        summary(tasks=42, success=14, drift_tasks=35, drift_success=10), "fast"
    )["passed"]
    assert not check_gate(
        summary(tasks=42, success=15, drift_tasks=35, drift_success=9), "fast"
    )["passed"]


def test_full_dev_gate_requires_35_percent_and_strictly_beats_base_drift() -> None:
    assert check_gate(summary(tasks=169, success=60, drift_tasks=154, drift_success=41), "dev")[
        "passed"
    ]
    assert not check_gate(
        summary(tasks=169, success=59, drift_tasks=154, drift_success=41), "dev"
    )["passed"]
    assert not check_gate(
        summary(tasks=169, success=60, drift_tasks=154, drift_success=40), "dev"
    )["passed"]


def test_fast_gate_must_beat_same_protocol_base_when_supplied() -> None:
    candidate = summary(tasks=42, success=20, drift_tasks=35, drift_success=17)
    baseline = summary(tasks=42, success=19, drift_tasks=35, drift_success=16)
    candidate["alias"] = "candidate"
    baseline["alias"] = "base-controller"
    result = check_gate(candidate, "fast", baseline)
    assert result["passed"] is True
    assert result["baseline_alias"] == "base-controller"

    tied = summary(tasks=42, success=19, drift_tasks=35, drift_success=16)
    failed = check_gate(tied, "fast", baseline)
    assert failed["passed"] is False
    assert failed["checks"]["beats_same_protocol_baseline_overall"] is False
    assert failed["checks"]["beats_same_protocol_baseline_non_clean"] is False
