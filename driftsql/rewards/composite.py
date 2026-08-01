"""Small reward contract shared by the MVP and future VERL adapter."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardBreakdown:
    task_success: float
    drift_detection: float
    semantic_validation: float
    tool_cost: float
    unsafe_action: float

    @property
    def total(self) -> float:
        return (
            self.task_success
            + self.drift_detection
            + self.semantic_validation
            - self.tool_cost
            - self.unsafe_action
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "task_success": self.task_success,
            "drift_detection": self.drift_detection,
            "semantic_validation": self.semantic_validation,
            "tool_cost": self.tool_cost,
            "unsafe_action": self.unsafe_action,
            "total": self.total,
        }


def compute_reward(
    success: bool,
    inspected_drift: bool,
    validated_result: bool,
    tool_cost: float,
    unsafe: bool = False,
) -> RewardBreakdown:
    return RewardBreakdown(
        task_success=1.0 if success else 0.0,
        drift_detection=0.2 if inspected_drift else 0.0,
        semantic_validation=0.1 if validated_result else 0.0,
        tool_cost=0.1 * max(tool_cost, 0.0),
        unsafe_action=1.0 if unsafe else 0.0,
    )
