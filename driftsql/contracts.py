"""Shared data contracts used by generation, training, evaluation, and serving."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Task:
    task_id: str
    db_id: str
    db_version: str
    question: str
    metric_version: str = "v1"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentStep:
    action: str
    arguments: dict[str, Any]
    observation: Any
    reward_items: dict[str, float] = field(default_factory=dict)


@dataclass
class Trajectory:
    task: Task
    steps: list[AgentStep] = field(default_factory=list)
    final_sql: str | None = None
    success: bool = False
    total_reward: float = 0.0
    total_cost: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FailureRecord:
    trajectory_id: str
    failure_type: str
    severity: str
    drift_type: str | None = None
    validated: bool = False
    details: dict[str, Any] = field(default_factory=dict)
