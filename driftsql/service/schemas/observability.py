"""Operational metrics, failure-explorer and external-run contracts."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import Field

from .common import SessionStatus, StrictModel


class DriftMetric(StrictModel):
    drift_type: str
    sessions: int = Field(ge=0)
    successful: int = Field(ge=0)
    success_rate: float = Field(ge=0, le=1)


class DailyMetric(StrictModel):
    day: date
    sessions: int = Field(ge=0)
    successful: int = Field(ge=0)
    failed: int = Field(ge=0)


class ModelDeployment(StrictModel):
    base_model: str
    adapter: str
    adapter_sha256: str
    sessions: int = Field(ge=0)
    successful: int = Field(ge=0)
    success_rate: float = Field(ge=0, le=1)


class OperationsSummary(StrictModel):
    generated_at: datetime
    total_sessions: int = Field(ge=0)
    terminal_sessions: int = Field(ge=0)
    active_sessions: int = Field(ge=0)
    successful_sessions: int = Field(ge=0)
    failed_sessions: int = Field(ge=0)
    unsafe_sessions: int = Field(ge=0)
    timed_out_sessions: int = Field(ge=0)
    success_rate: float = Field(ge=0, le=1)
    average_latency_ms: float = Field(ge=0)
    average_model_calls: float = Field(ge=0)
    average_tool_calls: float = Field(ge=0)
    total_prompt_tokens: int = Field(ge=0)
    total_response_tokens: int = Field(ge=0)
    tool_failure_rate: float = Field(ge=0, le=1)
    drift_metrics: list[DriftMetric]
    daily_metrics: list[DailyMetric]
    deployments: list[ModelDeployment]


FailureType = Literal[
    "unsafe",
    "timed_out",
    "budget_exhausted",
    "cancelled",
    "service_error",
    "task_failure",
]


class FailureRead(StrictModel):
    session_id: str
    scenario_id: str
    db_id: str
    drift_type: str
    status: SessionStatus
    failure_type: FailureType
    termination_reason: str | None
    error: str | None
    created_at: datetime
    completed_at: datetime | None
    model_calls: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    response_tokens: int = Field(ge=0)
    elapsed_ms: float = Field(ge=0)
    adapter_sha256: str


class FailureList(StrictModel):
    failures: list[FailureRead]
    total: int = Field(ge=0)
    counts: dict[str, int]


class WandbRunRead(StrictModel):
    run_id: str
    name: str
    state: str
    url: str
    created_at: datetime | None
    summary_metrics: dict[str, float]


class WandbRunList(StrictModel):
    provider: Literal["wandb"] = "wandb"
    configured: bool
    status: Literal["disabled", "ready", "error"]
    entity: str | None
    project: str
    project_url: str | None
    runs: list[WandbRunRead]
    error: str | None = None


class WandbMetricPoint(StrictModel):
    step: int = Field(ge=0)
    value: float


class WandbMetricSeries(StrictModel):
    name: str
    points: list[WandbMetricPoint]


class WandbRunHistory(StrictModel):
    provider: Literal["wandb"] = "wandb"
    configured: bool
    status: Literal["disabled", "ready", "error"]
    run_id: str
    series: list[WandbMetricSeries]
    error: str | None = None
