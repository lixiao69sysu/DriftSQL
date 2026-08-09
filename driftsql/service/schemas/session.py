"""Session lifecycle and demo-scenario API contracts."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field

from .common import InferenceBudget, ModelMetadata, SessionStatus, StrictModel, UsageMetrics


class ScenarioRead(StrictModel):
    scenario_id: str
    db_id: str
    question: str
    stale_sql: str
    drift_type: str
    wildcard_profile: str | None = None
    difficulty: str | None = None
    schema_diff: dict[str, Any]


class DatabaseRead(StrictModel):
    db_id: str
    scenario_count: int
    drift_types: list[str]


class DatabasePathRead(StrictModel):
    """A safe logical database path exposed to CLI context completion."""

    path: str
    kind: Literal["database", "table", "column"]
    db_id: str
    table: str | None = None
    column: str | None = None
    data_type: str | None = None


class SessionCreate(StrictModel):
    scenario_id: str
    question: str | None = Field(default=None, min_length=1, max_length=4000)
    labels: dict[str, str] = Field(default_factory=dict)


class QuerySessionCreate(StrictModel):
    db_id: str = Field(min_length=1, max_length=200)
    question: str = Field(min_length=1, max_length=4000)
    locale: Literal["en-US", "zh-CN"] = "zh-CN"
    labels: dict[str, str] = Field(default_factory=dict)


class SessionMode(StrEnum):
    recovery = "recovery"
    query = "query"


class RunCreate(StrictModel):
    max_turns: int | None = Field(default=None, ge=1, le=64)
    timeout_seconds: float | None = Field(default=None, gt=0, le=3600)
    max_tool_calls: int | None = Field(default=None, ge=1, le=128)
    max_new_tokens: int | None = Field(default=None, ge=16, le=4096)
    max_total_tokens: int | None = Field(default=None, ge=64, le=65536)


class SessionRead(StrictModel):
    session_id: str
    scenario_id: str
    db_id: str
    db_version: str
    mode: SessionMode = SessionMode.recovery
    status: SessionStatus
    question: str
    stale_sql: str
    drift_type: str
    wildcard_profile: str | None = None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    termination_reason: str | None = None
    final_sql: str | None = None
    success: bool | None = None
    cancellation_requested: bool = False
    sandbox_isolated: bool
    sandbox_ref: str
    source_db_sha256: str
    model: ModelMetadata
    budget: InferenceBudget | None = None
    usage: UsageMetrics = Field(default_factory=UsageMetrics)
    labels: dict[str, str] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)


class SessionList(StrictModel):
    sessions: list[SessionRead]
    total: int
