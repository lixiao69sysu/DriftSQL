"""Shared API enumerations and immutable inference metadata."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SessionStatus(StrEnum):
    created = "created"
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"
    timed_out = "timed_out"
    budget_exhausted = "budget_exhausted"


TERMINAL_STATUSES = {
    SessionStatus.completed,
    SessionStatus.failed,
    SessionStatus.cancelled,
    SessionStatus.timed_out,
    SessionStatus.budget_exhausted,
}


class EventType(StrEnum):
    session = "session"
    queued = "queued"
    status = "status"
    model = "model"
    tool = "tool"
    reward = "reward"
    budget = "budget"
    error = "error"
    cancelled = "cancelled"


class ModelMetadata(StrictModel):
    backend: str
    base_model: str
    adapter: str
    adapter_sha256: str
    frozen_candidate_sha256: str
    persistent: bool = True
    loaded: bool = False


class InferenceBudget(StrictModel):
    max_turns: int = Field(ge=1)
    timeout_seconds: float = Field(gt=0)
    max_tool_calls: int = Field(ge=1)
    max_new_tokens: int = Field(ge=1)
    max_total_tokens: int = Field(ge=1)


class UsageMetrics(StrictModel):
    model_calls: int = 0
    tool_calls: int = 0
    prompt_tokens: int = 0
    response_tokens: int = 0
    elapsed_ms: float = 0.0


class HealthRead(StrictModel):
    status: str
    service: str
    version: str
    model: ModelMetadata
    active_sessions: int
    max_concurrent_sessions: int
    repository: str
    timestamp: datetime
    details: dict[str, Any] = Field(default_factory=dict)
