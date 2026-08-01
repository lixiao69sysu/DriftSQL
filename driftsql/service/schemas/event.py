"""Append-only trajectory event contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from .common import EventType, StrictModel


class TrajectoryEvent(StrictModel):
    session_id: str
    sequence: int = Field(ge=1)
    event_type: EventType
    created_at: datetime
    payload: dict[str, Any]


class ToolEventPayload(StrictModel):
    turn: int = Field(ge=1)
    tool: str
    arguments: dict[str, Any]
    observation: Any
    metrics: dict[str, Any] = Field(default_factory=dict)
    reward: float = 0.0
    elapsed_ms: float = 0.0
    success: bool


class ModelEventPayload(StrictModel):
    turn: int = Field(ge=1)
    content: str
    tool_name: str | None = None
    prompt_tokens: int = 0
    response_tokens: int = 0
    elapsed_ms: float = 0.0
