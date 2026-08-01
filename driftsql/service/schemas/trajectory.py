"""Complete replayable trajectory response."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from .common import StrictModel
from .event import TrajectoryEvent
from .session import SessionRead


class TrajectoryRead(StrictModel):
    session: SessionRead
    events: list[TrajectoryEvent]
    messages: list[dict[str, Any]] = Field(default_factory=list)
    reward: dict[str, Any] = Field(default_factory=dict)
