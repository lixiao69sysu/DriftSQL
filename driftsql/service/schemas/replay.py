"""Human-review contracts for P4 failure replay candidates."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from .common import StrictModel

ReplayDecision = Literal["approve", "reject"]
ReplayReviewStatus = Literal["pending", "approve", "reject"]


class ReplayReviewCreate(StrictModel):
    decision: ReplayDecision
    reviewer: str = Field(min_length=2, max_length=100)
    reason: str = Field(min_length=8, max_length=1000)


class ReplayCandidateRead(StrictModel):
    candidate_id: str
    session_id: str
    scenario_id: str
    db_id: str
    drift_type: str
    wildcard_profile: str | None
    added_column_count: int | None
    failure_type: str
    failure_class: str
    session_status: str
    termination_reason: str | None
    success: bool | None
    model_calls: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    tool_sequence: list[str]
    final_sql: str | None
    reward_score: float | None
    trajectory_sha256: str
    review_status: ReplayReviewStatus
    reviewer: str | None = None
    review_reason: str | None = None
    reviewed_at: datetime | None = None


class ReplayCandidateList(StrictModel):
    available: bool
    candidates: list[ReplayCandidateRead]
    total: int = Field(ge=0)
    counts: dict[str, int]

