"""Sanitized experiment-comparison contracts for the Studio dashboard."""

from __future__ import annotations

from pydantic import Field

from .common import StrictModel


class ExperimentRead(StrictModel):
    experiment_id: str
    display_name: str
    category: str
    tasks: int = Field(ge=0)
    task_success_rate: float = Field(ge=0, le=1)
    executable_rate: float = Field(ge=0, le=1)
    average_model_calls: float = Field(ge=0)
    average_tool_calls: float = Field(ge=0)
    unsafe_tasks: int = Field(ge=0)
    selected: bool = False


class ExperimentList(StrictModel):
    experiments: list[ExperimentRead]
    selected_experiment_id: str
