"""Operational aggregation and optional W&B experiment discovery."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from numbers import Real
from typing import Any

from driftsql.service.repository import SQLiteSessionRepository
from driftsql.service.schemas import (
    DailyMetric,
    DriftMetric,
    EventType,
    FailureList,
    FailureRead,
    ModelDeployment,
    OperationsSummary,
    SessionRead,
    SessionStatus,
    WandbMetricPoint,
    WandbMetricSeries,
    WandbRunHistory,
    WandbRunList,
    WandbRunRead,
)
from driftsql.service.settings import ServiceSettings

ACTIVE_STATUSES = {SessionStatus.created, SessionStatus.queued, SessionStatus.running}
FAILURE_STATUSES = {
    SessionStatus.failed,
    SessionStatus.cancelled,
    SessionStatus.timed_out,
    SessionStatus.budget_exhausted,
}
WANDB_METRIC_HINTS = ("reward", "loss", "kl", "learning_rate", "lr", "throughput", "val/", "test/")


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _failure_type(session: SessionRead) -> str | None:
    if bool(session.result.get("unsafe")):
        return "unsafe"
    if session.status is SessionStatus.timed_out:
        return "timed_out"
    if session.status is SessionStatus.budget_exhausted:
        return "budget_exhausted"
    if session.status is SessionStatus.cancelled:
        return "cancelled"
    if session.status is SessionStatus.failed or session.result.get("error"):
        return "service_error"
    if session.status is SessionStatus.completed and not session.success:
        return "task_failure"
    return None


class OperationsService:
    def __init__(self, repository: SQLiteSessionRepository) -> None:
        self.repository = repository

    def summary(self) -> OperationsSummary:
        sessions = self.repository.list_sessions(limit=10000)
        terminal = [session for session in sessions if session.status not in ACTIVE_STATUSES]
        successful = [session for session in terminal if session.success is True]
        failed = [session for session in terminal if session.success is not True]
        unsafe = [session for session in terminal if bool(session.result.get("unsafe"))]
        timed_out = [session for session in terminal if session.status is SessionStatus.timed_out]

        tool_payloads = self.repository.list_event_payloads(EventType.tool)
        tool_failures = sum(not bool(payload.get("success")) for payload in tool_payloads)

        by_drift: dict[str, list[SessionRead]] = defaultdict(list)
        by_day: dict[Any, list[SessionRead]] = defaultdict(list)
        by_deployment: dict[tuple[str, str, str], list[SessionRead]] = defaultdict(list)
        for session in terminal:
            by_drift[session.drift_type].append(session)
            by_day[session.created_at.date()].append(session)
            model = session.model
            by_deployment[(model.base_model, model.adapter, model.adapter_sha256)].append(session)

        return OperationsSummary(
            generated_at=datetime.now(UTC),
            total_sessions=len(sessions),
            terminal_sessions=len(terminal),
            active_sessions=len(sessions) - len(terminal),
            successful_sessions=len(successful),
            failed_sessions=len(failed),
            unsafe_sessions=len(unsafe),
            timed_out_sessions=len(timed_out),
            success_rate=_rate(len(successful), len(terminal)),
            average_latency_ms=(sum(item.usage.elapsed_ms for item in terminal) / len(terminal) if terminal else 0),
            average_model_calls=(sum(item.usage.model_calls for item in terminal) / len(terminal) if terminal else 0),
            average_tool_calls=(sum(item.usage.tool_calls for item in terminal) / len(terminal) if terminal else 0),
            total_prompt_tokens=sum(item.usage.prompt_tokens for item in sessions),
            total_response_tokens=sum(item.usage.response_tokens for item in sessions),
            tool_failure_rate=_rate(tool_failures, len(tool_payloads)),
            drift_metrics=[
                DriftMetric(
                    drift_type=drift_type,
                    sessions=len(items),
                    successful=sum(item.success is True for item in items),
                    success_rate=_rate(sum(item.success is True for item in items), len(items)),
                )
                for drift_type, items in sorted(by_drift.items())
            ],
            daily_metrics=[
                DailyMetric(
                    day=day,
                    sessions=len(items),
                    successful=sum(item.success is True for item in items),
                    failed=sum(item.success is not True for item in items),
                )
                for day, items in sorted(by_day.items())
            ][-30:],
            deployments=[
                ModelDeployment(
                    base_model=key[0],
                    adapter=key[1],
                    adapter_sha256=key[2],
                    sessions=len(items),
                    successful=sum(item.success is True for item in items),
                    success_rate=_rate(sum(item.success is True for item in items), len(items)),
                )
                for key, items in sorted(by_deployment.items())
            ],
        )

    def failures(
        self,
        *,
        limit: int,
        offset: int,
        failure_type: str | None,
        drift_type: str | None,
    ) -> FailureList:
        failures: list[FailureRead] = []
        for session in self.repository.list_sessions(limit=10000):
            classified = _failure_type(session)
            if classified is None:
                continue
            if failure_type and classified != failure_type:
                continue
            if drift_type and session.drift_type != drift_type:
                continue
            failures.append(
                FailureRead(
                    session_id=session.session_id,
                    scenario_id=session.scenario_id,
                    db_id=session.db_id,
                    drift_type=session.drift_type,
                    status=session.status,
                    failure_type=classified,
                    termination_reason=session.termination_reason,
                    error=str(session.result.get("error")) if session.result.get("error") else None,
                    created_at=session.created_at,
                    completed_at=session.completed_at,
                    model_calls=session.usage.model_calls,
                    tool_calls=session.usage.tool_calls,
                    response_tokens=session.usage.response_tokens,
                    elapsed_ms=session.usage.elapsed_ms,
                    adapter_sha256=session.model.adapter_sha256,
                )
            )
        counts = Counter(item.failure_type for item in failures)
        return FailureList(
            failures=failures[offset : offset + limit],
            total=len(failures),
            counts=dict(sorted(counts.items())),
        )


class WandbService:
    def __init__(self, settings: ServiceSettings) -> None:
        self.settings = settings

    @property
    def configured(self) -> bool:
        return bool(self.settings.wandb_enabled and self.settings.wandb_entity)

    def list_runs(self) -> WandbRunList:
        entity = self.settings.wandb_entity
        project = self.settings.wandb_project
        project_url = f"https://wandb.ai/{entity}/{project}" if entity else None
        if not self.configured:
            return WandbRunList(
                configured=False,
                status="disabled",
                entity=entity,
                project=project,
                project_url=project_url,
                runs=[],
            )
        try:
            import wandb

            api_key = self.settings.wandb_api_key.get_secret_value() if self.settings.wandb_api_key else None
            api = wandb.Api(timeout=self.settings.wandb_timeout_seconds, api_key=api_key)
            runs = api.runs(f"{entity}/{project}", per_page=self.settings.wandb_max_runs)
            output: list[WandbRunRead] = []
            for run in list(runs)[: self.settings.wandb_max_runs]:
                raw_summary = dict(getattr(run.summary, "_json_dict", {}) or {})
                metrics = {
                    str(key): float(value)
                    for key, value in raw_summary.items()
                    if isinstance(value, Real)
                    and not isinstance(value, bool)
                    and any(hint in str(key).lower() for hint in WANDB_METRIC_HINTS)
                }
                created_at = getattr(run, "created_at", None)
                output.append(
                    WandbRunRead(
                        run_id=str(run.id),
                        name=str(run.name or run.id),
                        state=str(run.state or "unknown"),
                        url=str(run.url),
                        created_at=datetime.fromisoformat(created_at.replace("Z", "+00:00")) if created_at else None,
                        summary_metrics=metrics,
                    )
                )
            # W&B may return runs in creation order.  The Studio has limited
            # vertical space, so always surface the newest experiments first;
            # otherwise old connectivity tests and failed retries can hide the
            # selected SFT/GRPO curves.
            output.sort(
                key=lambda item: item.created_at or datetime.min.replace(tzinfo=UTC),
                reverse=True,
            )
            return WandbRunList(
                configured=True,
                status="ready",
                entity=entity,
                project=project,
                project_url=project_url,
                runs=output,
            )
        except Exception as error:  # noqa: BLE001 - optional provider must degrade safely.
            return WandbRunList(
                configured=True,
                status="error",
                entity=entity,
                project=project,
                project_url=project_url,
                runs=[],
                error=f"{type(error).__name__}: {error}",
            )

    def run_history(self, run_id: str) -> WandbRunHistory:
        if not self.configured:
            return WandbRunHistory(configured=False, status="disabled", run_id=run_id, series=[])
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", run_id):
            return WandbRunHistory(
                configured=True,
                status="error",
                run_id=run_id,
                series=[],
                error="Invalid W&B run id",
            )
        try:
            import wandb

            api_key = self.settings.wandb_api_key.get_secret_value() if self.settings.wandb_api_key else None
            api = wandb.Api(timeout=self.settings.wandb_timeout_seconds, api_key=api_key)
            run = api.run(f"{self.settings.wandb_entity}/{self.settings.wandb_project}/{run_id}")
            rows = run.history(samples=200, pandas=False)
            collected: dict[str, list[WandbMetricPoint]] = defaultdict(list)
            for row_index, row in enumerate(rows):
                step = int(row.get("_step", row_index))
                for key, value in row.items():
                    normalized = str(key).lower()
                    if (
                        key != "_step"
                        and isinstance(value, Real)
                        and not isinstance(value, bool)
                        and any(hint in normalized for hint in WANDB_METRIC_HINTS)
                    ):
                        collected[str(key)].append(WandbMetricPoint(step=max(0, step), value=float(value)))
            return WandbRunHistory(
                configured=True,
                status="ready",
                run_id=run_id,
                series=[
                    WandbMetricSeries(name=name, points=points) for name, points in sorted(collected.items()) if points
                ],
            )
        except Exception as error:  # noqa: BLE001 - optional provider must degrade safely.
            return WandbRunHistory(
                configured=True,
                status="error",
                run_id=run_id,
                series=[],
                error=f"{type(error).__name__}: {error}",
            )
