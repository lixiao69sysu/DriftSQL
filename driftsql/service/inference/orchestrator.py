"""Asynchronous service wrapper around the synchronous-turn DriftSQL agent."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from driftsql.rewards.agentic import compute_score
from driftsql.service.catalog import ScenarioCatalog
from driftsql.service.repository import SQLiteSessionRepository
from driftsql.service.schemas import (
    TERMINAL_STATUSES,
    EventType,
    InferenceBudget,
    RunCreate,
    SessionCreate,
    SessionRead,
    SessionStatus,
    TrajectoryEvent,
)
from driftsql.service.settings import ServiceSettings
from driftsql.tool_calls import find_tool_calls, remove_tool_call_payloads

from .backend import GenerationRequest, ModelBackend
from .tools import ToolRuntime


def utcnow() -> datetime:
    return datetime.now(UTC)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class ActiveSession:
    scenario_id: str
    create_kwargs: dict[str, Any]
    tool_names: list[str]
    messages: list[dict[str, Any]]
    tool_events: list[dict[str, Any]] = field(default_factory=list)
    model_outputs: list[str] = field(default_factory=list)
    cancel_requested: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task[None] | None = None


class SessionConflictError(RuntimeError):
    pass


class SessionOrchestrator:
    """Own session state, bounded GPU scheduling and append-only event delivery."""

    def __init__(
        self,
        settings: ServiceSettings,
        repository: SQLiteSessionRepository,
        catalog: ScenarioCatalog,
        backend: ModelBackend,
        tools: ToolRuntime,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.catalog = catalog
        self.backend = backend
        self.tools = tools
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_sessions)
        self._active: dict[str, ActiveSession] = {}
        self._conditions: dict[str, asyncio.Condition] = {}

    @property
    def active_count(self) -> int:
        return sum(
            self.repository.get_session(session_id).status not in TERMINAL_STATUSES for session_id in list(self._active)
        )

    async def create_session(self, request: SessionCreate) -> SessionRead:
        scenario = self.catalog.public_scenario(request.scenario_id)
        create_kwargs = self.catalog.create_kwargs(request.scenario_id)
        # The deterministic backend is used by CI and intentionally avoids a
        # worker thread (some GPU-enabled test environments cannot safely
        # initialize torch and SQLite thread pools together). Production vLLM
        # keeps database copy/query work off the event loop.
        if self.backend.metadata.backend == "scripted":
            create_kwargs["sync_io"] = True
        source_db = Path(str(create_kwargs["source_db"])).resolve()
        if not source_db.is_file():
            raise FileNotFoundError(source_db)
        session_id = str(uuid4())
        tool_names = self.catalog.tool_names(request.scenario_id)
        sandbox = await self.tools.initialize_session(session_id, tool_names, create_kwargs)
        if not sandbox["sandbox_isolated"]:
            await self.tools.release_session(session_id)
            raise RuntimeError("Sandbox tool did not create an isolated database session")
        now = utcnow()
        session = SessionRead(
            session_id=session_id,
            scenario_id=request.scenario_id,
            db_id=scenario.db_id,
            db_version=str(create_kwargs.get("db_version", "unknown")),
            status=SessionStatus.created,
            question=request.question or scenario.question,
            stale_sql=scenario.stale_sql,
            drift_type=scenario.drift_type,
            wildcard_profile=scenario.wildcard_profile,
            created_at=now,
            updated_at=now,
            sandbox_isolated=True,
            sandbox_ref=f"sandbox:{session_id}",
            source_db_sha256=file_sha256(source_db),
            model=self.backend.metadata,
            labels=request.labels,
            result={"database_session": "isolated-copy", "sandbox_engine": "sqlite"},
        )
        try:
            self.repository.create_session(session)
        except Exception:
            await self.tools.release_session(session_id)
            raise
        self._active[session_id] = ActiveSession(
            scenario_id=request.scenario_id,
            create_kwargs=create_kwargs,
            tool_names=tool_names,
            messages=self.catalog.prompt(request.scenario_id, question=request.question),
        )
        self._conditions[session_id] = asyncio.Condition()
        await self._emit(
            session_id,
            EventType.session,
            {
                "status": session.status.value,
                "scenario_id": session.scenario_id,
                "db_id": session.db_id,
                "db_version": session.db_version,
                "sandbox_ref": session.sandbox_ref,
                "model": session.model.model_dump(mode="json"),
            },
        )
        return session

    def _budget(self, request: RunCreate) -> InferenceBudget:
        maximum_turns = request.max_turns or self.settings.default_max_turns
        timeout = request.timeout_seconds or self.settings.default_timeout_seconds
        if maximum_turns > self.settings.maximum_max_turns:
            raise ValueError(f"max_turns exceeds service maximum {self.settings.maximum_max_turns}")
        if timeout > self.settings.maximum_timeout_seconds:
            raise ValueError(f"timeout_seconds exceeds service maximum {self.settings.maximum_timeout_seconds}")
        return InferenceBudget(
            max_turns=maximum_turns,
            timeout_seconds=timeout,
            max_tool_calls=request.max_tool_calls or self.settings.default_max_tool_calls,
            max_new_tokens=request.max_new_tokens or self.settings.default_max_new_tokens,
            max_total_tokens=request.max_total_tokens or self.settings.default_max_total_tokens,
        )

    async def start_run(self, session_id: str, request: RunCreate) -> SessionRead:
        session = self.repository.get_session(session_id)
        active = self._active.get(session_id)
        if active is None:
            raise SessionConflictError("Session is not resumable in this service process")
        if session.status is not SessionStatus.created:
            raise SessionConflictError(f"Session is already {session.status.value}")
        budget = self._budget(request)
        session = self.repository.save_session(
            session.model_copy(update={"status": SessionStatus.queued, "budget": budget})
        )
        await self._emit(session_id, EventType.queued, {"budget": budget.model_dump()})
        active.task = asyncio.create_task(self._run_guarded(session_id), name=f"driftsql-{session_id}")
        return session

    async def cancel(self, session_id: str) -> SessionRead:
        session = self.repository.get_session(session_id)
        if session.status in TERMINAL_STATUSES:
            return session
        active = self._active.get(session_id)
        if active is None:
            raise SessionConflictError("Session is not active in this service process")
        active.cancel_requested.set()
        session = self.repository.save_session(session.model_copy(update={"cancellation_requested": True}))
        await self._emit(session_id, EventType.cancelled, {"requested": True})
        await self.backend.abort(session_id)
        return session

    async def _run_guarded(self, session_id: str) -> None:
        started = time.perf_counter()
        terminal = SessionStatus.failed
        reason = "internal_error"
        error_detail = ""
        try:
            session = self.repository.get_session(session_id)
            assert session.budget is not None
            async with asyncio.timeout(session.budget.timeout_seconds):
                async with self._semaphore:
                    if self._active[session_id].cancel_requested.is_set():
                        raise asyncio.CancelledError
                    session = self.repository.save_session(
                        session.model_copy(update={"status": SessionStatus.running, "started_at": utcnow()})
                    )
                    await self._emit(session_id, EventType.status, {"status": "running"})
                    terminal, reason = await self._agent_loop(session_id)
        except TimeoutError:
            terminal, reason = SessionStatus.timed_out, "timeout"
            await self.backend.abort(session_id)
        except asyncio.CancelledError:
            terminal, reason = SessionStatus.cancelled, "cancelled_by_client"
        except Exception as error:  # noqa: BLE001 - event log must capture service failures.
            if self._active[session_id].cancel_requested.is_set():
                terminal, reason = SessionStatus.cancelled, "cancelled_by_client"
            else:
                terminal, reason, error_detail = SessionStatus.failed, "internal_error", repr(error)
                await self._emit(session_id, EventType.error, {"error": error_detail})
        finally:
            await self._finalize(session_id, terminal, reason, started, error_detail)

    async def _agent_loop(self, session_id: str) -> tuple[SessionStatus, str]:
        active = self._active[session_id]
        session = self.repository.get_session(session_id)
        budget = session.budget
        assert budget is not None
        for turn in range(1, budget.max_turns + 1):
            if active.cancel_requested.is_set():
                raise asyncio.CancelledError
            schemas = self.tools.schemas_for(session_id, active.tool_events)
            generation = await self.backend.generate(
                GenerationRequest(
                    session_id=session_id,
                    scenario_id=active.scenario_id,
                    messages=active.messages,
                    tools=schemas,
                    create_kwargs=active.create_kwargs,
                    tool_events=active.tool_events,
                    max_new_tokens=budget.max_new_tokens,
                )
            )
            active.model_outputs.append(generation.text)
            usage = session.usage.model_copy(
                update={
                    "model_calls": session.usage.model_calls + 1,
                    "prompt_tokens": session.usage.prompt_tokens + generation.prompt_tokens,
                    "response_tokens": session.usage.response_tokens + generation.response_tokens,
                }
            )
            session = self.repository.save_session(session.model_copy(update={"usage": usage}))
            calls = find_tool_calls(generation.text)
            await self._emit(
                session_id,
                EventType.model,
                {
                    "turn": turn,
                    "content": generation.text,
                    "tool_name": calls[0].name if calls else None,
                    "prompt_tokens": generation.prompt_tokens,
                    "response_tokens": generation.response_tokens,
                    "elapsed_ms": generation.elapsed_ms,
                },
            )
            if usage.prompt_tokens + usage.response_tokens > budget.max_total_tokens:
                await self._emit(session_id, EventType.budget, {"exhausted": "max_total_tokens"})
                return SessionStatus.budget_exhausted, "max_total_tokens"
            if not calls:
                await self._emit(session_id, EventType.error, {"turn": turn, "error": "missing_tool_call"})
                continue
            call = calls[0]
            assistant_content = remove_tool_call_payloads(generation.text, [call])
            active.messages.append(
                {
                    "role": "assistant",
                    "content": assistant_content,
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                        }
                    ],
                }
            )
            execution = await self.tools.execute(
                session_id,
                call.name,
                call.arguments,
                active.tool_events,
            )
            tool_event = {
                "turn": turn,
                "tool": call.name,
                "arguments": call.arguments,
                "response": execution.observation,
                "metrics": execution.metrics,
                "reward": execution.reward,
            }
            active.tool_events.append(tool_event)
            active.messages.append(
                {
                    "role": "tool",
                    "content": execution.observation,
                }
            )
            usage = session.usage.model_copy(update={"tool_calls": session.usage.tool_calls + 1})
            update: dict[str, Any] = {"usage": usage}
            if call.name == "submit_solution" and execution.metrics.get("submitted"):
                update["final_sql"] = str(call.arguments.get("sql", ""))
            session = self.repository.save_session(session.model_copy(update=update))
            await self._emit(
                session_id,
                EventType.tool,
                {
                    "turn": turn,
                    "tool": call.name,
                    "arguments": call.arguments,
                    "observation": execution.observation,
                    "metrics": execution.metrics,
                    "reward": execution.reward,
                    "elapsed_ms": execution.elapsed_ms,
                    "success": not bool(
                        execution.metrics.get("execution_error")
                        or execution.metrics.get("action_masked")
                        or execution.metrics.get("duplicate_retrieval")
                    ),
                },
            )
            self.repository.save_trajectory_state(session_id, messages=active.messages, reward={})
            if usage.tool_calls >= budget.max_tool_calls and call.name != "submit_solution":
                await self._emit(session_id, EventType.budget, {"exhausted": "max_tool_calls"})
                return SessionStatus.budget_exhausted, "max_tool_calls"
            if call.name == "submit_solution" and execution.metrics.get("submitted"):
                return SessionStatus.completed, "submitted"
        await self._emit(session_id, EventType.budget, {"exhausted": "max_turns"})
        return SessionStatus.budget_exhausted, "max_turns"

    async def _finalize(
        self,
        session_id: str,
        terminal: SessionStatus,
        reason: str,
        started: float,
        error_detail: str,
    ) -> None:
        active = self._active.get(session_id)
        if active is None:
            return
        try:
            session = self.repository.get_session(session_id)
            reward_extra = self.catalog.reward_extra_info(active.scenario_id)
            reward_extra.update(
                {
                    "environment_events": active.tool_events,
                    "response_tokens": session.usage.response_tokens,
                    "trajectory_timed_out": terminal is SessionStatus.timed_out,
                    "trajectory_turn_limit": reason == "max_turns",
                }
            )
            reward_args = (
                "driftsql_service",
                "\n".join(active.model_outputs),
                None,
                reward_extra,
            )
            try:
                if self.backend.metadata.backend == "scripted":
                    reward = compute_score(*reward_args)
                else:
                    reward = await asyncio.to_thread(compute_score, *reward_args)
            except Exception as reward_error:  # noqa: BLE001 - preserve terminal state.
                reward = {
                    "score": 0.0,
                    "task_success": False,
                    "unsafe": False,
                    "error": f"reward_error: {reward_error!r}",
                }
            elapsed_ms = (time.perf_counter() - started) * 1000
            usage = session.usage.model_copy(update={"elapsed_ms": elapsed_ms})
            result = dict(session.result)
            result.update(
                {
                    "reward": reward.get("score"),
                    "task_success": bool(reward.get("task_success")),
                    "unsafe": bool(reward.get("unsafe")),
                }
            )
            if error_detail:
                result["error"] = error_detail
            session = self.repository.save_session(
                session.model_copy(
                    update={
                        "status": terminal,
                        "termination_reason": reason,
                        "completed_at": utcnow(),
                        "success": bool(reward.get("task_success")),
                        "usage": usage,
                        "result": result,
                    }
                )
            )
            self.repository.save_trajectory_state(
                session_id,
                messages=active.messages,
                reward=reward,
            )
            await self._emit(session_id, EventType.reward, reward)
            await self._emit(
                session_id,
                EventType.status,
                {"status": terminal.value, "termination_reason": reason},
            )
        finally:
            await self.tools.release_session(session_id)
            self._active.pop(session_id, None)
            self._conditions.pop(session_id, None)

    async def _emit(
        self,
        session_id: str,
        event_type: EventType,
        payload: dict[str, Any],
    ) -> TrajectoryEvent:
        event = self.repository.append_event(session_id, event_type, payload)
        condition = self._conditions.get(session_id)
        if condition is not None:
            async with condition:
                condition.notify_all()
        return event

    async def event_stream(
        self,
        session_id: str,
        after_sequence: int = 0,
    ) -> AsyncIterator[TrajectoryEvent | None]:
        self.repository.get_session(session_id)
        cursor = after_sequence
        while True:
            events = self.repository.list_events(session_id, after_sequence=cursor)
            for event in events:
                cursor = event.sequence
                yield event
            session = self.repository.get_session(session_id)
            if session.status in TERMINAL_STATUSES:
                return
            condition = self._conditions.setdefault(session_id, asyncio.Condition())
            try:
                async with condition:
                    await asyncio.wait_for(condition.wait(), timeout=15.0)
            except TimeoutError:
                yield None

    async def shutdown(self) -> None:
        tasks = []
        for active in self._active.values():
            if active.task is not None and not active.task.done():
                active.cancel_requested.set()
                active.task.cancel()
                tasks.append(active.task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self.tools.shutdown()
