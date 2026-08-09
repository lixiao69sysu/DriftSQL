"""Product HTTP and server-sent-event API."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from hmac import compare_digest
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from driftsql.service.auth import AuthSessionStore
from driftsql.service.catalog import (
    ExperimentCatalog,
    ScenarioCatalog,
    ScenarioNotFoundError,
)
from driftsql.service.inference import SessionOrchestrator
from driftsql.service.inference.orchestrator import SessionConflictError
from driftsql.service.model_catalog import ModelCatalog, ModelNotFoundError, ModelUnavailableError
from driftsql.service.observability import OperationsService, WandbService
from driftsql.service.replay import (
    ReplayCandidateNotFoundError,
    ReplayReviewStore,
    ReplayReviewUnavailableError,
)
from driftsql.service.repository import SessionNotFoundError, SQLiteSessionRepository
from driftsql.service.schemas import (
    AuthLogin,
    AuthStatus,
    DatabasePathRead,
    DatabaseRead,
    ExperimentList,
    FailureList,
    HealthRead,
    ModelActivate,
    ModelList,
    OperationsSummary,
    QuerySessionCreate,
    ReplayCandidateList,
    ReplayCandidateRead,
    ReplayReviewCreate,
    RunCreate,
    ScenarioRead,
    SessionCreate,
    SessionList,
    SessionRead,
    TrajectoryRead,
    WandbRunHistory,
    WandbRunList,
)
from driftsql.service.settings import ServiceSettings
from driftsql.service.translation import TranslationUnavailableError

from .dependencies import (
    get_auth_sessions,
    get_catalog,
    get_experiment_catalog,
    get_model_catalog,
    get_operations,
    get_orchestrator,
    get_replay_reviews,
    get_repository,
    get_settings,
    get_wandb,
)

router = APIRouter()
AUTH_COOKIE = "driftsql_session"


@router.get("/auth/status", response_model=AuthStatus, tags=["authentication"], summary="Browser authentication status")
async def auth_status(
    request: Request,
    settings: Annotated[ServiceSettings, Depends(get_settings)],
    sessions: Annotated[AuthSessionStore, Depends(get_auth_sessions)],
) -> AuthStatus:
    if not settings.auth_enabled:
        return AuthStatus(enabled=False, authenticated=True)
    authenticated = sessions.validate(request.cookies.get(AUTH_COOKIE))
    return AuthStatus(enabled=True, authenticated=authenticated)


@router.post(
    "/auth/login",
    response_model=AuthStatus,
    tags=["authentication"],
    summary="Exchange API key for HttpOnly browser session",
)
async def auth_login(
    body: AuthLogin,
    settings: Annotated[ServiceSettings, Depends(get_settings)],
    sessions: Annotated[AuthSessionStore, Depends(get_auth_sessions)],
) -> JSONResponse:
    if not settings.auth_enabled:
        return JSONResponse(AuthStatus(enabled=False, authenticated=True).model_dump(mode="json"))
    expected = settings.api_key.get_secret_value() if settings.api_key else ""
    supplied = body.api_key.get_secret_value()
    if not supplied or not compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Invalid DriftSQL API key")
    token, expires_at = sessions.create()
    payload = AuthStatus(enabled=True, authenticated=True, expires_at=expires_at)
    response = JSONResponse(payload.model_dump(mode="json"))
    response.set_cookie(
        key=AUTH_COOKIE,
        value=token,
        max_age=settings.auth_session_ttl_seconds,
        httponly=True,
        secure=settings.auth_cookie_secure,
        samesite="strict",
        path="/",
    )
    return response


@router.post("/auth/logout", response_model=AuthStatus, tags=["authentication"], summary="Revoke browser session")
async def auth_logout(
    request: Request,
    settings: Annotated[ServiceSettings, Depends(get_settings)],
    sessions: Annotated[AuthSessionStore, Depends(get_auth_sessions)],
) -> JSONResponse:
    sessions.revoke(request.cookies.get(AUTH_COOKIE))
    payload = AuthStatus(
        enabled=settings.auth_enabled,
        authenticated=not settings.auth_enabled,
    )
    response = JSONResponse(payload.model_dump(mode="json"))
    response.delete_cookie(AUTH_COOKIE, path="/")
    return response


@router.get("/health", response_model=HealthRead, tags=["operations"], summary="Service readiness")
async def health(
    settings: Annotated[ServiceSettings, Depends(get_settings)],
    orchestrator: Annotated[SessionOrchestrator, Depends(get_orchestrator)],
) -> HealthRead:
    return HealthRead(
        status="ready" if orchestrator.backend.metadata.loaded else "starting",
        service=settings.service_name,
        version=settings.service_version,
        model=orchestrator.backend.metadata,
        active_sessions=orchestrator.active_count,
        max_concurrent_sessions=settings.max_concurrent_sessions,
        repository="sqlite",
        timestamp=datetime.now(UTC),
        details={
            "sandbox": "isolated-sqlite",
            "sql_policy": "read-only",
            "translation": {
                "enabled": settings.translation_enabled,
                "model_id": orchestrator.translator.model_id,
                "available": orchestrator.translator.available,
                "loaded": orchestrator.translator.loaded,
                "device": "cpu",
            },
        },
    )


@router.get("/api/models", response_model=ModelList, tags=["catalog"], summary="Registered model catalogue")
async def models(
    orchestrator: Annotated[SessionOrchestrator, Depends(get_orchestrator)],
    catalog: Annotated[ModelCatalog, Depends(get_model_catalog)],
) -> ModelList:
    return catalog.list_models(orchestrator.backend.metadata)


@router.post(
    "/api/models/activate",
    response_model=ModelList,
    tags=["catalog"],
    summary="Activate a registered model for new sessions",
)
async def activate_model(
    body: ModelActivate,
    orchestrator: Annotated[SessionOrchestrator, Depends(get_orchestrator)],
    catalog: Annotated[ModelCatalog, Depends(get_model_catalog)],
) -> ModelList:
    if orchestrator.active_count:
        raise HTTPException(status_code=409, detail="Cannot switch model while a Session is active")
    try:
        model = catalog.get(body.model_id)
    except ModelNotFoundError as error:
        raise HTTPException(status_code=404, detail=f"Unknown model: {body.model_id}") from error
    except ModelUnavailableError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    try:
        await orchestrator.backend.activate_model(
            model_id=model.model_id,
            base_model_path=model.base_model,
            adapter_path=model.adapter,
            adapter_sha256=model.adapter_sha256,
        )
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return catalog.list_models(orchestrator.backend.metadata)


@router.get("/api/scenarios", response_model=list[ScenarioRead], tags=["catalog"], summary="Tune-only demos")
async def scenarios(catalog: Annotated[ScenarioCatalog, Depends(get_catalog)]) -> list[ScenarioRead]:
    return catalog.list_scenarios()


@router.get("/api/databases", response_model=list[DatabaseRead], tags=["catalog"], summary="Demo databases")
async def databases(catalog: Annotated[ScenarioCatalog, Depends(get_catalog)]) -> list[DatabaseRead]:
    return catalog.list_databases()


@router.get(
    "/api/database-paths",
    response_model=list[DatabasePathRead],
    tags=["catalog"],
    summary="Safe logical database paths for CLI context completion",
)
async def database_paths(catalog: Annotated[ScenarioCatalog, Depends(get_catalog)]) -> list[DatabasePathRead]:
    return catalog.list_database_paths()


@router.get(
    "/api/experiments",
    response_model=ExperimentList,
    tags=["catalog"],
    summary="Frozen Tune experiment comparison",
)
async def experiments(
    catalog: Annotated[ExperimentCatalog, Depends(get_experiment_catalog)],
) -> ExperimentList:
    return catalog.list_experiments()


@router.get(
    "/api/observability/summary",
    response_model=OperationsSummary,
    tags=["observability"],
    summary="Aggregate persisted product-run metrics",
)
async def operations_summary(
    operations: Annotated[OperationsService, Depends(get_operations)],
) -> OperationsSummary:
    return operations.summary()


@router.get(
    "/api/observability/failures",
    response_model=FailureList,
    tags=["observability"],
    summary="Classified failed sessions for replay",
)
async def failures(
    operations: Annotated[OperationsService, Depends(get_operations)],
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    failure_type: str | None = Query(
        default=None, pattern="^(unsafe|timed_out|budget_exhausted|cancelled|service_error|task_failure)$"
    ),
    drift_type: str | None = Query(default=None, min_length=1, max_length=100),
) -> FailureList:
    return operations.failures(
        limit=limit,
        offset=offset,
        failure_type=failure_type,
        drift_type=drift_type,
    )


@router.get(
    "/api/replay/candidates",
    response_model=ReplayCandidateList,
    tags=["replay"],
    summary="List immutable P4 failure candidates and latest human decisions",
)
async def replay_candidates(
    reviews: Annotated[ReplayReviewStore, Depends(get_replay_reviews)],
    review_status: str | None = Query(default=None, pattern="^(pending|approve|reject)$"),
) -> ReplayCandidateList:
    try:
        return reviews.list_candidates(review_status=review_status)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.post(
    "/api/replay/candidates/{candidate_id}/reviews",
    response_model=ReplayCandidateRead,
    tags=["replay"],
    summary="Append a hash-bound human replay decision",
)
async def review_replay_candidate(
    candidate_id: str,
    body: ReplayReviewCreate,
    reviews: Annotated[ReplayReviewStore, Depends(get_replay_reviews)],
) -> ReplayCandidateRead:
    try:
        return reviews.review(candidate_id, body)
    except ReplayCandidateNotFoundError as error:
        raise HTTPException(status_code=404, detail="Replay candidate not found") from error
    except ReplayReviewUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.get(
    "/api/observability/wandb/runs",
    response_model=WandbRunList,
    tags=["observability"],
    summary="Optional server-side W&B experiment discovery",
)
async def wandb_runs(
    wandb: Annotated[WandbService, Depends(get_wandb)],
) -> WandbRunList:
    if not wandb.configured:
        return wandb.list_runs()
    return await asyncio.to_thread(wandb.list_runs)


@router.get(
    "/api/observability/wandb/runs/{run_id}/history",
    response_model=WandbRunHistory,
    tags=["observability"],
    summary="Sampled reward, KL, loss and learning curves for one W&B run",
)
async def wandb_run_history(
    run_id: str,
    wandb: Annotated[WandbService, Depends(get_wandb)],
) -> WandbRunHistory:
    if not wandb.configured:
        return wandb.run_history(run_id)
    return await asyncio.to_thread(wandb.run_history, run_id)


@router.post(
    "/api/sessions",
    response_model=SessionRead,
    status_code=status.HTTP_201_CREATED,
    tags=["sessions"],
    summary="Create an isolated database session",
)
async def create_session(
    body: SessionCreate,
    orchestrator: Annotated[SessionOrchestrator, Depends(get_orchestrator)],
) -> SessionRead:
    try:
        return await orchestrator.create_session(body)
    except ScenarioNotFoundError as error:
        raise HTTPException(status_code=404, detail=f"Unknown scenario: {body.scenario_id}") from error


@router.post(
    "/api/query-sessions",
    response_model=SessionRead,
    status_code=status.HTTP_201_CREATED,
    tags=["sessions"],
    summary="Create an execution-verified free-form database query session",
)
async def create_query_session(
    body: QuerySessionCreate,
    orchestrator: Annotated[SessionOrchestrator, Depends(get_orchestrator)],
) -> SessionRead:
    try:
        return await orchestrator.create_query_session(body)
    except ScenarioNotFoundError as error:
        raise HTTPException(status_code=404, detail=f"Unknown database: {body.db_id}") from error
    except TranslationUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@router.get("/api/sessions", response_model=SessionList, tags=["sessions"], summary="List sessions")
async def list_sessions(
    repository: Annotated[SQLiteSessionRepository, Depends(get_repository)],
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> SessionList:
    return SessionList(
        sessions=repository.list_sessions(limit=limit, offset=offset),
        total=repository.count_sessions(),
    )


@router.get(
    "/api/sessions/{session_id}",
    response_model=SessionRead,
    tags=["sessions"],
    summary="Read session status",
)
async def get_session(
    session_id: str,
    repository: Annotated[SQLiteSessionRepository, Depends(get_repository)],
) -> SessionRead:
    try:
        return repository.get_session(session_id)
    except SessionNotFoundError as error:
        raise HTTPException(status_code=404, detail="Session not found") from error


@router.post(
    "/api/sessions/{session_id}/run",
    response_model=SessionRead,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["inference"],
    summary="Queue the agent run",
)
async def run_session(
    session_id: str,
    body: RunCreate,
    orchestrator: Annotated[SessionOrchestrator, Depends(get_orchestrator)],
) -> SessionRead:
    try:
        return await orchestrator.start_run(session_id, body)
    except SessionNotFoundError as error:
        raise HTTPException(status_code=404, detail="Session not found") from error
    except (SessionConflictError, ValueError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.post(
    "/api/sessions/{session_id}/cancel",
    response_model=SessionRead,
    tags=["inference"],
    summary="Cancel a queued or running agent",
)
async def cancel_session(
    session_id: str,
    orchestrator: Annotated[SessionOrchestrator, Depends(get_orchestrator)],
) -> SessionRead:
    try:
        return await orchestrator.cancel(session_id)
    except SessionNotFoundError as error:
        raise HTTPException(status_code=404, detail="Session not found") from error
    except SessionConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.get(
    "/api/sessions/{session_id}/events",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Replayable trajectory event stream; terminal status closes the stream.",
            "content": {"text/event-stream": {"schema": {"type": "string"}}},
        }
    },
    tags=["inference"],
    summary="Stream trajectory events",
)
async def stream_events(
    session_id: str,
    orchestrator: Annotated[SessionOrchestrator, Depends(get_orchestrator)],
    after_sequence: int = Query(default=0, ge=0),
) -> StreamingResponse:
    try:
        orchestrator.repository.get_session(session_id)
    except SessionNotFoundError as error:
        raise HTTPException(status_code=404, detail="Session not found") from error

    async def generate():
        async for event in orchestrator.event_stream(session_id, after_sequence):
            if event is None:
                yield ": keepalive\n\n"
                continue
            payload = event.model_dump(mode="json")
            # `error` is reserved by the browser EventSource API for transport
            # failures. Keep the persisted EventType unchanged, but use a
            # distinct wire name so recoverable Agent errors do not look like
            # a disconnected stream.
            wire_type = "agent_error" if event.event_type.value == "error" else event.event_type.value
            yield f"id: {event.sequence}\nevent: {wire_type}\ndata: {json.dumps(payload)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get(
    "/api/sessions/{session_id}/trajectory",
    response_model=TrajectoryRead,
    tags=["inference"],
    summary="Replay complete trajectory",
)
async def trajectory(
    session_id: str,
    repository: Annotated[SQLiteSessionRepository, Depends(get_repository)],
) -> TrajectoryRead:
    try:
        return repository.get_trajectory(session_id)
    except SessionNotFoundError as error:
        raise HTTPException(status_code=404, detail="Session not found") from error
