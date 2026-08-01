"""DriftSQL service application factory."""

from __future__ import annotations

import mimetypes
from hmac import compare_digest
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from driftsql.service.api import router
from driftsql.service.catalog import ExperimentCatalog, ScenarioCatalog
from driftsql.service.inference import ModelBackend, ScriptedModelBackend, SessionOrchestrator, ToolRuntime, VLLMBackend
from driftsql.service.observability import OperationsService, WandbService
from driftsql.service.repository import SQLiteSessionRepository
from driftsql.service.settings import ServiceSettings


def create_app(
    settings: ServiceSettings | None = None,
    *,
    backend: ModelBackend | None = None,
) -> FastAPI:
    resolved_settings = settings or ServiceSettings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        resolved_settings.ensure_directories()
        repository = SQLiteSessionRepository(resolved_settings.repository_path)
        repository.initialize()
        interrupted = repository.mark_interrupted_sessions_failed()
        catalog = ScenarioCatalog(resolved_settings.scenario_path)
        catalog.load()
        experiment_catalog = ExperimentCatalog(resolved_settings.frozen_candidate_path)
        experiment_catalog.load()
        model_backend = backend
        if model_backend is None:
            model_backend = (
                VLLMBackend(resolved_settings) if resolved_settings.model_backend == "vllm" else ScriptedModelBackend()
            )
        tools = ToolRuntime(
            resolved_settings.tool_config_path,
            resolved_settings.temporary_root,
            executor_max_rows=resolved_settings.executor_max_rows,
            schema_max_chars=resolved_settings.schema_max_chars,
            knowledge_max_results=resolved_settings.knowledge_max_results,
        )
        tools.load()
        await model_backend.start()
        orchestrator = SessionOrchestrator(
            resolved_settings,
            repository,
            catalog,
            model_backend,
            tools,
        )
        application.state.settings = resolved_settings
        application.state.repository = repository
        application.state.catalog = catalog
        application.state.experiment_catalog = experiment_catalog
        application.state.backend = model_backend
        application.state.tools = tools
        application.state.orchestrator = orchestrator
        application.state.operations = OperationsService(repository)
        application.state.wandb = WandbService(resolved_settings)
        application.state.interrupted_sessions = interrupted
        try:
            yield
        finally:
            await orchestrator.shutdown()
            await model_backend.shutdown()
            repository.close()

    application = FastAPI(
        title=resolved_settings.service_name,
        version=resolved_settings.service_version,
        description=(
            "Persistent Stage-8 SFT20 DriftSQL inference with isolated read-only SQLite "
            "sessions and replayable trajectory events."
        ),
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def authenticate_api(request: Request, call_next):
        if not resolved_settings.auth_enabled or not request.url.path.startswith("/api/"):
            return await call_next(request)
        expected = resolved_settings.api_key.get_secret_value() if resolved_settings.api_key else ""
        bearer = request.headers.get("authorization", "")
        supplied = (
            bearer.removeprefix("Bearer ").strip()
            if bearer.startswith("Bearer ")
            else request.headers.get("x-driftsql-api-key", "").strip()
        )
        if not supplied or not compare_digest(supplied, expected):
            return JSONResponse(
                {"detail": "Invalid or missing DriftSQL API key"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
        return await call_next(request)

    application.include_router(router)
    if resolved_settings.serve_frontend:
        assets = (resolved_settings.frontend_dist_path / "assets").resolve()
        if assets.is_dir():

            @application.get("/assets/{asset_path:path}", include_in_schema=False)
            async def studio_asset(asset_path: str) -> Response:
                target = (assets / asset_path).resolve()
                if assets not in target.parents or not target.is_file():
                    raise HTTPException(status_code=404, detail="Asset not found")
                media_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
                return Response(
                    content=target.read_bytes(),
                    media_type=media_type,
                    headers={"Cache-Control": "public, max-age=31536000, immutable"},
                )

        @application.get("/", include_in_schema=False)
        async def studio() -> Response:
            index = resolved_settings.frontend_dist_path / "index.html"
            if index.is_file():
                return HTMLResponse(
                    index.read_text(encoding="utf-8"),
                    headers={"Cache-Control": "no-cache"},
                )
            return HTMLResponse(
                "<h1>DriftSQL Studio is not built</h1>"
                "<p>Run <code>bash scripts/build_frontend.sh</code> and restart the service.</p>",
                status_code=503,
            )

    return application


app = create_app()
