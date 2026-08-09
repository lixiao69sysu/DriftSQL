"""DriftSQL service application factory."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from hmac import compare_digest
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from driftsql.drift.factory import fingerprint_query, materialize_schema_diff
from driftsql.service.api import router
from driftsql.service.api.routes import AUTH_COOKIE
from driftsql.service.auth import AuthSessionStore
from driftsql.service.catalog import ExperimentCatalog, ScenarioCatalog
from driftsql.service.inference import ModelBackend, ScriptedModelBackend, SessionOrchestrator, ToolRuntime, VLLMBackend
from driftsql.service.model_catalog import ModelCatalog
from driftsql.service.observability import OperationsService, WandbService
from driftsql.service.replay import ReplayReviewStore
from driftsql.service.repository import SQLiteSessionRepository
from driftsql.service.settings import PROJECT_ROOT, ServiceSettings
from driftsql.service.translation import (
    PassthroughTranslationService,
    QwenChineseEnglishTranslator,
    TranslationService,
)


def _ensure_test_scenario_catalog(settings: ServiceSettings) -> Path:
    default_path = Path(ServiceSettings.model_fields["scenario_path"].default)
    scenario_path = settings.scenario_path
    if settings.environment != "test" or scenario_path.is_file():
        return scenario_path
    if scenario_path.resolve() != default_path.resolve():
        return scenario_path

    fixture_root = settings.temporary_root / "__service_test_fixture__"
    fixture_root.mkdir(parents=True, exist_ok=True)
    source_db = fixture_root / "book_publishing_company.sqlite"
    if not source_db.is_file():
        with sqlite3.connect(source_db) as connection:
            connection.executescript(
                """
                CREATE TABLE authors (
                    au_id TEXT PRIMARY KEY,
                    name TEXT
                );
                INSERT INTO authors (au_id, name) VALUES ('A-1', 'Ada');

                CREATE TABLE TeamsSC (
                    year INTEGER,
                    tmID TEXT,
                    W INTEGER
                );
                INSERT INTO TeamsSC (year, tmID, W) VALUES (2024, 'ATL', 88);
                """
            )
    schema_diff = {
        "operations": [{"type": "add_column", "table": "authors", "new_name": "country", "declared_type": "TEXT"}]
    }
    ground_truth = "SELECT COUNT(*) AS team_count FROM TeamsSC"
    projected = fixture_root / "book_publishing_company_v2.sqlite"
    materialize_schema_diff(source_db, projected, schema_diff)
    fingerprint = fingerprint_query(projected, ground_truth)
    try:
        projected.unlink()
    except FileNotFoundError:
        pass

    generated = fixture_root / "tune_agent_eval.jsonl"
    create_kwargs = {
        "db_id": "book_publishing_company",
        "db_version": "v2",
        "source_db": str(source_db),
        "schema_diff": schema_diff,
        "query": "Count teams in the current season.",
        "stale_sql": "SELECT * FROM TeamsSC LIMIT 1",
        "ground_truth": ground_truth,
        "result_fingerprint": {
            "row_count": fingerprint.row_count,
            "value_hash": fingerprint.value_hash,
        },
    }
    record = {
        "prompt": [
            {"role": "system", "content": "You are a read-only SQLite recovery agent."},
            {
                "role": "user",
                "content": (
                    "## Analytics request\nCount teams in the current season.\n\n"
                    "## Previously valid cached SQL\nSELECT * FROM TeamsSC LIMIT 1"
                ),
            },
        ],
        "extra_info": {
            "instance_id": "service-test-add-column",
            "db_id": "book_publishing_company",
            "source_db": str(source_db),
            "stale_sql": create_kwargs["stale_sql"],
            "drift_type": "add_column",
            "wildcard_profile": "single_table",
            "schema_diff": schema_diff,
            "result_fingerprint": create_kwargs["result_fingerprint"],
            "tool_selection": [
                "get_schema",
                "get_schema_version",
                "inspect_schema_diff",
                "execute_sql",
                "submit_solution",
            ],
            "tools_kwargs": {
                "execute_sql": {
                    "create_kwargs": create_kwargs,
                }
            },
        },
    }
    generated.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    settings.scenario_path = generated
    return generated


def create_app(
    settings: ServiceSettings | None = None,
    *,
    backend: ModelBackend | None = None,
    translator: TranslationService | None = None,
) -> FastAPI:
    resolved_settings = settings or ServiceSettings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        resolved_settings.ensure_directories()
        repository = SQLiteSessionRepository(resolved_settings.repository_path)
        repository.initialize()
        interrupted = repository.mark_interrupted_sessions_failed()
        catalog = ScenarioCatalog(_ensure_test_scenario_catalog(resolved_settings))
        catalog.load()
        experiment_catalog = ExperimentCatalog(resolved_settings.experiment_catalog_path)
        experiment_catalog.load()
        model_catalog = ModelCatalog(resolved_settings.model_registry_path, PROJECT_ROOT)
        model_catalog.load()
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
        translation_service = translator
        if translation_service is None:
            translation_service = (
                QwenChineseEnglishTranslator(
                    resolved_settings.translation_model_path,
                    max_input_tokens=resolved_settings.translation_max_input_tokens,
                    max_new_tokens=resolved_settings.translation_max_new_tokens,
                )
                if resolved_settings.translation_enabled
                else PassthroughTranslationService()
            )
        await model_backend.start()
        orchestrator = SessionOrchestrator(
            resolved_settings,
            repository,
            catalog,
            model_backend,
            tools,
            translation_service,
        )
        application.state.settings = resolved_settings
        application.state.repository = repository
        application.state.catalog = catalog
        application.state.experiment_catalog = experiment_catalog
        application.state.model_catalog = model_catalog
        application.state.backend = model_backend
        application.state.tools = tools
        application.state.translator = translation_service
        application.state.orchestrator = orchestrator
        application.state.operations = OperationsService(repository)
        application.state.wandb = WandbService(resolved_settings)
        application.state.replay_reviews = ReplayReviewStore(resolved_settings.replay_review_dir)
        application.state.auth_sessions = AuthSessionStore(resolved_settings.auth_session_ttl_seconds)
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
            "Persistent DriftSQL inference with isolated read-only SQLite "
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
        cookie_authenticated = request.app.state.auth_sessions.validate(request.cookies.get(AUTH_COOKIE))
        header_authenticated = bool(supplied) and compare_digest(supplied, expected)
        if not cookie_authenticated and not header_authenticated:
            return JSONResponse(
                {"detail": "Invalid or missing DriftSQL API key"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
        return await call_next(request)

    application.include_router(router)

    return application


app = create_app()
