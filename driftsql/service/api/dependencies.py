"""Typed FastAPI state dependencies."""

from fastapi import Request

from driftsql.service.catalog import ExperimentCatalog, ScenarioCatalog
from driftsql.service.inference import SessionOrchestrator
from driftsql.service.observability import OperationsService, WandbService
from driftsql.service.repository import SQLiteSessionRepository
from driftsql.service.settings import ServiceSettings


async def get_settings(request: Request) -> ServiceSettings:
    return request.app.state.settings


async def get_catalog(request: Request) -> ScenarioCatalog:
    return request.app.state.catalog


async def get_experiment_catalog(request: Request) -> ExperimentCatalog:
    return request.app.state.experiment_catalog


async def get_repository(request: Request) -> SQLiteSessionRepository:
    return request.app.state.repository


async def get_orchestrator(request: Request) -> SessionOrchestrator:
    return request.app.state.orchestrator


async def get_operations(request: Request) -> OperationsService:
    return request.app.state.operations


async def get_wandb(request: Request) -> WandbService:
    return request.app.state.wandb
