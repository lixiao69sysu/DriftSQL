"""Validated model registry exposed to the CLI and inference service."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from driftsql.service.schemas import ModelList, ModelMetadata, ModelRead


@lru_cache(maxsize=32)
def _directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        digest.update(str(item.relative_to(path)).encode())
        file_digest = hashlib.sha256()
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                file_digest.update(chunk)
        digest.update(file_digest.digest())
    return digest.hexdigest() if files else ""


@dataclass(frozen=True)
class RuntimeModelSpec:
    model_id: str
    display_name: str
    category: str
    base_model: Path
    adapter: Path | None
    adapter_sha256: str
    metrics: dict[str, Any]
    notes: str

    @property
    def available(self) -> bool:
        return self.base_model.is_dir() and (self.adapter is None or self.adapter.is_dir())


class ModelNotFoundError(KeyError):
    pass


class ModelUnavailableError(RuntimeError):
    pass


class ModelCatalog:
    def __init__(self, path: Path, project_root: Path) -> None:
        self.path = Path(path)
        self.project_root = Path(project_root)
        self._models: dict[str, RuntimeModelSpec] = {}

    def _resolve(self, value: str | None) -> Path | None:
        if not value:
            return None
        path = Path(value).expanduser()
        return (self.project_root / path).resolve() if not path.is_absolute() else path.resolve()

    def load(self) -> None:
        payload = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        models: dict[str, RuntimeModelSpec] = {}
        for raw in payload.get("models", []):
            model_id = str(raw.get("model_id", "")).strip()
            if not model_id or model_id in models:
                raise ValueError(f"Invalid or duplicate model_id in {self.path}: {model_id!r}")
            base_model = self._resolve(str(raw.get("base_model", "")))
            if base_model is None:
                raise ValueError(f"Model {model_id} has no base_model")
            adapter = self._resolve(raw.get("adapter"))
            models[model_id] = RuntimeModelSpec(
                model_id=model_id,
                display_name=str(raw.get("display_name", model_id)),
                category=str(raw.get("category", "unknown")),
                base_model=base_model,
                adapter=adapter,
                adapter_sha256=_directory_sha256(adapter) if adapter and adapter.is_dir() else "",
                metrics=dict(raw.get("metrics", {})),
                notes=str(raw.get("notes", "")),
            )
        if not models:
            raise ValueError(f"No models found in {self.path}")
        self._models = models

    def get(self, model_id: str, *, allow_unavailable: bool = False) -> RuntimeModelSpec:
        try:
            model = self._models[model_id]
        except KeyError as error:
            raise ModelNotFoundError(model_id) from error
        if not allow_unavailable and not model.available:
            raise ModelUnavailableError(f"Model files are unavailable: {model_id}")
        return model

    def identify(self, metadata: ModelMetadata) -> str | None:
        adapter = Path(metadata.adapter).resolve() if metadata.adapter else None
        base = Path(metadata.base_model).resolve() if metadata.base_model else None
        for model in self._models.values():
            if model.base_model.resolve() != base:
                continue
            if (model.adapter.resolve() if model.adapter else None) == adapter:
                return model.model_id
        return metadata.model_id

    def list_models(self, metadata: ModelMetadata) -> ModelList:
        active_id = self.identify(metadata)
        return ModelList(
            active_model_id=active_id,
            models=[
                ModelRead(
                    model_id=model.model_id,
                    display_name=model.display_name,
                    category=model.category,
                    base_model=str(model.base_model),
                    adapter=str(model.adapter) if model.adapter else None,
                    adapter_sha256=model.adapter_sha256,
                    available=model.available,
                    active=model.model_id == active_id,
                    metrics=model.metrics,
                    notes=model.notes,
                )
                for model in self._models.values()
            ],
        )
