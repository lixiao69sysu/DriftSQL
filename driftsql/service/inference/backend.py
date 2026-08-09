"""Persistent model backends for production and deterministic service tests."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from driftsql.service.schemas import ModelMetadata
from driftsql.service.settings import ServiceSettings


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _adapter_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise FileNotFoundError(f"Adapter directory is empty: {path}")
    for item in files:
        digest.update(str(item.relative_to(path)).encode())
        digest.update(bytes.fromhex(_sha256(item)))
    return digest.hexdigest()


@dataclass(frozen=True)
class GenerationRequest:
    session_id: str
    scenario_id: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    create_kwargs: dict[str, Any]
    tool_events: list[dict[str, Any]]
    max_new_tokens: int


@dataclass(frozen=True)
class GenerationResult:
    text: str
    prompt_tokens: int
    response_tokens: int
    elapsed_ms: float


class ModelBackend(ABC):
    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def shutdown(self) -> None: ...

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResult: ...

    @abstractmethod
    async def abort(self, session_id: str) -> None: ...

    @abstractmethod
    async def activate_model(
        self,
        *,
        model_id: str,
        base_model_path: Path,
        adapter_path: Path | None,
        adapter_sha256: str,
    ) -> None: ...

    @property
    @abstractmethod
    def metadata(self) -> ModelMetadata: ...


class VLLMBackend(ModelBackend):
    """One process-lifetime vLLM engine with one pinned frozen LoRA adapter."""

    def __init__(self, settings: ServiceSettings) -> None:
        self.settings = settings
        self._engine: Any = None
        self._tokenizer: Any = None
        self._lora_request: Any = None
        self._request_ids: dict[str, set[str]] = {}
        self._metadata = ModelMetadata(
            model_id=settings.default_model_id,
            backend="vllm",
            base_model=str(settings.base_model_path),
            adapter=str(settings.adapter_path),
            adapter_sha256="",
            frozen_candidate_sha256="",
            loaded=False,
        )

    def _verify_initial_model(self) -> None:
        settings = self.settings
        if not settings.base_model_path.is_dir():
            raise FileNotFoundError(settings.base_model_path)
        if not settings.adapter_path.is_dir():
            raise FileNotFoundError(settings.adapter_path)
        self._metadata = self._metadata.model_copy(
            update={
                "adapter_sha256": _adapter_sha256(settings.adapter_path),
                "frozen_candidate_sha256": "",
            }
        )

    async def start(self) -> None:
        if self._engine is not None:
            return
        self._verify_initial_model()
        # The service imports the VERL/torch tool stack before lifespan. vLLM
        # tensor-parallel workers must therefore use spawn; CUDA cannot be
        # re-initialized safely in a forked child.
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        from transformers import AutoTokenizer
        from vllm import AsyncEngineArgs, AsyncLLMEngine
        from vllm.lora.request import LoRARequest

        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.settings.base_model_path),
            local_files_only=True,
            trust_remote_code=True,
        )
        engine_args = AsyncEngineArgs(
            model=str(self.settings.base_model_path),
            dtype="bfloat16",
            tensor_parallel_size=self.settings.tensor_parallel_size,
            gpu_memory_utilization=self.settings.gpu_memory_utilization,
            max_model_len=self.settings.max_model_len,
            enable_lora=True,
            max_lora_rank=self.settings.max_lora_rank,
            max_loras=1,
            enforce_eager=True,
            enable_prefix_caching=False,
            async_scheduling=False,
            generation_config="vllm",
            trust_remote_code=True,
        )
        self._engine = AsyncLLMEngine.from_engine_args(engine_args)
        self._lora_request = LoRARequest(self.settings.default_model_id, 1, str(self.settings.adapter_path))
        await self._engine.add_lora(self._lora_request)
        await self._engine.pin_lora(1)
        self._metadata = self._metadata.model_copy(update={"loaded": True})

    async def shutdown(self) -> None:
        if self._engine is not None:
            self._engine.shutdown()
        self._engine = None
        self._tokenizer = None
        self._metadata = self._metadata.model_copy(update={"loaded": False})

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        if self._engine is None or self._tokenizer is None:
            raise RuntimeError("vLLM backend has not been started")
        from vllm import SamplingParams

        prompt = self._tokenizer.apply_chat_template(
            request.messages,
            tools=request.tools,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_tokens = len(self._tokenizer.encode(prompt))
        request_id = f"{request.session_id}-{uuid4()}"
        self._request_ids.setdefault(request.session_id, set()).add(request_id)
        started = time.perf_counter()
        final = None
        try:
            stream = self._engine.generate(
                prompt,
                SamplingParams(
                    temperature=0.0,
                    max_tokens=request.max_new_tokens,
                    seed=42,
                ),
                request_id,
                lora_request=self._lora_request,
            )
            async for output in stream:
                final = output
        finally:
            self._request_ids.get(request.session_id, set()).discard(request_id)
        if final is None:
            raise RuntimeError("vLLM returned no generation")
        text = final.outputs[0].text
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            response_tokens=len(final.outputs[0].token_ids),
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    async def abort(self, session_id: str) -> None:
        if self._engine is None:
            return
        await asyncio.gather(
            *(self._engine.abort(request_id) for request_id in list(self._request_ids.get(session_id, set()))),
            return_exceptions=True,
        )

    async def activate_model(
        self,
        *,
        model_id: str,
        base_model_path: Path,
        adapter_path: Path | None,
        adapter_sha256: str,
    ) -> None:
        from vllm.lora.request import LoRARequest

        if self._engine is None:
            raise RuntimeError("vLLM backend has not been started")
        if base_model_path.resolve() != self.settings.base_model_path.resolve():
            raise ValueError("Hot activation only supports adapters for the loaded base model")
        old_request = self._lora_request
        old_metadata = self._metadata
        if old_request is not None:
            await self._engine.remove_lora(old_request.lora_int_id)
        try:
            if adapter_path is None:
                next_request = None
            else:
                next_request = LoRARequest(model_id, 1, str(adapter_path))
                await self._engine.add_lora(next_request)
                await self._engine.pin_lora(1)
            self._lora_request = next_request
            self._metadata = self._metadata.model_copy(
                update={
                    "model_id": model_id,
                    "base_model": str(base_model_path),
                    "adapter": str(adapter_path) if adapter_path else "",
                    "adapter_sha256": adapter_sha256,
                    "frozen_candidate_sha256": "",
                }
            )
        except Exception:
            if old_request is not None:
                await self._engine.add_lora(old_request)
                await self._engine.pin_lora(old_request.lora_int_id)
            self._lora_request = old_request
            self._metadata = old_metadata
            raise

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata


class ScriptedModelBackend(ModelBackend):
    """Deterministic backend that exercises the real tools, sandbox and API."""

    def __init__(
        self,
        *,
        scripts: dict[str, list[dict[str, Any]]] | None = None,
        delay_seconds: float = 0.0,
    ) -> None:
        self.scripts = scripts or {}
        self.delay_seconds = delay_seconds
        self._cancelled: set[str] = set()
        self.active_generations = 0
        self.max_active_generations = 0
        self._metadata = ModelMetadata(
            model_id="stage8-sft20-legacy",
            backend="scripted",
            base_model="Qwen2.5-Coder-7B-Instruct-test-double",
            adapter="stage8-sft20-test-double",
            adapter_sha256="scripted-adapter-sha256",
            frozen_candidate_sha256="scripted-manifest-sha256",
            loaded=False,
        )

    async def start(self) -> None:
        self._metadata = self._metadata.model_copy(update={"loaded": True})

    async def shutdown(self) -> None:
        self._metadata = self._metadata.model_copy(update={"loaded": False})

    def _default_call(self, request: GenerationRequest) -> dict[str, Any]:
        index = len(request.tool_events)
        ground_truth = str(request.create_kwargs.get("ground_truth", ""))
        if request.create_kwargs.get("verification_mode") == "execution_only":
            ground_truth = "SELECT 1 AS result"
            query_calls = [
                {"name": "get_schema", "arguments": {"query": ""}},
                {"name": "execute_sql", "arguments": {"sql": ground_truth}},
                {"name": "submit_solution", "arguments": {"sql": ground_truth}},
            ]
            return query_calls[min(index, len(query_calls) - 1)]
        calls = [
            {"name": "get_schema_version", "arguments": {}},
            {"name": "inspect_schema_diff", "arguments": {}},
            {"name": "execute_sql", "arguments": {"sql": ground_truth}},
            {"name": "submit_solution", "arguments": {"sql": ground_truth}},
        ]
        return calls[min(index, len(calls) - 1)]

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.active_generations += 1
        self.max_active_generations = max(self.max_active_generations, self.active_generations)
        started = time.perf_counter()
        try:
            if self.delay_seconds:
                await asyncio.sleep(self.delay_seconds)
            if request.session_id in self._cancelled:
                raise asyncio.CancelledError
            script = self.scripts.get(request.scenario_id)
            if script:
                call = script[min(len(request.tool_events), len(script) - 1)]
            else:
                call = self._default_call(request)
            text = f"<think>deterministic service validation</think>\n{json.dumps(call)}"
            return GenerationResult(
                text=text,
                prompt_tokens=max(1, sum(len(str(message.get("content", ""))) for message in request.messages) // 4),
                response_tokens=max(1, len(text) // 4),
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )
        finally:
            self.active_generations -= 1

    async def abort(self, session_id: str) -> None:
        self._cancelled.add(session_id)

    async def activate_model(
        self,
        *,
        model_id: str,
        base_model_path: Path,
        adapter_path: Path | None,
        adapter_sha256: str,
    ) -> None:
        self._metadata = self._metadata.model_copy(
            update={
                "model_id": model_id,
                "base_model": str(base_model_path),
                "adapter": str(adapter_path) if adapter_path else "",
                "adapter_sha256": adapter_sha256,
                "frozen_candidate_sha256": "",
            }
        )

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata
