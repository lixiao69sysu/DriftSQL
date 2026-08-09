"""Production tool registry reusing the exact VERL DriftSQL tools."""

from __future__ import annotations

import importlib
import io
import json
import os
import time
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Import the torch/VERL-backed tool module on the application's main thread.
# Starlette's TestClient starts lifespan in a portal thread, where first-time
# torch imports can deadlock on some CUDA/Python builds.
from driftsql.integrations import verl_tools as _verl_tools  # noqa: F401
from driftsql.integrations.state_policy import (
    duplicate_retrieval_response,
    dynamic_mask_response,
    is_exact_duplicate_retrieval,
    select_dynamic_tool_names,
)
from driftsql.integrations.verl_compat import OpenAIFunctionToolSchema


@dataclass(frozen=True)
class ToolExecution:
    observation: str
    reward: float
    metrics: dict[str, Any]
    elapsed_ms: float


class ToolRuntime:
    """One shared tool object per type and one isolated state key per session."""

    def __init__(
        self,
        config_path: Path,
        temporary_root: Path,
        *,
        executor_max_rows: int = 5,
        schema_max_chars: int = 3500,
        knowledge_max_results: int = 1,
    ) -> None:
        self.config_path = Path(config_path)
        self.temporary_root = Path(temporary_root)
        self._tools: dict[str, Any] = {}
        self._schemas: dict[str, dict[str, Any]] = {}
        self._session_tools: dict[str, list[str]] = {}
        self._session_modes: dict[str, str] = {}
        self._runtime_limits = {
            "execute_sql": ("max_rows", executor_max_rows),
            "get_schema": ("max_chars", schema_max_chars),
            "get_knowledge_definition": ("max_results", knowledge_max_results),
        }

    def load(self) -> None:
        payload = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        for entry in payload.get("tools", []):
            module_name, class_name = str(entry["class_name"]).rsplit(".", 1)
            tool_class = getattr(importlib.import_module(module_name), class_name)
            schema = OpenAIFunctionToolSchema.model_validate(entry["tool_schema"])
            name = schema.function.name
            if name in self._tools:
                raise ValueError(f"Duplicate tool name: {name}")
            config = dict(entry.get("config", {}))
            if name in self._runtime_limits:
                key, value = self._runtime_limits[name]
                config[key] = value
            # VERL's BaseTool prints every schema during construction. Keep
            # service startup logs structured and expose schemas via OpenAPI.
            with redirect_stdout(io.StringIO()):
                self._tools[name] = tool_class(config, schema)
            self._schemas[name] = schema.model_dump(mode="json")

    async def initialize_session(
        self,
        session_id: str,
        tool_names: list[str],
        create_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        unknown = set(tool_names) - set(self._tools)
        if unknown:
            raise ValueError(f"Unknown configured tools: {sorted(unknown)}")
        os.environ.setdefault("DRIFTSQL_TMPDIR", str(self.temporary_root))
        self.temporary_root.mkdir(parents=True, exist_ok=True)
        initialized: set[str] = set()
        try:
            # Initialize the executor first so sandbox creation is part of
            # session creation, not delayed until the first SQL call.
            ordered = sorted(tool_names, key=lambda name: name != "execute_sql")
            for name in ordered:
                await self._tools[name].create(
                    instance_id=session_id,
                    ground_truth=str(create_kwargs.get("ground_truth", "")),
                    create_kwargs=create_kwargs,
                )
                initialized.add(name)
        except Exception:
            for name in initialized:
                await self._tools[name].release(session_id)
            raise
        self._session_tools[session_id] = list(tool_names)
        self._session_modes[session_id] = str(create_kwargs.get("verification_mode", "result_contract"))
        executor = self._tools.get("execute_sql")
        state = executor._state(session_id) if executor and "execute_sql" in initialized else {}
        return {
            "sandbox_ref": str(state.get("db_path", "")),
            "sandbox_isolated": bool(state.get("session_isolated", False)),
        }

    def schemas_for(self, session_id: str, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        configured = self._session_tools[session_id]
        allowed = select_dynamic_tool_names(
            events,
            configured,
            execution_only=self._session_modes.get(session_id) == "execution_only",
        )
        return [self._schemas[name] for name in configured if name in allowed]

    async def execute(
        self,
        session_id: str,
        name: str,
        arguments: dict[str, Any],
        events: list[dict[str, Any]],
    ) -> ToolExecution:
        configured = set(self._session_tools[session_id])
        available = select_dynamic_tool_names(
            events,
            configured,
            execution_only=self._session_modes.get(session_id) == "execution_only",
        )
        started = time.perf_counter()
        if name not in configured or name not in available:
            observation, metrics = dynamic_mask_response(name, available)
            if name == "execute_sql":
                payload = json.loads(observation)
                payload.update(
                    {
                        "success": False,
                        "error": (
                            "Readonly state policy rejected execute_sql: "
                            + str(payload.get("message", "tool unavailable"))
                        ),
                        "rolled_back": True,
                    }
                )
                observation = json.dumps(payload, ensure_ascii=False)
            return ToolExecution(observation, 0.0, metrics, (time.perf_counter() - started) * 1000)
        if is_exact_duplicate_retrieval(events, name, arguments):
            observation, metrics = duplicate_retrieval_response(name)
            return ToolExecution(observation, 0.0, metrics, (time.perf_counter() - started) * 1000)
        if name == "submit_solution" and self._session_modes.get(session_id) == "execution_only":
            executions = [event for event in events if event.get("tool") == "execute_sql"]
            last = executions[-1] if executions else {}
            executed = str(last.get("arguments", {}).get("sql", "")).strip().rstrip(";").casefold()
            submitted = str(arguments.get("sql", "")).strip().rstrip(";").casefold()
            if not executed or executed != submitted or not bool(last.get("metrics", {}).get("execution_success")):
                payload = {
                    "action_masked": True,
                    "tool": "submit_solution",
                    "message": "Free-query mode may submit only the exact SQL from the latest successful execution.",
                }
                return ToolExecution(
                    json.dumps(payload, ensure_ascii=False),
                    0.0,
                    {"action_masked": True, "state_guarded": True, "submitted": False},
                    (time.perf_counter() - started) * 1000,
                )
        response, reward, metrics = await self._tools[name].execute(
            instance_id=session_id,
            parameters=arguments,
        )
        response_text = response.text
        metrics = dict(metrics or {})
        if name == "execute_sql" and self._session_modes.get(session_id) == "execution_only":
            try:
                payload = json.loads(response_text)
            except (json.JSONDecodeError, TypeError):
                payload = {}
            execution_success = bool(payload.get("success"))
            payload.update(
                {
                    "validated_for_submit": execution_success,
                    "semantic_verified": False,
                    "verification_scope": "execution_only",
                }
            )
            response_text = json.dumps(payload, ensure_ascii=False, default=str)
            metrics.update(
                {
                    "validated_for_submit": execution_success,
                    "semantic_verified": False,
                    "verification_scope": "execution_only",
                }
            )
        return ToolExecution(
            response_text,
            float(reward),
            metrics,
            (time.perf_counter() - started) * 1000,
        )

    async def release_session(self, session_id: str) -> None:
        names = self._session_tools.pop(session_id, [])
        self._session_modes.pop(session_id, None)
        for name in names:
            await self._tools[name].release(session_id)

    async def shutdown(self) -> None:
        for session_id in list(self._session_tools):
            await self.release_session(session_id)

    def schema_names(self) -> set[str]:
        return set(self._schemas)
