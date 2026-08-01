"""VERL-native tools for versioned DriftSQL trajectories.

This module is imported only inside the full training environment, where the
pinned VERL source is installed or present on ``PYTHONPATH``.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

from driftsql.drift import materialize_schema_diff
from driftsql.integrations.state_policy import schema_diff_recovery_guidance
from driftsql.planning import plan_projection_contract


class _TrajectoryStateTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instances: dict[str, dict[str, Any]] = {}

    async def create(
        self,
        instance_id: str | None = None,
        ground_truth: str | None = None,
        **kwargs: Any,
    ) -> tuple[str, ToolResponse]:
        instance_id = instance_id or str(uuid4())
        if instance_id in self._instances:
            return instance_id, ToolResponse(text="DriftSQL trajectory state reused.")
        create_kwargs = dict(kwargs.get("create_kwargs", {}))
        if not create_kwargs.get("db_id"):
            raise ValueError("db_id is required in create_kwargs")
        self._instances[instance_id] = create_kwargs
        return instance_id, ToolResponse(text="DriftSQL trajectory state initialized.")

    async def calc_reward(self, instance_id: str, **kwargs: Any) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs: Any) -> None:
        self._instances.pop(instance_id, None)

    def _state(self, instance_id: str) -> dict[str, Any]:
        if instance_id not in self._instances:
            raise KeyError(f"Unknown tool instance: {instance_id}")
        return self._instances[instance_id]


class GetSchemaVersionTool(_TrajectoryStateTool):
    @rollout_trace_op
    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs: Any,
    ) -> tuple[ToolResponse, float, dict]:
        state = self._state(instance_id)
        observation = {
            "db_id": state["db_id"],
            "db_version": state.get("db_version"),
            "metric_version": state.get("metric_version"),
        }
        return ToolResponse(text=json.dumps(observation, ensure_ascii=False)), 0.0, {
            "schema_version_checked": True
        }


def _ordered_sqlite_schema(database: Path) -> dict[str, list[str]]:
    schema: dict[str, list[str]] = {}
    uri = database.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        for (raw_table,) in tables:
            table = str(raw_table)
            escaped = table.replace('"', '""')
            schema[table] = [
                str(row[1])
                for row in connection.execute(f'PRAGMA table_info("{escaped}")')
            ]
    return schema


def _active_schema_for_projection(
    source_db: Path,
    schema_diff: dict[str, Any],
) -> dict[str, list[str]]:
    """Reconstruct active ordered columns from source schema plus audited adds."""
    schema = _ordered_sqlite_schema(source_db)
    table_lookup = {table.casefold(): table for table in schema}
    for operation in schema_diff.get("operations", []) or []:
        if not isinstance(operation, dict) or operation.get("type") != "add_column":
            continue
        table = table_lookup.get(str(operation.get("table", "")).casefold())
        name = str(operation.get("new_name", "")).strip()
        if not table or not name:
            raise ValueError(f"Invalid add-column operation: {operation}")
        if name.casefold() not in {column.casefold() for column in schema[table]}:
            schema[table].append(name)
    return schema


class InspectSchemaDiffTool(_TrajectoryStateTool):
    @rollout_trace_op
    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs: Any,
    ) -> tuple[ToolResponse, float, dict]:
        state = self._state(instance_id)
        observation = dict(state.get("schema_diff", {"operations": []}))
        guidance = schema_diff_recovery_guidance(observation)
        if guidance:
            observation["recovery_guidance"] = guidance
        projection_planned = False
        projection_error = None
        operations = observation.get("operations", []) or []
        if any(
            isinstance(operation, dict) and operation.get("type") == "add_column"
            for operation in operations
        ):
            try:
                source_db = Path(str(state.get("source_db", "")))
                if not source_db.is_file():
                    raise FileNotFoundError("source_db is unavailable for projection planning")
                active_schema = _active_schema_for_projection(source_db, observation)
                plan = plan_projection_contract(
                    str(state.get("stale_sql", "")),
                    observation,
                    active_schema,
                )
                observation["projection_contract_plan"] = plan.to_dict()
                projection_planned = True
            except (FileNotFoundError, ValueError) as error:
                # Non-wildcard add-column cases still receive the audited diff
                # and normal recovery guidance; planner failure is observable.
                projection_error = str(error)
        return ToolResponse(text=json.dumps(observation, ensure_ascii=False)), 0.0, {
            "schema_diff_inspected": True,
            "recovery_guidance_count": len(guidance),
            "projection_contract_planned": projection_planned,
            "projection_contract_error": projection_error,
        }


def _terms(text: str) -> set[str]:
    return {term.casefold() for term in re.findall(r"[A-Za-z0-9_]+", text) if len(term) > 1}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class AskUserTool(_TrajectoryStateTool):
    """Deterministic, guarded Mini-Interact user simulator.

    The simulator answers only ambiguity points published with the current
    task. It never invents an answer or reveals unrelated ambiguity points.
    """

    async def create(self, *args: Any, **kwargs: Any) -> tuple[str, ToolResponse]:
        instance_id, response = await super().create(*args, **kwargs)
        state = self._state(instance_id)
        state.setdefault("_asked_terms", [])
        state.setdefault("_question_count", 0)
        return instance_id, response

    @rollout_trace_op
    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs: Any,
    ) -> tuple[ToolResponse, float, dict]:
        state = self._state(instance_id)
        question = str(parameters.get("question", "")).strip()
        if not question:
            return ToolResponse(text="Please ask one concrete clarification question."), 0.0, {
                "clarification_matched": False,
                "invalid_question": True,
            }

        maximum = int(self.config.get("max_questions", 3))
        if int(state["_question_count"]) >= maximum:
            return ToolResponse(text="The user has no remaining clarification turns."), 0.0, {
                "clarification_matched": False,
                "question_budget_exhausted": True,
            }
        state["_question_count"] = int(state["_question_count"]) + 1

        ambiguity = state.get("user_query_ambiguity", {}) or {}
        candidates: list[dict[str, Any]] = []
        for priority, key in enumerate(("critical_ambiguity", "non_critical_ambiguity")):
            for item in ambiguity.get(key, []) or []:
                candidate = dict(item)
                candidate["_priority"] = priority
                candidates.append(candidate)

        question_terms = _terms(question)
        ranked: list[tuple[int, int, dict[str, Any]]] = []
        for item in candidates:
            term = str(item.get("term", "")).strip()
            overlap = len(question_terms & _terms(term))
            phrase_match = bool(term and term.casefold() in question.casefold())
            if overlap or phrase_match:
                ranked.append((1 if phrase_match else 0, overlap, item))
        if not ranked:
            return ToolResponse(
                text="I cannot answer that from the stated business requirements. Please ask about one ambiguous term in the request."
            ), 0.0, {"clarification_matched": False, "unanswerable_question": True}

        ranked.sort(key=lambda entry: (entry[0], entry[1], -entry[2]["_priority"]), reverse=True)
        selected = ranked[0][2]
        term = str(selected.get("term", "")).strip()
        if term in state["_asked_terms"]:
            return ToolResponse(text=f"I already clarified '{term}'. Please use that definition."), 0.0, {
                "clarification_matched": True,
                "duplicate_question": True,
                "term": term,
            }
        state["_asked_terms"].append(term)
        definition = str(selected.get("sql_snippet", "")).strip()
        answer = f"For '{term}', the intended business definition is: {definition}"
        return ToolResponse(text=answer), 0.0, {
            "clarification_matched": True,
            "duplicate_question": False,
            "term": term,
            "ambiguity_type": selected.get("type"),
        }


class GetSchemaTool(_TrajectoryStateTool):
    """Retrieve the active schema, optionally ranked by a search query."""

    @rollout_trace_op
    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs: Any,
    ) -> tuple[ToolResponse, float, dict]:
        state = self._state(instance_id)
        schema = str(state.get("schema", ""))
        if not schema:
            schema_path = Path(str(state.get("schema_path", "")))
            if schema_path.is_file():
                schema = schema_path.read_text(encoding="utf-8", errors="replace")
        if not schema:
            return ToolResponse(text="No schema asset is available for this trajectory."), 0.0, {
                "schema_retrieved": False
            }

        query = str(parameters.get("query", "")).strip()
        blocks = [block.strip() for block in re.split(r"\n\s*\n(?=CREATE TABLE)", schema) if block.strip()]
        if query:
            query_terms = _terms(query)
            ranked = sorted(
                blocks,
                key=lambda block: len(query_terms & _terms(block)),
                reverse=True,
            )
            positive = [block for block in ranked if query_terms & _terms(block)]
            blocks = positive or ranked
        maximum = int(self.config.get("max_chars", 12000))
        selected = "\n\n".join(blocks)
        truncated = len(selected) > maximum
        selected = selected[:maximum]
        payload = {"query": query, "schema": selected, "truncated": truncated}
        return ToolResponse(text=json.dumps(payload, ensure_ascii=False)), 0.0, {
            "schema_retrieved": True,
            "schema_query_used": bool(query),
            "schema_truncated": truncated,
        }


class GetKnowledgeDefinitionTool(_TrajectoryStateTool):
    """Search the per-database hierarchical/business knowledge base."""

    async def create(self, *args: Any, **kwargs: Any) -> tuple[str, ToolResponse]:
        instance_id, response = await super().create(*args, **kwargs)
        state = self._state(instance_id)
        if "_knowledge_entries" not in state:
            inline = state.get("knowledge_entries")
            if isinstance(inline, list):
                state["_knowledge_entries"] = [dict(entry) for entry in inline]
            else:
                path = Path(str(state.get("knowledge_base_path", "")))
                state["_knowledge_entries"] = _load_jsonl(path)
        return instance_id, response

    @rollout_trace_op
    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs: Any,
    ) -> tuple[ToolResponse, float, dict]:
        state = self._state(instance_id)
        query = str(parameters.get("name", parameters.get("query", ""))).strip()
        if not query:
            return ToolResponse(text="A knowledge name or search query is required."), 0.0, {
                "knowledge_retrieved": False,
                "invalid_query": True,
            }
        query_terms = _terms(query)
        entries = list(state.get("_knowledge_entries", []))
        ranked: list[tuple[int, int, dict[str, Any]]] = []
        for entry in entries:
            name = str(entry.get("knowledge", ""))
            searchable = " ".join(
                str(entry.get(key, "")) for key in ("knowledge", "description", "definition", "type")
            )
            phrase_match = int(query.casefold() in name.casefold() or name.casefold() in query.casefold())
            overlap = len(query_terms & _terms(searchable))
            if phrase_match or overlap:
                ranked.append((phrase_match, overlap, entry))
        ranked.sort(key=lambda value: (value[0], value[1]), reverse=True)
        limit = int(self.config.get("max_results", 3))
        matches = [entry for _, _, entry in ranked[:limit]]
        return ToolResponse(text=json.dumps({"query": query, "matches": matches}, ensure_ascii=False)), 0.0, {
            "knowledge_retrieved": bool(matches),
            "knowledge_matches": len(matches),
        }


def _backup_sqlite(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True) as source_connection:
        with sqlite3.connect(target) as target_connection:
            source_connection.backup(target_connection)


def _execute_read_only(db_path: Path, sql: str, timeout_seconds: float, max_rows: int) -> dict[str, Any]:
    started = time.monotonic()
    if not sql.strip():
        return {
            "success": False,
            "error": "No SQL provided",
            "columns": [],
            "rows": [],
            "truncated": False,
            "rolled_back": False,
            "elapsed_ms": 0.0,
        }

    resolved = db_path.resolve()
    if not resolved.is_file():
        return {
            "success": False,
            "error": f"Database not found: {resolved}",
            "columns": [],
            "rows": [],
            "truncated": False,
            "rolled_back": False,
            "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
        }

    deadline = time.monotonic() + timeout_seconds
    denied_actions = {
        sqlite3.SQLITE_INSERT,
        sqlite3.SQLITE_UPDATE,
        sqlite3.SQLITE_DELETE,
        sqlite3.SQLITE_CREATE_INDEX,
        sqlite3.SQLITE_CREATE_TABLE,
        sqlite3.SQLITE_CREATE_TEMP_INDEX,
        sqlite3.SQLITE_CREATE_TEMP_TABLE,
        sqlite3.SQLITE_CREATE_TEMP_TRIGGER,
        sqlite3.SQLITE_CREATE_TEMP_VIEW,
        sqlite3.SQLITE_CREATE_TRIGGER,
        sqlite3.SQLITE_CREATE_VIEW,
        sqlite3.SQLITE_CREATE_VTABLE,
        sqlite3.SQLITE_DROP_INDEX,
        sqlite3.SQLITE_DROP_TABLE,
        sqlite3.SQLITE_DROP_TEMP_INDEX,
        sqlite3.SQLITE_DROP_TEMP_TABLE,
        sqlite3.SQLITE_DROP_TEMP_TRIGGER,
        sqlite3.SQLITE_DROP_TEMP_VIEW,
        sqlite3.SQLITE_DROP_TRIGGER,
        sqlite3.SQLITE_DROP_VIEW,
        sqlite3.SQLITE_DROP_VTABLE,
        sqlite3.SQLITE_ALTER_TABLE,
        sqlite3.SQLITE_ATTACH,
        sqlite3.SQLITE_DETACH,
        sqlite3.SQLITE_REINDEX,
    }

    connection = sqlite3.connect(resolved.as_uri() + "?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only = ON")
        connection.set_authorizer(
            lambda action, _arg1, _arg2, _db_name, _trigger: (
                sqlite3.SQLITE_DENY if action in denied_actions else sqlite3.SQLITE_OK
            )
        )
        connection.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 1000)
        connection.execute("SAVEPOINT driftsql_read_action")
        cursor = connection.execute(sql)
        columns = [item[0] for item in (cursor.description or ())]
        rows = cursor.fetchmany(max_rows + 1)
        truncated = len(rows) > max_rows
        rows = rows[:max_rows]
        return {
            "success": True,
            "error": None,
            "columns": columns,
            "rows": rows,
            "truncated": truncated,
            "rolled_back": True,
            "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
        }
    except Exception as error:
        return {
            "success": False,
            "error": str(error),
            "columns": [],
            "rows": [],
            "truncated": False,
            "rolled_back": True,
            "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
        }
    finally:
        connection.set_progress_handler(None, 0)
        try:
            connection.execute("ROLLBACK TO driftsql_read_action")
            connection.execute("RELEASE driftsql_read_action")
        except sqlite3.Error:
            pass
        connection.close()


class VersionedSqlExecutorTool(_TrajectoryStateTool):
    async def create(
        self,
        instance_id: str | None = None,
        ground_truth: str | None = None,
        **kwargs: Any,
    ) -> tuple[str, ToolResponse]:
        # VERL calls ``create`` before every tool invocation.  A trajectory may
        # execute SQL several times, but it must keep one database session and
        # one TemporaryDirectory until the loop releases the trajectory.
        reused = instance_id is not None and instance_id in self._instances
        instance_id, _ = await super().create(
            instance_id=instance_id,
            ground_truth=ground_truth,
            **kwargs,
        )
        state = self._state(instance_id)
        if reused:
            return instance_id, ToolResponse(
                text=(
                    f"Active database '{state['db_id']}' session reused "
                    f"(isolated={state.get('session_isolated', False)})."
                )
            )
        existing_db = Path(str(state.get("db_path", ""))).resolve()
        if existing_db.is_file():
            isolate = bool(
                state.get(
                    "isolate_db",
                    self.config.get("isolate_existing_db", True),
                )
            )
            if isolate:
                temporary = tempfile.TemporaryDirectory(
                    prefix="driftsql-session-",
                    dir=os.environ.get("DRIFTSQL_TMPDIR"),
                    ignore_cleanup_errors=True,
                )
                target = Path(temporary.name) / f"{state['db_id']}__session.sqlite"
                try:
                    if state.get("sync_io"):
                        _backup_sqlite(existing_db, target)
                    else:
                        await asyncio.to_thread(_backup_sqlite, existing_db, target)
                except Exception:
                    temporary.cleanup()
                    await super().release(instance_id)
                    raise
                state["source_db_path"] = str(existing_db)
                state["db_path"] = str(target)
                state["_temporary"] = temporary
                state["session_isolated"] = True
            else:
                state["db_path"] = str(existing_db)
                state["session_isolated"] = False
            return instance_id, ToolResponse(
                text=(
                    f"Active database '{state['db_id']}' is ready at schema "
                    f"version {state.get('db_version', 'unknown')} "
                    f"(isolated={state['session_isolated']})."
                )
            )

        source_db = Path(str(state.get("source_db", ""))).resolve()
        schema_diff = state.get("schema_diff")
        if not source_db.is_file():
            await super().release(instance_id)
            raise FileNotFoundError(source_db)
        if not schema_diff:
            await super().release(instance_id)
            raise ValueError("schema_diff is required in create_kwargs")

        temporary = tempfile.TemporaryDirectory(
            prefix="driftsql-rollout-",
            dir=os.environ.get("DRIFTSQL_TMPDIR"),
            ignore_cleanup_errors=True,
        )
        target = Path(temporary.name) / f"{state['db_id']}__v2.sqlite"
        try:
            if state.get("sync_io"):
                materialize_schema_diff(source_db, target, schema_diff)
            else:
                await asyncio.to_thread(
                    materialize_schema_diff,
                    source_db,
                    target,
                    schema_diff,
                )
        except Exception:
            temporary.cleanup()
            await super().release(instance_id)
            raise
        state["db_path"] = str(target)
        state["_temporary"] = temporary
        state["session_isolated"] = True
        return instance_id, ToolResponse(
            text=(
                f"Active database '{state['db_id']}' materialized at schema "
                f"version {state.get('db_version', 'unknown')}."
            )
        )

    @rollout_trace_op
    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs: Any,
    ) -> tuple[ToolResponse, float, dict]:
        state = self._state(instance_id)
        sql = str(parameters.get("sql", ""))
        timeout_seconds = float(self.config.get("timeout_seconds", 30))
        max_rows = int(self.config.get("max_rows", 100))

        if state.get("sync_io"):
            result = _execute_read_only(
                Path(state["db_path"]), sql, timeout_seconds, max_rows
            )
        else:
            result = await asyncio.to_thread(
                _execute_read_only,
                Path(state["db_path"]),
                sql,
                timeout_seconds,
                max_rows,
            )
        return (
            ToolResponse(text=json.dumps(result, ensure_ascii=False, default=str)),
            0.0,
            {
                "execution_success": bool(result["success"]),
                "execution_error": result["error"],
                "execution_elapsed_ms": result.get("elapsed_ms", 0.0),
                "session_isolated": bool(state.get("session_isolated", False)),
                "rolled_back": bool(result.get("rolled_back", False)),
            },
        )

    async def release(self, instance_id: str, **kwargs: Any) -> None:
        state = self._instances.pop(instance_id, None)
        if state and state.get("_temporary"):
            if state.get("sync_io"):
                state["_temporary"].cleanup()
            else:
                await asyncio.to_thread(state["_temporary"].cleanup)


class SubmitSolutionTool(_TrajectoryStateTool):
    @rollout_trace_op
    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs: Any,
    ) -> tuple[ToolResponse, float, dict]:
        self._state(instance_id)
        sql = str(parameters.get("sql", "")).strip()
        if not sql:
            return (
                ToolResponse(text="Error: No SQL provided in solution."),
                0.0,
                {"submitted": False},
            )
        return (
            ToolResponse(
                text="Solution submitted. It will be execution-verified."
            ),
            0.0,
            {"submitted": True},
        )
