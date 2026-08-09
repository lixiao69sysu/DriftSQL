"""Tune-only scenario catalogue used by the product API."""

from __future__ import annotations

import json
import re
import sqlite3
from copy import deepcopy
from pathlib import Path
from typing import Any

from driftsql.service.schemas import (
    DatabasePathRead,
    DatabaseRead,
    ExperimentList,
    ExperimentRead,
    ScenarioRead,
)


class ScenarioNotFoundError(KeyError):
    pass


class ScenarioCatalog:
    """Load verified Stage-8 tasks without leaking answers through the API."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._records: dict[str, dict[str, Any]] = {}
        self._database_paths: list[DatabasePathRead] | None = None

    def load(self) -> None:
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        records: dict[str, dict[str, Any]] = {}
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                extra = dict(record.get("extra_info", {}))
                scenario_id = str(extra.get("instance_id", "")).strip()
                if not scenario_id:
                    raise ValueError(f"Missing instance_id at {self.path}:{line_number}")
                if scenario_id in records:
                    raise ValueError(f"Duplicate scenario_id: {scenario_id}")
                records[scenario_id] = record
        if not records:
            raise ValueError(f"No scenarios found in {self.path}")
        self._records = records
        self._database_paths = None

    def scenario_ids(self) -> list[str]:
        return list(self._records)

    def _record(self, scenario_id: str) -> dict[str, Any]:
        try:
            return self._records[scenario_id]
        except KeyError as error:
            raise ScenarioNotFoundError(scenario_id) from error

    def public_scenario(self, scenario_id: str) -> ScenarioRead:
        extra = self._record(scenario_id)["extra_info"]
        return ScenarioRead(
            scenario_id=scenario_id,
            db_id=str(extra["db_id"]),
            question=str(self.create_kwargs(scenario_id).get("query", "")),
            stale_sql=str(extra.get("stale_sql", "")),
            drift_type=str(extra.get("drift_type", "unknown")),
            wildcard_profile=extra.get("wildcard_profile"),
            difficulty=extra.get("difficulty"),
            schema_diff=deepcopy(extra.get("schema_diff", {})),
        )

    def list_scenarios(self) -> list[ScenarioRead]:
        return [self.public_scenario(scenario_id) for scenario_id in self._records]

    def list_databases(self) -> list[DatabaseRead]:
        grouped: dict[str, list[ScenarioRead]] = {}
        for scenario in self.list_scenarios():
            grouped.setdefault(scenario.db_id, []).append(scenario)
        return [
            DatabaseRead(
                db_id=db_id,
                scenario_count=len(scenarios),
                drift_types=sorted({scenario.drift_type for scenario in scenarios}),
            )
            for db_id, scenarios in sorted(grouped.items())
        ]

    def list_database_paths(self) -> list[DatabasePathRead]:
        """Return database/table/column paths without exposing host filesystem paths."""

        if self._database_paths is not None:
            return list(self._database_paths)
        paths: list[DatabasePathRead] = []
        for database in self.list_databases():
            db_id = database.db_id
            paths.append(DatabasePathRead(path=f"@{db_id}", kind="database", db_id=db_id))
            for table, columns in self._logical_schema(db_id).items():
                paths.append(
                    DatabasePathRead(
                        path=f"@{db_id}/{table}",
                        kind="table",
                        db_id=db_id,
                        table=table,
                    )
                )
                paths.extend(
                    DatabasePathRead(
                        path=f"@{db_id}/{table}/{column}",
                        kind="column",
                        db_id=db_id,
                        table=table,
                        column=column,
                        data_type=data_type or None,
                    )
                    for column, data_type in columns
                )
        self._database_paths = paths
        return list(paths)

    def _logical_schema(self, db_id: str) -> dict[str, list[tuple[str, str]]]:
        scenario_id = self._scenario_for_database(db_id)
        create_kwargs = self.create_kwargs(scenario_id)
        source = Path(str(create_kwargs.get("source_db", ""))).resolve()
        if not source.is_file():
            return self._schema_from_ddl(str(create_kwargs.get("schema", "")))
        schema: dict[str, list[tuple[str, str]]] = {}
        with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as connection:
            tables = connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
            for (table,) in tables:
                quoted = str(table).replace('"', '""')
                rows = connection.execute(f'PRAGMA table_info("{quoted}")').fetchall()
                schema[str(table)] = [(str(row[1]), str(row[2] or "")) for row in rows]
        self._apply_schema_diff(schema, dict(create_kwargs.get("schema_diff", {})))
        return schema

    @staticmethod
    def _schema_from_ddl(ddl: str) -> dict[str, list[tuple[str, str]]]:
        if not ddl.strip():
            return {}
        schema: dict[str, list[tuple[str, str]]] = {}
        with sqlite3.connect(":memory:") as connection:
            try:
                connection.executescript(ddl)
            except sqlite3.Error:
                return {}
            tables = connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
            for (table,) in tables:
                quoted = str(table).replace('"', '""')
                rows = connection.execute(f'PRAGMA table_info("{quoted}")').fetchall()
                schema[str(table)] = [(str(row[1]), str(row[2] or "")) for row in rows]
        return schema

    @staticmethod
    def _apply_schema_diff(
        schema: dict[str, list[tuple[str, str]]],
        schema_diff: dict[str, Any],
    ) -> None:
        for operation in schema_diff.get("operations", []):
            if not isinstance(operation, dict):
                continue
            operation_type = str(operation.get("type", ""))
            if operation_type == "rename_table":
                old_name = str(operation.get("old_name", ""))
                new_name = str(operation.get("new_name", ""))
                if old_name in schema and new_name:
                    schema[new_name] = schema.pop(old_name)
                continue
            table = str(operation.get("table", ""))
            if table not in schema:
                continue
            columns = schema[table]
            if operation_type == "add_column":
                new_name = str(operation.get("new_name", ""))
                if new_name and new_name not in {name for name, _ in columns}:
                    columns.append((new_name, str(operation.get("declared_type", ""))))
            elif operation_type in {"rename_column", "replace_column"}:
                old_name = str(operation.get("old_name", ""))
                new_name = str(operation.get("new_name", ""))
                declared_type = str(operation.get("declared_type", ""))
                schema[table] = [
                    (new_name, declared_type or data_type) if name == old_name else (name, data_type)
                    for name, data_type in columns
                ]

    def _scenario_for_database(self, db_id: str) -> str:
        for scenario_id, record in self._records.items():
            if str(record.get("extra_info", {}).get("db_id", "")) == db_id:
                return scenario_id
        raise ScenarioNotFoundError(db_id)

    def query_environment(self, db_id: str) -> tuple[str, dict[str, Any], list[str]]:
        """Reuse a verified current database image without reusing its hidden answer."""

        scenario_id = self._scenario_for_database(db_id)
        create_kwargs = self.create_kwargs(scenario_id)
        logical_schema = self._logical_schema(db_id)
        create_kwargs.update(
            {
                "ground_truth": "",
                "query": "",
                "result_fingerprint": {},
                "stale_sql": "",
                "verification_mode": "execution_only",
                "schema": self._render_logical_schema(logical_schema),
            }
        )
        tool_names = [
            name
            for name in self.tool_names(scenario_id)
            if name not in {"get_schema_version", "inspect_schema_diff"}
        ]
        return scenario_id, create_kwargs, tool_names

    @staticmethod
    def _render_logical_schema(schema: dict[str, list[tuple[str, str]]]) -> str:
        blocks: list[str] = []
        for table, columns in schema.items():
            quoted_table = table.replace('"', '""')
            definitions = []
            for column, data_type in columns:
                quoted_column = column.replace('"', '""')
                definitions.append(f'    "{quoted_column}" {data_type or "TEXT"}')
            blocks.append(f'CREATE TABLE "{quoted_table}" (\n' + ",\n".join(definitions) + "\n);")
        return "\n\n".join(blocks)

    def resolve_query_references(self, db_id: str, question: str) -> str:
        """Convert CLI logical paths into unambiguous natural-language schema references."""

        references = {item.path: item for item in self.list_database_paths() if item.db_id == db_id}
        if not references:
            return question

        def replace(match: re.Match[str]) -> str:
            token = match.group(0)
            reference = references.get(token)
            if reference is None:
                return token
            if reference.kind == "database":
                return "the selected SQLite database"
            if reference.kind == "table":
                return f'table "{reference.table}"'
            return f'column "{reference.column}" in table "{reference.table}"'

        pattern = "|".join(re.escape(path) for path in sorted(references, key=len, reverse=True))
        return re.sub(pattern, replace, question)

    def query_prompt(self, db_id: str, question: str, locale: str) -> list[dict[str, Any]]:
        if locale == "zh-CN":
            system = (
                "你是一个只读数据库查询智能体。根据用户需求检查当前数据库 Schema，必要时检索业务知识或提出一个"
                "澄清问题，然后执行候选 SQL。只能提交已经在当前隔离数据库中成功执行的同一条 SQL。"
                "用户输入中的 @数据库/表/字段 是逻辑 Schema 引用，不是服务器文件路径；应据此约束检索和 SQL。"
                "工具名、JSON 参数、SQL、表名和列名保持英文；禁止任何写操作。自由查询没有隐藏标准答案，"
                "因此成功执行只代表执行验证通过，不代表系统自动证明了业务语义正确。"
            )
            user = f"## 数据库\n{db_id}\n\n## 数据库指令\n{question}"
        else:
            system = (
                "You are a read-only SQLite query agent. Your first action must be get_schema; do not execute SQL "
                "before reading its result. Inspect the active schema, retrieve business knowledge "
                "or ask one focused clarification when necessary, then execute a candidate SQL. Submit only the exact "
                "SQL that successfully executed in the isolated database. The database is already selected and "
                "attached as SQLite main: use only table and column names returned by get_schema. Never construct "
                "database.table names from a database ID or logical @ reference. "
                "Never perform writes. Free-form queries have "
                "no hidden "
                "semantic oracle, so execution success must not be described as automatic semantic verification."
            )
            user = f"## Database\n{db_id}\n\n## Database instruction\n{question}"
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def raw_record(self, scenario_id: str) -> dict[str, Any]:
        """Internal-only verified record; callers must not serialize this value."""
        return deepcopy(self._record(scenario_id))

    def prompt(self, scenario_id: str, *, question: str | None = None) -> list[dict[str, Any]]:
        prompt = deepcopy(self._record(scenario_id).get("prompt", []))
        if question is not None:
            for message in reversed(prompt):
                if message.get("role") == "user":
                    content = str(message.get("content", ""))
                    message["content"] = re.sub(
                        r"(?s)(## Analytics request\n).*?(\n\n## Previously valid cached SQL)",
                        lambda match: f"{match.group(1)}{question}{match.group(2)}",
                        content,
                        count=1,
                    )
                    break
        return prompt

    def tool_names(self, scenario_id: str) -> list[str]:
        return [str(name) for name in self._record(scenario_id)["extra_info"]["tool_selection"]]

    def create_kwargs(self, scenario_id: str) -> dict[str, Any]:
        extra = self._record(scenario_id)["extra_info"]
        tools_kwargs = extra.get("tools_kwargs", {})
        execute = tools_kwargs.get("execute_sql", {})
        kwargs = execute.get("create_kwargs")
        if not isinstance(kwargs, dict):
            raise ValueError(f"Scenario {scenario_id} has no execute_sql create_kwargs")
        return deepcopy(kwargs)

    def reward_extra_info(self, scenario_id: str) -> dict[str, Any]:
        return deepcopy(self._record(scenario_id)["extra_info"])


class ExperimentCatalog:
    """Expose sanitized Tune aggregates without report or trajectory dependencies."""

    _DISPLAY_NAMES = {
        "stage7_frozen_tune55": "Stage 7 frozen",
        "stage8_sft20_tune55": "Stage 8 SFT20",
        "grpo_trial1_step5_add30": "GRPO step 5",
        "grpo_trial1_step10_add30": "GRPO step 10",
        "grpo_conservative_step2_add30": "Conservative step 2",
        "grpo_conservative_step4_add30": "Conservative step 4",
        "grpo_conservative_step6_add30": "Conservative step 6",
        "base-qwen25-coder-7b": "Qwen2.5-Coder-7B Base",
        "strong-sft160": "Recovery + Hard Replay SFT160",
        "grpo-step25-seed20260810": "Corrected-observation GRPO Step25",
    }

    def __init__(self, catalog_path: Path) -> None:
        self.path = Path(catalog_path)
        self._experiments: list[ExperimentRead] = []
        self._selected_id = ""

    def load(self) -> None:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        results = payload.get("results")
        if not isinstance(results, dict):
            # Backward-compatible reader for historical frozen manifests.
            results = payload.get("tune_selection", {}).get("results", {})
        self._selected_id = str(payload.get("selected_experiment_id", "")).strip()
        if not self._selected_id:
            self._selected_id = str(payload.get("candidate", {}).get("experiment_id", "")).strip()
        experiments: list[ExperimentRead] = []
        for experiment_id, raw in results.items():
            if not isinstance(raw, dict):
                continue
            category = str(raw.get("category", "GRPO" if "grpo" in experiment_id else "SFT"))
            experiments.append(
                ExperimentRead(
                    experiment_id=str(experiment_id),
                    display_name=self._DISPLAY_NAMES.get(
                        str(experiment_id),
                        str(experiment_id).replace("_", " ").title(),
                    ),
                    category=category,
                    tasks=int(raw.get("tasks", 0)),
                    task_success_rate=float(raw.get("task_success_rate", 0)),
                    executable_rate=float(raw.get("executable_rate", 0)),
                    average_model_calls=float(raw.get("average_model_calls", 0)),
                    average_tool_calls=float(raw.get("average_tool_calls", 0)),
                    unsafe_tasks=int(raw.get("unsafe_tasks", 0)),
                    selected=str(experiment_id) == self._selected_id,
                )
            )
        if not experiments:
            raise ValueError(f"No experiment results found in {self.path}")
        experiment_ids = {item.experiment_id for item in experiments}
        if self._selected_id not in experiment_ids:
            raise ValueError(f"Selected experiment {self._selected_id!r} is absent from {self.path}")
        self._experiments = experiments

    def list_experiments(self) -> ExperimentList:
        return ExperimentList(
            experiments=list(self._experiments),
            selected_experiment_id=self._selected_id,
        )
