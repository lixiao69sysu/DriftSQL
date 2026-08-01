"""Tune-only scenario catalogue used by the product API."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from driftsql.service.schemas import (
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
    """Expose frozen Tune aggregates without local paths or trajectory answers."""

    _DISPLAY_NAMES = {
        "stage7_frozen_tune55": "Stage 7 frozen",
        "stage8_sft20_tune55": "Stage 8 SFT20",
        "grpo_trial1_step5_add30": "GRPO step 5",
        "grpo_trial1_step10_add30": "GRPO step 10",
        "grpo_conservative_step2_add30": "Conservative step 2",
        "grpo_conservative_step4_add30": "Conservative step 4",
        "grpo_conservative_step6_add30": "Conservative step 6",
    }

    def __init__(self, frozen_candidate_path: Path) -> None:
        self.path = Path(frozen_candidate_path)
        self._experiments: list[ExperimentRead] = []
        self._selected_id = "stage8_sft20_tune55"

    def load(self) -> None:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        results = payload.get("tune_selection", {}).get("results", {})
        experiments: list[ExperimentRead] = []
        for experiment_id, raw in results.items():
            if not isinstance(raw, dict):
                continue
            category = "GRPO" if "grpo" in experiment_id else "SFT"
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
        self._experiments = experiments

    def list_experiments(self) -> ExperimentList:
        return ExperimentList(
            experiments=list(self._experiments),
            selected_experiment_id=self._selected_id,
        )
