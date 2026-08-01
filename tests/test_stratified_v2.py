from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str) -> ModuleType:
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(task_id: str, db_id: str, drift_type: str) -> dict:
    operations = [] if drift_type == "clean" else [{"type": drift_type}]
    if drift_type == "compound":
        operations = [{"type": "rename_table"}, {"type": "rename_column"}]
    return {
        "task_id": task_id,
        "db_id": db_id,
        "source": "fixture",
        "stale_sql": "SELECT value FROM t",
        "stale_error": None if drift_type == "clean" else "no such column",
        "schema_diff": {"operations": operations},
    }


def test_v2_profiles_are_balanced_inside_each_drift_family() -> None:
    factory = load_script("build_stratified_drift_data_v2")
    rows = [
        row(f"{kind}-{index}", f"db-{index % 5}", kind)
        for kind in ("add_column", "rename_column", "rename_table", "replace_column", "compound")
        for index in range(20)
    ] + [row(f"clean-{index}", f"clean-db-{index % 3}", "clean") for index in range(10)]
    enriched = factory.enrich(rows, set())

    for kind in ("add_column", "rename_column", "rename_table", "replace_column", "compound"):
        profiles = Counter(item["interaction_profile"] for item in enriched if item["drift_type"] == kind)
        assert profiles == {"must_ask": 6, "knowledge_only": 5, "schema_only": 9}
    assert Counter(item["interaction_profile"] for item in enriched if item["drift_type"] == "clean") == {
        "direct_clean": 10
    }


def test_database_split_is_disjoint_and_keeps_frozen_database_in_test() -> None:
    splitter = load_script("split_stratified_drift_v2")
    rows = []
    for db_index in range(12):
        for item_index, kind in enumerate(("add_column", "rename_column", "compound", "clean")):
            item = row(f"task-{db_index}-{item_index}", f"db-{db_index}", kind)
            item.update(
                {
                    "scenario_type": "clean" if kind == "clean" else "compound" if kind == "compound" else "atomic",
                    "drift_type": kind,
                    "interaction_profile": "direct_clean" if kind == "clean" else "schema_only",
                    "difficulty": "hard" if kind == "compound" else "easy",
                    "failure_mode": "clean_no_drift" if kind == "clean" else "explicit_schema_error",
                }
            )
            rows.append(item)
    splits, _ = splitter.assign_databases(
        rows,
        frozen_test_databases={"db-0"},
        fractions={"train": 0.70, "dev": 0.15, "test": 0.15},
        seed=42,
        trials=10,
    )
    databases = {
        split: {item["db_id"] for item in values}
        for split, values in splits.items()
    }
    assert "db-0" in databases["test"]
    assert not (databases["train"] & databases["dev"])
    assert not (databases["train"] & databases["test"])
    assert not (databases["dev"] & databases["test"])


def test_v2_interaction_profiles_have_expected_action_budgets() -> None:
    prepare = load_script("prepare_stratified_five_tool_data_v2")
    assert len(prepare.expected_tool_sequence("must_ask")) == 6
    assert len(prepare.expected_tool_sequence("knowledge_only")) == 5
    assert len(prepare.expected_tool_sequence("schema_only")) == 4
    assert prepare.expected_tool_sequence("direct_clean") == ["execute_sql", "submit_solution"]
