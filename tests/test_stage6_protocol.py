from __future__ import annotations

import pytest

from scripts.split_stage6_train_tune_gate import assign_databases, validate_protocol


def _row(task_id: str, db_id: str, label: str = "rename_column") -> dict[str, str]:
    return {
        "task_id": task_id,
        "db_id": db_id,
        "scenario_type": "atomic",
        "drift_type": label,
        "interaction_profile": "schema_only",
        "difficulty": "easy",
        "failure_mode": "explicit_schema_error",
        "source": "fixture",
    }


def test_stage6_assignment_is_complete_and_database_disjoint() -> None:
    rows = [_row(f"task-{db}-{index}", f"db-{db}") for db in range(12) for index in range(2)]
    split_rows, _ = assign_databases(
        rows,
        fractions={"train": 0.5, "tune": 0.25, "gate": 0.25},
        seed=7,
        trials=30,
    )
    result = validate_protocol(rows, split_rows, [_row("old", "historical-db")])
    assert result["source_task_ids_preserved"] is True
    assert all(not values for values in result["database_overlap"].values())
    assert all(split_rows[split] for split in ("train", "tune", "gate"))


def test_stage6_validation_rejects_historical_database_leakage() -> None:
    rows = [_row("a", "db-a"), _row("b", "db-b"), _row("c", "db-c")]
    split_rows = {"train": [rows[0]], "tune": [rows[1]], "gate": [rows[2]]}
    with pytest.raises(RuntimeError, match="Historical dev/test"):
        validate_protocol(rows, split_rows, [_row("old", "db-c")])
