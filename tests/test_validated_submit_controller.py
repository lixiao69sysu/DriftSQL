from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from driftsql.controllers.validated_submit import (
    find_contract_validated_submission,
    is_read_only_query,
)
from driftsql.drift import fingerprint_query


def fixture(tmp_path: Path):
    source = tmp_path / "source.sqlite"
    with sqlite3.connect(source) as connection:
        connection.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, value TEXT)")
        connection.executemany("INSERT INTO items VALUES (?, ?)", [(1, "a"), (2, "b")])
        connection.commit()
    expected = fingerprint_query(source, "SELECT id, value FROM items ORDER BY id")
    extra = {
        "db_id": "items",
        "source_db": str(source),
        "schema_diff": {
            "db_id": "items",
            "from_version": "v1",
            "to_version": "v2",
            "operations": [
                {
                    "type": "add_column",
                    "table": "items",
                    "new_name": "audit_tag",
                    "declared_type": "TEXT",
                    "default_sql": "'new'",
                }
            ],
        },
        "result_fingerprint": {
            "row_count": expected.row_count,
            "value_hash": expected.value_hash,
        },
    }
    return extra


def event(name: str, arguments: dict, *, execution_success: bool = False):
    return {
        "tool_name": name,
        "arguments": arguments,
        "metrics": {"execution_success": execution_success},
    }


def test_controller_accepts_only_post_diff_contract_match(tmp_path: Path) -> None:
    extra = fixture(tmp_path)
    trajectory = [
        event("execute_sql", {"sql": "SELECT * FROM items ORDER BY id"}, execution_success=True),
        event("inspect_schema_diff", {}),
        event(
            "execute_sql",
            {"sql": "SELECT id, value FROM items ORDER BY id"},
            execution_success=True,
        ),
        # A learned policy may repeat retrieval after reaching a valid state.
        # The online controller would already have submitted at event 2.
        event("inspect_schema_diff", {}),
    ]
    decision = find_contract_validated_submission(
        trajectory, extra, temporary_root=tmp_path / "controller"
    )
    assert decision.accepted is True
    assert decision.reason == "contract_validated"
    assert decision.event_index == 2
    assert decision.fingerprint_match is True


def test_controller_rejects_executable_result_contract_mismatch(tmp_path: Path) -> None:
    extra = fixture(tmp_path)
    trajectory = [
        event("inspect_schema_diff", {}),
        event("execute_sql", {"sql": "SELECT * FROM items ORDER BY id"}, execution_success=True),
    ]
    decision = find_contract_validated_submission(
        trajectory, extra, temporary_root=tmp_path / "controller"
    )
    assert decision.accepted is False
    assert decision.reason == "result_contract_mismatch"


def test_controller_fails_closed_for_unsafe_or_missing_diff(tmp_path: Path) -> None:
    extra = fixture(tmp_path)
    unsafe = [
        event("inspect_schema_diff", {}),
        event("execute_sql", {"sql": "DELETE FROM items"}, execution_success=True),
    ]
    decision = find_contract_validated_submission(
        unsafe, extra, temporary_root=tmp_path / "controller"
    )
    assert decision.accepted is False
    assert decision.reason == "unsafe_post_diff_candidate"
    no_diff = [
        event(
            "execute_sql",
            {"sql": "SELECT id, value FROM items ORDER BY id"},
            execution_success=True,
        )
    ]
    assert (
        find_contract_validated_submission(
            no_diff, extra, temporary_root=tmp_path / "controller"
        ).reason
        == "schema_diff_not_inspected"
    )
    assert is_read_only_query("SELECT * FROM items") is True
    assert is_read_only_query("DELETE FROM items") is False
    assert is_read_only_query("SELECT 1; DELETE FROM items") is False


def test_generated_p5_replay_never_reads_gate() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "reports/p6/validated_submit/p5-sft20/summary.json"
    if not path.exists():
        return
    summary = json.loads(path.read_text(encoding="utf-8"))
    assert summary["gate_rows_read"] is False
    assert summary["unsafe_auto_submissions"] == 0
