from __future__ import annotations

import runpy
from pathlib import Path


RUNNER = Path(__file__).resolve().parents[1] / "scripts/run_stage1_baseline.py"
parse_fallback_action = runpy.run_path(str(RUNNER))["parse_fallback_action"]


def test_fallback_parser_accepts_explicit_react_variants() -> None:
    assert parse_fallback_action("<execute_sql>SELECT 1</execute_sql>") == (
        "execute_sql",
        ["SELECT 1"],
        False,
    )
    assert parse_fallback_action(
        "Action: submit_solution\n```sql\nSELECT count(*) FROM t\n```"
    ) == ("submit_solution", ["SELECT count(*) FROM t"], True)


def test_fallback_parser_only_infers_bare_fence_when_agent_supplies_default() -> None:
    response = "```sql\nSELECT * FROM t LIMIT 5\n```"
    assert parse_fallback_action(response) == (None, [], False)
    assert parse_fallback_action(response, "execute_sql") == (
        "execute_sql",
        ["SELECT * FROM t LIMIT 5"],
        False,
    )
