from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "stage7_failure_balance", ROOT / "scripts/build_stage7_failure_balanced_grpo.py"
)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def _row(*, success: bool = False, reason: str = "turn_limit", final_sql: str = "", calls=()):
    return {
        "task_success": success,
        "termination_reason": reason,
        "final_sql": final_sql,
        "trajectory": [
            {"tool_name": name, "arguments": {"sql": sql}} for name, sql in calls
        ],
    }


def test_classifies_real_policy_failures() -> None:
    ground_truth = 'SELECT "a", "b" FROM "t"'
    stale = _row(
        reason="submitted",
        final_sql='SELECT * FROM "t"',
        calls=(("execute_sql", 'SELECT * FROM "t"'), ("submit_solution", 'SELECT * FROM "t"')),
    )
    assert module.classify_failure(stale, ground_truth)[0] == "premature_stale_submit"

    no_submit = _row(calls=(("execute_sql", ground_truth),))
    assert module.classify_failure(no_submit, ground_truth)[0] == "repaired_not_submitted"

    success = _row(success=True, reason="submitted", final_sql=ground_truth)
    assert module.classify_failure(success, ground_truth)[0] == "success"


def test_largest_remainder_allocation_is_exact() -> None:
    allocated = module.allocate_counts([6, 6, 5, 4], 242)
    assert sum(allocated) == 242
    assert allocated[0] == allocated[1]
    assert allocated[0] > allocated[2] > allocated[3]
