from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_stage7_process_isolated_eval.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("stage7_process_eval", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
import run_stage6_eval


def test_select_records_preserves_source_order_for_multi_type_offsets() -> None:
    records = [
        {"id": "a", "extra_info": {"drift_type": "clean"}},
        {"id": "b", "extra_info": {"drift_type": "add_column"}},
        {"id": "c", "extra_info": {"drift_type": "rename_table"}},
        {"id": "d", "extra_info": {"drift_type": "compound"}},
        {"id": "e", "extra_info": {"drift_type": "clean"}},
    ]

    selected = MODULE.select_records(records, ("clean", "compound", "rename_table"))

    assert [row["id"] for row in selected] == ["a", "c", "d", "e"]


def test_completion_budget_is_bounded_by_remaining_context() -> None:
    assert run_stage6_eval.bounded_generation_tokens([7000], 512, 8192) == 512
    assert run_stage6_eval.bounded_generation_tokens([8000], 512, 8192) == 192
    assert run_stage6_eval.bounded_generation_tokens([8191], 512, 8192) == 1
