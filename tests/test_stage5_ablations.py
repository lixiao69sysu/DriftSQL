from __future__ import annotations

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path

import pytest


def _load(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ablations = _load("prepare_stage5_tool_ablations")
replay = _load("build_stage5_failure_replay")
comparison = _load("compare_stage5_eval")
selection = _load("select_stage5_checkpoint")


def _row() -> dict:
    return {
        "prompt": [
            {"role": "system", "content": "original"},
            {"role": "user", "content": "query"},
        ],
        "reward_model": {"ground_truth": "SELECT 1"},
        "extra_info": {
            "instance_id": "case-1",
            "db_id": "db-1",
            "tool_selection": [
                "get_schema",
                "ask_user",
                "get_knowledge_definition",
                "execute_sql",
                "submit_solution",
            ],
            "result_fingerprint": {"row_count": 1, "value_hash": "abc"},
        },
    }


def test_no_ask_user_changes_only_prompt_and_tool_selection() -> None:
    source = _row()
    before = deepcopy(source)
    result = ablations.ablate_row(source, "no_ask_user")
    assert source == before
    assert "ask_user" not in result["extra_info"]["tool_selection"]
    assert "get_knowledge_definition" in result["extra_info"]["tool_selection"]
    assert result["reward_model"] == source["reward_model"]
    assert result["extra_info"]["result_fingerprint"] == source["extra_info"]["result_fingerprint"]
    assert "cannot ask the user" in result["prompt"][0]["content"]


def test_no_hkb_changes_only_prompt_and_tool_selection() -> None:
    source = _row()
    result = ablations.ablate_row(source, "no_hkb")
    assert "get_knowledge_definition" not in result["extra_info"]["tool_selection"]
    assert "ask_user" in result["extra_info"]["tool_selection"]
    assert result["extra_info"]["instance_id"] == "case-1"
    assert "No business-knowledge retriever" in result["prompt"][0]["content"]


def test_failure_miner_uses_success_rate_and_builds_equal_size_control() -> None:
    rollouts = [
        {"instance_id": "case-1", "task_success": False, "score": -0.2, "turn_limit": True},
        {"instance_id": "case-1", "task_success": False, "score": 0.0},
        {"instance_id": "case-2", "task_success": True, "score": 1.1},
        {"instance_id": "case-2", "task_success": False, "score": -0.1},
    ]
    hard_ids, diagnostics = replay.mine_failures(
        rollouts,
        success_rate_threshold=0.5,
    )
    assert hard_ids == ["case-1"]
    assert diagnostics[0]["turn_limit"] == 1

    source = [
        {"extra_info": {"instance_id": f"case-{index}"}}
        for index in range(1, 5)
    ]
    hard, mixed, control = replay.sample_replay_rows(
        source,
        hard_ids,
        hard_fraction=0.5,
        seed=7,
    )
    assert len(hard) == 1
    assert len(mixed) == len(control) == len(source)
    assert sum(row["extra_info"]["instance_id"] == "case-1" for row in mixed) >= 2


def test_rollout_segments_select_resume_lineage_and_reject_overlap(tmp_path: Path) -> None:
    first = tmp_path / "first"
    resumed = tmp_path / "resumed"
    first.mkdir()
    resumed.mkdir()
    for directory, steps in ((first, (1, 2, 3)), (resumed, (3, 4))):
        for step in steps:
            (directory / f"{step}.jsonl").write_text("{}\n", encoding="utf-8")

    selected = replay.load_rollout_files([(first, 1, 2), (resumed, 3, 4)])
    assert [int(path.stem) for path in selected] == [1, 2, 3, 4]
    assert selected[2].parent == resumed
    with pytest.raises(ValueError, match="step 3"):
        replay.load_rollout_files([(first, None, None), (resumed, None, None)])


def test_parse_rollout_segment() -> None:
    directory, first, last = replay.parse_rollout_segment("run/rollouts@11-40")
    assert directory == Path("run/rollouts")
    assert (first, last) == (11, 40)
    with pytest.raises(Exception):
        replay.parse_rollout_segment("run/rollouts")


def test_stage5_comparison_reports_slices_and_exact_pair_test() -> None:
    rows = [
        {
            "task_success": True,
            "executable": True,
            "termination_reason": "submitted",
            "drift_type": "column_rename",
            "difficulty": "easy",
            "safety": {"unsafe": False, "timed_out": False},
            "usage": {"model_calls": 2, "tool_calls": 2},
        },
        {
            "task_success": False,
            "executable": False,
            "termination_reason": "turn_limit",
            "drift_type": "column_rename",
            "difficulty": "hard",
            "safety": {"unsafe": False, "timed_out": True},
            "usage": {"model_calls": 3, "tool_calls": 3},
        },
    ]
    overall = comparison.metrics(rows)
    assert overall["task_success_rate"] == 0.5
    assert overall["turn_limit"] == 1
    assert comparison.slices(rows)["difficulty"]["hard"]["timeout_tasks"] == 1
    assert comparison.exact_mcnemar(3, 0) == 0.25


def test_checkpoint_selection_uses_fixed_metric_order_then_earlier_step() -> None:
    common = {
        "tasks": 169,
        "task_success": 11,
        "submitted": 29,
        "unsafe_actions": 0,
        "timeout_tasks": 0,
        "turn_limit": 140,
        "duplicate_question_tasks": 7,
        "duplicate_execution_tasks": 13,
        "average_tool_calls": 6.5,
    }
    candidates = [
        {**common, "variant": "grpo-step10", "executable": 27},
        {**common, "variant": "grpo-step20", "executable": 25},
        {**common, "variant": "grpo-step30", "executable": 27},
    ]
    assert selection.select_checkpoint(candidates)["variant"] == "grpo-step10"
    candidates[1]["task_success"] = 12
    assert selection.select_checkpoint(candidates)["variant"] == "grpo-step20"
