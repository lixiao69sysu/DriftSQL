from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from driftsql.data.verl_sft_dataset import ActionNameSFTDataset, DecisionPrefixSFTDataset
from scripts.build_p6_on_policy_recovery_sft import (
    first_error_index,
    state_aware_post_error_target_index,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/p6_on_policy_recovery_sft_round1"
STATE_AWARE_DATA = ROOT / "data/processed/p6_on_policy_recovery_sft_round1_state_aware"
TRAIN_ORACLE = ROOT / "data/processed/p6_generalized_protocol/train_trajectories.parquet"


def read_rows(path: Path) -> list[dict]:
    return pq.read_table(path).to_pylist()


def target_payload(row: dict) -> dict:
    message = row["messages"][-1]
    assert message["role"] == "assistant"
    assert not message.get("tool_calls")
    return json.loads(str(message["content"]).rsplit("\n", 1)[-1])


def assistant(name: str, sql: str = "") -> dict:
    arguments = {"sql": sql} if sql else {}
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


def test_first_error_uses_action_sequence_not_sql_string_identity() -> None:
    canonical = [
        assistant("execute_sql", "SELECT a FROM t"),
        assistant("submit_solution", "SELECT a FROM t"),
    ]
    successful_equivalent = {
        "task_success": True,
        "trajectory": [
            {"tool_name": "execute_sql", "arguments": {"sql": 'SELECT "a" FROM "t"'}},
            {"tool_name": "submit_solution", "arguments": {"sql": 'SELECT "a" FROM "t"'}},
        ],
    }
    failed_same_sequence = {**successful_equivalent, "task_success": False}
    wrong_terminal_action = {
        "task_success": False,
        "trajectory": [
            {"tool_name": "execute_sql", "arguments": {"sql": "SELECT a FROM t"}},
            {"tool_name": "ask_user", "arguments": {"question": "again?"}},
        ],
    }

    assert first_error_index(successful_equivalent, canonical) == (None, "none")
    assert first_error_index(failed_same_sequence, canonical) == (1, "outcome_mismatch")
    assert first_error_index(wrong_terminal_action, canonical) == (1, "action_name")


def _event(name: str, *, sql: str = "", success: bool = False) -> dict:
    return {
        "tool_name": name,
        "arguments": {"sql": sql} if sql else {},
        "metrics": {"execution_success": success},
        "observation": json.dumps({"success": success}),
    }


def test_post_error_target_skips_version_after_diff_already_completed() -> None:
    canonical = [
        assistant("execute_sql", "SELECT old_name FROM t"),
        assistant("get_schema_version"),
        assistant("inspect_schema_diff"),
        assistant("execute_sql", "SELECT new_name FROM t"),
        assistant("submit_solution", "SELECT new_name FROM t"),
    ]
    events = [
        _event("execute_sql", sql="SELECT old_name FROM t"),
        _event("inspect_schema_diff"),
    ]
    target = state_aware_post_error_target_index(events, canonical)
    assert canonical[target]["tool_calls"][0]["function"]["name"] == "execute_sql"


def test_post_error_target_requires_diff_when_user_was_asked_too_early() -> None:
    canonical = [
        assistant("execute_sql", "SELECT old_name FROM t"),
        assistant("get_schema_version"),
        assistant("inspect_schema_diff"),
        assistant("ask_user"),
        assistant("get_knowledge_definition"),
        assistant("execute_sql", "SELECT new_name FROM t"),
        assistant("submit_solution", "SELECT new_name FROM t"),
    ]
    events = [
        _event("execute_sql", sql="SELECT old_name FROM t"),
        _event("get_schema_version"),
        _event("ask_user"),
    ]
    target = state_aware_post_error_target_index(events, canonical)
    assert canonical[target]["tool_calls"][0]["function"]["name"] == "inspect_schema_diff"


def test_post_error_target_submits_only_verified_canonical_repair() -> None:
    canonical = [
        assistant("execute_sql", "SELECT old_name FROM t"),
        assistant("get_schema_version"),
        assistant("inspect_schema_diff"),
        assistant("execute_sql", "SELECT new_name FROM t"),
        assistant("submit_solution", "SELECT new_name FROM t"),
    ]
    events = [
        _event("execute_sql", sql="SELECT old_name FROM t"),
        _event("get_schema_version"),
        _event("inspect_schema_diff"),
        _event("execute_sql", sql='SELECT "new_name" FROM "t"', success=True),
        _event("get_schema_version"),
    ]
    target = state_aware_post_error_target_index(events, canonical)
    assert canonical[target]["tool_calls"][0]["function"]["name"] == "submit_solution"


def test_recovery_data_is_train_only_and_database_isolated() -> None:
    summary = json.loads((DATA / "summary.json").read_text(encoding="utf-8"))
    train_rows = read_rows(DATA / "train.parquet")
    dev_rows = read_rows(DATA / "dev.parquet")
    oracle_ids = {
        str(value)
        for value in pq.read_table(TRAIN_ORACLE, columns=["task_id"])["task_id"].to_pylist()
    }

    assert train_rows and dev_rows
    assert len(train_rows) == summary["train_examples"]
    assert len(dev_rows) == summary["dev_examples"]
    assert not {row["db_id"] for row in train_rows} & {row["db_id"] for row in dev_rows}
    assert summary["database_overlap"] == []
    assert summary["split_guards"] == {
        "source_split": "train_only",
        "dev_rows_read": False,
        "test_rows_read": False,
        "gate_rows_read": False,
    }
    assert {str(row["task_id"]) for row in train_rows + dev_rows} <= oracle_ids


def test_every_target_is_policy_valid_plain_json_action() -> None:
    rows = read_rows(DATA / "train.parquet") + read_rows(DATA / "dev.parquet")
    for row in rows:
        payload = target_payload(row)
        assert payload["name"] == row["target_action"]
        assert row["target_action"] in row["available_tools"]
        assert isinstance(payload["arguments"], dict)
        assert row["source"] == "real_on_policy_train_rollout+verified_train_oracle"
        assert row["curriculum_stage"] == "single"
        assert row["recovery_context"] in {"pre_error", "post_error"}
        assert row["first_error_kind"] != "none"
        assert int(row["token_count"]) <= 3072


def test_examples_come_only_from_failed_actual_rollouts() -> None:
    audit = [
        json.loads(line)
        for line in (DATA / "audit.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {str(row["task_id"]): row for row in audit}
    rows = read_rows(DATA / "train.parquet") + read_rows(DATA / "dev.parquet")
    contexts = {str(row["recovery_context"]) for row in rows}

    assert contexts == {"pre_error", "post_error"}
    for row in rows:
        source = by_id[str(row["task_id"])]
        assert source["task_success"] is False
        assert source["first_error_index"] is not None
        assert row["recovery_context"] in source["examples"]
        if row["recovery_context"] == "post_error":
            assert sum(message["role"] == "assistant" for message in row["messages"]) >= 2


def test_state_aware_data_never_teaches_version_after_consumed_diff() -> None:
    rows = read_rows(STATE_AWARE_DATA / "train.parquet") + read_rows(
        STATE_AWARE_DATA / "dev.parquet"
    )
    post_error = [row for row in rows if row["recovery_context"] == "post_error"]
    assert post_error
    assert all(row["recovery_target_strategy"] == "state_aware" for row in post_error)
    assert all(row["target_action"] in row["available_tools"] for row in post_error)
    for row in post_error:
        prior_assistants = [
            message for message in row["messages"][:-1] if message["role"] == "assistant"
        ]
        if not prior_assistants:
            continue
        calls = prior_assistants[-1].get("tool_calls") or []
        actual = str(calls[0]["function"]["name"]) if len(calls) == 1 else ""
        assert not (
            actual == "inspect_schema_diff" and row["target_action"] == "get_schema_version"
        )


def test_action_name_dataset_supervises_only_the_target_tool_name() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / "models/Qwen2.5-Coder-7B-Instruct", local_files_only=True
    )
    dataset = ActionNameSFTDataset(
        parquet_files=str(DATA / "train.parquet"),
        tokenizer=tokenizer,
        processor=None,
        config={
            "messages_key": "messages",
            "tools_key": "tools",
            "enable_thinking_key": "enable_thinking",
            "max_length": 3072,
            "truncation": "error",
            "shuffle": False,
        },
    )
    rows = read_rows(DATA / "train.parquet")
    first_by_target: dict[str, int] = {}
    for index, row in enumerate(rows):
        first_by_target.setdefault(str(row["target_action"]), index)
    assert set(first_by_target) == {
        "ask_user",
        "execute_sql",
        "get_knowledge_definition",
        "get_schema_version",
        "submit_solution",
    }
    for target, index in first_by_target.items():
        item = dataset[index]
        selected = item["input_ids"][item["loss_mask"].bool()].tolist()
        assert selected == tokenizer.encode(target, add_special_tokens=False)


def test_decision_prefix_dataset_supervises_thought_through_action_not_arguments() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / "models/Qwen2.5-Coder-7B-Instruct", local_files_only=True
    )
    dataset = DecisionPrefixSFTDataset(
        parquet_files=str(DATA / "train.parquet"),
        tokenizer=tokenizer,
        processor=None,
        config={
            "messages_key": "messages",
            "tools_key": "tools",
            "enable_thinking_key": "enable_thinking",
            "max_length": 3072,
            "truncation": "error",
            "shuffle": False,
        },
    )
    rows = read_rows(DATA / "train.parquet")
    execute_index = next(
        index for index, row in enumerate(rows) if row["target_action"] == "execute_sql"
    )
    item = dataset[execute_index]
    selected_ids = item["input_ids"][item["loss_mask"].bool()].tolist()
    selected_text = tokenizer.decode(selected_ids)

    assert "<think>" in selected_text
    assert "execute_sql" in selected_text
    assert '"sql"' not in selected_text
    assert len(selected_ids) > len(tokenizer.encode("execute_sql", add_special_tokens=False))
