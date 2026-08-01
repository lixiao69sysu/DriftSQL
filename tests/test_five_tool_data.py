from __future__ import annotations

from driftsql.data.tool_sft import (
    build_five_tool_messages,
    clarification_spec,
    expand_next_action_messages,
    use_plain_json_for_last_action,
)


def test_five_tool_trajectory_contains_clarification_retrieval_and_repair() -> None:
    manifest = {
        "question": "How many active customers are there?",
        "evidence": "active customers refers to customers with status = 'active';",
        "stale_sql": "SELECT COUNT(*) FROM customer WHERE old_status = 'active'",
        "schema_diff": {
            "operations": [
                {
                    "type": "rename_column",
                    "table": "customer",
                    "old_name": "old_status",
                    "new_name": "status",
                }
            ]
        },
    }
    spec = clarification_spec(manifest)
    assert spec["term"] == "active customers"
    assert "status" in spec["definition"]
    actions = [
        "get_schema",
        "ask_user",
        "get_knowledge_definition",
        "execute_sql",
        "execute_sql",
        "submit_solution",
    ]
    steps = [
        {
            "action": action,
            "thought_key": (
                "execute_sql_stale"
                if index == 3
                else "execute_sql_repaired"
                if index == 4
                else action
            ),
            "arguments": {"sql": "SELECT 1"} if action in {"execute_sql", "submit_solution"} else {},
            "observation": "ok",
        }
        for index, action in enumerate(actions)
    ]
    messages = build_five_tool_messages(
        question=manifest["question"], stale_sql=manifest["stale_sql"], steps=steps
    )
    assistant_messages = [message for message in messages if message["role"] == "assistant"]
    tool_messages = [message for message in messages if message["role"] == "tool"]
    assert len(assistant_messages) == 6
    assert len(tool_messages) == 5
    assert all(len(message["tool_calls"]) == 1 for message in assistant_messages)
    assert assistant_messages[1]["tool_calls"][0]["function"]["name"] == "ask_user"

    examples = expand_next_action_messages(messages)
    assert len(examples) == 6
    assert [example[-1]["tool_calls"][0]["function"]["name"] for example in examples] == actions
    assert all(example[-1]["role"] == "assistant" for example in examples)
    assert examples[0] == messages[:3]
    assert examples[-1] == messages

    plain = use_plain_json_for_last_action(examples[1])
    assert "tool_calls" not in plain[-1]
    assert '"name": "ask_user"' in plain[-1]["content"]
    assert plain[-3].get("tool_calls")
