from __future__ import annotations

import json

from driftsql.integrations.state_policy import (
    duplicate_retrieval_response,
    dynamic_mask_response,
    is_exact_duplicate_retrieval,
    schema_diff_recovery_guidance,
    select_dynamic_tool_names,
    select_dynamic_tool_schemas,
)


def test_exact_duplicate_retrieval_is_guarded() -> None:
    events = [{"tool_name": "get_schema", "arguments": {"query": "orders"}}]
    assert is_exact_duplicate_retrieval(events, "get_schema", {"query": "orders"})
    assert not is_exact_duplicate_retrieval(events, "get_schema", {"query": "customers"})
    assert not is_exact_duplicate_retrieval(events, "execute_sql", {"sql": "select 1"})


def test_live_event_shape_and_response_are_auditable() -> None:
    events = [{"tool": "inspect_schema_diff", "arguments": {}}]
    assert is_exact_duplicate_retrieval(events, "inspect_schema_diff", {})
    text, metrics = duplicate_retrieval_response("inspect_schema_diff")
    assert json.loads(text)["duplicate_retrieval"] is True
    assert metrics == {"duplicate_retrieval": True, "state_guarded": True}


def _schemas(*names: str) -> list[dict]:
    return [{"type": "function", "function": {"name": name}} for name in names]


def test_dynamic_mask_removes_used_retrieval_and_forces_submit_after_repair() -> None:
    schemas = _schemas("get_schema_version", "inspect_schema_diff", "get_schema", "execute_sql", "submit_solution")
    messages = [
        {"role": "assistant", "tool_calls": [{"function": {"name": "get_schema_version"}}]},
        {"role": "tool", "content": "{}"},
    ]
    selected = select_dynamic_tool_schemas(messages, schemas)
    assert "get_schema_version" not in {row["function"]["name"] for row in selected}

    messages.extend(
        [
            {"role": "assistant", "tool_calls": [{"function": {"name": "inspect_schema_diff"}}]},
            {"role": "tool", "content": "{}"},
            {"role": "assistant", "tool_calls": [{"function": {"name": "execute_sql"}}]},
            {"role": "tool", "content": '{"success": true}'},
        ]
    )
    selected = select_dynamic_tool_schemas(messages, schemas)
    assert [row["function"]["name"] for row in selected] == ["submit_solution"]


def test_add_column_guidance_preserves_wildcard_result_contract() -> None:
    guidance = schema_diff_recovery_guidance(
        {
            "operations": [
                {
                    "type": "add_column",
                    "table": "orders",
                    "new_name": "audit_flag",
                }
            ]
        }
    )
    assert len(guidance) == 1
    assert "SELECT *" in guidance[0]
    assert "exclude 'audit_flag'" in guidance[0]


def test_rename_guidance_uses_only_audited_identifiers() -> None:
    guidance = schema_diff_recovery_guidance(
        {
            "operations": [
                {
                    "type": "rename_column",
                    "table": "orders",
                    "old_name": "amount",
                    "new_name": "net_amount",
                }
            ]
        }
    )
    assert guidance == [
        "Replace column identifier 'amount' with audited column 'net_amount' on orders; "
        "preserve aliases, predicates, grouping, and output intent."
    ]


def test_live_dynamic_names_remove_retrieval_and_force_submit() -> None:
    names = {
        "get_schema_version",
        "inspect_schema_diff",
        "get_schema",
        "execute_sql",
        "submit_solution",
    }
    events = [
        {"tool": "get_schema_version", "response": '{"db_version":"v2"}'},
        {"tool": "inspect_schema_diff", "response": '{"operations":[]}'},
    ]
    available = select_dynamic_tool_names(events, names)
    assert "get_schema_version" not in available
    assert "inspect_schema_diff" not in available
    assert "execute_sql" in available

    events.append({"tool": "execute_sql", "response": '{"success":true}'})
    assert select_dynamic_tool_names(events, names) == {"submit_solution"}

    text, metrics = dynamic_mask_response("get_schema", {"submit_solution"})
    assert json.loads(text)["available_tools"] == ["submit_solution"]
    assert metrics["action_masked"] is True


def test_successful_stale_execution_before_diff_does_not_force_submit() -> None:
    names = {
        "get_schema_version",
        "inspect_schema_diff",
        "execute_sql",
        "submit_solution",
    }
    events = [{"tool": "execute_sql", "response": '{"success":true}'}]
    available = select_dynamic_tool_names(events, names)
    assert available == names
