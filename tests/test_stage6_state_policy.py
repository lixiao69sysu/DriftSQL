from __future__ import annotations

import json

from driftsql.integrations.state_policy import (
    audited_repair_candidate,
    clarification_transition_response,
    duplicate_retrieval_response,
    dynamic_mask_response,
    hidden_tool_bad_words,
    is_exact_duplicate_retrieval,
    key_action_response_mask,
    schema_diff_recovery_guidance,
    select_dynamic_tool_names,
    select_dynamic_tool_schemas,
    should_apply_audited_repair,
)


def test_key_action_mask_keeps_action_suffixes_and_drops_rationale() -> None:
    original = [1] * 20 + [0] * 5 + [1] * 20
    events = [
        {
            "tool": "inspect_schema_diff",
            "model_token_start": 0,
            "model_token_end": 20,
            "metrics": {},
        },
        {
            "tool": "ask_user",
            "model_token_start": 25,
            "model_token_end": 45,
            "metrics": {},
        },
    ]
    selected = key_action_response_mask(original, events, suffix_tokens=6)
    assert sum(selected) == 6
    assert selected[:39] == [0] * 39
    assert selected[39:] == [1] * 6


def test_key_action_mask_falls_back_for_invalid_no_tool_output() -> None:
    assert key_action_response_mask([1] * 10, [], suffix_tokens=3) == [0] * 7 + [1] * 3


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
            {
                "role": "tool",
                "content": '{"success": true, "validated_for_submit": true}',
            },
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

    events.append(
        {
            "tool": "execute_sql",
            "response": '{"success":true,"validated_for_submit":true}',
        }
    )
    assert select_dynamic_tool_names(events, names) == {"submit_solution"}

    text, metrics = dynamic_mask_response("get_schema", {"submit_solution"})
    assert json.loads(text)["available_tools"] == ["submit_solution"]
    assert metrics["action_masked"] is True


def test_execution_only_policy_requires_schema_then_hides_submit_until_success() -> None:
    names = {"get_schema", "get_knowledge_definition", "execute_sql", "submit_solution"}
    assert select_dynamic_tool_names([], names, execution_only=True) == {"get_schema"}

    events = [{"tool": "get_schema", "response": '{"schema":"CREATE TABLE publishers (...)"}'}]
    available = select_dynamic_tool_names(events, names, execution_only=True)
    assert "get_schema" not in available
    assert "execute_sql" in available
    assert "submit_solution" not in available

    events.append({"tool": "execute_sql", "response": '{"success":false,"validated_for_submit":false}'})
    available = select_dynamic_tool_names(events, names, execution_only=True)
    assert "execute_sql" in available
    assert "submit_solution" not in available

    events.append({"tool": "execute_sql", "response": '{"success":true,"validated_for_submit":true}'})
    assert select_dynamic_tool_names(events, names, execution_only=True) == {"submit_solution"}


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


def test_contract_mismatch_forces_ordered_version_then_diff() -> None:
    names = {
        "get_schema_version",
        "inspect_schema_diff",
        "execute_sql",
        "submit_solution",
    }
    events = [
        {
            "tool": "execute_sql",
            "response": (
                '{"success":true,"result_contract_match":false,'
                '"validated_for_submit":false}'
            ),
        }
    ]
    assert select_dynamic_tool_names(events, names) == {"get_schema_version"}

    # An out-of-order diff does not satisfy the recovery protocol.
    events.append({"tool": "inspect_schema_diff", "response": '{"operations":[]}'})
    assert select_dynamic_tool_names(events, names) == {"get_schema_version"}

    events.append({"tool": "get_schema_version", "response": '{"db_version":"v2"}'})
    assert select_dynamic_tool_names(events, names) == {"inspect_schema_diff"}

    # The ordered diff is still available even though an earlier out-of-order
    # call used this normally one-shot tool.
    events.append({"tool": "inspect_schema_diff", "response": '{"operations":[]}'})
    available = select_dynamic_tool_names(events, names)
    assert "execute_sql" in available
    assert "submit_solution" in available


def test_failed_cached_sql_enters_the_same_ordered_recovery_state() -> None:
    names = {
        "get_schema_version",
        "inspect_schema_diff",
        "execute_sql",
        "submit_solution",
    }
    events = [
        {
            "tool": "execute_sql",
            "response": (
                '{"success":false,"error":"no such column: stale_name",'
                '"result_contract_match":null}'
            ),
        }
    ]
    assert select_dynamic_tool_names(events, names) == {"get_schema_version"}
    events.append({"tool": "get_schema_version", "response": '{"db_version":"v2"}'})
    assert select_dynamic_tool_names(events, names) == {"inspect_schema_diff"}


def test_executable_but_unvalidated_sql_is_never_forced_to_submit() -> None:
    names = {"inspect_schema_diff", "execute_sql", "submit_solution"}
    events = [
        {"tool": "inspect_schema_diff", "response": '{"operations":[]}'},
        {
            "tool": "execute_sql",
            "response": '{"success":true,"validated_for_submit":false}',
        },
    ]
    assert select_dynamic_tool_names(events, names) == {
        "execute_sql",
        "submit_solution",
    }


def test_prompt_schema_mask_matches_contract_recovery_state() -> None:
    schemas = _schemas(
        "get_schema_version",
        "inspect_schema_diff",
        "execute_sql",
        "submit_solution",
    )
    messages = [
        {"role": "assistant", "tool_calls": [{"function": {"name": "execute_sql"}}]},
        {
            "role": "tool",
            "content": '{"success":true,"result_contract_match":false}',
        },
    ]
    selected = select_dynamic_tool_schemas(messages, schemas)
    assert [row["function"]["name"] for row in selected] == ["get_schema_version"]

    messages.extend(
        [
            {
                "role": "assistant",
                "tool_calls": [{"function": {"name": "get_schema_version"}}],
            },
            {"role": "tool", "content": '{"db_version":"v2"}'},
        ]
    )
    selected = select_dynamic_tool_schemas(messages, schemas)
    assert [row["function"]["name"] for row in selected] == ["inspect_schema_diff"]


def test_ask_user_is_one_shot_and_requires_a_resolution_action() -> None:
    names = {
        "get_schema_version",
        "inspect_schema_diff",
        "ask_user",
        "get_knowledge_definition",
        "execute_sql",
        "submit_solution",
    }
    events = [{"tool": "ask_user", "response": "clarified"}]
    assert select_dynamic_tool_names(events, names) == {
        "get_knowledge_definition",
        "execute_sql",
    }

    events.append(
        {
            "tool": "get_schema_version",
            "observation": '{"action_masked":true}',
            "metrics": {"action_masked": True},
        }
    )
    assert select_dynamic_tool_names(events, names) == {
        "get_knowledge_definition",
        "execute_sql",
    }

    events.append(
        {"tool": "get_knowledge_definition", "response": '{"matches":[]}'},
    )
    available = select_dynamic_tool_names(events, names)
    assert "ask_user" not in available
    assert "get_knowledge_definition" not in available
    assert "execute_sql" in available


def test_prompt_mask_matches_post_clarification_dispatch_policy() -> None:
    schemas = _schemas(
        "get_schema_version",
        "inspect_schema_diff",
        "ask_user",
        "get_knowledge_definition",
        "execute_sql",
        "submit_solution",
    )
    messages = [
        {"role": "assistant", "tool_calls": [{"function": {"name": "ask_user"}}]},
        {"role": "tool", "content": "clarified"},
    ]
    assert {
        row["function"]["name"] for row in select_dynamic_tool_schemas(messages, schemas)
    } == {"get_knowledge_definition", "execute_sql"}

    messages.extend(
        [
            {
                "role": "assistant",
                "tool_calls": [{"function": {"name": "get_knowledge_definition"}}],
            },
            {"role": "tool", "content": '{"matches":[]}'},
        ]
    )
    available = {
        row["function"]["name"] for row in select_dynamic_tool_schemas(messages, schemas)
    }
    assert "ask_user" not in available
    assert "get_knowledge_definition" not in available
    assert "execute_sql" in available


def test_knowledge_first_clarification_routing_matches_prompt_and_dispatch() -> None:
    names = {
        "ask_user",
        "get_knowledge_definition",
        "execute_sql",
        "submit_solution",
    }
    events = [{"tool": "ask_user", "response": "clarified"}]
    assert select_dynamic_tool_names(
        events,
        names,
        knowledge_first_after_ask=True,
    ) == {"get_knowledge_definition"}

    schemas = _schemas(*sorted(names))
    messages = [
        {"role": "assistant", "tool_calls": [{"function": {"name": "ask_user"}}]},
        {"role": "tool", "content": "clarified"},
    ]
    assert {
        row["function"]["name"]
        for row in select_dynamic_tool_schemas(
            messages,
            schemas,
            knowledge_first_after_ask=True,
        )
    } == {"get_knowledge_definition"}

    text, metrics = clarification_transition_response(
        "Use the governed definition.",
        "get_knowledge_definition",
    )
    assert json.loads(text)["controller_state"]["required_next_action"] == (
        "get_knowledge_definition"
    )
    assert metrics["clarification_transition"] is True
    assert hidden_tool_bad_words(
        {"get_knowledge_definition"},
        names,
    ) == ["ask_user", "execute_sql", "submit_solution"]


def test_audited_repair_only_replaces_an_exact_stale_query_after_diff() -> None:
    stale = "SELECT T2.city FROM stores AS T2 WHERE T2.city = 'city'"
    repaired = "SELECT T2.municipality FROM stores AS T2 WHERE T2.municipality = 'city'"
    events = [
        {
            "tool": "inspect_schema_diff",
            "response": json.dumps(
                {
                    "repair_candidate": {
                        "original_sql": stale,
                        "repaired_sql": repaired,
                        "changed": True,
                    }
                }
            ),
        }
    ]
    assert audited_repair_candidate(events) == repaired
    assert should_apply_audited_repair(events, stale + ";", stale) == repaired
    assert should_apply_audited_repair(events, "SELECT city FROM stores LIMIT 1", stale) == ""
    assert should_apply_audited_repair([], stale, stale) == ""
