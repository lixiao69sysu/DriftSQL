from scripts.build_p6_on_policy_recovery_sft import verified_terminal_recovery


def assistant(name: str, sql: str = "") -> dict:
    arguments = {"sql": sql} if sql else {}
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"type": "function", "function": {"name": name, "arguments": arguments}}
        ],
    }


def test_verified_terminal_requires_post_diff_canonical_sql() -> None:
    canonical = [
        assistant("execute_sql", "SELECT old FROM t"),
        assistant("inspect_schema_diff"),
        assistant("execute_sql", "SELECT new FROM t"),
        assistant("submit_solution", "SELECT new FROM t"),
    ]
    rollout = {
        "trajectory": [
            {
                "tool_name": "execute_sql",
                "arguments": {"sql": "SELECT old FROM t"},
                "metrics": {"execution_success": True},
            },
            {"tool_name": "inspect_schema_diff", "arguments": {}, "metrics": {}},
            {
                "tool_name": "execute_sql",
                "arguments": {"sql": "SELECT new FROM t"},
                "metrics": {"execution_success": True},
            },
        ]
    }
    assert verified_terminal_recovery(rollout, canonical) == (2, 3)


def test_stale_success_before_diff_is_not_safe_to_submit() -> None:
    canonical = [
        assistant("execute_sql", "SELECT old FROM t"),
        assistant("inspect_schema_diff"),
        assistant("execute_sql", "SELECT new FROM t"),
        assistant("submit_solution", "SELECT new FROM t"),
    ]
    rollout = {
        "trajectory": [
            {
                "tool_name": "execute_sql",
                "arguments": {"sql": "SELECT old FROM t"},
                "metrics": {"execution_success": True},
            }
        ]
    }
    assert verified_terminal_recovery(rollout, canonical) is None
