from scripts.mine_p6_scaleup_failures import classify_failure


def event(name: str, *, success: bool = False) -> dict:
    return {
        "tool_name": name,
        "observation": {"success": success},
        "arguments": {},
    }


def failed_row(**overrides) -> dict:
    row = {
        "task_success": False,
        "scenario_type": "atomic",
        "interaction_profile": "schema_only",
        "termination_reason": "turn_limit",
        "trajectory": [],
        "safety": {"unsafe": False, "unsafe_actions": 0},
    }
    row.update(overrides)
    return row


def test_successful_post_diff_execute_without_submit_is_terminal_failure() -> None:
    result = classify_failure(
        failed_row(
            trajectory=[
                event("inspect_schema_diff"),
                event("execute_sql", success=True),
            ]
        )
    )
    assert result["primary_failure"] == "successful_execute_no_submit"
    assert result["terminal"]["post_diff_successful_execute_no_submit"] is True


def test_post_diff_wrong_retrieval_and_compound_are_multilabel() -> None:
    result = classify_failure(
        failed_row(
            scenario_type="compound",
            interaction_profile="knowledge_only",
            trajectory=[event("inspect_schema_diff"), event("get_schema")],
        )
    )
    assert result["primary_failure"] == "post_diff_wrong_retrieval"
    assert "compound_recovery" in result["labels"]


def test_must_ask_omission_and_alias_are_recorded() -> None:
    result = classify_failure(
        failed_row(
            interaction_profile="must_ask",
            trajectory=[event("inspect_schema_diff"), event("ask-user")],
        )
    )
    assert "must_ask_error" in result["labels"]
    assert "required_ask_omitted" in result["must_ask"]["subtypes"]
    assert "malformed_ask_tool" in result["must_ask"]["subtypes"]
    assert "ask-user" in result["invalid_tools"]
