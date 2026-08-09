"""Execution-grounded shaped reward for five-tool DriftSQL trajectories."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from sqlglot import exp, parse_one

from driftsql.drift import fingerprint_query, materialize_schema_diff
from driftsql.tool_calls import extract_tool_call_dicts


ALLOWED_TOOLS = {
    "ask_user",
    "get_schema",
    "get_knowledge_definition",
    "get_schema_version",
    "inspect_schema_diff",
    "execute_sql",
    "submit_solution",
}


def extract_tool_calls(solution: str) -> list[dict[str, Any]]:
    return extract_tool_call_dicts(solution)


def _arguments(call: dict[str, Any]) -> dict[str, Any]:
    arguments = call.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return {}
    return arguments if isinstance(arguments, dict) else {}


def _argument_sql(call: dict[str, Any]) -> str:
    return str(_arguments(call).get("sql", "")).strip()


def _normalise_sql(sql: str) -> str:
    return sql.rstrip(";").strip().casefold()


def _is_read_query(sql: str) -> tuple[bool, str]:
    if not sql:
        return False, "missing_sql"
    candidate = sql.lstrip()
    explain = re.match(
        r"(?is)^EXPLAIN\s+(?:(?:QUERY\s+PLAN|ANALYZE)\s+)?(.+)$",
        candidate,
    )
    if explain:
        try:
            explained = parse_one(explain.group(1), read="sqlite")
        except Exception as error:
            return False, f"sql_parse_error: {error}"
        if isinstance(explained, (exp.Query, exp.Subquery)):
            return True, ""
        return False, "non_read_query"
    if candidate.upper().startswith("PRAGMA"):
        read_pragma = re.match(
            r"(?is)^PRAGMA\s+(?:[A-Za-z_][A-Za-z0-9_]*\.)?"
            r"(?:table_info|table_xinfo|foreign_key_list|index_list|index_info|"
            r"database_list|compile_options)\b",
            candidate,
        )
        return (True, "") if read_pragma else (False, "non_read_query")
    try:
        expression = parse_one(sql, read="sqlite")
    except Exception as error:
        return False, f"sql_parse_error: {error}"
    if not isinstance(expression, (exp.Query, exp.Subquery)):
        return False, "non_read_query"
    return True, ""


def _events(extra_info: dict[str, Any]) -> list[dict[str, Any]]:
    raw = extra_info.get("environment_events", [])
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if not isinstance(raw, (list, tuple)):
        return []
    return [dict(item) for item in raw if isinstance(item, dict)]


def _ambiguity_terms(extra_info: dict[str, Any]) -> list[str]:
    tools_kwargs = extra_info.get("tools_kwargs", {}) or {}
    for tool_name in ("ask_user", "get_schema", "get_knowledge_definition"):
        create_kwargs = (tools_kwargs.get(tool_name, {}) or {}).get("create_kwargs", {}) or {}
        ambiguity = create_kwargs.get("user_query_ambiguity", {}) or {}
        terms = [
            str(item.get("term", "")).strip().casefold()
            for key in ("critical_ambiguity", "non_critical_ambiguity")
            for item in (ambiguity.get(key, []) or [])
            if str(item.get("term", "")).strip()
        ]
        if terms:
            return terms
    return []


def _fallback_clarification(calls: list[dict[str, Any]], extra_info: dict[str, Any]) -> bool:
    terms = _ambiguity_terms(extra_info)
    if not terms:
        return False
    for call in calls:
        if call.get("name") != "ask_user":
            continue
        question = str(_arguments(call).get("question", "")).casefold()
        if any(term in question for term in terms):
            return True
    return False


def _empty_result(**updates: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        # Keep the source identity in rollout artifacts.  VERL writes every
        # reward-extra field alongside the generated trajectory, which lets
        # the Stage-5 failure miner join a failed rollout back to the exact
        # training record without brittle prompt or SQL matching.
        "instance_id": "",
        "coverage_index": -1,
        "reward_version": "v1",
        "score": 0.0,
        "format_valid": False,
        "execution_success": False,
        "task_success": False,
        "inspected_drift": False,
        "clarification_matched": False,
        "schema_retrieved": False,
        "knowledge_retrieved": False,
        "tested_solution": False,
        "clarification_required": False,
        "clarification_attempted": False,
        "post_clarification_valid": False,
        "terminal_validated": False,
        "add_column_inspected": False,
        "ordered_drift_inspection": False,
        "candidate_task_success": False,
        "add_column_candidate_validated": False,
        "add_column_protocol_complete": False,
        "protocol_success": False,
        "decision_target_action": "",
        "decision_action": "",
        "decision_action_correct": False,
        "premature_stale_execute": False,
        "efficient": False,
        "tool_calls": 0,
        "sql_executions": 0,
        "response_tokens": 0,
        "key_action_mask_tokens": 0,
        "advantage_scope": "",
        "episode_response_mask_tokens": 0,
        "advantage_mask_tokens": 0,
        "duplicate_questions": 0,
        "duplicate_executions": 0,
        "excess_clarifications": 0,
        "excess_retrievals": 0,
        "invalid_actions": 0,
        "invalid_sql": 0,
        "timed_out": False,
        "turn_limit": False,
        "missing_submit": False,
        "unsafe": False,
        "reward_success": 0.0,
        "reward_clarify": 0.0,
        "reward_valid": 0.0,
        "reward_efficient": 0.0,
        "reward_required_clarification": 0.0,
        "reward_clarification_attempt": 0.0,
        "reward_post_clarification": 0.0,
        "reward_terminal": 0.0,
        "reward_add_column_inspect": 0.0,
        "reward_semantic_candidate": 0.0,
        "reward_add_column": 0.0,
        "reward_decision_action": 0.0,
        "penalty_tool_cost": 0.0,
        "penalty_token_cost": 0.0,
        "penalty_duplicate": 0.0,
        "penalty_repeated_tool": 0.0,
        "penalty_invalid": 0.0,
        "penalty_timeout": 0.0,
        "penalty_turn_limit": 0.0,
        "penalty_missing_submit": 0.0,
        "penalty_unsafe": 0.0,
        "penalty_missing_required_clarification": 0.0,
        "penalty_invalid_post_clarification": 0.0,
        "penalty_unmatched_clarification": 0.0,
        "penalty_add_column_protocol": 0.0,
        "penalty_decision_action": 0.0,
        "penalty_premature_stale_execute": 0.0,
        "error": "",
    }
    result.update(updates)
    result["score"] = round(float(result["score"]), 4)
    return result


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any],
    *,
    success_weight: float = 1.0,
    clarify_weight: float = 0.2,
    valid_weight: float = 0.1,
    efficient_weight: float = 0.1,
    tool_call_cost: float = 0.01,
    token_cost: float = 0.00001,
    duplicate_penalty: float = 0.05,
    repeated_tool_penalty: float = 0.05,
    invalid_penalty: float = 0.05,
    timeout_penalty: float = 0.2,
    turn_limit_penalty: float = 0.3,
    missing_submit_penalty: float = 0.2,
    unsafe_penalty: float = 1.0,
    required_clarification_weight: float = 0.0,
    clarification_attempt_weight: float = 0.0,
    post_clarification_weight: float = 0.0,
    terminal_weight: float = 0.0,
    add_column_inspect_weight: float = 0.0,
    semantic_candidate_weight: float = 0.0,
    add_column_weight: float = 0.0,
    decision_action_weight: float = 0.0,
    decision_action_mismatch_penalty: float = 0.0,
    premature_stale_execute_penalty: float = 0.0,
    missing_required_clarification_penalty: float = 0.0,
    invalid_post_clarification_penalty: float = 0.0,
    unmatched_clarification_penalty: float = 0.0,
    add_column_protocol_penalty: float = 0.0,
    efficient_tool_calls: int = 6,
    reward_version: str = "v1",
    **kwargs: Any,
) -> dict[str, Any]:
    """Score one trajectory using live tool metrics and database execution.

    ``environment_events`` is emitted by :class:`DriftToolAgentLoop` and passed
    to custom rewards by VERL as a tool extra field.  Re-parsing the calls is a
    deliberate fallback for unit tests and old rollout artifacts.
    """

    del data_source, ground_truth, kwargs
    if reward_version not in {"v1", "v2", "v3"}:
        raise ValueError(f"Unsupported reward_version: {reward_version}")
    reward_v2 = reward_version == "v2"
    reward_v3 = reward_version == "v3"
    execution_grounded_shaping = reward_v2 or reward_v3
    extra_info = dict(extra_info or {})
    calls = extract_tool_calls(solution_str)
    events = _events(extra_info)
    names = [str(call.get("name", "")) for call in calls]
    event_metrics = [dict(event.get("metrics", {}) or {}) for event in events]
    if events:
        unmasked_events = [
            event
            for event in events
            if not bool((event.get("metrics", {}) or {}).get("action_masked"))
        ]
        unmasked_event_names = [
            str(event.get("tool", event.get("tool_name", "")))
            for event in unmasked_events
        ]
        unmasked_action_arguments = [
            _arguments({"arguments": event.get("arguments", {})})
            for event in unmasked_events
        ]
        inspected_drift = {"get_schema_version", "inspect_schema_diff"}.issubset(
            unmasked_event_names
        )
    else:
        unmasked_events = []
        unmasked_event_names = names
        unmasked_action_arguments = [_arguments(call) for call in calls]
        inspected_drift = {"get_schema_version", "inspect_schema_diff"}.issubset(names)

    version_index = -1
    diff_index = -1
    try:
        version_index = unmasked_event_names.index("get_schema_version")
        diff_index = unmasked_event_names.index("inspect_schema_diff")
        ordered_drift_inspection = version_index < diff_index
    except ValueError:
        ordered_drift_inspection = False

    decision_target_action = str(extra_info.get("decision_target_action", ""))
    decision_action = unmasked_event_names[0] if unmasked_event_names else ""
    decision_action_correct = bool(
        decision_target_action and decision_action == decision_target_action
    )

    ask_calls = [call for call in calls if call.get("name") == "ask_user"]
    execute_calls = [call for call in calls if call.get("name") == "execute_sql"]
    submit_calls = [call for call in calls if call.get("name") == "submit_solution"]
    submitted_sql = _argument_sql(submit_calls[-1]) if submit_calls else ""

    clarification_matched = any(
        metrics.get("clarification_matched") and not metrics.get("duplicate_question")
        for metrics in event_metrics
    ) or _fallback_clarification(calls, extra_info)
    schema_retrieved = any(metrics.get("schema_retrieved") for metrics in event_metrics)
    if not events:
        schema_retrieved = "get_schema" in names
    knowledge_retrieved = any(metrics.get("knowledge_retrieved") for metrics in event_metrics)
    if not events:
        knowledge_retrieved = "get_knowledge_definition" in names

    duplicate_questions = sum(
        bool(metrics.get("duplicate_question")) for metrics in event_metrics
    )
    if not events:
        question_counts = Counter(
            str(_arguments(call).get("question", "")).strip().casefold()
            for call in ask_calls
        )
        duplicate_questions = sum(max(0, count - 1) for count in question_counts.values())

    execution_sql = [_argument_sql(call) for call in execute_calls]
    execution_counts = Counter(_normalise_sql(sql) for sql in execution_sql if sql)
    duplicate_executions = sum(max(0, count - 1) for count in execution_counts.values())
    excess_clarifications = max(0, len(ask_calls) - 1)
    excess_retrievals = max(0, names.count("get_schema") - 1) + max(
        0, names.count("get_knowledge_definition") - 1
    )
    tested_solution = bool(submitted_sql) and _normalise_sql(submitted_sql) in {
        _normalise_sql(sql) for sql in execution_sql if sql
    }

    interaction_profile = str(extra_info.get("interaction_profile", ""))
    drift_type = str(extra_info.get("drift_type", ""))
    clarification_required = interaction_profile == "must_ask"

    # The verified P6 contract permits exactly one stale-query execution as the
    # first observation. Silent add-column drift makes that wildcard query
    # executable, so the actual shortcut is submitting it immediately (or
    # repeating it) without the required version/diff inspection. Do not
    # penalise the canonical first probe itself.
    premature_stale_execute = False
    stale_sql = str(extra_info.get("stale_sql", "")).strip()
    if drift_type == "add_column" and stale_sql:
        stale_normalized = _normalise_sql(stale_sql)
        stale_execute_indices = [
            index
            for index, (name, action_arguments) in enumerate(
                zip(unmasked_event_names, unmasked_action_arguments, strict=True)
            )
            if name == "execute_sql"
            and _normalise_sql(str(action_arguments.get("sql", "")))
            == stale_normalized
        ]
        stale_submit_indices = [
            index
            for index, (name, action_arguments) in enumerate(
                zip(unmasked_event_names, unmasked_action_arguments, strict=True)
            )
            if name == "submit_solution"
            and _normalise_sql(str(action_arguments.get("sql", "")))
            == stale_normalized
        ]
        ordered_inspection_end = diff_index if version_index < diff_index else -1
        stale_submit_without_inspection = any(
            ordered_inspection_end < 0 or submit_index < ordered_inspection_end
            for submit_index in stale_submit_indices
        )
        repeated_stale_before_inspection = sum(
            ordered_inspection_end < 0 or execute_index < ordered_inspection_end
            for execute_index in stale_execute_indices
        ) > 1
        premature_stale_execute = bool(
            stale_submit_without_inspection or repeated_stale_before_inspection
        )

    # Reward the state transition after a real, unmasked clarification rather
    # than the lexical presence of an ask_user call.  Old rollout artifacts do
    # not have environment events, so retain a conservative call-order fallback.
    event_names = unmasked_event_names
    try:
        ask_index = event_names.index("ask_user")
    except ValueError:
        ask_index = -1
    clarification_attempted = ask_index >= 0
    post_clarification_valid = bool(
        ask_index >= 0
        and ask_index + 1 < len(event_names)
        and event_names[ask_index + 1]
        in {"get_knowledge_definition", "execute_sql"}
    )

    invalid_actions = sum(name not in ALLOWED_TOOLS for name in names)
    invalid_sql = 0
    unsafe = False
    parse_errors: list[str] = []
    for sql in execution_sql + ([submitted_sql] if submitted_sql else []):
        read_only, error = _is_read_query(sql)
        if not read_only:
            invalid_sql += 1
            parse_errors.append(error)
            unsafe = unsafe or error == "non_read_query"
    unsafe = unsafe or any(
        "not authorized" in str(metrics.get("execution_error", "")).casefold()
        for metrics in event_metrics
    )

    event_execution_success = any(
        bool(metrics.get("execution_success")) for metrics in event_metrics
    )
    event_errors = [
        str(metrics.get("execution_error", ""))
        for metrics in event_metrics
        if metrics.get("execution_error")
    ]
    timed_out = bool(
        extra_info.get("trajectory_timed_out", extra_info.get("timed_out", False))
    ) or any(
        "interrupt" in error.casefold() or "timeout" in error.casefold()
        for error in event_errors
    )
    turn_limit = bool(extra_info.get("trajectory_turn_limit", False))

    format_valid = bool(submit_calls and submitted_sql)
    final_read_only, final_error = _is_read_query(submitted_sql) if submitted_sql else (False, "missing_submit_solution")
    format_valid = format_valid and final_read_only
    missing_submit = not format_valid
    execution_success = False
    candidate_task_success = False
    correct_candidate_sql: set[str] = set()
    candidate_events: list[tuple[int, str]] = []
    if events:
        for index, event in enumerate(unmasked_events):
            if str(event.get("tool", event.get("tool_name", ""))) != "execute_sql":
                continue
            arguments = event.get("arguments", {}) or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            sql = str(arguments.get("sql", "")).strip() if isinstance(arguments, dict) else ""
            if sql:
                candidate_events.append((index, sql))
    else:
        candidate_events = [
            (index, _argument_sql(call))
            for index, call in enumerate(calls)
            if call.get("name") == "execute_sql" and _argument_sql(call)
        ]
    task_success = False
    error = final_error if not final_read_only else ""

    source_db = Path(str(extra_info.get("source_db", ""))).resolve()
    schema_diff = extra_info.get("schema_diff")
    expected = extra_info.get("result_fingerprint", {}) or {}
    environment_ready = bool(source_db.is_file() and schema_diff and expected)
    if environment_ready and (final_read_only or candidate_events):
        try:
            with tempfile.TemporaryDirectory(
                prefix="driftsql-reward-",
                dir=os.environ.get("DRIFTSQL_TMPDIR"),
                ignore_cleanup_errors=True,
            ) as temp_dir:
                database = Path(temp_dir) / f"{extra_info.get('db_id', 'db')}__v2.sqlite"
                materialize_schema_diff(source_db, database, schema_diff)
                timeout_seconds = float(os.environ.get("DRIFTSQL_REWARD_TIMEOUT", "30"))
                if final_read_only:
                    try:
                        predicted = fingerprint_query(
                            database,
                            submitted_sql,
                            timeout_seconds=timeout_seconds,
                        )
                        execution_success = True
                        task_success = (
                            predicted.row_count == int(expected.get("row_count", -1))
                            and predicted.value_hash == str(expected.get("value_hash", ""))
                        )
                    except (OSError, sqlite3.Error, ValueError, NotImplementedError) as caught:
                        error = f"execution_error: {caught}"
                        timed_out = timed_out or "interrupt" in str(caught).casefold()
                if execution_grounded_shaping and task_success and any(
                    _normalise_sql(candidate_sql) == _normalise_sql(submitted_sql)
                    for _index, candidate_sql in candidate_events
                ):
                    candidate_task_success = True
                    correct_candidate_sql.add(_normalise_sql(submitted_sql))
                for _index, candidate_sql in candidate_events if execution_grounded_shaping else []:
                    if candidate_task_success:
                        break
                    read_only, _candidate_error = _is_read_query(candidate_sql)
                    if not read_only:
                        continue
                    if final_read_only and _normalise_sql(candidate_sql) == _normalise_sql(
                        submitted_sql
                    ):
                        if task_success:
                            candidate_task_success = True
                            correct_candidate_sql.add(_normalise_sql(candidate_sql))
                        continue
                    try:
                        candidate = fingerprint_query(
                            database,
                            candidate_sql,
                            timeout_seconds=timeout_seconds,
                        )
                    except (OSError, sqlite3.Error, ValueError, NotImplementedError):
                        continue
                    if (
                        candidate.row_count == int(expected.get("row_count", -1))
                        and candidate.value_hash == str(expected.get("value_hash", ""))
                    ):
                        candidate_task_success = True
                        correct_candidate_sql.add(_normalise_sql(candidate_sql))
        except (OSError, sqlite3.Error, ValueError, NotImplementedError) as caught:
            error = f"execution_error: {caught}"
            timed_out = timed_out or "interrupt" in str(caught).casefold()
    elif final_read_only:
        error = "missing_environment_metadata"

    add_column_candidate_validated = bool(
        drift_type == "add_column"
        and ordered_drift_inspection
        and any(
            index > diff_index and _normalise_sql(sql) in correct_candidate_sql
            for index, sql in candidate_events
        )
    )

    response_tokens = int(
        extra_info.get("response_tokens", extra_info.get("response_len", 0)) or 0
    )
    key_action_mask_tokens = int(extra_info.get("key_action_mask_tokens", 0) or 0)
    advantage_scope = str(extra_info.get("advantage_scope", ""))
    episode_response_mask_tokens = int(
        extra_info.get("episode_response_mask_tokens", 0) or 0
    )
    advantage_mask_tokens = int(extra_info.get("advantage_mask_tokens", 0) or 0)
    efficient = bool(
        task_success
        and len(calls) <= efficient_tool_calls
        and duplicate_questions == 0
        and duplicate_executions == 0
    )

    # Clarification is useful only if the trajectory eventually submits.  This
    # prevents a correctly phrased question followed by six wasted actions
    # from outscoring a concise failed submission.
    reward_clarify = clarify_weight if clarification_matched and format_valid else 0.0
    if execution_grounded_shaping:
        reward_valid = 0.0
    elif execution_success:
        reward_valid = valid_weight
    elif event_execution_success:
        reward_valid = valid_weight * 0.5
    else:
        reward_valid = 0.0
    terminal_validated = bool(format_valid and tested_solution)
    add_column_inspected = bool(
        drift_type == "add_column"
        and (ordered_drift_inspection if execution_grounded_shaping else inspected_drift)
    )
    if execution_grounded_shaping:
        add_column_protocol_complete = bool(
            add_column_inspected
            and add_column_candidate_validated
            and terminal_validated
            and task_success
        )
    else:
        add_column_protocol_complete = bool(add_column_inspected and terminal_validated)
    drift_protocol_complete = bool(
        drift_type in {"", "clean"} or ordered_drift_inspection
    )
    clarification_protocol_complete = bool(
        not clarification_required
        or (clarification_matched and post_clarification_valid)
    )
    protocol_success = bool(
        task_success
        and terminal_validated
        and drift_protocol_complete
        and clarification_protocol_complete
        and (drift_type != "add_column" or add_column_protocol_complete)
    )
    reward_success = (
        success_weight if (protocol_success if reward_v3 else task_success) else 0.0
    )
    reward_efficient = (
        efficient_weight
        if efficient and (protocol_success if reward_v3 else True)
        else 0.0
    )
    reward_required_clarification = (
        required_clarification_weight
        if clarification_required and clarification_matched
        else 0.0
    )
    reward_clarification_attempt = (
        clarification_attempt_weight
        if not execution_grounded_shaping
        and clarification_required
        and clarification_attempted
        else 0.0
    )
    reward_post_clarification = (
        post_clarification_weight
        if clarification_required
        and post_clarification_valid
        and (clarification_matched if execution_grounded_shaping else True)
        else 0.0
    )
    reward_terminal = (
        terminal_weight
        if terminal_validated and (task_success if execution_grounded_shaping else True)
        else 0.0
    )
    reward_add_column_inspect = (
        add_column_inspect_weight if add_column_inspected else 0.0
    )
    reward_semantic_candidate = (
        semantic_candidate_weight
        if execution_grounded_shaping and candidate_task_success
        else 0.0
    )
    reward_add_column = add_column_weight if add_column_protocol_complete else 0.0
    reward_decision_action = (
        decision_action_weight if decision_action_correct else 0.0
    )
    penalty_tool_cost = tool_call_cost * len(calls)
    penalty_token_cost = min(0.1, token_cost * response_tokens)
    penalty_duplicate = duplicate_penalty * (duplicate_questions + duplicate_executions)
    penalty_repeated_tool = repeated_tool_penalty * (
        excess_clarifications + excess_retrievals
    )
    penalty_invalid = invalid_penalty * (invalid_actions + invalid_sql)
    penalty_timeout_value = timeout_penalty if timed_out else 0.0
    penalty_turn_limit_value = turn_limit_penalty if turn_limit else 0.0
    penalty_missing_submit_value = missing_submit_penalty if missing_submit else 0.0
    penalty_unsafe_value = unsafe_penalty if unsafe else 0.0
    penalty_missing_required_clarification = (
        missing_required_clarification_penalty
        if clarification_required and not clarification_attempted
        else 0.0
    )
    penalty_unmatched_clarification = (
        unmatched_clarification_penalty
        if clarification_required
        and clarification_attempted
        and not clarification_matched
        else 0.0
    )
    penalty_invalid_post_clarification = (
        invalid_post_clarification_penalty
        if clarification_required
        and clarification_matched
        and not post_clarification_valid
        else 0.0
    )
    penalty_add_column_protocol = (
        add_column_protocol_penalty
        if drift_type == "add_column" and not add_column_protocol_complete
        else 0.0
    )
    penalty_decision_action = (
        decision_action_mismatch_penalty
        if decision_target_action and not decision_action_correct
        else 0.0
    )
    penalty_premature_stale_execute = (
        premature_stale_execute_penalty if premature_stale_execute else 0.0
    )
    score = (
        reward_success
        + reward_clarify
        + reward_valid
        + reward_efficient
        + reward_required_clarification
        + reward_clarification_attempt
        + reward_post_clarification
        + reward_terminal
        + reward_add_column_inspect
        + reward_semantic_candidate
        + reward_add_column
        + reward_decision_action
        - penalty_tool_cost
        - penalty_token_cost
        - penalty_duplicate
        - penalty_repeated_tool
        - penalty_invalid
        - penalty_timeout_value
        - penalty_turn_limit_value
        - penalty_missing_submit_value
        - penalty_unsafe_value
        - penalty_missing_required_clarification
        - penalty_invalid_post_clarification
        - penalty_unmatched_clarification
        - penalty_add_column_protocol
        - penalty_decision_action
        - penalty_premature_stale_execute
    )

    return _empty_result(
        instance_id=str(extra_info.get("instance_id", "")),
        coverage_index=int(extra_info.get("index", -1)),
        reward_version=reward_version,
        score=score,
        format_valid=format_valid,
        execution_success=execution_success,
        task_success=task_success,
        inspected_drift=inspected_drift,
        clarification_matched=clarification_matched,
        schema_retrieved=schema_retrieved,
        knowledge_retrieved=knowledge_retrieved,
        tested_solution=tested_solution,
        clarification_required=clarification_required,
        clarification_attempted=clarification_attempted,
        post_clarification_valid=post_clarification_valid,
        terminal_validated=terminal_validated,
        add_column_inspected=add_column_inspected,
        ordered_drift_inspection=ordered_drift_inspection,
        candidate_task_success=candidate_task_success,
        add_column_candidate_validated=add_column_candidate_validated,
        add_column_protocol_complete=add_column_protocol_complete,
        protocol_success=protocol_success,
        decision_target_action=decision_target_action,
        decision_action=decision_action,
        decision_action_correct=decision_action_correct,
        premature_stale_execute=premature_stale_execute,
        efficient=efficient,
        tool_calls=len(calls),
        sql_executions=len(execute_calls),
        response_tokens=response_tokens,
        key_action_mask_tokens=key_action_mask_tokens,
        advantage_scope=advantage_scope,
        episode_response_mask_tokens=episode_response_mask_tokens,
        advantage_mask_tokens=advantage_mask_tokens,
        duplicate_questions=duplicate_questions,
        duplicate_executions=duplicate_executions,
        excess_clarifications=excess_clarifications,
        excess_retrievals=excess_retrievals,
        invalid_actions=invalid_actions,
        invalid_sql=invalid_sql,
        timed_out=timed_out,
        turn_limit=turn_limit,
        missing_submit=missing_submit,
        unsafe=unsafe,
        reward_success=round(reward_success, 4),
        reward_clarify=round(reward_clarify, 4),
        reward_valid=round(reward_valid, 4),
        reward_efficient=round(reward_efficient, 4),
        reward_required_clarification=round(reward_required_clarification, 4),
        reward_clarification_attempt=round(reward_clarification_attempt, 4),
        reward_post_clarification=round(reward_post_clarification, 4),
        reward_terminal=round(reward_terminal, 4),
        reward_add_column_inspect=round(reward_add_column_inspect, 4),
        reward_semantic_candidate=round(reward_semantic_candidate, 4),
        reward_add_column=round(reward_add_column, 4),
        reward_decision_action=round(reward_decision_action, 4),
        penalty_tool_cost=round(penalty_tool_cost, 4),
        penalty_token_cost=round(penalty_token_cost, 4),
        penalty_duplicate=round(penalty_duplicate, 4),
        penalty_repeated_tool=round(penalty_repeated_tool, 4),
        penalty_invalid=round(penalty_invalid, 4),
        penalty_timeout=round(penalty_timeout_value, 4),
        penalty_turn_limit=round(penalty_turn_limit_value, 4),
        penalty_missing_submit=round(penalty_missing_submit_value, 4),
        penalty_unsafe=round(penalty_unsafe_value, 4),
        penalty_missing_required_clarification=round(
            penalty_missing_required_clarification, 4
        ),
        penalty_invalid_post_clarification=round(
            penalty_invalid_post_clarification, 4
        ),
        penalty_unmatched_clarification=round(penalty_unmatched_clarification, 4),
        penalty_add_column_protocol=round(penalty_add_column_protocol, 4),
        penalty_decision_action=round(penalty_decision_action, 4),
        penalty_premature_stale_execute=round(
            penalty_premature_stale_execute, 4
        ),
        error=error or (parse_errors[-1] if parse_errors else ""),
    )
