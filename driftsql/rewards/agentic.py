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
        "score": 0.0,
        "format_valid": False,
        "execution_success": False,
        "task_success": False,
        "inspected_drift": False,
        "clarification_matched": False,
        "schema_retrieved": False,
        "knowledge_retrieved": False,
        "tested_solution": False,
        "efficient": False,
        "tool_calls": 0,
        "sql_executions": 0,
        "response_tokens": 0,
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
        "penalty_tool_cost": 0.0,
        "penalty_token_cost": 0.0,
        "penalty_duplicate": 0.0,
        "penalty_repeated_tool": 0.0,
        "penalty_invalid": 0.0,
        "penalty_timeout": 0.0,
        "penalty_turn_limit": 0.0,
        "penalty_missing_submit": 0.0,
        "penalty_unsafe": 0.0,
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
    efficient_tool_calls: int = 6,
    **kwargs: Any,
) -> dict[str, Any]:
    """Score one trajectory using live tool metrics and database execution.

    ``environment_events`` is emitted by :class:`DriftToolAgentLoop` and passed
    to custom rewards by VERL as a tool extra field.  Re-parsing the calls is a
    deliberate fallback for unit tests and old rollout artifacts.
    """

    del data_source, ground_truth, kwargs
    extra_info = dict(extra_info or {})
    calls = extract_tool_calls(solution_str)
    events = _events(extra_info)
    names = [str(call.get("name", "")) for call in calls]
    inspected_drift = {"get_schema_version", "inspect_schema_diff"}.issubset(names)
    event_metrics = [dict(event.get("metrics", {}) or {}) for event in events]

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
    task_success = False
    error = final_error if not final_read_only else ""

    source_db = Path(str(extra_info.get("source_db", ""))).resolve()
    schema_diff = extra_info.get("schema_diff")
    expected = extra_info.get("result_fingerprint", {}) or {}
    if final_read_only and source_db.is_file() and schema_diff and expected:
        try:
            with tempfile.TemporaryDirectory(
                prefix="driftsql-reward-",
                dir=os.environ.get("DRIFTSQL_TMPDIR"),
                ignore_cleanup_errors=True,
            ) as temp_dir:
                database = Path(temp_dir) / f"{extra_info.get('db_id', 'db')}__v2.sqlite"
                materialize_schema_diff(source_db, database, schema_diff)
                predicted = fingerprint_query(
                    database,
                    submitted_sql,
                    timeout_seconds=float(os.environ.get("DRIFTSQL_REWARD_TIMEOUT", "30")),
                )
            execution_success = True
            task_success = (
                predicted.row_count == int(expected.get("row_count", -1))
                and predicted.value_hash == str(expected.get("value_hash", ""))
            )
        except (OSError, sqlite3.Error, ValueError, NotImplementedError) as caught:
            error = f"execution_error: {caught}"
            timed_out = timed_out or "interrupt" in str(caught).casefold()
    elif final_read_only:
        error = "missing_environment_metadata"

    response_tokens = int(
        extra_info.get("response_tokens", extra_info.get("response_len", 0)) or 0
    )
    efficient = bool(
        task_success
        and len(calls) <= efficient_tool_calls
        and duplicate_questions == 0
        and duplicate_executions == 0
    )

    reward_success = success_weight if task_success else 0.0
    # Clarification is useful only if the trajectory eventually submits.  This
    # prevents a correctly phrased question followed by six wasted actions
    # from outscoring a concise failed submission.
    reward_clarify = clarify_weight if clarification_matched and format_valid else 0.0
    if execution_success:
        reward_valid = valid_weight
    elif event_execution_success:
        reward_valid = valid_weight * 0.5
    else:
        reward_valid = 0.0
    reward_efficient = efficient_weight if efficient else 0.0
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
    score = (
        reward_success
        + reward_clarify
        + reward_valid
        + reward_efficient
        - penalty_tool_cost
        - penalty_token_cost
        - penalty_duplicate
        - penalty_repeated_tool
        - penalty_invalid
        - penalty_timeout_value
        - penalty_turn_limit_value
        - penalty_missing_submit_value
        - penalty_unsafe_value
    )

    return _empty_result(
        instance_id=str(extra_info.get("instance_id", "")),
        score=score,
        format_valid=format_valid,
        execution_success=execution_success,
        task_success=task_success,
        inspected_drift=inspected_drift,
        clarification_matched=clarification_matched,
        schema_retrieved=schema_retrieved,
        knowledge_retrieved=knowledge_retrieved,
        tested_solution=tested_solution,
        efficient=efficient,
        tool_calls=len(calls),
        sql_executions=len(execute_calls),
        response_tokens=response_tokens,
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
        penalty_tool_cost=round(penalty_tool_cost, 4),
        penalty_token_cost=round(penalty_token_cost, 4),
        penalty_duplicate=round(penalty_duplicate, 4),
        penalty_repeated_tool=round(penalty_repeated_tool, 4),
        penalty_invalid=round(penalty_invalid, 4),
        penalty_timeout=round(penalty_timeout_value, 4),
        penalty_turn_limit=round(penalty_turn_limit_value, 4),
        penalty_missing_submit=round(penalty_missing_submit_value, 4),
        penalty_unsafe=round(penalty_unsafe_value, 4),
        error=error or (parse_errors[-1] if parse_errors else ""),
    )
