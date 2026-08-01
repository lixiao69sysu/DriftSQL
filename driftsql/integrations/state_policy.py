"""Pure state-policy helpers shared by live and offline Stage 6 agent loops."""

from __future__ import annotations

import json
from typing import Any


RETRIEVAL_TOOLS = {
    "get_schema_version",
    "inspect_schema_diff",
    "get_schema",
    "get_knowledge_definition",
}


def schema_diff_recovery_guidance(schema_diff: dict[str, Any]) -> list[str]:
    """Translate audited drift operations into generic recovery constraints.

    The guidance deliberately describes *how to preserve the old query
    contract* without generating a candidate SQL statement.  This keeps the
    policy model responsible for the actual repair while making subtle drift
    semantics, especially ``SELECT *`` after an added column, explicit.
    """

    guidance: list[str] = []
    for operation in schema_diff.get("operations", []) or []:
        if not isinstance(operation, dict):
            continue
        drift_type = str(operation.get("type", ""))
        table = str(operation.get("table") or "the affected table")
        old_name = str(operation.get("old_name") or "")
        new_name = str(operation.get("new_name") or "")
        if drift_type == "add_column" and new_name:
            guidance.append(
                f"Column '{new_name}' was added to {table}. Preserve the cached result-column "
                "contract: if the cached SQL uses SELECT * or alias.*, expand the wildcard to "
                f"the pre-change columns and exclude '{new_name}', unless the analytics request "
                "explicitly requires the newly added column."
            )
        elif drift_type == "rename_table" and old_name and new_name:
            guidance.append(
                f"Replace table identifier '{old_name}' with audited name '{new_name}'; "
                "preserve the rest of the query semantics."
            )
        elif drift_type in {"rename_column", "replace_column"} and old_name and new_name:
            guidance.append(
                f"Replace column identifier '{old_name}' with audited column '{new_name}' "
                f"on {table}; preserve aliases, predicates, grouping, and output intent."
            )
        elif drift_type == "metric_definition_change":
            metric_name = str(operation.get("metric_name") or "the requested metric")
            old_expression = str(operation.get("old_expression") or "")
            new_expression = str(operation.get("new_expression") or "")
            reason = str(operation.get("reason") or "the active business definition changed")
            clarification = bool(operation.get("requires_clarification"))
            guidance.append(
                f"Metric '{metric_name}' changed from '{old_expression}' to '{new_expression}' "
                f"because {reason}. Use the active definition"
                + (" after confirming the requested metric version with the user." if clarification else ".")
            )
    return guidance


def canonical_arguments(arguments: dict[str, Any]) -> str:
    return json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def is_exact_duplicate_retrieval(
    events: list[dict[str, Any]],
    tool_name: str,
    arguments: dict[str, Any],
) -> bool:
    """Return true only when the identical retrieval already has an observation."""

    if tool_name not in RETRIEVAL_TOOLS:
        return False
    target = canonical_arguments(arguments)
    for event in events:
        prior_name = str(event.get("tool", event.get("tool_name", "")))
        prior_arguments = event.get("arguments", {})
        if (
            prior_name == tool_name
            and isinstance(prior_arguments, dict)
            and canonical_arguments(prior_arguments) == target
            and not event.get("error")
        ):
            return True
    return False


def duplicate_retrieval_response(tool_name: str) -> tuple[str, dict[str, Any]]:
    payload = {
        "duplicate_retrieval": True,
        "tool": tool_name,
        "message": (
            "This exact retrieval result is already present in the conversation. "
            "Do not retrieve it again; use the existing evidence to execute_sql, "
            "or submit_solution if the candidate already executed successfully."
        ),
    }
    return json.dumps(payload, ensure_ascii=False), {
        "duplicate_retrieval": True,
        "state_guarded": True,
    }


def select_dynamic_tool_names(
    events: list[dict[str, Any]],
    tool_names: set[str] | list[str] | tuple[str, ...],
) -> set[str]:
    """Return the tools allowed by the current trajectory state.

    Unlike schema rendering, this helper consumes the compact event format
    emitted by the live VERL loop, so it can also enforce the policy at tool
    dispatch time when earlier prompt tokens cannot be rewritten.
    """

    available = set(tool_names)
    called_retrievals = {
        str(event.get("tool", event.get("tool_name", "")))
        for event in events
        if str(event.get("tool", event.get("tool_name", ""))) in RETRIEVAL_TOOLS
    }
    available.difference_update(called_retrievals)

    last_diff = max(
        (
            index
            for index, event in enumerate(events)
            if str(event.get("tool", event.get("tool_name", ""))) == "inspect_schema_diff"
        ),
        default=-1,
    )
    repaired_success = False
    if last_diff >= 0:
        for index, event in enumerate(events):
            if index <= last_diff or str(event.get("tool", event.get("tool_name", ""))) != "execute_sql":
                continue
            try:
                observation = json.loads(str(event.get("response", event.get("observation", ""))))
            except json.JSONDecodeError:
                observation = {}
            if bool(observation.get("success")):
                repaired_success = True
                break
    if repaired_success and "submit_solution" in available:
        return {"submit_solution"}
    return available


def dynamic_mask_response(tool_name: str, available: set[str]) -> tuple[str, dict[str, Any]]:
    payload = {
        "action_masked": True,
        "tool": tool_name,
        "available_tools": sorted(available),
        "message": (
            "This tool is no longer valid in the current trajectory state. "
            "Use one of the currently available tools and do not repeat completed retrievals."
        ),
    }
    return json.dumps(payload, ensure_ascii=False), {
        "action_masked": True,
        "state_guarded": True,
    }


def _message_tool_name(message: dict[str, Any]) -> str:
    calls = message.get("tool_calls", [])
    if len(calls) != 1:
        return ""
    return str(calls[0].get("function", {}).get("name", ""))


def select_dynamic_tool_schemas(
    messages: list[dict[str, Any]],
    schemas: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Mask one-shot retrievals and force submit after a validated repair.

    This is action masking, not action generation: the model must still emit
    the SQL and choose ``submit_solution`` itself.
    """

    called: list[str] = []
    execution_events: list[tuple[int, bool]] = []
    pending_tool = ""
    for index, message in enumerate(messages):
        if message.get("role") == "assistant":
            pending_tool = _message_tool_name(message)
            if pending_tool:
                called.append(pending_tool)
        elif message.get("role") == "tool" and pending_tool:
            if pending_tool == "execute_sql":
                try:
                    observation = json.loads(str(message.get("content", "")))
                except json.JSONDecodeError:
                    observation = {}
                execution_events.append((index, bool(observation.get("success"))))
            pending_tool = ""

    names = {
        str(schema.get("function", {}).get("name", "")): schema
        for schema in schemas
    }
    available = set(names)
    available.difference_update(RETRIEVAL_TOOLS & set(called))

    # Once an audited diff is in context, a later successful execution is the
    # validated repair candidate.  A production agent should submit it rather
    # than spend the remaining budget on unrelated retrievals.
    diff_positions = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "assistant"
        and _message_tool_name(message) == "inspect_schema_diff"
    ]
    last_diff = max(diff_positions, default=-1)
    repaired_success = any(index > last_diff and success for index, success in execution_events) if last_diff >= 0 else False
    if repaired_success and "submit_solution" in names:
        available = {"submit_solution"}

    return [schema for schema in schemas if str(schema.get("function", {}).get("name", "")) in available]
