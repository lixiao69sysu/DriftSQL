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
ONE_SHOT_TOOLS = RETRIEVAL_TOOLS | {"ask_user"}
POST_CLARIFICATION_TOOLS = {"get_knowledge_definition", "execute_sql"}


def _event_name(event: dict[str, Any]) -> str:
    return str(event.get("tool", event.get("tool_name", "")))


def _event_was_masked(event: dict[str, Any]) -> bool:
    metrics = event.get("metrics", {})
    if isinstance(metrics, dict) and bool(metrics.get("action_masked")):
        return True
    try:
        observation = json.loads(str(event.get("response", event.get("observation", ""))))
    except json.JSONDecodeError:
        observation = {}
    return bool(observation.get("action_masked"))


def _event_observation(event: dict[str, Any]) -> dict[str, Any]:
    value = event.get("response", event.get("observation", ""))
    if isinstance(value, dict):
        return value
    try:
        observation = json.loads(str(value))
    except (json.JSONDecodeError, TypeError):
        return {}
    return observation if isinstance(observation, dict) else {}


def _required_contract_recovery_action(
    completed_events: list[tuple[int, str, dict[str, Any]]],
) -> str:
    """Return the next mandatory action after an observed contract mismatch.

    A mismatch that happens before a completed version/diff recovery sequence
    must be followed by ``get_schema_version`` and then
    ``inspect_schema_diff``.  A mismatch from a later repair attempt is not
    sent through the same one-shot retrievals again; the policy remains free
    to revise and re-execute its candidate.
    """

    valid = [
        (index, name, observation)
        for index, name, observation in completed_events
        if not bool(observation.get("action_masked"))
    ]
    recovery_trigger_positions = [
        index
        for index, name, observation in valid
        if name == "execute_sql"
        and (
            observation.get("success") is False
            or observation.get("result_contract_match") is False
        )
    ]
    for mismatch_index in reversed(recovery_trigger_positions):
        versions_before = [
            index
            for index, name, _ in valid
            if name == "get_schema_version" and index < mismatch_index
        ]
        already_recovered = any(
            version_index < diff_index < mismatch_index
            for version_index in versions_before
            for diff_index, name, _ in valid
            if name == "inspect_schema_diff"
        )
        if already_recovered:
            continue

        versions_after = [
            index
            for index, name, _ in valid
            if name == "get_schema_version" and index > mismatch_index
        ]
        if not versions_after:
            return "get_schema_version"
        first_version = min(versions_after)
        has_ordered_diff = any(
            index > first_version and name == "inspect_schema_diff"
            for index, name, _ in valid
        )
        if not has_ordered_diff:
            return "inspect_schema_diff"
    return ""


def _has_validated_execution(
    completed_events: list[tuple[int, str, dict[str, Any]]],
) -> bool:
    """Return true only when the newest valid execution is safe to submit."""

    executions = [
        observation
        for _, name, observation in completed_events
        if name == "execute_sql" and not bool(observation.get("action_masked"))
    ]
    return bool(executions and executions[-1].get("validated_for_submit") is True)


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


def clarification_transition_response(
    user_response: str,
    required_next_action: str,
) -> tuple[str, dict[str, Any]]:
    if required_next_action not in POST_CLARIFICATION_TOOLS:
        raise ValueError(required_next_action)
    payload = {
        "user_response": user_response,
        "controller_state": {
            "clarification_complete": True,
            "ask_user_available": False,
            "required_next_action": required_next_action,
        },
    }
    return json.dumps(payload, ensure_ascii=False), {
        "clarification_transition": True,
        "required_next_action": required_next_action,
        "state_guarded": True,
    }


def post_clarification_tools(
    available: set[str],
    *,
    knowledge_first: bool,
) -> set[str]:
    allowed = available & POST_CLARIFICATION_TOOLS
    if not knowledge_first:
        return allowed
    if "get_knowledge_definition" in allowed:
        return {"get_knowledge_definition"}
    return allowed & {"execute_sql"}


def hidden_tool_bad_words(available: set[str], enabled: set[str]) -> list[str]:
    """Return deterministic lexical bans for generation-time action masking."""

    return sorted(enabled - available)


def audited_repair_candidate(events: list[dict[str, Any]]) -> str:
    """Return the newest changed repair candidate from an audited diff.

    The value is observation-derived and therefore contains no ground-truth
    SQL.  Empty, malformed, unchanged, or masked observations are ignored.
    """

    for event in reversed(events):
        if _event_name(event) != "inspect_schema_diff" or _event_was_masked(event):
            continue
        try:
            observation = json.loads(
                str(event.get("response", event.get("observation", "")))
            )
        except (json.JSONDecodeError, TypeError):
            continue
        candidate = observation.get("repair_candidate", {})
        if not isinstance(candidate, dict) or not bool(candidate.get("changed")):
            continue
        repaired_sql = str(candidate.get("repaired_sql", "")).strip()
        if repaired_sql:
            return repaired_sql
    return ""


def should_apply_audited_repair(
    events: list[dict[str, Any]],
    proposed_sql: str,
    stale_sql: str,
) -> str:
    """Return a repair only when the model repeats the exact cached query."""

    def normalize(value: str) -> str:
        return value.rstrip(";").strip().casefold()

    if not normalize(proposed_sql) or normalize(proposed_sql) != normalize(stale_sql):
        return ""
    candidate = audited_repair_candidate(events)
    if candidate and normalize(candidate) != normalize(proposed_sql):
        return candidate
    return ""


def select_dynamic_tool_names(
    events: list[dict[str, Any]],
    tool_names: set[str] | list[str] | tuple[str, ...],
    *,
    knowledge_first_after_ask: bool = False,
    execution_only: bool = False,
) -> set[str]:
    """Return the tools allowed by the current trajectory state.

    Unlike schema rendering, this helper consumes the compact event format
    emitted by the live VERL loop, so it can also enforce the policy at tool
    dispatch time when earlier prompt tokens cannot be rewritten.
    """

    available = set(tool_names)
    if execution_only:
        completed = [
            (_event_name(event), _event_observation(event))
            for event in events
            if not _event_was_masked(event)
        ]
        schema_retrieved = any(
            name == "get_schema" and observation.get("schema")
            for name, observation in completed
        )
        if not schema_retrieved and "get_schema" in available:
            return {"get_schema"}
        called_one_shot = {name for name, _ in completed if name in ONE_SHOT_TOOLS}
        available.difference_update(called_one_shot)
        if _has_validated_execution(
            [(index, name, observation) for index, (name, observation) in enumerate(completed)]
        ):
            return {"submit_solution"} if "submit_solution" in available else set()
        available.discard("submit_solution")
        return available
    completed_events = [
        (index, _event_name(event), _event_observation(event))
        for index, event in enumerate(events)
    ]
    called_one_shot = {
        _event_name(event)
        for event in events
        if _event_name(event) in ONE_SHOT_TOOLS and not _event_was_masked(event)
    }
    available.difference_update(called_one_shot)

    required_recovery_action = _required_contract_recovery_action(completed_events)
    if required_recovery_action in set(tool_names):
        return {required_recovery_action}

    if _has_validated_execution(completed_events) and "submit_solution" in available:
        return {"submit_solution"}

    last_ask = max(
        (
            index
            for index, event in enumerate(events)
            if _event_name(event) == "ask_user" and not _event_was_masked(event)
        ),
        default=-1,
    )
    clarification_followed_up = any(
        index > last_ask
        and _event_name(event) in POST_CLARIFICATION_TOOLS
        and not _event_was_masked(event)
        for index, event in enumerate(events)
    )
    if last_ask >= 0 and not clarification_followed_up:
        available.intersection_update(
            post_clarification_tools(
                available,
                knowledge_first=knowledge_first_after_ask,
            )
        )

    return available


def key_action_response_mask(
    response_mask: list[int],
    events: list[dict[str, Any]],
    *,
    suffix_tokens: int = 96,
) -> list[int]:
    """Keep loss only on target action suffixes for Agentic RL.

    Tool observations already have zeroes in ``response_mask``.  Each live
    event records the generated assistant-token span that selected it.  We
    retain the suffix of clarification, post-diff, execution, and terminal
    actions so trajectory reward updates the actual decision/arguments rather
    than hundreds of low-value rationale tokens.  Invalid/no-tool responses
    fall back to their final generated suffix so negative reward still trains
    a corrective signal.
    """

    if suffix_tokens <= 0:
        raise ValueError("suffix_tokens must be positive")
    output = [0] * len(response_mask)
    saw_diff = False
    for event in events:
        name = _event_name(event)
        masked = _event_was_masked(event)
        is_post_diff = saw_diff
        if name == "inspect_schema_diff" and not masked:
            saw_diff = True
        key_action = (
            name in {"ask_user", "submit_solution"}
            or (is_post_diff and not masked)
            or (name == "execute_sql" and not masked)
        )
        if not key_action:
            continue
        start = max(0, int(event.get("model_token_start", 0)))
        end = min(len(response_mask), int(event.get("model_token_end", 0)))
        start = max(start, end - suffix_tokens)
        for index in range(start, end):
            if response_mask[index]:
                output[index] = 1
    if any(output):
        return output
    active = [index for index, value in enumerate(response_mask) if value]
    for index in active[-suffix_tokens:]:
        output[index] = 1
    return output


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
    *,
    knowledge_first_after_ask: bool = False,
) -> list[dict[str, Any]]:
    """Mask one-shot retrievals and force submit after a validated repair.

    This is action masking, not action generation: the model must still emit
    the SQL and choose ``submit_solution`` itself.
    """

    completed_events: list[tuple[int, str, dict[str, Any]]] = []
    pending_tool = ""
    for index, message in enumerate(messages):
        if message.get("role") == "assistant":
            pending_tool = _message_tool_name(message)
        elif message.get("role") == "tool" and pending_tool:
            try:
                observation = json.loads(str(message.get("content", "")))
            except json.JSONDecodeError:
                observation = {}
            completed_events.append((index, pending_tool, observation))
            pending_tool = ""

    names = {
        str(schema.get("function", {}).get("name", "")): schema
        for schema in schemas
    }
    available = set(names)
    valid_called = {
        name
        for _, name, observation in completed_events
        if not bool(observation.get("action_masked"))
    }
    available.difference_update(ONE_SHOT_TOOLS & valid_called)

    required_recovery_action = _required_contract_recovery_action(completed_events)
    if required_recovery_action in names:
        available = {required_recovery_action}
        return [
            schema
            for schema in schemas
            if str(schema.get("function", {}).get("name", "")) in available
        ]

    if _has_validated_execution(completed_events) and "submit_solution" in names:
        available = {"submit_solution"}
        return [
            schema
            for schema in schemas
            if str(schema.get("function", {}).get("name", "")) in available
        ]

    last_ask = max(
        (
            index
            for index, name, observation in completed_events
            if name == "ask_user" and not bool(observation.get("action_masked"))
        ),
        default=-1,
    )
    clarification_followed_up = any(
        index > last_ask
        and name in POST_CLARIFICATION_TOOLS
        and not bool(observation.get("action_masked"))
        for index, name, observation in completed_events
    )
    if last_ask >= 0 and not clarification_followed_up:
        available.intersection_update(
            post_clarification_tools(
                available,
                knowledge_first=knowledge_first_after_ask,
            )
        )

    return [schema for schema in schemas if str(schema.get("function", {}).get("name", "")) in available]
