"""Construction helpers for execution-verified five-tool SFT trajectories."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any

from driftsql.data.interactive import INTERACTIVE_SYSTEM_PROMPT


FIVE_TOOL_USER_TEMPLATE = """## Analytics request
{question}

## Previously valid cached SQL
{stale_sql}

The active database schema or business interpretation may have changed.
Use the interactive tools, validate the query, and submit the corrected SQL."""


def clarification_spec(manifest: dict[str, Any]) -> dict[str, Any]:
    """Derive a guarded clarification/HKB entry from public task metadata."""

    evidence = str(manifest.get("evidence", "")).strip()
    if evidence:
        segments = [segment.strip() for segment in evidence.split(";") if segment.strip()]
        matched = next(
            (
                match
                for segment in segments
                if (match := re.match(r"(.+?)\s+refers? to\s+(.+)", segment, re.IGNORECASE))
            ),
            None,
        )
        if matched and matched.group(1).strip():
            term = matched.group(1).strip(" `\"'")
        else:
            words = re.findall(r"[A-Za-z0-9_'-]+", str(manifest["question"]))
            term = " ".join(words[: min(6, len(words))]) or "requested business rule"
        definition = evidence
        ambiguity_type = "business_evidence_ambiguity"
    else:
        operation = dict(manifest["schema_diff"]["operations"][0])
        operation_type = str(operation["type"])
        if operation_type in {"rename_column", "replace_column"}:
            term = str(operation["old_name"])
            definition = (
                f"The active schema replaces `{operation['old_name']}` with "
                f"`{operation['new_name']}` in table `{operation['table']}`."
            )
        elif operation_type == "rename_table":
            term = str(operation["old_name"])
            definition = (
                f"The active schema uses table `{operation['new_name']}` instead of "
                f"`{operation['old_name']}`."
            )
        else:
            term = "result column contract"
            definition = (
                f"Return only the requested business columns; the new audit column "
                f"`{operation['new_name']}` is not part of the requested result."
            )
        ambiguity_type = "schema_policy_ambiguity"
    return {
        "term": term,
        "definition": definition,
        "ambiguity_type": ambiguity_type,
        "question": f"What does {term} mean for this request?",
        "knowledge_entry": {
            "knowledge": term,
            "description": "Task-specific business/schema definition used by the active database.",
            "definition": definition,
            "type": ambiguity_type,
        },
    }


def tool_call(name: str, arguments: dict[str, Any]) -> str:
    payload = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
    return f"<tool_call>{payload}</tool_call>"


def build_five_tool_messages(
    *,
    question: str,
    stale_sql: str,
    steps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": INTERACTIVE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": FIVE_TOOL_USER_TEMPLATE.format(question=question, stale_sql=stale_sql),
        },
    ]
    thoughts = {
        "get_schema": "I will retrieve the active schema before editing identifiers.",
        "ask_user": "I need one focused clarification instead of guessing the intended definition.",
        "get_knowledge_definition": "I will retrieve the governed definition for the clarified concept.",
        "execute_sql_stale": "I will execute the cached SQL to observe its active behavior.",
        "execute_sql_clean": "The cached SQL may still be valid, so I will verify it directly before submission.",
        "execute_sql_repaired": "I will execute the corrected SQL to validate it before submission.",
        "submit_clean": "The cached SQL remains valid after execution, so I will submit it unchanged.",
        "submit_solution": "The corrected SQL is validated, so I will submit it.",
    }
    for index, step in enumerate(steps):
        action = str(step["action"])
        thought_key = str(step.get("thought_key", action))
        messages.append(
            {
                "role": "assistant",
                "content": f"<think>{thoughts[thought_key]}</think>",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": action,
                            "arguments": json.dumps(step["arguments"], ensure_ascii=False),
                        },
                    }
                ],
            }
        )
        if action != "submit_solution" and index < len(steps) - 1:
            messages.append(
                {
                    "role": "tool",
                    "content": str(step["observation"]),
                }
            )
    return messages


def expand_next_action_messages(
    messages: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Expand one verified trajectory into next-action supervision examples.

    Each example ends at exactly one assistant action.  The preceding tool
    observations remain in the prompt, while the dataset loss mask supervises
    only the final assistant message.  This prevents a model from learning to
    emit the entire scripted trajectory in a single generation.
    """

    examples: list[list[dict[str, Any]]] = []
    for index, message in enumerate(messages):
        if message["role"] == "assistant":
            examples.append(messages[: index + 1])
    return examples


def use_plain_json_for_last_action(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Render only the target action as parser-compatible plain JSON.

    Previous assistant actions stay in Qwen's native structured form so the
    training history matches the live agent loop.  Plain JSON avoids relying
    on a poorly calibrated frozen special-token output embedding.
    """

    converted = deepcopy(messages)
    target = converted[-1]
    if target.get("role") != "assistant" or len(target.get("tool_calls", [])) != 1:
        raise ValueError("Expected the conversation prefix to end in one assistant tool call")
    function = target["tool_calls"][0]["function"]
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    payload = {"name": str(function["name"]), "arguments": arguments}
    target["content"] = f"{target.get('content', '').rstrip()}\n{json.dumps(payload, ensure_ascii=False)}"
    target.pop("tool_calls")
    return converted
