"""Parsing utilities for tool calls emitted in common JSON wrappers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

_EMPTY_FENCE_PATTERN = re.compile(r"```(?:json)?\s*```", re.IGNORECASE)
_TOOL_TAG_PATTERN = re.compile(r"</?tool_call>", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedToolCall:
    """A normalized tool call and its character span in model output."""

    name: str
    arguments: dict[str, Any]
    start: int
    end: int

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments}


def _normalize_payload(payload: Any) -> tuple[str, dict[str, Any]] | None:
    if not isinstance(payload, dict):
        return None

    function = payload.get("function")
    if isinstance(function, dict):
        payload = function

    name = payload.get("name")
    arguments = payload.get("arguments", {})
    if not isinstance(name, str) or not name.strip():
        return None
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        return None
    return name.strip(), arguments


def find_tool_calls(text: str) -> list[ParsedToolCall]:
    """Find tool-call JSON objects whether tagged, fenced, or bare.

    The scanner deliberately accepts only objects with a tool ``name`` and an
    object-valued ``arguments`` field. JSON observations returned by tools are
    therefore ignored when a complete multi-turn trajectory is scanned.
    """

    decoder = json.JSONDecoder()
    calls: list[ParsedToolCall] = []
    cursor = 0
    while cursor < len(text):
        start = text.find("{", cursor)
        if start < 0:
            break
        try:
            payload, consumed = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            cursor = start + 1
            continue

        end = start + consumed
        normalized = _normalize_payload(payload)
        if normalized is not None:
            name, arguments = normalized
            calls.append(
                ParsedToolCall(
                    name=name,
                    arguments=arguments,
                    start=start,
                    end=end,
                )
            )
        cursor = max(end, start + 1)
    return calls


def extract_tool_call_dicts(text: str) -> list[dict[str, Any]]:
    return [call.as_dict() for call in find_tool_calls(text or "")]


def remove_tool_call_payloads(text: str, calls: list[ParsedToolCall]) -> str:
    """Remove parsed JSON payloads and now-empty wrappers from assistant text."""

    if not calls:
        return text
    chunks: list[str] = []
    cursor = 0
    for call in calls:
        chunks.append(text[cursor : call.start])
        cursor = call.end
    chunks.append(text[cursor:])
    content = "".join(chunks)
    content = _TOOL_TAG_PATTERN.sub("", content)
    content = _EMPTY_FENCE_PATTERN.sub("", content)
    return content.strip()
