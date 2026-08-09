"""Parsing utilities for tool calls emitted in common JSON wrappers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

_EMPTY_FENCE_PATTERN = re.compile(r"```(?:json)?\s*```", re.IGNORECASE)
_TOOL_TAG_PATTERN = re.compile(r"</?tool_call>", re.IGNORECASE)
_FUNCTION_TAG_PATTERN = re.compile(
    r"<function\b(?P<attributes>.*?)/\s*>",
    re.IGNORECASE | re.DOTALL,
)
_FUNCTION_NAME_PATTERN = re.compile(
    r"\bname\s*=\s*([\"'])(?P<name>[^\"']+)\1",
    re.IGNORECASE,
)
_FUNCTION_ARGUMENTS_PREFIX = re.compile(
    r"\barguments\s*=\s*([\"'])",
    re.IGNORECASE,
)


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


def _function_tag_call(match: re.Match[str]) -> ParsedToolCall | None:
    """Parse the native ``<function name=... arguments=... />`` variant.

    Qwen occasionally places JSON arguments inside a single-quoted attribute.
    The JSON string itself can legitimately contain apostrophes, so the closing
    delimiter must be the final matching quote in the tag rather than the first
    one encountered by a conventional attribute regex.
    """

    attributes = match.group("attributes")
    name_match = _FUNCTION_NAME_PATTERN.search(attributes)
    arguments_match = _FUNCTION_ARGUMENTS_PREFIX.search(attributes)
    if name_match is None or arguments_match is None:
        return None
    quote = arguments_match.group(1)
    arguments_start = arguments_match.end()
    arguments_end = attributes.rfind(quote)
    if arguments_end < arguments_start:
        return None
    try:
        arguments = json.loads(attributes[arguments_start:arguments_end])
    except json.JSONDecodeError:
        return None
    if not isinstance(arguments, dict):
        return None
    return ParsedToolCall(
        name=name_match.group("name").strip(),
        arguments=arguments,
        start=match.start(),
        end=match.end(),
    )


def find_tool_calls(text: str) -> list[ParsedToolCall]:
    """Find tool-call JSON objects whether tagged, fenced, or bare.

    The scanner deliberately accepts only objects with a tool ``name`` and an
    object-valued ``arguments`` field. JSON observations returned by tools are
    therefore ignored when a complete multi-turn trajectory is scanned.
    """

    decoder = json.JSONDecoder()
    calls: list[ParsedToolCall] = []
    function_spans: list[tuple[int, int]] = []
    for match in _FUNCTION_TAG_PATTERN.finditer(text):
        function_spans.append((match.start(), match.end()))
        call = _function_tag_call(match)
        if call is not None:
            calls.append(call)
    cursor = 0
    while cursor < len(text):
        start = text.find("{", cursor)
        if start < 0:
            break
        containing_tag = next(
            ((span_start, span_end) for span_start, span_end in function_spans if span_start <= start < span_end),
            None,
        )
        if containing_tag is not None:
            cursor = containing_tag[1]
            continue
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
    return sorted(calls, key=lambda call: call.start)


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
