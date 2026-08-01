"""Normalize Qwen JSON calls before delegating to BIRD-RL's reward."""

from __future__ import annotations

import json
from typing import Any

from bird_rl.rewards.critic_reward_agentic import compute_score as bird_compute_score

from driftsql.tool_calls import find_tool_calls


def _append_tagged_calls(solution_str: str) -> str:
    calls = find_tool_calls(solution_str or "")
    if not calls:
        return solution_str
    normalized = [
        "<tool_call>"
        + json.dumps(call.as_dict(), ensure_ascii=False)
        + "</tool_call>"
        for call in calls
    ]
    return (solution_str or "") + "\n" + "\n".join(normalized)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict,
    **kwargs,
) -> dict:
    return bird_compute_score(
        data_source=data_source,
        solution_str=_append_tagged_calls(solution_str),
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )
