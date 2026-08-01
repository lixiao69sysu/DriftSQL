"""Qwen2.5 JSON compatibility for BIRD-RL's stateful agent loop."""

from __future__ import annotations

import json

from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl_patch.tool_agent_loop_with_db_cleanup import ToolAgentLoopWithDBCleanup

from driftsql.tool_calls import find_tool_calls, remove_tool_call_payloads


@ToolParser.register("bird-json-compat")
class BirdJsonCompatToolParser(ToolParser):
    """Accept BIRD's tagged JSON plus fenced or bare Qwen JSON calls."""

    async def extract_tool_calls(self, responses_ids, tools=None):
        del tools
        text = self.tokenizer.decode(responses_ids)
        parsed = find_tool_calls(text)
        calls = [
            FunctionCall(name=call.name, arguments=json.dumps(call.arguments, ensure_ascii=False))
            for call in parsed
        ]
        return remove_tool_call_payloads(text, parsed), calls


class BirdJsonCompatAgentLoop(ToolAgentLoopWithDBCleanup):
    """BIRD-RL's original cleanup loop with only parser registration changed."""
