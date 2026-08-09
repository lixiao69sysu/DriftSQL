from __future__ import annotations

import asyncio
import json
from pathlib import Path

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput
from verl.experimental.agent_loop.tool_agent_loop import AgentData, ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.tools.schemas import OpenAIFunctionToolSchema

from driftsql.integrations.agent_loop import DriftToolAgentLoop
from driftsql.integrations.verl_tools import AskUserTool


class FakeTokenizer:
    pad_token_id = 0

    def decode(self, token_ids) -> str:
        return "decoded:" + ",".join(str(value) for value in token_ids)


def test_text_only_tool_response_can_be_truncated_without_null_media() -> None:
    tool_schema = OpenAIFunctionToolSchema.model_validate(
        {
            "type": "function",
            "function": {
                "name": "ask_user",
                "description": "clarify",
                "parameters": {
                    "type": "object",
                    "properties": {"question": {"type": "string"}},
                    "required": ["question"],
                },
            },
        }
    )
    ask = AskUserTool({"max_questions": 1}, tool_schema)
    state = {
        "db_id": "truncate_db",
        "user_query_ambiguity": {
            "critical_ambiguity": [
                {"term": "active", "sql_snippet": "x = 1", "type": "condition"}
            ],
            "non_critical_ambiguity": [],
        }
    }
    tools_kwargs = {"ask_user": {"create_kwargs": state}}
    loop = object.__new__(DriftToolAgentLoop)
    loop.tools = {"ask_user": ask}
    loop.max_tool_response_length = 16
    loop.tool_response_truncate_side = "right"
    agent_data = AgentData(
        messages=[],
        image_data=[],
        video_data=[],
        audio_data=None,
        mm_processor_kwargs=None,
        metrics={},
        request_id="truncate-response",
        tools_kwargs=tools_kwargs,
    )
    response, _, _ = asyncio.run(
        loop._call_tool(
            FunctionCall(name="ask_user", arguments='{"question":"What is active?"}'),
            tools_kwargs,
            agent_data,
        )
    )
    assert response.text.endswith("...(truncated)")
    assert response.image is None
    assert response.video is None


def test_agent_loop_reuses_state_releases_it_and_writes_full_trace(
    monkeypatch, tmp_path: Path
) -> None:
    tool_schema = OpenAIFunctionToolSchema.model_validate(
        {
            "type": "function",
            "function": {
                "name": "ask_user",
                "description": "clarify",
                "parameters": {
                    "type": "object",
                    "properties": {"question": {"type": "string"}},
                    "required": ["question"],
                },
            },
        }
    )
    ask = AskUserTool({"max_questions": 3}, tool_schema)
    state = {
        "db_id": "retail",
        "user_query_ambiguity": {
            "critical_ambiguity": [
                {
                    "term": "active customer",
                    "sql_snippet": "last_order_date >= date('now', '-90 day')",
                    "type": "condition_ambiguity",
                }
            ],
            "non_critical_ambiguity": [],
        },
    }
    tools_kwargs = {"ask_user": {"create_kwargs": state}}

    loop = object.__new__(DriftToolAgentLoop)
    loop.tools = {"ask_user": ask}
    loop.max_tool_response_length = 4096
    loop.tool_response_truncate_side = "right"
    loop.tokenizer = FakeTokenizer()
    loop.max_assistant_turns = 2

    async def fake_parent_run(self, sampling_params, **kwargs):
        del sampling_params
        agent_data = AgentData(
            messages=list(kwargs["raw_prompt"]),
            image_data=[],
            video_data=[],
            audio_data=None,
            mm_processor_kwargs=None,
            metrics={"fake": 1.0},
            request_id="stable-request",
            tools_kwargs=kwargs["tools_kwargs"],
        )
        first, _, first_metrics = await self._call_tool(
            FunctionCall(
                name="ask_user",
                arguments=json.dumps({"question": "What is an active customer?"}),
            ),
            agent_data.tools_kwargs,
            agent_data,
        )
        second, _, second_metrics = await self._call_tool(
            FunctionCall(
                name="ask_user",
                arguments=json.dumps({"question": "Clarify active customer again"}),
            ),
            agent_data.tools_kwargs,
            agent_data,
        )
        assert "90 day" in first.text
        assert first_metrics["clarification_matched"]
        assert second_metrics["duplicate_question"]
        agent_data.messages.extend(
            [
                {"role": "tool", "content": first.text},
                {"role": "tool", "content": second.text},
            ]
        )
        agent_data.assistant_turns = 2
        agent_data.user_turns = 2
        return AgentLoopOutput(
            prompt_ids=[1],
            response_ids=[2, 3],
            response_mask=[1, 1],
            metrics=AgentLoopMetrics(),
            num_turns=5,
            extra_fields={},
        )

    monkeypatch.setattr(ToolAgentLoop, "run", fake_parent_run)
    monkeypatch.setenv("DRIFTSQL_TRAJECTORY_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("DRIFTSQL_KEY_ACTION_MASK", "0")
    output = asyncio.run(
        loop.run(
            {},
            raw_prompt=[{"role": "user", "content": "Show active customers"}],
            extra_info={"instance_id": "sample-1"},
            tools_kwargs=tools_kwargs,
        )
    )

    assert len(output.extra_fields["environment_events"]) == 2
    assert output.extra_fields["response_tokens"] == 2
    assert output.extra_fields["trajectory_timed_out"] is False
    assert output.extra_fields["trajectory_turn_limit"] is True
    assert output.extra_fields["advantage_scope"] == "episode"
    assert output.extra_fields["episode_response_mask_tokens"] == 2
    assert output.extra_fields["advantage_mask_tokens"] == 2
    assert ask._instances == {}
    traces = list(tmp_path.glob("*.json"))
    assert len(traces) == 1
    trace = json.loads(traces[0].read_text(encoding="utf-8"))
    assert trace["request_id"] == "stable-request"
    assert trace["status"] == "completed"
    assert trace["decoded_response"] == "decoded:2,3"
    assert len(trace["events"]) == 2
    assert trace["events"][1]["metrics"]["duplicate_question"]
