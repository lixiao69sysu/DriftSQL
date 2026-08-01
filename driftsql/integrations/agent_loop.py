"""VERL agent-loop specialization that treats submission as terminal."""

from __future__ import annotations

import asyncio
import contextvars
import json
import os
import time
from pathlib import Path
from uuid import uuid4

from verl.experimental.agent_loop.tool_agent_loop import (
    AgentData,
    AgentState,
    ToolAgentLoop,
)
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput
from verl.experimental.agent_loop.tool_parser import (
    FunctionCall,
    ToolParser,
)
from verl.tools.function_tool import FunctionTool
from verl.tools.schemas import ToolResponse

from driftsql.tool_calls import find_tool_calls, remove_tool_call_payloads
from driftsql.integrations.state_policy import (
    dynamic_mask_response,
    duplicate_retrieval_response,
    is_exact_duplicate_retrieval,
    select_dynamic_tool_names,
)


_TRAJECTORY_CONTEXT: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "driftsql_trajectory_context", default=None
)


def _write_trace(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@ToolParser.register("driftsql-json")
class DriftJsonToolParser(ToolParser):
    """Tool parser accepting tagged, fenced, and bare JSON tool calls."""

    async def extract_tool_calls(self, responses_ids, tools=None):
        del tools
        text = self.tokenizer.decode(responses_ids)
        parsed = find_tool_calls(text)
        calls = [
            FunctionCall(
                name=call.name,
                arguments=json.dumps(call.arguments, ensure_ascii=False),
            )
            for call in parsed
        ]
        return remove_tool_call_payloads(text, parsed), calls


class DriftToolAgentLoop(ToolAgentLoop):
    """Trajectory-stateful VERL loop with bounded runtime and auditable logs."""

    async def run(self, sampling_params: dict, **kwargs) -> AgentLoopOutput:
        started = time.time()
        context: dict = {
            "trace_id": uuid4().hex,
            "request_id": None,
            "status": "running",
            "tools": {},
            "events": [],
            "raw_prompt": kwargs.get("raw_prompt", []),
            "extra_info": kwargs.get("extra_info", {}),
            "agent_data": None,
        }
        token = _TRAJECTORY_CONTEXT.set(context)
        output: AgentLoopOutput | None = None
        timeout = float(os.environ.get("DRIFTSQL_TRAJECTORY_TIMEOUT", "300"))
        try:
            output = await asyncio.wait_for(
                super().run(sampling_params, **kwargs),
                timeout=timeout,
            )
            context["status"] = "completed"
            submitted = any(
                event.get("tool") == "submit_solution"
                for event in context["events"]
            )
            max_assistant_turns = int(getattr(self, "max_assistant_turns", 7))
            trajectory_turn_limit = bool(
                not submitted and len(context["events"]) >= max_assistant_turns
            )
            output.extra_fields["environment_events"] = list(context["events"])
            output.extra_fields["response_tokens"] = len(output.response_ids)
            output.extra_fields["trajectory_timed_out"] = False
            output.extra_fields["trajectory_turn_limit"] = trajectory_turn_limit
            return output
        except asyncio.TimeoutError:
            context["status"] = "timeout"
            pad_token_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
            output = AgentLoopOutput(
                prompt_ids=[pad_token_id],
                response_ids=[pad_token_id],
                response_mask=[0],
                metrics=AgentLoopMetrics(),
                num_turns=0,
                extra_fields={
                    "environment_events": list(context["events"]),
                    "response_tokens": 0,
                    "trajectory_timed_out": True,
                    "trajectory_turn_limit": False,
                },
            )
            return output
        except Exception as error:
            context["status"] = "error"
            context["error"] = repr(error)
            raise
        finally:
            request_id = context.get("request_id")
            for tool in context["tools"].values():
                if request_id:
                    try:
                        await tool.release(request_id)
                    except (KeyError, RuntimeError):
                        pass

            agent_data = context.get("agent_data")
            trace = {
                "trace_id": context["trace_id"],
                "request_id": request_id,
                "status": context["status"],
                "started_at_unix": started,
                "elapsed_seconds": round(time.time() - started, 6),
                "raw_prompt": context["raw_prompt"],
                "extra_info": context["extra_info"],
                "events": context["events"],
                "messages": getattr(agent_data, "messages", []),
                "assistant_turns": getattr(agent_data, "assistant_turns", 0),
                "user_turns": getattr(agent_data, "user_turns", 0),
                "metrics": getattr(agent_data, "metrics", {}),
                "error": context.get("error"),
            }
            if output is not None:
                trace["response_ids"] = list(output.response_ids)
                try:
                    trace["decoded_response"] = self.tokenizer.decode(output.response_ids)
                except Exception:
                    trace["decoded_response"] = ""
            log_dir = os.environ.get("DRIFTSQL_TRAJECTORY_LOG_DIR")
            if log_dir:
                name = request_id or context["trace_id"]
                # The trace is a small atomic JSON write. Keeping it in the
                # event-loop thread avoids a rare lost-wakeup interaction
                # between asyncio's default executor and the Torch/Ray thread
                # pools initialized by VERL on this machine.
                _write_trace(
                    Path(log_dir) / f"{os.getpid()}-{name}.json",
                    trace,
                )
            _TRAJECTORY_CONTEXT.reset(token)

    async def _call_tool(
        self,
        tool_call: FunctionCall,
        tools_kwargs: dict,
        agent_data: AgentData,
    ) -> tuple[ToolResponse, float, dict]:
        active_tools = getattr(agent_data, "_active_tools", self.tools)
        if not hasattr(agent_data, "_base_active_tools"):
            agent_data._base_active_tools = dict(active_tools)
        base_active_tools = agent_data._base_active_tools
        tool_name = tool_call.name
        started = time.monotonic()
        event = {
            "index": len((_TRAJECTORY_CONTEXT.get() or {}).get("events", [])),
            "tool": tool_name,
            "arguments_raw": tool_call.arguments,
        }
        try:
            context = _TRAJECTORY_CONTEXT.get()
            dynamic_mask_enabled = os.environ.get("DRIFTSQL_DYNAMIC_TOOL_MASK", "0") == "1"
            if dynamic_mask_enabled:
                allowed = select_dynamic_tool_names(
                    (context or {}).get("events", []), set(base_active_tools)
                )
                active_tools = {
                    name: tool for name, tool in base_active_tools.items() if name in allowed
                }
                agent_data._active_tools = active_tools
                agent_data._active_tool_schemas = [
                    tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True)
                    for tool in active_tools.values()
                ]
                if tool_name not in active_tools:
                    text, metrics = dynamic_mask_response(tool_name, allowed)
                    event.update(
                        {
                            "response": text,
                            "reward": 0.0,
                            "metrics": metrics,
                            "success": False,
                        }
                    )
                    return ToolResponse(text=text), 0.0, metrics
            if tool_name not in active_tools:
                raise KeyError(f"Unknown tool '{tool_name}'; available: {sorted(active_tools)}")
            try:
                tool_args = json.loads(tool_call.arguments)
            except (json.JSONDecodeError, TypeError) as error:
                raise ValueError(f"Invalid JSON arguments for '{tool_name}': {error}") from error
            if not isinstance(tool_args, dict):
                raise ValueError(f"Arguments for '{tool_name}' must be an object")
            event["arguments"] = tool_args
            guards_enabled = os.environ.get("DRIFTSQL_STATE_GUARDS", "0") == "1"
            if (
                guards_enabled
                and context is not None
                and is_exact_duplicate_retrieval(context["events"], tool_name, tool_args)
            ):
                text, metrics = duplicate_retrieval_response(tool_name)
                response = ToolResponse(text=text)
                event.update(
                    {
                        "response": text,
                        "reward": 0.0,
                        "metrics": metrics,
                        "success": False,
                    }
                )
                return response, 0.0, metrics
            tool = active_tools[tool_name]
            if isinstance(tool, FunctionTool):
                response, reward, metrics = await super()._call_tool(
                    tool_call, tools_kwargs, agent_data
                )
            else:
                instance_id = agent_data.request_id
                kwargs = tools_kwargs.get(tool_name, {})
                await tool.create(
                    instance_id=instance_id,
                    create_kwargs=kwargs.get("create_kwargs", {}),
                )
                context = _TRAJECTORY_CONTEXT.get()
                if context is not None:
                    context["request_id"] = instance_id
                    context["tools"][tool_name] = tool
                    context["agent_data"] = agent_data
                response, reward, metrics = await tool.execute(
                    instance_id,
                    tool_args,
                    agent_data=agent_data,
                )

                text = response.text
                if text and len(text) > self.max_tool_response_length:
                    length = self.max_tool_response_length
                    if self.tool_response_truncate_side == "left":
                        text = "(truncated)..." + text[-length:]
                    elif self.tool_response_truncate_side == "right":
                        text = text[:length] + "...(truncated)"
                    else:
                        half = length // 2
                        text = text[:half] + "...(truncated)..." + text[-half:]
                    # Newer VERL/Pydantic schemas accept omitted media fields,
                    # but reject an explicit ``None`` because image/video must
                    # be lists when present.  Text-only SQL tools commonly hit
                    # this branch after their observations are truncated.
                    response_kwargs: dict[str, Any] = {"text": text}
                    if response.image is not None:
                        response_kwargs["image"] = response.image
                    if response.video is not None:
                        response_kwargs["video"] = response.video
                    response = ToolResponse(**response_kwargs)
            event.update(
                {
                    "response": response.text,
                    "reward": reward,
                    "metrics": metrics,
                    "success": not str(response.text or "").startswith("Error"),
                }
            )
            return response, reward, metrics
        except Exception as error:
            event.update({"response": f"Error executing tool '{tool_name}': {error}", "success": False})
            return ToolResponse(text=event["response"]), 0.0, {}
        finally:
            event["elapsed_ms"] = round((time.monotonic() - started) * 1000, 3)
            context = _TRAJECTORY_CONTEXT.get()
            if context is not None:
                context["events"].append(event)
                if os.environ.get("DRIFTSQL_DYNAMIC_TOOL_MASK", "0") == "1":
                    base_tools = getattr(agent_data, "_base_active_tools", self.tools)
                    allowed = select_dynamic_tool_names(context["events"], set(base_tools))
                    agent_data._active_tools = {
                        name: tool for name, tool in base_tools.items() if name in allowed
                    }
                    agent_data._active_tool_schemas = [
                        tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True)
                        for tool in agent_data._active_tools.values()
                    ]

    async def _handle_processing_tools_state(
        self,
        agent_data: AgentData,
    ) -> AgentState:
        state = await super()._handle_processing_tools_state(agent_data)
        if any(
            call.name == "submit_solution"
            for call in agent_data.tool_calls[: self.max_parallel_calls]
        ):
            return AgentState.TERMINATED
        return state
