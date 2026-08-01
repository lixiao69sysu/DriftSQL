#!/usr/bin/env python3
"""Run execution-grounded multi-turn evaluation for Base and SFT LoRAs."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlglot import exp, parse_one

from driftsql.drift import fingerprint_query
from driftsql.tool_calls import find_tool_calls, remove_tool_call_payloads


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data/processed/five_tool_sft/val_agent_eval.jsonl"
DEFAULT_MODEL = PROJECT_ROOT / "models/Qwen2.5-Coder-3B-Instruct"
DEFAULT_TOOLS = PROJECT_ROOT / "configs/tools/drift_tools.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports/stage3/five_tool_eval"
TOOL_NAMES = (
    "get_schema",
    "ask_user",
    "get_knowledge_definition",
    "execute_sql",
    "submit_solution",
)


def unsafe_sql(sql: str) -> bool:
    if not sql.strip():
        return False
    if sql.lstrip().upper().startswith("EXPLAIN"):
        return False
    try:
        expression = parse_one(sql, read="sqlite")
    except Exception:
        return False
    return not isinstance(expression, (exp.Query, exp.Subquery))


def apply_terminal_submit_fallback(state: "EvalState") -> bool:
    """Submit the last successfully executed read-only SQL at the turn limit.

    The fallback is deliberately conservative: it only uses SQL that already
    passed the sandboxed ``execute_sql`` tool.  It never invents or rewrites a
    query, and callers must opt in explicitly.
    """

    for event in reversed(state.trajectory):
        if event.get("tool_name") != "execute_sql":
            continue
        sql = str(event.get("arguments", {}).get("sql", "")).strip()
        execution_success = bool(event.get("metrics", {}).get("execution_success"))
        if sql and execution_success and not unsafe_sql(sql):
            state.final_sql = sql
            state.termination_reason = "fallback_submitted"
            return True
    return False


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)


@dataclass
class EvalState:
    record: dict[str, Any]
    variant: str
    conversation: list[dict[str, Any]]
    instance_id: str
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    created_tools: set[str] = field(default_factory=set)
    termination_reason: str = "running"
    final_sql: str = ""
    model_calls: int = 0
    prompt_tokens: int = 0
    new_tokens: int = 0


async def execute_action(
    state: EvalState,
    tools: dict[str, Any],
    name: str,
    arguments: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    if name not in tools:
        return (
            f"Error executing tool '{name}': unknown tool; available: {sorted(tools)}",
            {"unknown_tool": True},
        )
    tool = tools[name]
    if name not in state.created_tools:
        tool_kwargs = state.record["extra_info"]["tools_kwargs"][name]
        await tool.create(instance_id=state.instance_id, **tool_kwargs)
        state.created_tools.add(name)
    response, _, metrics = await tool.execute(state.instance_id, arguments)
    return str(response.text), dict(metrics)


async def execute_turn(
    actions: list[tuple[EvalState, str, dict[str, Any], str, int]],
    tools: dict[str, Any],
) -> None:
    async def one(item: tuple[EvalState, str, dict[str, Any], str, int]) -> None:
        state, name, arguments, raw_response, parsed_calls = item
        turn = len(state.trajectory)
        entry: dict[str, Any] = {
            "turn": turn,
            "raw_response": raw_response,
            "tool_name": name,
            "arguments": arguments,
            "parsed_tool_calls": parsed_calls,
        }
        try:
            observation, metrics = await execute_action(state, tools, name, arguments)
            entry["observation"] = observation
            entry["metrics"] = metrics
            state.trajectory.append(entry)
            if name == "submit_solution":
                state.final_sql = str(arguments.get("sql", "")).strip()
                state.termination_reason = "submitted" if metrics.get("submitted") else "invalid_submit"
            else:
                state.conversation.append(
                    {"role": "tool", "content": observation}
                )
        except Exception as error:
            entry["error"] = f"{type(error).__name__}: {error}"
            observation = f"Error executing tool '{name}': {error}"
            entry["observation"] = observation
            state.trajectory.append(entry)
            state.conversation.append({"role": "tool", "content": observation})

    await asyncio.gather(*(one(item) for item in actions))


async def score_and_release(states: list[EvalState], tools: dict[str, Any]) -> list[dict[str, Any]]:
    async def score_one(state: EvalState) -> dict[str, Any]:
        executable = False
        task_success = False
        error = ""
        if state.final_sql:
            try:
                if "execute_sql" not in state.created_tools:
                    kwargs = state.record["extra_info"]["tools_kwargs"]["execute_sql"]
                    await tools["execute_sql"].create(instance_id=state.instance_id, **kwargs)
                    state.created_tools.add("execute_sql")
                active_db = Path(tools["execute_sql"]._state(state.instance_id)["db_path"])
                # The parent process already owns vLLM/Torch thread pools.
                # A tiny number of local SQLite scoring calls is safer and
                # deterministic on the event-loop thread than through the
                # default asyncio executor on this machine.
                predicted = fingerprint_query(active_db, state.final_sql)
                expected = state.record["extra_info"]["result_fingerprint"]
                executable = True
                task_success = (
                    predicted.row_count == int(expected["row_count"])
                    and predicted.value_hash == str(expected["value_hash"])
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

        called = [str(item.get("tool_name", "")) for item in state.trajectory]
        sql_calls = [
            str(item.get("arguments", {}).get("sql", "")).strip()
            for item in state.trajectory
            if item.get("tool_name") in {"execute_sql", "submit_solution"}
        ]
        unsafe_actions = sum(unsafe_sql(sql) for sql in sql_calls)
        duplicate_questions = sum(
            bool(item.get("metrics", {}).get("duplicate_question"))
            for item in state.trajectory
        )
        normalized_executions = [
            str(item.get("arguments", {}).get("sql", "")).rstrip(";").strip().casefold()
            for item in state.trajectory
            if item.get("tool_name") == "execute_sql"
            and str(item.get("arguments", {}).get("sql", "")).strip()
        ]
        duplicate_executions = sum(
            max(0, count - 1) for count in Counter(normalized_executions).values()
        )
        timed_out = any(
            "timeout" in str(item.get("metrics", {}).get("execution_error", "")).casefold()
            or "interrupt" in str(item.get("metrics", {}).get("execution_error", "")).casefold()
            for item in state.trajectory
        )
        payload = {
            "variant": state.variant,
            "instance_id": state.record["extra_info"]["instance_id"],
            "db_id": state.record["extra_info"]["db_id"],
            "data_source": state.record["data_source"],
            "drift_type": state.record["extra_info"].get("drift_type", ""),
            "difficulty": state.record["extra_info"].get("difficulty", ""),
            "scenario_type": state.record["extra_info"].get("scenario_type", ""),
            "interaction_profile": state.record["extra_info"].get("interaction_profile", ""),
            "failure_mode": state.record["extra_info"].get("failure_mode", ""),
            "termination_reason": state.termination_reason,
            "final_sql": state.final_sql,
            "executable": executable,
            "task_success": task_success,
            "error": error,
            "called_tools": called,
            "all_five_tools": set(TOOL_NAMES).issubset(set(called)),
            "safety": {
                "unsafe": unsafe_actions > 0,
                "unsafe_actions": unsafe_actions,
                "timed_out": timed_out,
                "duplicate_questions": duplicate_questions,
                "duplicate_executions": duplicate_executions,
            },
            "usage": {
                "model_calls": state.model_calls,
                "tool_calls": len(called),
                "sql_executions": called.count("execute_sql"),
                "prompt_tokens": state.prompt_tokens,
                "new_tokens": state.new_tokens,
            },
            "trajectory": state.trajectory,
        }
        for name in list(state.created_tools):
            try:
                await tools[name].release(state.instance_id)
            except KeyError:
                pass
        return payload

    return await asyncio.gather(*(score_one(state) for state in states))


def summarize(alias: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    calls = Counter(name for row in rows for name in row["called_tools"])
    return {
        "variant": alias,
        "tasks": total,
        "task_success": sum(bool(row["task_success"]) for row in rows),
        "task_success_rate": sum(bool(row["task_success"]) for row in rows) / total,
        "executable": sum(bool(row["executable"]) for row in rows),
        "executable_rate": sum(bool(row["executable"]) for row in rows) / total,
        "submitted": sum(
            row["termination_reason"] in {"submitted", "fallback_submitted"}
            for row in rows
        ),
        "fallback_submitted": sum(
            row["termination_reason"] == "fallback_submitted" for row in rows
        ),
        "all_five_tools": sum(bool(row["all_five_tools"]) for row in rows),
        "turn_limit": sum(row["termination_reason"] == "turn_limit" for row in rows),
        "unsafe_tasks": sum(bool(row["safety"]["unsafe"]) for row in rows),
        "unsafe_actions": sum(int(row["safety"]["unsafe_actions"]) for row in rows),
        "timeout_tasks": sum(bool(row["safety"]["timed_out"]) for row in rows),
        "duplicate_question_tasks": sum(
            int(row["safety"]["duplicate_questions"] > 0) for row in rows
        ),
        "duplicate_execution_tasks": sum(
            int(row["safety"]["duplicate_executions"] > 0) for row in rows
        ),
        "average_model_calls": sum(row["usage"]["model_calls"] for row in rows) / total,
        "average_tool_calls": sum(row["usage"]["tool_calls"] for row in rows) / total,
        "tool_calls": dict(sorted(calls.items())),
        "termination_reasons": dict(sorted(Counter(row["termination_reason"] for row in rows).items())),
    }


def run_variant(
    *,
    llm: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    tools: dict[str, Any],
    schemas: list[dict[str, Any]],
    alias: str,
    batch_size: int,
    max_turns: int,
    max_new_tokens: int,
    max_model_len: int,
    lora_request: Any,
    terminal_submit_fallback: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from vllm import SamplingParams

    states = [
        EvalState(
            record=record,
            variant=alias,
            conversation=[dict(message) for message in record["prompt"]],
            instance_id=f"eval-{alias}-{record['extra_info']['instance_id']}",
        )
        for record in records
    ]
    sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, seed=42)
    for turn in range(max_turns):
        active = [state for state in states if state.termination_reason == "running"]
        if not active:
            break
        actions: list[tuple[EvalState, str, dict[str, Any], str, int]] = []
        for start in range(0, len(active), batch_size):
            batch = active[start : start + batch_size]
            admitted: list[EvalState] = []
            prompts: list[str] = []
            for state in batch:
                prompt = tokenizer.apply_chat_template(
                    state.conversation,
                    tools=schemas,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                prompt_length = len(tokenizer.encode(prompt))
                if prompt_length >= max_model_len:
                    state.trajectory.append(
                        {
                            "turn": turn,
                            "error": "prompt_too_long",
                            "prompt_tokens": prompt_length,
                            "max_model_len": max_model_len,
                        }
                    )
                    state.termination_reason = "prompt_too_long"
                    continue
                admitted.append(state)
                prompts.append(prompt)
            if not prompts:
                continue
            outputs = llm.generate(
                prompts,
                sampling,
                lora_request=lora_request,
                use_tqdm=False,
            )
            for state, output in zip(admitted, outputs, strict=True):
                generated = output.outputs[0]
                response = str(generated.text)
                state.model_calls += 1
                state.prompt_tokens += len(output.prompt_token_ids)
                state.new_tokens += len(generated.token_ids)
                parsed = find_tool_calls(response)
                if not parsed:
                    state.conversation.append({"role": "assistant", "content": response})
                    state.trajectory.append({"turn": turn, "raw_response": response, "error": "no_tool_call"})
                    state.termination_reason = "invalid_output"
                    continue
                call = parsed[0]
                assistant_content = remove_tool_call_payloads(response, parsed)
                state.conversation.append(
                    {
                        "role": "assistant",
                        "content": assistant_content,
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                                },
                            }
                        ],
                    }
                )
                actions.append((state, call.name, call.arguments, response, len(parsed)))
            print(
                f"{alias}: turn {turn + 1}, generated {min(start + batch_size, len(active))}/{len(active)}",
                flush=True,
            )
        if actions:
            asyncio.run(execute_turn(actions, tools))
    for state in states:
        if state.termination_reason == "running":
            if terminal_submit_fallback and apply_terminal_submit_fallback(state):
                continue
            state.termination_reason = "turn_limit"
    rows = asyncio.run(score_and_release(states, tools))
    return rows, summarize(alias, rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tools", type=Path, default=DEFAULT_TOOLS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--adapter-spec", action="append", default=[], help="Repeat alias=adapter_path")
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-turns", type=int, default=7)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument(
        "--disable-tool",
        action="append",
        default=[],
        choices=TOOL_NAMES,
        help="Repeat to remove a tool from both the advertised schema and runtime.",
    )
    parser.add_argument(
        "--terminal-submit-fallback",
        action="store_true",
        help="At the turn limit, submit the last SQL already validated by execute_sql.",
    )
    args = parser.parse_args()

    adapter_specs: list[tuple[str, Path]] = []
    for value in args.adapter_spec:
        if "=" not in value:
            parser.error("--adapter-spec must use alias=path")
        alias, raw_path = value.split("=", 1)
        path = Path(raw_path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        adapter_specs.append((alias.strip(), path))

    from vllm import LLM
    from vllm.lora.request import LoRARequest

    records = load_jsonl(args.data)
    if args.limit > 0:
        records = records[: args.limit]
    llm = LLM(
        model=str(args.model.resolve()),
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        enforce_eager=True,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=bool(adapter_specs),
        max_lora_rank=32,
        max_loras=max(1, len(adapter_specs)),
        generation_config="vllm",
    )
    tokenizer = llm.get_tokenizer()
    # Loading VERL tools touches torch/CUDA in the parent process.  Start the
    # fork-based vLLM workers first so they never inherit initialized CUDA.
    from verl.tools.tool_registry import load_all_tools

    loaded_tools = load_all_tools(tool_config_path=str(args.tools), function_tool_path=None)
    enabled_tool_names = tuple(name for name in TOOL_NAMES if name not in set(args.disable_tool))
    required = {"execute_sql", "submit_solution"}
    if not required.issubset(enabled_tool_names):
        parser.error("execute_sql and submit_solution cannot be disabled")
    tools = {tool.name: tool for tool in loaded_tools if tool.name in enabled_tool_names}
    if "get_schema" in tools:
        tools["get_schema"].config["max_chars"] = 3500
    if "get_knowledge_definition" in tools:
        tools["get_knowledge_definition"].config["max_results"] = 1
    tools["execute_sql"].config["max_rows"] = 5
    schemas = [tools[name].tool_schema.model_dump(mode="json") for name in enabled_tool_names]
    base_alias = f"{args.model.name.casefold()}-base"
    variants: list[tuple[str, Any]] = [] if args.skip_base else [(base_alias, None)]
    variants.extend(
        (alias, LoRARequest(alias, index, str(path)))
        for index, (alias, path) in enumerate(adapter_specs, 1)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for alias, request in variants:
        rows, summary = run_variant(
            llm=llm,
            tokenizer=tokenizer,
            records=records,
            tools=tools,
            schemas=schemas,
            alias=alias,
            batch_size=args.batch_size,
            max_turns=args.max_turns,
            max_new_tokens=args.max_new_tokens,
            max_model_len=args.max_model_len,
            lora_request=request,
            terminal_submit_fallback=args.terminal_submit_fallback,
        )
        write_jsonl(args.output_dir / f"{alias}.jsonl", rows)
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "protocol": "five_tool_execution_eval_v1",
                "enabled_tools": list(enabled_tool_names),
                "disabled_tools": list(args.disable_tool),
                "variants": summaries,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
