#!/usr/bin/env python3
"""Run Direct SQL or fixed ReAct with a local vLLM model under one budget."""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from driftsql.evaluation import BirdEvalBudget, extract_candidate_sql, get_schema_from_db
from driftsql.evaluation.bird import column_descriptions, execute_for_agent
from driftsql.tool_calls import find_tool_calls


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BIRD_RL_ROOT = PROJECT_ROOT / "third_party/BIRD-RL"
sys.path.insert(0, str(BIRD_RL_ROOT))

from bird_rl.inference.bird.generate_prompts import build_history_from_trajectory  # noqa: E402
from bird_rl.inference.parse_responses import parse_response  # noqa: E402
from bird_rl.prompts.bird_sft_training import (  # noqa: E402
    SFT_TRAINING_SYSTEM_PROMPT,
    SFT_TRAINING_USER_TEMPLATE,
)


DIRECT_SYSTEM_PROMPT = """You are an expert SQLite text-to-SQL model.
Return exactly one final SQL query for the supplied database question.
Do not call exploratory tools. Prefer this exact wrapper:
<tool_call>{"name":"submit_solution","arguments":{"sql":"SELECT ..."}}</tool_call>
Do not include explanations after the query."""

DIRECT_USER_TEMPLATE = """## Database Schema
{schema}

## Column Descriptions
{column_descriptions}

## Question
{question}

## Evidence
{evidence}

Produce the final SQLite query now."""


@dataclass
class State:
    task: dict[str, Any]
    schema: str
    descriptions: str
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    final_sql: str = ""
    termination_reason: str = "running"
    model_calls: int = 0
    tool_calls: int = 0
    sql_executions: int = 0
    prompt_tokens: int = 0
    new_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.new_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--column-meaning", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--modes", default="direct,react")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=20000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sql-workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def load_states(data_path: Path, meanings: dict[str, str], limit: int | None) -> list[State]:
    tasks = json.loads(data_path.read_text(encoding="utf-8"))
    if limit is not None:
        tasks = tasks[:limit]
    schemas: dict[str, str] = {}
    descriptions: dict[str, str] = {}
    states: list[State] = []
    for task in tasks:
        db_id = str(task["db_id"])
        if db_id not in schemas:
            schemas[db_id] = get_schema_from_db(Path(task["db_path"]))
            descriptions[db_id] = column_descriptions(db_id, meanings)
        states.append(State(task=task, schema=schemas[db_id], descriptions=descriptions[db_id]))
    return states


def direct_messages(state: State) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": DIRECT_USER_TEMPLATE.format(
                schema=state.schema,
                column_descriptions=state.descriptions,
                question=state.task["question"],
                evidence=state.task.get("evidence") or "(No evidence provided)",
            ),
        },
    ]


def react_messages(state: State, budget: BirdEvalBudget) -> list[dict[str, str]]:
    user = SFT_TRAINING_USER_TEMPLATE.format(
        schema=state.schema,
        column_descriptions=state.descriptions,
        question=state.task["question"],
        evidence=state.task.get("evidence") or "(No evidence provided)",
        max_turns=budget.max_model_calls,
    )
    history = build_history_from_trajectory(state.trajectory)
    if history:
        user += "\n\n" + history
    return [
        {
            "role": "system",
            "content": SFT_TRAINING_SYSTEM_PROMPT.format(
                max_turns=budget.max_model_calls,
                prev_turns=budget.max_model_calls - 1,
            ),
        },
        {"role": "user", "content": user},
    ]


def parse_fallback_action(
    response: str,
    default_tool: str | None = None,
) -> tuple[str | None, list[str], bool]:
    """Accept common ReAct syntax when the upstream BIRD parser finds no tool."""
    xml_action = re.search(
        r"<(execute_sql|submit_solution)>\s*(.*?)\s*</\1>",
        response,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if xml_action:
        tool_name = xml_action.group(1).lower()
        sql = xml_action.group(2).strip()
        return tool_name, [sql] if sql else [], tool_name == "submit_solution"

    action = re.search(
        r"(?:Action|Tool)\s*:\s*(execute_sql|submit_solution)",
        response,
        flags=re.IGNORECASE,
    )
    sql_match = re.search(r"```(?:sql)?\s*(.*?)```", response, flags=re.IGNORECASE | re.DOTALL)
    if not sql_match or (not action and not default_tool):
        return None, [], False
    tool_name = action.group(1).lower() if action else default_tool
    sql = sql_match.group(1).strip()
    return tool_name, [sql] if sql else [], tool_name == "submit_solution"


def generate_batch(
    llm: Any,
    tokenizer: Any,
    states: list[State],
    messages: list[list[dict]],
    budget: BirdEvalBudget,
    batch_size: int,
):
    from vllm import SamplingParams

    prompts: list[str] = []
    admitted: list[State] = []
    for state, conversation in zip(states, messages, strict=True):
        rendered = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        prompt_tokens = len(tokenizer.encode(rendered))
        if prompt_tokens > budget.max_prompt_tokens:
            state.termination_reason = "prompt_too_long"
            continue
        if state.total_tokens + prompt_tokens >= budget.max_total_tokens:
            state.termination_reason = "total_token_budget"
            continue
        prompts.append(rendered)
        admitted.append(state)

    outputs: list[tuple[State, str, int, int]] = []
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=budget.max_new_tokens_per_call,
        stop=["<tool_response>", "</tool_response>"],
    )
    for start in range(0, len(prompts), batch_size):
        prompt_batch = prompts[start : start + batch_size]
        state_batch = admitted[start : start + batch_size]
        generated = llm.generate(prompt_batch, sampling, use_tqdm=False)
        for state, prompt, output in zip(state_batch, prompt_batch, generated, strict=True):
            prompt_count = len(tokenizer.encode(prompt))
            token_ids = output.outputs[0].token_ids
            remaining_new = max(0, budget.max_new_tokens - state.new_tokens)
            remaining_total = max(0, budget.max_total_tokens - state.total_tokens - prompt_count)
            allowed = min(len(token_ids), remaining_new, remaining_total)
            response = output.outputs[0].text if allowed == len(token_ids) else tokenizer.decode(token_ids[:allowed])
            state.model_calls += 1
            state.prompt_tokens += prompt_count
            state.new_tokens += allowed
            outputs.append((state, response, prompt_count, allowed))
        print(
            f"Generated {min(start + batch_size, len(prompts))}/{len(prompts)} prompts",
            flush=True,
        )
    return outputs


def result_record(state: State, baseline: str) -> dict[str, Any]:
    usage = {
        "model_calls": state.model_calls,
        "tool_calls": state.tool_calls,
        "sql_executions": state.sql_executions,
        "prompt_tokens": state.prompt_tokens,
        "new_tokens": state.new_tokens,
        "total_tokens": state.total_tokens,
    }
    return state.task | {
        "baseline": baseline,
        "final_sql": state.final_sql,
        "termination_reason": state.termination_reason,
        "usage": usage,
        "trajectory": state.trajectory,
    }


def write_results(path: Path, states: list[State], baseline: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for state in states:
            handle.write(json.dumps(result_record(state, baseline), ensure_ascii=False) + "\n")
    temporary.replace(path)


def run_direct(
    llm: Any,
    tokenizer: Any,
    states: list[State],
    budget: BirdEvalBudget,
    batch_size: int,
) -> None:
    outputs = generate_batch(
        llm,
        tokenizer,
        states,
        [direct_messages(state) for state in states],
        budget,
        batch_size,
    )
    for state, response, _, _ in outputs:
        state.final_sql = extract_candidate_sql(response)
        state.termination_reason = "submitted" if state.final_sql else "invalid_output"
        state.trajectory.append({"turn": 0, "raw_response": response, "final_sql": state.final_sql})
    for state in states:
        if state.termination_reason == "running":
            state.termination_reason = "generation_not_run"


def run_react(
    llm: Any,
    tokenizer: Any,
    states: list[State],
    budget: BirdEvalBudget,
    batch_size: int,
    sql_workers: int,
    checkpoint_path: Path | None = None,
    baseline: str = "",
) -> None:
    for turn in range(budget.max_model_calls):
        active = [state for state in states if state.termination_reason == "running"]
        if not active:
            break
        outputs = generate_batch(
            llm,
            tokenizer,
            active,
            [react_messages(state, budget) for state in active],
            budget,
            batch_size,
        )
        generated_states = {id(state) for state, _, _, _ in outputs}
        for state in active:
            if id(state) not in generated_states and state.termination_reason == "running":
                state.termination_reason = "generation_not_run"

        pending_sql: list[tuple[State, dict[str, Any], str, str]] = []
        for state, response, _, _ in outputs:
            thought, tool_name, sqls, end_flag = parse_response(response)
            if not tool_name:
                calls = find_tool_calls(response)
                if calls:
                    call = calls[0]
                    tool_name = call.name
                    sql_arg = call.arguments.get("sql", "")
                    sql_list = call.arguments.get("sql_list", [])
                    if isinstance(sql_arg, str) and sql_arg.strip():
                        sqls = [sql_arg.strip()]
                    elif isinstance(sql_list, list) and sql_list:
                        sqls = [str(sql_list[0]).strip()]
                    end_flag = tool_name == "submit_solution"
            if not tool_name:
                default_tool = (
                    "submit_solution" if turn == budget.max_model_calls - 1 else "execute_sql"
                )
                tool_name, sqls, end_flag = parse_fallback_action(response, default_tool)
            sql = str(sqls[0]).strip() if sqls else ""
            turn_record: dict[str, Any] = {
                "turn": turn,
                "thought": thought,
                "raw_response": response,
                "tool_name": tool_name,
                "sql": sql,
                "end_flag": end_flag,
            }
            if not tool_name or not sql:
                state.trajectory.append(turn_record)
                state.termination_reason = "invalid_action"
                continue
            if state.tool_calls >= budget.max_tool_calls:
                state.trajectory.append(turn_record)
                state.termination_reason = "tool_budget"
                continue
            state.tool_calls += 1
            if tool_name == "submit_solution":
                state.final_sql = sql
                state.termination_reason = "submitted"
                state.trajectory.append(turn_record)
                continue
            if tool_name != "execute_sql":
                state.trajectory.append(turn_record)
                state.termination_reason = "unknown_tool"
                continue
            if state.sql_executions >= budget.max_sql_executions:
                state.trajectory.append(turn_record)
                state.termination_reason = "sql_execution_budget"
                continue
            state.sql_executions += 1
            pending_sql.append((state, turn_record, thought, response))

        def execute_pending(item: tuple[State, dict[str, Any], str, str]):
            state, turn_record, thought, response = item
            observation = execute_for_agent(
                Path(state.task["db_path"]),
                str(turn_record["sql"]),
                timeout_seconds=budget.sql_timeout_seconds,
                max_rows=budget.max_result_rows_for_agent,
            )
            return state, turn_record, thought, response, observation

        with ThreadPoolExecutor(max_workers=max(1, sql_workers)) as executor:
            executed = executor.map(execute_pending, pending_sql)
            for state, turn_record, thought, response, observation in executed:
                turn_record["observation"] = json.dumps(
                    observation, ensure_ascii=False, default=str
                )
                state.trajectory.append(
                    {
                        "thought": thought,
                        "action": response,
                        "observation": turn_record["observation"],
                        "end_flag": False,
                        **turn_record,
                    }
                )
        if checkpoint_path is not None:
            write_results(checkpoint_path, states, baseline)
        remaining = sum(state.termination_reason == "running" for state in states)
        print(f"Completed ReAct turn {turn + 1}; {remaining} trajectories remain", flush=True)

    for state in states:
        if state.termination_reason == "running":
            state.termination_reason = "turn_limit"


def main() -> None:
    args = parse_args()
    budget = BirdEvalBudget()
    meanings = json.loads(args.column_meaning.read_text(encoding="utf-8"))

    from vllm import LLM

    llm = LLM(
        model=str(args.model),
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    tokenizer = llm.get_tokenizer()

    for mode in [value.strip() for value in args.modes.split(",") if value.strip()]:
        states = load_states(args.data, meanings, args.limit)
        baseline = f"{args.model_alias}_{mode}"
        output = args.output_dir / f"{baseline}.jsonl"
        if mode == "direct":
            run_direct(llm, tokenizer, states, budget, args.batch_size)
        elif mode == "react":
            run_react(
                llm,
                tokenizer,
                states,
                budget,
                args.batch_size,
                args.sql_workers,
                output,
                baseline,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
        write_results(output, states, baseline)
        print(f"Saved {len(states)} predictions to {output}", flush=True)


if __name__ == "__main__":
    main()
