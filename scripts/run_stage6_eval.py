#!/usr/bin/env python3
"""Run the Stage 6 six-tool evaluator without modifying the frozen Stage 5 runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import run_five_tool_eval as evaluator

from driftsql.integrations.state_policy import (
    duplicate_retrieval_response,
    is_exact_duplicate_retrieval,
    select_dynamic_tool_schemas,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data/processed/stage6_ablation/b1/tune_agent_eval.jsonl"
DEFAULT_MODEL = PROJECT_ROOT / "models/Qwen2.5-Coder-7B-Instruct"
DEFAULT_TOOLS = PROJECT_ROOT / "configs/tools/drift_tools.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports/stage6/b1_tune"
TOOL_NAMES = (
    "get_schema_version",
    "inspect_schema_diff",
    "get_schema",
    "get_knowledge_definition",
    "execute_sql",
    "submit_solution",
)


def bounded_generation_tokens(
    prompt_lengths: list[int], max_new_tokens: int, max_model_len: int
) -> int:
    """Keep every request inside vLLM's configured context window."""
    if not prompt_lengths:
        return max_new_tokens
    remaining = max_model_len - max(prompt_lengths)
    return max(1, min(max_new_tokens, remaining))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tools", type=Path, default=DEFAULT_TOOLS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--adapter-spec", action="append", default=[], help="Repeat alias=adapter_path")
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--drift-type",
        action="append",
        default=[],
        help="Tune-only diagnostic filter; repeat to select drift types.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-turns", type=int, default=7)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument(
        "--disable-async-scheduling",
        action="store_true",
        help="Use vLLM synchronous scheduling for stable multi-turn LoRA evaluation.",
    )
    parser.add_argument(
        "--disable-prefix-caching",
        action="store_true",
        help="Disable vLLM prefix caching for multi-turn LoRA stability.",
    )
    parser.add_argument(
        "--episode-major",
        action="store_true",
        help="Finish and release one full agent episode before starting the next.",
    )
    parser.add_argument(
        "--disable-tool", action="append", default=[], choices=TOOL_NAMES
    )
    parser.add_argument("--terminal-submit-fallback", action="store_true")
    parser.add_argument(
        "--state-guards",
        action="store_true",
        help="Reject exact duplicate retrievals with a next-action hint.",
    )
    parser.add_argument(
        "--dynamic-tool-mask",
        action="store_true",
        help="Expose retrievals once and only submit after a successful post-diff execution.",
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

    records = evaluator.load_jsonl(args.data)
    if args.drift_type:
        selected_types = set(args.drift_type)
        records = [
            row for row in records
            if str(row.get("extra_info", {}).get("drift_type", "")) in selected_types
        ]
        if not records:
            raise RuntimeError(f"No records matched --drift-type={sorted(selected_types)}")
    if args.offset < 0:
        parser.error("--offset must be non-negative")
    if args.offset:
        records = records[args.offset :]
    if args.limit > 0:
        records = records[: args.limit]
    if not records:
        raise RuntimeError("No evaluation records remain after filtering/offset/limit")
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
        async_scheduling=not args.disable_async_scheduling,
        enable_prefix_caching=not args.disable_prefix_caching,
    )
    tokenizer = llm.get_tokenizer()

    # The shared Stage 5 evaluator only rejects prompts that already fill the
    # context window.  It does not account for the requested completion.  A
    # near-full multi-turn tool prompt can therefore be admitted with another
    # 512 tokens and wait forever in vLLM's scheduler.  Bound only the active
    # request; the configured maximum remains unchanged for shorter prompts.
    unbounded_generate = llm.generate

    def context_bounded_generate(prompts, sampling_params, *generate_args, **generate_kwargs):
        if hasattr(sampling_params, "max_tokens"):
            prompt_lengths = [len(tokenizer.encode(prompt)) for prompt in prompts]
            sampling_params.max_tokens = bounded_generation_tokens(
                prompt_lengths, args.max_new_tokens, args.max_model_len
            )
        return unbounded_generate(
            prompts, sampling_params, *generate_args, **generate_kwargs
        )

    llm.generate = context_bounded_generate
    from verl.tools.tool_registry import load_all_tools

    loaded_tools = load_all_tools(tool_config_path=str(args.tools), function_tool_path=None)
    enabled_names = tuple(name for name in TOOL_NAMES if name not in set(args.disable_tool))
    if not {"execute_sql", "submit_solution"}.issubset(enabled_names):
        parser.error("execute_sql and submit_solution cannot be disabled")
    tools: dict[str, Any] = {tool.name: tool for tool in loaded_tools if tool.name in enabled_names}
    missing = sorted(set(enabled_names) - set(tools))
    if missing:
        raise RuntimeError(f"Missing Stage 6 tools: {missing}")
    if "get_schema" in tools:
        tools["get_schema"].config["max_chars"] = 3500
    if "get_knowledge_definition" in tools:
        tools["get_knowledge_definition"].config["max_results"] = 1
    tools["execute_sql"].config["max_rows"] = 5
    schemas = [tools[name].tool_schema.model_dump(mode="json") for name in enabled_names]

    if args.dynamic_tool_mask:
        base_tokenizer = tokenizer

        class DynamicToolTokenizer:
            def __getattr__(self, name):
                return getattr(base_tokenizer, name)

            def apply_chat_template(self, conversation, *, tools=None, **kwargs):
                active = select_dynamic_tool_schemas(conversation, list(tools or []))
                return base_tokenizer.apply_chat_template(
                    conversation, tools=active, **kwargs
                )

        tokenizer = DynamicToolTokenizer()

    # The reused evaluator computes its all-tools diagnostic from this module
    # global.  Patch the imported module only in this process; the frozen Stage
    # 5 source file and protocol remain unchanged.
    evaluator.TOOL_NAMES = enabled_names
    if args.state_guards:
        unguarded_execute_action = evaluator.execute_action

        async def guarded_execute_action(state, active_tools, name, arguments):
            if is_exact_duplicate_retrieval(state.trajectory, name, arguments):
                return duplicate_retrieval_response(name)
            return await unguarded_execute_action(state, active_tools, name, arguments)

        evaluator.execute_action = guarded_execute_action
    variants: list[tuple[str, Any]] = []
    if not args.skip_base:
        variants.append((f"{args.model.name.casefold()}-base", None))
    variants.extend(
        (alias, LoRARequest(alias, index, str(path)))
        for index, (alias, path) in enumerate(adapter_specs, 1)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for alias, request in variants:
        if args.episode_major:
            rows = []
            for index, record in enumerate(records, 1):
                episode_rows, _ = evaluator.run_variant(
                    llm=llm,
                    tokenizer=tokenizer,
                    records=[record],
                    tools=tools,
                    schemas=schemas,
                    alias=alias,
                    batch_size=1,
                    max_turns=args.max_turns,
                    max_new_tokens=args.max_new_tokens,
                    max_model_len=args.max_model_len,
                    lora_request=request,
                    terminal_submit_fallback=args.terminal_submit_fallback,
                )
                rows.extend(episode_rows)
                print(f"{alias}: completed episode {index}/{len(records)}", flush=True)
            summary = evaluator.summarize(alias, rows)
        else:
            rows, summary = evaluator.run_variant(
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
        evaluator.write_jsonl(args.output_dir / f"{alias}.jsonl", rows)
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "protocol": "stage6_version_diff_execution_eval_v1",
                "dataset_role": "tune_or_gate_selected_by_input_path",
                "enabled_tools": list(enabled_names),
                "disabled_tools": list(args.disable_tool),
                "state_guards": bool(args.state_guards),
                "dynamic_tool_mask": bool(args.dynamic_tool_mask),
                "async_scheduling": not args.disable_async_scheduling,
                "prefix_caching": not args.disable_prefix_caching,
                "episode_major": bool(args.episode_major),
                "context_bounded_generation": True,
                "offset": args.offset,
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
