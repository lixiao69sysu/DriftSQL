#!/usr/bin/env python3
"""Generate paired Base/LoRA Reasoning predictions with one local vLLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from driftsql.evaluation.reasoning import extract_reasoning_sql, reasoning_format


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data/evaluation/stage3_reasoning/reasoning_val_128.jsonl"
DEFAULT_MODEL = PROJECT_ROOT / "models/Qwen2.5-Coder-3B-Instruct"
DEFAULT_ADAPTER = (
    PROJECT_ROOT
    / "checkpoints/stage3_reasoning_sft_3b_smoke/global_step_10/merged/lora_adapter"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "reports/stage3/reasoning_eval"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def generate_variant(
    *,
    llm: Any,
    tokenizer: Any,
    tasks: list[dict[str, Any]],
    alias: str,
    batch_size: int,
    lora_request: Any = None,
) -> list[dict[str, Any]]:
    from vllm import SamplingParams

    prompts = [
        tokenizer.apply_chat_template(
            task["messages"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for task in tasks
    ]
    sampling = SamplingParams(temperature=0.0, max_tokens=768, seed=42)
    predictions: list[dict[str, Any]] = []
    for start in range(0, len(tasks), batch_size):
        task_batch = tasks[start : start + batch_size]
        prompt_batch = prompts[start : start + batch_size]
        outputs = llm.generate(
            prompt_batch,
            sampling,
            lora_request=lora_request,
            use_tqdm=False,
        )
        for task, output in zip(task_batch, outputs, strict=True):
            generated = output.outputs[0]
            response = str(generated.text)
            sql = extract_reasoning_sql(response)
            format_state = reasoning_format(response)
            predictions.append(
                task
                | {
                    "baseline": alias,
                    "raw_response": response,
                    "final_sql": sql,
                    "format": format_state,
                    "termination_reason": "submitted" if sql else "invalid_output",
                    "usage": {
                        "model_calls": 1,
                        "tool_calls": 0,
                        "sql_executions": 0,
                        "prompt_tokens": len(output.prompt_token_ids),
                        "new_tokens": len(generated.token_ids),
                        "total_tokens": len(output.prompt_token_ids) + len(generated.token_ids),
                    },
                }
            )
        print(f"{alias}: generated {min(start + batch_size, len(tasks))}/{len(tasks)}", flush=True)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument(
        "--adapter-spec",
        action="append",
        default=[],
        help="Repeat alias=/absolute/or/project-relative/adapter/path to compare multiple LoRAs.",
    )
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    adapter_specs: list[tuple[str, Path]] = []
    if args.adapter_spec:
        for value in args.adapter_spec:
            if "=" not in value:
                parser.error("--adapter-spec must use alias=path")
            alias, raw_path = value.split("=", 1)
            path = Path(raw_path).expanduser().resolve()
            adapter_specs.append((alias.strip(), path))
    else:
        adapter_specs.append(("qwen2.5-coder-3b-reasoning-sft-step10", args.adapter.resolve()))
    for _, path in adapter_specs:
        if not path.is_dir():
            raise FileNotFoundError(path)

    from vllm import LLM
    from vllm.lora.request import LoRARequest

    tasks = load_jsonl(args.data)
    llm = LLM(
        model=str(args.model.resolve()),
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=4096,
        trust_remote_code=True,
        enforce_eager=True,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=True,
        max_lora_rank=32,
        generation_config="vllm",
    )
    tokenizer = llm.get_tokenizer()
    variants = [] if args.skip_base else [("qwen2.5-coder-3b-base-reasoning", None)]
    variants.extend(
        (
            alias,
            LoRARequest(f"stage3-reasoning-{index}", index, str(path)),
        )
        for index, (alias, path) in enumerate(adapter_specs, 1)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for alias, request in variants:
        predictions = generate_variant(
            llm=llm,
            tokenizer=tokenizer,
            tasks=tasks,
            alias=alias,
            batch_size=args.batch_size,
            lora_request=request,
        )
        write_jsonl(args.output_dir / f"{alias}.jsonl", predictions)


if __name__ == "__main__":
    main()
