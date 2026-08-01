#!/usr/bin/env python3
"""Run one vLLM process per agent episode and aggregate exact results.

vLLM 0.15.1 can retain invalid LoRA/multi-turn state across repeated
``LLM.generate`` calls.  This runner makes the evaluator process itself the
isolation boundary while parallelizing independent episodes across GPUs.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import run_five_tool_eval as evaluator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data/processed/stage7_add_column_sft/tune_agent_eval.jsonl"
DEFAULT_MODEL = PROJECT_ROOT / "models/Qwen2.5-Coder-7B-Instruct"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)


def select_records(
    records: list[dict[str, Any]], drift_types: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Filter without regrouping so child-process offsets remain exact."""
    selected = set(drift_types)
    return [
        row
        for row in records
        if str(row.get("extra_info", {}).get("drift_type", "")) in selected
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter-alias", required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument(
        "--drift-type",
        action="append",
        default=[],
        help=(
            "Repeat to evaluate multiple drift types in one isolated run. "
            "Defaults to add_column when omitted."
        ),
    )
    parser.add_argument("--gpus", default="0,2,3")
    parser.add_argument("--max-turns", type=int, default=7)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--episode-timeout-seconds", type=int, default=480)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.data = args.data.resolve()
    args.model = args.model.resolve()
    args.adapter_path = args.adapter_path.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.adapter_path.is_dir():
        raise FileNotFoundError(args.adapter_path)
    selected_types = tuple(dict.fromkeys(args.drift_type or ["add_column"]))
    all_records = load_jsonl(args.data)
    records = select_records(all_records, selected_types)
    if not records:
        raise RuntimeError(f"No records for drift_types={list(selected_types)}")
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}; use --resume")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_root = args.output_dir / "episodes"
    task_root.mkdir(exist_ok=True)
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("At least one GPU is required")
    if args.episode_timeout_seconds <= 0 or args.max_attempts <= 0:
        raise ValueError("Episode timeout and max attempts must be positive")

    base_command = [
        str(PROJECT_ROOT / ".venv/bin/python"),
        str(PROJECT_ROOT / "scripts/run_stage6_eval.py"),
        "--data", str(args.data),
        "--model", str(args.model),
        "--skip-base",
        "--tensor-parallel-size", "1",
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--batch-size", "1",
        "--max-turns", str(args.max_turns),
        "--max-new-tokens", str(args.max_new_tokens),
        "--max-model-len", str(args.max_model_len),
        "--state-guards",
        "--dynamic-tool-mask",
        "--disable-async-scheduling",
        "--disable-prefix-caching",
        "--episode-major",
        "--limit", "1",
        "--adapter-spec", f"{args.adapter_alias}={args.adapter_path}",
    ]
    for drift_type in selected_types:
        base_command.extend(["--drift-type", drift_type])

    def run_gpu_queue(gpu: str, indices: list[int]) -> list[dict[str, Any]]:
        completed: list[dict[str, Any]] = []
        for index in indices:
            episode_dir = task_root / f"episode_{index:04d}"
            result_path = episode_dir / f"{args.adapter_alias}.jsonl"
            log_path = episode_dir / "run.log"
            if result_path.is_file():
                rows = load_jsonl(result_path)
                if len(rows) != 1:
                    raise RuntimeError(f"Invalid resumed result: {result_path}")
                completed.append({"index": index, "row": rows[0]})
                continue
            episode_dir.mkdir(parents=True, exist_ok=True)
            command = base_command + [
                "--offset", str(index),
                "--output-dir", str(episode_dir),
            ]
            environment = dict(os.environ)
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": gpu,
                    "PYTHONPATH": (
                        f"{PROJECT_ROOT}:{PROJECT_ROOT / 'third_party/verl'}"
                        + (f":{environment['PYTHONPATH']}" if environment.get("PYTHONPATH") else "")
                    ),
                    "TOKENIZERS_PARALLELISM": "false",
                    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                    "DRIFTSQL_REWARD_TIMEOUT": "20",
                }
            )
            returncode = -1
            for attempt in range(1, args.max_attempts + 1):
                mode = "w" if attempt == 1 else "a"
                with log_path.open(mode, encoding="utf-8") as log:
                    if attempt > 1:
                        log.write(f"\n=== isolated retry {attempt}/{args.max_attempts} ===\n")
                        log.flush()
                    process = subprocess.Popen(
                        command,
                        cwd=PROJECT_ROOT,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    try:
                        returncode = process.wait(timeout=args.episode_timeout_seconds)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        returncode = 124
                        log.write(
                            f"\n=== episode timed out after "
                            f"{args.episode_timeout_seconds}s ===\n"
                        )
                        log.flush()
                if returncode == 0:
                    break
            if returncode != 0:
                raise RuntimeError(
                    f"GPU {gpu} episode {index} failed with {returncode} after "
                    f"{args.max_attempts} attempt(s); see {log_path}"
                )
            rows = load_jsonl(result_path)
            if len(rows) != 1:
                raise RuntimeError(f"Expected one row: {result_path}")
            completed.append({"index": index, "row": rows[0]})
            print(
                f"{args.adapter_alias}: GPU {gpu} completed episode {index + 1}/{len(records)}",
                flush=True,
            )
        return completed

    queues = [list(range(offset, len(records), len(gpus))) for offset in range(len(gpus))]
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(run_gpu_queue, gpu, indices) for gpu, indices in zip(gpus, queues)]
        indexed = [item for future in futures for item in future.result()]
    indexed.sort(key=lambda item: int(item["index"]))
    if [int(item["index"]) for item in indexed] != list(range(len(records))):
        raise RuntimeError("Episode index coverage is incomplete")
    rows = [dict(item["row"]) for item in indexed]
    expected_ids = [str(row["extra_info"]["instance_id"]) for row in records]
    actual_ids = [str(row["instance_id"]) for row in rows]
    if actual_ids != expected_ids or len(set(actual_ids)) != len(actual_ids):
        raise RuntimeError("Aggregated episode identity/order mismatch")

    write_jsonl(args.output_dir / f"{args.adapter_alias}.jsonl", rows)
    summary = {
        "protocol": "stage7_process_isolated_agent_eval_v1",
        "isolation_unit": "one OS process and one vLLM engine per episode",
        "drift_type": selected_types[0] if len(selected_types) == 1 else None,
        "drift_types": list(selected_types),
        "gpus": gpus,
        "episodes": len(rows),
        "adapter": str(args.adapter_path),
        "inference": {
            "async_scheduling": False,
            "prefix_caching": False,
            "episode_major": True,
            "dynamic_tool_mask": True,
            "state_guards": True,
            "context_bounded_generation": True,
            "max_turns": args.max_turns,
            "max_new_tokens": args.max_new_tokens,
            "max_model_len": args.max_model_len,
            "episode_timeout_seconds": args.episode_timeout_seconds,
            "max_attempts": args.max_attempts,
        },
        "result": evaluator.summarize(args.adapter_alias, rows),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
