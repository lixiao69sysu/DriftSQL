#!/usr/bin/env python3
"""Evaluate the unadapted base model with one vLLM process per episode.

This is intentionally separate from the frozen P5 evaluator sources.  It is
used for post-hoc Tune baselines only and never reads or evaluates a Gate row.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import run_five_tool_eval as evaluator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data/processed/p5_grpo/tune_agent_eval.jsonl"
DEFAULT_MODEL = ROOT / "models/Qwen2.5-Coder-7B-Instruct"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, required=True)
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
    args.output_dir = args.output_dir.resolve()
    if not args.data.is_file() or not args.model.is_dir():
        raise FileNotFoundError(args.data if not args.data.is_file() else args.model)
    records = load_jsonl(args.data)
    if not records:
        raise RuntimeError("Base evaluation input is empty")
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}; use --resume")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episode_root = args.output_dir / "episodes"
    episode_root.mkdir(exist_ok=True)
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("At least one GPU is required")
    if args.episode_timeout_seconds <= 0 or args.max_attempts <= 0:
        raise ValueError("Episode timeout and max attempts must be positive")

    base_alias = f"{args.model.name.casefold()}-base"
    command_prefix = [
        str(ROOT / ".venv/bin/python"),
        str(ROOT / "scripts/run_stage6_eval.py"),
        "--data", str(args.data),
        "--model", str(args.model),
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
        "--drift-type", "add_column",
    ]

    def run_queue(gpu: str, indices: list[int]) -> list[dict[str, Any]]:
        completed: list[dict[str, Any]] = []
        for index in indices:
            episode_dir = episode_root / f"episode_{index:04d}"
            result_path = episode_dir / f"{base_alias}.jsonl"
            log_path = episode_dir / "run.log"
            if result_path.is_file():
                rows = load_jsonl(result_path)
                if len(rows) != 1:
                    raise RuntimeError(f"Invalid resumed result: {result_path}")
                completed.append({"index": index, "row": rows[0]})
                continue
            episode_dir.mkdir(parents=True, exist_ok=True)
            command = command_prefix + [
                "--offset", str(index),
                "--output-dir", str(episode_dir),
            ]
            environment = dict(os.environ)
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": gpu,
                    "PYTHONPATH": (
                        f"{ROOT}:{ROOT / 'third_party/verl'}"
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
                        cwd=ROOT,
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
                            f"\n=== episode timed out after {args.episode_timeout_seconds}s ===\n"
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
            print(f"{base_alias}: GPU {gpu} completed {index + 1}/{len(records)}", flush=True)
        return completed

    queues = [list(range(offset, len(records), len(gpus))) for offset in range(len(gpus))]
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(run_queue, gpu, indices) for gpu, indices in zip(gpus, queues)]
        indexed = [item for future in futures for item in future.result()]
    indexed.sort(key=lambda item: int(item["index"]))
    if [int(item["index"]) for item in indexed] != list(range(len(records))):
        raise RuntimeError("Episode index coverage is incomplete")
    rows = [dict(item["row"]) for item in indexed]
    expected_ids = [str(row["extra_info"]["instance_id"]) for row in records]
    actual_ids = [str(row["instance_id"]) for row in rows]
    if actual_ids != expected_ids or len(set(actual_ids)) != len(actual_ids):
        raise RuntimeError("Aggregated episode identity/order mismatch")

    write_jsonl(args.output_dir / f"{base_alias}.jsonl", rows)
    summary = {
        "protocol": "driftsql_process_isolated_base_eval_v1",
        "purpose": "post-hoc P5 Tune baseline; excluded from candidate selection",
        "gate_rows_read": False,
        "isolation_unit": "one OS process and one vLLM engine per episode",
        "data": str(args.data),
        "model": str(args.model),
        "gpus": gpus,
        "episodes": len(rows),
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
        "result": evaluator.summarize(base_alias, rows),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
