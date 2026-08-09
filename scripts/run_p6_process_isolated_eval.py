#!/usr/bin/env python3
"""Evaluate one P6 system with one vLLM process per database episode.

The isolation boundary prevents vLLM LoRA/multi-turn state from leaking across
episodes.  The optional result-contract controller only submits read-only SQL
that the model already executed successfully after inspecting an audited diff.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import signal
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import run_five_tool_eval as base_evaluator

from driftsql.controllers.validated_submit import (
    find_contract_validated_submission,
    is_read_only_query,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data/processed/p6_generalized_protocol/dev_agent_eval.jsonl"
DEFAULT_MODEL = ROOT / "models/Qwen2.5-Coder-7B-Instruct"
TOOLS = (
    "get_schema_version",
    "inspect_schema_diff",
    "get_schema",
    "ask_user",
    "get_knowledge_definition",
    "execute_sql",
    "submit_solution",
)
SUBMITTED_REASONS = {
    "submitted",
    "fallback_submitted",
    "contract_validated_auto_submit",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)


def requested_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = len(rows)
    drift_rows = [row for row in rows if str(row.get("drift_type", "")) != "clean"]
    submitted = [row for row in rows if row.get("termination_reason") in SUBMITTED_REASONS]
    safe_submitted = [
        row
        for row in submitted
        if bool(row.get("executable"))
        and is_read_only_query(str(row.get("final_sql", "")))
        and not bool(row.get("safety", {}).get("unsafe"))
        and not bool(row.get("safety", {}).get("timed_out"))
    ]
    success = sum(bool(row.get("task_success")) for row in rows)
    drift_success = sum(bool(row.get("task_success")) for row in drift_rows)
    tool_calls = [int(row.get("usage", {}).get("tool_calls", 0)) for row in rows]
    model_calls = [int(row.get("usage", {}).get("model_calls", 0)) for row in rows]
    return {
        "tasks": tasks,
        "execution_success": success,
        "execution_success_rate": success / tasks,
        "drift_tasks": len(drift_rows),
        "drift_recovery": drift_success,
        "drift_recovery_rate": drift_success / len(drift_rows) if drift_rows else 0.0,
        "submitted": len(submitted),
        "submission_rate": len(submitted) / tasks,
        "safe_submitted": len(safe_submitted),
        "safe_submission_rate": len(safe_submitted) / tasks,
        "safe_submission_precision": len(safe_submitted) / len(submitted) if submitted else 0.0,
        "average_tool_calls": sum(tool_calls) / tasks,
        "total_tool_calls": sum(tool_calls),
        "average_model_calls": sum(model_calls) / tasks,
        "unsafe_tasks": sum(bool(row.get("safety", {}).get("unsafe")) for row in rows),
        "timeout_tasks": sum(bool(row.get("safety", {}).get("timed_out")) for row in rows),
        "termination_reasons": dict(
            sorted(Counter(str(row.get("termination_reason")) for row in rows).items())
        ),
    }


def apply_controller(
    rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    *,
    temporary_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output = copy.deepcopy(rows)
    metadata = {str(row["extra_info"]["instance_id"]): row["extra_info"] for row in records}
    decisions: list[dict[str, Any]] = []
    for row in output:
        extra_info = metadata[str(row["instance_id"])]
        decision = find_contract_validated_submission(
            list(row.get("trajectory", [])),
            extra_info,
            temporary_root=temporary_root,
            timeout_seconds=30.0,
        )
        decisions.append({"instance_id": row["instance_id"], **decision.to_dict()})
        if bool(row.get("task_success")) or not decision.accepted:
            continue
        row["final_sql"] = decision.sql
        row["termination_reason"] = "contract_validated_auto_submit"
        row["executable"] = True
        row["task_success"] = True
        row["error"] = ""
        row.setdefault("called_tools", []).append("submit_solution")
        row.setdefault("usage", {})["tool_calls"] = int(row.get("usage", {}).get("tool_calls", 0)) + 1
        row["controller"] = {
            "name": "contract_validated_submit_v1",
            "model_calls_added": 0,
            "tool_calls_added": 1,
            "decision": decision.to_dict(),
        }
        row.setdefault("trajectory", []).append(
            {
                "turn": len(row.get("trajectory", [])),
                "tool_name": "submit_solution",
                "arguments": {"sql": decision.sql},
                "observation": json.dumps(
                    {
                        "accepted": True,
                        "controller": "contract_validated_submit_v1",
                        "result_contract_match": True,
                    },
                    ensure_ascii=False,
                ),
                "metrics": {
                    "submitted": True,
                    "controller_applied": True,
                    "result_contract_match": True,
                },
            }
        )
    return output, decisions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--contract-controller", action="store_true")
    parser.add_argument("--constrained-tool-names", action="store_true")
    parser.add_argument("--knowledge-first-after-ask", action="store_true")
    parser.add_argument("--audited-repair-controller", action="store_true")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=7)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--episode-timeout-seconds", type=int, default=480)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--temporary-root", type=Path, default=ROOT / "data/tmp")
    args = parser.parse_args()

    args.data = args.data.resolve()
    args.model = args.model.resolve()
    args.output_dir = args.output_dir.resolve()
    args.temporary_root = args.temporary_root.resolve()
    if args.adapter_path is not None:
        args.adapter_path = args.adapter_path.resolve()
    if not args.data.is_file() or not args.model.is_dir():
        raise FileNotFoundError(args.data if not args.data.is_file() else args.model)
    if args.adapter_path is not None and not (args.adapter_path / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(args.adapter_path / "adapter_model.safetensors")
    records = load_jsonl(args.data)
    if args.limit > 0:
        records = records[: args.limit]
    if not records:
        raise RuntimeError("Evaluation input is empty")
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}; use --resume")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episode_root = args.output_dir / "episodes"
    episode_root.mkdir(exist_ok=True)
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("At least one GPU is required")

    child_alias = args.alias if args.adapter_path is not None else f"{args.model.name.casefold()}-base"
    command_prefix = [
        str(ROOT / ".venv/bin/python"),
        str(ROOT / "scripts/run_p6_generalized_eval.py"),
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
    ]
    if args.adapter_path is not None:
        command_prefix.extend(["--skip-base", "--adapter-spec", f"{args.alias}={args.adapter_path}"])
    if args.constrained_tool_names:
        command_prefix.append("--constrained-tool-names")
    if args.knowledge_first_after_ask:
        command_prefix.append("--knowledge-first-after-ask")
    if args.audited_repair_controller:
        command_prefix.append("--audited-repair-controller")

    def run_queue(gpu: str, indices: list[int]) -> list[dict[str, Any]]:
        completed: list[dict[str, Any]] = []
        for index in indices:
            episode_dir = episode_root / f"episode_{index:04d}"
            result_path = episode_dir / f"{child_alias}.jsonl"
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
                with log_path.open("w" if attempt == 1 else "a", encoding="utf-8") as log:
                    if attempt > 1:
                        log.write(f"\n=== isolated retry {attempt}/{args.max_attempts} ===\n")
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
                        log.write(f"\n=== episode timed out after {args.episode_timeout_seconds}s ===\n")
                if returncode == 0:
                    break
            if returncode != 0:
                raise RuntimeError(
                    f"GPU {gpu} episode {index} failed with {returncode}; see {log_path}"
                )
            rows = load_jsonl(result_path)
            if len(rows) != 1:
                raise RuntimeError(f"Expected one row: {result_path}")
            completed.append({"index": index, "row": rows[0]})
            print(f"{args.alias}: GPU {gpu} completed {index + 1}/{len(records)}", flush=True)
        return completed

    queues = [list(range(offset, len(records), len(gpus))) for offset in range(len(gpus))]
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(run_queue, gpu, indices) for gpu, indices in zip(gpus, queues)]
        indexed = [item for future in futures for item in future.result()]
    indexed.sort(key=lambda item: int(item["index"]))
    if [int(item["index"]) for item in indexed] != list(range(len(records))):
        raise RuntimeError("Episode index coverage is incomplete")
    raw_rows = [dict(item["row"]) for item in indexed]
    expected_ids = [str(row["extra_info"]["instance_id"]) for row in records]
    if [str(row["instance_id"]) for row in raw_rows] != expected_ids:
        raise RuntimeError("Aggregated episode identity/order mismatch")
    for row in raw_rows:
        row["variant"] = args.alias
    write_jsonl(args.output_dir / f"raw_{args.alias}.jsonl", raw_rows)

    decisions: list[dict[str, Any]] = []
    rows = raw_rows
    if args.contract_controller:
        rows, decisions = apply_controller(
            raw_rows,
            records,
            temporary_root=args.temporary_root,
        )
        write_jsonl(args.output_dir / "controller_decisions.jsonl", decisions)
    write_jsonl(args.output_dir / f"{args.alias}.jsonl", rows)

    base_evaluator.TOOL_NAMES = TOOLS
    summary = {
        "protocol": "p6_generalized_process_isolated_eval_v1",
        "data": str(args.data),
        "model": str(args.model),
        "adapter": str(args.adapter_path) if args.adapter_path is not None else None,
        "alias": args.alias,
        "episodes": len(rows),
        "gpus": gpus,
        "contract_controller": bool(args.contract_controller),
        "controller_accepted": sum(bool(item.get("accepted")) for item in decisions),
        "controller_applied": sum(bool(row.get("controller")) for row in rows),
        "inference": {
            "temperature": 0.0,
            "seed": 42,
            "max_turns": args.max_turns,
            "max_new_tokens": args.max_new_tokens,
            "max_model_len": args.max_model_len,
            "tools": list(TOOLS),
            "dynamic_tool_mask": True,
            "constrained_tool_names": bool(args.constrained_tool_names),
            "knowledge_first_after_ask": bool(args.knowledge_first_after_ask),
            "audited_repair_controller": bool(args.audited_repair_controller),
            "state_guards": True,
            "one_process_per_episode": True,
        },
        "requested_metrics": requested_metrics(rows),
        "raw_result": base_evaluator.summarize(args.alias, raw_rows),
        "final_result": base_evaluator.summarize(args.alias, rows),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
