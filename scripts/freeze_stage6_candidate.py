#!/usr/bin/env python3
"""Freeze the selected Stage 6 candidate before the sealed Gate112 pass."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROWS = (
    PROJECT_ROOT
    / "reports/stage6/b2_step20_dynamic_mask_guidance_guards_full_tune"
    / "b2-step20-final-tune-candidate.jsonl"
)
DEFAULT_SUMMARY = DEFAULT_ROWS.parent / "summary.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports/stage6/final_candidate/frozen_candidate.json"
DEFAULT_ADAPTER = (
    PROJECT_ROOT
    / "checkpoints/stage6_repair_sft_7b/global_step_20/merged/lora_adapter"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def group(field: str) -> dict[str, Any]:
        counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
        for row in rows:
            values = counts[str(row[field])]
            values[0] += 1
            values[1] += int(bool(row["task_success"]))
            values[2] += int(row["termination_reason"] == "submitted")
            values[3] += int(row["termination_reason"] == "turn_limit")
        return {
            name: {
                "tasks": values[0],
                "success": values[1],
                "success_rate": values[1] / values[0],
                "submitted": values[2],
                "turn_limit": values[3],
            }
            for name, values in sorted(counts.items())
        }

    non_clean = [row for row in rows if row["scenario_type"] != "clean"]
    total = len(rows)
    return {
        "overall": {
            "tasks": total,
            "success": sum(bool(row["task_success"]) for row in rows),
            "success_rate": sum(bool(row["task_success"]) for row in rows) / total,
            "submitted": sum(row["termination_reason"] == "submitted" for row in rows),
            "submission_rate": sum(row["termination_reason"] == "submitted" for row in rows) / total,
            "turn_limit": sum(row["termination_reason"] == "turn_limit" for row in rows),
            "turn_limit_rate": sum(row["termination_reason"] == "turn_limit" for row in rows) / total,
            "unsafe": sum(bool(row["safety"]["unsafe"]) for row in rows),
            "timeout": sum(bool(row["safety"]["timed_out"]) for row in rows),
        },
        "non_clean": {
            "tasks": len(non_clean),
            "success": sum(bool(row["task_success"]) for row in non_clean),
            "success_rate": sum(bool(row["task_success"]) for row in non_clean) / len(non_clean),
        },
        "by_interaction_profile": group("interaction_profile"),
        "by_drift_type": group("drift_type"),
        "by_difficulty": group("difficulty"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Candidate is already frozen: {args.output}")
    rows = [json.loads(line) for line in args.rows.read_text(encoding="utf-8").splitlines() if line.strip()]
    evaluator_summary = json.loads(args.summary.read_text(encoding="utf-8"))
    results = summarize(rows)
    overall = results["overall"]
    schema_only = results["by_interaction_profile"]["schema_only"]
    acceptance = {
        "overall_success_gte_0_20": overall["success_rate"] >= 0.20,
        "non_clean_success_gte_2x_b0": results["non_clean"]["success"] >= 14,
        "schema_only_success_gte_0_10": schema_only["success_rate"] >= 0.10,
        "submission_gte_0_40": overall["submission_rate"] >= 0.40,
        "turn_limit_lte_0_55": overall["turn_limit_rate"] <= 0.55,
        "unsafe_eq_0": overall["unsafe"] == 0,
        "timeout_eq_0": overall["timeout"] == 0,
    }
    if not all(acceptance.values()):
        raise RuntimeError(f"Tune candidate failed acceptance: {acceptance}")

    locked_files = [
        PROJECT_ROOT / "models.lock.json",
        PROJECT_ROOT / "configs/tools/drift_tools.yaml",
        PROJECT_ROOT / "scripts/run_stage6_eval.py",
        PROJECT_ROOT / "scripts/run_five_tool_eval.py",
        PROJECT_ROOT / "driftsql/integrations/state_policy.py",
        PROJECT_ROOT / "driftsql/integrations/verl_tools.py",
        PROJECT_ROOT / "driftsql/integrations/agent_loop.py",
        PROJECT_ROOT / "data/processed/stage6_ablation/b1/tune_agent_eval.jsonl",
    ]
    adapter_files = sorted(path for path in args.adapter.iterdir() if path.is_file())
    payload = {
        "protocol": "driftsql_stage6_frozen_candidate_v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": {
            "name": "b2-step20-typed-diff-dynamic-mask-guards",
            "base_model": {
                "repo_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
                "revision": "c03e6d358207e414f1eca0bb1891e29f1db0e242",
                "path": "models/Qwen2.5-Coder-7B-Instruct",
            },
            "adapter_path": str(args.adapter.resolve().relative_to(PROJECT_ROOT)),
            "adapter_files_sha256": {
                str(path.relative_to(PROJECT_ROOT)): sha256(path) for path in adapter_files
            },
            "inference": {
                "dynamic_tool_mask": True,
                "state_guards": True,
                "terminal_submit_fallback": False,
                "max_turns": 7,
                "max_new_tokens": 512,
                "max_model_len": 8192,
                "tensor_parallel_size": 2,
                "dtype": "bfloat16",
                "decoding": "deterministic evaluator default",
                "enabled_tools": evaluator_summary["enabled_tools"],
            },
        },
        "tune109": {
            "rows_path": str(args.rows.resolve().relative_to(PROJECT_ROOT)),
            "rows_sha256": sha256(args.rows),
            "summary_path": str(args.summary.resolve().relative_to(PROJECT_ROOT)),
            "results": results,
            "acceptance": acceptance,
        },
        "locked_files_sha256": {
            str(path.relative_to(PROJECT_ROOT)): sha256(path) for path in locked_files
        },
        "selection": {
            "selected_over": [
                "B2 step40 (Tune regression)",
                "failure-recovery SFT step20 (Tune regression)",
                "B2 step20 without dynamic routing (lower success and submission)",
            ],
            "grpo_decision": (
                "Do not add B3 before Gate: the Stage5-GRPO-initialized B2 candidate already "
                "clears every threshold, and additional updates showed catastrophic-forgetting risk."
            ),
        },
        "sealed_gate": {
            "dataset": "data/processed/stage6_ablation/b1/gate_agent_eval.jsonl",
            "tasks": 112,
            "access_before_freeze": "not used for inference or model selection",
            "allowed_runs_after_freeze": 1,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "acceptance": acceptance}, indent=2))


if __name__ == "__main__":
    main()
