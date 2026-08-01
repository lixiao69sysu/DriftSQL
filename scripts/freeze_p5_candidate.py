#!/usr/bin/env python3
"""Select from P5 Tune only, then freeze one candidate before opening Gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ALIASES = ("p5-sft20", "p5-grpo-step5", "p5-grpo-step10")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def reference(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def summarize(rows: list[dict[str, Any]], hard_ids: set[str]) -> dict[str, Any]:
    hard = [row for row in rows if str(row["instance_id"]) in hard_ids]
    success = sum(bool(row["task_success"]) for row in rows)
    hard_success = sum(bool(row["task_success"]) for row in hard)
    return {
        "tasks": len(rows),
        "task_success": success,
        "task_success_rate": success / len(rows),
        "turn_limit_focus_tasks": len(hard),
        "turn_limit_focus_success": hard_success,
        "turn_limit_focus_success_rate": hard_success / len(hard),
        "termination_reasons": dict(sorted(Counter(str(row["termination_reason"]) for row in rows).items())),
        "unsafe_tasks": sum(bool(row["safety"]["unsafe"]) for row in rows),
        "timeout_tasks": sum(bool(row["safety"]["timed_out"]) for row in rows),
        "average_model_calls": sum(int(row["usage"]["model_calls"]) for row in rows) / len(rows),
        "average_tool_calls": sum(int(row["usage"]["tool_calls"]) for row in rows) / len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/p5_grpo")
    parser.add_argument("--protocol-dir", type=Path, default=ROOT / "data/processed/p5_isolated_protocol")
    parser.add_argument("--report-root", type=Path, default=ROOT / "reports/p5/tune")
    parser.add_argument(
        "--sft20-adapter",
        type=Path,
        default=ROOT / "checkpoints/stage8_fresh_sft_7b/global_step_20/merged/lora_adapter",
    )
    parser.add_argument("--grpo-root", type=Path, default=ROOT / "checkpoints/p5_reviewed_grpo_7b")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/p5/final_candidate/frozen_candidate.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"P5 candidate is already frozen: {args.output}")

    metadata = load_jsonl(args.data_dir / "tune_agent_eval.jsonl")
    expected_ids = [str(row["extra_info"]["instance_id"]) for row in metadata]
    hard_ids = {
        str(row["extra_info"]["instance_id"])
        for row in metadata
        if bool(row["extra_info"]["p5_turn_limit_focus"])
    }
    if len(expected_ids) != 18 or len(hard_ids) != 12:
        raise RuntimeError("P5 Tune cardinality or turn-limit slice changed")
    adapter_paths = {
        "p5-sft20": args.sft20_adapter,
        "p5-grpo-step5": args.grpo_root / "global_step_5/merged/lora_adapter",
        "p5-grpo-step10": args.grpo_root / "global_step_10/merged/lora_adapter",
    }
    report_paths = {alias: args.report_root / alias / f"{alias}.jsonl" for alias in ALIASES}
    rows_by_alias = {alias: load_jsonl(path) for alias, path in report_paths.items()}
    for alias, rows in rows_by_alias.items():
        ids = [str(row["instance_id"]) for row in rows]
        if ids != expected_ids or len(set(ids)) != 18:
            raise RuntimeError(f"P5 Tune identity/order mismatch for {alias}")
        if not (adapter_paths[alias] / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(adapter_paths[alias])
    results = {alias: summarize(rows, hard_ids) for alias, rows in rows_by_alias.items()}
    eligible = [
        alias
        for alias in ALIASES
        if results[alias]["unsafe_tasks"] == 0 and results[alias]["timeout_tasks"] == 0
    ]
    if not eligible:
        raise RuntimeError("No safe P5 Tune candidate")

    def rank(alias: str) -> tuple[float, ...]:
        result = results[alias]
        return (
            result["task_success"],
            result["turn_limit_focus_success"],
            -result["termination_reasons"].get("turn_limit", 0),
            -result["average_tool_calls"],
            -result["average_model_calls"],
            float(alias == "p5-sft20"),
        )

    selected = max(eligible, key=rank)
    baseline = results["p5-sft20"]
    chosen = results[selected]
    acceptance = {
        "all_candidates_have_18_tune_tasks": all(result["tasks"] == 18 for result in results.values()),
        "turn_limit_slice_has_12_tasks": all(result["turn_limit_focus_tasks"] == 12 for result in results.values()),
        "selected_overall_not_worse_than_sft20": chosen["task_success"] >= baseline["task_success"],
        "selected_hard_slice_not_worse_than_sft20": (
            chosen["turn_limit_focus_success"] >= baseline["turn_limit_focus_success"]
        ),
        "selected_unsafe_eq_0": chosen["unsafe_tasks"] == 0,
        "selected_timeout_eq_0": chosen["timeout_tasks"] == 0,
    }
    if not all(acceptance.values()):
        raise RuntimeError(f"P5 Tune acceptance failed: {acceptance}")

    protocol_summary_path = args.protocol_dir / "summary.json"
    protocol_summary = json.loads(protocol_summary_path.read_text(encoding="utf-8"))
    if protocol_summary["gate"]["status"] != "sealed_unopened":
        raise RuntimeError("P5 Gate is not sealed and unopened")
    selected_adapter = adapter_paths[selected]
    adapter_files = sorted(path for path in selected_adapter.iterdir() if path.is_file())
    locked = [
        ROOT / "models.lock.json",
        ROOT / "configs/tools/drift_tools.yaml",
        ROOT / "driftsql/rewards/agentic.py",
        ROOT / "driftsql/integrations/verl_tools.py",
        ROOT / "driftsql/integrations/state_policy.py",
        ROOT / "driftsql/planning/projection_contract.py",
        ROOT / "scripts/run_five_tool_eval.py",
        ROOT / "scripts/run_stage6_eval.py",
        ROOT / "scripts/run_stage7_process_isolated_eval.py",
        ROOT / "scripts/prepare_p5_grpo.py",
        ROOT / "scripts/verify_p5_training_inputs.py",
        ROOT / "scripts/freeze_p5_candidate.py",
        protocol_summary_path,
        args.data_dir / "summary.json",
        args.data_dir / "tune_agent_eval.jsonl",
        *report_paths.values(),
    ]
    payload = {
        "protocol": "driftsql_p5_tune_frozen_candidate_v1",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "candidate": {
            "name": selected,
            "base_model": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "adapter_path": reference(selected_adapter),
            "adapter_files_sha256": {
                reference(path): sha256(path) for path in adapter_files
            },
            "inference": {
                "max_turns": 7,
                "max_new_tokens": 512,
                "max_model_len": 8192,
                "dynamic_tool_mask": True,
                "state_guards": True,
                "isolation": "one OS process and one vLLM engine per episode",
            },
        },
        "tune_selection": {
            "results": results,
            "rank_rule": "success, hard-slice success, fewer turn limits/tools/model calls; SFT wins exact tie",
            "selected": selected,
            "acceptance": acceptance,
            "artifacts_sha256": {
                reference(path): sha256(path) for path in report_paths.values()
            },
        },
        "locked_files_sha256": {
            reference(path): sha256(path) for path in locked
        },
        "one_shot_gate": {
            "rows": 18,
            "turn_limit_focus_rows": 12,
            "sealed_input_sha256": protocol_summary["gate"]["sha256"],
            "allowed_candidate_runs": 1,
            "acceptance_precommitted": {
                "overall_success_gte_0_50": 0.50,
                "turn_limit_focus_success_gte_0_33": 1 / 3,
                "turn_limit_terminations_lte_6": 6,
                "unsafe_eq_0": 0,
                "timeout_eq_0": 0,
            },
        },
        "gate55_policy": "Stage-8 Gate55 remains permanently sealed and was not read for P5 selection.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "selected": selected,
                "acceptance": acceptance,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
