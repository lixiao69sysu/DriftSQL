#!/usr/bin/env python3
"""Build paired raw/controller Tune432 checkpoint comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_p6_process_isolated_eval import apply_controller, requested_metrics, write_jsonl
from summarize_p6_eval_matrix import summarize_variant


ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def selection_key(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(metrics["unsafe_tasks"] == 0 and metrics["timeout_tasks"] == 0),
        float(metrics["execution_success_rate"]),
        float(metrics["drift_recovery_rate"]),
        -float(metrics["average_tool_calls"]),
    )


def paired_against(
    reference: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
) -> dict[str, int]:
    ids = set(reference)
    gains = sum(
        not bool(reference[key].get("task_success"))
        and bool(candidate[key].get("task_success"))
        for key in ids
    )
    losses = sum(
        bool(reference[key].get("task_success"))
        and not bool(candidate[key].get("task_success"))
        for key in ids
    )
    return {"gains": gains, "losses": losses, "net_gain": gains - losses}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variant", action="append", required=True, help="alias=raw.jsonl")
    parser.add_argument("--temporary-root", type=Path, default=ROOT / "data/tmp")
    args = parser.parse_args()

    records = load_jsonl(args.data.resolve())
    expected_ids = [str(row["extra_info"]["instance_id"]) for row in records]
    if len(records) != 432:
        raise RuntimeError(f"Expected Tune432, got {len(records)}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    controlled_root = args.output_root / "controlled"
    controlled_root.mkdir(exist_ok=True)

    raw_rows: dict[str, list[dict[str, Any]]] = {}
    controlled_rows: dict[str, list[dict[str, Any]]] = {}
    sources: dict[str, str] = {}
    prefix_equivalence: dict[str, dict[str, int]] = {}
    for specification in args.variant:
        alias, separator, raw_path = specification.partition("=")
        if not separator or not alias or alias in raw_rows:
            raise ValueError(f"Invalid or duplicate variant: {specification}")
        path = Path(raw_path).resolve()
        rows = load_jsonl(path)
        actual_ids = [str(row["instance_id"]) for row in rows]
        if len(rows) != 432 or actual_ids != expected_ids:
            raise RuntimeError(f"Tune432 identity/order mismatch: {alias}")
        partial_path = path.parent / f".{alias}.partial.jsonl"
        partial = load_jsonl(partial_path) if partial_path.is_file() else []
        outcome_fields = ("instance_id", "task_success", "executable", "termination_reason", "final_sql")
        mismatches = sum(
            any(reference.get(field) != rows[index].get(field) for field in outcome_fields)
            for index, reference in enumerate(partial)
        )
        prefix_equivalence[alias] = {
            "episode_major_reference_tasks": len(partial),
            "outcome_mismatches": mismatches,
        }
        if mismatches:
            raise RuntimeError(
                f"Batched/episode-major deterministic prefix mismatch for {alias}: {mismatches}"
            )
        controlled, decisions = apply_controller(
            rows,
            records,
            temporary_root=args.temporary_root.resolve(),
        )
        raw_rows[alias] = rows
        controlled_rows[alias] = controlled
        sources[alias] = str(path)
        write_jsonl(controlled_root / f"{alias}.jsonl", controlled)
        write_jsonl(controlled_root / f"{alias}_decisions.jsonl", decisions)

    raw_metrics = {alias: requested_metrics(rows) for alias, rows in raw_rows.items()}
    controller_metrics = {
        alias: requested_metrics(rows) for alias, rows in controlled_rows.items()
    }
    raw_selected = max(raw_metrics, key=lambda alias: selection_key(raw_metrics[alias]))
    controller_selected = max(
        controller_metrics,
        key=lambda alias: selection_key(controller_metrics[alias]),
    )
    reference_raw = {str(row["instance_id"]): row for row in raw_rows["sft160"]}
    reference_controlled = {
        str(row["instance_id"]): row for row in controlled_rows["sft160"]
    }
    paired = {
        alias: {
            "raw_vs_sft160": paired_against(
                reference_raw,
                {str(row["instance_id"]): row for row in raw_rows[alias]},
            ),
            "controller_vs_sft160": paired_against(
                reference_controlled,
                {str(row["instance_id"]): row for row in controlled_rows[alias]},
            ),
        }
        for alias in raw_rows
        if alias != "sft160"
    }
    stratified_raw = {
        alias: summarize_variant(alias, Path(sources[alias])) for alias in raw_rows
    }
    stratified_controller = {
        alias: summarize_variant(alias, controlled_root / f"{alias}.jsonl")
        for alias in controlled_rows
    }
    for values in (stratified_raw, stratified_controller):
        for row in values.values():
            row.pop("instance_ids", None)

    result = {
        "protocol": "p6_scaleup_tune432_checkpoint_matrix_v1",
        "data": str(args.data.resolve()),
        "tasks": 432,
        "fresh_blind_reads": 0,
        "inference": {
            "temperature": 0.0,
            "seed": 42,
            "max_turns": 7,
            "max_new_tokens": 512,
            "max_model_len": 6144,
            "state_guards": True,
            "dynamic_tool_mask": True,
            "batch_size": 32,
            "episode_chunk_size": 32,
            "episode_major": False,
            "async_scheduling": False,
            "prefix_caching": False,
        },
        "selection_rule": (
            "zero unsafe/timeouts, then execution success, drift recovery, lower tool calls"
        ),
        "sources": sources,
        "batched_prefix_equivalence": prefix_equivalence,
        "raw_metrics": raw_metrics,
        "controller_metrics": controller_metrics,
        "raw_selected": raw_selected,
        "controller_selected": controller_selected,
        "paired": paired,
        "stratified_raw": stratified_raw,
        "stratified_controller": stratified_controller,
    }
    (args.output_root / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# P6 Scale-up Tune432 checkpoint comparison",
        "",
        "All variants use identical Tune432 tasks, deterministic decoding, seven tools, "
        "state guards, dynamic tool masks, and a seven-turn budget.",
        "",
        "| Variant | Raw success | Raw drift | Raw safe submit | Raw tools | Controller success | Controller safe submit | Unsafe / timeout |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for alias in raw_rows:
        raw = raw_metrics[alias]
        controlled = controller_metrics[alias]
        lines.append(
            f"| {alias} | {raw['execution_success']}/{raw['tasks']} "
            f"({raw['execution_success_rate']:.2%}) | "
            f"{raw['drift_recovery']}/{raw['drift_tasks']} "
            f"({raw['drift_recovery_rate']:.2%}) | "
            f"{raw['safe_submitted']}/{raw['tasks']} "
            f"({raw['safe_submission_rate']:.2%}) | "
            f"{raw['average_tool_calls']:.2f} | "
            f"{controlled['execution_success']}/{controlled['tasks']} "
            f"({controlled['execution_success_rate']:.2%}) | "
            f"{controlled['safe_submitted']}/{controlled['tasks']} "
            f"({controlled['safe_submission_rate']:.2%}) | "
            f"{raw['unsafe_tasks']} / {raw['timeout_tasks']} |"
        )
    lines.extend(
        [
            "",
            f"Raw model selected: **{raw_selected}**",
            "",
            f"Controller-assisted selected: **{controller_selected}**",
            "",
            "Fresh Blind320 reads: **0**",
        ]
    )
    (args.output_root / "comparison.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "raw_selected": raw_selected,
        "controller_selected": controller_selected,
        "tasks": 432,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
