#!/usr/bin/env python3
"""Produce paired, execution-grounded comparisons for Stage-5 Dev runs."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


SLICE_FIELDS = ("drift_type", "difficulty", "scenario_type", "interaction_profile")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def exact_mcnemar(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(gains, losses) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {"tasks": 0}
    success = sum(bool(row.get("task_success")) for row in rows)
    executable = sum(bool(row.get("executable")) for row in rows)
    submitted = sum(row.get("termination_reason") == "submitted" for row in rows)
    turn_limit = sum(row.get("termination_reason") == "turn_limit" for row in rows)
    unsafe = sum(bool(row.get("safety", {}).get("unsafe")) for row in rows)
    timeout = sum(bool(row.get("safety", {}).get("timed_out")) for row in rows)
    return {
        "tasks": total,
        "task_success": success,
        "task_success_rate": success / total,
        "executable": executable,
        "executable_rate": executable / total,
        "submitted": submitted,
        "turn_limit": turn_limit,
        "unsafe_tasks": unsafe,
        "timeout_tasks": timeout,
        "average_model_calls": sum(int(row.get("usage", {}).get("model_calls", 0)) for row in rows) / total,
        "average_tool_calls": sum(int(row.get("usage", {}).get("tool_calls", 0)) for row in rows) / total,
    }


def slices(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for field in SLICE_FIELDS:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            value = str(row.get(field, "")).strip()
            if value:
                groups[value].append(row)
        if groups:
            result[field] = {key: metrics(value) for key, value in sorted(groups.items())}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help="Repeat alias=path/to/eval_rows.jsonl",
    )
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    variants: dict[str, list[dict[str, Any]]] = {}
    sources = {}
    for spec in args.variant:
        if "=" not in spec:
            parser.error("--variant must use alias=path")
        alias, raw_path = spec.split("=", 1)
        path = Path(raw_path).resolve()
        if alias in variants:
            parser.error(f"Duplicate variant alias: {alias}")
        variants[alias] = load_jsonl(path)
        sources[alias] = str(path)
    if args.reference not in variants:
        parser.error("--reference must name one supplied --variant alias")

    by_id = {
        alias: {str(row["instance_id"]): row for row in rows}
        for alias, rows in variants.items()
    }
    reference_ids = set(by_id[args.reference])
    for alias, rows in by_id.items():
        if set(rows) != reference_ids:
            raise RuntimeError(
                f"Task mismatch for {alias}: missing={len(reference_ids - set(rows))}, "
                f"extra={len(set(rows) - reference_ids)}"
            )

    paired = {}
    reference = by_id[args.reference]
    for alias, candidate in by_id.items():
        if alias == args.reference:
            continue
        gains = sum(
            not bool(reference[key]["task_success"])
            and bool(candidate[key]["task_success"])
            for key in reference_ids
        )
        losses = sum(
            bool(reference[key]["task_success"])
            and not bool(candidate[key]["task_success"])
            for key in reference_ids
        )
        paired[alias] = {
            "gains": gains,
            "losses": losses,
            "net_gain": gains - losses,
            "exact_mcnemar_p": exact_mcnemar(gains, losses),
        }

    result = {
        "protocol": "stage5_paired_execution_eval_v1",
        "reference": args.reference,
        "tasks": len(reference_ids),
        "sources": sources,
        "variants": {
            alias: {"overall": metrics(rows), "slices": slices(rows)}
            for alias, rows in variants.items()
        },
        "paired_vs_reference": paired,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Stage 5 paired execution comparison",
        "",
        "| Variant | Success | Executable | Turn limit | Unsafe | Avg tools | Gains/losses vs ref |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for alias, payload in result["variants"].items():
        item = payload["overall"]
        pair = paired.get(alias)
        pair_text = "reference" if pair is None else f"{pair['gains']}/{pair['losses']}"
        lines.append(
            f"| {alias} | {item['task_success']}/{item['tasks']} | "
            f"{item['executable']}/{item['tasks']} | {item['turn_limit']} | "
            f"{item['unsafe_tasks']} | {item['average_tool_calls']:.2f} | {pair_text} |"
        )
    (args.output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
