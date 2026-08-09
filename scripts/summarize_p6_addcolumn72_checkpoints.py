#!/usr/bin/env python3
"""Summarize deterministic AddColumn72 checkpoint evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalized_sql(sql: str) -> str:
    return sql.rstrip(";").strip().casefold()


def summarize(
    alias: str,
    rows: list[dict[str, Any]],
    source_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    ordered = 0
    inspect_before_execute = 0
    stale_shortcuts = 0
    safe_submitted = 0
    unsafe = 0
    timed_out = 0
    tool_total = 0
    for row in rows:
        tools = list(row.get("called_tools", []) or [])
        tool_total += len(tools)
        try:
            version_index = tools.index("get_schema_version")
            diff_index = tools.index("inspect_schema_diff")
            is_ordered = version_index < diff_index
        except ValueError:
            diff_index = -1
            is_ordered = False
        ordered += is_ordered
        try:
            execute_index = tools.index("execute_sql")
        except ValueError:
            execute_index = -1
        inspect_before_execute += bool(is_ordered and execute_index > diff_index)
        source = source_by_id[str(row["instance_id"])]
        stale_sql = str(source["extra_info"].get("stale_sql", ""))
        stale_shortcuts += bool(
            tools == ["execute_sql", "submit_solution"]
            and normalized_sql(str(row.get("final_sql", ""))) == normalized_sql(stale_sql)
        )
        safety = row.get("safety", {}) or {}
        row_unsafe = bool(safety.get("unsafe"))
        row_timeout = bool(safety.get("timed_out"))
        unsafe += row_unsafe
        timed_out += row_timeout
        safe_submitted += bool(
            row.get("termination_reason") in {"submitted", "fallback_submitted"}
            and row.get("final_sql")
            and not row_unsafe
        )
    total = len(rows)
    return {
        "variant": alias,
        "tasks": total,
        "task_success": sum(bool(row.get("task_success")) for row in rows),
        "executable": sum(bool(row.get("executable")) for row in rows),
        "safe_submitted": safe_submitted,
        "ordered_inspection": ordered,
        "inspect_before_execute": inspect_before_execute,
        "stale_execute_submit_shortcut": stale_shortcuts,
        "unsafe_tasks": unsafe,
        "timeout_tasks": timed_out,
        "average_tool_calls": tool_total / total,
    }


def selection_key(metrics: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(metrics["unsafe_tasks"] == 0 and metrics["timeout_tasks"] == 0),
        float(metrics["task_success"]),
        float(metrics["ordered_inspection"]),
        -float(metrics["stale_execute_submit_shortcut"]),
        float(metrics["safe_submitted"]),
        -float(metrics["average_tool_calls"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variant", action="append", required=True, help="alias=result.jsonl")
    parser.add_argument(
        "--title",
        default="P6 AddColumn72 SFT checkpoint curve",
        help="Markdown report title.",
    )
    parser.add_argument(
        "--protocol",
        default="p6_addcolumn72_sft_checkpoint_curve_v1",
        help="Machine-readable report protocol identifier.",
    )
    parser.add_argument(
        "--selection-label",
        default="Selected diagnostic checkpoint",
        help="Label used for the selected variant in Markdown.",
    )
    args = parser.parse_args()

    source = [
        row
        for row in load_jsonl(args.data.resolve())
        if str(row["extra_info"].get("drift_type", "")) == "add_column"
    ]
    if len(source) != 72:
        raise RuntimeError(f"Expected AddColumn72 source rows, got {len(source)}")
    expected_ids = [str(row["extra_info"]["instance_id"]) for row in source]
    source_by_id = {str(row["extra_info"]["instance_id"]): row for row in source}

    metrics: dict[str, dict[str, Any]] = {}
    sources: dict[str, str] = {}
    for specification in args.variant:
        alias, separator, raw_path = specification.partition("=")
        if not separator or not alias or alias in metrics:
            raise ValueError(f"Invalid or duplicate variant: {specification}")
        path = Path(raw_path).resolve()
        rows = load_jsonl(path)
        actual_ids = [str(row.get("instance_id", "")) for row in rows]
        if len(rows) != 72 or actual_ids != expected_ids:
            raise RuntimeError(f"AddColumn72 identity/order mismatch: {alias}")
        metrics[alias] = summarize(alias, rows, source_by_id)
        sources[alias] = str(path)

    selected = max(metrics, key=lambda alias: selection_key(metrics[alias]))
    result = {
        "protocol": args.protocol,
        "data": str(args.data.resolve()),
        "tasks": 72,
        "inference": {
            "temperature": 0.0,
            "seed": 42,
            "max_turns": 7,
            "state_guards": True,
            "dynamic_tool_mask": True,
        },
        "sources": sources,
        "metrics": metrics,
        "selected": selected,
        "fresh_blind_reads": 0,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        f"# {args.title}",
        "",
        "| Variant | Success | Executable | Safe submit | Ordered inspect | Inspect before execute | Stale shortcut | Tools | Unsafe/timeout |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for alias, row in metrics.items():
        lines.append(
            f"| {alias} | {row['task_success']}/72 | {row['executable']}/72 | "
            f"{row['safe_submitted']}/72 | {row['ordered_inspection']}/72 | "
            f"{row['inspect_before_execute']}/72 | "
            f"{row['stale_execute_submit_shortcut']}/72 | "
            f"{row['average_tool_calls']:.2f} | "
            f"{row['unsafe_tasks']}/{row['timeout_tasks']} |"
        )
    lines.extend(
        ["", f"{args.selection_label}: **{selected}**", "", "Fresh Blind320 reads: **0**"]
    )
    (args.output_root / "comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"selected": selected, "tasks": 72}, ensure_ascii=False))


if __name__ == "__main__":
    main()
