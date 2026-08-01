#!/usr/bin/env python3
"""Build paired B0/B1 records for the Stage 6 optimization protocol.

B0 preserves the selected Stage 5 no-ask-user policy.  B1 changes only the
system instruction and exposes the already implemented schema-version and
audited-diff tools.  Task payloads, fingerprints, and database identities stay
identical, which makes the comparison causal.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    PROJECT_ROOT / "data/processed/stage5_tool_ablations/no_ask_user/rl_train.parquet"
)
DEFAULT_SPLITS = PROJECT_ROOT / "data/processed/stage6_protocol"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/processed/stage6_ablation"
SPLITS = ("train", "tune", "gate")
B0_TOOLS = (
    "get_schema",
    "get_knowledge_definition",
    "execute_sql",
    "submit_solution",
)
B1_TOOLS = (
    "get_schema_version",
    "inspect_schema_diff",
    "get_schema",
    "get_knowledge_definition",
    "execute_sql",
    "submit_solution",
)

B1_SYSTEM_PROMPT = """You are a production SQL recovery agent operating on a versioned database.
Previously valid SQL may contain stale identifiers or a stale result-column contract. You cannot ask
the user follow-up questions. First check the active schema version. When the database has changed,
inspect the audited schema diff before editing SQL. Retrieve the active schema and governed business
knowledge only when they are needed, execute a read-only candidate for validation, then submit it.

Never guess a renamed identifier that is available in inspect_schema_diff. Do not repeat an identical
tool call or SQL execution. After a successful execute_sql, either repair the query using new evidence
or submit the validated SQL. Every assistant turn must contain one concise <think> block followed by
exactly one tool call. Finish with submit_solution within the tool budget."""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def build_b1(row: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(row)
    if not result["prompt"] or result["prompt"][0].get("role") != "system":
        raise ValueError("Expected a system message at prompt[0]")
    result["prompt"][0]["content"] = B1_SYSTEM_PROMPT
    extra = result["extra_info"]
    kwargs = extra["tools_kwargs"]
    state = copy.deepcopy(kwargs["execute_sql"]["create_kwargs"])
    for name in ("get_schema_version", "inspect_schema_diff"):
        kwargs[name] = {"create_kwargs": copy.deepcopy(state)}
    extra["tool_selection"] = list(B1_TOOLS)
    extra["stage6_variant"] = "b1_explicit_schema_diff"
    return result


def build_b0(row: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(row)
    selected = tuple(result["extra_info"]["tool_selection"])
    if selected != B0_TOOLS:
        raise ValueError(f"Unexpected Stage 5 no-ask tool order: {selected}")
    result["extra_info"]["stage6_variant"] = "b0_stage5_selected_policy"
    return result


def task_id(record: dict[str, Any]) -> str:
    return str(record["extra_info"]["instance_id"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source_rows = pq.read_table(args.source).to_pylist()
    by_id = {task_id(row): row for row in source_rows}
    if len(by_id) != len(source_rows):
        raise RuntimeError("Duplicate instance IDs in Stage 5 source parquet")

    summary: dict[str, Any] = {
        "name": "driftsql_stage6_paired_b0_b1_v1",
        "source": str(args.source.resolve()),
        "variants": {},
    }
    all_selected: set[str] = set()
    for variant, transform, tool_names in (
        ("b0", build_b0, B0_TOOLS),
        ("b1", build_b1, B1_TOOLS),
    ):
        variant_summary: dict[str, Any] = {"tools": list(tool_names), "splits": {}}
        for split in SPLITS:
            split_rows = load_jsonl(args.split_dir / f"{split}.jsonl")
            split_ids = [str(row["task_id"]) for row in split_rows]
            missing = sorted(set(split_ids) - set(by_id))
            if missing:
                raise RuntimeError(f"{split}: tasks missing from Stage 5 source: {missing[:5]}")
            records = [transform(by_id[instance_id]) for instance_id in split_ids]
            output_dir = args.output_dir / variant
            output_dir.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                pa.Table.from_pylist(records),
                output_dir / f"rl_{split}.parquet",
                compression="zstd",
            )
            write_jsonl(output_dir / f"{split}_agent_eval.jsonl", records)
            ids = {task_id(row) for row in records}
            if ids != set(split_ids):
                raise RuntimeError(f"{variant}/{split}: output task IDs changed")
            if variant == "b0":
                if all_selected & ids:
                    raise RuntimeError(f"Stage 6 split task overlap detected in {split}")
                all_selected.update(ids)
            variant_summary["splits"][split] = {
                "rows": len(records),
                "databases": len({row["extra_info"]["db_id"] for row in records}),
            }
        summary["variants"][variant] = variant_summary

    summary["paired_task_ids"] = len(all_selected)
    summary["gate_policy"] = "Prepared but prohibited from inference until candidate freeze."
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
