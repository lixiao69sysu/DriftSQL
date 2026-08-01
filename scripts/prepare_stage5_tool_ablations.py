#!/usr/bin/env python3
"""Build tool-availability ablations without changing the Stage-5 DB split.

Only ``extra_info.tool_selection`` and the matching system instruction are
changed.  Labels, database identities, fingerprints, and tool state remain
byte-for-byte equivalent at the Python-object level so the comparison isolates
the availability of ``ask_user`` or the business-knowledge (HKB) retriever.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


VARIANTS = {
    "no_ask_user": "ask_user",
    "no_hkb": "get_knowledge_definition",
}


SYSTEM_PROMPTS = {
    "no_ask_user": """You are a production analytics SQL agent.
The user's request can contain genuine ambiguity and organization-specific
business concepts. You cannot ask the user follow-up questions: retrieve the
active schema and available business knowledge, validate SQL in the isolated
database session, and submit one final read-only query.

Do not invent user requirements or metric definitions. Use
get_knowledge_definition for documented business knowledge. Every assistant
turn must contain one concise <think> block followed by exactly one tool call.
Finish with submit_solution.""",
    "no_hkb": """You are a production analytics SQL agent.
The user's request can contain genuine ambiguity and organization-specific
business concepts. No business-knowledge retriever is available: ask one
focused clarification question when required, retrieve the active schema,
validate SQL in the isolated database session, and submit one final read-only
query.

Do not invent user requirements or metric definitions. Use ask_user only for
ambiguities in the request. Every assistant turn must contain one concise
<think> block followed by exactly one tool call. Finish with submit_solution.""",
}


def ablate_row(row: dict[str, Any], variant: str) -> dict[str, Any]:
    removed_tool = VARIANTS[variant]
    result = dict(row)
    result["prompt"] = [dict(message) for message in row["prompt"]]
    if not result["prompt"] or result["prompt"][0].get("role") != "system":
        raise ValueError("Expected the first prompt message to be the system instruction")
    result["prompt"][0]["content"] = SYSTEM_PROMPTS[variant]

    result["extra_info"] = dict(row["extra_info"])
    selected = list(result["extra_info"].get("tool_selection") or [])
    if removed_tool not in selected:
        raise ValueError(
            f"{result['extra_info'].get('instance_id')}: {removed_tool} is not selected"
        )
    result["extra_info"]["tool_selection"] = [
        name for name in selected if name != removed_tool
    ]
    return result


def write_variant(source: Path, destination: Path, variant: str) -> dict[str, Any]:
    table = pq.read_table(source)
    rows = [ablate_row(row, variant) for row in table.to_pylist()]
    output = pa.Table.from_pylist(rows, schema=table.schema)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(output, destination, compression="zstd")
    return {
        "source": str(source.resolve()),
        "output": str(destination.resolve()),
        "rows": len(rows),
        "removed_tool": VARIANTS[variant],
        "db_ids": len({row["extra_info"]["db_id"] for row in rows}),
        "instance_ids": len({row["extra_info"]["instance_id"] for row in rows}),
    }


def write_eval_variant(source: Path, destination: Path, variant: str) -> dict[str, Any]:
    rows = [
        ablate_row(json.loads(line), variant)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(destination)
    return {
        "source": str(source.resolve()),
        "output": str(destination.resolve()),
        "rows": len(rows),
        "removed_tool": VARIANTS[variant],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed/stratified_five_tool_v2"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/stage5_tool_ablations"),
    )
    args = parser.parse_args()

    summary: dict[str, Any] = {"protocol": "stage5_tool_ablation_v1", "variants": {}}
    for variant in VARIANTS:
        variant_summary = {}
        for split in ("train", "dev"):
            variant_summary[split] = write_variant(
                args.input_dir / f"rl_{split}.parquet",
                args.output_dir / variant / f"rl_{split}.parquet",
                variant,
            )
        variant_summary["dev_agent_eval"] = write_eval_variant(
            args.input_dir / "dev_agent_eval.jsonl",
            args.output_dir / variant / "dev_agent_eval.jsonl",
            variant,
        )
        summary["variants"][variant] = variant_summary

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
