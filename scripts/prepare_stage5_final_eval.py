#!/usr/bin/env python3
"""Materialize the frozen no-ask-user Test/Frozen evaluation inputs once."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from prepare_stage5_tool_ablations import write_eval_variant


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_ids(path: Path) -> list[str]:
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row: dict[str, Any] = json.loads(line)
            ids.append(str(row["extra_info"]["instance_id"]))
    return ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed/stratified_five_tool_v2",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed/stage5_final_sealed/no_ask_user",
    )
    args = parser.parse_args()

    sources = {
        "test181": args.input_dir / "test_agent_eval.jsonl",
        "frozen78": args.input_dir / "frozen_regression_78_agent_eval.jsonl",
    }
    destinations = {
        "test181": args.output_dir / "test_agent_eval.jsonl",
        "frozen78": args.output_dir / "frozen_regression_78_agent_eval.jsonl",
    }
    existing = [str(path) for path in destinations.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite sealed evaluation inputs: " + ", ".join(existing)
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    materialized: dict[str, Any] = {}
    for name in ("test181", "frozen78"):
        materialized[name] = write_eval_variant(
            sources[name], destinations[name], "no_ask_user"
        )

    test_ids = load_ids(destinations["test181"])
    frozen_ids = load_ids(destinations["frozen78"])
    if len(test_ids) != 181 or len(set(test_ids)) != 181:
        raise RuntimeError(f"Expected 181 unique Test IDs, got {len(set(test_ids))}")
    if len(frozen_ids) != 78 or len(set(frozen_ids)) != 78:
        raise RuntimeError(f"Expected 78 unique frozen IDs, got {len(set(frozen_ids))}")
    if not set(frozen_ids).issubset(set(test_ids)):
        raise RuntimeError("Frozen78 is not a subset of Test181")

    for name in materialized:
        materialized[name]["source_sha256"] = sha256(sources[name])
        materialized[name]["output_sha256"] = sha256(destinations[name])
    summary = {
        "protocol": "stage5_final_eval_input_v1",
        "status": "materialized_after_model_freeze",
        "variant": "no_ask_user",
        "invariants": {
            "test_unique_ids": len(set(test_ids)),
            "frozen_unique_ids": len(set(frozen_ids)),
            "frozen_is_test_subset": True,
        },
        "artifacts": materialized,
    }
    summary_path = args.output_dir / "preparation_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
