#!/usr/bin/env python3
"""Derive Frozen78 metrics from the single Test181 inference output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from compare_stage5_eval import slices
from run_five_tool_eval import load_jsonl, summarize, write_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-results", type=Path, required=True)
    parser.add_argument("--frozen-data", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports/stage5/final_sealed_eval",
    )
    parser.add_argument("--alias", default="final-no-ask-user")
    args = parser.parse_args()

    test_rows: list[dict[str, Any]] = load_jsonl(args.test_results)
    frozen_source: list[dict[str, Any]] = load_jsonl(args.frozen_data)
    test_ids = [str(row["instance_id"]) for row in test_rows]
    frozen_ids = {
        str(row["extra_info"]["instance_id"])
        for row in frozen_source
    }
    if len(test_rows) != 181 or len(set(test_ids)) != 181:
        raise RuntimeError("Expected exactly 181 unique Test inference rows")
    if len(frozen_ids) != 78:
        raise RuntimeError("Expected exactly 78 unique frozen IDs")
    frozen_rows = [row for row in test_rows if str(row["instance_id"]) in frozen_ids]
    if len(frozen_rows) != 78:
        missing = sorted(frozen_ids - set(test_ids))
        raise RuntimeError(f"Frozen results missing from Test output: {missing[:5]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "frozen78_from_test181.jsonl", frozen_rows)
    test_summary = summarize(args.alias, test_rows)
    frozen_summary = summarize(args.alias, frozen_rows)
    payload = {
        "protocol": "stage5_final_sealed_eval_v1",
        "inference_passes": {
            "test181": 1,
            "frozen78": 0,
            "note": "Frozen78 metrics are filtered from the single Test181 output.",
        },
        "test181": {"overall": test_summary, "slices": slices(test_rows)},
        "frozen78": {"overall": frozen_summary, "slices": slices(frozen_rows)},
    }
    (args.output_dir / "sealed_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
