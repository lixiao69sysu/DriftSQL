#!/usr/bin/env python3
"""Compare Stage 4 GRPO with the locked Base and Tool-SFT baselines."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from sqlglot import exp, parse_one


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINES = PROJECT_ROOT / "reports/stage3/five_tool_eval_native_v4_json_final/summary.json"
DEFAULT_SFT_ROWS = PROJECT_ROOT / "reports/stage3/five_tool_eval_native_v4_json_final/tool-json-v4-step80.jsonl"
DEFAULT_CANDIDATE_ROWS = PROJECT_ROOT / "reports/stage4/five_tool_eval/tool-grpo-step40.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports/stage4/comparison"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def unsafe_sql(sql: str) -> bool:
    if not sql.strip():
        return False
    if sql.lstrip().upper().startswith("EXPLAIN"):
        return False
    try:
        expression = parse_one(sql, read="sqlite")
    except Exception:
        return False
    return not isinstance(expression, (exp.Query, exp.Subquery))


def row_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    unsafe_tasks = 0
    timeout_tasks = 0
    for row in rows:
        trajectory = row.get("trajectory", []) or []
        sqls = [
            str(item.get("arguments", {}).get("sql", ""))
            for item in trajectory
            if item.get("tool_name") in {"execute_sql", "submit_solution"}
        ]
        unsafe_tasks += int(any(unsafe_sql(sql) for sql in sqls))
        timeout_tasks += int(
            any(
                "timeout" in str(item.get("metrics", {}).get("execution_error", "")).casefold()
                or "interrupt" in str(item.get("metrics", {}).get("execution_error", "")).casefold()
                for item in trajectory
            )
        )
    return {
        "tasks": len(rows),
        "task_success": sum(bool(row.get("task_success")) for row in rows),
        "executable": sum(bool(row.get("executable")) for row in rows),
        "turn_limit": sum(row.get("termination_reason") == "turn_limit" for row in rows),
        "unsafe_tasks": unsafe_tasks,
        "timeout_tasks": timeout_tasks,
        "average_tool_calls": sum(int(row.get("usage", {}).get("tool_calls", 0)) for row in rows)
        / len(rows),
    }


def exact_mcnemar(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(gains, losses) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--sft-rows", type=Path, default=DEFAULT_SFT_ROWS)
    parser.add_argument("--candidate-rows", type=Path, default=DEFAULT_CANDIDATE_ROWS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--minimum-turn-limit-reduction", type=float, default=0.20)
    args = parser.parse_args()

    baselines = json.loads(args.baseline_summary.read_text(encoding="utf-8"))["variants"]
    base = next(row for row in baselines if row["variant"] == "qwen2.5-coder-3b-base")
    sft_rows = load_jsonl(args.sft_rows)
    candidate_rows = load_jsonl(args.candidate_rows)
    sft_by_id = {str(row["instance_id"]): row for row in sft_rows}
    candidate_by_id = {str(row["instance_id"]): row for row in candidate_rows}
    if set(sft_by_id) != set(candidate_by_id):
        missing = sorted(set(sft_by_id) - set(candidate_by_id))
        extra = sorted(set(candidate_by_id) - set(sft_by_id))
        raise RuntimeError(f"Evaluation task mismatch: missing={missing}, extra={extra}")

    sft = row_metrics(sft_rows)
    candidate = row_metrics(candidate_rows)
    gains = sum(
        not bool(sft_by_id[key]["task_success"]) and bool(candidate_by_id[key]["task_success"])
        for key in sft_by_id
    )
    losses = sum(
        bool(sft_by_id[key]["task_success"]) and not bool(candidate_by_id[key]["task_success"])
        for key in sft_by_id
    )
    required_turn_limit = math.floor(
        sft["turn_limit"] * (1.0 - args.minimum_turn_limit_reduction)
    )
    gates = {
        "same_78_tasks": candidate["tasks"] == sft["tasks"] == 78,
        "correctness_not_regressed": candidate["task_success"] >= sft["task_success"],
        "turn_limit_reduced_at_least_20pct": candidate["turn_limit"] <= required_turn_limit,
        "unsafe_not_regressed": candidate["unsafe_tasks"] <= sft["unsafe_tasks"],
    }
    result = {
        "protocol": "stage4_five_tool_grpo_promotion_v1",
        "base": {
            "tasks": int(base["tasks"]),
            "task_success": int(base["task_success"]),
            "turn_limit": int(base.get("turn_limit", base["termination_reasons"].get("turn_limit", 0))),
        },
        "tool_sft": sft,
        "tool_sft_grpo": candidate,
        "paired": {
            "grpo_gains": gains,
            "grpo_losses": losses,
            "exact_mcnemar_p": exact_mcnemar(gains, losses),
        },
        "gates": gates,
        "passed": all(gates.values()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown = f"""# Stage 4 GRPO comparison

| Model | Success | Executable | Turn limit | Unsafe tasks | Avg tool calls |
|---|---:|---:|---:|---:|---:|
| Base 3B | {base['task_success']}/78 | {base['executable']}/78 | {base.get('turn_limit', base['termination_reasons'].get('turn_limit', 0))} | n/a | {base['average_tool_calls']:.2f} |
| Tool-SFT step 80 | {sft['task_success']}/78 | {sft['executable']}/78 | {sft['turn_limit']} | {sft['unsafe_tasks']} | {sft['average_tool_calls']:.2f} |
| Tool-SFT + GRPO | {candidate['task_success']}/78 | {candidate['executable']}/78 | {candidate['turn_limit']} | {candidate['unsafe_tasks']} | {candidate['average_tool_calls']:.2f} |

- Paired GRPO gains/losses: {gains}/{losses}; exact McNemar p={result['paired']['exact_mcnemar_p']:.6g}.
- Promotion gate: **{'PASS' if result['passed'] else 'FAIL'}**.
- Gate details: `{json.dumps(gates, ensure_ascii=False)}`
"""
    (args.output_dir / "comparison.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
