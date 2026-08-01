#!/usr/bin/env python3
"""Evaluate standard Stage-1 prediction JSONL with one shared EX implementation."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from driftsql.evaluation import BirdEvalBudget, evaluate_prediction, summarize_results
from driftsql.evaluation.bird import dump_json


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def score_prediction(prediction: dict) -> dict:
    budget = BirdEvalBudget()
    scored = evaluate_prediction(
        predicted_sql=str(prediction.get("final_sql", "")),
        gold_sql=str(prediction["gold_sql"]),
        db_path=Path(prediction["db_path"]),
        timeout_seconds=budget.sql_timeout_seconds,
    )
    usage = prediction.get("usage", {})
    within_budget = (
        int(usage.get("model_calls", 0)) <= budget.max_model_calls
        and int(usage.get("tool_calls", 0)) <= budget.max_tool_calls
        and int(usage.get("sql_executions", 0)) <= budget.max_sql_executions
        and int(usage.get("new_tokens", 0)) <= budget.max_new_tokens
        and int(usage.get("total_tokens", 0)) <= budget.max_total_tokens
    )
    return prediction | scored | {"within_budget": within_budget}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    budget = BirdEvalBudget()
    predictions = load_jsonl(args.predictions)
    if args.workers <= 1:
        results = [score_prediction(prediction) for prediction in predictions]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(executor.map(score_prediction, predictions))

    report = {
        "baseline": results[0].get("baseline") if results else None,
        "budget": budget.to_dict(),
        "summary": summarize_results(results),
        "results": results,
    }
    dump_json(args.output, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
