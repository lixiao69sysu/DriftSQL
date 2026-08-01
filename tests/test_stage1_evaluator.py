from __future__ import annotations

import sqlite3
from pathlib import Path

from driftsql.evaluation import evaluate_prediction, extract_candidate_sql, summarize_results


def test_extract_candidate_sql_from_supported_surfaces() -> None:
    tagged = '<tool_call>{"name":"submit_solution","arguments":{"sql":"SELECT 1"}}</tool_call>'
    assert extract_candidate_sql(tagged) == "SELECT 1"
    assert extract_candidate_sql("```sql\nSELECT 2\n```") == "SELECT 2"
    assert extract_candidate_sql("<solution>SELECT 3</solution>") == "SELECT 3"


def test_execution_accuracy_and_summary(tmp_path: Path) -> None:
    database = tmp_path / "eval.sqlite"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            "CREATE TABLE items(id INTEGER, name TEXT);"
            "INSERT INTO items VALUES (1, 'A'), (2, 'B');"
        )

    correct = evaluate_prediction(
        predicted_sql="SELECT name FROM items ORDER BY id DESC",
        gold_sql="SELECT name FROM items ORDER BY id",
        db_path=database,
    )
    wrong = evaluate_prediction(
        predicted_sql="SELECT name FROM items WHERE id = 1",
        gold_sql="SELECT name FROM items",
        db_path=database,
    )
    invalid = evaluate_prediction(
        predicted_sql="SELECT missing FROM items",
        gold_sql="SELECT name FROM items",
        db_path=database,
    )
    assert correct == {"correct": True, "pred_executable": True, "error": ""}
    assert wrong["correct"] is False and wrong["pred_executable"] is True
    assert invalid["correct"] is False and invalid["pred_executable"] is False

    report = summarize_results(
        [
            correct | {"difficulty": "simple", "termination_reason": "submitted"},
            wrong | {"difficulty": "moderate", "termination_reason": "submitted"},
            invalid | {"difficulty": "moderate", "termination_reason": "invalid_action"},
        ]
    )
    assert report["overall"]["execution_accuracy"] == 1 / 3
    assert report["overall"]["executable_rate"] == 2 / 3
    assert report["by_difficulty"]["moderate"]["total"] == 2
