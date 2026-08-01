from __future__ import annotations

from driftsql.evaluation.reasoning import (
    extract_reasoning_sql,
    reasoning_format,
    stratified_database_sample,
)


def test_reasoning_parser_prefers_sql_wrapper_and_tracks_format() -> None:
    response = "<plan>\n1. Read t.\n</plan>\n<sql>\nSELECT value FROM t\n</sql>"
    assert extract_reasoning_sql(response) == "SELECT value FROM t"
    assert reasoning_format(response) == {
        "plan_tag": True,
        "sql_tag": True,
        "exact_wrapper": True,
    }
    assert extract_reasoning_sql("```sql\nSELECT 1\n```") == "SELECT 1"


def test_reasoning_eval_sample_is_stable_and_database_stratified() -> None:
    rows = [
        {"db_id": db_id, "source_index": index, "question": f"q{index}"}
        for index, db_id in enumerate(["a"] * 5 + ["b"] * 5 + ["c"] * 5)
    ]
    first = stratified_database_sample(rows, 6)
    second = stratified_database_sample(list(reversed(rows)), 6)
    assert [row["source_index"] for row in first] == [row["source_index"] for row in second]
    assert {row["db_id"] for row in first} == {"a", "b", "c"}
    assert all(sum(row["db_id"] == db_id for row in first) == 2 for db_id in "abc")
