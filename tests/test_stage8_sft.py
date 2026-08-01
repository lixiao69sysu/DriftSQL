from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/stage8_fresh_sft"
PROTOCOL = ROOT / "data/processed/stage8_fresh_protocol"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_stage8_sft_counts_weights_and_db_isolation() -> None:
    summary = json.loads((DATA / "summary.json").read_text(encoding="utf-8"))
    assert summary["stage7_gate106_read"] is False
    assert summary["stage8_gate_read"] is False
    assert summary["train_tune_database_overlap"] == []
    assert summary["splits"]["train"]["supervision_examples"] == 2120
    assert summary["splits"]["tune"]["supervision_examples"] == 250
    assert summary["splits"]["train"]["target_actions"]["submit_solution"] == 900
    assert pq.read_table(DATA / "train.parquet").num_rows == 2120
    assert pq.read_table(DATA / "tune.parquet").num_rows == 250


def test_stage8_agent_records_match_train_tune_protocol_only() -> None:
    split_dbs = {}
    for split, expected in (("train", 220), ("tune", 55)):
        records = load_jsonl(DATA / f"{split}_agent_eval.jsonl")
        assert len(records) == expected
        ids = {str(row["extra_info"]["instance_id"]) for row in records}
        protocol_rows = (
            load_jsonl(PROTOCOL / f"{split}_add_column.jsonl")
            + load_jsonl(PROTOCOL / f"{split}_general_replay.jsonl")
        )
        assert ids == {str(row["task_id"]) for row in protocol_rows}
        split_dbs[split] = {str(row["extra_info"]["db_id"]) for row in records}
        assert all(row["extra_info"]["stage8_variant"] for row in records)
        assert all(
            state["create_kwargs"]["metric_version"] == "stage8-v1"
            for row in records
            for state in row["extra_info"]["tools_kwargs"].values()
        )
    gate_rows = (
        load_jsonl(PROTOCOL / "gate_add_column.jsonl")
        + load_jsonl(PROTOCOL / "gate_general_replay.jsonl")
    )
    gate_ids = {str(row["task_id"]) for row in gate_rows}
    train_tune_ids = {
        str(row["extra_info"]["instance_id"])
        for split in ("train", "tune")
        for row in load_jsonl(DATA / f"{split}_agent_eval.jsonl")
    }
    assert not (train_tune_ids & gate_ids)
    assert not (split_dbs["train"] & split_dbs["tune"])


def test_stage8_submit_weighting_is_stronger_for_add_column() -> None:
    rows = pq.read_table(DATA / "train.parquet").to_pylist()
    by_source_action = Counter((row["replay_source"], row["target_action"]) for row in rows)
    assert by_source_action[("stage8_add_column", "submit_solution")] == 600
    assert by_source_action[("stage8_general_replay", "submit_solution")] == 300
