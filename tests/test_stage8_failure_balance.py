from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/stage8_failure_balanced_grpo"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_stage8_failure_replay_counts_and_seals() -> None:
    summary = json.loads((DATA / "summary.json").read_text(encoding="utf-8"))
    assert pq.read_table(DATA / "train.parquet").num_rows == 440
    assert summary["add_column_rows"] == 308
    assert summary["general_replay_rows"] == 132
    assert summary["train_tune_task_overlap"] == []
    assert summary["train_tune_database_overlap"] == []
    assert summary["stage6_gate112_read"] is False
    assert summary["stage7_gate106_read"] is False
    assert summary["stage8_gate_read"] is False


def test_stage8_failure_replay_matches_real_tune_failures_but_uses_train_ids() -> None:
    manifest = load_jsonl(DATA / "sampling_manifest.jsonl")
    diagnostics = load_jsonl(DATA / "failure_diagnostics.jsonl")
    assert Counter(row["failure_type"] for row in diagnostics) == {
        "success": 33,
        "repaired_not_submitted": 14,
        "premature_stale_submit": 4,
        "repair_not_reached": 3,
        "wrong_submit": 1,
    }
    assert Counter(row["replay_source"] for row in manifest) == {
        "real_tune_failure_matched_train": 308,
        "general_drift_replay": 132,
    }
    tune_ids = {row["instance_id"] for row in diagnostics}
    assert not ({row["source_instance_id"] for row in manifest} & tune_ids)
    assert sum(
        row.get("wildcard_profile", "").startswith("multi_table") for row in manifest
    ) == 221
    assert Counter(
        row.get("drift_type")
        for row in manifest
        if row["replay_source"] == "general_drift_replay"
    ) == {
        "compound": 56,
        "clean": 19,
        "rename_column": 19,
        "rename_table": 19,
        "replace_column": 19,
    }
