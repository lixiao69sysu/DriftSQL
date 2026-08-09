from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from scripts.replay_p6_first_action_reward import premature_stale

ROOT = Path(__file__).resolve().parents[1]


def test_first_action_focus200_release_is_isolated_and_hidden() -> None:
    data = ROOT / "data/processed/p6_first_action_focus200_v2"
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    rows = pq.read_table(data / "train.parquet").to_pylist()

    assert summary["train_rows"] == 200
    assert summary["training_steps_at_batch8"] == 25
    assert summary["unique_train_tasks"] == 200
    assert summary["train_tune_database_overlap"] == []
    assert summary["train_tune_task_overlap"] == []
    assert summary["fresh_blind_rows_read"] == 0
    assert Counter(row["extra_info"]["decision_target_action"] for row in rows) == {
        "execute_sql": 200,
    }
    assert {int(row["extra_info"]["index"]) for row in rows} == set(range(200))
    assert all("decision_target_action" not in json.dumps(row["prompt"]) for row in rows)


def test_offline_stale_shortcut_requires_missing_ordered_inspection() -> None:
    stale = "SELECT * FROM items"
    direct = '\n'.join(
        [
            '{"name":"execute_sql","arguments":{"sql":"SELECT * FROM items"}}',
            '{"name":"submit_solution","arguments":{"sql":"SELECT * FROM items"}}',
        ]
    )
    canonical = '\n'.join(
        [
            '{"name":"execute_sql","arguments":{"sql":"SELECT * FROM items"}}',
            '{"name":"get_schema_version","arguments":{}}',
            '{"name":"inspect_schema_diff","arguments":{}}',
            '{"name":"execute_sql","arguments":{"sql":"SELECT id FROM items"}}',
            '{"name":"submit_solution","arguments":{"sql":"SELECT id FROM items"}}',
        ]
    )

    assert premature_stale(direct, stale)
    assert not premature_stale(canonical, stale)
