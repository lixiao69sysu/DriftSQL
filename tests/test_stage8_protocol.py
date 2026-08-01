from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "data/processed/stage8_fresh_protocol"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def digest_task_ids(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for task_id in sorted(str(row["task_id"]) for row in rows):
        digest.update(task_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def test_stage8_protocol_is_db_disjoint_and_has_expected_size() -> None:
    summary = json.loads((PROTOCOL / "summary.json").read_text(encoding="utf-8"))
    expected = {
        "train": (20, 120, 100),
        "tune": (5, 30, 25),
        "gate": (5, 30, 25),
    }
    db_sets = {}
    all_task_ids = []
    for split, (database_count, add_count, general_count) in expected.items():
        add_rows = load_jsonl(PROTOCOL / f"{split}_add_column.jsonl")
        general_rows = load_jsonl(PROTOCOL / f"{split}_general_replay.jsonl")
        assert len(add_rows) == add_count
        assert len(general_rows) == general_count
        db_sets[split] = {str(row["db_id"]) for row in add_rows}
        assert len(db_sets[split]) == database_count
        assert db_sets[split] == {str(row["db_id"]) for row in general_rows}
        assert db_sets[split] == set(summary["splits"][split]["database_ids"])
        assert digest_task_ids(add_rows) == summary["splits"][split]["add_column"]["task_id_sha256"]
        assert digest_task_ids(general_rows) == summary["splits"][split]["general_replay"]["task_id_sha256"]
        all_task_ids.extend(row["task_id"] for row in add_rows + general_rows)
    assert not (db_sets["train"] & db_sets["tune"])
    assert not (db_sets["train"] & db_sets["gate"])
    assert not (db_sets["tune"] & db_sets["gate"])
    assert len(all_task_ids) == len(set(all_task_ids))
    assert summary["stage7_database_overlap"] == []


def test_stage8_add_column_profiles_and_oracles_are_balanced() -> None:
    expected_per_db = {
        "single_table_plain": 1,
        "single_table_qualified": 1,
        "multi_table_plain": 2,
        "multi_table_qualified": 2,
    }
    for split in ("train", "tune", "gate"):
        rows = load_jsonl(PROTOCOL / f"{split}_add_column.jsonl")
        by_db: dict[str, Counter[str]] = {}
        for row in rows:
            by_db.setdefault(str(row["db_id"]), Counter())[row["wildcard_profile"]] += 1
            actions = [step["action"] for step in row["oracle_steps"]]
            assert actions == [
                "execute_sql",
                "get_schema_version",
                "inspect_schema_diff",
                "execute_sql",
                "submit_solution",
            ]
            assert row["stage8"]["submit_decision_focus"] is True
            assert row["stale_sql"] != row["repaired_sql"]
            assert row["oracle_steps"][-1]["observation"]["accepted"] is True
        assert all(dict(counts) == expected_per_db for counts in by_db.values())


def test_stage8_general_replay_is_family_balanced() -> None:
    expected = {"clean", "compound", "rename_column", "rename_table", "replace_column"}
    for split in ("train", "tune", "gate"):
        rows = load_jsonl(PROTOCOL / f"{split}_general_replay.jsonl")
        by_db: dict[str, set[str]] = {}
        for row in rows:
            by_db.setdefault(str(row["db_id"]), set()).add(str(row["drift_type"]))
        assert all(families == expected for families in by_db.values())


def test_stage7_gate106_seal_hashes_match() -> None:
    from scripts.verify_stage7_gate106_seal import verify

    assert verify() == {"sealed": True, "files": 7}
