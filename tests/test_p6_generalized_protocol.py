from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/p6_generalized_protocol"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_generalized_protocol_preserves_all_database_isolated_rows() -> None:
    summary = json.loads((DATA / "summary.json").read_text(encoding="utf-8"))
    assert summary["protocol"] == "driftsql_p6_generalized_current_protocol_v1"
    assert summary["total_trajectories"] == 1102
    assert summary["splits"]["train"]["trajectories"] == 752
    assert summary["splits"]["dev"]["trajectories"] == 169
    assert summary["splits"]["test"]["trajectories"] == 181
    assert summary["database_overlap"] == {
        "train_dev": [],
        "train_test": [],
        "dev_test": [],
    }
    assert summary["evaluation_policy"]["test"].startswith("historical regression")
    assert summary["evaluation_policy"]["fresh_final_gate_required"] is True


def test_every_non_clean_audit_satisfies_the_result_contract() -> None:
    audits = [
        row
        for split in ("train", "dev", "test")
        for row in load_jsonl(DATA / f"{split}_contract_audit.jsonl")
    ]
    assert len(audits) == 1102
    drift = [row for row in audits if row["drift_type"] != "clean"]
    clean = [row for row in audits if row["drift_type"] == "clean"]
    assert len(drift) == 1002
    assert all(row["accepted"] is True for row in drift)
    assert all(row["reason"] == "contract_validated" for row in drift)
    assert all(row["read_only"] is True for row in drift)
    assert all(row["fingerprint_match"] is True for row in drift)
    assert len(clean) == 100
    assert all(row["accepted"] is False for row in clean)
    assert all(row["reason"] == "schema_diff_not_inspected" for row in clean)


def test_dynamic_mask_forces_submit_after_non_clean_validation() -> None:
    for split in ("train", "dev"):
        frame = pd.read_parquet(DATA / f"{split}.parquet")
        assert len(frame) > 0
        submits = frame[(frame["target_action"] == "submit_solution")]
        non_clean = submits[submits["drift_type"] != "clean"]
        assert len(non_clean) > 0
        for row in non_clean.itertuples(index=False):
            assert list(row.available_tools) == ["submit_solution"]
            schemas = json.loads(row.tools)
            assert [schema["function"]["name"] for schema in schemas] == [
                "submit_solution"
            ]
            final = list(row.messages)[-1]
            assert final["role"] == "assistant"
            marker = final["content"].find('{"name"')
            assert marker >= 0
            action = json.loads(final["content"][marker:])
            assert action["name"] == "submit_solution"


def test_rl_records_advertise_the_current_seven_tools_and_contract() -> None:
    expected = [
        "get_schema_version",
        "inspect_schema_diff",
        "get_schema",
        "ask_user",
        "get_knowledge_definition",
        "execute_sql",
        "submit_solution",
    ]
    for split, count in (("train", 752), ("dev", 169), ("test", 181)):
        rows = load_jsonl(DATA / f"{split}_agent_eval.jsonl")
        assert len(rows) == count
        for row in rows:
            extra = row["extra_info"]
            assert extra["tool_selection"] == expected
            assert sorted(extra["tools_kwargs"]) == sorted(expected)
            assert extra["p6_protocol"] == "current-state-policy+result-contract-v1"
            assert extra["result_fingerprint"]["value_hash"]


def test_no_supervised_test_split_is_emitted() -> None:
    assert not (DATA / "test.parquet").exists()
    assert (DATA / "rl_test.parquet").is_file()
    assert (DATA / "test_agent_eval.jsonl").is_file()
