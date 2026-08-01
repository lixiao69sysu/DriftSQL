from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(relative: str):
    path = ROOT / relative
    if not path.exists():
        pytest.skip(f"P5 generated protocol artifact is not present: {relative}")
    return json.loads(path.read_text(encoding="utf-8"))


def test_p5_uses_new_database_isolated_splits() -> None:
    summary = load("data/processed/p5_isolated_protocol/summary.json")
    splits = {
        name: set(summary["splits"][name]["database_ids"])
        for name in ("train", "tune", "gate")
    }

    assert tuple(len(splits[name]) for name in ("train", "tune", "gate")) == (6, 3, 3)
    assert not (splits["train"] & splits["tune"])
    assert not (splits["train"] & splits["gate"])
    assert not (splits["tune"] & splits["gate"])
    assert summary["stage7_database_overlap"] == []
    assert summary["stage8_database_overlap"] == []
    assert summary["stage8_gate55_rows_read"] is False


def test_p5_gate_is_sealed_from_tuning_and_replay() -> None:
    summary = load("data/processed/p5_isolated_protocol/summary.json")
    seal = load("reports/p5/stage8_gate55_permanent_seal.json")

    assert summary["gate"]["status"] == "sealed_unopened"
    assert set(summary["gate"]["forbidden_uses"]) == {
        "training", "tuning", "failure_mining", "replay_generation"
    }
    assert seal["gate55_rows_read"] is False
    assert seal["p5_database_overlap"] == []
