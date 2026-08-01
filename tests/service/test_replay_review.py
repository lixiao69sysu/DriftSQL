from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from driftsql.service import create_app
from driftsql.service.inference.backend import ScriptedModelBackend
from driftsql.service.replay import ReplayReviewStore
from driftsql.service.schemas import ReplayReviewCreate
from driftsql.service.settings import ServiceSettings


def prepare_review_dir(path: Path) -> Path:
    path.mkdir(parents=True)
    candidate = {
        "candidate_id": "p4r_fixture",
        "session_id": "session-fixture",
        "scenario_id": "scenario-fixture",
        "db_id": "database-fixture",
        "drift_type": "add_column",
        "wildcard_profile": "multi_table_qualified",
        "added_column_count": 1,
        "failure_type": "budget_exhausted",
        "failure_class": "turn_limit",
        "status": "budget_exhausted",
        "termination_reason": "max_turns",
        "success": False,
        "model_calls": 7,
        "tool_calls": 6,
        "tool_sequence": ["execute_sql", "get_schema_version"],
        "final_sql": None,
        "reward": {"score": -0.2},
        "trajectory_sha256": "a" * 64,
    }
    (path / "candidates.jsonl").write_text(
        json.dumps(candidate, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (path / "reviews.jsonl").touch()
    return path


def test_replay_review_is_append_only_and_bound_to_trajectory_hash(tmp_path: Path) -> None:
    review_dir = prepare_review_dir(tmp_path / "review")
    store = ReplayReviewStore(review_dir)

    pending = store.list_candidates()
    assert pending.available is True
    assert pending.counts == {"pending": 1}
    assert pending.candidates[0].review_status == "pending"

    approved = store.review(
        "p4r_fixture",
        ReplayReviewCreate(
            decision="approve",
            reviewer="human-reviewer",
            reason="The failure label and tool-loop evidence are correct.",
        ),
    )
    assert approved.review_status == "approve"
    first_record = json.loads((review_dir / "reviews.jsonl").read_text(encoding="utf-8"))
    assert first_record["candidate_trajectory_sha256"] == "a" * 64

    store.review(
        "p4r_fixture",
        ReplayReviewCreate(
            decision="reject",
            reviewer="second-reviewer",
            reason="The candidate is superseded after a second manual inspection.",
        ),
    )
    records = (review_dir / "reviews.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(records) == 2
    assert store.list_candidates().counts == {"reject": 1}
    assert store.list_candidates(review_status="approve").total == 0


def test_replay_review_refuses_stale_hash_decision(tmp_path: Path) -> None:
    review_dir = prepare_review_dir(tmp_path / "review")
    (review_dir / "reviews.jsonl").write_text(
        json.dumps(
            {
                "candidate_id": "p4r_fixture",
                "decision": "approve",
                "reviewer": "reviewer",
                "reason": "The evidence is complete and actionable.",
                "reviewed_at": "2026-08-01T00:00:00+00:00",
                "candidate_trajectory_sha256": "b" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Stale or tampered"):
        ReplayReviewStore(review_dir).list_candidates()


def test_replay_review_api_exposes_candidates_and_appends_human_decision(tmp_path: Path) -> None:
    async def run() -> None:
        review_dir = prepare_review_dir(tmp_path / "review")
        settings = ServiceSettings(
            environment="test",
            model_backend="scripted",
            repository_path=tmp_path / "repository.sqlite",
            temporary_root=tmp_path / "sandboxes",
            replay_review_dir=review_dir,
        )
        app = create_app(settings, backend=ScriptedModelBackend())
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                candidates = await client.get("/api/replay/candidates")
                assert candidates.status_code == 200
                assert candidates.json()["candidates"][0]["failure_class"] == "turn_limit"

                invalid = await client.post(
                    "/api/replay/candidates/p4r_fixture/reviews",
                    json={"decision": "approve", "reviewer": "x", "reason": "too short"},
                )
                assert invalid.status_code == 422

                reviewed = await client.post(
                    "/api/replay/candidates/p4r_fixture/reviews",
                    json={
                        "decision": "approve",
                        "reviewer": "portfolio-owner",
                        "reason": "The failure is actionable and contains no sensitive content.",
                    },
                )
                assert reviewed.status_code == 200
                assert reviewed.json()["review_status"] == "approve"
                approved = await client.get("/api/replay/candidates?review_status=approve")
                assert approved.json()["total"] == 1

    asyncio.run(run())

