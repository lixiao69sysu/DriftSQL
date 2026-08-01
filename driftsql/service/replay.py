"""Append-only, hash-bound human review for P4 replay candidates."""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

from driftsql.service.schemas import (
    ReplayCandidateList,
    ReplayCandidateRead,
    ReplayReviewCreate,
)


class ReplayCandidateNotFoundError(KeyError):
    pass


class ReplayReviewUnavailableError(RuntimeError):
    pass


class ReplayReviewStore:
    """Read immutable candidates and append hash-bound human decisions."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.candidates_path = self.directory / "candidates.jsonl"
        self.reviews_path = self.directory / "reviews.jsonl"
        self._write_lock = Lock()

    @property
    def available(self) -> bool:
        return self.candidates_path.is_file() and self.reviews_path.is_file()

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
                if not isinstance(row, dict):
                    raise ValueError(f"Expected object at {path}:{line_number}")
                rows.append(row)
        return rows

    def _state(self) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        if not self.available:
            raise ReplayReviewUnavailableError(
                f"Replay candidates are unavailable under {self.directory}"
            )
        candidates: dict[str, dict[str, Any]] = {}
        for row in self._load_jsonl(self.candidates_path):
            candidate_id = str(row.get("candidate_id", ""))
            if not candidate_id or candidate_id in candidates:
                raise ValueError(f"Missing or duplicate replay candidate: {candidate_id!r}")
            trajectory_sha256 = str(row.get("trajectory_sha256", ""))
            if len(trajectory_sha256) != 64:
                raise ValueError(f"Invalid trajectory hash for {candidate_id}")
            candidates[candidate_id] = row

        reviews: dict[str, dict[str, Any]] = {}
        for row in self._load_jsonl(self.reviews_path):
            candidate_id = str(row.get("candidate_id", ""))
            candidate = candidates.get(candidate_id)
            if candidate is None:
                raise ValueError(f"Review references unknown candidate: {candidate_id}")
            if row.get("decision") not in {"approve", "reject"}:
                raise ValueError(f"Invalid review decision for {candidate_id}")
            if row.get("candidate_trajectory_sha256") != candidate["trajectory_sha256"]:
                raise ValueError(f"Stale or tampered review hash for {candidate_id}")
            reviews[candidate_id] = row
        return candidates, reviews

    @staticmethod
    def _public(candidate: dict[str, Any], review: dict[str, Any] | None) -> ReplayCandidateRead:
        reward = candidate.get("reward")
        reward_score = reward.get("score") if isinstance(reward, dict) else None
        return ReplayCandidateRead(
            candidate_id=str(candidate["candidate_id"]),
            session_id=str(candidate["session_id"]),
            scenario_id=str(candidate["scenario_id"]),
            db_id=str(candidate["db_id"]),
            drift_type=str(candidate["drift_type"]),
            wildcard_profile=candidate.get("wildcard_profile"),
            added_column_count=candidate.get("added_column_count"),
            failure_type=str(candidate["failure_type"]),
            failure_class=str(candidate["failure_class"]),
            session_status=str(candidate["status"]),
            termination_reason=candidate.get("termination_reason"),
            success=candidate.get("success"),
            model_calls=int(candidate.get("model_calls", 0)),
            tool_calls=int(candidate.get("tool_calls", 0)),
            tool_sequence=[str(tool) for tool in candidate.get("tool_sequence", [])],
            final_sql=candidate.get("final_sql"),
            reward_score=float(reward_score) if reward_score is not None else None,
            trajectory_sha256=str(candidate["trajectory_sha256"]),
            review_status=review["decision"] if review else "pending",
            reviewer=str(review["reviewer"]) if review else None,
            review_reason=str(review["reason"]) if review else None,
            reviewed_at=datetime.fromisoformat(str(review["reviewed_at"])) if review else None,
        )

    def list_candidates(self, *, review_status: str | None = None) -> ReplayCandidateList:
        if not self.available:
            return ReplayCandidateList(available=False, candidates=[], total=0, counts={})
        candidates, reviews = self._state()
        public = [
            self._public(candidate, reviews.get(candidate_id))
            for candidate_id, candidate in sorted(candidates.items())
        ]
        counts = dict(sorted(Counter(candidate.review_status for candidate in public).items()))
        if review_status:
            public = [candidate for candidate in public if candidate.review_status == review_status]
        return ReplayCandidateList(
            available=True,
            candidates=public,
            total=len(public),
            counts=counts,
        )

    def review(self, candidate_id: str, decision: ReplayReviewCreate) -> ReplayCandidateRead:
        with self._write_lock:
            candidates, _reviews = self._state()
            try:
                candidate = candidates[candidate_id]
            except KeyError as error:
                raise ReplayCandidateNotFoundError(candidate_id) from error
            record = {
                "candidate_id": candidate_id,
                "decision": decision.decision,
                "reviewer": decision.reviewer.strip(),
                "reason": decision.reason.strip(),
                "reviewed_at": datetime.now(UTC).isoformat(),
                "candidate_trajectory_sha256": candidate["trajectory_sha256"],
            }
            with self.reviews_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            return self._public(candidate, record)

