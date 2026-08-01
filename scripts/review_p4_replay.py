#!/usr/bin/env python3
"""List or append human decisions for immutable P4 replay candidates."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "data/processed/p4_replay_review"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def latest_reviews(path: Path) -> dict[str, dict]:
    return {row["candidate_id"]: row for row in load_jsonl(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_DIR)
    subparsers = parser.add_subparsers(dest="command", required=True)
    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("--status", choices=("pending", "approve", "reject", "all"), default="pending")
    decide = subparsers.add_parser("decide")
    decide.add_argument("--candidate-id", required=True)
    decide.add_argument("--decision", choices=("approve", "reject"), required=True)
    decide.add_argument("--reviewer", required=True)
    decide.add_argument("--reason", required=True)
    args = parser.parse_args()

    candidates = {row["candidate_id"]: row for row in load_jsonl(args.review_dir / "candidates.jsonl")}
    reviews_path = args.review_dir / "reviews.jsonl"
    reviews = latest_reviews(reviews_path)
    if args.command == "list":
        output = []
        for candidate_id, candidate in candidates.items():
            review = reviews.get(candidate_id)
            status = review["decision"] if review else "pending"
            if args.status not in ("all", status):
                continue
            output.append(
                {
                    "candidate_id": candidate_id,
                    "status": status,
                    "failure_class": candidate["failure_class"],
                    "termination_reason": candidate["termination_reason"],
                    "wildcard_profile": candidate["wildcard_profile"],
                    "tool_sequence": candidate["tool_sequence"],
                    "final_sql": candidate["final_sql"],
                }
            )
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return

    if args.candidate_id not in candidates:
        raise KeyError(f"Unknown candidate: {args.candidate_id}")
    if len(args.reviewer.strip()) < 2 or len(args.reason.strip()) < 8:
        raise ValueError("Reviewer and a specific review reason are required")
    decision = {
        "candidate_id": args.candidate_id,
        "decision": args.decision,
        "reviewer": args.reviewer.strip(),
        "reason": args.reason.strip(),
        "reviewed_at": datetime.now(UTC).isoformat(),
        "candidate_trajectory_sha256": candidates[args.candidate_id]["trajectory_sha256"],
    }
    with reviews_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(decision, ensure_ascii=False) + "\n")
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
