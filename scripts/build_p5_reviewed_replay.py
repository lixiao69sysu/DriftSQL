#!/usr/bin/env python3
"""Match human-approved P4 failure strata to P5 Train rows only."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-dir", type=Path, default=ROOT / "data/processed/p4_replay_review")
    parser.add_argument("--p5-train", type=Path, default=ROOT / "data/processed/p5_isolated_protocol/train.jsonl")
    parser.add_argument("--p5-summary", type=Path, default=ROOT / "data/processed/p5_isolated_protocol/summary.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/p5_reviewed_replay")
    parser.add_argument("--rows-per-candidate", type=int, default=8)
    parser.add_argument("--seed", type=int, default=520260802)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")

    candidates = {row["candidate_id"]: row for row in load_jsonl(args.review_dir / "candidates.jsonl")}
    reviews = {row["candidate_id"]: row for row in load_jsonl(args.review_dir / "reviews.jsonl")}
    for candidate_id, review in reviews.items():
        candidate = candidates.get(candidate_id)
        if candidate is None:
            raise RuntimeError(f"Review references unknown candidate: {candidate_id}")
        if review.get("candidate_trajectory_sha256") != candidate.get("trajectory_sha256"):
            raise RuntimeError(f"Review hash does not match immutable candidate: {candidate_id}")
    approved = [
        (candidates[candidate_id], review)
        for candidate_id, review in reviews.items()
        if review["decision"] == "approve" and candidate_id in candidates
    ]
    if not approved:
        raise RuntimeError("No human-approved P4 replay candidates")

    p5_summary = json.loads(args.p5_summary.read_text(encoding="utf-8"))
    gate_dbs = set(p5_summary["splits"]["gate"]["database_ids"])
    tune_dbs = set(p5_summary["splits"]["tune"]["database_ids"])
    train_rows = load_jsonl(args.p5_train)
    if {row["db_id"] for row in train_rows} & (gate_dbs | tune_dbs):
        raise RuntimeError("P5 Train source overlaps Tune or sealed Gate")

    pools: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in train_rows:
        pools[(str(row["wildcard_profile"]), int(row["added_column_count"]))].append(row)
    rng = random.Random(args.seed)
    output: list[dict] = []
    manifest: list[dict] = []
    for candidate, review in approved:
        stratum = (str(candidate["wildcard_profile"]), int(candidate["added_column_count"]))
        pool = pools.get(stratum, [])
        if not pool:
            raise RuntimeError(f"No P5 Train rows for approved stratum: {stratum}")
        for _ in range(args.rows_per_candidate):
            source = rng.choice(pool)
            row = dict(source)
            row["p5_replay"] = {
                "candidate_id": candidate["candidate_id"],
                "failure_class": candidate["failure_class"],
                "failure_type": candidate["failure_type"],
                "reviewer": review["reviewer"],
                "reviewed_at": review["reviewed_at"],
                "policy": "P4 Tune failure defines stratum; row comes only from P5 Train",
            }
            output.append(row)
            manifest.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "source_task_id": source["task_id"],
                    "source_db_id": source["db_id"],
                    "wildcard_profile": source["wildcard_profile"],
                    "failure_class": candidate["failure_class"],
                }
            )
    order = list(range(len(output)))
    rng.shuffle(order)
    output = [output[index] for index in order]
    manifest = [manifest[index] | {"replay_index": replay_index} for replay_index, index in enumerate(order)]
    args.output_dir.mkdir(parents=True)
    write_jsonl(args.output_dir / "train.jsonl", output)
    write_jsonl(args.output_dir / "sampling_manifest.jsonl", manifest)
    summary = {
        "protocol": "driftsql_p5_human_reviewed_failure_replay_v1",
        "approved_candidates": len(approved),
        "approved_candidate_ids": sorted(candidate["candidate_id"] for candidate, _ in approved),
        "output_rows": len(output),
        "failure_classes": dict(sorted(Counter(row[0]["failure_class"] for row in approved).items())),
        "source_database_ids": sorted({row["db_id"] for row in output}),
        "p5_tune_database_overlap": sorted({row["db_id"] for row in output} & tune_dbs),
        "p5_gate_database_overlap": sorted({row["db_id"] for row in output} & gate_dbs),
        "p4_tune_rows_copied": False,
        "p5_gate_read": False,
        "stage8_gate55_read": False,
        "seed": args.seed,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
