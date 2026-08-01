#!/usr/bin/env python3
"""Freeze the BIRD Mini-Dev Stage-1 evaluation data and budget protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from driftsql.evaluation import BirdEvalBudget, build_column_meanings
from driftsql.evaluation.bird import dump_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "data/raw/bird-mini-dev/data/mini_dev_sqlite-00000-of-00001.json"
DEFAULT_DB_ROOT = PROJECT_ROOT / "data/raw/bird-mini-dev/full/dev_20240627/dev_databases"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/evaluation/stage1"


def stable_score(row: dict) -> str:
    identity = f"{row['db_id']}|{row['difficulty']}|{row['question_id']}"
    return hashlib.sha256(identity.encode()).hexdigest()


def stratified_pilot(rows: list[dict], size: int) -> list[dict]:
    strata: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        strata.setdefault((row["db_id"], row["difficulty"]), []).append(row)
    for values in strata.values():
        values.sort(key=stable_score)

    selected: list[dict] = []
    keys = sorted(strata)
    while len(selected) < min(size, len(rows)):
        progressed = False
        for key in keys:
            if strata[key] and len(selected) < size:
                selected.append(strata[key].pop(0))
                progressed = True
        if not progressed:
            break
    return sorted(selected, key=lambda row: int(row["instance_idx"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--database-root", type=Path, default=DEFAULT_DB_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pilot-size", type=int, default=64)
    args = parser.parse_args()

    source_rows = json.loads(args.source.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for instance_idx, source in enumerate(source_rows):
        db_id = str(source["db_id"])
        db_path = args.database_root / db_id / f"{db_id}.sqlite"
        if not db_path.is_file():
            raise FileNotFoundError(db_path)
        rows.append(
            {
                "instance_idx": instance_idx,
                "question_id": int(source["question_id"]),
                "db_id": db_id,
                "question": str(source["question"]),
                "evidence": str(source.get("evidence", "")),
                "gold_sql": str(source["SQL"]),
                "difficulty": str(source["difficulty"]),
                "db_path": str(db_path.resolve()),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dump_json(args.output_dir / "bird_mini_dev_500.json", rows)
    pilot = stratified_pilot(rows, args.pilot_size)
    dump_json(args.output_dir / f"bird_mini_dev_pilot_{len(pilot)}.json", pilot)
    meanings = build_column_meanings(args.database_root)
    dump_json(args.output_dir / "column_meaning.json", meanings)
    protocol = {
        "name": "stage1_bird_mini_dev_ex_v1",
        "source": str(args.source.resolve()),
        "database_root": str(args.database_root.resolve()),
        "evaluation_metric": "BIRD-RL set-normalized execution accuracy (EX)",
        "decoding": {"temperature": 0.0},
        "budget": BirdEvalBudget().to_dict(),
        "baselines": ["qwen_base_direct", "qwen_base_react", "bird_zeno_react"],
        "full": {
            "rows": len(rows),
            "databases": len({row["db_id"] for row in rows}),
            "difficulty": dict(Counter(row["difficulty"] for row in rows)),
        },
        "pilot": {
            "rows": len(pilot),
            "databases": len({row["db_id"] for row in pilot}),
            "difficulty": dict(Counter(row["difficulty"] for row in pilot)),
        },
    }
    dump_json(args.output_dir / "protocol.json", protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
