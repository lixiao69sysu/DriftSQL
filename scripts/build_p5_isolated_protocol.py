#!/usr/bin/env python3
"""Build the P5 add-column/turn-limit protocol on entirely unseen databases."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_stage7_add_column_protocol import choose_canonical_database, task_digest, write_jsonl
from scripts.build_stage8_fresh_protocol import build_add_examples


ROOT = Path(__file__).resolve().parents[1]
STAGE7_SUMMARY = ROOT / "data/processed/stage7_add_column_protocol/summary.json"
STAGE8_SUMMARY = ROOT / "data/processed/stage8_fresh_protocol/summary.json"
BIRD_DATABASES = ROOT / "data/raw/bird23-train-filtered/full/train/train_databases"
SIX_DATABASES = ROOT / "data/raw/six-gym-sqlite/database"
CRITIC_DATABASES = ROOT / "data/raw/bird-critic-sqlite/database"
DEFAULT_OUTPUT = ROOT / "data/processed/p5_isolated_protocol"
DEFAULT_SEAL = ROOT / "reports/p5/stage8_gate55_permanent_seal.json"
SPLIT_COUNTS = {
    "train": {"bird_critic": 6},
    "tune": {"bird_critic": 3},
    "gate": {"bird_critic": 3},
}
SEALED_STAGE8_GATE = (
    ROOT / "reports/stage8/final_candidate/frozen_candidate.json",
    ROOT / "reports/stage8/final_candidate/gate55_result.json",
    ROOT / "data/processed/stage8_gate_eval/summary.json",
    ROOT / "data/processed/stage8_gate_eval/gate55_agent_eval.jsonl",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_database_ids(summary: dict[str, Any]) -> set[str]:
    return set().union(*(set(value["database_ids"]) for value in summary["splits"].values()))


def database_candidates() -> tuple[dict[str, set[Path]], dict[str, str]]:
    candidates: dict[str, set[Path]] = defaultdict(set)
    cohorts: dict[str, str] = {}
    for path in BIRD_DATABASES.glob("*/*.sqlite"):
        candidates[path.parent.name].add(path.resolve())
        cohorts[path.parent.name] = "unused_train"
    for path in SIX_DATABASES.glob("*/*_template.sqlite"):
        candidates[path.parent.name].add(path.resolve())
        cohorts[path.parent.name] = "unused_train"
    for path in CRITIC_DATABASES.glob("*/*_template.sqlite"):
        candidates[path.parent.name].add(path.resolve())
        cohorts[path.parent.name] = "bird_critic"
    return candidates, cohorts


def allocate(cohorts: dict[str, str], *, seed: int) -> dict[str, list[str]]:
    available = {
        cohort: sorted(db_id for db_id, value in cohorts.items() if value == cohort)
        for cohort in ("unused_train", "bird_critic")
    }
    for cohort, values in available.items():
        random.Random(f"p5:{seed}:{cohort}").shuffle(values)

    offsets = Counter()
    result: dict[str, list[str]] = {}
    for split in ("train", "tune", "gate"):
        selected: list[str] = []
        for cohort, count in SPLIT_COUNTS[split].items():
            start = offsets[cohort]
            chosen = available[cohort][start : start + count]
            if len(chosen) != count:
                raise RuntimeError(f"Insufficient {cohort} databases for {split}: {len(chosen)}/{count}")
            selected.extend(chosen)
            offsets[cohort] += count
        result[split] = sorted(selected)
    return result


def describe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "databases": len({row["db_id"] for row in rows}),
        "task_id_sha256": task_digest(rows),
        "wildcard_profiles": dict(sorted(Counter(row["wildcard_profile"] for row in rows).items())),
        "failure_focus": dict(
            sorted(Counter(focus for row in rows for focus in row["p5"]["failure_focus"]).items())
        ),
        "source_cohorts": dict(sorted(Counter(row["p5"]["source_cohort"] for row in rows).items())),
    }


def write_gate55_seal(path: Path, selected_dbs: set[str], stage8_dbs: set[str]) -> None:
    missing = [str(candidate) for candidate in SEALED_STAGE8_GATE if not candidate.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot seal missing Gate55 artifacts: {missing}")
    seal = {
        "protocol": "driftsql_stage8_gate55_permanent_seal_v1",
        "created_for": "P5 unseen-database add-column/turn-limit development",
        "policy": (
            "Hash evidence only. P5 must never parse Gate55 rows, tune from Gate55 metrics, "
            "mine Gate55 failures, or include Gate55 tasks in replay."
        ),
        "files_sha256": {
            str(candidate.relative_to(ROOT)): sha256(candidate)
            for candidate in SEALED_STAGE8_GATE
        },
        "gate55_rows_read": False,
        "p5_database_overlap": sorted(selected_dbs & stage8_dbs),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(seal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seal-output", type=Path, default=DEFAULT_SEAL)
    parser.add_argument("--seed", type=int, default=520260801)
    parser.add_argument("--seal-only", action="store_true")
    args = parser.parse_args()
    if args.seal_only:
        if args.seal_output.exists():
            raise FileExistsError(f"P5 Gate55 seal already exists: {args.seal_output}")
        summary_path = args.output_dir / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        summary = load_json(summary_path)
        selected_dbs = set().union(
            *(set(value["database_ids"]) for value in summary["splits"].values())
        )
        stage8_dbs = split_database_ids(load_json(STAGE8_SUMMARY))
        if selected_dbs & stage8_dbs:
            raise RuntimeError("Existing P5 protocol overlaps Stage 8 databases")
        gate_path = args.output_dir / "sealed_gate.jsonl"
        if sha256(gate_path) != summary["gate"]["sha256"]:
            raise RuntimeError("Existing P5 sealed Gate hash mismatch")
        write_gate55_seal(args.seal_output, selected_dbs, stage8_dbs)
        print(args.seal_output)
        return
    if args.output_dir.exists() or args.seal_output.exists():
        raise FileExistsError("P5 protocol/seal already exists; refusing to overwrite")

    stage7_dbs = split_database_ids(load_json(STAGE7_SUMMARY))
    stage8_dbs = split_database_ids(load_json(STAGE8_SUMMARY))
    candidates, cohorts = database_candidates()
    eligible = sorted(set(candidates) - stage7_dbs - stage8_dbs)
    resolved: dict[str, Path] = {}
    resolution: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    for db_id in eligible:
        try:
            selected, audit = choose_canonical_database(candidates[db_id])
        except Exception as error:
            skipped[db_id] = f"{type(error).__name__}: {error}"
            continue
        resolved[db_id] = selected
        resolution[db_id] = audit

    cohort_by_db = {db_id: cohorts[db_id] for db_id in resolved}
    built_rows: dict[str, list[dict[str, Any]]] = {}
    for db_index, db_id in enumerate(
        sorted(db_id for db_id in resolved if cohort_by_db[db_id] == "bird_critic")
    ):
        print(f"validating P5 database={db_id} cohort=bird_critic", flush=True)
        try:
            built_rows[db_id] = build_add_examples(
                db_id,
                resolved[db_id],
                seed=args.seed,
                db_index=db_index,
            )
        except Exception as error:
            skipped[db_id] = f"{type(error).__name__}: {error}"
            print(f"rejected P5 database={db_id}: {skipped[db_id]}", flush=True)
            continue
        print(f"validated P5 database={db_id} rows={len(built_rows[db_id])}", flush=True)

    expected = sum(sum(values.values()) for values in SPLIT_COUNTS.values())
    if len(built_rows) < expected:
        raise RuntimeError(f"Only {len(built_rows)} execution-verified P5 databases; need {expected}")
    split_dbs = allocate(
        {db_id: cohort_by_db[db_id] for db_id in built_rows},
        seed=args.seed,
    )
    selected_dbs = set().union(*(set(values) for values in split_dbs.values()))
    if len(selected_dbs) != expected:
        raise RuntimeError("P5 database allocation contains duplicates")
    if selected_dbs & (stage7_dbs | stage8_dbs):
        raise RuntimeError("P5 database leakage from Stage 7 or Stage 8")

    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "tune", "gate"):
        split_rows: list[dict[str, Any]] = []
        for db_id in split_dbs[split]:
            rows = built_rows[db_id]
            for row in rows:
                profile = str(row["wildcard_profile"])
                turn_limit_focus = profile.startswith("multi_table")
                row["source"] = f"p5_{cohort_by_db[db_id]}_projection_contract"
                row.pop("stage8", None)
                row["p5"] = {
                    "split": split,
                    "source_cohort": cohort_by_db[db_id],
                    "failure_focus": ["add_column", *( ["turn_limit"] if turn_limit_focus else [])],
                    "turn_limit_focus": turn_limit_focus,
                    "target_max_model_calls": 5,
                    "target_max_tool_calls": 5,
                    "gate_policy": "sealed_one_shot" if split == "gate" else f"{split}_only",
                }
                split_rows.append(row)
        split_rows.sort(key=lambda row: str(row["task_id"]))
        rows_by_split[split] = split_rows

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.output_dir / "train.jsonl", rows_by_split["train"])
    write_jsonl(args.output_dir / "tune.jsonl", rows_by_split["tune"])
    write_jsonl(args.output_dir / "sealed_gate.jsonl", rows_by_split["gate"])
    gate_path = args.output_dir / "sealed_gate.jsonl"
    summary = {
        "protocol": "driftsql_p5_unseen_db_add_column_turn_limit_v1",
        "seed": args.seed,
        "split_unit": "db_id",
        "database_counts": {split: len(values) for split, values in split_dbs.items()},
        "stage7_database_overlap": sorted(selected_dbs & stage7_dbs),
        "stage8_database_overlap": sorted(selected_dbs & stage8_dbs),
        "stage8_gate55_rows_read": False,
        "splits": {
            split: {
                "database_ids": split_dbs[split],
                "cohorts": dict(sorted(Counter(cohort_by_db[db] for db in split_dbs[split]).items())),
                "tasks": describe(rows_by_split[split]),
            }
            for split in ("train", "tune", "gate")
        },
        "source_resolution": {db_id: resolution[db_id] for db_id in sorted(selected_dbs)},
        "skipped_databases": skipped,
        "gate": {
            "path": str(gate_path.relative_to(ROOT)),
            "sha256": sha256(gate_path),
            "rows": len(rows_by_split["gate"]),
            "status": "sealed_unopened",
            "allowed_reads": "Exactly one final evaluation after candidate freeze.",
            "forbidden_uses": ["training", "tuning", "failure_mining", "replay_generation"],
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    write_gate55_seal(args.seal_output, selected_dbs, stage8_dbs)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
