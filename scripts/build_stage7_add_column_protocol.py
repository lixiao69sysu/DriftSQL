#!/usr/bin/env python3
"""Build the DB-isolated Stage 7 additive projection-contract protocol.

The only task source parsed by this script is the historical Stage 6 *train*
partition.  Stage 6 Tune/Gate artifacts are hashed into a seal manifest but
are never loaded as rows or used for assignment, generation, or selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from driftsql.drift import build_add_column_projection_example
from scripts.split_stage6_train_tune_gate import assign_databases


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "data/processed/stage6_protocol/train.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/processed/stage7_add_column_protocol"
DEFAULT_SEAL = PROJECT_ROOT / "reports/stage7/stage6_gate112_seal.json"
SPLITS = ("train", "tune", "gate")
FRACTIONS = {"train": 2 / 3, "tune": 1 / 6, "gate": 1 / 6}

SEALED_STAGE6_FILES = (
    PROJECT_ROOT / "data/processed/stage6_protocol/gate.jsonl",
    PROJECT_ROOT / "data/processed/stage6_ablation/b1/gate_agent_eval.jsonl",
    PROJECT_ROOT / "reports/stage6/final_candidate/frozen_candidate.json",
    PROJECT_ROOT / "reports/stage6/final_gate112/summary.json",
    PROJECT_ROOT / "reports/stage6/final_gate112/stage6-frozen-candidate.jsonl",
    PROJECT_ROOT / "reports/stage6/final_gate112/audit.json",
)

PROFILES = (
    "single_table_plain",
    "single_table_qualified",
    "multi_table_qualified",
    "multi_table_plain",
)

AUDIT_SPECS = {
    "single_table_plain": ("ingestion_audit_id", "INTEGER", "0"),
    "single_table_qualified": ("source_sync_flag", "INTEGER", "0"),
    "multi_table_qualified": ("record_lineage_tag", "TEXT", "'unknown'"),
    "multi_table_plain": ("quality_review_state", "TEXT", "'pending'"),
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_digest(rows: list[dict[str, Any]]) -> str:
    payload = "\n".join(sorted(str(row["task_id"]) for row in rows)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def quote(identifier: str) -> str:
    return f'"{identifier.replace(chr(34), chr(34) * 2)}"'


def usable_tables(database: Path) -> list[tuple[str, list[str]]]:
    uri = database.resolve().as_uri() + "?mode=ro"
    result: list[tuple[str, list[str]]] = []
    with sqlite3.connect(uri, uri=True) as connection:
        tables = [
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        for table in tables:
            columns = [
                str(row[1])
                for row in connection.execute(f"PRAGMA table_info({quote(table)})")
            ]
            if not columns:
                continue
            try:
                # Explicit rowid ordering prevents SQLite from changing result
                # order when an expanded projection enables a covering index.
                nonempty = connection.execute(
                    f"SELECT rowid FROM {quote(table)} ORDER BY rowid LIMIT 1"
                ).fetchone()
            except sqlite3.Error:
                continue
            if nonempty is not None:
                result.append((table, columns))
    if len(result) < 2:
        raise ValueError(f"At least two non-empty tables are required: {database}")
    return result


def choose_canonical_database(paths: set[Path]) -> tuple[Path, dict[str, Any]]:
    """Resolve duplicate logical DB paths without weakening DB-level isolation.

    BIRD and Six-Gym contain several byte-identical database copies.  One
    historical ``airline`` copy differs; for deterministic scenario coverage
    we select the copy with the most usable tables, then break ties by path.
    All copies still share one logical ``db_id`` and therefore one split.
    """
    candidates = []
    for path in sorted(paths, key=str):
        tables = usable_tables(path)
        candidates.append(
            {
                "path": path,
                "sha256": sha256(path),
                "usable_tables": len(tables),
                "usable_columns": sum(len(columns) for _, columns in tables),
            }
        )
    selected = max(
        candidates,
        key=lambda item: (item["usable_tables"], item["usable_columns"], str(item["path"])),
    )
    audit = {
        "selected": str(selected["path"]),
        "selection_rule": "max_usable_tables_then_columns_then_path",
        "candidates": [
            {
                "path": str(item["path"]),
                "sha256": item["sha256"],
                "usable_tables": item["usable_tables"],
                "usable_columns": item["usable_columns"],
            }
            for item in candidates
        ],
    }
    return selected["path"], audit


def unique_added_name(base: str, columns: list[str]) -> str:
    existing = {name.casefold() for name in columns}
    candidate = base
    suffix = 2
    while candidate.casefold() in existing:
        candidate = f"{base}_{suffix}"
        suffix += 1
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", candidate):
        raise ValueError(f"Unsafe generated audit column: {candidate}")
    return candidate


def synthetic_query(profile: str, left: str, right: str) -> tuple[str, str, list[str]]:
    if profile == "single_table_plain":
        return (
            f"SELECT * FROM {quote(left)} ORDER BY rowid LIMIT 20",
            f"Return up to 20 complete records from {left} using the original result-column contract.",
            [left],
        )
    if profile == "single_table_qualified":
        return (
            f"SELECT src.* FROM {quote(left)} AS src ORDER BY src.rowid LIMIT 20",
            f"Return up to 20 complete {left} records through alias src while preserving the original columns.",
            [left],
        )
    if profile == "multi_table_qualified":
        return (
            f"SELECT lhs.*, rhs.* FROM {quote(left)} AS lhs "
            f"CROSS JOIN {quote(right)} AS rhs ORDER BY lhs.rowid, rhs.rowid LIMIT 20",
            f"Return paired complete records from {left} and {right}, preserving both original column contracts.",
            [left, right],
        )
    if profile == "multi_table_plain":
        return (
            f"SELECT * FROM {quote(left)} AS lhs CROSS JOIN {quote(right)} AS rhs "
            f"ORDER BY lhs.rowid, rhs.rowid LIMIT 20",
            f"Return up to 20 complete joined rows from {left} and {right} using the original combined contract.",
            [left, right],
        )
    raise ValueError(profile)


def build_database_examples(
    db_id: str,
    source_db: Path,
    *,
    db_index: int,
    seed: int,
) -> list[dict[str, Any]]:
    tables = usable_tables(source_db)
    random.Random(f"{seed}:{db_id}").shuffle(tables)
    rows: list[dict[str, Any]] = []
    for profile_index, profile in enumerate(PROFILES):
        left, left_columns = tables[(profile_index * 2) % len(tables)]
        right, right_columns = tables[(profile_index * 2 + 1) % len(tables)]
        sql, question, targets = synthetic_query(profile, left, right)
        base_name, declared_type, default_sql = AUDIT_SPECS[profile]
        column_map = {left: left_columns, right: right_columns}
        specs = [
            {
                "table": table,
                "new_name": unique_added_name(base_name, column_map[table]),
                "declared_type": declared_type,
                "default_sql": default_sql,
            }
            for table in targets
        ]
        try:
            example = build_add_column_projection_example(
                source="stage7_synthetic_projection_contract",
                source_index=db_index * len(PROFILES) + profile_index,
                db_id=db_id,
                question=question,
                evidence="",
                sql=sql,
                database=source_db,
                added_column_specs=specs,
            ).to_dict()
        except Exception as error:
            raise RuntimeError(
                f"Failed db={db_id} profile={profile} sql={sql!r} specs={specs!r}"
            ) from error
        example["stage7"] = {
            "wildcard_profile": profile,
            "synthetic": True,
            "verification": "old_result_equals_repaired_active_result",
        }
        rows.append(example)
    return rows


def describe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "databases": len({str(row["db_id"]) for row in rows}),
        "task_id_sha256": task_digest(rows),
        "wildcard_profiles": dict(sorted(Counter(str(row["wildcard_profile"]) for row in rows).items())),
        "added_column_count": dict(sorted(Counter(str(row["added_column_count"]) for row in rows).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seal-output", type=Path, default=DEFAULT_SEAL)
    parser.add_argument("--seed", type=int, default=72027)
    parser.add_argument("--trials", type=int, default=5000)
    args = parser.parse_args()
    if args.output_dir.exists() or args.seal_output.exists():
        raise FileExistsError("Stage 7 protocol/seal already exists; refusing to overwrite")

    source_rows = load_jsonl(args.source)
    split_general, balance_score = assign_databases(
        source_rows,
        fractions=FRACTIONS,
        seed=args.seed,
        trials=args.trials,
    )
    split_databases = {
        split: {str(row["db_id"]) for row in split_general[split]}
        for split in SPLITS
    }
    if any(
        split_databases[left] & split_databases[right]
        for left, right in (("train", "tune"), ("train", "gate"), ("tune", "gate"))
    ):
        raise RuntimeError("Stage 7 database leakage")

    source_candidates: dict[str, set[Path]] = {}
    for row in source_rows:
        db_id = str(row["db_id"])
        source_db = Path(str(row["source_db"])).resolve()
        source_candidates.setdefault(db_id, set()).add(source_db)
    db_sources: dict[str, Path] = {}
    source_resolution: dict[str, dict[str, Any]] = {}
    for db_id, paths in sorted(source_candidates.items()):
        db_sources[db_id], source_resolution[db_id] = choose_canonical_database(paths)

    generated_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        generated: list[dict[str, Any]] = []
        for db_index, db_id in enumerate(sorted(split_databases[split])):
            examples = build_database_examples(
                db_id,
                db_sources[db_id],
                db_index=db_index,
                seed=args.seed,
            )
            generated.extend(examples)
            print(f"{split}: generated {len(generated)} add-column examples", flush=True)
        generated.sort(key=lambda row: str(row["task_id"]))
        general_replay = [row for row in split_general[split] if str(row["drift_type"]) != "add_column"]
        legacy_add = [row for row in split_general[split] if str(row["drift_type"]) == "add_column"]
        write_jsonl(args.output_dir / f"{split}_add_column.jsonl", generated)
        write_jsonl(args.output_dir / f"{split}_general_replay.jsonl", general_replay)
        write_jsonl(args.output_dir / f"{split}_legacy_add_column.jsonl", legacy_add)
        generated_by_split[split] = generated

    summary = {
        "protocol": "driftsql_stage7_add_column_db_isolated_v1",
        "parent_source": str(args.source.resolve()),
        "parent_rows": len(source_rows),
        "seed": args.seed,
        "source_resolution": source_resolution,
        "trials": args.trials,
        "split_unit": "db_id",
        "fractions": FRACTIONS,
        "balance_score": balance_score,
        "splits": {
            split: {
                "database_ids": sorted(split_databases[split]),
                "add_column": describe(generated_by_split[split]),
                "general_replay_rows": sum(
                    str(row["drift_type"]) != "add_column" for row in split_general[split]
                ),
                "legacy_add_column_rows": sum(
                    str(row["drift_type"]) == "add_column" for row in split_general[split]
                ),
            }
            for split in SPLITS
        },
        "database_overlap": {
            "train_tune": sorted(split_databases["train"] & split_databases["tune"]),
            "train_gate": sorted(split_databases["train"] & split_databases["gate"]),
            "tune_gate": sorted(split_databases["tune"] & split_databases["gate"]),
        },
        "gate_policy": {
            "tune": "May be used for Stage 7 failure mining and candidate selection.",
            "gate": "Run exactly once only after the Stage 7 candidate is frozen.",
            "stage6_gate112": "Permanently sealed; prohibited as Stage 7 input or tuning evidence.",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    missing_sealed = [str(path) for path in SEALED_STAGE6_FILES if not path.is_file()]
    if missing_sealed:
        raise FileNotFoundError(f"Cannot seal missing Stage 6 artifacts: {missing_sealed}")
    seal = {
        "protocol": "driftsql_stage6_gate112_permanent_seal_v1",
        "policy": (
            "These hashes are audit evidence only. Stage 7 code must never parse these rows, "
            "train on them, select on them, or rerun inference over them."
        ),
        "files_sha256": {
            str(path.relative_to(PROJECT_ROOT)): sha256(path) for path in SEALED_STAGE6_FILES
        },
        "stage7_parent_source": str(args.source.relative_to(PROJECT_ROOT)),
        "stage7_parent_is_stage6_train_only": True,
    }
    args.seal_output.parent.mkdir(parents=True, exist_ok=True)
    args.seal_output.write_text(json.dumps(seal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
