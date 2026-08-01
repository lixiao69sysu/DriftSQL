#!/usr/bin/env python3
"""Build a fresh DB-disjoint Stage 8 protocol from databases unused by Stage 7.

Stage 7 Gate106 files are hashed but never parsed.  All Stage 8 task rows come
from original BIRD/Six-Gym training sources whose logical database IDs did not
occur in any Stage 7 split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import signal
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driftsql.drift import (
    build_add_column_projection_example,
    build_clean_example,
    build_column_rename_example,
    build_column_replacement_example,
    build_compound_drift_example,
    build_table_rename_example,
)
from driftsql.drift.factory import fingerprint_query
from scripts.build_stage7_add_column_protocol import (
    choose_canonical_database,
    quote,
    task_digest,
    unique_added_name,
    usable_tables,
    write_jsonl,
)
from scripts.build_stratified_drift_data_v2 import enrich
from scripts.split_stage6_train_tune_gate import assign_databases


ROOT = Path(__file__).resolve().parents[1]
STAGE7 = ROOT / "data/processed/stage7_add_column_protocol/summary.json"
BIRD_TASKS = ROOT / "data/raw/bird23-train-filtered/data/train-00000-of-00001.jsonl"
BIRD_DATABASES = ROOT / "data/raw/bird23-train-filtered/full/train/train_databases"
SIX_TASKS = ROOT / "data/raw/six-gym-sqlite/train.jsonl"
SIX_DATABASES = ROOT / "data/raw/six-gym-sqlite/database"
DEFAULT_OUTPUT = ROOT / "data/processed/stage8_fresh_protocol"
DEFAULT_SEAL = ROOT / "reports/stage8/stage7_gate106_seal.json"
SPLITS = ("train", "tune", "gate")
FRACTIONS = {"train": 2 / 3, "tune": 1 / 6, "gate": 1 / 6}

GENERAL_BUILDERS: dict[str, Callable[..., Any]] = {
    "rename_column": build_column_rename_example,
    "rename_table": build_table_rename_example,
    "replace_column": build_column_replacement_example,
    "compound": build_compound_drift_example,
    "clean": build_clean_example,
}

SEALED_STAGE7_GATE = (
    ROOT / "reports/stage7/final_candidate/frozen_candidate.json",
    ROOT / "data/processed/stage7_gate106/summary.json",
    ROOT / "data/processed/stage7_gate106/agent_eval.jsonl",
    ROOT / "reports/stage7/final_gate106/summary.json",
    ROOT / "reports/stage7/final_gate106/stage7-frozen-candidate.jsonl",
    ROOT / "reports/stage7/final_gate106/audit.json",
    ROOT / "reports/stage7/final_gate106/profile_addendum.json",
)

AUDIT_SPECS = (
    ("ingestion_audit_id", "INTEGER", "0"),
    ("source_sync_flag", "INTEGER", "0"),
    ("record_lineage_tag", "TEXT", "'unknown'"),
    ("quality_review_state", "TEXT", "'pending'"),
    ("pipeline_batch_id", "INTEGER", "0"),
    ("compliance_trace_tag", "TEXT", "'unreviewed'"),
)


class DatabaseBudgetExpired(BaseException):
    """Escape per-task Exception handlers when a whole database is too costly."""


def _database_budget_expired(signum: int, frame: Any) -> None:
    del signum, frame
    raise DatabaseBudgetExpired("database-level 90 second construction budget expired")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_pool() -> dict[str, list[dict[str, Any]]]:
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(load_jsonl(BIRD_TASKS)):
        pools[str(row["db_id"])].append(
            {
                "source": "bird23_train_filtered_stage8",
                "source_index": index,
                "question": str(row["question"]),
                "evidence": str(row.get("evidence", "")),
                "sql": str(row["SQL"]),
            }
        )
    six_offset = 10_000_000
    for index, row in enumerate(load_jsonl(SIX_TASKS)):
        solution = row.get("sol_sql")
        if (
            str(row.get("dialect", "")).casefold() != "sqlite"
            or not isinstance(solution, list)
            or len(solution) != 1
            or row.get("preprocess_sql")
            or row.get("clean_up_sql")
        ):
            continue
        pools[str(row["db_id"])].append(
            {
                "source": "six_gym_sqlite_stage8",
                "source_index": six_offset + index,
                "question": str(row["query"]),
                "evidence": "",
                "sql": str(solution[0]),
            }
        )
    return pools


def database_candidates() -> dict[str, set[Path]]:
    candidates: dict[str, set[Path]] = defaultdict(set)
    for path in BIRD_DATABASES.glob("*/*.sqlite"):
        candidates[path.parent.name].add(path.resolve())
    for path in SIX_DATABASES.glob("*/*_template.sqlite"):
        candidates[path.parent.name].add(path.resolve())
    return candidates


def synthetic_add_sql(variant: int, left: str, right: str) -> tuple[str, str, list[str], str]:
    if variant == 0:
        return (
            f"SELECT * FROM {quote(left)} ORDER BY rowid LIMIT 20",
            f"Return complete {left} records while preserving the cached result columns.",
            [left],
            "single_table_plain",
        )
    if variant == 1:
        return (
            f"SELECT src.* FROM {quote(left)} AS src ORDER BY src.rowid LIMIT 20",
            f"Return complete {left} records through alias src without exposing new audit fields.",
            [left],
            "single_table_qualified",
        )
    if variant in (2, 4):
        offset = 0 if variant == 2 else 5
        return (
            f"SELECT * FROM {quote(left)} AS lhs CROSS JOIN {quote(right)} AS rhs "
            f"LIMIT 20 OFFSET {offset}",
            f"Return joined {left}/{right} rows with the original combined projection contract.",
            [left, right],
            "multi_table_plain",
        )
    offset = 0 if variant == 3 else 5
    return (
        f"SELECT rhs.*, lhs.* FROM {quote(left)} AS lhs CROSS JOIN {quote(right)} AS rhs "
        f"LIMIT 20 OFFSET {offset}",
        f"Return joined {right}/{left} rows in cached alias order without new audit fields.",
        [right, left],
        "multi_table_qualified",
    )


def build_add_examples(db_id: str, database: Path, *, seed: int, db_index: int) -> list[dict[str, Any]]:
    tables = usable_tables(database)
    random.Random(f"stage8-add:{seed}:{db_id}").shuffle(tables)
    columns = {table: names for table, names in tables}
    examples = []
    for variant in range(6):
        left = tables[(2 * variant) % len(tables)][0]
        right = tables[(2 * variant + 1) % len(tables)][0]
        sql, question, targets, expected_profile = synthetic_add_sql(variant, left, right)
        base, declared_type, default_sql = AUDIT_SPECS[variant]
        specs = [
            {
                "table": table,
                "new_name": unique_added_name(base, columns[table]),
                "declared_type": declared_type,
                "default_sql": default_sql,
            }
            for table in targets
        ]
        item = build_add_column_projection_example(
            source="stage8_fresh_projection_contract",
            source_index=db_index * 6 + variant,
            db_id=db_id,
            question=question,
            evidence="",
            sql=sql,
            database=database,
            added_column_specs=specs,
        ).to_dict()
        if item["wildcard_profile"] != expected_profile:
            raise RuntimeError(
                f"Wildcard profile mismatch {db_id}/{variant}: {item['wildcard_profile']}"
            )
        item["stage8"] = {
            "scenario_variant": variant,
            "submit_decision_focus": True,
            "synthetic": True,
        }
        examples.append(item)
    return examples


def build_general_examples(
    db_id: str,
    database: Path,
    tasks: list[dict[str, Any]],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Counter[str]]]:
    ordered = list(tasks)
    random.Random(f"stage8-general:{seed}:{db_id}").shuffle(ordered)
    results: list[dict[str, Any]] = []
    rejected = {name: Counter() for name in GENERAL_BUILDERS}
    viable: list[dict[str, Any]] = []
    for task in ordered[:25]:
        try:
            fingerprint_query(database, task["sql"], timeout_seconds=1.0)
        except Exception as error:
            for counts in rejected.values():
                counts[f"prefilter_{type(error).__name__}"] += 1
            continue
        viable.append(task)
        if len(viable) >= 12:
            break
    if not viable:
        raise RuntimeError(f"No fast executable source task for {db_id}")
    for family, builder in GENERAL_BUILDERS.items():
        for task in viable:
            try:
                example = builder(
                    source=task["source"],
                    source_index=int(task["source_index"]),
                    db_id=db_id,
                    question=task["question"],
                    evidence=task["evidence"],
                    sql=task["sql"],
                    database=database,
                )
            except Exception as error:  # recorded aggregate only
                rejected[family][type(error).__name__] += 1
                continue
            results.append(example.to_dict())
            break
        else:
            raise RuntimeError(f"Unable to build {family} example for {db_id}")
    return results, rejected


def describe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def drift_type(row: dict[str, Any]) -> str:
        if row.get("drift_type"):
            return str(row["drift_type"])
        if row.get("wildcard_profile"):
            return "add_column"
        return "unknown"

    return {
        "rows": len(rows),
        "databases": len({str(row["db_id"]) for row in rows}),
        "task_id_sha256": task_digest(rows),
        "drift_types": dict(sorted(Counter(drift_type(row) for row in rows).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seal-output", type=Path, default=DEFAULT_SEAL)
    parser.add_argument("--seed", type=int, default=82028)
    parser.add_argument("--trials", type=int, default=5000)
    args = parser.parse_args()
    if args.output_dir.exists() or args.seal_output.exists():
        raise FileExistsError("Stage 8 protocol/seal already exists; refusing to overwrite")

    stage7_summary = json.loads(STAGE7.read_text(encoding="utf-8"))
    stage7_dbs = set().union(
        *(set(value["database_ids"]) for value in stage7_summary["splits"].values())
    )
    pools = task_pool()
    candidates = database_candidates()
    eligible = sorted((set(pools) & set(candidates)) - stage7_dbs)
    sources: dict[str, Path] = {}
    resolution: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    for db_id in eligible:
        try:
            source, audit = choose_canonical_database(candidates[db_id])
        except Exception as error:
            skipped[db_id] = f"{type(error).__name__}: {error}"
            continue
        sources[db_id] = source
        resolution[db_id] = audit
    if len(sources) < 30:
        raise RuntimeError(f"Only {len(sources)} fresh usable databases")

    all_general = []
    all_add: dict[str, list[dict[str, Any]]] = {}
    usable_sources: dict[str, Path] = {}
    rejection_totals: dict[str, Counter[str]] = {
        name: Counter() for name in GENERAL_BUILDERS
    }
    for db_index, db_id in enumerate(sorted(sources)):
        previous_alarm_handler = signal.signal(signal.SIGALRM, _database_budget_expired)
        signal.setitimer(signal.ITIMER_REAL, 90.0)
        try:
            general, rejected = build_general_examples(
                db_id, sources[db_id], pools[db_id], seed=args.seed
            )
            add_rows = build_add_examples(
                db_id, sources[db_id], seed=args.seed, db_index=db_index
            )
        except (Exception, DatabaseBudgetExpired) as error:
            skipped[db_id] = f"{type(error).__name__}: {error}"
            resolution.pop(db_id, None)
            print(
                f"skipped {db_id}: {type(error).__name__}: {error}",
                flush=True,
            )
            continue
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous_alarm_handler)
        usable_sources[db_id] = sources[db_id]
        all_general.extend(general)
        all_add[db_id] = add_rows
        for family, counts in rejected.items():
            rejection_totals[family].update(counts)
        print(
            f"built {db_index + 1}/{len(sources)} fresh DBs: {db_id} "
            f"general={len(general)} add={len(all_add[db_id])}",
            flush=True,
        )

    sources = usable_sources
    if len(sources) < 30:
        raise RuntimeError(
            f"Only {len(sources)} fresh databases passed full execution verification"
        )

    enriched = enrich(all_general, set())
    split_general, score = assign_databases(
        enriched, fractions=FRACTIONS, seed=args.seed, trials=args.trials
    )
    split_dbs = {
        split: {str(row["db_id"]) for row in split_general[split]}
        for split in SPLITS
    }
    if any(
        split_dbs[a] & split_dbs[b]
        for a, b in (("train", "tune"), ("train", "gate"), ("tune", "gate"))
    ):
        raise RuntimeError("Stage 8 database leakage")
    if set().union(*split_dbs.values()) & stage7_dbs:
        raise RuntimeError("Stage 8 reused a Stage 7 database")

    for split in SPLITS:
        add_rows = [row for db_id in sorted(split_dbs[split]) for row in all_add[db_id]]
        general_rows = sorted(split_general[split], key=lambda row: str(row["task_id"]))
        add_rows.sort(key=lambda row: str(row["task_id"]))
        write_jsonl(args.output_dir / f"{split}_add_column.jsonl", add_rows)
        write_jsonl(args.output_dir / f"{split}_general_replay.jsonl", general_rows)

    summary = {
        "protocol": "driftsql_stage8_fresh_db_isolated_v1",
        "seed": args.seed,
        "split_unit": "db_id",
        "fractions": FRACTIONS,
        "balance_score": score,
        "stage7_database_overlap": [],
        "source_resolution": resolution,
        "skipped_databases": skipped,
        "splits": {
            split: {
                "database_ids": sorted(split_dbs[split]),
                "add_column": describe(
                    [row for db_id in split_dbs[split] for row in all_add[db_id]]
                ),
                "general_replay": describe(split_general[split]),
            }
            for split in SPLITS
        },
        "rejections": {
            family: dict(sorted(counts.items()))
            for family, counts in rejection_totals.items()
        },
        "gate_policy": {
            "tune": "Tune-only selection and failure mining.",
            "gate": "Exactly one run after Stage 8 candidate freeze.",
            "stage7_gate106": "Permanently sealed; never parsed or replayed.",
            "stage6_gate112": "Permanently sealed; never parsed or replayed.",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    missing = [str(path) for path in SEALED_STAGE7_GATE if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot seal missing Stage 7 artifacts: {missing}")
    seal = {
        "protocol": "driftsql_stage7_gate106_permanent_seal_v1",
        "policy": (
            "Hash evidence only. Stage 8 must not parse Gate106 rows, mine failures, "
            "rerun inference, or select models from these files."
        ),
        "files_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in SEALED_STAGE7_GATE
        },
        "stage8_source": "original BIRD/Six-Gym tasks on Stage-7-unused databases",
        "stage7_gate_rows_read": False,
    }
    args.seal_output.parent.mkdir(parents=True, exist_ok=True)
    args.seal_output.write_text(
        json.dumps(seal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
