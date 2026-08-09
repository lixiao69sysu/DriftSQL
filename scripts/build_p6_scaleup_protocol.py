#!/usr/bin/env python3
"""Build the DB-isolated P6 scale-up corpus requested for Agentic RL.

The builder preserves the current 50-database Train752 corpus, adds ten
previously unused databases, and reserves every remaining unused database for
either Train-derived tune or a fresh blind gate.  Every generated row is
execution verified by the drift factory before it can enter an output split.

Output split names intentionally remain train/dev/test so the existing
five-tool and P6 protocol adapters can consume the result.  In this protocol
``dev`` means Train-derived tune and ``test`` means the sealed fresh blind set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from driftsql.drift import (
    build_add_column_projection_example,
    build_clean_example,
    build_column_rename_example,
    build_column_replacement_example,
    build_compound_drift_example,
    build_table_rename_example,
)
from scripts.build_stage7_add_column_protocol import (
    choose_canonical_database,
    quote,
    unique_added_name,
    usable_tables,
)
from scripts.build_stratified_drift_data_v2 import enrich


CURRENT_TRAIN = ROOT / "data/processed/stratified_v2/train.jsonl"
DEFAULT_OUTPUT = ROOT / "data/processed/p6_scaleup_v1_raw"
SEED = 20260805

# These historical DBs remain reserved.  The builder never opens or parses the
# corresponding Dev169/Test181 task rows.
HISTORICAL_RESERVED_DBS = {
    "book_publishing_company",
    "cars",
    "olympics",
    "student_loan",
    "works_cycles",
    "chinook",
    "college_completion",
    "computer_student",
    "donor",
    "food_inspection",
    "food_inspection_2",
    "ice_hockey_draft",
    "netflix",
    "public_review_platform",
    "trains",
    "university",
    "world",
}

FAMILIES = (
    "add_column",
    "rename_column",
    "rename_table",
    "replace_column",
    "compound",
    "clean",
)
GENERAL_BUILDERS: dict[str, Callable[..., Any]] = {
    "rename_column": build_column_rename_example,
    "rename_table": build_table_rename_example,
    "replace_column": build_column_replacement_example,
    "compound": build_compound_drift_example,
    "clean": build_clean_example,
}
AUDIT_SPECS = (
    ("ingestion_audit_id", "INTEGER", "0"),
    ("source_sync_flag", "INTEGER", "0"),
    ("record_lineage_tag", "TEXT", "'unknown'"),
    ("quality_review_state", "TEXT", "'pending'"),
    ("pipeline_batch_id", "INTEGER", "0"),
    ("compliance_trace_tag", "TEXT", "'unreviewed'"),
    ("policy_revision_id", "INTEGER", "0"),
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def exact_task_signature(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["db_id"]),
        str(row["stale_sql"]),
        json.dumps(row["schema_diff"], sort_keys=True, separators=(",", ":")),
    )


def deduplicate_exact_tasks(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    kept: list[dict[str, Any]] = []
    removed: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        signature = exact_task_signature(row)
        if signature in seen:
            removed.append(str(row["task_id"]))
            continue
        seen.add(signature)
        kept.append(row)
    return kept, removed


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_order(values: set[str], namespace: str) -> list[str]:
    return sorted(
        values,
        key=lambda value: hashlib.sha256(f"{SEED}:{namespace}:{value}".encode()).hexdigest(),
    )


def database_candidates() -> tuple[dict[str, set[Path]], dict[str, set[str]]]:
    candidates: dict[str, set[Path]] = defaultdict(set)
    provenance: dict[str, set[str]] = defaultdict(set)
    roots = {
        "bird_train": ROOT / "data/raw/bird23-train-filtered/full/train/train_databases",
        "six_gym": ROOT / "data/raw/six-gym-sqlite/database",
        "bird_critic": ROOT / "data/raw/bird-critic-sqlite/database",
        "mini_interact": ROOT / "data/raw/mini-interact",
    }
    for source, root in roots.items():
        for path in root.glob("*/*.sqlite"):
            if source == "bird_train" or path.name.endswith("_template.sqlite"):
                candidates[path.parent.name].add(path.resolve())
                provenance[path.parent.name].add(source)
    return candidates, provenance


def resolve_databases(
    ids: set[str],
    candidates: dict[str, set[Path]],
    preferred: dict[str, Path],
) -> tuple[dict[str, Path], dict[str, Any]]:
    paths: dict[str, Path] = {}
    audits: dict[str, Any] = {}
    for db_id in sorted(ids):
        if db_id in preferred:
            paths[db_id] = preferred[db_id].resolve()
            audits[db_id] = {
                "selected": str(paths[db_id]),
                "selection_rule": "reuse_current_execution_verified_train_source",
            }
            continue
        if db_id not in candidates:
            raise KeyError(f"No local SQLite candidate for {db_id}")
        options = sorted(candidates[db_id], key=str)
        if len(options) == 1:
            paths[db_id] = options[0]
            tables = usable_tables(paths[db_id])
            audits[db_id] = {
                "selected": str(paths[db_id]),
                "selection_rule": "single_fresh_candidate",
                "usable_tables": len(tables),
                "usable_columns": sum(len(columns) for _, columns in tables),
            }
        else:
            # Duplicate logical IDs remain one isolation unit.  Resolve by
            # schema coverage without hashing multi-GB SQLite files up front.
            scored = []
            for path in options:
                tables = usable_tables(path)
                scored.append((len(tables), sum(len(columns) for _, columns in tables), path))
            _, _, selected = max(scored, key=lambda item: (item[0], item[1], str(item[2])))
            paths[db_id] = selected
            audits[db_id] = {
                "selected": str(selected),
                "selection_rule": "max_usable_tables_then_columns_then_path",
                "candidates": [str(path) for path in options],
            }
    return paths, audits


def allocate_databases(
    current_train_ids: set[str],
    candidates: dict[str, set[Path]],
    provenance: dict[str, set[str]],
) -> dict[str, list[str]]:
    unused = set(candidates) - current_train_ids - HISTORICAL_RESERVED_DBS
    bird_unused = {db for db in unused if "bird_train" in provenance[db]}
    critic_unused = {db for db in unused if "bird_critic" in provenance[db]}
    mini_unused = {db for db in unused if "mini_interact" in provenance[db]}
    if len(bird_unused) < 6 or len(critic_unused) < 15 or len(mini_unused) < 27:
        raise RuntimeError(
            f"Insufficient fresh DBs: bird={len(bird_unused)} critic={len(critic_unused)} "
            f"mini={len(mini_unused)}"
        )

    train_new = set(stable_order(bird_unused, "train-bird")[:6])
    train_new.update(stable_order(mini_unused, "train-mini")[:4])
    remaining_critic = set(critic_unused) - train_new
    remaining_mini = set(mini_unused) - train_new
    tune = set(stable_order(remaining_critic, "tune-critic")[:8])
    tune.update(stable_order(remaining_mini, "tune-mini")[:10])
    blind = unused - train_new - tune
    splits = {
        "train": sorted(current_train_ids | train_new),
        "dev": sorted(tune),
        "test": sorted(blind),
    }
    if len(splits["train"]) != 60 or len(splits["dev"]) != 18 or len(splits["test"]) != 20:
        raise RuntimeError({name: len(values) for name, values in splits.items()})
    if any(
        set(splits[left]) & set(splits[right])
        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test"))
    ):
        raise RuntimeError("Database split leakage")
    return splits


def general_query(database: Path, db_id: str, variant: int) -> tuple[str, str]:
    tables = usable_tables(database)
    random.Random(f"{SEED}:general:{db_id}").shuffle(tables)
    table, columns = tables[variant % len(tables)]
    width = min(len(columns), 1 + (variant % min(3, len(columns))))
    start = (variant // max(1, len(tables))) % len(columns)
    selected = [columns[(start + offset) % len(columns)] for offset in range(width)]
    alias = f"t{variant % 5}"
    projection = ", ".join(quote(column) for column in selected)
    qualified = ", ".join(f"{quote(alias)}.{quote(column)}" for column in selected)
    limit = 5 + variant % 16
    offset = variant % 7
    mode = variant % 4
    if mode == 0:
        sql = f"SELECT {projection} FROM {quote(table)} ORDER BY rowid LIMIT {limit} OFFSET {offset}"
    elif mode == 1:
        sql = (
            f"SELECT {qualified} FROM {quote(table)} AS {quote(alias)} "
            f"WHERE {quote(alias)}.{quote(selected[0])} IS NOT NULL "
            f"ORDER BY {quote(alias)}.rowid LIMIT {limit} OFFSET {offset}"
        )
    elif mode == 2:
        sql = f"SELECT {projection} FROM {quote(table)} WHERE rowid >= 1 LIMIT {limit} OFFSET {offset}"
    else:
        sql = f"SELECT {projection} FROM {quote(table)} ORDER BY rowid DESC LIMIT {limit} OFFSET {offset}"
    question = (
        f"Return the requested {', '.join(selected)} fields from {table} for scale-up scenario "
        f"{variant}, preserving the cached result contract."
    )
    return sql, question


def add_query(database: Path, db_id: str, variant: int) -> tuple[str, str, list[str], dict[str, list[str]]]:
    tables = usable_tables(database)
    random.Random(f"{SEED}:add:{db_id}").shuffle(tables)
    left, left_columns = tables[(2 * variant) % len(tables)]
    right, right_columns = tables[(2 * variant + 1) % len(tables)]
    limit = 10 + variant % 11
    offset = variant % 6
    mode = variant % 4
    if mode == 0:
        sql = f"SELECT * FROM {quote(left)} ORDER BY rowid LIMIT {limit} OFFSET {offset}"
        targets = [left]
    elif mode == 1:
        sql = (
            f"SELECT src.* FROM {quote(left)} AS src ORDER BY src.rowid "
            f"LIMIT {limit} OFFSET {offset}"
        )
        targets = [left]
    elif mode == 2:
        sql = (
            f"SELECT lhs.*, rhs.* FROM {quote(left)} AS lhs CROSS JOIN {quote(right)} AS rhs "
            f"ORDER BY lhs.rowid, rhs.rowid LIMIT {limit} OFFSET {offset}"
        )
        targets = [left, right]
    else:
        sql = (
            f"SELECT * FROM {quote(left)} AS lhs CROSS JOIN {quote(right)} AS rhs "
            f"ORDER BY lhs.rowid, rhs.rowid LIMIT {limit} OFFSET {offset}"
        )
        targets = [left, right]
    question = f"Return complete cached records for scale-up projection scenario {variant}."
    return sql, question, targets, {left: left_columns, right: right_columns}


def source_index(split: str, family: str, db_index: int, variant: int) -> int:
    split_index = {"train": 1, "dev": 2, "test": 3}[split]
    family_index = FAMILIES.index(family) + 1
    return split_index * 100_000_000 + family_index * 10_000_000 + db_index * 10_000 + variant


def build_one(
    *, split: str, family: str, db_id: str, database: Path, db_index: int, variant: int
) -> dict[str, Any]:
    index = source_index(split, family, db_index, variant)
    source = f"p6_scaleup_v1_{split}_synthetic"
    if family == "add_column":
        sql, question, targets, columns = add_query(database, db_id, variant)
        base, declared_type, default_sql = AUDIT_SPECS[variant % len(AUDIT_SPECS)]
        specs = [
            {
                "table": table,
                "new_name": unique_added_name(f"{base}_{variant}", columns[table]),
                "declared_type": declared_type,
                "default_sql": default_sql,
            }
            for table in targets
        ]
        example = build_add_column_projection_example(
            source=source,
            source_index=index,
            db_id=db_id,
            question=question,
            evidence="",
            sql=sql,
            database=database,
            added_column_specs=specs,
        )
    else:
        # Column-level mutations require qualified references to remain
        # unambiguous on realistic multi-table schemas.  Keep the scenario
        # sequence diverse while selecting the qualified SQL shape directly;
        # rejected candidates still go through the same execution verifier.
        query_variant = (
            4 * variant + 1
            if family in {"rename_column", "replace_column", "compound"}
            else variant
        )
        sql, question = general_query(database, db_id, query_variant)
        example = GENERAL_BUILDERS[family](
            source=source,
            source_index=index,
            db_id=db_id,
            question=question,
            evidence="",
            sql=sql,
            database=database,
        )
    row = example.to_dict()
    row["scaleup"] = {
        "protocol": "p6_scaleup_v1",
        "synthetic": True,
        "generation_family": family,
        "execution_verified_at_generation": True,
        "database_split": split,
        "variant": variant,
    }
    return row


def generate_family(
    *,
    split: str,
    family: str,
    target: int,
    db_ids: list[str],
    coverage_ids: list[str],
    paths: dict[str, Path],
    cache_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows_path = cache_dir / f"{split}_{family}.jsonl"
    rejected_path = cache_dir / f"{split}_{family}_rejected.jsonl"
    rows = load_jsonl(rows_path) if rows_path.is_file() else []
    rejected = load_jsonl(rejected_path) if rejected_path.is_file() else []
    if len(rows) > target:
        rows = rows[:target]
    # Backfill caches created by an earlier resumable run.  This taxonomy
    # distinguishes a multi-table add-column projection from the deliberately
    # mixed-operation compound drift family.
    for row in rows:
        row.setdefault("scaleup", {})["generation_family"] = family
    variants = Counter()
    for row in rows:
        variants[str(row["db_id"])] = max(
            variants[str(row["db_id"])], int(row["scaleup"]["variant"]) + 1
        )
    for row in rejected:
        variants[str(row["db_id"])] = max(
            variants[str(row["db_id"])], int(row["variant"]) + 1
        )

    # Every newly introduced logical database must contribute at least one
    # task.  Large databases are touched once here, then excluded from the
    # repeated mutation pool to avoid multi-terabyte aggregate copy I/O.
    covered = {str(row["db_id"]) for row in rows}
    for db_id in coverage_ids:
        while db_id not in covered:
            variant = variants[db_id]
            variants[db_id] += 1
            try:
                row = build_one(
                    split=split,
                    family=family,
                    db_id=db_id,
                    database=paths[db_id],
                    db_index=coverage_ids.index(db_id),
                    variant=variant,
                )
                rows.append(row)
                append_jsonl(rows_path, row)
                covered.add(db_id)
            except Exception as error:
                rejection = {
                    "split": split,
                    "family": family,
                    "db_id": db_id,
                    "variant": variant,
                    "error": f"{type(error).__name__}: {error}",
                }
                rejected.append(rejection)
                append_jsonl(rejected_path, rejection)
            if variants[db_id] > 80:
                raise RuntimeError(f"Could not create coverage task for {split}/{family}/{db_id}")
        print(f"{split}/{family}: covered {db_id} ({len(rows)}/{target})", flush=True)

    # Some drift builders intentionally accept only particular SQL shapes.
    # For example, column-level mutations need qualified references on schemas
    # where common names (id/name/etc.) occur in several tables.  The query
    # generator cycles through four shapes, so three consecutive dry rounds
    # are expected and must not be treated as exhaustion.
    rounds_without_progress = 0
    while len(rows) < target:
        before = len(rows)
        for db_index, db_id in enumerate(db_ids):
            if len(rows) >= target:
                break
            variant = variants[db_id]
            variants[db_id] += 1
            try:
                row = build_one(
                    split=split,
                    family=family,
                    db_id=db_id,
                    database=paths[db_id],
                    db_index=db_index,
                    variant=variant,
                )
                rows.append(row)
                append_jsonl(rows_path, row)
            except Exception as error:  # rejection is an auditable data artifact
                rejection = {
                    "split": split,
                    "family": family,
                    "db_id": db_id,
                    "variant": variant,
                    "error": f"{type(error).__name__}: {error}",
                }
                rejected.append(rejection)
                append_jsonl(rejected_path, rejection)
        if len(rows) == before:
            rounds_without_progress += 1
        else:
            rounds_without_progress = 0
        if rounds_without_progress >= 12 or max(variants.values(), default=0) > 160:
            raise RuntimeError(
                f"Could not fill {split}/{family}: {len(rows)}/{target}; "
                f"rejections={Counter(item['error'] for item in rejected).most_common(5)}"
            )
        print(
            f"{split}/{family}: generated={len(rows)}/{target} rejected={len(rejected)}",
            flush=True,
        )
    return rows, rejected


def restratify_generation_families(
    rows: list[dict[str, Any]], legacy_profiles: dict[str, str]
) -> list[dict[str, Any]]:
    """Balance interaction profiles by intended drift family.

    The shared v2 enricher treats any schema diff with multiple operations as
    compound.  A two-table add-column projection can contain two homogeneous
    add operations, but remains part of the add-column capability we need to
    measure.  Use the audited generator family for synthetic rows, preserve
    the historical family for existing rows, and allocate 30/25/45 inside
    each resulting family (126/105/189 for every 420-row Train family).
    """

    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        scaleup = row.get("scaleup") or {}
        family = str(scaleup.get("generation_family") or row["drift_type"])
        if family != "clean":
            families[family].append(row)
        row["drift_type"] = family

    profiles: dict[str, str] = dict(legacy_profiles)
    for family, family_rows in sorted(families.items()):
        generated = sorted(
            (row for row in family_rows if str(row["task_id"]) not in legacy_profiles),
            key=lambda row: hashlib.sha256(str(row["task_id"]).encode()).hexdigest(),
        )
        desired = {
            "must_ask": round(len(family_rows) * 0.30),
            "knowledge_only": round(len(family_rows) * 0.25),
        }
        desired["schema_only"] = len(family_rows) - sum(desired.values())
        legacy_counts = Counter(
            legacy_profiles[str(row["task_id"])]
            for row in family_rows
            if str(row["task_id"]) in legacy_profiles
        )
        remaining = {
            profile: desired[profile] - legacy_counts[profile]
            for profile in ("must_ask", "knowledge_only", "schema_only")
        }
        if min(remaining.values(), default=0) < 0 or sum(remaining.values()) != len(generated):
            raise RuntimeError(
                f"Cannot preserve legacy profiles while balancing {family}: "
                f"desired={desired} legacy={dict(legacy_counts)} generated={len(generated)}"
            )
        cursor = 0
        for profile in ("must_ask", "knowledge_only", "schema_only"):
            for row in generated[cursor : cursor + remaining[profile]]:
                profiles[str(row["task_id"])] = profile
            cursor += remaining[profile]

    for row in rows:
        family = str(row["drift_type"])
        row["interaction_profile"] = (
            "direct_clean" if family == "clean" else profiles[str(row["task_id"])]
        )
        row["stratum"] = "|".join(
            str(row[key])
            for key in (
                "scenario_type",
                "drift_type",
                "interaction_profile",
                "difficulty",
                "failure_mode",
            )
        )
    return rows


def describe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cells = Counter((str(row["drift_type"]), str(row["interaction_profile"])) for row in rows)
    return {
        "tasks": len(rows),
        "databases": len({str(row["db_id"]) for row in rows}),
        "drift_types": dict(sorted(Counter(str(row["drift_type"]) for row in rows).items())),
        "profiles": dict(sorted(Counter(str(row["interaction_profile"]) for row in rows).items())),
        "drift_profile_cells": {
            f"{drift}|{profile}": count for (drift, profile), count in sorted(cells.items())
        },
        "unique_task_ids": len({str(row["task_id"]) for row in rows}),
        "unique_db_sql_drift": len(
            {(str(row["db_id"]), str(row["stale_sql"]), str(row["drift_type"])) for row in rows}
        ),
        "unique_db_sql_exact_schema_diff": len({exact_task_signature(row) for row in rows}),
        "execution_verified": sum(
            bool(row.get("result_fingerprint")) and bool(row.get("oracle_steps")) for row in rows
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--parts-dir",
        type=Path,
        default=ROOT / "data/processed/.p6_scaleup_v1_parts",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")

    original_base_train = load_jsonl(CURRENT_TRAIN)
    current_train_ids = {str(row["db_id"]) for row in original_base_train}
    if len(original_base_train) != 752 or len(current_train_ids) != 50:
        raise RuntimeError("Current Train752 invariant changed")
    base_train, removed_duplicate_ids = deduplicate_exact_tasks(original_base_train)
    if len(base_train) != 720 or len({str(row["db_id"]) for row in base_train}) != 50:
        raise RuntimeError(
            f"Unexpected exact-dedup result: rows={len(base_train)} "
            f"dbs={len({str(row['db_id']) for row in base_train})}"
        )
    candidates, provenance = database_candidates()
    split_dbs = allocate_databases(current_train_ids, candidates, provenance)
    if args.smoke:
        split_dbs = {name: values[:1] for name, values in split_dbs.items()}
        targets = {
            "train": {family: 1 for family in FAMILIES},
            "dev": {family: 1 for family in FAMILIES},
            "test": {family: 1 for family in FAMILIES},
        }
        base_train = []
    else:
        base_counts = Counter(str(row["drift_type"]) for row in base_train)
        targets = {
            "train": {
                "add_column": 420 - base_counts["add_column"],
                "rename_column": 420 - base_counts["rename_column"],
                "rename_table": 420 - base_counts["rename_table"],
                "replace_column": 420 - base_counts["replace_column"],
                "compound": 420 - base_counts["compound"],
                "clean": 300 - base_counts["clean"],
            },
            "dev": {family: 72 for family in FAMILIES},
            "test": {
                "add_column": 60,
                "rename_column": 60,
                "rename_table": 60,
                "replace_column": 60,
                "compound": 60,
                "clean": 20,
            },
        }
    all_ids = set().union(*(set(values) for values in split_dbs.values()))
    preferred: dict[str, Path] = {}
    for row in base_train:
        preferred.setdefault(str(row["db_id"]), Path(str(row["source_db"])))
    paths, resolution = resolve_databases(all_ids, candidates, preferred)
    args.parts_dir.mkdir(parents=True, exist_ok=True)

    output_rows: dict[str, list[dict[str, Any]]] = {}
    rejections: list[dict[str, Any]] = []
    for split in ("train", "dev", "test"):
        generated: list[dict[str, Any]] = []
        small_databases = sorted(
            (db_id for db_id in split_dbs[split] if paths[db_id].stat().st_size <= 128 * 1024 * 1024),
            key=lambda db_id: (paths[db_id].stat().st_size, db_id),
        )
        minimum_small = 1 if args.smoke else 10
        if len(small_databases) < minimum_small:
            raise RuntimeError(f"Too few bounded-I/O databases for {split}: {len(small_databases)}")
        for family in FAMILIES:
            coverage_ids: list[str] = []
            if family == "clean":
                coverage_ids = (
                    sorted(set(split_dbs[split]) - current_train_ids)
                    if split == "train"
                    else list(split_dbs[split])
                )
            family_rows, family_rejected = generate_family(
                split=split,
                family=family,
                target=targets[split][family],
                db_ids=small_databases,
                coverage_ids=coverage_ids,
                paths=paths,
                cache_dir=args.parts_dir,
            )
            generated.extend(family_rows)
            rejections.extend(family_rejected)
        rows = ([*base_train, *generated] if split == "train" else generated)
        legacy_profiles = {
            str(row["task_id"]): str(row["interaction_profile"])
            for row in base_train
        }
        rows = restratify_generation_families(
            enrich(rows, set(legacy_profiles)), legacy_profiles
        )
        rows.sort(key=lambda row: str(row["task_id"]))
        output_rows[split] = rows

    descriptions = {split: describe(rows) for split, rows in output_rows.items()}
    if not args.smoke:
        expected = {"train": 2400, "dev": 432, "test": 320}
        if {split: len(rows) for split, rows in output_rows.items()} != expected:
            raise RuntimeError({split: len(rows) for split, rows in output_rows.items()})
        non_unique = {
            split: len(rows) - descriptions[split]["unique_db_sql_exact_schema_diff"]
            for split, rows in output_rows.items()
            if descriptions[split]["unique_db_sql_exact_schema_diff"] != len(rows)
        }
        if non_unique:
            raise RuntimeError(f"Non-independent exact task signatures: {non_unique}")
        train_cells = descriptions["train"]["drift_profile_cells"]
        thin = {
            key: count
            for key, count in train_cells.items()
            if not key.startswith("clean|") and count < 100
        }
        if thin:
            raise RuntimeError(f"Underfilled train drift/profile cells: {thin}")
    if any(
        set(split_dbs[left]) & set(split_dbs[right])
        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test"))
    ):
        raise RuntimeError("Final database leakage")
    all_task_ids = [str(row["task_id"]) for rows in output_rows.values() for row in rows]
    if len(all_task_ids) != len(set(all_task_ids)):
        raise RuntimeError("Task ID leakage or duplication")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    for split, rows in output_rows.items():
        write_jsonl(args.output_dir / f"{split}.jsonl", rows)
    write_jsonl(args.output_dir / "rejected_generation.jsonl", rejections)
    summary = {
        "protocol": "driftsql_p6_scaleup_v1",
        "seed": SEED,
        "split_semantics": {"train": "training", "dev": "train-derived tune", "test": "fresh blind"},
        "database_ids": split_dbs,
        "database_overlap": {"train_dev": [], "train_test": [], "dev_test": []},
        "historical_reserved_databases": sorted(HISTORICAL_RESERVED_DBS),
        "historical_dev_or_test_rows_read": False,
        "legacy_exact_duplicates_removed": {
            "count": len(removed_duplicate_ids),
            "task_ids": removed_duplicate_ids,
        },
        "source_resolution": {db_id: resolution[db_id] for db_id in sorted(all_ids)},
        "source_provenance": {
            db_id: sorted(provenance[db_id]) for db_id in sorted(all_ids)
        },
        "splits": descriptions,
        "generation_rejections": {
            "total": len(rejections),
            "top": Counter(item["error"] for item in rejections).most_common(20),
        },
        "validation": "drift factory executed original, stale-active, and repaired-active SQL",
        "fresh_blind_policy": "Never use test rows for training, failure mining, reward tuning, or model selection.",
    }
    for split in ("train", "dev", "test"):
        summary.setdefault("sha256", {})[split] = sha256(args.output_dir / f"{split}.jsonl")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    seal = {
        "protocol": "driftsql_p6_scaleup_fresh_blind_seal_v1",
        "blind_rows": len(output_rows["test"]),
        "blind_databases": len(split_dbs["test"]),
        "blind_database_ids": split_dbs["test"],
        "blind_sha256": summary["sha256"]["test"],
        "policy": summary["fresh_blind_policy"],
    }
    (args.output_dir / "fresh_blind_seal.json").write_text(
        json.dumps(seal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
