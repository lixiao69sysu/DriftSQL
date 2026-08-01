#!/usr/bin/env python3
"""Build Dataset V2 with atomic, compound, and clean SQL-drift strata."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from sqlglot import exp, parse_one

from driftsql.drift import (
    DriftExample,
    build_add_column_star_example,
    build_clean_example,
    build_column_rename_example,
    build_column_replacement_example,
    build_compound_drift_example,
    build_table_rename_example,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = PROJECT_ROOT / "data/generated/schema_drift/train.jsonl"
DEFAULT_BIRD = PROJECT_ROOT / "data/raw/bird23-train-filtered/data/train-00000-of-00001.jsonl"
DEFAULT_BIRD_DATABASES = PROJECT_ROOT / "data/raw/bird23-train-filtered/full/train/train_databases"
DEFAULT_SIX = PROJECT_ROOT / "data/raw/six-gym-sqlite/train.jsonl"
DEFAULT_SIX_DATABASES = PROJECT_ROOT / "data/raw/six-gym-sqlite/database"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/generated/stratified_v2/tasks.jsonl"


ATOMIC_BUILDERS: dict[str, Callable[..., DriftExample]] = {
    "rename_column": build_column_rename_example,
    "rename_table": build_table_rename_example,
    "replace_column": build_column_replacement_example,
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def difficulty(sql: str, *, compound: bool) -> str:
    if compound:
        return "hard"
    try:
        tree = parse_one(sql, read="sqlite")
    except Exception:
        return "hard"
    tables = {table.name.casefold() for table in tree.find_all(exp.Table) if table.name}
    joins = sum(1 for _ in tree.find_all(exp.Join))
    subqueries = sum(1 for _ in tree.find_all(exp.Subquery))
    complex_nodes = sum(
        1
        for node_type in (exp.Group, exp.Having, exp.Union, exp.Intersect, exp.Except)
        for _ in tree.find_all(node_type)
    )
    aggregates = sum(1 for _ in tree.find_all(exp.AggFunc))
    if len(tables) <= 1 and not joins and not subqueries and not complex_nodes and not aggregates:
        return "easy"
    if len(tables) <= 2 and subqueries == 0 and complex_nodes <= 1:
        return "medium"
    return "hard"


def operation_type(row: dict[str, Any]) -> str:
    operations = row["schema_diff"]["operations"]
    if not operations:
        return "clean"
    if len(operations) > 1:
        return "compound"
    return str(operations[0]["type"])


def enrich(rows: list[dict[str, Any]], legacy_ids: set[str]) -> list[dict[str, Any]]:
    clean_rows = [row for row in rows if operation_type(row) == "clean"]
    profiles: dict[str, str] = {}
    # Allocate the 30/25/45 mix independently inside every drift family.  A
    # global lexical sort would correlate task-id prefixes (and therefore
    # drift type) with interaction style, creating a shortcut for the model.
    drift_families = sorted(
        {operation_type(row) for row in rows if operation_type(row) != "clean"}
    )
    for family in drift_families:
        family_rows = sorted(
            (row for row in rows if operation_type(row) == family),
            key=lambda row: hashlib.sha256(str(row["task_id"]).encode()).hexdigest(),
        )
        must_ask = round(len(family_rows) * 0.30)
        knowledge_only = round(len(family_rows) * 0.25)
        for index, row in enumerate(family_rows):
            if index < must_ask:
                profile = "must_ask"
            elif index < must_ask + knowledge_only:
                profile = "knowledge_only"
            else:
                profile = "schema_only"
            profiles[str(row["task_id"])] = profile
    profiles.update({str(row["task_id"]): "direct_clean" for row in clean_rows})

    enriched: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        kind = operation_type(item)
        failure_mode = (
            "clean_no_drift"
            if kind == "clean"
            else "silent_result_mismatch"
            if str(item.get("stale_error")) == "silent_result_schema_mismatch"
            else "explicit_schema_error"
        )
        item.update(
            {
                "dataset_version": "stratified-v2",
                "scenario_type": (
                    "compound" if kind == "compound" else "clean" if kind == "clean" else "atomic"
                ),
                "drift_type": kind,
                "interaction_profile": profiles[str(item["task_id"])],
                "difficulty": difficulty(
                    str(item["stale_sql"]), compound=kind == "compound"
                ),
                "failure_mode": failure_mode,
                "legacy_task": str(item["task_id"]) in legacy_ids,
            }
        )
        item["stratum"] = "|".join(
            str(item[key])
            for key in (
                "scenario_type",
                "drift_type",
                "interaction_profile",
                "difficulty",
                "failure_mode",
            )
        )
        enriched.append(item)
    return enriched


def summary(rows: list[dict[str, Any]], rejected: dict[str, Counter[str]]) -> dict[str, Any]:
    dimensions = {}
    for key in (
        "scenario_type",
        "drift_type",
        "interaction_profile",
        "difficulty",
        "failure_mode",
        "source",
    ):
        dimensions[key] = dict(sorted(Counter(str(row[key]) for row in rows).items()))
    return {
        "name": "driftsql_stratified_dataset_v2",
        "tasks": len(rows),
        "databases": len({str(row["db_id"]) for row in rows}),
        "legacy_tasks": sum(bool(row["legacy_task"]) for row in rows),
        "dimensions": dimensions,
        "rejections": {
            key: {"total": sum(values.values()), "top": values.most_common(5)}
            for key, values in rejected.items()
        },
        "validation": "execution verified at generation; databases materialized per episode",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--bird-tasks", type=Path, default=DEFAULT_BIRD)
    parser.add_argument("--bird-databases", type=Path, default=DEFAULT_BIRD_DATABASES)
    parser.add_argument("--six-tasks", type=Path, default=DEFAULT_SIX)
    parser.add_argument("--six-databases", type=Path, default=DEFAULT_SIX_DATABASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--atomic-per-type",
        type=int,
        default=0,
        help="Override every per-atomic-family quota (mainly for smoke tests)",
    )
    parser.add_argument("--add-column", type=int, default=154)
    parser.add_argument("--rename-column", type=int, default=216)
    parser.add_argument("--rename-table", type=int, default=216)
    parser.add_argument("--replace-column", type=int, default=216)
    parser.add_argument("--compound", type=int, default=200)
    parser.add_argument("--clean", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--parts-dir",
        type=Path,
        default=None,
        help="Incremental generation cache (defaults beside the output)",
    )
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    atomic_targets = {
        "add_column": args.add_column,
        "rename_column": args.rename_column,
        "rename_table": args.rename_table,
        "replace_column": args.replace_column,
    }
    if args.atomic_per_type > 0:
        atomic_targets = {name: args.atomic_per_type for name in atomic_targets}

    base = load_jsonl(args.base_manifest)
    legacy_ids = {str(row["task_id"]) for row in base}
    parts_dir = args.parts_dir or args.output.parent / ".parts"
    if args.no_resume:
        for name in ("add_column", *ATOMIC_BUILDERS, "compound", "clean"):
            write_jsonl(parts_dir / f"{name}.jsonl", [])
            write_jsonl(parts_dir / f"rejected_{name}.jsonl", [])

    def cached(name: str) -> list[dict[str, Any]]:
        path = parts_dir / f"{name}.jsonl"
        return load_jsonl(path) if path.is_file() else []

    atomic: dict[str, list[dict[str, Any]]] = {
        name: [row for row in base if operation_type(row) == name] + cached(name)
        for name in ("add_column", *ATOMIC_BUILDERS)
    }
    rejected = {
        name: Counter(str(row["error"]) for row in cached(f"rejected_{name}"))
        for name in ("add_column", *ATOMIC_BUILDERS, "compound", "clean")
    }
    rejected_indices = {
        name: {
            int(row["source_index"])
            for row in cached(f"rejected_{name}")
        }
        for name in rejected
    }

    existing_add_indices = {int(row["source_index"]) for row in atomic["add_column"]}
    for source_index, row in enumerate(load_jsonl(args.six_tasks)):
        if len(atomic["add_column"]) >= atomic_targets["add_column"]:
            break
        if source_index in existing_add_indices:
            continue
        if source_index in rejected_indices["add_column"]:
            continue
        solution = row.get("sol_sql", [])
        if (
            str(row.get("dialect", "")).casefold() != "sqlite"
            or not isinstance(solution, list)
            or len(solution) != 1
            or row.get("preprocess_sql")
            or row.get("clean_up_sql")
        ):
            continue
        db_id = str(row["db_id"])
        database = args.six_databases / db_id / f"{db_id}_template.sqlite"
        try:
            example = build_add_column_star_example(
                source="six_gym_sqlite",
                source_index=source_index,
                db_id=db_id,
                question=str(row["query"]),
                evidence="",
                sql=str(solution[0]),
                database=database,
            )
        except Exception as error:  # noqa: BLE001 - recorded data rejection
            message = f"{type(error).__name__}: {error}"
            rejected["add_column"][message] += 1
            append_jsonl(
                parts_dir / "rejected_add_column.jsonl",
                {"source_index": source_index, "error": message},
            )
            continue
        atomic["add_column"].append(example.to_dict())
        append_jsonl(parts_dir / "add_column.jsonl", example.to_dict())
        print(f"add_column {len(atomic['add_column'])}/{atomic_targets['add_column']}", flush=True)

    bird_rows = list(enumerate(load_jsonl(args.bird_tasks)))
    random.Random(args.seed).shuffle(bird_rows)
    existing_indices = {
        name: {int(row["source_index"]) for row in values}
        for name, values in atomic.items()
        if name != "add_column"
    }
    compound: list[dict[str, Any]] = cached("compound")
    clean: list[dict[str, Any]] = cached("clean")
    existing_compound_indices = {int(row["source_index"]) for row in compound}
    existing_clean_indices = {int(row["source_index"]) for row in clean}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for source_index, row in bird_rows:
            db_id = str(row["db_id"])
            database = args.bird_databases / db_id / f"{db_id}.sqlite"
            common = {
                "source": "bird23_train_filtered",
                "source_index": source_index,
                "db_id": db_id,
                "question": str(row["question"]),
                "evidence": str(row.get("evidence", "")),
                "sql": str(row["SQL"]),
                "database": database,
            }
            jobs = {}
            for name, builder in ATOMIC_BUILDERS.items():
                if (
                    len(atomic[name]) < atomic_targets[name]
                    and source_index not in existing_indices[name]
                    and source_index not in rejected_indices[name]
                ):
                    jobs[name] = executor.submit(builder, **common)
            if (
                len(compound) < args.compound
                and source_index not in existing_compound_indices
                and source_index not in rejected_indices["compound"]
            ):
                jobs["compound"] = executor.submit(build_compound_drift_example, **common)
            if (
                len(clean) < args.clean
                and source_index not in existing_clean_indices
                and source_index not in rejected_indices["clean"]
            ):
                jobs["clean"] = executor.submit(build_clean_example, **common)

            for name, future in jobs.items():
                try:
                    example = future.result()
                except Exception as error:  # noqa: BLE001
                    message = f"{type(error).__name__}: {error}"
                    rejected[name][message] += 1
                    append_jsonl(
                        parts_dir / f"rejected_{name}.jsonl",
                        {"source_index": source_index, "error": message},
                    )
                    continue
                item = example.to_dict()
                if name in atomic:
                    atomic[name].append(item)
                    append_jsonl(parts_dir / f"{name}.jsonl", item)
                    print(f"{name} {len(atomic[name])}/{atomic_targets[name]}", flush=True)
                elif name == "compound":
                    compound.append(item)
                    append_jsonl(parts_dir / "compound.jsonl", item)
                    print(f"compound {len(compound)}/{args.compound}", flush=True)
                else:
                    clean.append(item)
                    append_jsonl(parts_dir / "clean.jsonl", item)
                    print(f"clean {len(clean)}/{args.clean}", flush=True)
            if (
                all(len(values) >= atomic_targets[name] for name, values in atomic.items())
                and len(compound) >= args.compound
                and len(clean) >= args.clean
            ):
                break

    missing = {
        **{
            name: atomic_targets[name] - len(values)
            for name, values in atomic.items()
            if len(values) < atomic_targets[name]
        },
        **({"compound": args.compound - len(compound)} if len(compound) < args.compound else {}),
        **({"clean": args.clean - len(clean)} if len(clean) < args.clean else {}),
    }
    if missing:
        raise RuntimeError(f"Could not fill Dataset V2 quotas: {missing}")

    rows = []
    for index in range(max(atomic_targets.values())):
        for name in ("add_column", "rename_column", "rename_table", "replace_column"):
            if index < atomic_targets[name]:
                rows.append(atomic[name][index])
    rows += compound + clean
    rows = enrich(rows, legacy_ids)
    if len({str(row["task_id"]) for row in rows}) != len(rows):
        raise RuntimeError("Duplicate task_id in Dataset V2")
    write_jsonl(args.output, rows)
    result = summary(rows, rejected)
    summary_path = args.output.with_name("summary.json")
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
