#!/usr/bin/env python3
"""Mine and deduplicate real P6 Train on-policy failure trajectories.

The output keeps one full representative per (task, exact trajectory) and a
small manifest for every rollout.  Failure labels are multi-valued; a stable
primary label is also assigned for balanced replay and reporting.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POOL = ROOT / "data/processed/p6_scaleup_v1_rollout_pool600/train_agent_eval.jsonl"
DEFAULT_OUTPUT = ROOT / "data/processed/p6_scaleup_v1_on_policy_failures"
DEFAULT_ADAPTER = (
    ROOT
    / "checkpoints/p6_on_policy_recovery_sft_round2_mixed_7b/global_step_10/merged/lora_adapter"
)
DEFAULT_SOURCES = (
    (211, ROOT / "reports/p6_scaleup/on_policy_seed211/strong-sft.jsonl"),
    (307, ROOT / "reports/p6_scaleup/on_policy_seed307/strong-sft.jsonl"),
    (401, ROOT / "reports/p6_scaleup/on_policy_seed401/strong-sft.jsonl"),
    (101, ROOT / "reports/p6_scaleup/on_policy_seed101_supplement500/strong-sft.jsonl"),
    (503, ROOT / "reports/p6_scaleup/on_policy_seed503_targeted100/strong-sft.jsonl"),
)
TOOL_NAMES = {
    "get_schema_version",
    "inspect_schema_diff",
    "get_schema",
    "ask_user",
    "get_knowledge_definition",
    "execute_sql",
    "submit_solution",
}
EXPECTED_POST_DIFF = {
    "must_ask": "ask_user",
    "knowledge_only": "get_knowledge_definition",
    "schema_only": "execute_sql",
}
PRIMARY_PRIORITY = (
    "successful_execute_no_submit",
    "post_diff_wrong_retrieval",
    "must_ask_error",
    "compound_recovery",
    "wrong_submit",
    "invalid_tool_alias",
    "turn_limit",
    "execution_failure",
)


def digest(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)


def observation_succeeded(event: dict[str, Any]) -> bool:
    metrics = event.get("metrics")
    if isinstance(metrics, dict) and bool(
        metrics.get("execution_success") or metrics.get("success")
    ):
        return True
    value = event.get("observation", event.get("response", ""))
    if isinstance(value, dict):
        return value.get("success") is True
    if not isinstance(value, str):
        return False
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return '"success": true' in value.casefold()
    return isinstance(parsed, dict) and parsed.get("success") is True


def classify_failure(row: dict[str, Any]) -> dict[str, Any]:
    """Return stable multi-label classification for one failed rollout."""

    if bool(row.get("task_success")):
        raise ValueError("classify_failure requires a failed rollout")
    trajectory = list(row.get("trajectory") or [])
    tools = [str(event.get("tool_name", "")) for event in trajectory]
    invalid_tools = sorted({name for name in tools if name and name not in TOOL_NAMES})

    diff_index = next(
        (index for index, name in enumerate(tools) if name == "inspect_schema_diff"), None
    )
    post_diff_tool = (
        tools[diff_index + 1]
        if diff_index is not None and diff_index + 1 < len(tools)
        else None
    )
    expected_post_diff = EXPECTED_POST_DIFF.get(str(row.get("interaction_profile")))
    post_diff_wrong = bool(
        post_diff_tool is not None
        and expected_post_diff is not None
        and post_diff_tool != expected_post_diff
    )

    successful_execute_indices = [
        index
        for index, event in enumerate(trajectory)
        if tools[index] == "execute_sql" and observation_succeeded(event)
    ]
    post_diff_success_indices = [
        index
        for index in successful_execute_indices
        if diff_index is not None and index > diff_index
    ]
    successful_without_later_submit = any(
        not any(name == "submit_solution" for name in tools[index + 1 :])
        for index in successful_execute_indices
    )
    post_diff_success_without_later_submit = any(
        not any(name == "submit_solution" for name in tools[index + 1 :])
        for index in post_diff_success_indices
    )

    profile = str(row.get("interaction_profile", ""))
    exact_asks = [index for index, name in enumerate(tools) if name == "ask_user"]
    ask_aliases = sorted(
        {
            name
            for name in tools
            if name != "ask_user" and name.casefold().replace("-", "_") == "ask_user"
        }
    )
    must_ask_subtypes: list[str] = []
    if profile == "must_ask":
        if not exact_asks:
            must_ask_subtypes.append("required_ask_omitted")
        if ask_aliases:
            must_ask_subtypes.append("malformed_ask_tool")
        if exact_asks and diff_index is not None and exact_asks[0] < diff_index:
            must_ask_subtypes.append("ask_before_diff")
        if len(exact_asks) > 1:
            must_ask_subtypes.append("repeated_ask")
        if exact_asks and not must_ask_subtypes:
            must_ask_subtypes.append("asked_but_recovery_failed")
    elif exact_asks or ask_aliases:
        must_ask_subtypes.append("unnecessary_ask")

    labels: list[str] = []
    if successful_without_later_submit:
        labels.append("successful_execute_no_submit")
    if post_diff_wrong:
        labels.append("post_diff_wrong_retrieval")
    if must_ask_subtypes:
        labels.append("must_ask_error")
    if row.get("scenario_type") == "compound":
        labels.append("compound_recovery")
    if row.get("termination_reason") == "submitted":
        labels.append("wrong_submit")
    if invalid_tools:
        labels.append("invalid_tool_alias")
    if row.get("termination_reason") == "turn_limit":
        labels.append("turn_limit")
    if any(name == "execute_sql" for name in tools) and not successful_execute_indices:
        labels.append("execution_failure")
    safety = row.get("safety") or {}
    if bool(safety.get("unsafe")) or int(safety.get("unsafe_actions", 0) or 0) > 0:
        labels.append("unsafe")
    if not labels:
        labels.append("other")

    primary = next((name for name in PRIMARY_PRIORITY if name in labels), labels[0])
    return {
        "labels": labels,
        "primary_failure": primary,
        "post_diff": {
            "reached": diff_index is not None,
            "diff_index": diff_index,
            "expected_next_tool": expected_post_diff,
            "actual_next_tool": post_diff_tool,
            "wrong_retrieval": post_diff_wrong,
        },
        "terminal": {
            "successful_execute_indices": successful_execute_indices,
            "post_diff_successful_execute_indices": post_diff_success_indices,
            "successful_execute_no_submit": successful_without_later_submit,
            "post_diff_successful_execute_no_submit": (
                post_diff_success_without_later_submit
            ),
        },
        "must_ask": {
            "required": profile == "must_ask",
            "exact_ask_count": len(exact_asks),
            "aliases": ask_aliases,
            "subtypes": must_ask_subtypes,
        },
        "invalid_tools": invalid_tools,
    }


def parse_sources(values: list[str] | None) -> list[tuple[int, Path]]:
    if not values:
        return list(DEFAULT_SOURCES)
    result: list[tuple[int, Path]] = []
    for value in values:
        seed_text, separator, path_text = value.partition("=")
        if not separator:
            raise ValueError(f"Expected SEED=PATH, got {value!r}")
        result.append((int(seed_text), Path(path_text)))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument(
        "--rollout", action="append", help="Completed Train rollout as SEED=PATH"
    )
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--minimum-unique-failures", type=int, default=1000)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    pool_rows = load_jsonl(args.pool)
    train_ids = {str(row["extra_info"]["instance_id"]) for row in pool_rows}
    if len(pool_rows) != 600 or len(train_ids) != 600:
        raise RuntimeError("Failure Miner requires the unique Train pool600")

    sources = parse_sources(args.rollout)
    if len({seed for seed, _ in sources}) != len(sources):
        raise RuntimeError("Collection seeds must be unique")
    for _, path in sources:
        path_parts = {part.casefold() for part in path.resolve().parts}
        if path_parts & {"fresh", "blind", "tune", "test", "dev", "gate"}:
            raise RuntimeError(f"Non-Train source is forbidden: {path}")

    adapter_files = [
        args.adapter / "adapter_config.json",
        args.adapter / "adapter_model.safetensors",
    ]
    if not all(path.is_file() for path in adapter_files):
        raise FileNotFoundError(f"Incomplete adapter at {args.adapter}")
    adapter_file_hashes = {path.name: file_sha256(path) for path in adapter_files}
    policy_hash = digest(adapter_file_hashes)

    manifest: list[dict[str, Any]] = []
    failure_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    input_counts: dict[str, int] = {}
    input_failures: Counter[str] = Counter()
    task_exposures: Counter[str] = Counter()
    seed_by_rollout_id: dict[str, int] = {}
    for seed, path in sources:
        rows = load_jsonl(path)
        input_counts[str(seed)] = len(rows)
        seen: set[str] = set()
        for row in rows:
            task_id = str(row["instance_id"])
            if task_id not in train_ids:
                raise RuntimeError(f"Task outside Train pool: {task_id}")
            if task_id in seen:
                raise RuntimeError(f"Duplicate task for seed {seed}: {task_id}")
            seen.add(task_id)
            task_exposures[task_id] += 1
            trajectory_hash = digest(row.get("trajectory") or [])
            dedupe_key = digest({"task_id": task_id, "trajectory": trajectory_hash})
            rollout_id = digest(
                {
                    "policy_hash": policy_hash,
                    "task_id": task_id,
                    "seed": seed,
                    "trajectory": trajectory_hash,
                }
            )
            if rollout_id in seed_by_rollout_id:
                raise RuntimeError(f"Duplicate rollout identity: {rollout_id}")
            seed_by_rollout_id[rollout_id] = seed
            classification = None
            if not bool(row.get("task_success")):
                classification = classify_failure(row)
                input_failures[str(seed)] += 1
                failure_groups[dedupe_key].append(
                    {
                        "seed": seed,
                        "rollout_id": rollout_id,
                        "trajectory_sha256": trajectory_hash,
                        "classification": classification,
                        "row": row,
                    }
                )
            manifest.append(
                {
                    "rollout_id": rollout_id,
                    "collection_seed": seed,
                    "task_id": task_id,
                    "db_id": str(row.get("db_id", "")),
                    "task_success": bool(row.get("task_success")),
                    "termination_reason": str(row.get("termination_reason", "")),
                    "trajectory_sha256": trajectory_hash,
                    "dedupe_key": dedupe_key,
                    "primary_failure": (
                        classification["primary_failure"] if classification else None
                    ),
                    "failure_labels": classification["labels"] if classification else [],
                }
            )

    if sum(input_counts.values()) != 2400:
        raise RuntimeError(f"Expected 2,400 rollouts, got {sum(input_counts.values())}")
    if len(task_exposures) != 600:
        raise RuntimeError(f"Expected coverage of 600 Train tasks, got {len(task_exposures)}")

    failures: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for dedupe_key, group in sorted(failure_groups.items()):
        representative = group[0]
        classification_hashes = {digest(item["classification"]) for item in group}
        if len(classification_hashes) != 1:
            raise RuntimeError(f"Classification differs inside exact duplicate {dedupe_key}")
        output = copy.deepcopy(representative["row"])
        output["_failure_miner"] = {
            "dedupe_key": dedupe_key,
            "trajectory_sha256": representative["trajectory_sha256"],
            "representative_rollout_id": representative["rollout_id"],
            "collection_seeds": sorted(item["seed"] for item in group),
            "duplicate_count": len(group),
            "policy_hash": policy_hash,
            "classification": representative["classification"],
        }
        failures.append(output)
        if len(group) > 1:
            duplicates.append(
                {
                    "dedupe_key": dedupe_key,
                    "task_id": str(output["instance_id"]),
                    "trajectory_sha256": representative["trajectory_sha256"],
                    "rollout_ids": [item["rollout_id"] for item in group],
                    "collection_seeds": sorted(item["seed"] for item in group),
                    "collapsed_rows": len(group) - 1,
                }
            )

    raw_failure_total = sum(input_failures.values())
    if len(failures) < args.minimum_unique_failures:
        raise RuntimeError(
            f"Unique failures below target: {len(failures)}/{args.minimum_unique_failures}"
        )

    label_counts = Counter(
        label
        for row in failures
        for label in row["_failure_miner"]["classification"]["labels"]
    )
    primary_counts = Counter(
        row["_failure_miner"]["classification"]["primary_failure"] for row in failures
    )
    must_ask_subtypes = Counter(
        subtype
        for row in failures
        for subtype in row["_failure_miner"]["classification"]["must_ask"]["subtypes"]
    )
    failures.sort(
        key=lambda row: (
            row["_failure_miner"]["classification"]["primary_failure"],
            str(row["instance_id"]),
            row["_failure_miner"]["trajectory_sha256"],
        )
    )
    manifest.sort(key=lambda row: (row["collection_seed"], row["task_id"]))

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.output_dir / "failures.jsonl", failures)
    write_jsonl(args.output_dir / "all_rollouts_manifest.jsonl", manifest)
    write_jsonl(args.output_dir / "duplicates.jsonl", duplicates)
    summary = {
        "protocol": "p6_scaleup_real_on_policy_failure_miner_v1",
        "source_split": "train_only",
        "fresh_blind_rows_read": False,
        "tune_rows_read": False,
        "policy_adapter": str(args.adapter.resolve()),
        "policy_hash": policy_hash,
        "adapter_file_hashes": adapter_file_hashes,
        "input_rollouts": input_counts,
        "rollouts": sum(input_counts.values()),
        "covered_tasks": len(task_exposures),
        "covered_databases": len({str(row.get("db_id", "")) for row in failures}),
        "task_exposures": dict(sorted(Counter(task_exposures.values()).items())),
        "raw_failures_by_seed": dict(sorted(input_failures.items())),
        "raw_failures": raw_failure_total,
        "unique_failures": len(failures),
        "duplicate_failure_rows_removed": raw_failure_total - len(failures),
        "duplicate_groups": len(duplicates),
        "primary_failures": dict(sorted(primary_counts.items())),
        "multi_label_failures": dict(sorted(label_counts.items())),
        "must_ask_subtypes": dict(sorted(must_ask_subtypes.items())),
        "drift_types": dict(
            sorted(Counter(str(row.get("drift_type", "")) for row in failures).items())
        ),
        "profiles": dict(
            sorted(
                Counter(str(row.get("interaction_profile", "")) for row in failures).items()
            )
        ),
        "termination_reasons": dict(
            sorted(
                Counter(str(row.get("termination_reason", "")) for row in failures).items()
            )
        ),
        "guards": {
            "minimum_unique_failures": args.minimum_unique_failures,
            "unique_target_met": len(failures) >= args.minimum_unique_failures,
            "all_task_ids_in_train_pool600": True,
            "fresh_blind_read": False,
            "tune_read": False,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
