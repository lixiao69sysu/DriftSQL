#!/usr/bin/env python3
"""Build recovery supervision from real Train rollouts at their first error.

Each failed rollout yields a correction before the first action-sequence
divergence and, when the environment can continue safely, another correction
after observing that bad action. If the full sequence matches but the outcome
fails, the final canonical action corrects the SQL/result contract. Targets and
SQL arguments are copied only from the immutable, execution-verified Train
trajectory; Dev and Test are forbidden inputs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from sqlglot import exp, parse_one
from transformers import AutoTokenizer

from driftsql.tool_calls import find_tool_calls, remove_tool_call_payloads


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "data/processed/p6_generalized_protocol"
MODEL = ROOT / "models/Qwen2.5-Coder-7B-Instruct"
TOOLS_CONFIG = ROOT / "configs/tools/drift_tools.yaml"
TOOL_NAMES = (
    "get_schema_version",
    "inspect_schema_diff",
    "get_schema",
    "ask_user",
    "get_knowledge_definition",
    "execute_sql",
    "submit_solution",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
    temporary.replace(path)


def load_tool_schemas(path: Path) -> list[dict[str, Any]]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    by_name = {
        str(item["tool_schema"]["function"]["name"]): item["tool_schema"]
        for item in config["tools"]
    }
    missing = sorted(set(TOOL_NAMES) - set(by_name))
    if missing:
        raise RuntimeError(f"Missing tools: {missing}")
    return [by_name[name] for name in TOOL_NAMES]


def assistant_name(message: dict[str, Any]) -> str:
    calls = message.get("tool_calls", [])
    if len(calls) != 1:
        raise ValueError("Expected one canonical tool call")
    return str(calls[0]["function"]["name"])


def assistant_arguments(message: dict[str, Any]) -> dict[str, Any]:
    value = message["tool_calls"][0]["function"].get("arguments", {})
    return json.loads(value) if isinstance(value, str) else dict(value)


def first_error_index(
    rollout: dict[str, Any], canonical_assistants: list[dict[str, Any]]
) -> tuple[int | None, str]:
    events = list(rollout.get("trajectory", []))
    for index in range(min(len(events), len(canonical_assistants))):
        event = events[index]
        expected = canonical_assistants[index]
        actual_name = str(event.get("tool_name", ""))
        expected_name = assistant_name(expected)
        if actual_name != expected_name:
            return index, "action_name"
    if len(events) < len(canonical_assistants):
        return len(events), "premature_termination"
    if len(events) > len(canonical_assistants):
        return len(canonical_assistants) - 1, "extra_action_after_canonical_sequence"
    if not bool(rollout.get("task_success")):
        return len(canonical_assistants) - 1, "outcome_mismatch"
    return None, "none"


def _event_is_valid(event: dict[str, Any]) -> bool:
    if event.get("error"):
        return False
    metrics = event.get("metrics", {})
    if isinstance(metrics, dict) and bool(metrics.get("action_masked")):
        return False
    try:
        observation = json.loads(str(event.get("observation", event.get("response", ""))))
    except json.JSONDecodeError:
        observation = {}
    return not bool(observation.get("action_masked"))


def _sql_equivalent(left: str, right: str) -> bool:
    if not left.strip() or not right.strip():
        return False
    try:
        expressions = [parse_one(value, read="sqlite") for value in (left, right)]
        for expression in expressions:
            for identifier in expression.find_all(exp.Identifier):
                identifier.set("quoted", False)
        return expressions[0] == expressions[1]
    except Exception:
        return left.rstrip(";").strip().casefold() == right.rstrip(";").strip().casefold()


def state_aware_post_error_target_index(
    events: list[dict[str, Any]],
    canonical_assistants: list[dict[str, Any]],
) -> int:
    """Choose the next canonical action from the state actually reached.

    A post-error example cannot blindly replay the action at the original
    divergence position.  For example, if the model inspected the audited diff
    before checking the version, asking it to go back to the version makes the
    already-consumed one-shot diff impossible to revisit.  This helper treats
    the canonical trajectory as a small dependency graph and skips state that
    has already been established by valid observations.
    """

    names = [assistant_name(message) for message in canonical_assistants]
    valid = [
        (index, str(event.get("tool_name", "")), event)
        for index, event in enumerate(events)
        if str(event.get("tool_name", "")) and _event_is_valid(event)
    ]

    def canonical_index(name: str, *, last: bool = False) -> int | None:
        positions = [index for index, value in enumerate(names) if value == name]
        if not positions:
            return None
        return positions[-1] if last else positions[0]

    submit_index = canonical_index("submit_solution")
    repair_execute_index = canonical_index("execute_sql", last=True)
    diff_index = canonical_index("inspect_schema_diff")
    diff_events = [item for item in valid if item[1] == "inspect_schema_diff"]
    diff_position = diff_events[0][0] if diff_events else -1

    if repair_execute_index is not None and submit_index is not None:
        canonical_sql = str(
            assistant_arguments(canonical_assistants[repair_execute_index]).get("sql", "")
        )
        for event_position, name, event in valid:
            post_audit = diff_index is None or (
                diff_position >= 0 and event_position > diff_position
            )
            if name != "execute_sql" or not post_audit:
                continue
            metrics = event.get("metrics", {})
            succeeded = isinstance(metrics, dict) and bool(
                metrics.get("execution_success") or metrics.get("success")
            )
            candidate_sql = str(event.get("arguments", {}).get("sql", ""))
            if succeeded and _sql_equivalent(candidate_sql, canonical_sql):
                return submit_index

    # Clean tasks have no audited diff.  Until the canonical cached SQL has
    # executed successfully, their only recovery action is that first execute.
    if diff_index is None:
        if repair_execute_index is None:
            raise ValueError("Canonical recovery sequence has no execute_sql")
        return repair_execute_index

    if diff_position < 0:
        version_index = canonical_index("get_schema_version")
        version_done = any(name == "get_schema_version" for _, name, _ in valid)
        if version_index is not None and not version_done:
            return version_index
        return diff_index

    ask_index = canonical_index("ask_user")
    ask_events = [item for item in valid if item[1] == "ask_user" and item[0] > diff_position]
    if ask_index is not None and not ask_events:
        return ask_index

    prerequisite_position = ask_events[0][0] if ask_events else diff_position
    knowledge_index = canonical_index("get_knowledge_definition")
    knowledge_done = any(
        name == "get_knowledge_definition" and index > prerequisite_position
        for index, name, _ in valid
    )
    if knowledge_index is not None and not knowledge_done:
        return knowledge_index

    if repair_execute_index is None:
        raise ValueError("Canonical recovery sequence has no repaired execute_sql")
    return repair_execute_index


def verified_terminal_recovery(
    rollout: dict[str, Any],
    canonical_assistants: list[dict[str, Any]],
) -> tuple[int, int] | None:
    """Locate a verified repaired execution that should be submitted next.

    A merely executable stale query is not a safe terminal state.  The event
    must occur after the audited diff, report successful execution, and use SQL
    equivalent to the execution-verified canonical repair.
    """

    names = [assistant_name(message) for message in canonical_assistants]
    execute_indices = [index for index, name in enumerate(names) if name == "execute_sql"]
    submit_indices = [index for index, name in enumerate(names) if name == "submit_solution"]
    if not execute_indices or not submit_indices:
        return None
    canonical_sql = str(
        assistant_arguments(canonical_assistants[execute_indices[-1]]).get("sql", "")
    )
    events = list(rollout.get("trajectory", []))
    diff_position = next(
        (
            index
            for index, event in enumerate(events)
            if str(event.get("tool_name", "")) == "inspect_schema_diff"
        ),
        None,
    )
    if diff_position is None:
        return None
    candidates: list[int] = []
    for index, event in enumerate(events):
        if index <= diff_position or str(event.get("tool_name", "")) != "execute_sql":
            continue
        metrics = event.get("metrics", {})
        succeeded = isinstance(metrics, dict) and bool(
            metrics.get("execution_success") or metrics.get("success")
        )
        candidate_sql = str(event.get("arguments", {}).get("sql", ""))
        later_submit = any(
            str(item.get("tool_name", "")) == "submit_solution"
            for item in events[index + 1 :]
        )
        if succeeded and not later_submit and _sql_equivalent(candidate_sql, canonical_sql):
            candidates.append(index)
    if not candidates:
        return None
    return candidates[-1], submit_indices[-1]


def event_messages(event: dict[str, Any]) -> list[dict[str, Any]]:
    name = str(event.get("tool_name", "")).strip()
    if not name:
        return []
    raw_response = str(event.get("raw_response", ""))
    thought = remove_tool_call_payloads(raw_response, find_tool_calls(raw_response))
    if not thought:
        thought = "<think>I will use the observed state and choose the next safe action.</think>"
    assistant = {
        "role": "assistant",
        "content": thought,
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(event.get("arguments", {}), ensure_ascii=False),
                },
            }
        ],
    }
    observation = str(event.get("observation", ""))
    if not observation:
        observation = json.dumps(
            {"error": event.get("error", "missing observation")}, ensure_ascii=False
        )
    return [assistant, {"role": "tool", "content": observation}]


def plain_json_target(message: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(message)
    arguments = assistant_arguments(result)
    name = assistant_name(result)
    result["content"] = (
        f"{str(result.get('content', '')).rstrip()}\n"
        + json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
    )
    result.pop("tool_calls", None)
    return result


def dynamic_tools(history: list[dict[str, Any]], schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from driftsql.integrations.state_policy import select_dynamic_tool_schemas

    return select_dynamic_tool_schemas(history, schemas)


def build_example(
    *,
    task: dict[str, Any],
    rollout: dict[str, Any],
    canonical_messages: list[dict[str, Any]],
    first_error: int,
    target_index: int,
    include_error: bool,
    schemas: list[dict[str, Any]],
    tokenizer: Any,
    max_tokens: int,
    error_kind: str,
    curriculum_stage: str,
    recovery_context_override: str | None = None,
) -> dict[str, Any] | None:
    canonical_assistants = [message for message in canonical_messages if message["role"] == "assistant"]
    target = canonical_assistants[target_index]
    history = copy.deepcopy(task["prompt"])
    prefix_events = list(rollout.get("trajectory", []))[: first_error + int(include_error)]
    for event in prefix_events:
        history.extend(event_messages(event))
    available_schemas = dynamic_tools(history, schemas)
    available = [schema["function"]["name"] for schema in available_schemas]
    target_action = assistant_name(target)
    if target_action not in available:
        return None
    messages = history + [plain_json_target(target)]
    token_count = len(
        tokenizer.apply_chat_template(
            messages,
            tools=available_schemas,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    )
    if token_count > max_tokens:
        return None
    extra = task["extra_info"]
    miner = rollout.get("_failure_miner") or {}
    classification = miner.get("classification") or {}
    return {
        "messages": messages,
        "tools": json.dumps(available_schemas, ensure_ascii=False),
        "enable_thinking": False,
        "target_action": target_action,
        "task_id": str(extra["instance_id"]),
        "db_id": str(extra["db_id"]),
        "scenario_type": str(extra["scenario_type"]),
        "drift_type": str(extra["drift_type"]),
        "interaction_profile": str(extra["interaction_profile"]),
        "difficulty": str(extra["difficulty"]),
        "failure_mode": str(extra["failure_mode"]),
        "recovery_context": recovery_context_override
        or ("post_error" if include_error else "pre_error"),
        "first_error_kind": error_kind,
        "first_error_index": first_error,
        "recovery_target_index": target_index,
        "recovery_target_strategy": "state_aware" if include_error else "first_divergence",
        "available_tools": available,
        "token_count": token_count,
        "curriculum_stage": curriculum_stage,
        "failure_dedupe_key": str(miner.get("dedupe_key", "")),
        "failure_primary": str(classification.get("primary_failure", "")),
        "failure_labels": list(classification.get("labels") or []),
        "target_source": "execution_verified_train_oracle",
        "source": "real_on_policy_train_rollout+verified_train_oracle",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--agent-records", type=Path, required=True)
    parser.add_argument(
        "--canonical-trajectories",
        type=Path,
        default=PROTOCOL / "train_trajectories.parquet",
    )
    parser.add_argument("--tools", type=Path, default=TOOLS_CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=MODEL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--curriculum-stage", required=True, choices=("single", "compound", "mixed")
    )
    parser.add_argument("--max-tokens", type=int, default=3072)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--target-examples", type=int, default=1800)
    parser.add_argument("--minimum-examples", type=int, default=1500)
    parser.add_argument("--maximum-examples", type=int, default=2000)
    parser.add_argument(
        "--post-error-only",
        action="store_true",
        help="Emit only corrections conditioned on the actually reached failed state.",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    forbidden = {"dev", "test", "gate"}
    for path in (args.rollouts, args.agent_records, args.canonical_trajectories):
        if forbidden & {part.casefold() for part in path.resolve().parts}:
            raise RuntimeError(f"Recovery SFT forbids Dev/Test/Gate inputs: {path}")

    rollouts = load_jsonl(args.rollouts)
    tasks = load_jsonl(args.agent_records)
    task_by_id = {str(row["extra_info"]["instance_id"]): row for row in tasks}
    trajectories = pq.read_table(args.canonical_trajectories).to_pylist()
    canonical_by_id = {str(row["task_id"]): row for row in trajectories}
    rollout_ids = {str(row["instance_id"]) for row in rollouts}
    if not rollout_ids or not rollout_ids.issubset(task_by_id) or not rollout_ids.issubset(canonical_by_id):
        raise RuntimeError("Rollout IDs are not a subset of the selected Train protocol")
    failure_keys = [
        str((row.get("_failure_miner") or {}).get("dedupe_key", "")) for row in rollouts
    ]
    if any(bool(row.get("task_success")) for row in rollouts):
        raise RuntimeError("Recovery SFT accepts failed rollouts only")
    if any(not key for key in failure_keys) or len(set(failure_keys)) != len(rollouts):
        raise RuntimeError("Recovery SFT requires unique Failure Miner dedupe keys")
    if not (
        0 < args.minimum_examples <= args.target_examples <= args.maximum_examples
    ):
        parser.error("Expected 0 < minimum <= target <= maximum examples")
    schemas = load_tool_schemas(args.tools)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    examples: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    verified_terminal_states = 0
    for rollout in rollouts:
        task_id = str(rollout["instance_id"])
        canonical_messages = list(canonical_by_id[task_id]["messages"])
        canonical_assistants = [message for message in canonical_messages if message["role"] == "assistant"]
        index, error_kind = first_error_index(rollout, canonical_assistants)
        audit_row = {
            "task_id": task_id,
            "failure_dedupe_key": str(rollout["_failure_miner"]["dedupe_key"]),
            "task_success": bool(rollout.get("task_success")),
            "task_joined": task_id in task_by_id,
            "canonical_joined": task_id in canonical_by_id,
            "first_error_index": index,
            "first_error_kind": error_kind,
            "examples": [],
        }
        # Recovery supervision must represent states reached by failed policy
        # rollouts.  A successful rollout may legitimately use a different
        # action sequence from the oracle and must not be "corrected" merely
        # for that difference.
        if bool(rollout.get("task_success")):
            skipped["successful_rollout"] += 1
            audit.append(audit_row)
            continue
        if index is None:
            raise RuntimeError(f"Failed rollout has no recoverable divergence: {task_id}")
        for include_error in ((True,) if args.post_error_only else (False, True)):
            if include_error:
                events = list(rollout.get("trajectory", []))
                if index >= len(events) or not str(events[index].get("tool_name", "")):
                    skipped["post_error_unavailable"] += 1
                    continue
                if str(events[index].get("tool_name")) == "submit_solution":
                    skipped["post_submit_is_terminal"] += 1
                    continue
            recovery_target_index = (
                state_aware_post_error_target_index(
                    list(rollout.get("trajectory", []))[: index + 1],
                    canonical_assistants,
                )
                if include_error
                else index
            )
            example = build_example(
                task=task_by_id[task_id],
                rollout=rollout,
                canonical_messages=canonical_messages,
                first_error=index,
                target_index=recovery_target_index,
                include_error=include_error,
                schemas=schemas,
                tokenizer=tokenizer,
                max_tokens=args.max_tokens,
                error_kind=error_kind,
                curriculum_stage=args.curriculum_stage,
            )
            if example is None:
                skipped["target_masked_or_too_long"] += 1
                continue
            examples.append(example)
            audit_row["examples"].append(example["recovery_context"])

        labels = set(
            rollout["_failure_miner"]["classification"].get("labels") or []
        )
        if "successful_execute_no_submit" in labels:
            terminal = verified_terminal_recovery(rollout, canonical_assistants)
            if terminal is None:
                skipped["terminal_state_not_verified_canonical"] += 1
            else:
                verified_terminal_states += 1
                event_index, submit_index = terminal
                terminal_example = build_example(
                    task=task_by_id[task_id],
                    rollout=rollout,
                    canonical_messages=canonical_messages,
                    first_error=event_index,
                    target_index=submit_index,
                    include_error=True,
                    schemas=schemas,
                    tokenizer=tokenizer,
                    max_tokens=args.max_tokens,
                    error_kind="terminal_missing_after_verified_execute",
                    curriculum_stage=args.curriculum_stage,
                    recovery_context_override="terminal_missing",
                )
                if terminal_example is None:
                    skipped["terminal_submit_masked_or_too_long"] += 1
                else:
                    if terminal_example["target_action"] != "submit_solution":
                        raise RuntimeError("Verified terminal recovery target is not submit_solution")
                    examples.append(terminal_example)
                    audit_row["examples"].append("terminal_missing")
        audit.append(audit_row)

    raw_example_count = len(examples)
    if raw_example_count < args.minimum_examples:
        raise RuntimeError(
            f"Only {raw_example_count} recovery examples; need at least {args.minimum_examples}"
        )
    desired = min(args.target_examples, raw_example_count)
    context_priority = {"terminal_missing": 0, "post_error": 1, "pre_error": 2}
    first_per_failure: dict[str, dict[str, Any]] = {}
    for example in sorted(
        examples,
        key=lambda row: (
            context_priority.get(str(row["recovery_context"]), 9),
            str(row["failure_dedupe_key"]),
        ),
    ):
        first_per_failure.setdefault(str(example["failure_dedupe_key"]), example)
    selected_ids = {id(example) for example in first_per_failure.values()}
    selected_examples = list(first_per_failure.values())
    remaining = [example for example in examples if id(example) not in selected_ids]
    remaining.sort(
        key=lambda row: hashlib.sha256(
            (
                f"{args.seed}:{row['failure_dedupe_key']}:"
                f"{row['recovery_context']}:{row['target_action']}:"
                f"{row['first_error_index']}:{row['recovery_target_index']}"
            ).encode()
        ).hexdigest()
    )
    selected_examples.extend(remaining[: max(0, desired - len(selected_examples))])
    examples = selected_examples
    if not args.minimum_examples <= len(examples) <= args.maximum_examples:
        raise RuntimeError(f"Recovery example count outside requested bounds: {len(examples)}")
    terminal_examples = [
        row for row in examples if row["recovery_context"] == "terminal_missing"
    ]
    if not terminal_examples or any(
        row["target_action"] != "submit_solution" for row in terminal_examples
    ):
        raise RuntimeError("Terminal recovery examples must all target submit_solution")
    database_ids = sorted({str(row["db_id"]) for row in examples})
    dev_databases = {
        value
        for value in database_ids
        if int(hashlib.sha256(f"{args.seed}:{value}".encode()).hexdigest(), 16) % 5 == 0
    }
    if not dev_databases:
        dev_databases = {database_ids[-1]}
    train_rows = [row for row in examples if row["db_id"] not in dev_databases]
    dev_rows = [row for row in examples if row["db_id"] in dev_databases]
    if not train_rows or not dev_rows:
        raise RuntimeError("Database-isolated recovery train/dev split is empty")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_parquet(args.output_dir / "train.parquet", train_rows)
    write_parquet(args.output_dir / "dev.parquet", dev_rows)
    write_jsonl(args.output_dir / "audit.jsonl", audit)
    summary = {
        "protocol": "p6_real_on_policy_recovery_sft_v2_state_aware",
        "curriculum_stage": args.curriculum_stage,
        "post_error_only": bool(args.post_error_only),
        "rollouts": len(rollouts),
        "unique_failure_trajectories": len(set(failure_keys)),
        "task_associations": sum(row["task_joined"] for row in audit),
        "canonical_associations": sum(row["canonical_joined"] for row in audit),
        "rollout_success": sum(bool(row.get("task_success")) for row in rollouts),
        "failed_rollouts": sum(not bool(row.get("task_success")) for row in rollouts),
        "raw_examples_before_cap": raw_example_count,
        "examples": len(examples),
        "train_examples": len(train_rows),
        "dev_examples": len(dev_rows),
        "train_databases": len({row["db_id"] for row in train_rows}),
        "dev_databases": len({row["db_id"] for row in dev_rows}),
        "database_overlap": sorted(
            {row["db_id"] for row in train_rows} & {row["db_id"] for row in dev_rows}
        ),
        "target_actions": dict(sorted(Counter(row["target_action"] for row in examples).items())),
        "post_error_target_actions": dict(
            sorted(
                Counter(
                    row["target_action"]
                    for row in examples
                    if row["recovery_context"] == "post_error"
                ).items()
            )
        ),
        "state_aware_retargeted": sum(
            row["recovery_context"] == "post_error"
            and int(row["recovery_target_index"]) != int(row["first_error_index"])
            for row in examples
        ),
        "recovery_contexts": dict(sorted(Counter(row["recovery_context"] for row in examples).items())),
        "verified_terminal_states": verified_terminal_states,
        "terminal_missing_examples": len(terminal_examples),
        "terminal_targets": dict(
            sorted(Counter(row["target_action"] for row in terminal_examples).items())
        ),
        "target_sources": dict(sorted(Counter(row["target_source"] for row in examples).items())),
        "max_token_count": max(int(row["token_count"]) for row in examples),
        "first_error_kinds": dict(sorted(Counter(row["first_error_kind"] for row in examples).items())),
        "drift_types": dict(sorted(Counter(row["drift_type"] for row in examples).items())),
        "skipped": dict(sorted(skipped.items())),
        "split_guards": {
            "source_split": "train_only",
            "dev_rows_read": False,
            "tune_rows_read": False,
            "test_rows_read": False,
            "gate_rows_read": False,
            "fresh_blind_rows_read": False,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
