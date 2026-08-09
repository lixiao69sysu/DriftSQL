#!/usr/bin/env python3
"""Adapt Dataset V2 to the current P6 state policy and result contract.

The source Train/Dev/Test split is preserved at database granularity.  Only
Train and Dev produce supervised next-action examples; Test is emitted as a
regression evaluator input and is explicitly not promoted to a fresh Gate.
Every non-clean canonical trajectory is replayed through the conservative
result-contract controller before it is accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
import tempfile
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from driftsql.controllers.validated_submit import (
    ContractDecision,
    find_contract_validated_submission,
    is_read_only_query,
)
from driftsql.data.tool_sft import (
    clarification_spec,
    expand_next_action_messages,
    use_plain_json_for_last_action,
)
from driftsql.data.trajectory import relevant_schema_ddl
from driftsql.drift import fingerprint_query, materialize_schema_diff
from driftsql.integrations.state_policy import (
    schema_diff_recovery_guidance,
    select_dynamic_tool_schemas,
)
from driftsql.integrations.verl_tools import _active_schema_for_projection, _execute_read_only
from driftsql.planning import plan_projection_contract


SPLITS = ("train", "dev", "test")
SUPERVISED_SPLITS = ("train", "dev")
TOOL_NAMES = (
    "get_schema_version",
    "inspect_schema_diff",
    "get_schema",
    "ask_user",
    "get_knowledge_definition",
    "execute_sql",
    "submit_solution",
)

P6_GENERAL_SYSTEM_PROMPT = """You are a production SQL recovery agent operating on a versioned database.
Previously valid SQL may contain stale identifiers, a stale result-column contract, or a business
term that needs clarification. Execute the cached read-only SQL once. If it fails or changes its
result contract, check the active schema version and inspect the audited schema diff before editing.
Ask one focused user question only for a documented critical ambiguity, and retrieve governed
business knowledge only when needed. Validate one repaired read-only SQL candidate in the isolated
sandbox, then submit that exact SQL.

Never guess an identifier available in inspect_schema_diff. Never repeat an identical retrieval or
SQL execution. After a post-diff SQL execution succeeds, submit the validated SQL immediately. Every
assistant turn must contain one concise <think> block followed by exactly one tool call. Finish with
submit_solution within seven assistant turns."""

P6_GENERAL_USER_TEMPLATE = """## Analytics request
{question}

## Previously valid cached SQL
{stale_sql}

The active database schema or governed business interpretation may have changed. Use the available
tools, preserve the requested result contract, validate the SQL, and submit it."""

THOUGHTS = {
    "execute_stale": "I will execute the cached read-only SQL once to observe its active behavior.",
    "execute_clean": "I will validate the cached SQL once before submitting it unchanged.",
    "version": "The cached SQL failed or changed its result contract, so I will check the schema version.",
    "diff": "I will inspect the audited schema diff instead of guessing the repair.",
    "ask": "A documented business ambiguity remains, so I will ask one focused clarification.",
    "knowledge": "I will retrieve the governed definition for the clarified business term.",
    "execute_repaired": "I will validate the repaired SQL once in the isolated database.",
    "submit": "The SQL executed successfully, so I will submit that exact validated SQL now.",
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


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
    temporary.replace(path)


def load_verified_agents(directory: Path, split: str) -> list[dict[str, Any]]:
    jsonl = directory / f"{split}_agent_eval.jsonl"
    if jsonl.is_file():
        return load_jsonl(jsonl)
    parquet = directory / f"rl_{split}.parquet"
    if parquet.is_file():
        return pq.read_table(parquet).to_pylist()
    raise FileNotFoundError(f"No verified agent records for {split} in {directory}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_tool_schemas(path: Path) -> list[dict[str, Any]]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    by_name = {
        item["tool_schema"]["function"]["name"]: item["tool_schema"]
        for item in config["tools"]
    }
    missing = sorted(set(TOOL_NAMES) - set(by_name))
    if missing:
        raise RuntimeError(f"Missing P6 tool schemas: {missing}")
    return [by_name[name] for name in TOOL_NAMES]


def assistant(thought: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": f"<think>{THOUGHTS[thought]}</think>",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        ],
    }


def tool_observation(payload: Any) -> dict[str, Any]:
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, default=str)
    return {"role": "tool", "content": text}


def append_step(
    messages: list[dict[str, Any]],
    events: list[dict[str, Any]],
    *,
    thought: str,
    name: str,
    arguments: dict[str, Any],
    observation: Any | None,
    metrics: dict[str, Any] | None = None,
) -> None:
    messages.append(assistant(thought, name, arguments))
    event = {
        "tool_name": name,
        "arguments": arguments,
        "metrics": dict(metrics or {}),
    }
    if observation is not None:
        messages.append(tool_observation(observation))
        event["observation"] = (
            observation if isinstance(observation, str) else json.dumps(observation, ensure_ascii=False)
        )
    events.append(event)


def expected_sequence(row: dict[str, Any]) -> list[str]:
    if str(row["drift_type"]) == "clean":
        return ["execute_sql", "submit_solution"]
    sequence = ["execute_sql", "get_schema_version", "inspect_schema_diff"]
    profile = str(row["interaction_profile"])
    if profile == "must_ask":
        sequence.append("ask_user")
    if profile in {"must_ask", "knowledge_only"}:
        sequence.append("get_knowledge_definition")
    sequence.extend(["execute_sql", "submit_solution"])
    return sequence


def diff_observation(row: dict[str, Any]) -> tuple[dict[str, Any], bool, bool]:
    result = deepcopy(row["schema_diff"])
    guidance = schema_diff_recovery_guidance(result)
    if guidance:
        result["recovery_guidance"] = guidance
    operations = result.get("operations", []) or []
    projection_planned = False
    projection_matches_verified_sql = False
    if any(isinstance(operation, dict) and operation.get("type") == "add_column" for operation in operations):
        active_schema = _active_schema_for_projection(Path(row["source_db"]), result)
        plan = plan_projection_contract(str(row["stale_sql"]), result, active_schema)
        result["projection_contract_plan"] = plan.to_dict()
        projection_planned = True
        # Older execution-verified add-column rows may use a different but
        # result-equivalent explicit projection.  The immutable result
        # fingerprint, not SQL string identity, is the acceptance contract.
        projection_matches_verified_sql = plan.repaired_sql == str(row["repaired_sql"])
    return result, projection_planned, projection_matches_verified_sql


def replay_database(
    row: dict[str, Any],
    *,
    temporary_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], str, dict[str, Any]]:
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="p6-general-", dir=temporary_root) as directory:
        active = Path(directory) / f"{row['db_id']}__v2.sqlite"
        materialize_schema_diff(Path(row["source_db"]), active, row["schema_diff"])
        stale = _execute_read_only(active, str(row["stale_sql"]), 30.0, 5)
        repaired = (
            stale
            if str(row["stale_sql"]).strip() == str(row["repaired_sql"]).strip()
            else _execute_read_only(active, str(row["repaired_sql"]), 30.0, 5)
        )
        actual = fingerprint_query(active, str(row["repaired_sql"]), timeout_seconds=30.0)
        schema = relevant_schema_ddl(active, str(row["repaired_sql"]))
        stale_fingerprint = None
        if bool(stale.get("success")):
            stale_fingerprint = fingerprint_query(active, str(row["stale_sql"]), timeout_seconds=30.0)
    stale = dict(stale)
    repaired = dict(repaired)
    stale["elapsed_ms"] = 0.0
    repaired["elapsed_ms"] = 0.0
    expected = row["result_fingerprint"]
    validations = {
        "repaired_execution_success": bool(repaired.get("success")),
        "repaired_fingerprint_matches": (
            actual.row_count == int(expected["row_count"])
            and actual.value_hash == str(expected["value_hash"])
        ),
        "read_only_rollback": bool(repaired.get("rolled_back")),
    }
    failure_mode = str(row["failure_mode"])
    if failure_mode == "clean_no_drift":
        validations["stale_behavior_verified"] = (
            bool(stale.get("success"))
            and stale_fingerprint == actual
            and str(row["stale_sql"]).strip() == str(row["repaired_sql"]).strip()
        )
    elif failure_mode == "silent_result_mismatch":
        validations["stale_behavior_verified"] = (
            bool(stale.get("success")) and stale_fingerprint is not None and stale_fingerprint != actual
        )
    else:
        validations["stale_behavior_verified"] = not bool(stale.get("success"))
    if not all(validations.values()):
        raise RuntimeError(f"Live replay validation failed for {row['task_id']}: {validations}")
    return stale, repaired, schema, validations


def oracle_execution(row: dict[str, Any], sql: str, *, last: bool) -> dict[str, Any]:
    matches = [
        dict(step.get("observation", {}))
        for step in row.get("oracle_steps", [])
        if step.get("action") == "execute_sql"
        and str(step.get("arguments", {}).get("sql", "")).strip() == sql.strip()
    ]
    if not matches:
        raise RuntimeError(f"Missing immutable execution audit for {row['task_id']}")
    payload = matches[-1 if last else 0]
    payload["success"] = bool(payload.pop("ok", payload.get("success", False)))
    payload.setdefault("error", None)
    payload.setdefault("columns", [])
    payload.setdefault("rows", [])
    payload.setdefault("truncated", False)
    payload["rolled_back"] = True
    payload["elapsed_ms"] = 0.0
    payload["source"] = "execution_verified_dataset_v2"
    return payload


def recorded_contract_decision(
    events: list[dict[str, Any]],
    extra: dict[str, Any],
) -> ContractDecision:
    """Apply the production acceptance predicates to immutable execution audits.

    Dataset V2 already materialized and executed every episode when it was
    built.  Rebuilding multi-gigabyte databases again is deferred to model
    regression; data conversion checks the same ordering, read-only and result
    fingerprint predicates against those sealed observations.
    """

    expected = dict(extra["result_fingerprint"])
    expected_count = int(expected["row_count"])
    expected_hash = str(expected["value_hash"])
    first_diff = next(
        (
            index
            for index, event in enumerate(events)
            if event.get("tool_name") == "inspect_schema_diff" and not event.get("error")
        ),
        -1,
    )
    if first_diff < 0:
        return ContractDecision(False, "schema_diff_not_inspected")
    for index, event in enumerate(events):
        if index <= first_diff or event.get("tool_name") != "execute_sql":
            continue
        sql = str(event.get("arguments", {}).get("sql", "")).strip()
        read_only = is_read_only_query(sql)
        success = bool(event.get("metrics", {}).get("execution_success"))
        if not read_only:
            return ContractDecision(False, "unsafe_post_diff_candidate")
        if not success:
            continue
        try:
            observation = json.loads(str(event.get("observation", "{}")))
            actual_count = int(observation["row_count"])
            actual_hash = str(observation["value_hash"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        if actual_count == expected_count and actual_hash == expected_hash:
            return ContractDecision(
                accepted=True,
                reason="contract_validated",
                sql=sql,
                event_index=index,
                read_only=True,
                diff_inspected_before_execution=True,
                sandbox_execution_succeeded=True,
                fingerprint_match=True,
                expected_row_count=expected_count,
                actual_row_count=actual_count,
                expected_value_hash=expected_hash,
                actual_value_hash=actual_hash,
            )
    return ContractDecision(
        False,
        "result_contract_mismatch",
        read_only=True,
        diff_inspected_before_execution=True,
        expected_row_count=expected_count,
        expected_value_hash=expected_hash,
    )


def build_trajectory(
    row: dict[str, Any],
    verified: dict[str, Any],
    verified_agent: dict[str, Any],
    *,
    schemas: list[dict[str, Any]],
    temporary_root: Path,
    contract_root: Path,
    live_replay: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    required_prior = {
        "repaired_execution_success",
        "submitted",
        "fingerprint_matches",
        "stale_behavior_verified",
    }
    prior = dict(verified.get("validations", {}))
    if not all(bool(prior.get(name)) for name in required_prior):
        raise RuntimeError(f"Prior verification incomplete for {row['task_id']}")

    if live_replay:
        stale, repaired, active_schema, live_validations = replay_database(
            row, temporary_root=temporary_root
        )
    else:
        stale = oracle_execution(row, str(row["stale_sql"]), last=False)
        repaired = oracle_execution(row, str(row["repaired_sql"]), last=True)
        old_state = verified_agent["extra_info"]["tools_kwargs"]["execute_sql"]["create_kwargs"]
        active_schema = str(old_state.get("schema", ""))
        if not active_schema:
            raise RuntimeError(f"Missing verified active schema for {row['task_id']}")
        live_validations = {
            "repaired_execution_success": bool(prior["repaired_execution_success"]),
            "repaired_fingerprint_matches": bool(prior["fingerprint_matches"]),
            "read_only_rollback": bool(prior["rolled_back"]),
            "stale_behavior_verified": bool(prior["stale_behavior_verified"]),
        }
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": P6_GENERAL_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": P6_GENERAL_USER_TEMPLATE.format(
                question=row["question"], stale_sql=row["stale_sql"]
            ),
        },
    ]
    events: list[dict[str, Any]] = []
    clean = str(row["drift_type"]) == "clean"
    if clean:
        append_step(
            messages,
            events,
            thought="execute_clean",
            name="execute_sql",
            arguments={"sql": str(row["repaired_sql"])},
            observation=repaired,
            metrics={"execution_success": True},
        )
    else:
        append_step(
            messages,
            events,
            thought="execute_stale",
            name="execute_sql",
            arguments={"sql": str(row["stale_sql"])},
            observation=stale,
            metrics={"execution_success": bool(stale.get("success"))},
        )
        append_step(
            messages,
            events,
            thought="version",
            name="get_schema_version",
            arguments={},
            observation={
                "db_id": row["db_id"],
                "db_version": row["schema_diff"].get("to_version", "v2"),
                "metric_version": "p6-general-v1",
            },
            metrics={"schema_version_checked": True},
        )
        diff, projection_planned, projection_matches_verified_sql = diff_observation(row)
        append_step(
            messages,
            events,
            thought="diff",
            name="inspect_schema_diff",
            arguments={},
            observation=diff,
            metrics={
                "schema_diff_inspected": True,
                "projection_contract_planned": projection_planned,
                "projection_contract_matches_verified_sql": projection_matches_verified_sql,
            },
        )
        profile = str(row["interaction_profile"])
        spec = clarification_spec(row) if profile in {"must_ask", "knowledge_only"} else None
        if profile == "must_ask":
            assert spec is not None
            answer = (
                f"For '{spec['term']}', the intended business definition is: "
                f"{spec['definition']}"
            )
            append_step(
                messages,
                events,
                thought="ask",
                name="ask_user",
                arguments={"question": spec["question"]},
                observation=answer,
                metrics={"clarification_matched": True, "duplicate_question": False},
            )
        if profile in {"must_ask", "knowledge_only"}:
            assert spec is not None
            append_step(
                messages,
                events,
                thought="knowledge",
                name="get_knowledge_definition",
                arguments={"name": spec["term"]},
                observation={"query": spec["term"], "matches": [spec["knowledge_entry"]]},
                metrics={"knowledge_retrieved": True, "knowledge_matches": 1},
            )
        append_step(
            messages,
            events,
            thought="execute_repaired",
            name="execute_sql",
            arguments={"sql": str(row["repaired_sql"])},
            observation=repaired,
            metrics={"execution_success": True},
        )
    append_step(
        messages,
        events,
        thought="submit",
        name="submit_solution",
        arguments={"sql": str(row["repaired_sql"])},
        observation=None,
        metrics={"submitted": True},
    )

    actions = [str(event["tool_name"]) for event in events]
    if actions != expected_sequence(row):
        raise RuntimeError(f"P6 sequence mismatch for {row['task_id']}: {actions}")
    if len(actions) > 7 or actions[-1] != "submit_solution":
        raise RuntimeError(f"P6 turn contract failed for {row['task_id']}: {actions}")

    state = {
        "db_id": str(row["db_id"]),
        "db_version": str(row["schema_diff"].get("to_version", "v2")),
        "metric_version": "p6-general-v1",
        "source_db": str(Path(row["source_db"]).resolve()),
        "schema_diff": row["schema_diff"],
        "query": str(row["question"]),
        "stale_sql": str(row["stale_sql"]),
        "ground_truth": str(row["repaired_sql"]),
        "result_fingerprint": row["result_fingerprint"],
        "schema": active_schema,
        "sync_io": True,
        "knowledge_entries": [],
        "user_query_ambiguity": {"critical_ambiguity": [], "non_critical_ambiguity": []},
    }
    profile = str(row["interaction_profile"])
    if profile in {"must_ask", "knowledge_only"}:
        spec = clarification_spec(row)
        state["knowledge_entries"] = [spec["knowledge_entry"]]
        if profile == "must_ask":
            state["user_query_ambiguity"]["critical_ambiguity"] = [
                {
                    "term": spec["term"],
                    "sql_snippet": spec["definition"],
                    "type": spec["ambiguity_type"],
                }
            ]
    tools_kwargs = {name: {"create_kwargs": deepcopy(state)} for name in TOOL_NAMES}
    extra = {
        "instance_id": str(row["task_id"]),
        "db_id": str(row["db_id"]),
        "source_db": state["source_db"],
        "schema_diff": row["schema_diff"],
        "result_fingerprint": row["result_fingerprint"],
        "stale_sql": str(row["stale_sql"]),
        "scenario_type": str(row["scenario_type"]),
        "drift_type": str(row["drift_type"]),
        "interaction_profile": profile,
        "difficulty": str(row["difficulty"]),
        "failure_mode": str(row["failure_mode"]),
        "need_tools_kwargs": True,
        "tools_kwargs": tools_kwargs,
        "tool_selection": list(TOOL_NAMES),
        "p6_protocol": "current-state-policy+result-contract-v1",
    }
    agent_record = {
        "data_source": (
            f"driftsql/p6/general/{row['scenario_type']}/{row['drift_type']}/{profile}"
        ),
        "prompt": messages[:2],
        "ability": "interactive_sql_drift_recovery",
        "reward_model": {"ground_truth": str(row["repaired_sql"])},
        "extra_info": extra,
        "return_raw_chat": True,
        "agent_name": "driftsql_tool_agent",
    }

    contract = (
        find_contract_validated_submission(
            events,
            extra,
            temporary_root=contract_root,
            timeout_seconds=30.0,
        )
        if live_replay
        else recorded_contract_decision(events, extra)
    )
    expected_contract_acceptance = not clean
    if contract.accepted != expected_contract_acceptance:
        raise RuntimeError(
            f"Result-contract audit failed for {row['task_id']}: {contract.to_dict()}"
        )
    contract_audit = {
        "task_id": str(row["task_id"]),
        "db_id": str(row["db_id"]),
        "drift_type": str(row["drift_type"]),
        "eligible": expected_contract_acceptance,
        "expected_behavior": (
            "contract_validated" if expected_contract_acceptance else "direct_clean_model_submit"
        ),
        **contract.to_dict(),
    }
    trajectory = {
        "messages": messages,
        "task_id": str(row["task_id"]),
        "db_id": str(row["db_id"]),
        "scenario_type": str(row["scenario_type"]),
        "drift_type": str(row["drift_type"]),
        "interaction_profile": profile,
        "difficulty": str(row["difficulty"]),
        "failure_mode": str(row["failure_mode"]),
        "tool_sequence": actions,
    }
    manifest = {
        key: trajectory[key]
        for key in (
            "task_id",
            "db_id",
            "scenario_type",
            "drift_type",
            "interaction_profile",
            "difficulty",
            "failure_mode",
            "tool_sequence",
        )
    }
    manifest["prior_execution_verified"] = True
    manifest["current_live_validations"] = live_validations
    manifest["contract_behavior"] = contract_audit["expected_behavior"]
    manifest["validation_mode"] = (
        "fresh_database_replay" if live_replay else "immutable_execution_audit_replay"
    )
    manifest["projection_contract_planned"] = any(
        bool(event.get("metrics", {}).get("projection_contract_planned")) for event in events
    )
    manifest["projection_contract_matches_verified_sql"] = any(
        bool(event.get("metrics", {}).get("projection_contract_matches_verified_sql"))
        for event in events
    )
    manifest["available_tools"] = [schema["function"]["name"] for schema in schemas]
    return trajectory, agent_record, manifest, contract_audit


def expand_supervision(
    trajectory: dict[str, Any],
    *,
    schemas: list[dict[str, Any]],
    tokenizer: Any,
    max_tokens: int,
    train: bool,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for prefix in expand_next_action_messages(trajectory["messages"]):
        final = prefix[-1]
        action = str(final["tool_calls"][0]["function"]["name"])
        history = prefix[:-1]
        dynamic_schemas = select_dynamic_tool_schemas(history, schemas)
        available = [schema["function"]["name"] for schema in dynamic_schemas]
        if action not in available:
            raise RuntimeError(
                f"Dynamic mask removed target {action} for {trajectory['task_id']}: {available}"
            )
        converted = use_plain_json_for_last_action(prefix)
        token_count = len(
            tokenizer.apply_chat_template(
                converted,
                tools=dynamic_schemas,
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        )
        if token_count > max_tokens:
            raise RuntimeError(
                f"Token budget exceeded for {trajectory['task_id']}:{action}: "
                f"{token_count}>{max_tokens}"
            )
        prior_actions = [
            str(message["tool_calls"][0]["function"]["name"])
            for message in history
            if message.get("role") == "assistant" and message.get("tool_calls")
        ]
        repeats = 1
        if train:
            repeats += int(action == "inspect_schema_diff")
            repeats += int(action == "execute_sql" and "inspect_schema_diff" in prior_actions)
            repeats += 3 * int(action == "submit_solution")
        payload = {
            "messages": converted,
            "tools": json.dumps(dynamic_schemas, ensure_ascii=False),
            "enable_thinking": False,
            "target_action": action,
            "task_id": trajectory["task_id"],
            "db_id": trajectory["db_id"],
            "scenario_type": trajectory["scenario_type"],
            "drift_type": trajectory["drift_type"],
            "interaction_profile": trajectory["interaction_profile"],
            "difficulty": trajectory["difficulty"],
            "failure_mode": trajectory["failure_mode"],
            "available_tools": available,
            "token_count": token_count,
        }
        examples.extend(deepcopy(payload) for _ in range(repeats))
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", type=Path, default=ROOT / "data/processed/stratified_v2"
    )
    parser.add_argument(
        "--verified-dir", type=Path, default=ROOT / "data/processed/stratified_five_tool_v2"
    )
    parser.add_argument("--tools", type=Path, default=ROOT / "configs/tools/drift_tools.yaml")
    parser.add_argument(
        "--tokenizer", type=Path, default=ROOT / "models/Qwen2.5-Coder-7B-Instruct"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/processed/p6_generalized_protocol"
    )
    parser.add_argument("--temporary-root", type=Path, default=ROOT / "data/tmp")
    parser.add_argument("--max-tokens", type=int, default=6144)
    parser.add_argument("--seed", type=int, default=620260803)
    parser.add_argument("--limit-per-split", type=int, default=0)
    parser.add_argument(
        "--live-replay",
        action="store_true",
        help="Re-materialize every database. Intended for small smoke/regression slices only.",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")

    schemas = load_tool_schemas(args.tools)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    output: dict[str, dict[str, Any]] = {}
    all_manifests: dict[str, list[dict[str, Any]]] = {}
    all_contracts: dict[str, list[dict[str, Any]]] = {}
    all_examples: dict[str, list[dict[str, Any]]] = {}
    all_agents: dict[str, list[dict[str, Any]]] = {}
    all_trajectories: dict[str, list[dict[str, Any]]] = {}

    for split_index, split in enumerate(SPLITS):
        source_path = args.input_dir / f"{split}.jsonl"
        verified_path = args.verified_dir / f"{split}_manifest.jsonl"
        rows = load_jsonl(source_path)
        verified_rows = load_jsonl(verified_path)
        verified_by_id = {str(row["task_id"]): row for row in verified_rows}
        verified_agents = load_verified_agents(args.verified_dir, split)
        verified_agent_by_id = {
            str(row["extra_info"]["instance_id"]): row for row in verified_agents
        }
        if args.limit_per_split > 0:
            rows = rows[: args.limit_per_split]
        missing = sorted(str(row["task_id"]) for row in rows if str(row["task_id"]) not in verified_by_id)
        if missing:
            raise RuntimeError(f"{split}: missing prior verification for {missing[:5]}")
        missing_agents = sorted(
            str(row["task_id"])
            for row in rows
            if str(row["task_id"]) not in verified_agent_by_id
        )
        if missing_agents:
            raise RuntimeError(f"{split}: missing verified agent state for {missing_agents[:5]}")

        trajectories: list[dict[str, Any]] = []
        agents: list[dict[str, Any]] = []
        manifests: list[dict[str, Any]] = []
        contracts: list[dict[str, Any]] = []
        examples: list[dict[str, Any]] = []
        for index, row in enumerate(rows, 1):
            trajectory, agent, manifest, contract = build_trajectory(
                row,
                verified_by_id[str(row["task_id"])],
                verified_agent_by_id[str(row["task_id"])],
                schemas=schemas,
                temporary_root=args.temporary_root / "p6-general-live",
                contract_root=args.temporary_root / "p6-general-contract",
                live_replay=args.live_replay,
            )
            trajectories.append(trajectory)
            agents.append(agent)
            manifests.append(manifest)
            contracts.append(contract)
            if split in SUPERVISED_SPLITS:
                examples.extend(
                    expand_supervision(
                        trajectory,
                        schemas=schemas,
                        tokenizer=tokenizer,
                        max_tokens=args.max_tokens,
                        train=split == "train",
                    )
                )
            if index % 50 == 0 or index == len(rows):
                print(f"{split}: adapted {index}/{len(rows)} trajectories", flush=True)

        random.Random(args.seed + split_index).shuffle(examples)
        all_trajectories[split] = trajectories
        all_agents[split] = agents
        all_manifests[split] = manifests
        all_contracts[split] = contracts
        all_examples[split] = examples
        output[split] = {
            "trajectories": len(trajectories),
            "databases": len({row["db_id"] for row in manifests}),
            "supervision_examples": len(examples),
            "drift_types": dict(sorted(Counter(row["drift_type"] for row in manifests).items())),
            "profiles": dict(sorted(Counter(row["interaction_profile"] for row in manifests).items())),
            "difficulty": dict(sorted(Counter(row["difficulty"] for row in manifests).items())),
            "target_actions": dict(sorted(Counter(row["target_action"] for row in examples).items())),
            "contract_validated": sum(bool(row["accepted"]) for row in contracts),
            "direct_clean_model_submit": sum(
                row["expected_behavior"] == "direct_clean_model_submit" for row in contracts
            ),
        }

    databases = {
        split: {row["db_id"] for row in manifests}
        for split, manifests in all_manifests.items()
    }
    overlaps = {
        "train_dev": sorted(databases["train"] & databases["dev"]),
        "train_test": sorted(databases["train"] & databases["test"]),
        "dev_test": sorted(databases["dev"] & databases["test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Database isolation failed: {overlaps}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    for split in SPLITS:
        write_parquet(args.output_dir / f"rl_{split}.parquet", all_agents[split])
        write_jsonl(args.output_dir / f"{split}_agent_eval.jsonl", all_agents[split])
        write_jsonl(args.output_dir / f"{split}_manifest.jsonl", all_manifests[split])
        write_jsonl(args.output_dir / f"{split}_contract_audit.jsonl", all_contracts[split])
        write_parquet(args.output_dir / f"{split}_trajectories.parquet", all_trajectories[split])
        if split in SUPERVISED_SPLITS:
            write_parquet(args.output_dir / f"{split}.parquet", all_examples[split])

    token_counts = [
        int(row["token_count"])
        for split in SUPERVISED_SPLITS
        for row in all_examples[split]
    ]
    summary = {
        "protocol": "driftsql_p6_generalized_current_protocol_v1",
        "source": "execution-verified DriftSQL Dataset V2",
        "policy": (
            "current seven-tool state policy + conservative result-contract validation; "
            "database-isolated source split preserved"
        ),
        "seed": args.seed,
        "adaptation_validation_mode": (
            "fresh_database_replay"
            if args.live_replay
            else "immutable execution audit + production contract predicates"
        ),
        "splits": output,
        "database_overlap": overlaps,
        "total_trajectories": sum(len(rows) for rows in all_manifests.values()),
        "total_supervision_examples": sum(len(all_examples[split]) for split in SUPERVISED_SPLITS),
        "contract": {
            "eligible_non_clean": sum(
                row["eligible"] for rows in all_contracts.values() for row in rows
            ),
            "validated_non_clean": sum(
                row["accepted"] for rows in all_contracts.values() for row in rows
            ),
            "unsafe_auto_submissions": 0,
            "clean_policy": "model submits after one successful read-only execution; controller abstains",
        },
        "token_length": {
            "min": min(token_counts),
            "median": statistics.median(token_counts),
            "max": max(token_counts),
            "budget": args.max_tokens,
        },
        "evaluation_policy": {
            "dev": "development only",
            "test": "historical regression only; not a fresh blind Gate",
            "fresh_final_gate_required": True,
        },
        "tool_names": list(TOOL_NAMES),
        "source_sha256": {
            split: {
                "rows": sha256(args.input_dir / f"{split}.jsonl"),
                "verification": sha256(args.verified_dir / f"{split}_manifest.jsonl"),
            }
            for split in SPLITS
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
