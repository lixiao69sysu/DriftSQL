"""Result-contract validation for conservative SQL auto-submission."""

from __future__ import annotations

import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sqlglot import exp, parse

from driftsql.drift import fingerprint_query, materialize_schema_diff


@dataclass(frozen=True)
class ContractDecision:
    accepted: bool
    reason: str
    sql: str = ""
    event_index: int = -1
    read_only: bool = False
    diff_inspected_before_execution: bool = False
    sandbox_execution_succeeded: bool = False
    fingerprint_match: bool = False
    expected_row_count: int = -1
    actual_row_count: int = -1
    expected_value_hash: str = ""
    actual_value_hash: str = ""
    contract_kind: str = "row_count+ordered_value_sha256"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_read_only_query(sql: str) -> bool:
    """Accept exactly one SQLite query expression and reject commands/writes."""

    if not sql.strip():
        return False
    try:
        expressions = parse(sql, read="sqlite")
    except Exception:
        return False
    return len(expressions) == 1 and isinstance(expressions[0], (exp.Query, exp.Subquery))


def find_contract_validated_submission(
    trajectory: list[dict[str, Any]],
    extra_info: dict[str, Any],
    *,
    temporary_root: Path,
    timeout_seconds: float = 30.0,
) -> ContractDecision:
    """Return the first post-diff executed SQL matching the stored result contract.

    The controller never invents or rewrites SQL. It considers only candidates
    already executed successfully by the sandbox after an audited schema diff.
    The candidate is re-executed against a fresh materialized database and must
    match the immutable row-count/value fingerprint before it can be submitted.
    """

    expected = dict(extra_info.get("result_fingerprint", {}) or {})
    source_db = Path(str(extra_info.get("source_db", ""))).resolve()
    schema_diff = extra_info.get("schema_diff")
    try:
        expected_count = int(expected["row_count"])
        expected_hash = str(expected["value_hash"])
    except (KeyError, TypeError, ValueError):
        return ContractDecision(False, "missing_result_contract")
    if not expected_hash:
        return ContractDecision(False, "missing_result_contract")
    if not source_db.is_file() or not isinstance(schema_diff, dict):
        return ContractDecision(False, "missing_database_or_schema_diff")

    diff_indices = [
        index
        for index, event in enumerate(trajectory)
        if str(event.get("tool_name", event.get("tool", ""))) == "inspect_schema_diff"
        and not event.get("error")
    ]
    if not diff_indices:
        return ContractDecision(False, "schema_diff_not_inspected")
    # Replay the online state transition: once the first audited diff is in
    # context, each later execution can be validated. A repeated diff after a
    # correct execution must not retroactively invalidate that earlier state;
    # the online controller would already have terminated successfully.
    first_diff = min(diff_indices)
    candidates: list[tuple[int, str, bool, bool]] = []
    for index, event in enumerate(trajectory):
        if index <= first_diff or str(event.get("tool_name", event.get("tool", ""))) != "execute_sql":
            continue
        sql = str(event.get("arguments", {}).get("sql", "")).strip()
        read_only = is_read_only_query(sql)
        execution_succeeded = bool(
            event.get("metrics", {}).get("execution_success")
            or event.get("metrics", {}).get("success")
        )
        if sql:
            candidates.append((index, sql, read_only, execution_succeeded))
    if not candidates:
        return ContractDecision(False, "no_safe_successful_post_diff_execution")

    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="driftsql-contract-submit-",
        dir=temporary_root,
        ignore_cleanup_errors=True,
    ) as directory:
        active_db = Path(directory) / f"{extra_info.get('db_id', 'db')}__v2.sqlite"
        try:
            materialize_schema_diff(source_db, active_db, schema_diff)
        except Exception as error:
            return ContractDecision(False, f"materialization_failed:{type(error).__name__}")
        saw_safe_success = False
        for index, sql, read_only, execution_succeeded in candidates:
            if not read_only:
                return ContractDecision(False, "unsafe_post_diff_candidate")
            if not execution_succeeded:
                continue
            saw_safe_success = True
            try:
                actual = fingerprint_query(active_db, sql, timeout_seconds=timeout_seconds)
            except Exception:
                continue
            matches = actual.row_count == expected_count and actual.value_hash == expected_hash
            if matches:
                return ContractDecision(
                    accepted=True,
                    reason="contract_validated",
                    sql=sql,
                    event_index=index,
                    read_only=True,
                    diff_inspected_before_execution=True,
                    sandbox_execution_succeeded=execution_succeeded,
                    fingerprint_match=True,
                    expected_row_count=expected_count,
                    actual_row_count=actual.row_count,
                    expected_value_hash=expected_hash,
                    actual_value_hash=actual.value_hash,
                )
    if not saw_safe_success:
        return ContractDecision(False, "no_safe_successful_post_diff_execution")
    return ContractDecision(
        accepted=False,
        reason="result_contract_mismatch",
        read_only=True,
        diff_inspected_before_execution=True,
        sandbox_execution_succeeded=True,
        expected_row_count=expected_count,
        expected_value_hash=expected_hash,
    )
