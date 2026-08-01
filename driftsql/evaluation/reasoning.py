"""Helpers for paired SQL Reasoning SFT evaluation."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Any

from driftsql.evaluation.bird import extract_candidate_sql


_CLOSED_SQL = re.compile(r"<sql>\s*(.*?)\s*</sql>", re.IGNORECASE | re.DOTALL)
_OPEN_SQL = re.compile(r"<sql>\s*(.*)", re.IGNORECASE | re.DOTALL)
_PLAN = re.compile(r"<plan>\s*(.*?)\s*</plan>", re.IGNORECASE | re.DOTALL)


def extract_reasoning_sql(text: str) -> str:
    """Prefer the Stage-3 SQL tag, then use the shared BIRD fallback parser."""

    matches = _CLOSED_SQL.findall(text or "")
    if matches:
        return matches[-1].strip().strip("`")
    match = _OPEN_SQL.search(text or "")
    if match:
        candidate = match.group(1).strip()
        candidate = re.split(r"</?(?:plan|answer|final)>", candidate, maxsplit=1)[0]
        return candidate.strip().strip("`")
    return extract_candidate_sql(text or "")


def reasoning_format(text: str) -> dict[str, bool]:
    return {
        "plan_tag": bool(_PLAN.search(text or "")),
        "sql_tag": bool(_CLOSED_SQL.search(text or "")),
        "exact_wrapper": bool(_PLAN.search(text or "") and _CLOSED_SQL.search(text or "")),
    }


def stable_task_score(row: dict[str, Any]) -> str:
    identity = f"{row['db_id']}|{row['source_index']}|{row['question']}"
    return hashlib.sha256(identity.encode()).hexdigest()


def stratified_database_sample(rows: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    """Round-robin across databases, using a stable hash within each DB."""

    if size <= 0:
        raise ValueError("size must be positive")
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row["db_id"])].append(row)
    for bucket in buckets.values():
        bucket.sort(key=stable_task_score)

    selected: list[dict[str, Any]] = []
    database_ids = sorted(buckets)
    while len(selected) < min(size, len(rows)):
        progressed = False
        for db_id in database_ids:
            if buckets[db_id] and len(selected) < size:
                selected.append(buckets[db_id].pop(0))
                progressed = True
        if not progressed:
            break
    return sorted(selected, key=lambda row: int(row["source_index"]))
