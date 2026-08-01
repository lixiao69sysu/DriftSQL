#!/usr/bin/env python3
"""Run and record the real P4 product acceptance against a live service."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx


TERMINAL_STATUSES = {
    "completed",
    "failed",
    "cancelled",
    "timed_out",
    "budget_exhausted",
}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


async def get_json(client: httpx.AsyncClient, path: str) -> Any:
    response = await client.get(path)
    response.raise_for_status()
    return response.json()


async def run_acceptance(args: argparse.Namespace) -> dict[str, Any]:
    timeout = httpx.Timeout(args.http_timeout_seconds)
    async with httpx.AsyncClient(base_url=args.base_url, timeout=timeout) as client:
        health = await get_json(client, "/health")
        scenarios = await get_json(client, "/api/scenarios")
        wandb_runs = await get_json(client, "/api/observability/wandb/runs")
        wandb_history = await get_json(
            client,
            f"/api/observability/wandb/runs/{args.wandb_run_id}/history",
        )

        scenario = next(
            (
                row
                for row in scenarios
                if row["scenario_id"] == args.scenario_id
                and row["drift_type"] == "add_column"
            ),
            None,
        )
        if scenario is None:
            raise RuntimeError(f"No add-column scenario found: {args.scenario_id}")

        created_response = await client.post(
            "/api/sessions",
            json={
                "scenario_id": scenario["scenario_id"],
                "labels": {"source": "p4-real-acceptance"},
            },
        )
        created_response.raise_for_status()
        session = created_response.json()
        run_response = await client.post(
            f"/api/sessions/{session['session_id']}/run",
            json={
                "max_turns": args.max_turns,
                "timeout_seconds": args.agent_timeout_seconds,
                "max_tool_calls": args.max_tool_calls,
                "max_new_tokens": args.max_new_tokens,
                "max_total_tokens": args.max_total_tokens,
            },
        )
        run_response.raise_for_status()

        deadline = asyncio.get_running_loop().time() + args.agent_timeout_seconds + 30
        while True:
            session = await get_json(client, f"/api/sessions/{session['session_id']}")
            if session["status"] in TERMINAL_STATUSES:
                break
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("P4 real Agent did not reach a terminal state")
            await asyncio.sleep(args.poll_seconds)

        trajectory = await get_json(client, f"/api/sessions/{session['session_id']}/trajectory")
        replayed = await get_json(client, f"/api/sessions/{session['session_id']}/trajectory")
        operations = await get_json(client, "/api/observability/summary")
        failures = await get_json(client, "/api/observability/failures")

    event_counts: dict[str, int] = {}
    tool_trace: list[dict[str, Any]] = []
    for event in trajectory["events"]:
        event_type = str(event["event_type"])
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
        if event_type == "tool":
            payload = event["payload"]
            tool_trace.append(
                {
                    "tool": payload.get("tool"),
                    "success": bool(payload.get("success")),
                    "action_masked": bool(payload.get("metrics", {}).get("action_masked")),
                }
            )

    metric_names = {series["name"] for series in wandb_history.get("series", [])}
    run_ids = {run["run_id"] for run in wandb_runs.get("runs", [])}
    checks = {
        "real_vllm_backend_loaded": (
            health["status"] == "ready"
            and health["model"]["backend"] == "vllm"
            and health["model"]["loaded"] is True
            and health["model"]["persistent"] is True
        ),
        "frozen_sft20_adapter_loaded": (
            "stage8_fresh_sft_7b/global_step_20" in health["model"]["adapter"]
            and bool(health["model"]["adapter_sha256"])
            and bool(health["model"]["frozen_candidate_sha256"])
        ),
        "wandb_online_and_target_run_visible": (
            wandb_runs["configured"] is True
            and wandb_runs["status"] == "ready"
            and args.wandb_run_id in run_ids
        ),
        "wandb_reward_and_kl_curves_loaded": (
            wandb_history["status"] == "ready"
            and "actor/ppo_kl" in metric_names
            and "critic/rewards/mean" in metric_names
            and all(series["points"] for series in wandb_history["series"])
        ),
        "real_add_column_session_terminal": (
            session["drift_type"] == "add_column"
            and session["status"] in TERMINAL_STATUSES
            and session["usage"]["model_calls"] > 0
        ),
        "isolated_read_only_sandbox_used": (
            session["sandbox_isolated"] is True
            and bool(session["sandbox_ref"])
            and bool(session["source_db_sha256"])
            and health["details"]["sql_policy"] == "read-only"
        ),
        "model_and_tool_trajectory_persisted": (
            event_counts.get("model", 0) > 0
            and event_counts.get("tool", 0) > 0
            and len(trajectory["events"]) > 0
        ),
        "trajectory_replay_is_byte_stable": canonical_sha256(trajectory) == canonical_sha256(replayed),
        "operations_include_acceptance_session": operations["total_sessions"] > 0,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "protocol": "driftsql_p4_real_acceptance_v1",
        "verified_at": datetime.now(UTC).isoformat(),
        "base_url": args.base_url,
        "passed": not failed,
        "checks": checks,
        "failed": failed,
        "evidence": {
            "service": {
                "status": health["status"],
                "backend": health["model"]["backend"],
                "model_loaded": health["model"]["loaded"],
                "adapter_sha256": health["model"]["adapter_sha256"],
                "frozen_candidate_sha256": health["model"]["frozen_candidate_sha256"],
                "sandbox": health["details"]["sandbox"],
                "sql_policy": health["details"]["sql_policy"],
            },
            "wandb": {
                "entity": wandb_runs["entity"],
                "project": wandb_runs["project"],
                "run_id": args.wandb_run_id,
                "metric_names": sorted(metric_names),
                "series_points": {
                    series["name"]: len(series["points"])
                    for series in wandb_history.get("series", [])
                },
            },
            "session": {
                "session_id": session["session_id"],
                "scenario_id": session["scenario_id"],
                "db_id": session["db_id"],
                "drift_type": session["drift_type"],
                "status": session["status"],
                "termination_reason": session["termination_reason"],
                "success": session["success"],
                "model_calls": session["usage"]["model_calls"],
                "tool_calls": session["usage"]["tool_calls"],
                "elapsed_ms": session["usage"]["elapsed_ms"],
                "sandbox_ref": session["sandbox_ref"],
                "source_db_sha256": session["source_db_sha256"],
                "final_sql": session["final_sql"],
            },
            "trajectory": {
                "event_count": len(trajectory["events"]),
                "event_counts": event_counts,
                "tool_trace": tool_trace,
                "sha256": canonical_sha256(trajectory),
                "replay_sha256": canonical_sha256(replayed),
            },
            "operations": {
                "total_sessions": operations["total_sessions"],
                "terminal_sessions": operations["terminal_sessions"],
                "success_rate": operations["success_rate"],
                "classified_failures": failures["total"],
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--scenario-id", default="drift_coladd_336a8e6d4010d75e")
    parser.add_argument("--wandb-run-id", default="i57aenm4")
    parser.add_argument("--output", type=Path, default=Path("reports/service/p4_real_acceptance.json"))
    parser.add_argument("--agent-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--http-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    parser.add_argument("--max-turns", type=int, default=7)
    parser.add_argument("--max-tool-calls", type=int, default=7)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-total-tokens", type=int, default=32768)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = asyncio.run(run_acceptance(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
