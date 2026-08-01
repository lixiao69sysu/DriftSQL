#!/usr/bin/env python3
"""Run one real-backend product API trajectory without opening a TCP port."""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from pathlib import Path

import httpx

from driftsql.service import create_app
from driftsql.service.settings import ServiceSettings

TERMINAL = {"completed", "failed", "cancelled", "timed_out", "budget_exhausted"}


async def smoke(args: argparse.Namespace) -> None:
    with tempfile.TemporaryDirectory(prefix="driftsql-service-smoke-") as temporary:
        root = Path(temporary)
        settings = ServiceSettings(
            environment="production",
            model_backend="vllm",
            tensor_parallel_size=args.tensor_parallel_size,
            repository_path=root / "repository.sqlite",
            temporary_root=root / "sandboxes",
            default_timeout_seconds=args.timeout_seconds,
            maximum_timeout_seconds=args.timeout_seconds,
        )
        app = create_app(settings)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://service-smoke",
            ) as client:
                health = (await client.get("/health")).json()
                scenarios = (await client.get("/api/scenarios")).json()
                scenario = next(
                    (item for item in scenarios if item["scenario_id"] == args.scenario_id),
                    scenarios[0],
                )
                created = await client.post(
                    "/api/sessions",
                    json={"scenario_id": scenario["scenario_id"]},
                )
                created.raise_for_status()
                session = created.json()
                queued = await client.post(
                    f"/api/sessions/{session['session_id']}/run",
                    json={"timeout_seconds": args.timeout_seconds},
                )
                queued.raise_for_status()
                for _ in range(int(args.timeout_seconds * 10) + 100):
                    current = (await client.get(f"/api/sessions/{session['session_id']}")).json()
                    if current["status"] in TERMINAL:
                        break
                    await asyncio.sleep(0.1)
                else:
                    raise TimeoutError("Product service smoke did not terminate")
                trajectory = (await client.get(f"/api/sessions/{session['session_id']}/trajectory")).json()
                tool_trace = [
                    {
                        "tool": event["payload"]["tool"],
                        "success": event["payload"]["success"],
                        "action_masked": bool(event["payload"].get("metrics", {}).get("action_masked")),
                        "execution_success": event["payload"].get("metrics", {}).get("execution_success"),
                    }
                    for event in trajectory["events"]
                    if event["event_type"] == "tool"
                ]
                print(
                    json.dumps(
                        {
                            "health": health["status"],
                            "backend": health["model"]["backend"],
                            "model_loaded": health["model"]["loaded"],
                            "scenario_id": scenario["scenario_id"],
                            "status": current["status"],
                            "termination_reason": current["termination_reason"],
                            "success": current["success"],
                            "model_calls": current["usage"]["model_calls"],
                            "tool_calls": current["usage"]["tool_calls"],
                            "tool_trace": tool_trace,
                            "event_count": len(trajectory["events"]),
                            "reward_error": trajectory["reward"].get("error", ""),
                            "adapter_sha256": health["model"]["adapter_sha256"],
                        },
                        indent=2,
                    )
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario-id",
        default="drift_coladd_336a8e6d4010d75e",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(smoke(parse_args()))
