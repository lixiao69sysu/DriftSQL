"""Small typed HTTP/SSE client shared by one-shot and interactive CLI modes."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx


class DriftSQLApiError(RuntimeError):
    def __init__(self, message: str, status_code: int = 0) -> None:
        super().__init__(message)
        self.status_code = status_code


class DriftSQLClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8001",
        *,
        api_key: str | None = None,
        timeout: float = 180.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        headers = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout,
            transport=transport,
        )

    def close(self) -> None:
        self.http.close()

    def _json(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self.http.request(method, path, **kwargs)
        if response.is_error:
            try:
                detail = response.json().get("detail", response.text)
            except (ValueError, AttributeError):
                detail = response.text
            raise DriftSQLApiError(str(detail or response.reason_phrase), response.status_code)
        return response.json()

    def health(self) -> dict[str, Any]:
        return self._json("GET", "/health")

    def models(self) -> dict[str, Any]:
        return self._json("GET", "/api/models")

    def activate_model(self, model_id: str) -> dict[str, Any]:
        return self._json("POST", "/api/models/activate", json={"model_id": model_id})

    def databases(self) -> list[dict[str, Any]]:
        return self._json("GET", "/api/databases")

    def database_paths(self) -> list[dict[str, Any]]:
        return self._json("GET", "/api/database-paths")

    def scenarios(self) -> list[dict[str, Any]]:
        return self._json("GET", "/api/scenarios")

    def experiments(self) -> dict[str, Any]:
        return self._json("GET", "/api/experiments")

    def operations(self) -> dict[str, Any]:
        return self._json("GET", "/api/observability/summary")

    def failures(self, failure_type: str | None = None) -> dict[str, Any]:
        suffix = f"?limit=100&failure_type={failure_type}" if failure_type else "?limit=100"
        return self._json("GET", f"/api/observability/failures{suffix}")

    def replay_candidates(self, review_status: str | None = None) -> dict[str, Any]:
        suffix = f"?review_status={review_status}" if review_status else ""
        return self._json("GET", f"/api/replay/candidates{suffix}")

    def review_replay(
        self,
        candidate_id: str,
        decision: str,
        reviewer: str,
        reason: str,
    ) -> dict[str, Any]:
        return self._json(
            "POST",
            f"/api/replay/candidates/{candidate_id}/reviews",
            json={"decision": decision, "reviewer": reviewer, "reason": reason},
        )

    def wandb_runs(self) -> dict[str, Any]:
        return self._json("GET", "/api/observability/wandb/runs")

    def wandb_history(self, run_id: str) -> dict[str, Any]:
        return self._json("GET", f"/api/observability/wandb/runs/{run_id}/history")

    def sessions(self, limit: int = 20) -> dict[str, Any]:
        return self._json("GET", f"/api/sessions?limit={limit}")

    def create_query(self, db_id: str, question: str, locale: str = "zh-CN") -> dict[str, Any]:
        return self._json(
            "POST",
            "/api/query-sessions",
            json={"db_id": db_id, "question": question, "locale": locale, "labels": {"source": "cli"}},
        )

    def create_recovery(self, scenario_id: str) -> dict[str, Any]:
        return self._json(
            "POST",
            "/api/sessions",
            json={"scenario_id": scenario_id, "labels": {"source": "cli"}},
        )

    def run_session(self, session_id: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._json("POST", f"/api/sessions/{session_id}/run", json=options or {})

    def session(self, session_id: str) -> dict[str, Any]:
        return self._json("GET", f"/api/sessions/{session_id}")

    def trajectory(self, session_id: str) -> dict[str, Any]:
        return self._json("GET", f"/api/sessions/{session_id}/trajectory")

    def cancel(self, session_id: str) -> dict[str, Any]:
        return self._json("POST", f"/api/sessions/{session_id}/cancel")

    def stream_events(self, session_id: str, after_sequence: int = 0) -> Iterator[dict[str, Any]]:
        path = f"/api/sessions/{session_id}/events?after_sequence={after_sequence}"
        with self.http.stream("GET", path, headers={"Accept": "text/event-stream"}) as response:
            if response.is_error:
                response.read()
                raise DriftSQLApiError(response.text or response.reason_phrase, response.status_code)
            data_lines: list[str] = []
            for line in response.iter_lines():
                if not line:
                    if data_lines:
                        yield json.loads("\n".join(data_lines))
                        data_lines.clear()
                    continue
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
            if data_lines:
                yield json.loads("\n".join(data_lines))
