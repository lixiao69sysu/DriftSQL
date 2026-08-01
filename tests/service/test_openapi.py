from pathlib import Path

from driftsql.service import create_app
from driftsql.service.settings import ServiceSettings


def test_openapi_describes_all_product_interfaces(tmp_path: Path) -> None:
    app = create_app(
        ServiceSettings(
            environment="test",
            model_backend="scripted",
            repository_path=tmp_path / "repository.sqlite",
            temporary_root=tmp_path / "sandboxes",
        )
    )
    schema = app.openapi()
    expected_paths = {
        "/health",
        "/auth/status",
        "/auth/login",
        "/auth/logout",
        "/api/models",
        "/api/observability/failures",
        "/api/observability/summary",
        "/api/observability/wandb/runs",
        "/api/observability/wandb/runs/{run_id}/history",
        "/api/replay/candidates",
        "/api/replay/candidates/{candidate_id}/reviews",
        "/api/scenarios",
        "/api/databases",
        "/api/experiments",
        "/api/sessions",
        "/api/sessions/{session_id}",
        "/api/sessions/{session_id}/run",
        "/api/sessions/{session_id}/cancel",
        "/api/sessions/{session_id}/events",
        "/api/sessions/{session_id}/trajectory",
    }
    assert expected_paths == set(schema["paths"])
    components = schema["components"]["schemas"]
    for contract in (
        "ExperimentList",
        "AuthLogin",
        "AuthStatus",
        "FailureList",
        "OperationsSummary",
        "ReplayCandidateList",
        "ReplayReviewCreate",
        "SessionCreate",
        "SessionRead",
        "TrajectoryEvent",
        "TrajectoryRead",
        "WandbRunList",
        "WandbRunHistory",
        "RunCreate",
    ):
        assert contract in components
    event_content = schema["paths"]["/api/sessions/{session_id}/events"]["get"]["responses"]["200"]["content"]
    assert "text/event-stream" in event_content
