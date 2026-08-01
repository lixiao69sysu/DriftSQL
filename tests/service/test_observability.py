from __future__ import annotations

import sys
from types import SimpleNamespace

from driftsql.service.observability import WandbService
from driftsql.service.settings import ServiceSettings


def test_wandb_catalog_whitelists_numeric_training_metrics(monkeypatch) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "test-secret-that-must-not-be-serialized")
    summary = SimpleNamespace(
        _json_dict={
            "reward/mean": 0.82,
            "actor/kl": 0.013,
            "train/loss": 1.7,
            "learning_rate": 2e-6,
            "hostname": "internal-worker",
            "finished": True,
        }
    )
    run = SimpleNamespace(
        id="run-1",
        name="grpo-step-10",
        state="finished",
        url="https://wandb.ai/example/driftsql-rl/runs/run-1",
        created_at="2026-08-01T00:00:00Z",
        summary=summary,
    )
    captured: dict[str, object] = {}

    class FakeApi:
        def __init__(self, *, timeout, api_key) -> None:
            captured.update(timeout=timeout, api_key=api_key)

        def runs(self, path: str, *, per_page: int):
            captured.update(path=path, per_page=per_page)
            return [run]

        def run(self, path: str):
            captured["run_path"] = path
            return SimpleNamespace(
                history=lambda **kwargs: [
                    {"_step": 0, "reward/mean": 0.2, "actor/kl": 0.03, "hostname": "worker-1"},
                    {"_step": 5, "reward/mean": 0.8, "actor/kl": 0.01, "finished": True},
                ]
            )

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Api=FakeApi))
    settings = ServiceSettings(
        environment="test",
        model_backend="scripted",
        wandb_enabled=True,
        wandb_entity="example",
        wandb_project="driftsql-rl",
    )

    result = WandbService(settings).list_runs()

    assert result.status == "ready"
    assert result.configured is True
    assert result.runs[0].summary_metrics == {
        "reward/mean": 0.82,
        "actor/kl": 0.013,
        "train/loss": 1.7,
        "learning_rate": 2e-6,
    }
    assert captured == {
        "timeout": 10,
        "api_key": "test-secret-that-must-not-be-serialized",
        "path": "example/driftsql-rl",
        "per_page": 20,
    }
    assert "test-secret" not in result.model_dump_json()

    history = WandbService(settings).run_history("run-1")
    assert history.status == "ready"
    assert [series.name for series in history.series] == ["actor/kl", "reward/mean"]
    assert [point.value for point in history.series[1].points] == [0.2, 0.8]
    assert captured["run_path"] == "example/driftsql-rl/run-1"
