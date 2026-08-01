from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from driftsql.service import create_app
from driftsql.service.inference.backend import ScriptedModelBackend
from driftsql.service.settings import ServiceSettings


def test_api_key_auth_protects_api_but_keeps_health_public(tmp_path: Path) -> None:
    async def run() -> None:
        settings = ServiceSettings(
            environment="test",
            model_backend="scripted",
            auth_enabled=True,
            api_key="production-secret",
            auth_cookie_secure=False,
            repository_path=tmp_path / "repository.sqlite",
            temporary_root=tmp_path / "sandboxes",
        )
        app = create_app(settings, backend=ScriptedModelBackend())
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                assert (await client.get("/health")).status_code == 200
                status = await client.get("/auth/status")
                assert status.json() == {
                    "enabled": True,
                    "authenticated": False,
                    "expires_at": None,
                }
                unauthorized = await client.get("/api/scenarios")
                assert unauthorized.status_code == 401
                assert unauthorized.headers["www-authenticate"] == "Bearer"
                assert (
                    await client.get(
                        "/api/scenarios",
                        headers={"X-DriftSQL-API-Key": "production-secret"},
                    )
                ).status_code == 200
                invalid_login = await client.post(
                    "/auth/login",
                    json={"api_key": "wrong-secret"},
                )
                assert invalid_login.status_code == 401
                login = await client.post(
                    "/auth/login",
                    json={"api_key": "production-secret"},
                )
                assert login.status_code == 200
                assert login.json()["authenticated"] is True
                assert "httponly" in login.headers["set-cookie"].lower()
                assert "production-secret" not in login.text
                assert (await client.get("/api/scenarios")).status_code == 200
                assert (await client.get("/auth/status")).json()["authenticated"] is True
                logout = await client.post("/auth/logout")
                assert logout.status_code == 200
                assert (await client.get("/api/scenarios")).status_code == 401
                assert (
                    await client.get(
                        "/api/scenarios",
                        headers={"Authorization": "Bearer production-secret"},
                    )
                ).status_code == 200

    asyncio.run(run())


def test_auth_enabled_requires_nonempty_key() -> None:
    try:
        ServiceSettings(auth_enabled=True)
    except ValueError as error:
        assert "DRIFTSQL_SERVICE_API_KEY" in str(error)
    else:
        raise AssertionError("Authentication accepted a missing API key")


def test_auth_disabled_requires_no_login(tmp_path: Path) -> None:
    async def run() -> None:
        settings = ServiceSettings(
            environment="test",
            model_backend="scripted",
            repository_path=tmp_path / "repository.sqlite",
            temporary_root=tmp_path / "sandboxes",
        )
        app = create_app(settings, backend=ScriptedModelBackend())
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                assert (await client.get("/auth/status")).json() == {
                    "enabled": False,
                    "authenticated": True,
                    "expires_at": None,
                }

    asyncio.run(run())
