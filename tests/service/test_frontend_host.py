from __future__ import annotations

import asyncio
import re
from pathlib import Path

import httpx

from driftsql.service import create_app
from driftsql.service.settings import ServiceSettings


def settings(tmp_path: Path, frontend: Path) -> ServiceSettings:
    return ServiceSettings(
        environment="test",
        model_backend="scripted",
        repository_path=tmp_path / "repository.sqlite",
        temporary_root=tmp_path / "sandboxes",
        frontend_dist_path=frontend,
    )


def test_fastapi_serves_built_studio_and_hashed_assets(tmp_path: Path) -> None:
    frontend = tmp_path / "dist"
    assets = frontend / "assets"
    assets.mkdir(parents=True)
    (frontend / "index.html").write_text(
        '<div id="root"></div><script type="module" src="/assets/index-demo.js"></script>',
        encoding="utf-8",
    )
    (assets / "index-demo.js").write_text('document.title="DriftSQL Studio";', encoding="utf-8")
    app = create_app(settings(tmp_path, frontend))

    async def verify() -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            index = await client.get("/")
            assert index.status_code == 200
            assert index.headers["cache-control"] == "no-cache"
            asset_path = re.search(r'src="([^"]+)"', index.text).group(1)  # type: ignore[union-attr]
            asset = await client.get(asset_path)
            assert asset.status_code == 200
            assert "immutable" in asset.headers["cache-control"]
            assert "DriftSQL Studio" in asset.text

    asyncio.run(verify())


def test_unbuilt_studio_returns_actionable_503(tmp_path: Path) -> None:
    app = create_app(settings(tmp_path, tmp_path / "missing"))

    async def verify() -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.get("/")
            assert response.status_code == 503
            assert "build_frontend.sh" in response.text

    asyncio.run(verify())
