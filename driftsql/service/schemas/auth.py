"""Authentication request and browser-session status contracts."""

from __future__ import annotations

from datetime import datetime

from pydantic import SecretStr

from .common import StrictModel


class AuthLogin(StrictModel):
    api_key: SecretStr


class AuthStatus(StrictModel):
    enabled: bool
    authenticated: bool
    expires_at: datetime | None = None

