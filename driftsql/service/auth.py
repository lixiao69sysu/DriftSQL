"""Short-lived in-memory browser sessions derived from the deployment API key."""

from __future__ import annotations

import secrets
from datetime import UTC, datetime, timedelta
from threading import Lock


class AuthSessionStore:
    def __init__(self, ttl_seconds: int) -> None:
        self.ttl = timedelta(seconds=ttl_seconds)
        self._sessions: dict[str, datetime] = {}
        self._lock = Lock()

    def create(self) -> tuple[str, datetime]:
        token = secrets.token_urlsafe(32)
        expires_at = datetime.now(UTC) + self.ttl
        with self._lock:
            self._purge_locked()
            self._sessions[token] = expires_at
        return token, expires_at

    def validate(self, token: str | None) -> bool:
        if not token:
            return False
        with self._lock:
            self._purge_locked()
            return token in self._sessions

    def revoke(self, token: str | None) -> None:
        if not token:
            return
        with self._lock:
            self._sessions.pop(token, None)

    def _purge_locked(self) -> None:
        now = datetime.now(UTC)
        expired = [token for token, expiry in self._sessions.items() if expiry <= now]
        for token in expired:
            self._sessions.pop(token, None)

