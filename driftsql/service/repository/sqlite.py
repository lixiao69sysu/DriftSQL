"""SQLite-backed session metadata and append-only trajectory event store."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from driftsql.service.schemas import (
    EventType,
    SessionRead,
    SessionStatus,
    TrajectoryEvent,
    TrajectoryRead,
)


def utcnow() -> datetime:
    return datetime.now(UTC)


class SessionNotFoundError(KeyError):
    pass


class SQLiteSessionRepository:
    """Small durable repository using one short SQLite transaction per write."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        return connection

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_status_updated
                    ON sessions(status, updated_at DESC);

                CREATE TABLE IF NOT EXISTS events (
                    session_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY(session_id, sequence),
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_events_session_sequence
                    ON events(session_id, sequence);

                CREATE TABLE IF NOT EXISTS trajectories (
                    session_id TEXT PRIMARY KEY,
                    messages_json TEXT NOT NULL DEFAULT '[]',
                    reward_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                );
                """
            )

    def close(self) -> None:
        # Connections are intentionally short-lived; this method makes the
        # lifespan interface explicit and keeps alternate repositories simple.
        return None

    def create_session(self, session: SessionRead) -> SessionRead:
        payload = session.model_dump_json()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO sessions(session_id,status,created_at,updated_at,payload_json) VALUES(?,?,?,?,?)",
                (
                    session.session_id,
                    session.status.value,
                    session.created_at.isoformat(),
                    session.updated_at.isoformat(),
                    payload,
                ),
            )
            connection.execute(
                "INSERT INTO trajectories(session_id,messages_json,reward_json) VALUES(?,?,?)",
                (session.session_id, "[]", "{}"),
            )
        return session

    def get_session(self, session_id: str) -> SessionRead:
        with self._connect() as connection:
            row = connection.execute("SELECT payload_json FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        if row is None:
            raise SessionNotFoundError(session_id)
        return SessionRead.model_validate_json(str(row["payload_json"]))

    def save_session(self, session: SessionRead) -> SessionRead:
        session = session.model_copy(update={"updated_at": utcnow()})
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE sessions SET status=?,updated_at=?,payload_json=? WHERE session_id=?",
                (
                    session.status.value,
                    session.updated_at.isoformat(),
                    session.model_dump_json(),
                    session.session_id,
                ),
            )
        if cursor.rowcount != 1:
            raise SessionNotFoundError(session.session_id)
        return session

    def list_sessions(self, *, limit: int = 100, offset: int = 0) -> list[SessionRead]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM sessions ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [SessionRead.model_validate_json(str(row["payload_json"])) for row in rows]

    def count_sessions(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM sessions").fetchone()
        return int(row["count"])

    def list_event_payloads(self, event_type: EventType | None = None) -> list[dict[str, Any]]:
        """Return persisted event payloads for aggregate operational metrics."""
        query = "SELECT payload_json FROM events"
        parameters: tuple[str, ...] = ()
        if event_type is not None:
            query += " WHERE event_type=?"
            parameters = (event_type.value,)
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [json.loads(str(row["payload_json"])) for row in rows]

    def append_event(
        self,
        session_id: str,
        event_type: EventType,
        payload: dict[str, Any],
    ) -> TrajectoryEvent:
        created = utcnow()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            exists = connection.execute("SELECT 1 FROM sessions WHERE session_id=?", (session_id,)).fetchone()
            if exists is None:
                raise SessionNotFoundError(session_id)
            row = connection.execute(
                "SELECT COALESCE(MAX(sequence),0)+1 AS sequence FROM events WHERE session_id=?",
                (session_id,),
            ).fetchone()
            sequence = int(row["sequence"])
            event = TrajectoryEvent(
                session_id=session_id,
                sequence=sequence,
                event_type=event_type,
                created_at=created,
                payload=payload,
            )
            connection.execute(
                "INSERT INTO events(session_id,sequence,event_type,created_at,payload_json) VALUES(?,?,?,?,?)",
                (
                    session_id,
                    sequence,
                    event_type.value,
                    created.isoformat(),
                    json.dumps(payload, ensure_ascii=False, default=str),
                ),
            )
        return event

    def list_events(self, session_id: str, *, after_sequence: int = 0) -> list[TrajectoryEvent]:
        self.get_session(session_id)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT sequence,event_type,created_at,payload_json FROM events "
                "WHERE session_id=? AND sequence>? ORDER BY sequence",
                (session_id, after_sequence),
            ).fetchall()
        return [
            TrajectoryEvent(
                session_id=session_id,
                sequence=int(row["sequence"]),
                event_type=EventType(str(row["event_type"])),
                created_at=datetime.fromisoformat(str(row["created_at"])),
                payload=json.loads(str(row["payload_json"])),
            )
            for row in rows
        ]

    def save_trajectory_state(
        self,
        session_id: str,
        *,
        messages: list[dict[str, Any]],
        reward: dict[str, Any],
    ) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE trajectories SET messages_json=?,reward_json=? WHERE session_id=?",
                (
                    json.dumps(messages, ensure_ascii=False, default=str),
                    json.dumps(reward, ensure_ascii=False, default=str),
                    session_id,
                ),
            )
        if cursor.rowcount != 1:
            raise SessionNotFoundError(session_id)

    def get_trajectory(self, session_id: str) -> TrajectoryRead:
        session = self.get_session(session_id)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT messages_json,reward_json FROM trajectories WHERE session_id=?",
                (session_id,),
            ).fetchone()
        if row is None:
            raise SessionNotFoundError(session_id)
        return TrajectoryRead(
            session=session,
            events=self.list_events(session_id),
            messages=json.loads(str(row["messages_json"])),
            reward=json.loads(str(row["reward_json"])),
        )

    def mark_interrupted_sessions_failed(self) -> int:
        active = {
            SessionStatus.created.value,
            SessionStatus.queued.value,
            SessionStatus.running.value,
        }
        changed = 0
        for session in self.list_sessions(limit=10000):
            if session.status.value not in active:
                continue
            failed = session.model_copy(
                update={
                    "status": SessionStatus.failed,
                    "termination_reason": "service_restart",
                    "completed_at": utcnow(),
                }
            )
            self.save_session(failed)
            changed += 1
        return changed
