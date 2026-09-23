"""SQLite persistence for SafeVision incidents."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class Database:
    """Store and retrieve incidents created by the safety pipeline."""

    def __init__(self, path: str | Path = "safevision.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.init()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    def init(self) -> None:
        """Create the legacy-compatible incident table if necessary."""
        with self._lock, self.connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS incidents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    risk_score REAL NOT NULL,
                    reason TEXT,
                    track_ids TEXT
                )
                """
            )

    def add(
        self,
        severity: str,
        risk_score: float,
        reason: str | None,
        track_ids: Any,
    ) -> int:
        """Save an incident and return its database row ID."""
        try:
            serialized_track_ids = json.dumps(track_ids)
        except (TypeError, ValueError):
            serialized_track_ids = str(track_ids)

        with self._lock, self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO incidents (timestamp, severity, risk_score, reason, track_ids)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    str(severity),
                    float(risk_score),
                    reason,
                    serialized_track_ids,
                ),
            )
            return int(cursor.lastrowid)

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return newest incidents first, with JSON track IDs restored when possible."""
        try:
            limit = max(1, int(limit))
        except (TypeError, ValueError):
            limit = 20

        with self._lock, self.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, timestamp, severity, risk_score, reason, track_ids
                FROM incidents
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        incidents = [dict(row) for row in rows]
        for incident in incidents:
            try:
                incident["track_ids"] = json.loads(incident["track_ids"])
            except (TypeError, json.JSONDecodeError):
                pass
        return incidents
