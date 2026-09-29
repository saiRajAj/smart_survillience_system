"""SQLite persistence for SafeVision incidents.

Import this module with ``from database_manager import Database``.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class Database:
    """Store and retrieve SafeVision incidents in a local SQLite database."""

    def __init__(self, db_path: str | Path = "safevision.db") -> None:
        self.db_path = str(db_path)
        self._lock = threading.RLock()
        self._create_tables()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    def _create_tables(self) -> None:
        with self._lock, self._connect() as connection:
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

    @staticmethod
    def _serialize_track_ids(track_ids: Any) -> str:
        """Persist lists/dicts as JSON while accepting simple values too."""
        if isinstance(track_ids, str):
            return track_ids
        try:
            return json.dumps(track_ids)
        except (TypeError, ValueError):
            return str(track_ids)

    def add(
        self,
        severity: str,
        risk_score: float,
        reason: str | None,
        track_ids: Any,
    ) -> int:
        """Save an incident and return its database ID."""
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO incidents (timestamp, severity, risk_score, reason, track_ids)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    str(severity),
                    float(risk_score),
                    reason,
                    self._serialize_track_ids(track_ids),
                ),
            )
            return int(cursor.lastrowid)

    def recent(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return the newest incidents first as ordinary dictionaries."""
        try:
            limit = max(1, int(limit))
        except (TypeError, ValueError):
            limit = 10

        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, timestamp, severity, risk_score, reason, track_ids
                FROM incidents
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        incidents: list[dict[str, Any]] = []
        for row in rows:
            incident = dict(row)
            try:
                incident["track_ids"] = json.loads(incident["track_ids"])
            except (TypeError, json.JSONDecodeError):
                pass
            incidents.append(incident)
        return incidents

    def clear(self) -> None:
        """Remove every stored incident without deleting the database file."""
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM incidents")
