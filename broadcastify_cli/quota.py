from __future__ import annotations

import os
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


PROVIDER_ARCHIVE_REQUEST_LIMIT = 250
AUTOMATED_ARCHIVE_REQUEST_LIMIT = 240
USER_ARCHIVE_REQUEST_RESERVE = (
    PROVIDER_ARCHIVE_REQUEST_LIMIT - AUTOMATED_ARCHIVE_REQUEST_LIMIT
)
ARCHIVE_REQUEST_WINDOW_SECONDS = 24 * 60 * 60.0
RATE_LIMIT_RELEASE_GRACE_SECONDS = 5.0
DEFAULT_ARCHIVE_QUOTA_FILENAME = ".broadcastify-archive-quota.sqlite3"


class ArchiveRequestBudgetExceeded(RuntimeError):
    pass


def archive_quota_path(base_dir: str | Path | None = None) -> Path:
    configured = str(os.getenv("BROADCASTIFY_QUOTA_LEDGER") or "").strip()
    root = Path(base_dir or Path.cwd()).expanduser().resolve()
    if configured:
        path = Path(configured).expanduser()
        return path.resolve() if path.is_absolute() else (root / path).resolve()
    return root / DEFAULT_ARCHIVE_QUOTA_FILENAME


class ArchiveRequestLedger:
    """Process-safe, installation-local rolling archive request ledger."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        base_dir: str | Path | None = None,
        limit: int = AUTOMATED_ARCHIVE_REQUEST_LIMIT,
        provider_limit: int = PROVIDER_ARCHIVE_REQUEST_LIMIT,
        window_seconds: float = ARCHIVE_REQUEST_WINDOW_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = (
            Path(path).expanduser().resolve()
            if path is not None
            else archive_quota_path(base_dir)
        )
        self.limit = max(1, int(limit))
        self.provider_limit = max(self.limit, int(provider_limit))
        self.window_seconds = max(60.0, float(window_seconds))
        self.clock = clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @property
    def user_reserve(self) -> int:
        return self.provider_limit - self.limit

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS archive_quota_state (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    instance_id TEXT NOT NULL,
                    blocked_until REAL NOT NULL DEFAULT 0,
                    blocked_reason TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS archive_request_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    requested_at REAL NOT NULL,
                    feed_id TEXT NOT NULL,
                    archive_date TEXT NOT NULL,
                    archive_id TEXT NOT NULL,
                    outcome TEXT NOT NULL DEFAULT 'started',
                    http_status INTEGER
                );
                CREATE INDEX IF NOT EXISTS idx_archive_request_attempts_time
                ON archive_request_attempts(requested_at);
                """
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO archive_quota_state(
                    singleton, instance_id, blocked_until, blocked_reason
                ) VALUES (1, ?, 0, '')
                """,
                (uuid.uuid4().hex,),
            )

    @staticmethod
    def _timestamp(value: float | None) -> str:
        if value is None:
            return ""
        return datetime.fromtimestamp(value, timezone.utc).isoformat()

    def _status_locked(
        self,
        connection: sqlite3.Connection,
        now: float,
    ) -> dict[str, Any]:
        cutoff = now - self.window_seconds
        row = connection.execute(
            """
            SELECT COUNT(*) AS used, MIN(requested_at) AS oldest
            FROM archive_request_attempts
            WHERE requested_at > ?
            """,
            (cutoff,),
        ).fetchone()
        state = connection.execute(
            """
            SELECT instance_id, blocked_until, blocked_reason
            FROM archive_quota_state WHERE singleton = 1
            """
        ).fetchone()
        used = int(row["used"] if row is not None else 0)
        oldest = (
            float(row["oldest"])
            if row is not None and row["oldest"] is not None
            else None
        )
        blocked_until_value = float(state["blocked_until"] if state else 0.0)
        blocked_reason = str(state["blocked_reason"] or "") if state else ""
        # A provider 429 is rolling-window state. Once this installation's
        # oldest known request ages out, one upstream slot should be available.
        # Clamp legacy/full-window blocks to that earliest known release.
        if blocked_until_value > now and oldest is not None:
            stored_blocked_until = blocked_until_value
            blocked_until_value = min(
                blocked_until_value,
                oldest + self.window_seconds + RATE_LIMIT_RELEASE_GRACE_SECONDS,
            )
            if blocked_until_value < stored_blocked_until:
                blocked_reason = (
                    "Broadcastify returned HTTP 429; this installation is "
                    "waiting for its next known rolling-window release."
                )
        blocked = blocked_until_value > now
        capacity_available_at = (
            oldest + self.window_seconds
            if used >= self.limit and oldest is not None
            else None
        )
        next_request_at = capacity_available_at
        if blocked:
            next_request_at = max(
                blocked_until_value,
                capacity_available_at or blocked_until_value,
            )
        remaining = max(0, self.limit - used)
        return {
            "instance_id": str(state["instance_id"] if state else ""),
            "provider_limit": self.provider_limit,
            "automated_limit": self.limit,
            "user_reserve": self.user_reserve,
            "window_seconds": int(self.window_seconds),
            "used": used,
            "remaining": remaining,
            "available": remaining > 0 and not blocked,
            "blocked": blocked,
            "blocked_reason": blocked_reason if blocked else "",
            "blocked_until": self._timestamp(blocked_until_value if blocked else None),
            "next_request_at": self._timestamp(next_request_at),
            "next_request_seconds": (
                max(0, int(next_request_at - now))
                if next_request_at is not None
                else 0
            ),
        }

    def status(self) -> dict[str, Any]:
        now = float(self.clock())
        with self._connect() as connection:
            return self._status_locked(connection, now)

    def reserve(
        self,
        *,
        feed_id: str,
        archive_date: str,
        archive_id: str,
    ) -> int:
        now = float(self.clock())
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            # Retain a bounded audit tail while keeping the rolling calculation
            # exact. Six windows is enough to diagnose recent client behavior.
            connection.execute(
                "DELETE FROM archive_request_attempts WHERE requested_at <= ?",
                (now - self.window_seconds * 6,),
            )
            status = self._status_locked(connection, now)
            if not status["available"]:
                reason = str(status.get("blocked_reason") or "").strip()
                if reason:
                    message = reason
                elif int(status["remaining"]) <= 0:
                    message = (
                        f"This installation has used its {self.limit} automated "
                        "archive requests in the rolling 24-hour window."
                    )
                else:
                    message = "Archive requests are temporarily paused for this installation."
                if status.get("next_request_at"):
                    message += f" The next safe request time is {status['next_request_at']}."
                raise ArchiveRequestBudgetExceeded(message)
            cursor = connection.execute(
                """
                INSERT INTO archive_request_attempts(
                    requested_at, feed_id, archive_date, archive_id, outcome
                ) VALUES (?, ?, ?, ?, 'started')
                """,
                (now, str(feed_id), str(archive_date), str(archive_id)),
            )
            connection.commit()
            return int(cursor.lastrowid)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def finish(
        self,
        request_id: int,
        *,
        outcome: str,
        http_status: int | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE archive_request_attempts
                SET outcome = ?, http_status = ?
                WHERE id = ?
                """,
                (str(outcome)[:80], http_status, int(request_id)),
            )

    def mark_rate_limited(self, reason: str) -> dict[str, Any]:
        now = float(self.clock())
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            cutoff = now - self.window_seconds
            oldest = connection.execute(
                """
                SELECT MIN(requested_at) AS oldest
                FROM archive_request_attempts
                WHERE requested_at > ?
                """,
                (cutoff,),
            ).fetchone()
            oldest_value = (
                float(oldest["oldest"])
                if oldest is not None and oldest["oldest"] is not None
                else now
            )
            blocked_until = min(
                now + self.window_seconds,
                oldest_value
                + self.window_seconds
                + RATE_LIMIT_RELEASE_GRACE_SECONDS,
            )
            current = connection.execute(
                "SELECT blocked_until FROM archive_quota_state WHERE singleton = 1"
            ).fetchone()
            connection.execute(
                """
                UPDATE archive_quota_state
                SET blocked_until = ?, blocked_reason = ?
                WHERE singleton = 1
                """,
                (blocked_until, str(reason)[:800]),
            )
            status = self._status_locked(connection, now)
            connection.commit()
            return status
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
