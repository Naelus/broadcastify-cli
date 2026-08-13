from __future__ import annotations

import os
import re
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
DEFAULT_ACCOUNT_PROFILE_ID = "default"
_ACCOUNT_PROFILE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class ArchiveRequestBudgetExceeded(RuntimeError):
    pass


def archive_quota_path(base_dir: str | Path | None = None) -> Path:
    configured = str(os.getenv("BROADCASTIFY_QUOTA_LEDGER") or "").strip()
    root = Path(base_dir or Path.cwd()).expanduser().resolve()
    if configured:
        path = Path(configured).expanduser()
        return path.resolve() if path.is_absolute() else (root / path).resolve()
    return root / DEFAULT_ARCHIVE_QUOTA_FILENAME


def normalize_account_profile_id(value: str | None = None) -> str:
    """Return the durable, non-secret identifier for one authorized account."""

    profile_id = str(
        value
        if value is not None
        else os.getenv("BROADCASTIFY_ACCOUNT_PROFILE")
        or DEFAULT_ACCOUNT_PROFILE_ID
    ).strip().lower()
    if not _ACCOUNT_PROFILE_PATTERN.fullmatch(profile_id):
        raise ValueError(
            "Account profile IDs must start with a letter or number and contain "
            "only letters, numbers, underscores, or hyphens."
        )
    return profile_id


class ArchiveRequestLedger:
    """Process-safe rolling archive request ledger scoped to one account."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        base_dir: str | Path | None = None,
        limit: int = AUTOMATED_ARCHIVE_REQUEST_LIMIT,
        provider_limit: int = PROVIDER_ARCHIVE_REQUEST_LIMIT,
        window_seconds: float = ARCHIVE_REQUEST_WINDOW_SECONDS,
        clock: Callable[[], float] = time.time,
        account_profile_id: str | None = None,
        request_spacing_seconds: float | None = None,
        sleeper: Callable[[float], None] = time.sleep,
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
        self.account_profile_id = normalize_account_profile_id(account_profile_id)
        configured_spacing = request_spacing_seconds
        if configured_spacing is None:
            try:
                configured_spacing = float(
                    os.getenv("BROADCASTIFY_GLOBAL_REQUEST_SPACING_SECONDS") or 0
                )
            except ValueError:
                configured_spacing = 0.0
        self.request_spacing_seconds = max(0.0, float(configured_spacing))
        self.sleeper = sleeper
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
                    account_profile_id TEXT NOT NULL DEFAULT 'default',
                    feed_id TEXT NOT NULL,
                    archive_date TEXT NOT NULL,
                    archive_id TEXT NOT NULL,
                    outcome TEXT NOT NULL DEFAULT 'started',
                    http_status INTEGER
                );
                CREATE INDEX IF NOT EXISTS idx_archive_request_attempts_time
                ON archive_request_attempts(requested_at);
                CREATE TABLE IF NOT EXISTS archive_quota_profile_state (
                    account_profile_id TEXT PRIMARY KEY,
                    instance_id TEXT NOT NULL,
                    blocked_until REAL NOT NULL DEFAULT 0,
                    blocked_reason TEXT NOT NULL DEFAULT ''
                );
                """
            )
            attempt_columns = {
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_info(archive_request_attempts)"
                ).fetchall()
            }
            if "account_profile_id" not in attempt_columns:
                try:
                    connection.execute(
                        "ALTER TABLE archive_request_attempts ADD COLUMN "
                        "account_profile_id TEXT NOT NULL DEFAULT 'default'"
                    )
                except sqlite3.OperationalError as exc:
                    if "duplicate column name" not in str(exc).lower():
                        raise
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_archive_request_attempts_profile_time "
                "ON archive_request_attempts(account_profile_id, requested_at)"
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO archive_quota_state(
                    singleton, instance_id, blocked_until, blocked_reason
                ) VALUES (1, ?, 0, '')
                """,
                (uuid.uuid4().hex,),
            )
            if self.account_profile_id != DEFAULT_ACCOUNT_PROFILE_ID:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO archive_quota_profile_state(
                        account_profile_id, instance_id, blocked_until, blocked_reason
                    ) VALUES (?, ?, 0, '')
                    """,
                    (self.account_profile_id, uuid.uuid4().hex),
                )

    def _state_locked(self, connection: sqlite3.Connection) -> sqlite3.Row | None:
        if self.account_profile_id == DEFAULT_ACCOUNT_PROFILE_ID:
            return connection.execute(
                """
                SELECT instance_id, blocked_until, blocked_reason
                FROM archive_quota_state WHERE singleton = 1
                """
            ).fetchone()
        return connection.execute(
            """
            SELECT instance_id, blocked_until, blocked_reason
            FROM archive_quota_profile_state WHERE account_profile_id = ?
            """,
            (self.account_profile_id,),
        ).fetchone()

    def _update_state_locked(
        self,
        connection: sqlite3.Connection,
        *,
        blocked_until: float,
        blocked_reason: str,
    ) -> None:
        if self.account_profile_id == DEFAULT_ACCOUNT_PROFILE_ID:
            connection.execute(
                """
                UPDATE archive_quota_state
                SET blocked_until = ?, blocked_reason = ?
                WHERE singleton = 1
                """,
                (blocked_until, blocked_reason),
            )
            return
        connection.execute(
            """
            UPDATE archive_quota_profile_state
            SET blocked_until = ?, blocked_reason = ?
            WHERE account_profile_id = ?
            """,
            (blocked_until, blocked_reason, self.account_profile_id),
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
            WHERE account_profile_id = ? AND requested_at > ?
            """,
            (self.account_profile_id, cutoff),
        ).fetchone()
        state = self._state_locked(connection)
        used = int(row["used"] if row is not None else 0)
        oldest = (
            float(row["oldest"])
            if row is not None and row["oldest"] is not None
            else None
        )
        blocked_until_value = float(state["blocked_until"] if state else 0.0)
        blocked_reason = str(state["blocked_reason"] or "") if state else ""
        if blocked_until_value > now:
            blocked_until_value, blocked_reason = (
                self._migrate_legacy_rate_limit_block_locked(
                    connection,
                    now,
                    blocked_until_value,
                    blocked_reason,
                )
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
            "account_profile_id": self.account_profile_id,
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

    def _migrate_legacy_rate_limit_block_locked(
        self,
        connection: sqlite3.Connection,
        now: float,
        blocked_until_value: float,
        blocked_reason: str,
    ) -> tuple[float, str]:
        latest_429 = connection.execute(
            """
            SELECT requested_at
            FROM archive_request_attempts
            WHERE account_profile_id = ? AND http_status = 429
            ORDER BY requested_at DESC
            LIMIT 1
            """,
            (self.account_profile_id,),
        ).fetchone()
        if latest_429 is None:
            return blocked_until_value, blocked_reason
        limited_at = float(latest_429["requested_at"])
        legacy_full_window = limited_at + self.window_seconds
        if blocked_until_value < legacy_full_window - RATE_LIMIT_RELEASE_GRACE_SECONDS:
            return blocked_until_value, blocked_reason

        oldest_at_limit = connection.execute(
            """
            SELECT MIN(requested_at) AS oldest
            FROM archive_request_attempts
            WHERE account_profile_id = ?
              AND requested_at > ? AND requested_at <= ?
            """,
            (
                self.account_profile_id,
                limited_at - self.window_seconds,
                limited_at,
            ),
        ).fetchone()
        if oldest_at_limit is None or oldest_at_limit["oldest"] is None:
            return blocked_until_value, blocked_reason
        migrated_until = min(
            blocked_until_value,
            float(oldest_at_limit["oldest"])
            + self.window_seconds
            + RATE_LIMIT_RELEASE_GRACE_SECONDS,
        )
        if migrated_until >= blocked_until_value:
            return blocked_until_value, blocked_reason
        migrated_reason = (
            "Broadcastify returned HTTP 429; this installation is waiting for "
            "its next known rolling-window release."
        )
        self._update_state_locked(
            connection,
            blocked_until=migrated_until,
            blocked_reason=migrated_reason,
        )
        return migrated_until, migrated_reason

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
        while True:
            now = float(self.clock())
            connection = self._connect()
            spacing_delay = 0.0
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
                            f"Account profile {self.account_profile_id} has used its "
                            f"{self.limit} automated archive requests in the rolling "
                            "24-hour window."
                        )
                    else:
                        message = (
                            f"Archive requests are temporarily paused for account "
                            f"profile {self.account_profile_id}."
                        )
                    if status.get("next_request_at"):
                        message += (
                            f" The next safe request time is "
                            f"{status['next_request_at']}."
                        )
                    raise ArchiveRequestBudgetExceeded(message)
                if self.request_spacing_seconds > 0:
                    latest = connection.execute(
                        "SELECT MAX(requested_at) AS latest "
                        "FROM archive_request_attempts"
                    ).fetchone()
                    latest_at = (
                        float(latest["latest"])
                        if latest is not None and latest["latest"] is not None
                        else None
                    )
                    if latest_at is not None:
                        spacing_delay = max(
                            0.0,
                            min(
                                self.request_spacing_seconds,
                                latest_at + self.request_spacing_seconds - now,
                            ),
                        )
                if spacing_delay <= 0:
                    cursor = connection.execute(
                        """
                        INSERT INTO archive_request_attempts(
                            requested_at, account_profile_id, feed_id, archive_date,
                            archive_id, outcome
                        ) VALUES (?, ?, ?, ?, ?, 'started')
                        """,
                        (
                            now,
                            self.account_profile_id,
                            str(feed_id),
                            str(archive_date),
                            str(archive_id),
                        ),
                    )
                    connection.commit()
                    return int(cursor.lastrowid)
                connection.rollback()
            except Exception:
                if connection.in_transaction:
                    connection.rollback()
                raise
            finally:
                connection.close()
            self.sleeper(spacing_delay)

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
                WHERE id = ? AND account_profile_id = ?
                """,
                (
                    str(outcome)[:80],
                    http_status,
                    int(request_id),
                    self.account_profile_id,
                ),
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
                WHERE account_profile_id = ? AND requested_at > ?
                """,
                (self.account_profile_id, cutoff),
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
            self._update_state_locked(
                connection,
                blocked_until=blocked_until,
                blocked_reason=str(reason)[:800],
            )
            status = self._status_locked(connection, now)
            connection.commit()
            return status
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
