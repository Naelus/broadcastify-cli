from __future__ import annotations

import json
import secrets
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PIPELINE_SYNC_DATABASE = ".broadcastify-pipeline-sync.sqlite3"
PIPELINE_SYNC_SCHEMA_VERSION = 1
PIPELINE_EVENT_KINDS = {"source", "result"}
PIPELINE_ROLES = {"master", "follower"}


def normalize_pipeline_role(value: object = None) -> str:
    normalized = str(value or "master").strip().lower()
    if normalized not in PIPELINE_ROLES:
        raise ValueError("LAN pipeline role must be master or follower.")
    return normalized


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _feed_id(value: object) -> str:
    normalized = str(value or "").strip()
    if not normalized.isdigit():
        raise ValueError("A numeric feed ID is required.")
    return normalized


def _date_value(value: object) -> date:
    parsed = date.fromisoformat(str(value or "").strip())
    if parsed > date.today():
        raise ValueError("Pipeline dates cannot be in the future.")
    return parsed


class PipelineSyncStore:
    """Small durable control plane for master/follower delta synchronization."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / PIPELINE_SYNC_DATABASE
        self.connection = sqlite3.connect(self.path, timeout=30.0)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA busy_timeout=30000")
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL,
                feed_id TEXT NOT NULL,
                archive_date TEXT NOT NULL,
                processing_fingerprint TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                UNIQUE(kind, feed_id, archive_date, processing_fingerprint)
            );
            CREATE INDEX IF NOT EXISTS events_sequence_idx
                ON events(sequence);
            CREATE TABLE IF NOT EXISTS cursors (
                peer_url TEXT NOT NULL,
                peer_node_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(peer_url, peer_node_id)
            );
            CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feed_id TEXT NOT NULL,
                feed_name TEXT NOT NULL DEFAULT '',
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                requester_node_id TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(feed_id, start_date, end_date, requester_node_id)
            );
            """
        )
        self.connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES('schema_version', ?)",
            (str(PIPELINE_SYNC_SCHEMA_VERSION),),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def node_id(self) -> str:
        row = self.connection.execute(
            "SELECT value FROM metadata WHERE key='node_id'"
        ).fetchone()
        if row is not None and str(row["value"]):
            return str(row["value"])
        value = secrets.token_hex(12)
        self.connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES('node_id', ?)",
            (value,),
        )
        self.connection.commit()
        return value

    def __enter__(self) -> PipelineSyncStore:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def record_source(self, feed_id: object, archive_date: object) -> int:
        # A rolling day can gain additional retained blocks while keeping the
        # same feed/date identity. Republish it so the master observes the
        # newer completion proof instead of leaving its cursor on an older
        # source snapshot.
        return self._record(
            "source",
            feed_id,
            archive_date,
            "",
            replace_existing=True,
        )

    def record_result(
        self,
        feed_id: object,
        archive_date: object,
        processing_fingerprint: object,
    ) -> int:
        fingerprint = str(processing_fingerprint or "").strip().lower()
        if len(fingerprint) != 64 or any(value not in "0123456789abcdef" for value in fingerprint):
            raise ValueError("A valid processing fingerprint is required.")
        return self._record(
            "result",
            feed_id,
            archive_date,
            fingerprint,
            replace_existing=True,
        )

    def _record(
        self,
        kind: str,
        feed_id: object,
        archive_date: object,
        fingerprint: str,
        *,
        replace_existing: bool = False,
    ) -> int:
        if kind not in PIPELINE_EVENT_KINDS:
            raise ValueError("Unsupported pipeline event kind.")
        normalized_feed = _feed_id(feed_id)
        normalized_date = _date_value(archive_date).isoformat()
        now = _utc_now()
        if replace_existing:
            # A rolling day can produce newer artifacts without changing the
            # model fingerprint. Replace its prior journal entry so every peer
            # cursor observes the updated authoritative result.
            self.connection.execute(
                """
                DELETE FROM events
                WHERE kind=? AND feed_id=? AND archive_date=?
                  AND processing_fingerprint=?
                """,
                (kind, normalized_feed, normalized_date, fingerprint),
            )
        self.connection.execute(
            """
            INSERT INTO events(
                kind, feed_id, archive_date, processing_fingerprint, created_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(kind, feed_id, archive_date, processing_fingerprint)
            DO NOTHING
            """,
            (kind, normalized_feed, normalized_date, fingerprint, now),
        )
        row = self.connection.execute(
            """
            SELECT sequence FROM events
            WHERE kind=? AND feed_id=? AND archive_date=?
              AND processing_fingerprint=?
            """,
            (kind, normalized_feed, normalized_date, fingerprint),
        ).fetchone()
        self.connection.commit()
        assert row is not None
        return int(row["sequence"])

    def seed_sources(self) -> int:
        """One-time cheap migration from completion markers; never hashes media."""

        seeded = self.connection.execute(
            "SELECT value FROM metadata WHERE key='sources_seeded'"
        ).fetchone()
        if seeded is not None:
            return 0
        discovered: list[tuple[str, date]] = []
        for feed_dir in self._safe_directories(self.output_dir):
            if not feed_dir.name.isdigit():
                continue
            for day_dir in self._safe_directories(feed_dir):
                if len(day_dir.name) != 8 or not day_dir.name.isdigit():
                    continue
                marker = day_dir / ".broadcastify-archive-complete.json"
                try:
                    payload = json.loads(marker.read_text(encoding="utf-8"))
                    archive_date = date.fromisoformat(
                        str(payload.get("archive_date") or "")
                    )
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    continue
                if (
                    str(payload.get("feed_id") or "") == feed_dir.name
                    and archive_date.strftime("%Y%m%d") == day_dir.name
                ):
                    discovered.append((feed_dir.name, archive_date))
        for feed_id, archive_date in discovered:
            self.record_source(feed_id, archive_date)
        self.connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES('sources_seeded', ?)",
            (_utc_now(),),
        )
        self.connection.commit()
        return len(discovered)

    @staticmethod
    def _safe_directories(root: Path) -> Iterable[Path]:
        try:
            candidates = list(root.iterdir())
        except OSError:
            return ()
        values: list[Path] = []
        for candidate in candidates:
            try:
                if (
                    candidate.is_dir()
                    and not candidate.is_symlink()
                    and candidate.resolve().parent == root
                ):
                    values.append(candidate)
            except OSError:
                continue
        return values

    def changes(self, after: int, *, limit: int = 128) -> dict[str, Any]:
        cursor = max(0, int(after))
        bounded = min(512, max(1, int(limit)))
        rows = self.connection.execute(
            """
            SELECT sequence, kind, feed_id, archive_date,
                   processing_fingerprint, created_at
            FROM events WHERE sequence>? ORDER BY sequence LIMIT ?
            """,
            (cursor, bounded + 1),
        ).fetchall()
        has_more = len(rows) > bounded
        selected = rows[:bounded]
        return {
            "events": [dict(row) for row in selected],
            "cursor": int(selected[-1]["sequence"]) if selected else cursor,
            "has_more": has_more,
        }

    def list_source_spans(self) -> list[dict[str, Any]]:
        """Return compact source coverage from the durable delta journal."""

        rows = self.connection.execute(
            """
            SELECT feed_id,
                   MIN(archive_date) AS start_date,
                   MAX(archive_date) AS end_date,
                   COUNT(*) AS day_count
            FROM events
            WHERE kind='source'
            GROUP BY feed_id
            ORDER BY feed_id
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def cursor(self, peer_url: str, peer_node_id: str) -> int:
        row = self.connection.execute(
            "SELECT sequence FROM cursors WHERE peer_url=? AND peer_node_id=?",
            (str(peer_url), str(peer_node_id)),
        ).fetchone()
        return int(row["sequence"]) if row is not None else 0

    def save_cursor(self, peer_url: str, peer_node_id: str, sequence: int) -> None:
        self.connection.execute(
            """
            INSERT INTO cursors(peer_url, peer_node_id, sequence, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(peer_url, peer_node_id) DO UPDATE SET
                sequence=MAX(cursors.sequence, excluded.sequence),
                updated_at=excluded.updated_at
            """,
            (str(peer_url), str(peer_node_id), max(0, int(sequence)), _utc_now()),
        )
        self.connection.commit()

    def request(
        self,
        feed_id: object,
        start_date: object,
        end_date: object,
        *,
        requester_node_id: str,
        feed_name: str = "",
    ) -> dict[str, Any]:
        normalized_feed = _feed_id(feed_id)
        start = _date_value(start_date)
        end = _date_value(end_date)
        if start > end or (end - start).days > 3_650:
            raise ValueError("The requested pipeline range is not valid.")
        requester = str(requester_node_id or "").strip()
        if not requester or len(requester) > 80:
            raise ValueError("A bounded requester node ID is required.")
        now = _utc_now()
        self.connection.execute(
            """
            INSERT INTO requests(
                feed_id, feed_name, start_date, end_date,
                requester_node_id, state, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)
            ON CONFLICT(feed_id, start_date, end_date, requester_node_id)
            DO UPDATE SET state='pending', updated_at=excluded.updated_at
            """,
            (
                normalized_feed,
                " ".join(str(feed_name or "").split())[:200],
                start.isoformat(),
                end.isoformat(),
                requester,
                now,
                now,
            ),
        )
        self.connection.commit()
        row = self.connection.execute(
            """
            SELECT * FROM requests WHERE feed_id=? AND start_date=?
              AND end_date=? AND requester_node_id=?
            """,
            (normalized_feed, start.isoformat(), end.isoformat(), requester),
        ).fetchone()
        assert row is not None
        return dict(row)

    def pending_requests(self, *, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM requests WHERE state='pending' ORDER BY id LIMIT ?",
            (min(500, max(1, int(limit))),),
        ).fetchall()
        return [dict(row) for row in rows]

    def mark_scheduled(self, request_id: int) -> None:
        self.connection.execute(
            "UPDATE requests SET state='scheduled', updated_at=? WHERE id=?",
            (_utc_now(), int(request_id)),
        )
        self.connection.commit()
