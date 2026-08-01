from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, time as datetime_time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ImportedDay:
    day_id: int
    feed_id: str
    archive_date: date
    segment_count: int
    transcript_sha256: str


class AnalysisStore:
    """Durable local evidence store for transcripts and derived analysis."""

    def __init__(self, path: str | Path = "archives/broadcastify-analysis.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = NORMAL")
        self._initialize()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "AnalysisStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self.connection
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def _initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_info (
                version INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS feed_days (
                id INTEGER PRIMARY KEY,
                feed_id TEXT NOT NULL,
                archive_date TEXT NOT NULL,
                audio_path TEXT,
                transcript_path TEXT NOT NULL,
                manifest_path TEXT,
                audio_sha256 TEXT,
                transcript_sha256 TEXT NOT NULL,
                duration_seconds REAL,
                transcription_model TEXT,
                diarization_model TEXT,
                has_diarization INTEGER NOT NULL DEFAULT 0,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(feed_id, archive_date)
            );

            CREATE TABLE IF NOT EXISTS transcript_segments (
                id INTEGER PRIMARY KEY,
                day_id INTEGER NOT NULL REFERENCES feed_days(id) ON DELETE CASCADE,
                segment_index INTEGER NOT NULL,
                start_seconds REAL NOT NULL,
                end_seconds REAL NOT NULL,
                speaker TEXT,
                text TEXT NOT NULL,
                UNIQUE(day_id, segment_index)
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS transcript_fts USING fts5(
                text,
                content='transcript_segments',
                content_rowid='id',
                tokenize='porter unicode61'
            );

            CREATE TRIGGER IF NOT EXISTS transcript_segments_ai AFTER INSERT ON transcript_segments BEGIN
                INSERT INTO transcript_fts(rowid, text) VALUES (new.id, new.text);
            END;
            CREATE TRIGGER IF NOT EXISTS transcript_segments_ad AFTER DELETE ON transcript_segments BEGIN
                INSERT INTO transcript_fts(transcript_fts, rowid, text)
                VALUES ('delete', old.id, old.text);
            END;
            CREATE TRIGGER IF NOT EXISTS transcript_segments_au AFTER UPDATE ON transcript_segments BEGIN
                INSERT INTO transcript_fts(transcript_fts, rowid, text)
                VALUES ('delete', old.id, old.text);
                INSERT INTO transcript_fts(rowid, text) VALUES (new.id, new.text);
            END;

            CREATE TABLE IF NOT EXISTS passages (
                id INTEGER PRIMARY KEY,
                day_id INTEGER NOT NULL REFERENCES feed_days(id) ON DELETE CASCADE,
                passage_index INTEGER NOT NULL,
                start_seconds REAL NOT NULL,
                end_seconds REAL NOT NULL,
                text TEXT NOT NULL,
                segment_ids_json TEXT NOT NULL,
                UNIQUE(day_id, passage_index)
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS passage_fts USING fts5(
                text,
                content='passages',
                content_rowid='id',
                tokenize='porter unicode61'
            );

            CREATE TRIGGER IF NOT EXISTS passages_ai AFTER INSERT ON passages BEGIN
                INSERT INTO passage_fts(rowid, text) VALUES (new.id, new.text);
            END;
            CREATE TRIGGER IF NOT EXISTS passages_ad AFTER DELETE ON passages BEGIN
                INSERT INTO passage_fts(passage_fts, rowid, text)
                VALUES ('delete', old.id, old.text);
            END;
            CREATE TRIGGER IF NOT EXISTS passages_au AFTER UPDATE ON passages BEGIN
                INSERT INTO passage_fts(passage_fts, rowid, text)
                VALUES ('delete', old.id, old.text);
                INSERT INTO passage_fts(rowid, text) VALUES (new.id, new.text);
            END;

            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY,
                day_id INTEGER NOT NULL REFERENCES feed_days(id) ON DELETE CASCADE,
                fingerprint TEXT NOT NULL,
                event_type TEXT NOT NULL,
                title TEXT NOT NULL,
                summary TEXT NOT NULL,
                location_text TEXT,
                start_seconds REAL NOT NULL,
                end_seconds REAL NOT NULL,
                priority INTEGER NOT NULL,
                confidence REAL NOT NULL,
                evidence_json TEXT NOT NULL,
                attributes_json TEXT NOT NULL DEFAULT '{}',
                model TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(day_id, fingerprint, model, prompt_version)
            );

            CREATE TABLE IF NOT EXISTS analysis_window_checkpoints (
                id INTEGER PRIMARY KEY,
                day_id INTEGER NOT NULL REFERENCES feed_days(id) ON DELETE CASCADE,
                model TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                transcript_sha256 TEXT NOT NULL,
                window_index INTEGER NOT NULL,
                window_fingerprint TEXT NOT NULL,
                incidents_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(
                    day_id,
                    model,
                    prompt_version,
                    transcript_sha256,
                    window_index
                )
            );

            CREATE TABLE IF NOT EXISTS daily_summaries (
                day_id INTEGER PRIMARY KEY REFERENCES feed_days(id) ON DELETE CASCADE,
                summary TEXT NOT NULL,
                notable_incident_ids_json TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                transcript_sha256 TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS weekly_summaries (
                id INTEGER PRIMARY KEY,
                feed_id TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                summary TEXT NOT NULL,
                notable_incident_ids_json TEXT NOT NULL,
                days_available INTEGER NOT NULL,
                incident_count INTEGER NOT NULL,
                model TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                source_fingerprint TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(feed_id, start_date, end_date)
            );

            CREATE TABLE IF NOT EXISTS feed_catalog (
                feed_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                location TEXT NOT NULL DEFAULT '',
                description TEXT NOT NULL DEFAULT '',
                genre TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS feed_schedules (
                id INTEGER PRIMARY KEY,
                feed_id TEXT NOT NULL UNIQUE,
                feed_name TEXT NOT NULL,
                run_time_local TEXT NOT NULL,
                lookback_days INTEGER NOT NULL DEFAULT 2,
                job_json TEXT NOT NULL,
                analyze INTEGER NOT NULL DEFAULT 1,
                enabled INTEGER NOT NULL DEFAULT 1,
                state TEXT NOT NULL DEFAULT 'scheduled',
                message TEXT NOT NULL DEFAULT '',
                last_run_date TEXT NOT NULL DEFAULT '',
                last_started_at TEXT NOT NULL DEFAULT '',
                last_finished_at TEXT NOT NULL DEFAULT '',
                not_before TEXT NOT NULL DEFAULT '',
                lease_until TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS area_profiles (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                zip_codes_json TEXT NOT NULL,
                feeds_json TEXT NOT NULL,
                coverage_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS area_acquisition_runs (
                id INTEGER PRIMARY KEY,
                profile_id INTEGER NOT NULL REFERENCES area_profiles(id) ON DELETE CASCADE,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                processing_fingerprint TEXT NOT NULL,
                processing_json TEXT NOT NULL,
                status TEXT NOT NULL,
                stop_reason TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT,
                UNIQUE(profile_id, start_date, end_date, processing_fingerprint)
            );

            CREATE TABLE IF NOT EXISTS area_acquisition_items (
                id INTEGER PRIMARY KEY,
                run_id INTEGER NOT NULL REFERENCES area_acquisition_runs(id) ON DELETE CASCADE,
                feed_id TEXT NOT NULL,
                feed_name TEXT NOT NULL,
                priority_rank INTEGER NOT NULL,
                distance_miles REAL,
                status TEXT NOT NULL,
                requested_days INTEGER NOT NULL DEFAULT 0,
                completed_days INTEGER NOT NULL DEFAULT 0,
                missing_days_json TEXT NOT NULL DEFAULT '[]',
                download_limited INTEGER NOT NULL DEFAULT 0,
                result_json TEXT NOT NULL DEFAULT '{}',
                message TEXT NOT NULL DEFAULT '',
                attempt_count INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                UNIQUE(run_id, feed_id)
            );

            CREATE TABLE IF NOT EXISTS area_story_digests (
                id INTEGER PRIMARY KEY,
                profile_id INTEGER NOT NULL REFERENCES area_profiles(id) ON DELETE CASCADE,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                summary TEXT NOT NULL,
                stories_json TEXT NOT NULL,
                coverage_json TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                source_fingerprint TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(profile_id, start_date, end_date)
            );

            CREATE TABLE IF NOT EXISTS embeddings (
                entity_type TEXT NOT NULL,
                entity_id INTEGER NOT NULL,
                model TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                text_sha256 TEXT NOT NULL,
                vector BLOB NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(entity_type, entity_id, model)
            );

            CREATE TABLE IF NOT EXISTS qa_history (
                id INTEGER PRIMARY KEY,
                feed_id TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                model TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        area_profile_columns = {
            str(row["name"])
            for row in self.connection.execute("PRAGMA table_info(area_profiles)").fetchall()
        }
        if "coverage_json" not in area_profile_columns:
            self.connection.execute(
                "ALTER TABLE area_profiles ADD COLUMN coverage_json TEXT NOT NULL DEFAULT '{}'"
            )
        row = self.connection.execute("SELECT version FROM schema_info LIMIT 1").fetchone()
        if row is None:
            self.connection.execute(
                "INSERT INTO schema_info(version) VALUES (?)", (SCHEMA_VERSION,)
            )
        elif int(row["version"]) != SCHEMA_VERSION:
            raise RuntimeError(
                f"Unsupported analysis database version {row['version']}; expected {SCHEMA_VERSION}."
            )
        self.connection.commit()

    @staticmethod
    def _schedule_time(value: str) -> str:
        text = str(value or "").strip()
        try:
            parsed = datetime_time.fromisoformat(text)
        except ValueError as exc:
            raise ValueError("Schedule time must use HH:MM local time.") from exc
        return f"{parsed.hour:02d}:{parsed.minute:02d}"

    @staticmethod
    def _aware_local(value: datetime | None = None) -> datetime:
        current = value or datetime.now().astimezone()
        return current.astimezone() if current.tzinfo is None else current

    @staticmethod
    def _utc_value(value: str) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _feed_schedule(
        self,
        row: sqlite3.Row,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        current = self._aware_local(now)
        current_utc = current.astimezone(timezone.utc)
        hour, minute = (int(value) for value in str(row["run_time_local"]).split(":"))
        scheduled_today = datetime.combine(
            current.date(),
            datetime_time(hour=hour, minute=minute),
            tzinfo=current.tzinfo,
        )
        not_before = self._utc_value(str(row["not_before"] or ""))
        lease_until = self._utc_value(str(row["lease_until"] or ""))
        running = str(row["state"]) == "running" and bool(
            lease_until and lease_until > current_utc
        )
        due = bool(row["enabled"]) and (
            current >= scheduled_today
            and str(row["last_run_date"] or "") != current.date().isoformat()
            and not running
            and (not_before is None or not_before <= current_utc)
        )
        if due:
            next_run = current
        elif not_before is not None and not_before > current_utc:
            next_run = max(scheduled_today, not_before.astimezone(current.tzinfo))
        elif running and lease_until is not None:
            next_run = lease_until.astimezone(current.tzinfo)
        elif current < scheduled_today and str(row["last_run_date"] or "") != current.date().isoformat():
            next_run = scheduled_today
        else:
            next_run = scheduled_today + timedelta(days=1)
        return {
            "id": int(row["id"]),
            "feed_id": str(row["feed_id"]),
            "feed_name": str(row["feed_name"]),
            "run_time_local": str(row["run_time_local"]),
            "lookback_days": int(row["lookback_days"]),
            "job": json.loads(str(row["job_json"])),
            "analyze": bool(row["analyze"]),
            "enabled": bool(row["enabled"]),
            "state": str(row["state"]),
            "message": str(row["message"]),
            "last_run_date": str(row["last_run_date"]),
            "last_started_at": str(row["last_started_at"]),
            "last_finished_at": str(row["last_finished_at"]),
            "not_before": str(row["not_before"]),
            "next_run_at": next_run.isoformat(timespec="seconds"),
            "due": due,
        }

    def save_feed_schedule(self, payload: dict[str, Any]) -> dict[str, Any]:
        feed_id = str(payload.get("feed_id") or "").strip()
        if not feed_id.isdigit():
            raise ValueError("A numeric feed ID is required for a schedule.")
        feed_name = str(payload.get("feed_name") or f"Feed {feed_id}").strip()[:200]
        run_time = self._schedule_time(str(payload.get("run_time_local") or "02:00"))
        lookback_days = max(1, min(14, int(payload.get("lookback_days") or 2)))
        job = dict(payload.get("job") or {})
        for key in (
            "feed_id",
            "feed_name",
            "start_date",
            "end_date",
            "huggingface_token",
            "analysis_api_key",
        ):
            job.pop(key, None)
        job["download_jobs"] = 1
        job["keep_originals"] = True
        now = utc_now()
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO feed_schedules(
                    feed_id, feed_name, run_time_local, lookback_days,
                    job_json, analyze, enabled, state, message,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'scheduled', '', ?, ?)
                ON CONFLICT(feed_id) DO UPDATE SET
                    feed_name=excluded.feed_name,
                    run_time_local=excluded.run_time_local,
                    lookback_days=excluded.lookback_days,
                    job_json=excluded.job_json,
                    analyze=excluded.analyze,
                    enabled=excluded.enabled,
                    state=CASE
                        WHEN feed_schedules.state='running' THEN feed_schedules.state
                        ELSE 'scheduled'
                    END,
                    message='',
                    not_before='',
                    updated_at=excluded.updated_at
                """,
                (
                    feed_id,
                    feed_name,
                    run_time,
                    lookback_days,
                    json.dumps(job, sort_keys=True),
                    int(bool(payload.get("analyze", True))),
                    int(bool(payload.get("enabled", True))),
                    now,
                    now,
                ),
            )
        row = self.connection.execute(
            "SELECT * FROM feed_schedules WHERE feed_id=?", (feed_id,)
        ).fetchone()
        assert row is not None
        return self._feed_schedule(row)

    def list_feed_schedules(
        self,
        *,
        now: datetime | None = None,
    ) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM feed_schedules ORDER BY enabled DESC, run_time_local, feed_name"
        ).fetchall()
        return [self._feed_schedule(row, now=now) for row in rows]

    def delete_feed_schedule(self, schedule_id: int) -> bool:
        with self.transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM feed_schedules WHERE id=?", (int(schedule_id),)
            )
        return cursor.rowcount > 0

    def recover_feed_schedules(self, *, now: datetime | None = None) -> int:
        current = self._aware_local(now).astimezone(timezone.utc)
        timestamp = current.isoformat(timespec="seconds")
        not_before = (current + timedelta(minutes=1)).isoformat(timespec="seconds")
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE feed_schedules
                SET state='deferred',
                    message='The previous app session ended during this scheduled run; resuming from retained work.',
                    not_before=?, lease_until='', updated_at=?
                WHERE state='running'
                """,
                (not_before, timestamp),
            )
        return int(cursor.rowcount)

    def claim_due_feed_schedule(
        self,
        *,
        now: datetime | None = None,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any] | None:
        current = self._aware_local(now)
        current_utc = current.astimezone(timezone.utc)
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            rows = self.connection.execute(
                "SELECT * FROM feed_schedules ORDER BY run_time_local, id"
            ).fetchall()
            selected = next(
                (
                    (row, self._feed_schedule(row, now=current))
                    for row in rows
                    if self._feed_schedule(row, now=current)["due"]
                ),
                None,
            )
            if selected is None:
                self.connection.commit()
                return None
            row, schedule = selected
            started = current_utc.isoformat(timespec="seconds")
            lease_until = (current_utc + timedelta(hours=24)).isoformat(timespec="seconds")
            self.connection.execute(
                """
                UPDATE feed_schedules
                SET state='running', message='', last_started_at=?,
                    lease_until=?, updated_at=?
                WHERE id=?
                """,
                (started, lease_until, started, int(row["id"])),
            )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        due_date = current.date()
        lookback = int(schedule["lookback_days"])
        job = {
            **dict(schedule["job"]),
            "feed_id": schedule["feed_id"],
            "feed_name": schedule["feed_name"],
            "start_date": (due_date - timedelta(days=lookback - 1)).isoformat(),
            "end_date": due_date.isoformat(),
            "download_jobs": 1,
            "keep_originals": True,
        }
        if output_dir is not None:
            # The process hosting the scheduler owns the active Library
            # selection. It must win over an older absolute path saved in the
            # schedule; otherwise changing Library settings can make the next
            # run fetch the same upstream archive IDs into a second root.
            job["output_dir"] = str(Path(output_dir).expanduser().resolve())
        else:
            configured_output = Path(str(job.get("output_dir") or "archives"))
            if not configured_output.is_absolute():
                # Older desktop schedules saved the literal relative value
                # ``archives``. Installed workers run from a private data
                # directory, so replaying that value can silently split one
                # library across two roots. The evidence database already
                # identifies the canonical library for this schedule; resolve
                # legacy relative values there.
                job["output_dir"] = str(self.path.resolve().parent)
        schedule["state"] = "running"
        schedule["due"] = False
        schedule["due_date"] = due_date.isoformat()
        schedule["job"] = job
        return schedule

    def finish_feed_schedule(
        self,
        schedule_id: int,
        *,
        due_date: str,
        status: str,
        message: str = "",
        next_request_at: str = "",
        now: datetime | None = None,
    ) -> dict[str, Any]:
        allowed = {"complete", "waiting_quota", "failed", "canceled", "deferred"}
        if status not in allowed:
            raise ValueError("Unsupported schedule completion status.")
        current = self._aware_local(now).astimezone(timezone.utc)
        finished = current.isoformat(timespec="seconds")
        last_run_date = due_date if status in {"complete", "failed", "canceled"} else ""
        not_before = ""
        if status == "waiting_quota":
            parsed = self._utc_value(next_request_at)
            not_before = (
                parsed.isoformat(timespec="seconds")
                if parsed is not None and parsed > current
                else (current + timedelta(minutes=5)).isoformat(timespec="seconds")
            )
        elif status == "deferred":
            not_before = (current + timedelta(minutes=5)).isoformat(timespec="seconds")
        with self.transaction() as connection:
            connection.execute(
                """
                UPDATE feed_schedules
                SET state=?, message=?,
                    last_run_date=CASE WHEN ?='' THEN last_run_date ELSE ? END,
                    last_finished_at=?, not_before=?, lease_until='', updated_at=?
                WHERE id=?
                """,
                (
                    status,
                    str(message)[:1000],
                    last_run_date,
                    last_run_date,
                    finished,
                    not_before,
                    finished,
                    int(schedule_id),
                ),
            )
        row = self.connection.execute(
            "SELECT * FROM feed_schedules WHERE id=?", (int(schedule_id),)
        ).fetchone()
        if row is None:
            raise ValueError("Feed schedule was not found.")
        return self._feed_schedule(row, now=now)

    def import_transcript(
        self,
        feed_id: str,
        archive_date: date,
        transcript_path: str | Path,
        audio_path: str | Path | None = None,
        manifest_path: str | Path | None = None,
    ) -> ImportedDay:
        transcript = Path(transcript_path).resolve()
        resolved_audio = Path(audio_path).resolve() if audio_path else None
        resolved_manifest = Path(manifest_path).resolve() if manifest_path else None
        transcript_hash = sha256_file(transcript)
        existing = self.connection.execute(
            """
            SELECT id, transcript_sha256 FROM feed_days
            WHERE feed_id=? AND archive_date=?
            """,
            (feed_id, archive_date.isoformat()),
        ).fetchone()
        if existing is not None and existing["transcript_sha256"] == transcript_hash:
            # A retained library can be moved between Windows, Linux, and a NAS
            # without changing its transcript bytes. Refresh the physical paths
            # even when the evidence import itself is already current.
            audio_value = (
                str(resolved_audio)
                if resolved_audio is not None and resolved_audio.is_file()
                else None
            )
            audio_hash = sha256_file(resolved_audio) if audio_value else None
            with self.transaction() as connection:
                connection.execute(
                    """
                    UPDATE feed_days SET
                        audio_path=COALESCE(?, audio_path),
                        transcript_path=?,
                        manifest_path=COALESCE(?, manifest_path),
                        audio_sha256=COALESCE(?, audio_sha256),
                        updated_at=?
                    WHERE id=?
                    """,
                    (
                        audio_value,
                        str(transcript),
                        str(resolved_manifest) if resolved_manifest else None,
                        audio_hash,
                        utc_now(),
                        int(existing["id"]),
                    ),
                )
            segment_count = int(
                self.connection.execute(
                    "SELECT COUNT(*) FROM transcript_segments WHERE day_id=?",
                    (int(existing["id"]),),
                ).fetchone()[0]
            )
            return ImportedDay(
                day_id=int(existing["id"]),
                feed_id=feed_id,
                archive_date=archive_date,
                segment_count=segment_count,
                transcript_sha256=transcript_hash,
            )

        payload = json.loads(transcript.read_text(encoding="utf-8"))
        segments = payload.get("segments")
        if not isinstance(segments, list):
            raise ValueError(f"Transcript has no segment list: {transcript}")

        audio_hash = (
            sha256_file(resolved_audio)
            if resolved_audio is not None and resolved_audio.exists()
            else None
        )
        now = utc_now()
        metadata = {
            key: payload.get(key)
            for key in ("language", "language_probability", "compute_type")
            if payload.get(key) is not None
        }
        has_diarization = any(segment.get("speaker") for segment in segments)
        diarization_model = payload.get("diarization_model") if has_diarization else None

        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO feed_days(
                    feed_id, archive_date, audio_path, transcript_path, manifest_path,
                    audio_sha256, transcript_sha256, duration_seconds,
                    transcription_model, diarization_model, has_diarization,
                    metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(feed_id, archive_date) DO UPDATE SET
                    audio_path=excluded.audio_path,
                    transcript_path=excluded.transcript_path,
                    manifest_path=excluded.manifest_path,
                    audio_sha256=excluded.audio_sha256,
                    transcript_sha256=excluded.transcript_sha256,
                    duration_seconds=excluded.duration_seconds,
                    transcription_model=excluded.transcription_model,
                    diarization_model=excluded.diarization_model,
                    has_diarization=excluded.has_diarization,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at
                """,
                (
                    feed_id,
                    archive_date.isoformat(),
                    str(resolved_audio) if resolved_audio else None,
                    str(transcript),
                    str(resolved_manifest) if resolved_manifest else None,
                    audio_hash,
                    transcript_hash,
                    payload.get("duration"),
                    payload.get("model"),
                    diarization_model,
                    int(has_diarization),
                    json.dumps(metadata, ensure_ascii=False),
                    now,
                    now,
                ),
            )
            day_id = int(
                connection.execute(
                    "SELECT id FROM feed_days WHERE feed_id=? AND archive_date=?",
                    (feed_id, archive_date.isoformat()),
                ).fetchone()["id"]
            )
            connection.execute("DELETE FROM incidents WHERE day_id=?", (day_id,))
            connection.execute("DELETE FROM daily_summaries WHERE day_id=?", (day_id,))
            connection.execute("DELETE FROM transcript_segments WHERE day_id=?", (day_id,))
            connection.execute("DELETE FROM passages WHERE day_id=?", (day_id,))
            connection.execute("DELETE FROM embeddings WHERE entity_type='passage' AND entity_id NOT IN (SELECT id FROM passages)")

            for index, segment in enumerate(segments):
                text = str(segment.get("text") or "").strip()
                if not text:
                    continue
                connection.execute(
                    """
                    INSERT INTO transcript_segments(
                        day_id, segment_index, start_seconds, end_seconds, speaker, text
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        day_id,
                        index,
                        float(segment.get("start", 0.0)),
                        float(segment.get("end", segment.get("start", 0.0))),
                        segment.get("speaker"),
                        text,
                    ),
                )

            self._create_passages(connection, day_id)

        return ImportedDay(
            day_id=day_id,
            feed_id=feed_id,
            archive_date=archive_date,
            segment_count=len(segments),
            transcript_sha256=transcript_hash,
        )

    @staticmethod
    def _create_passages(connection: sqlite3.Connection, day_id: int) -> None:
        rows = connection.execute(
            """
            SELECT id, segment_index, start_seconds, end_seconds, speaker, text
            FROM transcript_segments WHERE day_id=? ORDER BY start_seconds, segment_index
            """,
            (day_id,),
        ).fetchall()
        groups: list[list[sqlite3.Row]] = []
        for row in rows:
            if not groups:
                groups.append([row])
                continue
            group = groups[-1]
            span = float(row["end_seconds"]) - float(group[0]["start_seconds"])
            characters = sum(len(str(item["text"])) for item in group) + len(str(row["text"]))
            if span <= 300 and characters <= 1_600:
                group.append(row)
            else:
                groups.append([row])

        for passage_index, group in enumerate(groups):
            lines = []
            for row in group:
                speaker = f" {row['speaker']}:" if row["speaker"] else ""
                lines.append(
                    f"[{float(row['start_seconds']):.2f}-{float(row['end_seconds']):.2f}]"
                    f"{speaker} {row['text']}"
                )
            connection.execute(
                """
                INSERT INTO passages(
                    day_id, passage_index, start_seconds, end_seconds, text, segment_ids_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    day_id,
                    passage_index,
                    float(group[0]["start_seconds"]),
                    float(group[-1]["end_seconds"]),
                    "\n".join(lines),
                    json.dumps([int(row["id"]) for row in group]),
                ),
            )

    def _with_local_day_paths(self, raw: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        """Rebase stale absolute paths when a retained library changed hosts."""

        value = dict(raw)
        feed_id = str(value.get("feed_id") or "")
        archive_key = str(value.get("archive_date") or "").replace("-", "")
        if not feed_id or len(archive_key) != 8 or not archive_key.isdigit():
            return value

        day_directory = self.path.resolve().parent / feed_id / archive_key
        stem = f"combined_{feed_id}_{archive_key}"
        candidates = {
            "audio_path": day_directory / f"{stem}.mp3",
            "transcript_path": day_directory / "transcripts" / f"{stem}.json",
            "manifest_path": day_directory / f"{stem}.manifest.json",
        }
        for field, candidate in candidates.items():
            stored = Path(str(value.get(field) or ""))
            if stored.is_file():
                value[field] = str(stored.resolve())
            elif candidate.is_file():
                value[field] = str(candidate.resolve())
        return value

    def get_day(self, feed_id: str, archive_date: date) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM feed_days WHERE feed_id=? AND archive_date=?",
            (feed_id, archive_date.isoformat()),
        ).fetchone()
        return self._with_local_day_paths(row) if row is not None else None

    def list_days(self, feed_id: str | None = None) -> list[dict[str, Any]]:
        where = "WHERE d.feed_id=?" if feed_id else ""
        parameters: tuple[object, ...] = (feed_id,) if feed_id else ()
        rows = self.connection.execute(
            f"""
            SELECT d.*,
                   (SELECT COUNT(*) FROM transcript_segments s WHERE s.day_id=d.id)
                       AS segment_count,
                   (SELECT COUNT(DISTINCT s.speaker) FROM transcript_segments s
                    WHERE s.day_id=d.id AND s.speaker IS NOT NULL)
                       AS speaker_count,
                   (SELECT COUNT(*) FROM passages p WHERE p.day_id=d.id)
                       AS passage_count,
                   (SELECT COUNT(*) FROM incidents i WHERE i.day_id=d.id)
                       AS incident_count,
                   EXISTS(SELECT 1 FROM daily_summaries ds WHERE ds.day_id=d.id)
                       AS has_summary,
                   (SELECT ds.prompt_version FROM daily_summaries ds WHERE ds.day_id=d.id)
                       AS summary_prompt_version,
                   (SELECT ds.transcript_sha256 FROM daily_summaries ds WHERE ds.day_id=d.id)
                       AS summary_transcript_sha256,
                   (SELECT ds.model FROM daily_summaries ds WHERE ds.day_id=d.id)
                       AS summary_model
            FROM feed_days d
            {where}
            ORDER BY d.archive_date DESC, d.feed_id
            """,
            parameters,
        ).fetchall()
        return [self._with_local_day_paths(row) for row in rows]

    def save_feed_catalog(self, feeds: Sequence[dict[str, Any]]) -> None:
        now = utc_now()
        with self.transaction() as connection:
            for feed in feeds:
                feed_id = str(feed.get("feed_id") or "").strip()
                name = str(feed.get("name") or "").strip()
                if not feed_id or not name:
                    continue
                known = {"feed_id", "name", "location", "description", "genre", "status"}
                metadata = {key: value for key, value in feed.items() if key not in known}
                connection.execute(
                    """
                    INSERT INTO feed_catalog(
                        feed_id, name, location, description, genre, status,
                        metadata_json, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(feed_id) DO UPDATE SET
                        name=excluded.name,
                        location=excluded.location,
                        description=excluded.description,
                        genre=excluded.genre,
                        status=excluded.status,
                        metadata_json=excluded.metadata_json,
                        updated_at=excluded.updated_at
                    """,
                    (
                        feed_id,
                        name,
                        str(feed.get("location") or ""),
                        str(feed.get("description") or ""),
                        str(feed.get("genre") or ""),
                        str(feed.get("status") or ""),
                        json.dumps(metadata, ensure_ascii=False),
                        now,
                    ),
                )

    def list_feed_catalog(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM feed_catalog ORDER BY name COLLATE NOCASE, feed_id"
        ).fetchall()
        results = []
        for row in rows:
            result = dict(row)
            try:
                result.update(json.loads(str(result.pop("metadata_json") or "{}")))
            except (TypeError, json.JSONDecodeError):
                result.pop("metadata_json", None)
            results.append(result)
        return results

    def get_segments(self, day_id: int) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT id, segment_index, start_seconds, end_seconds, speaker, text
            FROM transcript_segments WHERE day_id=? ORDER BY start_seconds, segment_index
            """,
            (day_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_passages(
        self,
        feed_id: str,
        start_date: date,
        end_date: date,
    ) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT p.*, d.feed_id, d.archive_date, d.manifest_path
            FROM passages p JOIN feed_days d ON d.id=p.day_id
            WHERE d.feed_id=? AND d.archive_date BETWEEN ? AND ?
            ORDER BY d.archive_date, p.start_seconds
            """,
            (feed_id, start_date.isoformat(), end_date.isoformat()),
        ).fetchall()
        return [dict(row) for row in rows]

    def search_passages(
        self,
        feed_id: str,
        start_date: date,
        end_date: date,
        query: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        tokens = [token for token in query.replace("'", " ").split() if len(token) >= 2]
        if not tokens:
            return []
        match_query = " OR ".join(f'"{token}"' for token in tokens[:20])
        rows = self.connection.execute(
            """
            SELECT p.*, d.feed_id, d.archive_date, d.manifest_path,
                   bm25(passage_fts) AS rank
            FROM passage_fts
            JOIN passages p ON p.id=passage_fts.rowid
            JOIN feed_days d ON d.id=p.day_id
            WHERE passage_fts MATCH ? AND d.feed_id=? AND d.archive_date BETWEEN ? AND ?
            ORDER BY rank LIMIT ?
            """,
            (
                match_query,
                feed_id,
                start_date.isoformat(),
                end_date.isoformat(),
                max(1, limit),
            ),
        ).fetchall()
        return [dict(row) for row in rows]

    def replace_incidents(
        self,
        day_id: int,
        incidents: Sequence[dict[str, Any]],
        model: str,
        prompt_version: str,
    ) -> list[int]:
        ids: list[int] = []
        with self.transaction() as connection:
            connection.execute(
                "DELETE FROM incidents WHERE day_id=?",
                (day_id,),
            )
            connection.execute("DELETE FROM daily_summaries WHERE day_id=?", (day_id,))
            for incident in incidents:
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO incidents(
                        day_id, fingerprint, event_type, title, summary, location_text,
                        start_seconds, end_seconds, priority, confidence, evidence_json,
                        attributes_json, model, prompt_version, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        day_id,
                        incident["fingerprint"],
                        incident["event_type"],
                        incident["title"],
                        incident["summary"],
                        incident.get("location"),
                        float(incident["start_seconds"]),
                        float(incident["end_seconds"]),
                        int(incident["priority"]),
                        float(incident["confidence"]),
                        json.dumps(incident.get("evidence", []), ensure_ascii=False),
                        json.dumps(incident.get("attributes", {}), ensure_ascii=False),
                        model,
                        prompt_version,
                        utc_now(),
                    ),
                )
                if cursor.lastrowid:
                    ids.append(int(cursor.lastrowid))
        return ids

    def get_analysis_window_checkpoint(
        self,
        day_id: int,
        model: str,
        prompt_version: str,
        transcript_sha256: str,
        window_index: int,
        window_fingerprint: str,
    ) -> list[dict[str, Any]] | None:
        row = self.connection.execute(
            """
            SELECT incidents_json FROM analysis_window_checkpoints
            WHERE day_id=? AND model=? AND prompt_version=?
              AND window_index=? AND window_fingerprint=?
            ORDER BY
              CASE WHEN transcript_sha256=? THEN 0 ELSE 1 END,
              updated_at DESC
            LIMIT 1
            """,
            (
                day_id,
                model,
                prompt_version,
                window_index,
                window_fingerprint,
                transcript_sha256,
            ),
        ).fetchone()
        if row is None:
            return None
        try:
            value = json.loads(str(row["incidents_json"]))
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(value, list) or not all(
            isinstance(item, dict) for item in value
        ):
            return None
        return value

    def save_analysis_window_checkpoint(
        self,
        day_id: int,
        model: str,
        prompt_version: str,
        transcript_sha256: str,
        window_index: int,
        window_fingerprint: str,
        incidents: Sequence[dict[str, Any]],
    ) -> None:
        now = utc_now()
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO analysis_window_checkpoints(
                    day_id, model, prompt_version, transcript_sha256,
                    window_index, window_fingerprint, incidents_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(
                    day_id, model, prompt_version, transcript_sha256, window_index
                ) DO UPDATE SET
                    window_fingerprint=excluded.window_fingerprint,
                    incidents_json=excluded.incidents_json,
                    updated_at=excluded.updated_at
                """,
                (
                    day_id,
                    model,
                    prompt_version,
                    transcript_sha256,
                    window_index,
                    window_fingerprint,
                    json.dumps(list(incidents), ensure_ascii=False, sort_keys=True),
                    now,
                    now,
                ),
            )

    def clear_analysis_window_checkpoints(
        self,
        day_id: int,
        model: str,
        prompt_version: str,
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                """
                DELETE FROM analysis_window_checkpoints
                WHERE day_id=? AND model=? AND prompt_version=?
                """,
                (day_id, model, prompt_version),
            )

    def prune_analysis_window_checkpoints(
        self,
        day_id: int,
        model: str,
        prompt_version: str,
        transcript_sha256: str,
    ) -> None:
        """Keep only the completed current transcript revision for one run."""

        with self.transaction() as connection:
            connection.execute(
                """
                DELETE FROM analysis_window_checkpoints
                WHERE day_id=? AND model=? AND prompt_version=?
                  AND transcript_sha256<>?
                """,
                (day_id, model, prompt_version, transcript_sha256),
            )

    def get_incidents(
        self,
        feed_id: str,
        start_date: date,
        end_date: date,
        *,
        prompt_version: str | None = None,
    ) -> list[dict[str, Any]]:
        prompt_clause = " AND i.prompt_version=?" if prompt_version else ""
        parameters: tuple[object, ...] = (
            feed_id,
            start_date.isoformat(),
            end_date.isoformat(),
            *((prompt_version,) if prompt_version else ()),
        )
        rows = self.connection.execute(
            f"""
            SELECT i.*, d.feed_id, d.archive_date, d.manifest_path,
                   d.audio_path, d.audio_sha256, d.transcript_sha256,
                   d.has_diarization
            FROM incidents i JOIN feed_days d ON d.id=i.day_id
            WHERE d.feed_id=? AND d.archive_date BETWEEN ? AND ?{prompt_clause}
            ORDER BY d.archive_date, i.start_seconds, i.priority DESC
            """,
            parameters,
        ).fetchall()
        values = []
        for row in rows:
            value = self._with_local_day_paths(row)
            value["location"] = value.pop("location_text")
            value["evidence"] = json.loads(value.pop("evidence_json"))
            value["attributes"] = json.loads(value.pop("attributes_json"))
            values.append(value)
        return values

    def get_incident(self, incident_id: int) -> dict[str, Any] | None:
        row = self.connection.execute(
            """
            SELECT i.*, d.feed_id, d.archive_date, d.manifest_path,
                   d.audio_path, d.audio_sha256, d.transcript_sha256,
                   d.has_diarization
            FROM incidents i JOIN feed_days d ON d.id=i.day_id
            WHERE i.id=?
            """,
            (int(incident_id),),
        ).fetchone()
        if row is None:
            return None
        value = self._with_local_day_paths(row)
        value["location"] = value.pop("location_text")
        value["evidence"] = json.loads(value.pop("evidence_json"))
        value["attributes"] = json.loads(value.pop("attributes_json"))
        return value

    def get_incidents_for_run(
        self, day_id: int, model: str, prompt_version: str
    ) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT i.*, d.feed_id, d.archive_date, d.manifest_path
            FROM incidents i JOIN feed_days d ON d.id=i.day_id
            WHERE i.day_id=? AND i.model=? AND i.prompt_version=?
            ORDER BY i.start_seconds, i.priority DESC
            """,
            (day_id, model, prompt_version),
        ).fetchall()
        values = []
        for row in rows:
            value = self._with_local_day_paths(row)
            value["location"] = value.pop("location_text")
            value["evidence"] = json.loads(value.pop("evidence_json"))
            value["attributes"] = json.loads(value.pop("attributes_json"))
            values.append(value)
        return values

    def get_incidents_for_feeds(
        self,
        feed_ids: Sequence[str],
        start_date: date,
        end_date: date,
        *,
        prompt_version: str | None = None,
    ) -> list[dict[str, Any]]:
        normalized = list(dict.fromkeys(str(value) for value in feed_ids if str(value)))
        if not normalized:
            return []
        placeholders = ",".join("?" for _ in normalized)
        prompt_clause = " AND i.prompt_version=?" if prompt_version else ""
        parameters: tuple[object, ...] = (
            *normalized,
            start_date.isoformat(),
            end_date.isoformat(),
            *((prompt_version,) if prompt_version else ()),
        )
        rows = self.connection.execute(
            f"""
            SELECT i.*, d.feed_id, d.archive_date, d.manifest_path,
                   d.audio_path, d.audio_sha256, d.transcript_sha256,
                   d.has_diarization
            FROM incidents i JOIN feed_days d ON d.id=i.day_id
            WHERE d.feed_id IN ({placeholders}) AND d.archive_date BETWEEN ? AND ?{prompt_clause}
            ORDER BY d.archive_date, i.start_seconds, i.priority DESC
            """,
            parameters,
        ).fetchall()
        values = []
        for row in rows:
            value = self._with_local_day_paths(row)
            value["location"] = value.pop("location_text")
            value["evidence"] = json.loads(value.pop("evidence_json"))
            value["attributes"] = json.loads(value.pop("attributes_json"))
            values.append(value)
        return values

    def save_area_profile(
        self,
        name: str,
        zip_codes: Sequence[str],
        feeds: Sequence[dict[str, Any]],
        coverage: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Area profile name is required.")
        if len(clean_name) > 80:
            raise ValueError("Area profile name must be 80 characters or fewer.")
        clean_zips = list(dict.fromkeys(str(value).strip() for value in zip_codes))
        if not clean_zips or any(not value.isdigit() or len(value) != 5 for value in clean_zips):
            raise ValueError("Area profiles require one or more five-digit ZIP codes.")
        raw_coverage = coverage or {}
        coverage_mode = str(raw_coverage.get("mode") or "zip-list").strip().lower()
        if coverage_mode not in {"radius", "zip-list"}:
            raise ValueError("Area coverage mode must be radius or zip-list.")
        center_zip = str(raw_coverage.get("center_zip") or clean_zips[0]).strip()
        if center_zip not in clean_zips:
            raise ValueError("The center ZIP must be included in the discovered ZIP list.")
        radius_value = raw_coverage.get("radius_miles")
        radius_miles = float(radius_value) if radius_value not in {None, ""} else None
        if coverage_mode == "radius" and (radius_miles is None or not 1 <= radius_miles <= 100):
            raise ValueError("Radius area profiles require 1 through 100 miles.")
        max_zip_codes = int(raw_coverage.get("max_zip_codes") or len(clean_zips))
        if not 1 <= max_zip_codes <= 20:
            raise ValueError("Area profiles may search at most 20 ZIP areas.")
        clean_coverage = {
            "mode": coverage_mode,
            "center_zip": center_zip,
            "radius_miles": radius_miles if coverage_mode == "radius" else None,
            "max_zip_codes": max_zip_codes,
            "searched_zip_codes": [
                {
                    "zip_code": zip_code,
                    "distance_miles": next(
                        (
                            round(float(candidate["distance_miles"]), 2)
                            if candidate.get("distance_miles") is not None
                            else None
                            for candidate in raw_coverage.get("searched_zip_codes", [])
                            if isinstance(candidate, dict)
                            and str(candidate.get("zip_code") or "").strip() == zip_code
                        ),
                        None,
                    ),
                }
                for zip_code in clean_zips
            ],
            "distance_basis": str(raw_coverage.get("distance_basis") or ""),
        }
        clean_feeds: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in feeds:
            feed_id = str(raw.get("feed_id") or "").strip()
            if not feed_id.isdigit() or feed_id in seen:
                continue
            seen.add(feed_id)
            clean_feeds.append(
                {
                    "feed_id": feed_id,
                    "name": str(raw.get("name") or f"Feed {feed_id}"),
                    "location": str(raw.get("location") or ""),
                    "description": str(raw.get("description") or ""),
                    "genre": str(raw.get("genre") or ""),
                    "listeners": int(raw.get("listeners") or 0),
                    "status": str(raw.get("status") or ""),
                    "matched_zip_codes": list(raw.get("matched_zip_codes") or []),
                    "nearest_zip_code": str(raw.get("nearest_zip_code") or ""),
                    "distance_miles": (
                        round(float(raw["distance_miles"]), 2)
                        if raw.get("distance_miles") is not None
                        else None
                    ),
                    "priority_rank": int(raw.get("priority_rank") or len(clean_feeds) + 1),
                }
            )
        if not clean_feeds:
            raise ValueError("Select at least one feed for the area profile.")
        clean_feeds.sort(
            key=lambda value: (
                int(value["priority_rank"]),
                float(value["distance_miles"])
                if value["distance_miles"] is not None
                else float("inf"),
                str(value["name"]).lower(),
            )
        )
        for rank, feed in enumerate(clean_feeds, start=1):
            feed["priority_rank"] = rank
        now = utc_now()
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO area_profiles(
                    name, zip_codes_json, feeds_json, coverage_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    zip_codes_json=excluded.zip_codes_json,
                    feeds_json=excluded.feeds_json,
                    coverage_json=excluded.coverage_json,
                    updated_at=excluded.updated_at
                """,
                (
                    clean_name,
                    json.dumps(clean_zips),
                    json.dumps(clean_feeds, ensure_ascii=False),
                    json.dumps(clean_coverage, ensure_ascii=False),
                    now,
                    now,
                ),
            )
        profile = self.get_area_profile(clean_name)
        assert profile is not None
        return profile

    @staticmethod
    def _area_profile(row: sqlite3.Row) -> dict[str, Any]:
        value = dict(row)
        value["zip_codes"] = json.loads(value.pop("zip_codes_json"))
        value["feeds"] = json.loads(value.pop("feeds_json"))
        value["coverage"] = json.loads(value.pop("coverage_json") or "{}")
        value["feed_ids"] = [str(feed["feed_id"]) for feed in value["feeds"]]
        return value

    def get_area_profile(self, name: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM area_profiles WHERE name=?", (name.strip(),)
        ).fetchone()
        return self._area_profile(row) if row is not None else None

    def list_area_profiles(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM area_profiles ORDER BY name COLLATE NOCASE"
        ).fetchall()
        return [self._area_profile(row) for row in rows]

    @staticmethod
    def _area_acquisition_fingerprint(
        processing: dict[str, Any], feeds: Sequence[dict[str, Any]]
    ) -> str:
        payload = {
            "processing": processing,
            "feeds": [
                {
                    "feed_id": str(feed.get("feed_id") or ""),
                    "priority_rank": int(feed.get("priority_rank") or index),
                    "distance_miles": feed.get("distance_miles"),
                }
                for index, feed in enumerate(feeds, start=1)
            ],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def ensure_area_acquisition_run(
        self,
        profile_id: int,
        start_date: date,
        end_date: date,
        processing: dict[str, Any],
        feeds: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        if start_date > end_date:
            raise ValueError("Area acquisition start date must be on or before its end date.")
        if not feeds:
            raise ValueError("Area acquisition requires at least one selected feed.")
        fingerprint = self._area_acquisition_fingerprint(processing, feeds)
        serialized_processing = json.dumps(processing, sort_keys=True, ensure_ascii=False)
        now = utc_now()
        with self.transaction() as connection:
            row = connection.execute(
                """
                SELECT id FROM area_acquisition_runs
                WHERE profile_id=? AND start_date=? AND end_date=?
                  AND processing_fingerprint=?
                """,
                (
                    profile_id,
                    start_date.isoformat(),
                    end_date.isoformat(),
                    fingerprint,
                ),
            ).fetchone()
            if row is None:
                cursor = connection.execute(
                    """
                    INSERT INTO area_acquisition_runs(
                        profile_id, start_date, end_date, processing_fingerprint,
                        processing_json, status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, 'running', ?, ?)
                    """,
                    (
                        profile_id,
                        start_date.isoformat(),
                        end_date.isoformat(),
                        fingerprint,
                        serialized_processing,
                        now,
                        now,
                    ),
                )
                run_id = int(cursor.lastrowid)
            else:
                run_id = int(row["id"])
                connection.execute(
                    """
                    UPDATE area_acquisition_runs
                    SET processing_json=?, status='running', stop_reason='',
                        updated_at=?, completed_at=NULL
                    WHERE id=?
                    """,
                    (serialized_processing, now, run_id),
                )
                connection.execute(
                    """
                    UPDATE area_acquisition_items
                    SET status='pending', message='Recovered an interrupted local worker.', updated_at=?
                    WHERE run_id=? AND status='running'
                    """,
                    (now, run_id),
                )
            for index, feed in enumerate(feeds, start=1):
                feed_id = str(feed.get("feed_id") or "").strip()
                if not feed_id.isdigit():
                    raise ValueError("Area acquisition feed IDs must contain only digits.")
                distance = feed.get("distance_miles")
                connection.execute(
                    """
                    INSERT INTO area_acquisition_items(
                        run_id, feed_id, feed_name, priority_rank, distance_miles,
                        status, updated_at
                    ) VALUES (?, ?, ?, ?, ?, 'pending', ?)
                    ON CONFLICT(run_id, feed_id) DO UPDATE SET
                        feed_name=excluded.feed_name,
                        priority_rank=excluded.priority_rank,
                        distance_miles=excluded.distance_miles,
                        updated_at=excluded.updated_at
                    """,
                    (
                        run_id,
                        feed_id,
                        str(feed.get("name") or f"Feed {feed_id}"),
                        int(feed.get("priority_rank") or index),
                        float(distance) if distance is not None else None,
                        now,
                    ),
                )
        result = self.get_area_acquisition_run(run_id)
        assert result is not None
        return result

    def start_area_acquisition_item(self, run_id: int, feed_id: str) -> None:
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE area_acquisition_items
                SET status='running', message='', attempt_count=attempt_count+1, updated_at=?
                WHERE run_id=? AND feed_id=?
                """,
                (utc_now(), run_id, feed_id),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"Feed {feed_id} is not in area acquisition run {run_id}.")

    def finish_area_acquisition_item(
        self,
        run_id: int,
        feed_id: str,
        *,
        status: str,
        result: dict[str, Any] | None = None,
        message: str = "",
    ) -> None:
        if status not in {"complete", "partial", "failed", "pending"}:
            raise ValueError(f"Unsupported area acquisition item status: {status}")
        value = result or {}
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE area_acquisition_items
                SET status=?, requested_days=?, completed_days=?, missing_days_json=?,
                    download_limited=?, result_json=?, message=?, updated_at=?
                WHERE run_id=? AND feed_id=?
                """,
                (
                    status,
                    int(value.get("requested_days") or 0),
                    int(value.get("completed_days") or 0),
                    json.dumps(list(value.get("missing_days") or [])),
                    int(bool(value.get("download_limited"))),
                    json.dumps(value, ensure_ascii=False),
                    message,
                    utc_now(),
                    run_id,
                    feed_id,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"Feed {feed_id} is not in area acquisition run {run_id}.")

    def finish_area_acquisition_run(
        self, run_id: int, status: str, stop_reason: str = ""
    ) -> dict[str, Any]:
        if status not in {"complete", "partial", "quota_limited", "failed", "canceled"}:
            raise ValueError(f"Unsupported area acquisition run status: {status}")
        now = utc_now()
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE area_acquisition_runs
                SET status=?, stop_reason=?, updated_at=?, completed_at=?
                WHERE id=?
                """,
                (status, stop_reason, now, now, run_id),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"Area acquisition run {run_id} was not found.")
        result = self.get_area_acquisition_run(run_id)
        assert result is not None
        return result

    @staticmethod
    def _area_acquisition_item(row: sqlite3.Row) -> dict[str, Any]:
        value = dict(row)
        value["download_limited"] = bool(value["download_limited"])
        value["missing_days"] = json.loads(value.pop("missing_days_json") or "[]")
        value["result"] = json.loads(value.pop("result_json") or "{}")
        return value

    def get_area_acquisition_run(self, run_id: int) -> dict[str, Any] | None:
        row = self.connection.execute(
            """
            SELECT r.*, p.name AS profile_name
            FROM area_acquisition_runs r
            JOIN area_profiles p ON p.id=r.profile_id
            WHERE r.id=?
            """,
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        value = dict(row)
        value["processing"] = json.loads(value.pop("processing_json") or "{}")
        items = self.connection.execute(
            """
            SELECT * FROM area_acquisition_items
            WHERE run_id=? ORDER BY priority_rank, id
            """,
            (run_id,),
        ).fetchall()
        value["items"] = [self._area_acquisition_item(item) for item in items]
        return value

    def list_area_acquisition_runs(
        self, profile_name: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        parameters: list[Any] = []
        where = ""
        if profile_name:
            where = "WHERE p.name=?"
            parameters.append(profile_name.strip())
        parameters.append(max(1, min(100, int(limit))))
        rows = self.connection.execute(
            f"""
            SELECT r.id
            FROM area_acquisition_runs r
            JOIN area_profiles p ON p.id=r.profile_id
            {where}
            ORDER BY r.updated_at DESC, r.id DESC
            LIMIT ?
            """,
            parameters,
        ).fetchall()
        return [
            result
            for row in rows
            if (result := self.get_area_acquisition_run(int(row["id"]))) is not None
        ]

    def get_area_story_digest(
        self,
        profile_id: int,
        start_date: date,
        end_date: date,
        model: str,
        prompt_version: str,
        source_fingerprint: str,
    ) -> sqlite3.Row | None:
        return self.connection.execute(
            """
            SELECT * FROM area_story_digests
            WHERE profile_id=? AND start_date=? AND end_date=? AND model=?
              AND prompt_version=? AND source_fingerprint=?
            """,
            (
                profile_id,
                start_date.isoformat(),
                end_date.isoformat(),
                model,
                prompt_version,
                source_fingerprint,
            ),
        ).fetchone()

    def get_latest_area_story_digest(
        self, profile_name: str, prompt_version: str | None = None
    ) -> sqlite3.Row | None:
        prompt_clause = " AND d.prompt_version=?" if prompt_version else ""
        parameters: tuple[object, ...] = (
            profile_name.strip(),
            *((prompt_version,) if prompt_version else ()),
        )
        return self.connection.execute(
            f"""
            SELECT d.*, p.name AS profile_name
            FROM area_story_digests d
            JOIN area_profiles p ON p.id=d.profile_id
            WHERE p.name=?{prompt_clause}
            ORDER BY d.created_at DESC, d.id DESC
            LIMIT 1
            """,
            parameters,
        ).fetchone()

    def save_area_story_digest(
        self,
        profile_id: int,
        start_date: date,
        end_date: date,
        summary: str,
        stories: Sequence[dict[str, Any]],
        coverage: dict[str, Any],
        model: str,
        prompt_version: str,
        source_fingerprint: str,
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO area_story_digests(
                    profile_id, start_date, end_date, summary, stories_json,
                    coverage_json, model, prompt_version, source_fingerprint, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(profile_id, start_date, end_date) DO UPDATE SET
                    summary=excluded.summary,
                    stories_json=excluded.stories_json,
                    coverage_json=excluded.coverage_json,
                    model=excluded.model,
                    prompt_version=excluded.prompt_version,
                    source_fingerprint=excluded.source_fingerprint,
                    created_at=excluded.created_at
                """,
                (
                    profile_id,
                    start_date.isoformat(),
                    end_date.isoformat(),
                    summary,
                    json.dumps(list(stories), ensure_ascii=False),
                    json.dumps(coverage, ensure_ascii=False),
                    model,
                    prompt_version,
                    source_fingerprint,
                    utc_now(),
                ),
            )

    def get_daily_summary(
        self,
        day_id: int,
        model: str,
        prompt_version: str,
        transcript_sha256: str,
    ) -> sqlite3.Row | None:
        return self.connection.execute(
            """
            SELECT * FROM daily_summaries
            WHERE day_id=? AND model=? AND prompt_version=? AND transcript_sha256=?
            """,
            (day_id, model, prompt_version, transcript_sha256),
        ).fetchone()

    def get_latest_daily_summary(self, day_id: int) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM daily_summaries WHERE day_id=?",
            (day_id,),
        ).fetchone()

    def save_daily_summary(
        self,
        day_id: int,
        summary: str,
        notable_incident_ids: Sequence[int],
        model: str,
        prompt_version: str,
        transcript_sha256: str,
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO daily_summaries(
                    day_id, summary, notable_incident_ids_json, model,
                    prompt_version, transcript_sha256, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(day_id) DO UPDATE SET
                    summary=excluded.summary,
                    notable_incident_ids_json=excluded.notable_incident_ids_json,
                    model=excluded.model,
                    prompt_version=excluded.prompt_version,
                    transcript_sha256=excluded.transcript_sha256,
                    created_at=excluded.created_at
                """,
                (
                    day_id,
                    summary,
                    json.dumps(list(notable_incident_ids)),
                    model,
                    prompt_version,
                    transcript_sha256,
                    utc_now(),
                ),
            )

    def get_weekly_summary(
        self,
        feed_id: str,
        start_date: date,
        end_date: date,
        model: str,
        prompt_version: str,
        source_fingerprint: str,
    ) -> sqlite3.Row | None:
        return self.connection.execute(
            """
            SELECT * FROM weekly_summaries
            WHERE feed_id=? AND start_date=? AND end_date=?
              AND model=? AND prompt_version=? AND source_fingerprint=?
            """,
            (
                feed_id,
                start_date.isoformat(),
                end_date.isoformat(),
                model,
                prompt_version,
                source_fingerprint,
            ),
        ).fetchone()

    def get_latest_weekly_summary(
        self,
        feed_id: str,
        start_date: date,
        end_date: date,
        prompt_version: str | None = None,
    ) -> sqlite3.Row | None:
        prompt_clause = " AND prompt_version=?" if prompt_version else ""
        parameters: tuple[object, ...] = (
            feed_id,
            start_date.isoformat(),
            end_date.isoformat(),
            *((prompt_version,) if prompt_version else ()),
        )
        return self.connection.execute(
            f"""
            SELECT * FROM weekly_summaries
            WHERE feed_id=? AND start_date=? AND end_date=?
              {prompt_clause}
            """,
            parameters,
        ).fetchone()

    def save_weekly_summary(
        self,
        feed_id: str,
        start_date: date,
        end_date: date,
        summary: str,
        notable_incident_ids: Sequence[int],
        days_available: int,
        incident_count: int,
        model: str,
        prompt_version: str,
        source_fingerprint: str,
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO weekly_summaries(
                    feed_id, start_date, end_date, summary,
                    notable_incident_ids_json, days_available, incident_count,
                    model, prompt_version, source_fingerprint, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(feed_id, start_date, end_date) DO UPDATE SET
                    summary=excluded.summary,
                    notable_incident_ids_json=excluded.notable_incident_ids_json,
                    days_available=excluded.days_available,
                    incident_count=excluded.incident_count,
                    model=excluded.model,
                    prompt_version=excluded.prompt_version,
                    source_fingerprint=excluded.source_fingerprint,
                    created_at=excluded.created_at
                """,
                (
                    feed_id,
                    start_date.isoformat(),
                    end_date.isoformat(),
                    summary,
                    json.dumps(list(notable_incident_ids)),
                    days_available,
                    incident_count,
                    model,
                    prompt_version,
                    source_fingerprint,
                    utc_now(),
                ),
            )

    def passages_missing_embeddings(self, model: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT p.* FROM passages p
            LEFT JOIN embeddings e
              ON e.entity_type='passage' AND e.entity_id=p.id AND e.model=?
            WHERE e.entity_id IS NULL
            ORDER BY p.id
            """,
            (model,),
        ).fetchall()
        return [dict(row) for row in rows]

    def save_embeddings(
        self,
        entity_type: str,
        values: Iterable[tuple[int, str, int, bytes]],
        model: str,
    ) -> None:
        with self.transaction() as connection:
            connection.executemany(
                """
                INSERT INTO embeddings(
                    entity_type, entity_id, model, dimensions,
                    text_sha256, vector, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(entity_type, entity_id, model) DO UPDATE SET
                    dimensions=excluded.dimensions,
                    text_sha256=excluded.text_sha256,
                    vector=excluded.vector,
                    created_at=excluded.created_at
                """,
                [
                    (entity_type, entity_id, model, dimensions, text_hash, vector, utc_now())
                    for entity_id, text_hash, dimensions, vector in values
                ],
            )

    def passage_embeddings(
        self,
        feed_id: str,
        start_date: date,
        end_date: date,
        model: str,
    ) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT p.*, d.archive_date, d.manifest_path, e.dimensions, e.vector
            FROM embeddings e
            JOIN passages p ON p.id=e.entity_id AND e.entity_type='passage'
            JOIN feed_days d ON d.id=p.day_id
            WHERE e.model=? AND d.feed_id=? AND d.archive_date BETWEEN ? AND ?
            """,
            (model, feed_id, start_date.isoformat(), end_date.isoformat()),
        ).fetchall()
        return [dict(row) for row in rows]

    def save_qa(
        self,
        feed_id: str,
        start_date: date,
        end_date: date,
        question: str,
        answer: str,
        evidence: Sequence[dict[str, Any]],
        model: str,
    ) -> int:
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO qa_history(
                    feed_id, start_date, end_date, question, answer,
                    evidence_json, model, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    feed_id,
                    start_date.isoformat(),
                    end_date.isoformat(),
                    question,
                    answer,
                    json.dumps(list(evidence), ensure_ascii=False),
                    model,
                    utc_now(),
                ),
            )
            return int(cursor.lastrowid)

    def stats(self) -> dict[str, int]:
        return {
            table: int(self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in (
                "feed_days",
                "transcript_segments",
                "passages",
                "incidents",
                "analysis_window_checkpoints",
                "daily_summaries",
                "weekly_summaries",
                "area_profiles",
                "area_acquisition_runs",
                "area_acquisition_items",
                "area_story_digests",
                "embeddings",
                "qa_history",
            )
        }
