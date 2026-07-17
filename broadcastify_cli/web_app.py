from __future__ import annotations

import argparse
import hmac
import json
import mimetypes
import os
import platform
import re
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from email.utils import formatdate
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, quote, unquote, urlparse

from dotenv import dotenv_values

from .analysis import PROMPT_VERSION, WEEKLY_PROMPT_VERSION
from .area_watch import AREA_PROMPT_VERSION
from .library import scan_local_library
from .storage import AnalysisStore


SESSION_COOKIE = "radio_archive_session"
MAX_BODY_BYTES = 1_048_576
MAX_EVENTS = 500
FEED_ID_PATTERN = re.compile(r"^\d+$")
ZIP_PATTERN = re.compile(r"^\d{5}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _readiness_environment(working_dir: Path) -> tuple[dict[str, str], str]:
    """Return non-secret setup flags using the worker's environment precedence."""

    values = {key: str(value) for key, value in os.environ.items()}
    loaded_path = ""
    candidates = [working_dir / ".env"]
    configured = os.getenv("BROADCASTIFY_ENV_FILE")
    if configured:
        candidates.append(Path(configured).expanduser())
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            parsed = dotenv_values(candidate)
        except (OSError, UnicodeError):
            continue
        values.update(
            {key: str(value) for key, value in parsed.items() if value is not None}
        )
        loaded_path = str(candidate.resolve())
    return values, loaded_path


def _runtime_readiness(state: "WebAppState") -> dict[str, Any]:
    values, environment_file = _readiness_environment(state.working_dir)
    username = values.get("BROADCASTIFY_USERNAME") or values.get("USERNAME")
    password = values.get("BROADCASTIFY_PASSWORD") or values.get("PASSWORD")
    session_path = state.working_dir / "cookies.json"
    probe = state.output_dir if state.output_dir.exists() else state.output_dir.parent
    storage_ready = bool(
        probe.is_dir()
        and os.access(probe, os.R_OK | os.W_OK)
        and (not state.output_dir.exists() or state.output_dir.is_dir())
    )
    credentials_configured = bool(username and password)
    saved_session = session_path.is_file()
    return {
        "storage_ready": storage_ready,
        "account": {
            "configured": credentials_configured or saved_session,
            "credentials_configured": credentials_configured,
            "saved_session_available": saved_session,
            "environment_file_available": bool(environment_file),
        },
    }


class WebRequestError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class JobRecord:
    id: str
    command: str
    status: str = "queued"
    created_at: str = field(default_factory=utc_now)
    started_at: str = ""
    finished_at: str = ""
    events: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str = ""
    cancel_requested: bool = False
    process: subprocess.Popen[str] | None = field(default=None, repr=False)

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "command": self.command,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "events": list(self.events),
            "result": self.result,
            "error": self.error,
        }


def _retained_media_url(
    output_dir: str | Path,
    path_value: str | Path | None,
) -> str:
    if not path_value:
        return ""
    root = Path(output_dir).resolve()
    path = Path(path_value).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError:
        return ""
    return f"/media?path={quote(relative.as_posix(), safe='/')}"


def _area_stories_for_web(
    output_dir: str | Path,
    stories: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add safe clip URLs without exposing workstation paths to the browser."""

    rendered: list[dict[str, Any]] = []
    for raw_story in stories:
        story = dict(raw_story)
        references: list[dict[str, Any]] = []
        for raw_reference in raw_story.get("incident_references", []):
            reference = dict(raw_reference)
            clip_path = str(reference.pop("clip_path", "") or "")
            reference.pop("source_audio_path", None)
            reference["media_url"] = _retained_media_url(output_dir, clip_path)
            reference["filename"] = Path(clip_path).name if reference["media_url"] else ""
            references.append(reference)
        story["incident_references"] = references
        rendered.append(story)
    return rendered


class JobManager:
    """Run the existing JSON worker behind a small, bounded local job API."""

    def __init__(self, output_dir: Path, database_path: Path, working_dir: Path) -> None:
        self.output_dir = output_dir
        self.database_path = database_path
        self.working_dir = working_dir
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.RLock()

    def start(self, command: str, payload: dict[str, Any]) -> dict[str, Any]:
        arguments, stdin_payload = self._worker_request(command, payload)
        with self._lock:
            if any(job.status in {"queued", "running", "canceling"} for job in self._jobs.values()):
                raise WebRequestError(
                    HTTPStatus.CONFLICT,
                    "Another archive or model job is already active. Wait for it or cancel it first.",
                )
            job = JobRecord(id=secrets.token_hex(8), command=command)
            self._jobs[job.id] = job
        threading.Thread(
            target=self._run,
            args=(job, arguments, stdin_payload),
            name=f"radio-job-{job.id}",
            daemon=True,
        ).start()
        return job.snapshot()

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise WebRequestError(HTTPStatus.NOT_FOUND, "That local job was not found.")
            return job.snapshot()

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise WebRequestError(HTTPStatus.NOT_FOUND, "That local job was not found.")
            if job.status not in {"queued", "running", "canceling"}:
                return job.snapshot()
            job.cancel_requested = True
            job.status = "canceling"
            process = job.process
        if process is not None and process.poll() is None:
            try:
                if os.name == "nt":
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    os.killpg(process.pid, signal.SIGINT)
            except (OSError, ValueError):
                try:
                    process.terminate()
                except OSError:
                    pass

            def force_stop() -> None:
                time.sleep(3)
                if process.poll() is None:
                    try:
                        process.kill()
                    except OSError:
                        pass

            threading.Thread(target=force_stop, daemon=True).start()
        return self.get(job_id)

    def _append(self, job: JobRecord, event: dict[str, Any]) -> None:
        with self._lock:
            if event.get("type") == "incident_clip" and isinstance(event.get("clip"), dict):
                clip = dict(event["clip"])
                try:
                    clip_path = Path(str(clip.get("path") or "")).resolve()
                    relative = clip_path.relative_to(self.output_dir)
                    clip["media_url"] = f"/media?path={quote(relative.as_posix(), safe='/')}"
                    clip["filename"] = clip_path.name
                except ValueError:
                    clip["media_url"] = ""
                event = {**event, "clip": clip}
            if event.get("type") == "area_digest" and isinstance(event.get("result"), dict):
                result = dict(event["result"])
                result["stories"] = _area_stories_for_web(
                    self.output_dir,
                    list(result.get("stories") or []),
                )
                event = {**event, "result": result}
            event = {**event, "event_index": len(job.events), "received_at": utc_now()}
            job.events.append(event)
            if len(job.events) > MAX_EVENTS:
                job.events = job.events[-MAX_EVENTS:]
            if event.get("type") not in {"stage", "progress", "log"}:
                job.result = event

    def _run(
        self,
        job: JobRecord,
        arguments: list[str],
        stdin_payload: dict[str, Any] | None,
    ) -> None:
        with self._lock:
            job.status = "running"
            job.started_at = utc_now()
        environment = os.environ.copy()
        environment["BROADCASTIFY_ANALYSIS_DB"] = str(self.database_path)
        creation_flags = 0
        start_new_session = os.name != "nt"
        if os.name == "nt":
            creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        stderr_lines: list[str] = []
        try:
            process = subprocess.Popen(
                [sys.executable, "-m", "broadcastify_cli.worker", *arguments],
                cwd=self.working_dir,
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creation_flags,
                start_new_session=start_new_session,
            )
            with self._lock:
                job.process = process

            def read_stderr() -> None:
                assert process.stderr is not None
                for value in process.stderr:
                    line = value.strip()
                    if line:
                        stderr_lines.append(line)
                        del stderr_lines[:-20]

            stderr_thread = threading.Thread(target=read_stderr, daemon=True)
            stderr_thread.start()
            assert process.stdin is not None
            if stdin_payload is not None:
                process.stdin.write(json.dumps(stdin_payload, ensure_ascii=False))
            process.stdin.close()
            assert process.stdout is not None
            for raw_line in process.stdout:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    if not isinstance(event, dict):
                        event = {"type": "log", "message": str(event)}
                except json.JSONDecodeError:
                    event = {"type": "log", "message": line}
                self._append(job, event)
            return_code = process.wait()
            stderr_thread.join(timeout=1)
            with self._lock:
                if job.cancel_requested:
                    job.status = "canceled"
                    job.error = "Canceled by the user. Completed files remain resumable."
                elif return_code == 0:
                    job.status = "completed"
                else:
                    job.status = "failed"
                    error_event = next(
                        (
                            event
                            for event in reversed(job.events)
                            if event.get("type") == "error" and event.get("message")
                        ),
                        None,
                    )
                    job.error = str(
                        (error_event or {}).get("message")
                        or (stderr_lines[-1] if stderr_lines else "The Python worker failed.")
                    )
                job.finished_at = utc_now()
                job.process = None
        except Exception as exc:  # pragma: no cover - defensive process boundary
            with self._lock:
                job.status = "canceled" if job.cancel_requested else "failed"
                job.error = str(exc)
                job.finished_at = utc_now()
                job.process = None

    def _worker_request(
        self, command: str, raw_payload: dict[str, Any]
    ) -> tuple[list[str], dict[str, Any] | None]:
        if not isinstance(raw_payload, dict):
            raise WebRequestError(HTTPStatus.BAD_REQUEST, "The job payload must be a JSON object.")
        payload = dict(raw_payload)
        if command == "search":
            query = str(payload.get("query") or "").strip()
            if not query or len(query) > 160:
                raise WebRequestError(HTTPStatus.BAD_REQUEST, "Enter a feed search of 160 characters or fewer.")
            return ["search", "--query", query], None
        if command == "area-search":
            center_zip = str(payload.get("center_zip") or "").strip()
            if center_zip:
                if not ZIP_PATTERN.fullmatch(center_zip):
                    raise WebRequestError(HTTPStatus.BAD_REQUEST, "Enter a five-digit center ZIP code.")
                try:
                    radius_miles = float(payload.get("radius_miles", 25))
                    max_zip_codes = int(payload.get("max_zip_codes", 12))
                except (TypeError, ValueError):
                    raise WebRequestError(
                        HTTPStatus.BAD_REQUEST, "Radius and ZIP limit must be numeric."
                    ) from None
                if not 1 <= radius_miles <= 100:
                    raise WebRequestError(HTTPStatus.BAD_REQUEST, "Radius must be between 1 and 100 miles.")
                if not 1 <= max_zip_codes <= 20:
                    raise WebRequestError(HTTPStatus.BAD_REQUEST, "ZIP limit must be between 1 and 20.")
                return [
                    "area-search",
                    "--center-zip",
                    center_zip,
                    "--radius-miles",
                    str(radius_miles),
                    "--max-zip-codes",
                    str(max_zip_codes),
                ], None
            zip_codes = list(dict.fromkeys(str(value).strip() for value in payload.get("zip_codes", [])))
            if not zip_codes or any(not ZIP_PATTERN.fullmatch(value) for value in zip_codes):
                raise WebRequestError(HTTPStatus.BAD_REQUEST, "Enter one or more five-digit ZIP codes.")
            arguments = ["area-search"]
            for zip_code in zip_codes[:20]:
                arguments.extend(["--zip", zip_code])
            return arguments, None
        if command == "incident-clip":
            try:
                incident_id = int(payload.get("incident_id"))
            except (TypeError, ValueError):
                raise WebRequestError(HTTPStatus.BAD_REQUEST, "A numeric incident ID is required.") from None
            if incident_id < 1:
                raise WebRequestError(HTTPStatus.BAD_REQUEST, "A positive incident ID is required.")
            return ["incident-clip", "--incident-id", str(incident_id)], None
        if command == "diagnostics":
            return ["diagnostics"], None

        stdin_commands = {
            "analysis-provider-diagnostics",
            "analyze-day",
            "asr-self-test",
            "diarization-self-test",
            "ask",
            "authenticate",
            "continue-local",
            "run",
            "run-area",
            "save-area-profile",
            "summarize-area",
            "summarize-week",
        }
        if command not in stdin_commands:
            raise WebRequestError(HTTPStatus.BAD_REQUEST, "That local job type is not supported.")
        if command in {"run", "continue-local", "analyze-day"}:
            payload["output_dir"] = str(self.output_dir)
        if command == "run":
            # Broadcastify's numeric limit is unknown and shared. The browser UI
            # deliberately keeps a single downloader and preserves source blocks.
            payload["download_jobs"] = 1
            payload["keep_originals"] = True
            if payload.get("diarize"):
                payload["combine"] = True
                payload["transcribe"] = True
        if command == "run-area":
            job_payload = dict(payload.get("job") or {})
            job_payload["output_dir"] = str(self.output_dir)
            job_payload["download_jobs"] = 1
            job_payload["keep_originals"] = True
            if job_payload.get("diarize"):
                job_payload["combine"] = True
                job_payload["transcribe"] = True
            payload["job"] = job_payload
        return [command], payload


@dataclass(frozen=True)
class WebAppState:
    output_dir: Path
    database_path: Path
    working_dir: Path
    static_dir: Path
    session_token: str
    jobs: JobManager


def _safe_media_path(state: WebAppState, relative_value: str) -> Path:
    relative = Path(unquote(relative_value))
    if relative.is_absolute() or ".." in relative.parts:
        raise WebRequestError(HTTPStatus.BAD_REQUEST, "The media path is not valid.")
    candidate = (state.output_dir / relative).resolve()
    try:
        candidate.relative_to(state.output_dir)
    except ValueError:
        raise WebRequestError(HTTPStatus.FORBIDDEN, "The media path is outside the archive library.") from None
    if not candidate.is_file():
        raise WebRequestError(HTTPStatus.NOT_FOUND, "That retained media file was not found.")
    return candidate


def _media_url(state: WebAppState, path_value: str | Path | None) -> str:
    return _retained_media_url(state.output_dir, path_value)


def _library_payload(state: WebAppState) -> dict[str, Any]:
    days = scan_local_library(state.output_dir, state.database_path)
    return {
        "days": days,
        "summary": {
            "feed_count": len({value["feed_id"] for value in days}),
            "day_count": len(days),
            "complete_count": sum(bool(value["is_complete"]) for value in days),
            "attention_count": sum(not bool(value["is_complete"]) for value in days),
            "storage_bytes": sum(int(value["storage_bytes"]) for value in days),
        },
    }


def _compact_incident(value: dict[str, Any]) -> dict[str, Any]:
    evidence = list(value.get("evidence") or [])
    quote_text = ""
    for item in evidence:
        if isinstance(item, dict):
            quote_text = str(item.get("text") or item.get("quote") or "").strip()
            if quote_text:
                break
    return {
        "id": int(value["id"]),
        "event_type": str(value.get("event_type") or "other"),
        "title": str(value.get("title") or "Untitled radio event"),
        "summary": str(value.get("summary") or ""),
        "location": str(value.get("location") or ""),
        "priority": int(value.get("priority") or 0),
        "confidence": float(value.get("confidence") or 0),
        "start_seconds": float(value.get("start_seconds") or 0),
        "end_seconds": float(value.get("end_seconds") or 0),
        "quote": quote_text,
    }


def _day_payload(state: WebAppState, feed_id: str, archive_date: date) -> dict[str, Any]:
    library = scan_local_library(state.output_dir, state.database_path)
    day_state = next(
        (
            value
            for value in library
            if value["feed_id"] == feed_id and value["archive_date"] == archive_date.isoformat()
        ),
        None,
    )
    if day_state is None:
        raise WebRequestError(HTTPStatus.NOT_FOUND, "That feed day is not in the local library.")
    summary = ""
    incidents: list[dict[str, Any]] = []
    with AnalysisStore(state.database_path) as store:
        stored_day = store.get_day(feed_id, archive_date)
        if stored_day is not None:
            stored_summary = store.get_latest_daily_summary(int(stored_day["id"]))
            analysis_current = bool(
                stored_summary
                and str(stored_summary["prompt_version"]) == PROMPT_VERSION
            )
            summary = (
                str(stored_summary["summary"])
                if stored_summary and analysis_current
                else ""
            )
            if analysis_current:
                incidents = [
                    _compact_incident(value)
                    for value in store.get_incidents(
                        feed_id,
                        archive_date,
                        archive_date,
                        prompt_version=PROMPT_VERSION,
                    )
                ]
    return {
        "state": day_state,
        "summary": summary,
        "incidents": incidents,
        "audio_url": _media_url(state, day_state.get("combined_path")),
    }


def _transcript_payload(
    state: WebAppState,
    feed_id: str,
    archive_date: date,
    offset: int,
    limit: int,
    query: str,
) -> dict[str, Any]:
    segments: list[dict[str, Any]] = []
    with AnalysisStore(state.database_path) as store:
        stored_day = store.get_day(feed_id, archive_date)
        if stored_day is not None:
            segments = store.get_segments(int(stored_day["id"]))
    if not segments:
        day = _day_payload(state, feed_id, archive_date)["state"]
        transcript_path = Path(str(day.get("transcript_path") or ""))
        if transcript_path.is_file():
            try:
                raw_segments = json.loads(transcript_path.read_text(encoding="utf-8")).get("segments", [])
                segments = [
                    {
                        "segment_index": index,
                        "start_seconds": float(value.get("start", 0)),
                        "end_seconds": float(value.get("end", 0)),
                        "speaker": value.get("speaker"),
                        "text": str(value.get("text") or "").strip(),
                    }
                    for index, value in enumerate(raw_segments)
                    if isinstance(value, dict)
                ]
            except (OSError, ValueError, TypeError):
                segments = []
    clean_query = query.strip().casefold()
    if clean_query:
        segments = [
            value
            for value in segments
            if clean_query in str(value.get("text") or "").casefold()
            or clean_query in str(value.get("speaker") or "").casefold()
        ]
    total = len(segments)
    return {
        "segments": segments[offset : offset + limit],
        "offset": offset,
        "limit": limit,
        "total": total,
        "has_more": offset + limit < total,
    }


def create_server(
    output_dir: str | Path = "archives",
    database_path: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    working_dir: str | Path | None = None,
) -> ThreadingHTTPServer:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("The web app may only bind to a loopback address.")
    root = Path(output_dir).expanduser().resolve()
    database = (
        Path(database_path).expanduser().resolve()
        if database_path
        else (root / "broadcastify-analysis.sqlite3").resolve()
    )
    work = Path(working_dir or Path.cwd()).resolve()
    static_dir = Path(__file__).with_name("web_static").resolve()
    token = secrets.token_urlsafe(32)
    state = WebAppState(
        output_dir=root,
        database_path=database,
        working_dir=work,
        static_dir=static_dir,
        session_token=token,
        jobs=JobManager(root, database, work),
    )

    class Handler(BaseHTTPRequestHandler):
        server_version = "RadioArchiveLocal/0.3"

        def log_message(self, format_value: str, *args: object) -> None:
            if getattr(self.server, "quiet", False):
                return
            super().log_message(format_value, *args)

        def do_GET(self) -> None:  # noqa: N802
            try:
                self._get()
            except WebRequestError as exc:
                self._json(exc.status, {"error": str(exc)})
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception:
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "The local web service hit an unexpected error."})

        def do_POST(self) -> None:  # noqa: N802
            try:
                self._require_session(write=True)
                self._post()
            except WebRequestError as exc:
                self._json(exc.status, {"error": str(exc)})
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception:
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "The local web service hit an unexpected error."})

        def _get(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                self._json(HTTPStatus.OK, {"status": "ok", "scope": "loopback-only"})
                return
            if parsed.path == "/":
                index_path = state.static_dir / "index.html"
                html = index_path.read_text(encoding="utf-8").replace("__SESSION_TOKEN__", state.session_token)
                self._bytes(
                    HTTPStatus.OK,
                    html.encode("utf-8"),
                    "text/html; charset=utf-8",
                    cookie=True,
                    no_store=True,
                )
                return
            if parsed.path.startswith("/static/"):
                name = parsed.path.removeprefix("/static/")
                if not name or "/" in name or "\\" in name or name.startswith("."):
                    raise WebRequestError(HTTPStatus.NOT_FOUND, "Static file not found.")
                path = state.static_dir / name
                if not path.is_file():
                    raise WebRequestError(HTTPStatus.NOT_FOUND, "Static file not found.")
                content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                self._bytes(HTTPStatus.OK, path.read_bytes(), content_type, no_store=True)
                return

            self._require_session(write=False)
            query = parse_qs(parsed.query)
            if parsed.path == "/api/bootstrap":
                library = _library_payload(state)
                with AnalysisStore(state.database_path) as store:
                    profiles = store.list_area_profiles()
                    area_runs = store.list_area_acquisition_runs(limit=20)
                self._json(
                    HTTPStatus.OK,
                    {
                        **library,
                        "profiles": profiles,
                        "area_runs": area_runs,
                        "runtime": {
                            "platform": platform.system(),
                            "platform_release": platform.release(),
                            "python": platform.python_version(),
                            "output_dir": str(state.output_dir),
                            "database_path": str(state.database_path),
                            "loopback_only": True,
                            **_runtime_readiness(state),
                        },
                    },
                )
                return
            if parsed.path == "/api/day":
                feed_id, archive_date = self._feed_date(query)
                self._json(HTTPStatus.OK, _day_payload(state, feed_id, archive_date))
                return
            if parsed.path == "/api/transcript":
                feed_id, archive_date = self._feed_date(query)
                offset = max(0, self._query_int(query, "offset", 0))
                limit = min(1000, max(1, self._query_int(query, "limit", 250)))
                search = str((query.get("q") or [""])[0])[:200]
                self._json(
                    HTTPStatus.OK,
                    _transcript_payload(state, feed_id, archive_date, offset, limit, search),
                )
                return
            if parsed.path.startswith("/api/jobs/"):
                job_id = parsed.path.removeprefix("/api/jobs/")
                self._json(HTTPStatus.OK, state.jobs.get(job_id))
                return
            if parsed.path == "/api/saved-area-digest":
                profile_name = str((query.get("profile_name") or [""])[0]).strip()
                if not profile_name:
                    raise WebRequestError(HTTPStatus.BAD_REQUEST, "An area profile name is required.")
                with AnalysisStore(state.database_path) as store:
                    latest_any = store.get_latest_area_story_digest(profile_name)
                    row = store.get_latest_area_story_digest(
                        profile_name,
                        prompt_version=AREA_PROMPT_VERSION,
                    )
                result = None
                stale = latest_any is not None and row is None
                if row is not None:
                    coverage = json.loads(str(row["coverage_json"]))
                    if str(coverage.get("incident_prompt_version") or "") != PROMPT_VERSION:
                        stale = True
                    else:
                        result = {
                            "profile_name": str(row["profile_name"]),
                            "start_date": str(row["start_date"]),
                            "end_date": str(row["end_date"]),
                            "summary": str(row["summary"]),
                            "stories": _area_stories_for_web(
                                state.output_dir,
                                json.loads(str(row["stories_json"])),
                            ),
                            "coverage": coverage,
                        }
                self._json(HTTPStatus.OK, {"result": result, "stale": stale})
                return
            if parsed.path == "/api/saved-week":
                feed_id = str((query.get("feed_id") or [""])[0]).strip()
                week_ending = self._date_value(str((query.get("week_ending") or [""])[0]))
                if not FEED_ID_PATTERN.fullmatch(feed_id):
                    raise WebRequestError(HTTPStatus.BAD_REQUEST, "A numeric feed ID is required.")
                week_start = week_ending - timedelta(days=6)
                with AnalysisStore(state.database_path) as store:
                    row = store.get_latest_weekly_summary(
                        feed_id,
                        week_start,
                        week_ending,
                        prompt_version=WEEKLY_PROMPT_VERSION,
                    )
                result = None
                if row is not None:
                    result = dict(row)
                    result["notable_incident_ids"] = json.loads(
                        str(result.pop("notable_incident_ids_json") or "[]")
                    )
                self._json(HTTPStatus.OK, {"result": result})
                return
            if parsed.path == "/media":
                media_value = str((query.get("path") or [""])[0])
                if not media_value:
                    raise WebRequestError(HTTPStatus.BAD_REQUEST, "A retained media path is required.")
                self._media(_safe_media_path(state, media_value))
                return
            raise WebRequestError(HTTPStatus.NOT_FOUND, "Page not found.")

        def _post(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/jobs":
                body = self._body()
                command = str(body.get("command") or "").strip()
                payload = body.get("payload") or {}
                if not isinstance(payload, dict):
                    raise WebRequestError(HTTPStatus.BAD_REQUEST, "The job payload must be a JSON object.")
                self._json(HTTPStatus.ACCEPTED, state.jobs.start(command, payload))
                return
            if parsed.path.startswith("/api/jobs/") and parsed.path.endswith("/cancel"):
                job_id = parsed.path.removeprefix("/api/jobs/").removesuffix("/cancel")
                self._json(HTTPStatus.OK, state.jobs.cancel(job_id))
                return
            raise WebRequestError(HTTPStatus.NOT_FOUND, "Page not found.")

        def _require_session(self, write: bool) -> None:
            cookie = SimpleCookie(self.headers.get("Cookie", ""))
            supplied = cookie.get(SESSION_COOKIE)
            if supplied is None or not hmac.compare_digest(supplied.value, state.session_token):
                raise WebRequestError(HTTPStatus.FORBIDDEN, "This request is not from the active local app session.")
            if not write:
                return
            header_token = self.headers.get("X-Radio-Archive-Token", "")
            if not hmac.compare_digest(header_token, state.session_token):
                raise WebRequestError(HTTPStatus.FORBIDDEN, "The local action token is missing or invalid.")
            origin = self.headers.get("Origin")
            expected_origins = {
                f"http://{self.headers.get('Host', '')}",
                f"http://127.0.0.1:{self.server.server_port}",
                f"http://localhost:{self.server.server_port}",
            }
            if origin and origin not in expected_origins:
                raise WebRequestError(HTTPStatus.FORBIDDEN, "Cross-origin local actions are blocked.")

        def _body(self) -> dict[str, Any]:
            content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if content_type != "application/json":
                raise WebRequestError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Local actions require application/json.")
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                raise WebRequestError(HTTPStatus.BAD_REQUEST, "Invalid request length.") from None
            if length < 1 or length > MAX_BODY_BYTES:
                raise WebRequestError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "The local action payload is too large.")
            try:
                value = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise WebRequestError(HTTPStatus.BAD_REQUEST, "The request body is not valid JSON.") from None
            if not isinstance(value, dict):
                raise WebRequestError(HTTPStatus.BAD_REQUEST, "The request body must be a JSON object.")
            return value

        def _feed_date(self, query: dict[str, list[str]]) -> tuple[str, date]:
            feed_id = str((query.get("feed_id") or [""])[0]).strip()
            if not FEED_ID_PATTERN.fullmatch(feed_id):
                raise WebRequestError(HTTPStatus.BAD_REQUEST, "A numeric feed ID is required.")
            archive_date = self._date_value(str((query.get("date") or [""])[0]))
            return feed_id, archive_date

        @staticmethod
        def _date_value(value: str) -> date:
            try:
                return date.fromisoformat(value)
            except ValueError:
                raise WebRequestError(HTTPStatus.BAD_REQUEST, "Use an ISO date such as 2026-07-16.") from None

        @staticmethod
        def _query_int(query: dict[str, list[str]], name: str, default: int) -> int:
            try:
                return int(str((query.get(name) or [default])[0]))
            except (TypeError, ValueError):
                raise WebRequestError(HTTPStatus.BAD_REQUEST, f"{name} must be an integer.") from None

        def _json(self, status: int, value: dict[str, Any]) -> None:
            self._bytes(
                status,
                json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"),
                "application/json; charset=utf-8",
                no_store=True,
            )

        def _bytes(
            self,
            status: int,
            body: bytes,
            content_type: str,
            *,
            cookie: bool = False,
            no_store: bool = False,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; media-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
            )
            if no_store:
                self.send_header("Cache-Control", "no-store")
            else:
                self.send_header("Cache-Control", "public, max-age=3600")
            if cookie:
                self.send_header(
                    "Set-Cookie",
                    f"{SESSION_COOKIE}={state.session_token}; Path=/; HttpOnly; SameSite=Strict",
                )
            self.end_headers()
            self.wfile.write(body)

        def _media(self, path: Path) -> None:
            size = path.stat().st_size
            start = 0
            end = max(0, size - 1)
            status = HTTPStatus.OK
            range_value = self.headers.get("Range", "")
            if range_value:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_value.strip())
                if not match:
                    raise WebRequestError(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, "Unsupported media range.")
                first, last = match.groups()
                if first:
                    start = int(first)
                    end = int(last) if last else end
                elif last:
                    suffix = int(last)
                    start = max(0, size - suffix)
                if start >= size or end < start:
                    raise WebRequestError(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, "Media range is outside the file.")
                end = min(end, size - 1)
                status = HTTPStatus.PARTIAL_CONTENT
            length = max(0, end - start + 1)
            self.send_response(status)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "audio/mpeg")
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Last-Modified", formatdate(path.stat().st_mtime, usegmt=True))
            self.send_header("Cache-Control", "private, no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as handle:
                handle.seek(start)
                remaining = length
                while remaining:
                    chunk = handle.read(min(256 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

    class LocalThreadingHTTPServer(ThreadingHTTPServer):
        address_family = socket.AF_INET6 if host == "::1" else socket.AF_INET
        daemon_threads = True

    server = LocalThreadingHTTPServer((host, port), Handler)
    server.state = state  # type: ignore[attr-defined]
    server.quiet = False  # type: ignore[attr-defined]
    return server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the loopback-only cross-platform radio archive web app."
    )
    parser.add_argument("--output-dir", default="archives")
    parser.add_argument("--database")
    parser.add_argument(
        "--working-dir",
        help=(
            "Directory used for cookies, .env discovery, and worker processes. "
            "Defaults to the current directory."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1", "localhost", "::1"])
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", dest="open_browser")
    return parser


def run_web_app(
    *,
    output_dir: str | Path = "archives",
    database_path: str | Path | None = None,
    working_dir: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = False,
) -> int:
    server = create_server(
        output_dir=output_dir,
        database_path=database_path,
        working_dir=working_dir,
        host=host,
        port=port,
    )
    bound_host, bound_port = server.server_address[:2]
    display_host = "127.0.0.1" if bound_host in {"0.0.0.0", "::"} else bound_host
    url = f"http://{display_host}:{bound_port}/"
    print(f"Radio Archive Intelligence is available at {url}", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def main() -> int:
    arguments = build_parser().parse_args()
    return run_web_app(
        output_dir=arguments.output_dir,
        database_path=arguments.database,
        working_dir=arguments.working_dir,
        host=arguments.host,
        port=arguments.port,
        open_browser=arguments.open_browser,
    )


if __name__ == "__main__":
    raise SystemExit(main())
