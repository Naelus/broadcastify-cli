from __future__ import annotations

import argparse
import hmac
import ipaddress
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

from . import __version__
from .analysis import (
    PROMPT_VERSION,
    WEEKLY_PROMPT_VERSION,
    current_weekly_summary_source_fingerprint,
)
from .area_watch import (
    AREA_PROMPT_VERSION,
    _public_quote,
    current_area_story_source_fingerprint,
)
from .audio import select_incident_evidence_window
from .credential_store import CredentialStoreError, EncryptedCredentialStore
from .lan_sync import (
    LAN_PROTOCOL,
    LanArchiveCatalog,
    LanArchiveReconciler,
    LanArchiveSyncClient,
    LanDiscoveryResponder,
    LanSyncError,
    normalize_peer_url,
    normalize_peer_urls,
)
from .library import (
    build_library_feed_coverage,
    completed_library_catchup_feed_ids,
    require_current_range_evidence,
    scan_local_library,
)
from .quota import (
    DEFAULT_ACCOUNT_PROFILE_ID,
    ArchiveRequestLedger,
    ArchiveRequestBudgetExceeded,
    archive_request_ledger,
    normalize_account_profile_id,
    normalize_archive_request_id,
)
from .storage import AnalysisStore


SESSION_COOKIE = "radio_archive_session"
MAX_BODY_BYTES = 1_048_576
MAX_EVENTS = 500
FEED_ID_PATTERN = re.compile(r"^\d+$")
ZIP_PATTERN = re.compile(r"^\d{5}$")
PROCESSING_DEFAULT_ENVIRONMENT = {
    "hardware_profile": "BROADCASTIFY_DEFAULT_HARDWARE_PROFILE",
    "model": "BROADCASTIFY_DEFAULT_WHISPER_MODEL",
    "asr_engine": "BROADCASTIFY_DEFAULT_ASR_ENGINE",
    "device": "BROADCASTIFY_DEFAULT_DEVICE",
    "diarization_engine": "BROADCASTIFY_DEFAULT_DIARIZATION_ENGINE",
    "diarization_device": "BROADCASTIFY_DEFAULT_DIARIZATION_DEVICE",
}
PROCESSING_DEFAULT_CHOICES = {
    "hardware_profile": {
        "auto",
        "cuda",
        "vulkan",
        "openvino",
        "metal",
        "windowsml",
        "qwen",
        "cpu",
    },
    "model": {
        "turbo",
        "distil-large-v3",
        "large-v3",
        "medium.en",
        "small.en",
        "base.en",
        "tiny.en",
        "qwen3-asr-0.6b-int8",
    },
    "asr_engine": {
        "auto",
        "faster-whisper",
        "whisper.cpp",
        "openvino",
        "windows-ml",
        "qwen3-asr",
    },
    "device": {
        "auto",
        "cpu",
        "cuda",
        "vulkan",
        "metal",
        "openvino-auto",
        "openvino-cpu",
        "openvino-gpu",
        "openvino-npu",
        "gpu",
        "npu",
        "windows-ml",
        "directml",
    },
    "diarization_engine": {"community-1", "sherpa-onnx"},
    "diarization_device": {"auto", "cpu", "cuda"},
}


def validate_bind_host(host: str) -> str:
    """Allow only loopback or an explicitly selected trusted-LAN interface."""

    value = str(host).strip()
    if value == "localhost":
        return value
    try:
        address = ipaddress.ip_address(value)
    except ValueError as exc:
        raise ValueError(
            "The web host must be localhost or a numeric loopback/private/link-local "
            "address; use 0.0.0.0 or :: only when every attached network is trusted."
        ) from exc
    if address.is_multicast or not (
        address.is_loopback
        or address.is_private
        or address.is_link_local
        or address.is_unspecified
    ):
        raise ValueError(
            "The web app refuses a public or multicast bind address. "
            "Choose a loopback or trusted-LAN address."
        )
    return value


def bind_is_loopback(host: str) -> bool:
    value = validate_bind_host(host)
    return value == "localhost" or ipaddress.ip_address(value).is_loopback


def bind_scope(host: str) -> str:
    return "loopback-only" if bind_is_loopback(host) else "trusted-lan"


def format_web_url(host: str, port: int) -> str:
    value = validate_bind_host(host)
    if ":" in value and not value.startswith("["):
        value = f"[{value}]"
    return f"http://{value}:{int(port)}/"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _readiness_environment(working_dir: Path) -> tuple[dict[str, str], str]:
    """Return non-secret setup flags using the worker's environment precedence."""

    values = {key: str(value) for key, value in os.environ.items()}
    loaded_path = ""
    candidates = [working_dir / ".env", working_dir / ".env.accounts"]
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


def _account_cookie_path(working_dir: Path, profile_id: str) -> Path:
    clean_profile_id = normalize_account_profile_id(profile_id)
    if clean_profile_id == DEFAULT_ACCOUNT_PROFILE_ID:
        return working_dir / "cookies.json"
    return working_dir / "account-sessions" / f"{clean_profile_id}.json"


def _account_profile_environment_suffix(profile_id: str) -> str:
    return "".join(
        value if value.isalnum() else "_"
        for value in normalize_account_profile_id(profile_id).upper()
    )


def _account_pool_profiles(
    working_dir: Path,
    credential_store: EncryptedCredentialStore,
) -> dict[str, Any]:
    """Return non-secret configured profiles and their independent ledgers."""

    values, _environment_file = _readiness_environment(working_dir)
    credential_status = credential_store.status(values)
    authorized = _environment_flag(
        values,
        "BROADCASTIFY_AUTHORIZED_ACCOUNT_POOL",
    )
    requested_ids = [DEFAULT_ACCOUNT_PROFILE_ID]
    if authorized:
        requested_ids.extend(
            value.strip()
            for value in str(
                values.get("BROADCASTIFY_ACCOUNT_PROFILES") or ""
            ).split(",")
            if value.strip()
        )
        requested_ids.extend(
            str(value.get("id") or "")
            for value in credential_status.get("broadcastify_profiles", [])
        )
        session_dir = working_dir / "account-sessions"
        if session_dir.is_dir():
            requested_ids.extend(path.stem for path in session_dir.glob("*.json"))

    encrypted_profiles = {
        normalize_account_profile_id(str(value.get("id") or "default")): value
        for value in credential_status.get("broadcastify_profiles", [])
        if value.get("id")
    }
    profiles: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_profile_id in requested_ids:
        try:
            profile_id = normalize_account_profile_id(raw_profile_id)
        except ValueError:
            continue
        if profile_id in seen:
            continue
        seen.add(profile_id)
        cookie_path = _account_cookie_path(working_dir, profile_id)
        if profile_id == DEFAULT_ACCOUNT_PROFILE_ID:
            default_status = dict(credential_status.get("broadcastify") or {})
            configured = bool(default_status.get("configured"))
            username = str(default_status.get("username") or "")
            source = str(default_status.get("source") or "")
            saved = bool(default_status.get("saved"))
            label = "Primary / default"
        else:
            suffix = _account_profile_environment_suffix(profile_id)
            environment_username = str(
                values.get(f"BROADCASTIFY_ACCOUNT_{suffix}_USERNAME") or ""
            ).strip()
            environment_password = str(
                values.get(f"BROADCASTIFY_ACCOUNT_{suffix}_PASSWORD") or ""
            )
            encrypted = dict(encrypted_profiles.get(profile_id) or {})
            configured = bool(
                (environment_username and environment_password) or encrypted
            )
            username = str(encrypted.get("username") or environment_username)
            source = "encrypted-store" if encrypted else (
                "environment" if environment_username and environment_password else ""
            )
            saved = bool(encrypted)
            label = str(encrypted.get("label") or profile_id.replace("_", " ").title())
        session_available = cookie_path.is_file()
        configured = configured or session_available
        if profile_id != DEFAULT_ACCOUNT_PROFILE_ID and not configured:
            continue
        quota = archive_request_ledger(
            base_dir=working_dir,
            account_profile_id=profile_id,
        ).status()
        profiles.append(
            {
                "id": profile_id,
                "label": label,
                "username": username,
                "configured": configured,
                "saved": saved,
                "source": source,
                "session_available": session_available,
                "quota": quota,
            }
        )

    configured_profiles = [value for value in profiles if value["configured"]]
    quota_profiles = configured_profiles or profiles[:1]
    quotas = [dict(value["quota"]) for value in quota_profiles]
    next_values = sorted(
        str(value.get("next_request_at") or "")
        for value in quotas
        if value.get("next_request_at")
    )
    aggregate = {
        "account_profile_id": "automatic" if len(quotas) > 1 else (
            str(quotas[0].get("account_profile_id") or "default") if quotas else "default"
        ),
        "account_count": len(quotas),
        "authorized_pool": authorized,
        "provider_limit": sum(int(value.get("provider_limit") or 0) for value in quotas),
        "automated_limit": sum(int(value.get("automated_limit") or 0) for value in quotas),
        "user_reserve": sum(int(value.get("user_reserve") or 0) for value in quotas),
        "used": sum(int(value.get("used") or 0) for value in quotas),
        "remaining": sum(int(value.get("remaining") or 0) for value in quotas),
        "available": any(bool(value.get("available")) for value in quotas),
        "blocked": bool(quotas) and all(bool(value.get("blocked")) for value in quotas),
        "blocked_reason": "; ".join(
            str(value.get("blocked_reason") or "")
            for value in quotas
            if value.get("blocked_reason")
        ),
        "next_request_at": next_values[0] if next_values else "",
        "next_request_seconds": min(
            (int(value.get("next_request_seconds") or 0) for value in quotas),
            default=0,
        ),
        "instance_id": (
            str(quotas[0].get("instance_id") or "") if len(quotas) == 1 else "pooled"
        ),
        "profiles": profiles,
    }
    return {
        "authorized": authorized,
        "profiles": profiles,
        "configured_profile_ids": [value["id"] for value in configured_profiles],
        "quota": aggregate,
    }


def _select_account_profile(
    working_dir: Path,
    credential_store: EncryptedCredentialStore,
    requested_profile_id: str,
    *,
    excluded_profile_ids: set[str] | None = None,
    allow_unconfigured_requested: bool = False,
) -> str:
    excluded = {
        normalize_account_profile_id(value) for value in excluded_profile_ids or set()
    }
    pool = _account_pool_profiles(working_dir, credential_store)
    requested = str(requested_profile_id or "default").strip().lower()
    if requested != "automatic":
        profile_id = normalize_account_profile_id(requested)
        configured = profile_id in set(pool["configured_profile_ids"])
        if not configured and not allow_unconfigured_requested:
            raise WebRequestError(
                HTTPStatus.BAD_REQUEST,
                f"Account profile {profile_id} is not configured on this app server.",
            )
        return profile_id

    candidates = [
        value
        for value in pool["profiles"]
        if value["configured"] and value["id"] not in excluded
    ]
    if not candidates:
        raise WebRequestError(
            HTTPStatus.TOO_MANY_REQUESTS,
            "No untried authorized account profile is available for this job.",
        )
    candidates.sort(
        key=lambda value: (
            not bool(value["quota"].get("available")),
            -int(value["quota"].get("remaining") or 0),
            int(value["quota"].get("next_request_seconds") or 0),
            str(value["id"]),
        )
    )
    return str(candidates[0]["id"])


def _processing_defaults(values: dict[str, str]) -> dict[str, Any]:
    """Return a validated, non-secret processing preset for this deployment."""

    defaults: dict[str, Any] = {}
    for field, environment_name in PROCESSING_DEFAULT_ENVIRONMENT.items():
        value = str(values.get(environment_name) or "").strip().lower()
        if value and value in PROCESSING_DEFAULT_CHOICES[field]:
            defaults[field] = value
    batch_value = str(
        values.get("BROADCASTIFY_DEFAULT_BATCH_SIZE") or ""
    ).strip()
    if batch_value:
        try:
            batch_size = int(batch_value)
        except ValueError:
            batch_size = 0
        if 1 <= batch_size <= 128:
            defaults["batch_size"] = batch_size
    return defaults if defaults.get("hardware_profile") else {}


def _environment_flag(
    values: dict[str, str],
    name: str,
    *,
    default: bool = False,
) -> bool:
    raw = str(values.get(name) or "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def _apply_automatic_processing_defaults(
    payload: dict[str, Any],
    defaults: dict[str, Any],
) -> dict[str, Any]:
    """Resolve Automatic against a deployment's installed runtime.

    New browsers name their selected hardware profile. The legacy test below
    keeps an already-open pre-upgrade browser safe without overriding an
    explicit custom/runtime selection.
    """

    if not defaults:
        return payload
    selected_profile = str(payload.get("hardware_profile") or "").strip().lower()
    legacy_automatic = (
        not selected_profile
        and str(payload.get("asr_engine") or "auto").strip().lower() == "auto"
        and str(payload.get("device") or "auto").strip().lower() == "auto"
    )
    if selected_profile != "auto" and not legacy_automatic:
        return payload
    resolved = dict(payload)
    for field in (
        "hardware_profile",
        "model",
        "asr_engine",
        "device",
        "diarization_engine",
        "diarization_device",
        "batch_size",
    ):
        if field in defaults:
            resolved[field] = defaults[field]
    return resolved


def _runtime_readiness(state: "WebAppState") -> dict[str, Any]:
    values, environment_file = _readiness_environment(state.working_dir)
    credentials = state.credential_store.status(values)
    account_pool = _account_pool_profiles(
        state.working_dir,
        state.credential_store,
    )
    probe = state.output_dir if state.output_dir.exists() else state.output_dir.parent
    storage_ready = bool(
        probe.is_dir()
        and os.access(probe, os.R_OK | os.W_OK)
        and (not state.output_dir.exists() or state.output_dir.is_dir())
    )
    credentials_configured = any(
        bool(value.get("configured")) for value in account_pool["profiles"]
    )
    saved_session = any(
        bool(value.get("session_available"))
        for value in account_pool["profiles"]
    )
    return {
        "storage_ready": storage_ready,
        "credentials": credentials,
        "account": {
            "configured": credentials_configured or saved_session,
            "credentials_configured": credentials_configured,
            "encrypted_credentials_saved": any(
                bool(value.get("saved")) for value in account_pool["profiles"]
            ),
            "saved_session_available": saved_session,
            "environment_file_available": bool(environment_file),
        },
        "account_pool": account_pool,
    }


class WebRequestError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class JobRecord:
    id: str
    command: str
    account_profile_id: str = DEFAULT_ACCOUNT_PROFILE_ID
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
            "account_profile_id": self.account_profile_id,
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
    """Prepare redacted evidence and safe clip URLs for the browser."""

    rendered: list[dict[str, Any]] = []
    for raw_story in stories:
        story = dict(raw_story)
        location = str(story.get("location") or "")
        references: list[dict[str, Any]] = []
        for raw_reference in raw_story.get("incident_references", []):
            reference = dict(raw_reference)
            safe_quote, quote_changed = _public_quote(
                str(reference.get("quote") or ""),
                location=location,
            )
            reference["quote"] = safe_quote
            reference["quote_redacted"] = bool(
                reference.get("quote_redacted")
            ) or quote_changed
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

    def __init__(
        self,
        output_dir: Path,
        database_path: Path,
        working_dir: Path,
        credential_store: EncryptedCredentialStore | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.database_path = database_path
        self.working_dir = working_dir
        self.credential_store = credential_store or (
            EncryptedCredentialStore.for_working_directory(working_dir)
        )
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.RLock()

    def start(self, command: str, payload: dict[str, Any]) -> dict[str, Any]:
        request_payload = dict(payload)
        requested_profile_id = str(
            request_payload.pop("account_profile_id", "default") or "default"
        )
        excluded_profile_ids = {
            normalize_account_profile_id(str(value))
            for value in request_payload.pop("exclude_account_profile_ids", [])
        }
        account_profile_id = _select_account_profile(
            self.working_dir,
            self.credential_store,
            requested_profile_id,
            excluded_profile_ids=excluded_profile_ids,
            allow_unconfigured_requested=(
                command
                not in {"search", "area-search", "run", "run-scheduled", "run-area"}
                or (
                    command == "authenticate"
                    and bool(request_payload.get("username"))
                    and bool(request_payload.get("password"))
                )
            ),
        )
        arguments, stdin_payload = self._worker_request(command, request_payload)
        with self._lock:
            if any(job.status in {"queued", "running", "canceling"} for job in self._jobs.values()):
                raise WebRequestError(
                    HTTPStatus.CONFLICT,
                    "Another archive or model job is already active. Wait for it or cancel it first.",
                )
            job = JobRecord(
                id=secrets.token_hex(8),
                command=command,
                account_profile_id=account_profile_id,
            )
            self._jobs[job.id] = job
        threading.Thread(
            target=self._run,
            args=(job, arguments, stdin_payload, account_profile_id),
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
        account_profile_id: str,
    ) -> None:
        with self._lock:
            job.status = "running"
            job.started_at = utc_now()
        environment = os.environ.copy()
        environment["BROADCASTIFY_ANALYSIS_DB"] = str(self.database_path)
        environment["BROADCASTIFY_SECURE_ANALYSIS_DB"] = str(
            self.database_path
        )
        environment["BROADCASTIFY_LIBRARY_ROOT"] = str(self.output_dir)
        environment["PYTHONIOENCODING"] = "utf-8"
        environment["PYTHONUTF8"] = "1"
        for name in (
            "BROADCASTIFY_SECURE_USERNAME",
            "BROADCASTIFY_SECURE_PASSWORD",
            "BROADCASTIFY_ACCOUNT_PROFILE",
            "BROADCASTIFY_COOKIE_PATH",
        ):
            environment.pop(name, None)
        environment.update(
            self.credential_store.worker_environment(account_profile_id)
        )
        environment["BROADCASTIFY_ACCOUNT_PROFILE"] = account_profile_id
        environment["BROADCASTIFY_COOKIE_PATH"] = str(
            _account_cookie_path(self.working_dir, account_profile_id)
        )
        environment.setdefault("BROADCASTIFY_GLOBAL_REQUEST_SPACING_SECONDS", "5")
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
        processing_commands = {
            "asr-self-test",
            "continue-local",
            "diagnostics-selected",
            "diarization-self-test",
            "prepare-asr-model",
            "profile-self-test",
            "run",
        }
        if command in processing_commands:
            values, _environment_file = _readiness_environment(self.working_dir)
            payload = _apply_automatic_processing_defaults(
                payload,
                _processing_defaults(values),
            )
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
            "analysis-self-test",
            "analyze-day",
            "asr-self-test",
            "prepare-asr-model",
            "diarization-self-test",
            "diagnostics-selected",
            "profile-self-test",
            "ask",
            "authenticate",
            "continue-local",
            "run",
            "run-scheduled",
            "run-area",
            "save-area-profile",
            "summarize-area",
            "summarize-week",
        }
        if command not in stdin_commands:
            raise WebRequestError(HTTPStatus.BAD_REQUEST, "That local job type is not supported.")
        if command in {"run", "continue-local", "analyze-day", "ask"}:
            payload["output_dir"] = str(self.output_dir)
        if command == "run":
            # The browser UI keeps a single upstream downloader, preserves
            # source blocks, and asks trusted-LAN peers before website access.
            payload["download_jobs"] = 1
            payload["keep_originals"] = True
            payload.setdefault("lan_sync_enabled", True)
            payload.setdefault("lan_discovery_enabled", True)
            if payload.get("diarize"):
                payload["combine"] = True
                payload["transcribe"] = True
        if command == "run-scheduled":
            job_payload = dict(payload.get("job") or {})
            values, _environment_file = _readiness_environment(self.working_dir)
            job_payload = _apply_automatic_processing_defaults(
                job_payload,
                _processing_defaults(values),
            )
            job_payload["output_dir"] = str(self.output_dir)
            job_payload["download_jobs"] = 1
            job_payload["keep_originals"] = True
            job_payload.setdefault("lan_sync_enabled", True)
            job_payload.setdefault("lan_discovery_enabled", True)
            if job_payload.get("diarize"):
                job_payload["combine"] = True
                job_payload["transcribe"] = True
            payload["job"] = job_payload
        if command == "run-area":
            job_payload = dict(payload.get("job") or {})
            values, _environment_file = _readiness_environment(self.working_dir)
            job_payload = _apply_automatic_processing_defaults(
                job_payload,
                _processing_defaults(values),
            )
            job_payload["output_dir"] = str(self.output_dir)
            job_payload["download_jobs"] = 1
            job_payload["keep_originals"] = True
            job_payload.setdefault("lan_sync_enabled", True)
            job_payload.setdefault("lan_discovery_enabled", True)
            if job_payload.get("diarize"):
                job_payload["combine"] = True
                job_payload["transcribe"] = True
            payload["job"] = job_payload
        return [command], payload


class FeedScheduleCoordinator:
    """Run persisted feed schedules while the Web/NAS service is alive."""

    def __init__(
        self,
        jobs: JobManager,
        database_path: Path,
        working_dir: Path,
        *,
        poll_seconds: float = 30.0,
    ) -> None:
        self.jobs = jobs
        self.database_path = database_path
        self.working_dir = working_dir
        self.poll_seconds = max(0.05, float(poll_seconds))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._active: tuple[str, dict[str, Any], set[str], str] | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        with AnalysisStore(self.database_path) as store:
            store.recover_feed_schedules()
        self._thread = threading.Thread(
            target=self._run,
            name="radio-feed-schedules",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def status(self) -> dict[str, Any]:
        """Return a bounded, non-secret view for trusted coordinated clients."""

        with AnalysisStore(self.database_path) as store:
            schedules = store.list_feed_schedules()
        active = self._active
        active_status: dict[str, Any] | None = None
        if active is not None:
            job_id, schedule, _attempted_profile_ids, phase = active
            try:
                snapshot = self.jobs.get(job_id)
            except WebRequestError:
                snapshot = {}
            latest_stage: dict[str, Any] = {}
            for event in reversed(list(snapshot.get("events") or [])):
                if not isinstance(event, dict):
                    continue
                stage = str(event.get("stage") or "").strip().lower()
                if not re.fullmatch(r"[a-z0-9_-]{1,40}", stage):
                    continue
                message = str(event.get("message") or "")
                archive_match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", message)
                latest_stage = {
                    "stage": stage,
                    "archive_date": archive_match.group(0) if archive_match else "",
                    "current": max(0, int(event.get("current") or 0)),
                    "total": max(0, int(event.get("total") or 0)),
                    "updated_at": str(event.get("received_at") or "")[:40],
                }
                break
            active_status = {
                "feed_id": str(schedule.get("feed_id") or "")[:40],
                "feed_name": str(schedule.get("feed_name") or "")[:200],
                "phase": phase,
                "status": str(snapshot.get("status") or "")[:40],
                "account_profile_id": normalize_account_profile_id(
                    str(snapshot.get("account_profile_id") or "default")
                ),
                **latest_stage,
            }
        return {
            "active": active_status,
            "schedules": [
                {
                    "id": int(schedule.get("id") or 0),
                    "feed_id": str(schedule.get("feed_id") or "")[:40],
                    "feed_name": str(schedule.get("feed_name") or "")[:200],
                    "state": str(schedule.get("state") or "")[:40],
                    "enabled": bool(schedule.get("enabled")),
                    "account_profile_id": str(
                        schedule.get("account_profile_id") or "automatic"
                    )[:64],
                    "next_run_at": str(schedule.get("next_run_at") or "")[:40],
                    "last_started_at": str(
                        schedule.get("last_started_at") or ""
                    )[:40],
                    "message": " ".join(
                        str(schedule.get("message") or "").split()
                    )[:240],
                }
                for schedule in schedules[:100]
            ],
        }

    def check_once(self) -> None:
        if self._active is not None:
            job_id, schedule, attempted_profile_ids, phase = self._active
            snapshot = self.jobs.get(job_id)
            if snapshot["status"] in {"queued", "running", "canceling"}:
                return
            selected_profile_id = normalize_account_profile_id(
                str(snapshot.get("account_profile_id") or "default")
            )
            attempted_profile_ids.add(selected_profile_id)
            automatic_pool = str(
                schedule.get("account_profile_id") or "automatic"
            ) == "automatic"
            if snapshot["status"] == "completed":
                event = snapshot.get("result") or {}
                result = event.get("result") if isinstance(event, dict) else {}
                result = result if isinstance(result, dict) else {}
                limited = bool(result.get("download_limited"))
                incomplete = bool(result.get("missing_days")) or bool(
                    result.get("pending_processing_days")
                )
                if limited and automatic_pool:
                    next_phase = (
                        "acquisition"
                        if self._uses_multi_account_acquisition(schedule)
                        else "full"
                    )
                    if self._try_next_account(
                        schedule,
                        attempted_profile_ids,
                        phase=next_phase,
                    ):
                        return
                if phase == "acquisition":
                    try:
                        job = self._start_schedule_job(
                            schedule,
                            set(),
                            account_profile_id=selected_profile_id,
                        )
                    except WebRequestError as exc:
                        with AnalysisStore(self.database_path) as store:
                            store.finish_feed_schedule(
                                int(schedule["id"]),
                                due_date=str(schedule["due_date"]),
                                status="deferred",
                                message=(
                                    "Archive acquisition was checkpointed, but local "
                                    f"processing could not start: {exc}"
                                ),
                                next_request_at="",
                            )
                        self._active = None
                        return
                    self._active = (
                        str(job["id"]),
                        schedule,
                        set(attempted_profile_ids),
                        "processing",
                    )
                    return
                quota = self._quota_status()
                status = (
                    "waiting_quota"
                    if limited
                    else "deferred"
                    if incomplete
                    else "complete"
                )
                message = (
                    "Waiting for the next rolling archive-request slot; retained "
                    "trusted-LAN/local work will be rechecked within five minutes."
                    if limited
                    else "Some archive days were deferred; retrying retained work shortly."
                    if incomplete
                    else "Scheduled feed run completed."
                )
                next_request_at = str(quota.get("next_request_at") or "") if limited else ""
            else:
                message = str(snapshot.get("error") or "Scheduled feed job failed.")
                if (
                    automatic_pool
                    and snapshot["status"] == "failed"
                    and re.search(
                        r"auth|credential|forbidden|login|premium|unauthori[sz]ed",
                        message,
                        flags=re.IGNORECASE,
                    )
                    and self._try_next_account(
                        schedule,
                        attempted_profile_ids,
                        phase=(
                            "acquisition"
                            if self._uses_multi_account_acquisition(schedule)
                            else "full"
                        ),
                    )
                ):
                    return
                status = "canceled" if snapshot["status"] == "canceled" else "failed"
                next_request_at = ""
            with AnalysisStore(self.database_path) as store:
                store.finish_feed_schedule(
                    int(schedule["id"]),
                    due_date=str(schedule["due_date"]),
                    status=status,
                    message=message,
                    next_request_at=next_request_at,
                )
            self._active = None
            return

        with AnalysisStore(self.database_path) as store:
            schedule = store.claim_due_feed_schedule(
                output_dir=self.jobs.output_dir,
            )
        if schedule is None:
            return
        split_acquisition = self._uses_multi_account_acquisition(schedule)
        try:
            job = self._start_schedule_job(
                schedule,
                set(),
                acquisition_only=split_acquisition,
            )
        except WebRequestError as exc:
            with AnalysisStore(self.database_path) as store:
                store.finish_feed_schedule(
                    int(schedule["id"]),
                    due_date=str(schedule["due_date"]),
                    status=(
                        "waiting_quota"
                        if exc.status == HTTPStatus.TOO_MANY_REQUESTS
                        else "deferred"
                    ),
                    message=str(exc),
                    next_request_at=str(
                        self._quota_status().get("next_request_at") or ""
                    ),
                )
            return
        self._active = (
            str(job["id"]),
            schedule,
            set(),
            "acquisition" if split_acquisition else "full",
        )

    def _start_schedule_job(
        self,
        schedule: dict[str, Any],
        attempted_profile_ids: set[str],
        *,
        acquisition_only: bool = False,
        account_profile_id: str = "",
    ) -> dict[str, Any]:
        job_payload = dict(schedule["job"])
        analyze = bool(schedule["analyze"])
        if acquisition_only:
            job_payload.update(
                {
                    "combine": False,
                    "transcribe": False,
                    "diarize": False,
                }
            )
            analyze = False
        return self.jobs.start(
            "run-scheduled",
            {
                "job": job_payload,
                "analyze": analyze,
                "account_profile_id": account_profile_id
                or str(schedule.get("account_profile_id") or "automatic"),
                "exclude_account_profile_ids": (
                    [] if account_profile_id else sorted(attempted_profile_ids)
                ),
            },
        )

    def _try_next_account(
        self,
        schedule: dict[str, Any],
        attempted_profile_ids: set[str],
        *,
        phase: str,
    ) -> bool:
        try:
            job = self._start_schedule_job(
                schedule,
                attempted_profile_ids,
                acquisition_only=phase == "acquisition",
            )
        except WebRequestError:
            return False
        self._active = (
            str(job["id"]),
            schedule,
            set(attempted_profile_ids),
            phase,
        )
        return True

    def _uses_multi_account_acquisition(
        self,
        schedule: dict[str, Any],
    ) -> bool:
        if str(schedule.get("account_profile_id") or "automatic") != "automatic":
            return False
        credential_store = getattr(self.jobs, "credential_store", None)
        if credential_store is None:
            return False
        pool = _account_pool_profiles(
            self.working_dir,
            credential_store,
        )
        return bool(pool["authorized"]) and len(pool["configured_profile_ids"]) > 1

    def _quota_status(self) -> dict[str, Any]:
        credential_store = getattr(self.jobs, "credential_store", None)
        if credential_store is None:
            return archive_request_ledger(base_dir=self.working_dir).status()
        return _account_pool_profiles(
            self.working_dir,
            credential_store,
        )["quota"]

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.check_once()
            except Exception:
                # Keep the long-running Web/NAS service alive. The claimed
                # schedule lease and startup recovery prevent duplicate work.
                pass
            self._stop.wait(self.poll_seconds)


@dataclass(frozen=True)
class WebAppState:
    output_dir: Path
    database_path: Path
    working_dir: Path
    static_dir: Path
    session_token: str
    jobs: JobManager
    scheduler: FeedScheduleCoordinator
    bind_host: str
    access_scope: str
    loopback_only: bool
    lan_catalog: LanArchiveCatalog
    lan_reconciler: LanArchiveReconciler
    credential_store: EncryptedCredentialStore


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
    with AnalysisStore(state.database_path) as store:
        schedules = store.list_feed_schedules()
        catchups = store.list_library_catchups()
        for feed_id in completed_library_catchup_feed_ids(days, catchups):
            store.delete_library_catchup(feed_id)
        if catchups:
            catchups = store.list_library_catchups()
    return {
        "days": days,
        "feed_coverage": build_library_feed_coverage(
            days,
            schedules,
            catchups,
        ),
        "catchups": catchups,
        "summary": {
            "feed_count": len({value["feed_id"] for value in days}),
            "day_count": len(days),
            "complete_count": sum(bool(value["is_complete"]) for value in days),
            "attention_count": sum(not bool(value["is_complete"]) for value in days),
            "storage_bytes": sum(int(value["storage_bytes"]) for value in days),
            "working_storage_bytes": sum(
                int(value["working_storage_bytes"]) for value in days
            ),
        },
    }


def _compact_incident(value: dict[str, Any]) -> dict[str, Any]:
    evidence = list(value.get("evidence") or [])
    quote_parts: list[str] = []
    evidence_start, evidence_end = select_incident_evidence_window(value)
    for item in evidence:
        if isinstance(item, dict):
            try:
                segment_start = float(item.get("start_seconds", 0.0))
                segment_end = float(item.get("end_seconds", segment_start))
            except (TypeError, ValueError):
                continue
            text = str(item.get("text") or item.get("quote") or "").strip()
            if (
                text
                and segment_end >= evidence_start
                and segment_start <= evidence_end
                and text not in quote_parts
            ):
                quote_parts.append(text)
    quote_text, quote_redacted = _public_quote(
        " ".join(quote_parts),
        location=str(value.get("location") or ""),
    )
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
        "quote_redacted": quote_redacted,
    }


def _local_day_state(
    state: WebAppState,
    feed_id: str,
    archive_date: date,
) -> dict[str, Any]:
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
    return day_state


def _day_payload(state: WebAppState, feed_id: str, archive_date: date) -> dict[str, Any]:
    day_state = _local_day_state(state, feed_id, archive_date)
    summary = ""
    incidents: list[dict[str, Any]] = []
    if day_state["has_analysis"]:
        with AnalysisStore(state.database_path) as store:
            stored_day = store.get_day(feed_id, archive_date)
            if stored_day is not None:
                stored_summary = store.get_latest_daily_summary(
                    int(stored_day["id"])
                )
                analysis_current = bool(
                    stored_summary
                    and str(stored_summary["prompt_version"])
                    == PROMPT_VERSION
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
    day = _local_day_state(state, feed_id, archive_date)
    if not day["has_transcript"]:
        return {
            "segments": [],
            "offset": offset,
            "limit": limit,
            "total": 0,
            "has_more": False,
        }
    segments: list[dict[str, Any]] = []
    if day["has_imported_transcript"]:
        with AnalysisStore(state.database_path) as store:
            stored_day = store.get_day(feed_id, archive_date)
            if stored_day is not None:
                segments = store.get_segments(int(stored_day["id"]))
    if not segments:
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
    credential_store_path: str | Path | None = None,
    background_sync_enabled: bool = False,
) -> ThreadingHTTPServer:
    host = validate_bind_host(host)
    access_scope = bind_scope(host)
    loopback_only = access_scope == "loopback-only"
    root = Path(output_dir).expanduser().resolve()
    database = (
        Path(database_path).expanduser().resolve()
        if database_path
        else (root / "broadcastify-analysis.sqlite3").resolve()
    )
    work = Path(working_dir or Path.cwd()).resolve()
    static_dir = Path(__file__).with_name("web_static").resolve()
    token = secrets.token_urlsafe(32)
    readiness_values, _environment_file = _readiness_environment(work)
    configured_advertisement = str(
        readiness_values.get("BROADCASTIFY_LAN_ADVERTISE_URL") or ""
    ).strip()
    if configured_advertisement:
        configured_advertisement = normalize_peer_url(
            configured_advertisement
        )
    lan_catalog = LanArchiveCatalog(
        root,
        enabled=_environment_flag(
            readiness_values,
            "BROADCASTIFY_LAN_SHARING",
            default=False,
        ),
        sync_key=str(readiness_values.get("BROADCASTIFY_LAN_SYNC_KEY") or ""),
        peer_urls=normalize_peer_urls(
            readiness_values.get("BROADCASTIFY_LAN_PEERS"),
        ),
        queue_enabled=_environment_flag(
            readiness_values,
            "BROADCASTIFY_LAN_QUEUE_ENABLED",
            default=True,
        ),
    )
    lan_discovery_enabled = _environment_flag(
        readiness_values,
        "BROADCASTIFY_LAN_DISCOVERY_ENABLED",
        default=True,
    )
    try:
        lan_reconcile_seconds = float(
            readiness_values.get("BROADCASTIFY_LAN_RECONCILE_SECONDS")
            or 5 * 60.0
        )
    except ValueError:
        lan_reconcile_seconds = 5 * 60.0
    lan_reconciler = LanArchiveReconciler(
        root,
        LanArchiveSyncClient(
            enabled=lan_catalog.enabled,
            peer_urls=lan_catalog.peer_urls,
            discovery_enabled=lan_discovery_enabled,
            sync_key=lan_catalog.sync_key,
            queue_enabled=False,
        ),
        enabled=(
            background_sync_enabled
            and _environment_flag(
                readiness_values,
                "BROADCASTIFY_LAN_BACKGROUND_SYNC",
                default=True,
            )
        ),
        poll_seconds=min(60 * 60.0, max(30.0, lan_reconcile_seconds)),
    )
    credential_store = (
        EncryptedCredentialStore(credential_store_path)
        if credential_store_path is not None
        else EncryptedCredentialStore.for_working_directory(work)
    )
    jobs = JobManager(root, database, work, credential_store)
    scheduler = FeedScheduleCoordinator(jobs, database, work)
    state = WebAppState(
        output_dir=root,
        database_path=database,
        working_dir=work,
        static_dir=static_dir,
        session_token=token,
        jobs=jobs,
        scheduler=scheduler,
        bind_host=host,
        access_scope=access_scope,
        loopback_only=loopback_only,
        lan_catalog=lan_catalog,
        lan_reconciler=lan_reconciler,
        credential_store=credential_store,
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
                parsed = urlparse(self.path)
                if parsed.path.startswith("/api/lan/v1/acquisition/"):
                    self._require_lan_access()
                    self._post_lan_acquisition(parsed.path)
                    return
                if parsed.path.startswith("/api/lan/v1/quota/"):
                    self._require_lan_access()
                    self._post_lan_quota(parsed.path)
                    return
                if parsed.path.startswith("/api/lan/v1/processing/"):
                    self._require_lan_access()
                    self._post_lan_processing(parsed.path)
                    return
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
                self._json(
                    HTTPStatus.OK,
                    {"status": "ok", "scope": state.access_scope},
                )
                return
            if parsed.path == "/api/lan/v1/info":
                self._require_lan_access()
                info = state.lan_catalog.info()
                info["scheduler"] = state.scheduler.status()
                info["reconciliation"] = state.lan_reconciler.status()
                self._json(HTTPStatus.OK, info)
                return
            if parsed.path == "/api/lan/v1/feed-days":
                self._require_lan_access()
                query = parse_qs(parsed.query)
                feed_id = str((query.get("feed_id") or [""])[0]).strip()
                try:
                    dates = state.lan_catalog.feed_dates(feed_id)
                except LanSyncError as exc:
                    raise WebRequestError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
                self._json(
                    HTTPStatus.OK,
                    {
                        "protocol": LAN_PROTOCOL,
                        "node_id": state.lan_catalog.node_id,
                        "feed_id": feed_id,
                        "dates": [value.isoformat() for value in dates],
                        "peers": list(state.lan_catalog.peer_urls),
                    },
                )
                return
            if parsed.path == "/api/lan/v1/blocks":
                self._require_lan_access()
                query = parse_qs(parsed.query)
                feed_id, archive_date = self._feed_date(query)
                try:
                    blocks = state.lan_catalog.inventory(feed_id, archive_date)
                    complete, completion_blocks = (
                        state.lan_catalog.completion_inventory(
                            feed_id,
                            archive_date,
                        )
                    )
                except LanSyncError as exc:
                    raise WebRequestError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
                self._json(
                    HTTPStatus.OK,
                    {
                        "protocol": LAN_PROTOCOL,
                        "node_id": state.lan_catalog.node_id,
                        "feed_id": feed_id,
                        "archive_date": archive_date.isoformat(),
                        "blocks": [block.to_dict() for block in blocks],
                        "complete": complete,
                        "completion_blocks": [
                            block.to_dict() for block in completion_blocks
                        ],
                        "peers": list(state.lan_catalog.peer_urls),
                    },
                )
                return
            if parsed.path == "/api/lan/v1/acquisition":
                self._require_lan_access()
                if not state.lan_catalog.acquisition_queue.enabled:
                    raise WebRequestError(
                        HTTPStatus.NOT_FOUND,
                        "LAN acquisition coordination is not enabled.",
                    )
                query = parse_qs(parsed.query)
                feed_id, archive_date = self._feed_date(query)
                quota_scope = str(
                    (query.get("quota_scope") or ["default"])[0]
                ).strip()
                try:
                    value = state.lan_catalog.acquisition_queue.status(
                        quota_scope,
                        feed_id,
                        archive_date,
                    )
                except LanSyncError as exc:
                    raise WebRequestError(
                        HTTPStatus.BAD_REQUEST,
                        str(exc),
                    ) from exc
                self._json(HTTPStatus.OK, value)
                return
            if parsed.path == "/api/lan/v1/quota":
                self._require_lan_access()
                query = parse_qs(parsed.query)
                try:
                    profile_id = normalize_account_profile_id(
                        str(
                            (query.get("account_profile_id") or ["default"])[0]
                        )
                    )
                except ValueError as exc:
                    raise WebRequestError(
                        HTTPStatus.BAD_REQUEST,
                        str(exc),
                    ) from exc
                status = ArchiveRequestLedger(
                    base_dir=state.working_dir,
                    account_profile_id=profile_id,
                ).status()
                self._json(
                    HTTPStatus.OK,
                    {"protocol": LAN_PROTOCOL, **status},
                )
                return
            if parsed.path == "/api/lan/v1/transcript-fingerprints":
                self._require_lan_access()
                query = parse_qs(parsed.query)
                feed_id, archive_date = self._feed_date(query)
                try:
                    fingerprints = state.lan_catalog.transcript_fingerprints(
                        feed_id,
                        archive_date,
                    )
                except LanSyncError as exc:
                    raise WebRequestError(
                        HTTPStatus.BAD_REQUEST,
                        str(exc),
                    ) from exc
                self._json(
                    HTTPStatus.OK,
                    {
                        "protocol": LAN_PROTOCOL,
                        "node_id": state.lan_catalog.node_id,
                        "feed_id": feed_id,
                        "archive_date": archive_date.isoformat(),
                        "processing_fingerprints": list(fingerprints),
                        "peers": list(state.lan_catalog.peer_urls),
                    },
                )
                return
            if parsed.path == "/api/lan/v1/transcripts":
                self._require_lan_access()
                query = parse_qs(parsed.query)
                feed_id, archive_date = self._feed_date(query)
                fingerprint = str(
                    (query.get("processing_fingerprint") or [""])[0]
                ).strip()
                try:
                    artifacts = state.lan_catalog.transcript_inventory(
                        feed_id,
                        archive_date,
                        fingerprint,
                    )
                except LanSyncError as exc:
                    raise WebRequestError(
                        HTTPStatus.BAD_REQUEST,
                        str(exc),
                    ) from exc
                self._json(
                    HTTPStatus.OK,
                    {
                        "protocol": LAN_PROTOCOL,
                        "node_id": state.lan_catalog.node_id,
                        "feed_id": feed_id,
                        "archive_date": archive_date.isoformat(),
                        "processing_fingerprint": fingerprint,
                        "artifacts": [value.to_dict() for value in artifacts],
                        "peers": list(state.lan_catalog.peer_urls),
                    },
                )
                return
            if parsed.path == "/api/lan/v1/processing":
                self._require_lan_access()
                if not state.lan_catalog.processing_queue.enabled:
                    raise WebRequestError(
                        HTTPStatus.NOT_FOUND,
                        "LAN processing coordination is not enabled.",
                    )
                query = parse_qs(parsed.query)
                feed_id, archive_date = self._feed_date(query)
                fingerprint = str(
                    (query.get("processing_fingerprint") or [""])[0]
                ).strip()
                try:
                    value = state.lan_catalog.processing_queue.status(
                        fingerprint,
                        feed_id,
                        archive_date,
                    )
                except LanSyncError as exc:
                    raise WebRequestError(
                        HTTPStatus.BAD_REQUEST,
                        str(exc),
                    ) from exc
                self._json(HTTPStatus.OK, value)
                return
            if parsed.path.startswith("/api/lan/v1/blocks/"):
                self._require_lan_access()
                parts = parsed.path.removeprefix("/api/lan/v1/blocks/").split("/")
                if len(parts) != 3:
                    raise WebRequestError(
                        HTTPStatus.NOT_FOUND,
                        "Archive block not found.",
                    )
                feed_id, date_value, filename = (unquote(value) for value in parts)
                if not FEED_ID_PATTERN.fullmatch(feed_id):
                    raise WebRequestError(
                        HTTPStatus.BAD_REQUEST,
                        "A numeric feed ID is required.",
                    )
                archive_date = self._date_value(date_value)
                try:
                    path, block = state.lan_catalog.resolve_block(
                        feed_id,
                        archive_date,
                        filename,
                    )
                except FileNotFoundError:
                    raise WebRequestError(
                        HTTPStatus.NOT_FOUND,
                        "Archive block not found.",
                    ) from None
                except LanSyncError as exc:
                    raise WebRequestError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
                self._lan_block(path, block.sha256)
                return
            if parsed.path.startswith("/api/lan/v1/transcripts/"):
                self._require_lan_access()
                parts = parsed.path.removeprefix(
                    "/api/lan/v1/transcripts/"
                ).split("/")
                if len(parts) != 4:
                    raise WebRequestError(
                        HTTPStatus.NOT_FOUND,
                        "Transcript artifact not found.",
                    )
                feed_id, date_value, fingerprint, filename = (
                    unquote(value) for value in parts
                )
                if not FEED_ID_PATTERN.fullmatch(feed_id):
                    raise WebRequestError(
                        HTTPStatus.BAD_REQUEST,
                        "A numeric feed ID is required.",
                    )
                archive_date = self._date_value(date_value)
                try:
                    path, artifact = state.lan_catalog.resolve_transcript_artifact(
                        feed_id,
                        archive_date,
                        fingerprint,
                        filename,
                    )
                except FileNotFoundError:
                    raise WebRequestError(
                        HTTPStatus.NOT_FOUND,
                        "Transcript artifact not found.",
                    ) from None
                except LanSyncError as exc:
                    raise WebRequestError(
                        HTTPStatus.BAD_REQUEST,
                        str(exc),
                    ) from exc
                self._lan_artifact(path, artifact.sha256)
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
            if parsed.path == "/api/system-activity":
                account_pool = _account_pool_profiles(
                    state.working_dir,
                    state.credential_store,
                )
                info = state.lan_catalog.info()
                self._json(
                    HTTPStatus.OK,
                    {
                        **info,
                        "scheduler": state.scheduler.status(),
                        "reconciliation": state.lan_reconciler.status(),
                        "account_pool": account_pool,
                    },
                )
                return
            if parsed.path == "/api/bootstrap":
                library = _library_payload(state)
                readiness_values, _environment_file = _readiness_environment(
                    state.working_dir
                )
                account_pool = _account_pool_profiles(
                    state.working_dir,
                    state.credential_store,
                )
                with AnalysisStore(state.database_path) as store:
                    profiles = store.list_area_profiles()
                    area_runs = store.list_area_acquisition_runs(limit=20)
                    schedules = store.list_feed_schedules()
                self._json(
                    HTTPStatus.OK,
                    {
                        **library,
                        "profiles": profiles,
                        "area_runs": area_runs,
                        "schedules": schedules,
                        "runtime": {
                            "version": __version__,
                            "source_commit": str(
                                os.getenv("BROADCASTIFY_SOURCE_COMMIT") or ""
                            ).strip(),
                            "platform": platform.system(),
                            "platform_release": platform.release(),
                            "python": platform.python_version(),
                            "output_dir": str(state.output_dir),
                            "database_path": str(state.database_path),
                            "loopback_only": state.loopback_only,
                            "access_scope": state.access_scope,
                            "bind_host": state.bind_host,
                            "archive_quota": account_pool["quota"],
                            "processing_defaults": _processing_defaults(
                                readiness_values
                            ),
                            "lan_sync": {
                                "sharing_enabled": state.lan_catalog.enabled,
                                "key_required": bool(state.lan_catalog.sync_key),
                                "configured_peer_count": len(
                                    state.lan_catalog.peer_urls
                                ),
                                "discovery_available": (
                                    state.lan_catalog.discovery_available
                                ),
                                "discovery_error": (
                                    state.lan_catalog.discovery_error
                                ),
                                "acquisition_queue_available": bool(
                                    state.lan_catalog.acquisition_queue.enabled
                                ),
                                "processing_queue_available": bool(
                                    state.lan_catalog.processing_queue.enabled
                                ),
                            },
                            **_runtime_readiness(state),
                        },
                    },
                )
                return
            if parsed.path == "/api/credentials":
                readiness_values, _environment_file = _readiness_environment(
                    state.working_dir
                )
                self._json(
                    HTTPStatus.OK,
                    {
                        "credentials": state.credential_store.status(
                            readiness_values
                        )
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
                        profile = store.get_area_profile(profile_name)
                        if (
                            str(coverage.get("incident_prompt_version") or "")
                            != PROMPT_VERSION
                            or profile is None
                        ):
                            stale = True
                        else:
                            try:
                                require_current_range_evidence(
                                    store,
                                    [
                                        str(value)
                                        for value in profile["feed_ids"]
                                    ],
                                    date.fromisoformat(str(row["start_date"])),
                                    date.fromisoformat(str(row["end_date"])),
                                    require_analysis=True,
                                    purpose="Saved area summary",
                                )
                            except ValueError:
                                stale = True
                            else:
                                current_fingerprint = (
                                    current_area_story_source_fingerprint(
                                        store,
                                        profile,
                                        date.fromisoformat(
                                            str(row["start_date"])
                                        ),
                                        date.fromisoformat(
                                            str(row["end_date"])
                                        ),
                                    )
                                )
                                if (
                                    str(row["source_fingerprint"])
                                    != current_fingerprint
                                ):
                                    stale = True
                                else:
                                    result = {
                                        "profile_name": str(
                                            row["profile_name"]
                                        ),
                                        "start_date": str(row["start_date"]),
                                        "end_date": str(row["end_date"]),
                                        "summary": str(row["summary"]),
                                        "stories": _area_stories_for_web(
                                            state.output_dir,
                                            json.loads(
                                                str(row["stories_json"])
                                            ),
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
                    stale = False
                    if row is not None:
                        try:
                            require_current_range_evidence(
                                store,
                                [feed_id],
                                week_start,
                                week_ending,
                                require_analysis=True,
                                purpose="Saved weekly summary",
                            )
                        except ValueError:
                            row = None
                            stale = True
                        else:
                            current_fingerprint = (
                                current_weekly_summary_source_fingerprint(
                                    store,
                                    feed_id,
                                    week_start,
                                    week_ending,
                                )
                            )
                            if (
                                str(row["source_fingerprint"])
                                != current_fingerprint
                            ):
                                row = None
                                stale = True
                result = None
                if row is not None:
                    result = dict(row)
                    result["notable_incident_ids"] = json.loads(
                        str(result.pop("notable_incident_ids_json") or "[]")
                    )
                self._json(
                    HTTPStatus.OK,
                    {"result": result, "stale": stale},
                )
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
            if parsed.path == "/api/credentials":
                body = self._body()
                kind = str(body.get("kind") or "").strip().lower()
                action = str(body.get("action") or "save").strip().lower()
                try:
                    if kind == "broadcastify":
                        profile_id = str(body.get("profile_id") or "default")
                        if action == "clear":
                            state.credential_store.clear_broadcastify(profile_id)
                        elif action == "save":
                            state.credential_store.save_broadcastify(
                                str(body.get("username") or ""),
                                str(body.get("secret") or ""),
                                profile_id=profile_id,
                                label=str(body.get("label") or ""),
                            )
                        else:
                            raise ValueError("Unknown credential action.")
                    elif kind == "huggingface":
                        if action == "clear":
                            state.credential_store.clear_huggingface()
                        elif action == "save":
                            state.credential_store.save_huggingface(
                                str(body.get("secret") or "")
                            )
                        else:
                            raise ValueError("Unknown credential action.")
                    else:
                        raise ValueError("Choose Broadcastify or Hugging Face credentials.")
                    readiness_values, _environment_file = _readiness_environment(
                        state.working_dir
                    )
                    self._json(
                        HTTPStatus.OK,
                        {
                            "credentials": state.credential_store.status(
                                readiness_values
                            )
                        },
                    )
                except (CredentialStoreError, ValueError) as exc:
                    raise WebRequestError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
                return
            if parsed.path == "/api/jobs":
                body = self._body()
                command = str(body.get("command") or "").strip()
                payload = body.get("payload") or {}
                if not isinstance(payload, dict):
                    raise WebRequestError(HTTPStatus.BAD_REQUEST, "The job payload must be a JSON object.")
                self._json(HTTPStatus.ACCEPTED, state.jobs.start(command, payload))
                return
            if parsed.path == "/api/schedules":
                body = self._body()
                requested_profile_id = str(
                    body.get("account_profile_id") or "automatic"
                ).strip().lower()
                if requested_profile_id != "automatic":
                    _select_account_profile(
                        state.working_dir,
                        state.credential_store,
                        requested_profile_id,
                    )
                with AnalysisStore(state.database_path) as store:
                    schedule = store.save_feed_schedule(body)
                self._json(HTTPStatus.OK, {"schedule": schedule})
                return
            if parsed.path == "/api/catchups":
                body = self._body()
                action = str(body.get("action") or "save").strip().lower()
                try:
                    with AnalysisStore(state.database_path) as store:
                        if action == "save":
                            catchup = store.save_library_catchup(
                                {
                                    **body,
                                    "through_current": True,
                                    "end_date": "",
                                }
                            )
                            self._json(
                                HTTPStatus.OK,
                                {"catchup": catchup},
                            )
                        elif action == "clear":
                            deleted = store.delete_library_catchup(
                                str(body.get("feed_id") or "")
                            )
                            self._json(
                                HTTPStatus.OK,
                                {"deleted": deleted},
                            )
                        else:
                            raise ValueError("Unknown catch-up action.")
                except ValueError as exc:
                    raise WebRequestError(
                        HTTPStatus.BAD_REQUEST,
                        str(exc),
                    ) from exc
                return
            if parsed.path.startswith("/api/schedules/") and parsed.path.endswith("/delete"):
                value = parsed.path.removeprefix("/api/schedules/").removesuffix("/delete")
                try:
                    schedule_id = int(value)
                except ValueError:
                    raise WebRequestError(HTTPStatus.BAD_REQUEST, "A numeric schedule ID is required.") from None
                with AnalysisStore(state.database_path) as store:
                    deleted = store.delete_feed_schedule(schedule_id)
                if not deleted:
                    raise WebRequestError(HTTPStatus.NOT_FOUND, "That feed schedule was not found.")
                self._json(HTTPStatus.OK, {"deleted": True})
                return
            if parsed.path.startswith("/api/jobs/") and parsed.path.endswith("/cancel"):
                job_id = parsed.path.removeprefix("/api/jobs/").removesuffix("/cancel")
                self._json(HTTPStatus.OK, state.jobs.cancel(job_id))
                return
            raise WebRequestError(HTTPStatus.NOT_FOUND, "Page not found.")

        def _post_lan_acquisition(self, path: str) -> None:
            if not state.lan_catalog.acquisition_queue.enabled:
                raise WebRequestError(
                    HTTPStatus.NOT_FOUND,
                    "LAN acquisition coordination is not enabled.",
                )
            action = path.removeprefix("/api/lan/v1/acquisition/")
            body = self._body()
            feed_id = str(body.get("feed_id") or "").strip()
            if not FEED_ID_PATTERN.fullmatch(feed_id):
                raise WebRequestError(
                    HTTPStatus.BAD_REQUEST,
                    "A numeric feed ID is required.",
                )
            archive_date = self._date_value(
                str(body.get("archive_date") or "")
            )
            quota_scope = str(body.get("quota_scope") or "default").strip()
            try:
                if action == "claim":
                    owner_node_id = str(body.get("owner_node_id") or "")
                    producer_url = str(body.get("producer_url") or "")
                    value = state.lan_catalog.acquisition_queue.claim(
                        quota_scope,
                        feed_id,
                        archive_date,
                        owner_node_id=owner_node_id,
                        producer_url=producer_url,
                        requester_address=str(self.client_address[0]),
                        allow_multihomed_self=bool(
                            configured_advertisement
                            and owner_node_id == state.lan_catalog.node_id
                            and normalize_peer_url(producer_url)
                            == configured_advertisement
                        ),
                    )
                elif action == "renew":
                    value = state.lan_catalog.acquisition_queue.renew(
                        quota_scope,
                        feed_id,
                        archive_date,
                        lease_token=str(body.get("lease_token") or ""),
                    )
                elif action == "finish":
                    value = state.lan_catalog.acquisition_queue.finish(
                        quota_scope,
                        feed_id,
                        archive_date,
                        lease_token=str(body.get("lease_token") or ""),
                        outcome=str(body.get("outcome") or ""),
                        block_count=int(body.get("block_count") or 0),
                        blocks=body.get("blocks") or (),  # type: ignore[arg-type]
                        retry_after_seconds=body.get("retry_after_seconds"),  # type: ignore[arg-type]
                    )
                else:
                    raise WebRequestError(
                        HTTPStatus.NOT_FOUND,
                        "LAN acquisition action not found.",
                    )
            except PermissionError as exc:
                raise WebRequestError(HTTPStatus.FORBIDDEN, str(exc)) from exc
            except (LanSyncError, TypeError, ValueError) as exc:
                raise WebRequestError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
            self._json(HTTPStatus.OK, value)

        def _post_lan_quota(self, path: str) -> None:
            action = path.removeprefix("/api/lan/v1/quota/")
            body = self._body()
            try:
                profile_id = normalize_account_profile_id(
                    str(body.get("account_profile_id") or "default")
                )
            except ValueError as exc:
                raise WebRequestError(
                    HTTPStatus.BAD_REQUEST,
                    str(exc),
                ) from exc
            ledger = ArchiveRequestLedger(
                base_dir=state.working_dir,
                account_profile_id=profile_id,
            )
            try:
                if action == "reserve":
                    feed_id = str(body.get("feed_id") or "").strip()
                    archive_date = str(body.get("archive_date") or "").strip()
                    archive_id = normalize_archive_request_id(
                        body.get("archive_id")
                    )
                    if not FEED_ID_PATTERN.fullmatch(feed_id):
                        raise ValueError("A numeric feed ID is required.")
                    _ = date.fromisoformat(archive_date)
                    request_id = ledger.reserve(
                        feed_id=feed_id,
                        archive_date=archive_date,
                        archive_id=archive_id,
                    )
                    value: dict[str, Any] = {"request_id": request_id}
                elif action == "finish":
                    request_id = int(body.get("request_id") or 0)
                    if request_id < 1:
                        raise ValueError("A positive quota request ID is required.")
                    http_status_value = body.get("http_status")
                    http_status = (
                        int(http_status_value)
                        if http_status_value is not None
                        else None
                    )
                    if http_status is not None and not 100 <= http_status <= 599:
                        raise ValueError("The archive HTTP status is not valid.")
                    ledger.finish(
                        request_id,
                        outcome=str(body.get("outcome") or "")[:80],
                        http_status=http_status,
                    )
                    value = ledger.status()
                elif action == "rate-limit":
                    reason = str(body.get("reason") or "").strip()
                    if not reason:
                        raise ValueError("A rate-limit reason is required.")
                    value = ledger.mark_rate_limited(reason[:800])
                else:
                    raise WebRequestError(
                        HTTPStatus.NOT_FOUND,
                        "LAN quota action not found.",
                    )
            except ArchiveRequestBudgetExceeded as exc:
                raise WebRequestError(
                    HTTPStatus.TOO_MANY_REQUESTS,
                    str(exc),
                ) from exc
            except (TypeError, ValueError) as exc:
                raise WebRequestError(
                    HTTPStatus.BAD_REQUEST,
                    str(exc),
                ) from exc
            self._json(
                HTTPStatus.OK,
                {"protocol": LAN_PROTOCOL, **value},
            )

        def _post_lan_processing(self, path: str) -> None:
            if not state.lan_catalog.processing_queue.enabled:
                raise WebRequestError(
                    HTTPStatus.NOT_FOUND,
                    "LAN processing coordination is not enabled.",
                )
            action = path.removeprefix("/api/lan/v1/processing/")
            body = self._body()
            feed_id = str(body.get("feed_id") or "").strip()
            if not FEED_ID_PATTERN.fullmatch(feed_id):
                raise WebRequestError(
                    HTTPStatus.BAD_REQUEST,
                    "A numeric feed ID is required.",
                )
            archive_date = self._date_value(
                str(body.get("archive_date") or "")
            )
            fingerprint = str(
                body.get("processing_fingerprint") or ""
            ).strip()
            try:
                if action == "claim":
                    owner_node_id = str(body.get("owner_node_id") or "")
                    producer_url = str(body.get("producer_url") or "")
                    value = state.lan_catalog.processing_queue.claim(
                        fingerprint,
                        feed_id,
                        archive_date,
                        owner_node_id=owner_node_id,
                        producer_url=producer_url,
                        requester_address=str(self.client_address[0]),
                        allow_multihomed_self=bool(
                            configured_advertisement
                            and owner_node_id == state.lan_catalog.node_id
                            and normalize_peer_url(producer_url)
                            == configured_advertisement
                        ),
                    )
                elif action == "renew":
                    value = state.lan_catalog.processing_queue.renew(
                        fingerprint,
                        feed_id,
                        archive_date,
                        lease_token=str(body.get("lease_token") or ""),
                    )
                elif action == "finish":
                    value = state.lan_catalog.processing_queue.finish(
                        fingerprint,
                        feed_id,
                        archive_date,
                        lease_token=str(body.get("lease_token") or ""),
                        outcome=str(body.get("outcome") or ""),
                        artifact_count=int(body.get("artifact_count") or 0),
                    )
                else:
                    raise WebRequestError(
                        HTTPStatus.NOT_FOUND,
                        "LAN processing action not found.",
                    )
            except PermissionError as exc:
                raise WebRequestError(HTTPStatus.FORBIDDEN, str(exc)) from exc
            except (LanSyncError, TypeError, ValueError) as exc:
                raise WebRequestError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
            self._json(HTTPStatus.OK, value)

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

        def _require_lan_access(self) -> None:
            if not state.lan_catalog.enabled:
                raise WebRequestError(
                    HTTPStatus.NOT_FOUND,
                    "LAN archive sharing is not enabled on this app.",
                )
            supplied_key = self.headers.get("X-Radio-Archive-LAN-Key", "")
            if not state.lan_catalog.authorized(supplied_key):
                raise WebRequestError(
                    HTTPStatus.FORBIDDEN,
                    "The LAN archive key is missing or invalid.",
                )

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

        def _lan_block(self, path: Path, sha256: str) -> None:
            size = path.stat().st_size
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "audio/mpeg")
            self.send_header("Content-Length", str(size))
            self.send_header("X-Radio-Archive-Protocol", LAN_PROTOCOL)
            self.send_header("X-Radio-Archive-SHA256", sha256)
            self.send_header("Cache-Control", "private, no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.end_headers()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(256 * 1024), b""):
                    self.wfile.write(chunk)

        def _lan_artifact(self, path: Path, sha256: str) -> None:
            size = path.stat().st_size
            self.send_response(HTTPStatus.OK)
            content_type = {
                ".json": "application/json; charset=utf-8",
                ".mp3": "audio/mpeg",
            }.get(path.suffix.lower(), "text/plain; charset=utf-8")
            self.send_header(
                "Content-Type",
                content_type,
            )
            self.send_header("Content-Length", str(size))
            self.send_header("X-Radio-Archive-Protocol", LAN_PROTOCOL)
            self.send_header("X-Radio-Archive-SHA256", sha256)
            self.send_header("Cache-Control", "private, no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.end_headers()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(256 * 1024), b""):
                    self.wfile.write(chunk)

    class LocalThreadingHTTPServer(ThreadingHTTPServer):
        address_family = (
            socket.AF_INET6
            if host != "localhost" and ipaddress.ip_address(host).version == 6
            else socket.AF_INET
        )
        daemon_threads = True

        def server_close(self) -> None:
            state.scheduler.close()
            state.lan_reconciler.close()
            responder = getattr(self, "lan_discovery", None)
            if responder is not None:
                responder.close()
            super().server_close()

    server = LocalThreadingHTTPServer((host, port), Handler)
    server.state = state  # type: ignore[attr-defined]
    server.quiet = False  # type: ignore[attr-defined]
    server.lan_discovery = None  # type: ignore[attr-defined]
    state.scheduler.start()
    if state.lan_catalog.enabled and lan_discovery_enabled:
        def advertised_url(remote_address: str) -> str:
            if configured_advertisement:
                return configured_advertisement
            if host not in {"0.0.0.0", "::", "localhost"}:
                return format_web_url(host, server.server_port).rstrip("/")
            if host == "localhost" or bind_is_loopback(host):
                return format_web_url("127.0.0.1", server.server_port).rstrip("/")
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
                route.connect((remote_address, 9))
                local_address = str(route.getsockname()[0])
            return format_web_url(
                local_address,
                server.server_port,
            ).rstrip("/")

        responder = LanDiscoveryResponder(
            state.lan_catalog,
            advertised_url,
        )
        responder.start()
        server.lan_discovery = responder  # type: ignore[attr-defined]
    state.lan_reconciler.start()
    return server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the private cross-platform radio archive web app. "
            "It defaults to loopback and supports an explicit trusted-LAN bind."
        )
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
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        type=validate_bind_host,
        help=(
            "Loopback or a trusted private/link-local interface address. "
            "0.0.0.0/:: listens on every interface and must not be port-forwarded."
        ),
    )
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
        background_sync_enabled=True,
    )
    bound_host, bound_port = server.server_address[:2]
    display_host = (
        "127.0.0.1"
        if bound_host == "0.0.0.0"
        else "::1" if bound_host == "::" else str(bound_host)
    )
    url = format_web_url(display_host, bound_port)
    print(f"Radio Archive Intelligence is available at {url}", flush=True)
    if not bind_is_loopback(host):
        print(
            "Trusted-LAN mode is active: anyone who can reach this address can "
            "open retained transcripts and audio.",
            flush=True,
        )
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
