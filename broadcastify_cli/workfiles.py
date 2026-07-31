from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import socket
import time
import uuid
from pathlib import Path
from typing import Iterable


_PYANNOTE_RAW_PATTERN = re.compile(
    r"^\..+\.(?P<owner>[a-f0-9]{12})\.(?P<pid>\d+)\.\d+"
    r"\.pyannote\.f32le$",
    re.IGNORECASE,
)
_PYANNOTE_PART_PATTERN = re.compile(
    r"^\..+\.(?P<owner>[a-f0-9]{12})\.(?P<pid>\d+)\.\d+"
    r"\.pyannote\.part\.flac$",
    re.IGNORECASE,
)
_COMBINED_PART_PATTERN = re.compile(
    r"^\.combined_.+\.(?P<owner>[a-f0-9]{12})\.(?P<pid>\d+)"
    r"\.[^.]+\.part\.(?:mp3|wav|flac)$",
    re.IGNORECASE,
)
_WORK_FILE_OWNER_TOKEN = hashlib.sha256(
    (
        str(os.getenv("BROADCASTIFY_WORK_OWNER_ID") or "").strip()
        or f"{socket.gethostname() or 'unknown-host'}|{uuid.getnode():012x}"
    ).encode("utf-8", errors="replace")
).hexdigest()[:12]


def work_file_owner_token() -> str:
    """Return an opaque host token used to distinguish shared-path workers."""

    return _WORK_FILE_OWNER_TOKEN


def is_temporary_audio_work_file(path: str | Path) -> bool:
    """Identify disposable audio preparation files, not retained evidence."""

    name = Path(path).name.lower()
    return (
        (
            name.startswith(".combined_")
            and ".part." in name
            and name.endswith((".mp3", ".wav", ".flac"))
        )
        or name.endswith(".pyannote.part.flac")
        or name.endswith(".pyannote.flac")
        or name.endswith(".pyannote.f32le")
    )


def directory_storage_usage(path: str | Path) -> tuple[int, int]:
    """Return retained and temporary-working bytes below a day directory."""

    root = Path(path)
    if not root.is_dir():
        return 0, 0
    retained = 0
    working = 0
    for item in root.rglob("*"):
        try:
            if not item.is_file():
                continue
            size = item.stat().st_size
        except OSError:
            continue
        if is_temporary_audio_work_file(item):
            working += size
        else:
            retained += size
    return retained, working


def files_storage_usage(paths: Iterable[str | Path]) -> tuple[int, int]:
    """Return retained and working bytes for an explicit artifact set."""

    retained = 0
    working = 0
    seen: set[Path] = set()
    for value in paths:
        item = Path(value)
        try:
            identity = item.resolve()
            if identity in seen or not item.is_file():
                continue
            seen.add(identity)
            size = item.stat().st_size
        except OSError:
            continue
        if is_temporary_audio_work_file(item):
            working += size
        else:
            retained += size
    return retained, working


def _windows_process_running(pid: int) -> bool:
    process_query_limited_information = 0x1000
    still_active = 259
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    open_process = kernel32.OpenProcess
    open_process.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    open_process.restype = ctypes.c_void_p
    get_exit_code = kernel32.GetExitCodeProcess
    get_exit_code.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    get_exit_code.restype = ctypes.c_int
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [ctypes.c_void_p]
    close_handle.restype = ctypes.c_int
    handle = open_process(
        process_query_limited_information,
        False,
        pid,
    )
    if not handle:
        # ERROR_INVALID_PARAMETER is the normal "PID does not exist" result.
        # Access denied and other indeterminate failures must be treated as
        # alive so a read-oriented refresh never deletes active work.
        return ctypes.get_last_error() != 87
    try:
        exit_code = ctypes.c_ulong()
        if not get_exit_code(handle, ctypes.byref(exit_code)):
            return False
        return exit_code.value == still_active
    finally:
        close_handle(handle)


def process_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if os.name == "nt":
        return _windows_process_running(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _completed_diarization_cache_matches(prepared: Path) -> bool:
    """Return whether a prepared FLAC has an authoritative completed cache."""

    suffix = ".pyannote.flac"
    if (
        not prepared.name.lower().endswith(suffix)
        or prepared.name.lower().endswith(".pyannote.part.flac")
        or prepared.parent.name != ".cache"
        or prepared.parent.parent.name != "transcripts"
    ):
        return False
    stem = prepared.name[: -len(suffix)]
    day_directory = prepared.parent.parent.parent
    audio_path = next(
        (
            candidate
            for candidate in (
                day_directory / f"{stem}.mp3",
                day_directory / f"{stem}.wav",
                day_directory / f"{stem}.flac",
            )
            if candidate.is_file()
        ),
        None,
    )
    cache_path = prepared.parent.parent / f"{stem}.diarization.json"
    if audio_path is None or not cache_path.is_file():
        return False
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        audio_stat = audio_path.stat()
        return (
            str(payload.get("engine") or "community-1") == "community-1"
            and int(payload.get("audio_size", -1)) == audio_stat.st_size
            and int(payload.get("audio_mtime_ns", -1))
            == audio_stat.st_mtime_ns
            and isinstance(payload.get("turns"), list)
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def cleanup_orphaned_audio_work_files(
    path: str | Path,
    *,
    minimum_age_seconds: float = 3_600,
    now: float | None = None,
) -> list[Path]:
    """Remove old incomplete outputs and dead-process raw scratch audio.

    A completed ``*.pyannote.flac`` is a valid retry cache and is retained
    until an exact completed diarization cache proves it is no longer needed.
    It is reported as working storage, rather than retained archive data.
    """

    root = Path(path)
    if not root.is_dir():
        return []
    cutoff = (time.time() if now is None else float(now)) - max(
        0.0,
        float(minimum_age_seconds),
    )
    removed: list[Path] = []
    for item in root.rglob("*"):
        try:
            if not item.is_file() or item.stat().st_mtime > cutoff:
                continue
        except OSError:
            continue
        name = item.name.lower()
        if name.endswith(".pyannote.flac") and not name.endswith(
            ".pyannote.part.flac"
        ):
            if not _completed_diarization_cache_matches(item):
                continue
            removable = True
        else:
            removable = False
        raw_match = _PYANNOTE_RAW_PATTERN.match(item.name)
        partial_match = _PYANNOTE_PART_PATTERN.match(item.name)
        combined_match = _COMBINED_PART_PATTERN.match(item.name)
        owned_match = raw_match or partial_match or combined_match
        if owned_match:
            # A PID has meaning only on its originating host. Foreign-host
            # work on a shared NAS path is never eligible for local cleanup.
            if owned_match.group("owner").lower() != work_file_owner_token():
                continue
            if process_running(int(owned_match.group("pid"))):
                continue
        # Legacy partial names do not carry ownership, so their worker cannot
        # be proven dead. Count them as temporary but never unlink them from a
        # read-oriented Library refresh. New writers include a PID.
        removable = removable or (
            raw_match is not None
            or partial_match is not None
            or combined_match is not None
        )
        if not removable:
            continue
        try:
            if item.stat().st_mtime > cutoff:
                continue
        except OSError:
            continue
        try:
            item.unlink()
        except OSError:
            continue
        removed.append(item)
    return removed
