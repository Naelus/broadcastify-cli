from __future__ import annotations

import json
import os
import re
import threading
import time
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any


ARCHIVE_CACHE_INDEX_FILENAME = ".broadcastify-archive-index.json"
ARCHIVE_CACHE_INDEX_SCHEMA_VERSION = 1
ARCHIVE_CACHE_COMPLETION_FILENAME = ".broadcastify-archive-complete.json"
ARCHIVE_CACHE_COMPLETION_SCHEMA_VERSION = 1

_INDEX_WRITE_LOCK = threading.RLock()


def _write_index(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _index_path(day_directory: str | Path) -> Path:
    return Path(day_directory) / ARCHIVE_CACHE_INDEX_FILENAME


def _completion_path(day_directory: str | Path) -> Path:
    return Path(day_directory) / ARCHIVE_CACHE_COMPLETION_FILENAME


def _empty_index(feed_id: str, archive_date: date) -> dict[str, Any]:
    return {
        "schema_version": ARCHIVE_CACHE_INDEX_SCHEMA_VERSION,
        "feed_id": str(feed_id),
        "archive_date": archive_date.isoformat(),
        "archives": {},
    }


def _load_index(
    day_directory: str | Path,
    feed_id: str,
    archive_date: date | None = None,
) -> dict[str, Any]:
    path = _index_path(day_directory)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return _empty_index(feed_id, archive_date or date.min)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != ARCHIVE_CACHE_INDEX_SCHEMA_VERSION
        or str(payload.get("feed_id") or "") != str(feed_id)
        or not isinstance(payload.get("archives"), dict)
    ):
        return _empty_index(feed_id, archive_date or date.min)
    if archive_date is not None and str(payload.get("archive_date") or "") not in {
        "",
        archive_date.isoformat(),
    }:
        return _empty_index(feed_id, archive_date)
    return payload


def cached_archive_for_id(
    day_directory: str | Path,
    feed_id: str,
    archive_id: str,
) -> Path | None:
    """Resolve an exact provider archive ID without relying on display timestamps."""

    day = Path(day_directory)
    payload = _load_index(day, feed_id)
    raw = payload["archives"].get(str(archive_id))
    if not isinstance(raw, dict):
        return None
    filename = str(raw.get("filename") or "")
    if not filename or Path(filename).name != filename:
        return None
    candidate = day / filename
    try:
        expected_size = int(raw.get("size") or 0)
        actual_size = candidate.stat().st_size
        if (
            candidate.is_symlink()
            or not candidate.is_file()
            or expected_size <= 0
            or actual_size != expected_size
        ):
            return None
    except (OSError, TypeError, ValueError):
        return None
    return candidate


def archive_identity_for_filename(
    day_directory: str | Path,
    feed_id: str,
    filename: str,
) -> tuple[str, str, bool]:
    """Return ``(archive_id, listing_prefix, file_matches_index)``."""

    if Path(filename).name != filename:
        return "", "", False
    payload = _load_index(day_directory, feed_id)
    for archive_id, raw in payload["archives"].items():
        if not isinstance(raw, dict) or str(raw.get("filename") or "") != filename:
            continue
        valid = False
        try:
            source = Path(day_directory) / filename
            expected_size = int(raw.get("size") or 0)
            valid = (
                not source.is_symlink()
                and source.is_file()
                and expected_size > 0
                and source.stat().st_size == expected_size
            )
        except (OSError, TypeError, ValueError):
            pass
        return (
            str(archive_id),
            str(raw.get("listing_prefix") or ""),
            valid,
        )
    return "", "", False


def archive_identities_for_filename(
    day_directory: str | Path,
    feed_id: str,
    filename: str,
) -> tuple[tuple[str, str], ...]:
    """Return every exact provider identity proven for one retained file.

    Broadcastify can expose more than one archive-list ID whose authenticated
    download resolves to the same retained MP3 filename. Those are aliases,
    not duplicate audio files. Only identities whose indexed size still
    matches the local file are returned.
    """

    if Path(filename).name != filename:
        return ()
    payload = _load_index(day_directory, feed_id)
    try:
        source = Path(day_directory) / filename
        if source.is_symlink() or not source.is_file():
            return ()
        actual_size = source.stat().st_size
    except OSError:
        return ()
    identities: list[tuple[str, str]] = []
    for archive_id, raw in payload["archives"].items():
        if not isinstance(raw, dict) or str(raw.get("filename") or "") != filename:
            continue
        try:
            expected_size = int(raw.get("size") or 0)
        except (TypeError, ValueError):
            continue
        if expected_size <= 0 or expected_size != actual_size:
            continue
        identities.append(
            (str(archive_id), str(raw.get("listing_prefix") or ""))
        )
    return tuple(identities)


def remember_complete_archive_day(
    day_directory: str | Path,
    feed_id: str,
    archive_date: date,
    archive_ids: Sequence[str],
) -> bool:
    """Persist a locally verifiable, network-free completion snapshot.

    An exact identity index can represent either a partial or a complete day.
    This separate marker is therefore written only after an authenticated
    listing has been satisfied in full, or after an exact trusted-LAN
    completion manifest has been assembled. Every referenced identity is
    revalidated against its retained file before the marker is published.
    """

    normalized_ids = list(
        dict.fromkeys(str(value).strip() for value in archive_ids)
    )
    if (
        len(normalized_ids) > 1_000
        or any(not value or len(value) > 200 for value in normalized_ids)
    ):
        return False
    day = Path(day_directory)
    with _INDEX_WRITE_LOCK:
        for archive_id in normalized_ids:
            if cached_archive_for_id(day, feed_id, archive_id) is None:
                return False
        day.mkdir(parents=True, exist_ok=True)
        _write_index(
            _completion_path(day),
            {
                "schema_version": ARCHIVE_CACHE_COMPLETION_SCHEMA_VERSION,
                "feed_id": str(feed_id),
                "archive_date": archive_date.isoformat(),
                "archive_ids": normalized_ids,
                "completed_at_unix": round(time.time(), 6),
            },
        )
    return True


def complete_cached_archive_day(
    day_directory: str | Path,
    feed_id: str,
    archive_date: date,
) -> tuple[list[Path], int] | None:
    """Return a proven complete snapshot using local files only.

    ``None`` means no valid completion proof is available. ``([], 0)`` is a
    valid authenticated empty-day snapshot. Multiple provider identities may
    intentionally resolve to one retained file, so the returned paths are
    unique while the integer is the completed identity count.
    """

    day = Path(day_directory)
    try:
        payload = json.loads(
            _completion_path(day).read_text(encoding="utf-8")
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version")
        != ARCHIVE_CACHE_COMPLETION_SCHEMA_VERSION
        or str(payload.get("feed_id") or "") != str(feed_id)
        or str(payload.get("archive_date") or "") != archive_date.isoformat()
        or not isinstance(payload.get("archive_ids"), list)
    ):
        return None
    raw_ids = payload["archive_ids"]
    if len(raw_ids) > 1_000 or any(
        not isinstance(value, str) or not value or len(value) > 200
        for value in raw_ids
    ):
        return None
    archive_ids = list(dict.fromkeys(raw_ids))
    if len(archive_ids) != len(raw_ids):
        return None
    files: list[Path] = []
    for archive_id in archive_ids:
        cached = cached_archive_for_id(day, feed_id, archive_id)
        if cached is None:
            return None
        files.append(cached)
    return sorted(dict.fromkeys(files)), len(archive_ids)


def remember_archive_identity(
    day_directory: str | Path,
    feed_id: str,
    archive_date: date,
    archive_id: str,
    source_file: str | Path,
    *,
    listing_prefix: str | None = None,
    allow_filename_alias: bool = False,
) -> None:
    """Atomically retain the exact website-ID-to-file relationship.

    Broadcastify's archive-list ``startTs`` and its download filename are not
    interchangeable identifiers.  Some feeds have exhibited offsets of more
    than thirty minutes.  Keeping the provider ID beside the retained MP3 is
    the only safe way to prove that a later cache hit is the same request.
    ``allow_filename_alias`` is reserved for an authenticated media response
    (or a hash-verified LAN copy of one) that proves multiple IDs resolve to
    the same retained file.
    """

    day = Path(day_directory)
    source = Path(source_file)
    try:
        if (
            source.parent.resolve() != day.resolve()
            or source.name != Path(source.name).name
            or source.is_symlink()
            or not source.is_file()
            or source.stat().st_size <= 0
        ):
            return
    except OSError:
        return
    normalized_id = str(archive_id).strip()
    if not normalized_id or len(normalized_id) > 200:
        return

    path = _index_path(day)
    day.mkdir(parents=True, exist_ok=True)
    with _INDEX_WRITE_LOCK:
        payload = _load_index(day, feed_id, archive_date)
        payload["schema_version"] = ARCHIVE_CACHE_INDEX_SCHEMA_VERSION
        payload["feed_id"] = str(feed_id)
        payload["archive_date"] = archive_date.isoformat()
        payload["updated_at_unix"] = round(time.time(), 6)
        if not allow_filename_alias:
            for existing_id, raw in list(payload["archives"].items()):
                if (
                    existing_id != normalized_id
                    and isinstance(raw, dict)
                    and str(raw.get("filename") or "") == source.name
                ):
                    del payload["archives"][existing_id]
        payload["archives"][normalized_id] = {
            "filename": source.name,
            "listing_prefix": str(listing_prefix or "")[:12],
            "size": source.stat().st_size,
        }
        _write_index(path, payload)


def reconcile_complete_legacy_day(
    day_directory: str | Path,
    feed_id: str,
    archive_date: date,
    archive_ids: Sequence[str],
    listing_prefixes: Mapping[str, str],
) -> int:
    """Index a complete older day whose filename timestamps drifted.

    Before exact IDs were persisted, cache reuse had to compare a listing
    timestamp with the timestamp in ``Content-Disposition``. Those clocks can
    differ substantially. A one-to-one chronological migration is safe only
    when every unique listing row and every retained source block is present,
    all timestamps are unambiguous, and any already-known mapping agrees.
    Partial or irregular days deliberately remain on the conservative
    per-block fallback.
    """

    normalized_ids = list(dict.fromkeys(str(value).strip() for value in archive_ids))
    if not normalized_ids or any(not value for value in normalized_ids):
        return 0
    identities: list[tuple[datetime, str, str]] = []
    for archive_id in normalized_ids:
        prefix = str(listing_prefixes.get(archive_id) or "")
        if not re.fullmatch(r"\d{12}", prefix):
            return 0
        try:
            started = datetime.strptime(prefix, "%Y%m%d%H%M")
        except ValueError:
            return 0
        identities.append((started, archive_id, prefix))
    if len({value[0] for value in identities}) != len(identities):
        return 0

    day = Path(day_directory)
    if not day.is_dir():
        return 0
    pattern = re.compile(
        rf"^(?P<stamp>\d{{12}})-.+-{re.escape(str(feed_id))}\.mp3$",
        re.IGNORECASE,
    )
    sources: list[tuple[datetime, Path]] = []
    try:
        for source in day.glob("*.mp3"):
            match = pattern.fullmatch(source.name)
            if match is None or source.is_symlink() or not source.is_file():
                continue
            stat = source.stat()
            if stat.st_size <= 0:
                continue
            stamped = datetime.strptime(match.group("stamp"), "%Y%m%d%H%M")
            sources.append((stamped, source))
    except (OSError, ValueError):
        return 0
    if len(sources) != len(identities):
        return 0
    if len({value[0] for value in sources}) != len(sources):
        return 0

    identities.sort(key=lambda value: value[0])
    sources.sort(key=lambda value: value[0])
    # The filename clock may be offset, but a complete day should retain the
    # same sequence. Reject a migration if relative gaps diverge dramatically
    # or if the clock offset itself jumps by more than twenty minutes.
    offsets = [
        (source_stamp - listing_stamp).total_seconds()
        for (listing_stamp, _archive_id, _prefix), (source_stamp, _source) in zip(
            identities,
            sources,
            strict=True,
        )
    ]
    if any(abs(value) > 3 * 60 * 60 for value in offsets):
        return 0
    if max(offsets) - min(offsets) > 20 * 60:
        return 0
    for index in range(1, len(identities)):
        listing_gap = (identities[index][0] - identities[index - 1][0]).total_seconds()
        source_gap = (sources[index][0] - sources[index - 1][0]).total_seconds()
        if abs(listing_gap - source_gap) > 20 * 60:
            return 0

    path = _index_path(day)
    with _INDEX_WRITE_LOCK:
        payload = _load_index(day, feed_id, archive_date)
        archives = payload["archives"]
        pairs = list(zip(identities, sources, strict=True))
        for (_started, archive_id, _prefix), (_stamped, source) in pairs:
            existing = archives.get(archive_id)
            if isinstance(existing, dict) and str(existing.get("filename") or "") not in {
                "",
                source.name,
            }:
                return 0
            for other_id, raw in archives.items():
                if (
                    other_id != archive_id
                    and isinstance(raw, dict)
                    and str(raw.get("filename") or "") == source.name
                ):
                    return 0

        added = 0
        for (_started, archive_id, prefix), (_stamped, source) in pairs:
            if archive_id not in archives:
                added += 1
            archives[archive_id] = {
                "filename": source.name,
                "listing_prefix": prefix,
                "size": source.stat().st_size,
            }
        if added:
            payload["schema_version"] = ARCHIVE_CACHE_INDEX_SCHEMA_VERSION
            payload["feed_id"] = str(feed_id)
            payload["archive_date"] = archive_date.isoformat()
            payload["updated_at_unix"] = round(time.time(), 6)
            day.mkdir(parents=True, exist_ok=True)
            _write_index(path, payload)
        return added
