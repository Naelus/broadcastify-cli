from __future__ import annotations

import ctypes
import hashlib
import hmac
import ipaddress
import json
import math
import os
import re
import secrets
import socket
import sys
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse, urlunparse

import requests

from .audio import combined_output_is_current
from .archive_cache import (
    archive_identities_for_filename,
    complete_cached_archive_day,
    remember_archive_identity,
    remember_complete_archive_day,
)

LAN_PROTOCOL = "radio-archive-lan/1"
LAN_DISCOVERY_MAGIC = b"RADIO-ARCHIVE-LAN-DISCOVER/1 "
LAN_DISCOVERY_PORT = 48_765
LAN_MULTICAST_ADDRESS = "239.255.77.77"
MAX_LAN_PEERS = 24
MAX_FEED_DAYS = 20_000
MAX_BLOCKS_PER_DAY = 128
MAX_ARCHIVE_BLOCK_BYTES = 256 * 1024 * 1024
MAX_INVENTORY_BYTES = 1024 * 1024
MAX_LAN_QUEUE_ENTRIES = 512
MAX_TRANSCRIPT_ARTIFACTS_PER_DAY = 512
MAX_TRANSCRIPT_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_DERIVED_AUDIO_BYTES = 8 * 1024 * 1024 * 1024
LAN_QUEUE_LEASE_SECONDS = 90.0
LAN_PROCESSING_LEASE_SECONDS = 180.0
LAN_QUEUE_RESULT_SECONDS = 24 * 60 * 60.0
LAN_QUEUE_ROLLING_RESULT_SECONDS = 5 * 60.0
LAN_QUEUE_REQUEST_BYTES = 256 * 1024
LAN_QUEUE_RESPONSE_BYTES = 256 * 1024
LAN_QUEUE_NODE_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,80}$")
LAN_QUEUE_SCOPE_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
PROCESSING_FINGERPRINT_PATTERN = re.compile(r"^[0-9a-f]{64}$")
DERIVED_VARIANTS_DIRECTORY = ".broadcastify-derived"
RAW_ARCHIVE_PATTERN = re.compile(
    r"^(?P<stamp>\d{12})-(?P<archive_id>\d+)-(?P<feed_id>\d+)\.mp3$",
    re.IGNORECASE,
)


def _archive_stamp_matches_day(stamp: str, archive_date: date) -> bool:
    """Accept the final website block when its drifting label crosses midnight."""

    try:
        timestamp = datetime.strptime(stamp, "%Y%m%d%H%M")
    except ValueError:
        return False
    day_start = datetime.combine(archive_date, datetime.min.time())
    return timedelta(0) <= timestamp - day_start < timedelta(hours=30)

ProgressCallback = Callable[[str], None]


class LanSyncError(RuntimeError):
    pass


def environment_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def environment_float(
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    try:
        value = float(os.getenv(name) or default)
    except ValueError:
        value = default
    return min(maximum, max(minimum, value))


def discovery_port() -> int:
    try:
        value = int(os.getenv("BROADCASTIFY_LAN_DISCOVERY_PORT") or LAN_DISCOVERY_PORT)
    except ValueError:
        return LAN_DISCOVERY_PORT
    return value if 1_024 <= value <= 65_535 else LAN_DISCOVERY_PORT


def normalize_peer_url(value: str) -> str:
    """Return a private, path-free HTTP peer URL.

    Numeric addresses avoid DNS rebinding at this local file-transfer boundary.
    Loopback is accepted for same-machine testing and multi-instance use.
    """

    raw = str(value or "").strip()
    if not raw:
        raise ValueError("A LAN peer URL cannot be blank.")
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("LAN peer URLs must use http:// or https://.")
    if parsed.username or parsed.password:
        raise ValueError("LAN peer URLs cannot contain credentials.")
    if parsed.query or parsed.fragment or parsed.params:
        raise ValueError("LAN peer URLs cannot contain a query or fragment.")
    if parsed.path not in {"", "/"}:
        raise ValueError("LAN peer URLs must point to the app root, without a path.")
    hostname = (parsed.hostname or "").strip().lower()
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError as exc:
        raise ValueError(
            "Use a numeric private/link-local LAN address for archive peers."
        ) from exc
    if address.is_unspecified or address.is_multicast or not (
        address.is_private or address.is_link_local or address.is_loopback
    ):
        raise ValueError(
            "Archive peers must use a private, link-local, or loopback address."
        )
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("The LAN peer port is not valid.") from exc
    if port is not None and not 1 <= port <= 65_535:
        raise ValueError("The LAN peer port must be between 1 and 65535.")
    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = f"{host}:{port}" if port is not None else host
    return urlunparse((parsed.scheme, netloc, "", "", "", ""))


def normalize_peer_urls(
    values: str | Sequence[str] | None,
    *,
    strict: bool = True,
) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        raw_values = re.split(r"[\s,;]+", values)
    else:
        raw_values = [str(value) for value in values]
    peers: list[str] = []
    errors: list[str] = []
    for raw in raw_values:
        if not raw.strip():
            continue
        try:
            peer = normalize_peer_url(raw)
        except ValueError as exc:
            errors.append(f"{raw!r}: {exc}")
            continue
        if peer not in peers:
            peers.append(peer)
        if len(peers) >= MAX_LAN_PEERS:
            break
    if strict and errors:
        raise ValueError("Invalid LAN peer URL: " + errors[0])
    return tuple(peers)


def configured_peer_urls(
    request_values: str | Sequence[str] | None = None,
) -> tuple[str, ...]:
    requested = normalize_peer_urls(request_values)
    environment = normalize_peer_urls(
        os.getenv("BROADCASTIFY_LAN_PEERS"),
        strict=True,
    )
    return tuple(dict.fromkeys((*requested, *environment)))[:MAX_LAN_PEERS]


@dataclass(frozen=True)
class ArchiveIdentity:
    archive_id: str
    listing_prefix: str = ""

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ArchiveIdentity":
        identity = cls(
            archive_id=str(value.get("archive_id") or ""),
            listing_prefix=str(value.get("listing_prefix") or ""),
        )
        identity.validate()
        return identity

    def validate(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,200}", self.archive_id):
            raise LanSyncError("A LAN peer advertised an invalid archive identity.")
        if self.listing_prefix and not re.fullmatch(r"\d{12}", self.listing_prefix):
            raise LanSyncError("A LAN peer advertised an invalid archive timestamp.")


@dataclass(frozen=True)
class ArchiveBlock:
    feed_id: str
    archive_date: str
    filename: str
    size: int
    sha256: str
    modified_ns: int
    archive_id: str = ""
    listing_prefix: str = ""
    archive_identities: tuple[ArchiveIdentity, ...] = ()

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        expected_feed_id: str,
        expected_date: date,
    ) -> "ArchiveBlock":
        raw_identities = value.get("archive_identities") or ()
        if isinstance(raw_identities, (str, bytes)) or not isinstance(
            raw_identities, Sequence
        ):
            raise LanSyncError("A LAN peer advertised invalid archive identities.")
        if len(raw_identities) > 64:
            raise LanSyncError("A LAN peer advertised too many archive identities.")
        identities = tuple(
            ArchiveIdentity.from_mapping(raw)
            for raw in raw_identities
            if isinstance(raw, Mapping)
        )
        if len(identities) != len(raw_identities):
            raise LanSyncError("A LAN peer advertised invalid archive identities.")
        block = cls(
            feed_id=str(value.get("feed_id") or ""),
            archive_date=str(value.get("archive_date") or ""),
            filename=str(value.get("filename") or ""),
            size=int(value.get("size") or 0),
            sha256=str(value.get("sha256") or "").lower(),
            modified_ns=int(value.get("modified_ns") or 0),
            archive_id=str(value.get("archive_id") or ""),
            listing_prefix=str(value.get("listing_prefix") or ""),
            archive_identities=identities,
        )
        block.validate(expected_feed_id=expected_feed_id, expected_date=expected_date)
        return block

    def validate(self, *, expected_feed_id: str, expected_date: date) -> None:
        expected_date_value = expected_date.isoformat()
        match = RAW_ARCHIVE_PATTERN.fullmatch(self.filename)
        if (
            self.feed_id != expected_feed_id
            or self.archive_date != expected_date_value
            or match is None
            or match.group("feed_id") != expected_feed_id
            or not _archive_stamp_matches_day(match.group("stamp"), expected_date)
        ):
            raise LanSyncError("A LAN peer advertised an invalid archive block name.")
        if not 0 < self.size <= MAX_ARCHIVE_BLOCK_BYTES:
            raise LanSyncError("A LAN peer advertised an invalid archive block size.")
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise LanSyncError("A LAN peer advertised an invalid archive block hash.")
        if self.archive_id and not re.fullmatch(r"[A-Za-z0-9_.-]{1,200}", self.archive_id):
            raise LanSyncError("A LAN peer advertised an invalid archive identity.")
        if self.listing_prefix and not re.fullmatch(r"\d{12}", self.listing_prefix):
            raise LanSyncError("A LAN peer advertised an invalid archive timestamp.")
        if len(self.archive_identities) > 1:
            raise LanSyncError(
                "A LAN peer collapsed multiple archive timeline positions into one file."
            )
        for identity in self.archive_identities:
            identity.validate()
        identity_ids = [value.archive_id for value in self.archive_identities]
        if len(identity_ids) != len(set(identity_ids)):
            raise LanSyncError("A LAN peer advertised duplicate archive identities.")
        if (
            self.archive_identities
            and self.archive_id
            and self.archive_identities[0].archive_id != self.archive_id
        ):
            raise LanSyncError("A LAN peer advertised conflicting archive identities.")
        if (
            self.archive_identities
            and self.listing_prefix
            and self.archive_identities[0].listing_prefix != self.listing_prefix
        ):
            raise LanSyncError("A LAN peer advertised conflicting archive timestamps.")

    def identities(self) -> tuple[ArchiveIdentity, ...]:
        if self.archive_identities:
            return self.archive_identities
        if self.archive_id:
            return (ArchiveIdentity(self.archive_id, self.listing_prefix),)
        return ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TranscriptArtifact:
    """One hash-verified transcript file tied to exact audio and model input."""

    feed_id: str
    archive_date: str
    audio_filename: str
    audio_sha256: str
    processing_fingerprint: str
    filename: str
    kind: str
    size: int
    sha256: str
    modified_ns: int

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        expected_feed_id: str,
        expected_date: date,
        expected_fingerprint: str,
    ) -> "TranscriptArtifact":
        artifact = cls(
            feed_id=str(value.get("feed_id") or ""),
            archive_date=str(value.get("archive_date") or ""),
            audio_filename=str(value.get("audio_filename") or ""),
            audio_sha256=str(value.get("audio_sha256") or "").lower(),
            processing_fingerprint=str(
                value.get("processing_fingerprint") or ""
            ).lower(),
            filename=str(value.get("filename") or ""),
            kind=str(value.get("kind") or ""),
            size=int(value.get("size") or 0),
            sha256=str(value.get("sha256") or "").lower(),
            modified_ns=int(value.get("modified_ns") or 0),
        )
        artifact.validate(
            expected_feed_id=expected_feed_id,
            expected_date=expected_date,
            expected_fingerprint=expected_fingerprint,
        )
        return artifact

    def validate(
        self,
        *,
        expected_feed_id: str,
        expected_date: date,
        expected_fingerprint: str,
    ) -> None:
        if (
            self.feed_id != expected_feed_id
            or self.archive_date != expected_date.isoformat()
            or self.processing_fingerprint != expected_fingerprint
            or not PROCESSING_FINGERPRINT_PATTERN.fullmatch(
                self.processing_fingerprint
            )
        ):
            raise LanSyncError(
                "A LAN peer advertised an incompatible transcript artifact."
            )
        if not _valid_day_audio_filename(
            self.audio_filename,
            expected_feed_id,
            expected_date,
        ):
            raise LanSyncError(
                "A LAN peer advertised an invalid transcript audio identity."
            )
        expected_stem = Path(self.audio_filename).stem
        expected_names = {
            "audio": self.audio_filename,
            "json": f"{expected_stem}.json",
            "text": f"{expected_stem}.txt",
        }
        if self.audio_filename == (
            f"combined_{expected_feed_id}_{expected_date:%Y%m%d}.mp3"
        ):
            expected_names["manifest"] = f"{expected_stem}.manifest.json"
        if self.kind not in expected_names or self.filename != expected_names[self.kind]:
            raise LanSyncError(
                "A LAN peer advertised an invalid transcript artifact name."
            )
        minimum_size = 0 if self.kind == "text" else 1
        maximum_size = (
            MAX_DERIVED_AUDIO_BYTES
            if self.kind == "audio"
            else MAX_TRANSCRIPT_ARTIFACT_BYTES
        )
        if not minimum_size <= self.size <= maximum_size:
            raise LanSyncError(
                "A LAN peer advertised an invalid transcript artifact size."
            )
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise LanSyncError(
                "A LAN peer advertised an invalid transcript artifact hash."
            )
        if not re.fullmatch(r"[0-9a-f]{64}", self.audio_sha256):
            raise LanSyncError(
                "A LAN peer advertised an invalid transcript audio hash."
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _valid_day_audio_filename(
    filename: str,
    feed_id: str,
    archive_date: date,
) -> bool:
    if Path(filename).name != filename or not filename.lower().endswith(".mp3"):
        return False
    if filename == f"combined_{feed_id}_{archive_date:%Y%m%d}.mp3":
        return True
    match = RAW_ARCHIVE_PATTERN.fullmatch(filename)
    return bool(
        match
        and match.group("feed_id") == feed_id
        and _archive_stamp_matches_day(match.group("stamp"), archive_date)
    )


def _block_archive_identities(
    day_directory: Path,
    feed_id: str,
    filename: str,
) -> tuple[str, str, tuple[ArchiveIdentity, ...]]:
    identities = tuple(
        ArchiveIdentity(archive_id, listing_prefix)
        for archive_id, listing_prefix in archive_identities_for_filename(
            day_directory,
            feed_id,
            filename,
        )
    )
    if not identities:
        return "", "", ()
    return identities[0].archive_id, identities[0].listing_prefix, identities


def _remember_block_archive_identities(
    day_directory: Path,
    feed_id: str,
    archive_date: date,
    block: ArchiveBlock,
    source_file: Path,
) -> None:
    for identity in block.identities():
        # The peer supplied this identity with a hash- and size-verified block.
        # Equal bytes still need a distinct filename for each timeline position.
        remember_archive_identity(
            day_directory,
            feed_id,
            archive_date,
            identity.archive_id,
            source_file,
            listing_prefix=identity.listing_prefix,
        )


@dataclass(frozen=True)
class LanSyncResult:
    enabled: bool
    peers_considered: int = 0
    peers_reached: int = 0
    blocks_available: int = 0
    blocks_already_local: int = 0
    blocks_copied: int = 0
    bytes_copied: int = 0
    conflicts: int = 0
    completion_proven: bool = False
    failures: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LanTranscriptSyncResult:
    enabled: bool
    peers_considered: int = 0
    peers_reached: int = 0
    artifacts_available: int = 0
    artifacts_already_local: int = 0
    artifacts_copied: int = 0
    bytes_copied: int = 0
    conflicts: int = 0
    transcripts: tuple[Path, ...] = ()
    failures: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["transcripts"] = [str(path) for path in self.transcripts]
        return value


@dataclass(frozen=True)
class LanFeedSyncResult:
    """Feed-wide pull result used to converge followed feeds across nodes."""

    enabled: bool
    dates_discovered: tuple[str, ...] = ()
    days_considered: int = 0
    days_with_download_changes: int = 0
    days_with_transcript_changes: int = 0
    blocks_copied: int = 0
    transcript_artifacts_copied: int = 0
    bytes_copied: int = 0
    failures: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def merge_lan_sync_results(
    values: Iterable[LanSyncResult],
    *,
    enabled: bool = True,
) -> LanSyncResult:
    results = list(values)
    failures = tuple(
        dict.fromkeys(
            failure
            for result in results
            for failure in result.failures
        )
    )
    return LanSyncResult(
        enabled=enabled and any(result.enabled for result in results),
        peers_considered=max(
            (result.peers_considered for result in results),
            default=0,
        ),
        peers_reached=max(
            (result.peers_reached for result in results),
            default=0,
        ),
        blocks_available=max(
            (result.blocks_available for result in results),
            default=0,
        ),
        blocks_already_local=max(
            (result.blocks_already_local for result in results),
            default=0,
        ),
        blocks_copied=sum(result.blocks_copied for result in results),
        bytes_copied=sum(result.bytes_copied for result in results),
        conflicts=sum(result.conflicts for result in results),
        completion_proven=any(result.completion_proven for result in results),
        failures=failures[:50],
    )


@dataclass(frozen=True)
class LanDownloadTurn:
    """One client's role for a shared feed/day archive acquisition."""

    role: str
    feed_id: str = ""
    archive_date: str = ""
    coordinator_url: str = ""
    producer_url: str = ""
    owner_node_id: str = ""
    lease_token: str = ""
    lease_seconds: float = 0.0
    block_count: int = 0
    rolling: bool = False
    blocks: tuple[ArchiveBlock, ...] = ()
    audio_files: tuple[Path, ...] = ()
    sync_result: LanSyncResult = LanSyncResult(enabled=False)
    warnings: tuple[str, ...] = ()

    @property
    def is_leader(self) -> bool:
        return self.role == "leader"


@dataclass(frozen=True)
class LanProcessingTurn:
    """One node's role for a model-specific feed/day transcript operation."""

    role: str
    feed_id: str = ""
    archive_date: str = ""
    processing_fingerprint: str = ""
    coordinator_url: str = ""
    producer_url: str = ""
    owner_node_id: str = ""
    lease_token: str = ""
    lease_seconds: float = 0.0
    artifact_count: int = 0
    transcripts: tuple[Path, ...] = ()
    sync_result: LanTranscriptSyncResult = LanTranscriptSyncResult(enabled=False)
    warnings: tuple[str, ...] = ()

    @property
    def is_leader(self) -> bool:
        return self.role == "leader"


@dataclass
class _LanQueueEntry:
    state: str
    owner_node_id: str
    producer_url: str
    lease_token: str
    expires_at: float
    block_count: int = 0
    rolling: bool = False
    blocks: tuple[ArchiveBlock, ...] = ()


class LanAcquisitionQueue:
    """A bounded in-memory lease queue; retained MP3s are the durable state."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        lease_seconds: float = LAN_QUEUE_LEASE_SECONDS,
        result_seconds: float = LAN_QUEUE_RESULT_SECONDS,
        rolling_result_seconds: float = LAN_QUEUE_ROLLING_RESULT_SECONDS,
        maximum_entries: int = MAX_LAN_QUEUE_ENTRIES,
        clock: Callable[[], float] = time.monotonic,
        today: Callable[[], date] = date.today,
    ) -> None:
        self.enabled = bool(enabled)
        self.lease_seconds = min(300.0, max(15.0, float(lease_seconds)))
        self.result_seconds = min(
            24 * 60 * 60.0,
            max(60.0, float(result_seconds)),
        )
        self.rolling_result_seconds = min(
            30 * 60.0,
            max(30.0, float(rolling_result_seconds)),
        )
        self.maximum_entries = min(
            MAX_LAN_QUEUE_ENTRIES,
            max(16, int(maximum_entries)),
        )
        self._clock = clock
        self._today = today
        self._entries: dict[tuple[str, str, str], _LanQueueEntry] = {}
        self._lock = threading.RLock()

    def status(
        self,
        quota_scope: str,
        feed_id: str,
        archive_date: date,
    ) -> dict[str, Any]:
        scope = self._validate_key(quota_scope, feed_id)
        key = (scope, feed_id, archive_date.isoformat())
        with self._lock:
            now = self._clock()
            self._cleanup_locked(now)
            return self._payload_locked(key, now)

    def active_activity(self) -> dict[str, Any] | None:
        """Return the one non-secret live upstream lease, if any."""

        with self._lock:
            now = self._clock()
            self._cleanup_locked(now)
            for (scope, feed_id, archive_date), entry in self._entries.items():
                if entry.state != "active":
                    continue
                return {
                    "quota_scope": scope,
                    "feed_id": feed_id,
                    "archive_date": archive_date,
                    "owner_node_id": entry.owner_node_id,
                    "producer_url": entry.producer_url,
                    "lease_seconds": round(
                        max(0.0, entry.expires_at - now),
                        3,
                    ),
                }
        return None

    def claim(
        self,
        quota_scope: str,
        feed_id: str,
        archive_date: date,
        *,
        owner_node_id: str,
        producer_url: str,
        requester_address: str = "",
        allow_multihomed_self: bool = False,
    ) -> dict[str, Any]:
        scope = self._validate_key(quota_scope, feed_id)
        owner = self._validate_node_id(owner_node_id)
        producer = normalize_peer_url(producer_url)
        self._validate_requester(
            producer,
            requester_address,
            allow_multihomed_self=allow_multihomed_self,
        )
        key = (scope, feed_id, archive_date.isoformat())
        with self._lock:
            now = self._clock()
            self._cleanup_locked(now)
            existing = self._entries.get(key)
            if existing is not None:
                value = self._payload_locked(key, now)
                value["granted"] = False
                return value
            # Broadcastify requested one globally sequential archive stream.
            # Completed result records may coexist, but a different active
            # feed/day must release its renewable lease before another website
            # producer can start, even when a different authorized account is
            # selected.
            if any(entry.state == "active" for entry in self._entries.values()):
                value = self._payload_locked(key, now)
                value["granted"] = False
                return value
            if len(self._entries) >= self.maximum_entries:
                raise LanSyncError(
                    "The LAN acquisition queue is full; retry after older leases expire."
                )
            token = secrets.token_urlsafe(32)
            self._entries[key] = _LanQueueEntry(
                state="active",
                owner_node_id=owner,
                producer_url=producer,
                lease_token=token,
                expires_at=now + self.lease_seconds,
            )
            value = self._payload_locked(key, now)
            value["granted"] = True
            value["lease_token"] = token
            return value

    def renew(
        self,
        quota_scope: str,
        feed_id: str,
        archive_date: date,
        *,
        lease_token: str,
    ) -> dict[str, Any]:
        scope = self._validate_key(quota_scope, feed_id)
        key = (scope, feed_id, archive_date.isoformat())
        with self._lock:
            now = self._clock()
            self._cleanup_locked(now)
            entry = self._authorized_active_entry(key, lease_token)
            entry.expires_at = now + self.lease_seconds
            return self._payload_locked(key, now)

    def finish(
        self,
        quota_scope: str,
        feed_id: str,
        archive_date: date,
        *,
        lease_token: str,
        outcome: str,
        block_count: int = 0,
        blocks: Sequence[Mapping[str, Any]] = (),
        retry_after_seconds: float | None = None,
    ) -> dict[str, Any]:
        scope = self._validate_key(quota_scope, feed_id)
        if outcome not in {"complete", "quota_limited", "failed"}:
            raise LanSyncError("The LAN acquisition outcome is not valid.")
        if not 0 <= int(block_count) <= MAX_BLOCKS_PER_DAY:
            raise LanSyncError("The LAN acquisition block count is not valid.")
        retry_delay: float | None = None
        if retry_after_seconds is not None:
            try:
                retry_delay = float(retry_after_seconds)
            except (TypeError, ValueError) as exc:
                raise LanSyncError(
                    "The LAN acquisition retry delay is not valid."
                ) from exc
            if (
                outcome != "quota_limited"
                or not math.isfinite(retry_delay)
                or not 0.0 <= retry_delay <= 24 * 60 * 60.0
            ):
                raise LanSyncError(
                    "The LAN acquisition retry delay is not valid."
                )
        completion_blocks: tuple[ArchiveBlock, ...] = ()
        raw_blocks = tuple(blocks)
        if outcome == "complete":
            if len(raw_blocks) != int(block_count):
                raise LanSyncError(
                    "The completed LAN acquisition manifest does not match "
                    "its block count."
                )
            parsed_blocks: list[ArchiveBlock] = []
            names: set[str] = set()
            for value in raw_blocks:
                if not isinstance(value, Mapping):
                    raise LanSyncError(
                        "The completed LAN acquisition manifest is not valid."
                    )
                try:
                    block = ArchiveBlock.from_mapping(
                        value,
                        expected_feed_id=feed_id,
                        expected_date=archive_date,
                    )
                except (TypeError, ValueError) as exc:
                    raise LanSyncError(
                        "The completed LAN acquisition manifest is not valid."
                    ) from exc
                if block.filename in names:
                    raise LanSyncError(
                        "The completed LAN acquisition manifest contains "
                        "a duplicate block."
                    )
                names.add(block.filename)
                parsed_blocks.append(block)
            completion_blocks = tuple(
                sorted(parsed_blocks, key=lambda value: value.filename)
            )
        elif raw_blocks:
            raise LanSyncError(
                "Only a completed LAN acquisition may publish a block manifest."
            )
        key = (scope, feed_id, archive_date.isoformat())
        with self._lock:
            now = self._clock()
            self._cleanup_locked(now)
            entry = self._authorized_active_entry(key, lease_token)
            if outcome == "failed":
                del self._entries[key]
                return self._payload_locked(key, now)
            entry.state = outcome
            entry.lease_token = ""
            entry.block_count = int(block_count)
            current_date = self._today()
            entry.rolling = bool(
                outcome == "complete"
                and current_date - timedelta(days=1)
                <= archive_date
                <= current_date
            )
            entry.blocks = completion_blocks
            if outcome == "quota_limited" and retry_delay is not None:
                result_lifetime = max(5.0, retry_delay)
            else:
                result_lifetime = (
                    self.rolling_result_seconds
                    if entry.rolling
                    else self.result_seconds
                )
            entry.expires_at = now + result_lifetime
            return self._payload_locked(key, now)

    def _authorized_active_entry(
        self,
        key: tuple[str, str, str],
        lease_token: str,
    ) -> _LanQueueEntry:
        entry = self._entries.get(key)
        supplied = str(lease_token or "")
        if (
            entry is None
            or entry.state != "active"
            or not supplied
            or not hmac.compare_digest(entry.lease_token, supplied)
        ):
            raise PermissionError("The LAN acquisition lease is missing or expired.")
        return entry

    def _payload_locked(
        self,
        key: tuple[str, str, str],
        now: float,
    ) -> dict[str, Any]:
        scope, feed_id, archive_date = key
        entry = self._entries.get(key)
        value: dict[str, Any] = {
            "protocol": LAN_PROTOCOL,
            "quota_scope": scope,
            "feed_id": feed_id,
            "archive_date": archive_date,
            "state": "available",
            "producer_url": "",
            "owner_node_id": "",
            "lease_seconds": 0.0,
            "block_count": 0,
            "rolling": False,
            "blocks": [],
            "global_busy": False,
        }
        if entry is None:
            active = [
                candidate
                for candidate in self._entries.values()
                if candidate.state == "active"
            ]
            if active:
                value["global_busy"] = True
                value["lease_seconds"] = round(
                    max(max(0.0, candidate.expires_at - now) for candidate in active),
                    3,
                )
            return value
        value.update(
            {
                "state": entry.state,
                "producer_url": entry.producer_url,
                "owner_node_id": entry.owner_node_id,
                "lease_seconds": round(max(0.0, entry.expires_at - now), 3),
                "block_count": entry.block_count,
                "rolling": entry.rolling,
                "blocks": [block.to_dict() for block in entry.blocks],
            }
        )
        return value

    def _cleanup_locked(self, now: float) -> None:
        expired = [
            key
            for key, entry in self._entries.items()
            if entry.expires_at <= now
        ]
        for key in expired:
            self._entries.pop(key, None)

    @staticmethod
    def _validate_key(quota_scope: str, feed_id: str) -> str:
        scope = str(quota_scope or "default").strip()
        if not LAN_QUEUE_SCOPE_PATTERN.fullmatch(scope):
            raise LanSyncError("The LAN quota scope is not valid.")
        if not str(feed_id or "").isdigit():
            raise LanSyncError("A numeric feed ID is required.")
        return scope

    @staticmethod
    def _validate_node_id(value: str) -> str:
        node_id = str(value or "").strip()
        if not LAN_QUEUE_NODE_PATTERN.fullmatch(node_id):
            raise LanSyncError("The LAN producer node identity is not valid.")
        return node_id

    @staticmethod
    def _validate_requester(
        producer_url: str,
        requester_address: str,
        *,
        allow_multihomed_self: bool = False,
    ) -> None:
        if not requester_address:
            return
        try:
            producer = ipaddress.ip_address(urlparse(producer_url).hostname or "")
            requester = ipaddress.ip_address(requester_address)
            if isinstance(requester, ipaddress.IPv6Address) and requester.ipv4_mapped:
                requester = requester.ipv4_mapped
        except ValueError as exc:
            raise LanSyncError("The LAN producer address is not valid.") from exc
        if producer == requester or (producer.is_loopback and requester.is_loopback):
            return
        if allow_multihomed_self and (
            requester.is_private
            or requester.is_link_local
            or requester.is_loopback
        ):
            return
        raise PermissionError(
            "A LAN client may claim work only for its own reachable producer node."
        )


@dataclass
class _LanProcessingEntry:
    state: str
    owner_node_id: str
    producer_url: str
    lease_token: str
    expires_at: float
    artifact_count: int = 0


class LanProcessingQueue:
    """Bounded renewable leases for model-specific transcript work."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        lease_seconds: float = LAN_PROCESSING_LEASE_SECONDS,
        result_seconds: float = LAN_QUEUE_RESULT_SECONDS,
        maximum_entries: int = MAX_LAN_QUEUE_ENTRIES,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.enabled = bool(enabled)
        self.lease_seconds = min(15 * 60.0, max(30.0, float(lease_seconds)))
        self.result_seconds = min(
            24 * 60 * 60.0,
            max(5 * 60.0, float(result_seconds)),
        )
        self.maximum_entries = min(
            MAX_LAN_QUEUE_ENTRIES,
            max(16, int(maximum_entries)),
        )
        self._clock = clock
        self._entries: dict[tuple[str, str, str], _LanProcessingEntry] = {}
        self._lock = threading.RLock()

    def status(
        self,
        processing_fingerprint: str,
        feed_id: str,
        archive_date: date,
    ) -> dict[str, Any]:
        fingerprint = self._validate_key(processing_fingerprint, feed_id)
        key = (fingerprint, feed_id, archive_date.isoformat())
        with self._lock:
            now = self._clock()
            self._cleanup_locked(now)
            return self._payload_locked(key, now)

    def active_activities(self, *, limit: int = 32) -> list[dict[str, Any]]:
        """Return bounded non-secret model/day leases for status surfaces."""

        bounded = min(32, max(1, int(limit)))
        with self._lock:
            now = self._clock()
            self._cleanup_locked(now)
            values = [
                {
                    "feed_id": feed_id,
                    "archive_date": archive_date,
                    "owner_node_id": entry.owner_node_id,
                    "producer_url": entry.producer_url,
                    "lease_seconds": round(
                        max(0.0, entry.expires_at - now),
                        3,
                    ),
                }
                for (_fingerprint, feed_id, archive_date), entry
                in self._entries.items()
                if entry.state == "active"
            ]
        return sorted(
            values,
            key=lambda value: (
                str(value["feed_id"]),
                str(value["archive_date"]),
                str(value["owner_node_id"]),
            ),
        )[:bounded]

    def claim(
        self,
        processing_fingerprint: str,
        feed_id: str,
        archive_date: date,
        *,
        owner_node_id: str,
        producer_url: str,
        requester_address: str = "",
        allow_multihomed_self: bool = False,
    ) -> dict[str, Any]:
        fingerprint = self._validate_key(processing_fingerprint, feed_id)
        owner = LanAcquisitionQueue._validate_node_id(owner_node_id)
        producer = normalize_peer_url(producer_url)
        LanAcquisitionQueue._validate_requester(
            producer,
            requester_address,
            allow_multihomed_self=allow_multihomed_self,
        )
        key = (fingerprint, feed_id, archive_date.isoformat())
        with self._lock:
            now = self._clock()
            self._cleanup_locked(now)
            if key in self._entries:
                value = self._payload_locked(key, now)
                value["granted"] = False
                return value
            if len(self._entries) >= self.maximum_entries:
                raise LanSyncError(
                    "The LAN processing queue is full; retry after older leases expire."
                )
            token = secrets.token_urlsafe(32)
            self._entries[key] = _LanProcessingEntry(
                state="active",
                owner_node_id=owner,
                producer_url=producer,
                lease_token=token,
                expires_at=now + self.lease_seconds,
            )
            value = self._payload_locked(key, now)
            value["granted"] = True
            value["lease_token"] = token
            return value

    def renew(
        self,
        processing_fingerprint: str,
        feed_id: str,
        archive_date: date,
        *,
        lease_token: str,
    ) -> dict[str, Any]:
        fingerprint = self._validate_key(processing_fingerprint, feed_id)
        key = (fingerprint, feed_id, archive_date.isoformat())
        with self._lock:
            now = self._clock()
            self._cleanup_locked(now)
            entry = self._authorized_active_entry(key, lease_token)
            entry.expires_at = now + self.lease_seconds
            return self._payload_locked(key, now)

    def finish(
        self,
        processing_fingerprint: str,
        feed_id: str,
        archive_date: date,
        *,
        lease_token: str,
        outcome: str,
        artifact_count: int = 0,
    ) -> dict[str, Any]:
        fingerprint = self._validate_key(processing_fingerprint, feed_id)
        if outcome not in {"complete", "failed"}:
            raise LanSyncError("The LAN processing outcome is not valid.")
        if not 0 <= int(artifact_count) <= MAX_TRANSCRIPT_ARTIFACTS_PER_DAY:
            raise LanSyncError("The LAN transcript artifact count is not valid.")
        if outcome == "complete" and int(artifact_count) < 3:
            raise LanSyncError(
                "Completed LAN transcript work must publish audio, JSON, and text."
            )
        key = (fingerprint, feed_id, archive_date.isoformat())
        with self._lock:
            now = self._clock()
            self._cleanup_locked(now)
            entry = self._authorized_active_entry(key, lease_token)
            if outcome == "failed":
                del self._entries[key]
                return self._payload_locked(key, now)
            entry.state = "complete"
            entry.lease_token = ""
            entry.artifact_count = int(artifact_count)
            entry.expires_at = now + self.result_seconds
            return self._payload_locked(key, now)

    def _authorized_active_entry(
        self,
        key: tuple[str, str, str],
        lease_token: str,
    ) -> _LanProcessingEntry:
        entry = self._entries.get(key)
        supplied = str(lease_token or "")
        if (
            entry is None
            or entry.state != "active"
            or not supplied
            or not hmac.compare_digest(entry.lease_token, supplied)
        ):
            raise PermissionError("The LAN processing lease is missing or expired.")
        return entry

    def _payload_locked(
        self,
        key: tuple[str, str, str],
        now: float,
    ) -> dict[str, Any]:
        fingerprint, feed_id, archive_date = key
        entry = self._entries.get(key)
        value: dict[str, Any] = {
            "protocol": LAN_PROTOCOL,
            "processing_fingerprint": fingerprint,
            "feed_id": feed_id,
            "archive_date": archive_date,
            "state": "available",
            "producer_url": "",
            "owner_node_id": "",
            "lease_seconds": 0.0,
            "artifact_count": 0,
        }
        if entry is not None:
            value.update(
                {
                    "state": entry.state,
                    "producer_url": entry.producer_url,
                    "owner_node_id": entry.owner_node_id,
                    "lease_seconds": round(
                        max(0.0, entry.expires_at - now),
                        3,
                    ),
                    "artifact_count": entry.artifact_count,
                }
            )
        return value

    def _cleanup_locked(self, now: float) -> None:
        for key in [
            value
            for value, entry in self._entries.items()
            if entry.expires_at <= now
        ]:
            self._entries.pop(key, None)

    @staticmethod
    def _validate_key(processing_fingerprint: str, feed_id: str) -> str:
        fingerprint = str(processing_fingerprint or "").strip().lower()
        if not PROCESSING_FINGERPRINT_PATTERN.fullmatch(fingerprint):
            raise LanSyncError("The LAN processing fingerprint is not valid.")
        if not str(feed_id or "").isdigit():
            raise LanSyncError("A numeric feed ID is required.")
        return fingerprint


class ArchiveHashCache:
    """Bounded, thread-safe hash cache keyed by immutable file metadata."""

    def __init__(self, maximum_entries: int = 4_096) -> None:
        self.maximum_entries = max(32, maximum_entries)
        self._values: OrderedDict[tuple[str, int, int], str] = OrderedDict()
        self._lock = threading.Lock()

    def sha256(self, path: Path) -> str:
        initial = path.stat()
        key = (str(path.resolve()), initial.st_size, initial.st_mtime_ns)
        with self._lock:
            cached = self._values.get(key)
            if cached:
                self._values.move_to_end(key)
                return cached
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        final = path.stat()
        if (
            initial.st_size != final.st_size
            or initial.st_mtime_ns != final.st_mtime_ns
        ):
            raise LanSyncError(f"{path.name} changed while its hash was calculated.")
        value = digest.hexdigest()
        with self._lock:
            self._values[key] = value
            self._values.move_to_end(key)
            while len(self._values) > self.maximum_entries:
                self._values.popitem(last=False)
        return value


class LanArchiveCatalog:
    """Read-only catalog of original archive blocks under one library root."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        enabled: bool,
        sync_key: str = "",
        peer_urls: Sequence[str] = (),
        node_id: str | None = None,
        acquisition_queue: LanAcquisitionQueue | None = None,
        processing_queue: LanProcessingQueue | None = None,
        queue_enabled: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.enabled = bool(enabled)
        self.sync_key = str(sync_key or "")
        self.peer_urls = normalize_peer_urls(peer_urls)
        self.node_id = node_id or secrets.token_hex(12)
        self.hashes = ArchiveHashCache()
        self.discovery_available = False
        self.discovery_error = ""
        self.acquisition_queue = acquisition_queue or LanAcquisitionQueue(
            enabled=self.enabled and queue_enabled,
            lease_seconds=environment_float(
                "BROADCASTIFY_LAN_QUEUE_LEASE_SECONDS",
                LAN_QUEUE_LEASE_SECONDS,
                minimum=30.0,
                maximum=300.0,
            ),
            result_seconds=environment_float(
                "BROADCASTIFY_LAN_QUEUE_RESULT_SECONDS",
                LAN_QUEUE_RESULT_SECONDS,
                minimum=5 * 60.0,
                maximum=24 * 60 * 60.0,
            ),
            rolling_result_seconds=environment_float(
                "BROADCASTIFY_LAN_QUEUE_ROLLING_RESULT_SECONDS",
                LAN_QUEUE_ROLLING_RESULT_SECONDS,
                minimum=30.0,
                maximum=30 * 60.0,
            ),
        )
        self.processing_queue = processing_queue or LanProcessingQueue(
            enabled=self.enabled and queue_enabled,
            lease_seconds=environment_float(
                "BROADCASTIFY_LAN_PROCESSING_LEASE_SECONDS",
                LAN_PROCESSING_LEASE_SECONDS,
                minimum=30.0,
                maximum=15 * 60.0,
            ),
            result_seconds=environment_float(
                "BROADCASTIFY_LAN_PROCESSING_RESULT_SECONDS",
                LAN_QUEUE_RESULT_SECONDS,
                minimum=5 * 60.0,
                maximum=24 * 60 * 60.0,
            ),
        )

    def authorized(self, supplied_key: str) -> bool:
        if not self.sync_key:
            return True
        return hmac.compare_digest(self.sync_key, supplied_key)

    def info(self) -> dict[str, Any]:
        return {
            "protocol": LAN_PROTOCOL,
            "node_id": self.node_id,
            "sharing": self.enabled,
            "key_required": bool(self.sync_key),
            "peers": list(self.peer_urls),
            "discovery_available": self.discovery_available,
            "acquisition_queue_available": bool(
                self.enabled and self.acquisition_queue.enabled
            ),
            "processing_queue_available": bool(
                self.enabled and self.processing_queue.enabled
            ),
            "activity": {
                "acquisition": self.acquisition_queue.active_activity(),
                "processing": self.processing_queue.active_activities(),
            },
        }

    def feed_dates(self, feed_id: str) -> list[date]:
        """List retained dates for one feed without exposing unrelated paths."""

        if not feed_id.isdigit():
            raise LanSyncError("A numeric feed ID is required.")
        feed_dir = (self.output_dir / feed_id).resolve()
        try:
            feed_dir.relative_to(self.output_dir)
        except ValueError as exc:
            raise LanSyncError("The feed is outside the archive library.") from exc
        if (
            not feed_dir.is_dir()
            or feed_dir.is_symlink()
            or feed_dir.parent != self.output_dir
        ):
            return []
        values: list[date] = []
        for candidate in sorted(feed_dir.iterdir(), key=lambda value: value.name):
            if (
                len(values) >= MAX_FEED_DAYS
                or candidate.is_symlink()
                or not candidate.is_dir()
                or not re.fullmatch(r"\d{8}", candidate.name)
            ):
                continue
            try:
                archive_date = datetime.strptime(candidate.name, "%Y%m%d").date()
            except ValueError:
                continue
            if candidate.resolve().parent != feed_dir:
                continue
            complete = complete_cached_archive_day(
                candidate,
                feed_id,
                archive_date,
            )
            has_source = any(
                RAW_ARCHIVE_PATTERN.fullmatch(path.name)
                and not path.is_symlink()
                and path.is_file()
                for path in candidate.glob("*.mp3")
            )
            transcript_dir = candidate / "transcripts"
            has_derived_transcript = (
                transcript_dir.is_dir()
                and not transcript_dir.is_symlink()
                and any(transcript_dir.glob("*.json"))
            )
            if complete is not None or has_source or has_derived_transcript:
                values.append(archive_date)
        return values

    def inventory(self, feed_id: str, archive_date: date) -> list[ArchiveBlock]:
        if not feed_id.isdigit():
            raise LanSyncError("A numeric feed ID is required.")
        day_dir = self._day_directory(feed_id, archive_date)
        if not day_dir.is_dir():
            return []
        blocks: list[ArchiveBlock] = []
        for path in sorted(day_dir.glob("*.mp3")):
            match = RAW_ARCHIVE_PATTERN.fullmatch(path.name)
            if (
                match is None
                or match.group("feed_id") != feed_id
                or not _archive_stamp_matches_day(match.group("stamp"), archive_date)
                or path.is_symlink()
            ):
                continue
            resolved = path.resolve()
            if resolved.parent != day_dir or not resolved.is_file():
                continue
            stat = resolved.stat()
            if not 0 < stat.st_size <= MAX_ARCHIVE_BLOCK_BYTES:
                continue
            archive_id, listing_prefix, archive_identities = _block_archive_identities(
                day_dir,
                feed_id,
                path.name,
            )
            blocks.append(
                ArchiveBlock(
                    feed_id=feed_id,
                    archive_date=archive_date.isoformat(),
                    filename=path.name,
                    size=stat.st_size,
                    sha256=self.hashes.sha256(resolved),
                    modified_ns=stat.st_mtime_ns,
                    archive_id=archive_id,
                    listing_prefix=listing_prefix,
                    archive_identities=archive_identities,
                )
            )
            if len(blocks) >= MAX_BLOCKS_PER_DAY:
                break
        return blocks

    def completion_inventory(
        self,
        feed_id: str,
        archive_date: date,
    ) -> tuple[bool, tuple[ArchiveBlock, ...]]:
        """Return a durable exact day-completion proof, including empty days."""

        day_dir = self._day_directory(feed_id, archive_date)
        complete = complete_cached_archive_day(day_dir, feed_id, archive_date)
        if complete is None:
            return False, ()
        files, expected_count = complete
        by_name = {value.filename: value for value in self.inventory(feed_id, archive_date)}
        blocks = tuple(by_name[path.name] for path in files if path.name in by_name)
        if len(blocks) != expected_count or len(blocks) != len(files):
            return False, ()
        return True, blocks

    def resolve_block(
        self,
        feed_id: str,
        archive_date: date,
        filename: str,
    ) -> tuple[Path, ArchiveBlock]:
        if Path(filename).name != filename:
            raise LanSyncError("The archive block name is not valid.")
        match = RAW_ARCHIVE_PATTERN.fullmatch(filename)
        if (
            not feed_id.isdigit()
            or match is None
            or match.group("feed_id") != feed_id
            or not _archive_stamp_matches_day(match.group("stamp"), archive_date)
        ):
            raise LanSyncError("The archive block does not match that feed and date.")
        day_dir = self._day_directory(feed_id, archive_date)
        candidate = day_dir / filename
        if candidate.is_symlink():
            raise FileNotFoundError(filename)
        path = candidate.resolve()
        if path.parent != day_dir or not path.is_file():
            raise FileNotFoundError(filename)
        stat = path.stat()
        if not 0 < stat.st_size <= MAX_ARCHIVE_BLOCK_BYTES:
            raise FileNotFoundError(filename)
        archive_id, listing_prefix, archive_identities = _block_archive_identities(
            day_dir,
            feed_id,
            filename,
        )
        block = ArchiveBlock(
            feed_id=feed_id,
            archive_date=archive_date.isoformat(),
            filename=filename,
            size=stat.st_size,
            sha256=self.hashes.sha256(path),
            modified_ns=stat.st_mtime_ns,
            archive_id=archive_id,
            listing_prefix=listing_prefix,
            archive_identities=archive_identities,
        )
        return path, block

    def transcript_inventory(
        self,
        feed_id: str,
        archive_date: date,
        processing_fingerprint: str,
    ) -> list[TranscriptArtifact]:
        fingerprint = LanProcessingQueue._validate_key(
            processing_fingerprint,
            feed_id,
        )
        day_dir = self._day_directory(feed_id, archive_date)
        roots: list[tuple[Path, Path, str]] = [
            (day_dir, day_dir / "transcripts", ""),
        ]
        variants_root = day_dir / DERIVED_VARIANTS_DIRECTORY
        fingerprint_root = variants_root / fingerprint
        if (
            variants_root.is_dir()
            and not variants_root.is_symlink()
            and variants_root.resolve().parent == day_dir
            and fingerprint_root.is_dir()
            and not fingerprint_root.is_symlink()
            and fingerprint_root.resolve().parent == variants_root.resolve()
        ):
            for candidate in sorted(
                fingerprint_root.iterdir(),
                key=lambda value: value.name,
            ):
                if (
                    not PROCESSING_FINGERPRINT_PATTERN.fullmatch(candidate.name)
                    or candidate.is_symlink()
                    or not candidate.is_dir()
                    or candidate.resolve().parent != fingerprint_root.resolve()
                ):
                    continue
                roots.append((candidate.resolve(), candidate / "transcripts", candidate.name))

        artifacts: list[TranscriptArtifact] = []
        selected_audio: set[str] = set()
        for artifact_root, transcript_dir, expected_audio_hash in roots:
            values = self._transcript_inventory_from_root(
                day_dir,
                artifact_root,
                transcript_dir,
                feed_id,
                archive_date,
                fingerprint,
                expected_audio_hash=expected_audio_hash,
            )
            grouped: dict[str, list[TranscriptArtifact]] = {}
            for value in values:
                grouped.setdefault(value.audio_filename, []).append(value)
            for audio_filename, group in grouped.items():
                # The conventional day paths remain the preferred result. A
                # model/audio variant is retained and served only when that
                # filename was not already represented by an earlier safe root.
                if audio_filename in selected_audio:
                    continue
                if len(artifacts) + len(group) > MAX_TRANSCRIPT_ARTIFACTS_PER_DAY:
                    return artifacts
                selected_audio.add(audio_filename)
                artifacts.extend(group)
        return artifacts

    def _transcript_inventory_from_root(
        self,
        day_dir: Path,
        artifact_root: Path,
        transcript_dir: Path,
        feed_id: str,
        archive_date: date,
        fingerprint: str,
        *,
        expected_audio_hash: str = "",
    ) -> list[TranscriptArtifact]:
        if (
            not artifact_root.is_dir()
            or artifact_root.is_symlink()
            or not transcript_dir.is_dir()
            or transcript_dir.is_symlink()
            or transcript_dir.resolve().parent != artifact_root.resolve()
        ):
            return []
        artifacts: list[TranscriptArtifact] = []
        for json_path in sorted(transcript_dir.glob("*.json")):
            if len(artifacts) + 4 > MAX_TRANSCRIPT_ARTIFACTS_PER_DAY:
                break
            if json_path.is_symlink() or json_path.resolve().parent != transcript_dir:
                continue
            try:
                json_stat = json_path.stat()
                if not 0 < json_stat.st_size <= MAX_TRANSCRIPT_ARTIFACT_BYTES:
                    continue
                payload = json.loads(json_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, ValueError):
                continue
            if not isinstance(payload, Mapping):
                continue
            audio_filename = str(payload.get("audio_file") or "")
            audio_sha256 = str(payload.get("audio_sha256") or "").lower()
            if (
                str(payload.get("processing_fingerprint") or "").lower()
                != fingerprint
                or not _valid_day_audio_filename(
                    audio_filename,
                    feed_id,
                    archive_date,
                )
                or not re.fullmatch(r"[0-9a-f]{64}", audio_sha256)
                or (expected_audio_hash and audio_sha256 != expected_audio_hash)
                or json_path.name != f"{Path(audio_filename).stem}.json"
            ):
                continue
            audio_path = artifact_root / audio_filename
            if (
                audio_path.is_symlink()
                or not audio_path.is_file()
                or audio_path.resolve().parent != artifact_root.resolve()
            ):
                continue
            try:
                if self.hashes.sha256(audio_path) != audio_sha256:
                    continue
            except OSError:
                continue
            text_path = transcript_dir / f"{Path(audio_filename).stem}.txt"
            if (
                text_path.is_symlink()
                or not text_path.is_file()
                or text_path.resolve().parent != transcript_dir
            ):
                continue
            try:
                text_stat = text_path.stat()
                if not 0 <= text_stat.st_size <= MAX_TRANSCRIPT_ARTIFACT_BYTES:
                    continue
                rendered_hash = str(
                    payload.get("rendered_text_sha256") or ""
                ).lower()
                if (
                    not re.fullmatch(r"[0-9a-f]{64}", rendered_hash)
                    or self.hashes.sha256(text_path) != rendered_hash
                ):
                    continue
                artifact_set = [
                    TranscriptArtifact(
                        feed_id=feed_id,
                        archive_date=archive_date.isoformat(),
                        audio_filename=audio_filename,
                        audio_sha256=audio_sha256,
                        processing_fingerprint=fingerprint,
                        filename=audio_filename,
                        kind="audio",
                        size=audio_path.stat().st_size,
                        sha256=audio_sha256,
                        modified_ns=audio_path.stat().st_mtime_ns,
                    ),
                    TranscriptArtifact(
                        feed_id=feed_id,
                        archive_date=archive_date.isoformat(),
                        audio_filename=audio_filename,
                        audio_sha256=audio_sha256,
                        processing_fingerprint=fingerprint,
                        filename=json_path.name,
                        kind="json",
                        size=json_stat.st_size,
                        sha256=self.hashes.sha256(json_path),
                        modified_ns=json_stat.st_mtime_ns,
                    ),
                    TranscriptArtifact(
                        feed_id=feed_id,
                        archive_date=archive_date.isoformat(),
                        audio_filename=audio_filename,
                        audio_sha256=audio_sha256,
                        processing_fingerprint=fingerprint,
                        filename=text_path.name,
                        kind="text",
                        size=text_stat.st_size,
                        sha256=rendered_hash,
                        modified_ns=text_stat.st_mtime_ns,
                    ),
                ]
                if audio_filename == f"combined_{feed_id}_{archive_date:%Y%m%d}.mp3":
                    manifest_path = (
                        artifact_root / f"{Path(audio_filename).stem}.manifest.json"
                    )
                    source_files = [
                        day_dir / block.filename
                        for block in self.inventory(feed_id, archive_date)
                    ]
                    if (
                        manifest_path.is_symlink()
                        or not manifest_path.is_file()
                        or manifest_path.resolve().parent != artifact_root.resolve()
                        or not combined_output_is_current(
                            audio_path,
                            manifest_path,
                            source_files,
                        )
                    ):
                        continue
                    manifest_stat = manifest_path.stat()
                    if not 0 < manifest_stat.st_size <= MAX_TRANSCRIPT_ARTIFACT_BYTES:
                        continue
                    artifact_set.append(
                        TranscriptArtifact(
                            feed_id=feed_id,
                            archive_date=archive_date.isoformat(),
                            audio_filename=audio_filename,
                            audio_sha256=audio_sha256,
                            processing_fingerprint=fingerprint,
                            filename=manifest_path.name,
                            kind="manifest",
                            size=manifest_stat.st_size,
                            sha256=self.hashes.sha256(manifest_path),
                            modified_ns=manifest_stat.st_mtime_ns,
                        )
                    )
                artifacts.extend(artifact_set)
            except OSError:
                continue
        return artifacts

    def resolve_transcript_artifact(
        self,
        feed_id: str,
        archive_date: date,
        processing_fingerprint: str,
        filename: str,
    ) -> tuple[Path, TranscriptArtifact]:
        if Path(filename).name != filename:
            raise LanSyncError("The transcript artifact name is not valid.")
        for artifact in self.transcript_inventory(
            feed_id,
            archive_date,
            processing_fingerprint,
        ):
            if artifact.filename != filename:
                continue
            day_dir = self._day_directory(feed_id, archive_date)
            roots = [
                day_dir,
                day_dir
                / DERIVED_VARIANTS_DIRECTORY
                / artifact.processing_fingerprint
                / artifact.audio_sha256,
            ]
            for root in roots:
                candidate = (
                    root / filename
                    if artifact.kind in {"audio", "manifest"}
                    else root / "transcripts" / filename
                )
                if candidate.is_symlink():
                    continue
                path = candidate.resolve()
                expected_parent = (
                    root.resolve()
                    if artifact.kind in {"audio", "manifest"}
                    else (root / "transcripts").resolve()
                )
                if path.parent != expected_parent or not path.is_file():
                    continue
                try:
                    if (
                        path.stat().st_size == artifact.size
                        and self.hashes.sha256(path) == artifact.sha256
                    ):
                        return path, artifact
                except OSError:
                    continue
        raise FileNotFoundError(filename)

    def _day_directory(self, feed_id: str, archive_date: date) -> Path:
        candidate = (
            self.output_dir / feed_id / archive_date.strftime("%Y%m%d")
        ).resolve()
        try:
            candidate.relative_to(self.output_dir)
        except ValueError as exc:
            raise LanSyncError("The archive day is outside the library.") from exc
        return candidate


_discovery_cache_lock = threading.Lock()
_discovery_cache: tuple[float, tuple[str, ...]] = (0.0, ())


def _usable_directed_broadcast(
    address: ipaddress.IPv4Address,
    broadcast: ipaddress.IPv4Address,
) -> bool:
    return (
        address != broadcast
        and not address.is_loopback
        and not address.is_unspecified
        and not address.is_multicast
        and not broadcast.is_loopback
        and not broadcast.is_unspecified
        and not broadcast.is_multicast
        and (address.is_private or address.is_link_local)
    )


def _windows_directed_broadcasts() -> tuple[str, ...]:
    """Read IPv4 interface masks without parsing localized command output."""

    class MibIpAddressRow(ctypes.Structure):
        _fields_ = [
            ("address", ctypes.c_uint32),
            ("interface_index", ctypes.c_uint32),
            ("mask", ctypes.c_uint32),
            ("broadcast_flag", ctypes.c_uint32),
            ("reassembly_size", ctypes.c_uint32),
            ("unused", ctypes.c_uint16),
            ("address_type", ctypes.c_uint16),
        ]

    try:
        get_table = ctypes.WinDLL("iphlpapi").GetIpAddrTable
    except (AttributeError, OSError):
        return ()
    get_table.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.c_int,
    ]
    get_table.restype = ctypes.c_ulong
    size = ctypes.c_ulong(0)
    result = int(get_table(None, ctypes.byref(size), 0))
    if result not in {0, 122} or size.value < ctypes.sizeof(ctypes.c_uint32):
        return ()
    buffer = ctypes.create_string_buffer(size.value)
    if int(get_table(buffer, ctypes.byref(size), 0)) != 0:
        return ()
    count = int(ctypes.c_uint32.from_buffer_copy(buffer.raw[:4]).value)
    row_size = ctypes.sizeof(MibIpAddressRow)
    broadcasts: list[str] = []
    for index in range(min(count, 256)):
        offset = 4 + (index * row_size)
        if offset + row_size > size.value:
            break
        row = MibIpAddressRow.from_buffer_copy(buffer.raw[offset : offset + row_size])
        try:
            address = ipaddress.IPv4Address(
                int(row.address).to_bytes(4, byteorder="little")
            )
            mask = ipaddress.IPv4Address(
                int(row.mask).to_bytes(4, byteorder="little")
            )
            network = ipaddress.IPv4Network(f"{address}/{mask}", strict=False)
            broadcast = network.broadcast_address
        except (ipaddress.AddressValueError, ipaddress.NetmaskValueError, ValueError):
            continue
        if _usable_directed_broadcast(address, broadcast):
            broadcasts.append(str(broadcast))
    return tuple(dict.fromkeys(broadcasts))


def _posix_directed_broadcasts() -> tuple[str, ...]:
    """Read getifaddrs broadcast addresses on Linux, macOS, and BSD."""

    class IfAddrs(ctypes.Structure):
        pass

    IfAddrsPointer = ctypes.POINTER(IfAddrs)
    IfAddrs._fields_ = [
        ("next", IfAddrsPointer),
        ("name", ctypes.c_char_p),
        ("flags", ctypes.c_uint),
        ("address", ctypes.c_void_p),
        ("netmask", ctypes.c_void_p),
        ("broadcast_or_destination", ctypes.c_void_p),
        ("data", ctypes.c_void_p),
    ]
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        getifaddrs = libc.getifaddrs
        freeifaddrs = libc.freeifaddrs
    except (AttributeError, OSError):
        return ()
    getifaddrs.argtypes = [ctypes.POINTER(IfAddrsPointer)]
    getifaddrs.restype = ctypes.c_int
    freeifaddrs.argtypes = [IfAddrsPointer]
    head = IfAddrsPointer()
    if getifaddrs(ctypes.byref(head)) != 0:
        return ()
    broadcasts: list[str] = []
    try:
        current = head
        for _index in range(2_048):
            if not current:
                break
            entry = current.contents
            current = entry.next
            if (
                not entry.address
                or not entry.broadcast_or_destination
                or not (entry.flags & 0x2)  # IFF_BROADCAST
            ):
                continue
            address_bytes = ctypes.string_at(entry.address, 16)
            broadcast_bytes = ctypes.string_at(entry.broadcast_or_destination, 16)
            if sys.platform.startswith(("darwin", "freebsd", "openbsd", "netbsd")):
                family = address_bytes[1]
                broadcast_family = broadcast_bytes[1]
            else:
                family = int.from_bytes(address_bytes[:2], byteorder=sys.byteorder)
                broadcast_family = int.from_bytes(
                    broadcast_bytes[:2],
                    byteorder=sys.byteorder,
                )
            if family != socket.AF_INET or broadcast_family != socket.AF_INET:
                continue
            address = ipaddress.IPv4Address(address_bytes[4:8])
            broadcast = ipaddress.IPv4Address(broadcast_bytes[4:8])
            if _usable_directed_broadcast(address, broadcast):
                broadcasts.append(str(broadcast))
    finally:
        freeifaddrs(head)
    return tuple(dict.fromkeys(broadcasts))


def local_ipv4_broadcasts() -> tuple[str, ...]:
    """Return real interface-directed broadcasts for one-hop discovery."""

    if os.name == "nt":
        return _windows_directed_broadcasts()
    if os.name == "posix":
        return _posix_directed_broadcasts()
    return ()


def discovery_destinations() -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                "255.255.255.255",
                LAN_MULTICAST_ADDRESS,
                *local_ipv4_broadcasts(),
            )
        )
    )


def discover_lan_peers(
    *,
    timeout: float = 0.45,
    port: int | None = None,
    destinations: Iterable[str] | None = None,
    cache_ttl: float = 60.0,
) -> tuple[str, ...]:
    """Best-effort, one-hop discovery with a short process-wide cache."""

    global _discovery_cache
    now = time.monotonic()
    with _discovery_cache_lock:
        cached_at, cached = _discovery_cache
        if cache_ttl > 0 and now - cached_at < cache_ttl:
            return cached
    nonce = secrets.token_urlsafe(12)
    request = LAN_DISCOVERY_MAGIC + nonce.encode("ascii")
    found: list[str] = []
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)
        sock.bind(("", 0))
        sock.settimeout(max(0.05, timeout))
        target_port = port or discovery_port()
        probe_destinations = (
            discovery_destinations() if destinations is None else destinations
        )
        for destination in probe_destinations:
            try:
                sock.sendto(request, (destination, target_port))
            except OSError:
                continue
        deadline = time.monotonic() + max(0.05, timeout)
        while time.monotonic() < deadline:
            sock.settimeout(max(0.01, deadline - time.monotonic()))
            try:
                payload, _source = sock.recvfrom(4_096)
            except socket.timeout:
                break
            except OSError:
                continue
            try:
                value = json.loads(payload.decode("utf-8"))
                if (
                    not isinstance(value, dict)
                    or value.get("protocol") != LAN_PROTOCOL
                    or value.get("nonce") != nonce
                ):
                    continue
                peer = normalize_peer_url(str(value.get("url") or ""))
            except (UnicodeDecodeError, ValueError, TypeError, json.JSONDecodeError):
                continue
            if peer not in found:
                found.append(peer)
                if len(found) >= MAX_LAN_PEERS:
                    break
    finally:
        sock.close()
    result = tuple(found)
    if cache_ttl > 0:
        with _discovery_cache_lock:
            _discovery_cache = (time.monotonic(), result)
    return result


class LanDiscoveryResponder:
    """Advertise an enabled read-only catalog to one trusted LAN hop."""

    def __init__(
        self,
        catalog: LanArchiveCatalog,
        url_for_remote: Callable[[str], str],
        *,
        port: int | None = None,
    ) -> None:
        self.catalog = catalog
        self.url_for_remote = url_for_remote
        self.port = port or discovery_port()
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> bool:
        if not self.catalog.enabled or self._thread is not None:
            return False
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("", self.port))
            try:
                membership = socket.inet_aton(LAN_MULTICAST_ADDRESS) + socket.inet_aton(
                    "0.0.0.0"
                )
                sock.setsockopt(
                    socket.IPPROTO_IP,
                    socket.IP_ADD_MEMBERSHIP,
                    membership,
                )
            except OSError:
                pass
            sock.settimeout(0.5)
        except OSError as exc:
            sock.close()
            self.catalog.discovery_error = str(exc)
            return False
        self._socket = sock
        self._thread = threading.Thread(
            target=self._run,
            name="radio-archive-lan-discovery",
            daemon=True,
        )
        self._thread.start()
        self.catalog.discovery_available = True
        self.catalog.discovery_error = ""
        return True

    def close(self) -> None:
        self._stop.set()
        sock = self._socket
        if sock is not None:
            sock.close()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=1)
        self._thread = None
        self._socket = None
        self.catalog.discovery_available = False

    def _run(self) -> None:
        assert self._socket is not None
        while not self._stop.is_set():
            try:
                payload, source = self._socket.recvfrom(1_024)
            except socket.timeout:
                continue
            except OSError:
                break
            if not payload.startswith(LAN_DISCOVERY_MAGIC):
                continue
            nonce_bytes = payload[len(LAN_DISCOVERY_MAGIC) :]
            if not 1 <= len(nonce_bytes) <= 128:
                continue
            try:
                nonce = nonce_bytes.decode("ascii")
                source_address = ipaddress.ip_address(source[0])
            except (UnicodeDecodeError, ValueError):
                continue
            if not (
                source_address.is_private
                or source_address.is_link_local
                or source_address.is_loopback
            ):
                continue
            try:
                advertised_url = normalize_peer_url(self.url_for_remote(source[0]))
            except (OSError, ValueError):
                continue
            response = json.dumps(
                {
                    "protocol": LAN_PROTOCOL,
                    "nonce": nonce,
                    "url": advertised_url,
                    "node_id": self.catalog.node_id,
                },
                separators=(",", ":"),
            ).encode("utf-8")
            try:
                self._socket.sendto(response, source)
            except OSError:
                continue


class LanArchiveSyncClient:
    """Copy blocks and coordinate one upstream downloader across LAN peers."""

    def __init__(
        self,
        *,
        enabled: bool,
        peer_urls: Sequence[str] = (),
        discovery_enabled: bool = True,
        sync_key: str = "",
        connect_timeout: float = 2.0,
        read_timeout: float = 120.0,
        queue_enabled: bool = True,
        quota_scope: str = "default",
        coordinator_url: str = "",
        producer_url: str = "",
        producer_port: int = 0,
        queue_poll_interval: float = 2.0,
        queue_max_wait: float = 30 * 60.0,
        queue_consumer_grace: float = 3.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.enabled = bool(enabled)
        self.peer_urls = normalize_peer_urls(peer_urls)
        self.discovery_enabled = bool(discovery_enabled)
        self.sync_key = str(sync_key or "")
        self.connect_timeout = max(0.1, float(connect_timeout))
        self.read_timeout = max(1.0, float(read_timeout))
        self.hashes = ArchiveHashCache()
        self.queue_enabled = self.enabled and bool(queue_enabled)
        self.quota_scope = LanAcquisitionQueue._validate_key(
            quota_scope,
            "1",
        )
        self.coordinator_url = (
            normalize_peer_url(coordinator_url)
            if str(coordinator_url).strip()
            else ""
        )
        self.producer_url = (
            normalize_peer_url(producer_url) if str(producer_url).strip() else ""
        )
        self.producer_port = int(producer_port or 0)
        if self.producer_port and not 1_024 <= self.producer_port <= 65_535:
            raise ValueError("The LAN producer port must be between 1024 and 65535.")
        self.queue_poll_interval = min(
            30.0,
            max(0.05, float(queue_poll_interval)),
        )
        self.queue_max_wait = min(
            24 * 60 * 60.0,
            max(0.1, float(queue_max_wait)),
        )
        self.queue_consumer_grace = min(
            30.0,
            max(0.0, float(queue_consumer_grace)),
        )
        self._sleep = sleep
        self._recent_peers: OrderedDict[str, float] = OrderedDict()
        self._peer_lock = threading.Lock()
        self._producer_cache: dict[str, tuple[float, str, str]] = {}

    @classmethod
    def from_settings(
        cls,
        *,
        enabled: bool,
        peer_urls: str | Sequence[str] | None,
        discovery_enabled: bool,
    ) -> "LanArchiveSyncClient":
        environment_enabled = environment_flag(
            "BROADCASTIFY_LAN_SYNC_ENABLED",
            default=True,
        )
        environment_discovery = environment_flag(
            "BROADCASTIFY_LAN_DISCOVERY_ENABLED",
            default=True,
        )
        try:
            producer_port = int(os.getenv("BROADCASTIFY_LAN_SELF_PORT") or 0)
        except ValueError:
            producer_port = 0

        base_quota_scope = str(
            os.getenv("BROADCASTIFY_LAN_QUOTA_SCOPE") or "default"
        ).strip()
        account_profile_id = str(
            os.getenv("BROADCASTIFY_ACCOUNT_PROFILE") or "default"
        ).strip().lower()
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", account_profile_id):
            raise ValueError("The Broadcastify account profile ID is not valid.")
        # A quota-limited result belongs to one authorized account, while the
        # queue itself still permits only one active website stream globally.
        # Including the non-secret profile ID lets a scheduled retry rotate to
        # another account without mistaking the first account's rolling limit
        # for a pool-wide provider limit.
        quota_scope = f"{base_quota_scope}.{account_profile_id}"
        if len(quota_scope) > 64:
            quota_scope = (
                f"account-{account_profile_id[:32]}-"
                f"{hashlib.sha256(quota_scope.encode('utf-8')).hexdigest()[:16]}"
            )

        return cls(
            enabled=enabled and environment_enabled,
            peer_urls=configured_peer_urls(peer_urls),
            discovery_enabled=discovery_enabled and environment_discovery,
            sync_key=os.getenv("BROADCASTIFY_LAN_SYNC_KEY") or "",
            queue_enabled=environment_flag(
                "BROADCASTIFY_LAN_QUEUE_ENABLED",
                default=True,
            ),
            quota_scope=quota_scope,
            coordinator_url=(
                os.getenv("BROADCASTIFY_LAN_COORDINATOR")
                or os.getenv("BROADCASTIFY_LAN_QUOTA_COORDINATOR")
                or ""
            ),
            producer_url=(
                os.getenv("BROADCASTIFY_LAN_SELF_URL")
                or os.getenv("BROADCASTIFY_LAN_ADVERTISE_URL")
                or ""
            ),
            producer_port=producer_port,
            queue_poll_interval=environment_float(
                "BROADCASTIFY_LAN_QUEUE_POLL_SECONDS",
                2.0,
                minimum=0.25,
                maximum=30.0,
            ),
            queue_max_wait=environment_float(
                "BROADCASTIFY_LAN_QUEUE_MAX_WAIT_SECONDS",
                30 * 60.0,
                minimum=30.0,
                maximum=24 * 60 * 60.0,
            ),
            queue_consumer_grace=environment_float(
                "BROADCASTIFY_LAN_QUEUE_CONSUMER_GRACE_SECONDS",
                3.0,
                minimum=0.0,
                maximum=30.0,
            ),
        )

    def sync_day(
        self,
        output_dir: str | Path,
        feed_id: str,
        archive_date: date,
        *,
        progress: ProgressCallback | None = None,
        additional_peer_urls: Sequence[str] = (),
    ) -> LanSyncResult:
        if not self.enabled:
            return LanSyncResult(enabled=False)
        if not feed_id.isdigit():
            raise ValueError("A numeric feed ID is required for LAN archive sync.")
        seeds = list(
            dict.fromkeys(
                (
                    *self.peer_urls,
                    *normalize_peer_urls(additional_peer_urls, strict=False),
                    *self._recent_peer_urls(),
                )
            )
        )
        failures: list[str] = []
        if self.discovery_enabled:
            try:
                seeds.extend(discover_lan_peers())
            except OSError as exc:
                failures.append(f"LAN discovery: {exc}")
        queue = list(dict.fromkeys(seeds))[:MAX_LAN_PEERS]
        considered: list[str] = []
        reachable: list[str] = []
        reached = 0
        candidates: dict[str, list[tuple[str, ArchiveBlock]]] = {}
        completion_manifests: list[tuple[str, tuple[ArchiveBlock, ...]]] = []
        while queue and len(considered) < MAX_LAN_PEERS:
            peer = queue.pop(0)
            if peer in considered:
                continue
            considered.append(peer)
            try:
                blocks, advertised_peers, complete, completion_blocks = self._inventory(
                    peer,
                    feed_id,
                    archive_date,
                )
                reached += 1
                reachable.append(peer)
                for block in blocks:
                    candidates.setdefault(block.filename, []).append((peer, block))
                if complete:
                    completion_manifests.append((peer, completion_blocks))
                for advertised in advertised_peers:
                    if (
                        advertised not in considered
                        and advertised not in queue
                        and len(considered) + len(queue) < MAX_LAN_PEERS
                    ):
                        queue.append(advertised)
            except (LanSyncError, requests.RequestException, ValueError) as exc:
                failures.append(f"{peer}: {exc}")
        self._remember_peers(reachable)
        if progress and considered:
            progress(
                f"LAN archive pool reached {reached}/{len(considered)} peer"
                f"{'s' if len(considered) != 1 else ''} for {archive_date.isoformat()}."
            )

        output_root = Path(output_dir).expanduser().resolve()
        day_dir = (
            output_root
            / feed_id
            / archive_date.strftime("%Y%m%d")
        ).resolve()
        try:
            day_dir.relative_to(output_root)
        except ValueError as exc:
            raise LanSyncError(
                "The LAN sync target is outside the archive library."
            ) from exc
        day_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        copied_bytes = 0
        already_local = 0
        conflicts = 0
        completion_proven = False
        for filename in sorted(candidates):
            sources = candidates[filename]
            signatures = {(block.size, block.sha256) for _peer, block in sources}
            if len(signatures) != 1:
                conflicts += 1
                failures.append(
                    f"{filename}: peers disagree on the retained block hash or size."
                )
                continue
            target = day_dir / filename
            expected = sources[0][1]
            if target.is_symlink():
                conflicts += 1
                failures.append(
                    f"{filename}: a local symbolic link already uses this name."
                )
                continue
            if target.exists():
                try:
                    if (
                        target.is_file()
                        and target.stat().st_size == expected.size
                        and self.hashes.sha256(target) == expected.sha256
                    ):
                        already_local += 1
                        _remember_block_archive_identities(
                            day_dir,
                            feed_id,
                            archive_date,
                            expected,
                            target,
                        )
                    else:
                        conflicts += 1
                        failures.append(
                            f"{filename}: a different local file already uses this name."
                        )
                except OSError as exc:
                    failures.append(f"{filename}: {exc}")
                continue
            copied_from_peer = False
            for peer, block in sources:
                try:
                    transferred = self._download_block(
                        peer,
                        block,
                        target,
                    )
                    copied += 1
                    copied_bytes += transferred
                    copied_from_peer = True
                    _remember_block_archive_identities(
                        day_dir,
                        feed_id,
                        archive_date,
                        block,
                        target,
                    )
                    if progress:
                        progress(
                            f"Copied archive block {filename} from LAN peer "
                            f"({copied} copied, "
                            f"{copied_bytes / (1024 * 1024):.1f} MiB total)."
                        )
                    break
                except (LanSyncError, requests.RequestException, OSError) as exc:
                    failures.append(f"{peer} / {filename}: {exc}")
            if not copied_from_peer:
                continue
        if completion_manifests:
            signatures = {
                tuple(
                    (
                        block.filename,
                        block.size,
                        block.sha256,
                        tuple(
                            (identity.archive_id, identity.listing_prefix)
                            for identity in block.identities()
                        ),
                    )
                    for block in manifest
                )
                for _peer, manifest in completion_manifests
            }
            if len(signatures) != 1:
                conflicts += 1
                failures.append(
                    "LAN peers disagree on the exact completion proof for this day."
                )
            else:
                completion_blocks = completion_manifests[0][1]
                verified = self.verified_local_blocks(
                    output_root,
                    feed_id,
                    archive_date,
                    completion_blocks,
                )
                archive_ids = [
                    identity.archive_id
                    for block in completion_blocks
                    for identity in block.identities()
                ]
                if (
                    len(verified) == len(completion_blocks)
                    and len(archive_ids) == len(completion_blocks)
                    and remember_complete_archive_day(
                        day_dir,
                        feed_id,
                        archive_date,
                        archive_ids,
                    )
                ):
                    completion_proven = True
                else:
                    failures.append(
                        "The LAN completion proof could not be verified locally."
                    )
        return LanSyncResult(
            enabled=True,
            peers_considered=len(considered),
            peers_reached=reached,
            blocks_available=len(candidates),
            blocks_already_local=already_local,
            blocks_copied=copied,
            bytes_copied=copied_bytes,
            conflicts=conflicts,
            completion_proven=completion_proven,
            failures=tuple(failures[:50]),
        )

    def sync_transcripts(
        self,
        output_dir: str | Path,
        feed_id: str,
        archive_date: date,
        processing_fingerprint: str,
        *,
        progress: ProgressCallback | None = None,
        additional_peer_urls: Sequence[str] = (),
    ) -> LanTranscriptSyncResult:
        """Pull hash-verified audio/JSON/text produced by an equivalent model."""

        if not self.enabled:
            return LanTranscriptSyncResult(enabled=False)
        fingerprint = LanProcessingQueue._validate_key(
            processing_fingerprint,
            feed_id,
        )
        seeds = list(
            dict.fromkeys(
                (
                    *self.peer_urls,
                    *normalize_peer_urls(additional_peer_urls, strict=False),
                    *self._recent_peer_urls(),
                )
            )
        )
        failures: list[str] = []
        if self.discovery_enabled:
            try:
                seeds.extend(discover_lan_peers())
            except OSError as exc:
                failures.append(f"LAN discovery: {exc}")
        queue = list(dict.fromkeys(seeds))[:MAX_LAN_PEERS]
        considered: list[str] = []
        reachable: list[str] = []
        candidates: dict[
            tuple[str, str], list[tuple[str, TranscriptArtifact]]
        ] = {}
        while queue and len(considered) < MAX_LAN_PEERS:
            peer = queue.pop(0)
            if peer in considered:
                continue
            considered.append(peer)
            try:
                artifacts, advertised_peers = self._transcript_inventory(
                    peer,
                    feed_id,
                    archive_date,
                    fingerprint,
                )
                reachable.append(peer)
                for artifact in artifacts:
                    candidates.setdefault(
                        (artifact.kind, artifact.filename),
                        [],
                    ).append((peer, artifact))
                for advertised in advertised_peers:
                    if (
                        advertised not in considered
                        and advertised not in queue
                        and len(considered) + len(queue) < MAX_LAN_PEERS
                    ):
                        queue.append(advertised)
            except (LanSyncError, requests.RequestException, ValueError) as exc:
                failures.append(f"{peer}: {exc}")
        self._remember_peers(reachable)

        output_root = Path(output_dir).expanduser().resolve()
        day_dir = (
            output_root / feed_id / archive_date.strftime("%Y%m%d")
        ).resolve()
        try:
            day_dir.relative_to(output_root)
        except ValueError as exc:
            raise LanSyncError(
                "The LAN transcript target is outside the archive library."
            ) from exc
        day_dir.mkdir(parents=True, exist_ok=True)
        transcript_dir = day_dir / "transcripts"
        if transcript_dir.exists() and (
            transcript_dir.is_symlink() or not transcript_dir.is_dir()
        ):
            raise LanSyncError("The local transcript directory is not safe to use.")
        transcript_dir.mkdir(parents=True, exist_ok=True)
        if transcript_dir.resolve().parent != day_dir:
            raise LanSyncError("The local transcript directory escaped the archive day.")

        copied = 0
        copied_bytes = 0
        already_local = 0
        conflicts = 0
        order = {"audio": 0, "manifest": 1, "json": 2, "text": 3}
        usable: dict[
            tuple[str, str], list[tuple[str, TranscriptArtifact]]
        ] = {}
        for key, sources in candidates.items():
            signatures = {
                (
                    artifact.size,
                    artifact.sha256,
                    artifact.audio_filename,
                    artifact.audio_sha256,
                )
                for _peer, artifact in sources
            }
            if len(signatures) != 1:
                conflicts += 1
                failures.append(
                    f"{key[1]}: peers disagree on the transcript artifact."
                )
                continue
            usable[key] = sources

        def target_for(root: Path, artifact: TranscriptArtifact) -> Path:
            return (
                root / artifact.filename
                if artifact.kind in {"audio", "manifest"}
                else root / "transcripts" / artifact.filename
            )

        def matches(path: Path, artifact: TranscriptArtifact) -> bool:
            return bool(
                not path.is_symlink()
                and path.is_file()
                and path.stat().st_size == artifact.size
                and self.hashes.sha256(path) == artifact.sha256
            )

        grouped: dict[tuple[str, str], list[TranscriptArtifact]] = {}
        for sources in usable.values():
            artifact = sources[0][1]
            grouped.setdefault(
                (artifact.audio_filename, artifact.audio_sha256),
                [],
            ).append(artifact)

        target_roots: dict[tuple[str, str], Path] = {}
        for group, artifacts in grouped.items():
            canonical_conflict = False
            for artifact in artifacts:
                target = target_for(day_dir, artifact)
                if target.exists() or target.is_symlink():
                    try:
                        if not matches(target, artifact):
                            canonical_conflict = True
                            break
                    except OSError:
                        canonical_conflict = True
                        break
            if not canonical_conflict:
                target_roots[group] = day_dir
                continue

            # Different model/audio output must not overwrite the conventional
            # day result. Keep a complete, hash-bound variant under its model
            # fingerprint and audio hash so both nodes' retained evidence stays
            # available and can be served to another peer later.
            variant_parts = (
                day_dir / DERIVED_VARIANTS_DIRECTORY,
                day_dir / DERIVED_VARIANTS_DIRECTORY / fingerprint,
                day_dir
                / DERIVED_VARIANTS_DIRECTORY
                / fingerprint
                / group[1],
            )
            parent = day_dir
            for candidate in variant_parts:
                if candidate.exists() and (
                    candidate.is_symlink() or not candidate.is_dir()
                ):
                    raise LanSyncError(
                        "The local derived-variant directory is not safe to use."
                    )
                candidate.mkdir(exist_ok=True)
                if candidate.resolve().parent != parent.resolve():
                    raise LanSyncError(
                        "The local derived-variant directory escaped the archive day."
                    )
                parent = candidate
            variant_root = variant_parts[-1].resolve()
            variant_transcripts = variant_root / "transcripts"
            if variant_transcripts.exists() and (
                variant_transcripts.is_symlink()
                or not variant_transcripts.is_dir()
            ):
                raise LanSyncError(
                    "The local derived transcript directory is not safe to use."
                )
            variant_transcripts.mkdir(exist_ok=True)
            if variant_transcripts.resolve().parent != variant_root:
                raise LanSyncError(
                    "The local derived transcript directory escaped its variant."
                )
            target_roots[group] = variant_root

        for key in sorted(usable, key=lambda value: (order[value[0]], value[1])):
            sources = usable[key]
            expected = sources[0][1]
            target = target_for(
                target_roots[(expected.audio_filename, expected.audio_sha256)],
                expected,
            )
            if target.is_symlink():
                conflicts += 1
                failures.append(
                    f"{expected.filename}: a local symbolic link uses this name."
                )
                continue
            if target.exists():
                try:
                    if matches(target, expected):
                        already_local += 1
                    else:
                        conflicts += 1
                        failures.append(
                            f"{expected.filename}: different local content uses this name."
                        )
                except OSError as exc:
                    failures.append(f"{expected.filename}: {exc}")
                continue
            for peer, artifact in sources:
                try:
                    transferred = self._download_transcript_artifact(
                        peer,
                        artifact,
                        target,
                    )
                    copied += 1
                    copied_bytes += transferred
                    if progress:
                        progress(
                            f"Copied {artifact.kind} artifact {artifact.filename} "
                            f"from a LAN peer ({copied_bytes / (1024 * 1024):.1f} MiB)."
                        )
                    break
                except (LanSyncError, requests.RequestException, OSError) as exc:
                    failures.append(f"{peer} / {artifact.filename}: {exc}")

        local_catalog = LanArchiveCatalog(
            output_root,
            enabled=True,
            queue_enabled=False,
        )
        verified = local_catalog.transcript_inventory(
            feed_id,
            archive_date,
            fingerprint,
        )
        resolved_transcripts: list[Path] = []
        for value in verified:
            if value.kind != "json":
                continue
            try:
                path, _artifact = local_catalog.resolve_transcript_artifact(
                    feed_id,
                    archive_date,
                    fingerprint,
                    value.filename,
                )
                resolved_transcripts.append(path)
            except (FileNotFoundError, LanSyncError, OSError) as exc:
                failures.append(f"{value.filename}: {exc}")
        transcripts = tuple(resolved_transcripts)
        return LanTranscriptSyncResult(
            enabled=True,
            peers_considered=len(considered),
            peers_reached=len(reachable),
            artifacts_available=len(candidates),
            artifacts_already_local=already_local,
            artifacts_copied=copied,
            bytes_copied=copied_bytes,
            conflicts=conflicts,
            transcripts=transcripts,
            failures=tuple(failures[:50]),
        )

    def sync_feed(
        self,
        output_dir: str | Path,
        feed_id: str,
        *,
        processing_fingerprint: str = "",
        progress: ProgressCallback | None = None,
    ) -> LanFeedSyncResult:
        """Converge every retained peer date for one followed feed."""

        if not self.enabled:
            return LanFeedSyncResult(enabled=False)
        if not feed_id.isdigit():
            raise ValueError("A numeric feed ID is required for LAN feed sync.")
        fingerprint = str(processing_fingerprint or "").strip().lower()
        if fingerprint:
            fingerprint = LanProcessingQueue._validate_key(fingerprint, feed_id)
        dates, peers, failures = self._feed_date_union(feed_id)
        day_results: list[LanSyncResult] = []
        transcript_results: list[LanTranscriptSyncResult] = []
        for current, archive_date in enumerate(dates, start=1):
            if progress:
                progress(
                    f"Reconciling retained feed {feed_id} day {archive_date.isoformat()} "
                    f"({current}/{len(dates)})."
                )
            try:
                day_result = self.sync_day(
                    output_dir,
                    feed_id,
                    archive_date,
                    additional_peer_urls=peers,
                )
                day_results.append(day_result)
                failures.extend(day_result.failures)
            except (LanSyncError, requests.RequestException, OSError, ValueError) as exc:
                failures.append(f"{archive_date.isoformat()}: {exc}")
            if fingerprint:
                try:
                    transcript_result = self.sync_transcripts(
                        output_dir,
                        feed_id,
                        archive_date,
                        fingerprint,
                        additional_peer_urls=peers,
                    )
                    transcript_results.append(transcript_result)
                    failures.extend(transcript_result.failures)
                except (
                    LanSyncError,
                    requests.RequestException,
                    OSError,
                    ValueError,
                ) as exc:
                    failures.append(
                        f"{archive_date.isoformat()} transcripts: {exc}"
                    )
        block_bytes = sum(value.bytes_copied for value in day_results)
        transcript_bytes = sum(value.bytes_copied for value in transcript_results)
        return LanFeedSyncResult(
            enabled=True,
            dates_discovered=tuple(value.isoformat() for value in dates),
            days_considered=len(dates),
            days_with_download_changes=sum(
                value.blocks_copied > 0 for value in day_results
            ),
            days_with_transcript_changes=sum(
                value.artifacts_copied > 0 for value in transcript_results
            ),
            blocks_copied=sum(value.blocks_copied for value in day_results),
            transcript_artifacts_copied=sum(
                value.artifacts_copied for value in transcript_results
            ),
            bytes_copied=block_bytes + transcript_bytes,
            failures=tuple(dict.fromkeys(failures))[:50],
        )

    def wait_for_download_turn(
        self,
        output_dir: str | Path,
        feed_id: str,
        archive_date: date,
        *,
        progress: ProgressCallback | None = None,
        defer_active: bool = False,
    ) -> LanDownloadTurn:
        """Wait behind a producer or claim the one upstream download lease."""

        if not self.queue_enabled:
            return LanDownloadTurn(role="uncoordinated")
        selected = self._select_coordinator()
        if selected is None:
            return LanDownloadTurn(
                role="uncoordinated",
                warnings=("No LAN peer offered acquisition coordination.",),
            )
        coordinator, _coordinator_info = selected
        producer = self._producer_identity(coordinator)
        started = time.monotonic()
        grace_deadline = started + self.queue_consumer_grace
        observed_shared_work = bool(self.coordinator_url)
        queue_failures = 0
        sync_results: list[LanSyncResult] = []
        latest_sync: LanSyncResult | None = None
        warnings: list[str] = []
        last_progress_at = 0.0

        while True:
            now = time.monotonic()
            elapsed = now - started
            if elapsed >= self.queue_max_wait:
                role = "deferred" if observed_shared_work else "uncoordinated"
                warnings.append(
                    "The LAN acquisition wait window ended before a shared "
                    "producer finished."
                )
                return LanDownloadTurn(
                    role=role,
                    coordinator_url=coordinator,
                    sync_result=merge_lan_sync_results(sync_results),
                    warnings=tuple(dict.fromkeys(warnings)),
                )
            try:
                status = self._queue_status(
                    coordinator,
                    feed_id,
                    archive_date,
                )
                queue_failures = 0
            except (LanSyncError, requests.RequestException, ValueError) as exc:
                queue_failures += 1
                warnings.append(f"{coordinator}: {exc}")
                if queue_failures >= 3:
                    return LanDownloadTurn(
                        role=(
                            "deferred"
                            if observed_shared_work
                            else "uncoordinated"
                        ),
                        coordinator_url=coordinator,
                        sync_result=merge_lan_sync_results(sync_results),
                        warnings=tuple(dict.fromkeys(warnings)),
                    )
                self._sleep(self.queue_poll_interval)
                continue

            state = str(status["state"])
            producer_url = str(status.get("producer_url") or "")
            if state == "available":
                if bool(status.get("global_busy")):
                    observed_shared_work = True
                    if progress:
                        progress(
                            "Another LAN producer owns the one sequential "
                            "Broadcastify request stream; continuing with other "
                            "local work instead of opening a second stream."
                        )
                    if defer_active:
                        return LanDownloadTurn(
                            role="deferred",
                            coordinator_url=coordinator,
                            sync_result=merge_lan_sync_results(sync_results),
                            warnings=tuple(dict.fromkeys(warnings)),
                        )
                    remaining = max(0.05, float(status["lease_seconds"]))
                    self._sleep(min(self.queue_poll_interval, remaining))
                    continue
                if producer is not None:
                    producer_url, owner_node_id = producer
                    try:
                        claim = self._queue_action(
                            coordinator,
                            "claim",
                            feed_id,
                            archive_date,
                            {
                                "producer_url": producer_url,
                                "owner_node_id": owner_node_id,
                            },
                        )
                    except (
                        LanSyncError,
                        requests.RequestException,
                        ValueError,
                    ) as exc:
                        warnings.append(f"{coordinator}: {exc}")
                        self._sleep(self.queue_poll_interval)
                        continue
                    if bool(claim.get("granted")):
                        if progress:
                            progress(
                                "This client owns the shared LAN acquisition "
                                f"lease for {archive_date.isoformat()}."
                            )
                        return LanDownloadTurn(
                            role="leader",
                            feed_id=feed_id,
                            archive_date=archive_date.isoformat(),
                            coordinator_url=coordinator,
                            producer_url=producer_url,
                            owner_node_id=owner_node_id,
                            lease_token=str(claim["lease_token"]),
                            lease_seconds=float(claim["lease_seconds"]),
                            sync_result=merge_lan_sync_results(sync_results),
                            warnings=tuple(dict.fromkeys(warnings)),
                        )
                    if bool(claim.get("global_busy")) and defer_active:
                        return LanDownloadTurn(
                            role="deferred",
                            coordinator_url=coordinator,
                            sync_result=merge_lan_sync_results(sync_results),
                            warnings=tuple(dict.fromkeys(warnings)),
                        )
                    continue
                if now < grace_deadline:
                    if progress and last_progress_at == 0:
                        progress(
                            "Waiting briefly for a LAN peer that can seed this "
                            "feed/day to claim the shared download."
                        )
                        last_progress_at = now
                    self._sleep(self.queue_poll_interval)
                    continue
                warnings.append(
                    "This client is not serving original blocks, so it cannot "
                    "hold a shared upstream lease."
                )
                return LanDownloadTurn(
                    role="uncoordinated",
                    coordinator_url=coordinator,
                    sync_result=merge_lan_sync_results(sync_results),
                    warnings=tuple(dict.fromkeys(warnings)),
                )

            if state in {"active", "complete", "quota_limited"}:
                observed_shared_work = True
                if producer_url:
                    try:
                        latest_sync = self.sync_day(
                            output_dir,
                            feed_id,
                            archive_date,
                            progress=progress,
                            additional_peer_urls=(producer_url,),
                        )
                        sync_results.append(latest_sync)
                    except (
                        LanSyncError,
                        requests.RequestException,
                        OSError,
                        ValueError,
                    ) as exc:
                        warnings.append(f"{producer_url}: {exc}")

            if state == "complete":
                block_count = int(status["block_count"])
                completion_blocks = tuple(status["blocks"])
                audio_files = self.verified_local_blocks(
                    output_dir,
                    feed_id,
                    archive_date,
                    completion_blocks,
                )
                if len(audio_files) == block_count:
                    if progress:
                        progress(
                            f"LAN producer completed {archive_date.isoformat()}; "
                            f"{len(audio_files)} verified source blocks are local."
                        )
                    return LanDownloadTurn(
                        role="completed",
                        coordinator_url=coordinator,
                        producer_url=producer_url,
                        owner_node_id=str(status.get("owner_node_id") or ""),
                        block_count=block_count,
                        rolling=bool(status["rolling"]),
                        blocks=completion_blocks,
                        audio_files=tuple(audio_files),
                        sync_result=merge_lan_sync_results(sync_results),
                        warnings=tuple(dict.fromkeys(warnings)),
                    )
                if defer_active:
                    return LanDownloadTurn(
                        role="deferred",
                        coordinator_url=coordinator,
                        producer_url=producer_url,
                        owner_node_id=str(status.get("owner_node_id") or ""),
                        block_count=block_count,
                        blocks=completion_blocks,
                        sync_result=merge_lan_sync_results(sync_results),
                        warnings=tuple(dict.fromkeys(warnings)),
                    )
                if progress and now - last_progress_at >= 15.0:
                    progress(
                        "The shared download is complete; waiting for its "
                        f"verified blocks ({len(audio_files)}/{block_count} local)."
                    )
                    last_progress_at = now
                self._sleep(self.queue_poll_interval)
                continue

            if state == "quota_limited":
                if progress:
                    progress(
                        "A LAN producer reached the shared Broadcastify quota; "
                        "this client will not repeat those archive requests."
                    )
                return LanDownloadTurn(
                    role="quota_limited",
                    coordinator_url=coordinator,
                    producer_url=producer_url,
                    owner_node_id=str(status.get("owner_node_id") or ""),
                    block_count=int(status["block_count"]),
                    audio_files=tuple(
                        self.local_source_files(
                            output_dir,
                            feed_id,
                            archive_date,
                        )
                    ),
                    sync_result=merge_lan_sync_results(sync_results),
                    warnings=tuple(dict.fromkeys(warnings)),
                )

            if state == "active":
                if progress and now - last_progress_at >= 15.0:
                    progress(
                        "Another LAN producer owns this feed/day; pulling "
                        "completed blocks as they appear instead of contacting "
                        "Broadcastify."
                    )
                    last_progress_at = now
                if defer_active:
                    return LanDownloadTurn(
                        role="deferred",
                        coordinator_url=coordinator,
                        producer_url=producer_url,
                        owner_node_id=str(status.get("owner_node_id") or ""),
                        sync_result=merge_lan_sync_results(sync_results),
                        warnings=tuple(dict.fromkeys(warnings)),
                    )
                remaining = max(0.05, float(status["lease_seconds"]))
                self._sleep(min(self.queue_poll_interval, remaining))
                continue

            raise LanSyncError("The LAN coordinator returned an unknown queue state.")

    def maintain_download_lease(
        self,
        turn: LanDownloadTurn,
    ) -> AbstractContextManager["_LanLeaseHeartbeat"]:
        if not turn.is_leader:
            raise ValueError("Only the LAN acquisition leader has a renewable lease.")
        return _LanLeaseHeartbeat(self, turn)

    def finish_download_turn(
        self,
        turn: LanDownloadTurn,
        *,
        outcome: str,
        block_count: int = 0,
        source_files: Sequence[str | Path] = (),
        retry_after_seconds: float | None = None,
    ) -> str:
        if not turn.is_leader:
            return ""
        try:
            blocks = (
                self.completion_blocks(
                    source_files,
                    turn.feed_id,
                    date.fromisoformat(turn.archive_date),
                )
                if outcome == "complete"
                else ()
            )
            if outcome == "complete" and len(blocks) != int(block_count):
                raise LanSyncError(
                    "The completed LAN acquisition files do not match "
                    "the reported block count."
                )
            self._queue_action(
                turn.coordinator_url,
                "finish",
                turn.feed_id,
                date.fromisoformat(turn.archive_date),
                {
                    "lease_token": turn.lease_token,
                    "outcome": outcome,
                    "block_count": int(block_count),
                    "blocks": [block.to_dict() for block in blocks],
                    "retry_after_seconds": retry_after_seconds,
                },
            )
        except (LanSyncError, requests.RequestException, ValueError) as exc:
            return f"The LAN acquisition result could not be published: {exc}"
        return ""

    def claim_processing_turn(
        self,
        feed_id: str,
        archive_date: date,
        processing_fingerprint: str,
        *,
        progress: ProgressCallback | None = None,
    ) -> LanProcessingTurn:
        """Claim one model/day without waiting behind another node's model."""

        fingerprint = LanProcessingQueue._validate_key(
            processing_fingerprint,
            feed_id,
        )
        if not self.queue_enabled:
            return LanProcessingTurn(role="uncoordinated")
        selected = self._select_coordinator("processing_queue_available")
        if selected is None:
            return LanProcessingTurn(
                role="uncoordinated",
                warnings=("No LAN peer offered processing coordination.",),
            )
        coordinator, _coordinator_info = selected
        try:
            status = self._processing_queue_status(
                coordinator,
                feed_id,
                archive_date,
                fingerprint,
            )
        except (LanSyncError, requests.RequestException, ValueError) as exc:
            return LanProcessingTurn(
                role="deferred",
                feed_id=feed_id,
                archive_date=archive_date.isoformat(),
                processing_fingerprint=fingerprint,
                coordinator_url=coordinator,
                warnings=(
                    "The LAN processing coordinator could not confirm a safe "
                    f"turn: {exc}",
                ),
            )

        state = str(status["state"])
        if state == "complete":
            if progress:
                progress(
                    "An equivalent model run completed on another LAN node; "
                    "its verified artifacts can be reused."
                )
            return LanProcessingTurn(
                role="completed",
                feed_id=feed_id,
                archive_date=archive_date.isoformat(),
                processing_fingerprint=fingerprint,
                coordinator_url=coordinator,
                producer_url=str(status.get("producer_url") or ""),
                owner_node_id=str(status.get("owner_node_id") or ""),
                artifact_count=int(status.get("artifact_count") or 0),
            )
        if state == "active":
            if progress:
                progress(
                    "Another LAN node is processing this model/day; moving to "
                    "the next day instead of waiting or duplicating work."
                )
            return LanProcessingTurn(
                role="deferred",
                feed_id=feed_id,
                archive_date=archive_date.isoformat(),
                processing_fingerprint=fingerprint,
                coordinator_url=coordinator,
                producer_url=str(status.get("producer_url") or ""),
                owner_node_id=str(status.get("owner_node_id") or ""),
            )
        if state != "available":
            raise LanSyncError("The LAN coordinator returned an unknown processing state.")

        producer = self._producer_identity(coordinator)
        if producer is None:
            return LanProcessingTurn(
                role="uncoordinated",
                coordinator_url=coordinator,
                warnings=(
                    "This client is not serving retained artifacts, so it cannot "
                    "publish a shared processing result.",
                ),
            )
        producer_url, owner_node_id = producer
        try:
            claim = self._processing_queue_action(
                coordinator,
                "claim",
                feed_id,
                archive_date,
                fingerprint,
                {
                    "producer_url": producer_url,
                    "owner_node_id": owner_node_id,
                },
            )
        except (LanSyncError, requests.RequestException, ValueError) as exc:
            return LanProcessingTurn(
                role="deferred",
                feed_id=feed_id,
                archive_date=archive_date.isoformat(),
                processing_fingerprint=fingerprint,
                coordinator_url=coordinator,
                warnings=(f"The LAN processing lease could not be claimed: {exc}",),
            )
        if not bool(claim.get("granted")):
            return LanProcessingTurn(
                role=("completed" if claim["state"] == "complete" else "deferred"),
                feed_id=feed_id,
                archive_date=archive_date.isoformat(),
                processing_fingerprint=fingerprint,
                coordinator_url=coordinator,
                producer_url=str(claim.get("producer_url") or ""),
                owner_node_id=str(claim.get("owner_node_id") or ""),
                artifact_count=int(claim.get("artifact_count") or 0),
            )
        if progress:
            progress(
                "This client owns the shared model/day lease; other nodes may "
                "process different days in parallel."
            )
        return LanProcessingTurn(
            role="leader",
            feed_id=feed_id,
            archive_date=archive_date.isoformat(),
            processing_fingerprint=fingerprint,
            coordinator_url=coordinator,
            producer_url=producer_url,
            owner_node_id=owner_node_id,
            lease_token=str(claim["lease_token"]),
            lease_seconds=float(claim["lease_seconds"]),
        )

    def maintain_processing_lease(
        self,
        turn: LanProcessingTurn,
    ) -> AbstractContextManager["_LanProcessingLeaseHeartbeat"]:
        if not turn.is_leader:
            raise ValueError("Only the LAN processing leader has a renewable lease.")
        return _LanProcessingLeaseHeartbeat(self, turn)

    def finish_processing_turn(
        self,
        turn: LanProcessingTurn,
        *,
        outcome: str,
        artifact_count: int = 0,
    ) -> str:
        if not turn.is_leader:
            return ""
        try:
            self._processing_queue_action(
                turn.coordinator_url,
                "finish",
                turn.feed_id,
                date.fromisoformat(turn.archive_date),
                turn.processing_fingerprint,
                {
                    "lease_token": turn.lease_token,
                    "outcome": outcome,
                    "artifact_count": int(artifact_count),
                },
            )
        except (LanSyncError, requests.RequestException, ValueError) as exc:
            return f"The LAN processing result could not be published: {exc}"
        return ""

    def _select_coordinator(
        self,
        capability: str = "acquisition_queue_available",
    ) -> tuple[str, dict[str, Any]] | None:
        if capability not in {
            "acquisition_queue_available",
            "processing_queue_available",
        }:
            raise ValueError("Unsupported LAN coordinator capability.")
        # When an authoritative coordinator is configured, never independently
        # elect a different peer. Both Windows and NAS workers must make the
        # same decision even if discovery is asymmetric or temporarily down.
        # The endpoint call that follows will fail closed if it is unavailable.
        if self.coordinator_url:
            return self.coordinator_url, {
                "node_id": "configured-coordinator",
                capability: True,
            }
        seeds = list(dict.fromkeys((*self.peer_urls, *self._recent_peer_urls())))
        if self.discovery_enabled:
            try:
                seeds.extend(discover_lan_peers())
            except OSError:
                pass
        pending = list(dict.fromkeys(seeds))[:MAX_LAN_PEERS]
        seen: set[str] = set()
        coordinators: list[tuple[str, str, dict[str, Any]]] = []
        while pending and len(seen) < MAX_LAN_PEERS:
            batch: list[str] = []
            while pending and len(seen) + len(batch) < MAX_LAN_PEERS:
                peer = pending.pop(0)
                if peer not in seen and peer not in batch:
                    batch.append(peer)
            if not batch:
                break
            workers = min(8, len(batch))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(self._peer_info, peer): peer
                    for peer in batch
                }
                for future in as_completed(futures):
                    peer = futures[future]
                    seen.add(peer)
                    try:
                        info = future.result()
                    except (
                        LanSyncError,
                        requests.RequestException,
                        ValueError,
                    ):
                        continue
                    self._remember_peers((peer,))
                    for advertised in info["peers"]:
                        if (
                            advertised not in seen
                            and advertised not in pending
                            and len(seen) + len(pending) < MAX_LAN_PEERS
                        ):
                            pending.append(advertised)
                    if info[capability]:
                        coordinators.append((str(info["node_id"]), peer, info))
        if not coordinators:
            return None
        _node_id, peer, info = min(
            coordinators,
            key=lambda value: (value[0], value[1]),
        )
        return peer, info

    def coordinated_status(self) -> dict[str, Any]:
        """Read the trusted coordinator's non-secret work/status surface."""

        if not self.enabled:
            return {
                "connected": False,
                "coordinator_url": "",
                "node_id": "",
                "peer_count": 0,
                "activity": {"acquisition": None, "processing": []},
                "scheduler": {"active": None, "schedules": []},
                "error": "Trusted-LAN coordination is disabled.",
            }
        try:
            if self.coordinator_url:
                coordinator = self.coordinator_url
                info = self._peer_info(coordinator)
            else:
                selected = self._select_coordinator()
                if selected is None:
                    raise LanSyncError(
                        "No trusted-LAN acquisition coordinator answered."
                    )
                coordinator, info = selected
        except (LanSyncError, requests.RequestException, ValueError) as exc:
            return {
                "connected": False,
                "coordinator_url": self.coordinator_url,
                "node_id": "",
                "peer_count": 0,
                "activity": {"acquisition": None, "processing": []},
                "scheduler": {"active": None, "schedules": []},
                "error": str(exc)[:300],
            }
        return {
            "connected": True,
            "coordinator_url": coordinator,
            "node_id": str(info["node_id"]),
            "peer_count": len(info["peers"]),
            "activity": info["activity"],
            "scheduler": info["scheduler"],
            "error": "",
        }

    def _producer_identity(self, coordinator: str) -> tuple[str, str] | None:
        cached = self._producer_cache.get(coordinator)
        now = time.monotonic()
        if cached and now - cached[0] <= 30.0:
            return cached[1], cached[2]
        producer = self.producer_url
        if not producer and self.producer_port:
            parsed = urlparse(coordinator)
            address = ipaddress.ip_address(parsed.hostname or "")
            family = socket.AF_INET6 if address.version == 6 else socket.AF_INET
            with socket.socket(family, socket.SOCK_DGRAM) as route:
                route.connect((str(address), parsed.port or 80))
                local_address = str(route.getsockname()[0])
            display = (
                f"[{local_address}]" if address.version == 6 else local_address
            )
            producer = normalize_peer_url(
                f"http://{display}:{self.producer_port}"
            )
        if not producer:
            return None
        try:
            info = self._peer_info(producer)
        except (LanSyncError, requests.RequestException, ValueError):
            return None
        if not info["sharing"]:
            return None
        value = (producer, str(info["node_id"]))
        self._producer_cache[coordinator] = (now, *value)
        return value

    def _queue_status(
        self,
        coordinator: str,
        feed_id: str,
        archive_date: date,
    ) -> dict[str, Any]:
        with self._lan_session() as session:
            with session.get(
                f"{coordinator}/api/lan/v1/acquisition",
                params={
                    "quota_scope": self.quota_scope,
                    "feed_id": feed_id,
                    "date": archive_date.isoformat(),
                },
                headers=self._headers(),
                timeout=(self.connect_timeout, min(self.read_timeout, 10.0)),
                allow_redirects=False,
                stream=True,
            ) as response:
                response.raise_for_status()
                payload = self._bounded_json(response, LAN_QUEUE_RESPONSE_BYTES)
        return self._validate_queue_payload(
            payload,
            feed_id,
            archive_date,
        )

    def _queue_action(
        self,
        coordinator: str,
        action: str,
        feed_id: str,
        archive_date: date,
        extra: Mapping[str, Any],
    ) -> dict[str, Any]:
        body = {
            "quota_scope": self.quota_scope,
            "feed_id": feed_id,
            "archive_date": archive_date.isoformat(),
            **dict(extra),
        }
        headers = self._headers()
        headers["Content-Type"] = "application/json"
        with self._lan_session() as session:
            with session.post(
                f"{coordinator}/api/lan/v1/acquisition/{quote(action, safe='')}",
                headers=headers,
                data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
                timeout=(self.connect_timeout, min(self.read_timeout, 10.0)),
                allow_redirects=False,
                stream=True,
            ) as response:
                response.raise_for_status()
                payload = self._bounded_json(response, LAN_QUEUE_RESPONSE_BYTES)
        return self._validate_queue_payload(payload, feed_id, archive_date)

    def _processing_queue_status(
        self,
        coordinator: str,
        feed_id: str,
        archive_date: date,
        processing_fingerprint: str,
    ) -> dict[str, Any]:
        with self._lan_session() as session:
            with session.get(
                f"{coordinator}/api/lan/v1/processing",
                params={
                    "processing_fingerprint": processing_fingerprint,
                    "feed_id": feed_id,
                    "date": archive_date.isoformat(),
                },
                headers=self._headers(),
                timeout=(self.connect_timeout, min(self.read_timeout, 10.0)),
                allow_redirects=False,
                stream=True,
            ) as response:
                response.raise_for_status()
                payload = self._bounded_json(response, LAN_QUEUE_RESPONSE_BYTES)
        return self._validate_processing_queue_payload(
            payload,
            feed_id,
            archive_date,
            processing_fingerprint,
        )

    def _processing_queue_action(
        self,
        coordinator: str,
        action: str,
        feed_id: str,
        archive_date: date,
        processing_fingerprint: str,
        extra: Mapping[str, Any],
    ) -> dict[str, Any]:
        body = {
            "processing_fingerprint": processing_fingerprint,
            "feed_id": feed_id,
            "archive_date": archive_date.isoformat(),
            **dict(extra),
        }
        headers = self._headers()
        headers["Content-Type"] = "application/json"
        with self._lan_session() as session:
            with session.post(
                f"{coordinator}/api/lan/v1/processing/{quote(action, safe='')}",
                headers=headers,
                data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
                timeout=(self.connect_timeout, min(self.read_timeout, 10.0)),
                allow_redirects=False,
                stream=True,
            ) as response:
                response.raise_for_status()
                payload = self._bounded_json(response, LAN_QUEUE_RESPONSE_BYTES)
        return self._validate_processing_queue_payload(
            payload,
            feed_id,
            archive_date,
            processing_fingerprint,
        )

    def _peer_info(self, peer: str) -> dict[str, Any]:
        with self._lan_session() as session:
            with session.get(
                f"{peer}/api/lan/v1/info",
                headers=self._headers(),
                timeout=(min(self.connect_timeout, 1.0), 5.0),
                allow_redirects=False,
                stream=True,
            ) as response:
                response.raise_for_status()
                payload = self._bounded_json(response, LAN_QUEUE_RESPONSE_BYTES)
        if not isinstance(payload, Mapping) or payload.get("protocol") != LAN_PROTOCOL:
            raise LanSyncError("The peer returned incompatible node information.")
        node_id = str(payload.get("node_id") or "")
        if not LAN_QUEUE_NODE_PATTERN.fullmatch(node_id):
            raise LanSyncError("The peer returned an invalid node identity.")
        return {
            "node_id": node_id,
            "sharing": bool(payload.get("sharing")),
            "acquisition_queue_available": bool(
                payload.get("acquisition_queue_available")
            ),
            "processing_queue_available": bool(
                payload.get("processing_queue_available")
            ),
            "peers": normalize_peer_urls(payload.get("peers") or (), strict=False),
            "activity": self._validate_activity_surface(
                payload.get("activity")
            ),
            "scheduler": self._validate_scheduler_surface(
                payload.get("scheduler")
            ),
        }

    @staticmethod
    def _validate_activity_surface(payload: Any) -> dict[str, Any]:
        value = payload if isinstance(payload, Mapping) else {}

        def activity_item(
            candidate: Any,
            *,
            quota: bool,
        ) -> dict[str, Any] | None:
            if not isinstance(candidate, Mapping):
                return None
            feed_id = str(candidate.get("feed_id") or "")
            archive_date = str(candidate.get("archive_date") or "")
            owner = str(candidate.get("owner_node_id") or "")
            producer_url = str(candidate.get("producer_url") or "")
            scope = str(candidate.get("quota_scope") or "") if quota else ""
            try:
                date.fromisoformat(archive_date)
                lease_seconds = float(candidate.get("lease_seconds") or 0.0)
            except (TypeError, ValueError):
                return None
            if (
                not feed_id.isdigit()
                or not LAN_QUEUE_NODE_PATTERN.fullmatch(owner)
                or not 0.0 <= lease_seconds <= 24 * 60 * 60.0
                or (quota and not LAN_QUEUE_SCOPE_PATTERN.fullmatch(scope))
            ):
                return None
            try:
                producer_url = normalize_peer_url(producer_url)
            except ValueError:
                return None
            result = {
                "feed_id": feed_id,
                "archive_date": archive_date,
                "owner_node_id": owner,
                "producer_url": producer_url,
                "lease_seconds": lease_seconds,
            }
            if quota:
                result["quota_scope"] = scope
            return result

        raw_processing = value.get("processing")
        processing = [
            result
            for candidate in (
                raw_processing[:32]
                if isinstance(raw_processing, list)
                else []
            )
            if (result := activity_item(candidate, quota=False)) is not None
        ]
        return {
            "acquisition": activity_item(
                value.get("acquisition"),
                quota=True,
            ),
            "processing": processing,
        }

    @staticmethod
    def _validate_scheduler_surface(payload: Any) -> dict[str, Any]:
        value = payload if isinstance(payload, Mapping) else {}

        def clean_text(candidate: Any, maximum: int) -> str:
            return " ".join(str(candidate or "").split())[:maximum]

        def clean_non_negative_int(candidate: Any) -> int:
            try:
                return max(0, int(candidate or 0))
            except (TypeError, ValueError, OverflowError):
                return 0

        def clean_schedule(candidate: Any) -> dict[str, Any] | None:
            if not isinstance(candidate, Mapping):
                return None
            feed_id = clean_text(candidate.get("feed_id"), 40)
            if not feed_id.isdigit():
                return None
            return {
                "id": clean_non_negative_int(candidate.get("id")),
                "feed_id": feed_id,
                "feed_name": clean_text(candidate.get("feed_name"), 200),
                "state": clean_text(candidate.get("state"), 40),
                "enabled": bool(candidate.get("enabled")),
                "account_profile_id": clean_text(
                    candidate.get("account_profile_id"),
                    64,
                ),
                "next_run_at": clean_text(candidate.get("next_run_at"), 40),
                "last_started_at": clean_text(
                    candidate.get("last_started_at"),
                    40,
                ),
                "message": clean_text(candidate.get("message"), 240),
            }

        active: dict[str, Any] | None = None
        raw_active = value.get("active")
        if isinstance(raw_active, Mapping):
            feed_id = clean_text(raw_active.get("feed_id"), 40)
            owner_profile = clean_text(
                raw_active.get("account_profile_id"),
                64,
            )
            if feed_id.isdigit() and re.fullmatch(
                r"[a-z0-9][a-z0-9_-]{0,63}",
                owner_profile,
            ):
                active = {
                    "feed_id": feed_id,
                    "feed_name": clean_text(raw_active.get("feed_name"), 200),
                    "phase": clean_text(raw_active.get("phase"), 40),
                    "status": clean_text(raw_active.get("status"), 40),
                    "account_profile_id": owner_profile,
                    "stage": clean_text(raw_active.get("stage"), 40),
                    "archive_date": clean_text(
                        raw_active.get("archive_date"),
                        10,
                    ),
                    "current": clean_non_negative_int(
                        raw_active.get("current")
                    ),
                    "total": clean_non_negative_int(raw_active.get("total")),
                    "updated_at": clean_text(
                        raw_active.get("updated_at"),
                        40,
                    ),
                }
        raw_schedules = value.get("schedules")
        schedules = [
            result
            for candidate in (
                raw_schedules[:100]
                if isinstance(raw_schedules, list)
                else []
            )
            if (result := clean_schedule(candidate)) is not None
        ]
        return {"active": active, "schedules": schedules}

    def _validate_queue_payload(
        self,
        payload: Any,
        feed_id: str,
        archive_date: date,
    ) -> dict[str, Any]:
        if (
            not isinstance(payload, Mapping)
            or payload.get("protocol") != LAN_PROTOCOL
            or str(payload.get("quota_scope") or "") != self.quota_scope
            or str(payload.get("feed_id") or "") != feed_id
            or str(payload.get("archive_date") or "") != archive_date.isoformat()
        ):
            raise LanSyncError("The peer returned an incompatible queue response.")
        state = str(payload.get("state") or "")
        if state not in {"available", "active", "complete", "quota_limited"}:
            raise LanSyncError("The peer returned an invalid queue state.")
        producer_url = str(payload.get("producer_url") or "")
        if producer_url:
            producer_url = normalize_peer_url(producer_url)
        owner_node_id = str(payload.get("owner_node_id") or "")
        if owner_node_id and not LAN_QUEUE_NODE_PATTERN.fullmatch(owner_node_id):
            raise LanSyncError("The peer returned an invalid queue owner.")
        try:
            lease_seconds = float(payload.get("lease_seconds") or 0.0)
            block_count = int(payload.get("block_count") or 0)
        except (TypeError, ValueError) as exc:
            raise LanSyncError("The peer returned invalid queue counters.") from exc
        if (
            not 0.0 <= lease_seconds <= 24 * 60 * 60.0
            or not 0 <= block_count <= MAX_BLOCKS_PER_DAY
        ):
            raise LanSyncError("The peer returned out-of-range queue counters.")
        raw_blocks = payload.get("blocks")
        if not isinstance(raw_blocks, list) or len(raw_blocks) > MAX_BLOCKS_PER_DAY:
            raise LanSyncError("The peer returned an invalid completion manifest.")
        if any(not isinstance(value, Mapping) for value in raw_blocks):
            raise LanSyncError("The peer returned an invalid completion block.")
        try:
            blocks = tuple(
                ArchiveBlock.from_mapping(
                    value,
                    expected_feed_id=feed_id,
                    expected_date=archive_date,
                )
                for value in raw_blocks
            )
        except (TypeError, ValueError) as exc:
            raise LanSyncError(
                "The peer returned an invalid completion block."
            ) from exc
        if len({block.filename for block in blocks}) != len(blocks):
            raise LanSyncError("The peer returned a duplicate completion block.")
        if (
            (state == "complete" and len(blocks) != block_count)
            or (state != "complete" and blocks)
        ):
            raise LanSyncError(
                "The peer completion manifest does not match its queue state."
            )
        value = {
            "protocol": LAN_PROTOCOL,
            "quota_scope": self.quota_scope,
            "feed_id": feed_id,
            "archive_date": archive_date.isoformat(),
            "state": state,
            "producer_url": producer_url,
            "owner_node_id": owner_node_id,
            "lease_seconds": lease_seconds,
            "block_count": block_count,
            "rolling": bool(payload.get("rolling")),
            "blocks": blocks,
            "granted": bool(payload.get("granted")),
            "global_busy": bool(payload.get("global_busy")),
        }
        lease_token = str(payload.get("lease_token") or "")
        if value["granted"]:
            if not re.fullmatch(r"[A-Za-z0-9_-]{24,128}", lease_token):
                raise LanSyncError("The peer returned an invalid acquisition lease.")
            value["lease_token"] = lease_token
        return value

    def _validate_processing_queue_payload(
        self,
        payload: Any,
        feed_id: str,
        archive_date: date,
        processing_fingerprint: str,
    ) -> dict[str, Any]:
        if (
            not isinstance(payload, Mapping)
            or payload.get("protocol") != LAN_PROTOCOL
            or str(payload.get("processing_fingerprint") or "").lower()
            != processing_fingerprint
            or str(payload.get("feed_id") or "") != feed_id
            or str(payload.get("archive_date") or "") != archive_date.isoformat()
        ):
            raise LanSyncError(
                "The peer returned an incompatible processing queue response."
            )
        state = str(payload.get("state") or "")
        if state not in {"available", "active", "complete"}:
            raise LanSyncError("The peer returned an invalid processing queue state.")
        producer_url = str(payload.get("producer_url") or "")
        if producer_url:
            producer_url = normalize_peer_url(producer_url)
        owner_node_id = str(payload.get("owner_node_id") or "")
        if owner_node_id and not LAN_QUEUE_NODE_PATTERN.fullmatch(owner_node_id):
            raise LanSyncError("The peer returned an invalid processing queue owner.")
        try:
            lease_seconds = float(payload.get("lease_seconds") or 0.0)
            artifact_count = int(payload.get("artifact_count") or 0)
        except (TypeError, ValueError) as exc:
            raise LanSyncError(
                "The peer returned invalid processing queue counters."
            ) from exc
        if (
            not 0.0 <= lease_seconds <= 24 * 60 * 60.0
            or not 0 <= artifact_count <= MAX_TRANSCRIPT_ARTIFACTS_PER_DAY
        ):
            raise LanSyncError(
                "The peer returned out-of-range processing queue counters."
            )
        value = {
            "protocol": LAN_PROTOCOL,
            "processing_fingerprint": processing_fingerprint,
            "feed_id": feed_id,
            "archive_date": archive_date.isoformat(),
            "state": state,
            "producer_url": producer_url,
            "owner_node_id": owner_node_id,
            "lease_seconds": lease_seconds,
            "artifact_count": artifact_count,
            "granted": bool(payload.get("granted")),
        }
        lease_token = str(payload.get("lease_token") or "")
        if value["granted"]:
            if not re.fullmatch(r"[A-Za-z0-9_-]{24,128}", lease_token):
                raise LanSyncError("The peer returned an invalid processing lease.")
            value["lease_token"] = lease_token
        return value

    @staticmethod
    def _bounded_json(response: requests.Response, maximum_bytes: int) -> Any:
        try:
            content_length = int(response.headers.get("Content-Length") or 0)
        except ValueError as exc:
            raise LanSyncError("The peer returned an invalid response length.") from exc
        if content_length > maximum_bytes:
            raise LanSyncError("The peer response exceeded the size limit.")
        body = bytearray()
        for chunk in response.iter_content(chunk_size=16 * 1024):
            body.extend(chunk)
            if len(body) > maximum_bytes:
                raise LanSyncError("The peer response exceeded the size limit.")
        try:
            return json.loads(body)
        except (UnicodeDecodeError, ValueError) as exc:
            raise LanSyncError("The peer response was not valid JSON.") from exc

    @staticmethod
    def local_source_files(
        output_dir: str | Path,
        feed_id: str,
        archive_date: date,
    ) -> list[Path]:
        output_root = Path(output_dir).expanduser().resolve()
        day_dir = (
            output_root / feed_id / archive_date.strftime("%Y%m%d")
        ).resolve()
        try:
            day_dir.relative_to(output_root)
        except ValueError as exc:
            raise LanSyncError(
                "The LAN archive day is outside the local library."
            ) from exc
        if not day_dir.is_dir():
            return []
        values: list[Path] = []
        for path in sorted(day_dir.glob("*.mp3")):
            match = RAW_ARCHIVE_PATTERN.fullmatch(path.name)
            if (
                match is not None
                and match.group("feed_id") == feed_id
                and _archive_stamp_matches_day(match.group("stamp"), archive_date)
                and not path.is_symlink()
                and path.is_file()
                and 0 < path.stat().st_size <= MAX_ARCHIVE_BLOCK_BYTES
            ):
                values.append(path.resolve())
        return values[:MAX_BLOCKS_PER_DAY]

    def completion_blocks(
        self,
        source_files: Sequence[str | Path],
        feed_id: str,
        archive_date: date,
    ) -> tuple[ArchiveBlock, ...]:
        """Build the exact immutable manifest published by a completed leader."""

        blocks: list[ArchiveBlock] = []
        names: set[str] = set()
        for source in source_files:
            path = Path(source)
            match = RAW_ARCHIVE_PATTERN.fullmatch(path.name)
            if (
                match is None
                or match.group("feed_id") != feed_id
                or not _archive_stamp_matches_day(match.group("stamp"), archive_date)
                or path.is_symlink()
                or not path.is_file()
            ):
                raise LanSyncError(
                    "A completed acquisition contains an invalid source block."
                )
            resolved = path.resolve()
            stat = resolved.stat()
            if not 0 < stat.st_size <= MAX_ARCHIVE_BLOCK_BYTES:
                raise LanSyncError(
                    "A completed acquisition source block has an invalid size."
                )
            if path.name in names:
                raise LanSyncError(
                    "A completed acquisition contains a duplicate source block."
                )
            names.add(path.name)
            archive_id, listing_prefix, archive_identities = _block_archive_identities(
                path.parent,
                feed_id,
                path.name,
            )
            blocks.append(
                ArchiveBlock(
                    feed_id=feed_id,
                    archive_date=archive_date.isoformat(),
                    filename=path.name,
                    size=stat.st_size,
                    sha256=self.hashes.sha256(resolved),
                    modified_ns=stat.st_mtime_ns,
                    archive_id=archive_id,
                    listing_prefix=listing_prefix,
                    archive_identities=archive_identities,
                )
            )
        if len(blocks) > MAX_BLOCKS_PER_DAY:
            raise LanSyncError("A completed acquisition contains too many blocks.")
        return tuple(sorted(blocks, key=lambda value: value.filename))

    def verified_local_blocks(
        self,
        output_dir: str | Path,
        feed_id: str,
        archive_date: date,
        blocks: Sequence[ArchiveBlock],
    ) -> list[Path]:
        """Return files only when every exact completion-manifest block is local."""

        output_root = Path(output_dir).expanduser().resolve()
        day_dir = (
            output_root / feed_id / archive_date.strftime("%Y%m%d")
        ).resolve()
        try:
            day_dir.relative_to(output_root)
        except ValueError as exc:
            raise LanSyncError(
                "The LAN archive day is outside the local library."
            ) from exc
        verified: list[Path] = []
        try:
            for block in blocks:
                block.validate(
                    expected_feed_id=feed_id,
                    expected_date=archive_date,
                )
                path = day_dir / block.filename
                if (
                    path.is_symlink()
                    or not path.is_file()
                    or path.resolve().parent != day_dir
                    or path.stat().st_size != block.size
                    or self.hashes.sha256(path) != block.sha256
                ):
                    return []
                _remember_block_archive_identities(
                    day_dir,
                    feed_id,
                    archive_date,
                    block,
                    path,
                )
                verified.append(path.resolve())
        except (LanSyncError, OSError):
            return []
        return verified

    def _remember_peers(self, peers: Iterable[str]) -> None:
        with self._peer_lock:
            for peer in peers:
                normalized = normalize_peer_url(peer)
                self._recent_peers[normalized] = time.monotonic()
                self._recent_peers.move_to_end(normalized)
            while len(self._recent_peers) > MAX_LAN_PEERS:
                self._recent_peers.popitem(last=False)

    def _recent_peer_urls(self) -> tuple[str, ...]:
        with self._peer_lock:
            return tuple(self._recent_peers)

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "User-Agent": "RadioArchiveLAN/1",
        }
        if self.sync_key:
            headers["X-Radio-Archive-LAN-Key"] = self.sync_key
        return headers

    @staticmethod
    def _lan_session() -> requests.Session:
        # LAN control/data traffic must never inherit HTTP(S)_PROXY. Besides
        # breaking source-address lease validation, a proxy would receive the
        # optional LAN key and opaque lease token.
        session = requests.Session()
        session.trust_env = False
        return session

    def _inventory(
        self,
        peer: str,
        feed_id: str,
        archive_date: date,
    ) -> tuple[
        list[ArchiveBlock],
        tuple[str, ...],
        bool,
        tuple[ArchiveBlock, ...],
    ]:
        with self._lan_session() as session:
            with session.get(
                f"{peer}/api/lan/v1/blocks",
                params={"feed_id": feed_id, "date": archive_date.isoformat()},
                headers=self._headers(),
                timeout=(self.connect_timeout, min(self.read_timeout, 30.0)),
                allow_redirects=False,
                stream=True,
            ) as response:
                response.raise_for_status()
                try:
                    content_length = int(
                        response.headers.get("Content-Length") or 0
                    )
                except ValueError as exc:
                    raise LanSyncError(
                        "The peer returned an invalid inventory length."
                    ) from exc
                if content_length > MAX_INVENTORY_BYTES:
                    raise LanSyncError("The peer inventory exceeded the size limit.")
                inventory = bytearray()
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    inventory.extend(chunk)
                    if len(inventory) > MAX_INVENTORY_BYTES:
                        raise LanSyncError(
                            "The peer inventory exceeded the size limit."
                        )
        try:
            payload = json.loads(inventory)
        except (UnicodeDecodeError, ValueError) as exc:
            raise LanSyncError("The peer inventory was not valid JSON.") from exc
        if (
            not isinstance(payload, dict)
            or payload.get("protocol") != LAN_PROTOCOL
            or str(payload.get("feed_id") or "") != feed_id
            or str(payload.get("archive_date") or "") != archive_date.isoformat()
        ):
            raise LanSyncError("The peer returned an incompatible archive inventory.")
        raw_blocks = payload.get("blocks")
        if not isinstance(raw_blocks, list) or len(raw_blocks) > MAX_BLOCKS_PER_DAY:
            raise LanSyncError("The peer returned too many or invalid archive blocks.")
        if any(not isinstance(value, Mapping) for value in raw_blocks):
            raise LanSyncError("The peer returned an invalid archive block entry.")
        blocks = [
            ArchiveBlock.from_mapping(
                value,
                expected_feed_id=feed_id,
                expected_date=archive_date,
            )
            for value in raw_blocks
        ]
        if len({block.filename for block in blocks}) != len(blocks):
            raise LanSyncError("The peer returned duplicate archive blocks.")
        complete = bool(payload.get("complete"))
        raw_completion = payload.get("completion_blocks") or []
        if (
            not isinstance(raw_completion, list)
            or len(raw_completion) > MAX_BLOCKS_PER_DAY
            or any(not isinstance(value, Mapping) for value in raw_completion)
        ):
            raise LanSyncError("The peer returned an invalid completion proof.")
        completion_blocks = tuple(
            ArchiveBlock.from_mapping(
                value,
                expected_feed_id=feed_id,
                expected_date=archive_date,
            )
            for value in raw_completion
        )
        available = {
            (block.filename, block.size, block.sha256): block for block in blocks
        }
        if (
            len({block.filename for block in completion_blocks})
            != len(completion_blocks)
            or (not complete and completion_blocks)
            or any(
                (block.filename, block.size, block.sha256) not in available
                for block in completion_blocks
            )
        ):
            raise LanSyncError("The peer returned an incompatible completion proof.")
        peers = normalize_peer_urls(payload.get("peers") or (), strict=False)
        return blocks, peers, complete, completion_blocks

    def _feed_date_union(
        self,
        feed_id: str,
    ) -> tuple[tuple[date, ...], tuple[str, ...], list[str]]:
        seeds = list(dict.fromkeys((*self.peer_urls, *self._recent_peer_urls())))
        failures: list[str] = []
        if self.discovery_enabled:
            try:
                seeds.extend(discover_lan_peers())
            except OSError as exc:
                failures.append(f"LAN discovery: {exc}")
        queue = list(dict.fromkeys(seeds))[:MAX_LAN_PEERS]
        considered: list[str] = []
        reachable: list[str] = []
        dates: set[date] = set()
        while queue and len(considered) < MAX_LAN_PEERS:
            peer = queue.pop(0)
            if peer in considered:
                continue
            considered.append(peer)
            try:
                peer_dates, advertised = self._peer_feed_dates(peer, feed_id)
                reachable.append(peer)
                dates.update(peer_dates)
                if len(dates) > MAX_FEED_DAYS:
                    raise LanSyncError(
                        "The LAN feed date union exceeded the safety limit."
                    )
                for value in advertised:
                    if (
                        value not in considered
                        and value not in queue
                        and len(considered) + len(queue) < MAX_LAN_PEERS
                    ):
                        queue.append(value)
            except (LanSyncError, requests.RequestException, ValueError) as exc:
                failures.append(f"{peer}: {exc}")
        self._remember_peers(reachable)
        # Reconcile the most recent retained coverage first. This makes a
        # month-scale follower useful immediately while the older tail keeps
        # converging, and matches archive catch-up acquisition order.
        return tuple(sorted(dates, reverse=True)), tuple(reachable), failures

    def _peer_feed_dates(
        self,
        peer: str,
        feed_id: str,
    ) -> tuple[tuple[date, ...], tuple[str, ...]]:
        with self._lan_session() as session:
            with session.get(
                f"{peer}/api/lan/v1/feed-days",
                params={"feed_id": feed_id},
                headers=self._headers(),
                timeout=(self.connect_timeout, min(self.read_timeout, 30.0)),
                allow_redirects=False,
                stream=True,
            ) as response:
                response.raise_for_status()
                payload = self._bounded_json(response, MAX_INVENTORY_BYTES)
        if (
            not isinstance(payload, Mapping)
            or payload.get("protocol") != LAN_PROTOCOL
            or str(payload.get("feed_id") or "") != feed_id
        ):
            raise LanSyncError("The peer returned an incompatible feed date list.")
        raw_dates = payload.get("dates")
        if not isinstance(raw_dates, list) or len(raw_dates) > MAX_FEED_DAYS:
            raise LanSyncError("The peer returned too many or invalid feed dates.")
        try:
            dates = tuple(date.fromisoformat(str(value)) for value in raw_dates)
        except ValueError as exc:
            raise LanSyncError("The peer returned an invalid feed date.") from exc
        if len(set(dates)) != len(dates):
            raise LanSyncError("The peer returned duplicate feed dates.")
        peers = normalize_peer_urls(payload.get("peers") or (), strict=False)
        return dates, peers

    def _transcript_inventory(
        self,
        peer: str,
        feed_id: str,
        archive_date: date,
        processing_fingerprint: str,
    ) -> tuple[list[TranscriptArtifact], tuple[str, ...]]:
        with self._lan_session() as session:
            with session.get(
                f"{peer}/api/lan/v1/transcripts",
                params={
                    "feed_id": feed_id,
                    "date": archive_date.isoformat(),
                    "processing_fingerprint": processing_fingerprint,
                },
                headers=self._headers(),
                timeout=(self.connect_timeout, min(self.read_timeout, 30.0)),
                allow_redirects=False,
                stream=True,
            ) as response:
                response.raise_for_status()
                payload = self._bounded_json(response, MAX_INVENTORY_BYTES)
        if (
            not isinstance(payload, Mapping)
            or payload.get("protocol") != LAN_PROTOCOL
            or str(payload.get("feed_id") or "") != feed_id
            or str(payload.get("archive_date") or "") != archive_date.isoformat()
            or str(payload.get("processing_fingerprint") or "").lower()
            != processing_fingerprint
        ):
            raise LanSyncError(
                "The peer returned an incompatible transcript inventory."
            )
        raw_artifacts = payload.get("artifacts")
        if (
            not isinstance(raw_artifacts, list)
            or len(raw_artifacts) > MAX_TRANSCRIPT_ARTIFACTS_PER_DAY
            or any(not isinstance(value, Mapping) for value in raw_artifacts)
        ):
            raise LanSyncError("The peer returned invalid transcript artifacts.")
        artifacts = [
            TranscriptArtifact.from_mapping(
                value,
                expected_feed_id=feed_id,
                expected_date=archive_date,
                expected_fingerprint=processing_fingerprint,
            )
            for value in raw_artifacts
        ]
        if len({(value.kind, value.filename) for value in artifacts}) != len(
            artifacts
        ):
            raise LanSyncError("The peer returned duplicate transcript artifacts.")
        groups: dict[str, list[TranscriptArtifact]] = {}
        for artifact in artifacts:
            groups.setdefault(artifact.audio_filename, []).append(artifact)
        for audio_filename, group in groups.items():
            required = {"audio", "json", "text"}
            if audio_filename.startswith("combined_"):
                required.add("manifest")
            if {value.kind for value in group} != required or len(
                {value.audio_sha256 for value in group}
            ) != 1:
                raise LanSyncError(
                    "The peer returned an incomplete transcript artifact set."
                )
        peers = normalize_peer_urls(payload.get("peers") or (), strict=False)
        return artifacts, peers

    def _download_transcript_artifact(
        self,
        peer: str,
        artifact: TranscriptArtifact,
        target: Path,
    ) -> int:
        endpoint = (
            f"{peer}/api/lan/v1/transcripts/{quote(artifact.feed_id, safe='')}/"
            f"{quote(artifact.archive_date, safe='')}/"
            f"{quote(artifact.processing_fingerprint, safe='')}/"
            f"{quote(artifact.filename, safe='')}"
        )
        headers = self._headers()
        headers["Accept"] = "application/octet-stream"
        partial = target.with_name(
            f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.lan-part"
        )
        digest = hashlib.sha256()
        received = 0
        try:
            with self._lan_session() as session:
                with session.get(
                    endpoint,
                    headers=headers,
                    stream=True,
                    timeout=(self.connect_timeout, self.read_timeout),
                    allow_redirects=False,
                ) as response:
                    response.raise_for_status()
                    if str(
                        response.headers.get("X-Radio-Archive-SHA256") or ""
                    ).lower() != artifact.sha256:
                        raise LanSyncError(
                            "The transcript response hash changed after inventory."
                        )
                    try:
                        content_length = int(
                            response.headers.get("Content-Length") or 0
                        )
                    except ValueError as exc:
                        raise LanSyncError(
                            "The peer returned an invalid artifact length."
                        ) from exc
                    if content_length != artifact.size:
                        raise LanSyncError(
                            "The transcript response size changed after inventory."
                        )
                    with partial.open("xb") as handle:
                        for chunk in response.iter_content(chunk_size=256 * 1024):
                            if not chunk:
                                continue
                            received += len(chunk)
                            if received > artifact.size:
                                raise LanSyncError(
                                    "The peer sent more transcript data than advertised."
                                )
                            digest.update(chunk)
                            handle.write(chunk)
                        handle.flush()
                        os.fsync(handle.fileno())
            if received != artifact.size or digest.hexdigest() != artifact.sha256:
                raise LanSyncError(
                    "The copied transcript artifact failed size/hash verification."
                )
            try:
                os.link(partial, target)
            except FileExistsError as exc:
                raise LanSyncError(
                    "A local artifact appeared before the transfer completed."
                ) from exc
            except OSError as exc:
                raise LanSyncError(
                    "The verified transcript artifact could not be published atomically."
                ) from exc
            partial.unlink()
            return received
        finally:
            try:
                partial.unlink(missing_ok=True)
            except OSError:
                pass

    def _download_block(
        self,
        peer: str,
        block: ArchiveBlock,
        target: Path,
    ) -> int:
        endpoint = (
            f"{peer}/api/lan/v1/blocks/{quote(block.feed_id, safe='')}/"
            f"{quote(block.archive_date, safe='')}/{quote(block.filename, safe='')}"
        )
        headers = self._headers()
        headers["Accept"] = "audio/mpeg, application/octet-stream"
        partial = target.with_name(
            f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.lan-part"
        )
        digest = hashlib.sha256()
        received = 0
        try:
            with self._lan_session() as session:
                with session.get(
                    endpoint,
                    headers=headers,
                    stream=True,
                    timeout=(self.connect_timeout, self.read_timeout),
                    allow_redirects=False,
                ) as response:
                    response.raise_for_status()
                    response_hash = str(
                        response.headers.get("X-Radio-Archive-SHA256") or ""
                    ).lower()
                    if response_hash != block.sha256:
                        raise LanSyncError(
                            "The peer response hash changed after inventory."
                        )
                    try:
                        content_length = int(
                            response.headers.get("Content-Length") or 0
                        )
                    except ValueError as exc:
                        raise LanSyncError(
                            "The peer returned an invalid block length."
                        ) from exc
                    if content_length != block.size:
                        raise LanSyncError(
                            "The peer response size changed after inventory."
                        )
                    with partial.open("xb") as handle:
                        for chunk in response.iter_content(chunk_size=1024 * 256):
                            if not chunk:
                                continue
                            received += len(chunk)
                            if received > block.size:
                                raise LanSyncError(
                                    "The peer sent more data than advertised."
                                )
                            digest.update(chunk)
                            handle.write(chunk)
                        handle.flush()
                        os.fsync(handle.fileno())
            if received != block.size:
                raise LanSyncError("The peer transfer ended before the block was complete.")
            if digest.hexdigest() != block.sha256:
                raise LanSyncError("The copied archive block failed SHA-256 verification.")
            try:
                # A hard-link publish is atomic and cannot replace a local file
                # that appeared during transfer. The partial is in the same
                # directory/filesystem and is unlinked immediately afterward.
                os.link(partial, target)
            except FileExistsError as exc:
                raise LanSyncError(
                    "A local file appeared before the transfer completed."
                ) from exc
            except OSError as exc:
                raise LanSyncError(
                    "The verified block could not be published atomically."
                ) from exc
            partial.unlink()
            return received
        finally:
            try:
                partial.unlink(missing_ok=True)
            except OSError:
                pass


class _LanLeaseHeartbeat(AbstractContextManager["_LanLeaseHeartbeat"]):
    def __init__(
        self,
        client: LanArchiveSyncClient,
        turn: LanDownloadTurn,
    ) -> None:
        self.client = client
        self.turn = turn
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._warnings: list[str] = []
        self._last_success = time.monotonic()
        self._lost_reason = ""
        self._state_lock = threading.Lock()

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(self._warnings)

    def assert_active(self) -> None:
        with self._state_lock:
            reason = self._lost_reason
        if reason:
            raise LanSyncError(reason)

    def __enter__(self) -> "_LanLeaseHeartbeat":
        self._thread = threading.Thread(
            target=self._run,
            name="radio-archive-lan-lease-heartbeat",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None

    def _run(self) -> None:
        interval = min(
            30.0,
            max(5.0, float(self.turn.lease_seconds or 30.0) / 3.0),
        )
        archive_date = date.fromisoformat(self.turn.archive_date)
        while not self._stop.wait(interval):
            try:
                status = self.client._queue_action(
                    self.turn.coordinator_url,
                    "renew",
                    self.turn.feed_id,
                    archive_date,
                    {"lease_token": self.turn.lease_token},
                )
                if status["state"] != "active":
                    reason = "The LAN acquisition lease is no longer active."
                    with self._state_lock:
                        self._lost_reason = reason
                    self._warnings.append(reason)
                    return
                with self._state_lock:
                    self._last_success = time.monotonic()
            except (
                LanSyncError,
                requests.RequestException,
                ValueError,
            ) as exc:
                warning = (
                    "The LAN acquisition heartbeat could not reach its "
                    f"coordinator: {exc}"
                )
                if len(self._warnings) < 20:
                    self._warnings.append(warning)
                with self._state_lock:
                    elapsed = time.monotonic() - self._last_success
                    if elapsed >= max(
                        10.0,
                        float(self.turn.lease_seconds) * 0.75,
                    ):
                        self._lost_reason = (
                            "The LAN acquisition lease could not be renewed; "
                            "new upstream archive requests were stopped."
                        )


class _LanProcessingLeaseHeartbeat(
    AbstractContextManager["_LanProcessingLeaseHeartbeat"]
):
    def __init__(
        self,
        client: LanArchiveSyncClient,
        turn: LanProcessingTurn,
    ) -> None:
        self.client = client
        self.turn = turn
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._warnings: list[str] = []
        self._last_success = time.monotonic()
        self._lost_reason = ""
        self._state_lock = threading.Lock()

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(self._warnings)

    def assert_active(self) -> None:
        with self._state_lock:
            reason = self._lost_reason
        if reason:
            raise LanSyncError(reason)

    def __enter__(self) -> "_LanProcessingLeaseHeartbeat":
        self._thread = threading.Thread(
            target=self._run,
            name="radio-archive-lan-processing-heartbeat",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None

    def _run(self) -> None:
        interval = min(
            30.0,
            max(5.0, float(self.turn.lease_seconds or 30.0) / 3.0),
        )
        archive_date = date.fromisoformat(self.turn.archive_date)
        while not self._stop.wait(interval):
            try:
                status = self.client._processing_queue_action(
                    self.turn.coordinator_url,
                    "renew",
                    self.turn.feed_id,
                    archive_date,
                    self.turn.processing_fingerprint,
                    {"lease_token": self.turn.lease_token},
                )
                if status["state"] != "active":
                    reason = "The LAN processing lease is no longer active."
                    with self._state_lock:
                        self._lost_reason = reason
                    self._warnings.append(reason)
                    return
                with self._state_lock:
                    self._last_success = time.monotonic()
            except (
                LanSyncError,
                requests.RequestException,
                ValueError,
            ) as exc:
                warning = (
                    "The LAN processing heartbeat could not reach its "
                    f"coordinator: {exc}"
                )
                if len(self._warnings) < 20:
                    self._warnings.append(warning)
                with self._state_lock:
                    elapsed = time.monotonic() - self._last_success
                    if elapsed >= max(
                        10.0,
                        float(self.turn.lease_seconds) * 0.75,
                    ):
                        self._lost_reason = (
                            "The LAN processing lease could not be renewed; "
                            "model work was stopped before publishing a shared result."
                        )
