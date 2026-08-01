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

from .archive_cache import (
    archive_identities_for_filename,
    remember_archive_identity,
)

LAN_PROTOCOL = "radio-archive-lan/1"
LAN_DISCOVERY_MAGIC = b"RADIO-ARCHIVE-LAN-DISCOVER/1 "
LAN_DISCOVERY_PORT = 48_765
LAN_MULTICAST_ADDRESS = "239.255.77.77"
MAX_LAN_PEERS = 24
MAX_BLOCKS_PER_DAY = 128
MAX_ARCHIVE_BLOCK_BYTES = 256 * 1024 * 1024
MAX_INVENTORY_BYTES = 1024 * 1024
MAX_LAN_QUEUE_ENTRIES = 512
LAN_QUEUE_LEASE_SECONDS = 90.0
LAN_QUEUE_RESULT_SECONDS = 24 * 60 * 60.0
LAN_QUEUE_ROLLING_RESULT_SECONDS = 5 * 60.0
LAN_QUEUE_REQUEST_BYTES = 256 * 1024
LAN_QUEUE_RESPONSE_BYTES = 256 * 1024
LAN_QUEUE_NODE_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,80}$")
LAN_QUEUE_SCOPE_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
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
        if len(self.archive_identities) > 64:
            raise LanSyncError("A LAN peer advertised too many archive identities.")
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
        # Preserve aliases when multiple provider IDs resolve to that same MP3.
        remember_archive_identity(
            day_directory,
            feed_id,
            archive_date,
            identity.archive_id,
            source_file,
            listing_prefix=identity.listing_prefix,
            allow_filename_alias=True,
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
        }
        if entry is None:
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
        }

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

        return cls(
            enabled=enabled and environment_enabled,
            peer_urls=configured_peer_urls(peer_urls),
            discovery_enabled=discovery_enabled and environment_discovery,
            sync_key=os.getenv("BROADCASTIFY_LAN_SYNC_KEY") or "",
            queue_enabled=environment_flag(
                "BROADCASTIFY_LAN_QUEUE_ENABLED",
                default=True,
            ),
            quota_scope=os.getenv("BROADCASTIFY_LAN_QUOTA_SCOPE") or "default",
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
        while queue and len(considered) < MAX_LAN_PEERS:
            peer = queue.pop(0)
            if peer in considered:
                continue
            considered.append(peer)
            try:
                blocks, advertised_peers = self._inventory(
                    peer,
                    feed_id,
                    archive_date,
                )
                reached += 1
                reachable.append(peer)
                for block in blocks:
                    candidates.setdefault(block.filename, []).append((peer, block))
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
        return LanSyncResult(
            enabled=True,
            peers_considered=len(considered),
            peers_reached=reached,
            blocks_available=len(candidates),
            blocks_already_local=already_local,
            blocks_copied=copied,
            bytes_copied=copied_bytes,
            conflicts=conflicts,
            failures=tuple(failures[:50]),
        )

    def wait_for_download_turn(
        self,
        output_dir: str | Path,
        feed_id: str,
        archive_date: date,
        *,
        progress: ProgressCallback | None = None,
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
        observed_shared_work = False
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

    def _select_coordinator(self) -> tuple[str, dict[str, Any]] | None:
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
                    if info["acquisition_queue_available"]:
                        coordinators.append((str(info["node_id"]), peer, info))
        if not coordinators:
            return None
        _node_id, peer, info = min(
            coordinators,
            key=lambda value: (value[0], value[1]),
        )
        return peer, info

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
            "peers": normalize_peer_urls(payload.get("peers") or (), strict=False),
        }

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
        }
        lease_token = str(payload.get("lease_token") or "")
        if value["granted"]:
            if not re.fullmatch(r"[A-Za-z0-9_-]{24,128}", lease_token):
                raise LanSyncError("The peer returned an invalid acquisition lease.")
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
    ) -> tuple[list[ArchiveBlock], tuple[str, ...]]:
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
        peers = normalize_peer_urls(payload.get("peers") or (), strict=False)
        return blocks, peers

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
