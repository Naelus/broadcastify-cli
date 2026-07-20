from __future__ import annotations

import ctypes
import hashlib
import hmac
import ipaddress
import json
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
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse, urlunparse

import requests


LAN_PROTOCOL = "radio-archive-lan/1"
LAN_DISCOVERY_MAGIC = b"RADIO-ARCHIVE-LAN-DISCOVER/1 "
LAN_DISCOVERY_PORT = 48_765
LAN_MULTICAST_ADDRESS = "239.255.77.77"
MAX_LAN_PEERS = 24
MAX_BLOCKS_PER_DAY = 128
MAX_ARCHIVE_BLOCK_BYTES = 256 * 1024 * 1024
MAX_INVENTORY_BYTES = 1024 * 1024
RAW_ARCHIVE_PATTERN = re.compile(
    r"^(?P<stamp>\d{12})-(?P<archive_id>\d+)-(?P<feed_id>\d+)\.mp3$",
    re.IGNORECASE,
)

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
class ArchiveBlock:
    feed_id: str
    archive_date: str
    filename: str
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
    ) -> "ArchiveBlock":
        block = cls(
            feed_id=str(value.get("feed_id") or ""),
            archive_date=str(value.get("archive_date") or ""),
            filename=str(value.get("filename") or ""),
            size=int(value.get("size") or 0),
            sha256=str(value.get("sha256") or "").lower(),
            modified_ns=int(value.get("modified_ns") or 0),
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
            or match.group("stamp")[:8] != expected_date.strftime("%Y%m%d")
        ):
            raise LanSyncError("A LAN peer advertised an invalid archive block name.")
        if not 0 < self.size <= MAX_ARCHIVE_BLOCK_BYTES:
            raise LanSyncError("A LAN peer advertised an invalid archive block size.")
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise LanSyncError("A LAN peer advertised an invalid archive block hash.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    ) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.enabled = bool(enabled)
        self.sync_key = str(sync_key or "")
        self.peer_urls = normalize_peer_urls(peer_urls)
        self.node_id = node_id or secrets.token_hex(12)
        self.hashes = ArchiveHashCache()
        self.discovery_available = False
        self.discovery_error = ""

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
                or match.group("stamp")[:8] != archive_date.strftime("%Y%m%d")
                or path.is_symlink()
            ):
                continue
            resolved = path.resolve()
            if resolved.parent != day_dir or not resolved.is_file():
                continue
            stat = resolved.stat()
            if not 0 < stat.st_size <= MAX_ARCHIVE_BLOCK_BYTES:
                continue
            blocks.append(
                ArchiveBlock(
                    feed_id=feed_id,
                    archive_date=archive_date.isoformat(),
                    filename=path.name,
                    size=stat.st_size,
                    sha256=self.hashes.sha256(resolved),
                    modified_ns=stat.st_mtime_ns,
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
            or match.group("stamp")[:8] != archive_date.strftime("%Y%m%d")
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
        block = ArchiveBlock(
            feed_id=feed_id,
            archive_date=archive_date.isoformat(),
            filename=filename,
            size=stat.st_size,
            sha256=self.hashes.sha256(path),
            modified_ns=stat.st_mtime_ns,
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
    """Copy missing source blocks from a bounded trusted-LAN peer pool."""

    def __init__(
        self,
        *,
        enabled: bool,
        peer_urls: Sequence[str] = (),
        discovery_enabled: bool = True,
        sync_key: str = "",
        connect_timeout: float = 2.0,
        read_timeout: float = 120.0,
    ) -> None:
        self.enabled = bool(enabled)
        self.peer_urls = normalize_peer_urls(peer_urls)
        self.discovery_enabled = bool(discovery_enabled)
        self.sync_key = str(sync_key or "")
        self.connect_timeout = max(0.1, float(connect_timeout))
        self.read_timeout = max(1.0, float(read_timeout))
        self.hashes = ArchiveHashCache()

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
        return cls(
            enabled=enabled and environment_enabled,
            peer_urls=configured_peer_urls(peer_urls),
            discovery_enabled=discovery_enabled and environment_discovery,
            sync_key=os.getenv("BROADCASTIFY_LAN_SYNC_KEY") or "",
        )

    def sync_day(
        self,
        output_dir: str | Path,
        feed_id: str,
        archive_date: date,
        *,
        progress: ProgressCallback | None = None,
    ) -> LanSyncResult:
        if not self.enabled:
            return LanSyncResult(enabled=False)
        if not feed_id.isdigit():
            raise ValueError("A numeric feed ID is required for LAN archive sync.")
        seeds = list(self.peer_urls)
        failures: list[str] = []
        if self.discovery_enabled:
            try:
                seeds.extend(discover_lan_peers())
            except OSError as exc:
                failures.append(f"LAN discovery: {exc}")
        queue = list(dict.fromkeys(seeds))[:MAX_LAN_PEERS]
        considered: list[str] = []
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
                    if progress:
                        progress(
                            f"Copied {copied} missing archive block"
                            f"{'s' if copied != 1 else ''} from the LAN "
                            f"({copied_bytes / (1024 * 1024):.1f} MiB)."
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

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "User-Agent": "RadioArchiveLAN/1",
        }
        if self.sync_key:
            headers["X-Radio-Archive-LAN-Key"] = self.sync_key
        return headers

    def _inventory(
        self,
        peer: str,
        feed_id: str,
        archive_date: date,
    ) -> tuple[list[ArchiveBlock], tuple[str, ...]]:
        with requests.get(
            f"{peer}/api/lan/v1/blocks",
            params={"feed_id": feed_id, "date": archive_date.isoformat()},
            headers=self._headers(),
            timeout=(self.connect_timeout, min(self.read_timeout, 30.0)),
            allow_redirects=False,
            stream=True,
        ) as response:
            response.raise_for_status()
            try:
                content_length = int(response.headers.get("Content-Length") or 0)
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
                    raise LanSyncError("The peer inventory exceeded the size limit.")
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
            with requests.get(
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
                    raise LanSyncError("The peer response hash changed after inventory.")
                try:
                    content_length = int(response.headers.get("Content-Length") or 0)
                except ValueError as exc:
                    raise LanSyncError("The peer returned an invalid block length.") from exc
                if content_length != block.size:
                    raise LanSyncError("The peer response size changed after inventory.")
                with partial.open("xb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 256):
                        if not chunk:
                            continue
                        received += len(chunk)
                        if received > block.size:
                            raise LanSyncError("The peer sent more data than advertised.")
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
