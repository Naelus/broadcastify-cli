from __future__ import annotations

import hashlib
import http.client
import json
import socket
import threading
from datetime import date
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from broadcastify_cli.lan_sync import (
    LanArchiveCatalog,
    LanArchiveSyncClient,
    LanDiscoveryResponder,
    LanSyncResult,
    discover_lan_peers,
    normalize_peer_url,
    normalize_peer_urls,
)
from broadcastify_cli.models import JobRequest
from broadcastify_cli.lan_node import create_lan_node_server, validate_lan_host
from broadcastify_cli.web_app import create_server


def _raw_day(root: Path) -> tuple[Path, list[Path]]:
    day = root / "90001" / "20260712"
    day.mkdir(parents=True)
    raw = [
        day / "202607120000-123456-90001.mp3",
        day / "202607120030-654321-90001.mp3",
    ]
    raw[0].write_bytes(b"first retained archive block")
    raw[1].write_bytes(b"second retained archive block")
    (day / "combined_90001_20260712.mp3").write_bytes(b"private derived audio")
    (day / "notes.mp3").write_bytes(b"not a source block")
    return day, raw


def test_peer_urls_are_limited_to_numeric_private_addresses() -> None:
    assert normalize_peer_url("http://10.20.30.40:8765/") == (
        "http://10.20.30.40:8765"
    )
    assert normalize_peer_urls(
        "http://192.168.1.2:8765, http://192.168.1.2:8765"
    ) == ("http://192.168.1.2:8765",)

    for value in [
        "https://example.com",
        "http://8.8.8.8:8765",
        "http://localhost:8765",
        "http://user:secret@10.0.0.2:8765",
        "http://10.0.0.2:8765/untrusted/path",
    ]:
        with pytest.raises(ValueError):
            normalize_peer_url(value)


def test_catalog_exposes_only_original_blocks_for_the_requested_day(
    tmp_path: Path,
) -> None:
    _day, raw = _raw_day(tmp_path)
    catalog = LanArchiveCatalog(tmp_path, enabled=True)

    blocks = catalog.inventory("90001", date(2026, 7, 12))

    assert [block.filename for block in blocks] == [path.name for path in raw]
    assert [block.sha256 for block in blocks] == [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in raw
    ]
    assert all(block.feed_id == "90001" for block in blocks)
    assert catalog.inventory("90001", date(2026, 7, 13)) == []


def test_hash_verified_peer_sync_copies_missing_blocks_without_a_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "peer"
    _day, raw = _raw_day(source)
    monkeypatch.setenv("BROADCASTIFY_LAN_SHARING", "true")
    monkeypatch.setenv("BROADCASTIFY_LAN_SYNC_KEY", "shared-test-key")
    monkeypatch.setenv("BROADCASTIFY_LAN_DISCOVERY_ENABLED", "false")
    server = create_server(source, port=0, working_dir=tmp_path)
    server.quiet = True  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    peer_url = f"http://127.0.0.1:{server.server_port}"
    target = tmp_path / "client"
    client = LanArchiveSyncClient(
        enabled=True,
        peer_urls=[peer_url],
        discovery_enabled=False,
        sync_key="shared-test-key",
    )
    try:
        first = client.sync_day(target, "90001", date(2026, 7, 12))
        second = client.sync_day(target, "90001", date(2026, 7, 12))

        copied = sorted((target / "90001" / "20260712").glob("*.mp3"))
        assert first.peers_reached == 1
        assert first.blocks_available == 2
        assert first.blocks_copied == 2
        assert first.bytes_copied == sum(path.stat().st_size for path in raw)
        assert first.failures == ()
        assert second.blocks_copied == 0
        assert second.blocks_already_local == 2
        assert [path.name for path in copied] == [path.name for path in raw]
        assert [path.read_bytes() for path in copied] == [
            path.read_bytes() for path in raw
        ]

        wrong_key = LanArchiveSyncClient(
            enabled=True,
            peer_urls=[peer_url],
            discovery_enabled=False,
            sync_key="wrong-key",
        ).sync_day(tmp_path / "wrong-key", "90001", date(2026, 7, 12))
        assert wrong_key.peers_reached == 0
        assert wrong_key.blocks_copied == 0
        assert any("403" in failure for failure in wrong_key.failures)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_disabled_web_peer_does_not_expose_an_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BROADCASTIFY_LAN_SHARING", "false")
    monkeypatch.setenv("BROADCASTIFY_LAN_DISCOVERY_ENABLED", "false")
    server = create_server(tmp_path, port=0, working_dir=tmp_path)
    server.quiet = True  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection(
        "127.0.0.1",
        server.server_port,
        timeout=5,
    )
    try:
        connection.request(
            "GET",
            "/api/lan/v1/blocks?feed_id=90001&date=2026-07-12",
        )
        response = connection.getresponse()
        payload = json.loads(response.read())
        assert response.status == 404
        assert "not enabled" in payload["error"].lower()
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_minimal_native_lan_node_seeds_blocks_without_web_app_data(
    tmp_path: Path,
) -> None:
    source = tmp_path / "native-peer"
    _day, raw = _raw_day(source)
    server = create_lan_node_server(
        source,
        host="127.0.0.1",
        port=0,
        sync_key="native-test-key",
        discovery_enabled=False,
    )
    server.quiet = True  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    target = tmp_path / "consumer"
    try:
        result = LanArchiveSyncClient(
            enabled=True,
            peer_urls=[f"http://127.0.0.1:{server.server_port}"],
            discovery_enabled=False,
            sync_key="native-test-key",
        ).sync_day(target, "90001", date(2026, 7, 12))

        copied = sorted((target / "90001" / "20260712").glob("*.mp3"))
        assert result.blocks_copied == 2
        assert [path.read_bytes() for path in copied] == [
            path.read_bytes() for path in raw
        ]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_corrupt_peer_body_is_never_promoted_to_the_archive_library(
    tmp_path: Path,
) -> None:
    expected = b"good"
    expected_hash = hashlib.sha256(expected).hexdigest()
    filename = "202607120000-123456-90001.mp3"

    class CorruptHandler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args: object) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            if self.path.startswith("/api/lan/v1/blocks?"):
                body = json.dumps(
                    {
                        "protocol": "radio-archive-lan/1",
                        "feed_id": "90001",
                        "archive_date": "2026-07-12",
                        "peers": [],
                        "blocks": [
                            {
                                "feed_id": "90001",
                                "archive_date": "2026-07-12",
                                "filename": filename,
                                "size": len(expected),
                                "sha256": expected_hash,
                                "modified_ns": 1,
                            }
                        ],
                    }
                ).encode()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            body = b"evil"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "audio/mpeg")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Radio-Archive-SHA256", expected_hash)
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), CorruptHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = LanArchiveSyncClient(
            enabled=True,
            peer_urls=[f"http://127.0.0.1:{server.server_port}"],
            discovery_enabled=False,
        ).sync_day(tmp_path / "target", "90001", date(2026, 7, 12))

        day = tmp_path / "target" / "90001" / "20260712"
        assert result.blocks_copied == 0
        assert any("SHA-256" in failure for failure in result.failures)
        assert not (day / filename).exists()
        assert list(day.glob("*.lan-part")) == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_native_lan_node_rejects_public_bind_addresses() -> None:
    with pytest.raises(ValueError, match="public"):
        validate_lan_host("8.8.8.8")


def test_one_hop_discovery_finds_a_read_only_lan_peer(tmp_path: Path) -> None:
    catalog = LanArchiveCatalog(tmp_path, enabled=True)
    reservation = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    reservation.bind(("127.0.0.1", 0))
    port = int(reservation.getsockname()[1])
    reservation.close()
    responder = LanDiscoveryResponder(
        catalog,
        lambda _remote: "http://127.0.0.1:8766",
        port=port,
    )
    try:
        assert responder.start() is True
        found = discover_lan_peers(
            timeout=0.5,
            port=port,
            destinations=("127.0.0.1",),
            cache_ttl=0,
        )
        assert found == ("http://127.0.0.1:8766",)
    finally:
        responder.close()


def test_job_request_rejects_public_peer_urls() -> None:
    with pytest.raises(ValueError, match="private"):
        JobRequest.from_dict(
            {
                "feed_id": "90001",
                "start_date": "2026-07-12",
                "end_date": "2026-07-12",
                "lan_sync_enabled": True,
                "lan_peer_urls": ["http://8.8.8.8:8765"],
            }
        )


def test_lan_result_remains_serializable_for_worker_events() -> None:
    result = LanSyncResult(
        enabled=True,
        peers_considered=2,
        peers_reached=1,
        blocks_copied=3,
        bytes_copied=123,
        failures=("one peer was unavailable",),
    )

    assert json.loads(json.dumps(result.to_dict()))["blocks_copied"] == 3
