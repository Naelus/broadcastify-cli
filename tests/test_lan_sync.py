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

import broadcastify_cli.lan_sync as lan_sync
from broadcastify_cli.lan_sync import (
    LanArchiveCatalog,
    LanArchiveSyncClient,
    LanAcquisitionQueue,
    LanDiscoveryResponder,
    LanDownloadTurn,
    LanSyncResult,
    discovery_destinations,
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


def _force_unusable_system_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        monkeypatch.setenv(name, "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")


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


def test_shared_acquisition_queue_grants_one_expiring_producer_lease() -> None:
    now = [100.0]
    queue = LanAcquisitionQueue(
        lease_seconds=15.0,
        result_seconds=60.0,
        clock=lambda: now[0],
    )
    archive_date = date(2026, 7, 12)

    first = queue.claim(
        "default",
        "90001",
        archive_date,
        owner_node_id="producer_one",
        producer_url="http://10.20.30.40:8766",
        requester_address="10.20.30.40",
    )
    second = queue.claim(
        "default",
        "90001",
        archive_date,
        owner_node_id="producer_two",
        producer_url="http://10.20.30.41:8766",
        requester_address="10.20.30.41",
    )

    assert first["granted"] is True
    assert first["state"] == "active"
    assert second["granted"] is False
    assert second["owner_node_id"] == "producer_one"

    now[0] += 16.0
    takeover = queue.claim(
        "default",
        "90001",
        archive_date,
        owner_node_id="producer_two",
        producer_url="http://10.20.30.41:8766",
        requester_address="10.20.30.41",
    )
    assert takeover["granted"] is True
    assert takeover["owner_node_id"] == "producer_two"


def test_shared_quota_result_prevents_follower_retry_until_it_expires() -> None:
    now = [200.0]
    queue = LanAcquisitionQueue(
        lease_seconds=15.0,
        result_seconds=60.0,
        clock=lambda: now[0],
    )
    archive_date = date(2026, 7, 12)
    claim = queue.claim(
        "premium-account",
        "90001",
        archive_date,
        owner_node_id="producer_one",
        producer_url="http://10.20.30.40:8766",
        requester_address="10.20.30.40",
    )
    limited = queue.finish(
        "premium-account",
        "90001",
        archive_date,
        lease_token=str(claim["lease_token"]),
        outcome="quota_limited",
        block_count=7,
    )
    follower = queue.claim(
        "premium-account",
        "90001",
        archive_date,
        owner_node_id="producer_two",
        producer_url="http://10.20.30.41:8766",
        requester_address="10.20.30.41",
    )

    assert limited["state"] == "quota_limited"
    assert limited["block_count"] == 7
    assert follower["granted"] is False
    assert follower["state"] == "quota_limited"

    now[0] += 61.0
    assert queue.status("premium-account", "90001", archive_date)["state"] == (
        "available"
    )


def test_web_queue_elects_one_producer_and_follower_pulls_completed_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "producer"
    source.mkdir()
    monkeypatch.setenv("BROADCASTIFY_LAN_SHARING", "true")
    monkeypatch.setenv("BROADCASTIFY_LAN_SYNC_KEY", "queue-test-key")
    monkeypatch.setenv("BROADCASTIFY_LAN_DISCOVERY_ENABLED", "false")
    _force_unusable_system_proxy(monkeypatch)
    server = create_server(source, port=0, working_dir=tmp_path)
    server.quiet = True  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    peer_url = f"http://127.0.0.1:{server.server_port}"
    archive_date = date(2026, 7, 12)
    leader = LanArchiveSyncClient(
        enabled=True,
        peer_urls=(peer_url,),
        discovery_enabled=False,
        sync_key="queue-test-key",
        producer_url=peer_url,
        queue_poll_interval=0.05,
        queue_max_wait=3.0,
    )
    follower = LanArchiveSyncClient(
        enabled=True,
        peer_urls=(peer_url,),
        discovery_enabled=False,
        sync_key="queue-test-key",
        queue_poll_interval=0.05,
        queue_max_wait=3.0,
        queue_consumer_grace=0.05,
    )
    follower_result: list[LanDownloadTurn] = []
    try:
        leader_turn = leader.wait_for_download_turn(
            source,
            "90001",
            archive_date,
        )
        assert leader_turn.role == "leader"

        follower_thread = threading.Thread(
            target=lambda: follower_result.append(
                follower.wait_for_download_turn(
                    tmp_path / "consumer",
                    "90001",
                    archive_date,
                )
            ),
            daemon=True,
        )
        follower_thread.start()
        day = source / "90001" / "20260712"
        day.mkdir(parents=True)
        block = day / "202607120000-123456-90001.mp3"
        block.write_bytes(b"completed by the elected producer")
        assert (
            leader.finish_download_turn(
                leader_turn,
                outcome="complete",
                block_count=1,
                source_files=(block,),
            )
            == ""
        )
        follower_thread.join(timeout=5)

        assert len(follower_result) == 1
        assert follower_result[0].role == "completed"
        assert follower_result[0].sync_result.blocks_copied == 1
        copied = follower_result[0].audio_files
        assert len(copied) == 1
        assert copied[0].read_bytes() == block.read_bytes()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_completed_manifest_is_assembled_from_multiple_peer_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_root = tmp_path / "manifest-source"
    _day, raw = _raw_day(manifest_root)
    seed_roots = (tmp_path / "seed-one", tmp_path / "seed-two")
    for seed_root, source in zip(seed_roots, raw, strict=True):
        seed_day = seed_root / "90001" / "20260712"
        seed_day.mkdir(parents=True)
        (seed_day / source.name).write_bytes(source.read_bytes())

    monkeypatch.setenv("BROADCASTIFY_LAN_QUEUE_ENABLED", "false")
    seeds = [
        create_lan_node_server(
            seed_root,
            host="127.0.0.1",
            port=0,
            sync_key="multi-source-key",
            discovery_enabled=False,
        )
        for seed_root in seed_roots
    ]
    seed_threads = [
        threading.Thread(target=server.serve_forever, daemon=True)
        for server in seeds
    ]
    for server, thread in zip(seeds, seed_threads, strict=True):
        server.quiet = True  # type: ignore[attr-defined]
        thread.start()
    seed_urls = tuple(
        f"http://127.0.0.1:{server.server_port}" for server in seeds
    )

    monkeypatch.setenv("BROADCASTIFY_LAN_QUEUE_ENABLED", "true")
    monkeypatch.setenv("BROADCASTIFY_LAN_SHARING", "true")
    monkeypatch.setenv("BROADCASTIFY_LAN_SYNC_KEY", "multi-source-key")
    monkeypatch.setenv("BROADCASTIFY_LAN_DISCOVERY_ENABLED", "false")
    monkeypatch.setenv("BROADCASTIFY_LAN_PEERS", " ".join(seed_urls))
    coordinator_root = tmp_path / "coordinator"
    coordinator_root.mkdir()
    coordinator = create_server(
        coordinator_root,
        port=0,
        working_dir=tmp_path,
    )
    coordinator.quiet = True  # type: ignore[attr-defined]
    coordinator_thread = threading.Thread(
        target=coordinator.serve_forever,
        daemon=True,
    )
    coordinator_thread.start()
    coordinator_url = f"http://127.0.0.1:{coordinator.server_port}"
    archive_date = date(2026, 7, 12)
    leader = LanArchiveSyncClient(
        enabled=True,
        peer_urls=(coordinator_url,),
        discovery_enabled=False,
        sync_key="multi-source-key",
        producer_url=seed_urls[0],
        queue_poll_interval=0.05,
        queue_max_wait=2.0,
    )
    follower = LanArchiveSyncClient(
        enabled=True,
        peer_urls=(coordinator_url,),
        discovery_enabled=False,
        sync_key="multi-source-key",
        queue_poll_interval=0.05,
        queue_max_wait=2.0,
    )
    try:
        turn = leader.wait_for_download_turn(
            manifest_root,
            "90001",
            archive_date,
        )
        assert turn.role == "leader"
        assert (
            leader.finish_download_turn(
                turn,
                outcome="complete",
                block_count=2,
                source_files=raw,
            )
            == ""
        )

        completed = follower.wait_for_download_turn(
            tmp_path / "assembled",
            "90001",
            archive_date,
        )

        assert completed.role == "completed"
        assert completed.sync_result.blocks_copied == 2
        assert completed.sync_result.peers_reached >= 3
        assert [path.read_bytes() for path in completed.audio_files] == [
            path.read_bytes() for path in raw
        ]
    finally:
        coordinator.shutdown()
        coordinator.server_close()
        coordinator_thread.join(timeout=3)
        for server, thread in zip(seeds, seed_threads, strict=True):
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)


def test_queue_completion_never_accepts_a_conflicting_local_block(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "producer"
    day = source / "90001" / "20260712"
    day.mkdir(parents=True)
    block = day / "202607120000-123456-90001.mp3"
    block.write_bytes(b"verified producer bytes")
    monkeypatch.setenv("BROADCASTIFY_LAN_SHARING", "true")
    monkeypatch.setenv("BROADCASTIFY_LAN_DISCOVERY_ENABLED", "false")
    server = create_server(source, port=0, working_dir=tmp_path)
    server.quiet = True  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    peer_url = f"http://127.0.0.1:{server.server_port}"
    archive_date = date(2026, 7, 12)
    leader = LanArchiveSyncClient(
        enabled=True,
        peer_urls=(peer_url,),
        discovery_enabled=False,
        producer_url=peer_url,
        queue_poll_interval=0.05,
        queue_max_wait=1.0,
    )
    target = tmp_path / "consumer"
    target_day = target / "90001" / "20260712"
    target_day.mkdir(parents=True)
    (target_day / block.name).write_bytes(b"different local bytes")
    follower = LanArchiveSyncClient(
        enabled=True,
        peer_urls=(peer_url,),
        discovery_enabled=False,
        queue_poll_interval=0.05,
        queue_max_wait=0.25,
    )
    try:
        turn = leader.wait_for_download_turn(source, "90001", archive_date)
        assert turn.role == "leader"
        assert (
            leader.finish_download_turn(
                turn,
                outcome="complete",
                block_count=1,
                source_files=(block,),
            )
            == ""
        )

        result = follower.wait_for_download_turn(
            target,
            "90001",
            archive_date,
        )

        assert result.role == "deferred"
        assert result.sync_result.conflicts > 0
        assert (target_day / block.name).read_bytes() == b"different local bytes"
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


def test_native_lan_node_coordinates_one_producer_lease(
    tmp_path: Path,
) -> None:
    source = tmp_path / "native-queue"
    source.mkdir()
    server = create_lan_node_server(
        source,
        host="127.0.0.1",
        port=0,
        sync_key="native-queue-key",
        discovery_enabled=False,
    )
    server.quiet = True  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    peer_url = f"http://127.0.0.1:{server.server_port}"
    archive_date = date(2026, 7, 12)
    producer = LanArchiveSyncClient(
        enabled=True,
        peer_urls=(peer_url,),
        discovery_enabled=False,
        sync_key="native-queue-key",
        producer_url=peer_url,
        queue_poll_interval=0.05,
        queue_max_wait=1.0,
    )
    follower = LanArchiveSyncClient(
        enabled=True,
        peer_urls=(peer_url,),
        discovery_enabled=False,
        sync_key="native-queue-key",
        producer_url=peer_url,
        queue_poll_interval=0.05,
        queue_max_wait=0.15,
    )
    try:
        first = producer.wait_for_download_turn(
            source,
            "90001",
            archive_date,
        )
        second = follower.wait_for_download_turn(
            tmp_path / "native-follower",
            "90001",
            archive_date,
        )

        assert first.role == "leader"
        assert second.role == "deferred"
        assert producer.finish_download_turn(first, outcome="failed") == ""

        takeover = follower.wait_for_download_turn(
            tmp_path / "native-follower",
            "90001",
            archive_date,
        )
        assert takeover.role == "leader"
        assert follower.finish_download_turn(takeover, outcome="failed") == ""
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


def test_discovery_includes_real_interface_directed_broadcasts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        lan_sync,
        "local_ipv4_broadcasts",
        lambda: ("10.20.31.255", "192.168.50.255", "10.20.31.255"),
    )

    assert discovery_destinations() == (
        "255.255.255.255",
        "239.255.77.77",
        "10.20.31.255",
        "192.168.50.255",
    )


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
