from __future__ import annotations

import hashlib
import http.client
import json
import os
import socket
import threading
import time
from datetime import date, timedelta
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import broadcastify_cli.lan_sync as lan_sync
from broadcastify_cli.archive_cache import (
    cached_archive_for_id,
    complete_cached_archive_day,
    remember_archive_identity,
    remember_complete_archive_day,
)
from broadcastify_cli.lan_sync import (
    LanArchiveCatalog,
    LanArchiveSyncClient,
    LanAcquisitionQueue,
    LanDiscoveryResponder,
    LanDownloadTurn,
    LanProcessingQueue,
    LanSyncResult,
    discovery_destinations,
    discover_lan_peers,
    normalize_peer_url,
    normalize_peer_urls,
)
from broadcastify_cli.models import JobRequest
from broadcastify_cli.lan_node import (
    _load_environment,
    create_lan_node_server,
    validate_lan_host,
)
from broadcastify_cli.quota import (
    ArchiveQuotaCoordinatorUnavailable,
    ArchiveRequestLedger,
    RemoteArchiveRequestLedger,
)
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


def _retained_transcribed_feed_day(
    root: Path,
    feed_id: str,
    archive_date: date,
    processing_fingerprint: str,
) -> None:
    day = root / feed_id / archive_date.strftime("%Y%m%d")
    transcripts = day / "transcripts"
    transcripts.mkdir(parents=True)
    archive_id = f"{archive_date:%Y%m%d}0000"
    raw = day / f"{archive_date:%Y%m%d}0000-{archive_id}-{feed_id}.mp3"
    raw.write_bytes(f"raw {feed_id} {archive_date.isoformat()}".encode())
    remember_archive_identity(
        day,
        feed_id,
        archive_date,
        archive_id,
        raw,
        listing_prefix=f"{archive_date:%Y%m%d}0000",
    )
    assert remember_complete_archive_day(
        day,
        feed_id,
        archive_date,
        [archive_id],
    )
    audio = day / f"combined_{feed_id}_{archive_date:%Y%m%d}.mp3"
    audio.write_bytes(f"combined {feed_id} {archive_date.isoformat()}".encode())
    (day / f"{audio.stem}.manifest.json").write_text(
        json.dumps(
            {
                "timeline_version": 2,
                "feed_id": feed_id,
                "archive_date": archive_date.isoformat(),
                "combined_file": audio.name,
                "sources": [
                    {
                        "source_file": raw.name,
                        "archive_start": f"{archive_date.isoformat()}T00:00:00",
                        "combined_start_seconds": 0.0,
                        "trimmed_duration_seconds": 1800.0,
                        "duration_source": "archive_interval",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    rendered = f"[{archive_date.isoformat()} 00:00:00] Dispatch retained.\n"
    text_path = transcripts / f"{audio.stem}.txt"
    text_path.write_text(rendered, encoding="utf-8")
    (transcripts / f"{audio.stem}.json").write_text(
        json.dumps(
            {
                "audio_file": audio.name,
                "audio_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
                "processing_fingerprint": processing_fingerprint,
                "rendered_text_sha256": hashlib.sha256(
                    text_path.read_bytes()
                ).hexdigest(),
                "segments": [],
            }
        ),
        encoding="utf-8",
    )


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


def _lan_info(port: int) -> tuple[int, dict[str, object]]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    connection.request("GET", "/api/lan/v1/info")
    response = connection.getresponse()
    status = response.status
    payload = json.loads(response.read())
    connection.close()
    return status, payload


def test_remote_quota_ledger_counts_once_at_the_lan_coordinator_and_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    central_path = tmp_path / "central-quota.sqlite3"
    monkeypatch.setenv("BROADCASTIFY_QUOTA_LEDGER", str(central_path))
    server = create_lan_node_server(
        tmp_path / "coordinator-library",
        host="127.0.0.1",
        port=0,
        sync_key="shared-test-key",
        discovery_enabled=False,
    )
    server.quiet = True  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    coordinator = f"http://127.0.0.1:{server.server_port}"
    remote = RemoteArchiveRequestLedger(
        coordinator,
        local_path=tmp_path / "windows-mirror.sqlite3",
        sync_key="shared-test-key",
        timeout_seconds=2,
    )
    try:
        request_id = remote.reserve(
            feed_id="90001",
            archive_date="2026-07-12",
            archive_id="90001-1783140752",
        )
        remote.finish(request_id, outcome="http_200", http_status=200)

        shared = ArchiveRequestLedger(central_path).status()
        mirror = ArchiveRequestLedger(tmp_path / "windows-mirror.sqlite3").status()
        status = remote.status()
        assert shared["used"] == 1
        assert mirror["used"] == 1
        assert status["used"] == 1
        assert status["remaining"] == 239
        assert status["coordinated"] is True

        rejected = RemoteArchiveRequestLedger(
            coordinator,
            local_path=tmp_path / "rejected-mirror.sqlite3",
            sync_key="wrong-key",
            timeout_seconds=2,
        )
        assert rejected.status()["available"] is False
        with pytest.raises(ArchiveQuotaCoordinatorUnavailable):
            rejected.reserve(
                feed_id="90001",
                archive_date="2026-07-12",
                archive_id="654321",
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


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


def test_catalog_keeps_drifted_final_block_that_crosses_midnight(
    tmp_path: Path,
) -> None:
    archive_date = date(2026, 7, 12)
    day = tmp_path / "90001" / "20260712"
    day.mkdir(parents=True)
    final_block = day / "202607130031-777777-90001.mp3"
    final_block.write_bytes(b"last block from the requested website archive day")
    too_late = day / "202607130700-888888-90001.mp3"
    too_late.write_bytes(b"not part of the requested archive day")
    catalog = LanArchiveCatalog(tmp_path, enabled=True)

    blocks = catalog.inventory("90001", archive_date)

    assert [block.filename for block in blocks] == [final_block.name]
    resolved, block = catalog.resolve_block("90001", archive_date, final_block.name)
    assert resolved == final_block.resolve()
    assert block.filename == final_block.name
    with pytest.raises(lan_sync.LanSyncError, match="does not match"):
        catalog.resolve_block("90001", archive_date, too_late.name)

    client = LanArchiveSyncClient(enabled=True, discovery_enabled=False)
    completion = client.completion_blocks(
        [final_block], "90001", archive_date
    )
    assert [value.filename for value in completion] == [final_block.name]


def test_hash_verified_peer_sync_copies_missing_blocks_without_a_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "peer"
    source_day, raw = _raw_day(source)
    remember_archive_identity(
        source_day,
        "90001",
        date(2026, 7, 12),
        "90001-exact-provider-id",
        raw[0],
        listing_prefix="202607120001",
    )
    remember_archive_identity(
        source_day,
        "90001",
        date(2026, 7, 12),
        "90001-provider-alias",
        raw[0],
        listing_prefix="202607120101",
        allow_filename_alias=True,
    )
    assert cached_archive_for_id(
        source_day,
        "90001",
        "90001-provider-alias",
    ) is None
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
        assert cached_archive_for_id(
            target / "90001" / "20260712",
            "90001",
            "90001-exact-provider-id",
        ) == target / "90001" / "20260712" / raw[0].name
        assert cached_archive_for_id(
            target / "90001" / "20260712",
            "90001",
            "90001-provider-alias",
        ) is None

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


def test_peer_sync_promotes_a_compatible_completion_proof_superset(
    tmp_path: Path,
) -> None:
    feed_id = "90001"
    archive_date = date(2026, 7, 12)
    smaller = tmp_path / "smaller-peer"
    larger = tmp_path / "larger-peer"
    smaller_day = smaller / feed_id / "20260712"
    larger_day, larger_blocks = _raw_day(larger)
    smaller_day.mkdir(parents=True)
    smaller_block = smaller_day / larger_blocks[0].name
    smaller_block.write_bytes(larger_blocks[0].read_bytes())
    archive_ids = ["90001-1783140752", "90001-1783142552"]
    for day, blocks, ids in (
        (smaller_day, [smaller_block], archive_ids[:1]),
        (larger_day, larger_blocks, archive_ids),
    ):
        for block, archive_id in zip(blocks, ids, strict=True):
            remember_archive_identity(
                day,
                feed_id,
                archive_date,
                archive_id,
                block,
                listing_prefix=block.name[:12],
            )
        assert remember_complete_archive_day(
            day,
            feed_id,
            archive_date,
            ids,
        )

    servers = [
        create_lan_node_server(
            root,
            host="127.0.0.1",
            port=0,
            discovery_enabled=False,
        )
        for root in (smaller, larger)
    ]
    threads = [
        threading.Thread(target=server.serve_forever, daemon=True)
        for server in servers
    ]
    for server, thread in zip(servers, threads, strict=True):
        server.quiet = True  # type: ignore[attr-defined]
        thread.start()
    target = tmp_path / "target"
    try:
        result = LanArchiveSyncClient(
            enabled=True,
            peer_urls=[
                f"http://127.0.0.1:{server.server_port}"
                for server in servers
            ],
            discovery_enabled=False,
        ).sync_day(target, feed_id, archive_date)

        completion = complete_cached_archive_day(
            target / feed_id / "20260712",
            feed_id,
            archive_date,
        )
        assert result.failures == ()
        assert result.conflicts == 0
        assert result.completion_proven is True
        assert completion is not None
        assert completion[1] == 2
        assert [path.name for path in completion[0]] == [
            path.name for path in larger_blocks
        ]
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=3)


def test_peer_sync_rejects_divergent_completion_proofs(
    tmp_path: Path,
) -> None:
    feed_id = "90001"
    archive_date = date(2026, 7, 12)
    roots = [tmp_path / "peer-a", tmp_path / "peer-b"]
    archive_ids = ["90001-provider-a", "90001-provider-b"]
    for root, archive_id in zip(roots, archive_ids, strict=True):
        day, blocks = _raw_day(root)
        remember_archive_identity(
            day,
            feed_id,
            archive_date,
            archive_id,
            blocks[0],
            listing_prefix=blocks[0].name[:12],
        )
        assert remember_complete_archive_day(
            day,
            feed_id,
            archive_date,
            [archive_id],
        )

    servers = [
        create_lan_node_server(
            root,
            host="127.0.0.1",
            port=0,
            discovery_enabled=False,
        )
        for root in roots
    ]
    threads = [
        threading.Thread(target=server.serve_forever, daemon=True)
        for server in servers
    ]
    for server, thread in zip(servers, threads, strict=True):
        server.quiet = True  # type: ignore[attr-defined]
        thread.start()
    try:
        result = LanArchiveSyncClient(
            enabled=True,
            peer_urls=[
                f"http://127.0.0.1:{server.server_port}"
                for server in servers
            ],
            discovery_enabled=False,
        ).sync_day(tmp_path / "target", feed_id, archive_date)

        assert result.completion_proven is False
        assert result.conflicts == 1
        assert any("completion proof" in value for value in result.failures)
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=3)


def test_followed_feed_reconciliation_converges_month_and_few_day_nodes(
    tmp_path: Path,
) -> None:
    feed_id = "91059"
    fingerprint = "a" * 64
    month = [date(2026, 7, 1) + timedelta(days=value) for value in range(31)]
    node_a = tmp_path / "node-a"
    node_b = tmp_path / "node-b"
    for archive_date in month[:28]:
        _retained_transcribed_feed_day(
            node_a,
            feed_id,
            archive_date,
            fingerprint,
        )
    for archive_date in month[28:]:
        _retained_transcribed_feed_day(
            node_b,
            feed_id,
            archive_date,
            fingerprint,
        )

    server_a = create_lan_node_server(
        node_a,
        host="127.0.0.1",
        port=0,
        sync_key="shared-test-key",
        discovery_enabled=False,
    )
    server_b = create_lan_node_server(
        node_b,
        host="127.0.0.1",
        port=0,
        sync_key="shared-test-key",
        discovery_enabled=False,
    )
    server_a.quiet = True  # type: ignore[attr-defined]
    server_b.quiet = True  # type: ignore[attr-defined]
    thread_a = threading.Thread(target=server_a.serve_forever, daemon=True)
    thread_b = threading.Thread(target=server_b.serve_forever, daemon=True)
    thread_a.start()
    thread_b.start()
    url_a = f"http://127.0.0.1:{server_a.server_port}"
    url_b = f"http://127.0.0.1:{server_b.server_port}"
    try:
        progress_a: list[str] = []
        result_a = LanArchiveSyncClient(
            enabled=True,
            peer_urls=[url_b],
            discovery_enabled=False,
            sync_key="shared-test-key",
        ).sync_feed(
            node_a,
            feed_id,
            processing_fingerprint=fingerprint,
            progress=progress_a.append,
        )
        result_b = LanArchiveSyncClient(
            enabled=True,
            peer_urls=[url_a],
            discovery_enabled=False,
            sync_key="shared-test-key",
        ).sync_feed(
            node_b,
            feed_id,
            processing_fingerprint=fingerprint,
        )

        assert result_a.dates_discovered == tuple(
            value.isoformat() for value in reversed(month[28:])
        )
        assert result_a.blocks_copied == 3
        assert result_a.transcript_artifacts_copied == 12
        assert result_b.dates_discovered == tuple(
            value.isoformat() for value in reversed(month)
        )
        assert month[-1].isoformat() in progress_a[0]
        assert result_b.blocks_copied == 28
        assert result_b.transcript_artifacts_copied == 112
        assert result_a.failures == ()
        assert result_b.failures == ()

        for root in (node_a, node_b):
            assert len(list((root / feed_id).glob("*/transcripts/*.json"))) == 31
            for archive_date in month:
                day = root / feed_id / archive_date.strftime("%Y%m%d")
                complete = complete_cached_archive_day(
                    day,
                    feed_id,
                    archive_date,
                )
                assert complete is not None
                assert len(complete[0]) == 1
                assert (
                    day / "transcripts" / f"combined_{feed_id}_{archive_date:%Y%m%d}.json"
                ).is_file()
                assert (
                    day / f"combined_{feed_id}_{archive_date:%Y%m%d}.manifest.json"
                ).is_file()
    finally:
        server_a.shutdown()
        server_b.shutdown()
        server_a.server_close()
        server_b.server_close()
        thread_a.join(timeout=3)
        thread_b.join(timeout=3)


def test_transcript_sync_preserves_a_different_local_model_variant(
    tmp_path: Path,
) -> None:
    feed_id = "91059"
    archive_date = date(2026, 7, 12)
    local_fingerprint = "a" * 64
    peer_fingerprint = "b" * 64
    local_root = tmp_path / "local"
    peer_root = tmp_path / "peer"
    _retained_transcribed_feed_day(
        local_root,
        feed_id,
        archive_date,
        local_fingerprint,
    )
    _retained_transcribed_feed_day(
        peer_root,
        feed_id,
        archive_date,
        peer_fingerprint,
    )

    peer_day = peer_root / feed_id / archive_date.strftime("%Y%m%d")
    stem = f"combined_{feed_id}_{archive_date:%Y%m%d}"
    peer_audio = peer_day / f"{stem}.mp3"
    peer_text = peer_day / "transcripts" / f"{stem}.txt"
    peer_json = peer_day / "transcripts" / f"{stem}.json"
    peer_audio.write_bytes(b"a separately encoded peer audio result")
    peer_text.write_text("[2026-07-12 00:00:00] Peer model result.\n", encoding="utf-8")
    payload = json.loads(peer_json.read_text(encoding="utf-8"))
    payload["audio_sha256"] = hashlib.sha256(peer_audio.read_bytes()).hexdigest()
    payload["rendered_text_sha256"] = hashlib.sha256(
        peer_text.read_bytes()
    ).hexdigest()
    peer_json.write_text(json.dumps(payload), encoding="utf-8")

    local_day = local_root / feed_id / archive_date.strftime("%Y%m%d")
    conventional = {
        path.relative_to(local_day): path.read_bytes()
        for path in (
            local_day / f"{stem}.mp3",
            local_day / f"{stem}.manifest.json",
            local_day / "transcripts" / f"{stem}.json",
            local_day / "transcripts" / f"{stem}.txt",
        )
    }
    server = create_lan_node_server(
        peer_root,
        host="127.0.0.1",
        port=0,
        discovery_enabled=False,
    )
    server.quiet = True  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    peer_url = f"http://127.0.0.1:{server.server_port}"
    try:
        client = LanArchiveSyncClient(
            enabled=True,
            peer_urls=(peer_url,),
            discovery_enabled=False,
        )
        first = client.sync_transcripts(
            local_root,
            feed_id,
            archive_date,
            peer_fingerprint,
        )
        audio_hash = hashlib.sha256(peer_audio.read_bytes()).hexdigest()
        variant = (
            local_day
            / lan_sync.DERIVED_VARIANTS_DIRECTORY
            / peer_fingerprint
            / audio_hash
        )
        variant_json = variant / "transcripts" / f"{stem}.json"

        assert first.artifacts_copied == 4
        assert first.conflicts == 0
        assert first.failures == ()
        assert first.transcripts == (variant_json,)
        assert variant_json.read_bytes() == peer_json.read_bytes()
        assert (variant / f"{stem}.mp3").read_bytes() == peer_audio.read_bytes()
        for relative, content in conventional.items():
            assert (local_day / relative).read_bytes() == content

        catalog = LanArchiveCatalog(
            local_root,
            enabled=True,
            queue_enabled=False,
        )
        inventory = catalog.transcript_inventory(
            feed_id,
            archive_date,
            peer_fingerprint,
        )
        assert len(inventory) == 4
        resolved, artifact = catalog.resolve_transcript_artifact(
            feed_id,
            archive_date,
            peer_fingerprint,
            f"{stem}.json",
        )
        assert resolved == variant_json
        assert artifact.audio_sha256 == audio_hash

        second = client.sync_transcripts(
            local_root,
            feed_id,
            archive_date,
            peer_fingerprint,
        )
        assert second.artifacts_copied == 0
        assert second.artifacts_already_local == 4
        assert second.conflicts == 0
        assert second.transcripts == (variant_json,)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)

    # A node that retained the variant can serve it onward without promoting
    # or overwriting the different conventional model result.
    variant_server = create_lan_node_server(
        local_root,
        host="127.0.0.1",
        port=0,
        discovery_enabled=False,
    )
    variant_server.quiet = True  # type: ignore[attr-defined]
    variant_thread = threading.Thread(
        target=variant_server.serve_forever,
        daemon=True,
    )
    variant_thread.start()
    try:
        consumer = tmp_path / "consumer"
        forwarded = LanArchiveSyncClient(
            enabled=True,
            peer_urls=(f"http://127.0.0.1:{variant_server.server_port}",),
            discovery_enabled=False,
        ).sync_transcripts(
            consumer,
            feed_id,
            archive_date,
            peer_fingerprint,
        )
        assert forwarded.artifacts_copied == 4
        assert forwarded.conflicts == 0
        assert (
            consumer
            / feed_id
            / archive_date.strftime("%Y%m%d")
            / "transcripts"
            / f"{stem}.json"
        ).read_bytes() == peer_json.read_bytes()
    finally:
        variant_server.shutdown()
        variant_server.server_close()
        variant_thread.join(timeout=3)


def test_transcript_fingerprint_catalog_rejects_partial_artifacts(
    tmp_path: Path,
) -> None:
    feed_id = "91059"
    archive_date = date(2026, 7, 12)
    fingerprint = "c" * 64
    transcript_dir = tmp_path / feed_id / "20260712" / "transcripts"
    transcript_dir.mkdir(parents=True)
    (transcript_dir / f"combined_{feed_id}_20260712.json").write_text(
        json.dumps(
            {
                "audio_file": f"combined_{feed_id}_20260712.mp3",
                "audio_sha256": "d" * 64,
                "processing_fingerprint": fingerprint,
                "rendered_text_sha256": "e" * 64,
            }
        ),
        encoding="utf-8",
    )

    catalog = LanArchiveCatalog(tmp_path, enabled=True, queue_enabled=False)

    assert catalog.transcript_fingerprints(feed_id, archive_date) == ()


def test_feed_sync_keeps_pre_fingerprint_discovery_peers_compatible(
    tmp_path: Path,
) -> None:
    feed_id = "90001"
    archive_date = date(2026, 7, 12)

    class OldPeerHandler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args: object) -> None:
            return

        def _json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.startswith("/api/lan/v1/feed-days?"):
                self._json(
                    HTTPStatus.OK,
                    {
                        "protocol": lan_sync.LAN_PROTOCOL,
                        "feed_id": feed_id,
                        "dates": [archive_date.isoformat()],
                        "peers": [],
                    },
                )
                return
            if self.path.startswith("/api/lan/v1/blocks?"):
                self._json(
                    HTTPStatus.OK,
                    {
                        "protocol": lan_sync.LAN_PROTOCOL,
                        "feed_id": feed_id,
                        "archive_date": archive_date.isoformat(),
                        "blocks": [],
                        "complete": False,
                        "completion_blocks": [],
                        "peers": [],
                    },
                )
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "not available"})

    server = ThreadingHTTPServer(("127.0.0.1", 0), OldPeerHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = LanArchiveSyncClient(
            enabled=True,
            peer_urls=(f"http://127.0.0.1:{server.server_port}",),
            discovery_enabled=False,
        ).sync_feed(tmp_path / "consumer", feed_id)

        assert result.days_considered == 1
        assert result.failures == ()
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
    assert queue.active_activity() == {
        "quota_scope": "default",
        "feed_id": "90001",
        "archive_date": "2026-07-12",
        "owner_node_id": "producer_one",
        "producer_url": "http://10.20.30.40:8766",
        "lease_seconds": 15.0,
    }

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


def test_acquisition_queue_allows_only_one_global_website_stream() -> None:
    queue = LanAcquisitionQueue(lease_seconds=30.0)
    first_day = date(2026, 7, 12)
    second_day = date(2026, 7, 13)

    first = queue.claim(
        "primary-account",
        "90001",
        first_day,
        owner_node_id="producer_one",
        producer_url="http://10.20.30.40:8766",
        requester_address="10.20.30.40",
    )
    second = queue.claim(
        "secondary-account",
        "20305",
        second_day,
        owner_node_id="producer_two",
        producer_url="http://10.20.30.41:8766",
        requester_address="10.20.30.41",
    )

    assert first["granted"] is True
    assert second["granted"] is False
    assert second["state"] == "available"
    assert second["global_busy"] is True

    queue.finish(
        "primary-account",
        "90001",
        first_day,
        lease_token=str(first["lease_token"]),
        outcome="failed",
    )
    retry = queue.claim(
        "secondary-account",
        "20305",
        second_day,
        owner_node_id="producer_two",
        producer_url="http://10.20.30.41:8766",
        requester_address="10.20.30.41",
    )
    assert retry["granted"] is True


def test_quota_limited_account_does_not_block_a_different_account_scope() -> None:
    queue = LanAcquisitionQueue(lease_seconds=30.0)
    archive_date = date(2026, 7, 12)
    primary = queue.claim(
        "pool.default",
        "91059",
        archive_date,
        owner_node_id="producer_one",
        producer_url="http://10.20.30.40:8766",
        requester_address="10.20.30.40",
    )
    queue.finish(
        "pool.default",
        "91059",
        archive_date,
        lease_token=str(primary["lease_token"]),
        outcome="quota_limited",
        retry_after_seconds=60.0,
    )

    secondary = queue.claim(
        "pool.secondary",
        "91059",
        archive_date,
        owner_node_id="producer_two",
        producer_url="http://10.20.30.41:8766",
        requester_address="10.20.30.41",
    )

    assert secondary["granted"] is True


def test_settings_scope_queue_results_per_account_and_use_one_coordinator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BROADCASTIFY_LAN_QUOTA_SCOPE", "authorized-pool")
    monkeypatch.setenv(
        "BROADCASTIFY_LAN_QUOTA_COORDINATOR",
        "http://10.200.1.227:8765",
    )
    monkeypatch.setenv("BROADCASTIFY_ACCOUNT_PROFILE", "default")
    primary = LanArchiveSyncClient.from_settings(
        enabled=True,
        peer_urls=(),
        discovery_enabled=False,
    )
    monkeypatch.setenv("BROADCASTIFY_ACCOUNT_PROFILE", "secondary")
    secondary = LanArchiveSyncClient.from_settings(
        enabled=True,
        peer_urls=(),
        discovery_enabled=False,
    )

    assert primary.quota_scope == "authorized-pool.default"
    assert secondary.quota_scope == "authorized-pool.secondary"
    assert primary._select_coordinator() == (
        "http://10.200.1.227:8765",
        {
            "node_id": "configured-coordinator",
            "acquisition_queue_available": True,
        },
    )
    assert secondary._select_coordinator("processing_queue_available") == (
        "http://10.200.1.227:8765",
        {
            "node_id": "configured-coordinator",
            "processing_queue_available": True,
        },
    )


def test_processing_queue_releases_failure_and_expired_leases() -> None:
    now = [500.0]
    queue = LanProcessingQueue(
        lease_seconds=30.0,
        result_seconds=300.0,
        clock=lambda: now[0],
    )
    archive_date = date(2026, 7, 12)
    fingerprint = "b" * 64

    first = queue.claim(
        fingerprint,
        "91059",
        archive_date,
        owner_node_id="producer_one",
        producer_url="http://10.20.30.40:8766",
        requester_address="10.20.30.40",
    )
    queue.finish(
        fingerprint,
        "91059",
        archive_date,
        lease_token=str(first["lease_token"]),
        outcome="failed",
    )
    second = queue.claim(
        fingerprint,
        "91059",
        archive_date,
        owner_node_id="producer_two",
        producer_url="http://10.20.30.41:8766",
        requester_address="10.20.30.41",
    )
    assert second["granted"] is True
    assert queue.active_activities() == [
        {
            "feed_id": "91059",
            "archive_date": "2026-07-12",
            "owner_node_id": "producer_two",
            "producer_url": "http://10.20.30.41:8766",
            "lease_seconds": 30.0,
        }
    ]

    now[0] += 31.0
    third = queue.claim(
        fingerprint,
        "91059",
        archive_date,
        owner_node_id="producer_three",
        producer_url="http://10.20.30.42:8766",
        requester_address="10.20.30.42",
    )
    assert third["granted"] is True
    assert third["owner_node_id"] == "producer_three"


def test_coordinated_status_validation_rejects_unsafe_peer_values() -> None:
    scheduler = LanArchiveSyncClient._validate_scheduler_surface(
        {
            "active": {
                "feed_id": "91059",
                "feed_name": "  Example   Feed  ",
                "phase": "processing",
                "status": "running",
                "account_profile_id": "secondary",
                "stage": "transcribe",
                "archive_date": "2026-07-12",
                "current": "not-an-integer",
                "total": object(),
            },
            "schedules": [
                {
                    "id": "also-invalid",
                    "feed_id": "91059",
                    "feed_name": "  Example   Feed  ",
                    "state": "running",
                    "enabled": True,
                    "account_profile_id": "automatic",
                    "message": "  retained   work  ",
                },
                {"id": 2, "feed_id": "../../secrets"},
            ],
        }
    )

    assert scheduler["active"]["current"] == 0
    assert scheduler["active"]["total"] == 0
    assert scheduler["active"]["feed_name"] == "Example Feed"
    assert scheduler["schedules"] == [
        {
            "id": 0,
            "feed_id": "91059",
            "feed_name": "Example Feed",
            "state": "running",
            "enabled": True,
            "account_profile_id": "automatic",
            "next_run_at": "",
            "last_started_at": "",
            "message": "retained work",
        }
    ]

    reconciliation = LanArchiveSyncClient._validate_reconciliation_surface(
        {
            "enabled": True,
            "running": True,
            "active_feed_id": "../../secrets",
            "last_started_at": " 2026-08-15T04:00:00-05:00 ",
            "feeds_considered": -3,
            "days_considered": "89",
            "blocks_copied": object(),
            "transcript_artifacts_copied": "8",
            "bytes_copied": "190000000",
            "failures": ["  peer   unavailable  ", object()],
        }
    )
    assert reconciliation == {
        "enabled": True,
        "running": True,
        "active_feed_id": "",
        "last_started_at": "2026-08-15T04:00:00-05:00",
        "last_finished_at": "",
        "feeds_considered": 0,
        "days_considered": 89,
        "blocks_copied": 0,
        "transcript_artifacts_copied": 8,
        "bytes_copied": 190000000,
        "failures": ["peer unavailable"],
    }


def test_lan_clients_claim_different_model_days_without_duplicate_work(
    tmp_path: Path,
) -> None:
    root = tmp_path / "coordinator"
    root.mkdir()
    server = create_lan_node_server(
        root,
        host="127.0.0.1",
        port=0,
        discovery_enabled=False,
    )
    server.quiet = True  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    peer_url = f"http://127.0.0.1:{server.server_port}"
    first = LanArchiveSyncClient(
        enabled=True,
        peer_urls=(peer_url,),
        discovery_enabled=False,
        producer_url=peer_url,
    )
    second = LanArchiveSyncClient(
        enabled=True,
        peer_urls=(peer_url,),
        discovery_enabled=False,
        producer_url=peer_url,
    )
    fingerprint = "c" * 64
    first_day = date(2026, 7, 12)
    second_day = date(2026, 7, 13)
    try:
        first_turn = first.claim_processing_turn(
            "91059",
            first_day,
            fingerprint,
        )
        duplicate_turn = second.claim_processing_turn(
            "91059",
            first_day,
            fingerprint,
        )
        parallel_turn = second.claim_processing_turn(
            "91059",
            second_day,
            fingerprint,
        )

        assert first_turn.role == "leader"
        assert duplicate_turn.role == "deferred"
        assert parallel_turn.role == "leader"

        assert first.finish_processing_turn(
            first_turn,
            outcome="complete",
            artifact_count=4,
        ) == ""
        completed_turn = second.claim_processing_turn(
            "91059",
            first_day,
            fingerprint,
        )
        assert completed_turn.role == "completed"
        assert completed_turn.artifact_count == 4
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_truenas_web_node_coordinates_model_work_for_windows_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "truenas"
    root.mkdir()
    monkeypatch.setenv("BROADCASTIFY_LAN_SHARING", "true")
    monkeypatch.setenv("BROADCASTIFY_LAN_QUEUE_ENABLED", "true")
    monkeypatch.setenv("BROADCASTIFY_LAN_DISCOVERY_ENABLED", "false")
    server = create_server(root, port=0, working_dir=tmp_path)
    server.quiet = True  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    coordinator_url = f"http://127.0.0.1:{server.server_port}"
    windows = LanArchiveSyncClient(
        enabled=True,
        peer_urls=(coordinator_url,),
        discovery_enabled=False,
        coordinator_url=coordinator_url,
        producer_url=coordinator_url,
    )
    peer = LanArchiveSyncClient(
        enabled=True,
        peer_urls=(coordinator_url,),
        discovery_enabled=False,
        coordinator_url=coordinator_url,
        producer_url=coordinator_url,
    )
    fingerprint = "f" * 64
    archive_date = date(2026, 7, 14)
    try:
        leader = windows.claim_processing_turn(
            "91059",
            archive_date,
            fingerprint,
        )
        duplicate = peer.claim_processing_turn(
            "91059",
            archive_date,
            fingerprint,
        )

        assert leader.role == "leader"
        assert duplicate.role == "deferred"
        assert windows.finish_processing_turn(
            leader,
            outcome="complete",
            artifact_count=4,
        ) == ""
        completed = peer.claim_processing_turn(
            "91059",
            archive_date,
            fingerprint,
        )
        assert completed.role == "completed"
        assert completed.artifact_count == 4
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_multihomed_self_claim_requires_an_explicit_private_exception() -> None:
    queue = LanAcquisitionQueue()
    archive_date = date(2026, 7, 12)

    with pytest.raises(PermissionError, match="own reachable producer"):
        queue.claim(
            "default",
            "90001",
            archive_date,
            owner_node_id="producer_one",
            producer_url="http://10.200.1.227:8765",
            requester_address="10.200.1.99",
        )

    claim = queue.claim(
        "default",
        "90001",
        archive_date,
        owner_node_id="producer_one",
        producer_url="http://10.200.1.227:8765",
        requester_address="10.200.1.99",
        allow_multihomed_self=True,
    )
    assert claim["granted"] is True


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


def test_recent_completed_day_is_a_short_lived_rolling_snapshot() -> None:
    now = [300.0]
    today = date(2026, 7, 20)
    queue = LanAcquisitionQueue(
        lease_seconds=15.0,
        result_seconds=3600.0,
        rolling_result_seconds=300.0,
        clock=lambda: now[0],
        today=lambda: today,
    )

    current_claim = queue.claim(
        "premium-account",
        "90001",
        today,
        owner_node_id="producer_one",
        producer_url="http://10.20.30.40:8766",
        requester_address="10.20.30.40",
    )
    current = queue.finish(
        "premium-account",
        "90001",
        today,
        lease_token=str(current_claim["lease_token"]),
        outcome="complete",
    )

    assert current["state"] == "complete"
    assert current["rolling"] is True
    assert current["lease_seconds"] == 300.0

    now[0] += 301.0
    assert queue.status("premium-account", "90001", today)["state"] == "available"

    old_date = date(2026, 7, 18)
    old_claim = queue.claim(
        "premium-account",
        "90001",
        old_date,
        owner_node_id="producer_one",
        producer_url="http://10.20.30.40:8766",
        requester_address="10.20.30.40",
    )
    old = queue.finish(
        "premium-account",
        "90001",
        old_date,
        lease_token=str(old_claim["lease_token"]),
        outcome="complete",
    )

    assert old["rolling"] is False
    assert old["lease_seconds"] == 3600.0

    now[0] += 301.0
    assert queue.status("premium-account", "90001", old_date)["state"] == "complete"


def test_quota_limit_uses_the_producer_next_rolling_slot() -> None:
    now = [400.0]
    today = date(2026, 7, 20)
    queue = LanAcquisitionQueue(
        result_seconds=3600.0,
        rolling_result_seconds=300.0,
        clock=lambda: now[0],
        today=lambda: today,
    )
    claim = queue.claim(
        "premium-account",
        "90001",
        today,
        owner_node_id="producer_one",
        producer_url="http://10.20.30.40:8766",
        requester_address="10.20.30.40",
    )

    limited = queue.finish(
        "premium-account",
        "90001",
        today,
        lease_token=str(claim["lease_token"]),
        outcome="quota_limited",
        block_count=2,
        retry_after_seconds=45.0,
    )

    assert limited["rolling"] is False
    assert limited["lease_seconds"] == 45.0
    now[0] += 46.0
    assert queue.status("premium-account", "90001", today)["state"] == "available"


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


def test_job_mode_defers_an_active_download_without_sleeping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "producer"
    source.mkdir()
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
    )
    follower = LanArchiveSyncClient(
        enabled=True,
        peer_urls=(peer_url,),
        discovery_enabled=False,
        sleep=lambda _seconds: pytest.fail(
            "deferred job mode must continue without queue polling sleeps"
        ),
    )
    try:
        leader_turn = leader.wait_for_download_turn(
            source,
            "90001",
            archive_date,
        )
        assert leader_turn.role == "leader"

        deferred = follower.wait_for_download_turn(
            tmp_path / "consumer",
            "90001",
            archive_date,
            defer_active=True,
        )

        assert deferred.role == "deferred"
        assert deferred.owner_node_id == leader_turn.owner_node_id
        assert leader.finish_download_turn(leader_turn, outcome="failed") == ""
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


def test_web_peer_advertises_complete_transcript_fingerprints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feed_id = "91059"
    archive_date = date(2026, 7, 12)
    fingerprint = "a" * 64
    _retained_transcribed_feed_day(
        tmp_path,
        feed_id,
        archive_date,
        fingerprint,
    )
    monkeypatch.setenv("BROADCASTIFY_LAN_SHARING", "true")
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
            "/api/lan/v1/transcript-fingerprints"
            f"?feed_id={feed_id}&date={archive_date.isoformat()}",
        )
        response = connection.getresponse()
        payload = json.loads(response.read())

        assert response.status == 200
        assert payload["processing_fingerprints"] == [fingerprint]
        assert "output_dir" not in payload
        assert "sync_key" not in payload
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_web_queue_allows_only_its_exact_multihomed_advertised_self(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    advertised_url = "http://10.200.1.227:8765"
    monkeypatch.setenv("BROADCASTIFY_LAN_SHARING", "true")
    monkeypatch.setenv("BROADCASTIFY_LAN_DISCOVERY_ENABLED", "false")
    monkeypatch.setenv("BROADCASTIFY_LAN_ADVERTISE_URL", advertised_url)
    server = create_server(tmp_path, port=0, working_dir=tmp_path)
    server.quiet = True  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection(
        "127.0.0.1",
        server.server_port,
        timeout=5,
    )

    def claim(archive_date: str, producer_url: str) -> tuple[int, dict[str, object]]:
        body = json.dumps(
            {
                "quota_scope": "multihomed-test",
                "feed_id": "90001",
                "archive_date": archive_date,
                "owner_node_id": server.state.lan_catalog.node_id,  # type: ignore[attr-defined]
                "producer_url": producer_url,
            }
        )
        connection.request(
            "POST",
            "/api/lan/v1/acquisition/claim",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        return response.status, json.loads(response.read())

    try:
        status, payload = claim("2026-07-12", advertised_url)
        assert status == 200
        assert payload["granted"] is True
        assert payload["producer_url"] == advertised_url

        rejected_status, rejected = claim(
            "2026-07-13",
            "http://10.200.1.228:8765",
        )
        assert rejected_status == 403
        assert "own reachable producer" in str(rejected["error"])
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


def test_native_lan_node_loads_private_account_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / ".env").write_text(
        "BROADCASTIFY_LAN_PEERS=http://10.0.0.10:8765\n",
        encoding="utf-8",
    )
    (tmp_path / ".env.accounts").write_text(
        "BROADCASTIFY_LAN_PEERS=http://10.0.0.20:8765\n"
        "BROADCASTIFY_LAN_QUOTA_COORDINATOR=http://10.0.0.20:8765\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BROADCASTIFY_ENV_FILE", raising=False)
    monkeypatch.delenv("BROADCASTIFY_LAN_PEERS", raising=False)
    monkeypatch.delenv("BROADCASTIFY_LAN_QUOTA_COORDINATOR", raising=False)

    try:
        _load_environment()

        assert os.environ["BROADCASTIFY_LAN_PEERS"] == "http://10.0.0.20:8765"
        assert (
            os.environ["BROADCASTIFY_LAN_QUOTA_COORDINATOR"]
            == "http://10.0.0.20:8765"
        )
    finally:
        # python-dotenv mutates os.environ directly, outside MonkeyPatch's
        # assignment tracking. Remove those values before another offline test
        # can mistake this fixture coordinator for real deployment state.
        os.environ.pop("BROADCASTIFY_LAN_PEERS", None)
        os.environ.pop("BROADCASTIFY_LAN_QUOTA_COORDINATOR", None)


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


def test_native_lan_node_continuously_reconciles_peer_artifacts(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    feed_id = "90001"
    archive_date = date(2026, 7, 12)
    fingerprint = "a" * 64
    _retained_transcribed_feed_day(
        source_root,
        feed_id,
        archive_date,
        fingerprint,
    )
    (target_root / feed_id).mkdir(parents=True)
    source = create_lan_node_server(
        source_root,
        host="127.0.0.1",
        port=0,
        discovery_enabled=False,
    )
    source.quiet = True  # type: ignore[attr-defined]
    source_thread = threading.Thread(target=source.serve_forever, daemon=True)
    source_thread.start()
    target = create_lan_node_server(
        target_root,
        host="127.0.0.1",
        port=0,
        peer_urls=[f"http://127.0.0.1:{source.server_port}"],
        discovery_enabled=False,
        background_sync_enabled=True,
    )
    target.quiet = True  # type: ignore[attr-defined]
    target_thread = threading.Thread(target=target.serve_forever, daemon=True)
    target_thread.start()
    copied_transcript = (
        target_root
        / feed_id
        / archive_date.strftime("%Y%m%d")
        / "transcripts"
        / f"combined_{feed_id}_{archive_date:%Y%m%d}.json"
    )
    try:
        deadline = time.monotonic() + 10
        while not copied_transcript.is_file() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert copied_transcript.is_file()

        deadline = time.monotonic() + 10
        status = 0
        payload: dict[str, object] = {}
        while time.monotonic() < deadline:
            status, payload = _lan_info(target.server_port)
            if payload["reconciliation"]["last_finished_at"]:  # type: ignore[index]
                break
            time.sleep(0.05)
        assert status == HTTPStatus.OK
        assert payload["reconciliation"]["enabled"] is True
        assert payload["reconciliation"]["last_finished_at"]
        assert payload["reconciliation"]["blocks_copied"] == 1
        assert (
            payload["reconciliation"]["transcript_artifacts_copied"]
            == 4
        )
        assert payload["reconciliation"]["failures"] == []
    finally:
        target.shutdown()
        target.server_close()
        target_thread.join(timeout=3)
        source.shutdown()
        source.server_close()
        source_thread.join(timeout=3)


def test_web_host_continuously_reconciles_peer_artifacts(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    working_dir = tmp_path / "work"
    working_dir.mkdir()
    feed_id = "90001"
    archive_date = date(2026, 7, 13)
    _retained_transcribed_feed_day(
        source_root,
        feed_id,
        archive_date,
        "b" * 64,
    )
    (target_root / feed_id).mkdir(parents=True)
    source = create_lan_node_server(
        source_root,
        host="127.0.0.1",
        port=0,
        discovery_enabled=False,
    )
    source.quiet = True  # type: ignore[attr-defined]
    source_thread = threading.Thread(target=source.serve_forever, daemon=True)
    source_thread.start()
    (working_dir / ".env").write_text(
        "BROADCASTIFY_LAN_SHARING=true\n"
        f"BROADCASTIFY_LAN_PEERS=http://127.0.0.1:{source.server_port}\n"
        "BROADCASTIFY_LAN_DISCOVERY_ENABLED=false\n",
        encoding="utf-8",
    )
    target = create_server(
        target_root,
        host="127.0.0.1",
        port=0,
        working_dir=working_dir,
        background_sync_enabled=True,
    )
    target.quiet = True  # type: ignore[attr-defined]
    target_thread = threading.Thread(target=target.serve_forever, daemon=True)
    target_thread.start()
    copied_transcript = (
        target_root
        / feed_id
        / archive_date.strftime("%Y%m%d")
        / "transcripts"
        / f"combined_{feed_id}_{archive_date:%Y%m%d}.json"
    )
    try:
        deadline = time.monotonic() + 10
        while not copied_transcript.is_file() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert copied_transcript.is_file()

        deadline = time.monotonic() + 10
        status = 0
        payload: dict[str, object] = {}
        while time.monotonic() < deadline:
            status, payload = _lan_info(target.server_port)
            if payload["reconciliation"]["last_finished_at"]:  # type: ignore[index]
                break
            time.sleep(0.05)
        assert status == HTTPStatus.OK
        assert payload["reconciliation"]["enabled"] is True
        assert payload["reconciliation"]["last_finished_at"]
        assert payload["reconciliation"]["blocks_copied"] == 1
        assert (
            payload["reconciliation"]["transcript_artifacts_copied"]
            == 4
        )
        assert payload["reconciliation"]["failures"] == []
    finally:
        target.shutdown()
        target.server_close()
        target_thread.join(timeout=3)
        source.shutdown()
        source.server_close()
        source_thread.join(timeout=3)
