from __future__ import annotations

import http.client
import json
import re
import threading
from datetime import date
from pathlib import Path

import pytest

from broadcastify_cli.storage import AnalysisStore
from broadcastify_cli.web_app import JobManager, WebRequestError, create_server


def _request(
    connection: http.client.HTTPConnection,
    method: str,
    path: str,
    *,
    cookie: str = "",
    token: str = "",
    body: dict[str, object] | None = None,
) -> tuple[http.client.HTTPResponse, bytes]:
    headers: dict[str, str] = {}
    payload = None
    if cookie:
        headers["Cookie"] = cookie
    if token:
        headers["X-Radio-Archive-Token"] = token
    if body is not None:
        headers["Content-Type"] = "application/json"
        payload = json.dumps(body)
    connection.request(method, path, body=payload, headers=headers)
    response = connection.getresponse()
    return response, response.read()


def _retained_day(root: Path, database: Path) -> None:
    day = root / "90001" / "20260712"
    transcript = day / "transcripts" / "combined_90001_20260712.json"
    audio = day / "combined_90001_20260712.mp3"
    transcript.parent.mkdir(parents=True)
    audio.write_bytes(b"0123456789")
    transcript.write_text(
        json.dumps(
            {
                "duration": 12.0,
                "diarization_completed": True,
                "segments": [
                    {
                        "start": 1.0,
                        "end": 2.5,
                        "speaker": "SPEAKER_00",
                        "text": "Unit responding to the retained call.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with AnalysisStore(database) as store:
        imported = store.import_transcript("90001", date(2026, 7, 12), transcript, audio)
        store.save_daily_summary(
            imported.day_id,
            "One retained dispatch call.",
            [],
            model="test",
            prompt_version="test",
            transcript_sha256=imported.transcript_sha256,
        )
        store.save_feed_catalog([{"feed_id": "90001", "name": "Example City Public Safety"}])


def test_loopback_web_app_serves_library_transcript_and_media(tmp_path: Path) -> None:
    output = tmp_path / "archives"
    database = output / "broadcastify-analysis.sqlite3"
    _retained_day(output, database)
    server = create_server(output, database, port=0, working_dir=Path.cwd())
    server.quiet = True  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        response, body = _request(connection, "GET", "/")
        assert response.status == 200
        cookie = response.getheader("Set-Cookie", "").split(";", 1)[0]
        token_match = re.search(rb'<meta name="app-token" content="([^"]+)">', body)
        assert cookie.startswith("radio_archive_session=")
        assert token_match is not None
        token = token_match.group(1).decode()

        response, body = _request(connection, "GET", "/api/bootstrap", cookie=cookie)
        bootstrap = json.loads(body)
        assert response.status == 200
        assert bootstrap["summary"]["day_count"] == 1
        assert bootstrap["days"][0]["feed_name"] == "Example City Public Safety"
        assert bootstrap["runtime"]["loopback_only"] is True

        response, body = _request(
            connection,
            "GET",
            "/api/day?feed_id=90001&date=2026-07-12",
            cookie=cookie,
        )
        detail = json.loads(body)
        assert response.status == 200
        assert detail["summary"] == "One retained dispatch call."
        assert detail["audio_url"].startswith("/media?path=")

        response, body = _request(
            connection,
            "GET",
            "/api/transcript?feed_id=90001&date=2026-07-12",
            cookie=cookie,
        )
        transcript = json.loads(body)
        assert response.status == 200
        assert transcript["total"] == 1
        assert transcript["segments"][0]["speaker"] == "SPEAKER_00"

        connection.request(
            "GET",
            detail["audio_url"],
            headers={"Cookie": cookie, "Range": "bytes=2-5"},
        )
        media = connection.getresponse()
        assert media.status == 206
        assert media.read() == b"2345"

        response, body = _request(
            connection,
            "POST",
            "/api/jobs",
            cookie=cookie,
            body={"command": "diagnostics", "payload": {}},
        )
        assert response.status == 403
        assert "token" in json.loads(body)["error"].lower()

        response, body = _request(
            connection,
            "POST",
            "/api/jobs",
            cookie=cookie,
            token=token,
            body={"command": "not-supported", "payload": {}},
        )
        assert response.status == 400
        assert "not supported" in json.loads(body)["error"].lower()
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_web_app_rejects_non_loopback_binding(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="loopback"):
        create_server(tmp_path, host="0.0.0.0", port=0)


def test_web_jobs_force_quota_safe_archive_defaults(tmp_path: Path) -> None:
    manager = JobManager(tmp_path, tmp_path / "analysis.sqlite3", tmp_path)
    arguments, payload = manager._worker_request(  # noqa: SLF001 - validates the security boundary
        "run",
        {
            "feed_id": "90001",
            "start_date": "2026-07-12",
            "end_date": "2026-07-12",
            "download_jobs": 12,
            "keep_originals": False,
            "combine": False,
            "transcribe": False,
            "diarize": True,
        },
    )

    assert arguments == ["run"]
    assert payload is not None
    assert payload["download_jobs"] == 1
    assert payload["keep_originals"] is True
    assert payload["combine"] is True
    assert payload["transcribe"] is True
    assert payload["output_dir"] == str(tmp_path)


def test_web_area_jobs_force_the_same_quota_boundary(tmp_path: Path) -> None:
    manager = JobManager(tmp_path, tmp_path / "analysis.sqlite3", tmp_path)
    arguments, payload = manager._worker_request(  # noqa: SLF001
        "run-area",
        {
            "profile_name": "Regional desk",
            "job": {
                "start_date": "2026-07-12",
                "end_date": "2026-07-12",
                "download_jobs": 8,
                "keep_originals": False,
                "diarize": True,
            },
        },
    )

    assert arguments == ["run-area"]
    assert payload is not None
    assert payload["job"]["download_jobs"] == 1
    assert payload["job"]["keep_originals"] is True
    assert payload["job"]["combine"] is True
    assert payload["job"]["transcribe"] is True
    assert payload["job"]["output_dir"] == str(tmp_path)


def test_web_jobs_validate_area_zip_codes(tmp_path: Path) -> None:
    manager = JobManager(tmp_path, tmp_path / "analysis.sqlite3", tmp_path)
    with pytest.raises(WebRequestError, match="five-digit"):
        manager._worker_request("area-search", {"zip_codes": ["not-a-zip"]})  # noqa: SLF001


def test_web_jobs_validate_and_forward_radius_discovery(tmp_path: Path) -> None:
    manager = JobManager(tmp_path, tmp_path / "analysis.sqlite3", tmp_path)
    arguments, payload = manager._worker_request(  # noqa: SLF001
        "area-search",
        {"center_zip": "12345", "radius_miles": 25, "max_zip_codes": 12},
    )

    assert arguments == [
        "area-search",
        "--center-zip",
        "12345",
        "--radius-miles",
        "25.0",
        "--max-zip-codes",
        "12",
    ]
    assert payload is None


def test_web_jobs_forward_explicit_asr_self_test_settings(tmp_path: Path) -> None:
    manager = JobManager(tmp_path, tmp_path / "analysis.sqlite3", tmp_path)
    arguments, payload = manager._worker_request(  # noqa: SLF001
        "asr-self-test",
        {
            "model": "tiny.en",
            "asr_engine": "openvino",
            "device": "openvino-npu",
            "huggingface_token": "session-only-test-token",
        },
    )

    assert arguments == ["asr-self-test"]
    assert payload is not None
    assert payload["model"] == "tiny.en"
    assert payload["device"] == "openvino-npu"
    assert payload["huggingface_token"] == "session-only-test-token"
