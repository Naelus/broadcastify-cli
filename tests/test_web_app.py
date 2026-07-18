from __future__ import annotations

import http.client
import json
import re
import threading
from datetime import date
from pathlib import Path

import pytest

from broadcastify_cli.analysis import PROMPT_VERSION
from broadcastify_cli.storage import AnalysisStore
from broadcastify_cli.web_app import (
    JobManager,
    WebRequestError,
    _area_stories_for_web,
    create_server,
)


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


def _retained_day(
    root: Path,
    database: Path,
    *,
    prompt_version: str = PROMPT_VERSION,
) -> None:
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
        incident_ids = store.replace_incidents(
            imported.day_id,
            [
                {
                    "fingerprint": "web-redaction-boundary",
                    "event_type": "welfare_check",
                    "title": "Welfare check requested",
                    "summary": "Radio traffic requested a welfare check.",
                    "location": "Retained Place",
                    "start_seconds": 1.0,
                    "end_seconds": 2.5,
                    "priority": 2,
                    "confidence": 0.9,
                    "evidence": [
                        {
                            "segment_index": 0,
                            "start_seconds": 1.0,
                            "end_seconds": 2.5,
                            "speaker": "SPEAKER_00",
                            "text": "Check the welfare of Jordan Example at the retained location.",
                        }
                    ],
                }
            ],
            model="test",
            prompt_version=prompt_version,
        )
        store.save_daily_summary(
            imported.day_id,
            "One retained dispatch call.",
            incident_ids,
            model="test",
            prompt_version=prompt_version,
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
        assert b'id="areaPublicSafetyOnly"' in body
        assert b'/static/app.js?v=24' in body
        assert b'value="qwen3-asr"' in body
        assert b'qwen3-asr-0.6b-int8' in body
        assert b'id="settingAnalysisDevice"' in body
        assert b'id="analysisSelfTestButton"' in body
        assert b'id="profileSelfTestButton"' in body
        assert b'id="profileNextActionButton"' in body
        assert b'id="setupProfileSelfTestButton"' in body
        assert b'id="profileSelfTestNotice"' in body
        assert b'id="settingsSectionTabs"' in body
        assert b'data-settings-panel="processing"' in body
        assert b'role="tabpanel"' in body
        assert b'/static/favicon.svg' in body
        token = token_match.group(1).decode()

        response, body = _request(connection, "GET", "/static/favicon.svg")
        assert response.status == 200
        assert response.getheader("Content-Type") == "image/svg+xml"
        assert b"<svg" in body

        response, body = _request(connection, "GET", "/static/app.js?v=24")
        assert response.status == 200
        assert b"areaSelectedStoryIndex" in body
        assert b"data-area-story-index" in body
        assert b"story-browser" in body
        assert b"analysis_device: state.settings.analysisDevice" in body
        assert b'analysis-self-test' in body
        assert b'profile-self-test' in body
        assert b"analysisSelfTest" in body
        assert b"function runProfileSelfTest" in body
        assert b"function runProfileNextAction" in body
        assert b"function currentProfileAction" in body
        assert b'setupProfileSelfTestButton").addEventListener' in body
        assert b"function syncPlatformProfileOptions" in body
        assert b"whisper.cpp has no managed distil-large-v3 mapping" in body
        assert b' qwen: ["qwen3-asr", "cpu", "cpu", "auto"]' in body
        assert b"Fast CPU preview uses managed Qwen3-ASR 0.6B INT8" in body
        assert b"updateHardwareProfileDescription();" in body
        assert b"function ensureAsrModelCompatibility" in body
        assert b'"windows-ml", "whisper.cpp", "qwen3-asr"' in body
        assert b"profile.configured" in body
        assert b"const nextSteps = isVerified" in body
        assert b"function setSettingsSection" in body
        assert b"function setViewFromLocation" in body
        assert b'window.addEventListener("hashchange", setViewFromLocation)' in body
        assert b'radioArchiveSettingsSection' in body
        assert b'setSettingsSection("processing")' in body

        response, body = _request(connection, "GET", "/static/app.css?v=20")
        assert response.status == 200
        assert b".story-browser" in body
        assert b".story-index-item.active" in body
        assert b".runtime-profile-card.configured" in body
        assert b".settings-section-tab.active" in body
        assert b".settings-panel[hidden]" in body

        response, body = _request(connection, "GET", "/api/bootstrap", cookie=cookie)
        bootstrap = json.loads(body)
        assert response.status == 200
        assert bootstrap["summary"]["day_count"] == 1
        assert bootstrap["days"][0]["feed_name"] == "Example City Public Safety"
        assert bootstrap["runtime"]["loopback_only"] is True
        assert bootstrap["runtime"]["storage_ready"] is True
        assert isinstance(bootstrap["runtime"]["account"]["configured"], bool)

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
        assert len(detail["incidents"]) == 1
        assert detail["incidents"][0]["quote_redacted"] is True
        assert detail["incidents"][0]["quote"] == (
            "Check the welfare of [private person] at the retained location."
        )

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


def test_loopback_web_app_hides_stale_daily_claims(tmp_path: Path) -> None:
    output = tmp_path / "archives"
    database = output / "broadcastify-analysis.sqlite3"
    _retained_day(output, database, prompt_version="older-evidence-rules")
    server = create_server(output, database, port=0, working_dir=Path.cwd())
    server.quiet = True  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        response, _body = _request(connection, "GET", "/")
        cookie = response.getheader("Set-Cookie", "").split(";", 1)[0]

        response, body = _request(connection, "GET", "/api/bootstrap", cookie=cookie)
        bootstrap = json.loads(body)
        assert response.status == 200
        assert bootstrap["days"][0]["has_analysis"] is False
        assert bootstrap["days"][0]["has_stale_analysis"] is True
        assert bootstrap["days"][0]["primary_action"] == "continue_local"

        response, body = _request(
            connection,
            "GET",
            "/api/day?feed_id=90001&date=2026-07-12",
            cookie=cookie,
        )
        detail = json.loads(body)
        assert response.status == 200
        assert detail["summary"] == ""
        assert detail["incidents"] == []
        assert detail["state"]["has_stale_analysis"] is True
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


def test_web_area_story_packages_use_safe_media_urls_without_local_paths(
    tmp_path: Path,
) -> None:
    output = tmp_path / "archives"
    clip = output / "90001" / "20260716" / "evidence-clips" / "I694.mp3"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"clip")
    stories = [
        {
            "story_id": "S1",
            "location": "9805",
            "incident_references": [
                {
                    "incident_id": 694,
                    "clip_path": str(clip),
                    "source_audio_path": str(output / "90001" / "day.mp3"),
                    "quote": "9805, Jordan Example, and for a theft report.",
                }
            ],
        }
    ]

    rendered = _area_stories_for_web(output, stories)
    reference = rendered[0]["incident_references"][0]

    assert reference["media_url"] == "/media?path=90001/20260716/evidence-clips/I694.mp3"
    assert reference["filename"] == "I694.mp3"
    assert reference["quote"] == "9805, [private person], and for a theft report."
    assert reference["quote_redacted"] is True
    assert "clip_path" not in reference
    assert "source_audio_path" not in reference
    assert "Jordan Example" in stories[0]["incident_references"][0]["quote"]
    assert "clip_path" in stories[0]["incident_references"][0]


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


def test_web_jobs_forward_explicit_asr_model_preparation_settings(
    tmp_path: Path,
) -> None:
    manager = JobManager(tmp_path, tmp_path / "analysis.sqlite3", tmp_path)
    arguments, payload = manager._worker_request(  # noqa: SLF001
        "prepare-asr-model",
        {
            "model": "tiny.en",
            "asr_engine": "windows-ml",
            "device": "windows-ml",
            "huggingface_token": "session-only-test-token",
        },
    )

    assert arguments == ["prepare-asr-model"]
    assert payload is not None
    assert payload["model"] == "tiny.en"
    assert payload["asr_engine"] == "windows-ml"
    assert payload["huggingface_token"] == "session-only-test-token"


def test_web_jobs_forward_selected_diagnostics_settings(tmp_path: Path) -> None:
    manager = JobManager(tmp_path, tmp_path / "analysis.sqlite3", tmp_path)
    arguments, payload = manager._worker_request(  # noqa: SLF001
        "diagnostics-selected",
        {
            "model": "tiny.en",
            "asr_engine": "whisper.cpp",
            "device": "vulkan",
        },
    )

    assert arguments == ["diagnostics-selected"]
    assert payload is not None
    assert payload["model"] == "tiny.en"


def test_web_jobs_forward_explicit_diarization_self_test_settings(
    tmp_path: Path,
) -> None:
    manager = JobManager(tmp_path, tmp_path / "analysis.sqlite3", tmp_path)
    arguments, payload = manager._worker_request(  # noqa: SLF001
        "diarization-self-test",
        {
            "diarization_device": "cpu",
            "huggingface_token": "session-only-test-token",
            "batch_size": 4,
        },
    )

    assert arguments == ["diarization-self-test"]
    assert payload is not None
    assert payload["diarization_device"] == "cpu"
    assert payload["huggingface_token"] == "session-only-test-token"


def test_web_jobs_forward_explicit_analysis_self_test_settings(
    tmp_path: Path,
) -> None:
    manager = JobManager(tmp_path, tmp_path / "analysis.sqlite3", tmp_path)
    arguments, payload = manager._worker_request(  # noqa: SLF001
        "analysis-self-test",
        {
            "analysis_provider": "local",
            "analysis_model": "local-model.gguf",
            "analysis_device": "cpu",
        },
    )

    assert arguments == ["analysis-self-test"]
    assert payload is not None
    assert payload["analysis_provider"] == "local"
    assert payload["analysis_model"] == "local-model.gguf"
    assert payload["analysis_device"] == "cpu"


def test_web_jobs_forward_combined_profile_self_test_settings(
    tmp_path: Path,
) -> None:
    manager = JobManager(tmp_path, tmp_path / "analysis.sqlite3", tmp_path)
    arguments, payload = manager._worker_request(  # noqa: SLF001
        "profile-self-test",
        {
            "model": "tiny.en",
            "asr_engine": "openvino",
            "device": "openvino-npu",
            "diarization_device": "cpu",
            "analysis_provider": "local",
            "analysis_model": "local-model.gguf",
            "analysis_device": "cpu",
            "huggingface_token": "session-only-test-token",
        },
    )

    assert arguments == ["profile-self-test"]
    assert payload is not None
    assert payload["asr_engine"] == "openvino"
    assert payload["device"] == "openvino-npu"
    assert payload["diarization_device"] == "cpu"
    assert payload["analysis_model"] == "local-model.gguf"
    assert payload["huggingface_token"] == "session-only-test-token"
