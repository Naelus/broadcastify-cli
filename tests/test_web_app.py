from __future__ import annotations

import http.client
import json
import os
import re
import threading
import time
from datetime import date
from pathlib import Path

import pytest

from broadcastify_cli.analysis import (
    PROMPT_VERSION,
    WEEKLY_PROMPT_VERSION,
    current_weekly_summary_source_fingerprint,
)
from broadcastify_cli.area_watch import (
    AREA_PROMPT_VERSION,
    current_area_story_source_fingerprint,
)
from broadcastify_cli.storage import AnalysisStore
from broadcastify_cli.web_app import (
    FeedScheduleCoordinator,
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


def test_web_schedule_coordinator_claims_and_finishes_due_feed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "analysis.sqlite3"
    monkeypatch.setenv(
        "BROADCASTIFY_QUOTA_LEDGER", str(tmp_path / "quota.sqlite3")
    )
    with AnalysisStore(database) as store:
        saved = store.save_feed_schedule(
            {
                "feed_id": "90001",
                "feed_name": "Example Public Safety",
                "run_time_local": "00:00",
                "lookback_days": 1,
                "job": {"combine": False, "transcribe": False},
            }
        )

    class FakeJobs:
        output_dir = tmp_path / "selected-library"

        def start(self, command: str, payload: dict[str, object]) -> dict[str, object]:
            assert command == "run-scheduled"
            assert payload["job"]["feed_id"] == "90001"  # type: ignore[index]
            assert payload["job"]["output_dir"] == str(self.output_dir.resolve())  # type: ignore[index]
            return {"id": "scheduled-job"}

        def get(self, job_id: str) -> dict[str, object]:
            assert job_id == "scheduled-job"
            return {
                "status": "completed",
                "result": {
                    "type": "scheduled_complete",
                    "result": {"download_limited": False},
                },
            }

    coordinator = FeedScheduleCoordinator(  # type: ignore[arg-type]
        FakeJobs(), database, tmp_path, poll_seconds=0.05
    )
    coordinator.check_once()
    coordinator.check_once()

    with AnalysisStore(database) as store:
        result = store.list_feed_schedules()[0]
    assert result["id"] == saved["id"]
    assert result["state"] == "complete"
    assert result["last_run_date"] == date.today().isoformat()


def test_web_schedule_coordinator_retries_deferred_missing_days(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "analysis.sqlite3"
    monkeypatch.setenv(
        "BROADCASTIFY_QUOTA_LEDGER", str(tmp_path / "quota.sqlite3")
    )
    with AnalysisStore(database) as store:
        store.save_feed_schedule(
            {
                "feed_id": "90001",
                "feed_name": "Example Public Safety",
                "run_time_local": "00:00",
                "lookback_days": 1,
                "backfill_start_date": "2026-07-03",
                "job": {"combine": False, "transcribe": False},
            }
        )

    class FakeJobs:
        output_dir = tmp_path / "selected-library"

        def start(self, command: str, payload: dict[str, object]) -> dict[str, object]:
            assert command == "run-scheduled"
            assert payload["job"]["start_date"] == "2026-07-03"  # type: ignore[index]
            return {"id": "scheduled-job"}

        def get(self, job_id: str) -> dict[str, object]:
            assert job_id == "scheduled-job"
            return {
                "status": "completed",
                "result": {
                    "type": "scheduled_complete",
                    "result": {
                        "download_limited": False,
                        "missing_days": ["2026-07-04"],
                    },
                },
            }

    coordinator = FeedScheduleCoordinator(  # type: ignore[arg-type]
        FakeJobs(), database, tmp_path, poll_seconds=0.05
    )
    coordinator.check_once()
    coordinator.check_once()

    with AnalysisStore(database) as store:
        result = store.list_feed_schedules()[0]
    assert result["state"] == "deferred"
    assert result["last_run_date"] == ""
    assert result["backfill_start_date"] == "2026-07-03"
    assert result["due"] is False


def test_loopback_web_app_serves_library_transcript_and_media(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "BROADCASTIFY_QUOTA_LEDGER",
        str(tmp_path / "archive-quota.sqlite3"),
    )
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
        assert b'/static/app.js?v=34' in body
        assert b'id="archiveQuotaNotice"' in body
        assert b'id="saveFeedScheduleButton"' in body
        assert b'id="scheduleBackfillStartDate"' in body
        assert b'id="scheduleEnabled"' in body
        assert b'id="accessScopeStatus"' in body
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
        assert b'id="settingLanSyncEnabled"' in body
        assert b'id="settingLanDiscoveryEnabled"' in body
        assert b'id="settingLanPeerUrls"' in body
        assert b'id="rememberBroadcastifyLogin"' in body
        assert b'id="huggingFaceCredentialForm"' in body
        assert b'href="https://huggingface.co/docs/hub/en/security-tokens"' in body
        assert b'role="tabpanel"' in body
        assert b'/static/favicon.svg' in body
        token = token_match.group(1).decode()

        response, body = _request(connection, "GET", "/static/favicon.svg")
        assert response.status == 200
        assert response.getheader("Content-Type") == "image/svg+xml"
        assert b"<svg" in body

        response, body = _request(connection, "GET", "/static/app.js?v=34")
        assert response.status == 200
        assert b"areaSelectedStoryIndex" in body
        assert b"data-area-story-index" in body
        assert b"data-edit-schedule" in body
        assert b"backfill_start_date" in body
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
        assert b"function applyRuntimeProcessingDefaults" in body
        assert b"hardwareProfile: defaults.hardware_profile" not in body
        assert b'state.settings.hardwareProfile = "auto"' in body
        assert b'profile === "auto"' in body
        assert b"deploymentDefaults.asr_engine" in body
        assert b"deploymentDefaults.model" in body
        assert b"installed ${words(deploymentDefaults.hardware_profile)} deployment preset" in body
        assert b"hardware_profile: state.settings.hardwareProfile" in body
        assert b'"Trusted LAN"' in body
        assert b"whisper.cpp has no managed distil-large-v3 mapping" in body
        assert b' qwen: ["qwen3-asr", "cpu", "cpu", "auto", "sherpa-onnx"]' in body
        assert b"diarization_engine: state.settings.diarizationEngine" in body
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
        assert b"lan_sync_enabled: Boolean(state.settings.lanSyncEnabled)" in body
        assert b"lan_peer_urls: state.settings.lanPeerUrls" in body
        assert b"function renderArchiveQuota" in body
        assert b"function renderFeedSchedules" in body
        assert b"function renderCredentials" in body
        assert b'api("/api/credentials"' in body

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
        assert bootstrap["runtime"]["processing_defaults"] == {}
        assert bootstrap["runtime"]["storage_ready"] is True
        assert isinstance(bootstrap["runtime"]["account"]["configured"], bool)
        assert "credentials" in bootstrap["runtime"]
        assert bootstrap["runtime"]["lan_sync"]["sharing_enabled"] is False
        assert bootstrap["runtime"]["archive_quota"]["provider_limit"] == 250
        assert bootstrap["runtime"]["archive_quota"]["automated_limit"] == 240
        assert bootstrap["runtime"]["archive_quota"]["user_reserve"] == 10
        assert bootstrap["runtime"]["archive_quota"]["remaining"] == 240
        assert bootstrap["schedules"] == []

        response, body = _request(
            connection,
            "POST",
            "/api/schedules",
            cookie=cookie,
            token=token,
            body={
                "feed_id": "90001",
                "feed_name": "Example City Public Safety",
                "run_time_local": "23:59",
                "lookback_days": 2,
                "enabled": False,
                "analyze": True,
                "job": {"combine": True, "transcribe": True, "diarize": True},
            },
        )
        saved_schedule = json.loads(body)["schedule"]
        assert response.status == 200
        assert saved_schedule["feed_id"] == "90001"
        assert saved_schedule["job"]["download_jobs"] == 1

        response, body = _request(
            connection,
            "POST",
            f"/api/schedules/{saved_schedule['id']}/delete",
            cookie=cookie,
            token=token,
            body={},
        )
        assert response.status == 200
        assert json.loads(body)["deleted"] is True

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
        assert detail["incidents"][0]["quote_redacted"] is False
        assert detail["incidents"][0]["quote"] == (
            "Check the welfare of Jordan Example at the retained location."
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


def test_web_credentials_are_encrypted_server_side_and_only_previewed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BROADCASTIFY_USERNAME", raising=False)
    monkeypatch.delenv("BROADCASTIFY_PASSWORD", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    output = tmp_path / "archives"
    output.mkdir()
    credential_path = tmp_path / "credentials.enc"
    server = create_server(
        output,
        port=0,
        working_dir=tmp_path,
        credential_store_path=credential_path,
    )
    server.quiet = True  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection(
        "127.0.0.1",
        server.server_port,
        timeout=5,
    )
    try:
        response, body = _request(connection, "GET", "/")
        assert response.status == 200
        cookie = response.getheader("Set-Cookie", "").split(";", 1)[0]
        token_match = re.search(rb'<meta name="app-token" content="([^"]+)">', body)
        assert token_match is not None
        token = token_match.group(1).decode()

        response, body = _request(
            connection,
            "POST",
            "/api/credentials",
            cookie=cookie,
            token=token,
            body={
                "kind": "broadcastify",
                "action": "save",
                "username": "example-user",
                "secret": "example-password",
            },
        )
        assert response.status == 200
        payload = json.loads(body)
        assert payload["credentials"]["broadcastify"]["password_preview"] == "ex••••"
        assert b"example-password" not in body

        response, body = _request(
            connection,
            "POST",
            "/api/credentials",
            cookie=cookie,
            token=token,
            body={
                "kind": "huggingface",
                "action": "save",
                "secret": "hf_example_read_token_123",
            },
        )
        assert response.status == 200
        payload = json.loads(body)
        assert payload["credentials"]["huggingface"]["token_preview"] == "hf_examp••••"
        assert b"hf_example_read_token_123" not in body

        encrypted = credential_path.read_text(encoding="utf-8")
        assert "example-user" not in encrypted
        assert "example-password" not in encrypted
        assert "hf_example_read_token_123" not in encrypted

        response, body = _request(
            connection,
            "GET",
            "/api/bootstrap",
            cookie=cookie,
        )
        assert response.status == 200
        credentials = json.loads(body)["runtime"]["credentials"]
        assert credentials["broadcastify"]["username"] == "example-user"
        assert credentials["broadcastify"]["password_preview"] == "ex••••"
        assert credentials["huggingface"]["token_preview"] == "hf_examp••••"
        assert "example-password" not in body.decode()
        assert "hf_example_read_token_123" not in body.decode()

        for kind in ("broadcastify", "huggingface"):
            response, _body = _request(
                connection,
                "POST",
                "/api/credentials",
                cookie=cookie,
                token=token,
                body={"kind": kind, "action": "clear"},
            )
            assert response.status == 200
        assert not credential_path.exists()
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


def test_loopback_web_app_hides_results_for_refreshed_combined_audio(
    tmp_path: Path,
) -> None:
    output = tmp_path / "archives"
    database = output / "broadcastify-analysis.sqlite3"
    _retained_day(output, database)
    audio = output / "90001" / "20260712" / "combined_90001_20260712.mp3"
    audio.write_bytes(b"refreshed combined recording")
    future = time.time() + 10
    os.utime(audio, (future, future))
    server = create_server(output, database, port=0, working_dir=Path.cwd())
    server.quiet = True  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection(
        "127.0.0.1", server.server_port, timeout=5
    )
    try:
        response, _body = _request(connection, "GET", "/")
        cookie = response.getheader("Set-Cookie", "").split(";", 1)[0]

        response, body = _request(
            connection, "GET", "/api/bootstrap", cookie=cookie
        )
        day = json.loads(body)["days"][0]
        assert response.status == 200
        assert day["has_transcript"] is False
        assert day["has_stale_transcript"] is True
        assert day["has_analysis"] is False
        assert day["segment_count"] == 0
        assert day["incident_count"] == 0

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

        response, body = _request(
            connection,
            "GET",
            "/api/transcript?feed_id=90001&date=2026-07-12",
            cookie=cookie,
        )
        transcript = json.loads(body)
        assert response.status == 200
        assert transcript["segments"] == []
        assert transcript["total"] == 0
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_loopback_web_app_reads_current_file_when_import_revision_is_older(
    tmp_path: Path,
) -> None:
    output = tmp_path / "archives"
    database = output / "broadcastify-analysis.sqlite3"
    _retained_day(output, database)
    transcript_path = (
        output
        / "90001"
        / "20260712"
        / "transcripts"
        / "combined_90001_20260712.json"
    )
    transcript_path.write_text(
        json.dumps(
            {
                "duration": 12.0,
                "diarization_completed": True,
                "segments": [
                    {
                        "start": 3.0,
                        "end": 4.0,
                        "speaker": "SPEAKER_01",
                        "text": "Current file text.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    server = create_server(output, database, port=0, working_dir=Path.cwd())
    server.quiet = True  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection(
        "127.0.0.1", server.server_port, timeout=5
    )
    try:
        response, _body = _request(connection, "GET", "/")
        cookie = response.getheader("Set-Cookie", "").split(";", 1)[0]

        response, body = _request(
            connection, "GET", "/api/bootstrap", cookie=cookie
        )
        day = json.loads(body)["days"][0]
        assert response.status == 200
        assert day["has_transcript"] is True
        assert day["has_imported_transcript"] is False
        assert day["has_analysis"] is False

        response, body = _request(
            connection,
            "GET",
            "/api/transcript?feed_id=90001&date=2026-07-12",
            cookie=cookie,
        )
        payload = json.loads(body)
        assert response.status == 200
        assert payload["total"] == 1
        assert payload["segments"][0]["text"] == "Current file text."
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_loopback_web_app_hides_saved_week_after_current_daily_summary_changes(
    tmp_path: Path,
) -> None:
    output = tmp_path / "archives"
    database = output / "broadcastify-analysis.sqlite3"
    _retained_day(output, database)
    week_ending = date(2026, 7, 12)
    with AnalysisStore(database) as store:
        fingerprint = current_weekly_summary_source_fingerprint(
            store,
            "90001",
            date(2026, 7, 6),
            week_ending,
        )
        store.save_weekly_summary(
            "90001",
            date(2026, 7, 6),
            week_ending,
            "Current saved week.",
            [],
            1,
            1,
            "test",
            WEEKLY_PROMPT_VERSION,
            fingerprint,
        )

    server = create_server(output, database, port=0, working_dir=Path.cwd())
    server.quiet = True  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection(
        "127.0.0.1", server.server_port, timeout=5
    )
    try:
        response, _body = _request(connection, "GET", "/")
        cookie = response.getheader("Set-Cookie", "").split(";", 1)[0]
        endpoint = "/api/saved-week?feed_id=90001&week_ending=2026-07-12"

        response, body = _request(connection, "GET", endpoint, cookie=cookie)
        current = json.loads(body)
        assert response.status == 200
        assert current["stale"] is False
        assert current["result"]["summary"] == "Current saved week."

        with AnalysisStore(database) as store:
            day = store.get_day("90001", week_ending)
            assert day is not None
            store.save_daily_summary(
                int(day["id"]),
                "A newer current daily summary.",
                [],
                model="test",
                prompt_version=PROMPT_VERSION,
                transcript_sha256=str(day["transcript_sha256"]),
            )
        response, body = _request(connection, "GET", endpoint, cookie=cookie)
        stale = json.loads(body)
        assert response.status == 200
        assert stale["stale"] is True
        assert stale["result"] is None
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_loopback_web_app_hides_saved_area_after_current_incidents_change(
    tmp_path: Path,
) -> None:
    output = tmp_path / "archives"
    database = output / "broadcastify-analysis.sqlite3"
    _retained_day(output, database)
    archive_date = date(2026, 7, 12)
    with AnalysisStore(database) as store:
        profile = store.save_area_profile(
            "ExampleArea",
            ["00000"],
            [{"feed_id": "90001", "name": "Example Public Safety"}],
        )
        fingerprint = current_area_story_source_fingerprint(
            store,
            profile,
            archive_date,
            archive_date,
        )
        store.save_area_story_digest(
            int(profile["id"]),
            archive_date,
            archive_date,
            "Current saved area brief.",
            [],
            {"incident_prompt_version": PROMPT_VERSION},
            "test",
            AREA_PROMPT_VERSION,
            fingerprint,
        )

    server = create_server(output, database, port=0, working_dir=Path.cwd())
    server.quiet = True  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection(
        "127.0.0.1", server.server_port, timeout=5
    )
    try:
        response, _body = _request(connection, "GET", "/")
        cookie = response.getheader("Set-Cookie", "").split(";", 1)[0]
        endpoint = "/api/saved-area-digest?profile_name=ExampleArea"

        response, body = _request(connection, "GET", endpoint, cookie=cookie)
        current = json.loads(body)
        assert response.status == 200
        assert current["stale"] is False
        assert current["result"]["summary"] == "Current saved area brief."

        with AnalysisStore(database) as store:
            day = store.get_day("90001", archive_date)
            assert day is not None
            incident_ids = store.replace_incidents(
                int(day["id"]),
                [
                    {
                        "fingerprint": "new-current-area-incident",
                        "event_type": "fire",
                        "title": "New current incident",
                        "summary": "A new current incident was extracted.",
                        "location": "Example Place",
                        "start_seconds": 3.0,
                        "end_seconds": 4.0,
                        "priority": 4,
                        "confidence": 0.9,
                        "evidence": [],
                        "attributes": {},
                    }
                ],
                model="test",
                prompt_version=PROMPT_VERSION,
            )
            store.save_daily_summary(
                int(day["id"]),
                "A newer current daily summary.",
                incident_ids,
                model="test",
                prompt_version=PROMPT_VERSION,
                transcript_sha256=str(day["transcript_sha256"]),
            )

        response, body = _request(connection, "GET", endpoint, cookie=cookie)
        stale = json.loads(body)
        assert response.status == 200
        assert stale["stale"] is True
        assert stale["result"] is None
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_web_app_allows_explicit_trusted_lan_binding(tmp_path: Path) -> None:
    server = create_server(tmp_path, host="0.0.0.0", port=0)
    server.quiet = True  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        response, body = _request(connection, "GET", "/health")
        assert response.status == 200
        assert json.loads(body) == {"status": "ok", "scope": "trusted-lan"}

        response, body = _request(connection, "GET", "/")
        cookie = response.getheader("Set-Cookie", "").split(";", 1)[0]
        assert response.status == 200
        response, body = _request(connection, "GET", "/api/bootstrap", cookie=cookie)
        assert response.status == 200
        runtime = json.loads(body)["runtime"]
        assert runtime["loopback_only"] is False
        assert runtime["access_scope"] == "trusted-lan"
        assert runtime["bind_host"] == "0.0.0.0"
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_web_app_advertises_validated_deployment_processing_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    defaults = {
        "BROADCASTIFY_DEFAULT_HARDWARE_PROFILE": "vulkan",
        "BROADCASTIFY_DEFAULT_WHISPER_MODEL": "tiny.en",
        "BROADCASTIFY_DEFAULT_ASR_ENGINE": "whisper.cpp",
        "BROADCASTIFY_DEFAULT_DEVICE": "vulkan",
        "BROADCASTIFY_DEFAULT_DIARIZATION_ENGINE": "sherpa-onnx",
        "BROADCASTIFY_DEFAULT_DIARIZATION_DEVICE": "cpu",
        "BROADCASTIFY_DEFAULT_BATCH_SIZE": "8",
    }
    for name, value in defaults.items():
        monkeypatch.setenv(name, value)
    server = create_server(tmp_path, port=0, working_dir=tmp_path)
    server.quiet = True  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection(
        "127.0.0.1", server.server_port, timeout=5
    )
    try:
        response, _body = _request(connection, "GET", "/")
        cookie = response.getheader("Set-Cookie", "").split(";", 1)[0]
        response, body = _request(
            connection,
            "GET",
            "/api/bootstrap",
            cookie=cookie,
        )
        assert response.status == 200
        assert json.loads(body)["runtime"]["processing_defaults"] == {
            "hardware_profile": "vulkan",
            "model": "tiny.en",
            "asr_engine": "whisper.cpp",
            "device": "vulkan",
            "diarization_engine": "sherpa-onnx",
            "diarization_device": "cpu",
            "batch_size": 8,
        }
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_web_app_rejects_public_binding(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="public"):
        create_server(tmp_path, host="8.8.8.8", port=0)


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
    assert payload["lan_sync_enabled"] is True
    assert payload["lan_discovery_enabled"] is True


def test_web_archive_questions_use_the_selected_local_library(tmp_path: Path) -> None:
    manager = JobManager(tmp_path, tmp_path / "analysis.sqlite3", tmp_path)
    arguments, payload = manager._worker_request(  # noqa: SLF001
        "ask",
        {
            "feed_id": "90001",
            "start_date": "2026-07-01",
            "end_date": "2026-07-31",
            "question": "What happened this month?",
        },
    )

    assert arguments == ["ask"]
    assert payload is not None
    assert payload["output_dir"] == str(tmp_path)


def test_web_jobs_resolve_automatic_against_the_installed_deployment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    defaults = {
        "BROADCASTIFY_DEFAULT_HARDWARE_PROFILE": "vulkan",
        "BROADCASTIFY_DEFAULT_WHISPER_MODEL": "tiny.en",
        "BROADCASTIFY_DEFAULT_ASR_ENGINE": "whisper.cpp",
        "BROADCASTIFY_DEFAULT_DEVICE": "vulkan",
        "BROADCASTIFY_DEFAULT_DIARIZATION_ENGINE": "sherpa-onnx",
        "BROADCASTIFY_DEFAULT_DIARIZATION_DEVICE": "cpu",
    }
    for name, value in defaults.items():
        monkeypatch.setenv(name, value)
    manager = JobManager(tmp_path, tmp_path / "analysis.sqlite3", tmp_path)
    _arguments, payload = manager._worker_request(  # noqa: SLF001
        "run",
        {
            "feed_id": "90001",
            "start_date": "2026-07-17",
            "end_date": "2026-07-18",
            "hardware_profile": "auto",
            "model": "turbo",
            "asr_engine": "auto",
            "device": "auto",
            "diarization_engine": "community-1",
            "diarization_device": "auto",
        },
    )

    assert payload is not None
    assert payload["hardware_profile"] == "vulkan"
    assert payload["model"] == "tiny.en"
    assert payload["asr_engine"] == "whisper.cpp"
    assert payload["device"] == "vulkan"
    assert payload["diarization_engine"] == "sherpa-onnx"
    assert payload["diarization_device"] == "cpu"

    _arguments, legacy = manager._worker_request(  # noqa: SLF001
        "continue-local",
        {
            "feed_id": "90001",
            "archive_date": "2026-07-17",
            "model": "turbo",
            "asr_engine": "auto",
            "device": "auto",
            "diarization_engine": "community-1",
        },
    )
    assert legacy is not None
    assert legacy["hardware_profile"] == "vulkan"
    assert legacy["model"] == "tiny.en"
    assert legacy["diarization_engine"] == "sherpa-onnx"

    _arguments, custom = manager._worker_request(  # noqa: SLF001
        "asr-self-test",
        {
            "hardware_profile": "custom",
            "model": "base.en",
            "asr_engine": "whisper.cpp",
            "device": "cpu",
        },
    )
    assert custom is not None
    assert custom["model"] == "base.en"
    assert custom["device"] == "cpu"


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
    assert payload["job"]["lan_sync_enabled"] is True
    assert payload["job"]["lan_discovery_enabled"] is True


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
    assert reference["quote"] == "9805, Jordan Example, and for a theft report."
    assert reference["quote_redacted"] is False
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
