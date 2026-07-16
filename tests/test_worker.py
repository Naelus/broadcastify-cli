import json
import os
from datetime import date
from pathlib import Path

from broadcastify_cli.storage import AnalysisStore
from broadcastify_cli.worker import _day_report, _incident_clip, load_worker_environment


def test_explicit_private_environment_overrides_repository_defaults(
    monkeypatch, tmp_path: Path
) -> None:
    (tmp_path / ".env").write_text(
        'BROADCASTIFY_USERNAME="repository-user"\n'
        'BROADCASTIFY_PASSWORD="repository-password"\n',
        encoding="utf-8",
    )
    bundled = tmp_path / "broadcastify-desktop.env"
    bundled.write_text(
        'BROADCASTIFY_USERNAME="bundled-user"\n'
        'BROADCASTIFY_PASSWORD="bundled-password"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BROADCASTIFY_ENV_FILE", str(bundled))
    monkeypatch.delenv("BROADCASTIFY_USERNAME", raising=False)
    monkeypatch.delenv("BROADCASTIFY_PASSWORD", raising=False)

    loaded = load_worker_environment()

    assert loaded == bundled
    assert os.environ["BROADCASTIFY_USERNAME"] == "bundled-user"
    assert os.environ["BROADCASTIFY_PASSWORD"] == "bundled-password"


def test_day_report_exposes_playback_metadata_without_raw_evidence(tmp_path: Path) -> None:
    archive_date = date(2026, 7, 12)
    audio = tmp_path / "combined_90001_20260712.mp3"
    transcript = tmp_path / "transcript.json"
    audio.write_bytes(b"audio")
    transcript.write_text(
        json.dumps(
            {
                "model": "turbo",
                "duration": 600.0,
                "segments": [
                    {
                        "start": 10.0,
                        "end": 15.0,
                        "text": "Dispatch reports shots fired near Main and First.",
                        "speaker": "SPEAKER_00",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with AnalysisStore(tmp_path / "analysis.sqlite3") as store:
        imported = store.import_transcript(
            "90001", archive_date, transcript, audio
        )
        incident_ids = store.replace_incidents(
            imported.day_id,
            [
                {
                    "fingerprint": "test-incident",
                    "event_type": "shots_fired",
                    "title": "Reported shots fired",
                    "summary": "Units were sent to check a shots-fired report.",
                    "location": "Main and First",
                    "start_seconds": 10.0,
                    "end_seconds": 15.0,
                    "priority": 5,
                    "confidence": 0.9,
                    "evidence": ["sensitive transcript evidence"],
                    "attributes": {"private": "detail"},
                }
            ],
            model="test-model",
            prompt_version="test-prompt",
        )
        store.save_daily_summary(
            imported.day_id,
            "A shots-fired report was dispatched.",
            incident_ids,
            model="test-model",
            prompt_version="test-prompt",
            transcript_sha256=imported.transcript_sha256,
        )

        report = _day_report(store, "90001", archive_date)

    assert report["audio_path"] == str(audio.resolve())
    assert report["summary"] == "A shots-fired report was dispatched."
    assert len(report["incidents"]) == 1
    incident = report["incidents"][0]
    assert incident["start_seconds"] == 10.0
    assert incident["end_seconds"] == 15.0
    assert set(incident) == {
        "id",
        "event_type",
        "title",
        "summary",
        "location",
        "priority",
        "confidence",
        "start_seconds",
        "end_seconds",
        "archive_time",
    }
    assert "evidence" not in incident
    assert "attributes" not in incident


def test_incident_clip_uses_a_timestamped_cache_key(
    monkeypatch, tmp_path: Path
) -> None:
    archive_date = date(2026, 7, 12)
    audio = tmp_path / "combined_90001_20260712.mp3"
    transcript = tmp_path / "transcript.json"
    audio.write_bytes(b"combined audio")
    transcript.write_text(
        json.dumps(
            {
                "model": "turbo",
                "duration": 600.0,
                "segments": [
                    {
                        "start": 100.0,
                        "end": 108.0,
                        "text": "Dispatch reports a vehicle fire at Main Street.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_extract(_source: Path, output: Path, start: float, end: float) -> Path:
        assert start == 92.0
        assert end == 120.0
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"exact evidence clip")
        return output

    monkeypatch.setattr("broadcastify_cli.worker.extract_audio_clip", fake_extract)
    with AnalysisStore(tmp_path / "analysis.sqlite3") as store:
        imported = store.import_transcript("90001", archive_date, transcript, audio)
        incident_id = store.replace_incidents(
            imported.day_id,
            [
                {
                    "fingerprint": "vehicle-fire",
                    "event_type": "fire",
                    "title": "Vehicle fire",
                    "summary": "A vehicle fire was reported at Main Street.",
                    "location": "Main Street",
                    "start_seconds": 100.0,
                    "end_seconds": 108.0,
                    "priority": 4,
                    "confidence": 0.9,
                    "evidence": [
                        {
                            "segment_index": 0,
                            "start_seconds": 100.0,
                            "end_seconds": 108.0,
                            "text": "Dispatch reports a vehicle fire at Main Street.",
                        }
                    ],
                    "attributes": {},
                }
            ],
            model="test-model",
            prompt_version="test-prompt",
        )[0]

        result = _incident_clip(store, incident_id)

    assert result["incident_id"] == incident_id
    assert result["duration_seconds"] == 28.0
    assert result["path"].endswith(f"I{incident_id}_92000-120000.mp3")
    assert len(result["sha256"]) == 64
