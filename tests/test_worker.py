import json
import os
from contextlib import nullcontext
from datetime import date
from pathlib import Path

from broadcastify_cli.storage import AnalysisStore
from broadcastify_cli.worker import (
    _day_report,
    _incident_clip,
    asr_self_test,
    diarization_self_test,
    load_worker_environment,
)


def test_asr_self_test_uses_selected_engine_without_returning_transcript_text(
    monkeypatch,
) -> None:
    emitted: list[dict[str, object]] = []
    initialized: dict[str, object] = {}

    class FakeTranscriber:
        asr_engine = "openvino"
        backend_description = "OpenVINO CPU (fallback from NPU)"
        device = "openvino-npu"

        def __init__(self, **kwargs: object) -> None:
            initialized.update(kwargs)

        def transcribe_file(self, audio_path: Path, progress=None) -> Path:
            assert audio_path.is_file()
            if progress:
                progress("Synthetic audio decoded")
            transcript_dir = audio_path.parent / "transcripts"
            transcript_dir.mkdir()
            result = transcript_dir / "silence.json"
            result.write_text(
                json.dumps(
                    {
                        "asr_engine": self.asr_engine,
                        "asr_backend": self.backend_description,
                        "device": self.device,
                        "text": "never return this transcript",
                        "segments": [],
                        "words": [],
                        "asr_metadata": {
                            "fallback_reason": "NPU compilation failed",
                            "fallback_stage": "initialization",
                        },
                    }
                ),
                encoding="utf-8",
            )
            return result

    monkeypatch.setattr("broadcastify_cli.worker.LocalTranscriber", FakeTranscriber)
    monkeypatch.setattr("broadcastify_cli.worker.emit", emitted.append)

    exit_code = asr_self_test(
        {
            "model": "tiny",
            "asr_engine": "openvino",
            "device": "openvino-npu",
            "batch_size": 4,
        }
    )

    result = next(value["result"] for value in emitted if value["type"] == "asr_self_test")
    assert exit_code == 0
    assert initialized["model_name"] == "tiny"
    assert initialized["device"] == "openvino-npu"
    assert result["ready"] is True
    assert result["fallback_stage"] == "initialization"
    assert "never return" not in json.dumps(result)


def test_diarization_self_test_executes_generated_audio_without_returning_token(
    monkeypatch,
) -> None:
    emitted: list[dict[str, object]] = []
    loaded: dict[str, object] = {}

    class FakeAnnotation:
        def itertracks(self, *, yield_label: bool = False):
            assert yield_label is True
            yield object(), object(), "SPEAKER_00"

    class FakePipeline:
        embedding_batch_size = 2

        def __call__(self, audio: dict[str, object]) -> FakeAnnotation:
            assert audio["sample_rate"] == 16_000
            return FakeAnnotation()

    def fake_load(**kwargs: object) -> tuple[FakePipeline, str]:
        loaded.update(kwargs)
        return FakePipeline(), "cpu"

    monkeypatch.setattr("broadcastify_cli.worker._load_diarization_pipeline", fake_load)
    monkeypatch.setattr(
        "broadcastify_cli.worker.decoded_diarization_audio",
        lambda _path: nullcontext({"waveform": object(), "sample_rate": 16_000}),
    )
    monkeypatch.setattr("broadcastify_cli.worker.emit", emitted.append)

    exit_code = diarization_self_test(
        {
            "huggingface_token": "private-test-token",
            "diarization_device": "cpu",
            "batch_size": 4,
        }
    )

    result = next(
        value["result"]
        for value in emitted
        if value["type"] == "diarization_self_test"
    )
    assert exit_code == 0
    assert loaded["token"] == "private-test-token"
    assert result["ready"] is True
    assert result["turn_count"] == 1
    assert "private-test-token" not in json.dumps(emitted)


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
