import json
from datetime import date
from pathlib import Path

from broadcastify_cli.analysis import PROMPT_VERSION
from broadcastify_cli.library import (
    LocalProcessingRequest,
    prepare_local_day,
    scan_local_library,
)
from broadcastify_cli.storage import AnalysisStore
from broadcastify_cli.transcription import LocalTranscriber, SpeakerTurn


def _day(tmp_path: Path, feed_id: str, value: str) -> Path:
    result = tmp_path / feed_id / value.replace("-", "")
    result.mkdir(parents=True)
    return result


def test_library_discovers_partial_and_analyzed_days(tmp_path: Path) -> None:
    partial = _day(tmp_path, "90003", "2026-07-07")
    (partial / "202607070000-1-90003.mp3").write_bytes(b"raw")
    (partial / "combined_90003_20260707.manifest.json").write_text(
        json.dumps(
            {
                "feed_id": "90003",
                "feed_name": "Example County Public Safety",
                "archive_date": "2026-07-07",
                "sources": [],
            }
        ),
        encoding="utf-8",
    )

    ready = _day(tmp_path, "90001", "2026-07-12")
    audio = ready / "combined_90001_20260712.mp3"
    transcript = ready / "transcripts" / "combined_90001_20260712.json"
    audio.write_bytes(b"audio")
    transcript.parent.mkdir()
    transcript.write_text(
        json.dumps(
            {
                "model": "turbo",
                "duration": 60.0,
                "segments": [
                    {
                        "start": 1.0,
                        "end": 2.0,
                        "text": "Unit responding.",
                        "speaker": "SPEAKER_00",
                    }
                ],
                "diarization_requested": True,
            }
        ),
        encoding="utf-8",
    )
    database = tmp_path / "analysis.sqlite3"
    with AnalysisStore(database) as store:
        imported = store.import_transcript("90001", date(2026, 7, 12), transcript, audio)
        store.save_daily_summary(
            imported.day_id,
            "One dispatch call was retained.",
            [],
            model="test",
            prompt_version=PROMPT_VERSION,
            transcript_sha256=imported.transcript_sha256,
        )
        store.save_feed_catalog(
            [{"feed_id": "90001", "name": "Example City Public Safety"}]
        )

    states = scan_local_library(tmp_path, database)

    assert len(states) == 2
    analyzed = next(value for value in states if value["feed_id"] == "90001")
    assert analyzed["feed_name"] == "Example City Public Safety"
    assert analyzed["has_diarization"] is True
    assert analyzed["has_analysis"] is True
    assert analyzed["is_complete"] is True
    assert analyzed["primary_action"] == "open_review"

    incomplete = next(value for value in states if value["feed_id"] == "90003")
    assert incomplete["feed_name"] == "Example County Public Safety"
    assert incomplete["raw_file_count"] == 1
    assert incomplete["needs_network"] is True
    assert incomplete["primary_action"] == "resume_download"


def test_library_marks_older_analysis_for_local_evidence_update(tmp_path: Path) -> None:
    ready = _day(tmp_path, "90001", "2026-07-11")
    audio = ready / "combined_90001_20260711.mp3"
    transcript = ready / "transcripts" / "combined_90001_20260711.json"
    audio.write_bytes(b"audio")
    transcript.parent.mkdir()
    transcript.write_text(
        json.dumps(
            {
                "duration": 60.0,
                "segments": [
                    {
                        "start": 1.0,
                        "end": 2.0,
                        "text": "Unit responding.",
                        "speaker": "SPEAKER_00",
                    }
                ],
                "diarization_requested": True,
                "diarization_model": "pyannote/test",
            }
        ),
        encoding="utf-8",
    )
    database = tmp_path / "analysis.sqlite3"
    with AnalysisStore(database) as store:
        imported = store.import_transcript("90001", date(2026, 7, 11), transcript, audio)
        store.save_daily_summary(
            imported.day_id,
            "An older summary.",
            [],
            model="test",
            prompt_version="older-evidence-rules",
            transcript_sha256=imported.transcript_sha256,
        )

    state = scan_local_library(tmp_path, database)[0]

    assert state["has_analysis"] is False
    assert state["has_stale_analysis"] is True
    assert state["status"] == "Analysis update available"
    assert state["next_step"] == "Re-run evidence analysis"
    assert state["primary_action"] == "continue_local"
    assert state["can_open_review"] is False
    assert state["needs_network"] is False


def test_prepare_local_day_uses_diarization_only_for_existing_transcript(
    monkeypatch, tmp_path: Path
) -> None:
    day = _day(tmp_path, "90001", "2026-07-12")
    audio = day / "combined_90001_20260712.mp3"
    transcript = day / "transcripts" / "combined_90001_20260712.json"
    audio.write_bytes(b"audio")
    transcript.parent.mkdir()
    transcript.write_text(
        json.dumps(
            {
                "model": "turbo",
                "segments": [{"start": 0.0, "end": 1.0, "text": "Radio"}],
                "words": [],
                "diarization_requested": False,
            }
        ),
        encoding="utf-8",
    )
    constructor_arguments = []

    class FakeTranscriber:
        def __init__(self, **kwargs: object) -> None:
            constructor_arguments.append(kwargs)

        def diarize_existing_transcript(
            self, _audio: Path, value: Path, progress=None
        ) -> Path:
            payload = json.loads(value.read_text(encoding="utf-8"))
            payload["diarization_requested"] = True
            value.write_text(json.dumps(payload), encoding="utf-8")
            return value

    monkeypatch.setattr("broadcastify_cli.library.LocalTranscriber", FakeTranscriber)
    result = prepare_local_day(
        LocalProcessingRequest(
            feed_id="90001",
            archive_date=date(2026, 7, 12),
            output_dir=tmp_path,
        )
    )

    assert result["operation"] == "diarized"
    assert constructor_arguments[0]["load_asr"] is False


def test_library_marks_existing_plain_transcript_for_diarization(tmp_path: Path) -> None:
    day = _day(tmp_path, "300", "2026-07-10")
    (day / "combined_300_20260710.mp3").write_bytes(b"audio")
    transcript = day / "transcripts" / "combined_300_20260710.json"
    transcript.parent.mkdir()
    transcript.write_text(
        json.dumps(
            {
                "model": "turbo",
                "segments": [{"start": 0.0, "end": 1.0, "text": "Dispatch"}],
                "diarization_requested": False,
            }
        ),
        encoding="utf-8",
    )

    state = scan_local_library(tmp_path, tmp_path / "analysis.sqlite3")[0]

    assert state["status"] == "Transcript ready"
    assert state["next_step"] == "Add speaker labels"
    assert state["primary_action"] == "continue_local"
    assert state["needs_network"] is False


def test_library_does_not_treat_a_request_flag_as_completed_diarization(
    tmp_path: Path,
) -> None:
    day = _day(tmp_path, "301", "2026-07-10")
    (day / "combined_301_20260710.mp3").write_bytes(b"audio")
    transcript = day / "transcripts" / "combined_301_20260710.json"
    transcript.parent.mkdir()
    transcript.write_text(
        json.dumps(
            {
                "model": "turbo",
                "segments": [{"start": 0.0, "end": 1.0, "text": "Dispatch"}],
                "diarization_requested": True,
                "diarization_model": "pyannote/test",
            }
        ),
        encoding="utf-8",
    )

    state = scan_local_library(tmp_path, tmp_path / "analysis.sqlite3")[0]

    assert state["has_diarization"] is False
    assert state["next_step"] == "Add speaker labels"


def test_diarize_existing_transcript_reuses_words_without_whisper(tmp_path: Path) -> None:
    audio = tmp_path / "combined.mp3"
    transcript = tmp_path / "transcripts" / "combined.json"
    audio.write_bytes(b"audio")
    transcript.parent.mkdir()
    transcript.write_text(
        json.dumps(
            {
                "model": "turbo",
                "segments": [{"start": 0.0, "end": 1.0, "text": "Dispatch calling"}],
                "words": [
                    {"start": 0.0, "end": 0.5, "text": " Dispatch"},
                    {"start": 0.5, "end": 1.0, "text": " calling"},
                ],
            }
        ),
        encoding="utf-8",
    )

    transcriber = object.__new__(LocalTranscriber)
    transcriber.diarize = True
    transcriber.diarization_device = "cpu"
    transcriber._diarization_pipeline = object()
    transcriber._diarize = lambda *_args, **_kwargs: [
        SpeakerTurn(0.0, 1.0, "SPEAKER_00")
    ]

    result = transcriber.diarize_existing_transcript(audio, transcript)
    payload = json.loads(result.read_text(encoding="utf-8"))

    assert payload["diarization_requested"] is True
    assert payload["diarization_device"] == "cpu"
    assert payload["segments"][0]["speaker"] == "SPEAKER_00"
    assert payload["words"][0]["speaker"] == "SPEAKER_00"
    assert "SPEAKER_00: Dispatch calling" in transcript.with_suffix(".txt").read_text(
        encoding="utf-8"
    )
