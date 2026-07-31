import json
import os
import time
from datetime import date
from pathlib import Path

from broadcastify_cli.analysis import PROMPT_VERSION
from broadcastify_cli.library import (
    LocalProcessingRequest,
    prepare_local_day,
    scan_local_library,
    transcript_satisfies_diarization,
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


def test_library_separates_working_audio_and_cleans_old_orphans(
    tmp_path: Path,
) -> None:
    day = _day(tmp_path, "90004", "2026-07-29")
    raw = day / "202607290000-1-90004.mp3"
    raw.write_bytes(b"retained")
    cache = day / "transcripts" / ".cache"
    cache.mkdir(parents=True)
    prepared = cache / "combined_90004_20260729.pyannote.flac"
    prepared.write_bytes(b"reusable retry input")
    active_raw = (
        cache
        / f".combined_90004_20260729.pyannote.{os.getpid()}.123.pyannote.f32le"
    )
    active_raw.write_bytes(b"active raw scratch")
    orphan_raw = (
        cache
        / ".combined_90004_20260729.pyannote.999999999.456.pyannote.f32le"
    )
    orphan_raw.write_bytes(b"orphan raw scratch")
    orphan_part = day / ".combined_90004_20260729.old.part.mp3"
    orphan_part.write_bytes(b"orphan combined output")
    old = time.time() - 7_200
    for path in (active_raw, orphan_raw, orphan_part):
        os.utime(path, (old, old))

    state = scan_local_library(tmp_path)[0]

    assert state["storage_bytes"] == raw.stat().st_size
    assert state["working_storage_bytes"] == (
        prepared.stat().st_size + active_raw.stat().st_size
    )
    assert prepared.exists()
    assert active_raw.exists()
    assert not orphan_raw.exists()
    assert not orphan_part.exists()


def test_library_removes_prepared_audio_after_exact_diarization_cache(
    tmp_path: Path,
) -> None:
    day = _day(tmp_path, "90004", "2026-07-29")
    audio = day / "combined_90004_20260729.mp3"
    audio.write_bytes(b"combined audio")
    cache = day / "transcripts" / ".cache"
    cache.mkdir(parents=True)
    prepared = cache / "combined_90004_20260729.pyannote.flac"
    prepared.write_bytes(b"completed preparation")
    stat = audio.stat()
    (cache.parent / "combined_90004_20260729.diarization.json").write_text(
        json.dumps(
            {
                "engine": "community-1",
                "audio_size": stat.st_size,
                "audio_mtime_ns": stat.st_mtime_ns,
                "turns": [],
            }
        ),
        encoding="utf-8",
    )
    old = time.time() - 7_200
    os.utime(prepared, (old, old))

    state = scan_local_library(tmp_path)[0]

    assert not prepared.exists()
    assert state["working_storage_bytes"] == 0
    assert state["storage_bytes"] == (
        audio.stat().st_size
        + (
            cache.parent / "combined_90004_20260729.diarization.json"
        ).stat().st_size
    )


def test_library_rejects_transcript_older_than_refreshed_combined_audio(
    tmp_path: Path,
) -> None:
    day = _day(tmp_path, "90005", "2026-07-29")
    audio = day / "combined_90005_20260729.mp3"
    audio.write_bytes(b"first combined audio")
    transcript = day / "transcripts" / "combined_90005_20260729.json"
    transcript.parent.mkdir()
    transcript.write_text(
        json.dumps(
            {
                "segments": [{"start": 0.0, "end": 1.0, "text": "old"}],
                "diarization_completed": True,
            }
        ),
        encoding="utf-8",
    )
    future = time.time() + 10
    audio.write_bytes(b"refreshed combined audio")
    os.utime(audio, (future, future))

    state = scan_local_library(tmp_path)[0]

    assert state["has_combined"] is True
    assert state["has_transcript"] is False
    assert state["has_diarization"] is False
    assert state["has_analysis"] is False
    assert state["status"] == "Audio ready"
    assert state["next_step"] == "Transcribe locally"


def test_library_does_not_present_older_combined_timeline_as_current(
    tmp_path: Path,
) -> None:
    day = _day(tmp_path, "90001", "2026-07-20")
    first = day / "202607200000-1-90001.mp3"
    newer = day / "202607200030-2-90001.mp3"
    first.write_bytes(b"first")
    combined = day / "combined_90001_20260720.mp3"
    combined.write_bytes(b"older combined audio")
    manifest = combined.with_suffix(".manifest.json")
    manifest.write_text(
        json.dumps({"sources": [{"source_file": first.name}]}),
        encoding="utf-8",
    )
    newer.write_bytes(b"newer")
    transcript = day / "transcripts" / "combined_90001_20260720.json"
    transcript.parent.mkdir()
    transcript.write_text(
        json.dumps(
            {
                "segments": [{"start": 0.0, "end": 1.0, "text": "old"}],
                "diarization_completed": True,
            }
        ),
        encoding="utf-8",
    )

    state = scan_local_library(tmp_path)[0]

    assert state["raw_file_count"] == 2
    assert state["has_stale_combined"] is True
    assert state["has_combined"] is False
    assert state["has_transcript"] is False
    assert state["has_diarization"] is False
    assert state["has_analysis"] is False
    assert state["can_open_review"] is False
    assert state["is_complete"] is False
    assert state["pipeline_percent"] == 20
    assert state["combined_path"] == ""
    assert state["status"] == "New audio pending combine"
    assert state["next_step"] == "Refresh archive day"
    assert state["primary_action"] == "resume_download"
    assert state["needs_network"] is True


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


def test_prepare_local_day_progress_uses_selected_transcription_model_language(
    monkeypatch, tmp_path: Path
) -> None:
    day = _day(tmp_path, "90001", "2026-07-13")
    audio = day / "combined_90001_20260713.mp3"
    audio.write_bytes(b"audio")
    messages: list[str] = []

    class FakeTranscriber:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def transcribe_file(self, _audio: Path, progress=None) -> Path:
            transcript = (
                day / "transcripts" / "combined_90001_20260713.json"
            )
            transcript.parent.mkdir()
            transcript.write_text(
                json.dumps({"segments": [], "words": []}),
                encoding="utf-8",
            )
            return transcript

    monkeypatch.setattr("broadcastify_cli.library.LocalTranscriber", FakeTranscriber)

    result = prepare_local_day(
        LocalProcessingRequest(
            feed_id="90001",
            archive_date=date(2026, 7, 13),
            output_dir=tmp_path,
            model="qwen3-asr-0.6b-int8",
            asr_engine="qwen3-asr",
            diarize=False,
        ),
        progress=messages.append,
    )

    assert result["operation"] == "transcribed"
    assert messages == [
        "Loading local transcription qwen3-asr-0.6b-int8 for 2026-07-13…"
    ]
    assert all("Whisper" not in message for message in messages)


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
    assert state["status_detail"] == (
        "Transcript exists; speaker labels can run without repeating transcription"
    )
    assert state["primary_action"] == "continue_local"
    assert state["needs_network"] is False


def test_library_describes_audio_ready_state_without_assuming_whisper(
    tmp_path: Path,
) -> None:
    day = _day(tmp_path, "299", "2026-07-10")
    (day / "combined_299_20260710.mp3").write_bytes(b"audio")

    state = scan_local_library(tmp_path, tmp_path / "analysis.sqlite3")[0]

    assert state["status"] == "Audio ready"
    assert state["next_step"] == "Transcribe locally"
    assert state["status_detail"] == (
        "Combined audio is ready for local transcription"
    )
    assert "Whisper" not in state["status_detail"]


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


def test_library_labels_portable_speakers_as_preview_with_accuracy_upgrade(
    tmp_path: Path,
) -> None:
    day = _day(tmp_path, "302", "2026-07-10")
    (day / "combined_302_20260710.mp3").write_bytes(b"audio")
    transcript = day / "transcripts" / "combined_302_20260710.json"
    transcript.parent.mkdir()
    transcript.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "text": "Dispatch",
                        "speaker": "SPEAKER_00",
                    }
                ],
                "speaker_turns": [
                    {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}
                ],
                "diarization_requested": True,
                "diarization_completed": True,
                "diarization_engine": "sherpa-onnx",
                "diarization_model": (
                    "pyannote-segmentation-3.0-int8+nemo-titanet-small"
                ),
                "diarization_quality": "preview",
            }
        ),
        encoding="utf-8",
    )

    state = scan_local_library(tmp_path, tmp_path / "analysis.sqlite3")[0]

    assert state["has_diarization"] is True
    assert state["diarization_engine"] == "sherpa-onnx"
    assert state["diarization_quality"] == "preview"
    assert state["speaker_upgrade_available"] is True
    assert transcript_satisfies_diarization(transcript, "sherpa-onnx") is True
    assert transcript_satisfies_diarization(transcript, "community-1") is False


def test_community_labels_satisfy_preview_without_being_downgraded(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "transcript.json"
    transcript.write_text(
        json.dumps(
            {
                "diarization_requested": True,
                "diarization_completed": True,
                "diarization_engine": "community-1",
                "diarization_model": "pyannote/speaker-diarization-community-1",
            }
        ),
        encoding="utf-8",
    )

    assert transcript_satisfies_diarization(transcript, "community-1") is True
    assert transcript_satisfies_diarization(transcript, "sherpa-onnx") is True


def test_prepare_local_day_upgrades_preview_without_loading_asr(
    monkeypatch, tmp_path: Path
) -> None:
    day = _day(tmp_path, "303", "2026-07-10")
    audio = day / "combined_303_20260710.mp3"
    transcript = day / "transcripts" / "combined_303_20260710.json"
    audio.write_bytes(b"audio")
    transcript.parent.mkdir()
    transcript.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "text": "Dispatch",
                        "speaker": "SPEAKER_00",
                    }
                ],
                "words": [],
                "diarization_requested": True,
                "diarization_completed": True,
                "diarization_engine": "sherpa-onnx",
                "diarization_model": (
                    "pyannote-segmentation-3.0-int8+nemo-titanet-small"
                ),
            }
        ),
        encoding="utf-8",
    )
    constructor_arguments: list[dict[str, object]] = []

    class FakeTranscriber:
        def __init__(self, **kwargs: object) -> None:
            constructor_arguments.append(kwargs)

        def diarize_existing_transcript(
            self, _audio: Path, value: Path, progress=None
        ) -> Path:
            payload = json.loads(value.read_text(encoding="utf-8"))
            payload.update(
                diarization_engine="community-1",
                diarization_model="pyannote/speaker-diarization-community-1",
                diarization_completed=True,
            )
            value.write_text(json.dumps(payload), encoding="utf-8")
            return value

    monkeypatch.setattr("broadcastify_cli.library.LocalTranscriber", FakeTranscriber)

    result = prepare_local_day(
        LocalProcessingRequest(
            feed_id="303",
            archive_date=date(2026, 7, 10),
            output_dir=tmp_path,
            diarization_engine="community-1",
        )
    )

    assert result["operation"] == "upgraded_diarization"
    assert result["diarization_engine"] == "community-1"
    assert constructor_arguments == [
        {
            "model_name": "turbo",
            "asr_engine": "auto",
            "device": "auto",
            "device_index": 0,
            "compute_type": "auto",
            "asr_model_path": None,
            "diarization_engine": "community-1",
            "diarization_device": "auto",
            "diarize": True,
            "huggingface_token": None,
            "batch_size": 8,
            "min_speakers": None,
            "max_speakers": None,
            "load_asr": False,
        }
    ]


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
