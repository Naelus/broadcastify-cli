import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from broadcastify_cli.transcription import (
    LocalTranscriber,
    SpeakerTurn,
    TranscriptWord,
    format_timestamp,
    group_words,
    speaker_for_interval,
)


def test_speaker_uses_largest_overlap_not_first_overlap() -> None:
    turns = [
        SpeakerTurn(0.0, 1.1, "SPEAKER_00"),
        SpeakerTurn(1.1, 5.0, "SPEAKER_01"),
    ]
    assert speaker_for_interval(0.9, 2.5, turns) == "SPEAKER_01"


def test_zero_duration_word_uses_containing_or_nearest_turn() -> None:
    turns = [SpeakerTurn(1.0, 2.0, "SPEAKER_00")]
    assert speaker_for_interval(1.5, 1.5, turns) == "SPEAKER_00"
    assert speaker_for_interval(2.25, 2.25, turns) == "SPEAKER_00"
    assert speaker_for_interval(3.0, 3.0, turns) is None


def test_words_group_only_while_speaker_is_unchanged() -> None:
    words = [
        TranscriptWord(0.0, 0.5, " Dispatch", "SPEAKER_00"),
        TranscriptWord(0.5, 1.0, " calling", "SPEAKER_00"),
        TranscriptWord(1.0, 1.5, " unit", "SPEAKER_01"),
    ]
    grouped = group_words(words)
    assert [segment.speaker for segment in grouped] == ["SPEAKER_00", "SPEAKER_01"]
    assert grouped[0].text == "Dispatch calling"
    assert grouped[1].text == "unit"


def test_timestamp_includes_milliseconds() -> None:
    assert format_timestamp(3661.234) == "01:01:01.234"


def test_compatible_transcript_is_a_cache_hit(tmp_path: Path) -> None:
    audio = tmp_path / "combined.mp3"
    audio.write_bytes(b"audio")
    transcript_dir = tmp_path / "transcripts"
    transcript_dir.mkdir()
    json_path = transcript_dir / "combined.json"
    txt_path = transcript_dir / "combined.txt"
    json_path.write_text(
        json.dumps({"model": "turbo", "segments": [], "diarization_requested": False}),
        encoding="utf-8",
    )
    txt_path.write_text("", encoding="utf-8")

    transcriber = object.__new__(LocalTranscriber)
    transcriber.model_name = "turbo"
    transcriber.diarize = False

    assert transcriber._existing_transcript_is_current(audio, json_path, txt_path)


def test_diarization_turn_cache_is_parameter_and_audio_specific(tmp_path: Path) -> None:
    audio = tmp_path / "combined.mp3"
    audio.write_bytes(b"audio")
    transcriber = object.__new__(LocalTranscriber)
    transcriber.min_speakers = 2
    transcriber.max_speakers = 8
    turns = [SpeakerTurn(1.0, 2.5, "SPEAKER_00")]

    transcriber._save_diarization_cache(audio, turns)
    assert transcriber._load_diarization_cache(audio) == turns

    transcriber.max_speakers = 9
    assert transcriber._load_diarization_cache(audio) is None


@pytest.mark.parametrize(
    ("pipeline_batch_size", "expected_batch_size"),
    [(1, 8), (32, 32)],
)
def test_diarization_reports_inner_pipeline_progress_without_lowering_model_batch(
    monkeypatch,
    tmp_path: Path,
    pipeline_batch_size: int,
    expected_batch_size: int,
) -> None:
    audio = tmp_path / "combined.mp3"
    audio.write_bytes(b"audio")
    prepared = tmp_path / "combined.pyannote.flac"
    prepared.write_bytes(b"prepared audio")
    messages: list[str] = []

    class FakePipeline:
        embedding_batch_size = pipeline_batch_size

        def __call__(self, _path: str, *, hook=None, **_kwargs: object):
            assert hook is not None
            assert self.embedding_batch_size == expected_batch_size
            hook("segmentation", None, file={"uri": "test"}, total=4, completed=1)
            hook("segmentation", None, file={"uri": "test"}, total=4, completed=4)
            hook("embeddings", None)
            return [(SimpleNamespace(start=1.0, end=2.0), "SPEAKER_00")]

    transcriber = object.__new__(LocalTranscriber)
    transcriber._diarization_pipeline = FakePipeline()
    transcriber.batch_size = 8
    transcriber.min_speakers = None
    transcriber.max_speakers = None
    monkeypatch.setattr(
        transcriber, "_prepare_diarization_input", lambda _path: (prepared, True)
    )

    turns = transcriber._diarize(audio, progress=messages.append)

    assert turns == [SpeakerTurn(1.0, 2.0, "SPEAKER_00")]
    assert "Diarization segmentation: 25% (1/4)" in messages
    assert "Diarization segmentation: 100% (4/4)" in messages
    assert "Diarization embeddings" in messages
    assert not prepared.exists()


def test_diarization_keeps_prepared_input_after_pipeline_failure(
    monkeypatch, tmp_path: Path
) -> None:
    audio = tmp_path / "combined.mp3"
    audio.write_bytes(b"audio")
    prepared = tmp_path / "combined.pyannote.flac"
    prepared.write_bytes(b"prepared audio")

    class FailingPipeline:
        embedding_batch_size = 32

        def __call__(self, _path: str, **_kwargs: object):
            raise RuntimeError("diarization failed")

    transcriber = object.__new__(LocalTranscriber)
    transcriber._diarization_pipeline = FailingPipeline()
    transcriber.batch_size = 8
    transcriber.min_speakers = None
    transcriber.max_speakers = None
    monkeypatch.setattr(
        transcriber, "_prepare_diarization_input", lambda _path: (prepared, True)
    )

    with pytest.raises(RuntimeError, match="diarization failed"):
        transcriber._diarize(audio)

    assert prepared.exists()


def test_diarization_reuses_completed_lossless_retry_input(
    monkeypatch, tmp_path: Path
) -> None:
    audio = tmp_path / "combined.mp3"
    audio.write_bytes(b"audio")
    prepared = (
        tmp_path
        / "transcripts"
        / ".cache"
        / "combined.pyannote.flac"
    )
    prepared.parent.mkdir(parents=True)
    prepared.write_bytes(b"lossless audio")
    monkeypatch.setattr(
        "broadcastify_cli.transcription.find_ffmpeg",
        lambda: (_ for _ in ()).throw(AssertionError("FFmpeg should not run")),
    )
    transcriber = object.__new__(LocalTranscriber)

    result, temporary = transcriber._prepare_diarization_input(audio)

    assert result == prepared
    assert temporary is True
