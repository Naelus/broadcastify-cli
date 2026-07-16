from datetime import date

import pytest

from broadcastify_cli.models import JobRequest


def test_date_range_is_inclusive() -> None:
    request = JobRequest(
        feed_id="5318",
        start_date=date(2026, 7, 10),
        end_date=date(2026, 7, 12),
    )
    assert list(request.dates()) == [
        date(2026, 7, 10),
        date(2026, 7, 11),
        date(2026, 7, 12),
    ]


def test_diarization_requires_transcription() -> None:
    request = JobRequest(
        feed_id="5318",
        start_date=date(2026, 7, 10),
        end_date=date(2026, 7, 10),
        diarize=True,
        transcribe=False,
    )
    with pytest.raises(ValueError, match="requires transcription"):
        request.validate()


def test_feed_id_must_be_numeric() -> None:
    request = JobRequest(
        feed_id="Dallas",
        start_date=date(2026, 7, 10),
        end_date=date(2026, 7, 10),
    )
    with pytest.raises(ValueError, match="digits"):
        request.validate()


def test_speaker_range_must_be_ordered() -> None:
    request = JobRequest(
        feed_id="5318",
        start_date=date(2026, 7, 10),
        end_date=date(2026, 7, 10),
        transcribe=True,
        diarize=True,
        combine=True,
        min_speakers=5,
        max_speakers=2,
    )
    with pytest.raises(ValueError, match="cannot exceed"):
        request.validate()


def test_diarization_requires_daily_combination() -> None:
    request = JobRequest(
        feed_id="5318",
        start_date=date(2026, 7, 10),
        end_date=date(2026, 7, 10),
        transcribe=True,
        diarize=True,
        combine=False,
    )
    with pytest.raises(ValueError, match="daily combination"):
        request.validate()


def test_non_cuda_transcription_profiles_are_valid() -> None:
    for engine, device in [
        ("whisper.cpp", "vulkan"),
        ("openvino", "openvino-gpu"),
        ("windows-ml", "windows-ml"),
        ("faster-whisper", "cpu"),
    ]:
        request = JobRequest(
            feed_id="5318",
            start_date=date(2026, 7, 10),
            end_date=date(2026, 7, 10),
            asr_engine=engine,
            device=device,
        )
        request.validate()
