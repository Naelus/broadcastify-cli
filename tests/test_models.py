from datetime import date

import pytest

from broadcastify_cli.models import JobRequest


def test_date_range_is_inclusive_and_newest_first_by_default() -> None:
    request = JobRequest(
        feed_id="5318",
        start_date=date(2026, 7, 10),
        end_date=date(2026, 7, 12),
    )
    assert list(request.dates()) == [
        date(2026, 7, 12),
        date(2026, 7, 11),
        date(2026, 7, 10),
    ]


def test_date_range_can_explicitly_use_chronological_order() -> None:
    request = JobRequest(
        feed_id="5318",
        start_date=date(2026, 7, 10),
        end_date=date(2026, 7, 12),
        newest_first=False,
    )
    assert list(request.dates()) == [
        date(2026, 7, 10),
        date(2026, 7, 11),
        date(2026, 7, 12),
    ]


def test_job_request_preserves_bounded_feed_name() -> None:
    request = JobRequest.from_dict(
        {
            "feed_id": "5318",
            "feed_name": f"  {'A' * 240}  ",
            "start_date": "2026-07-10",
            "end_date": "2026-07-10",
        }
    )

    assert request.feed_name == "A" * 200


def test_job_request_parses_optional_processing_day_limit() -> None:
    request = JobRequest.from_dict(
        {
            "feed_id": "5318",
            "start_date": "2026-07-10",
            "end_date": "2026-07-10",
            "max_processing_days": "1",
        }
    )

    assert request.max_processing_days == 1


def test_processing_day_limit_must_be_positive() -> None:
    request = JobRequest(
        feed_id="5318",
        start_date=date(2026, 7, 10),
        end_date=date(2026, 7, 10),
        max_processing_days=0,
    )

    with pytest.raises(ValueError, match="processing days"):
        request.validate()


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
        ("whisper.cpp", "metal"),
        ("openvino", "openvino-gpu"),
        ("windows-ml", "windows-ml"),
        ("qwen3-asr", "cpu"),
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
