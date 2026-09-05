from datetime import date

import pytest

from broadcastify_cli.models import JobRequest


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
