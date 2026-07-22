from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path

import pytest
import requests

from broadcastify_cli.broadcastify import (
    BroadcastifyClient,
    BroadcastifyError,
    DownloadLimitExceeded,
)


class _QueuedSession:
    def __init__(self, responses: list[requests.Response]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def get(self, url: str, **_: object) -> requests.Response:
        self.calls.append(url)
        return self.responses.pop(0)


def _response(
    status: int,
    *,
    headers: dict[str, str] | None = None,
    content: bytes = b"",
) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response.url = "https://www.broadcastify.com/archives/download/test-archive"
    response.headers.update(headers or {})
    response._content = content
    response._content_consumed = True
    return response


def test_download_archive_retries_429_then_succeeds(tmp_path: Path) -> None:
    session = _QueuedSession(
        [
            _response(429, headers={"Retry-After": "0"}),
            _response(
                200,
                headers={
                    "Content-Type": "audio/mpeg",
                    "Content-Disposition": (
                        'attachment; filename="202607120000-test-archive-90001.mp3"'
                    ),
                },
                content=b"audio",
            ),
        ]
    )
    notices: list[str] = []
    client = BroadcastifyClient(
        download_attempts=3,
        download_request_interval=0,
        download_backoff_base=0,
        download_backoff_max=0,
        random_uniform=lambda _start, _end: 0,
    )
    client.session = session  # type: ignore[assignment]

    result = client.download_archive(
        "90001",
        date(2026, 7, 12),
        "test-archive",
        tmp_path,
        notice=notices.append,
    )

    assert result.read_bytes() == b"audio"
    assert len(session.calls) == 2
    assert any("HTTP 429" in message for message in notices)
    assert any("one at a time" in message for message in notices)


def test_download_archive_stops_after_configured_attempts(tmp_path: Path) -> None:
    session = _QueuedSession(
        [_response(429, headers={"Retry-After": "0"}) for _ in range(3)]
    )
    client = BroadcastifyClient(
        download_attempts=3,
        download_request_interval=0,
        download_backoff_base=0,
        download_backoff_max=0,
        random_uniform=lambda _start, _end: 0,
    )
    client.session = session  # type: ignore[assignment]

    with pytest.raises(requests.HTTPError):
        client.download_archive(
            "90001", date(2026, 7, 12), "test-archive", tmp_path
        )

    assert len(session.calls) == 3


def test_download_archive_does_not_retry_exhausted_quota(tmp_path: Path) -> None:
    session = _QueuedSession(
        [
            _response(
                429,
                headers={"Content-Type": "text/html; charset=UTF-8"},
                content=(
                    b"Download limit exceeded - contact support@broadcastify.com "
                    b"for more details."
                ),
            )
        ]
    )
    notices: list[str] = []
    client = BroadcastifyClient(
        download_attempts=7,
        download_request_interval=0,
        random_uniform=lambda _start, _end: 0,
    )
    client.session = session  # type: ignore[assignment]

    with pytest.raises(DownloadLimitExceeded, match="quota is exhausted"):
        client.download_archive(
            "90003",
            date(2026, 7, 3),
            "90003-1783072731",
            tmp_path,
            notice=notices.append,
        )

    assert len(session.calls) == 1
    assert any("without retrying" in message for message in notices)


def test_retry_after_supports_seconds_and_http_dates() -> None:
    now = datetime(2026, 7, 14, 12, 0, tzinfo=timezone.utc)
    retry_date = format_datetime(now + timedelta(seconds=45), usegmt=True)

    assert BroadcastifyClient._retry_after_seconds("12", now=now) == 12
    assert BroadcastifyClient._retry_after_seconds(retry_date, now=now) == 45
    assert BroadcastifyClient._retry_after_seconds("not-a-date", now=now) is None


def test_429_without_retry_after_uses_long_rate_limit_backoff() -> None:
    client = BroadcastifyClient(random_uniform=lambda _start, _end: 0)
    response = _response(429)

    assert client._download_retry_delay(response, 1) == 30
    assert client._download_retry_delay(response, 4) == 240
    assert client._download_retry_delay(response, 7) == 300


def test_serialized_throttle_is_reused_across_days_in_one_job() -> None:
    client = BroadcastifyClient()
    first_day = client._shared_download_throttle(4)
    first_day.defer(0, serialize=True)

    second_day = client._shared_download_throttle(8)

    assert second_day is first_day
    assert second_day.serialized


def test_download_archive_requires_live_shared_lease_before_media_request(
    tmp_path: Path,
) -> None:
    client = BroadcastifyClient()
    day = tmp_path / "90001" / "20260712"
    day.mkdir(parents=True)
    requests_made = 0

    def forbidden_get(*_args: object, **_kwargs: object) -> None:
        nonlocal requests_made
        requests_made += 1
        raise AssertionError("The archive request must not start.")

    def lost_lease() -> None:
        raise RuntimeError("lease lost")

    client.session.get = forbidden_get  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="lease lost"):
        client.download_archive(
            "90001",
            date(2026, 7, 12),
            "123456",
            day,
            admit_download=lost_lease,
        )

    assert requests_made == 0


def test_download_day_does_not_count_failed_archives_as_downloaded(
    tmp_path: Path,
) -> None:
    client = BroadcastifyClient()
    client.authenticate = lambda force=False: None  # type: ignore[method-assign]
    client.get_archive_ids = lambda feed_id, archive_date: [  # type: ignore[method-assign]
        "first",
        "second",
        "failed",
    ]

    def fake_download(
        feed_id: str,
        archive_date: date,
        archive_id: str,
        day_dir: Path,
        *_: object,
        **__: object,
    ) -> Path:
        if archive_id == "failed":
            raise requests.HTTPError("429 Too Many Requests")
        result = day_dir / f"{archive_id}.mp3"
        result.write_bytes(b"audio")
        return result

    client.download_archive = fake_download  # type: ignore[method-assign]
    progress: list[tuple[int, int, str]] = []

    with pytest.raises(BroadcastifyError, match="failed"):
        client.download_day(
            "90001",
            date(2026, 7, 12),
            tmp_path,
            jobs=2,
            progress=lambda current, total, message: progress.append(
                (current, total, message)
            ),
        )

    assert max(current for current, _total, _message in progress) == 2
    assert all(
        message != "Ready 3/3 (cached or downloaded)"
        for _current, _total, message in progress
    )


def test_download_day_deduplicates_shared_quota_failure(tmp_path: Path) -> None:
    client = BroadcastifyClient()
    client.authenticate = lambda force=False: None  # type: ignore[method-assign]
    client.get_archive_ids = lambda feed_id, archive_date: [  # type: ignore[method-assign]
        "cached",
        "blocked-one",
        "blocked-two",
    ]

    def fake_download(
        feed_id: str,
        archive_date: date,
        archive_id: str,
        day_dir: Path,
        *_: object,
        **__: object,
    ) -> Path:
        if archive_id != "cached":
            raise DownloadLimitExceeded("archive quota exhausted")
        result = day_dir / "cached.mp3"
        result.write_bytes(b"audio")
        return result

    client.download_archive = fake_download  # type: ignore[method-assign]

    with pytest.raises(DownloadLimitExceeded) as raised:
        client.download_day("90003", date(2026, 7, 3), tmp_path, jobs=2)

    assert str(raised.value).count("archive quota exhausted") == 1


def test_download_day_progress_distinguishes_cache_from_website_download(
    tmp_path: Path,
) -> None:
    client = BroadcastifyClient(download_request_interval=0)
    client.authenticate = lambda force=False: None  # type: ignore[method-assign]
    client.get_archive_ids = lambda feed_id, archive_date: [  # type: ignore[method-assign]
        "cached",
        "fresh",
    ]
    day_dir = tmp_path / "90001" / "20260712"
    day_dir.mkdir(parents=True)
    (day_dir / "cached.mp3").write_bytes(b"cached audio")

    def fake_download(
        feed_id: str,
        archive_date: date,
        archive_id: str,
        target: Path,
        *_: object,
        **__: object,
    ) -> Path:
        result = target / f"{archive_id}.mp3"
        if not result.exists():
            result.write_bytes(b"downloaded audio")
        return result

    client.download_archive = fake_download  # type: ignore[method-assign]
    messages: list[str] = []

    client.download_day(
        "90001",
        date(2026, 7, 12),
        tmp_path,
        progress=lambda _current, _total, message: messages.append(message),
    )

    assert "Ready 1/2 — cached locally: cached.mp3" in messages
    assert "Ready 2/2 — downloaded from Broadcastify: fresh.mp3" in messages
    assert not any("cached or downloaded" in message for message in messages)


def test_download_day_acquires_current_and_previous_before_older_backlog(
    tmp_path: Path,
) -> None:
    client = BroadcastifyClient(download_request_interval=0)
    client.authenticate = lambda force=False: None  # type: ignore[method-assign]
    client.get_archive_ids = lambda feed_id, archive_date: [  # type: ignore[method-assign]
        "current",
        "previous",
        "older-one",
        "older-two",
    ]
    calls: list[str] = []

    def fake_download(
        feed_id: str,
        archive_date: date,
        archive_id: str,
        day_dir: Path,
        *_: object,
        **__: object,
    ) -> Path:
        calls.append(archive_id)
        result = day_dir / f"{archive_id}.mp3"
        result.write_bytes(b"audio")
        return result

    client.download_archive = fake_download  # type: ignore[method-assign]

    client.download_day(
        "90001",
        date(2026, 7, 12),
        tmp_path,
        jobs=2,
    )

    assert calls[:2] == ["current", "previous"]
    assert set(calls[2:]) == {"older-one", "older-two"}


def test_download_day_refreshes_a_growing_current_day_once(
    tmp_path: Path,
) -> None:
    client = BroadcastifyClient(download_request_interval=0)
    client.authenticate = lambda force=False: None  # type: ignore[method-assign]
    listings = iter(
        (
            ["current", "previous"],
            ["new-current", "current", "previous"],
        )
    )
    client.get_archive_ids = (  # type: ignore[method-assign]
        lambda feed_id, archive_date: next(listings)
    )
    client._is_current_archive_date = (  # type: ignore[method-assign]
        lambda feed_id, archive_date: True
    )
    calls: list[str] = []

    def fake_download(
        feed_id: str,
        archive_date: date,
        archive_id: str,
        day_dir: Path,
        *_: object,
        **__: object,
    ) -> Path:
        calls.append(archive_id)
        result = day_dir / f"{archive_id}.mp3"
        result.write_bytes(b"audio")
        return result

    client.download_archive = fake_download  # type: ignore[method-assign]

    downloaded = client.download_day(
        "90001",
        date(2026, 7, 20),
        tmp_path,
        jobs=2,
    )

    assert calls == ["current", "previous", "new-current"]
    assert {path.stem for path in downloaded} == {
        "current",
        "previous",
        "new-current",
    }
