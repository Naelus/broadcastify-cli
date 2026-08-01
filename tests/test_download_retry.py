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
from broadcastify_cli.archive_cache import (
    cached_archive_for_id,
    reconcile_complete_legacy_day,
    remember_archive_identity,
)
from broadcastify_cli.quota import ArchiveRequestLedger


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


def _ledger(tmp_path: Path, *, limit: int = 240) -> ArchiveRequestLedger:
    return ArchiveRequestLedger(
        tmp_path / "archive-quota.sqlite3",
        limit=limit,
        provider_limit=max(250, limit + 10),
    )


def test_download_archive_stops_on_any_429_without_retry(tmp_path: Path) -> None:
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
        quota_ledger=_ledger(tmp_path),
    )
    client.session = session  # type: ignore[assignment]

    with pytest.raises(DownloadLimitExceeded, match="HTTP 429"):
        client.download_archive(
            "90001",
            date(2026, 7, 12),
            "test-archive",
            tmp_path,
            notice=notices.append,
        )

    assert len(session.calls) == 1
    assert len(session.responses) == 1
    assert any("without retrying" in message for message in notices)
    assert client.archive_quota_status()["blocked"] is True


def test_download_archive_stops_after_configured_attempts(tmp_path: Path) -> None:
    session = _QueuedSession(
        [_response(503, headers={"Retry-After": "0"}) for _ in range(3)]
    )
    client = BroadcastifyClient(
        download_attempts=3,
        download_request_interval=0,
        download_backoff_base=0,
        download_backoff_max=0,
        random_uniform=lambda _start, _end: 0,
        quota_ledger=_ledger(tmp_path),
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
        quota_ledger=_ledger(tmp_path),
    )
    client.session = session  # type: ignore[assignment]

    with pytest.raises(DownloadLimitExceeded, match="limit is exhausted"):
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


def test_local_budget_blocks_before_an_extra_archive_request(tmp_path: Path) -> None:
    session = _QueuedSession(
        [
            _response(
                200,
                headers={
                    "Content-Type": "audio/mpeg",
                    "Content-Disposition": 'attachment; filename="first.mp3"',
                },
                content=b"audio",
            )
        ]
    )
    client = BroadcastifyClient(
        download_request_interval=0,
        quota_ledger=_ledger(tmp_path, limit=1),
    )
    client.session = session  # type: ignore[assignment]

    client.download_archive("90001", date(2026, 7, 12), "first", tmp_path)
    with pytest.raises(DownloadLimitExceeded, match="used its 1 automated"):
        client.download_archive("90001", date(2026, 7, 12), "second", tmp_path)

    assert len(session.calls) == 1
    assert client.archive_quota_status()["remaining"] == 0


def test_cached_archive_does_not_consume_local_budget(tmp_path: Path) -> None:
    cached = tmp_path / "existing.mp3"
    cached.write_bytes(b"audio")
    client = BroadcastifyClient(quota_ledger=_ledger(tmp_path, limit=1))

    result = client.download_archive(
        "90001", date(2026, 7, 12), "existing", tmp_path
    )

    assert result == cached
    assert client.archive_quota_status()["used"] == 0


def test_cached_archive_allows_one_minute_filename_boundary(
    tmp_path: Path,
) -> None:
    day_dir = tmp_path / "40590" / "20260719"
    day_dir.mkdir(parents=True)
    cached = day_dir / "202607191900-447318-40590.mp3"
    cached.write_bytes(b"audio")
    client = BroadcastifyClient(quota_ledger=_ledger(tmp_path, limit=1))
    client._archive_filename_prefixes["40590-1784502071"] = "202607191901"

    result = client.download_archive(
        "40590",
        date(2026, 7, 19),
        "40590-1784502071",
        day_dir,
    )

    assert result == cached
    assert client.archive_quota_status()["used"] == 0


def test_cached_archive_allows_observed_one_hour_filename_displacement(
    tmp_path: Path,
) -> None:
    day_dir = tmp_path / "40590" / "20260719"
    day_dir.mkdir(parents=True)
    cached = day_dir / "202607192259-888106-40590.mp3"
    cached.write_bytes(b"audio")
    client = BroadcastifyClient(quota_ledger=_ledger(tmp_path, limit=1))
    client._archive_filename_prefixes["40590-1784519971"] = "202607192359"

    result = client.download_archive(
        "40590",
        date(2026, 7, 19),
        "40590-1784519971",
        day_dir,
    )

    assert result == cached
    assert client.archive_quota_status()["used"] == 0


def test_download_archive_persists_exact_provider_identity(tmp_path: Path) -> None:
    day_dir = tmp_path / "90001" / "20260712"
    day_dir.mkdir(parents=True)
    session = _QueuedSession(
        [
            _response(
                200,
                headers={
                    "Content-Type": "audio/mpeg",
                    "Content-Disposition": (
                        'attachment; filename="202607120035-777-90001.mp3"'
                    ),
                },
                content=b"audio",
            )
        ]
    )
    first = BroadcastifyClient(
        download_request_interval=0,
        quota_ledger=_ledger(tmp_path / "first", limit=1),
    )
    first.session = session  # type: ignore[assignment]
    first._archive_filename_prefixes["90001-opaque"] = "202607120001"

    downloaded = first.download_archive(
        "90001",
        date(2026, 7, 12),
        "90001-opaque",
        day_dir,
    )

    assert downloaded.name == "202607120035-777-90001.mp3"
    assert (day_dir / ".broadcastify-archive-index.json").is_file()

    second_session = _QueuedSession([])
    second = BroadcastifyClient(
        download_request_interval=0,
        quota_ledger=_ledger(tmp_path / "second", limit=1),
    )
    second.session = second_session  # type: ignore[assignment]
    # A changed listing timestamp must not matter once the exact provider ID
    # has been retained beside the source file.
    second._archive_filename_prefixes["90001-opaque"] = "202607121200"

    reused = second.download_archive(
        "90001",
        date(2026, 7, 12),
        "90001-opaque",
        day_dir,
    )

    assert reused == downloaded
    assert second_session.calls == []
    assert second.archive_quota_status()["used"] == 0


def test_provider_confirmed_archive_aliases_reuse_one_retained_file(
    tmp_path: Path,
) -> None:
    day_dir = tmp_path / "90001" / "20260712"
    day_dir.mkdir(parents=True)
    shared_filename = "202607120035-777-90001.mp3"
    session = _QueuedSession(
        [
            _response(
                200,
                headers={
                    "Content-Type": "audio/mpeg",
                    "Content-Disposition": f'attachment; filename="{shared_filename}"',
                },
                content=b"same audio",
            ),
            _response(
                200,
                headers={
                    "Content-Type": "audio/mpeg",
                    "Content-Disposition": f'attachment; filename="{shared_filename}"',
                },
                content=b"same audio",
            ),
        ]
    )
    first = BroadcastifyClient(
        download_request_interval=0,
        quota_ledger=_ledger(tmp_path / "first", limit=2),
    )
    first.session = session  # type: ignore[assignment]
    first._archive_filename_prefixes.update(
        {
            "provider-first": "202607120000",
            "provider-alias": "202607121500",
        }
    )

    initial = first.download_archive(
        "90001", date(2026, 7, 12), "provider-first", day_dir
    )
    alias = first.download_archive(
        "90001", date(2026, 7, 12), "provider-alias", day_dir
    )

    assert initial == alias == day_dir / shared_filename
    assert len(session.calls) == 2
    assert cached_archive_for_id(day_dir, "90001", "provider-first") == initial
    assert cached_archive_for_id(day_dir, "90001", "provider-alias") == initial

    no_network = _QueuedSession([])
    second = BroadcastifyClient(
        download_request_interval=0,
        quota_ledger=_ledger(tmp_path / "second", limit=2),
    )
    second.session = no_network  # type: ignore[assignment]
    assert second.download_archive(
        "90001", date(2026, 7, 12), "provider-first", day_dir
    ) == initial
    assert second.download_archive(
        "90001", date(2026, 7, 12), "provider-alias", day_dir
    ) == initial
    assert no_network.calls == []


def test_exact_identity_prevents_one_file_from_satisfying_two_archives(
    tmp_path: Path,
) -> None:
    day_dir = tmp_path / "90001" / "20260712"
    day_dir.mkdir(parents=True)
    cached = day_dir / "202607120035-777-90001.mp3"
    cached.write_bytes(b"audio")

    remember_archive_identity(
        day_dir,
        "90001",
        date(2026, 7, 12),
        "first-provider-id",
        cached,
        listing_prefix="202607120001",
    )
    client = BroadcastifyClient(quota_ledger=_ledger(tmp_path, limit=1))

    assert (
        client._existing_archive(
            day_dir,
            "90001",
            "second-provider-id",
            "202607120035",
        )
        is None
    )


def test_exact_identity_rejects_changed_or_unsafe_cached_file(tmp_path: Path) -> None:
    day_dir = tmp_path / "90001" / "20260712"
    day_dir.mkdir(parents=True)
    cached = day_dir / "202607120035-777-90001.mp3"
    cached.write_bytes(b"audio")
    remember_archive_identity(
        day_dir,
        "90001",
        date(2026, 7, 12),
        "exact-provider-id",
        cached,
        listing_prefix="202607120001",
    )

    cached.write_bytes(b"changed")
    assert cached_archive_for_id(day_dir, "90001", "exact-provider-id") is None
    client = BroadcastifyClient(quota_ledger=_ledger(tmp_path, limit=1))
    assert (
        client._existing_archive(
            day_dir,
            "90001",
            "exact-provider-id",
            "202607120035",
        )
        is None
    )

    cached.write_bytes(b"audio")
    index = day_dir / ".broadcastify-archive-index.json"
    index.write_text(
        '{"schema_version":1,"feed_id":"90001","archive_date":"2026-07-12",'
        '"archives":{"exact-provider-id":{"filename":"../outside.mp3",'
        '"listing_prefix":"202607120001","size":5}}}',
        encoding="utf-8",
    )
    assert cached_archive_for_id(day_dir, "90001", "exact-provider-id") is None


def test_complete_legacy_day_reconciles_large_filename_clock_drift(
    tmp_path: Path,
) -> None:
    day_dir = tmp_path / "90001" / "20260712"
    day_dir.mkdir(parents=True)
    first = day_dir / "202607120035-111-90001.mp3"
    second = day_dir / "202607120106-222-90001.mp3"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    added = reconcile_complete_legacy_day(
        day_dir,
        "90001",
        date(2026, 7, 12),
        ["provider-second", "provider-first", "provider-second"],
        {
            "provider-first": "202607120000",
            "provider-second": "202607120030",
        },
    )

    assert added == 2
    assert cached_archive_for_id(day_dir, "90001", "provider-first") == first
    assert cached_archive_for_id(day_dir, "90001", "provider-second") == second


def test_complete_legacy_day_refuses_partial_or_conflicting_migration(
    tmp_path: Path,
) -> None:
    day_dir = tmp_path / "90001" / "20260712"
    day_dir.mkdir(parents=True)
    first = day_dir / "202607120035-111-90001.mp3"
    first.write_bytes(b"first")
    identities = {
        "provider-first": "202607120000",
        "provider-second": "202607120030",
    }

    assert (
        reconcile_complete_legacy_day(
            day_dir,
            "90001",
            date(2026, 7, 12),
            list(identities),
            identities,
        )
        == 0
    )

    second = day_dir / "202607120106-222-90001.mp3"
    second.write_bytes(b"second")
    remember_archive_identity(
        day_dir,
        "90001",
        date(2026, 7, 12),
        "provider-first",
        second,
        listing_prefix="202607120000",
    )
    assert (
        reconcile_complete_legacy_day(
            day_dir,
            "90001",
            date(2026, 7, 12),
            list(identities),
            identities,
        )
        == 0
    )
    assert cached_archive_for_id(day_dir, "90001", "provider-first") == second


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


def test_download_day_returns_unique_paths_when_archive_ids_share_file(
    tmp_path: Path,
) -> None:
    client = BroadcastifyClient(download_request_interval=0)
    client.authenticate = lambda force=False: None  # type: ignore[method-assign]
    client.get_archive_ids = lambda feed_id, archive_date: [  # type: ignore[method-assign]
        "first",
        "second",
    ]

    def fake_download(
        feed_id: str,
        archive_date: date,
        archive_id: str,
        target: Path,
        *_: object,
        **__: object,
    ) -> Path:
        result = target / "shared.mp3"
        result.write_bytes(b"audio")
        return result

    client.download_archive = fake_download  # type: ignore[method-assign]

    result = client.download_day("90001", date(2026, 7, 12), tmp_path)

    assert result == [tmp_path / "90001" / "20260712" / "shared.mp3"]


def test_download_day_deduplicates_repeated_listing_ids(tmp_path: Path) -> None:
    client = BroadcastifyClient(download_request_interval=0)
    client.authenticate = lambda force=False: None  # type: ignore[method-assign]
    client.get_archive_ids = lambda feed_id, archive_date: [  # type: ignore[method-assign]
        "current",
        "current",
        "previous",
    ]
    calls: list[str] = []
    totals: list[int] = []

    def fake_download(
        feed_id: str,
        archive_date: date,
        archive_id: str,
        target: Path,
        *_: object,
        **__: object,
    ) -> Path:
        calls.append(archive_id)
        result = target / f"{archive_id}.mp3"
        result.write_bytes(b"audio")
        return result

    client.download_archive = fake_download  # type: ignore[method-assign]

    client.download_day(
        "90001",
        date(2026, 7, 12),
        tmp_path,
        progress=lambda _current, total, _message: totals.append(total),
    )

    assert calls == ["current", "previous"]
    assert set(totals) == {2}


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
