from __future__ import annotations

from pathlib import Path

import pytest

from broadcastify_cli.quota import (
    ArchiveRequestBudgetExceeded,
    ArchiveRequestLedger,
)


def test_instance_ledger_mints_stable_identity_and_reserves_user_capacity(
    tmp_path: Path,
) -> None:
    now = 1_800_000_000.0
    path = tmp_path / "quota.sqlite3"
    first = ArchiveRequestLedger(path, clock=lambda: now)
    second = ArchiveRequestLedger(path, clock=lambda: now)

    assert first.status()["instance_id"] == second.status()["instance_id"]
    assert first.status()["provider_limit"] == 250
    assert first.status()["automated_limit"] == 240
    assert first.status()["user_reserve"] == 10

    limited_first = ArchiveRequestLedger(
        tmp_path / "shared.sqlite3",
        limit=1,
        provider_limit=2,
        clock=lambda: now,
    )
    limited_second = ArchiveRequestLedger(
        tmp_path / "shared.sqlite3",
        limit=1,
        provider_limit=2,
        clock=lambda: now,
    )
    limited_first.reserve(
        feed_id="90001", archive_date="2026-07-22", archive_id="first"
    )
    with pytest.raises(ArchiveRequestBudgetExceeded):
        limited_second.reserve(
            feed_id="90001", archive_date="2026-07-22", archive_id="second"
        )


def test_rolling_ledger_blocks_at_limit_and_releases_oldest_slot(
    tmp_path: Path,
) -> None:
    now = [1_800_000_000.0]
    ledger = ArchiveRequestLedger(
        tmp_path / "quota.sqlite3",
        limit=3,
        provider_limit=4,
        window_seconds=100,
        clock=lambda: now[0],
    )
    for index in range(3):
        request_id = ledger.reserve(
            feed_id="90001",
            archive_date="2026-07-22",
            archive_id=f"archive-{index}",
        )
        ledger.finish(request_id, outcome="http_200", http_status=200)
        now[0] += 1

    status = ledger.status()
    assert status["used"] == 3
    assert status["remaining"] == 0
    assert status["available"] is False
    assert status["next_request_seconds"] == 97
    with pytest.raises(ArchiveRequestBudgetExceeded, match="rolling 24-hour"):
        ledger.reserve(
            feed_id="90001",
            archive_date="2026-07-22",
            archive_id="blocked",
        )

    now[0] += 97
    assert ledger.status()["remaining"] == 1
    ledger.reserve(
        feed_id="90001",
        archive_date="2026-07-22",
        archive_id="released",
    )


def test_server_429_conservatively_blocks_instance_for_one_window(
    tmp_path: Path,
) -> None:
    now = [1_800_000_000.0]
    ledger = ArchiveRequestLedger(
        tmp_path / "quota.sqlite3",
        limit=3,
        provider_limit=4,
        window_seconds=100,
        clock=lambda: now[0],
    )
    request_id = ledger.reserve(
        feed_id="90001",
        archive_date="2026-07-22",
        archive_id="limited",
    )
    ledger.finish(request_id, outcome="http_429", http_status=429)
    status = ledger.mark_rate_limited("The upstream archive allowance was reached.")

    assert status["blocked"] is True
    assert status["remaining"] == 2
    assert status["next_request_seconds"] == 100
    with pytest.raises(ArchiveRequestBudgetExceeded, match="allowance was reached"):
        ledger.reserve(
            feed_id="90001",
            archive_date="2026-07-22",
            archive_id="still-blocked",
        )

    now[0] += 101
    assert ledger.status()["available"] is True
