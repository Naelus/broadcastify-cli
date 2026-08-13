from __future__ import annotations

from pathlib import Path

import pytest

from broadcastify_cli.quota import (
    ArchiveRequestBudgetExceeded,
    ArchiveRequestLedger,
    RATE_LIMIT_RELEASE_GRACE_SECONDS,
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


def test_server_429_reopens_when_oldest_known_request_ages_out(
    tmp_path: Path,
) -> None:
    now = [1_800_000_000.0]
    ledger = ArchiveRequestLedger(
        tmp_path / "quota.sqlite3",
        limit=4,
        provider_limit=5,
        window_seconds=100,
        clock=lambda: now[0],
    )
    ledger.reserve(
        feed_id="90001", archive_date="2026-07-22", archive_id="oldest"
    )
    now[0] += 90
    limited = ledger.reserve(
        feed_id="90001", archive_date="2026-07-22", archive_id="limited"
    )
    ledger.finish(limited, outcome="http_429", http_status=429)

    status = ledger.mark_rate_limited("rolling limit")

    assert status["next_request_seconds"] == 15
    now[0] += 14
    assert ledger.status()["available"] is False
    now[0] += 2
    assert ledger.status()["available"] is True


def test_legacy_429_migration_does_not_shift_release_forward(
    tmp_path: Path,
) -> None:
    now = [10.0]
    ledger = ArchiveRequestLedger(
        tmp_path / "quota.sqlite3",
        limit=4,
        provider_limit=5,
        window_seconds=100,
        clock=lambda: now[0],
    )
    first = ledger.reserve(
        feed_id="90001", archive_date="2026-07-22", archive_id="first"
    )
    ledger.finish(first, outcome="http_200", http_status=200)
    now[0] = 30.0
    limited = ledger.reserve(
        feed_id="90001", archive_date="2026-07-22", archive_id="limited"
    )
    ledger.finish(limited, outcome="http_429", http_status=429)

    with ledger._connect() as connection:
        connection.execute(
            """
            UPDATE archive_quota_state
            SET blocked_until = ?, blocked_reason = ?
            WHERE singleton = 1
            """,
            (130.0, "legacy full-window 429 block"),
        )

    now[0] = 40.0
    status = ledger.status()
    assert status["blocked"] is True
    assert status["next_request_seconds"] == int(
        10.0 + 100.0 + RATE_LIMIT_RELEASE_GRACE_SECONDS - 40.0
    )

    now[0] = 116.0
    status = ledger.status()
    assert status["available"] is True
    assert status["blocked"] is False


def test_authorized_accounts_have_independent_limits_in_one_durable_ledger(
    tmp_path: Path,
) -> None:
    now = [1_800_000_000.0]
    path = tmp_path / "quota.sqlite3"
    primary = ArchiveRequestLedger(
        path,
        limit=1,
        provider_limit=2,
        window_seconds=100,
        clock=lambda: now[0],
        account_profile_id="default",
    )
    secondary = ArchiveRequestLedger(
        path,
        limit=1,
        provider_limit=2,
        window_seconds=100,
        clock=lambda: now[0],
        account_profile_id="secondary",
    )

    primary.reserve(
        feed_id="90001",
        archive_date="2026-07-22",
        archive_id="primary-request",
    )

    assert primary.status()["available"] is False
    assert primary.status()["account_profile_id"] == "default"
    assert secondary.status()["available"] is True
    assert secondary.status()["account_profile_id"] == "secondary"
    secondary.reserve(
        feed_id="90001",
        archive_date="2026-07-22",
        archive_id="secondary-request",
    )
    assert secondary.status()["remaining"] == 0


def test_rate_limit_block_is_confined_to_the_account_that_received_it(
    tmp_path: Path,
) -> None:
    path = tmp_path / "quota.sqlite3"
    primary = ArchiveRequestLedger(path, account_profile_id="default")
    secondary = ArchiveRequestLedger(path, account_profile_id="secondary")

    primary.mark_rate_limited("primary account reached its provider boundary")

    assert primary.status()["blocked"] is True
    assert secondary.status()["blocked"] is False
    assert secondary.status()["remaining"] == 240


def test_account_rotation_obeys_one_cross_profile_spacing_gate(
    tmp_path: Path,
) -> None:
    now = [1_800_000_000.0]
    sleeps: list[float] = []

    def advance(delay: float) -> None:
        sleeps.append(delay)
        now[0] += delay

    path = tmp_path / "quota.sqlite3"
    primary = ArchiveRequestLedger(
        path,
        clock=lambda: now[0],
        account_profile_id="default",
        request_spacing_seconds=5,
        sleeper=advance,
    )
    secondary = ArchiveRequestLedger(
        path,
        clock=lambda: now[0],
        account_profile_id="secondary",
        request_spacing_seconds=5,
        sleeper=advance,
    )
    primary.reserve(
        feed_id="90001",
        archive_date="2026-07-22",
        archive_id="primary-request",
    )
    secondary.reserve(
        feed_id="90001",
        archive_date="2026-07-22",
        archive_id="secondary-request",
    )

    assert sleeps == [5.0]
