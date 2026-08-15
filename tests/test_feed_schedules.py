from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from broadcastify_cli.storage import AnalysisStore


def _payload(*, enabled: bool = True) -> dict[str, object]:
    return {
        "feed_id": "90001",
        "feed_name": "Example Public Safety",
        "run_time_local": "02:30",
        "lookback_days": 2,
        "analyze": True,
        "enabled": enabled,
        "job": {
            "output_dir": "archives",
            "combine": True,
            "keep_originals": False,
            "transcribe": True,
            "diarize": True,
            "download_jobs": 8,
        },
    }


def test_feed_schedule_is_specific_persistent_and_forces_safe_acquisition(
    tmp_path: Path,
) -> None:
    database = tmp_path / "analysis.sqlite3"
    now = datetime(2026, 7, 23, 3, 0, tzinfo=timezone(timedelta(hours=-5)))
    with AnalysisStore(database) as store:
        saved = store.save_feed_schedule(_payload())
        schedules = store.list_feed_schedules(now=now)

    assert saved["feed_id"] == "90001"
    assert schedules[0]["due"] is True
    assert schedules[0]["job"]["download_jobs"] == 1
    assert schedules[0]["job"]["keep_originals"] is True
    assert schedules[0]["account_profile_id"] == "automatic"

    with AnalysisStore(database) as store:
        assert store.list_feed_schedules(now=now)[0]["feed_name"] == (
            "Example Public Safety"
        )


def test_existing_schedule_database_adds_historical_and_recurring_catch_up_columns(
    tmp_path: Path,
) -> None:
    database = tmp_path / "analysis.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE feed_schedules (
                id INTEGER PRIMARY KEY,
                feed_id TEXT NOT NULL UNIQUE,
                feed_name TEXT NOT NULL,
                run_time_local TEXT NOT NULL,
                lookback_days INTEGER NOT NULL DEFAULT 2,
                job_json TEXT NOT NULL,
                analyze INTEGER NOT NULL DEFAULT 1,
                enabled INTEGER NOT NULL DEFAULT 1,
                state TEXT NOT NULL DEFAULT 'scheduled',
                message TEXT NOT NULL DEFAULT '',
                last_run_date TEXT NOT NULL DEFAULT '',
                last_started_at TEXT NOT NULL DEFAULT '',
                last_finished_at TEXT NOT NULL DEFAULT '',
                not_before TEXT NOT NULL DEFAULT '',
                lease_until TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

    with AnalysisStore(database) as store:
        columns = {
            str(row["name"])
            for row in store.connection.execute(
                "PRAGMA table_info(feed_schedules)"
            ).fetchall()
        }
        saved = store.save_feed_schedule(
            {**_payload(), "backfill_start_date": "2026-07-03"}
        )

    assert "backfill_start_date" in columns
    assert "recurring_catch_up" in columns
    assert "account_profile_id" in columns
    assert saved["backfill_start_date"] == "2026-07-03"
    assert saved["recurring_catch_up"] is False
    assert saved["account_profile_id"] == "automatic"


def test_schedule_persists_one_explicit_account_profile_without_credentials(
    tmp_path: Path,
) -> None:
    payload = _payload()
    payload["account_profile_id"] = "secondary"

    with AnalysisStore(tmp_path / "analysis.sqlite3") as store:
        saved = store.save_feed_schedule(payload)
        claimed = store.claim_due_feed_schedule(
            now=datetime(
                2026,
                7,
                23,
                3,
                0,
                tzinfo=timezone(timedelta(hours=-5)),
            )
        )

    assert saved["account_profile_id"] == "secondary"
    assert claimed is not None
    assert claimed["account_profile_id"] == "secondary"
    assert "username" not in claimed["job"]
    assert "password" not in claimed["job"]


def test_claim_is_atomic_and_quota_wait_rechecks_retained_work_before_rolling_slot(
    tmp_path: Path,
) -> None:
    database = tmp_path / "analysis.sqlite3"
    now = datetime(2026, 7, 23, 3, 0, tzinfo=timezone(timedelta(hours=-5)))
    with AnalysisStore(database) as first:
        saved = first.save_feed_schedule(_payload())
        claimed = first.claim_due_feed_schedule(now=now)
        assert claimed is not None
        assert claimed["due_date"] == "2026-07-23"
        assert claimed["job"]["start_date"] == "2026-07-22"
        assert claimed["job"]["end_date"] == "2026-07-23"
        assert claimed["job"]["output_dir"] == str(tmp_path.resolve())

    with AnalysisStore(database) as second:
        assert second.claim_due_feed_schedule(now=now) is None
        release = (now + timedelta(minutes=20)).astimezone(timezone.utc)
        waiting = second.finish_feed_schedule(
            int(saved["id"]),
            due_date="2026-07-23",
            status="waiting_quota",
            message="Waiting for one rolling slot.",
            next_request_at=release.isoformat(),
            now=now,
        )
        assert waiting["state"] == "waiting_quota"
        assert second.claim_due_feed_schedule(now=now + timedelta(minutes=4)) is None
        retried = second.claim_due_feed_schedule(now=now + timedelta(minutes=6))
        assert retried is not None
        completed = second.finish_feed_schedule(
            int(saved["id"]),
            due_date="2026-07-23",
            status="complete",
            message="Complete.",
            now=now + timedelta(minutes=22),
        )
        assert completed["last_run_date"] == "2026-07-23"
        assert second.claim_due_feed_schedule(now=now + timedelta(hours=1)) is None


def test_historical_catch_up_survives_quota_wait_and_clears_only_when_complete(
    tmp_path: Path,
) -> None:
    database = tmp_path / "analysis.sqlite3"
    now = datetime(2026, 7, 23, 3, 0, tzinfo=timezone(timedelta(hours=-5)))
    payload = _payload()
    payload["backfill_start_date"] = "2026-07-03"

    with AnalysisStore(database) as store:
        saved = store.save_feed_schedule(payload)
        claimed = store.claim_due_feed_schedule(now=now)
        assert claimed is not None
        assert claimed["backfill_start_date"] == "2026-07-03"
        assert claimed["job"]["start_date"] == "2026-07-03"
        assert claimed["job"]["end_date"] == "2026-07-23"

        release = (now + timedelta(minutes=20)).astimezone(timezone.utc)
        waiting = store.finish_feed_schedule(
            int(saved["id"]),
            due_date="2026-07-23",
            status="waiting_quota",
            next_request_at=release.isoformat(),
            now=now,
        )
        assert waiting["backfill_start_date"] == "2026-07-03"

        assert store.claim_due_feed_schedule(now=now + timedelta(minutes=4)) is None
        retried = store.claim_due_feed_schedule(now=now + timedelta(minutes=6))
        assert retried is not None
        assert retried["job"]["start_date"] == "2026-07-03"
        completed = store.finish_feed_schedule(
            int(saved["id"]),
            due_date="2026-07-23",
            status="complete",
            now=now + timedelta(minutes=22),
        )
        assert completed["backfill_start_date"] == ""


def test_failed_schedule_retries_same_day_after_bounded_backoff(tmp_path: Path) -> None:
    database = tmp_path / "analysis.sqlite3"
    now = datetime(2026, 7, 23, 3, 0, tzinfo=timezone(timedelta(hours=-5)))

    with AnalysisStore(database) as store:
        saved = store.save_feed_schedule(_payload())
        claimed = store.claim_due_feed_schedule(now=now)
        assert claimed is not None

        failed = store.finish_feed_schedule(
            int(saved["id"]),
            due_date="2026-07-23",
            status="failed",
            message="A transient worker failure occurred.",
            now=now,
        )

        assert failed["state"] == "failed"
        assert failed["last_run_date"] == ""
        assert store.claim_due_feed_schedule(now=now + timedelta(minutes=14)) is None
        retried = store.claim_due_feed_schedule(now=now + timedelta(minutes=16))
        assert retried is not None
        assert retried["due_date"] == "2026-07-23"


def test_recurring_catch_up_keeps_boundary_after_success_and_rechecks_next_day(
    tmp_path: Path,
) -> None:
    database = tmp_path / "analysis.sqlite3"
    now = datetime(2026, 7, 23, 3, 0, tzinfo=timezone(timedelta(hours=-5)))
    payload = _payload()
    payload["backfill_start_date"] = "2026-07-03"
    payload["recurring_catch_up"] = True

    with AnalysisStore(database) as store:
        saved = store.save_feed_schedule(payload)
        claimed = store.claim_due_feed_schedule(now=now)
        assert claimed is not None
        assert claimed["recurring_catch_up"] is True
        assert claimed["job"]["start_date"] == "2026-07-03"

        completed = store.finish_feed_schedule(
            int(saved["id"]),
            due_date="2026-07-23",
            status="complete",
            now=now + timedelta(minutes=5),
        )
        assert completed["backfill_start_date"] == "2026-07-03"
        assert completed["recurring_catch_up"] is True
        assert store.claim_due_feed_schedule(now=now + timedelta(hours=1)) is None

        next_run = store.claim_due_feed_schedule(now=now + timedelta(days=1))
        assert next_run is not None
        assert next_run["job"]["start_date"] == "2026-07-03"
        assert next_run["job"]["end_date"] == "2026-07-24"


def test_recurring_catch_up_requires_a_start_date(tmp_path: Path) -> None:
    payload = _payload()
    payload["recurring_catch_up"] = True

    with AnalysisStore(tmp_path / "analysis.sqlite3") as store:
        with pytest.raises(ValueError, match="requires a catch-up start date"):
            store.save_feed_schedule(payload)


def test_schedule_rejects_a_future_historical_catch_up_date(tmp_path: Path) -> None:
    payload = _payload()
    payload["backfill_start_date"] = (datetime.now().date() + timedelta(days=1)).isoformat()

    with AnalysisStore(tmp_path / "analysis.sqlite3") as store:
        try:
            store.save_feed_schedule(payload)
        except ValueError as exc:
            assert "cannot be in the future" in str(exc)
        else:
            raise AssertionError("A future catch-up date should be rejected.")


def test_disabled_schedule_does_not_claim_and_can_be_removed(tmp_path: Path) -> None:
    database = tmp_path / "analysis.sqlite3"
    now = datetime(2026, 7, 23, 3, 0, tzinfo=timezone.utc)
    with AnalysisStore(database) as store:
        saved = store.save_feed_schedule(_payload(enabled=False))
        assert store.claim_due_feed_schedule(now=now) is None
        assert store.delete_feed_schedule(int(saved["id"])) is True
        assert store.list_feed_schedules(now=now) == []


def test_claim_preserves_an_explicit_absolute_library(tmp_path: Path) -> None:
    database = tmp_path / "database" / "analysis.sqlite3"
    library = (tmp_path / "selected library").resolve()
    payload = _payload()
    payload["job"] = {**dict(payload["job"]), "output_dir": str(library)}
    now = datetime(2026, 7, 23, 3, 0, tzinfo=timezone.utc)

    with AnalysisStore(database) as store:
        store.save_feed_schedule(payload)
        claimed = store.claim_due_feed_schedule(now=now)

    assert claimed is not None
    assert claimed["job"]["output_dir"] == str(library)


def test_claim_runtime_library_overrides_stale_absolute_path(tmp_path: Path) -> None:
    database = tmp_path / "database" / "analysis.sqlite3"
    old_library = (tmp_path / "old library").resolve()
    selected_library = (tmp_path / "selected library").resolve()
    payload = _payload()
    payload["job"] = {
        **dict(payload["job"]),
        "output_dir": str(old_library),
    }
    now = datetime(2026, 7, 23, 3, 0, tzinfo=timezone.utc)

    with AnalysisStore(database) as store:
        store.save_feed_schedule(payload)
        claimed = store.claim_due_feed_schedule(
            now=now,
            output_dir=selected_library,
        )

    assert claimed is not None
    assert claimed["job"]["output_dir"] == str(selected_library)


def test_existing_schedule_can_update_timing_processing_and_enabled_state(
    tmp_path: Path,
) -> None:
    database = tmp_path / "analysis.sqlite3"
    now = datetime(2026, 7, 23, 3, 0, tzinfo=timezone.utc)
    with AnalysisStore(database) as store:
        original = store.save_feed_schedule(_payload())
        updated_payload = _payload(enabled=False)
        updated_payload["run_time_local"] = "04:15"
        updated_payload["lookback_days"] = 5
        updated_payload["analyze"] = False
        updated_payload["job"] = {
            **dict(updated_payload["job"]),
            "transcribe": False,
            "diarize": False,
            "model": "base",
        }
        updated = store.save_feed_schedule(updated_payload)
        schedules = store.list_feed_schedules(now=now)

    assert updated["id"] == original["id"]
    assert len(schedules) == 1
    assert schedules[0]["run_time_local"] == "04:15"
    assert schedules[0]["lookback_days"] == 5
    assert schedules[0]["enabled"] is False
    assert schedules[0]["analyze"] is False
    assert schedules[0]["job"]["transcribe"] is False
    assert schedules[0]["job"]["diarize"] is False
    assert schedules[0]["job"]["model"] == "base"
    assert schedules[0]["job"]["download_jobs"] == 1
    assert schedules[0]["job"]["keep_originals"] is True


def test_startup_recovery_releases_interrupted_schedule(tmp_path: Path) -> None:
    database = tmp_path / "analysis.sqlite3"
    now = datetime(2026, 7, 23, 3, 0, tzinfo=timezone.utc)
    with AnalysisStore(database) as store:
        store.save_feed_schedule(_payload())
        assert store.claim_due_feed_schedule(now=now) is not None
        assert store.recover_feed_schedules(now=now + timedelta(minutes=2)) == 1
        recovered = store.list_feed_schedules(now=now + timedelta(minutes=2))[0]
        assert recovered["state"] == "deferred"
        assert "resuming from retained work" in recovered["message"]
        assert recovered["last_run_date"] == ""
        assert store.claim_due_feed_schedule(now=now + timedelta(minutes=2)) is None
        assert store.claim_due_feed_schedule(now=now + timedelta(minutes=4)) is not None


def test_startup_recovery_rechecks_quota_paused_local_work(tmp_path: Path) -> None:
    database = tmp_path / "analysis.sqlite3"
    now = datetime(2026, 7, 23, 3, 0, tzinfo=timezone.utc)
    with AnalysisStore(database) as store:
        store.save_feed_schedule(_payload())
        claimed = store.claim_due_feed_schedule(now=now)
        assert claimed is not None
        waiting = store.finish_feed_schedule(
            int(claimed["id"]),
            due_date=str(claimed["due_date"]),
            status="waiting_quota",
            message="Waiting for the next rolling archive-request slot.",
            next_request_at=(now + timedelta(hours=20)).isoformat(),
            now=now + timedelta(minutes=1),
        )
        assert waiting["state"] == "waiting_quota"
        assert waiting["due"] is False

        assert store.recover_feed_schedules(now=now + timedelta(minutes=2)) == 1
        recovered = store.list_feed_schedules(now=now + timedelta(minutes=2))[0]
        assert recovered["state"] == "deferred"
        assert "retained local work" in recovered["message"]
        assert store.claim_due_feed_schedule(now=now + timedelta(minutes=2)) is None
        assert store.claim_due_feed_schedule(now=now + timedelta(minutes=4)) is not None
