from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

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

    with AnalysisStore(database) as store:
        assert store.list_feed_schedules(now=now)[0]["feed_name"] == (
            "Example Public Safety"
        )


def test_claim_is_atomic_and_quota_wait_reopens_at_next_rolling_slot(
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
        assert second.claim_due_feed_schedule(now=now + timedelta(minutes=19)) is None
        retried = second.claim_due_feed_schedule(now=now + timedelta(minutes=21))
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


def test_disabled_schedule_does_not_claim_and_can_be_removed(tmp_path: Path) -> None:
    database = tmp_path / "analysis.sqlite3"
    now = datetime(2026, 7, 23, 3, 0, tzinfo=timezone.utc)
    with AnalysisStore(database) as store:
        saved = store.save_feed_schedule(_payload(enabled=False))
        assert store.claim_due_feed_schedule(now=now) is None
        assert store.delete_feed_schedule(int(saved["id"])) is True
        assert store.list_feed_schedules(now=now) == []


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
