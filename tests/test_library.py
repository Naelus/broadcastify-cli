import json
import os
import time
from datetime import date
from pathlib import Path

import pytest

from broadcastify_cli.analysis import PROMPT_VERSION
from broadcastify_cli.archive_cache import (
    remember_archive_identity,
    remember_complete_archive_day,
)
from broadcastify_cli.library import (
    LocalProcessingRequest,
    build_archive_question_coverage,
    build_library_feed_coverage,
    build_library_resume_plan,
    cleanup_pending_library_deletions,
    compact_archive_date_ranges,
    completed_library_catchup_feed_ids,
    delete_local_library_feed,
    describe_archive_date_ranges,
    entire_archive_feed_range,
    prepare_local_day,
    scan_local_library,
    transcript_satisfies_diarization,
)
from broadcastify_cli.storage import AnalysisStore
from broadcastify_cli.transcription import LocalTranscriber, SpeakerTurn
from broadcastify_cli.workfiles import work_file_owner_token


def _day(tmp_path: Path, feed_id: str, value: str) -> Path:
    result = tmp_path / feed_id / value.replace("-", "")
    result.mkdir(parents=True)
    return result


def test_archive_question_coverage_distinguishes_ready_processing_and_missing_days() -> None:
    coverage = build_archive_question_coverage(
        [
            {
                "feed_id": "90001",
                "archive_date": "2026-07-01",
                "has_combined": True,
                "has_transcript": True,
                "has_imported_transcript": True,
                "has_analysis": True,
            },
            {
                "feed_id": "90001",
                "archive_date": "2026-07-02",
                "has_combined": True,
                "has_transcript": False,
                "has_imported_transcript": False,
                "has_analysis": False,
            },
            {
                "feed_id": "90001",
                "archive_date": "2026-07-03",
                "has_combined": True,
                "has_transcript": True,
                "has_imported_transcript": True,
                "has_analysis": False,
            },
            {
                "feed_id": "90002",
                "archive_date": "2026-07-04",
                "has_combined": True,
                "has_transcript": True,
                "has_imported_transcript": True,
                "has_analysis": True,
            },
        ],
        "90001",
        date(2026, 7, 1),
        date(2026, 7, 4),
    )

    assert coverage["requested_day_count"] == 4
    assert coverage["audio_day_count"] == 3
    assert coverage["question_ready_day_count"] == 2
    assert coverage["analyzed_day_count"] == 1
    assert coverage["question_ready_dates"] == ["2026-07-01", "2026-07-03"]
    assert coverage["analyzed_dates"] == ["2026-07-01"]
    assert coverage["local_processing_dates"] == ["2026-07-02"]
    assert coverage["missing_audio_dates"] == ["2026-07-04"]
    assert coverage["unavailable_dates"] == ["2026-07-02", "2026-07-04"]
    assert coverage["question_ready_ranges"] == ["2026-07-01", "2026-07-03"]
    assert coverage["unavailable_ranges"] == ["2026-07-02", "2026-07-04"]
    assert coverage["complete_coverage"] is False
    assert "2/4 requested days are question-ready" in coverage["summary"]


def test_entire_feed_range_and_compact_gap_descriptions_are_local_only() -> None:
    days = [
        {"feed_id": "90001", "archive_date": "2026-06-30"},
        {"feed_id": "90001", "archive_date": "2026-07-01"},
        {"feed_id": "90001", "archive_date": "2026-07-03"},
        {"feed_id": "90002", "archive_date": "2025-01-01"},
    ]

    assert entire_archive_feed_range(days, "90001") == (
        date(2026, 6, 30),
        date(2026, 7, 3),
    )
    values = ["2026-06-30", "2026-07-01", "2026-07-03"]
    assert compact_archive_date_ranges(values) == [
        "2026-06-30 through 2026-07-01",
        "2026-07-03",
    ]
    assert describe_archive_date_ranges(values) == (
        "2026-06-30 through 2026-07-01, 2026-07-03"
    )


def test_entire_feed_range_rejects_a_feed_without_retained_days() -> None:
    with pytest.raises(ValueError, match="No locally retained feed days"):
        entire_archive_feed_range([], "90001")


def test_library_resume_plan_is_local_first_and_never_starts_acquisition() -> None:
    days = [
        {
            "feed_id": "90003",
            "archive_date": "2026-07-03",
            "is_complete": False,
            "needs_network": True,
        },
        {
            "feed_id": "90001",
            "archive_date": "2026-07-02",
            "is_complete": False,
            "needs_network": False,
        },
        {
            "feed_id": "90002",
            "archive_date": "2026-07-01",
            "is_complete": True,
            "needs_network": False,
        },
    ]
    quota = {"available": False, "remaining": 0, "next_request_at": "later"}

    result = build_library_resume_plan(days, quota)

    assert [value["feed_id"] for value in result["days"]] == ["90001", "90003"]
    assert result["local_count"] == 1
    assert result["network_count"] == 1
    assert result["quota"] == quota


def test_library_coverage_and_resume_plan_include_missing_scheduled_days() -> None:
    days = [
        {
            "feed_id": "90001",
            "feed_name": "Example Public Safety",
            "archive_date": "2026-08-05",
            "is_complete": True,
            "needs_network": False,
            "source_check_due": False,
            "pipeline_percent": 100,
        }
    ]
    schedules = [
        {
            "feed_id": "90001",
            "feed_name": "Example Public Safety",
            "enabled": True,
            "lookback_days": 3,
            "backfill_start_date": "",
        }
    ]

    feeds = build_library_feed_coverage(
        days,
        schedules,
        today=date(2026, 8, 6),
    )
    plan = build_library_resume_plan(
        days,
        {"available": True, "remaining": 12},
        schedules,
        today=date(2026, 8, 6),
    )

    assert feeds[0]["target_start_date"] == "2026-08-04"
    assert feeds[0]["target_end_date"] == "2026-08-06"
    assert feeds[0]["missing_dates"] == ["2026-08-04", "2026-08-06"]
    assert feeds[0]["backlog_count"] == 2
    assert [value["archive_date"] for value in plan["days"]] == [
        "2026-08-04",
        "2026-08-06",
    ]
    assert plan["network_count"] == 2
    assert all(value["scheduled_missing"] for value in plan["days"])


def test_library_resume_plan_catches_up_only_missing_or_incomplete_through_current() -> None:
    days = [
        {
            "feed_id": "90001",
            "feed_name": "Example Public Safety",
            "archive_date": "2026-08-01",
            "is_complete": True,
            "needs_network": False,
            "source_check_due": True,
            "pipeline_percent": 100,
        },
        {
            "feed_id": "90001",
            "feed_name": "Example Public Safety",
            "archive_date": "2026-08-03",
            "is_complete": False,
            "needs_network": False,
            "source_check_due": False,
            "pipeline_percent": 60,
        },
        {
            "feed_id": "90002",
            "feed_name": "Other Feed",
            "archive_date": "2026-08-02",
            "is_complete": False,
            "needs_network": True,
            "source_check_due": True,
            "pipeline_percent": 0,
        },
    ]

    plan = build_library_resume_plan(
        days,
        {"available": True, "remaining": 40},
        today=date(2026, 8, 5),
        requested_feed_id="90001",
        requested_start_date=date(2026, 8, 1),
        requested_through_current=True,
    )

    assert plan["scope_feed_id"] == "90001"
    assert plan["scope_start_date"] == "2026-08-01"
    assert plan["scope_end_date"] == "2026-08-05"
    assert plan["scope_through_current"] is True
    assert [value["feed_id"] for value in plan["feeds"]] == ["90001"]
    assert plan["feeds"][0]["target_day_count"] == 5
    assert plan["feeds"][0]["missing_dates"] == [
        "2026-08-02",
        "2026-08-04",
        "2026-08-05",
    ]
    assert [value["archive_date"] for value in plan["days"]] == [
        "2026-08-03",
        "2026-08-02",
        "2026-08-04",
        "2026-08-05",
    ]
    assert plan["local_count"] == 1
    assert plan["network_count"] == 3
    assert "2026-08-01" not in {
        value["archive_date"] for value in plan["days"]
    }


def test_library_resume_plan_rejects_future_or_partial_catch_up_ranges() -> None:
    with pytest.raises(ValueError, match="all required"):
        build_library_resume_plan(
            [],
            {"available": True, "remaining": 40},
            today=date(2026, 8, 5),
            requested_feed_id="90001",
        )

    with pytest.raises(ValueError, match="future"):
        build_library_resume_plan(
            [],
            {"available": True, "remaining": 40},
            today=date(2026, 8, 5),
            requested_feed_id="90001",
            requested_start_date=date(2026, 8, 1),
            requested_end_date=date(2026, 8, 6),
        )

    with pytest.raises(ValueError, match="future"):
        build_library_resume_plan(
            [],
            {"available": True, "remaining": 40},
            today=date(2026, 8, 5),
            requested_feed_id="90001",
            requested_start_date=date(2026, 8, 6),
            requested_through_current=True,
        )


def test_saved_library_catchup_survives_restart_and_expands_global_resume(
    tmp_path: Path,
) -> None:
    database = tmp_path / "analysis.sqlite3"
    with AnalysisStore(database) as store:
        saved = store.save_library_catchup(
            {
                "feed_id": "90001",
                "feed_name": "Example Public Safety",
                "start_date": "2026-08-01",
                "end_date": "2026-08-05",
                "through_current": True,
            }
        )
    assert saved["start_date"] == "2026-08-01"
    assert saved["through_current"] is True

    with AnalysisStore(database) as reopened:
        catchups = reopened.list_library_catchups()

    days = [
        {
            "feed_id": "90001",
            "feed_name": "Example Public Safety",
            "archive_date": "2026-08-01",
            "is_complete": True,
            "needs_network": False,
            "source_check_due": True,
            "pipeline_percent": 100,
        }
    ]
    plan = build_library_resume_plan(
        days,
        {"available": True, "remaining": 40},
        catchups=catchups,
        today=date(2026, 8, 6),
    )

    assert plan["feeds"][0]["catch_up_saved"] is True
    assert plan["feeds"][0]["catch_up_through_current"] is True
    assert plan["feeds"][0]["catch_up_start_date"] == "2026-08-01"
    assert plan["feeds"][0]["target_end_date"] == "2026-08-06"
    assert plan["feeds"][0]["target_day_count"] == 6
    assert plan["feeds"][0]["source_check_due_count"] == 0
    assert [value["archive_date"] for value in plan["days"]] == [
        "2026-08-02",
        "2026-08-03",
        "2026-08-04",
        "2026-08-05",
        "2026-08-06",
    ]


def test_saved_through_current_catchup_defaults_snapshot_end_to_today(
    tmp_path: Path,
) -> None:
    with AnalysisStore(tmp_path / "analysis.sqlite3") as store:
        saved = store.save_library_catchup(
            {
                "feed_id": "90001",
                "feed_name": "Example Public Safety",
                "start_date": date.today().isoformat(),
                "through_current": True,
            }
        )

    assert saved["end_date"] == date.today().isoformat()
    assert saved["through_current"] is True


def test_saved_through_current_catchup_preserves_scheduled_source_refresh() -> None:
    current = date(2026, 8, 6)
    plan = build_library_resume_plan(
        [
            {
                "feed_id": "90001",
                "feed_name": "Example Public Safety",
                "archive_date": current.isoformat(),
                "is_complete": True,
                "needs_network": False,
                "source_check_due": True,
                "pipeline_percent": 100,
            }
        ],
        {"available": True, "remaining": 40},
        schedules=[
            {
                "feed_id": "90001",
                "feed_name": "Example Public Safety",
                "enabled": True,
                "lookback_days": 1,
            }
        ],
        catchups=[
            {
                "feed_id": "90001",
                "feed_name": "Example Public Safety",
                "start_date": "2026-08-01",
                "end_date": "2026-08-05",
                "through_current": True,
            }
        ],
        today=current,
    )

    planned_dates = [value["archive_date"] for value in plan["days"]]
    assert planned_dates == [
        "2026-08-01",
        "2026-08-02",
        "2026-08-03",
        "2026-08-04",
        "2026-08-05",
        current.isoformat(),
    ]
    assert plan["days"][-1]["status"] == "Source refresh due"
    assert plan["feeds"][0]["source_check_due_count"] == 1


def test_saved_through_current_catchup_clears_only_after_current_day_is_complete() -> None:
    catchups = [
        {
            "feed_id": "90001",
            "feed_name": "Example Public Safety",
            "start_date": "2026-08-01",
            "end_date": "2026-08-02",
            "through_current": True,
        }
    ]
    complete = {
        "feed_id": "90001",
        "feed_name": "Example Public Safety",
        "is_complete": True,
        "needs_network": False,
        "source_check_due": False,
        "pipeline_percent": 100,
    }

    assert completed_library_catchup_feed_ids(
        [{**complete, "archive_date": "2026-08-01"}],
        catchups,
        today=date(2026, 8, 3),
    ) == []
    assert completed_library_catchup_feed_ids(
        [
            {**complete, "archive_date": "2026-08-01"},
            {**complete, "archive_date": "2026-08-02"},
        ],
        catchups,
        today=date(2026, 8, 3),
    ) == []
    assert completed_library_catchup_feed_ids(
        [
            {**complete, "archive_date": "2026-08-01"},
            {**complete, "archive_date": "2026-08-02"},
            {**complete, "archive_date": "2026-08-03"},
        ],
        catchups,
        today=date(2026, 8, 3),
    ) == ["90001"]


def test_library_catchup_schema_migrates_existing_fixed_ranges(tmp_path: Path) -> None:
    database = tmp_path / "analysis.sqlite3"
    with AnalysisStore(database) as store:
        store.connection.executescript(
            """
            ALTER TABLE library_catchups RENAME TO library_catchups_newer;
            CREATE TABLE library_catchups (
                feed_id TEXT PRIMARY KEY,
                feed_name TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            INSERT INTO library_catchups(
                feed_id, feed_name, start_date, end_date, created_at, updated_at
            ) VALUES (
                '90001', 'Legacy feed', '2026-08-01', '2026-08-02',
                '2026-08-02T00:00:00+00:00', '2026-08-02T00:00:00+00:00'
            );
            DROP TABLE library_catchups_newer;
            """
        )

    with AnalysisStore(database) as reopened:
        columns = {
            str(row["name"])
            for row in reopened.connection.execute(
                "PRAGMA table_info(library_catchups)"
            ).fetchall()
        }
        catchup = reopened.list_library_catchups()[0]

    assert "through_current" in columns
    assert catchup["through_current"] is False


def test_current_day_source_snapshot_becomes_resume_candidate_when_stale(
    tmp_path: Path,
    monkeypatch,
) -> None:
    day = _day(tmp_path, "90001", date.today().isoformat())
    source = day / f"{date.today():%Y%m%d}0000-1-90001.mp3"
    source.write_bytes(b"source")
    combined = day / f"combined_90001_{date.today():%Y%m%d}.mp3"
    combined.write_bytes(b"combined")
    combined.with_suffix(".manifest.json").write_text(
        json.dumps({"sources": [{"source_file": source.name}]}),
        encoding="utf-8",
    )
    remember_archive_identity(
        day,
        "90001",
        date.today(),
        "archive-1",
        source,
    )
    assert remember_complete_archive_day(
        day,
        "90001",
        date.today(),
        ["archive-1"],
    )
    completion = day / ".broadcastify-archive-complete.json"
    payload = json.loads(completion.read_text(encoding="utf-8"))
    payload["completed_at_unix"] = time.time() - 3_600
    completion.write_text(json.dumps(payload), encoding="utf-8")

    state = scan_local_library(tmp_path, tmp_path / "analysis.sqlite3")[0]
    plan = build_library_resume_plan(
        [state],
        {"available": True, "remaining": 12},
    )

    assert state["known_source_count"] == 1
    assert state["retained_source_count"] == 1
    assert state["source_check_due"] is True
    assert plan["days"][0]["needs_local_processing"] is True
    assert plan["days"][0]["needs_network"] is True
    assert plan["local_count"] == 1
    assert plan["network_count"] == 1


def test_delete_feed_retries_a_transient_windows_directory_lock(
    tmp_path: Path,
    monkeypatch,
) -> None:
    feed = _day(tmp_path, "90001", "2026-08-05").parent
    (feed / "20260805" / "retained.txt").write_text("saved", encoding="utf-8")
    database = tmp_path / "analysis.sqlite3"
    with AnalysisStore(database):
        pass
    original_rename = Path.rename
    attempts = 0

    def flaky_rename(source: Path, target: Path) -> Path:
        nonlocal attempts
        if source == feed and attempts < 2:
            attempts += 1
            raise PermissionError(5, "Access is denied")
        return original_rename(source, target)

    monkeypatch.setattr(Path, "rename", flaky_rename)
    monkeypatch.setattr(
        "broadcastify_cli.library.DELETE_DETACH_RETRY_SECONDS",
        (0.0, 0.0, 0.0),
    )

    result = delete_local_library_feed(tmp_path, database, "90001")

    assert attempts == 2
    assert result["directory_deleted"] is True
    assert not feed.exists()


def test_delete_feed_preserves_files_and_records_when_windows_lock_persists(
    tmp_path: Path,
    monkeypatch,
) -> None:
    day = _day(tmp_path, "90001", "2026-08-05")
    audio = day / "combined_90001_20260805.mp3"
    transcript = day / "transcripts" / "combined_90001_20260805.json"
    audio.write_bytes(b"audio")
    transcript.parent.mkdir()
    transcript.write_text(
        json.dumps(
            {
                "segments": [
                    {"start": 0.0, "end": 1.0, "text": "retained"}
                ]
            }
        ),
        encoding="utf-8",
    )
    database = tmp_path / "analysis.sqlite3"
    with AnalysisStore(database) as store:
        store.import_transcript("90001", date(2026, 8, 5), transcript, audio)

    def locked_rename(_source: Path, _target: Path) -> Path:
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(Path, "rename", locked_rename)
    monkeypatch.setattr(
        "broadcastify_cli.library.DELETE_DETACH_RETRY_SECONDS",
        (0.0, 0.0),
    )

    with pytest.raises(PermissionError, match="still in use"):
        delete_local_library_feed(tmp_path, database, "90001")

    assert audio.is_file()
    assert transcript.is_file()
    with AnalysisStore(database) as store:
        assert len(store.list_days("90001")) == 1


def test_delete_local_library_feed_removes_only_selected_feed_and_schedule(
    tmp_path: Path,
) -> None:
    target = _day(tmp_path, "90001", "2026-07-12")
    other = _day(tmp_path, "90002", "2026-07-12")
    target_audio = target / "combined_90001_20260712.mp3"
    other_audio = other / "combined_90002_20260712.mp3"
    target_audio.write_bytes(b"target audio")
    other_audio.write_bytes(b"other audio")
    target_transcript = target / "transcripts" / "combined_90001_20260712.json"
    other_transcript = other / "transcripts" / "combined_90002_20260712.json"
    target_transcript.parent.mkdir()
    other_transcript.parent.mkdir()
    transcript_payload = {
        "model": "test",
        "segments": [{"start": 0.0, "end": 1.0, "text": "retained"}],
    }
    target_transcript.write_text(json.dumps(transcript_payload), encoding="utf-8")
    other_transcript.write_text(json.dumps(transcript_payload), encoding="utf-8")
    database = tmp_path / "analysis.sqlite3"
    with AnalysisStore(database) as store:
        store.import_transcript(
            "90001", date(2026, 7, 12), target_transcript, target_audio
        )
        store.import_transcript(
            "90002", date(2026, 7, 12), other_transcript, other_audio
        )
        store.save_feed_schedule(
            {
                "feed_id": "90001",
                "feed_name": "Target Feed",
                "run_time_local": "02:00",
                "job": {},
            }
        )
        store.save_library_catchup(
            {
                "feed_id": "90001",
                "feed_name": "Target Feed",
                "start_date": "2026-07-10",
                "end_date": "2026-07-12",
            }
        )
        store.save_library_catchup(
            {
                "feed_id": "90002",
                "feed_name": "Other Feed",
                "start_date": "2026-07-10",
                "end_date": "2026-07-12",
            }
        )
        store.save_feed_schedule(
            {
                "feed_id": "90002",
                "feed_name": "Other Feed",
                "run_time_local": "02:00",
                "job": {},
            }
        )

    result = delete_local_library_feed(
        tmp_path,
        database,
        "90001",
        remove_schedule=True,
    )

    assert result["directory_deleted"] is True
    assert result["cleanup_pending"] is False
    assert result["days_deleted"] == 1
    assert result["segments_deleted"] == 1
    assert result["schedules_deleted"] == 1
    assert result["catchups_deleted"] == 1
    assert not (tmp_path / "90001").exists()
    assert other_audio.is_file()
    with AnalysisStore(database) as store:
        assert store.list_days("90001") == []
        assert len(store.list_days("90002")) == 1
        assert [value["feed_id"] for value in store.list_feed_schedules()] == [
            "90002"
        ]
        assert [value["feed_id"] for value in store.list_library_catchups()] == [
            "90002"
        ]


def test_library_refresh_retries_only_detached_delete_cleanup(tmp_path: Path) -> None:
    pending = tmp_path / (".deleting-90001-" + "a" * 32)
    pending.mkdir()
    (pending / "old.mp3").write_bytes(b"old")
    ordinary = tmp_path / "notes"
    ordinary.mkdir()
    (ordinary / "keep.txt").write_text("keep", encoding="utf-8")
    lookalike = tmp_path / ".deleting-not-a-tombstone"
    lookalike.mkdir()

    assert cleanup_pending_library_deletions(tmp_path) == 1
    assert not pending.exists()
    assert (ordinary / "keep.txt").is_file()
    assert lookalike.is_dir()


def test_library_discovers_partial_and_analyzed_days(tmp_path: Path) -> None:
    partial = _day(tmp_path, "90003", "2026-07-07")
    (partial / "202607070000-1-90003.mp3").write_bytes(b"raw")
    (partial / "combined_90003_20260707.manifest.json").write_text(
        json.dumps(
            {
                "feed_id": "90003",
                "feed_name": "Example County Public Safety",
                "archive_date": "2026-07-07",
                "sources": [],
            }
        ),
        encoding="utf-8",
    )

    ready = _day(tmp_path, "90001", "2026-07-12")
    audio = ready / "combined_90001_20260712.mp3"
    transcript = ready / "transcripts" / "combined_90001_20260712.json"
    audio.write_bytes(b"audio")
    transcript.parent.mkdir()
    transcript.write_text(
        json.dumps(
            {
                "model": "turbo",
                "duration": 60.0,
                "segments": [
                    {
                        "start": 1.0,
                        "end": 2.0,
                        "text": "Unit responding.",
                        "speaker": "SPEAKER_00",
                    }
                ],
                "diarization_requested": True,
            }
        ),
        encoding="utf-8",
    )
    database = tmp_path / "analysis.sqlite3"
    with AnalysisStore(database) as store:
        imported = store.import_transcript("90001", date(2026, 7, 12), transcript, audio)
        store.save_daily_summary(
            imported.day_id,
            "One dispatch call was retained.",
            [],
            model="test",
            prompt_version=PROMPT_VERSION,
            transcript_sha256=imported.transcript_sha256,
        )
        store.save_feed_catalog(
            [{"feed_id": "90001", "name": "Example City Public Safety"}]
        )

    states = scan_local_library(tmp_path, database)

    assert len(states) == 2
    analyzed = next(value for value in states if value["feed_id"] == "90001")
    assert analyzed["feed_name"] == "Example City Public Safety"
    assert analyzed["has_diarization"] is True
    assert analyzed["has_analysis"] is True
    assert analyzed["is_complete"] is True
    assert analyzed["primary_action"] == "open_review"

    incomplete = next(value for value in states if value["feed_id"] == "90003")
    assert incomplete["feed_name"] == "Example County Public Safety"
    assert incomplete["raw_file_count"] == 1
    assert incomplete["needs_network"] is True
    assert incomplete["primary_action"] == "resume_download"


def test_library_separates_working_audio_and_cleans_old_orphans(
    tmp_path: Path,
) -> None:
    day = _day(tmp_path, "90004", "2026-07-29")
    raw = day / "202607290000-1-90004.mp3"
    raw.write_bytes(b"retained")
    cache = day / "transcripts" / ".cache"
    cache.mkdir(parents=True)
    prepared = cache / "combined_90004_20260729.pyannote.flac"
    prepared.write_bytes(b"reusable retry input")
    owner = work_file_owner_token()
    active_raw = (
        cache
        / f".combined_90004_20260729.pyannote.{owner}."
        f"{os.getpid()}.123.pyannote.f32le"
    )
    active_raw.write_bytes(b"active raw scratch")
    active_preparation = (
        cache
        / f".combined_90004_20260729.{owner}."
        f"{os.getpid()}.789.pyannote.part.flac"
    )
    active_preparation.write_bytes(b"active preparation")
    active_combine = (
        day
        / f".combined_90004_20260729.{owner}."
        f"{os.getpid()}.abcdefgh.part.mp3"
    )
    active_combine.write_bytes(b"active combined output")
    orphan_raw = (
        cache
        / f".combined_90004_20260729.pyannote.{owner}."
        "999999999.456.pyannote.f32le"
    )
    orphan_raw.write_bytes(b"orphan raw scratch")
    orphan_preparation = (
        cache
        / f".combined_90004_20260729.{owner}."
        "999999999.456.pyannote.part.flac"
    )
    orphan_preparation.write_bytes(b"orphan preparation")
    orphan_combine = (
        day
        / f".combined_90004_20260729.{owner}."
        "999999999.abcdefgh.part.mp3"
    )
    orphan_combine.write_bytes(b"orphan combined output")
    foreign_combine = (
        day
        / ".combined_90004_20260729.000000000000."
        "999999999.abcdefgh.part.mp3"
    )
    foreign_combine.write_bytes(b"foreign active output")
    orphan_part = day / ".combined_90004_20260729.old.part.mp3"
    orphan_part.write_bytes(b"orphan combined output")
    old = time.time() - 7_200
    for path in (
        active_raw,
        active_preparation,
        active_combine,
        orphan_raw,
        orphan_preparation,
        orphan_combine,
        foreign_combine,
        orphan_part,
    ):
        os.utime(path, (old, old))

    state = scan_local_library(tmp_path)[0]

    assert state["storage_bytes"] == raw.stat().st_size
    assert state["working_storage_bytes"] == (
        prepared.stat().st_size
        + active_raw.stat().st_size
        + active_preparation.stat().st_size
        + active_combine.stat().st_size
        + foreign_combine.stat().st_size
        + orphan_part.stat().st_size
    )
    assert prepared.exists()
    assert active_raw.exists()
    assert active_preparation.exists()
    assert active_combine.exists()
    assert not orphan_raw.exists()
    assert not orphan_preparation.exists()
    assert not orphan_combine.exists()
    assert foreign_combine.exists()
    assert orphan_part.exists()


def test_library_removes_prepared_audio_after_exact_diarization_cache(
    tmp_path: Path,
) -> None:
    day = _day(tmp_path, "90004", "2026-07-29")
    audio = day / "combined_90004_20260729.mp3"
    audio.write_bytes(b"combined audio")
    cache = day / "transcripts" / ".cache"
    cache.mkdir(parents=True)
    prepared = cache / "combined_90004_20260729.pyannote.flac"
    prepared.write_bytes(b"completed preparation")
    stat = audio.stat()
    (cache.parent / "combined_90004_20260729.diarization.json").write_text(
        json.dumps(
            {
                "engine": "community-1",
                "audio_size": stat.st_size,
                "audio_mtime_ns": stat.st_mtime_ns,
                "turns": [],
            }
        ),
        encoding="utf-8",
    )
    old = time.time() - 7_200
    os.utime(prepared, (old, old))

    state = scan_local_library(tmp_path)[0]

    assert not prepared.exists()
    assert state["working_storage_bytes"] == 0
    assert state["storage_bytes"] == (
        audio.stat().st_size
        + (
            cache.parent / "combined_90004_20260729.diarization.json"
        ).stat().st_size
    )


def test_library_rejects_transcript_older_than_refreshed_combined_audio(
    tmp_path: Path,
) -> None:
    day = _day(tmp_path, "90005", "2026-07-29")
    audio = day / "combined_90005_20260729.mp3"
    audio.write_bytes(b"first combined audio")
    transcript = day / "transcripts" / "combined_90005_20260729.json"
    transcript.parent.mkdir()
    transcript.write_text(
        json.dumps(
            {
                "segments": [{"start": 0.0, "end": 1.0, "text": "old"}],
                "diarization_completed": True,
            }
        ),
        encoding="utf-8",
    )
    future = time.time() + 10
    audio.write_bytes(b"refreshed combined audio")
    os.utime(audio, (future, future))

    state = scan_local_library(tmp_path)[0]

    assert state["has_combined"] is True
    assert state["has_transcript"] is False
    assert state["has_stale_transcript"] is True
    assert state["has_diarization"] is False
    assert state["has_analysis"] is False
    assert state["status"] == "Transcript update required"
    assert state["next_step"] == "Update local transcript"


def test_prepare_local_day_retranscribes_after_combined_audio_changes(
    monkeypatch, tmp_path: Path
) -> None:
    day = _day(tmp_path, "90005", "2026-07-29")
    audio = day / "combined_90005_20260729.mp3"
    audio.write_bytes(b"first combined audio")
    transcript = day / "transcripts" / "combined_90005_20260729.json"
    transcript.parent.mkdir()
    transcript.write_text(
        json.dumps(
            {
                "segments": [{"start": 0.0, "end": 1.0, "text": "old"}],
                "diarization_completed": True,
                "diarization_engine": "community-1",
            }
        ),
        encoding="utf-8",
    )
    audio.write_bytes(b"refreshed combined audio")
    future = time.time() + 10
    os.utime(audio, (future, future))
    constructor_arguments: list[dict[str, object]] = []

    class FakeTranscriber:
        def __init__(self, **kwargs: object) -> None:
            constructor_arguments.append(kwargs)

        def transcribe_file(self, _audio: Path, progress=None) -> Path:
            transcript.write_text(
                json.dumps(
                    {
                        "segments": [
                            {"start": 0.0, "end": 1.0, "text": "current"}
                        ],
                        "diarization_completed": True,
                        "diarization_engine": "community-1",
                    }
                ),
                encoding="utf-8",
            )
            return transcript

        def diarize_existing_transcript(self, *_args, **_kwargs) -> Path:
            raise AssertionError("Stale transcript must not be relabeled")

    monkeypatch.setattr(
        "broadcastify_cli.library.LocalTranscriber", FakeTranscriber
    )

    result = prepare_local_day(
        LocalProcessingRequest(
            feed_id="90005",
            archive_date=date(2026, 7, 29),
            output_dir=tmp_path,
        )
    )

    assert result["operation"] == "transcribed"
    assert constructor_arguments
    assert "load_asr" not in constructor_arguments[0]
    assert json.loads(transcript.read_text(encoding="utf-8"))["segments"][0][
        "text"
    ] == "current"


def test_library_does_not_trust_analysis_from_an_older_transcript_revision(
    tmp_path: Path,
) -> None:
    day = _day(tmp_path, "90006", "2026-07-29")
    audio = day / "combined_90006_20260729.mp3"
    audio.write_bytes(b"combined audio")
    transcript = day / "transcripts" / "combined_90006_20260729.json"
    transcript.parent.mkdir()
    transcript.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "text": "old",
                        "speaker": "SPEAKER_00",
                    }
                ],
                "diarization_completed": True,
            }
        ),
        encoding="utf-8",
    )
    database = tmp_path / "analysis.sqlite3"
    with AnalysisStore(database) as store:
        imported = store.import_transcript(
            "90006", date(2026, 7, 29), transcript, audio
        )
        store.save_daily_summary(
            imported.day_id,
            "Old summary",
            [],
            model="test",
            prompt_version=PROMPT_VERSION,
            transcript_sha256=imported.transcript_sha256,
        )

    transcript.write_text(
        json.dumps(
            {
                "segments": [
                    {"start": 0.0, "end": 1.0, "text": "current"}
                ],
                "diarization_completed": False,
            }
        ),
        encoding="utf-8",
    )

    state = scan_local_library(tmp_path, database)[0]

    assert state["has_transcript"] is True
    assert state["has_imported_transcript"] is False
    assert state["has_diarization"] is False
    assert state["has_analysis"] is False
    assert state["has_stale_analysis"] is True
    assert state["segment_count"] == 0
    assert state["incident_count"] == 0
    assert state["status"] == "Transcript ready"


def test_library_requires_summary_to_match_current_imported_transcript(
    tmp_path: Path,
) -> None:
    day = _day(tmp_path, "90008", "2026-07-29")
    audio = day / "combined_90008_20260729.mp3"
    audio.write_bytes(b"combined audio")
    transcript = day / "transcripts" / "combined_90008_20260729.json"
    transcript.parent.mkdir()
    transcript.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "text": "old",
                        "speaker": "SPEAKER_00",
                    }
                ],
                "diarization_completed": True,
            }
        ),
        encoding="utf-8",
    )
    database = tmp_path / "analysis.sqlite3"
    with AnalysisStore(database) as store:
        previous = store.import_transcript(
            "90008", date(2026, 7, 29), transcript, audio
        )
        transcript.write_text(
            json.dumps(
                {
                    "segments": [
                        {
                            "start": 0.0,
                            "end": 2.0,
                            "text": "current",
                            "speaker": "SPEAKER_01",
                        }
                    ],
                    "diarization_completed": True,
                }
            ),
            encoding="utf-8",
        )
        current = store.import_transcript(
            "90008", date(2026, 7, 29), transcript, audio
        )
        assert current.transcript_sha256 != previous.transcript_sha256
        store.save_daily_summary(
            current.day_id,
            "Summary for the previous transcript.",
            [],
            model="test",
            prompt_version=PROMPT_VERSION,
            transcript_sha256=previous.transcript_sha256,
        )

    state = scan_local_library(tmp_path, database)[0]

    assert state["has_transcript"] is True
    assert state["has_imported_transcript"] is True
    assert state["has_diarization"] is True
    assert state["has_analysis"] is False
    assert state["has_stale_analysis"] is True
    assert state["segment_count"] == 1
    assert state["incident_count"] == 0
    assert state["status"] == "Analysis update available"


def test_imported_day_storage_does_not_count_unrelated_sibling_files(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external"
    external.mkdir()
    audio = external / "combined_90007_20260729.mp3"
    transcript = external / "combined_90007_20260729.json"
    audio.write_bytes(b"audio")
    transcript.write_text(
        json.dumps({"segments": [], "diarization_completed": False}),
        encoding="utf-8",
    )
    (external / "unrelated-large-file.bin").write_bytes(b"x" * 10_000)
    output = tmp_path / "library"
    database = output / "analysis.sqlite3"
    output.mkdir()
    with AnalysisStore(database) as store:
        store.import_transcript(
            "90007", date(2026, 7, 29), transcript, audio
        )

    state = scan_local_library(output, database)[0]

    assert state["storage_bytes"] == (
        audio.stat().st_size + transcript.stat().st_size
    )


def test_imported_day_storage_includes_mixed_layout_transcript(
    tmp_path: Path,
) -> None:
    output = tmp_path / "library"
    day = _day(output, "90009", "2026-07-29")
    audio = day / "combined_90009_20260729.mp3"
    audio.write_bytes(b"audio")
    external = tmp_path / "external"
    external.mkdir()
    transcript = external / "combined_90009_20260729.json"
    transcript.write_text(
        json.dumps({"segments": [], "diarization_completed": False}),
        encoding="utf-8",
    )
    database = output / "analysis.sqlite3"
    with AnalysisStore(database) as store:
        store.import_transcript(
            "90009", date(2026, 7, 29), transcript, audio
        )

    state = scan_local_library(output, database)[0]

    assert state["storage_bytes"] == (
        audio.stat().st_size + transcript.stat().st_size
    )


def test_imported_day_storage_includes_matching_external_raw_blocks(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external"
    external.mkdir()
    audio = external / "combined_90010_20260729.mp3"
    audio.write_bytes(b"audio")
    raw = external / "202607290000-1-90010.mp3"
    raw.write_bytes(b"raw block")
    transcript = external / "combined_90010_20260729.json"
    transcript.write_text(
        json.dumps({"segments": [], "diarization_completed": False}),
        encoding="utf-8",
    )
    output = tmp_path / "library"
    output.mkdir()
    database = output / "analysis.sqlite3"
    with AnalysisStore(database) as store:
        store.import_transcript(
            "90010", date(2026, 7, 29), transcript, audio
        )

    state = scan_local_library(output, database)[0]

    assert state["raw_file_count"] == 1
    assert state["storage_bytes"] == (
        audio.stat().st_size
        + raw.stat().st_size
        + transcript.stat().st_size
    )


def test_library_does_not_present_older_combined_timeline_as_current(
    tmp_path: Path,
) -> None:
    day = _day(tmp_path, "90001", "2026-07-20")
    first = day / "202607200000-1-90001.mp3"
    newer = day / "202607200030-2-90001.mp3"
    first.write_bytes(b"first")
    combined = day / "combined_90001_20260720.mp3"
    combined.write_bytes(b"older combined audio")
    manifest = combined.with_suffix(".manifest.json")
    manifest.write_text(
        json.dumps({"sources": [{"source_file": first.name}]}),
        encoding="utf-8",
    )
    newer.write_bytes(b"newer")
    transcript = day / "transcripts" / "combined_90001_20260720.json"
    transcript.parent.mkdir()
    transcript.write_text(
        json.dumps(
            {
                "segments": [{"start": 0.0, "end": 1.0, "text": "old"}],
                "diarization_completed": True,
            }
        ),
        encoding="utf-8",
    )

    state = scan_local_library(tmp_path)[0]

    assert state["raw_file_count"] == 2
    assert state["has_stale_combined"] is True
    assert state["has_combined"] is False
    assert state["has_transcript"] is False
    assert state["has_diarization"] is False
    assert state["has_analysis"] is False
    assert state["can_open_review"] is False
    assert state["is_complete"] is False
    assert state["pipeline_percent"] == 20
    assert state["combined_path"] == ""
    assert state["status"] == "New audio pending combine"
    assert state["next_step"] == "Refresh archive day"
    assert state["primary_action"] == "resume_download"
    assert state["needs_network"] is True


def test_library_hides_timeline_with_collapsed_archive_identities(
    tmp_path: Path,
) -> None:
    archive_date = date(2026, 7, 31)
    day = _day(tmp_path, "90001", archive_date.isoformat())
    source = day / "202607310027-111-90001.mp3"
    source.write_bytes(b"source")
    combined = day / "combined_90001_20260731.mp3"
    combined.write_bytes(b"combined")
    combined.with_suffix(".manifest.json").write_text(
        json.dumps({"sources": [{"source_file": source.name}]}),
        encoding="utf-8",
    )
    remember_archive_identity(
        day,
        "90001",
        archive_date,
        "at-0027",
        source,
        listing_prefix="202607310027",
    )
    index = day / ".broadcastify-archive-index.json"
    payload = json.loads(index.read_text(encoding="utf-8"))
    payload["archives"]["at-0127"] = {
        "filename": source.name,
        "listing_prefix": "202607310127",
        "size": source.stat().st_size,
    }
    index.write_text(json.dumps(payload), encoding="utf-8")

    state = scan_local_library(tmp_path)[0]

    assert state["collapsed_identity_count"] == 1
    assert state["has_stale_combined"] is True
    assert state["has_combined"] is False
    assert state["can_open_review"] is False
    assert state["is_complete"] is False
    assert state["status"] == "Archive timeline repair required"
    assert state["next_step"] == "Verify & repair archive day"
    assert "collapsed onto another retained filename" in state["status_detail"]
    assert state["needs_network"] is True


def test_library_marks_older_analysis_for_local_evidence_update(tmp_path: Path) -> None:
    ready = _day(tmp_path, "90001", "2026-07-11")
    audio = ready / "combined_90001_20260711.mp3"
    transcript = ready / "transcripts" / "combined_90001_20260711.json"
    audio.write_bytes(b"audio")
    transcript.parent.mkdir()
    transcript.write_text(
        json.dumps(
            {
                "duration": 60.0,
                "segments": [
                    {
                        "start": 1.0,
                        "end": 2.0,
                        "text": "Unit responding.",
                        "speaker": "SPEAKER_00",
                    }
                ],
                "diarization_requested": True,
                "diarization_model": "pyannote/test",
            }
        ),
        encoding="utf-8",
    )
    database = tmp_path / "analysis.sqlite3"
    with AnalysisStore(database) as store:
        imported = store.import_transcript("90001", date(2026, 7, 11), transcript, audio)
        store.save_daily_summary(
            imported.day_id,
            "An older summary.",
            [],
            model="test",
            prompt_version="older-evidence-rules",
            transcript_sha256=imported.transcript_sha256,
        )

    state = scan_local_library(tmp_path, database)[0]

    assert state["has_analysis"] is False
    assert state["has_stale_analysis"] is True
    assert state["status"] == "Analysis update available"
    assert state["next_step"] == "Re-run evidence analysis"
    assert state["primary_action"] == "continue_local"
    assert state["can_open_review"] is False
    assert state["needs_network"] is False


def test_prepare_local_day_uses_diarization_only_for_existing_transcript(
    monkeypatch, tmp_path: Path
) -> None:
    day = _day(tmp_path, "90001", "2026-07-12")
    audio = day / "combined_90001_20260712.mp3"
    transcript = day / "transcripts" / "combined_90001_20260712.json"
    audio.write_bytes(b"audio")
    transcript.parent.mkdir()
    transcript.write_text(
        json.dumps(
            {
                "model": "turbo",
                "segments": [{"start": 0.0, "end": 1.0, "text": "Radio"}],
                "words": [],
                "diarization_requested": False,
            }
        ),
        encoding="utf-8",
    )
    constructor_arguments = []

    class FakeTranscriber:
        def __init__(self, **kwargs: object) -> None:
            constructor_arguments.append(kwargs)

        def diarize_existing_transcript(
            self, _audio: Path, value: Path, progress=None
        ) -> Path:
            payload = json.loads(value.read_text(encoding="utf-8"))
            payload["diarization_requested"] = True
            value.write_text(json.dumps(payload), encoding="utf-8")
            return value

    monkeypatch.setattr("broadcastify_cli.library.LocalTranscriber", FakeTranscriber)
    result = prepare_local_day(
        LocalProcessingRequest(
            feed_id="90001",
            archive_date=date(2026, 7, 12),
            output_dir=tmp_path,
        )
    )

    assert result["operation"] == "diarized"
    assert constructor_arguments[0]["load_asr"] is False


def test_prepare_local_day_progress_uses_selected_transcription_model_language(
    monkeypatch, tmp_path: Path
) -> None:
    day = _day(tmp_path, "90001", "2026-07-13")
    audio = day / "combined_90001_20260713.mp3"
    audio.write_bytes(b"audio")
    messages: list[str] = []

    class FakeTranscriber:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def transcribe_file(self, _audio: Path, progress=None) -> Path:
            transcript = (
                day / "transcripts" / "combined_90001_20260713.json"
            )
            transcript.parent.mkdir()
            transcript.write_text(
                json.dumps({"segments": [], "words": []}),
                encoding="utf-8",
            )
            return transcript

    monkeypatch.setattr("broadcastify_cli.library.LocalTranscriber", FakeTranscriber)

    result = prepare_local_day(
        LocalProcessingRequest(
            feed_id="90001",
            archive_date=date(2026, 7, 13),
            output_dir=tmp_path,
            model="qwen3-asr-0.6b-int8",
            asr_engine="qwen3-asr",
            diarize=False,
        ),
        progress=messages.append,
    )

    assert result["operation"] == "transcribed"
    assert messages == [
        "Loading local transcription qwen3-asr-0.6b-int8 for 2026-07-13…"
    ]
    assert all("Whisper" not in message for message in messages)


def test_library_marks_existing_plain_transcript_for_diarization(tmp_path: Path) -> None:
    day = _day(tmp_path, "300", "2026-07-10")
    (day / "combined_300_20260710.mp3").write_bytes(b"audio")
    transcript = day / "transcripts" / "combined_300_20260710.json"
    transcript.parent.mkdir()
    transcript.write_text(
        json.dumps(
            {
                "model": "turbo",
                "segments": [{"start": 0.0, "end": 1.0, "text": "Dispatch"}],
                "diarization_requested": False,
            }
        ),
        encoding="utf-8",
    )

    state = scan_local_library(tmp_path, tmp_path / "analysis.sqlite3")[0]

    assert state["status"] == "Transcript ready"
    assert state["next_step"] == "Add speaker labels"
    assert state["status_detail"] == (
        "Transcript exists; speaker labels can run without repeating transcription"
    )
    assert state["primary_action"] == "continue_local"
    assert state["needs_network"] is False


def test_library_describes_audio_ready_state_without_assuming_whisper(
    tmp_path: Path,
) -> None:
    day = _day(tmp_path, "299", "2026-07-10")
    (day / "combined_299_20260710.mp3").write_bytes(b"audio")

    state = scan_local_library(tmp_path, tmp_path / "analysis.sqlite3")[0]

    assert state["status"] == "Audio ready"
    assert state["next_step"] == "Transcribe locally"
    assert state["status_detail"] == (
        "Combined audio is ready for local transcription"
    )
    assert "Whisper" not in state["status_detail"]


def test_library_does_not_treat_a_request_flag_as_completed_diarization(
    tmp_path: Path,
) -> None:
    day = _day(tmp_path, "301", "2026-07-10")
    (day / "combined_301_20260710.mp3").write_bytes(b"audio")
    transcript = day / "transcripts" / "combined_301_20260710.json"
    transcript.parent.mkdir()
    transcript.write_text(
        json.dumps(
            {
                "model": "turbo",
                "segments": [{"start": 0.0, "end": 1.0, "text": "Dispatch"}],
                "diarization_requested": True,
                "diarization_model": "pyannote/test",
            }
        ),
        encoding="utf-8",
    )

    state = scan_local_library(tmp_path, tmp_path / "analysis.sqlite3")[0]

    assert state["has_diarization"] is False
    assert state["next_step"] == "Add speaker labels"


def test_library_labels_portable_speakers_as_preview_with_accuracy_upgrade(
    tmp_path: Path,
) -> None:
    day = _day(tmp_path, "302", "2026-07-10")
    (day / "combined_302_20260710.mp3").write_bytes(b"audio")
    transcript = day / "transcripts" / "combined_302_20260710.json"
    transcript.parent.mkdir()
    transcript.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "text": "Dispatch",
                        "speaker": "SPEAKER_00",
                    }
                ],
                "speaker_turns": [
                    {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}
                ],
                "diarization_requested": True,
                "diarization_completed": True,
                "diarization_engine": "sherpa-onnx",
                "diarization_model": (
                    "pyannote-segmentation-3.0-int8+nemo-titanet-small"
                ),
                "diarization_quality": "preview",
            }
        ),
        encoding="utf-8",
    )

    state = scan_local_library(tmp_path, tmp_path / "analysis.sqlite3")[0]

    assert state["has_diarization"] is True
    assert state["diarization_engine"] == "sherpa-onnx"
    assert state["diarization_quality"] == "preview"
    assert state["speaker_upgrade_available"] is True
    assert transcript_satisfies_diarization(transcript, "sherpa-onnx") is True
    assert transcript_satisfies_diarization(transcript, "community-1") is False


def test_community_labels_satisfy_preview_without_being_downgraded(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "transcript.json"
    transcript.write_text(
        json.dumps(
            {
                "diarization_requested": True,
                "diarization_completed": True,
                "diarization_engine": "community-1",
                "diarization_model": "pyannote/speaker-diarization-community-1",
            }
        ),
        encoding="utf-8",
    )

    assert transcript_satisfies_diarization(transcript, "community-1") is True
    assert transcript_satisfies_diarization(transcript, "sherpa-onnx") is True


def test_prepare_local_day_upgrades_preview_without_loading_asr(
    monkeypatch, tmp_path: Path
) -> None:
    day = _day(tmp_path, "303", "2026-07-10")
    audio = day / "combined_303_20260710.mp3"
    transcript = day / "transcripts" / "combined_303_20260710.json"
    audio.write_bytes(b"audio")
    transcript.parent.mkdir()
    transcript.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "text": "Dispatch",
                        "speaker": "SPEAKER_00",
                    }
                ],
                "words": [],
                "diarization_requested": True,
                "diarization_completed": True,
                "diarization_engine": "sherpa-onnx",
                "diarization_model": (
                    "pyannote-segmentation-3.0-int8+nemo-titanet-small"
                ),
            }
        ),
        encoding="utf-8",
    )
    constructor_arguments: list[dict[str, object]] = []

    class FakeTranscriber:
        def __init__(self, **kwargs: object) -> None:
            constructor_arguments.append(kwargs)

        def diarize_existing_transcript(
            self, _audio: Path, value: Path, progress=None
        ) -> Path:
            payload = json.loads(value.read_text(encoding="utf-8"))
            payload.update(
                diarization_engine="community-1",
                diarization_model="pyannote/speaker-diarization-community-1",
                diarization_completed=True,
            )
            value.write_text(json.dumps(payload), encoding="utf-8")
            return value

    monkeypatch.setattr("broadcastify_cli.library.LocalTranscriber", FakeTranscriber)

    result = prepare_local_day(
        LocalProcessingRequest(
            feed_id="303",
            archive_date=date(2026, 7, 10),
            output_dir=tmp_path,
            diarization_engine="community-1",
        )
    )

    assert result["operation"] == "upgraded_diarization"
    assert result["diarization_engine"] == "community-1"
    assert constructor_arguments == [
        {
            "model_name": "turbo",
            "asr_engine": "auto",
            "device": "auto",
            "device_index": 0,
            "compute_type": "auto",
            "asr_model_path": None,
            "diarization_engine": "community-1",
            "diarization_device": "auto",
            "diarize": True,
            "huggingface_token": None,
            "batch_size": 8,
            "min_speakers": None,
            "max_speakers": None,
            "load_asr": False,
        }
    ]


def test_diarize_existing_transcript_reuses_words_without_whisper(tmp_path: Path) -> None:
    audio = tmp_path / "combined.mp3"
    transcript = tmp_path / "transcripts" / "combined.json"
    audio.write_bytes(b"audio")
    transcript.parent.mkdir()
    transcript.write_text(
        json.dumps(
            {
                "model": "turbo",
                "segments": [{"start": 0.0, "end": 1.0, "text": "Dispatch calling"}],
                "words": [
                    {"start": 0.0, "end": 0.5, "text": " Dispatch"},
                    {"start": 0.5, "end": 1.0, "text": " calling"},
                ],
            }
        ),
        encoding="utf-8",
    )

    transcriber = object.__new__(LocalTranscriber)
    transcriber.diarize = True
    transcriber.diarization_device = "cpu"
    transcriber._diarization_pipeline = object()
    transcriber._diarize = lambda *_args, **_kwargs: [
        SpeakerTurn(0.0, 1.0, "SPEAKER_00")
    ]

    result = transcriber.diarize_existing_transcript(audio, transcript)
    payload = json.loads(result.read_text(encoding="utf-8"))

    assert payload["diarization_requested"] is True
    assert payload["diarization_device"] == "cpu"
    assert payload["segments"][0]["speaker"] == "SPEAKER_00"
    assert payload["words"][0]["speaker"] == "SPEAKER_00"
    assert "SPEAKER_00: Dispatch calling" in transcript.with_suffix(".txt").read_text(
        encoding="utf-8"
    )
