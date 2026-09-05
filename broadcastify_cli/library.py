from __future__ import annotations

import json
import re
import shutil
import time
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from .analysis import PROMPT_VERSION
from .archive_cache import (
    ARCHIVE_CACHE_COMPLETION_FILENAME,
    ARCHIVE_CACHE_COMPLETION_SCHEMA_VERSION,
    cached_archives_for_ids,
    collapsed_archive_identity_count,
)
from .audio import combined_output_is_current
from .portable_diarization import (
    COMMUNITY_DIARIZATION_ENGINE,
    COMMUNITY_DIARIZATION_QUALITY,
    PORTABLE_DIARIZATION_ENGINE,
    diarization_engine_satisfies,
    normalize_diarization_engine,
)
from .storage import AnalysisStore, sha256_file
from .transcription import LocalTranscriber
from .workfiles import (
    cleanup_orphaned_audio_work_files,
    directory_storage_usage,
    files_storage_usage,
)


RAW_ARCHIVE_PATTERN = re.compile(r"^\d{12}-\d+-(\d+)\.mp3$", re.IGNORECASE)
DAY_DIRECTORY_PATTERN = re.compile(r"^\d{8}$")
PENDING_DELETE_PATTERN = re.compile(r"^\.deleting-\d+-[0-9a-f]{32}$")
CURRENT_DAY_SOURCE_REFRESH = timedelta(minutes=30)
DELETE_DETACH_RETRY_SECONDS = (0.0, 0.15, 0.3, 0.6, 1.0, 1.5)


def _schedule_target_dates(
    schedule: dict[str, Any],
    today: date,
) -> list[date]:
    if not bool(schedule.get("enabled", True)):
        return []
    lookback = max(1, min(14, int(schedule.get("lookback_days") or 2)))
    start = today - timedelta(days=lookback - 1)
    backfill = str(schedule.get("backfill_start_date") or "").strip()
    if backfill:
        try:
            start = min(start, date.fromisoformat(backfill))
        except ValueError:
            pass
    return [
        start + timedelta(days=offset)
        for offset in range((today - start).days + 1)
    ]


def build_library_feed_coverage(
    days: list[dict[str, Any]],
    schedules: list[dict[str, Any]] | None = None,
    catchups: list[dict[str, Any]] | None = None,
    *,
    today: date | None = None,
    requested_ranges: dict[str, tuple[date, date]] | None = None,
) -> list[dict[str, Any]]:
    """Summarize retained and scheduled coverage without website access."""

    current = today or date.today()
    schedule_by_feed = {
        str(value.get("feed_id") or ""): value
        for value in schedules or []
        if str(value.get("feed_id") or "")
    }
    catchup_by_feed = {
        str(value.get("feed_id") or ""): value
        for value in catchups or []
        if str(value.get("feed_id") or "")
    }
    days_by_feed: dict[str, list[dict[str, Any]]] = {}
    for value in days:
        feed_id = str(value.get("feed_id") or "")
        if feed_id:
            days_by_feed.setdefault(feed_id, []).append(value)
    explicit_ranges = requested_ranges or {}
    feed_ids = sorted(
        set(days_by_feed)
        | set(schedule_by_feed)
        | set(catchup_by_feed)
        | set(explicit_ranges)
    )
    results: list[dict[str, Any]] = []
    for feed_id in feed_ids:
        retained = days_by_feed.get(feed_id, [])
        schedule = schedule_by_feed.get(feed_id)
        catchup = catchup_by_feed.get(feed_id)
        requested = explicit_ranges.get(feed_id)
        schedule_dates = _schedule_target_dates(schedule, current) if schedule else []
        schedule_values = {value.isoformat() for value in schedule_dates}
        strict_catchup_values: set[str] = set()
        catchup_through_current = bool(
            catchup and catchup.get("through_current", False)
        )
        if requested is not None:
            range_start, range_end = requested
            schedule_values = set()
            if range_start > range_end:
                raise ValueError("Catch-up start date must not be after its end date.")
            if range_end > current:
                raise ValueError("Catch-up end date cannot be in the future.")
            target_dates = [
                range_start + timedelta(days=offset)
                for offset in range((range_end - range_start).days + 1)
            ]
            strict_catchup_values = {value.isoformat() for value in target_dates}
        else:
            target_dates = list(schedule_dates)
            if catchup is not None:
                try:
                    catchup_start = date.fromisoformat(
                        str(catchup.get("start_date") or "")
                    )
                    catchup_end = date.fromisoformat(
                        str(catchup.get("end_date") or "")
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"Saved catch-up dates for feed {feed_id} are invalid."
                    ) from exc
                effective_catchup_end = (
                    current if catchup_through_current else catchup_end
                )
                if catchup_start > effective_catchup_end:
                    raise ValueError(
                        f"Saved catch-up start date for feed {feed_id} is after its end date."
                    )
                if not catchup_through_current and catchup_end > current:
                    raise ValueError(
                        f"Saved catch-up end date for feed {feed_id} is in the future."
                    )
                catchup_dates = [
                    catchup_start + timedelta(days=offset)
                    for offset in range(
                        (effective_catchup_end - catchup_start).days + 1
                    )
                ]
                target_dates.extend(catchup_dates)
                target_dates = sorted(set(target_dates))
                if catchup_through_current:
                    strict_catchup_values = {
                        value.isoformat() for value in catchup_dates
                    }
        target_values = {value.isoformat() for value in target_dates}
        retained_by_date = {
            str(value.get("archive_date") or ""): value
            for value in retained
            if str(value.get("archive_date") or "")
        }
        missing_dates = [
            value.isoformat()
            for value in target_dates
            if value.isoformat() not in retained_by_date
        ]
        target_days = (
            [retained_by_date[value] for value in sorted(target_values & set(retained_by_date))]
            if target_values
            else list(retained)
        )
        incomplete = [value for value in target_days if not bool(value.get("is_complete"))]
        source_due = [
            value
            for value in target_days
            if bool(value.get("source_check_due"))
            and (
                str(value.get("archive_date") or "") not in strict_catchup_values
                or str(value.get("archive_date") or "") in schedule_values
                or not bool(value.get("is_complete"))
            )
        ]
        network_days = {
            str(value.get("archive_date") or "")
            for value in target_days
            if (
                not bool(value.get("is_complete"))
                and (
                    bool(value.get("needs_network"))
                    or bool(value.get("source_check_due"))
                )
            )
            or (
                bool(value.get("is_complete"))
                and bool(value.get("source_check_due"))
                and (
                    str(value.get("archive_date") or "")
                    not in strict_catchup_values
                    or str(value.get("archive_date") or "") in schedule_values
                )
            )
        }
        network_days.update(missing_dates)
        local_processing_dates = {
            str(value.get("archive_date") or "")
            for value in incomplete
            if not bool(value.get("needs_network"))
        }
        local_dates = sorted(retained_by_date)
        latest_local = local_dates[-1] if local_dates else ""
        latest_date = date.fromisoformat(latest_local) if latest_local else None
        days_behind = max(0, (current - latest_date).days) if latest_date else None
        expected_count = len(target_dates) if target_dates else len(retained)
        progress_points = sum(
            max(0, min(100, int(value.get("pipeline_percent") or 0)))
            for value in target_days
        )
        progress_percent = (
            round(progress_points / expected_count)
            if expected_count
            else 0
        )
        names = [
            str(value.get("feed_name") or "").strip()
            for value in retained
            if str(value.get("feed_name") or "").strip()
        ]
        feed_name = str((schedule or {}).get("feed_name") or "").strip()
        if not feed_name:
            feed_name = str((catchup or {}).get("feed_name") or "").strip()
        if not feed_name and names:
            feed_name = names[0]
        if not feed_name:
            feed_name = f"Feed {feed_id}"
        checked_values = sorted(
            str(value.get("source_checked_at") or "")
            for value in retained
            if str(value.get("source_checked_at") or "")
        )
        known_source_blocks = sum(
            int(value.get("known_source_count") or 0)
            for value in target_days
        )
        retained_source_blocks = sum(
            int(value.get("retained_source_count") or 0)
            for value in target_days
        )
        missing_source_blocks = sum(
            int(value.get("missing_source_count") or 0)
            for value in target_days
        )
        scheduled = bool(schedule and schedule.get("enabled", True))
        backlog_count = len(set(missing_dates) | network_days | local_processing_dates)
        catchup_saved = catchup is not None
        if catchup_saved and backlog_count == 0:
            status = "Saved catch-up range is complete"
        elif catchup_saved:
            status = f"{backlog_count} catch-up day{'s' if backlog_count != 1 else ''} need work"
        elif scheduled and backlog_count == 0:
            status = "Caught up for the scheduled range"
        elif scheduled:
            status = f"{backlog_count} scheduled day{'s' if backlog_count != 1 else ''} need work"
        elif incomplete:
            status = f"{len(incomplete)} retained day{'s' if len(incomplete) != 1 else ''} need local work"
        elif latest_date is None:
            status = "No retained days yet"
        elif days_behind == 0:
            status = "Newest retained day is today"
        else:
            status = f"Newest retained day is {days_behind} day{'s' if days_behind != 1 else ''} old"
        results.append(
            {
                "feed_id": feed_id,
                "feed_name": feed_name,
                "scheduled": scheduled,
                "schedule_enabled": bool(schedule and schedule.get("enabled", True)),
                "catch_up_saved": catchup_saved,
                "catch_up_through_current": catchup_through_current,
                "catch_up_start_date": (
                    str(catchup.get("start_date") or "") if catchup else ""
                ),
                "catch_up_end_date": (
                    str(catchup.get("end_date") or "") if catchup else ""
                ),
                "target_start_date": target_dates[0].isoformat() if target_dates else "",
                "target_end_date": target_dates[-1].isoformat() if target_dates else "",
                "target_day_count": expected_count,
                "retained_day_count": len(target_days) if target_dates else len(retained),
                "ready_day_count": sum(bool(value.get("is_complete")) for value in target_days),
                "incomplete_day_count": len(incomplete),
                "missing_day_count": len(missing_dates),
                "missing_dates": missing_dates,
                "source_check_due_count": len(source_due),
                "network_day_count": len(network_days),
                "local_processing_day_count": len(local_processing_dates),
                "backlog_count": backlog_count,
                "latest_local_date": latest_local,
                "days_behind_today": days_behind,
                "last_source_check_at": checked_values[-1] if checked_values else "",
                "known_source_block_count": known_source_blocks,
                "retained_source_block_count": retained_source_blocks,
                "missing_source_block_count": missing_source_blocks,
                "progress_percent": progress_percent,
                "status": status,
            }
        )
    return sorted(
        results,
        key=lambda value: (
            not (
                bool(value["scheduled"])
                or bool(value.get("catch_up_saved"))
            ),
            -int(value["backlog_count"]),
            str(value["feed_name"]).lower(),
        ),
    )


def _missing_resume_day(
    feed_id: str,
    feed_name: str,
    archive_date: str,
) -> dict[str, Any]:
    return {
        "feed_id": feed_id,
        "feed_name": feed_name,
        "archive_date": archive_date,
        "status": "Scheduled day missing",
        "status_detail": "No retained source audio exists for this scheduled day",
        "next_step": "Acquire scheduled archive day",
        "primary_action": "resume_download",
        "pipeline_percent": 0,
        "pipeline_summary": "Audio missing  ·  Not combined  ·  Not transcribed  ·  Not diarized  ·  Not analyzed",
        "is_complete": False,
        "needs_local_processing": False,
        "needs_network": True,
        "source_check_due": True,
        "scheduled_missing": True,
    }


def _saved_through_current_contains(
    day: dict[str, Any],
    catchups: list[dict[str, Any]],
    schedules: list[dict[str, Any]],
    current: date,
) -> bool:
    """Return whether a complete day belongs only to a strict saved catch-up."""

    feed_id = str(day.get("feed_id") or "")
    archive_value = str(day.get("archive_date") or "")
    if not feed_id or not archive_value:
        return False
    for schedule in schedules:
        if str(schedule.get("feed_id") or "") != feed_id:
            continue
        if archive_value in {
            value.isoformat() for value in _schedule_target_dates(schedule, current)
        }:
            return False
    for catchup in catchups:
        if (
            str(catchup.get("feed_id") or "") != feed_id
            or not bool(catchup.get("through_current", False))
        ):
            continue
        try:
            start_date = date.fromisoformat(str(catchup.get("start_date") or ""))
            archive_date = date.fromisoformat(archive_value)
        except ValueError:
            continue
        if start_date <= archive_date <= current:
            return True
    return False


def build_library_resume_plan(
    days: list[dict[str, Any]],
    quota_status: dict[str, Any],
    schedules: list[dict[str, Any]] | None = None,
    catchups: list[dict[str, Any]] | None = None,
    *,
    today: date | None = None,
    requested_feed_id: str = "",
    requested_start_date: date | None = None,
    requested_end_date: date | None = None,
    requested_through_current: bool = False,
) -> dict[str, Any]:
    """Return a deterministic local-first queue without starting any work."""

    normalized_feed_id = str(requested_feed_id or "").strip()
    current = today or date.today()
    if requested_through_current and requested_end_date is not None:
        raise ValueError("Through-current catch-up does not accept a fixed end date.")
    if requested_start_date and requested_start_date > current:
        raise ValueError("Catch-up start date cannot be in the future.")
    resolved_requested_end = (
        current
        if requested_through_current and requested_start_date
        else requested_end_date
    )
    has_requested_range = bool(
        normalized_feed_id and requested_start_date and resolved_requested_end
    )
    if (
        any(
            (
                normalized_feed_id,
                requested_start_date,
                requested_end_date,
                requested_through_current,
            )
        )
        and not has_requested_range
    ):
        raise ValueError(
            "Feed ID and start date are required for through-current catch-up."
            if requested_through_current
            else "Feed ID, start date, and end date are all required for catch-up."
        )
    if normalized_feed_id and not normalized_feed_id.isdigit():
        raise ValueError("Feed ID must contain only digits.")
    requested_ranges = (
        {normalized_feed_id: (requested_start_date, resolved_requested_end)}
        if has_requested_range
        else None
    )
    coverage = build_library_feed_coverage(
        days,
        schedules,
        catchups,
        today=today,
        requested_ranges=requested_ranges,
    )
    if has_requested_range:
        coverage = [
            value for value in coverage
            if str(value.get("feed_id") or "") == normalized_feed_id
        ]
    candidates = [
        dict(value)
        for value in days
        if (
            not has_requested_range
            or (
                str(value.get("feed_id") or "") == normalized_feed_id
                and requested_start_date.isoformat()
                <= str(value.get("archive_date") or "")
                <= resolved_requested_end.isoformat()
            )
        )
        and (
            not bool(value.get("is_complete"))
            or (
                bool(value.get("source_check_due"))
                and not has_requested_range
                and not _saved_through_current_contains(
                    value,
                    catchups or [],
                    schedules or [],
                    current,
                )
            )
        )
    ]
    existing_keys = {
        (str(value.get("feed_id") or ""), str(value.get("archive_date") or ""))
        for value in candidates
    }
    for feed in coverage:
        for archive_value in feed["missing_dates"]:
            key = (str(feed["feed_id"]), archive_value)
            if key in existing_keys:
                continue
            candidates.append(
                _missing_resume_day(
                    str(feed["feed_id"]),
                    str(feed["feed_name"]),
                    archive_value,
                )
            )
            existing_keys.add(key)
    for value in candidates:
        value["needs_local_processing"] = bool(
            not value.get("is_complete")
            and not value.get("needs_network")
        )
        if bool(value.get("source_check_due")):
            value["needs_network"] = True
            if bool(value.get("is_complete")):
                value.update(
                    {
                        "status": "Source refresh due",
                        "status_detail": (
                            "The retained processing is complete, but the latest "
                            "source listing snapshot may have grown"
                        ),
                        "next_step": "Check for new source audio",
                        "primary_action": "resume_download",
                    }
                )
    ordered = sorted(
        candidates,
        key=lambda value: (
            not bool(value.get("needs_local_processing")),
            bool(value.get("needs_network")),
            -date.fromisoformat(str(value.get("archive_date") or "")).toordinal(),
            str(value.get("feed_id") or ""),
        ),
    )
    return {
        "days": ordered,
        "feeds": coverage,
        "local_count": sum(
            bool(value.get("needs_local_processing")) for value in ordered
        ),
        "network_count": sum(bool(value.get("needs_network")) for value in ordered),
        "quota": dict(quota_status),
        "scope_feed_id": normalized_feed_id if has_requested_range else "",
        "scope_start_date": (
            requested_start_date.isoformat() if has_requested_range else ""
        ),
        "scope_end_date": (
            resolved_requested_end.isoformat() if has_requested_range else ""
        ),
        "scope_through_current": bool(
            has_requested_range and requested_through_current
        ),
    }


def compact_archive_date_ranges(values: Sequence[str]) -> list[str]:
    """Collapse ISO archive dates into stable, human-readable contiguous ranges."""

    parsed: list[date] = []
    for raw in values:
        try:
            parsed.append(date.fromisoformat(str(raw)))
        except ValueError:
            continue
    ordered = sorted(set(parsed))
    if not ordered:
        return []
    ranges: list[str] = []
    start = ordered[0]
    end = ordered[0]
    for value in ordered[1:]:
        if value == end + timedelta(days=1):
            end = value
            continue
        ranges.append(
            start.isoformat()
            if start == end
            else f"{start.isoformat()} through {end.isoformat()}"
        )
        start = value
        end = value
    ranges.append(
        start.isoformat()
        if start == end
        else f"{start.isoformat()} through {end.isoformat()}"
    )
    return ranges


def describe_archive_date_ranges(
    values: Sequence[str],
    *,
    max_ranges: int = 12,
) -> str:
    """Describe large date sets without placing hundreds of dates in a prompt."""

    ranges = compact_archive_date_ranges(values)
    visible = ranges[: max(1, int(max_ranges))]
    description = ", ".join(visible) or "none"
    remaining = len(ranges) - len(visible)
    if remaining > 0:
        description += f", plus {remaining} more range{'s' if remaining != 1 else ''}"
    return description


def entire_archive_feed_range(
    days: Sequence[dict[str, Any]],
    feed_id: str,
) -> tuple[date, date]:
    """Return the earliest-to-latest locally retained span for one feed."""

    normalized_feed_id = str(feed_id or "").strip()
    if not normalized_feed_id.isdigit():
        raise ValueError("Feed ID must contain only digits.")
    retained_dates: list[date] = []
    for value in days:
        if str(value.get("feed_id") or "") != normalized_feed_id:
            continue
        retention_keys = {
            "raw_file_count",
            "has_combined",
            "has_transcript",
            "has_imported_transcript",
        }
        if retention_keys.intersection(value) and not (
            int(value.get("raw_file_count") or 0) > 0
            or bool(value.get("has_combined"))
            or bool(value.get("has_transcript"))
            or bool(value.get("has_imported_transcript"))
        ):
            # Scheduled catch-up coverage deliberately includes absent dates so
            # Resume can acquire them.  Those zero-file placeholders are gaps,
            # not boundaries of the "entire downloaded feed" question scope.
            continue
        try:
            retained_dates.append(date.fromisoformat(str(value.get("archive_date") or "")))
        except ValueError:
            continue
    if not retained_dates:
        raise ValueError(
            f"No locally retained feed days exist for feed {normalized_feed_id}."
        )
    return min(retained_dates), max(retained_dates)


def build_archive_question_coverage(
    days: list[dict[str, Any]],
    feed_id: str,
    start_date: date,
    end_date: date,
) -> dict[str, Any]:
    """Describe which retained days can safely support a range question."""

    normalized_feed_id = str(feed_id or "").strip()
    if not normalized_feed_id.isdigit():
        raise ValueError("Feed ID must contain only digits.")
    if start_date > end_date:
        raise ValueError("Question start date must not be after its end date.")
    requested_dates = [
        start_date + timedelta(days=offset)
        for offset in range((end_date - start_date).days + 1)
    ]
    states = {
        str(value.get("archive_date") or ""): value
        for value in days
        if str(value.get("feed_id") or "") == normalized_feed_id
        and start_date.isoformat()
        <= str(value.get("archive_date") or "")
        <= end_date.isoformat()
    }
    audio_dates: list[str] = []
    question_ready_dates: list[str] = []
    analyzed_dates: list[str] = []
    local_processing_dates: list[str] = []
    partial_audio_dates: list[str] = []
    missing_audio_dates: list[str] = []
    for requested_date in requested_dates:
        archive_value = requested_date.isoformat()
        state = states.get(archive_value)
        # A downloaded source block is retained audio even before the local
        # combine stage has produced the continuous daily file.  Treating only
        # combined files as audio made month/whole-feed coverage incorrectly
        # call downloaded raw-only days "No retained audio" while those
        # days were visibly present in the Library.  They are unavailable for
        # questions until local processing finishes, but they are not missing
        # downloads.
        has_audio = bool(
            state
            and (
                state.get("has_combined")
                or int(state.get("raw_file_count") or 0) > 0
            )
        )
        question_ready = bool(
            state
            and state.get("has_transcript")
            and state.get("has_imported_transcript")
        )
        if has_audio:
            audio_dates.append(archive_value)
        if question_ready:
            question_ready_dates.append(archive_value)
            if bool(state and state.get("has_analysis")):
                analyzed_dates.append(archive_value)
        elif has_audio and not bool(state and state.get("needs_network")):
            local_processing_dates.append(archive_value)
        elif has_audio:
            partial_audio_dates.append(archive_value)
        else:
            missing_audio_dates.append(archive_value)
    acquisition_values = set(partial_audio_dates) | set(missing_audio_dates)
    unavailable_values = set(local_processing_dates) | acquisition_values
    acquisition_needed_dates = [
        value.isoformat()
        for value in requested_dates
        if value.isoformat() in acquisition_values
    ]
    unavailable_dates = [
        value.isoformat()
        for value in requested_dates
        if value.isoformat() in unavailable_values
    ]
    requested_count = len(requested_dates)
    summary = (
        f"{len(question_ready_dates)}/{requested_count} requested days are "
        f"question-ready; audio is retained for {len(audio_dates)}/{requested_count}."
    )
    if local_processing_dates:
        summary += (
            f" Local processing needed for {len(local_processing_dates)} day(s): "
            + describe_archive_date_ranges(local_processing_dates)
            + "."
        )
    if partial_audio_dates:
        summary += (
            f" Additional archive acquisition needed for {len(partial_audio_dates)} "
            "partial day(s): "
            + describe_archive_date_ranges(partial_audio_dates)
            + "."
        )
    if missing_audio_dates:
        summary += (
            f" No retained audio for {len(missing_audio_dates)} day(s): "
            + describe_archive_date_ranges(missing_audio_dates)
            + "."
        )
    return {
        "scope": "range",
        "feed_id": normalized_feed_id,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "requested_day_count": requested_count,
        "audio_day_count": len(audio_dates),
        "question_ready_day_count": len(question_ready_dates),
        "analyzed_day_count": len(analyzed_dates),
        "audio_dates": audio_dates,
        "audio_ranges": compact_archive_date_ranges(audio_dates),
        "question_ready_dates": question_ready_dates,
        "question_ready_ranges": compact_archive_date_ranges(question_ready_dates),
        "analyzed_dates": analyzed_dates,
        "analyzed_ranges": compact_archive_date_ranges(analyzed_dates),
        "local_processing_dates": local_processing_dates,
        "local_processing_ranges": compact_archive_date_ranges(
            local_processing_dates
        ),
        "partial_audio_dates": partial_audio_dates,
        "partial_audio_ranges": compact_archive_date_ranges(partial_audio_dates),
        "missing_audio_dates": missing_audio_dates,
        "missing_audio_ranges": compact_archive_date_ranges(missing_audio_dates),
        "acquisition_needed_dates": acquisition_needed_dates,
        "acquisition_needed_ranges": compact_archive_date_ranges(
            acquisition_needed_dates
        ),
        "unavailable_dates": unavailable_dates,
        "unavailable_ranges": compact_archive_date_ranges(unavailable_dates),
        "complete_coverage": not unavailable_dates,
        "summary": summary,
    }


def completed_library_catchup_feed_ids(
    days: list[dict[str, Any]],
    catchups: list[dict[str, Any]],
    *,
    today: date | None = None,
) -> list[str]:
    """Identify saved explicit ranges whose every day is locally complete/current."""

    current = today or date.today()
    completed: list[str] = []
    for catchup in catchups:
        feed_id = str(catchup.get("feed_id") or "").strip()
        try:
            start_date = date.fromisoformat(str(catchup.get("start_date") or ""))
            end_date = date.fromisoformat(str(catchup.get("end_date") or ""))
        except ValueError:
            continue
        if bool(catchup.get("through_current", False)):
            end_date = current
        coverage = build_library_feed_coverage(
            days,
            today=current,
            requested_ranges={feed_id: (start_date, end_date)},
        )
        if coverage and int(coverage[0].get("backlog_count") or 0) == 0:
            completed.append(feed_id)
    return completed


def delete_local_library_feed(
    output_dir: str | Path,
    database_path: str | Path,
    feed_id: str,
    *,
    remove_schedule: bool = True,
) -> dict[str, Any]:
    """Atomically detach one feed directory, then remove its database evidence."""

    normalized = str(feed_id or "").strip()
    if not normalized.isdigit():
        raise ValueError("Feed ID must contain only digits.")
    output_root = Path(output_dir).expanduser().resolve()
    feed_directory = output_root / normalized
    tombstone: Path | None = None
    if feed_directory.exists():
        is_junction = getattr(feed_directory, "is_junction", lambda: False)
        if feed_directory.is_symlink() or is_junction() or not feed_directory.is_dir():
            raise ValueError(
                "The feed library path is not a regular directory and was not deleted."
            )
        if feed_directory.resolve().parent != output_root:
            raise ValueError("The feed library path escapes the selected library root.")
        tombstone = output_root / f".deleting-{normalized}-{uuid.uuid4().hex}"
        last_error: PermissionError | None = None
        for delay in DELETE_DETACH_RETRY_SECONDS:
            if delay:
                time.sleep(delay)
            try:
                feed_directory.rename(tombstone)
                last_error = None
                break
            except PermissionError as exc:
                last_error = exc
        if last_error is not None:
            raise PermissionError(
                f"Feed {normalized} is still in use by audio playback, an archive "
                "worker, File Explorer, or another process. Playback was released "
                "and the detach was retried, but Windows still denied it. Close "
                "anything using that feed folder and choose Delete feed again; no "
                "library records or files were removed."
            ) from last_error

    try:
        with AnalysisStore(database_path) as store:
            result = store.delete_library_feed(
                normalized,
                remove_schedule=remove_schedule,
            )
    except Exception:
        if tombstone is not None and tombstone.exists() and not feed_directory.exists():
            tombstone.rename(feed_directory)
        raise

    cleanup_pending = False
    if tombstone is not None:
        try:
            shutil.rmtree(tombstone)
        except OSError:
            cleanup_pending = True
    return {
        "feed_id": normalized,
        "directory_deleted": tombstone is not None,
        "cleanup_pending": cleanup_pending,
        **result,
    }


def cleanup_pending_library_deletions(output_dir: str | Path) -> int:
    """Retry cleanup of directories detached by a completed feed deletion."""

    output_root = Path(output_dir).expanduser().resolve()
    if not output_root.is_dir():
        return 0
    removed = 0
    for candidate in output_root.iterdir():
        if not PENDING_DELETE_PATTERN.match(candidate.name):
            continue
        is_junction = getattr(candidate, "is_junction", lambda: False)
        if candidate.is_symlink() or is_junction() or not candidate.is_dir():
            continue
        try:
            shutil.rmtree(candidate)
        except OSError:
            continue
        removed += 1
    return removed


def _transcript_tail(path: str | Path) -> str:
    transcript = Path(path)
    if not transcript.is_file():
        return ""
    try:
        with transcript.open("rb") as handle:
            handle.seek(max(0, transcript.stat().st_size - 512 * 1024))
            return handle.read().decode("utf-8", errors="ignore").lower()
    except OSError:
        return ""


def transcript_has_diarization(path: str | Path) -> bool:
    """Confirm that diarization finished, rather than merely being requested."""

    tail = _transcript_tail(path)
    if not tail:
        return False
    if '"diarization_completed": true' in tail:
        return True
    requested = '"diarization_requested": true' in tail
    model_recorded = '"diarization_model"' in tail
    completion_evidence = (
        '"speaker_turns"' in tail
        or '"speaker": "speaker_' in tail
        or '"speaker":"speaker_' in tail
    )
    return requested and model_recorded and completion_evidence


def transcript_diarization_engine(path: str | Path) -> str:
    if not transcript_has_diarization(path):
        return ""
    tail = _transcript_tail(path)
    match = re.search(r'"diarization_engine"\s*:\s*"([^"]+)"', tail)
    if match:
        try:
            return normalize_diarization_engine(match.group(1))
        except ValueError:
            return match.group(1)
    if "pyannote/speaker-diarization-community-1" in tail:
        return COMMUNITY_DIARIZATION_ENGINE
    if "pyannote-segmentation-3.0-int8+nemo-titanet-small" in tail:
        return PORTABLE_DIARIZATION_ENGINE
    # Completed transcripts written before engine metadata existed used
    # Community-1 exclusively.
    return COMMUNITY_DIARIZATION_ENGINE


def transcript_satisfies_diarization(
    path: str | Path, requested_engine: str
) -> bool:
    actual = transcript_diarization_engine(path)
    return diarization_engine_satisfies(actual, requested_engine)


def _friendly_feed_names(store: AnalysisStore) -> dict[str, str]:
    names: dict[str, str] = {}
    if hasattr(store, "list_feed_catalog"):
        for feed in store.list_feed_catalog():
            feed_id = str(feed.get("feed_id") or "")
            name = str(feed.get("name") or "")
            if feed_id and name:
                names[feed_id] = name
    for profile in store.list_area_profiles():
        for feed in profile.get("feeds", []):
            feed_id = str(feed.get("feed_id") or "")
            name = str(feed.get("name") or "")
            if feed_id and name:
                names.setdefault(feed_id, name)
    return names


def _manifest_feed_name(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("feed_name") or "").strip()[:200]


def _archive_source_snapshot(
    day_directory: Path,
    feed_id: str,
    archive_date: date,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Describe the last authenticated listing snapshot using local files only."""

    path = day_directory / ARCHIVE_CACHE_COMPLETION_FILENAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        payload = None
    valid = bool(
        isinstance(payload, dict)
        and payload.get("schema_version")
        == ARCHIVE_CACHE_COMPLETION_SCHEMA_VERSION
        and str(payload.get("feed_id") or "") == str(feed_id)
        and str(payload.get("archive_date") or "") == archive_date.isoformat()
        and isinstance(payload.get("archive_ids"), list)
    )
    archive_ids: list[str] = []
    checked_at = ""
    checked_value: datetime | None = None
    if valid and isinstance(payload, dict):
        raw_ids = payload.get("archive_ids") or []
        if (
            len(raw_ids) <= 1_000
            and all(
                isinstance(value, str) and value and len(value) <= 200
                for value in raw_ids
            )
            and len(set(raw_ids)) == len(raw_ids)
        ):
            archive_ids = list(raw_ids)
        else:
            valid = False
        try:
            timestamp = float(payload.get("completed_at_unix") or 0)
            if timestamp > 0:
                checked_value = datetime.fromtimestamp(timestamp, timezone.utc)
                checked_at = checked_value.isoformat(timespec="seconds")
        except (OSError, OverflowError, TypeError, ValueError):
            checked_value = None
            checked_at = ""
    retained = 0
    if valid:
        retained = sum(
            path is not None
            for path in cached_archives_for_ids(day_directory, feed_id, archive_ids)
        )
    current_local = now or datetime.now().astimezone()
    if current_local.tzinfo is None:
        current_local = current_local.astimezone()
    current_utc = current_local.astimezone(timezone.utc)
    source_check_due = archive_date >= current_local.date() and (
        checked_value is None
        or current_utc - checked_value >= CURRENT_DAY_SOURCE_REFRESH
    )
    return {
        "known_source_count": len(archive_ids) if valid else 0,
        "retained_source_count": retained if valid else 0,
        "missing_source_count": max(0, len(archive_ids) - retained) if valid else 0,
        "source_checked_at": checked_at,
        "source_snapshot_complete": bool(valid and retained == len(archive_ids)),
        "source_check_due": source_check_due,
    }


def _transcript_is_current(audio: Path, transcript: Path) -> bool:
    """Reject results created before the combined recording was refreshed."""

    try:
        return (
            audio.is_file()
            and transcript.is_file()
            and transcript.stat().st_size > 0
            and transcript.stat().st_mtime_ns >= audio.stat().st_mtime_ns
        )
    except OSError:
        return False


def _external_artifacts(
    combined: Path,
    manifest: Path,
    transcript: Path,
    raw_files: list[Path],
) -> list[Path]:
    """Select one imported feed-day's files without counting sibling data."""

    paths = [
        combined,
        manifest,
        transcript,
        transcript.with_suffix(".txt"),
        *raw_files,
    ]
    stem = combined.stem
    for directory in (transcript.parent, transcript.parent / ".cache"):
        if not directory.is_dir():
            continue
        paths.extend(directory.glob(f"{stem}*"))
        paths.extend(directory.glob(f".{stem}*"))
    return paths


def _state_for_day(
    output_root: Path,
    feed_id: str,
    archive_date: date,
    stored: dict[str, Any] | None,
    feed_name: str,
) -> dict[str, Any]:
    expected_day_directory = (
        output_root / feed_id / archive_date.strftime("%Y%m%d")
    )
    day_directory = expected_day_directory
    stem = f"combined_{feed_id}_{archive_date:%Y%m%d}"
    combined = day_directory / f"{stem}.mp3"
    manifest = day_directory / f"{stem}.manifest.json"
    transcript = day_directory / "transcripts" / f"{stem}.json"

    if stored:
        stored_audio = Path(str(stored.get("audio_path") or ""))
        stored_transcript = Path(str(stored.get("transcript_path") or ""))
        stored_manifest = Path(str(stored.get("manifest_path") or ""))
        if not combined.is_file() and stored_audio.is_file():
            combined = stored_audio
            day_directory = combined.parent
        if not transcript.is_file() and stored_transcript.is_file():
            transcript = stored_transcript
        if not manifest.is_file() and stored_manifest.is_file():
            manifest = stored_manifest

    expected_root = expected_day_directory.resolve()
    external_layout = False
    for artifact in (combined, manifest, transcript):
        if not artifact.is_file():
            continue
        try:
            artifact.resolve().relative_to(expected_root)
        except (OSError, ValueError):
            external_layout = True
            break

    raw_files = []
    if day_directory.is_dir():
        for path in day_directory.glob("*.mp3"):
            match = RAW_ARCHIVE_PATTERN.match(path.name)
            if match and match.group(1) == feed_id:
                raw_files.append(path)
        raw_files.sort()
    source_snapshot = _archive_source_snapshot(
        day_directory,
        feed_id,
        archive_date,
    )
    has_combined_file = combined.is_file() and combined.stat().st_size > 0
    collapsed_identity_count = collapsed_archive_identity_count(
        day_directory,
        feed_id,
    )
    # Imported/legacy combined recordings may legitimately have no retained raw
    # blocks.  When raw blocks are present, however, the manifest must describe
    # that exact set.  Otherwise an interrupted refresh can leave an older MP3,
    # transcript, and analysis that look complete even though newer blocks are
    # waiting to be combined.
    has_stale_combined = bool(
        has_combined_file
        and (
            collapsed_identity_count > 0
            or (
                raw_files
                and not combined_output_is_current(combined, manifest, raw_files)
            )
        )
    )
    has_combined = has_combined_file and not has_stale_combined
    transcript_file_exists = bool(
        transcript.is_file() and transcript.stat().st_size > 0
    )
    has_transcript = bool(
        has_combined and _transcript_is_current(combined, transcript)
    )
    has_stale_transcript = bool(
        has_combined and transcript_file_exists and not has_transcript
    )
    stored_matches_transcript = False
    if has_transcript and stored:
        expected_transcript_sha256 = str(
            stored.get("transcript_sha256") or ""
        )
        try:
            stored_matches_transcript = bool(
                expected_transcript_sha256
                and sha256_file(transcript) == expected_transcript_sha256
            )
        except OSError:
            stored_matches_transcript = False
    has_diarization = has_transcript and (
        transcript_has_diarization(transcript)
        or bool(stored_matches_transcript and stored.get("has_diarization"))
    )
    diarization_engine = (
        transcript_diarization_engine(transcript) if has_diarization else ""
    )
    speaker_upgrade_available = (
        diarization_engine == PORTABLE_DIARIZATION_ENGINE
    )
    has_saved_analysis = bool(stored and stored.get("has_summary"))
    analysis_prompt_version = (
        str(stored.get("summary_prompt_version") or "") if stored else ""
    )
    summary_transcript_sha256 = (
        str(stored.get("summary_transcript_sha256") or "")
        if stored
        else ""
    )
    has_analysis = bool(
        has_transcript
        and stored_matches_transcript
        and has_saved_analysis
        and analysis_prompt_version == PROMPT_VERSION
        and summary_transcript_sha256
        == str(stored.get("transcript_sha256") or "")
    )
    has_stale_analysis = bool(
        has_transcript and has_saved_analysis and not has_analysis
    )
    incident_count = (
        int(stored.get("incident_count") or 0)
        if stored_matches_transcript and has_analysis
        else 0
    )
    segment_count = (
        int(stored.get("segment_count") or 0)
        if stored_matches_transcript
        else 0
    )

    if int(source_snapshot["missing_source_count"]) > 0:
        next_step = "Restore missing source audio"
        action = "resume_download"
        status = "Archive source repair required"
        status_detail = (
            f"The last authenticated listing contained "
            f"{source_snapshot['known_source_count']} source segments, but "
            f"{source_snapshot['missing_source_count']} retained segment"
            f"{'s are' if source_snapshot['missing_source_count'] != 1 else ' is'} missing"
        )
    elif collapsed_identity_count > 0:
        next_step = "Verify & repair archive day"
        action = "resume_download"
        status = "Archive timeline repair required"
        status_detail = (
            f"{collapsed_identity_count} archive timeline position"
            f"{'s were' if collapsed_identity_count != 1 else ' was'} "
            "collapsed onto another retained filename by an older cache; "
            "the previous recording is preserved but hidden until the exact "
            "missing source positions are restored"
        )
    elif has_stale_combined:
        next_step = "Refresh archive day"
        action = "resume_download"
        status = "New audio pending combine"
        status_detail = (
            f"{len(raw_files)} retained source segment"
            f"{'s' if len(raw_files) != 1 else ''} do not match the older "
            "combined timeline; the older recording is preserved"
        )
    elif not has_combined:
        next_step = "Resume archive download"
        action = "resume_download"
        status = "Needs download"
        status_detail = (
            f"{len(raw_files)} local segment{'s' if len(raw_files) != 1 else ''}; "
            "completeness has not been verified"
            if raw_files
            else "No usable combined audio was found"
        )
    elif has_stale_transcript:
        next_step = "Update local transcript"
        action = "continue_local"
        status = "Transcript update required"
        status_detail = (
            "Combined audio changed; the previous transcript is preserved but "
            "hidden until local processing updates it"
        )
    elif not has_transcript:
        next_step = "Transcribe locally"
        action = "continue_local"
        status = "Audio ready"
        status_detail = "Combined audio is ready for local transcription"
    elif not has_diarization:
        next_step = "Add speaker labels"
        action = "continue_local"
        status = "Transcript ready"
        status_detail = (
            "Transcript exists; speaker labels can run without repeating transcription"
        )
    elif has_stale_analysis:
        next_step = "Re-run evidence analysis"
        action = "continue_local"
        status = "Analysis update available"
        status_detail = (
            "Saved results predate the current evidence rules; retained audio, "
            "transcript, and speaker labels will be reused"
        )
    elif not has_analysis:
        next_step = "Extract and summarize incidents"
        action = "continue_local"
        status = "Speakers ready"
        status_detail = "Diarized transcript is ready for local analysis"
    else:
        next_step = "Open review"
        action = "open_review"
        status = "Ready to review"
        status_detail = f"{incident_count} extracted incident{'s' if incident_count != 1 else ''}"

    completed_stages = sum(
        [bool(raw_files or has_combined), has_combined, has_transcript, has_diarization, has_analysis]
    )
    stage_parts = [
        f"Audio {len(raw_files)} segment{'s' if len(raw_files) != 1 else ''}"
        if raw_files
        else "Audio retained" if has_combined else "Audio missing",
        "Combined" if has_combined else "Not combined",
        f"Transcript {segment_count:,} segments"
        if has_transcript and segment_count
        else "Transcribed"
        if has_transcript
        else "Transcript update required"
        if has_stale_transcript
        else "Not transcribed",
        "Diarized" if has_diarization else "Not diarized",
        f"Analyzed {incident_count} incidents"
        if has_analysis
        else "Analysis update required"
        if has_stale_analysis
        else "Not analyzed",
    ]
    resolved_feed_name = feed_name or _manifest_feed_name(manifest)
    if external_layout:
        storage_bytes, working_storage_bytes = files_storage_usage(
            _external_artifacts(combined, manifest, transcript, raw_files)
        )
    else:
        storage_bytes, working_storage_bytes = directory_storage_usage(
            day_directory
        )
    return {
        "feed_id": feed_id,
        "feed_name": resolved_feed_name or f"Feed {feed_id}",
        "archive_date": archive_date.isoformat(),
        "day_directory": str(day_directory.resolve()),
        "raw_file_count": len(raw_files),
        "collapsed_identity_count": collapsed_identity_count,
        "combined_path": str(combined.resolve()) if has_combined else "",
        "transcript_path": str(transcript.resolve()) if has_transcript else "",
        "manifest_path": str(manifest.resolve()) if manifest.is_file() else "",
        "has_combined": has_combined,
        "has_stale_combined": has_stale_combined,
        "has_transcript": has_transcript,
        "has_stale_transcript": has_stale_transcript,
        "has_imported_transcript": stored_matches_transcript,
        "has_diarization": has_diarization,
        "diarization_engine": diarization_engine,
        "diarization_quality": (
            "preview"
            if diarization_engine == PORTABLE_DIARIZATION_ENGINE
            else COMMUNITY_DIARIZATION_QUALITY
            if diarization_engine == COMMUNITY_DIARIZATION_ENGINE
            else ""
        ),
        "speaker_upgrade_available": speaker_upgrade_available,
        "has_analysis": has_analysis,
        "has_stale_analysis": has_stale_analysis,
        "analysis_prompt_version": analysis_prompt_version,
        "expected_analysis_prompt_version": PROMPT_VERSION,
        "incident_count": incident_count,
        "segment_count": segment_count,
        "storage_bytes": storage_bytes,
        "working_storage_bytes": working_storage_bytes,
        "pipeline_percent": completed_stages * 20,
        "pipeline_summary": "  ·  ".join(stage_parts),
        "status": status,
        "status_detail": status_detail,
        "next_step": next_step,
        "primary_action": action,
        "can_open_review": has_analysis,
        "is_complete": has_diarization and has_analysis,
        "needs_network": (
            not has_combined
            or int(source_snapshot["missing_source_count"]) > 0
        ),
        "known_source_count": int(source_snapshot["known_source_count"]),
        "retained_source_count": int(source_snapshot["retained_source_count"]),
        "missing_source_count": int(source_snapshot["missing_source_count"]),
        "source_checked_at": str(source_snapshot["source_checked_at"]),
        "source_snapshot_complete": bool(source_snapshot["source_snapshot_complete"]),
        "source_check_due": bool(source_snapshot["source_check_due"]),
    }


def scan_local_library(
    output_dir: str | Path = "archives",
    database_path: str | Path | None = None,
    *,
    feed_id: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[dict[str, Any]]:
    """Read fresh retained state, inspecting only the requested feed and dates."""

    if feed_id is not None and not feed_id.isdigit():
        raise ValueError("Feed ID must contain only digits.")
    output_root = Path(output_dir)
    cleanup_pending_library_deletions(output_root)
    database = Path(database_path) if database_path else output_root / "broadcastify-analysis.sqlite3"
    with AnalysisStore(database) as store:
        stored_days = {
            (str(value["feed_id"]), date.fromisoformat(str(value["archive_date"]))): value
            for value in store.list_days(
                feed_id, start_date=start_date, end_date=end_date,
            )
        }
        feed_names = _friendly_feed_names(store)

    keys = set(stored_days)
    if output_root.is_dir():
        # Select directories before walking or hashing their contents. A day
        # report should not depend on the size or availability of other feeds.
        feed_directories = (
            (output_root / feed_id,) if feed_id is not None else output_root.iterdir()
        )
        for feed_directory in feed_directories:
            if not feed_directory.is_dir() or not feed_directory.name.isdigit():
                continue
            for day_directory in feed_directory.iterdir():
                if not day_directory.is_dir() or not DAY_DIRECTORY_PATTERN.match(day_directory.name):
                    continue
                try:
                    archive_date = date.fromisoformat(
                        f"{day_directory.name[:4]}-{day_directory.name[4:6]}-{day_directory.name[6:]}"
                    )
                except ValueError:
                    continue
                if (start_date is None or archive_date >= start_date) and (
                    end_date is None or archive_date <= end_date
                ):
                    keys.add((feed_directory.name, archive_date))

    results = []
    for feed_id, archive_date in keys:
        day_directory = (
            output_root / feed_id / archive_date.strftime("%Y%m%d")
        )
        cleanup_orphaned_audio_work_files(day_directory)
        results.append(
            _state_for_day(
                output_root,
                feed_id,
                archive_date,
                stored_days.get((feed_id, archive_date)),
                feed_names.get(feed_id, ""),
            )
        )
    return sorted(results, key=lambda value: (value["archive_date"], value["feed_id"]), reverse=True)


def require_current_range_evidence(
    store: AnalysisStore,
    feed_ids: list[str],
    start_date: date,
    end_date: date,
    *,
    require_analysis: bool,
    purpose: str,
    archive_dates: Sequence[str] | None = None,
) -> None:
    """Block DB consumers when retained files have moved to a newer revision."""

    normalized = list(dict.fromkeys(str(value) for value in feed_ids if str(value)))
    allowed_dates = (
        {str(value) for value in archive_dates if str(value)}
        if archive_dates is not None
        else None
    )
    relevant: set[tuple[str, str]] = set()
    days: dict[tuple[str, str], dict[str, Any]] = {}
    for feed_id in normalized:
        for day in store.list_days(feed_id, start_date=start_date, end_date=end_date):
            archive_value = str(day["archive_date"])
            if (
                start_date.isoformat() <= archive_value <= end_date.isoformat()
                and (allowed_dates is None or archive_value in allowed_dates)
            ):
                key = (feed_id, archive_value)
                days[key] = day
                if not require_analysis and (
                    int(day.get("segment_count") or 0) > 0
                    or int(day.get("passage_count") or 0) > 0
                ):
                    relevant.add(key)

    if require_analysis:
        for key, day in days.items():
            if (
                bool(day.get("has_summary"))
                and str(day.get("summary_prompt_version") or "")
                == PROMPT_VERSION
            ):
                relevant.add(key)
        for incident in store.get_incidents_for_feeds(
            normalized,
            start_date,
            end_date,
            prompt_version=PROMPT_VERSION,
        ):
            if (
                allowed_dates is not None
                and str(incident["archive_date"]) not in allowed_dates
            ):
                continue
            relevant.add(
                (
                    str(incident["feed_id"]),
                    str(incident["archive_date"]),
                )
            )

    stale: list[tuple[str, str]] = []
    for key in sorted(relevant):
        # Validate only evidence this consumer can actually use, against the
        # live files. No cross-request cache may hide a rewritten transcript.
        state = _state_for_day(
            store.path.parent, key[0], date.fromisoformat(key[1]), days.get(key), "",
        )
        current = bool(
            state
            and (
                state["has_analysis"]
                if require_analysis
                else (
                    state["has_transcript"]
                    and state["has_imported_transcript"]
                )
            )
        )
        if not current:
            stale.append(key)
    if not stale:
        return
    labels = ", ".join(
        f"feed {feed_id} on {archive_value}"
        for feed_id, archive_value in stale[:8]
    )
    if len(stale) > 8:
        labels += f", and {len(stale) - 8} more"
    raise ValueError(
        f"{purpose} is blocked because saved evidence is older than the "
        f"retained files for {labels}. Finish those local days first; no "
        "archive re-download is required."
    )


@dataclass(frozen=True)
class LocalProcessingRequest:
    feed_id: str
    archive_date: date
    output_dir: Path = Path("archives")
    model: str = "turbo"
    asr_engine: str = "auto"
    device: str = "auto"
    device_index: int = 0
    compute_type: str = "auto"
    asr_model_path: str | None = None
    diarization_engine: str = COMMUNITY_DIARIZATION_ENGINE
    diarization_device: str = "auto"
    batch_size: int = 8
    diarize: bool = True
    min_speakers: int | None = None
    max_speakers: int | None = None
    huggingface_token: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LocalProcessingRequest":
        return cls(
            feed_id=str(value["feed_id"]).strip(),
            archive_date=date.fromisoformat(str(value["archive_date"])),
            output_dir=Path(value.get("output_dir") or "archives"),
            model=str(value.get("model") or "turbo"),
            asr_engine=str(value.get("asr_engine") or "auto"),
            device=str(value.get("device") or "auto"),
            device_index=max(0, int(value.get("device_index", 0))),
            compute_type=str(value.get("compute_type") or "auto"),
            asr_model_path=(
                str(value["asr_model_path"]).strip()
                if value.get("asr_model_path")
                else None
            ),
            diarization_engine=normalize_diarization_engine(
                str(
                    value.get("diarization_engine")
                    or COMMUNITY_DIARIZATION_ENGINE
                )
            ),
            diarization_device=str(value.get("diarization_device") or "auto"),
            batch_size=max(1, int(value.get("batch_size", 8))),
            diarize=bool(value.get("diarize", True)),
            min_speakers=(
                int(value["min_speakers"])
                if value.get("min_speakers") not in {None, ""}
                else None
            ),
            max_speakers=(
                int(value["max_speakers"])
                if value.get("max_speakers") not in {None, ""}
                else None
            ),
            huggingface_token=(
                str(value["huggingface_token"]).strip()
                if value.get("huggingface_token")
                else None
            ),
        )


def prepare_local_day(
    request: LocalProcessingRequest,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    day_directory = request.output_dir / request.feed_id / request.archive_date.strftime("%Y%m%d")
    stem = f"combined_{request.feed_id}_{request.archive_date:%Y%m%d}"
    audio = day_directory / f"{stem}.mp3"
    transcript = day_directory / "transcripts" / f"{stem}.json"
    if not audio.is_file():
        raise FileNotFoundError(
            "This day has no combined local audio. Resume its archive download first."
        )

    shared = {
        "model_name": request.model,
        "asr_engine": request.asr_engine,
        "device": request.device,
        "device_index": request.device_index,
        "compute_type": request.compute_type,
        "asr_model_path": request.asr_model_path,
        "diarization_engine": request.diarization_engine,
        "diarization_device": request.diarization_device,
        "diarize": request.diarize,
        "huggingface_token": request.huggingface_token,
        "batch_size": request.batch_size,
        "min_speakers": request.min_speakers,
        "max_speakers": request.max_speakers,
    }
    operation = "reused"
    if not _transcript_is_current(audio, transcript):
        if progress:
            progress(
                f"Loading local transcription {request.model} "
                f"for {request.archive_date}…"
            )
        transcriber = LocalTranscriber(**shared)
        transcript = transcriber.transcribe_file(audio, progress=progress)
        operation = "transcribed"
    elif request.diarize and not transcript_satisfies_diarization(
        transcript, request.diarization_engine
    ):
        previous_engine = transcript_diarization_engine(transcript)
        if progress:
            progress(
                "Loading the selected speaker model without reloading transcription…"
            )
        transcriber = LocalTranscriber(**shared, load_asr=False)
        transcript = transcriber.diarize_existing_transcript(
            audio, transcript, progress=progress
        )
        operation = (
            "upgraded_diarization"
            if previous_engine == PORTABLE_DIARIZATION_ENGINE
            and request.diarization_engine == COMMUNITY_DIARIZATION_ENGINE
            else "diarized"
        )
    elif progress:
        progress("The existing local transcript already satisfies the selected stages.")

    return {
        "feed_id": request.feed_id,
        "archive_date": request.archive_date.isoformat(),
        "audio_path": str(audio.resolve()),
        "transcript_path": str(transcript.resolve()),
        "diarization_engine": transcript_diarization_engine(transcript),
        "operation": operation,
    }
