"""Cited catch-up from saved analysis; reading this view never runs a model."""
from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from .analysis import PROMPT_VERSION, format_archive_time
from .area_watch import _public_quote
from .audio import select_incident_evidence_window
from .library import scan_local_library
from .storage import AnalysisStore


def activity_scope(payload: dict[str, Any]) -> tuple[str, date, date]:
    feed_id = str(payload.get("feed_id") or "").strip()
    if not feed_id.isdigit():
        raise ValueError("Choose a retained feed.")
    start = date.fromisoformat(str(payload.get("start_date") or ""))
    end = date.fromisoformat(str(payload.get("end_date") or ""))
    if end < start or (end - start).days > 365:
        raise ValueError("Choose a date range of up to 366 days, with the start before the end.")
    return feed_id, start, end


def _saved_activity(
    store: AnalysisStore, days: dict[str, dict[str, Any]], feed_id: str, start: date, end: date,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    indexed_days = {
        str(row["archive_date"]): dict(row)
        for row in store.connection.execute(
            """SELECT d.archive_date, d.transcript_sha256,
                      s.prompt_version AS summary_prompt_version,
                      s.transcript_sha256 AS summary_transcript_sha256
               FROM feed_days d LEFT JOIN daily_summaries s ON s.day_id=d.id
               WHERE d.feed_id=? AND d.archive_date BETWEEN ? AND ?""",
            (feed_id, start.isoformat(), end.isoformat()),
        )
    }
    # A saved summary ties these incidents to the indexed transcript and prompt.
    # New discoveries may leave this evidence readable while a refresh is pending.
    eligible = {
        value for value, day in days.items()
        if day.get("has_searchable_analysis")
        and (indexed := indexed_days.get(value))
        and indexed.get("summary_prompt_version") == PROMPT_VERSION
        and indexed.get("summary_transcript_sha256") == indexed.get("transcript_sha256")
    }
    seen = store.activity_seen(feed_id, start, end)
    items = []
    for raw in store.get_incidents(feed_id, start, end, prompt_version=PROMPT_VERSION, archive_dates=sorted(eligible)):
        day = days[str(raw["archive_date"])]
        if day.get("evidence_update_pending"):
            raw["manifest_path"] = None
        evidence_start, evidence_end = select_incident_evidence_window(raw)
        quotes = []
        for segment in raw.get("evidence") or []:
            if not isinstance(segment, dict):
                continue
            try:
                left = float(segment.get("start_seconds", 0))
                right = float(segment.get("end_seconds", left))
            except (TypeError, ValueError):
                continue
            text = str(segment.get("text") or segment.get("quote") or "").strip()
            if text and right >= evidence_start and left <= evidence_end and text not in quotes:
                quotes.append(text)
        quote, _ = _public_quote(" ".join(quotes), location=str(raw.get("location") or ""))
        if not quote:
            continue  # This is a cited view, not a list of unsupported extractions.
        item = {key: raw.get(key) for key in (
            "id", "feed_id", "archive_date", "fingerprint", "event_type", "title",
            "summary", "location", "priority", "confidence", "start_seconds", "end_seconds",
        )}
        item.update(
            evidence_quote=quote,
            archive_time=format_archive_time(raw, float(raw["start_seconds"])),
            evidence_update_pending=bool(day.get("evidence_update_pending")),
        )
        # IDs and runtime timestamps change on identical analysis reruns. Only
        # meaningful displayed evidence changes should reappear as an update.
        content = {key: value for key, value in item.items() if key not in (
            "id", "confidence", "evidence_update_pending",
        )}
        digest = hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        prior = seen.get((int(raw["day_id"]), str(raw["fingerprint"])))
        if prior == digest:
            continue
        item.update(content_sha256=digest, change_kind="updated" if prior else "new", day_id=int(raw["day_id"]))
        items.append(item)
    dates = [(start + timedelta(days=index)).isoformat() for index in range((end - start).days + 1)]
    coverage = {
        "total_days": len(dates),
        "analyzed_days": len(eligible),
        "pending_dates": [value for value in dates if value not in eligible],
        "updating_dates": [value for value in sorted(eligible) if days[value].get("evidence_update_pending")],
    }
    return items, coverage


def activity_report(database: Path, output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    feed_id, start, end = activity_scope(payload)
    days = _activity_days(database, output_dir, feed_id, start, end)
    with AnalysisStore(database) as store, store.read_snapshot():
        items, coverage = _saved_activity(store, days, feed_id, start, end)
    # Bound the response and acknowledgement to the visible batch. Older unseen
    # batches remain available after marking this batch caught up.
    visible = [{key: value for key, value in item.items() if key != "day_id"} for item in items[:100]]
    return {
        "feed_id": feed_id, "start_date": start.isoformat(), "end_date": end.isoformat(),
        "items": visible, "remaining_count": len(items), "has_more": len(items) > len(visible),
        "coverage": coverage,
    }


def _activity_days(database: Path, output_dir: Path, feed_id: str, start: date, end: date) -> dict[str, dict[str, Any]]:
    # Scan retained files once, before opening a revision transaction. Library
    # inspection uses its own connection; it must not wait behind our write lock.
    return {
        str(day["archive_date"]): day
        for day in scan_local_library(output_dir, database, feed_id=feed_id, start_date=start, end_date=end)
    }


def acknowledge_activity(database: Path, output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    feed_id, start, end = activity_scope(payload)
    references = payload.get("items")
    if not isinstance(references, list) or len(references) > 100 or any(not isinstance(value, dict) for value in references):
        raise ValueError("Acknowledge at most 100 displayed incidents.")
    acknowledged = 0
    days = _activity_days(database, output_dir, feed_id, start, end)
    with AnalysisStore(database) as store, store.transaction():
        # Serialize revision validation with incident replacement. An event that
        # changes after loading the view must remain unread, even on a late click.
        store.connection.execute("BEGIN IMMEDIATE")
        current, _ = _saved_activity(store, days, feed_id, start, end)
        revisions = {(str(item["archive_date"]), str(item["fingerprint"]), str(item["content_sha256"])): item for item in current}
        for key in {(str(value.get("archive_date") or ""), str(value.get("fingerprint") or ""), str(value.get("content_sha256") or "")) for value in references}:
            if (item := revisions.get(key)) is not None:
                store.mark_activity_seen(int(item["day_id"]), str(item["fingerprint"]), str(item["content_sha256"]))
                acknowledged += 1
    return {"acknowledged_count": acknowledged, "submitted_count": len(references)}
