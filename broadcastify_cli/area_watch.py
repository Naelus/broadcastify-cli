from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from .analysis import (
    PROMPT_VERSION,
    archive_datetime_for_offset,
    format_archive_time,
    normalize_priority,
    redact_public_text,
)
from .audio import AudioClipError, extract_audio_clip
from .storage import AnalysisStore


AREA_PROMPT_VERSION = "police-radio-area-stories-v10-preserve-spoken-names"
MIN_STORY_SCORE = 48
MAX_STORIES = 30
EVIDENCE_CONTEXT_BEFORE_SECONDS = 8.0
EVIDENCE_CONTEXT_AFTER_SECONDS = 12.0

_STOP_WORDS = {
    "a",
    "an",
    "and",
    "at",
    "by",
    "call",
    "called",
    "dispatch",
    "for",
    "from",
    "in",
    "near",
    "of",
    "on",
    "possible",
    "reported",
    "report",
    "the",
    "to",
    "unit",
    "units",
    "was",
    "were",
}

_IMPACT_BONUS = {
    "fire": 12,
    "shots_fired": 12,
    "missing_person": 12,
    "robbery": 10,
    "assault": 9,
    "person_with_weapon": 9,
    "weapons": 8,
    "traffic_collision": 7,
    "burglary": 6,
    "vehicle_theft": 8,
    "vehicle_pursuit": 10,
    "self_harm_crisis": 5,
    "medical": 4,
    "eviction_civil": 10,
}


class TextModel(Protocol):
    model: str

    def chat_text(self, *, system: str, user: str, max_tokens: int) -> str: ...


def _tokens(value: object) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(value or "").lower())
        if len(token) > 1 and token not in _STOP_WORDS
    }


def _overlap(left: set[str], right: set[str], *, containment: bool = False) -> float:
    if not left or not right:
        return 0.0
    denominator = min(len(left), len(right)) if containment else len(left | right)
    return len(left & right) / max(1, denominator)


def _incident_time(value: dict[str, Any]) -> datetime:
    offset = float(value["start_seconds"])
    wall_time = archive_datetime_for_offset(value.get("manifest_path"), offset)
    if wall_time is not None:
        return wall_time.replace(tzinfo=None)
    return datetime.combine(date.fromisoformat(str(value["archive_date"])), time.min) + timedelta(
        seconds=offset
    )


def _public_quote(value: str, *, location: str = "") -> tuple[str, bool]:
    """Preserve spoken names while masking high-risk identifiers in an ASR quote."""

    del location  # Retained for compatibility with existing callers.
    quote, changed = redact_public_text(value)
    if len(quote) > 800:
        quote = quote[:797].rstrip() + "…"
        changed = True
    return quote, changed


def _usable_location(value: object) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()
    return bool(normalized and normalized not in {"unknown", "not specified", "not stated", "n a", "none"})


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evidence_reference(
    value: dict[str, Any],
    feed_names: dict[str, str],
    segment_lookup: dict[int, dict[int, dict[str, Any]]],
) -> dict[str, Any]:
    day_segments = segment_lookup.get(int(value["day_id"]), {})
    evidence_start = max(0.0, float(value["start_seconds"]) - EVIDENCE_CONTEXT_BEFORE_SECONDS)
    evidence_end = max(
        evidence_start + 1.0,
        float(value["end_seconds"]) + EVIDENCE_CONTEXT_AFTER_SECONDS,
    )
    evidence_end = min(evidence_end, evidence_start + 120.0)
    quote_parts: list[str] = []
    speakers: list[str] = []
    segment_indexes: list[int] = []
    source_segment_count = 0
    for raw in value.get("evidence", []):
        if not isinstance(raw, dict):
            continue
        source_segment_count += 1
        try:
            segment_index = int(raw.get("segment_index"))
        except (TypeError, ValueError):
            segment_index = -1
        segment = day_segments.get(segment_index, {})
        try:
            segment_start = float(raw.get("start_seconds", segment.get("start_seconds", value["start_seconds"])))
            segment_end = float(raw.get("end_seconds", segment.get("end_seconds", segment_start)))
        except (TypeError, ValueError):
            segment_start = float(value["start_seconds"])
            segment_end = segment_start
        # Quotes shown beside a clip must occur inside that clip. Long incident
        # extractions can cite follow-up traffic many minutes later; retain those
        # source IDs in SQLite, but show only the excerpt covered by this compact
        # verification clip.
        if segment_end < evidence_start or segment_start > evidence_end:
            continue
        text = str(raw.get("text") or segment.get("text") or "").strip()
        if text and text not in quote_parts:
            quote_parts.append(text)
        speaker = str(raw.get("speaker") or segment.get("speaker") or "").strip()
        if speaker and speaker not in speakers:
            speakers.append(speaker)
        if segment_index >= 0 and segment_index not in segment_indexes:
            segment_indexes.append(segment_index)

    quote, quote_redacted = _public_quote(
        " ".join(quote_parts),
        location=str(value.get("location") or ""),
    )
    source_audio = str(value.get("audio_path") or "")
    clip_path = ""
    if source_audio:
        source = Path(source_audio)
        clip_path = str(
            source.parent
            / "evidence-clips"
            / f"{value['feed_id']}_{value['archive_date']}_I{value['id']}.mp3"
        )
    return {
        "feed_id": str(value["feed_id"]),
        "feed_name": feed_names.get(str(value["feed_id"]), f"Feed {value['feed_id']}"),
        "incident_id": int(value["id"]),
        "archive_date": str(value["archive_date"]),
        "archive_time": format_archive_time(value, float(value["start_seconds"])),
        "priority": int(value["priority"]),
        "confidence": float(value["confidence"]),
        "start_seconds": float(value["start_seconds"]),
        "end_seconds": float(value["end_seconds"]),
        "quote": quote,
        "quote_redacted": quote_redacted,
        "evidence_segment_indexes": segment_indexes,
        "evidence_segment_count": len(segment_indexes),
        "source_evidence_segment_count": source_segment_count,
        "speaker_labels": speakers,
        "has_diarization": bool(value.get("has_diarization")),
        "source_audio_path": source_audio,
        "source_audio_sha256": str(value.get("audio_sha256") or ""),
        "transcript_sha256": str(value.get("transcript_sha256") or ""),
        "clip_start_seconds": evidence_start,
        "clip_end_seconds": evidence_end,
        "clip_path": clip_path,
        "clip_available": False,
        "clip_sha256": "",
        "clip_status": (
            "Pending evidence clip generation."
            if source_audio
            else "Retained source audio is unavailable; transcript evidence only."
        ),
    }


def _prepare_story_clips(stories: Sequence[dict[str, Any]]) -> bool:
    changed = False
    for story in stories:
        for reference in story.get("incident_references", []):
            before = (
                reference.get("clip_available"),
                reference.get("clip_sha256"),
                reference.get("clip_status"),
            )
            source_audio = str(reference.get("source_audio_path") or "")
            clip_path = str(reference.get("clip_path") or "")
            if not source_audio or not clip_path:
                reference["clip_available"] = False
                reference["clip_status"] = "Retained source audio is unavailable; transcript evidence only."
            else:
                try:
                    clip = extract_audio_clip(
                        source_audio,
                        clip_path,
                        float(reference["clip_start_seconds"]),
                        float(reference["clip_end_seconds"]),
                    )
                    reference["clip_path"] = str(clip.resolve())
                    reference["clip_available"] = True
                    reference["clip_sha256"] = _file_sha256(clip)
                    reference["clip_status"] = "Timestamped evidence clip ready."
                except (AudioClipError, OSError, ValueError) as exc:
                    reference["clip_available"] = False
                    reference["clip_sha256"] = ""
                    reference["clip_status"] = str(exc)[:240]
            after = (
                reference.get("clip_available"),
                reference.get("clip_sha256"),
                reference.get("clip_status"),
            )
            changed = changed or before != after
        references = story.get("incident_references", [])
        story["evidence_clip_count"] = sum(bool(value.get("clip_available")) for value in references)
        story["quote_count"] = sum(bool(str(value.get("quote") or "").strip()) for value in references)
    return changed


def _same_story(left: dict[str, Any], right: dict[str, Any]) -> bool:
    same_feed = str(left["feed_id"]) == str(right["feed_id"])
    if str(left["event_type"]) != str(right["event_type"]):
        return False
    if str(left["event_type"]) in {"other", "unknown"}:
        return False
    maximum_gap = 20 * 60 if same_feed else 90 * 60
    if abs((_incident_time(left) - _incident_time(right)).total_seconds()) > maximum_gap:
        return False

    left_location = _tokens(left.get("location"))
    right_location = _tokens(right.get("location"))
    location_match = _overlap(left_location, right_location, containment=True)
    left_text = _tokens(f"{left.get('title', '')} {left.get('summary', '')}")
    right_text = _tokens(f"{right.get('title', '')} {right.get('summary', '')}")
    shared_text = left_text & right_text
    content_match = _overlap(left_text, right_text)

    # A common category and nearby timestamp are not enough. Two conflicting
    # named places are a hard stop; descriptive overlap is only a fallback when
    # at least one extraction did not recover a location.
    if left_location and right_location:
        if same_feed:
            return location_match >= 0.8 and len(shared_text) >= 2
        return location_match >= 0.5
    if same_feed:
        return len(shared_text) >= 4 and content_match >= 0.6
    return len(shared_text) >= 3 and content_match >= 0.45


def _cluster_incidents(incidents: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    clusters: list[list[dict[str, Any]]] = []
    for incident in sorted(incidents, key=_incident_time):
        destination = next(
            (
                cluster
                for cluster in clusters
                if any(_same_story(incident, existing) for existing in cluster)
            ),
            None,
        )
        if destination is None:
            clusters.append([incident])
        else:
            destination.append(incident)
    return clusters


def _story_from_cluster(
    cluster: Sequence[dict[str, Any]],
    feed_names: dict[str, str],
    segment_lookup: dict[int, dict[int, dict[str, Any]]],
) -> dict[str, Any]:
    ranked = sorted(
        cluster,
        key=lambda value: (
            -int(value["priority"]),
            -float(value["confidence"]),
            _incident_time(value),
        ),
    )
    lead = ranked[0]
    feed_ids = sorted({str(value["feed_id"]) for value in cluster})
    max_priority = max(int(value["priority"]) for value in cluster)
    confidence = max(float(value["confidence"]) for value in cluster)
    event_type = str(lead["event_type"])
    locations = list(
        dict.fromkeys(str(value.get("location") or "").strip() for value in ranked)
    )
    locations = [value for value in locations if _usable_location(value)]

    score = max_priority * 10 + round(confidence * 15) + _IMPACT_BONUS.get(event_type, 2)
    if locations:
        score += 5
    if len(feed_ids) > 1:
        score += min(16, 10 + (len(feed_ids) - 2) * 3)
    # Emergency severity and editorial newsworthiness are different. A routine
    # individual medical/traffic/retail call should not become a headline solely
    # because its dispatch priority is high; cross-feed overlap can still surface
    # a broader public-impact event.
    if len(feed_ids) == 1 and event_type in {
        "medical",
        "self_harm_crisis",
        "traffic_stop",
        "theft",
        "theft_shoplifting",
    }:
        score = min(score, MIN_STORY_SCORE - 1)
    score = min(100, score)
    level = "Top lead" if score >= 75 else "Strong lead" if score >= 60 else "Monitor"

    reasons = [f"P{max_priority} dispatch report", f"{confidence:.0%} extraction confidence"]
    if len(feed_ids) > 1:
        reasons.append(
            f"overlapping traffic in {len(feed_ids)} selected feeds (which may rebroadcast the same radio traffic)"
        )
    if event_type in _IMPACT_BONUS:
        reasons.append(event_type.replace("_", " "))
    if locations:
        reasons.append("a usable reported location")

    story_key = ",".join(
        f"{value['feed_id']}:{value['id']}" for value in sorted(cluster, key=lambda item: int(item["id"]))
    )
    story_id = "S" + hashlib.sha1(story_key.encode("utf-8")).hexdigest()[:10].upper()
    references = [
        _evidence_reference(value, feed_names, segment_lookup)
        for value in sorted(cluster, key=_incident_time)
    ]
    topic_tags = [event_type, "public_safety"]
    return {
        "story_id": story_id,
        "headline": str(lead["title"]),
        "summary": str(lead["summary"]),
        "event_type": event_type,
        "location": locations[0] if locations else "",
        "first_reported": min(_incident_time(value) for value in cluster).isoformat(sep=" ", timespec="seconds"),
        "last_reported": max(_incident_time(value) for value in cluster).isoformat(sep=" ", timespec="seconds"),
        "newsworthiness_score": score,
        "interest_level": level,
        "priority": max_priority,
        "confidence": confidence,
        "feed_count": len(feed_ids),
        "feed_ids": feed_ids,
        "why_interesting": "; ".join(reasons),
        "incident_references": references,
        "quote_count": sum(bool(value["quote"]) for value in references),
        "evidence_clip_count": 0,
        "neighborhood_tags": locations[:5],
        "topic_tags": topic_tags,
        "subscription_eligible": bool(score >= 60 and locations and confidence >= 0.65),
        "publication_status": "review_required",
    }


def area_story_source_fingerprint(
    profile: dict[str, Any],
    days: Sequence[dict[str, Any]],
    incidents: Sequence[dict[str, Any]],
) -> str:
    """Hash the exact current inputs used to write an area digest."""

    source = {
        "incident_prompt_version": PROMPT_VERSION,
        "profile_updated_at": profile["updated_at"],
        "feed_ids": list(profile["feed_ids"]),
        "days": [
            (
                value["feed_id"],
                value["archive_date"],
                value["transcript_sha256"],
                value.get("audio_sha256"),
            )
            for value in days
        ],
        "incidents": [
            (value["id"], value["fingerprint"], value["created_at"])
            for value in incidents
        ],
    }
    return hashlib.sha256(
        json.dumps(source, sort_keys=True).encode("utf-8")
    ).hexdigest()


def current_area_story_source_fingerprint(
    store: AnalysisStore,
    profile: dict[str, Any],
    start_date: date,
    end_date: date,
) -> str:
    """Recompute a saved area digest's source identity without a model."""

    feed_ids = [str(value) for value in profile["feed_ids"]]
    incidents = store.get_incidents_for_feeds(
        feed_ids,
        start_date,
        end_date,
        prompt_version=PROMPT_VERSION,
    )
    days = [
        value
        for value in store.list_days()
        if str(value["feed_id"]) in feed_ids
        and start_date.isoformat()
        <= str(value["archive_date"])
        <= end_date.isoformat()
        and str(value.get("summary_prompt_version") or "")
        == PROMPT_VERSION
    ]
    return area_story_source_fingerprint(profile, days, incidents)


class AreaStoryAnalyzer:
    def __init__(
        self,
        store: AnalysisStore,
        client: TextModel,
        prompt_version: str = AREA_PROMPT_VERSION,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.store = store
        self.client = client
        self.prompt_version = prompt_version
        self.progress = progress or (lambda _message: None)

    def summarize(
        self,
        profile_name: str,
        start_date: date,
        end_date: date,
        force: bool = False,
    ) -> dict[str, Any]:
        if start_date > end_date:
            raise ValueError("Start date must be on or before end date.")
        if (end_date - start_date).days > 30:
            raise ValueError("Area story digests are limited to 31 days at a time.")
        profile = self.store.get_area_profile(profile_name)
        if profile is None:
            raise ValueError(f"Area profile not found: {profile_name}.")
        feed_ids = list(profile["feed_ids"])
        feed_names = {str(value["feed_id"]): str(value["name"]) for value in profile["feeds"]}
        incidents = [
            {
                **value,
                "priority": normalize_priority(
                    str(value.get("event_type") or "unknown"),
                    int(value.get("priority") or 1),
                ),
            }
            for value in self.store.get_incidents_for_feeds(
                feed_ids,
                start_date,
                end_date,
                prompt_version=PROMPT_VERSION,
            )
        ]
        retained_days = [
            value
            for value in self.store.list_days()
            if str(value["feed_id"]) in feed_ids
            and start_date.isoformat() <= str(value["archive_date"]) <= end_date.isoformat()
        ]
        stale_days = [
            value
            for value in retained_days
            if bool(value.get("has_summary"))
            and str(value.get("summary_prompt_version") or "") != PROMPT_VERSION
        ]
        days = [
            value
            for value in retained_days
            if str(value.get("summary_prompt_version") or "") == PROMPT_VERSION
        ]
        available = {(str(value["feed_id"]), str(value["archive_date"])) for value in days}
        date_count = (end_date - start_date).days + 1
        expected = [
            (feed_id, (start_date + timedelta(days=offset)).isoformat())
            for feed_id in feed_ids
            for offset in range(date_count)
        ]
        missing = [f"{feed_id}:{day}" for feed_id, day in expected if (feed_id, day) not in available]
        coverage = {
            "feed_count": len(feed_ids),
            "feeds_with_data": len({feed_id for feed_id, _day in available}),
            "feed_days_available": len(available),
            "feed_days_expected": len(expected),
            "missing_feed_days": missing,
            "retained_feed_days": len(retained_days),
            "stale_feed_days": [
                f"{value['feed_id']}:{value['archive_date']}" for value in stale_days
            ],
            "incident_count": len(incidents),
            "incident_prompt_version": PROMPT_VERSION,
            "area_prompt_version": self.prompt_version,
        }

        fingerprint = area_story_source_fingerprint(
            profile,
            days,
            incidents,
        )
        existing = None if force else self.store.get_area_story_digest(
            int(profile["id"]),
            start_date,
            end_date,
            self.client.model,
            self.prompt_version,
            fingerprint,
        )
        if existing is not None:
            self.progress(f"Reusing saved area digest for {start_date} through {end_date}.")
            cached_stories = json.loads(str(existing["stories_json"]))
            if _prepare_story_clips(cached_stories):
                self.store.save_area_story_digest(
                    int(profile["id"]),
                    start_date,
                    end_date,
                    str(existing["summary"]),
                    cached_stories,
                    json.loads(str(existing["coverage_json"])),
                    self.client.model,
                    self.prompt_version,
                    fingerprint,
                )
            return self._report(
                profile,
                start_date,
                end_date,
                str(existing["summary"]),
                cached_stories,
                json.loads(str(existing["coverage_json"])),
                cached=True,
            )

        segment_lookup = {
            int(day_id): {
                int(segment["segment_index"]): segment
                for segment in self.store.get_segments(int(day_id))
            }
            for day_id in {int(value["day_id"]) for value in incidents}
        }
        stories = [
            _story_from_cluster(cluster, feed_names, segment_lookup)
            for cluster in _cluster_incidents(incidents)
        ]
        stories = sorted(
            (value for value in stories if int(value["newsworthiness_score"]) >= MIN_STORY_SCORE),
            key=lambda value: (-int(value["newsworthiness_score"]), str(value["first_reported"])),
        )[:MAX_STORIES]
        if stories:
            self.progress(f"Preparing quotes and audio clips for {len(stories)} ranked lead(s).")
            _prepare_story_clips(stories)
        self.progress(
            f"Ranked {len(stories)} story lead(s) from {len(incidents)} saved incident(s) across {len(feed_ids)} feeds."
        )
        summary = self._write_digest(profile, start_date, end_date, stories, coverage)
        self.store.save_area_story_digest(
            int(profile["id"]),
            start_date,
            end_date,
            summary,
            stories,
            coverage,
            self.client.model,
            self.prompt_version,
            fingerprint,
        )
        return self._report(
            profile, start_date, end_date, summary, stories, coverage, cached=False
        )

    def _write_digest(
        self,
        profile: dict[str, Any],
        start_date: date,
        end_date: date,
        stories: Sequence[dict[str, Any]],
        coverage: dict[str, Any],
    ) -> str:
        coverage_line = (
            f"Coverage: {coverage['feed_days_available']}/{coverage['feed_days_expected']} "
            f"selected feed-days analyzed; {coverage['feeds_with_data']}/"
            f"{coverage['feed_count']} selected feeds had retained current analysis."
        )
        if not stories:
            return (
                coverage_line
                + "\n\nNo sufficiently supported, newsworthy story leads were found in the analyzed feed-days. "
                "This does not mean no events occurred; missing archives, radio coverage, and noisy ASR limit the result."
            )
        cards = [
            f"{value['story_id']} | score {value['newsworthiness_score']} | {value['interest_level']} | "
            f"P{value['priority']} | {value['event_type']} | {value['first_reported']} | "
            f"{value['headline']} — {value['summary']} | location: {value['location'] or 'not stated'} | "
            f"feeds: {', '.join(value['feed_ids'])}"
            for value in stories[:12]
        ]
        system = (
            "Write a concise local-news assignment brief from structured police-radio story leads. "
            "Use only supplied facts. Treat every item as an unconfirmed dispatch report, never as a proven crime or outcome. "
            "Prioritize public impact and cross-feed overlap, but state that feeds can rebroadcast the same traffic. "
            "Preserve person names present in supplied leads, but never infer or normalize an identity. Omit "
            "phone numbers, dates of birth, driver's-license numbers, and license plates. Do not restate "
            "archive/feed coverage counts; the application adds "
            "its exact SQLite-derived coverage line. A ZIP identifies the feed-discovery center, not an incident "
            "geofence; do not say events occurred within a ZIP unless a supplied story says so. "
            "Use plain text, under 250 words, with short section labels."
        )
        user = (
            f"Area profile: {profile['name']}; ZIPs: {', '.join(profile['zip_codes'])}; "
            f"range: {start_date} through {end_date}.\n"
            f"Coverage: {coverage['feed_days_available']}/{coverage['feed_days_expected']} feed-days; "
            f"{coverage['feeds_with_data']}/{coverage['feed_count']} feeds with data.\n\n"
            "RANKED STORY LEADS:\n" + "\n".join(cards)
        )
        summary = self.client.chat_text(system=system, user=user, max_tokens=2_048).strip()
        body = self._plain_text(summary or self._fallback(stories, coverage))
        body_lines = [
            line
            for line in body.splitlines()
            if not (
                re.search(r"\bfeed-days?\b", line, flags=re.I)
                or re.search(r"\bfeeds?\s+with\s+data\b", line, flags=re.I)
                or (
                    re.match(r"^\s*coverage\b", line, flags=re.I)
                    and re.search(r"\b(?:feed|data|archive)\b", line, flags=re.I)
                )
            )
        ]
        body = "\n".join(body_lines).strip()
        body = re.sub(r"\bfire\s*/\s*arson\b", "reported fire", body, flags=re.I)
        body = re.sub(r"\barson\b", "reported fire", body, flags=re.I)
        for zip_code in profile["zip_codes"]:
            body = re.sub(
                rf"\b(?:across|in)\s+(?:the\s+)?{re.escape(str(zip_code))}\s+area\b",
                "on the selected feeds",
                body,
                flags=re.I,
            )
        return coverage_line + (f"\n\n{body}" if body else "")

    @staticmethod
    def _plain_text(value: str) -> str:
        lines = []
        for original in value.splitlines():
            line = re.sub(r"\*\*(.+?)\*\*", r"\1", original.strip())
            line = re.sub(r"^#{1,6}\s*", "", line)
            line = re.sub(r"^[*-]\s+", "• ", line)
            line = line.replace("_", " ")
            lines.append(line.rstrip())
        return "\n".join(lines).strip()

    @staticmethod
    def _fallback(
        stories: Sequence[dict[str, Any]], coverage: dict[str, Any]
    ) -> str:
        leads = "; ".join(
            f"{value['headline']} ({value['event_type'].replace('_', ' ')}, score {value['newsworthiness_score']})"
            for value in stories[:6]
        )
        return (
            f"Leading unconfirmed dispatch reports: {leads}. These are newsroom leads, not verified events; "
            "feeds may overlap, archives may be missing, and radio ASR can be wrong."
        )

    @staticmethod
    def _report(
        profile: dict[str, Any],
        start_date: date,
        end_date: date,
        summary: str,
        stories: Sequence[dict[str, Any]],
        coverage: dict[str, Any],
        cached: bool,
    ) -> dict[str, Any]:
        return {
            "profile_name": profile["name"],
            "zip_codes": list(profile["zip_codes"]),
            "feed_ids": list(profile["feed_ids"]),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "summary": AreaStoryAnalyzer._plain_text(summary),
            "stories": list(stories),
            "coverage": coverage,
            "cached": cached,
        }
