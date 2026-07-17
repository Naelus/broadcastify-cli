import json
from datetime import date
from pathlib import Path

from broadcastify_cli.analysis import PROMPT_VERSION
from broadcastify_cli.area_watch import AreaStoryAnalyzer, _public_quote
from broadcastify_cli.storage import AnalysisStore


class FakeWriter:
    model = "fake-gemma-q4"

    def __init__(self) -> None:
        self.calls = 0

    def chat_text(self, **_kwargs: object) -> str:
        self.calls += 1
        return "**Top leads**\n* A shots-fired dispatch report appeared in overlapping selected feeds. Verify independently."


def test_public_quote_redacts_contextual_name_and_phone() -> None:
    quote, changed = _public_quote(
        "Check the welfare of Summer Gibson; call 309-555-0123."
    )

    assert changed is True
    assert "Summer Gibson" not in quote
    assert "309-555-0123" not in quote


def test_public_quote_redacts_a_name_after_the_known_incident_location() -> None:
    quote, changed = _public_quote(
        "The subjects live at 2134 Wellington, Jordan Example.",
        location="2134 Wellington",
    )

    assert changed is True
    assert "Jordan Example" not in quote
    assert quote.endswith("[private person].")


def test_public_quote_redacts_a_location_adjacent_name_before_dispatch_clause() -> None:
    quote, changed = _public_quote(
        "9805, Jordan Example, for an intrusion alarm on the garage door.",
        location="9805",
    )

    assert changed is True
    assert "Jordan Example" not in quote
    assert quote == "9805, [private person], for an intrusion alarm on the garage door."


def _transcript(path: Path, text: str) -> None:
    path.write_text(
        json.dumps(
            {
                "model": "turbo",
                "duration": 600,
                "segments": [{"start": 10, "end": 15, "text": text, "speaker": None}],
            }
        ),
        encoding="utf-8",
    )


def _incident(
    fingerprint: str,
    event_type: str,
    title: str,
    summary: str,
    location: str,
    priority: int,
) -> dict[str, object]:
    return {
        "fingerprint": fingerprint,
        "event_type": event_type,
        "title": title,
        "summary": summary,
        "location": location,
        "start_seconds": 100,
        "end_seconds": 120,
        "priority": priority,
        "confidence": 0.9,
        "evidence": [
            {
                "segment_index": 0,
                "start_seconds": 100,
                "end_seconds": 120,
                "text": summary,
            }
        ],
        "attributes": {},
    }


def test_area_digest_clusters_cross_feed_reports_and_is_cached(tmp_path: Path) -> None:
    archive_date = date(2026, 7, 12)
    writer = FakeWriter()
    with AnalysisStore(tmp_path / "analysis.sqlite3") as store:
        for feed_id in ("100", "200"):
            transcript = tmp_path / f"{feed_id}.json"
            _transcript(transcript, "Possible shots fired at Main Street and First Avenue.")
            imported = store.import_transcript(feed_id, archive_date, transcript)
            store.replace_incidents(
                imported.day_id,
                [
                    _incident(
                        f"shots-{feed_id}",
                        "shots_fired",
                        "Reported shots fired",
                        "Units were sent after a shots-fired report at Main and First.",
                        "Main Street and First Avenue",
                        4,
                    )
                ],
                "test-model",
                PROMPT_VERSION,
            )
            store.save_daily_summary(
                imported.day_id,
                "Current evidence-gated day.",
                [],
                model="test-model",
                prompt_version=PROMPT_VERSION,
                transcript_sha256=imported.transcript_sha256,
            )

        # This routine singleton should remain below the newsroom-interest threshold.
        day = store.get_day("100", archive_date)
        assert day is not None
        current = store.get_incidents("100", archive_date, archive_date)
        store.replace_incidents(
            int(day["id"]),
            [
                _incident(
                    "shots-100",
                    "shots_fired",
                    "Reported shots fired",
                    "Units were sent after a shots-fired report at Main and First.",
                    "Main Street and First Avenue",
                    4,
                ),
                _incident(
                    "traffic-100",
                    "traffic_stop",
                    "Routine traffic stop",
                    "A unit stopped a vehicle.",
                    "Second Street",
                    2,
                ),
            ],
            "test-model",
            PROMPT_VERSION,
        )
        store.save_daily_summary(
            int(day["id"]),
            "Current evidence-gated day.",
            [],
            model="test-model",
            prompt_version=PROMPT_VERSION,
            transcript_sha256=str(day["transcript_sha256"]),
        )
        assert len(current) == 1
        store.save_area_profile(
            "Metro desk",
            ["75201", "75202"],
            [
                {"feed_id": "100", "name": "Police North"},
                {"feed_id": "200", "name": "Police Central"},
            ],
        )

        analyzer = AreaStoryAnalyzer(store, writer)
        first = analyzer.summarize("Metro desk", archive_date, archive_date)
        second = analyzer.summarize("Metro desk", archive_date, archive_date)

        assert len(first["stories"]) == 1
        story = first["stories"][0]
        assert story["feed_count"] == 2
        assert story["event_type"] == "shots_fired"
        assert len(story["incident_references"]) == 2
        assert story["quote_count"] == 2
        assert story["evidence_clip_count"] == 0
        assert story["subscription_eligible"] is True
        assert story["publication_status"] == "review_required"
        assert all(value["quote"] for value in story["incident_references"])
        assert all(not value["clip_available"] for value in story["incident_references"])
        assert first["coverage"]["feed_days_available"] == 2
        assert "**" not in first["summary"]
        assert second["cached"] is True
        assert writer.calls == 1
        assert store.stats()["area_story_digests"] == 1
        latest = store.get_latest_area_story_digest("Metro desk")
        assert latest is not None
        assert latest["profile_name"] == "Metro desk"
        assert latest["start_date"] == archive_date.isoformat()


def test_area_digest_does_not_merge_same_category_without_shared_place(tmp_path: Path) -> None:
    archive_date = date(2026, 7, 12)
    writer = FakeWriter()
    with AnalysisStore(tmp_path / "analysis.sqlite3") as store:
        for feed_id, location in (("100", "Main and First"), ("200", "Oak and Ninth")):
            transcript = tmp_path / f"{feed_id}.json"
            _transcript(transcript, f"Possible shots fired at {location}.")
            imported = store.import_transcript(feed_id, archive_date, transcript)
            store.replace_incidents(
                imported.day_id,
                [
                    _incident(
                        f"shots-{feed_id}",
                        "shots_fired",
                        "Reported shots fired",
                        f"Units were sent to {location} after a shots-fired report.",
                        location,
                        4,
                    )
                ],
                "test-model",
                PROMPT_VERSION,
            )
            store.save_daily_summary(
                imported.day_id,
                "Current evidence-gated day.",
                [],
                model="test-model",
                prompt_version=PROMPT_VERSION,
                transcript_sha256=imported.transcript_sha256,
            )
        store.save_area_profile(
            "Metro desk",
            ["75201"],
            [
                {"feed_id": "100", "name": "Police North"},
                {"feed_id": "200", "name": "Police Central"},
            ],
        )
        result = AreaStoryAnalyzer(store, writer).summarize(
            "Metro desk", archive_date, archive_date
        )

    assert len(result["stories"]) == 2
    assert all(value["feed_count"] == 1 for value in result["stories"])


def test_area_digest_merges_near_duplicate_same_feed_reports(tmp_path: Path) -> None:
    archive_date = date(2026, 7, 12)
    writer = FakeWriter()
    with AnalysisStore(tmp_path / "analysis.sqlite3") as store:
        transcript = tmp_path / "100.json"
        _transcript(transcript, "Possible shots fired at Main and First.")
        imported = store.import_transcript("100", archive_date, transcript)
        store.replace_incidents(
            imported.day_id,
            [
                _incident(
                    "shots-a",
                    "shots_fired",
                    "Reported shots fired",
                    "Units were sent after shots were reported at Main and First.",
                    "Main and First",
                    4,
                ),
                _incident(
                    "shots-b",
                    "shots_fired",
                    "Follow-up to reported shots fired",
                    "Units checked Main and First after the shots-fired report.",
                    "Main and First",
                    4,
                ),
            ],
            "test-model",
            PROMPT_VERSION,
        )
        store.save_daily_summary(
            imported.day_id,
            "Current evidence-gated day.",
            [],
            model="test-model",
            prompt_version=PROMPT_VERSION,
            transcript_sha256=imported.transcript_sha256,
        )
        store.save_area_profile(
            "Metro desk", ["75201"], [{"feed_id": "100", "name": "Police"}]
        )
        result = AreaStoryAnalyzer(store, writer).summarize(
            "Metro desk", archive_date, archive_date
        )

    assert len(result["stories"]) == 1
    assert len(result["stories"][0]["incident_references"]) == 2


def test_area_digest_excludes_stale_daily_claims(tmp_path: Path) -> None:
    archive_date = date(2026, 7, 12)
    writer = FakeWriter()
    with AnalysisStore(tmp_path / "analysis.sqlite3") as store:
        transcript = tmp_path / "100.json"
        _transcript(transcript, "An older extracted event claim.")
        imported = store.import_transcript("100", archive_date, transcript)
        incident_ids = store.replace_incidents(
            imported.day_id,
            [
                _incident(
                    "old-claim",
                    "shots_fired",
                    "Older shots-fired claim",
                    "An older analysis classified a shots-fired report.",
                    "Main and First",
                    5,
                )
            ],
            "test-model",
            "older-evidence-rules",
        )
        store.save_daily_summary(
            imported.day_id,
            "Older daily summary.",
            incident_ids,
            model="test-model",
            prompt_version="older-evidence-rules",
            transcript_sha256=imported.transcript_sha256,
        )
        store.save_area_profile(
            "Metro desk", ["75201"], [{"feed_id": "100", "name": "Police"}]
        )

        result = AreaStoryAnalyzer(store, writer).summarize(
            "Metro desk", archive_date, archive_date
        )

    assert result["stories"] == []
    assert result["coverage"]["incident_count"] == 0
    assert result["coverage"]["feed_days_available"] == 0
    assert result["coverage"]["stale_feed_days"] == ["100:2026-07-12"]
