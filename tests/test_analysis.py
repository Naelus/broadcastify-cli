import json
from copy import deepcopy
from datetime import date
from pathlib import Path

import pytest

from broadcastify_cli.analysis import (
    IncidentAnalyzer,
    LlamaCppClient,
    WeeklySummaryAnalyzer,
    archive_datetime_for_offset,
    build_transcript_windows,
    find_llama_server,
    normalize_event_type,
    normalize_priority,
    prepare_llama_environment,
)
from broadcastify_cli.storage import AnalysisStore


class FakeLlamaClient:
    model = "fake-gemma-q4"

    def __init__(self) -> None:
        self.calls = 0

    def chat_json(self, *_args: object, **kwargs: object) -> dict[str, object]:
        self.calls += 1
        if kwargs["schema_name"] == "police_radio_incidents":
            return {
                "incidents": [
                    {
                        "event_type": "shots_fired",
                        "title": "Reported shots fired",
                        "summary": "Dispatch reported possible shots fired near Main and First.",
                        "location": "Main and First",
                        "priority": 4,
                        "confidence": 0.8,
                        "evidence_segment_ids": [0, 999],
                        "attributes": {},
                    }
                ]
            }
        return {"summary": "One notable report of possible shots fired was dispatched."}

    def chat_text(self, *_args: object, **_kwargs: object) -> str:
        self.calls += 1
        return "Two available days included reported shots-fired calls, with five dates missing from coverage."


class FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, object]) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self) -> dict[str, object]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_llama_json_retries_schema_parser_500_as_json_object(monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    responses = [
        FakeResponse(500, {"error": {"message": "Failed to parse input at pos 1188"}}),
        FakeResponse(
            200,
            {
                "choices": [
                    {
                        "message": {"content": '{"summary":"CPU fallback works."}'},
                        "finish_reason": "stop",
                    }
                ]
            },
        ),
    ]

    def fake_post(_url: str, *, json: dict[str, object], timeout: float) -> FakeResponse:
        assert timeout == 30
        calls.append(deepcopy(json))
        return responses.pop(0)

    monkeypatch.setattr("broadcastify_cli.analysis.requests.post", fake_post)
    result = LlamaCppClient(timeout=30).chat_json(
        system="Return JSON.",
        user="Summarize.",
        schema_name="summary",
        schema={
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
        },
    )

    assert result == {"summary": "CPU fallback works."}
    assert calls[0]["response_format"]["type"] == "json_schema"
    assert calls[1]["response_format"] == {"type": "json_object"}


def test_llama_json_does_not_hide_unrelated_server_500(monkeypatch) -> None:
    calls = 0

    def fake_post(_url: str, *, json: dict[str, object], timeout: float) -> FakeResponse:
        nonlocal calls
        calls += 1
        return FakeResponse(500, {"error": {"message": "model out of memory"}})

    monkeypatch.setattr("broadcastify_cli.analysis.requests.post", fake_post)
    with pytest.raises(RuntimeError, match="HTTP 500"):
        LlamaCppClient().chat_json(
            system="Return JSON.",
            user="Summarize.",
            schema_name="summary",
            schema={"type": "object"},
        )
    assert calls == 1


def test_daily_summary_schema_avoids_large_llama_grammar_repeat() -> None:
    captured: dict[str, object] = {}

    class CaptureClient:
        model = "test"

        def chat_json(self, **kwargs: object) -> dict[str, object]:
            captured.update(kwargs)
            return {"summary": "word " * 300}

    analyzer = IncidentAnalyzer(None, CaptureClient())  # type: ignore[arg-type]
    summary = analyzer._summarize_day(  # noqa: SLF001 - regression for local schema
        "90001",
        date(2026, 7, 16),
        [
            {
                "id": 1,
                "archive_date": "2026-07-16",
                "manifest_path": "",
                "start_seconds": 10.0,
                "priority": 4,
                "confidence": 0.8,
                "event_type": "shots_fired",
                "title": "Reported shots fired",
                "summary": "Dispatch reported possible shots.",
                "location": "Main Street",
            }
        ],
    )

    schema = captured["schema"]
    assert "maxLength" not in schema["properties"]["summary"]
    assert len(summary.split()) == 250


def test_daily_summary_retries_unknown_ids_and_unsupported_counts() -> None:
    calls = 0
    progress: list[str] = []

    class GroundingClient:
        model = "test"

        def chat_json(self, **_kwargs: object) -> dict[str, object]:
            nonlocal calls
            calls += 1
            if calls == 1:
                return {
                    "summary": (
                        "Four priority events were identified: I1 was reported, "
                        "and I2 through I4 require follow-up."
                    )
                }
            return {"summary": "I1 was a reported weapon-related call with no confirmed outcome."}

    analyzer = IncidentAnalyzer(  # type: ignore[arg-type]
        None,
        GroundingClient(),
        progress=progress.append,
    )
    summary = analyzer._summarize_day(  # noqa: SLF001
        "90001",
        date(2026, 7, 16),
        [
            {
                "id": 1,
                "archive_date": "2026-07-16",
                "manifest_path": "",
                "start_seconds": 10.0,
                "priority": 4,
                "confidence": 0.8,
                "event_type": "person_with_weapon",
                "title": "Reported person with weapon",
                "summary": "Dispatch audio may describe a person with a weapon.",
                "location": "not stated",
            }
        ],
    )

    assert calls == 2
    assert summary.startswith("I1")
    assert "I2" not in summary
    assert any("unsupported daily activity" in value for value in progress)


def test_daily_summary_uses_deterministic_fallback_after_two_ungrounded_briefs() -> None:
    progress: list[str] = []

    class UngroundedClient:
        model = "test"

        def chat_json(self, **_kwargs: object) -> dict[str, object]:
            return {"summary": "I99 was one of five priority incidents."}

    analyzer = IncidentAnalyzer(  # type: ignore[arg-type]
        None,
        UngroundedClient(),
        progress=progress.append,
    )
    summary = analyzer._summarize_day(  # noqa: SLF001
        "90001",
        date(2026, 7, 16),
        [
            {
                "id": 1,
                "archive_date": "2026-07-16",
                "manifest_path": "",
                "start_seconds": 10.0,
                "priority": 4,
                "confidence": 0.8,
                "event_type": "person_with_weapon",
                "title": "Reported person with weapon",
                "summary": "Dispatch audio may describe a person with a weapon.",
                "location": "not stated",
            }
        ],
    )

    assert "I99" not in summary
    assert "Reported person with weapon" in summary
    assert summary.endswith("not confirmed outcomes.")
    assert any("deterministic evidence summary" in value for value in progress)


def test_llama_lookup_tolerates_inaccessible_winget_cache(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("LLAMA_SERVER_PATH", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr("broadcastify_cli.analysis.shutil.which", lambda _name: None)

    def denied(_self: Path, _pattern: str):
        raise PermissionError("package cache is not readable")

    monkeypatch.setattr(Path, "glob", denied)

    assert find_llama_server() is None


def test_llama_environment_provides_rootless_cache_paths(
    tmp_path: Path,
) -> None:
    prepared = prepare_llama_environment(
        {"HOME": str(tmp_path / "missing"), "HUGGINGFACE_TOKEN": "test-token"},
        tmp_path / "runtime",
        platform_name="posix",
    )

    assert Path(prepared["HOME"]).is_dir()
    assert Path(prepared["LLAMA_CACHE"]).is_dir()
    assert Path(prepared["HF_HOME"]).is_dir()
    assert prepared["HF_TOKEN"] == "test-token"


def test_windows_cover_full_timeline() -> None:
    segments = [
        {"segment_index": 0, "start_seconds": 10.0, "end_seconds": 20.0, "text": "one"},
        {"segment_index": 1, "start_seconds": 8_000.0, "end_seconds": 8_010.0, "text": "two"},
    ]
    windows = build_transcript_windows(segments, window_seconds=7_200, overlap_seconds=0)
    assert len(windows) == 2
    assert windows[-1].segments[0]["segment_index"] == 1


def test_clear_evidence_corrects_category_and_routine_priority() -> None:
    assert normalize_event_type("Shoplifting in progress in the bathroom", "shots_fired") == "theft_shoplifting"
    assert normalize_event_type("A bald male has a silver handgun", "warrant_arrest") == "person_with_weapon"
    assert normalize_event_type("She reported a single gunshot", "other") == "shots_fired"
    assert normalize_event_type("Juveniles running northeast", "fire") == "suspicious_activity"
    assert normalize_event_type("Residential intrusion alarm", "warrant_arrest") == "burglary"
    assert normalize_priority("theft_shoplifting", 5) == 3


def test_incident_normalization_marks_radio_report_and_caps_coarse_asr_confidence() -> None:
    incident = IncidentAnalyzer._validate_incident(  # noqa: SLF001
        {
            "event_type": "person_with_weapon",
            "title": "Chase with gun",
            "summary": "A person was chased with a gun.",
            "location": "Unknown",
            "priority": 4,
            "confidence": 1.0,
            "evidence_segment_ids": [0],
            "attributes": {},
        },
        {0},
        {
            0: {
                "segment_index": 0,
                "start_seconds": 0.0,
                "end_seconds": 30.0,
                "speaker": "SPEAKER_00",
                "text": "I was chased with someone with a gun.",
            }
        },
    )

    assert incident is not None
    assert incident["summary"] == "Radio traffic reported: a person was chased with a gun."
    assert incident["confidence"] == 0.90


def test_manifest_maps_audio_offset_to_archive_wall_time(tmp_path: Path) -> None:
    manifest = tmp_path / "combined.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "archive_start": "2026-07-11T00:08:00",
                        "combined_start_seconds": 0,
                    },
                    {
                        "archive_start": "2026-07-11T00:38:00",
                        "combined_start_seconds": 1800,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    wall_time = archive_datetime_for_offset(manifest, 1860)
    assert wall_time is not None
    assert wall_time.isoformat() == "2026-07-11T00:39:00"


def test_incident_analysis_requires_valid_evidence(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.json"
    transcript.write_text(
        json.dumps(
            {
                "model": "turbo",
                "duration": 300.0,
                "segments": [
                    {
                        "start": 10.0,
                        "end": 15.0,
                        "text": "Possible shots fired near Main and First.",
                        "speaker": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with AnalysisStore(tmp_path / "analysis.sqlite3") as store:
        store.import_transcript("90001", date(2026, 7, 12), transcript)
        client = FakeLlamaClient()
        analyzer = IncidentAnalyzer(store, client)
        result = analyzer.analyze_day(
            "90001", date(2026, 7, 12)
        )
        first_call_count = client.calls
        resumed = analyzer.analyze_day("90001", date(2026, 7, 12))
        incidents = store.get_incidents(
            "90001", date(2026, 7, 12), date(2026, 7, 12)
        )

    assert result["incidents"] == 1
    assert incidents[0]["event_type"] == "shots_fired"
    assert incidents[0]["evidence"][0]["segment_index"] == 0
    assert all(item["segment_index"] != 999 for item in incidents[0]["evidence"])
    assert resumed["incidents"] == 1
    assert client.calls == first_call_count


def test_weekly_summary_covers_available_days_and_is_cached(tmp_path: Path) -> None:
    client = FakeLlamaClient()
    with AnalysisStore(tmp_path / "analysis.sqlite3") as store:
        for archive_date in (date(2026, 7, 11), date(2026, 7, 12)):
            transcript = tmp_path / f"{archive_date}.json"
            transcript.write_text(
                json.dumps(
                    {
                        "model": "turbo",
                        "duration": 300.0,
                        "segments": [
                            {
                                "start": 10.0,
                                "end": 15.0,
                                "text": "Possible shots fired near Main and First.",
                                "speaker": "SPEAKER_00",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            store.import_transcript("90001", archive_date, transcript)
            IncidentAnalyzer(store, client).analyze_day("90001", archive_date)

        summarizer = WeeklySummaryAnalyzer(store, client)
        first = summarizer.summarize("90001", date(2026, 7, 12))
        calls_after_first = client.calls
        resumed = summarizer.summarize("90001", date(2026, 7, 12))

        assert store.stats()["weekly_summaries"] == 1

    assert first["days_available"] == 2
    assert first["days_expected"] == 7
    assert len(first["missing_dates"]) == 5
    assert first["incident_count"] == 2
    assert first["cached"] is False
    assert resumed["cached"] is True
    assert client.calls == calls_after_first
