import json
import os
from copy import deepcopy
from datetime import date
from pathlib import Path

import pytest
import requests

from broadcastify_cli.analysis import (
    IncidentAnalyzer,
    LlamaCppClient,
    LlamaServerError,
    LlamaServerProcess,
    PROMPT_VERSION,
    RangeQuestionAnswerer,
    WINDOW_PROMPT_VERSION,
    WeeklySummaryAnalyzer,
    archive_datetime_for_offset,
    format_archive_time,
    build_transcript_windows,
    find_cached_huggingface_gguf,
    incident_claim_has_evidence_support,
    normalize_local_model_reference,
    prepare_llama_loader_environment,
    redact_public_text,
    resolve_local_llama_model,
)
from broadcastify_cli.storage import AnalysisStore


def test_range_question_withholds_an_uncited_generated_event(
    tmp_path: Path,
) -> None:
    archive_date = date(2026, 8, 5)
    transcript = tmp_path / "transcript.json"
    transcript.write_text(
        json.dumps(
            {
                "model": "test",
                "segments": [
                    {
                        "start": 10.0,
                        "end": 15.0,
                        "text": "Dispatch received one report of possible shots fired.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    class UncitedClient:
        model = "fake-uncited-model"

        def chat_json(self, **kwargs: object) -> dict[str, object]:
            return {
                "answer": "A specific shots-fired report happened.",
                "evidence_ids": [],
                "limitations": [],
            }

    with AnalysisStore(tmp_path / "analysis.sqlite3") as store:
        store.import_transcript("90001", archive_date, transcript)
        result = RangeQuestionAnswerer(store, UncitedClient()).ask(
            "90001",
            archive_date,
            archive_date,
            "What happened?",
        )
        saved = store.connection.execute(
            "SELECT answer FROM qa_history WHERE feed_id=?",
            ("90001",),
        ).fetchone()

    assert result["answer"] == (
        "The retained evidence did not support a fully cited answer. "
        "Try narrowing the date range or question."
    )
    assert result["evidence_ids"] == []
    assert result["limitations"] == [
        "The generated answer was withheld because every material "
        "statement did not include a valid retained-evidence citation."
    ]
    assert saved is not None
    assert saved["answer"] == result["answer"]


def test_range_question_withholds_mixed_cited_and_uncited_events(
    tmp_path: Path,
) -> None:
    archive_date = date(2026, 8, 5)
    transcript = tmp_path / "transcript.json"
    transcript.write_text(
        json.dumps(
            {
                "model": "test",
                "segments": [
                    {
                        "start": 10.0,
                        "end": 15.0,
                        "text": "Dispatch received one report of possible shots fired.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    class MixedCitationClient:
        model = "fake-mixed-citation-model"

        def chat_json(self, **kwargs: object) -> dict[str, object]:
            return {
                "answer": (
                    "One possible report was retained [E1]. "
                    "A second unsupported event happened."
                ),
                "evidence_ids": ["E1"],
                "limitations": [],
            }

    with AnalysisStore(tmp_path / "analysis.sqlite3") as store:
        store.import_transcript("90001", archive_date, transcript)
        result = RangeQuestionAnswerer(store, MixedCitationClient()).ask(
            "90001",
            archive_date,
            archive_date,
            "What happened?",
        )

    assert result["answer"].startswith(
        "The retained evidence did not support a fully cited answer."
    )
    assert result["evidence_ids"] == []
    assert result["limitations"] == [
        "The generated answer was withheld because every material "
        "statement did not include a valid retained-evidence citation."
    ]


def test_managed_llama_cpu_device_disables_every_gpu_layer() -> None:
    assert LlamaServerProcess(device="cpu")._offload_arguments() == [
        "--device",
        "none",
        "--n-gpu-layers",
        "0",
    ]
    assert LlamaServerProcess(
        device="Vulkan1", gpu_layers=17
    )._offload_arguments() == [
        "--device",
        "Vulkan1",
        "--n-gpu-layers",
        "17",
    ]


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


class InterruptingWindowClient:
    model = "interrupting-test-model"

    def __init__(self) -> None:
        self.incident_calls = 0
        self.failed_once = False

    def chat_json(self, *_args: object, **kwargs: object) -> dict[str, object]:
        if kwargs["schema_name"] != "police_radio_incidents":
            return {
                "summary": "Radio traffic contained supported shots-fired reports."
            }
        self.incident_calls += 1
        if self.incident_calls == 2 and not self.failed_once:
            self.failed_once = True
            raise RuntimeError("simulated process interruption")
        user = str(kwargs["user"])
        segment_id = 1 if "S1 " in user else 0
        location = "Oak and Ninth" if segment_id else "Main and First"
        return {
            "incidents": [
                {
                    "event_type": "shots_fired",
                    "title": "Reported shots fired",
                    "summary": f"Dispatch reported possible shots fired near {location}.",
                    "location": location,
                    "priority": 4,
                    "confidence": 0.8,
                    "evidence_segment_ids": [segment_id],
                    "attributes": {},
                }
            ]
        }


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
    assert calls[0]["temperature"] == 0.0
    assert calls[0]["top_p"] == 1.0
    assert calls[0]["seed"] == 0
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


def test_llama_json_reports_oversized_context_without_schema_retry(monkeypatch) -> None:
    calls = 0
    response = requests.Response()
    response.status_code = 400
    response.url = "http://127.0.0.1:8088/v1/chat/completions"
    response._content = json.dumps(
        {
            "error": {
                "message": (
                    "request (66558 tokens) exceeds the available context size "
                    "(32768 tokens)"
                )
            }
        }
    ).encode()

    def fake_post(_url: str, *, json: dict[str, object], timeout: float) -> requests.Response:
        nonlocal calls
        calls += 1
        return response

    monkeypatch.setattr("broadcastify_cli.analysis.requests.post", fake_post)
    with pytest.raises(LlamaServerError, match="larger than llama.cpp's context window"):
        LlamaCppClient().chat_json(
            system="Return JSON.",
            user="Dense transcript.",
            schema_name="summary",
            schema={"type": "object"},
        )
    assert calls == 1


def test_llama_json_reaches_full_length_retry_ceiling(monkeypatch) -> None:
    requested_tokens: list[int] = []

    def fake_post(_url: str, *, json: dict[str, object], timeout: float) -> FakeResponse:
        requested_tokens.append(int(json["max_tokens"]))
        if len(requested_tokens) < 4:
            return FakeResponse(
                200,
                {
                    "choices": [
                        {
                            "message": {"content": '{"incidents": ['},
                            "finish_reason": "length",
                        }
                    ]
                },
            )
        return FakeResponse(
            200,
            {
                "choices": [
                    {
                        "message": {"content": '{"incidents": []}'},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    monkeypatch.setattr("broadcastify_cli.analysis.requests.post", fake_post)
    result = LlamaCppClient().chat_json(
        system="Return JSON.",
        user="Extract all incidents.",
        schema_name="incidents",
        schema={"type": "object"},
    )

    assert result == {"incidents": []}
    assert requested_tokens == [2_048, 4_096, 8_192, 16_384]


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


def test_llama_loader_path_is_scoped_and_prefers_executable_siblings(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "llama-release" / "llama-server"
    executable.parent.mkdir()
    executable.write_bytes(b"binary")
    original = {
        "LD_LIBRARY_PATH": f"/shared/ggml{os.pathsep}{executable.parent.resolve()}",
        "UNRELATED": "retained",
    }

    prepared = prepare_llama_loader_environment(
        original,
        executable,
        platform_name="linux",
    )

    assert prepared["LD_LIBRARY_PATH"].split(os.pathsep) == [
        str(executable.parent.resolve()),
        "/shared/ggml",
    ]
    assert prepared["UNRELATED"] == "retained"
    assert original["LD_LIBRARY_PATH"].startswith("/shared/ggml")


def test_legacy_llama_model_reuses_main_gguf_from_older_hub_snapshot(
    tmp_path: Path,
) -> None:
    hub = tmp_path / "hub"
    model_root = hub / "models--ggml-org--gemma-4-12B-it-GGUF"
    (model_root / "refs").mkdir(parents=True)
    (model_root / "refs" / "main").write_text("new-revision", encoding="utf-8")
    old_snapshot = model_root / "snapshots" / "old-revision"
    old_snapshot.mkdir(parents=True)
    main_model = old_snapshot / "gemma-4-12B-it-Q4_K_M.gguf"
    main_model.write_bytes(b"model")
    (old_snapshot / "mmproj-gemma-4-12B-it-Q4_K_M.gguf").write_bytes(b"projector")
    legacy = "ggml-org/gemma-4-12B-it-GGUF:Q4_K_M"

    assert find_cached_huggingface_gguf(legacy, cache_roots=[hub]) == main_model
    assert normalize_local_model_reference(legacy, cache_roots=[hub]) == legacy
    assert resolve_local_llama_model(legacy, cache_roots=[hub]) == (
        main_model,
        legacy,
    )


def test_missing_legacy_llama_model_migrates_to_available_quant(
    tmp_path: Path,
) -> None:
    legacy = "ggml-org/gemma-4-12B-it-GGUF:Q4_K_M"

    assert normalize_local_model_reference(
        legacy, cache_roots=[tmp_path / "empty"]
    ) == "ggml-org/gemma-4-12B-it-GGUF:Q4_0"
    assert resolve_local_llama_model(
        legacy, cache_roots=[tmp_path / "empty"]
    ) == (None, "ggml-org/gemma-4-12B-it-GGUF:Q4_0")


def test_dense_transcript_windows_are_bounded_with_small_context_overlap() -> None:
    segments = [
        {
            "segment_index": index,
            "start_seconds": float(index),
            "end_seconds": float(index + 1),
            "text": "dispatch evidence " + ("x" * 180),
        }
        for index in range(500)
    ]

    windows = build_transcript_windows(
        segments,
        window_seconds=7_200,
        overlap_seconds=0,
        max_prompt_chars=10_000,
        overlap_prompt_chars=500,
    )

    assert len(windows) > 1
    assert all(len(window.prompt_text()) <= 10_000 for window in windows)
    assert {segment["segment_index"] for window in windows for segment in window.segments} == set(
        range(500)
    )
    assert any(
        set(segment["segment_index"] for segment in first.segments)
        & set(segment["segment_index"] for segment in second.segments)
        for first, second in zip(windows, windows[1:])
    )


def test_incident_validation_preserves_spoken_names_and_keeps_evidence() -> None:
    incident = IncidentAnalyzer._validate_incident(  # noqa: SLF001
        {
            "event_type": "trespassing",
            "title": "Logan Spangler refusing to leave",
            "summary": "Officers responded to a male, Logan Spangler, refusing to leave.",
            "location": "North Brandywine",
            "priority": 2,
            "confidence": 0.9,
            "evidence_segment_ids": [0],
            "attributes": {"subject_name": "Logan Spangler"},
        },
        {0},
        {
            0: {
                "segment_index": 0,
                "start_seconds": 0.0,
                "end_seconds": 8.0,
                "speaker": "SPEAKER_00",
                "text": "Trouble with Logan Spangler, refusing to leave North Brandywine.",
            }
        },
    )

    assert incident is not None
    assert incident["event_type"] == "trespassing"
    assert "Logan Spangler" in incident["title"]
    assert "Logan Spangler" in incident["summary"]
    assert incident["attributes"]["subject_name"] == "Logan Spangler"
    assert "Logan Spangler" in incident["evidence"][0]["text"]


def test_public_text_preserves_names_by_default_but_masks_high_risk_identifiers() -> None:
    text, changed = redact_public_text(
        "Trouble with Logan Spangler; caller 309-555-0123, DOB 1/2/1980."
    )

    assert changed is True
    assert "Logan Spangler" in text
    assert "309-555-0123" not in text
    assert "1/2/1980" not in text


def test_incident_validation_caps_medical_priority_and_removes_uncertain_welfare_suffix() -> None:
    breathing = IncidentAnalyzer._validate_incident(  # noqa: SLF001
        {
            "event_type": "medical",
            "title": "Medical emergency - breathing difficulty",
            "summary": "A female was reported having trouble breathing.",
            "location": "North Allen Road",
            "priority": 5,
            "confidence": 0.9,
            "evidence_segment_ids": [0],
            "attributes": {},
        },
        {0},
        {
            0: {
                "segment_index": 0,
                "start_seconds": 0.0,
                "end_seconds": 8.0,
                "speaker": None,
                "text": "North Allen Road, a female is having trouble breathing.",
            }
        },
    )
    welfare = IncidentAnalyzer._validate_incident(  # noqa: SLF001
        {
            "event_type": "welfare_check",
            "title": "Welfare check on female",
            "summary": "Officers were sent to North Delaware, Texas for a welfare check.",
            "location": "2306 North Delaware, Texas",
            "priority": 2,
            "confidence": 0.8,
            "evidence_segment_ids": [1],
            "attributes": {},
        },
        {1},
        {
            1: {
                "segment_index": 1,
                "start_seconds": 10.0,
                "end_seconds": 18.0,
                "speaker": None,
                "text": "2306 North Delaware, Texas welfare of Summer Gibson.",
            }
        },
    )

    assert breathing is not None
    assert breathing["priority"] == 4
    assert welfare is not None
    assert welfare["location"] == "2306 North Delaware"
    assert "Texas" not in welfare["summary"]


def test_daily_summary_rejects_unsupported_outcome_language() -> None:
    issues = IncidentAnalyzer._daily_summary_grounding_issues(  # noqa: SLF001
        "A shots-fired report was later identified as fireworks.",
        {1},
        1,
        [
            "Report of shots fired. Residents separately reported hearing fireworks."
        ],
    )

    assert any("identified fireworks" in issue for issue in issues)


def test_incident_validation_removes_unsupported_outcome_plate_and_arson_language() -> None:
    hit_and_run = IncidentAnalyzer._validate_incident(  # noqa: SLF001
        {
            "event_type": "traffic_collision",
            "title": "Hit and run",
            "summary": (
                "A hit and run involving a gray SUV was reported on Hightower. "
                "The driver was identified as a male in a green shirt."
            ),
            "location": "S-E-V-8-9-5-9-I-M-Charles-0-9-5",
            "priority": 4,
            "confidence": 0.8,
            "evidence_segment_ids": [0],
            "attributes": {},
        },
        {0},
        {
            0: {
                "segment_index": 0,
                "start_seconds": 0.0,
                "end_seconds": 12.0,
                "speaker": None,
                "text": (
                    "Hit and run, gray SUV north on Hightower. "
                    "The driver is a male in a green shirt."
                ),
            }
        },
    )
    fire = IncidentAnalyzer._validate_incident(  # noqa: SLF001
        {
            "event_type": "fire",
            "title": "Arson investigation at Fallen Oak",
            "summary": "A male was reported setting something on fire at Fallen Oak.",
            "location": "Fallen Oak",
            "priority": 3,
            "confidence": 0.8,
            "evidence_segment_ids": [1],
            "attributes": {},
        },
        {1},
        {
            1: {
                "segment_index": 1,
                "start_seconds": 20.0,
                "end_seconds": 28.0,
                "speaker": None,
                "text": "Fallen Oak, a male set something on fire on the patio.",
            }
        },
    )

    assert hit_and_run is not None
    assert "identified" not in hit_and_run["summary"].lower()
    assert hit_and_run["location"] is None
    assert fire is not None
    assert "arson" not in fire["title"].lower()
    assert fire["event_type"] == "fire"


def test_incident_claim_requires_meaningful_support_in_cited_evidence() -> None:
    assert incident_claim_has_evidence_support(
        "Hit and run on Hightower",
        "A gray minivan left northbound after a hit and run.",
        "Hightower",
        {},
        [{"text": "Hit and run, gray minivan went north on Hightower."}],
    )
    assert not incident_claim_has_evidence_support(
        "Stolen squad car in Example Township",
        "A stolen squad car was located at Bruch and Garfield.",
        "Example Township",
        {"vehicle_type": "squad car"},
        [
            {"text": "One subject is under arrest for transport."},
            {"text": "Recent domestic at this location; she is waiting in a black car."},
        ],
    )
    assert not incident_claim_has_evidence_support(
        "Search warrant execution on Main Street",
        "Officers executed a search warrant on Main Street.",
        "Main Street",
        {},
        [{"text": "Officers are searching for a subject near Main Street."}],
    )
    assert not incident_claim_has_evidence_support(
        "Vehicle pursuit on Jefferson",
        "A vehicle pursuit continued on Jefferson.",
        "Jefferson",
        {},
        [{"text": "A vehicle was traveling the wrong way on Jefferson."}],
    )


def test_incident_validation_rejects_unsupported_claim_and_scattered_quotes() -> None:
    segments = {
        0: {
            "segment_index": 0,
            "start_seconds": 0.0,
            "end_seconds": 5.0,
            "speaker": "SPEAKER_00",
            "text": "One subject is under arrest for transport.",
        },
        1: {
            "segment_index": 1,
            "start_seconds": 1_000.0,
            "end_seconds": 1_005.0,
            "speaker": "SPEAKER_01",
            "text": "Follow-up on reported shots fired at Main Street.",
        },
    }
    unsupported = IncidentAnalyzer._validate_incident(  # noqa: SLF001
        {
            "event_type": "vehicle_theft",
            "title": "Stolen squad car in Example Township",
            "summary": "A stolen squad car was located at Bruch and Garfield.",
            "location": "Example Township",
            "priority": 4,
            "confidence": 0.9,
            "evidence_segment_ids": [0],
            "attributes": {},
        },
        {0, 1},
        segments,
    )
    scattered = IncidentAnalyzer._validate_incident(  # noqa: SLF001
        {
            "event_type": "shots_fired",
            "title": "Reported shots fired at Main Street",
            "summary": "Radio traffic reported shots fired at Main Street.",
            "location": "Main Street",
            "priority": 4,
            "confidence": 0.9,
            "evidence_segment_ids": [0, 1],
            "attributes": {},
        },
        {0, 1},
        segments,
    )

    assert unsupported is None
    assert scattered is None


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


def test_incident_deduplication_merges_contained_evidence() -> None:
    common = {
        "event_type": "burglary",
        "location": "West Ham",
        "start_seconds": 10.0,
        "end_seconds": 30.0,
        "priority": 4,
        "confidence": 0.8,
        "attributes": {},
    }
    incidents = [
        {
            **common,
            "fingerprint": "first",
            "title": "Burglary in progress",
            "summary": "Four people were reported entering the property.",
            "evidence": [
                {"segment_index": 1},
                {"segment_index": 2},
                {"segment_index": 3},
            ],
        },
        {
            **common,
            "fingerprint": "second",
            "title": "Trespassers on property",
            "summary": "People were reported entering the property.",
            "evidence": [{"segment_index": 1}],
        },
    ]

    assert len(IncidentAnalyzer._deduplicate(incidents)) == 1  # noqa: SLF001


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


def test_archive_time_never_presents_an_audio_offset_as_clock_time() -> None:
    assert format_archive_time({"archive_date": "2026-07-11"}, 3_661) == (
        "2026-07-11 at archive offset 01:01:01 (clock time unavailable)"
    )


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


def test_critical_phrase_fallback_recovers_model_omissions_without_negated_calls(
    tmp_path: Path,
) -> None:
    archive_date = date(2026, 7, 17)
    transcript = tmp_path / "transcript.json"
    transcript.write_text(
        json.dumps(
            {
                "model": "base",
                "duration": 900.0,
                "segments": [
                    {
                        "start": 10.0,
                        "end": 15.0,
                        "text": "Example Township Police just had a squad car stolen.",
                    },
                    {
                        "start": 400.0,
                        "end": 405.0,
                        "text": "At least one caller in the area reported shots fire.",
                    },
                    {
                        "start": 800.0,
                        "end": 805.0,
                        "text": "No shots fired were reported at that address.",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    class OmissionClient:
        model = "omission-test-model"

        def __init__(self) -> None:
            self.incident_calls = 0

        def chat_json(
            self, *_args: object, **kwargs: object
        ) -> dict[str, object]:
            if kwargs["schema_name"] == "police_radio_incidents":
                self.incident_calls += 1
                return {"incidents": []}
            return {
                "summary": "I1 and I2 were reported from retained radio evidence."
            }

    messages: list[str] = []
    client = OmissionClient()
    with AnalysisStore(tmp_path / "analysis.sqlite3") as store:
        store.import_transcript("90001", archive_date, transcript)
        result = IncidentAnalyzer(
            store,
            client,
            progress=messages.append,
        ).analyze_day("90001", archive_date)
        incidents = store.get_incidents("90001", archive_date, archive_date)

    assert result["incidents"] == 2
    assert client.incident_calls == 1
    assert {value["event_type"] for value in incidents} == {
        "shots_fired",
        "vehicle_theft",
    }
    assert {
        value["evidence"][0]["segment_index"] for value in incidents
    } == {0, 1}
    assert all(
        "No shots fired" not in value["evidence"][0]["text"]
        for value in incidents
    )
    assert any("exact-phrase critical-event" in value for value in messages)


def test_evidence_policy_upgrade_reuses_unchanged_llm_window_checkpoint(
    tmp_path: Path,
) -> None:
    archive_date = date(2026, 7, 17)
    transcript = tmp_path / "transcript.json"
    transcript.write_text(
        json.dumps(
            {
                "model": "base",
                "duration": 300.0,
                "segments": [
                    {
                        "start": 10.0,
                        "end": 15.0,
                        "text": "Routine radio check.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    class CountingClient:
        model = "policy-upgrade-test-model"

        def __init__(self) -> None:
            self.incident_calls = 0

        def chat_json(
            self, *_args: object, **kwargs: object
        ) -> dict[str, object]:
            if kwargs["schema_name"] == "police_radio_incidents":
                self.incident_calls += 1
                return {"incidents": []}
            return {"summary": "No clearly supported eventful incidents."}

    messages: list[str] = []
    client = CountingClient()
    with AnalysisStore(tmp_path / "analysis.sqlite3") as store:
        store.import_transcript("90001", archive_date, transcript)
        IncidentAnalyzer(
            store,
            client,
            prompt_version=WINDOW_PROMPT_VERSION,
        ).analyze_day("90001", archive_date)
        day = store.get_day("90001", archive_date)
        assert day is not None
        with store.transaction() as connection:
            connection.execute(
                "DELETE FROM daily_summaries WHERE day_id=?",
                (int(day["id"]),),
            )
        upgraded = IncidentAnalyzer(
            store,
            client,
            prompt_version=PROMPT_VERSION,
            progress=messages.append,
        ).analyze_day("90001", archive_date)

        assert store.get_daily_summary(
            int(day["id"]),
            client.model,
            PROMPT_VERSION,
            str(day["transcript_sha256"]),
        ) is not None

    assert upgraded["windows"] == 1
    assert client.incident_calls == 1
    assert any(
        value.startswith("Reusing saved analysis window 1/1")
        for value in messages
    )


def test_evidence_policy_upgrade_migrates_exact_legacy_run_and_adds_fallback(
    tmp_path: Path,
) -> None:
    archive_date = date(2026, 7, 16)
    transcript = tmp_path / "transcript.json"
    transcript.write_text(
        json.dumps(
            {
                "model": "base",
                "duration": 300.0,
                "segments": [
                    {
                        "start": 10.0,
                        "end": 15.0,
                        "text": "Example Township Police just had a squad car stolen.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    class CountingClient:
        model = "policy-migration-test-model"

        def __init__(self) -> None:
            self.incident_calls = 0

        def chat_json(
            self, *_args: object, **kwargs: object
        ) -> dict[str, object]:
            if kwargs["schema_name"] == "police_radio_incidents":
                self.incident_calls += 1
                return {"incidents": []}
            return {"summary": "Retained radio evidence was reviewed."}

    messages: list[str] = []
    client = CountingClient()
    with AnalysisStore(tmp_path / "analysis.sqlite3") as store:
        store.import_transcript("90001", archive_date, transcript)
        initial = IncidentAnalyzer(
            store,
            client,
            prompt_version=WINDOW_PROMPT_VERSION,
        ).analyze_day("90001", archive_date)
        upgraded = IncidentAnalyzer(
            store,
            client,
            prompt_version=PROMPT_VERSION,
            progress=messages.append,
        ).analyze_day("90001", archive_date)
        incidents = store.get_incidents_for_run(
            int(store.get_day("90001", archive_date)["id"]),
            client.model,
            PROMPT_VERSION,
        )

    assert initial["incidents"] == 0
    assert upgraded["incidents"] == 1
    assert client.incident_calls == 1
    assert incidents[0]["event_type"] == "vehicle_theft"
    assert incidents[0]["evidence"][0]["segment_index"] == 0
    assert any(
        value.startswith("Reusing 0 saved model incidents")
        for value in messages
    )


def test_incident_analysis_resumes_after_the_last_completed_model_window(
    tmp_path: Path,
) -> None:
    archive_date = date(2026, 7, 12)
    transcript = tmp_path / "transcript.json"
    transcript.write_text(
        json.dumps(
            {
                "model": "turbo",
                "duration": 10_020.0,
                "segments": [
                    {
                        "start": 10.0,
                        "end": 15.0,
                        "text": "Possible shots fired near Main and First.",
                    },
                    {
                        "start": 10_000.0,
                        "end": 10_005.0,
                        "text": "Possible shots fired near Oak and Ninth.",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    client = InterruptingWindowClient()

    with AnalysisStore(tmp_path / "analysis.sqlite3") as store:
        store.import_transcript("90001", archive_date, transcript)
        analyzer = IncidentAnalyzer(store, client)
        with pytest.raises(RuntimeError, match="simulated process interruption"):
            analyzer.analyze_day("90001", archive_date)

        assert store.stats()["analysis_window_checkpoints"] == 1
        resumed = analyzer.analyze_day("90001", archive_date)
        incidents = store.get_incidents("90001", archive_date, archive_date)

    assert resumed["windows"] == 2
    assert resumed["incidents"] == 2
    assert len(incidents) == 2
    assert client.incident_calls == 3


def test_incident_analysis_subdivides_length_limited_window_and_checkpoints_parent(
    tmp_path: Path,
) -> None:
    archive_date = date(2026, 7, 12)
    transcript = tmp_path / "transcript.json"
    transcript.write_text(
        json.dumps(
            {
                "model": "turbo",
                "duration": 300.0,
                "segments": [
                    {
                        "start": float(index * 30),
                        "end": float(index * 30 + 10),
                        "text": f"Routine retained radio segment {index}.",
                    }
                    for index in range(4)
                ],
            }
        ),
        encoding="utf-8",
    )

    class LengthLimitedClient:
        model = "length-limited-test-model"

        def __init__(self) -> None:
            self.incident_calls = 0

        def chat_json(
            self, *_args: object, **kwargs: object
        ) -> dict[str, object]:
            if kwargs["schema_name"] == "police_radio_incidents":
                self.incident_calls += 1
                if self.incident_calls == 1:
                    raise LlamaServerError(
                        "Local model returned invalid JSON (finish_reason=length)"
                    )
                return {"incidents": []}
            return {"summary": "No clearly supported eventful incidents."}

    messages: list[str] = []
    client = LengthLimitedClient()
    with AnalysisStore(tmp_path / "analysis.sqlite3") as store:
        store.import_transcript("90001", archive_date, transcript)
        analyzer = IncidentAnalyzer(store, client, progress=messages.append)
        result = analyzer.analyze_day("90001", archive_date)
        first_call_count = client.incident_calls
        resumed = analyzer.analyze_day("90001", archive_date)
        checkpoint_count = store.stats()["analysis_window_checkpoints"]

    assert result["windows"] == 1
    assert resumed["windows"] == 1
    assert first_call_count == 3
    assert client.incident_calls == first_call_count
    assert checkpoint_count == 1
    assert any("subdividing" in message for message in messages)


def test_incident_analysis_subdivides_first_unfinished_window_on_resume(
    tmp_path: Path,
) -> None:
    archive_date = date(2026, 7, 12)
    transcript = tmp_path / "transcript.json"
    transcript.write_text(
        json.dumps(
            {
                "model": "turbo",
                "duration": 10_020.0,
                "segments": [
                    {"start": 10.0, "end": 15.0, "text": "First window."},
                    *[
                        {
                            "start": float(8_000 + index * 30),
                            "end": float(8_010 + index * 30),
                            "text": f"Second-window segment {index}.",
                        }
                        for index in range(4)
                    ],
                ],
            }
        ),
        encoding="utf-8",
    )

    class InterruptedDenseClient:
        model = "interrupted-dense-test-model"

        def __init__(self) -> None:
            self.incident_calls = 0
            self.segment_counts: list[int] = []

        def chat_json(
            self, *_args: object, **kwargs: object
        ) -> dict[str, object]:
            if kwargs["schema_name"] == "police_radio_incidents":
                self.incident_calls += 1
                self.segment_counts.append(
                    sum(
                        line.startswith("S")
                        for line in str(kwargs["user"]).splitlines()
                    )
                )
                if self.incident_calls == 2:
                    raise RuntimeError("simulated interruption in dense window")
                return {"incidents": []}
            return {"summary": "No clearly supported eventful incidents."}

    messages: list[str] = []
    client = InterruptedDenseClient()
    with AnalysisStore(tmp_path / "analysis.sqlite3") as store:
        store.import_transcript("90001", archive_date, transcript)
        analyzer = IncidentAnalyzer(store, client, progress=messages.append)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            analyzer.analyze_day("90001", archive_date)
        result = analyzer.analyze_day("90001", archive_date)

    assert result["windows"] == 2
    assert client.segment_counts == [1, 4, 2, 2]
    assert any("first unfinished" in message for message in messages)


def test_incident_analysis_reuses_unchanged_windows_after_append(
    tmp_path: Path,
) -> None:
    archive_date = date(2026, 7, 12)
    transcript = tmp_path / "transcript.json"

    class CountingClient:
        model = "append-window-test-model"

        def __init__(self) -> None:
            self.incident_calls = 0

        def chat_json(
            self, *_args: object, **kwargs: object
        ) -> dict[str, object]:
            if kwargs["schema_name"] == "police_radio_incidents":
                self.incident_calls += 1
                return {"incidents": []}
            return {"summary": "No clearly supported eventful incidents."}

    def write_transcript(include_append: bool) -> None:
        segments = [
            {"start": 10.0, "end": 15.0, "text": "Routine radio check."},
            {"start": 8_000.0, "end": 8_005.0, "text": "Routine radio check."},
            {"start": 14_500.0, "end": 14_505.0, "text": "Routine radio check."},
        ]
        if include_append:
            segments.append(
                {
                    "start": 22_000.0,
                    "end": 22_005.0,
                    "text": "Routine radio check.",
                }
            )
        transcript.write_text(
            json.dumps(
                {
                    "model": "turbo",
                    "duration": segments[-1]["end"],
                    "segments": segments,
                }
            ),
            encoding="utf-8",
        )

    client = CountingClient()
    with AnalysisStore(tmp_path / "analysis.sqlite3") as store:
        write_transcript(False)
        store.import_transcript("90001", archive_date, transcript)
        first = IncidentAnalyzer(store, client).analyze_day(
            "90001",
            archive_date,
        )
        assert first["windows"] == 3
        assert client.incident_calls == 3

        write_transcript(True)
        store.import_transcript("90001", archive_date, transcript)
        messages: list[str] = []
        second = IncidentAnalyzer(
            store,
            client,
            progress=messages.append,
        ).analyze_day("90001", archive_date)

        assert second["windows"] == 4
        assert client.incident_calls == 5
        assert sum(
            message.startswith("Reusing saved analysis window")
            for message in messages
        ) == 2
        assert store.stats()["analysis_window_checkpoints"] == 4


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
