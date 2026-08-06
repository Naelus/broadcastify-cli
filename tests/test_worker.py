import io
import json
import os
import sys
import time
from contextlib import nullcontext
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from broadcastify_cli.analysis import PROMPT_VERSION
from broadcastify_cli.area_watch import (
    AREA_PROMPT_VERSION,
    current_area_story_source_fingerprint,
)
from broadcastify_cli.library import require_current_range_evidence
from broadcastify_cli.storage import AnalysisStore
from broadcastify_cli.worker import (
    _day_report,
    _incident_clip,
    analysis_days,
    analysis_self_test,
    archive_quota_status,
    asr_self_test,
    diarization_self_test,
    emit,
    load_worker_environment,
    latest_area_digest,
    library_resume_plan,
    prepare_asr_model_command,
    profile_self_test,
)


def test_emit_is_safe_on_a_legacy_windows_console(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buffer = io.BytesIO()
    console = io.TextIOWrapper(buffer, encoding="cp1252", newline="\n")
    monkeypatch.setattr(sys, "stdout", console)

    emit({"message": "quota reached \u2192 resume later"})
    console.flush()

    encoded = buffer.getvalue().decode("cp1252")
    assert "\\u2192" in encoded
    assert json.loads(encoded)["message"] == "quota reached \u2192 resume later"


def test_archive_quota_status_mints_one_persistent_installation_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = tmp_path / "quota.sqlite3"
    monkeypatch.setenv("BROADCASTIFY_QUOTA_LEDGER", str(ledger))
    emitted: list[dict[str, object]] = []
    monkeypatch.setattr("broadcastify_cli.worker.emit", emitted.append)

    assert archive_quota_status() == 0
    assert archive_quota_status() == 0

    statuses = [value["status"] for value in emitted]
    assert statuses[0]["instance_id"] == statuses[1]["instance_id"]
    assert statuses[0]["automated_limit"] == 240
    assert statuses[0]["user_reserve"] == 10


def test_library_resume_planning_reads_only_local_state_and_quota(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitted: list[dict[str, object]] = []

    class FakeLedger:
        def status(self) -> dict[str, object]:
            return {"available": False, "remaining": 0}

    class ForbiddenBroadcastifyClient:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("resume planning must not create a website client")

    class FakeStore:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def list_feed_schedules(self) -> list[dict[str, object]]:
            return []

    monkeypatch.setattr(
        "broadcastify_cli.worker.scan_local_library",
        lambda *_args: [
            {
                "feed_id": "90001",
                "archive_date": "2026-07-12",
                "is_complete": False,
                "needs_network": True,
            }
        ],
    )
    monkeypatch.setattr("broadcastify_cli.worker.ArchiveRequestLedger", FakeLedger)
    monkeypatch.setattr("broadcastify_cli.worker.AnalysisStore", FakeStore)
    monkeypatch.setattr(
        "broadcastify_cli.worker.BroadcastifyClient",
        ForbiddenBroadcastifyClient,
    )
    monkeypatch.setattr("broadcastify_cli.worker.emit", emitted.append)

    assert library_resume_plan("unused") == 0
    assert emitted[0]["type"] == "library_resume_plan"
    assert emitted[0]["network_count"] == 1
    assert emitted[0]["quota"] == {"available": False, "remaining": 0}


def test_asr_self_test_uses_selected_engine_without_returning_transcript_text(
    monkeypatch,
) -> None:
    emitted: list[dict[str, object]] = []
    initialized: dict[str, object] = {}

    class FakeTranscriber:
        asr_engine = "openvino"
        backend_description = "OpenVINO CPU (fallback from NPU)"
        device = "openvino-npu"

        def __init__(self, **kwargs: object) -> None:
            initialized.update(kwargs)

        def transcribe_file(self, audio_path: Path, progress=None) -> Path:
            assert audio_path.is_file()
            if progress:
                progress("Synthetic audio decoded")
            transcript_dir = audio_path.parent / "transcripts"
            transcript_dir.mkdir()
            result = transcript_dir / "silence.json"
            result.write_text(
                json.dumps(
                    {
                        "asr_engine": self.asr_engine,
                        "asr_backend": self.backend_description,
                        "device": self.device,
                        "text": "never return this transcript",
                        "segments": [],
                        "words": [],
                        "asr_metadata": {
                            "fallback_reason": "NPU compilation failed",
                            "fallback_stage": "initialization",
                        },
                    }
                ),
                encoding="utf-8",
            )
            return result

    monkeypatch.setattr("broadcastify_cli.worker.LocalTranscriber", FakeTranscriber)
    monkeypatch.setattr("broadcastify_cli.worker.emit", emitted.append)

    exit_code = asr_self_test(
        {
            "model": "tiny",
            "asr_engine": "openvino",
            "device": "openvino-npu",
            "batch_size": 4,
        }
    )

    result = next(value["result"] for value in emitted if value["type"] == "asr_self_test")
    assert exit_code == 0
    assert initialized["model_name"] == "tiny"
    assert initialized["device"] == "openvino-npu"
    assert result["ready"] is True
    assert result["fallback_stage"] == "initialization"
    assert "never return" not in json.dumps(result)


def test_diarization_self_test_executes_generated_audio_without_returning_token(
    monkeypatch,
) -> None:
    emitted: list[dict[str, object]] = []
    loaded: dict[str, object] = {}

    class FakeAnnotation:
        def itertracks(self, *, yield_label: bool = False):
            assert yield_label is True
            yield object(), object(), "SPEAKER_00"

    class FakePipeline:
        embedding_batch_size = 2

        def __call__(self, audio: dict[str, object]) -> FakeAnnotation:
            assert audio["sample_rate"] == 16_000
            return FakeAnnotation()

    def fake_load(**kwargs: object) -> tuple[FakePipeline, str]:
        loaded.update(kwargs)
        return FakePipeline(), "cpu"

    monkeypatch.setattr("broadcastify_cli.worker._load_diarization_pipeline", fake_load)
    monkeypatch.setattr(
        "broadcastify_cli.worker.decoded_diarization_audio",
        lambda _path: nullcontext({"waveform": object(), "sample_rate": 16_000}),
    )
    monkeypatch.setattr("broadcastify_cli.worker.emit", emitted.append)

    exit_code = diarization_self_test(
        {
            "huggingface_token": "private-test-token",
            "diarization_device": "cpu",
            "batch_size": 4,
        }
    )

    result = next(
        value["result"]
        for value in emitted
        if value["type"] == "diarization_self_test"
    )
    assert exit_code == 0
    assert loaded["token"] == "private-test-token"
    assert result["ready"] is True
    assert result["turn_count"] == 1
    assert "private-test-token" not in json.dumps(emitted)


def test_prepare_asr_model_emits_managed_path_without_returning_token(
    monkeypatch, tmp_path: Path
) -> None:
    emitted: list[dict[str, object]] = []
    received: dict[str, object] = {}
    model_path = tmp_path / "whisper-tiny-fp32-cpu"

    def fake_prepare(settings, progress=None):
        received.update(settings)
        assert progress is not None
        progress("Prepared local model")
        return {
            "ready": True,
            "engine": "windows-ml",
            "model": "tiny",
            "source_model": "openai/whisper-tiny",
            "provider": "cpu",
            "precision": "fp32",
            "path": str(model_path),
            "reused": False,
            "bytes": 123,
            "message": "Prepared Windows ML tiny.",
        }

    monkeypatch.setattr("broadcastify_cli.worker.prepare_asr_model", fake_prepare)
    monkeypatch.setattr("broadcastify_cli.worker.emit", emitted.append)

    exit_code = prepare_asr_model_command(
        {
            "model": "tiny",
            "asr_engine": "windows-ml",
            "huggingface_token": "private-test-token",
        }
    )

    result = next(
        value["result"]
        for value in emitted
        if value["type"] == "asr_model_prepared"
    )
    assert exit_code == 0
    assert received["model"] == "tiny"
    assert result["path"] == str(model_path)
    assert "private-test-token" not in json.dumps(emitted)


def test_diarization_self_test_can_reuse_cached_model_without_token(monkeypatch) -> None:
    emitted: list[dict[str, object]] = []
    loaded: dict[str, object] = {}

    class FakeAnnotation:
        def itertracks(self, *, yield_label: bool = False):
            assert yield_label is True
            yield object(), object(), "SPEAKER_00"

    class FakePipeline:
        embedding_batch_size = 1

        def __call__(self, _audio: dict[str, object]) -> FakeAnnotation:
            return FakeAnnotation()

    def fake_load(**kwargs: object) -> tuple[FakePipeline, str]:
        loaded.update(kwargs)
        return FakePipeline(), "cpu"

    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr("broadcastify_cli.worker._load_diarization_pipeline", fake_load)
    monkeypatch.setattr(
        "broadcastify_cli.worker.decoded_diarization_audio",
        lambda _path: nullcontext({"waveform": object(), "sample_rate": 16_000}),
    )
    monkeypatch.setattr("broadcastify_cli.worker.emit", emitted.append)

    assert diarization_self_test({"diarization_device": "cpu"}) == 0
    assert loaded["token"] == ""
    assert any(value["type"] == "diarization_self_test" for value in emitted)


def test_portable_diarization_self_test_prepares_and_executes_public_models(
    monkeypatch,
) -> None:
    emitted: list[dict[str, object]] = []
    prepared: dict[str, object] = {}
    initialized: dict[str, object] = {}

    def fake_prepare(*, progress=None):
        prepared["called"] = True
        assert progress is not None
        progress("Verified public models")
        return {
            "ready": True,
            "model": "segmentation+titanet",
        }

    class FakeDiarizer:
        def __init__(self, **kwargs: object) -> None:
            initialized.update(kwargs)
            self.metadata = {
                "cluster_threshold": 0.95,
                "speaker_identity_scope": "processing-chunk",
            }

        def process(self, path: Path, progress=None):
            assert path.is_file()
            assert progress is not None
            progress("Portable execution complete")
            return [SimpleNamespace(start=0.0, end=1.0, speaker="SPEAKER_00")]

    monkeypatch.setattr(
        "broadcastify_cli.worker.prepare_portable_diarization_model",
        fake_prepare,
    )
    monkeypatch.setattr(
        "broadcastify_cli.worker.SherpaOnnxDiarizer", FakeDiarizer
    )
    monkeypatch.setattr("broadcastify_cli.worker.emit", emitted.append)

    assert (
        diarization_self_test(
            {
                "diarization_engine": "sherpa-onnx",
                "diarization_device": "cpu",
                "min_speakers": 2,
                "max_speakers": 8,
            }
        )
        == 0
    )

    result = next(
        value["result"]
        for value in emitted
        if value["type"] == "diarization_self_test"
    )
    assert prepared["called"] is True
    assert initialized == {"min_speakers": 2, "max_speakers": 8}
    assert result["engine"] == "sherpa-onnx"
    assert result["quality"] == "preview"
    assert result["turn_count"] == 1


def test_portable_diarization_self_test_rejects_cuda_before_model_work(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "broadcastify_cli.worker.prepare_portable_diarization_model",
        lambda **_kwargs: pytest.fail("model preparation should not start"),
    )

    with pytest.raises(RuntimeError, match="currently runs on CPU"):
        diarization_self_test(
            {
                "diarization_engine": "sherpa-onnx",
                "diarization_device": "cuda",
            }
        )


def test_analysis_self_test_executes_structured_synthetic_generation(
    monkeypatch,
) -> None:
    emitted: list[dict[str, object]] = []
    request: dict[str, object] = {}

    class FakeClient:
        model = "local-test-model.gguf"

        def chat_json(
            self,
            system_prompt: str,
            user_prompt: str,
            schema_name: str,
            schema: dict[str, object],
            *,
            max_tokens: int,
        ) -> dict[str, object]:
            request.update(
                {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "schema_name": schema_name,
                    "schema": schema,
                    "max_tokens": max_tokens,
                }
            )
            return {"ready": True}

    monkeypatch.setattr(
        "broadcastify_cli.worker.open_analysis_client",
        lambda _config: nullcontext(FakeClient()),
    )
    monkeypatch.setattr("broadcastify_cli.worker.emit", emitted.append)

    exit_code = analysis_self_test(
        {
            "analysis_provider": "local",
            "analysis_model": "requested-test-model.gguf",
            "analysis_device": "cpu",
            "analysis_api_key": "private-test-key",
        }
    )

    result = next(
        value["result"]
        for value in emitted
        if value["type"] == "analysis_self_test"
    )
    assert exit_code == 0
    assert request["schema_name"] == "runtime_readiness"
    assert request["max_tokens"] == 32
    assert "synthetic" in str(request["user_prompt"]).lower()
    assert result["ready"] is True
    assert result["verified"] is True
    assert result["device"] == "cpu"
    assert result["model"] == "local-test-model.gguf"
    assert "private-test-key" not in json.dumps(emitted)


def test_profile_self_test_runs_all_stages_and_returns_one_verification(
    monkeypatch,
) -> None:
    emitted: list[dict[str, object]] = []
    calls: list[str] = []

    def stage(name: str, **values: object):
        def run(_settings: dict[str, object]) -> dict[str, object]:
            calls.append(name)
            return {"ready": True, "message": f"{name} passed", **values}

        return run

    monkeypatch.setattr(
        "broadcastify_cli.worker._asr_self_test_result",
        stage("transcription", backend="whisper.cpp / Vulkan"),
    )
    monkeypatch.setattr(
        "broadcastify_cli.worker._diarization_self_test_result",
        stage("diarization", device="cpu"),
    )
    monkeypatch.setattr(
        "broadcastify_cli.worker._analysis_self_test_result",
        stage("analysis", verified=True, device="auto"),
    )
    monkeypatch.setattr("broadcastify_cli.worker.emit", emitted.append)

    assert profile_self_test({"analysis_api_key": "private-test-key"}) == 0

    result = next(
        value["result"]
        for value in emitted
        if value["type"] == "profile_self_test"
    )
    assert calls == ["transcription", "diarization", "analysis"]
    assert result["ready"] is True
    assert result["verified"] is True
    assert set(result["results"]) == {"transcription", "diarization", "analysis"}
    assert [value["status"] for value in emitted if value["type"] == "profile_self_test_stage"] == [
        "running",
        "passed",
        "running",
        "passed",
        "running",
        "passed",
    ]
    assert "private-test-key" not in json.dumps(emitted)


def test_profile_self_test_stops_at_first_failed_stage(monkeypatch) -> None:
    emitted: list[dict[str, object]] = []
    analysis_called = False

    monkeypatch.setattr(
        "broadcastify_cli.worker._asr_self_test_result",
        lambda _settings: {"ready": True, "message": "transcription passed"},
    )

    def fail_diarization(_settings: dict[str, object]) -> dict[str, object]:
        raise RuntimeError("Community-1 cache is incomplete")

    def analysis(_settings: dict[str, object]) -> dict[str, object]:
        nonlocal analysis_called
        analysis_called = True
        return {"ready": True}

    monkeypatch.setattr(
        "broadcastify_cli.worker._diarization_self_test_result",
        fail_diarization,
    )
    monkeypatch.setattr(
        "broadcastify_cli.worker._analysis_self_test_result",
        analysis,
    )
    monkeypatch.setattr("broadcastify_cli.worker.emit", emitted.append)

    assert profile_self_test({}) == 0

    result = next(
        value["result"]
        for value in emitted
        if value["type"] == "profile_self_test"
    )
    assert result["ready"] is False
    assert result["verified"] is False
    assert result["failed_stage"] == "diarization"
    assert "Community-1 cache is incomplete" in result["message"]
    assert set(result["results"]) == {"transcription"}
    assert result["recovery"]["kind"] == "configure-speakers"
    assert result["recovery"]["stage"] == "diarization"
    assert analysis_called is False


def test_profile_self_test_does_not_reuse_asr_model_for_analysis(
    monkeypatch,
) -> None:
    emitted: list[dict[str, object]] = []
    analysis_settings: dict[str, object] = {}
    monkeypatch.setattr(
        "broadcastify_cli.worker._asr_self_test_result",
        lambda _settings: {"ready": True, "message": "asr passed"},
    )
    monkeypatch.setattr(
        "broadcastify_cli.worker._diarization_self_test_result",
        lambda _settings: {"ready": True, "message": "speakers passed"},
    )

    def analysis(settings: dict[str, object]) -> dict[str, object]:
        analysis_settings.update(settings)
        return {
            "ready": True,
            "verified": True,
            "message": "analysis passed",
        }

    monkeypatch.setattr(
        "broadcastify_cli.worker._analysis_self_test_result",
        analysis,
    )
    monkeypatch.setattr("broadcastify_cli.worker.emit", emitted.append)

    assert profile_self_test({"model": "turbo", "analysis_model": ""}) == 0

    assert "model" not in analysis_settings
    assert analysis_settings["analysis_model"] == ""
    result = next(
        value["result"]
        for value in emitted
        if value["type"] == "profile_self_test"
    )
    assert result["ready"] is True


def test_profile_self_test_recovery_distinguishes_missing_whisper_runtime(
    monkeypatch,
) -> None:
    emitted: list[dict[str, object]] = []
    monkeypatch.setattr(
        "broadcastify_cli.worker._asr_self_test_result",
        lambda _settings: (_ for _ in ()).throw(
            RuntimeError("whisper-cli was not found")
        ),
    )
    monkeypatch.setattr("broadcastify_cli.worker.find_whisper_cpp", lambda: None)
    monkeypatch.setattr(
        "broadcastify_cli.worker.whisper_cpp_container_diagnostics",
        lambda: {"configured": False, "ready": False, "backend": "vulkan"},
    )
    monkeypatch.setattr(
        "broadcastify_cli.worker.whisper_cpp_backends",
        lambda _path: [],
    )
    monkeypatch.setattr("broadcastify_cli.worker.emit", emitted.append)

    assert profile_self_test(
        {"model": "turbo", "asr_engine": "whisper.cpp", "device": "vulkan"}
    ) == 0

    result = next(
        value["result"]
        for value in emitted
        if value["type"] == "profile_self_test"
    )
    assert result["recovery"]["kind"] == "configure-transcription"
    assert result["recovery"]["label"] == "Show Vulkan setup"


def test_profile_self_test_recovery_prepares_model_after_runtime_is_ready(
    monkeypatch,
) -> None:
    emitted: list[dict[str, object]] = []
    monkeypatch.setattr(
        "broadcastify_cli.worker._asr_self_test_result",
        lambda _settings: (_ for _ in ()).throw(
            RuntimeError("selected GGML model was not found")
        ),
    )
    monkeypatch.setattr(
        "broadcastify_cli.worker.find_whisper_cpp",
        lambda: "/opt/whisper-cli",
    )
    monkeypatch.setattr(
        "broadcastify_cli.worker.whisper_cpp_container_diagnostics",
        lambda: {"configured": False, "ready": False, "backend": "vulkan"},
    )
    monkeypatch.setattr(
        "broadcastify_cli.worker.whisper_cpp_backends",
        lambda _path: ["cpu", "vulkan"],
    )
    monkeypatch.setattr(
        "broadcastify_cli.worker.find_whisper_cpp_model",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr("broadcastify_cli.worker.emit", emitted.append)

    assert profile_self_test(
        {"model": "turbo", "asr_engine": "whisper.cpp", "device": "vulkan"}
    ) == 0

    result = next(
        value["result"]
        for value in emitted
        if value["type"] == "profile_self_test"
    )
    assert result["recovery"]["kind"] == "prepare-asr-model"
    assert result["recovery"]["label"] == "Download selected model"


def test_profile_self_test_bounds_native_runtime_dump(monkeypatch) -> None:
    emitted: list[dict[str, object]] = []
    runtime_dump = "\n".join(
        [
            "whisper_model_load: model metadata",
            "/source/ggml-backend.cpp:595: GGML_ASSERT(device) failed",
            *[f"/lib/frame-{index}.so(+0x1234)" for index in range(100)],
        ]
    )
    monkeypatch.setattr(
        "broadcastify_cli.worker._asr_self_test_result",
        lambda _settings: (_ for _ in ()).throw(RuntimeError(runtime_dump)),
    )
    monkeypatch.setattr("broadcastify_cli.worker.emit", emitted.append)

    assert profile_self_test({}) == 0

    result = next(
        value["result"]
        for value in emitted
        if value["type"] == "profile_self_test"
    )
    assert "GGML_ASSERT(device) failed" in result["message"]
    assert "individual transcription test" in result["message"]
    assert "frame-99" not in result["message"]
    assert len(result["message"]) < 700


def test_explicit_private_environment_overrides_repository_defaults(
    monkeypatch, tmp_path: Path
) -> None:
    (tmp_path / ".env").write_text(
        'BROADCASTIFY_USERNAME="repository-user"\n'
        'BROADCASTIFY_PASSWORD="repository-password"\n',
        encoding="utf-8",
    )
    bundled = tmp_path / "broadcastify-desktop.env"
    bundled.write_text(
        'BROADCASTIFY_USERNAME="bundled-user"\n'
        'BROADCASTIFY_PASSWORD="bundled-password"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BROADCASTIFY_ENV_FILE", str(bundled))
    monkeypatch.delenv("BROADCASTIFY_USERNAME", raising=False)
    monkeypatch.delenv("BROADCASTIFY_PASSWORD", raising=False)

    loaded = load_worker_environment()

    assert loaded == bundled
    assert os.environ["BROADCASTIFY_USERNAME"] == "bundled-user"
    assert os.environ["BROADCASTIFY_PASSWORD"] == "bundled-password"


def test_secure_store_values_override_private_environment(
    monkeypatch, tmp_path: Path
) -> None:
    bundled = tmp_path / "broadcastify-desktop.env"
    bundled.write_text(
        'BROADCASTIFY_USERNAME="bundled-user"\n'
        'BROADCASTIFY_PASSWORD="bundled-password"\n'
        'HUGGINGFACE_TOKEN="hf_bundled"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BROADCASTIFY_ENV_FILE", str(bundled))
    monkeypatch.setenv("BROADCASTIFY_SECURE_USERNAME", "secure-user")
    monkeypatch.setenv("BROADCASTIFY_SECURE_PASSWORD", "secure-password")
    monkeypatch.setenv("HUGGINGFACE_SECURE_TOKEN", "hf_secure")
    monkeypatch.setenv(
        "BROADCASTIFY_SECURE_ANALYSIS_DB",
        str(tmp_path / "selected-library.sqlite3"),
    )

    loaded = load_worker_environment()

    assert loaded == bundled
    assert os.environ["BROADCASTIFY_USERNAME"] == "secure-user"
    assert os.environ["BROADCASTIFY_PASSWORD"] == "secure-password"
    assert os.environ["HUGGINGFACE_TOKEN"] == "hf_secure"
    assert os.environ["BROADCASTIFY_ANALYSIS_DB"] == str(
        tmp_path / "selected-library.sqlite3"
    )


def test_day_report_exposes_playback_metadata_without_raw_evidence(tmp_path: Path) -> None:
    archive_date = date(2026, 7, 12)
    audio = tmp_path / "combined_90001_20260712.mp3"
    transcript = tmp_path / "transcript.json"
    audio.write_bytes(b"audio")
    transcript.write_text(
        json.dumps(
            {
                "model": "turbo",
                "duration": 600.0,
                "segments": [
                    {
                        "start": 10.0,
                        "end": 15.0,
                        "text": "Dispatch reports shots fired near Main and First.",
                        "speaker": "SPEAKER_00",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with AnalysisStore(tmp_path / "analysis.sqlite3") as store:
        imported = store.import_transcript(
            "90001", archive_date, transcript, audio
        )
        incident_ids = store.replace_incidents(
            imported.day_id,
            [
                {
                    "fingerprint": "test-incident",
                    "event_type": "shots_fired",
                    "title": "Reported shots fired",
                    "summary": "Units were sent to check a shots-fired report.",
                    "location": "Main and First",
                    "start_seconds": 10.0,
                    "end_seconds": 15.0,
                    "priority": 5,
                    "confidence": 0.9,
                    "evidence": ["sensitive transcript evidence"],
                    "attributes": {"private": "detail"},
                }
            ],
            model="test-model",
            prompt_version=PROMPT_VERSION,
        )
        store.save_daily_summary(
            imported.day_id,
            "A shots-fired report was dispatched.",
            incident_ids,
            model="test-model",
            prompt_version=PROMPT_VERSION,
            transcript_sha256=imported.transcript_sha256,
        )

        report = _day_report(store, "90001", archive_date)

    assert report["audio_path"] == str(audio.resolve())
    assert report["summary"] == "A shots-fired report was dispatched."
    assert len(report["incidents"]) == 1
    incident = report["incidents"][0]
    assert incident["start_seconds"] == 10.0
    assert incident["end_seconds"] == 15.0
    assert set(incident) == {
        "id",
        "event_type",
        "title",
        "summary",
        "location",
        "priority",
        "confidence",
        "start_seconds",
        "end_seconds",
        "evidence_quote",
        "archive_time",
    }
    assert incident["evidence_quote"] == ""
    assert "evidence" not in incident
    assert "attributes" not in incident


def test_day_report_hides_incidents_from_older_evidence_rules(tmp_path: Path) -> None:
    archive_date = date(2026, 7, 11)
    transcript = tmp_path / "transcript.json"
    transcript.write_text(
        json.dumps(
            {
                "duration": 30.0,
                "segments": [
                    {"start": 1.0, "end": 2.0, "text": "Older extracted claim."}
                ],
            }
        ),
        encoding="utf-8",
    )
    with AnalysisStore(tmp_path / "analysis.sqlite3") as store:
        imported = store.import_transcript("90001", archive_date, transcript)
        incident_ids = store.replace_incidents(
            imported.day_id,
            [
                {
                    "fingerprint": "old",
                    "event_type": "other",
                    "title": "Old claim",
                    "summary": "Old claim.",
                    "location": "",
                    "start_seconds": 1.0,
                    "end_seconds": 2.0,
                    "priority": 3,
                    "confidence": 0.7,
                    "evidence": [],
                    "attributes": {},
                }
            ],
            model="test-model",
            prompt_version="older-evidence-rules",
        )
        store.save_daily_summary(
            imported.day_id,
            "Older summary.",
            incident_ids,
            model="test-model",
            prompt_version="older-evidence-rules",
            transcript_sha256=imported.transcript_sha256,
        )

        report = _day_report(store, "90001", archive_date)

    assert report["summary"] == ""
    assert report["incidents"] == []
    assert report["analysis_current"] is False
    assert report["analysis_update_required"] is True


def _analyzed_worker_day(
    tmp_path: Path,
) -> tuple[Path, Path, Path, date, int]:
    archive_date = date(2026, 7, 29)
    audio = tmp_path / "combined_90001_20260729.mp3"
    transcript = tmp_path / "combined_90001_20260729.json"
    database = tmp_path / "analysis.sqlite3"
    audio.write_bytes(b"combined audio")
    transcript.write_text(
        json.dumps(
            {
                "duration": 30.0,
                "diarization_completed": True,
                "segments": [
                    {
                        "start": 1.0,
                        "end": 2.0,
                        "speaker": "SPEAKER_00",
                        "text": "Current retained evidence.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with AnalysisStore(database) as store:
        imported = store.import_transcript(
            "90001", archive_date, transcript, audio
        )
        incident_id = store.replace_incidents(
            imported.day_id,
            [
                {
                    "fingerprint": "current-worker-evidence",
                    "event_type": "other",
                    "title": "Current event",
                    "summary": "Current retained event.",
                    "location": "",
                    "start_seconds": 1.0,
                    "end_seconds": 2.0,
                    "priority": 3,
                    "confidence": 0.8,
                    "evidence": [],
                    "attributes": {},
                }
            ],
            model="test-model",
            prompt_version=PROMPT_VERSION,
        )[0]
        store.save_daily_summary(
            imported.day_id,
            "Current summary.",
            [incident_id],
            model="test-model",
            prompt_version=PROMPT_VERSION,
            transcript_sha256=imported.transcript_sha256,
        )
    return database, audio, transcript, archive_date, incident_id


def test_day_report_hides_current_prompt_results_after_audio_refresh(
    tmp_path: Path,
) -> None:
    database, audio, _transcript, archive_date, _incident_id = (
        _analyzed_worker_day(tmp_path)
    )
    audio.write_bytes(b"refreshed combined audio")
    future = time.time() + 10
    os.utime(audio, (future, future))

    with AnalysisStore(database) as store:
        report = _day_report(store, "90001", archive_date)

    assert report["summary"] == ""
    assert report["incidents"] == []
    assert report["analysis_current"] is False
    assert report["analysis_update_required"] is True


def test_day_report_hides_database_results_after_transcript_rewrite(
    tmp_path: Path,
) -> None:
    database, _audio, transcript, archive_date, _incident_id = (
        _analyzed_worker_day(tmp_path)
    )
    transcript.write_text(
        json.dumps(
            {
                "duration": 30.0,
                "segments": [
                    {
                        "start": 3.0,
                        "end": 4.0,
                        "text": "A newer transcript revision.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with AnalysisStore(database) as store:
        report = _day_report(store, "90001", archive_date)

    assert report["summary"] == ""
    assert report["incidents"] == []
    assert report["analysis_current"] is False
    assert report["analysis_update_required"] is True


def test_range_consumers_and_clip_reject_older_retained_revision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    database, audio, _transcript, archive_date, incident_id = (
        _analyzed_worker_day(tmp_path)
    )
    extracted = False

    def fail_extract(*_args, **_kwargs) -> Path:
        nonlocal extracted
        extracted = True
        raise AssertionError("Stale evidence must not be clipped")

    monkeypatch.setattr("broadcastify_cli.worker.extract_audio_clip", fail_extract)
    with AnalysisStore(database) as store:
        require_current_range_evidence(
            store,
            ["90001"],
            archive_date,
            archive_date,
            require_analysis=False,
            purpose="Question answering",
        )
        require_current_range_evidence(
            store,
            ["90001"],
            archive_date,
            archive_date,
            require_analysis=True,
            purpose="Summary",
        )

    audio.write_bytes(b"refreshed combined audio")
    future = time.time() + 10
    os.utime(audio, (future, future))
    with AnalysisStore(database) as store:
        with pytest.raises(ValueError, match="older than the retained files"):
            require_current_range_evidence(
                store,
                ["90001"],
                archive_date,
                archive_date,
                require_analysis=False,
                purpose="Question answering",
            )
        with pytest.raises(ValueError, match="older than the retained files"):
            require_current_range_evidence(
                store,
                ["90001"],
                archive_date,
                archive_date,
                require_analysis=True,
                purpose="Summary",
            )
        with pytest.raises(ValueError, match="older retained-evidence revision"):
            _incident_clip(store, incident_id)
    assert extracted is False


def test_analysis_day_list_uses_retained_revision_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    database, audio, _transcript, _archive_date, _incident_id = (
        _analyzed_worker_day(tmp_path)
    )
    emitted: list[dict[str, object]] = []
    monkeypatch.setattr("broadcastify_cli.worker.DEFAULT_DATABASE", database)
    monkeypatch.setattr("broadcastify_cli.worker.emit", emitted.append)

    assert analysis_days("90001") == 0
    current = emitted[-1]["days"][0]  # type: ignore[index]
    assert current["analysis_current"] is True
    assert current["incident_count"] == 1

    audio.write_bytes(b"refreshed combined audio")
    future = time.time() + 10
    os.utime(audio, (future, future))
    assert analysis_days("90001") == 0
    stale = emitted[-1]["days"][0]  # type: ignore[index]
    assert stale["analysis_current"] is False
    assert stale["analysis_update_required"] is True
    assert stale["segment_count"] == 0
    assert stale["incident_count"] == 0


def test_native_saved_area_digest_requires_exact_current_source_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    database, _audio, _transcript, archive_date, _incident_id = (
        _analyzed_worker_day(tmp_path)
    )
    with AnalysisStore(database) as store:
        profile = store.save_area_profile(
            "ExampleArea",
            ["00000"],
            [{"feed_id": "90001", "name": "Example Public Safety"}],
        )
        fingerprint = current_area_story_source_fingerprint(
            store,
            profile,
            archive_date,
            archive_date,
        )
        store.save_area_story_digest(
            int(profile["id"]),
            archive_date,
            archive_date,
            "Current saved area brief.",
            [],
            {"incident_prompt_version": PROMPT_VERSION},
            "test",
            AREA_PROMPT_VERSION,
            fingerprint,
        )

    emitted: list[dict[str, object]] = []
    monkeypatch.setattr("broadcastify_cli.worker.DEFAULT_DATABASE", database)
    monkeypatch.setattr("broadcastify_cli.worker.emit", emitted.append)
    assert latest_area_digest("ExampleArea") == 0
    current = emitted[-1]
    assert current["stale"] is False
    assert current["result"]["summary"] == "Current saved area brief."  # type: ignore[index]

    with AnalysisStore(database) as store:
        day = store.get_day("90001", archive_date)
        assert day is not None
        incident_ids = store.replace_incidents(
            int(day["id"]),
            [
                {
                    "fingerprint": "new-native-area-incident",
                    "event_type": "fire",
                    "title": "New current incident",
                    "summary": "A new current incident was extracted.",
                    "location": "",
                    "start_seconds": 3.0,
                    "end_seconds": 4.0,
                    "priority": 4,
                    "confidence": 0.9,
                    "evidence": [],
                    "attributes": {},
                }
            ],
            model="test",
            prompt_version=PROMPT_VERSION,
        )
        store.save_daily_summary(
            int(day["id"]),
            "New current daily summary.",
            incident_ids,
            model="test",
            prompt_version=PROMPT_VERSION,
            transcript_sha256=str(day["transcript_sha256"]),
        )

    assert latest_area_digest("ExampleArea") == 0
    stale = emitted[-1]
    assert stale["stale"] is True
    assert stale["result"] is None


def test_incident_clip_uses_a_timestamped_cache_key(
    monkeypatch, tmp_path: Path
) -> None:
    archive_date = date(2026, 7, 12)
    audio = tmp_path / "combined_90001_20260712.mp3"
    transcript = tmp_path / "transcript.json"
    audio.write_bytes(b"combined audio")
    transcript.write_text(
        json.dumps(
            {
                "model": "turbo",
                "duration": 600.0,
                "segments": [
                    {
                        "start": 100.0,
                        "end": 108.0,
                        "text": "Dispatch reports a vehicle fire at Main Street.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    extracted_windows: list[tuple[float, float]] = []

    def fake_extract(_source: Path, output: Path, start: float, end: float) -> Path:
        extracted_windows.append((start, end))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"exact evidence clip")
        return output

    monkeypatch.setattr("broadcastify_cli.worker.extract_audio_clip", fake_extract)
    with AnalysisStore(tmp_path / "analysis.sqlite3") as store:
        imported = store.import_transcript("90001", archive_date, transcript, audio)
        incident_id = store.replace_incidents(
            imported.day_id,
            [
                {
                    "fingerprint": "vehicle-fire",
                    "event_type": "fire",
                    "title": "Vehicle fire",
                    "summary": "A vehicle fire was reported at Main Street.",
                    "location": "Main Street",
                    "start_seconds": 100.0,
                    "end_seconds": 108.0,
                    "priority": 4,
                    "confidence": 0.9,
                    "evidence": [
                        {
                            "segment_index": 0,
                            "start_seconds": 100.0,
                            "end_seconds": 108.0,
                            "text": "Dispatch reports a vehicle fire at Main Street.",
                        }
                    ],
                    "attributes": {},
                }
            ],
            model="test-model",
            prompt_version=PROMPT_VERSION,
        )[0]

        result = _incident_clip(store, incident_id)
        context = _incident_clip(
            store,
            incident_id,
            include_surrounding_context=True,
        )

    assert result["incident_id"] == incident_id
    assert result["duration_seconds"] == 28.0
    assert result["clip_kind"] == "evidence"
    assert result["path"].endswith(f"I{incident_id}_92000-120000.mp3")
    assert len(result["sha256"]) == 64
    assert context["clip_kind"] == "context"
    assert context["duration_seconds"] == 180.0
    assert context["path"].endswith(f"I{incident_id}_context_0-180000.mp3")
    assert extracted_windows == [(92.0, 120.0), (0.0, 180.0)]
