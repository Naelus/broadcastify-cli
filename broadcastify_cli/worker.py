from __future__ import annotations

import argparse
import gc
import json
import math
import os
import struct
import sys
import tempfile
import time
import traceback
import warnings
import wave
from datetime import date
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .accelerators import (
    collect_accelerator_diagnostics,
    find_whisper_cpp,
    find_windows_ml_helper,
    module_available,
    whisper_cpp_backends,
    whisper_cpp_container_diagnostics,
)
from .audio import (
    extract_audio_clip,
    find_ffmpeg,
    select_incident_context_window,
    select_incident_evidence_window,
)
from .analysis import (
    DEFAULT_EMBEDDING_MODEL,
    PROMPT_VERSION,
    IncidentAnalyzer,
    RangeQuestionAnswerer,
    SemanticIndexer,
    WeeklySummaryAnalyzer,
    discover_day_paths,
    find_llama_server,
    format_archive_time,
)
from .analysis_providers import (
    AnalysisProviderConfig,
    diagnose_analysis_provider,
    open_analysis_client,
)
from .asr import (
    find_whisper_cpp_model,
    find_windows_ml_model,
    normalize_asr_engine,
    prepare_asr_model,
)
from .qwen_asr import find_qwen3_asr_model, qwen3_asr_diagnostics
from .portable_diarization import (
    COMMUNITY_DIARIZATION_ENGINE,
    COMMUNITY_DIARIZATION_QUALITY,
    PORTABLE_DIARIZATION_ENGINE,
    SherpaOnnxDiarizer,
    normalize_diarization_engine,
    portable_diarization_diagnostics,
    prepare_portable_diarization_model,
)
from .area_watch import AREA_PROMPT_VERSION, AreaStoryAnalyzer, _public_quote
from .area_acquisition import AreaAcquisitionRunner
from .broadcastify import BroadcastifyClient
from .quota import ArchiveRequestLedger
from .geography import CENSUS_ZCTA_YEAR, ZipCentroidCatalog
from .jobs import JobRunner
from .library import LocalProcessingRequest, prepare_local_day, scan_local_library
from .models import JobRequest
from .storage import AnalysisStore, sha256_file
from .transcription import LocalTranscriber, decoded_diarization_audio


DEFAULT_DATABASE = Path(
    os.getenv("BROADCASTIFY_ANALYSIS_DB") or "archives/broadcastify-analysis.sqlite3"
)
LOADED_ENVIRONMENT_FILE: Path | None = None


def emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False), flush=True)


def search_feeds(query: str) -> int:
    with BroadcastifyClient() as client:
        results = client.search_feeds(query)
    serialized = [result.to_dict() for result in results]
    with AnalysisStore(DEFAULT_DATABASE) as store:
        store.save_feed_catalog(serialized)
    emit({"type": "result", "results": serialized})
    return 0


def analysis_provider_diagnostics() -> int:
    payload = json.load(sys.stdin)
    result = diagnose_analysis_provider(AnalysisProviderConfig.from_mapping(payload))
    emit({"type": "analysis_provider_diagnostics", "result": result})
    return 0


def _analysis_self_test_result(settings: dict[str, Any]) -> dict[str, Any]:
    """Load the selected analysis model and return one structured proof."""

    config = AnalysisProviderConfig.from_mapping(settings)
    provider_name = {
        "local": "Local llama.cpp",
        "openai-responses": "OpenAI Responses",
        "openai-compatible": "OpenAI-compatible endpoint",
        "codex-cli": "Codex CLI",
    }.get(config.provider, config.provider)
    started = time.monotonic()
    emit(
        {
            "type": "progress",
            "phase": "analysis_self_test",
            "current": 0,
            "total": 0,
            "message": (
                f"Loading {provider_name} model {config.model or 'provider default'}; "
                "the test uses synthetic text only"
            ),
        }
    )
    schema = {
        "type": "object",
        "properties": {"ready": {"type": "boolean", "const": True}},
        "required": ["ready"],
        "additionalProperties": False,
    }
    with open_analysis_client(config) as client:
        response = client.chat_json(
            "You are a local runtime readiness probe. Follow the response schema exactly.",
            "Return ready=true. This is synthetic setup text and contains no archive evidence.",
            "runtime_readiness",
            schema,
            max_tokens=32,
        )
        effective_model = str(getattr(client, "model", "") or config.model)
    if response.get("ready") is not True:
        raise RuntimeError(
            "The analysis model responded, but did not pass the structured readiness check."
        )
    elapsed = time.monotonic() - started
    return {
        "provider": config.provider,
        "model": effective_model or "provider default",
        "device": config.device,
        "external": config.is_external,
        "ready": True,
        "verified": True,
        "elapsed_seconds": elapsed,
        "message": (
            f"{provider_name} generated valid structured output with "
            f"{effective_model or 'the selected model'} in {elapsed:.2f} seconds."
        ),
    }


def analysis_self_test(payload: dict[str, Any] | None = None) -> int:
    """Load the selected analysis model and prove one structured generation."""

    settings = payload if payload is not None else json.load(sys.stdin)
    result = _analysis_self_test_result(settings)
    emit({"type": "analysis_self_test", "result": result})
    return 0


def search_area(
    zip_codes: list[str],
    *,
    center_zip: str | None = None,
    radius_miles: float | None = None,
    max_zip_codes: int = 12,
) -> int:
    coverage: dict[str, Any]
    zip_distances: dict[str, float] | None = None
    if center_zip:
        if radius_miles is None:
            raise ValueError("Radius discovery requires a radius in miles.")
        emit(
            {
                "type": "log",
                "message": (
                    f"Finding up to {max_zip_codes} Census ZIP areas within "
                    f"{radius_miles:g} miles of {center_zip}..."
                ),
            }
        )
        nearby = ZipCentroidCatalog().nearest(
            center_zip, radius_miles, limit=max_zip_codes
        )
        zip_codes = [value.zip_code for value in nearby]
        zip_distances = {value.zip_code: value.distance_miles for value in nearby}
        coverage = {
            "mode": "radius",
            "center_zip": center_zip,
            "radius_miles": float(radius_miles),
            "max_zip_codes": int(max_zip_codes),
            "searched_zip_codes": [value.to_dict() for value in nearby],
            "distance_basis": (
                f"{CENSUS_ZCTA_YEAR} Census ZCTA internal-point centroids; "
                "feed distance is the nearest matched ZIP area, not a transmitter location."
            ),
        }
    else:
        coverage = {
            "mode": "zip-list",
            "center_zip": zip_codes[0] if zip_codes else "",
            "radius_miles": None,
            "max_zip_codes": len(zip_codes),
            "searched_zip_codes": [
                {"zip_code": value, "distance_miles": None} for value in zip_codes
            ],
            "distance_basis": "User-ordered ZIP priority; no mileage estimate.",
        }
    with BroadcastifyClient() as client:
        results = client.search_area_feeds(zip_codes, zip_distances=zip_distances)
    with AnalysisStore(DEFAULT_DATABASE) as store:
        store.save_feed_catalog(results)
    emit({"type": "area_search", "results": results, "coverage": coverage})
    return 0


def list_area_profiles() -> int:
    with AnalysisStore(DEFAULT_DATABASE) as store:
        profiles = store.list_area_profiles()
    emit({"type": "area_profiles", "profiles": profiles})
    return 0


def save_area_profile() -> int:
    payload = json.load(sys.stdin)
    with AnalysisStore(DEFAULT_DATABASE) as store:
        store.save_feed_catalog(list(payload.get("feeds") or []))
        profile = store.save_area_profile(
            str(payload.get("name") or ""),
            list(payload.get("zip_codes") or []),
            list(payload.get("feeds") or []),
            dict(payload.get("coverage") or {}),
        )
    emit(
        {
            "type": "area_profile_saved",
            "message": f"Saved area profile {profile['name']} with {len(profile['feed_ids'])} feeds.",
            "profile": profile,
        }
    )
    return 0


def run_job() -> int:
    payload = json.load(sys.stdin)
    request = JobRequest.from_dict(payload)
    if request.feed_name:
        with AnalysisStore(DEFAULT_DATABASE) as store:
            store.save_feed_catalog(
                [{"feed_id": request.feed_id, "name": request.feed_name}]
            )
    with BroadcastifyClient() as client:
        JobRunner(request, emit=emit, client=client).run()
    return 0


def run_area_acquisition() -> int:
    payload = json.load(sys.stdin)
    with AnalysisStore(DEFAULT_DATABASE) as store, BroadcastifyClient() as client:
        AreaAcquisitionRunner(payload, store, emit=emit, client=client).run()
    return 0


def list_area_acquisition_runs(profile_name: str | None = None) -> int:
    with AnalysisStore(DEFAULT_DATABASE) as store:
        runs = store.list_area_acquisition_runs(profile_name=profile_name)
    emit({"type": "area_runs", "runs": runs})
    return 0


def authenticate() -> int:
    payload = json.load(sys.stdin)
    username = str(payload.get("username") or "").strip()
    password = str(payload.get("password") or "")
    if not username or not password:
        raise ValueError("Broadcastify username and password are required.")
    with BroadcastifyClient(username=username, password=password) as client:
        client.authenticate(force=True)
    emit({"type": "authenticated", "message": "Broadcastify sign-in succeeded."})
    return 0


def archive_quota_status() -> int:
    emit({"type": "archive_quota_status", "status": ArchiveRequestLedger().status()})
    return 0


def diagnostics(settings: dict[str, Any] | None = None) -> int:
    llama_server = find_llama_server()
    selected_whisper_model: Path | None = None
    selected_windows_model: Path | None = None
    selected_qwen_model: Path | None = None
    selected_asr_engine: str | None = None
    selected_diarization_engine: str | None = None
    selected_asr_model: dict[str, Any] = {}
    if settings is not None:
        selected_diarization_engine = normalize_diarization_engine(
            str(
                settings.get("diarization_engine")
                or COMMUNITY_DIARIZATION_ENGINE
            )
        )
        model = str(settings.get("model") or "turbo")
        engine = normalize_asr_engine(
            str(settings.get("asr_engine") or "auto"),
            str(settings.get("device") or "auto"),
        )
        selected_asr_engine = engine
        selected_asr_model = {
            "engine": engine,
            "model": model,
            "path": "",
            "configured": False,
            "error": "",
        }
        try:
            if engine == "whisper.cpp":
                selected_whisper_model = find_whisper_cpp_model(
                    model, settings.get("asr_model_path") or None
                )
                selected_asr_model["path"] = (
                    str(selected_whisper_model) if selected_whisper_model else ""
                )
                selected_asr_model["configured"] = selected_whisper_model is not None
            elif engine == "windows-ml":
                selected_windows_model = find_windows_ml_model(
                    model, settings.get("asr_model_path") or None
                )
                selected_asr_model["path"] = (
                    str(selected_windows_model) if selected_windows_model else ""
                )
                selected_asr_model["configured"] = selected_windows_model is not None
            elif engine == "qwen3-asr":
                qwen_info = find_qwen3_asr_model(
                    model, settings.get("asr_model_path") or None
                )
                selected_qwen_model = qwen_info.path if qwen_info else None
                selected_asr_model["path"] = (
                    str(selected_qwen_model) if selected_qwen_model else ""
                )
                selected_asr_model["configured"] = bool(
                    qwen_info and qwen_info.vad_path
                )
        except (OSError, RuntimeError, ValueError) as exc:
            selected_asr_model["error"] = str(exc)
    payload: dict[str, Any] = {
        "type": "diagnostics",
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "ffmpeg": find_ffmpeg(),
        "cuda_available": False,
        "cuda_devices": [],
        "llama_server": llama_server,
        "huggingface_token_configured": bool(
            os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
        ),
        "broadcastify_credentials_configured": bool(
            (
                os.getenv("BROADCASTIFY_USERNAME")
                and os.getenv("BROADCASTIFY_PASSWORD")
            )
            or (os.getenv("USERNAME") and os.getenv("PASSWORD"))
        ),
        "saved_session_available": Path("cookies.json").is_file(),
        "environment_file": str(LOADED_ENVIRONMENT_FILE.resolve())
        if LOADED_ENVIRONMENT_FILE
        else "",
        "analysis_database": str(DEFAULT_DATABASE.resolve()),
        "analysis_stats": {},
        "archive_quota": ArchiveRequestLedger().status(),
    }
    try:
        import torch

        payload["cuda_available"] = bool(torch.cuda.is_available())
        payload["cuda_devices"] = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
    except ModuleNotFoundError:
        payload["transcription_dependencies"] = "not installed"
    payload["accelerators"] = collect_accelerator_diagnostics(
        llama_server,
        selected_asr_engine=selected_asr_engine,
        selected_whisper_model=selected_whisper_model,
        selected_windows_model=selected_windows_model,
        selected_qwen_model=selected_qwen_model,
        selected_diarization_engine=selected_diarization_engine,
    )
    if selected_asr_model:
        payload["selected_asr_model"] = selected_asr_model
    if DEFAULT_DATABASE.exists():
        with AnalysisStore(DEFAULT_DATABASE) as store:
            payload["analysis_stats"] = store.stats()
    emit(payload)
    return 0


def _asr_self_test_result(settings: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()

    def progress(message: str) -> None:
        emit(
            {
                "type": "progress",
                "phase": "asr_self_test",
                "current": 0,
                "total": 0,
                "message": str(message),
            }
        )

    model = str(settings.get("model") or "turbo")
    asr_engine = str(settings.get("asr_engine") or "auto")
    device = str(settings.get("device") or "auto")
    effective_engine = normalize_asr_engine(asr_engine, device)
    if effective_engine in {"whisper.cpp", "windows-ml", "qwen3-asr"}:
        model_note = "the explicitly prepared managed model must already be present"
    else:
        model_note = "a runtime-managed model may download now"
    progress(f"Loading {model} with {asr_engine} on {device}; {model_note}")
    transcriber = LocalTranscriber(
        model_name=model,
        asr_engine=asr_engine,
        device=device,
        device_index=max(0, int(settings.get("device_index", 0))),
        compute_type=str(settings.get("compute_type") or "auto"),
        asr_model_path=settings.get("asr_model_path") or None,
        diarization_device="cpu",
        diarize=False,
        huggingface_token=str(settings.get("huggingface_token") or "") or None,
        batch_size=max(1, int(settings.get("batch_size", 8))),
    )
    with tempfile.TemporaryDirectory(prefix="radio-archive-asr-test-") as temporary:
        audio_path = Path(temporary) / "silence.wav"
        with wave.open(str(audio_path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16_000)
            output.writeframes(bytes(16_000 * 2))
        transcript_path = transcriber.transcribe_file(audio_path, progress=progress)
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    metadata = dict(transcript.get("asr_metadata") or {})
    actual_model = str(
        metadata.get("model")
        or transcript.get("model")
        or model
    )
    result = {
        "ready": True,
        "engine": str(transcript.get("asr_engine") or transcriber.asr_engine),
        "backend": str(transcript.get("asr_backend") or transcriber.backend_description),
        "model": actual_model,
        "requested_model": model,
        "model_path": str(metadata.get("model_path") or ""),
        "provider": str(metadata.get("provider") or ""),
        "precision": str(metadata.get("precision") or ""),
        "device": str(transcript.get("device") or transcriber.device),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "segment_count": len(transcript.get("segments") or []),
        "word_count": len(transcript.get("words") or []),
        "fallback_reason": str(metadata.get("fallback_reason") or ""),
        "fallback_stage": str(metadata.get("fallback_stage") or ""),
    }
    result["message"] = (
        f"Local execution check passed with {actual_model} on "
        f"{result['backend']} in "
        f"{result['elapsed_seconds']:.1f} seconds. This generated-silence check "
        "does not measure radio accuracy."
    )
    return result


def asr_self_test(payload: dict[str, Any] | None = None) -> int:
    settings = payload if payload is not None else json.load(sys.stdin)
    result = _asr_self_test_result(settings)
    emit({"type": "asr_self_test", "result": result, "message": result["message"]})
    return 0


def prepare_asr_model_command(payload: dict[str, Any] | None = None) -> int:
    settings = payload if payload is not None else json.load(sys.stdin)

    def progress(message: str) -> None:
        emit(
            {
                "type": "progress",
                "phase": "asr_model_preparation",
                "current": 0,
                "total": 0,
                "message": str(message),
            }
        )

    result = prepare_asr_model(settings, progress=progress)
    emit(
        {
            "type": "asr_model_prepared",
            "result": result,
            "message": result["message"],
        }
    )
    return 0


def _load_diarization_pipeline(
    *,
    token: str,
    device: str,
    device_index: int,
) -> tuple[Any, str]:
    try:
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module=r"pyannote\.audio\.core\.io",
        )
        import torch
        from pyannote.audio import Pipeline
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pyannote.audio and PyTorch are required for speaker labeling."
        ) from exc

    selected_device = str(device or "auto").strip().lower()
    cuda_available = bool(torch.cuda.is_available())
    if selected_device == "auto":
        selected_device = "cuda" if cuda_available else "cpu"
    if selected_device not in {"cuda", "cpu"}:
        raise ValueError("Speaker-label testing supports CUDA or CPU.")
    if selected_device == "cuda" and not cuda_available:
        raise RuntimeError(
            "CUDA speaker labeling was selected, but PyTorch cannot see a CUDA device."
        )

    try:
        pipeline = Pipeline.from_pretrained(
            LocalTranscriber.DIARIZATION_MODEL, token=token or None
        )
    except Exception as exc:
        if not token:
            raise RuntimeError(
                "No Hugging Face read token was supplied and no usable cached "
                "speaker-label model could be loaded. Add a read token for the "
                "first download, then the cached model can run offline."
            ) from exc
        raise
    if pipeline is None:
        message = (
            "No Hugging Face read token was supplied and no usable cached speaker-label "
            "model was found. Add a read token for the first download."
            if not token
            else "The speaker-label model could not be loaded. Confirm that its Hugging Face terms were accepted."
        )
        raise RuntimeError(message)
    target = (
        torch.device(f"cuda:{max(0, int(device_index))}")
        if selected_device == "cuda"
        else torch.device("cpu")
    )
    pipeline.to(target)
    return pipeline, selected_device


def _write_diarization_test_audio(path: Path) -> None:
    sample_rate = 16_000
    frames = bytearray()
    # Alternating low-amplitude tones and gaps exercise the complete local
    # pipeline without using archive audio or pretending to be real speech.
    for index in range(sample_rate * 4):
        second = index / sample_rate
        active = int(second * 2) % 2 == 0
        frequency = 260.0 if second < 2.0 else 520.0
        sample = int(2400 * math.sin(2.0 * math.pi * frequency * second)) if active else 0
        frames.extend(struct.pack("<h", sample))
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(frames)


def _speaker_turn_count(output: Any) -> int:
    annotation = getattr(output, "exclusive_speaker_diarization", None)
    if annotation is None:
        annotation = getattr(output, "speaker_diarization", None)
    if annotation is None:
        annotation = output
    if hasattr(annotation, "itertracks"):
        return sum(1 for _value in annotation.itertracks(yield_label=True))
    try:
        return sum(1 for _value in annotation)
    except TypeError:
        return 0


def _diarization_self_test_result(settings: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    engine = normalize_diarization_engine(
        str(
            settings.get("diarization_engine")
            or COMMUNITY_DIARIZATION_ENGINE
        )
    )
    token = str(settings.get("huggingface_token") or "").strip()
    token = token or os.getenv("HUGGINGFACE_TOKEN", "") or os.getenv("HF_TOKEN", "")
    requested_device = str(settings.get("diarization_device") or "auto")
    def progress(message: str) -> None:
        emit(
            {
                "type": "progress",
                "phase": "diarization_self_test",
                "current": 0,
                "total": 0,
                "message": str(message),
            }
        )

    if engine == PORTABLE_DIARIZATION_ENGINE:
        if requested_device.lower() not in {"auto", "cpu"}:
            raise RuntimeError(
                "Fast portable speaker preview currently runs on CPU. "
                "Select CPU or Automatic."
            )
        progress(
            "Preparing the checksum-verified public sherpa-onnx preview models; "
            "a complete managed copy is reused offline"
        )
        prepared = prepare_portable_diarization_model(progress=progress)
        diarizer = SherpaOnnxDiarizer(
            min_speakers=(
                int(settings["min_speakers"])
                if settings.get("min_speakers") not in {None, ""}
                else None
            ),
            max_speakers=(
                int(settings["max_speakers"])
                if settings.get("max_speakers") not in {None, ""}
                else None
            ),
        )
        with tempfile.TemporaryDirectory(
            prefix="radio-archive-speaker-test-"
        ) as temporary:
            audio_path = Path(temporary) / "generated-test.wav"
            _write_diarization_test_audio(audio_path)
            turns = diarizer.process(audio_path, progress=progress)
        result = {
            "ready": True,
            "engine": engine,
            "model": str(prepared["model"]),
            "device": "cpu",
            "provider": "cpu",
            "quality": "preview",
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "turn_count": len(turns),
            "metadata": dict(diarizer.metadata),
        }
        result["message"] = (
            "Fast portable speaker-preview self-test passed on CPU in "
            f"{result['elapsed_seconds']:.1f} seconds. Synthetic-audio turn "
            f"count: {len(turns)}. Community-1 remains the accuracy default."
        )
        return result

    progress(
        "Loading Community-1; a complete cache is reused offline, and the "
        "first download requires a Hugging Face read token"
    )
    pipeline, selected_device = _load_diarization_pipeline(
        token=token,
        device=requested_device,
        device_index=max(0, int(settings.get("device_index", 0))),
    )
    if hasattr(pipeline, "embedding_batch_size"):
        pipeline.embedding_batch_size = max(
            int(getattr(pipeline, "embedding_batch_size", 1)),
            max(1, int(settings.get("batch_size", 8))),
        )
    with tempfile.TemporaryDirectory(prefix="radio-archive-speaker-test-") as temporary:
        audio_path = Path(temporary) / "generated-test.wav"
        _write_diarization_test_audio(audio_path)
        with decoded_diarization_audio(audio_path) as diarization_audio:
            output = pipeline(diarization_audio)
            turn_count = _speaker_turn_count(output)
    result = {
        "ready": True,
        "engine": engine,
        "model": LocalTranscriber.DIARIZATION_MODEL,
        "device": selected_device,
        "provider": selected_device,
        "quality": COMMUNITY_DIARIZATION_QUALITY,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "turn_count": turn_count,
    }
    result["message"] = (
        f"Speaker-label self-test passed on {selected_device} in "
        f"{result['elapsed_seconds']:.1f} seconds. Synthetic-audio turn count: {turn_count}."
    )
    return result


def diarization_self_test(payload: dict[str, Any] | None = None) -> int:
    settings = payload if payload is not None else json.load(sys.stdin)
    result = _diarization_self_test_result(settings)
    emit(
        {
            "type": "diarization_self_test",
            "result": result,
            "message": result["message"],
        }
    )
    return 0


def _release_profile_stage_memory() -> None:
    """Release model objects and accelerator caches between profile stages."""

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # Cleanup is deliberately best-effort. Some optional torch/runtime builds
    # can raise loader-specific exceptions while probing CUDA; that must not
    # replace the actual stage result with a cleanup failure.
    except Exception:
        pass


def _profile_failure_summary(exc: Exception, stage: str) -> str:
    """Keep native runtime dumps from overwhelming the setup UI."""

    raw = str(exc).strip()
    if not raw:
        return f"The {stage} self-test failed."
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(raw) <= 600 and len(lines) <= 4:
        return raw
    markers = (
        "error",
        "failed",
        "not found",
        "cannot",
        "could not",
        "requires",
        "unavailable",
        "unsupported",
        "invalid",
        "assert",
    )
    signal = next(
        (
            line
            for line in lines
            if any(marker in line.lower() for marker in markers)
        ),
        lines[0],
    )
    if "GGML_ASSERT" in signal:
        signal = signal[signal.index("GGML_ASSERT") :]
    if len(signal) > 500:
        signal = signal[:497].rstrip() + "..."
    return (
        f"{signal} Run the individual {stage} test for detailed diagnostics."
    )


def _profile_recovery_action(
    stage: str,
    settings: dict[str, Any],
    message: str,
) -> dict[str, str]:
    """Return the first UI action that can materially advance a failed proof."""

    if stage == "diarization":
        engine = normalize_diarization_engine(
            str(
                settings.get("diarization_engine")
                or COMMUNITY_DIARIZATION_ENGINE
            )
        )
        if engine == PORTABLE_DIARIZATION_ENGINE:
            diagnostics = portable_diarization_diagnostics()
            runtime_note = (
                "The runtime is installed; Test speakers will acquire or reuse "
                "the pinned public models."
                if diagnostics["runtime_installed"]
                else 'Install `python -m pip install -e ".[portable-diarization]"` first.'
            )
            return {
                "stage": stage,
                "kind": "configure-speakers",
                "label": "Review portable speakers",
                "message": (
                    f"{message} {runtime_note} This path is a fast CPU preview; "
                    "Community-1 remains available as the later accuracy upgrade."
                ),
            }
        return {
            "stage": stage,
            "kind": "configure-speakers",
            "label": "Review speaker setup",
            "message": (
                f"{message} Check the pyannote installation, Community-1 access or "
                "complete cache, and the selected CUDA/CPU device."
            ),
        }
    if stage == "analysis":
        return {
            "stage": stage,
            "kind": "configure-analysis",
            "label": "Review analysis setup",
            "message": (
                f"{message} Open Analysis & AI to check the selected provider, model, "
                "endpoint, credential, and device."
            ),
        }

    engine = normalize_asr_engine(
        str(settings.get("asr_engine") or "auto"),
        str(settings.get("device") or "auto"),
    )
    device = str(settings.get("device") or "auto").strip().lower()
    if engine == "whisper.cpp":
        executable = find_whisper_cpp()
        container = whisper_cpp_container_diagnostics()
        native_backends = set(whisper_cpp_backends(executable))
        requested_backend = "metal" if device == "metal" else "vulkan" if device == "vulkan" else "cpu"
        runtime_ready = bool(
            (
                container.get("configured")
                and container.get("ready")
                and container.get("backend") == requested_backend
            )
            or (executable and requested_backend in native_backends)
        )
        if not runtime_ready:
            backend_label = "Metal" if requested_backend == "metal" else "Vulkan" if requested_backend == "vulkan" else "native"
            return {
                "stage": stage,
                "kind": "configure-transcription",
                "label": f"Show {backend_label} setup",
                "message": (
                    f"{message} Configure a whisper.cpp runtime built for "
                    f"{backend_label}, set WHISPER_CPP_PATH, then refresh the check."
                ),
            }
        try:
            model = find_whisper_cpp_model(
                str(settings.get("model") or "turbo"),
                settings.get("asr_model_path") or None,
            )
        except (OSError, RuntimeError, ValueError):
            model = None
        if model is None:
            return {
                "stage": stage,
                "kind": "prepare-asr-model",
                "label": "Download selected model",
                "message": (
                    f"{message} The runtime is present; download and verify the "
                    "matching GGML model before retrying."
                ),
            }
    elif engine == "windows-ml":
        if not find_windows_ml_helper():
            return {
                "stage": stage,
                "kind": "configure-transcription",
                "label": "Show Windows ML setup",
                "message": (
                    f"{message} Use the verified Windows publish or configure "
                    "WINDOWS_ML_HELPER_PATH, then refresh the check."
                ),
            }
        try:
            model = find_windows_ml_model(
                str(settings.get("model") or "base"),
                settings.get("asr_model_path") or None,
            )
        except (OSError, RuntimeError, ValueError):
            model = None
        if model is None:
            return {
                "stage": stage,
                "kind": "prepare-asr-model",
                "label": "Build selected model",
                "message": (
                    f"{message} The helper is present; build and validate the "
                    "matching managed Whisper graph before retrying."
                ),
            }
    elif engine == "qwen3-asr":
        qwen = qwen3_asr_diagnostics(settings.get("asr_model_path") or None)
        if not qwen["runtime_installed"]:
            return {
                "stage": stage,
                "kind": "configure-transcription",
                "label": "Show Qwen setup",
                "message": (
                    f'{message} Install the optional runtime with `python -m pip '
                    'install -e ".[qwen]"`, then refresh the check.'
                ),
            }
        if not qwen["ready"]:
            return {
                "stage": stage,
                "kind": "prepare-asr-model",
                "label": "Download Qwen model",
                "message": (
                    f"{message} Download and checksum-verify the pinned Qwen graph "
                    "and Silero VAD before retrying."
                ),
            }
    elif engine == "openvino":
        if not module_available("openvino_genai"):
            return {
                "stage": stage,
                "kind": "configure-transcription",
                "label": "Show OpenVINO setup",
                "message": (
                    f'{message} Install the optional runtime with `python -m pip '
                    'install -e ".[openvino]"`, then refresh the check.'
                ),
            }
    elif not module_available("faster_whisper") or not module_available("torch"):
        return {
            "stage": stage,
            "kind": "configure-transcription",
            "label": "Show transcription setup",
            "message": (
                f'{message} Install the optional runtime with `python -m pip '
                'install -e ".[transcription]"`, then refresh the check.'
            ),
        }

    return {
        "stage": stage,
        "kind": "test-transcription",
        "label": "Run engine details",
        "message": (
            f"{message} The required runtime and selected model appear present; "
            "run Test engine to retain the focused backend error."
        ),
    }


def profile_self_test(payload: dict[str, Any] | None = None) -> int:
    """Execute ASR, diarization, and analysis proofs as one guided action."""

    settings = payload if payload is not None else json.load(sys.stdin)
    started = time.monotonic()
    results: dict[str, dict[str, Any]] = {}
    stages = [
        ("transcription", "asr_self_test", _asr_self_test_result),
        ("diarization", "diarization_self_test", _diarization_self_test_result),
        ("analysis", "analysis_self_test", _analysis_self_test_result),
    ]
    for stage, event_type, runner in stages:
        emit(
            {
                "type": "profile_self_test_stage",
                "stage": stage,
                "status": "running",
                "message": f"Verifying {stage} with generated local input.",
            }
        )
        try:
            stage_settings = settings
            if stage == "analysis":
                # ``model`` is the ASR model in this joined contract. The analysis
                # provider retains a legacy ``model`` fallback, so never let a blank
                # analysis-model field turn Whisper "turbo" into a llama.cpp repo.
                stage_settings = dict(settings)
                stage_settings.pop("model", None)
            result = runner(stage_settings)
        except Exception as exc:
            message = _profile_failure_summary(exc, stage)
            recovery = _profile_recovery_action(stage, settings, message)
            _release_profile_stage_memory()
            profile_result = {
                "ready": False,
                "verified": False,
                "failed_stage": stage,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "results": results,
                "recovery": recovery,
                "message": f"Profile verification stopped at {stage}: {message}",
            }
            emit(
                {
                    "type": "profile_self_test_stage",
                    "stage": stage,
                    "status": "failed",
                    "message": message,
                }
            )
            emit(
                {
                    "type": "profile_self_test",
                    "result": profile_result,
                    "message": profile_result["message"],
                }
            )
            return 0
        results[stage] = result
        emit(
            {
                "type": event_type,
                "result": result,
                "message": result.get("message", ""),
            }
        )
        emit(
            {
                "type": "profile_self_test_stage",
                "stage": stage,
                "status": "passed",
                "message": (
                    f"{stage.title()} verified; "
                    + (
                        "releasing its model before the next stage."
                        if stage != "analysis"
                        else "all requested stages have executed."
                    )
                ),
            }
        )
        # Do not retain one model stage while loading the next one. This keeps
        # the guided check viable on machines where all three models fit only
        # one at a time.
        _release_profile_stage_memory()

    elapsed = round(time.monotonic() - started, 3)
    profile_result = {
        "ready": True,
        "verified": True,
        "failed_stage": "",
        "elapsed_seconds": elapsed,
        "results": results,
        "message": (
            "Transcription, speaker labels, and analysis all executed "
            f"successfully in {elapsed:.1f} seconds."
        ),
    }
    emit(
        {
            "type": "profile_self_test",
            "result": profile_result,
            "message": profile_result["message"],
        }
    )
    return 0


def analysis_days(feed_id: str | None) -> int:
    with AnalysisStore(DEFAULT_DATABASE) as store:
        days = store.list_days(feed_id)
    for day in days:
        analysis_current = (
            str(day.get("summary_prompt_version") or "") == PROMPT_VERSION
        )
        day["analysis_current"] = analysis_current
        day["analysis_update_required"] = bool(day.get("has_summary")) and not analysis_current
    emit({"type": "analysis_days", "days": days})
    return 0


def library_days(output_dir: str) -> int:
    days = scan_local_library(Path(output_dir), DEFAULT_DATABASE)
    emit(
        {
            "type": "library_days",
            "days": days,
            "summary": {
                "feed_count": len({value["feed_id"] for value in days}),
                "day_count": len(days),
                "complete_count": sum(bool(value["is_complete"]) for value in days),
                "attention_count": sum(not bool(value["is_complete"]) for value in days),
                "storage_bytes": sum(int(value["storage_bytes"]) for value in days),
            },
        }
    )
    return 0


def _day_report(store: AnalysisStore, feed_id: str, archive_date: date) -> dict[str, Any]:
    day = store.get_day(feed_id, archive_date)
    if day is None:
        raise ValueError(f"No imported transcript for feed {feed_id} on {archive_date}.")
    summary = store.get_latest_daily_summary(int(day["id"]))
    analysis_prompt_version = str(summary["prompt_version"]) if summary else ""
    analysis_current = analysis_prompt_version == PROMPT_VERSION
    incidents = []
    for stored in store.get_incidents(
        feed_id,
        archive_date,
        archive_date,
        prompt_version=PROMPT_VERSION,
    ):
        evidence_start, evidence_end = select_incident_evidence_window(stored)
        quote_parts = []
        for raw in stored.get("evidence", []):
            if not isinstance(raw, dict):
                continue
            try:
                segment_start = float(raw.get("start_seconds", 0.0))
                segment_end = float(raw.get("end_seconds", segment_start))
            except (TypeError, ValueError):
                continue
            text = str(raw.get("text") or "").strip()
            if (
                text
                and segment_end >= evidence_start
                and segment_start <= evidence_end
                and text not in quote_parts
            ):
                quote_parts.append(text)
        evidence_quote, _ = _public_quote(
            " ".join(quote_parts),
            location=str(stored.get("location") or ""),
        )
        # Keep the routine list/report response compact. Raw transcript evidence
        # stays in the local database; only the redacted exact-clip quote crosses
        # the viewer boundary.
        incidents.append(
            {
                "id": stored["id"],
                "event_type": stored["event_type"],
                "title": stored["title"],
                "summary": stored["summary"],
                "location": stored["location"],
                "priority": stored["priority"],
                "confidence": stored["confidence"],
                "start_seconds": stored["start_seconds"],
                "end_seconds": stored["end_seconds"],
                "evidence_quote": evidence_quote,
                "archive_time": format_archive_time(
                    stored, float(stored["start_seconds"])
                ),
            }
        )
    return {
        "feed_id": feed_id,
        "archive_date": archive_date.isoformat(),
        "summary": str(summary["summary"]) if summary and analysis_current else "",
        "incidents": incidents,
        "audio_path": str(day["audio_path"] or ""),
        "has_diarization": bool(day["has_diarization"]),
        "analysis_current": analysis_current,
        "analysis_update_required": bool(summary) and not analysis_current,
        "analysis_prompt_version": analysis_prompt_version,
        "expected_analysis_prompt_version": PROMPT_VERSION,
    }


def report_day(feed_id: str, date_value: str) -> int:
    archive_date = date.fromisoformat(date_value)
    with AnalysisStore(DEFAULT_DATABASE) as store:
        report = _day_report(store, feed_id, archive_date)
    emit({"type": "day_report", "report": report})
    return 0


def _incident_clip(
    store: AnalysisStore,
    incident_id: int,
    *,
    include_surrounding_context: bool = False,
) -> dict[str, Any]:
    incident = store.get_incident(incident_id)
    if incident is None:
        raise ValueError(f"Incident I{incident_id} was not found in the local analysis database.")
    source_value = str(incident.get("audio_path") or "")
    source = Path(source_value)
    if not source.is_file():
        raise ValueError(
            f"The retained combined audio for incident I{incident_id} could not be found."
        )

    start_seconds, end_seconds = (
        select_incident_context_window(incident)
        if include_surrounding_context
        else select_incident_evidence_window(incident)
    )
    start_milliseconds = round(start_seconds * 1_000)
    end_milliseconds = round(end_seconds * 1_000)
    output = (
        source.parent
        / "evidence-clips"
        / (
            f"{incident['feed_id']}_{incident['archive_date']}_I{incident_id}_"
            f"{'context_' if include_surrounding_context else ''}"
            f"{start_milliseconds}-{end_milliseconds}.mp3"
        )
    )
    clip = extract_audio_clip(source, output, start_seconds, end_seconds).resolve()
    return {
        "incident_id": int(incident_id),
        "feed_id": str(incident["feed_id"]),
        "archive_date": str(incident["archive_date"]),
        "archive_time": format_archive_time(incident, start_seconds),
        "start_seconds": start_seconds,
        "end_seconds": end_seconds,
        "duration_seconds": end_seconds - start_seconds,
        "clip_kind": "context" if include_surrounding_context else "evidence",
        "path": str(clip),
        "sha256": sha256_file(clip),
    }


def incident_clip(incident_id: int, *, include_surrounding_context: bool = False) -> int:
    with AnalysisStore(DEFAULT_DATABASE) as store:
        result = _incident_clip(
            store,
            incident_id,
            include_surrounding_context=include_surrounding_context,
        )
    emit(
        {
            "type": "incident_clip",
            "message": (
                f"Surrounding local radio context ready for incident I{incident_id}."
                if include_surrounding_context
                else f"Exact local evidence clip ready for incident I{incident_id}."
            ),
            "clip": result,
        }
    )
    return 0


def _analyze_day_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], int, int]:
    feed_id = str(payload["feed_id"])
    archive_date = date.fromisoformat(str(payload["archive_date"]))
    output_dir = Path(payload.get("output_dir") or "archives")
    provider = AnalysisProviderConfig.from_mapping(payload)
    force = bool(payload.get("force", False))
    force_summary = bool(payload.get("force_summary", False))
    audio, transcript, manifest = discover_day_paths(output_dir, feed_id, archive_date)
    with AnalysisStore(DEFAULT_DATABASE) as store:
        imported = store.import_transcript(
            feed_id, archive_date, transcript, audio, manifest
        )
        emit(
            {
                "type": "stage",
                "stage": "analysis_import",
                "message": f"Imported {imported.segment_count} transcript segments.",
            }
        )
        day = store.get_day(feed_id, archive_date)
        complete = bool(
            day is not None
            and store.get_daily_summary(
                int(day["id"]),
                provider.cache_model,
                PROMPT_VERSION,
                str(day["transcript_sha256"]),
            )
        )
        emit(
            {
                "type": "stage",
                "stage": "analysis_provider",
                "message": (
                    f"Analysis provider: {provider.provider} / {provider.cache_model}"
                    + (" (external transcript excerpts allowed)." if provider.is_external else ".")
                ),
            }
        )
        with open_analysis_client(
            provider,
            launch_local_server=not (complete and not force and not force_summary),
        ) as client:
            result = IncidentAnalyzer(
                store,
                client,
                progress=lambda message: emit(
                    {"type": "stage", "stage": "analysis", "message": message}
                ),
            ).analyze_day(
                feed_id,
                archive_date,
                force=force,
                force_summary=force_summary,
            )
        emit(
            {
                "type": "stage",
                "stage": "embeddings",
                "message": "Updating semantic search index…",
            }
        )
        indexed = SemanticIndexer(store, model=DEFAULT_EMBEDDING_MODEL).index_missing()
        report = _day_report(store, feed_id, archive_date)
    return report, int(result["incidents"]), indexed


def analyze_day() -> int:
    payload = json.load(sys.stdin)
    report, incident_count, indexed = _analyze_day_payload(payload)
    emit(
        {
            "type": "analysis_complete",
            "message": f"Analysis complete: {incident_count} incidents, {indexed} new passages indexed.",
            "report": report,
        }
    )
    return 0


def continue_local_day() -> int:
    payload = json.load(sys.stdin)
    request = LocalProcessingRequest.from_dict(payload)
    emit(
        {
            "type": "stage",
            "stage": "local_prepare",
            "message": (
                f"Continuing feed {request.feed_id} for {request.archive_date} "
                "from retained local files…"
            ),
        }
    )
    prepared = prepare_local_day(
        request,
        progress=lambda message: emit(
            {"type": "stage", "stage": "local_prepare", "message": message}
        ),
    )
    report = None
    if bool(payload.get("analyze", True)):
        provider_fields = {
            key: payload[key]
            for key in (
                "analysis_provider",
                "analysis_model",
                "analysis_device",
                "analysis_endpoint",
                "analysis_api_key",
                "analysis_api_key_env",
                "codex_cli_path",
                "allow_external_analysis",
                "analysis_timeout",
            )
            if key in payload
        }
        report, incident_count, indexed = _analyze_day_payload(
            {
                "feed_id": request.feed_id,
                "archive_date": request.archive_date.isoformat(),
                "output_dir": str(request.output_dir),
                **provider_fields,
            }
        )
        emit(
            {
                "type": "stage",
                "stage": "analysis_complete",
                "message": (
                    f"Analysis complete: {incident_count} incidents, "
                    f"{indexed} new passages indexed."
                ),
            }
        )
    state = next(
        (
            value
            for value in scan_local_library(request.output_dir, DEFAULT_DATABASE)
            if value["feed_id"] == request.feed_id
            and value["archive_date"] == request.archive_date.isoformat()
        ),
        None,
    )
    emit(
        {
            "type": "local_complete",
            "message": f"Local processing finished for {request.archive_date}.",
            "prepared": prepared,
            "state": state,
            "report": report,
        }
    )
    return 0


def ask_archive() -> int:
    payload = json.load(sys.stdin)
    feed_id = str(payload["feed_id"])
    start_date = date.fromisoformat(str(payload["start_date"]))
    end_date = date.fromisoformat(str(payload["end_date"]))
    question = str(payload["question"]).strip()
    if not question:
        raise ValueError("A question is required.")
    provider = AnalysisProviderConfig.from_mapping(payload)
    emit(
        {
            "type": "stage",
            "stage": "retrieval",
            "message": "Retrieving transcript evidence…",
        }
    )
    with AnalysisStore(DEFAULT_DATABASE) as store:
        indexer = SemanticIndexer(store, model=DEFAULT_EMBEDDING_MODEL)
        indexer.index_missing()
        with open_analysis_client(provider) as client:
            result = RangeQuestionAnswerer(
                store,
                client,
                indexer=indexer,
            ).ask(feed_id, start_date, end_date, question)
    emit({"type": "answer", "message": "Question answered.", "result": result})
    return 0


def summarize_week() -> int:
    payload = json.load(sys.stdin)
    feed_id = str(payload["feed_id"])
    week_ending = date.fromisoformat(str(payload["week_ending"]))
    provider = AnalysisProviderConfig.from_mapping(payload)
    force = bool(payload.get("force", False))
    emit(
        {
            "type": "stage",
            "stage": "weekly_summary",
            "message": f"Building seven-day brief ending {week_ending}…",
        }
    )
    with AnalysisStore(DEFAULT_DATABASE) as store:
        with open_analysis_client(provider) as client:
            result = WeeklySummaryAnalyzer(
                store,
                client,
                progress=lambda message: emit(
                    {
                        "type": "stage",
                        "stage": "weekly_summary",
                        "message": message,
                    }
                ),
            ).summarize(feed_id, week_ending, force=force)
    emit(
        {
            "type": "weekly_summary",
            "message": (
                f"Weekly brief ready: {result['days_available']}/7 days, "
                f"{result['incident_count']} incidents."
            ),
            "result": result,
        }
    )
    return 0


def summarize_area() -> int:
    payload = json.load(sys.stdin)
    profile_name = str(payload["profile_name"])
    start_date = date.fromisoformat(str(payload["start_date"]))
    end_date = date.fromisoformat(str(payload["end_date"]))
    provider = AnalysisProviderConfig.from_mapping(payload)
    force = bool(payload.get("force", False))
    emit(
        {
            "type": "stage",
            "stage": "area_digest",
            "message": f"Building area story brief for {start_date} through {end_date}…",
        }
    )
    with AnalysisStore(DEFAULT_DATABASE) as store:
        with open_analysis_client(provider) as client:
            result = AreaStoryAnalyzer(
                store,
                client,
                progress=lambda message: emit(
                    {"type": "stage", "stage": "area_digest", "message": message}
                ),
            ).summarize(profile_name, start_date, end_date, force=force)
    emit(
        {
            "type": "area_digest",
            "message": f"Area brief ready: {len(result['stories'])} ranked story leads.",
            "result": result,
        }
    )
    return 0


def latest_area_digest(profile_name: str) -> int:
    with AnalysisStore(DEFAULT_DATABASE) as store:
        latest_any = store.get_latest_area_story_digest(profile_name)
        row = store.get_latest_area_story_digest(
            profile_name,
            prompt_version=AREA_PROMPT_VERSION,
        )
    result = None
    stale = latest_any is not None and row is None
    if row is not None:
        coverage = json.loads(str(row["coverage_json"]))
        if str(coverage.get("incident_prompt_version") or "") != PROMPT_VERSION:
            stale = True
        else:
            result = {
                "profile_name": str(row["profile_name"]),
                "start_date": str(row["start_date"]),
                "end_date": str(row["end_date"]),
                "summary": str(row["summary"]),
                "stories": json.loads(str(row["stories_json"])),
                "coverage": coverage,
                "cached": True,
            }
    emit({"type": "saved_area_digest", "result": result, "stale": stale})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="JSON worker for the Windows desktop UI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    search = subparsers.add_parser("search")
    search.add_argument("--query", required=True)
    area_search = subparsers.add_parser("area-search")
    area_search.add_argument("--zip", action="append", dest="zip_codes", default=[])
    area_search.add_argument("--center-zip")
    area_search.add_argument("--radius-miles", type=float)
    area_search.add_argument("--max-zip-codes", type=int, default=12)
    subparsers.add_parser("area-profiles")
    subparsers.add_parser("save-area-profile")
    subparsers.add_parser("run-area")
    area_runs = subparsers.add_parser("area-runs")
    area_runs.add_argument("--profile-name")
    subparsers.add_parser("summarize-area")
    saved_area = subparsers.add_parser("saved-area-digest")
    saved_area.add_argument("--profile-name", required=True)
    subparsers.add_parser("run")
    subparsers.add_parser("authenticate")
    subparsers.add_parser("quota-status")
    subparsers.add_parser("diagnostics")
    subparsers.add_parser("diagnostics-selected")
    subparsers.add_parser("prepare-asr-model")
    subparsers.add_parser("asr-self-test")
    subparsers.add_parser("diarization-self-test")
    subparsers.add_parser("analysis-provider-diagnostics")
    subparsers.add_parser("analysis-self-test")
    subparsers.add_parser("profile-self-test")
    library = subparsers.add_parser("library")
    library.add_argument("--output-dir", default="archives")
    subparsers.add_parser("continue-local")
    days = subparsers.add_parser("analysis-days")
    days.add_argument("--feed-id")
    report = subparsers.add_parser("report-day")
    report.add_argument("--feed-id", required=True)
    report.add_argument("--date", required=True)
    clip = subparsers.add_parser("incident-clip")
    clip.add_argument("--incident-id", required=True, type=int)
    clip.add_argument("--context", action="store_true")
    subparsers.add_parser("analyze-day")
    subparsers.add_parser("summarize-week")
    subparsers.add_parser("ask")
    return parser


def load_worker_environment() -> Path | None:
    """Load repository defaults, then an explicitly bundled private env file."""

    loaded: Path | None = None
    repository_env = Path.cwd() / ".env"
    if repository_env.is_file():
        load_dotenv(repository_env, override=True)
        loaded = repository_env
    configured = os.getenv("BROADCASTIFY_ENV_FILE")
    if configured:
        bundled_env = Path(configured)
        if bundled_env.is_file():
            load_dotenv(bundled_env, override=True)
            loaded = bundled_env
    return loaded


def main() -> int:
    global DEFAULT_DATABASE, LOADED_ENVIRONMENT_FILE
    # override=True preserves compatibility with the original USERNAME setting
    # on Windows, where USERNAME already exists in the parent environment.
    LOADED_ENVIRONMENT_FILE = load_worker_environment()
    DEFAULT_DATABASE = Path(
        os.getenv("BROADCASTIFY_ANALYSIS_DB")
        or "archives/broadcastify-analysis.sqlite3"
    )
    arguments = build_parser().parse_args()
    try:
        if arguments.command == "search":
            return search_feeds(arguments.query)
        if arguments.command == "area-search":
            return search_area(
                arguments.zip_codes,
                center_zip=arguments.center_zip,
                radius_miles=arguments.radius_miles,
                max_zip_codes=arguments.max_zip_codes,
            )
        if arguments.command == "area-profiles":
            return list_area_profiles()
        if arguments.command == "save-area-profile":
            return save_area_profile()
        if arguments.command == "run-area":
            return run_area_acquisition()
        if arguments.command == "area-runs":
            return list_area_acquisition_runs(arguments.profile_name)
        if arguments.command == "summarize-area":
            return summarize_area()
        if arguments.command == "saved-area-digest":
            return latest_area_digest(arguments.profile_name)
        if arguments.command == "run":
            return run_job()
        if arguments.command == "authenticate":
            return authenticate()
        if arguments.command == "quota-status":
            return archive_quota_status()
        if arguments.command == "diagnostics":
            return diagnostics()
        if arguments.command == "diagnostics-selected":
            return diagnostics(json.load(sys.stdin))
        if arguments.command == "prepare-asr-model":
            return prepare_asr_model_command()
        if arguments.command == "asr-self-test":
            return asr_self_test()
        if arguments.command == "diarization-self-test":
            return diarization_self_test()
        if arguments.command == "analysis-provider-diagnostics":
            return analysis_provider_diagnostics()
        if arguments.command == "analysis-self-test":
            return analysis_self_test()
        if arguments.command == "profile-self-test":
            return profile_self_test()
        if arguments.command == "library":
            return library_days(arguments.output_dir)
        if arguments.command == "continue-local":
            return continue_local_day()
        if arguments.command == "analysis-days":
            return analysis_days(arguments.feed_id)
        if arguments.command == "report-day":
            return report_day(arguments.feed_id, arguments.date)
        if arguments.command == "incident-clip":
            return incident_clip(
                arguments.incident_id,
                include_surrounding_context=arguments.context,
            )
        if arguments.command == "analyze-day":
            return analyze_day()
        if arguments.command == "summarize-week":
            return summarize_week()
        if arguments.command == "ask":
            return ask_archive()
        raise ValueError(f"Unknown command: {arguments.command}")
    except Exception as exc:
        emit(
            {
                "type": "error",
                "message": str(exc),
                "exception": type(exc).__name__,
                "traceback": traceback.format_exc(),
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
