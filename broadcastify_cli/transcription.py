from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import warnings
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .asr import (
    OpenVinoWhisperAsr,
    WhisperCppAsr,
    WindowsMlWhisperAsr,
    normalize_asr_engine,
    normalize_whisper_model_name,
)
from .accelerators import find_whisper_cpp, whisper_cpp_backends
from .audio import configure_ffmpeg_runtime, find_ffmpeg
from .portable_diarization import (
    COMMUNITY_DIARIZATION_ENGINE,
    COMMUNITY_DIARIZATION_QUALITY,
    PORTABLE_DIARIZATION_ENGINE,
    PORTABLE_DIARIZATION_MODEL,
    PORTABLE_DIARIZATION_QUALITY,
    PortableSpeakerTurn,
    SherpaOnnxDiarizer,
    diarization_engine_satisfies,
    normalize_diarization_engine,
)
from .qwen_asr import SherpaQwen3Asr, normalize_qwen3_asr_model_name
from .workfiles import work_file_owner_token


@dataclass(frozen=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: str


@dataclass(frozen=True)
class TranscriptWord:
    start: float
    end: float
    text: str
    speaker: str | None = None


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str
    speaker: str | None = None


def temporal_overlap(start: float, end: float, turn: SpeakerTurn) -> float:
    return max(0.0, min(end, turn.end) - max(start, turn.start))


def speaker_for_interval(
    start: float,
    end: float,
    turns: Sequence[SpeakerTurn],
    max_nearest_distance: float = 0.5,
) -> str | None:
    best_speaker: str | None = None
    best_overlap = 0.0
    for turn in turns:
        overlap = temporal_overlap(start, end, turn)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = turn.speaker
    if best_speaker is not None:
        return best_speaker

    midpoint = (start + end) / 2.0
    for turn in turns:
        if turn.start <= midpoint <= turn.end:
            return turn.speaker

    nearest_speaker: str | None = None
    nearest_distance = float("inf")
    for turn in turns:
        distance = min(abs(midpoint - turn.start), abs(midpoint - turn.end))
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_speaker = turn.speaker
    if nearest_distance <= max_nearest_distance:
        return nearest_speaker
    return best_speaker


def group_words(
    words: Sequence[TranscriptWord], max_gap: float = 1.5
) -> list[TranscriptSegment]:
    if not words:
        return []

    grouped: list[list[TranscriptWord]] = [[words[0]]]
    for word in words[1:]:
        previous = grouped[-1][-1]
        if word.speaker == previous.speaker and word.start - previous.end <= max_gap:
            grouped[-1].append(word)
        else:
            grouped.append([word])

    return [
        TranscriptSegment(
            start=group[0].start,
            end=group[-1].end,
            text="".join(word.text for word in group).strip(),
            speaker=group[0].speaker,
        )
        for group in grouped
    ]


def format_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    whole_seconds, milliseconds = divmod(milliseconds, 1_000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


class TranscriptionDependencyError(RuntimeError):
    pass


class TranscriptionQualityError(RuntimeError):
    pass


LOCALIZED_REPETITION_POLICY = "localized-repetition-collapse-v1"
TRANSCRIPTION_FINGERPRINT_VERSION = 1


def stable_file_sha256(path: str | Path) -> str:
    """Hash an immutable processing input and reject concurrent mutation."""

    value = Path(path)
    initial = value.stat()
    digest = hashlib.sha256()
    with value.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    final = value.stat()
    if (
        initial.st_size != final.st_size
        or initial.st_mtime_ns != final.st_mtime_ns
    ):
        raise RuntimeError(f"{value.name} changed while its hash was calculated.")
    return digest.hexdigest()


def transcription_processing_fingerprint(
    *,
    model_name: str,
    asr_engine: str,
    diarize: bool,
    diarization_engine: str = COMMUNITY_DIARIZATION_ENGINE,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> str:
    """Return the portable identity of transcript-affecting model settings."""

    engine = str(asr_engine or "faster-whisper").strip().lower()
    raw_model = str(model_name or "turbo").strip()
    try:
        model = (
            normalize_qwen3_asr_model_name(raw_model)
            if engine == "qwen3-asr"
            else normalize_whisper_model_name(raw_model)
        )
    except ValueError:
        model = raw_model.casefold()
    speaker_engine = "none"
    if diarize:
        try:
            speaker_engine = normalize_diarization_engine(diarization_engine)
        except ValueError:
            speaker_engine = str(diarization_engine or "").strip().casefold()
    payload = {
        "version": TRANSCRIPTION_FINGERPRINT_VERSION,
        "asr_engine": engine,
        "model": model,
        "diarize": bool(diarize),
        "diarization_engine": speaker_engine,
        "min_speakers": int(min_speakers) if min_speakers is not None else None,
        "max_speakers": int(max_speakers) if max_speakers is not None else None,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _is_localized_repetition_collapse(text: str) -> bool:
    """Identify a long, low-variety decoder loop inside one ASR segment."""

    tokens = re.findall(r"[\w]+", str(text or "").casefold(), flags=re.UNICODE)
    if len(tokens) < 50:
        return False
    return len(set(tokens)) / len(tokens) <= 0.12


def transcript_quality_report(texts: Sequence[str]) -> dict[str, object]:
    nonempty = [str(value or "").strip() for value in texts if str(value or "").strip()]
    normalized = [
        re.sub(r"[\W_]+", " ", value.casefold(), flags=re.UNICODE).strip()
        or re.sub(r"\s+", "", value)
        for value in nonempty
    ]
    counts = Counter(normalized)
    segment_count = len(normalized)
    unique_count = len(counts)
    dominant_count = max(counts.values(), default=0)
    unique_ratio = unique_count / segment_count if segment_count else 1.0
    dominant_ratio = dominant_count / segment_count if segment_count else 0.0
    alphanumeric_characters = sum(
        1 for value in nonempty for character in value if character.isalnum()
    )
    localized_repetition_count = sum(
        1 for value in nonempty if _is_localized_repetition_collapse(value)
    )
    rejected = bool(
        (segment_count >= 10 and alphanumeric_characters == 0)
        or localized_repetition_count
        or (
            segment_count >= 20
            and dominant_ratio >= 0.75
            and unique_ratio <= 0.20
        )
        or (
            segment_count >= 100
            and dominant_ratio >= 0.55
            and unique_ratio <= 0.10
        )
    )
    return {
        "status": "rejected" if rejected else "passed",
        "segment_count": segment_count,
        "unique_normalized_segments": unique_count,
        "unique_ratio": round(unique_ratio, 6),
        "dominant_segment_count": dominant_count,
        "dominant_segment_ratio": round(dominant_ratio, 6),
        "alphanumeric_characters": alphanumeric_characters,
        "localized_repetition_segment_count": localized_repetition_count,
        "policy": "repetition-collapse-v2",
    }


def _discard_localized_repetition_segments(
    segments: Sequence[TranscriptSegment],
) -> tuple[list[TranscriptSegment], list[TranscriptSegment]]:
    kept: list[TranscriptSegment] = []
    discarded: list[TranscriptSegment] = []
    for segment in segments:
        target = (
            discarded
            if _is_localized_repetition_collapse(segment.text)
            else kept
        )
        target.append(segment)
    return kept, discarded


def _render_transcript_segments(segments: Sequence[TranscriptSegment]) -> str:
    lines: list[str] = []
    for segment in segments:
        speaker = f" {segment.speaker}:" if segment.speaker else ""
        lines.append(
            f"[{format_timestamp(segment.start)}]{speaker} {segment.text}".rstrip()
        )
    return "\n".join(lines) + ("\n" if lines else "")


def _cleanup_metadata(
    discarded: Sequence[TranscriptSegment],
    *,
    previous_count: int = 0,
    previous_intervals: Sequence[dict[str, object]] = (),
) -> dict[str, object]:
    return {
        "policy": LOCALIZED_REPETITION_POLICY,
        "discarded_segment_count": previous_count + len(discarded),
        "discarded_intervals": list(previous_intervals)
        + [
            {
                "start": round(float(segment.start), 3),
                "end": round(float(segment.end), 3),
            }
            for segment in discarded
        ],
    }


def _repair_cached_localized_repetition(
    json_path: Path,
    txt_path: Path,
) -> bool:
    """Remove provable decoder loops from an otherwise useful cached transcript."""

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list):
        return False
    segments = [
        TranscriptSegment(
            start=float(value.get("start", 0.0)),
            end=float(value.get("end", value.get("start", 0.0))),
            text=str(value.get("text") or "").strip(),
            speaker=(
                str(value.get("speaker"))
                if value.get("speaker") is not None
                else None
            ),
        )
        for value in raw_segments
        if isinstance(value, dict) and str(value.get("text") or "").strip()
    ]
    kept, discarded = _discard_localized_repetition_segments(segments)
    if not discarded:
        return False

    quality = transcript_quality_report([segment.text for segment in kept])
    if quality["status"] == "rejected":
        return False

    intervals = [
        (float(segment.start), float(segment.end)) for segment in discarded
    ]
    raw_words = payload.get("words")
    if isinstance(raw_words, list):
        payload["words"] = [
            value
            for value in raw_words
            if not (
                isinstance(value, dict)
                and any(
                    start
                    <= (
                        float(value.get("start", 0.0))
                        + float(value.get("end", value.get("start", 0.0)))
                    )
                    / 2.0
                    <= end
                    for start, end in intervals
                )
            )
        ]

    prior_cleanup = payload.get("transcription_cleanup")
    if not isinstance(prior_cleanup, dict):
        prior_cleanup = {}
    prior_intervals = prior_cleanup.get("discarded_intervals")
    if not isinstance(prior_intervals, list):
        prior_intervals = []
    try:
        prior_count = int(prior_cleanup.get("discarded_segment_count", 0))
    except (TypeError, ValueError):
        prior_count = 0

    rendered_text = _render_transcript_segments(kept)
    payload["segments"] = [asdict(segment) for segment in kept]
    payload["text"] = " ".join(segment.text for segment in kept).strip()
    payload["transcription_quality"] = quality
    payload["transcription_cleanup"] = _cleanup_metadata(
        discarded,
        previous_count=prior_count,
        previous_intervals=[
            value for value in prior_intervals if isinstance(value, dict)
        ],
    )
    payload["rendered_text_sha256"] = hashlib.sha256(
        rendered_text.encode("utf-8")
    ).hexdigest()

    json_temp = json_path.with_suffix(json_path.suffix + ".tmp")
    txt_temp = txt_path.with_suffix(txt_path.suffix + ".tmp")
    try:
        txt_temp.write_text(rendered_text, encoding="utf-8", newline="\n")
        json_temp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        txt_temp.replace(txt_path)
        json_temp.replace(json_path)
    finally:
        txt_temp.unlink(missing_ok=True)
        json_temp.unlink(missing_ok=True)
    return True


_DIARIZATION_SOURCE_SIGNATURE_SCHEMA = 1


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _combined_source_signature(audio_path: Path) -> dict[str, Any] | None:
    """Describe the retained source audio that built a combined timeline."""

    manifest_path = audio_path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        not isinstance(manifest, dict)
        or str(manifest.get("combined_file") or "") != audio_path.name
        or not isinstance(manifest.get("sources"), list)
        or not manifest["sources"]
    ):
        return None

    sources: list[dict[str, Any]] = []
    previous_end = 0.0
    for value in manifest["sources"]:
        if not isinstance(value, dict):
            return None
        source_name = str(value.get("source_file") or "")
        if not source_name or Path(source_name).name != source_name:
            return None
        source_path = audio_path.parent / source_name
        if not source_path.is_file():
            return None
        try:
            start = float(value.get("combined_start_seconds"))
            duration = float(value.get("trimmed_duration_seconds"))
        except (TypeError, ValueError):
            return None
        if (
            not math.isfinite(start)
            or not math.isfinite(duration)
            or start < 0
            or duration <= 0
            or abs(start - previous_end) > 0.01
        ):
            return None
        try:
            source_stat = source_path.stat()
            source_sha256 = _file_sha256(source_path)
        except OSError:
            return None
        sources.append(
            {
                "source_file": source_name,
                "archive_start": value.get("archive_start"),
                "combined_start_seconds": start,
                "trimmed_duration_seconds": duration,
                "source_size": source_stat.st_size,
                "source_sha256": source_sha256,
            }
        )
        previous_end = start + duration
    return {
        "schema": _DIARIZATION_SOURCE_SIGNATURE_SCHEMA,
        "timeline_version": manifest.get("timeline_version"),
        "combined_file": audio_path.name,
        "sources": sources,
    }


def _unchanged_source_prefix_seconds(
    previous: object,
    current: object,
) -> float:
    """Return the decoded prefix proven identical by retained source hashes."""

    if not isinstance(previous, dict) or not isinstance(current, dict):
        return 0.0
    if (
        previous.get("schema") != _DIARIZATION_SOURCE_SIGNATURE_SCHEMA
        or current.get("schema") != _DIARIZATION_SOURCE_SIGNATURE_SCHEMA
        or previous.get("combined_file") != current.get("combined_file")
        or not isinstance(previous.get("sources"), list)
        or not isinstance(current.get("sources"), list)
    ):
        return 0.0

    unchanged_through = 0.0
    for old, new in zip(previous["sources"], current["sources"]):
        if not isinstance(old, dict) or not isinstance(new, dict):
            break
        identity_keys = (
            "source_file",
            "archive_start",
            "source_size",
            "source_sha256",
        )
        if any(old.get(key) != new.get(key) for key in identity_keys):
            break
        try:
            old_start = float(old.get("combined_start_seconds"))
            new_start = float(new.get("combined_start_seconds"))
            old_duration = float(old.get("trimmed_duration_seconds"))
            new_duration = float(new.get("trimmed_duration_seconds"))
        except (TypeError, ValueError):
            break
        if (
            not all(
                math.isfinite(value)
                for value in (old_start, new_start, old_duration, new_duration)
            )
            or abs(old_start - new_start) > 0.01
            or abs(new_start - unchanged_through) > 0.01
            or old_duration <= 0
            or new_duration <= 0
        ):
            break
        unchanged_through = new_start + min(old_duration, new_duration)
        if abs(old_duration - new_duration) > 0.01:
            break
    return unchanged_through


@contextmanager
def decoded_diarization_audio(audio_path: Path):
    """Give pyannote a waveform without relying on its TorchCodec file loader."""

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise TranscriptionDependencyError(
            "PyTorch is required for local speaker labeling."
        ) from exc
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required to decode audio for speaker labeling.")
    directory = audio_path.parent
    raw_path = directory / (
        f".{audio_path.stem}.{work_file_owner_token()}."
        f"{os.getpid()}.{time.time_ns()}.pyannote.f32le"
    )
    process = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(audio_path),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-f",
            "f32le",
            "-c:a",
            "pcm_f32le",
            "-y",
            str(raw_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0 or not raw_path.is_file() or raw_path.stat().st_size < 4:
        raw_path.unlink(missing_ok=True)
        raise RuntimeError(
            process.stderr.strip()
            or "FFmpeg could not decode audio for the speaker-label model."
        )
    sample_count = raw_path.stat().st_size // 4
    waveform = torch.from_file(
        str(raw_path),
        shared=False,
        size=sample_count,
        dtype=torch.float32,
    ).reshape(1, sample_count)
    audio = {"waveform": waveform, "sample_rate": 16_000}
    try:
        yield audio
    finally:
        audio.clear()
        del waveform
        gc.collect()
        try:
            raw_path.unlink(missing_ok=True)
        except OSError:
            # A force-terminated process may leave this uniquely named raw
            # scratch file behind; the retained FLAC cache is still reusable.
            pass


class LocalTranscriber:
    """Fast local ASR with an accuracy or portable-preview diarization stage."""

    DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"

    @property
    def processing_fingerprint(self) -> str:
        return transcription_processing_fingerprint(
            model_name=str(getattr(self, "model_name", "turbo")),
            asr_engine=str(getattr(self, "asr_engine", "faster-whisper")),
            diarize=bool(getattr(self, "diarize", False)),
            diarization_engine=str(
                getattr(
                    self,
                    "diarization_engine",
                    COMMUNITY_DIARIZATION_ENGINE,
                )
            ),
            min_speakers=getattr(self, "min_speakers", None),
            max_speakers=getattr(self, "max_speakers", None),
        )

    def __init__(
        self,
        model_name: str = "turbo",
        asr_engine: str = "auto",
        device: str = "auto",
        device_index: int = 0,
        compute_type: str = "auto",
        asr_model_path: str | Path | None = None,
        diarization_device: str = "auto",
        diarization_engine: str = COMMUNITY_DIARIZATION_ENGINE,
        diarize: bool = False,
        huggingface_token: str | None = None,
        batch_size: int = 8,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        load_asr: bool = True,
    ) -> None:
        configure_ffmpeg_runtime()
        self.model_name = model_name
        self.asr_engine = normalize_asr_engine(asr_engine, device)
        self.batch_size = max(1, batch_size)
        self.diarize = diarize
        self.diarization_engine = normalize_diarization_engine(diarization_engine)
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self._torch = None
        self._asr = None
        self._external_asr = None
        self._batched = False
        self.device_index = device_index
        torch = None
        if self.asr_engine == "faster-whisper" or (
            self.diarize
            and self.diarization_engine == COMMUNITY_DIARIZATION_ENGINE
        ):
            try:
                import torch
            except ModuleNotFoundError as exc:
                raise TranscriptionDependencyError(
                    "Local audio dependencies are missing from the active Python "
                    "runtime. In the installed app, select a prepared Python "
                    "environment under Settings → Processing → Optional Python "
                    'runtime. From source, install with: pip install -e ".[transcription]"'
                ) from exc
            self._torch = torch

        cuda_available = bool(torch and torch.cuda.is_available())
        requested_device = (device or "auto").strip().lower()
        if self.asr_engine == "faster-whisper":
            if requested_device == "auto":
                self.device = "cuda" if cuda_available else "cpu"
            elif requested_device in {"cuda", "cpu"}:
                self.device = requested_device
            else:
                raise RuntimeError(
                    f"faster-whisper cannot use {device!r}. Select CUDA/CPU, "
                    "whisper.cpp for Vulkan, or OpenVINO for Intel hardware."
                )
            if self.device == "cuda" and not cuda_available:
                raise RuntimeError("CUDA was requested but no CUDA device is available to PyTorch.")
            if compute_type == "auto":
                self.compute_type = "float16" if self.device == "cuda" else "int8"
            else:
                self.compute_type = compute_type
        elif self.asr_engine == "whisper.cpp":
            if requested_device == "auto":
                executable = find_whisper_cpp()
                backends = whisper_cpp_backends(executable)
                if sys.platform == "darwin" and "metal" in backends:
                    requested_device = "metal"
                elif "vulkan" in backends:
                    requested_device = "vulkan"
                else:
                    requested_device = "cpu"
            if requested_device not in {"vulkan", "metal", "cpu"}:
                raise RuntimeError(
                    "whisper.cpp currently accepts Vulkan, Apple Metal, or CPU in this app."
                )
            self.device = requested_device
            self.compute_type = "ggml quantized"
        elif self.asr_engine == "openvino":
            openvino_device = requested_device
            if openvino_device.startswith("openvino-"):
                openvino_device = openvino_device.removeprefix("openvino-")
            if openvino_device == "auto":
                openvino_device = "AUTO"
            self.device = f"openvino-{openvino_device.lower()}"
            self.compute_type = "int8"
        elif self.asr_engine == "windows-ml":
            self.device = "windows-ml"
            self.compute_type = "onnx"
        elif self.asr_engine == "qwen3-asr":
            if requested_device not in {"auto", "cpu"}:
                raise RuntimeError(
                    "Qwen3-ASR currently uses its portable sherpa-onnx CPU runtime."
                )
            self.device = "cpu"
            self.compute_type = "int8"
        else:
            raise ValueError(f"Unsupported transcription engine: {self.asr_engine}")

        if load_asr and self.asr_engine == "faster-whisper":
            try:
                from faster_whisper import BatchedInferencePipeline, WhisperModel
            except ModuleNotFoundError as exc:
                raise TranscriptionDependencyError(
                    "faster-whisper is missing from the active Python runtime. "
                    "In the installed app, select a prepared Python environment "
                    "under Settings → Processing → Optional Python runtime. "
                    'From source, install with: pip install -e ".[transcription]"'
                ) from exc
            model_args: dict[str, object] = {
                "device": self.device,
                "compute_type": self.compute_type,
            }
            if self.device == "cuda":
                model_args["device_index"] = self.device_index
            self._whisper_model = WhisperModel(self.model_name, **model_args)
            self._batched = self.batch_size > 1
            self._asr = (
                BatchedInferencePipeline(model=self._whisper_model)
                if self._batched
                else self._whisper_model
            )
        elif load_asr and self.asr_engine == "whisper.cpp":
            self._external_asr = WhisperCppAsr(
                model_name=self.model_name,
                device=self.device,
                device_index=self.device_index,
                model_path=asr_model_path,
            )
        elif load_asr and self.asr_engine == "openvino":
            self._external_asr = OpenVinoWhisperAsr(
                model_name=self.model_name,
                device=self.device,
                model_path=asr_model_path,
                huggingface_token=huggingface_token,
            )
        elif load_asr and self.asr_engine == "windows-ml":
            self._external_asr = WindowsMlWhisperAsr(
                model_name=self.model_name,
                model_path=asr_model_path,
            )
        elif load_asr and self.asr_engine == "qwen3-asr":
            self._external_asr = SherpaQwen3Asr(
                model_name=self.model_name,
                device=self.device,
                model_path=asr_model_path,
                batch_size=self.batch_size,
            )

        if self._external_asr is not None:
            self.backend_description = self._external_asr.backend
        else:
            self.backend_description = (
                f"faster-whisper {self.device}:{self.device_index} ({self.compute_type})"
                if self.asr_engine == "faster-whisper"
                else f"{self.asr_engine} {self.device}"
            )

        self._diarization_pipeline = None
        self._portable_diarizer = None
        self._diarization_details: dict[str, object] = {}
        if self.diarize and self.diarization_engine == COMMUNITY_DIARIZATION_ENGINE:
            try:
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    module=r"pyannote\.audio\.core\.io",
                )
                from pyannote.audio import Pipeline
            except ModuleNotFoundError as exc:
                raise TranscriptionDependencyError(
                    "pyannote.audio is missing from the active Python runtime. "
                    "In the installed app, select a prepared Python environment "
                    "under Settings → Processing → Optional Python runtime."
                ) from exc

            token = (
                huggingface_token
                or os.getenv("HUGGINGFACE_TOKEN")
                or os.getenv("HF_TOKEN")
            )
            try:
                self._diarization_pipeline = Pipeline.from_pretrained(
                    self.DIARIZATION_MODEL, token=token or None
                )
            except Exception as exc:
                if not token:
                    raise RuntimeError(
                        "No Hugging Face read token was supplied and no usable cached "
                        "speaker-label model could be loaded. Add a read token for the "
                        "first download, then the cached model can run offline."
                    ) from exc
                raise
            if self._diarization_pipeline is None:
                message = (
                    "No Hugging Face read token was supplied and no usable cached "
                    "speaker-label model was found. Add a read token for the first download."
                    if not token
                    else "Unable to load the diarization model. Confirm the Hugging Face model terms were accepted."
                )
                raise RuntimeError(message)
            requested_diarization_device = (diarization_device or "auto").lower()
            if requested_diarization_device == "auto":
                requested_diarization_device = "cuda" if cuda_available else "cpu"
            if requested_diarization_device not in {"cuda", "cpu"}:
                raise RuntimeError(
                    "pyannote diarization currently supports CUDA or CPU in this app."
                )
            if requested_diarization_device == "cuda" and not cuda_available:
                raise RuntimeError(
                    "CUDA diarization was requested but PyTorch cannot see a CUDA device."
                )
            self.diarization_device = requested_diarization_device
            assert torch is not None
            target = (
                torch.device(f"cuda:{self.device_index}")
                if self.diarization_device == "cuda"
                else torch.device("cpu")
            )
            self._diarization_pipeline.to(target)
            self.diarization_model = self.DIARIZATION_MODEL
            self.diarization_quality = COMMUNITY_DIARIZATION_QUALITY
        elif self.diarize:
            requested_diarization_device = (diarization_device or "auto").lower()
            if requested_diarization_device == "auto":
                requested_diarization_device = "cpu"
            if requested_diarization_device != "cpu":
                raise RuntimeError(
                    "The fast portable speaker preview currently uses sherpa-onnx "
                    "on CPU. Select CPU or Automatic."
                )
            self.diarization_device = "cpu"
            self._portable_diarizer = SherpaOnnxDiarizer(
                min_speakers=self.min_speakers,
                max_speakers=self.max_speakers,
            )
            self.diarization_model = PORTABLE_DIARIZATION_MODEL
            self.diarization_quality = PORTABLE_DIARIZATION_QUALITY
        else:
            self.diarization_device = "none"
            self.diarization_model = None
            self.diarization_quality = None

    def transcribe_files(
        self,
        audio_files: Iterable[str | Path],
        progress: Callable[[int, int, str], None] | None = None,
    ) -> list[Path]:
        files = sorted(Path(path) for path in audio_files)
        outputs: list[Path] = []
        for index, audio_file in enumerate(files, start=1):
            if progress:
                progress(index - 1, len(files), f"Transcribing {audio_file.name}")

            def detail(message: str) -> None:
                if progress:
                    progress(index - 1, len(files), message)

            outputs.append(self.transcribe_file(audio_file, progress=detail))
            if progress:
                progress(index, len(files), f"Transcribed {index}/{len(files)}")
        return outputs

    def current_transcripts(
        self,
        audio_files: Iterable[str | Path],
    ) -> list[Path]:
        """Return the complete matching cache set without starting model work."""

        outputs: list[Path] = []
        for audio_path in sorted(Path(path) for path in audio_files):
            transcript_dir = audio_path.parent / "transcripts"
            json_path = transcript_dir / f"{audio_path.stem}.json"
            text_path = transcript_dir / f"{audio_path.stem}.txt"
            if not self._existing_transcript_is_current(
                audio_path,
                json_path,
                text_path,
            ):
                return []
            outputs.append(json_path)
        return outputs

    def transcribe_file(
        self,
        audio_file: str | Path,
        progress: Callable[[str], None] | None = None,
    ) -> Path:
        if self._asr is None and self._external_asr is None:
            raise RuntimeError("The ASR model was not loaded for this local audio operation.")
        audio_path = Path(audio_file)
        transcript_dir = audio_path.parent / "transcripts"
        json_path = transcript_dir / f"{audio_path.stem}.json"
        txt_path = transcript_dir / f"{audio_path.stem}.txt"
        if self._existing_transcript_is_current(audio_path, json_path, txt_path):
            if (
                self.diarize
                and getattr(
                    self,
                    "diarization_engine",
                    COMMUNITY_DIARIZATION_ENGINE,
                )
                == PORTABLE_DIARIZATION_ENGINE
            ):
                # Migrate completed legacy caches to the append-resume schema
                # even when the final transcript itself needs no work.
                self._load_diarization_cache(audio_path)
            return json_path

        if self.diarize and progress:
            progress(f"Preparing diarization for {audio_path.name}")
        turns = self._diarize(audio_path, progress=progress) if self.diarize else []
        words: list[TranscriptWord] = []
        fallback_segments: list[TranscriptSegment] = []
        language = "en"
        language_probability = None
        duration = None
        asr_metadata: dict[str, object] = {}
        if self._external_asr is not None:
            if isinstance(self._external_asr, SherpaQwen3Asr):
                result = self._external_asr.transcribe(
                    audio_path,
                    progress=progress,
                    segment_hints=[
                        (float(turn.start), float(turn.end)) for turn in turns
                    ]
                    or None,
                )
            else:
                result = self._external_asr.transcribe(audio_path, progress=progress)
            if result.backend:
                self.backend_description = result.backend
            language = result.language
            language_probability = result.language_probability
            duration = result.duration
            asr_metadata = dict(result.metadata)
            if (
                isinstance(self._external_asr, SherpaQwen3Asr)
                and turns
                and getattr(
                    self,
                    "diarization_engine",
                    COMMUNITY_DIARIZATION_ENGINE,
                )
                == PORTABLE_DIARIZATION_ENGINE
            ):
                asr_metadata["timestamp_source"] = (
                    "sherpa-onnx-speaker-preview-turns"
                )
                asr_metadata["speaker_label_quality"] = (
                    PORTABLE_DIARIZATION_QUALITY
                )
            for word in result.words:
                words.append(
                    TranscriptWord(
                        start=float(word.start),
                        end=float(word.end),
                        text=str(word.text),
                        speaker=speaker_for_interval(
                            float(word.start), float(word.end), turns
                        ),
                    )
                )
            for segment in result.segments:
                fallback_segments.append(
                    TranscriptSegment(
                        start=float(segment.start),
                        end=float(segment.end),
                        text=str(segment.text).strip(),
                        speaker=speaker_for_interval(
                            float(segment.start), float(segment.end), turns
                        ),
                    )
                )
        else:
            if progress:
                progress(f"Running Whisper {self.model_name} on {audio_path.name}")
            transcribe_args: dict[str, object] = {
                "beam_size": 5,
                "condition_on_previous_text": False,
                "language": "en",
                "word_timestamps": True,
                "vad_filter": True,
                "no_speech_threshold": 0.6,
                "initial_prompt": "Police, fire, EMS, and public safety radio traffic.",
            }
            if self._batched:
                transcribe_args["batch_size"] = self.batch_size
            assert self._asr is not None
            raw_segments, info = self._asr.transcribe(str(audio_path), **transcribe_args)
            language = getattr(info, "language", "en")
            language_probability = getattr(info, "language_probability", None)
            duration = getattr(info, "duration", None)
            for segment in raw_segments:
                segment_text = str(segment.text).strip()

                segment_words = getattr(segment, "words", None) or []
                added_word = False
                for word in segment_words:
                    if word.start is None or word.end is None:
                        continue
                    speaker = speaker_for_interval(
                        float(word.start), float(word.end), turns
                    )
                    words.append(
                        TranscriptWord(
                            start=float(word.start),
                            end=float(word.end),
                            text=str(word.word),
                            speaker=speaker,
                        )
                    )
                    added_word = True

                if not added_word and segment_text:
                    fallback_segments.append(
                        TranscriptSegment(
                            start=float(segment.start),
                            end=float(segment.end),
                            text=segment_text,
                            speaker=speaker_for_interval(
                                float(segment.start), float(segment.end), turns
                            ),
                        )
                    )

        output_segments = group_words(words) if words else [
            value for value in fallback_segments if value.text
        ]
        output_segments, discarded_segments = _discard_localized_repetition_segments(
            output_segments
        )
        if discarded_segments and words:
            discarded_intervals = [
                (float(segment.start), float(segment.end))
                for segment in discarded_segments
            ]
            words = [
                word
                for word in words
                if not any(
                    start <= (float(word.start) + float(word.end)) / 2.0 <= end
                    for start, end in discarded_intervals
                )
            ]
        quality = transcript_quality_report(
            [segment.text for segment in output_segments]
        )
        if quality["status"] == "rejected":
            raise TranscriptionQualityError(
                "Transcription quality check rejected a repetitive or non-speech "
                f"collapse: {quality['unique_normalized_segments']} unique segments "
                f"across {quality['segment_count']}, with the most common output at "
                f"{float(quality['dominant_segment_ratio']) * 100:.1f}%. No new "
                "transcript or analysis was saved. Use a VAD-backed profile and Base "
                "or larger Whisper model, then retry the retained audio."
            )
        transcript_dir.mkdir(parents=True, exist_ok=True)

        actual_model = str(asr_metadata.get("model") or self.model_name)
        rendered_text = _render_transcript_segments(output_segments)
        payload = {
            "audio_file": audio_path.name,
            "audio_sha256": stable_file_sha256(audio_path),
            "processing_fingerprint": self.processing_fingerprint,
            "model": actual_model,
            "requested_model": self.model_name,
            "asr_engine": self.asr_engine,
            "asr_backend": self.backend_description,
            "device": self.device,
            "compute_type": self.compute_type,
            "language": language,
            "language_probability": language_probability,
            "duration": duration,
            "text": " ".join(segment.text for segment in output_segments).strip(),
            "segments": [asdict(segment) for segment in output_segments],
            "words": [asdict(word) for word in words],
            "speaker_turns": [asdict(turn) for turn in turns],
            "diarization_requested": self.diarize,
            "diarization_completed": self.diarize,
            "diarization_engine": (
                getattr(
                    self,
                    "diarization_engine",
                    COMMUNITY_DIARIZATION_ENGINE,
                )
                if self.diarize
                else None
            ),
            "diarization_model": (
                getattr(self, "diarization_model", self.DIARIZATION_MODEL)
                if self.diarize
                else None
            ),
            "diarization_quality": (
                getattr(
                    self,
                    "diarization_quality",
                    COMMUNITY_DIARIZATION_QUALITY,
                )
                if self.diarize
                else None
            ),
            "diarization_device": self.diarization_device,
            "diarization_metadata": (
                dict(getattr(self, "_diarization_details", {}))
                if self.diarize
                else {}
            ),
            "asr_metadata": asr_metadata,
            "transcription_quality": quality,
            "transcription_cleanup": _cleanup_metadata(discarded_segments),
            "rendered_text_sha256": hashlib.sha256(
                rendered_text.encode("utf-8")
            ).hexdigest(),
        }
        json_temp = json_path.with_suffix(json_path.suffix + ".tmp")
        txt_temp = txt_path.with_suffix(txt_path.suffix + ".tmp")
        try:
            txt_temp.write_text(rendered_text, encoding="utf-8", newline="\n")
            json_temp.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            txt_temp.replace(txt_path)
            json_temp.replace(json_path)
        finally:
            txt_temp.unlink(missing_ok=True)
            json_temp.unlink(missing_ok=True)
        return json_path

    def diarize_existing_transcript(
        self,
        audio_file: str | Path,
        transcript_file: str | Path,
        progress: Callable[[str], None] | None = None,
    ) -> Path:
        """Attach speaker labels without running transcription a second time."""

        if not self.diarize or (
            self._diarization_pipeline is None
            and getattr(self, "_portable_diarizer", None) is None
        ):
            raise RuntimeError("Speaker diarization is not enabled for this operation.")
        audio_path = Path(audio_file)
        json_path = Path(transcript_file)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Combined audio does not exist: {audio_path}")
        if not json_path.is_file():
            raise FileNotFoundError(f"Transcript does not exist: {json_path}")

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        raw_words = payload.get("words") or []
        raw_segments = payload.get("segments") or []
        if not isinstance(raw_words, list) or not isinstance(raw_segments, list):
            raise ValueError(f"Transcript has invalid word or segment data: {json_path}")

        if progress:
            progress(f"Adding speaker labels to {audio_path.name}")
        turns = self._diarize(audio_path, progress=progress)
        words = [
            TranscriptWord(
                start=float(value.get("start", 0.0)),
                end=float(value.get("end", value.get("start", 0.0))),
                text=str(value.get("text") or ""),
                speaker=speaker_for_interval(
                    float(value.get("start", 0.0)),
                    float(value.get("end", value.get("start", 0.0))),
                    turns,
                ),
            )
            for value in raw_words
            if str(value.get("text") or "")
        ]
        if words:
            segments = group_words(words)
        else:
            segments = [
                TranscriptSegment(
                    start=float(value.get("start", 0.0)),
                    end=float(value.get("end", value.get("start", 0.0))),
                    text=str(value.get("text") or "").strip(),
                    speaker=speaker_for_interval(
                        float(value.get("start", 0.0)),
                        float(value.get("end", value.get("start", 0.0))),
                        turns,
                    ),
                )
                for value in raw_segments
                if str(value.get("text") or "").strip()
            ]

        payload["segments"] = [asdict(segment) for segment in segments]
        payload["words"] = [asdict(word) for word in words]
        payload["speaker_turns"] = [asdict(turn) for turn in turns]
        payload["diarization_requested"] = True
        payload["diarization_completed"] = True
        payload["diarization_engine"] = getattr(
            self, "diarization_engine", COMMUNITY_DIARIZATION_ENGINE
        )
        payload["diarization_model"] = getattr(
            self, "diarization_model", self.DIARIZATION_MODEL
        )
        payload["diarization_quality"] = getattr(
            self,
            "diarization_quality",
            COMMUNITY_DIARIZATION_QUALITY,
        )
        payload["diarization_device"] = self.diarization_device
        payload["diarization_metadata"] = dict(
            getattr(self, "_diarization_details", {})
        )

        json_temp = json_path.with_suffix(json_path.suffix + ".tmp")
        json_temp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        json_temp.replace(json_path)

        txt_path = json_path.with_suffix(".txt")
        lines = []
        for segment in segments:
            speaker = f" {segment.speaker}:" if segment.speaker else ""
            lines.append(
                f"[{format_timestamp(segment.start)}]{speaker} {segment.text}".rstrip()
            )
        txt_temp = txt_path.with_suffix(txt_path.suffix + ".tmp")
        txt_temp.write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        txt_temp.replace(txt_path)
        if progress:
            progress(f"Speaker labels added to {json_path.name}")
        return json_path

    def _existing_transcript_is_current(
        self, audio_path: Path, json_path: Path, txt_path: Path
    ) -> bool:
        if not json_path.is_file() or not txt_path.is_file():
            return False
        if (
            min(json_path.stat().st_mtime, txt_path.stat().st_mtime)
            < audio_path.stat().st_mtime
        ):
            return False
        try:
            _repair_cached_localized_repetition(json_path, txt_path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return False
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        saved_audio_sha256 = str(payload.get("audio_sha256") or "")
        if saved_audio_sha256:
            try:
                if stable_file_sha256(audio_path) != saved_audio_sha256:
                    return False
            except (OSError, RuntimeError):
                return False
        saved_fingerprint = str(payload.get("processing_fingerprint") or "")
        if saved_fingerprint and saved_fingerprint != self.processing_fingerprint:
            return False
        quality = transcript_quality_report(
            [
                str(value.get("text") or "")
                for value in payload.get("segments", [])
                if isinstance(value, dict)
            ]
        )
        if quality["status"] == "rejected":
            return False
        saved_quality = payload.get("transcription_quality")
        if isinstance(saved_quality, dict) and saved_quality.get("status") == "rejected":
            return False
        expected_text_sha256 = str(payload.get("rendered_text_sha256") or "")
        if expected_text_sha256:
            try:
                actual_text_sha256 = hashlib.sha256(txt_path.read_bytes()).hexdigest()
            except OSError:
                return False
            if actual_text_sha256 != expected_text_sha256:
                return False
        expected_engine = getattr(self, "asr_engine", "faster-whisper")
        requested_model = str(payload.get("requested_model") or payload.get("model") or "")
        if expected_engine == "qwen3-asr":
            try:
                saved_model = normalize_qwen3_asr_model_name(requested_model)
                current_model = normalize_qwen3_asr_model_name(self.model_name)
            except ValueError:
                return False
        else:
            saved_model = normalize_whisper_model_name(requested_model)
            current_model = normalize_whisper_model_name(self.model_name)
        if saved_model != current_model:
            return False
        actual_engine = str(payload.get("asr_engine") or "faster-whisper")
        if actual_engine != expected_engine:
            return False
        requested = bool(payload.get("diarization_requested"))
        if "diarization_requested" not in payload:
            requested = bool(payload.get("diarization_model")) or any(
                value.get("speaker") for value in payload.get("segments", [])
            )
        if requested != self.diarize:
            return False
        if not self.diarize:
            return True
        completed = bool(payload.get("diarization_completed"))
        if "diarization_completed" not in payload:
            completed = bool(payload.get("diarization_model")) and (
                "speaker_turns" in payload
                or any(value.get("speaker") for value in payload.get("segments", []))
            )
        if not completed:
            return False
        actual_diarization_engine = str(
            payload.get("diarization_engine") or ""
        )
        if not actual_diarization_engine:
            actual_model = str(payload.get("diarization_model") or "")
            actual_diarization_engine = (
                PORTABLE_DIARIZATION_ENGINE
                if actual_model == PORTABLE_DIARIZATION_MODEL
                else COMMUNITY_DIARIZATION_ENGINE
            )
        return diarization_engine_satisfies(
            actual_diarization_engine,
            getattr(
                self,
                "diarization_engine",
                COMMUNITY_DIARIZATION_ENGINE,
            ),
        )

    def _diarize(
        self,
        audio_path: Path,
        progress: Callable[[str], None] | None = None,
    ) -> list[SpeakerTurn]:
        cached = self._load_diarization_cache(audio_path)
        if cached is not None:
            if (
                getattr(self, "diarization_engine", COMMUNITY_DIARIZATION_ENGINE)
                == PORTABLE_DIARIZATION_ENGINE
            ):
                self._portable_diarization_checkpoint_path(audio_path).unlink(
                    missing_ok=True
                )
            if progress:
                progress(f"Reusing cached diarization for {audio_path.name}")
            return cached
        portable = getattr(self, "_portable_diarizer", None)
        if portable is not None:
            checkpoint_path = self._portable_diarization_checkpoint_path(audio_path)
            stale_payload = getattr(self, "_stale_diarization_payload", None)
            if isinstance(stale_payload, dict):
                previous_signature = stale_payload.get("source_signature")
                current_signature = _combined_source_signature(audio_path)
                unchanged_through = _unchanged_source_prefix_seconds(
                    previous_signature,
                    current_signature,
                )
                metadata = stale_payload.get("metadata")
                previous_identity = (
                    metadata.get("checkpoint_identity")
                    if isinstance(metadata, dict)
                    else None
                )
                previous_chunk_count = (
                    metadata.get("chunk_count")
                    if isinstance(metadata, dict)
                    else None
                )
                previous_turns: list[PortableSpeakerTurn] = []
                previous_turns_valid = True
                for value in stale_payload.get("turns", []):
                    if not isinstance(value, dict):
                        previous_turns_valid = False
                        break
                    try:
                        previous_turns.append(
                            PortableSpeakerTurn(
                                start=float(value["start"]),
                                end=float(value["end"]),
                                speaker=str(value["speaker"]),
                            )
                        )
                    except (KeyError, TypeError, ValueError):
                        previous_turns_valid = False
                        break
                seeded = 0
                if (
                    isinstance(previous_identity, dict)
                    and isinstance(previous_chunk_count, int)
                    and not isinstance(previous_chunk_count, bool)
                    and previous_turns_valid
                    and unchanged_through > 0
                ):
                    seeded = portable.seed_checkpoint_from_completed_turns(
                        audio_path,
                        checkpoint_path=checkpoint_path,
                        previous_identity=previous_identity,
                        previous_chunk_count=previous_chunk_count,
                        turns=previous_turns,
                        unchanged_through_seconds=unchanged_through,
                    )
                if seeded and progress:
                    progress(
                        "Reusing "
                        f"{seeded} completed fast speaker preview chunks from "
                        f"{unchanged_through / 60:.1f} unchanged minutes."
                    )
            portable_turns = portable.process(
                audio_path,
                progress=progress,
                checkpoint_path=checkpoint_path,
                cleanup_checkpoint_on_success=False,
            )
            turns = [
                SpeakerTurn(value.start, value.end, value.speaker)
                for value in portable_turns
            ]
            self._diarization_details = dict(portable.metadata)
            self._save_diarization_cache(audio_path, turns)
            try:
                checkpoint_path.unlink(missing_ok=True)
            except OSError:
                # The completed final cache is authoritative. A stale chunk
                # checkpoint is harmless and will be removed when that cache
                # is reused.
                pass
            return turns
        if self._diarization_pipeline is None:
            return []

        diarization_args: dict[str, int] = {}
        if self.min_speakers is not None:
            diarization_args["min_speakers"] = self.min_speakers
        if self.max_speakers is not None:
            diarization_args["max_speakers"] = self.max_speakers
        if hasattr(self._diarization_pipeline, "embedding_batch_size"):
            # Some generic pyannote constructors default to one embedding at a
            # time, while downloaded pipelines can carry a larger tuned value
            # (community-1 currently uses 32). Never lower the model's own
            # setting; only let the bounded CLI/UI batch size raise it.
            existing_batch_size = max(
                1, int(self._diarization_pipeline.embedding_batch_size)
            )
            self._diarization_pipeline.embedding_batch_size = max(
                existing_batch_size,
                int(getattr(self, "batch_size", 1)),
            )
        diarization_input, temporary_input = self._prepare_diarization_input(audio_path)
        last_step = ""
        last_emit = 0.0

        def diarization_progress(
            step_name: str,
            _artifact: object,
            file: object | None = None,
            total: int | None = None,
            completed: int | None = None,
        ) -> None:
            nonlocal last_step, last_emit
            if progress is None:
                return
            now = time.monotonic()
            step = str(step_name).replace("_", " ")
            changed_step = step != last_step
            complete = total is not None and completed is not None and completed >= total
            if not changed_step and not complete and now - last_emit < 5.0:
                return
            if total and completed is not None:
                percent = max(0.0, min(100.0, completed * 100.0 / total))
                message = f"Diarization {step}: {percent:.0f}% ({completed}/{total})"
            else:
                message = f"Diarization {step}"
            progress(message)
            last_step = step
            last_emit = now

        with decoded_diarization_audio(diarization_input) as diarization_audio:
            output = self._diarization_pipeline(
                diarization_audio,
                hook=diarization_progress if progress is not None else None,
                **diarization_args,
            )
        annotation = getattr(output, "exclusive_speaker_diarization", None)
        if annotation is None:
            annotation = getattr(output, "speaker_diarization", None)
        if annotation is None:
            annotation = output

        turns: list[SpeakerTurn] = []
        if hasattr(annotation, "itertracks"):
            for segment, _, label in annotation.itertracks(yield_label=True):
                turns.append(
                    SpeakerTurn(float(segment.start), float(segment.end), str(label))
                )
        else:
            for item in annotation:
                if len(item) == 2:
                    segment, label = item
                elif len(item) == 3:
                    segment, _, label = item
                else:
                    continue
                turns.append(
                    SpeakerTurn(float(segment.start), float(segment.end), str(label))
                )
        turns.sort(key=lambda value: (value.start, value.end, value.speaker))
        self._save_diarization_cache(audio_path, turns)
        if temporary_input:
            # Keep this expensive lossless preparation when diarization or
            # cache persistence fails so the next attempt can resume cheaply.
            try:
                diarization_input.unlink(missing_ok=True)
            except OSError:
                pass
        return turns

    def _prepare_diarization_input(self, audio_path: Path) -> tuple[Path, bool]:
        if audio_path.suffix.lower() in {".wav", ".flac"}:
            return audio_path, False
        cache_dir = audio_path.parent / "transcripts" / ".cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        prepared = cache_dir / f"{audio_path.stem}.pyannote.flac"
        partial = cache_dir / (
            f".{audio_path.stem}.{work_file_owner_token()}."
            f"{os.getpid()}.{time.time_ns()}"
            ".pyannote.part.flac"
        )
        if (
            prepared.is_file()
            and prepared.stat().st_size > 0
            and prepared.stat().st_mtime_ns >= audio_path.stat().st_mtime_ns
        ):
            # A force-terminated diarization run can leave the fully prepared
            # lossless input behind. It was atomically promoted only after a
            # successful FFmpeg conversion, so reuse it on the retry.
            return prepared, True
        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            raise RuntimeError(
                "FFmpeg is required to prepare sample-accurate audio for diarization."
            )
        process = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(audio_path),
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "flac",
                "-compression_level",
                "5",
                "-y",
                str(partial),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if process.returncode != 0 or not partial.is_file() or partial.stat().st_size == 0:
            partial.unlink(missing_ok=True)
            raise RuntimeError(
                process.stderr.strip()
                or "FFmpeg could not prepare lossless diarization audio."
            )
        partial.replace(prepared)
        return prepared, True

    def _diarization_cache_path(self, audio_path: Path) -> Path:
        engine = getattr(
            self, "diarization_engine", COMMUNITY_DIARIZATION_ENGINE
        )
        suffix = (
            ".diarization.json"
            if engine == COMMUNITY_DIARIZATION_ENGINE
            else f".diarization.{engine}.json"
        )
        return audio_path.parent / "transcripts" / f"{audio_path.stem}{suffix}"

    def _portable_diarization_checkpoint_path(self, audio_path: Path) -> Path:
        return (
            audio_path.parent
            / "transcripts"
            / f"{audio_path.stem}.diarization.{PORTABLE_DIARIZATION_ENGINE}.chunks.json"
        )

    def _load_diarization_cache(self, audio_path: Path) -> list[SpeakerTurn] | None:
        cache_path = self._diarization_cache_path(audio_path)
        self._stale_diarization_payload = None
        if not cache_path.is_file():
            return None
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            expected_engine = getattr(
                self, "diarization_engine", COMMUNITY_DIARIZATION_ENGINE
            )
            cached_engine = str(
                payload.get("engine") or COMMUNITY_DIARIZATION_ENGINE
            )
            if cached_engine != expected_engine:
                return None
            expected_model = getattr(
                self, "diarization_model", self.DIARIZATION_MODEL
            )
            if payload.get("model") != expected_model:
                return None
            if payload.get("min_speakers") != self.min_speakers:
                return None
            if payload.get("max_speakers") != self.max_speakers:
                return None
            audio_stat = audio_path.stat()
            if (
                int(payload.get("audio_size", -1)) != audio_stat.st_size
                or int(payload.get("audio_mtime_ns", -1))
                != audio_stat.st_mtime_ns
            ):
                if expected_engine == PORTABLE_DIARIZATION_ENGINE:
                    self._stale_diarization_payload = payload
                return None
            turns = [
                SpeakerTurn(**value)
                for value in payload.get("turns", [])
            ]
            metadata = dict(payload.get("metadata") or {})
            portable = getattr(self, "_portable_diarizer", None)
            changed = False
            if expected_engine == PORTABLE_DIARIZATION_ENGINE and portable is not None:
                try:
                    if not isinstance(metadata.get("checkpoint_identity"), dict):
                        identity, _, _ = portable.checkpoint_identity(audio_path)
                        metadata["checkpoint_identity"] = identity
                        payload["metadata"] = metadata
                        changed = True
                    if not isinstance(payload.get("source_signature"), dict):
                        signature = _combined_source_signature(audio_path)
                        if signature is not None:
                            payload["source_signature"] = signature
                            changed = True
                except (OSError, RuntimeError, TypeError, ValueError):
                    # An exact completed cache remains usable even if optional
                    # append-resume metadata cannot be migrated on this host.
                    pass
            if changed:
                try:
                    self._write_diarization_cache_payload(cache_path, payload)
                except OSError:
                    # Append-resume enrichment is optional; the exact cache is
                    # still authoritative when its migration cannot be saved.
                    pass
            self._diarization_details = metadata
            return turns
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    @staticmethod
    def _write_diarization_cache_payload(
        cache_path: Path,
        payload: dict[str, Any],
    ) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        partial = cache_path.with_suffix(cache_path.suffix + ".tmp")
        try:
            partial.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            partial.replace(cache_path)
        finally:
            partial.unlink(missing_ok=True)

    def _save_diarization_cache(
        self, audio_path: Path, turns: Sequence[SpeakerTurn]
    ) -> None:
        cache_path = self._diarization_cache_path(audio_path)
        engine = getattr(
            self,
            "diarization_engine",
            COMMUNITY_DIARIZATION_ENGINE,
        )
        payload: dict[str, Any] = {
            "engine": engine,
            "model": getattr(
                self, "diarization_model", self.DIARIZATION_MODEL
            ),
            "quality": getattr(
                self,
                "diarization_quality",
                COMMUNITY_DIARIZATION_QUALITY,
            ),
            "audio_size": audio_path.stat().st_size,
            "audio_mtime_ns": audio_path.stat().st_mtime_ns,
            "min_speakers": self.min_speakers,
            "max_speakers": self.max_speakers,
            "device": self.diarization_device,
            "metadata": dict(
                getattr(self, "_diarization_details", {})
            ),
            "turns": [asdict(value) for value in turns],
        }
        if engine == PORTABLE_DIARIZATION_ENGINE:
            signature = _combined_source_signature(audio_path)
            if signature is not None:
                payload["source_signature"] = signature
        self._write_diarization_cache_payload(cache_path, payload)
