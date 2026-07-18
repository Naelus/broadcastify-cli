from __future__ import annotations

import json
import gc
import os
import subprocess
import sys
import time
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

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
    SherpaOnnxDiarizer,
    diarization_engine_satisfies,
    normalize_diarization_engine,
)
from .qwen_asr import SherpaQwen3Asr, normalize_qwen3_asr_model_name


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
        f".{audio_path.stem}.{os.getpid()}.{time.time_ns()}.pyannote.f32le"
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
                    'Local audio dependencies are missing. Install with: pip install -e ".[transcription]"'
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
                    'Transcription dependencies are missing. Install with: pip install -e ".[transcription]"'
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
                    "pyannote.audio is required when diarization is enabled."
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
            return json_path

        if self.diarize and progress:
            progress(f"Preparing diarization for {audio_path.name}")
        turns = self._diarize(audio_path, progress=progress) if self.diarize else []
        words: list[TranscriptWord] = []
        fallback_segments: list[TranscriptSegment] = []
        full_text: list[str] = []
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
            if result.text:
                full_text.append(result.text)
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
                if segment_text:
                    full_text.append(segment_text)

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
        transcript_dir.mkdir(parents=True, exist_ok=True)

        actual_model = str(asr_metadata.get("model") or self.model_name)
        payload = {
            "audio_file": audio_path.name,
            "model": actual_model,
            "requested_model": self.model_name,
            "asr_engine": self.asr_engine,
            "asr_backend": self.backend_description,
            "device": self.device,
            "compute_type": self.compute_type,
            "language": language,
            "language_probability": language_probability,
            "duration": duration,
            "text": " ".join(full_text).strip(),
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
        }
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        lines = []
        for segment in output_segments:
            speaker = f" {segment.speaker}:" if segment.speaker else ""
            lines.append(
                f"[{format_timestamp(segment.start)}]{speaker} {segment.text}".rstrip()
            )
        txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return json_path

    def diarize_existing_transcript(
        self,
        audio_file: str | Path,
        transcript_file: str | Path,
        progress: Callable[[str], None] | None = None,
    ) -> Path:
        """Attach speaker labels without running Whisper a second time."""

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
        if min(json_path.stat().st_mtime, txt_path.stat().st_mtime) < audio_path.stat().st_mtime:
            return False
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
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
            portable_turns = portable.process(
                audio_path,
                progress=progress,
                checkpoint_path=checkpoint_path,
            )
            turns = [
                SpeakerTurn(value.start, value.end, value.speaker)
                for value in portable_turns
            ]
            self._diarization_details = dict(portable.metadata)
            self._save_diarization_cache(audio_path, turns)
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
        partial = cache_dir / f"{audio_path.stem}.pyannote.part.flac"
        partial.unlink(missing_ok=True)
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
            if int(payload.get("audio_size", -1)) != audio_path.stat().st_size:
                return None
            if int(payload.get("audio_mtime_ns", -1)) != audio_path.stat().st_mtime_ns:
                return None
            if payload.get("min_speakers") != self.min_speakers:
                return None
            if payload.get("max_speakers") != self.max_speakers:
                return None
            self._diarization_details = dict(payload.get("metadata") or {})
            return [SpeakerTurn(**value) for value in payload.get("turns", [])]
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _save_diarization_cache(
        self, audio_path: Path, turns: Sequence[SpeakerTurn]
    ) -> None:
        cache_path = self._diarization_cache_path(audio_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        partial = cache_path.with_suffix(cache_path.suffix + ".tmp")
        partial.write_text(
            json.dumps(
                {
                    "engine": getattr(
                        self,
                        "diarization_engine",
                        COMMUNITY_DIARIZATION_ENGINE,
                    ),
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
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        partial.replace(cache_path)
