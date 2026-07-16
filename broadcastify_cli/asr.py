from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from .accelerators import find_whisper_cpp, whisper_cpp_backends
from .audio import find_ffmpeg


@dataclass(frozen=True)
class AsrWord:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class AsrSegment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class AsrResult:
    text: str
    language: str = "en"
    language_probability: float | None = None
    duration: float | None = None
    words: Sequence[AsrWord] = ()
    segments: Sequence[AsrSegment] = ()
    engine: str = ""
    backend: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class AsrDependencyError(RuntimeError):
    pass


def normalize_asr_engine(engine: str, device: str) -> str:
    normalized = (engine or "auto").strip().lower().replace("_", "-")
    if normalized == "auto":
        device_name = (device or "auto").strip().lower()
        if device_name == "vulkan":
            return "whisper.cpp"
        if device_name.startswith("openvino") or device_name in {"gpu", "npu"}:
            return "openvino"
        if device_name.startswith("windows") or device_name == "directml":
            return "windows-ml"
        return "faster-whisper"
    aliases = {
        "fasterwhisper": "faster-whisper",
        "whisper-cpp": "whisper.cpp",
        "whispercpp": "whisper.cpp",
        "windowsml": "windows-ml",
    }
    return aliases.get(normalized, normalized)


WHISPER_CPP_MODELS = {
    "turbo": "ggml-large-v3-turbo-q5_0.bin",
    "large-v3-turbo": "ggml-large-v3-turbo-q5_0.bin",
    "large-v3": "ggml-large-v3-q5_0.bin",
    "medium": "ggml-medium.en-q5_0.bin",
    "small": "ggml-small.en-q5_1.bin",
    "base": "ggml-base.en-q5_1.bin",
    "tiny": "ggml-tiny.en-q5_1.bin",
}


def whisper_cpp_model_filename(model_name: str) -> str:
    normalized = model_name.strip().lower()
    try:
        return WHISPER_CPP_MODELS[normalized]
    except KeyError as exc:
        raise ValueError(
            f"The whisper.cpp engine does not have a configured model mapping for {model_name!r}."
        ) from exc


def _default_model_root() -> Path:
    configured = os.getenv("BROADCASTIFY_MODEL_DIR")
    if configured:
        return Path(configured).expanduser()
    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "Broadcastify Desktop" / "models"
    return Path.home() / ".cache" / "broadcastify-desktop" / "models"


def find_whisper_cpp_model(
    model_name: str, explicit_path: str | Path | None = None
) -> Path | None:
    configured = explicit_path or os.getenv("WHISPER_CPP_MODEL_PATH")
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file():
            return candidate.resolve()
    filename = whisper_cpp_model_filename(model_name)
    roots = [
        _default_model_root() / "whisper.cpp",
        Path.cwd() / "models",
        Path.cwd() / "whisper.cpp" / "models",
    ]
    executable = find_whisper_cpp()
    if executable:
        roots.append(Path(executable).resolve().parent / "models")
    for root in roots:
        candidate = root / filename
        if candidate.is_file():
            return candidate.resolve()
    return None


class WhisperCppAsr:
    """Runs an installed whisper.cpp CLI and normalizes its JSON output."""

    def __init__(
        self,
        model_name: str,
        device: str = "vulkan",
        device_index: int = 0,
        executable: str | Path | None = None,
        model_path: str | Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = (device or "vulkan").lower()
        self.device_index = max(0, int(device_index))
        self.executable = str(executable or find_whisper_cpp() or "")
        if not self.executable or not Path(self.executable).is_file():
            raise AsrDependencyError(
                "whisper-cli was not found. Install or build whisper.cpp and set "
                "WHISPER_CPP_PATH to whisper-cli.exe. A Vulkan build must be compiled "
                "with GGML_VULKAN=1."
            )
        resolved_model = find_whisper_cpp_model(model_name, model_path)
        if resolved_model is None:
            expected = whisper_cpp_model_filename(model_name)
            raise AsrDependencyError(
                f"The whisper.cpp model {expected} was not found. Put it under "
                f"{_default_model_root() / 'whisper.cpp'} or set WHISPER_CPP_MODEL_PATH."
            )
        self.model_path = resolved_model
        self.backends = whisper_cpp_backends(self.executable)
        if self.device == "vulkan" and "vulkan" not in self.backends:
            raise AsrDependencyError(
                "The selected whisper.cpp executable does not expose ggml-vulkan. "
                "Use a build compiled with GGML_VULKAN=1 or select CPU."
            )
        self.backend = self.device if self.device != "auto" else ",".join(self.backends)

    def transcribe(
        self,
        audio_path: str | Path,
        progress: Callable[[str], None] | None = None,
    ) -> AsrResult:
        source = Path(audio_path)
        cache_dir = source.parent / "transcripts" / ".cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="whisper-cpp-", dir=cache_dir) as temporary:
            output_base = Path(temporary) / source.stem
            arguments = [
                self.executable,
                "--model",
                str(self.model_path),
                "--file",
                str(source),
                "--language",
                "en",
                "--beam-size",
                "5",
                "--no-speech-thold",
                "0.6",
                "--prompt",
                "Police, fire, EMS, and public safety radio traffic.",
                "--suppress-nst",
                "--output-json",
                "--output-file",
                str(output_base),
                "--print-progress",
                "--no-prints",
                "--device",
                str(self.device_index),
            ]
            if self.device == "cpu":
                arguments.append("--no-gpu")
            if progress:
                progress(
                    f"Running whisper.cpp {self.model_name} on {self.backend or 'auto'}"
                )
            process = subprocess.Popen(
                arguments,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            error_lines: list[str] = []
            assert process.stderr is not None
            for line in process.stderr:
                line = line.strip()
                if line:
                    error_lines.append(line)
                match = re.search(r"progress\s*=\s*(\d+)%", line)
                if match and progress:
                    progress(f"whisper.cpp transcription: {match.group(1)}%")
            return_code = process.wait()
            json_path = output_base.with_suffix(".json")
            if return_code != 0 or not json_path.is_file():
                detail = "\n".join(error_lines[-20:])
                raise RuntimeError(
                    detail or f"whisper.cpp exited with code {return_code}."
                )
            payload = json.loads(json_path.read_text(encoding="utf-8"))

        segments: list[AsrSegment] = []
        for value in payload.get("transcription", []):
            offsets = value.get("offsets") or {}
            start = float(offsets.get("from", 0.0)) / 1000.0
            end = float(offsets.get("to", offsets.get("from", 0.0))) / 1000.0
            text = str(value.get("text") or "").strip()
            if text:
                segments.append(AsrSegment(start, end, text))
        text = " ".join(value.text for value in segments).strip()
        language = str((payload.get("result") or {}).get("language") or "en")
        system_info = str(payload.get("systeminfo") or "")
        return AsrResult(
            text=text,
            language=language,
            duration=segments[-1].end if segments else None,
            segments=segments,
            engine="whisper.cpp",
            backend=self.backend,
            metadata={
                "system_info": system_info,
                "model_path": str(self.model_path),
                "available_backends": self.backends,
            },
        )


OPENVINO_MODELS = {
    "turbo": "OpenVINO/whisper-large-v3-turbo-int8-ov",
    "large-v3-turbo": "OpenVINO/whisper-large-v3-turbo-int8-ov",
    "large-v3": "OpenVINO/whisper-large-v3-int8-ov",
    "medium": "OpenVINO/whisper-medium-int8-ov",
    "small": "OpenVINO/whisper-small-int8-ov",
    "base": "OpenVINO/whisper-base-int8-ov",
    "tiny": "OpenVINO/whisper-tiny-int8-ov",
}


class OpenVinoWhisperAsr:
    """OpenVINO GenAI Whisper with bounded-memory long-recording chunking."""

    SAMPLE_RATE = 16_000

    def __init__(
        self,
        model_name: str,
        device: str = "AUTO",
        model_path: str | Path | None = None,
        huggingface_token: str | None = None,
        chunk_seconds: int = 300,
    ) -> None:
        try:
            import openvino_genai as ov_genai
        except ModuleNotFoundError as exc:
            raise AsrDependencyError(
                'OpenVINO transcription is not installed. Install with: pip install -e ".[openvino]"'
            ) from exc
        self.model_name = model_name
        self.device = self._normalize_device(device)
        self.chunk_seconds = max(30, int(chunk_seconds))
        configured = model_path or os.getenv("OPENVINO_WHISPER_MODEL_PATH")
        if configured and Path(configured).expanduser().is_dir():
            resolved_model = Path(configured).expanduser().resolve()
            model_id = str(resolved_model)
        else:
            try:
                model_id = OPENVINO_MODELS[model_name.strip().lower()]
            except KeyError as exc:
                raise ValueError(
                    f"The OpenVINO engine does not have a configured model mapping for {model_name!r}."
                ) from exc
            try:
                from huggingface_hub import snapshot_download
            except ModuleNotFoundError as exc:
                raise AsrDependencyError(
                    "huggingface-hub is required to download the OpenVINO Whisper model."
                ) from exc
            resolved_model = Path(
                snapshot_download(
                    repo_id=model_id,
                    token=huggingface_token
                    or os.getenv("HUGGINGFACE_TOKEN")
                    or os.getenv("HF_TOKEN"),
                )
            )
        constructor_args: dict[str, object] = {"word_timestamps": True}
        if self.device == "NPU":
            constructor_args["STATIC_PIPELINE"] = True
        self.model_id = model_id
        self.model_path = resolved_model
        self._ov_genai = ov_genai
        self._constructor_args = constructor_args
        self._pipeline = self._create_pipeline(self.device)
        self.backend = f"OpenVINO {self.device}"
        self._fallback_reason = ""

    @staticmethod
    def _normalize_device(value: str) -> str:
        normalized = (value or "AUTO").strip().upper().replace("OPENVINO-", "")
        if normalized in {"AUTO", "CPU", "GPU", "NPU"}:
            return normalized
        if normalized in {"CUDA", "VULKAN"}:
            return "AUTO"
        raise ValueError("OpenVINO device must be AUTO, CPU, GPU, or NPU.")

    def _create_pipeline(self, device: str):
        return self._ov_genai.WhisperPipeline(
            str(self.model_path), device, **self._constructor_args
        )

    def transcribe(
        self,
        audio_path: str | Path,
        progress: Callable[[str], None] | None = None,
    ) -> AsrResult:
        words: list[AsrWord] = []
        segments: list[AsrSegment] = []
        text_parts: list[str] = []
        total_samples = 0
        for index, samples in enumerate(self._audio_chunks(Path(audio_path)), start=1):
            offset = total_samples / self.SAMPLE_RATE
            if progress:
                progress(
                    f"OpenVINO {self.device}: transcribing audio at {offset / 3600:.1f} hours"
                )
            generation_args = {
                "language": "<|en|>",
                "task": "transcribe",
                "return_timestamps": True,
                "word_timestamps": True,
                "hotwords": "police fire EMS dispatch unit vehicle address county sheriff",
            }
            try:
                result = self._pipeline.generate(samples, **generation_args)
            except RuntimeError as exc:
                if self.device == "CPU":
                    raise
                failed_device = self.device
                self._fallback_reason = str(exc).splitlines()[0][:500]
                if progress:
                    progress(
                        f"OpenVINO {failed_device} rejected this model; retrying on CPU"
                    )
                self.device = "CPU"
                self.backend = f"OpenVINO CPU (fallback from {failed_device})"
                cpu_args = {"word_timestamps": True}
                self._constructor_args = cpu_args
                self._pipeline = self._create_pipeline("CPU")
                result = self._pipeline.generate(samples, **generation_args)
            result_text = self._result_text(result)
            if result_text:
                text_parts.append(result_text)
            for value in getattr(result, "words", None) or []:
                text = str(getattr(value, "word", ""))
                if text:
                    words.append(
                        AsrWord(
                            offset + float(getattr(value, "start_ts", 0.0)),
                            offset + float(getattr(value, "end_ts", 0.0)),
                            text,
                        )
                    )
            for value in getattr(result, "chunks", None) or []:
                text = str(getattr(value, "text", "")).strip()
                if text:
                    segments.append(
                        AsrSegment(
                            offset + float(getattr(value, "start_ts", 0.0)),
                            offset + float(getattr(value, "end_ts", 0.0)),
                            text,
                        )
                    )
            total_samples += len(samples)
            if progress:
                progress(f"OpenVINO completed audio chunk {index}")
        return AsrResult(
            text=" ".join(text_parts).strip(),
            language="en",
            duration=total_samples / self.SAMPLE_RATE if total_samples else None,
            words=words,
            segments=segments,
            engine="openvino",
            backend=getattr(self, "backend", f"OpenVINO {self.device}"),
            metadata={
                "model_id": self.model_id,
                "model_path": str(self.model_path),
                "fallback_reason": getattr(self, "_fallback_reason", ""),
            },
        )

    @staticmethod
    def _result_text(result: object) -> str:
        texts = getattr(result, "texts", None)
        if texts:
            return str(texts[0]).strip()
        text = getattr(result, "text", None)
        if text:
            return str(text).strip()
        value = str(result).strip()
        return value if value and not value.startswith("<") else ""

    def _audio_chunks(self, audio_path: Path) -> Iterator[list[float]]:
        try:
            import numpy as np
        except ModuleNotFoundError as exc:
            raise AsrDependencyError("NumPy is required for OpenVINO audio decoding.") from exc
        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            raise AsrDependencyError("FFmpeg is required for OpenVINO transcription.")
        bytes_per_chunk = self.SAMPLE_RATE * self.chunk_seconds * 2
        with tempfile.TemporaryFile() as error_log:
            process = subprocess.Popen(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(audio_path),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    str(self.SAMPLE_RATE),
                    "-c:a",
                    "pcm_s16le",
                    "-f",
                    "s16le",
                    "pipe:1",
                ],
                stdout=subprocess.PIPE,
                stderr=error_log,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            assert process.stdout is not None
            while True:
                block = self._read_block(process.stdout, bytes_per_chunk)
                if not block:
                    break
                values = np.frombuffer(block, dtype="<i2").astype("float32") / 32768.0
                yield values.tolist()
            return_code = process.wait()
            if return_code != 0:
                error_log.seek(0)
                detail = error_log.read().decode("utf-8", errors="replace").strip()
                raise RuntimeError(detail or f"FFmpeg exited with code {return_code}.")

    @staticmethod
    def _read_block(stream: Any, size: int) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        while remaining > 0:
            value = stream.read(remaining)
            if not value:
                break
            chunks.append(value)
            remaining -= len(value)
        return b"".join(chunks)
