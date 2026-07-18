from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from .accelerators import (
    find_container_runtime,
    find_whisper_cpp,
    find_windows_ml_helper,
    whisper_cpp_container_diagnostics,
    whisper_cpp_backends,
)
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


WHISPER_MODEL_ALIASES = {
    "large-v3-turbo": "turbo",
}


def normalize_whisper_model_name(model_name: str) -> str:
    """Return one stable identity across the supported Whisper runtimes."""

    normalized = str(model_name or "").strip().lower().replace("_", "-")
    if normalized.endswith(".en"):
        normalized = normalized[:-3]
    return WHISPER_MODEL_ALIASES.get(normalized, normalized)


def normalize_asr_engine(engine: str, device: str) -> str:
    normalized = (engine or "auto").strip().lower().replace("_", "-")
    if normalized == "auto":
        device_name = (device or "auto").strip().lower()
        if device_name == "vulkan":
            return "whisper.cpp"
        if device_name == "metal":
            return "whisper.cpp"
        if device_name == "auto" and sys.platform == "darwin":
            executable = find_whisper_cpp()
            if executable and "metal" in whisper_cpp_backends(executable):
                return "whisper.cpp"
        if device_name == "auto" and sys.platform.startswith("linux"):
            executable = find_whisper_cpp()
            if executable and "vulkan" in whisper_cpp_backends(executable):
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
        "qwen": "qwen3-asr",
        "qwen3": "qwen3-asr",
        "qwen3asr": "qwen3-asr",
        "sherpa-qwen3": "qwen3-asr",
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
    normalized = normalize_whisper_model_name(model_name)
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
    filename = whisper_cpp_model_filename(model_name)
    configured = explicit_path or os.getenv("WHISPER_CPP_MODEL_PATH")
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file():
            if candidate.name.lower() != filename.lower():
                raise AsrDependencyError(
                    f"The selected Whisper model expects {filename}, but the configured "
                    f"path points to {candidate.name}. Select the matching model or clear "
                    "the custom path."
                )
            return candidate.resolve()
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
    """Runs native or explicitly configured containerized whisper.cpp."""

    def __init__(
        self,
        model_name: str,
        device: str = "vulkan",
        device_index: int = 0,
        executable: str | Path | None = None,
        model_path: str | Path | None = None,
        container_image: str | None = None,
        container_runtime: str | Path | None = None,
        container_device: str | Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = (device or "vulkan").lower()
        self.device_index = max(0, int(device_index))
        self.container_image = str(
            container_image or os.getenv("WHISPER_CPP_CONTAINER_IMAGE") or ""
        ).strip()
        self.executable = str(
            executable
            or (find_whisper_cpp() if not self.container_image else "")
            or ""
        )
        self.container_runtime = find_container_runtime(container_runtime)
        self.container_backend = str(
            os.getenv("WHISPER_CPP_CONTAINER_BACKEND") or "vulkan"
        ).strip().lower()
        self.container_device = str(
            container_device
            or os.getenv("WHISPER_CPP_CONTAINER_DEVICE")
            or "/dev/dri"
        )
        self.container_executable = str(
            os.getenv("WHISPER_CPP_CONTAINER_EXECUTABLE")
            or "/app/build/bin/whisper-cli"
        ).strip()
        self.containerized = bool(
            (not self.executable or not Path(self.executable).is_file())
            and self.container_image
        )
        if self.containerized:
            if self.device == "metal":
                raise AsrDependencyError(
                    "Metal transcription requires a native macOS whisper.cpp build; "
                    "Docker and Podman do not expose Apple Metal to this adapter."
                )
            container = whisper_cpp_container_diagnostics(
                image=self.container_image,
                runtime=self.container_runtime,
                backend=self.container_backend,
                device=self.container_device,
            )
            if not container["runtime"]:
                raise AsrDependencyError(
                    "A whisper.cpp container image is configured, but Docker or Podman "
                    "was not found. Set BROADCASTIFY_CONTAINER_RUNTIME."
                )
            if not container["image_present"]:
                raise AsrDependencyError(
                    f"The configured whisper.cpp image {self.container_image!r} is not "
                    "present locally. Pull and verify it explicitly before processing."
                )
            if self.device == "vulkan" and not container["device_present"]:
                raise AsrDependencyError(
                    f"The Vulkan device {self.container_device!r} is unavailable to the "
                    "container runtime."
                )
        elif not self.executable or not Path(self.executable).is_file():
            if self.device == "metal":
                raise AsrDependencyError(
                    "whisper-cli was not found. Apple Metal requires a native macOS "
                    "whisper.cpp build compiled with GGML_METAL=ON; Docker and Podman "
                    "cannot expose Metal to this adapter. Set WHISPER_CPP_PATH after "
                    "building the native executable."
                )
            if self.device == "vulkan":
                raise AsrDependencyError(
                    "whisper-cli was not found. Install or build whisper.cpp with "
                    "GGML_VULKAN=1 and set WHISPER_CPP_PATH, or explicitly configure "
                    "a pre-pulled WHISPER_CPP_CONTAINER_IMAGE on Linux."
                )
            raise AsrDependencyError(
                "whisper-cli was not found. Install or build native whisper.cpp and "
                "set WHISPER_CPP_PATH to whisper-cli."
            )
        resolved_model = find_whisper_cpp_model(model_name, model_path)
        if resolved_model is None and self.executable:
            adjacent_model = (
                Path(self.executable).resolve().parent
                / "models"
                / whisper_cpp_model_filename(model_name)
            )
            if adjacent_model.is_file():
                resolved_model = adjacent_model
        if resolved_model is None:
            expected = whisper_cpp_model_filename(model_name)
            raise AsrDependencyError(
                f"The whisper.cpp model {expected} was not found. Put it under "
                f"{_default_model_root() / 'whisper.cpp'} or set WHISPER_CPP_MODEL_PATH."
            )
        self.model_path = resolved_model
        self.backends = (
            sorted({"cpu", self.container_backend})
            if self.containerized
            else whisper_cpp_backends(self.executable)
        )
        if self.device == "vulkan" and "vulkan" not in self.backends:
            raise AsrDependencyError(
                "The selected whisper.cpp executable does not expose ggml-vulkan. "
                "Use a build compiled with GGML_VULKAN=1 or select CPU."
            )
        if self.device == "metal" and "metal" not in self.backends:
            raise AsrDependencyError(
                "The selected whisper.cpp executable does not expose ggml-metal. "
                "Use a native macOS build with Metal enabled or select CPU."
            )
        selected_backend = (
            self.device if self.device != "auto" else ",".join(self.backends)
        )
        self.backend = (
            f"{selected_backend} (container)"
            if self.containerized
            else selected_backend
        )

    @staticmethod
    def _mount(source: Path, destination: str, *, readonly: bool = False) -> str:
        resolved = str(source.resolve())
        if "," in resolved:
            raise AsrDependencyError(
                "Containerized whisper.cpp cannot bind a path containing a comma."
            )
        value = f"type=bind,src={resolved},dst={destination}"
        return value + (",readonly" if readonly else "")

    def _container_group_id(self) -> int | None:
        configured = os.getenv("WHISPER_CPP_CONTAINER_GROUP_ID", "").strip()
        if configured:
            try:
                return int(configured)
            except ValueError as exc:
                raise AsrDependencyError(
                    "WHISPER_CPP_CONTAINER_GROUP_ID must be a numeric group ID."
                ) from exc
        device = Path(self.container_device)
        candidates = sorted(device.glob("renderD*")) if device.is_dir() else [device]
        for candidate in candidates:
            try:
                return candidate.stat().st_gid
            except OSError:
                continue
        return None

    def _container_arguments(
        self, source: Path, output_directory: Path
    ) -> tuple[list[str], str, str, str]:
        assert self.container_runtime is not None
        arguments = [
            self.container_runtime,
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--security-opt",
            "no-new-privileges",
            "--cap-drop",
            "ALL",
            "--env",
            "HOME=/tmp",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=512m",
        ]
        if hasattr(os, "getuid") and hasattr(os, "getgid"):
            arguments.extend(["--user", f"{os.getuid()}:{os.getgid()}"])
        if self.device == "vulkan":
            arguments.extend(["--device", self.container_device])
            group_id = self._container_group_id()
            if group_id is not None:
                arguments.extend(["--group-add", str(group_id)])
        arguments.extend(
            [
                "--mount",
                self._mount(source, "/input/audio.wav", readonly=True),
                "--mount",
                self._mount(self.model_path, "/models/model.bin", readonly=True),
                "--mount",
                self._mount(output_directory, "/output"),
                "--entrypoint",
                self.container_executable,
                self.container_image,
            ]
        )
        return arguments, "/input/audio.wav", "/models/model.bin", "/output/result"

    def _prepare_audio(
        self,
        source: Path,
        cache_dir: Path,
        progress: Callable[[str], None] | None,
    ) -> tuple[Path, bool]:
        if source.suffix.lower() == ".wav":
            return source, False
        prepared = cache_dir / f"{source.stem}.whisper.cpp.wav"
        partial = prepared.with_suffix(".part.wav")
        partial.unlink(missing_ok=True)
        if (
            prepared.is_file()
            and prepared.stat().st_size > 0
            and prepared.stat().st_mtime_ns >= source.stat().st_mtime_ns
        ):
            if progress:
                progress(f"Reusing prepared whisper.cpp audio for {source.name}")
            return prepared, True
        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            raise AsrDependencyError(
                "FFmpeg is required to prepare 16 kHz WAV input for whisper.cpp."
            )
        if progress:
            progress(f"Preparing 16 kHz audio for whisper.cpp: {source.name}")
        converted = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                "-y",
                str(partial),
            ],
            capture_output=True,
            text=True,
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if converted.returncode != 0 or not partial.is_file() or partial.stat().st_size == 0:
            partial.unlink(missing_ok=True)
            raise RuntimeError(
                converted.stderr.strip()
                or "FFmpeg could not prepare whisper.cpp audio."
            )
        partial.replace(prepared)
        return prepared, True

    def transcribe(
        self,
        audio_path: str | Path,
        progress: Callable[[str], None] | None = None,
    ) -> AsrResult:
        source = Path(audio_path)
        cache_dir = source.parent / "transcripts" / ".cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        prepared_source, remove_prepared = self._prepare_audio(
            source, cache_dir, progress
        )
        completed = False
        runtime_evidence: list[str] = []
        try:
            with tempfile.TemporaryDirectory(prefix="whisper-cpp-", dir=cache_dir) as temporary:
                output_directory = Path(temporary)
                if self.containerized:
                    arguments, input_path, model_path, output_path = (
                        self._container_arguments(prepared_source, output_directory)
                    )
                    output_base = output_directory / "result"
                else:
                    arguments = [self.executable]
                    input_path = str(prepared_source)
                    model_path = str(self.model_path)
                    output_base = output_directory / source.stem
                    output_path = str(output_base)
                arguments.extend(
                    [
                        "--model",
                        model_path,
                        "--file",
                        input_path,
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
                        output_path,
                        "--print-progress",
                        "--device",
                        str(self.device_index),
                    ]
                )
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
                        percent = min(100, max(0, int(match.group(1))))
                        progress(f"whisper.cpp transcription: {percent}%")
                return_code = process.wait()
                json_path = output_base.with_suffix(".json")
                if return_code != 0 or not json_path.is_file():
                    detail = "\n".join(error_lines[-20:])
                    raise RuntimeError(
                        detail or f"whisper.cpp exited with code {return_code}."
                    )
                runtime_evidence = [
                    line
                    for line in error_lines
                    if re.search(
                        r"(?:vulkan|cuda|sycl|openvino|metal|device|backend)",
                        line,
                        re.IGNORECASE,
                    )
                ][-40:]
                payload = json.loads(json_path.read_text(encoding="utf-8"))
                completed = True
        finally:
            if completed and remove_prepared:
                prepared_source.unlink(missing_ok=True)

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
                "model": normalize_whisper_model_name(self.model_name),
                "requested_model": self.model_name,
                "system_info": system_info,
                "model_path": str(self.model_path),
                "available_backends": self.backends,
                "container_image": self.container_image if self.containerized else "",
                "container_runtime": (
                    self.container_runtime if self.containerized else ""
                ),
                "runtime_evidence": runtime_evidence,
            },
        )


WINDOWS_ML_MODEL_MANIFEST = "broadcastify-model.json"
WINDOWS_ML_MODEL_SOURCES = {
    "tiny": "openai/whisper-tiny",
    "base": "openai/whisper-base",
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large-v3": "openai/whisper-large-v3",
    "turbo": "openai/whisper-large-v3-turbo",
    "distil-large-v3": "distil-whisper/distil-large-v3",
}
WINDOWS_ML_ARCHITECTURES = {
    (384, 4, 4): "tiny",
    (512, 6, 6): "base",
    (768, 12, 12): "small",
    (1024, 24, 24): "medium",
    (1280, 32, 32): "large-v3",
    (1280, 32, 4): "turbo",
    (1280, 32, 2): "distil-large-v3",
}


@dataclass(frozen=True)
class WindowsMlModelInfo:
    path: Path
    model: str
    source_model: str
    provider: str
    precision: str
    managed: bool


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _windows_ml_identity_from_directory(path: Path) -> str:
    names = (
        "distil-large-v3",
        "large-v3-turbo",
        "large-v3",
        "turbo",
        "medium",
        "small",
        "base",
        "tiny",
    )
    lowered = path.name.lower().replace("_", "-")
    for name in names:
        if re.search(rf"(?:^|-)({re.escape(name)})(?:-|$)", lowered):
            return normalize_whisper_model_name(name)
    return ""


def _windows_ml_identity_from_config(config: dict[str, Any]) -> str:
    model = config.get("model")
    if not isinstance(model, dict):
        return ""
    encoder = model.get("encoder")
    decoder = model.get("decoder")
    if not isinstance(encoder, dict) or not isinstance(decoder, dict):
        return ""
    try:
        signature = (
            int(encoder["hidden_size"]),
            int(encoder["num_hidden_layers"]),
            int(decoder["num_hidden_layers"]),
        )
    except (KeyError, TypeError, ValueError):
        return ""
    return WINDOWS_ML_ARCHITECTURES.get(signature, "")


def _windows_ml_provider(config: dict[str, Any]) -> str:
    model = config.get("model")
    if not isinstance(model, dict):
        return "unknown"
    providers: set[str] = set()
    for component_name in ("encoder", "decoder"):
        component = model.get(component_name)
        if not isinstance(component, dict):
            continue
        session = component.get("session_options")
        if not isinstance(session, dict):
            continue
        options = session.get("provider_options")
        if not isinstance(options, list):
            continue
        for value in options:
            if isinstance(value, dict):
                providers.update(str(key).lower() for key in value)
    if not providers:
        return "cpu"
    aliases = {"dml": "directml", "webgpu": "webgpu"}
    return "+".join(sorted(aliases.get(value, value) for value in providers))


def windows_ml_model_info(path: str | Path) -> WindowsMlModelInfo | None:
    candidate = Path(path).expanduser()
    config_path = candidate / "genai_config.json"
    if not config_path.is_file():
        return None
    config = _read_json_object(config_path)
    manifest = _read_json_object(candidate / WINDOWS_ML_MODEL_MANIFEST)
    manifest_identity = normalize_whisper_model_name(str(manifest.get("model") or ""))
    config_identity = _windows_ml_identity_from_config(config)
    directory_identity = _windows_ml_identity_from_directory(candidate)
    if manifest_identity and config_identity and manifest_identity != config_identity:
        raise AsrDependencyError(
            f"Windows ML model metadata in {candidate} says {manifest_identity}, "
            f"but the ONNX graph dimensions identify {config_identity}. Rebuild this "
            "managed model before using it."
        )
    identity = config_identity or manifest_identity or directory_identity
    precision = str(manifest.get("precision") or "").strip().lower()
    if not precision:
        match = re.search(r"(?:^|-)(int4|bf16|fp16|fp32)(?:-|$)", candidate.name.lower())
        precision = match.group(1) if match else "unknown"
    source_model = str(manifest.get("source_model") or "").strip()
    return WindowsMlModelInfo(
        path=candidate.resolve(),
        model=identity,
        source_model=source_model,
        provider=str(manifest.get("provider") or "").strip().lower()
        or _windows_ml_provider(config),
        precision=precision,
        managed=bool(manifest),
    )


def _windows_ml_model_roots() -> list[Path]:
    return [
        _default_model_root() / "windowsml",
        Path.cwd() / "models" / "windowsml",
        Path.cwd() / ".models" / "windowsml",
    ]


def _windows_ml_model_candidates() -> Iterator[WindowsMlModelInfo]:
    seen: set[Path] = set()
    for root in _windows_ml_model_roots():
        paths = [root]
        try:
            paths.extend(
                path.parent
                for path in root.glob("*/genai_config.json")
                if path.is_file()
            )
        except OSError:
            continue
        for path in paths:
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            info = windows_ml_model_info(resolved)
            if info is not None:
                yield info


def _windows_ml_model_score(info: WindowsMlModelInfo) -> tuple[int, int, int, str]:
    # CPU FP32 is the currently timed, dependable Windows ML path. Prefer it
    # over an unverified provider graph while preserving deterministic choice.
    return (
        1 if info.provider == "cpu" else 0,
        1 if info.precision == "fp32" else 0,
        1 if info.managed else 0,
        str(info.path).lower(),
    )


def find_windows_ml_model(
    model_name: str, explicit_path: str | Path | None = None
) -> Path | None:
    configured = explicit_path or os.getenv("WINDOWS_ML_WHISPER_MODEL_PATH")
    requested = normalize_whisper_model_name(model_name)
    if configured:
        candidate = Path(configured).expanduser()
        info = windows_ml_model_info(candidate)
        if info is None:
            raise AsrDependencyError(
                f"The configured Windows ML model path {candidate} does not contain "
                "genai_config.json."
            )
        if not info.model:
            raise AsrDependencyError(
                f"The Windows ML model at {candidate} has no verifiable Whisper model "
                f"identity. Rebuild it with this app or add {WINDOWS_ML_MODEL_MANIFEST}."
            )
        if info.model != requested:
            raise AsrDependencyError(
                f"The selected Whisper model is {requested}, but {candidate} contains "
                f"{info.model}. Select {info.model}, clear the custom model path, or "
                "prepare the selected model."
            )
        return info.path
    compatible = [
        info for info in _windows_ml_model_candidates() if info.model == requested
    ]
    if compatible:
        return max(compatible, key=_windows_ml_model_score).path
    return None


class WindowsMlWhisperAsr:
    """Runs the Windows ML ONNX Runtime GenAI helper once for many short chunks."""

    SAMPLE_RATE = 16_000

    def __init__(
        self,
        model_name: str,
        model_path: str | Path | None = None,
        helper_path: str | Path | None = None,
        chunk_seconds: int = 28,
    ) -> None:
        if os.name != "nt":
            raise AsrDependencyError("Windows ML transcription is available only on Windows.")
        self.requested_model_name = model_name
        self.model_name = normalize_whisper_model_name(model_name)
        self.helper = str(helper_path or find_windows_ml_helper() or "")
        if not self.helper or not Path(self.helper).is_file():
            raise AsrDependencyError(
                "The Windows ML helper was not found. Build BroadcastifyCli.WindowsML "
                "or set WINDOWS_ML_HELPER_PATH."
            )
        resolved_model = find_windows_ml_model(model_name, model_path)
        if resolved_model is None:
            raise AsrDependencyError(
                f"A compatible ONNX Runtime GenAI Whisper {self.model_name} model was "
                "not found. Use Prepare model or set WINDOWS_ML_WHISPER_MODEL_PATH to "
                "a matching directory containing genai_config.json."
            )
        self.model_path = resolved_model
        model_info = windows_ml_model_info(resolved_model)
        assert model_info is not None
        self.model_name = model_info.model
        self.model_source = model_info.source_model
        self.model_provider = model_info.provider
        self.model_precision = model_info.precision
        self.chunk_seconds = min(29, max(5, int(chunk_seconds)))
        provider = self.model_provider.upper() if self.model_provider == "cpu" else self.model_provider
        self.backend = f"Windows ML (ONNX Runtime GenAI · {provider})"

    def transcribe(
        self,
        audio_path: str | Path,
        progress: Callable[[str], None] | None = None,
    ) -> AsrResult:
        source = Path(audio_path)
        if not source.is_file():
            raise FileNotFoundError(f"Audio does not exist: {source}")
        cache_dir = source.parent / "transcripts" / ".cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        segments: list[AsrSegment] = []
        text_parts: list[str] = []
        total_seconds = 0.0
        with tempfile.TemporaryFile() as error_log:
            process = subprocess.Popen(
                [self.helper, "--model", str(self.model_path), "--stream"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=error_log,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            assert process.stdin is not None
            assert process.stdout is not None
            try:
                for index, block in enumerate(self._audio_chunks(source), start=1):
                    start = total_seconds
                    duration = len(block) / (self.SAMPLE_RATE * 2)
                    end = start + duration
                    chunk_path = cache_dir / f"winml-{os.getpid()}-{index:06d}.wav"
                    try:
                        self._write_wave(chunk_path, block)
                        if progress:
                            progress(
                                f"Windows ML: transcribing audio at {start / 3600:.1f} hours"
                            )
                        process.stdin.write(
                            json.dumps(
                                {"path": str(chunk_path.resolve()), "start": start, "end": end}
                            )
                            + "\n"
                        )
                        process.stdin.flush()
                        response_line = process.stdout.readline()
                        if not response_line:
                            raise RuntimeError("Windows ML helper stopped before returning a chunk.")
                        response = json.loads(response_line)
                        if response.get("error"):
                            raise RuntimeError(str(response["error"]))
                        response_backend = str(response.get("backend") or "").strip()
                        if response_backend:
                            self.backend = response_backend
                        text = str(response.get("text") or "").strip()
                        if text:
                            text_parts.append(text)
                            segments.append(AsrSegment(start, end, text))
                    finally:
                        chunk_path.unlink(missing_ok=True)
                    total_seconds = end
                    if progress:
                        progress(f"Windows ML completed audio chunk {index}")
                process.stdin.write('{"command":"stop"}\n')
                process.stdin.flush()
                process.stdin.close()
                return_code = process.wait(timeout=30)
            except Exception:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=10)
                raise
            if return_code != 0:
                error_log.seek(0)
                detail = error_log.read().decode("utf-8", errors="replace").strip()
                raise RuntimeError(
                    detail or f"Windows ML helper exited with code {return_code}."
                )
        return AsrResult(
            text=" ".join(text_parts).strip(),
            language="en",
            duration=total_seconds or None,
            segments=segments,
            engine="windows-ml",
            backend=self.backend,
            metadata={
                "model": self.model_name,
                "requested_model": getattr(
                    self, "requested_model_name", self.model_name
                ),
                "source_model": getattr(self, "model_source", ""),
                "provider": getattr(self, "model_provider", ""),
                "precision": getattr(self, "model_precision", ""),
                "model_path": str(self.model_path),
                "helper_path": self.helper,
                "chunk_seconds": self.chunk_seconds,
            },
        )

    def _audio_chunks(self, audio_path: Path) -> Iterator[bytes]:
        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            raise AsrDependencyError("FFmpeg is required for Windows ML transcription.")
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
                block = OpenVinoWhisperAsr._read_block(process.stdout, bytes_per_chunk)
                if not block:
                    break
                yield block
            return_code = process.wait()
            if return_code != 0:
                error_log.seek(0)
                detail = error_log.read().decode("utf-8", errors="replace").strip()
                raise RuntimeError(detail or f"FFmpeg exited with code {return_code}.")

    @classmethod
    def _write_wave(cls, path: Path, block: bytes) -> None:
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(cls.SAMPLE_RATE)
            output.writeframes(block)


def _model_directory_size(path: Path) -> int:
    total = 0
    try:
        for candidate in path.rglob("*"):
            if candidate.is_file():
                total += candidate.stat().st_size
    except OSError:
        return total
    return total


def _onnxruntime_genai_version() -> str:
    try:
        return importlib_metadata.version("onnxruntime-genai")
    except importlib_metadata.PackageNotFoundError:
        return ""


def prepare_windows_ml_model(
    model_name: str,
    *,
    explicit_path: str | Path | None = None,
    huggingface_token: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Build a selected CPU Whisper graph after an explicit user action."""

    requested = normalize_whisper_model_name(model_name)
    try:
        source_model = WINDOWS_ML_MODEL_SOURCES[requested]
    except KeyError as exc:
        raise ValueError(
            f"Windows ML model preparation does not support {model_name!r}."
        ) from exc

    if explicit_path:
        explicit = Path(explicit_path).expanduser()
        if explicit.exists():
            resolved = find_windows_ml_model(requested, explicit)
            assert resolved is not None
            info = windows_ml_model_info(resolved)
            assert info is not None
            return {
                "ready": True,
                "engine": "windows-ml",
                "model": info.model,
                "source_model": info.source_model or source_model,
                "provider": info.provider,
                "precision": info.precision,
                "path": str(info.path),
                "reused": True,
                "bytes": _model_directory_size(info.path),
                "message": (
                    f"Reusing the matching Windows ML {info.model} model at {info.path}."
                ),
            }

    compatible = [
        info for info in _windows_ml_model_candidates() if info.model == requested
    ]
    if compatible:
        info = max(compatible, key=_windows_ml_model_score)
        return {
            "ready": True,
            "engine": "windows-ml",
            "model": info.model,
            "source_model": info.source_model or source_model,
            "provider": info.provider,
            "precision": info.precision,
            "path": str(info.path),
            "reused": True,
            "bytes": _model_directory_size(info.path),
            "message": (
                f"Reusing the matching Windows ML {info.model} model at {info.path}."
            ),
        }

    try:
        import onnxruntime_genai.models.builder  # noqa: F401
    except ModuleNotFoundError as exc:
        raise AsrDependencyError(
            'Windows ML model preparation needs the optional builder. Install it with '
            '`pip install -e ".[windowsml]"` and try again.'
        ) from exc

    target_root = _default_model_root() / "windowsml"
    target = target_root / f"whisper-{requested}-fp32-cpu"
    if target.exists():
        raise AsrDependencyError(
            f"The managed target {target} already exists but is not a compatible, "
            "complete model. Move it aside and run Prepare model again."
        )
    target_root.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".whisper-{requested}-build-", dir=target_root)
    )
    staging = staging_root / "model"
    cache = _default_model_root() / "downloads"
    cache.mkdir(parents=True, exist_ok=True)
    if progress:
        progress(
            f"Building {source_model} as a Windows ML CPU FP32 graph. "
            "This explicit first-time step may download several files."
        )
    environment = os.environ.copy()
    if huggingface_token:
        environment["HF_TOKEN"] = huggingface_token
    arguments = [
        sys.executable,
        "-m",
        "onnxruntime_genai.models.builder",
        "-m",
        source_model,
        "-o",
        str(staging),
        "-p",
        "fp32",
        "-e",
        "cpu",
        "-c",
        str(cache),
    ]
    if not huggingface_token:
        # ORT GenAI's builder currently defaults its hf_token option to True,
        # which rejects even public models when no saved login exists.
        arguments.extend(["--extra_options", "hf_token=false"])
    lines: list[str] = []
    last_percent = -10
    last_phase = ""
    try:
        process = subprocess.Popen(
            arguments,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=environment,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", raw_line).strip()
            if not line:
                continue
            if huggingface_token:
                line = line.replace(huggingface_token, "[redacted]")
            lines.append(line)
            if progress:
                percentage = re.search(r"\b(\d{1,3})%", line)
                if percentage:
                    value = min(100, int(percentage.group(1)))
                    if value >= last_percent + 10 or (
                        value == 100 and last_percent < 100
                    ):
                        last_percent = value
                        progress(f"Windows ML model export: {value}%")
                else:
                    phase = next(
                        (
                            marker
                            for marker in (
                                "Downloading",
                                "Loading",
                                "Saving ONNX model",
                                "Saving processing files",
                            )
                            if marker.lower() in line.lower()
                        ),
                        "",
                    )
                    if phase and phase != last_phase:
                        last_phase = phase
                        progress(line[:500])
        return_code = process.wait()
        info = windows_ml_model_info(staging)
        if return_code != 0 or info is None:
            detail = "\n".join(lines[-12:])
            raise RuntimeError(
                detail
                or f"Windows ML model builder exited with code {return_code}."
            )
        if info.model and info.model != requested:
            raise RuntimeError(
                f"The builder returned {info.model}, not the requested {requested} graph."
            )
        manifest = {
            "schema_version": 1,
            "engine": "windows-ml",
            "model": requested,
            "source_model": source_model,
            "provider": "cpu",
            "precision": "fp32",
            "builder": "onnxruntime-genai",
            "builder_version": _onnxruntime_genai_version(),
        }
        (staging / WINDOWS_ML_MODEL_MANIFEST).write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        staging.replace(target)
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    shutil.rmtree(staging_root, ignore_errors=True)
    info = windows_ml_model_info(target)
    if info is None or info.model != requested:
        raise RuntimeError("The prepared Windows ML model failed final identity validation.")
    return {
        "ready": True,
        "engine": "windows-ml",
        "model": info.model,
        "source_model": info.source_model,
        "provider": info.provider,
        "precision": info.precision,
        "path": str(info.path),
        "reused": False,
        "bytes": _model_directory_size(info.path),
        "message": (
            f"Prepared Windows ML {info.model} as {info.precision.upper()} on "
            f"{info.provider.upper()} at {info.path}."
        ),
    }


def prepare_whisper_cpp_model(
    model_name: str,
    *,
    explicit_path: str | Path | None = None,
    huggingface_token: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Download one official public GGML file after an explicit user action."""

    requested = normalize_whisper_model_name(model_name)
    filename = whisper_cpp_model_filename(requested)
    existing = find_whisper_cpp_model(requested, explicit_path)
    if existing:
        return {
            "ready": True,
            "engine": "whisper.cpp",
            "model": requested,
            "source_model": "ggerganov/whisper.cpp",
            "provider": "local",
            "precision": "ggml-quantized",
            "path": str(existing),
            "reused": True,
            "bytes": existing.stat().st_size,
            "message": f"Reusing the matching whisper.cpp model at {existing}.",
        }
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise AsrDependencyError(
            'Managed whisper.cpp downloads need huggingface-hub. Install `.[openvino]` '
            "or `.[windowsml]`, then try again."
        ) from exc
    target_root = _default_model_root() / "whisper.cpp"
    target_root.mkdir(parents=True, exist_ok=True)
    if progress:
        progress(
            f"Downloading public whisper.cpp model {filename} from "
            "ggerganov/whisper.cpp. The runtime binary is kept separate."
        )
    downloaded = Path(
        hf_hub_download(
            repo_id="ggerganov/whisper.cpp",
            filename=filename,
            local_dir=target_root,
            token=huggingface_token or None,
        )
    )
    target = target_root / filename
    if not target.is_file() and downloaded.is_file():
        target = downloaded
    if not target.is_file() or target.stat().st_size <= 0:
        raise RuntimeError("The whisper.cpp model download did not produce a usable file.")
    return {
        "ready": True,
        "engine": "whisper.cpp",
        "model": requested,
        "source_model": "ggerganov/whisper.cpp",
        "provider": "local",
        "precision": "ggml-quantized",
        "path": str(target.resolve()),
        "reused": False,
        "bytes": target.stat().st_size,
        "message": f"Downloaded whisper.cpp {requested} model to {target.resolve()}.",
    }


def prepare_asr_model(
    settings: dict[str, Any],
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    model = str(settings.get("model") or "turbo")
    engine = normalize_asr_engine(
        str(settings.get("asr_engine") or "auto"),
        str(settings.get("device") or "auto"),
    )
    options = {
        "explicit_path": settings.get("asr_model_path") or None,
        "huggingface_token": str(settings.get("huggingface_token") or "") or None,
        "progress": progress,
    }
    if engine == "windows-ml":
        return prepare_windows_ml_model(model, **options)
    if engine == "whisper.cpp":
        return prepare_whisper_cpp_model(model, **options)
    if engine == "qwen3-asr":
        from .qwen_asr import prepare_qwen3_asr_model

        return prepare_qwen3_asr_model(
            model,
            explicit_path=options["explicit_path"],
            progress=progress,
        )
    raise ValueError(
        f"{engine} manages its selected model during Test engine; "
        "a separate Prepare model step is not required."
    )


OPENVINO_MODELS = {
    "turbo": "OpenVINO/whisper-large-v3-turbo-int8-ov",
    "large-v3-turbo": "OpenVINO/whisper-large-v3-turbo-int8-ov",
    "distil-large-v3": "OpenVINO/distil-whisper-large-v3-int8-ov",
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
                model_id = OPENVINO_MODELS[
                    model_name.strip().lower().removesuffix(".en")
                ]
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
        self.model_id = model_id
        self.model_path = resolved_model
        self._ov_genai = ov_genai
        self._constructor_args = constructor_args
        self.requested_device = self.device
        self._fallback_reason = ""
        self._fallback_stage = ""
        try:
            self._pipeline = self._create_pipeline(self.device)
            self.backend = f"OpenVINO {self.device}"
        except RuntimeError as exc:
            if self.device == "CPU":
                raise
            failed_device = self.device
            self._fallback_reason = str(exc).splitlines()[0][:500]
            self._fallback_stage = "initialization"
            self.device = "CPU"
            self._constructor_args = {"word_timestamps": True}
            self._pipeline = self._create_pipeline("CPU")
            self.backend = f"OpenVINO CPU (fallback from {failed_device})"

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
                self._fallback_stage = "generation"
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
                "model": normalize_whisper_model_name(
                    getattr(self, "model_name", self.model_id)
                ),
                "requested_model": getattr(self, "model_name", self.model_id),
                "model_id": self.model_id,
                "model_path": str(self.model_path),
                "requested_device": getattr(self, "requested_device", self.device),
                "fallback_reason": getattr(self, "_fallback_reason", ""),
                "fallback_stage": getattr(self, "_fallback_stage", ""),
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
