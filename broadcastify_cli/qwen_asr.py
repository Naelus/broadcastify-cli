from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Iterator, Sequence

from .asr import AsrDependencyError, AsrResult, AsrSegment, _default_model_root
from .audio import find_ffmpeg


QWEN3_ASR_ENGINE = "qwen3-asr"
QWEN3_ASR_MODEL = "qwen3-asr-0.6b-int8"
QWEN3_ASR_MODEL_DIRECTORY = "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25"
QWEN3_ASR_ARCHIVE = f"{QWEN3_ASR_MODEL_DIRECTORY}.tar.bz2"
QWEN3_ASR_ARCHIVE_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    f"{QWEN3_ASR_ARCHIVE}"
)
QWEN3_ASR_ARCHIVE_BYTES = 878_702_423
QWEN3_ASR_ARCHIVE_SHA256 = (
    "393f8a14e2f5fb96746aaab342997a40641001fbd5bf9592a080a8329178ee96"
)
SILERO_VAD_FILENAME = "silero_vad.onnx"
SILERO_VAD_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    f"{SILERO_VAD_FILENAME}"
)
SILERO_VAD_BYTES = 643_854
SILERO_VAD_SHA256 = (
    "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6"
)
QWEN3_ASR_MANIFEST = "broadcastify-model.json"
QWEN3_ASR_MIN_HINT_SECONDS = 0.75
QWEN3_ASR_REQUIRED_FILES = (
    "conv_frontend.onnx",
    "encoder.int8.onnx",
    "decoder.int8.onnx",
    "tokenizer/merges.txt",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
)
QWEN3_ASR_ALIASES = {
    "qwen3-asr": QWEN3_ASR_MODEL,
    "qwen3-asr-0.6b": QWEN3_ASR_MODEL,
    "qwen3-0.6b-int8": QWEN3_ASR_MODEL,
    "qwen3-asr-0.6b-int8": QWEN3_ASR_MODEL,
}


@dataclass(frozen=True)
class Qwen3AsrModelInfo:
    path: Path
    vad_path: Path | None
    model: str = QWEN3_ASR_MODEL
    source_model: str = "Qwen/Qwen3-ASR-0.6B"
    provider: str = "cpu"
    precision: str = "int8"


@dataclass(frozen=True)
class _SpeechSlice:
    start: float
    end: float
    samples: Any


def normalize_qwen3_asr_model_name(model_name: str) -> str:
    normalized = str(model_name or "").strip().lower().replace("_", "-")
    try:
        return QWEN3_ASR_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Qwen3-ASR supports {QWEN3_ASR_MODEL!r}; got {model_name!r}."
        ) from exc


def _qwen3_model_files_ready(path: Path) -> bool:
    try:
        return path.is_dir() and all(
            (path / relative).is_file() and (path / relative).stat().st_size > 0
            for relative in QWEN3_ASR_REQUIRED_FILES
        )
    except OSError:
        return False


def _managed_manifest_matches(path: Path) -> bool:
    manifest_path = path / QWEN3_ASR_MANIFEST
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return all(
        str(value.get(key) or "") == expected
        for key, expected in {
            "engine": QWEN3_ASR_ENGINE,
            "model": QWEN3_ASR_MODEL,
            "source_model": "Qwen/Qwen3-ASR-0.6B",
            "runtime": "sherpa-onnx",
            "provider": "cpu",
            "precision": "int8",
            "archive_sha256": QWEN3_ASR_ARCHIVE_SHA256,
            "vad_sha256": SILERO_VAD_SHA256,
        }.items()
    )


def _qwen3_model_candidates(
    explicit_path: str | Path | None = None,
) -> Iterator[Path]:
    configured = explicit_path or os.getenv("QWEN3_ASR_MODEL_PATH")
    if configured:
        candidate = Path(configured).expanduser()
        yield candidate
        yield candidate / QWEN3_ASR_MODEL_DIRECTORY
        return
    managed = _default_model_root() / "qwen3-asr"
    yield managed / QWEN3_ASR_MODEL_DIRECTORY
    yield Path.cwd() / "models" / "qwen3-asr" / QWEN3_ASR_MODEL_DIRECTORY
    yield Path.cwd() / ".models" / "qwen3-asr" / QWEN3_ASR_MODEL_DIRECTORY


def _find_qwen3_vad(path: Path) -> Path | None:
    configured = os.getenv("QWEN3_ASR_VAD_PATH")
    candidates = [
        Path(configured).expanduser() if configured else None,
        path / SILERO_VAD_FILENAME,
        path.parent / SILERO_VAD_FILENAME,
        _default_model_root() / "qwen3-asr" / SILERO_VAD_FILENAME,
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            try:
                if candidate.stat().st_size > 0:
                    return candidate.resolve()
            except OSError:
                continue
    return None


def find_qwen3_asr_model(
    model_name: str = QWEN3_ASR_MODEL,
    explicit_path: str | Path | None = None,
) -> Qwen3AsrModelInfo | None:
    normalize_qwen3_asr_model_name(model_name)
    configured = bool(explicit_path or os.getenv("QWEN3_ASR_MODEL_PATH"))
    managed_root = (_default_model_root() / "qwen3-asr").resolve()
    for candidate in _qwen3_model_candidates(explicit_path):
        if _qwen3_model_files_ready(candidate):
            resolved = candidate.resolve()
            if (
                not configured
                and (resolved == managed_root or managed_root in resolved.parents)
                and not _managed_manifest_matches(resolved)
            ):
                continue
            return Qwen3AsrModelInfo(
                path=resolved,
                vad_path=_find_qwen3_vad(resolved),
            )
    return None


def qwen3_asr_diagnostics(
    explicit_path: str | Path | None = None,
) -> dict[str, Any]:
    try:
        from importlib import metadata as importlib_metadata

        version = importlib_metadata.version("sherpa-onnx")
    except importlib_metadata.PackageNotFoundError:
        version = ""
    info = find_qwen3_asr_model(explicit_path=explicit_path)
    return {
        "runtime_installed": bool(version),
        "runtime_version": version,
        "model_ready": info is not None,
        "vad_ready": bool(info and info.vad_path),
        "ready": bool(version and info and info.vad_path),
        "model": info.model if info else QWEN3_ASR_MODEL,
        "model_path": str(info.path) if info else "",
        "vad_path": str(info.vad_path) if info and info.vad_path else "",
        "provider": "cpu",
        "precision": "int8",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_asset(
    path: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
) -> None:
    try:
        actual_bytes = path.stat().st_size
    except OSError as exc:
        raise RuntimeError(f"Managed model asset is unavailable: {path}") from exc
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"{path.name} has {actual_bytes:,} bytes; expected "
            f"{expected_bytes:,}. The incomplete asset was not installed."
        )
    actual_sha256 = _sha256(path)
    if actual_sha256.lower() != expected_sha256.lower():
        raise RuntimeError(
            f"{path.name} failed SHA-256 verification. The untrusted asset was "
            "not installed."
        )


def _download_verified_asset(
    *,
    url: str,
    destination: Path,
    expected_bytes: int,
    expected_sha256: str,
    progress: Callable[[str], None] | None = None,
) -> Path:
    if destination.is_file():
        try:
            _verify_asset(
                destination,
                expected_bytes=expected_bytes,
                expected_sha256=expected_sha256,
            )
            return destination
        except RuntimeError:
            destination.unlink(missing_ok=True)

    try:
        import requests
    except ModuleNotFoundError as exc:
        raise AsrDependencyError(
            "The base requests package is required for managed Qwen3-ASR downloads."
        ) from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    existing = partial.stat().st_size if partial.is_file() else 0
    if existing > expected_bytes:
        partial.unlink(missing_ok=True)
        existing = 0
    headers = {
        "User-Agent": "broadcastify-desktop/0.4 managed-local-model",
        "Accept": "application/octet-stream",
    }
    if existing:
        headers["Range"] = f"bytes={existing}-"
    mode = "ab" if existing else "wb"
    with requests.get(
        url,
        headers=headers,
        stream=True,
        allow_redirects=True,
        timeout=(30, 120),
    ) as response:
        if response.status_code == 416 and existing == expected_bytes:
            pass
        else:
            response.raise_for_status()
            if existing and response.status_code != 206:
                existing = 0
                mode = "wb"
            completed = existing
            next_report = max(1, int((completed * 20) / expected_bytes) + 1)
            with partial.open(mode) as output:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    if not block:
                        continue
                    output.write(block)
                    completed += len(block)
                    report = int((completed * 20) / expected_bytes)
                    if progress and report >= next_report:
                        next_report = report + 1
                        progress(
                            f"Downloading {destination.name}: "
                            f"{min(100, report * 5)}% "
                            f"({completed / (1024 ** 2):.0f} MiB)"
                        )
    try:
        _verify_asset(
            partial,
            expected_bytes=expected_bytes,
            expected_sha256=expected_sha256,
        )
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(destination)
    return destination


def _safe_extract_qwen_archive(archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    total_bytes = 0
    member_count = 0
    with tarfile.open(archive, mode="r:bz2") as package:
        for member in package:
            member_count += 1
            if member_count > 1_000:
                raise RuntimeError("The Qwen3-ASR archive contains too many entries.")
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(
                    "The Qwen3-ASR archive contains an unsafe path and was not installed."
                )
            if not (member.isdir() or member.isfile()):
                raise RuntimeError(
                    "The Qwen3-ASR archive contains a link or special file and was "
                    "not installed."
                )
            target = destination.joinpath(*relative.parts)
            resolved_target = target.resolve()
            if (
                resolved_target != destination_root
                and destination_root not in resolved_target.parents
            ):
                raise RuntimeError(
                    "The Qwen3-ASR archive would escape its managed directory."
                )
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            total_bytes += max(0, int(member.size))
            if total_bytes > 2_000_000_000:
                raise RuntimeError("The Qwen3-ASR archive expands beyond its safety limit.")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = package.extractfile(member)
            if source is None:
                raise RuntimeError(f"Unable to read {member.name} from the model archive.")
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
    model = destination / QWEN3_ASR_MODEL_DIRECTORY
    if not _qwen3_model_files_ready(model):
        raise RuntimeError(
            "The verified Qwen3-ASR archive did not contain the expected model files."
        )
    return model


def _write_qwen_manifest(path: Path) -> None:
    manifest = {
        "schema_version": 1,
        "engine": QWEN3_ASR_ENGINE,
        "model": QWEN3_ASR_MODEL,
        "source_model": "Qwen/Qwen3-ASR-0.6B",
        "runtime": "sherpa-onnx",
        "provider": "cpu",
        "precision": "int8",
        "archive_url": QWEN3_ASR_ARCHIVE_URL,
        "archive_sha256": QWEN3_ASR_ARCHIVE_SHA256,
        "vad_url": SILERO_VAD_URL,
        "vad_sha256": SILERO_VAD_SHA256,
        "installed_utc": datetime.now(timezone.utc).isoformat(),
        "timestamp_contract": "speech-segment bounds; no model token timestamps",
    }
    temporary = path / f".{QWEN3_ASR_MANIFEST}.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path / QWEN3_ASR_MANIFEST)


def prepare_qwen3_asr_model(
    model_name: str,
    *,
    explicit_path: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
    archive_path: str | Path | None = None,
    vad_path: str | Path | None = None,
) -> dict[str, Any]:
    """Install the pinned public INT8 export after an explicit user action."""

    requested = normalize_qwen3_asr_model_name(model_name)
    existing = find_qwen3_asr_model(requested, explicit_path)
    if existing and existing.vad_path:
        return {
            "ready": True,
            "engine": QWEN3_ASR_ENGINE,
            "model": existing.model,
            "source_model": existing.source_model,
            "provider": existing.provider,
            "precision": existing.precision,
            "path": str(existing.path),
            "reused": True,
            "bytes": sum(
                candidate.stat().st_size
                for candidate in existing.path.rglob("*")
                if candidate.is_file()
            ),
            "message": (
                f"Reusing Qwen3-ASR 0.6B INT8 and verified VAD at {existing.path}."
            ),
        }

    if existing:
        target = existing.path
        supplied_vad = Path(vad_path).expanduser() if vad_path else None
        if supplied_vad:
            _verify_asset(
                supplied_vad,
                expected_bytes=SILERO_VAD_BYTES,
                expected_sha256=SILERO_VAD_SHA256,
            )
            shutil.copy2(supplied_vad, target / SILERO_VAD_FILENAME)
        else:
            _download_verified_asset(
                url=SILERO_VAD_URL,
                destination=target / SILERO_VAD_FILENAME,
                expected_bytes=SILERO_VAD_BYTES,
                expected_sha256=SILERO_VAD_SHA256,
                progress=progress,
            )
        _write_qwen_manifest(target)
        final_info = find_qwen3_asr_model(requested, target)
        if final_info is None or final_info.vad_path is None:
            raise RuntimeError("Qwen3-ASR VAD preparation failed final validation.")
        return {
            "ready": True,
            "engine": QWEN3_ASR_ENGINE,
            "model": final_info.model,
            "source_model": final_info.source_model,
            "provider": final_info.provider,
            "precision": final_info.precision,
            "path": str(final_info.path),
            "reused": False,
            "bytes": sum(
                candidate.stat().st_size
                for candidate in final_info.path.rglob("*")
                if candidate.is_file()
            ),
            "message": (
                f"Added the verified VAD model to Qwen3-ASR at {final_info.path}."
            ),
        }

    target_root = _default_model_root() / "qwen3-asr"
    target_root.mkdir(parents=True, exist_ok=True)
    target = target_root / QWEN3_ASR_MODEL_DIRECTORY
    if target.exists():
        raise AsrDependencyError(
            f"The managed target {target} exists but does not contain a complete "
            "Qwen3-ASR model. Move it aside and run Download & test model again."
        )

    supplied_archive = Path(archive_path).expanduser() if archive_path else None
    if supplied_archive:
        _verify_asset(
            supplied_archive,
            expected_bytes=QWEN3_ASR_ARCHIVE_BYTES,
            expected_sha256=QWEN3_ASR_ARCHIVE_SHA256,
        )
        archive = supplied_archive
    else:
        archive = _download_verified_asset(
            url=QWEN3_ASR_ARCHIVE_URL,
            destination=target_root / QWEN3_ASR_ARCHIVE,
            expected_bytes=QWEN3_ASR_ARCHIVE_BYTES,
            expected_sha256=QWEN3_ASR_ARCHIVE_SHA256,
            progress=progress,
        )
    supplied_vad = Path(vad_path).expanduser() if vad_path else None
    if supplied_vad:
        _verify_asset(
            supplied_vad,
            expected_bytes=SILERO_VAD_BYTES,
            expected_sha256=SILERO_VAD_SHA256,
        )
        verified_vad = supplied_vad
    else:
        verified_vad = _download_verified_asset(
            url=SILERO_VAD_URL,
            destination=target_root / SILERO_VAD_FILENAME,
            expected_bytes=SILERO_VAD_BYTES,
            expected_sha256=SILERO_VAD_SHA256,
            progress=progress,
        )

    if progress:
        progress(
            "Extracting the verified Qwen3-ASR INT8 archive into a staging directory."
        )
    staging = Path(
        tempfile.mkdtemp(prefix=".qwen3-asr-install-", dir=target_root)
    )
    try:
        extracted = _safe_extract_qwen_archive(archive, staging)
        shutil.copy2(verified_vad, extracted / SILERO_VAD_FILENAME)
        _write_qwen_manifest(extracted)
        extracted.replace(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    shutil.rmtree(staging, ignore_errors=True)
    if not supplied_archive:
        archive.unlink(missing_ok=True)

    final_info = find_qwen3_asr_model(requested, target)
    if final_info is None or final_info.vad_path is None:
        raise RuntimeError("The prepared Qwen3-ASR model failed final validation.")
    total_bytes = sum(
        candidate.stat().st_size
        for candidate in final_info.path.rglob("*")
        if candidate.is_file()
    )
    return {
        "ready": True,
        "engine": QWEN3_ASR_ENGINE,
        "model": final_info.model,
        "source_model": final_info.source_model,
        "provider": final_info.provider,
        "precision": final_info.precision,
        "path": str(final_info.path),
        "reused": False,
        "bytes": total_bytes,
        "message": (
            f"Installed checksum-verified Qwen3-ASR 0.6B INT8 plus Silero VAD "
            f"at {final_info.path}."
        ),
    }


class SherpaQwen3Asr:
    """Fast CPU Qwen3-ASR with honest speech-region timestamp provenance."""

    SAMPLE_RATE = 16_000
    MAX_SEGMENT_SECONDS = 25.0

    def __init__(
        self,
        model_name: str = QWEN3_ASR_MODEL,
        *,
        device: str = "cpu",
        model_path: str | Path | None = None,
        batch_size: int = 8,
        num_threads: int | None = None,
    ) -> None:
        self.model_name = normalize_qwen3_asr_model_name(model_name)
        requested_device = str(device or "cpu").strip().lower()
        if requested_device not in {"auto", "cpu"}:
            raise ValueError(
                "The current sherpa Qwen3-ASR export supports CPU in this app."
            )
        try:
            import sherpa_onnx
        except ModuleNotFoundError as exc:
            raise AsrDependencyError(
                'Qwen3-ASR is optional. Install it with `pip install -e ".[qwen]"`, '
                "then choose Download & test model."
            ) from exc
        info = find_qwen3_asr_model(self.model_name, model_path)
        if info is None:
            raise AsrDependencyError(
                "Qwen3-ASR 0.6B INT8 is not installed. Choose Download & test "
                "model or set QWEN3_ASR_MODEL_PATH to a complete export."
            )
        if info.vad_path is None:
            raise AsrDependencyError(
                "The Qwen3-ASR model is present, but its verified Silero VAD is "
                "missing. Choose Download & test model to complete it."
            )
        self.model_path = info.path
        self.vad_path = info.vad_path
        self.device = "cpu"
        self.batch_size = max(1, min(16, int(batch_size)))
        default_threads = max(1, min(4, os.cpu_count() or 1))
        self.num_threads = max(
            1,
            int(
                num_threads
                or os.getenv("QWEN3_ASR_THREADS")
                or default_threads
            ),
        )
        self._sherpa = sherpa_onnx
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
            conv_frontend=str(info.path / "conv_frontend.onnx"),
            encoder=str(info.path / "encoder.int8.onnx"),
            decoder=str(info.path / "decoder.int8.onnx"),
            tokenizer=str(info.path / "tokenizer"),
            num_threads=self.num_threads,
            provider="cpu",
            max_total_len=512,
            max_new_tokens=256,
        )
        self.backend = "sherpa-onnx CPU / Qwen3-ASR 0.6B INT8"
        self._last_duration = 0.0
        self._probe_samples: Any = None

    def transcribe(
        self,
        audio_path: str | Path,
        progress: Callable[[str], None] | None = None,
        *,
        segment_hints: Sequence[tuple[float, float]] | None = None,
    ) -> AsrResult:
        source = (
            "pyannote-exclusive-speaker-turns"
            if segment_hints
            else "silero-vad-speech-bounds"
        )
        slices = (
            self._iter_hint_slices(Path(audio_path), segment_hints)
            if segment_hints
            else self._iter_vad_slices(Path(audio_path))
        )
        segments: list[AsrSegment] = []
        decoded_count = 0
        batch: list[_SpeechSlice] = []
        last_report = time.monotonic()
        for speech_slice in slices:
            batch.append(speech_slice)
            if len(batch) < self.batch_size:
                continue
            values = self._decode_batch(batch)
            decoded_count += len(batch)
            segments.extend(values)
            batch.clear()
            if progress and (
                decoded_count % 25 == 0 or time.monotonic() - last_report >= 5.0
            ):
                progress(
                    f"Qwen3-ASR CPU decoded {decoded_count} timestamped speech regions"
                )
                last_report = time.monotonic()
        if batch:
            segments.extend(self._decode_batch(batch))
            decoded_count += len(batch)

        decoder_probe = False
        if (
            decoded_count == 0
            and self._last_duration <= 2.1
            and self._probe_samples is not None
            and len(self._probe_samples)
        ):
            # Execution checks use one second of generated silence. Exercise the
            # decoder, discard any hallucinated text, and keep the transcript empty.
            probe = _SpeechSlice(0.0, self._last_duration, self._probe_samples)
            self._decode_batch([probe])
            decoder_probe = True

        if progress:
            progress(
                f"Qwen3-ASR CPU completed {decoded_count} speech regions; "
                "timestamps are source-region bounds"
            )
        text = " ".join(value.text for value in segments if value.text).strip()
        return AsrResult(
            text=text,
            language="en",
            duration=self._last_duration or None,
            segments=segments,
            engine=QWEN3_ASR_ENGINE,
            backend=self.backend,
            metadata={
                "model": self.model_name,
                "requested_model": self.model_name,
                "model_id": "Qwen/Qwen3-ASR-0.6B",
                "model_path": str(self.model_path),
                "vad_model_path": str(self.vad_path),
                "provider": "cpu",
                "precision": "int8",
                "runtime": "sherpa-onnx",
                "language_hint": "English",
                "timestamp_source": source,
                "word_timestamps": False,
                "decoded_speech_regions": decoded_count,
                "decoder_execution_probe": decoder_probe,
                "max_segment_seconds": self.MAX_SEGMENT_SECONDS,
                "minimum_diarization_region_seconds": QWEN3_ASR_MIN_HINT_SECONDS,
            },
        )

    def _decode_batch(self, values: Sequence[_SpeechSlice]) -> list[AsrSegment]:
        streams = []
        for value in values:
            stream = self._recognizer.create_stream()
            stream.set_option("language", "English")
            stream.accept_waveform(self.SAMPLE_RATE, value.samples)
            streams.append(stream)
        self._recognizer.decode_streams(streams)
        result: list[AsrSegment] = []
        for value, stream in zip(values, streams):
            text = str(stream.result.text or "").strip()
            if text and text not in {".", "The."}:
                result.append(AsrSegment(value.start, value.end, text))
        return result

    def _iter_pcm_blocks(
        self,
        audio_path: Path,
        *,
        seconds_per_block: int = 10,
    ) -> Iterator[tuple[int, Any]]:
        try:
            import numpy as np
        except ModuleNotFoundError as exc:
            raise AsrDependencyError(
                "NumPy is required for Qwen3-ASR audio decoding."
            ) from exc
        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            raise AsrDependencyError("FFmpeg is required for Qwen3-ASR transcription.")
        bytes_per_block = self.SAMPLE_RATE * max(1, seconds_per_block) * 2
        sample_offset = 0
        probe_parts: list[Any] = []
        probe_remaining = self.SAMPLE_RATE * 2
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
                block = self._read_block(process.stdout, bytes_per_block)
                if not block:
                    break
                samples = (
                    np.frombuffer(block, dtype="<i2").astype("float32") / 32768.0
                )
                if probe_remaining > 0:
                    probe = samples[:probe_remaining].copy()
                    probe_parts.append(probe)
                    probe_remaining -= len(probe)
                yield sample_offset, samples
                sample_offset += len(samples)
            return_code = process.wait()
            if return_code != 0:
                error_log.seek(0)
                detail = error_log.read().decode("utf-8", errors="replace").strip()
                raise RuntimeError(detail or f"FFmpeg exited with code {return_code}.")
        self._last_duration = sample_offset / self.SAMPLE_RATE
        self._probe_samples = (
            np.concatenate(probe_parts) if probe_parts else np.asarray([], dtype="float32")
        )

    def _iter_vad_slices(self, audio_path: Path) -> Iterator[_SpeechSlice]:
        try:
            import numpy as np
        except ModuleNotFoundError as exc:
            raise AsrDependencyError(
                "NumPy is required for Qwen3-ASR voice activity detection."
            ) from exc
        config = self._sherpa.VadModelConfig()
        config.silero_vad.model = str(self.vad_path)
        config.silero_vad.threshold = 0.2
        config.silero_vad.min_silence_duration = 0.25
        config.silero_vad.min_speech_duration = 0.15
        config.silero_vad.max_speech_duration = self.MAX_SEGMENT_SECONDS
        config.sample_rate = self.SAMPLE_RATE
        detector = self._sherpa.VoiceActivityDetector(
            config,
            buffer_size_in_seconds=60,
        )
        window_size = int(config.silero_vad.window_size)
        pending = np.asarray([], dtype="float32")
        for _, samples in self._iter_pcm_blocks(audio_path):
            pending = np.concatenate((pending, samples))
            while len(pending) >= window_size:
                detector.accept_waveform(pending[:window_size])
                pending = pending[window_size:]
                yield from self._drain_vad(detector, np)
        if len(pending):
            detector.accept_waveform(
                np.pad(pending, (0, max(0, window_size - len(pending))))
            )
        detector.flush()
        yield from self._drain_vad(detector, np)

    def _drain_vad(self, detector: Any, np: Any) -> Iterator[_SpeechSlice]:
        while not detector.empty():
            segment = detector.front
            samples = np.asarray(segment.samples, dtype="float32")
            start = float(segment.start) / self.SAMPLE_RATE
            end = start + len(samples) / self.SAMPLE_RATE
            detector.pop()
            if len(samples):
                yield _SpeechSlice(start, end, samples)

    def _iter_hint_slices(
        self,
        audio_path: Path,
        hints: Sequence[tuple[float, float]],
    ) -> Iterator[_SpeechSlice]:
        try:
            import numpy as np
        except ModuleNotFoundError as exc:
            raise AsrDependencyError(
                "NumPy is required for Qwen3-ASR segment extraction."
            ) from exc
        bounded = self._bounded_hints(hints)
        if not bounded:
            yield from self._iter_vad_slices(audio_path)
            return
        next_index = 0
        active: list[tuple[int, int, list[Any]]] = []
        for block_start, samples in self._iter_pcm_blocks(audio_path):
            block_end = block_start + len(samples)
            while next_index < len(bounded) and bounded[next_index][0] < block_end:
                start, end = bounded[next_index]
                active.append((start, end, []))
                next_index += 1
            remaining: list[tuple[int, int, list[Any]]] = []
            for start, end, parts in active:
                overlap_start = max(start, block_start)
                overlap_end = min(end, block_end)
                if overlap_end > overlap_start:
                    parts.append(
                        samples[
                            overlap_start - block_start : overlap_end - block_start
                        ].copy()
                    )
                if end <= block_end:
                    if parts:
                        combined = np.concatenate(parts)
                        actual_end = start + len(combined)
                        yield _SpeechSlice(
                            start / self.SAMPLE_RATE,
                            actual_end / self.SAMPLE_RATE,
                            combined,
                        )
                else:
                    remaining.append((start, end, parts))
            active = remaining
        for start, _, parts in active:
            if parts:
                combined = np.concatenate(parts)
                yield _SpeechSlice(
                    start / self.SAMPLE_RATE,
                    (start + len(combined)) / self.SAMPLE_RATE,
                    combined,
                )

    def _bounded_hints(
        self,
        hints: Sequence[tuple[float, float]],
    ) -> list[tuple[int, int]]:
        values: list[tuple[int, int]] = []
        maximum = int(self.MAX_SEGMENT_SECONDS * self.SAMPLE_RATE)
        # Very short radio/noise fragments are a frequent source of generative
        # ASR filler. This evidence-oriented preview excludes sub-second
        # acknowledgements rather than presenting invented text as a quotation.
        minimum = int(QWEN3_ASR_MIN_HINT_SECONDS * self.SAMPLE_RATE)
        for raw_start, raw_end in sorted(hints):
            start = max(0, int(float(raw_start) * self.SAMPLE_RATE))
            end = max(start, int(float(raw_end) * self.SAMPLE_RATE))
            while end - start > maximum:
                values.append((start, start + maximum))
                start += maximum
            if end - start >= minimum:
                values.append((start, end))
        return values

    @staticmethod
    def _read_block(stream: Any, size: int) -> bytes:
        blocks: list[bytes] = []
        remaining = size
        while remaining > 0:
            value = stream.read(remaining)
            if not value:
                break
            blocks.append(value)
            remaining -= len(value)
        return b"".join(blocks)
