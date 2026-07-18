from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterator, Sequence

from .asr import _default_model_root
from .audio import _probe_audio_duration, find_ffmpeg
from .qwen_asr import _download_verified_asset, _verify_asset


COMMUNITY_DIARIZATION_ENGINE = "community-1"
COMMUNITY_DIARIZATION_QUALITY = "accuracy-default"
PORTABLE_DIARIZATION_ENGINE = "sherpa-onnx"
PORTABLE_DIARIZATION_MODEL = (
    "pyannote-segmentation-3.0-int8+nemo-titanet-small"
)
PORTABLE_DIARIZATION_QUALITY = "preview"
PORTABLE_DIARIZATION_MANIFEST = "broadcastify-model.json"
PORTABLE_SEGMENTATION_DIRECTORY = "sherpa-onnx-pyannote-segmentation-3-0"
PORTABLE_SEGMENTATION_FILENAME = "model.int8.onnx"
PORTABLE_SEGMENTATION_ARCHIVE = f"{PORTABLE_SEGMENTATION_DIRECTORY}.tar.bz2"
PORTABLE_SEGMENTATION_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    f"speaker-segmentation-models/{PORTABLE_SEGMENTATION_ARCHIVE}"
)
PORTABLE_SEGMENTATION_BYTES = 6_958_444
PORTABLE_SEGMENTATION_SHA256 = (
    "24615ee884c897d9d2ba09bb4d30da6bb1b15e685065962db5b02e76e4996488"
)
PORTABLE_SEGMENTATION_MODEL_BYTES = 1_540_506
PORTABLE_SEGMENTATION_MODEL_SHA256 = (
    "d582f4b4c6b48205de7e0643c57df0df5615a3c176189be3fc461e9d18827b5d"
)
PORTABLE_EMBEDDING_FILENAME = "nemo_en_titanet_small.onnx"
# The upstream release tag contains the historical "recongition" spelling.
PORTABLE_EMBEDDING_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    f"speaker-recongition-models/{PORTABLE_EMBEDDING_FILENAME}"
)
PORTABLE_EMBEDDING_BYTES = 40_257_283
PORTABLE_EMBEDDING_SHA256 = (
    "ad4a1802485d8b34c722d2a9d04249662f2ece5d28a7a039063ca22f515a789e"
)
PORTABLE_DEFAULT_CLUSTER_THRESHOLD = 0.95
PORTABLE_DEFAULT_CHUNK_SECONDS = 15 * 60
PORTABLE_DEFAULT_OVERLAP_SECONDS = 5.0
PORTABLE_CHECKPOINT_SCHEMA = 1

_DIARIZATION_ALIASES = {
    "community": COMMUNITY_DIARIZATION_ENGINE,
    "community-1": COMMUNITY_DIARIZATION_ENGINE,
    "pyannote": COMMUNITY_DIARIZATION_ENGINE,
    "pyannote-community-1": COMMUNITY_DIARIZATION_ENGINE,
    "sherpa": PORTABLE_DIARIZATION_ENGINE,
    "sherpa-onnx": PORTABLE_DIARIZATION_ENGINE,
    "fast": PORTABLE_DIARIZATION_ENGINE,
    "portable": PORTABLE_DIARIZATION_ENGINE,
}


def normalize_diarization_engine(value: str | None) -> str:
    normalized = str(value or COMMUNITY_DIARIZATION_ENGINE).strip().lower()
    normalized = normalized.replace("_", "-")
    try:
        return _DIARIZATION_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(
            "Diarization engine must be community-1 or sherpa-onnx."
        ) from exc


def diarization_engine_satisfies(actual: str | None, requested: str | None) -> bool:
    """Return whether an existing engine meets the requested quality contract."""

    if not str(actual or "").strip():
        return False
    try:
        normalized_actual = normalize_diarization_engine(actual)
        normalized_requested = normalize_diarization_engine(requested)
    except ValueError:
        return False
    return normalized_actual == normalized_requested or (
        normalized_actual == COMMUNITY_DIARIZATION_ENGINE
        and normalized_requested == PORTABLE_DIARIZATION_ENGINE
    )


@dataclass(frozen=True)
class PortableDiarizationModelInfo:
    root: Path
    segmentation_path: Path
    embedding_path: Path
    engine: str = PORTABLE_DIARIZATION_ENGINE
    model: str = PORTABLE_DIARIZATION_MODEL
    provider: str = "cpu"
    quality: str = PORTABLE_DIARIZATION_QUALITY


@dataclass(frozen=True)
class PortableSpeakerTurn:
    start: float
    end: float
    speaker: str


def _runtime_version() -> str:
    try:
        from importlib import metadata as importlib_metadata

        return importlib_metadata.version("sherpa-onnx")
    except importlib_metadata.PackageNotFoundError:
        return ""


def _model_files_ready(root: Path) -> bool:
    segmentation = (
        root
        / PORTABLE_SEGMENTATION_DIRECTORY
        / PORTABLE_SEGMENTATION_FILENAME
    )
    embedding = root / PORTABLE_EMBEDDING_FILENAME
    try:
        return (
            segmentation.is_file()
            and segmentation.stat().st_size > 0
            and embedding.is_file()
            and embedding.stat().st_size > 0
        )
    except OSError:
        return False


def _managed_model_files_match(root: Path) -> bool:
    segmentation = (
        root
        / PORTABLE_SEGMENTATION_DIRECTORY
        / PORTABLE_SEGMENTATION_FILENAME
    )
    embedding = root / PORTABLE_EMBEDDING_FILENAME
    try:
        _verify_asset(
            segmentation,
            expected_bytes=PORTABLE_SEGMENTATION_MODEL_BYTES,
            expected_sha256=PORTABLE_SEGMENTATION_MODEL_SHA256,
        )
        _verify_asset(
            embedding,
            expected_bytes=PORTABLE_EMBEDDING_BYTES,
            expected_sha256=PORTABLE_EMBEDDING_SHA256,
        )
    except RuntimeError:
        return False
    return True


def _managed_manifest_matches(root: Path) -> bool:
    try:
        value = json.loads(
            (root / PORTABLE_DIARIZATION_MANIFEST).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return False
    return all(
        str(value.get(key) or "") == expected
        for key, expected in {
            "engine": PORTABLE_DIARIZATION_ENGINE,
            "model": PORTABLE_DIARIZATION_MODEL,
            "runtime": "sherpa-onnx",
            "provider": "cpu",
            "quality": PORTABLE_DIARIZATION_QUALITY,
            "segmentation_sha256": PORTABLE_SEGMENTATION_SHA256,
            "segmentation_model_sha256": PORTABLE_SEGMENTATION_MODEL_SHA256,
            "embedding_sha256": PORTABLE_EMBEDDING_SHA256,
            "embedding_source_model": "NVIDIA NeMo TitaNet-S (titanet_small)",
            "embedding_license": "Apache-2.0",
        }.items()
    )


def _model_info(root: Path) -> PortableDiarizationModelInfo:
    resolved = root.resolve()
    return PortableDiarizationModelInfo(
        root=resolved,
        segmentation_path=(
            resolved
            / PORTABLE_SEGMENTATION_DIRECTORY
            / PORTABLE_SEGMENTATION_FILENAME
        ),
        embedding_path=resolved / PORTABLE_EMBEDDING_FILENAME,
    )


def _portable_model_candidates(
    explicit_path: str | Path | None = None,
) -> Iterator[tuple[Path, bool]]:
    configured = explicit_path or os.getenv("SHERPA_DIARIZATION_MODEL_PATH")
    if configured:
        yield Path(configured).expanduser(), False
        return
    yield _default_model_root() / "speaker-diarization" / "sherpa-onnx", True
    yield Path.cwd() / ".models" / "speaker-diarization" / "sherpa-onnx", False
    yield Path.cwd() / "models" / "speaker-diarization" / "sherpa-onnx", False


def find_portable_diarization_model(
    explicit_path: str | Path | None = None,
) -> PortableDiarizationModelInfo | None:
    segmentation_override = os.getenv("SHERPA_DIARIZATION_SEGMENTATION_PATH")
    embedding_override = os.getenv("SHERPA_DIARIZATION_EMBEDDING_PATH")
    if segmentation_override or embedding_override:
        if not (segmentation_override and embedding_override):
            return None
        segmentation = Path(segmentation_override).expanduser()
        embedding = Path(embedding_override).expanduser()
        try:
            if (
                segmentation.is_file()
                and segmentation.stat().st_size > 0
                and embedding.is_file()
                and embedding.stat().st_size > 0
            ):
                return PortableDiarizationModelInfo(
                    root=segmentation.parent.resolve(),
                    segmentation_path=segmentation.resolve(),
                    embedding_path=embedding.resolve(),
                )
        except OSError:
            return None
        return None

    for candidate, managed in _portable_model_candidates(explicit_path):
        if not _model_files_ready(candidate):
            continue
        if managed and (
            not _managed_manifest_matches(candidate)
            or not _managed_model_files_match(candidate)
        ):
            continue
        return _model_info(candidate)
    return None


def portable_diarization_diagnostics(
    explicit_path: str | Path | None = None,
) -> dict[str, Any]:
    version = _runtime_version()
    info = find_portable_diarization_model(explicit_path)
    return {
        "runtime_installed": bool(version),
        "runtime_version": version,
        "model_ready": info is not None,
        "ready": bool(version and info),
        "engine": PORTABLE_DIARIZATION_ENGINE,
        "model": PORTABLE_DIARIZATION_MODEL,
        "model_path": str(info.root) if info else "",
        "segmentation_path": str(info.segmentation_path) if info else "",
        "embedding_path": str(info.embedding_path) if info else "",
        "provider": "cpu",
        "quality": PORTABLE_DIARIZATION_QUALITY,
        "cluster_threshold": PORTABLE_DEFAULT_CLUSTER_THRESHOLD,
    }


def _safe_extract_segmentation_archive(
    archive: Path, destination: Path
) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    member_count = 0
    total_bytes = 0
    with tarfile.open(archive, mode="r:bz2") as package:
        for member in package:
            member_count += 1
            if member_count > 100:
                raise RuntimeError(
                    "The portable diarization archive contains too many entries."
                )
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(
                    "The portable diarization archive contains an unsafe path."
                )
            if not (member.isdir() or member.isfile()):
                raise RuntimeError(
                    "The portable diarization archive contains a link or special file."
                )
            target = destination.joinpath(*relative.parts)
            resolved_target = target.resolve()
            if (
                resolved_target != destination_root
                and destination_root not in resolved_target.parents
            ):
                raise RuntimeError(
                    "The portable diarization archive would escape its managed directory."
                )
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            total_bytes += max(0, int(member.size))
            if total_bytes > 100_000_000:
                raise RuntimeError(
                    "The portable diarization archive expands beyond its safety limit."
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            source = package.extractfile(member)
            if source is None:
                raise RuntimeError(f"Unable to read {member.name} from the model archive.")
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
    model = (
        destination
        / PORTABLE_SEGMENTATION_DIRECTORY
        / PORTABLE_SEGMENTATION_FILENAME
    )
    if not model.is_file() or model.stat().st_size == 0:
        raise RuntimeError(
            "The verified archive did not contain the expected INT8 segmentation graph."
        )
    return model


def _write_manifest(root: Path) -> None:
    value = {
        "schema_version": 1,
        "engine": PORTABLE_DIARIZATION_ENGINE,
        "model": PORTABLE_DIARIZATION_MODEL,
        "runtime": "sherpa-onnx",
        "provider": "cpu",
        "quality": PORTABLE_DIARIZATION_QUALITY,
        "segmentation_source_model": "pyannote/segmentation-3.0",
        "segmentation_license": "MIT",
        "segmentation_url": PORTABLE_SEGMENTATION_URL,
        "segmentation_sha256": PORTABLE_SEGMENTATION_SHA256,
        "segmentation_model_sha256": PORTABLE_SEGMENTATION_MODEL_SHA256,
        "embedding_source_model": "NVIDIA NeMo TitaNet-S (titanet_small)",
        "embedding_source_url": (
            "https://catalog.ngc.nvidia.com/orgs/nvidia/nemo/models/titanet_small"
        ),
        "embedding_license": "Apache-2.0",
        "embedding_url": PORTABLE_EMBEDDING_URL,
        "embedding_sha256": PORTABLE_EMBEDDING_SHA256,
        "installed_utc": datetime.now(timezone.utc).isoformat(),
        "quality_contract": (
            "fast preview; Community-1 remains the accuracy/evidence default"
        ),
        "speaker_identity_contract": (
            "anonymous acoustic clusters scoped to each bounded processing chunk"
        ),
    }
    temporary = root / f".{PORTABLE_DIARIZATION_MANIFEST}.tmp"
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(root / PORTABLE_DIARIZATION_MANIFEST)


def prepare_portable_diarization_model(
    *,
    explicit_path: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
    archive_path: str | Path | None = None,
    embedding_path: str | Path | None = None,
) -> dict[str, Any]:
    """Install pinned public ONNX models after an explicit user action."""

    existing = find_portable_diarization_model(explicit_path)
    if existing is not None:
        return {
            "ready": True,
            "engine": existing.engine,
            "model": existing.model,
            "provider": existing.provider,
            "quality": existing.quality,
            "path": str(existing.root),
            "reused": True,
            "bytes": sum(
                candidate.stat().st_size
                for candidate in existing.root.rglob("*")
                if candidate.is_file()
            ),
            "message": (
                "Reusing the checksum-identified sherpa-onnx speaker preview "
                f"models at {existing.root}."
            ),
        }
    if explicit_path:
        raise RuntimeError(
            "The configured sherpa diarization model directory is incomplete."
        )

    target_parent = _default_model_root() / "speaker-diarization"
    target_parent.mkdir(parents=True, exist_ok=True)
    target = target_parent / "sherpa-onnx"
    if target.exists():
        if _model_files_ready(target):
            segmentation = (
                target
                / PORTABLE_SEGMENTATION_DIRECTORY
                / PORTABLE_SEGMENTATION_FILENAME
            )
            embedding = target / PORTABLE_EMBEDDING_FILENAME
            _verify_asset(
                segmentation,
                expected_bytes=PORTABLE_SEGMENTATION_MODEL_BYTES,
                expected_sha256=PORTABLE_SEGMENTATION_MODEL_SHA256,
            )
            _verify_asset(
                embedding,
                expected_bytes=PORTABLE_EMBEDDING_BYTES,
                expected_sha256=PORTABLE_EMBEDDING_SHA256,
            )
            _write_manifest(target)
            migrated = find_portable_diarization_model()
            if migrated is None:
                raise RuntimeError(
                    "The verified portable speaker models failed identity migration."
                )
            return {
                "ready": True,
                "engine": migrated.engine,
                "model": migrated.model,
                "provider": migrated.provider,
                "quality": migrated.quality,
                "path": str(migrated.root),
                "reused": True,
                "bytes": sum(
                    candidate.stat().st_size
                    for candidate in migrated.root.rglob("*")
                    if candidate.is_file()
                ),
                "message": (
                    "Reverified the managed portable speaker assets and refreshed "
                    f"their model/license identity at {migrated.root}."
                ),
            }
        raise RuntimeError(
            f"The managed target {target} exists but is incomplete. Move it "
            "aside, then test speakers again."
        )

    supplied_archive = Path(archive_path).expanduser() if archive_path else None
    if supplied_archive:
        _verify_asset(
            supplied_archive,
            expected_bytes=PORTABLE_SEGMENTATION_BYTES,
            expected_sha256=PORTABLE_SEGMENTATION_SHA256,
        )
        archive = supplied_archive
    else:
        archive = _download_verified_asset(
            url=PORTABLE_SEGMENTATION_URL,
            destination=target_parent / PORTABLE_SEGMENTATION_ARCHIVE,
            expected_bytes=PORTABLE_SEGMENTATION_BYTES,
            expected_sha256=PORTABLE_SEGMENTATION_SHA256,
            progress=progress,
        )

    supplied_embedding = (
        Path(embedding_path).expanduser() if embedding_path else None
    )
    if supplied_embedding:
        _verify_asset(
            supplied_embedding,
            expected_bytes=PORTABLE_EMBEDDING_BYTES,
            expected_sha256=PORTABLE_EMBEDDING_SHA256,
        )
        embedding = supplied_embedding
    else:
        embedding = _download_verified_asset(
            url=PORTABLE_EMBEDDING_URL,
            destination=target_parent / PORTABLE_EMBEDDING_FILENAME,
            expected_bytes=PORTABLE_EMBEDDING_BYTES,
            expected_sha256=PORTABLE_EMBEDDING_SHA256,
            progress=progress,
        )

    if progress:
        progress(
            "Extracting the verified segmentation graph into a staging directory."
        )
    staging = Path(
        tempfile.mkdtemp(prefix=".sherpa-diarization-install-", dir=target_parent)
    )
    try:
        _safe_extract_segmentation_archive(archive, staging)
        shutil.copy2(embedding, staging / PORTABLE_EMBEDDING_FILENAME)
        _write_manifest(staging)
        staging.replace(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if not supplied_archive:
        archive.unlink(missing_ok=True)
    if not supplied_embedding:
        embedding.unlink(missing_ok=True)

    final = find_portable_diarization_model()
    if final is None:
        raise RuntimeError(
            "The prepared portable diarization model failed final validation."
        )
    total_bytes = sum(
        candidate.stat().st_size
        for candidate in final.root.rglob("*")
        if candidate.is_file()
    )
    return {
        "ready": True,
        "engine": final.engine,
        "model": final.model,
        "provider": final.provider,
        "quality": final.quality,
        "path": str(final.root),
        "reused": False,
        "bytes": total_bytes,
        "message": (
            "Installed checksum-verified pyannote segmentation 3.0 INT8 and "
            f"NeMo TitaNet preview models at {final.root}."
        ),
    }


def _positive_env_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if math.isfinite(value) and value > 0 else default


def _positive_env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


class SherpaOnnxDiarizer:
    """Bounded-memory CPU speaker-label preview for long radio archives."""

    SAMPLE_RATE = 16_000

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        num_threads: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        cluster_threshold: float | None = None,
        chunk_seconds: int | None = None,
        overlap_seconds: float | None = None,
    ) -> None:
        info = find_portable_diarization_model(model_path)
        if info is None:
            raise RuntimeError(
                "The fast portable speaker models are not prepared. Select "
                "Fast portable preview and run Test speakers to download and "
                "checksum-verify them."
            )
        try:
            import numpy as np
            import sherpa_onnx
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                'Fast portable speaker labels require `pip install -e ".[portable-diarization]"`.'
            ) from exc

        cpu_count = os.cpu_count() or 1
        requested_threads = num_threads or _positive_env_int(
            "SHERPA_DIARIZATION_THREADS", min(8, cpu_count)
        )
        self.num_threads = max(1, min(int(requested_threads), cpu_count))
        self.cluster_threshold = float(
            cluster_threshold
            if cluster_threshold is not None
            else _positive_env_float(
                "SHERPA_DIARIZATION_THRESHOLD",
                PORTABLE_DEFAULT_CLUSTER_THRESHOLD,
            )
        )
        self.chunk_seconds = max(
            60,
            min(
                3600,
                int(
                    chunk_seconds
                    if chunk_seconds is not None
                    else _positive_env_int(
                        "SHERPA_DIARIZATION_CHUNK_SECONDS",
                        PORTABLE_DEFAULT_CHUNK_SECONDS,
                    )
                ),
            ),
        )
        self.overlap_seconds = max(
            0.0,
            min(
                30.0,
                float(
                    overlap_seconds
                    if overlap_seconds is not None
                    else _positive_env_float(
                        "SHERPA_DIARIZATION_OVERLAP_SECONDS",
                        PORTABLE_DEFAULT_OVERLAP_SECONDS,
                    )
                ),
            ),
        )
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        exact_speakers = (
            int(min_speakers)
            if min_speakers is not None
            and max_speakers is not None
            and int(min_speakers) == int(max_speakers)
            else -1
        )
        segmentation = sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(info.segmentation_path)
            ),
            num_threads=self.num_threads,
            debug=False,
            provider="cpu",
        )
        embedding = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(info.embedding_path),
            num_threads=self.num_threads,
            debug=False,
            provider="cpu",
        )
        clustering = sherpa_onnx.FastClusteringConfig(
            num_clusters=exact_speakers,
            threshold=self.cluster_threshold,
        )
        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=segmentation,
            embedding=embedding,
            clustering=clustering,
            min_duration_on=0.3,
            min_duration_off=0.5,
        )
        validate = getattr(config, "validate", None)
        if validate is not None and not validate():
            raise RuntimeError("The sherpa-onnx speaker-label configuration is invalid.")
        self._numpy = np
        self._engine = sherpa_onnx.OfflineSpeakerDiarization(config)
        self.model_info = info
        self.metadata: dict[str, Any] = {
            "engine": PORTABLE_DIARIZATION_ENGINE,
            "model": PORTABLE_DIARIZATION_MODEL,
            "provider": "cpu",
            "quality": PORTABLE_DIARIZATION_QUALITY,
            "runtime_version": _runtime_version(),
            "num_threads": self.num_threads,
            "cluster_threshold": self.cluster_threshold,
            "requested_min_speakers": min_speakers,
            "requested_max_speakers": max_speakers,
            "speaker_range_honored": (
                min_speakers is None and max_speakers is None
            )
            or exact_speakers > 0,
            "clustering_mode": (
                "exact-count" if exact_speakers > 0 else "threshold"
            ),
            "chunk_seconds": self.chunk_seconds,
            "overlap_seconds": self.overlap_seconds,
            "speaker_identity_scope": "processing-chunk",
        }

    def _decode_chunk(
        self,
        audio_path: Path,
        *,
        start: float,
        duration: float,
        ffmpeg: str,
    ) -> Any:
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(audio_path),
            "-vn",
            "-ar",
            str(self.SAMPLE_RATE),
            "-ac",
            "1",
            "-c:a",
            "pcm_f32le",
            "-f",
            "f32le",
            "pipe:1",
        ]
        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if process.returncode != 0 or not process.stdout:
            message = process.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                message or f"FFmpeg could not decode {audio_path.name} for diarization."
            )
        return self._numpy.frombuffer(process.stdout, dtype=self._numpy.float32)

    @staticmethod
    def _merge_turns(
        turns: Sequence[PortableSpeakerTurn],
        *,
        maximum_gap: float = 0.5,
    ) -> list[PortableSpeakerTurn]:
        merged: list[PortableSpeakerTurn] = []
        for turn in sorted(turns, key=lambda value: (value.start, value.end, value.speaker)):
            if (
                merged
                and merged[-1].speaker == turn.speaker
                and turn.start - merged[-1].end <= maximum_gap
            ):
                previous = merged[-1]
                merged[-1] = PortableSpeakerTurn(
                    previous.start,
                    max(previous.end, turn.end),
                    previous.speaker,
                )
            else:
                merged.append(turn)
        return merged

    def _checkpoint_identity(
        self,
        audio_path: Path,
        *,
        duration: float,
        chunk_count: int,
    ) -> dict[str, Any]:
        audio_stat = audio_path.stat()
        model_files: list[dict[str, Any]] = []
        model_info = getattr(self, "model_info", None)
        for role, attribute in (
            ("segmentation", "segmentation_path"),
            ("embedding", "embedding_path"),
        ):
            value = getattr(model_info, attribute, None)
            if value is None:
                continue
            path = Path(value)
            try:
                model_stat = path.stat()
            except OSError:
                continue
            model_files.append(
                {
                    "role": role,
                    "path": str(path.resolve()),
                    "size": model_stat.st_size,
                    "mtime_ns": model_stat.st_mtime_ns,
                }
            )
        metadata = getattr(self, "metadata", {})
        return {
            "schema": PORTABLE_CHECKPOINT_SCHEMA,
            "engine": metadata.get("engine", PORTABLE_DIARIZATION_ENGINE),
            "model": metadata.get("model", PORTABLE_DIARIZATION_MODEL),
            "provider": metadata.get("provider", "cpu"),
            "quality": metadata.get("quality", PORTABLE_DIARIZATION_QUALITY),
            "runtime_version": metadata.get("runtime_version", ""),
            "audio_path": str(audio_path.resolve()),
            "audio_size": audio_stat.st_size,
            "audio_mtime_ns": audio_stat.st_mtime_ns,
            "duration_seconds": duration,
            "chunk_count": chunk_count,
            "chunk_seconds": self.chunk_seconds,
            "overlap_seconds": self.overlap_seconds,
            "cluster_threshold": getattr(self, "cluster_threshold", None),
            "min_speakers": getattr(self, "min_speakers", None),
            "max_speakers": getattr(self, "max_speakers", None),
            "model_files": model_files,
        }

    @staticmethod
    def _checkpoint_turns(
        value: Any,
        *,
        chunk_index: int,
        chunk_count: int,
        chunk_seconds: int,
        duration: float,
    ) -> list[PortableSpeakerTurn] | None:
        if not isinstance(value, list):
            return None
        expected_prefix = (
            "SPEAKER_"
            if chunk_count == 1
            else f"SPEAKER_C{chunk_index:03d}_"
        )
        core_start = chunk_index * chunk_seconds
        core_end = min(duration, core_start + chunk_seconds)
        result: list[PortableSpeakerTurn] = []
        for item in value:
            if not isinstance(item, dict):
                return None
            start = item.get("start")
            end = item.get("end")
            speaker = item.get("speaker")
            if (
                isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, (int, float))
                or not isinstance(end, (int, float))
                or not math.isfinite(float(start))
                or not math.isfinite(float(end))
                or float(start) < 0
                or float(end) <= float(start)
                or float(end) > duration + 0.001
                or not isinstance(speaker, str)
                or not speaker.startswith(expected_prefix)
                or not speaker[len(expected_prefix) :].isdigit()
                or len(speaker) > 128
            ):
                return None
            midpoint = (float(start) + float(end)) / 2
            belongs = midpoint >= core_start and (
                midpoint < core_end
                or (
                    chunk_index == chunk_count - 1
                    and midpoint <= core_end
                )
            )
            if not belongs:
                return None
            result.append(
                PortableSpeakerTurn(
                    start=float(start),
                    end=float(end),
                    speaker=speaker,
                )
            )
        return result

    def _load_checkpoint(
        self,
        checkpoint_path: Path,
        *,
        identity: dict[str, Any],
        duration: float,
        chunk_count: int,
    ) -> tuple[dict[int, list[PortableSpeakerTurn]], bool]:
        if not checkpoint_path.is_file():
            return {}, False
        try:
            payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if (
                not isinstance(payload, dict)
                or payload.get("schema") != PORTABLE_CHECKPOINT_SCHEMA
                or payload.get("identity") != identity
                or not isinstance(payload.get("chunks"), list)
            ):
                return {}, True
            chunks: dict[int, list[PortableSpeakerTurn]] = {}
            for item in payload["chunks"]:
                if not isinstance(item, dict):
                    return {}, True
                index = item.get("index")
                if (
                    isinstance(index, bool)
                    or not isinstance(index, int)
                    or index < 0
                    or index >= chunk_count
                    or index in chunks
                ):
                    return {}, True
                turns = self._checkpoint_turns(
                    item.get("turns"),
                    chunk_index=index,
                    chunk_count=chunk_count,
                    chunk_seconds=self.chunk_seconds,
                    duration=duration,
                )
                if turns is None:
                    return {}, True
                chunks[index] = turns
            return chunks, False
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return {}, True

    @staticmethod
    def _write_checkpoint(
        checkpoint_path: Path,
        *,
        identity: dict[str, Any],
        chunks: dict[int, list[PortableSpeakerTurn]],
    ) -> None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        partial = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
        partial.write_text(
            json.dumps(
                {
                    "schema": PORTABLE_CHECKPOINT_SCHEMA,
                    "identity": identity,
                    "chunks": [
                        {
                            "index": index,
                            "turns": [
                                {
                                    "start": turn.start,
                                    "end": turn.end,
                                    "speaker": turn.speaker,
                                }
                                for turn in chunks[index]
                            ],
                        }
                        for index in sorted(chunks)
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        partial.replace(checkpoint_path)

    def process(
        self,
        audio_file: str | Path,
        progress: Callable[[str], None] | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> list[PortableSpeakerTurn]:
        audio_path = Path(audio_file)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio does not exist: {audio_path}")
        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            raise RuntimeError("FFmpeg is required for portable speaker labels.")
        duration = _probe_audio_duration(audio_path, ffmpeg)
        chunk_count = max(1, math.ceil(duration / self.chunk_seconds))
        checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
        checkpoint_identity = self._checkpoint_identity(
            audio_path,
            duration=duration,
            chunk_count=chunk_count,
        )
        checkpoint_chunks: dict[int, list[PortableSpeakerTurn]] = {}
        checkpoint_ignored = False
        if checkpoint is not None:
            checkpoint_chunks, checkpoint_ignored = self._load_checkpoint(
                checkpoint,
                identity=checkpoint_identity,
                duration=duration,
                chunk_count=chunk_count,
            )
        turns: list[PortableSpeakerTurn] = []
        started = time.monotonic()
        self.metadata.update(
            {
                "checkpoint_enabled": checkpoint is not None,
                "checkpoint_chunks_reused": len(checkpoint_chunks),
                "checkpoint_ignored": checkpoint_ignored,
            }
        )
        if progress and checkpoint_ignored:
            progress(
                "Ignoring an incompatible or incomplete fast speaker preview "
                "checkpoint and starting its chunks again."
            )
        if (
            progress
            and not self.metadata.get("speaker_range_honored", True)
        ):
            progress(
                "Fast speaker preview uses threshold clustering for a speaker "
                "range. Set minimum and maximum to the same value to require "
                "an exact count."
            )
        for chunk_index in range(chunk_count):
            if chunk_index in checkpoint_chunks:
                turns.extend(checkpoint_chunks[chunk_index])
                if progress:
                    progress(
                        "Reusing checkpointed fast speaker preview chunk "
                        f"{chunk_index + 1}/{chunk_count}"
                    )
                continue
            core_start = chunk_index * self.chunk_seconds
            core_end = min(duration, core_start + self.chunk_seconds)
            decode_start = max(0.0, core_start - self.overlap_seconds)
            decode_end = min(duration, core_end + self.overlap_seconds)
            if progress:
                progress(
                    "Fast speaker preview "
                    f"{chunk_index + 1}/{chunk_count}: decoding "
                    f"{core_start / 60:.1f}–{core_end / 60:.1f} minutes"
                )
            samples = self._decode_chunk(
                audio_path,
                start=decode_start,
                duration=decode_end - decode_start,
                ffmpeg=ffmpeg,
            )
            last_percent = -1

            def callback(processed: int, total: int) -> int:
                nonlocal last_percent
                if progress and total:
                    percent = int(processed * 100 / total)
                    if percent >= last_percent + 10 or processed >= total:
                        last_percent = percent
                        progress(
                            "Fast speaker preview "
                            f"{chunk_index + 1}/{chunk_count}: {min(100, percent)}%"
                        )
                return 0

            result = self._engine.process(samples, callback)
            del samples
            segments = (
                result.sort_by_start_time()
                if hasattr(result, "sort_by_start_time")
                else result
            )
            prefix = (
                "SPEAKER_"
                if chunk_count == 1
                else f"SPEAKER_C{chunk_index:03d}_"
            )
            chunk_turns: list[PortableSpeakerTurn] = []
            for segment in segments:
                local_start = float(segment.start)
                local_end = float(segment.end)
                absolute_start = max(0.0, decode_start + local_start)
                absolute_end = min(duration, decode_start + local_end)
                midpoint = (absolute_start + absolute_end) / 2
                belongs = (
                    midpoint >= core_start
                    and (
                        midpoint < core_end
                        or (
                            chunk_index == chunk_count - 1
                            and midpoint <= core_end
                        )
                    )
                )
                if not belongs or absolute_end <= absolute_start:
                    continue
                chunk_turns.append(
                    PortableSpeakerTurn(
                        start=absolute_start,
                        end=absolute_end,
                        speaker=f"{prefix}{int(segment.speaker):02d}",
                    )
                )
            turns.extend(chunk_turns)
            if checkpoint is not None:
                checkpoint_chunks[chunk_index] = chunk_turns
                self._write_checkpoint(
                    checkpoint,
                    identity=checkpoint_identity,
                    chunks=checkpoint_chunks,
                )
        merged = self._merge_turns(turns)
        self.metadata.update(
            {
                "duration_seconds": duration,
                "chunk_count": chunk_count,
                "turn_count": len(merged),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "speaker_identity_scope": "processing-chunk",
            }
        )
        if checkpoint is not None:
            try:
                checkpoint.unlink(missing_ok=True)
            except OSError as exc:
                self.metadata["checkpoint_cleanup_error"] = str(exc)
        if progress:
            progress(
                f"Fast speaker preview found {len(merged)} turns in "
                f"{self.metadata['elapsed_seconds']:.1f} seconds."
            )
        return merged
