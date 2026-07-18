from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import broadcastify_cli.portable_diarization as portable
from broadcastify_cli.portable_diarization import (
    PORTABLE_DIARIZATION_ENGINE,
    PORTABLE_DIARIZATION_MANIFEST,
    PORTABLE_SEGMENTATION_DIRECTORY,
    PORTABLE_SEGMENTATION_FILENAME,
    PortableSpeakerTurn,
    SherpaOnnxDiarizer,
    _safe_extract_segmentation_archive,
    diarization_engine_satisfies,
    find_portable_diarization_model,
    normalize_diarization_engine,
    prepare_portable_diarization_model,
)


def _write_archive(path: Path) -> None:
    with tarfile.open(path, "w:bz2") as package:
        payload = b"segmentation-model"
        info = tarfile.TarInfo(
            f"{PORTABLE_SEGMENTATION_DIRECTORY}/{PORTABLE_SEGMENTATION_FILENAME}"
        )
        info.size = len(payload)
        package.addfile(info, io.BytesIO(payload))
        license_payload = b"MIT"
        license_info = tarfile.TarInfo(
            f"{PORTABLE_SEGMENTATION_DIRECTORY}/LICENSE"
        )
        license_info.size = len(license_payload)
        package.addfile(license_info, io.BytesIO(license_payload))


def test_diarization_engine_aliases_are_explicit() -> None:
    assert normalize_diarization_engine("pyannote") == "community-1"
    assert normalize_diarization_engine("portable") == PORTABLE_DIARIZATION_ENGINE
    with pytest.raises(ValueError, match="community-1 or sherpa-onnx"):
        normalize_diarization_engine("magic-unified-model")


def test_accuracy_labels_satisfy_preview_but_preview_does_not_satisfy_accuracy() -> None:
    assert diarization_engine_satisfies("community-1", "sherpa-onnx") is True
    assert diarization_engine_satisfies("sherpa-onnx", "community-1") is False
    assert diarization_engine_satisfies("sherpa-onnx", "sherpa-onnx") is True
    assert diarization_engine_satisfies("", "community-1") is False


def test_managed_portable_model_requires_matching_manifest(
    monkeypatch, tmp_path: Path
) -> None:
    root = tmp_path / "models"
    target = root / "speaker-diarization" / "sherpa-onnx"
    segmentation = (
        target / PORTABLE_SEGMENTATION_DIRECTORY / PORTABLE_SEGMENTATION_FILENAME
    )
    segmentation.parent.mkdir(parents=True)
    segmentation_payload = b"segmentation"
    embedding_payload = b"embedding"
    segmentation.write_bytes(segmentation_payload)
    (target / portable.PORTABLE_EMBEDDING_FILENAME).write_bytes(embedding_payload)
    monkeypatch.setattr(portable, "_default_model_root", lambda: root)
    monkeypatch.setattr(
        portable, "PORTABLE_SEGMENTATION_MODEL_BYTES", len(segmentation_payload)
    )
    monkeypatch.setattr(
        portable,
        "PORTABLE_SEGMENTATION_MODEL_SHA256",
        hashlib.sha256(segmentation_payload).hexdigest(),
    )
    monkeypatch.setattr(
        portable, "PORTABLE_EMBEDDING_BYTES", len(embedding_payload)
    )
    monkeypatch.setattr(
        portable,
        "PORTABLE_EMBEDDING_SHA256",
        hashlib.sha256(embedding_payload).hexdigest(),
    )

    assert find_portable_diarization_model() is None
    portable._write_manifest(target)
    assert find_portable_diarization_model() is not None

    segmentation.write_bytes(b"tampered-model")
    assert find_portable_diarization_model() is None
    segmentation.write_bytes(segmentation_payload)

    manifest_path = target / PORTABLE_DIARIZATION_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["quality"] = "evidence"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert find_portable_diarization_model() is None


def test_portable_archive_rejects_parent_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar.bz2"
    with tarfile.open(archive, "w:bz2") as package:
        payload = b"unsafe"
        info = tarfile.TarInfo("../outside")
        info.size = len(payload)
        package.addfile(info, io.BytesIO(payload))

    with pytest.raises(RuntimeError, match="unsafe path"):
        _safe_extract_segmentation_archive(archive, tmp_path / "output")
    assert not (tmp_path / "outside").exists()


def test_portable_model_preparation_verifies_and_installs_local_assets(
    monkeypatch, tmp_path: Path
) -> None:
    archive = tmp_path / "segmentation.tar.bz2"
    embedding = tmp_path / "embedding.onnx"
    _write_archive(archive)
    embedding.write_bytes(b"embedding-model")
    monkeypatch.setattr(portable, "_default_model_root", lambda: tmp_path / "models")
    monkeypatch.setattr(portable, "PORTABLE_SEGMENTATION_BYTES", archive.stat().st_size)
    monkeypatch.setattr(
        portable,
        "PORTABLE_SEGMENTATION_SHA256",
        hashlib.sha256(archive.read_bytes()).hexdigest(),
    )
    segmentation_payload = b"segmentation-model"
    monkeypatch.setattr(
        portable,
        "PORTABLE_SEGMENTATION_MODEL_BYTES",
        len(segmentation_payload),
    )
    monkeypatch.setattr(
        portable,
        "PORTABLE_SEGMENTATION_MODEL_SHA256",
        hashlib.sha256(segmentation_payload).hexdigest(),
    )
    monkeypatch.setattr(portable, "PORTABLE_EMBEDDING_BYTES", embedding.stat().st_size)
    monkeypatch.setattr(
        portable,
        "PORTABLE_EMBEDDING_SHA256",
        hashlib.sha256(embedding.read_bytes()).hexdigest(),
    )

    result = prepare_portable_diarization_model(
        archive_path=archive,
        embedding_path=embedding,
    )

    installed = Path(result["path"])
    manifest = json.loads(
        (installed / PORTABLE_DIARIZATION_MANIFEST).read_text(encoding="utf-8")
    )
    assert result["ready"] is True
    assert result["reused"] is False
    assert (
        installed
        / PORTABLE_SEGMENTATION_DIRECTORY
        / PORTABLE_SEGMENTATION_FILENAME
    ).is_file()
    assert manifest["quality"] == "preview"
    assert manifest["segmentation_license"] == "MIT"
    assert manifest["embedding_license"] == "Apache-2.0"
    assert manifest["embedding_source_model"] == (
        "NVIDIA NeMo TitaNet-S (titanet_small)"
    )
    assert not list(installed.parent.glob(".sherpa-diarization-install-*"))


def test_chunked_preview_scopes_speakers_and_discards_overlap_duplicates(
    monkeypatch, tmp_path: Path
) -> None:
    audio = tmp_path / "radio.wav"
    audio.write_bytes(b"audio")

    class FakeEngine:
        def __init__(self) -> None:
            self.calls = 0

        def process(self, _samples: object, callback) -> list[SimpleNamespace]:
            self.calls += 1
            callback(1, 1)
            if self.calls == 1:
                return [
                    SimpleNamespace(start=8.0, end=12.0, speaker=0),
                    SimpleNamespace(start=58.0, end=62.0, speaker=1),
                ]
            return [
                # decode_start is 55s, so this maps to 58-62s. Its midpoint is
                # in the second core and is retained exactly once.
                SimpleNamespace(start=3.0, end=7.0, speaker=0),
                SimpleNamespace(start=12.0, end=14.0, speaker=1),
            ]

    adapter = object.__new__(SherpaOnnxDiarizer)
    adapter.chunk_seconds = 60
    adapter.overlap_seconds = 5.0
    adapter._numpy = SimpleNamespace(float32="float32", frombuffer=lambda *_a, **_k: [0.0])
    adapter._engine = FakeEngine()
    adapter.metadata = {}
    adapter._decode_chunk = lambda *_a, **_k: [0.0]
    monkeypatch.setattr(portable, "find_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(portable, "_probe_audio_duration", lambda *_args: 120.0)

    turns = adapter.process(audio)

    assert turns == [
        PortableSpeakerTurn(8.0, 12.0, "SPEAKER_C000_00"),
        PortableSpeakerTurn(58.0, 62.0, "SPEAKER_C001_00"),
        PortableSpeakerTurn(67.0, 69.0, "SPEAKER_C001_01"),
    ]
    assert adapter.metadata["chunk_count"] == 2
    assert adapter.metadata["speaker_identity_scope"] == "processing-chunk"


def test_chunk_checkpoint_resumes_after_failure_and_is_removed_on_success(
    monkeypatch, tmp_path: Path
) -> None:
    audio = tmp_path / "radio.wav"
    audio.write_bytes(b"audio")
    checkpoint = tmp_path / "radio.sherpa-onnx.chunks.json"

    class FailingEngine:
        def __init__(self) -> None:
            self.calls = 0

        def process(self, _samples: object, callback) -> list[SimpleNamespace]:
            self.calls += 1
            callback(1, 1)
            if self.calls == 2:
                raise RuntimeError("interrupted")
            return [SimpleNamespace(start=10.0, end=12.0, speaker=0)]

    class CompletingEngine:
        def __init__(self) -> None:
            self.calls = 0

        def process(self, _samples: object, callback) -> list[SimpleNamespace]:
            self.calls += 1
            callback(1, 1)
            return [SimpleNamespace(start=10.0, end=12.0, speaker=1)]

    def adapter(engine: object) -> SherpaOnnxDiarizer:
        value = object.__new__(SherpaOnnxDiarizer)
        value.chunk_seconds = 60
        value.overlap_seconds = 5.0
        value.cluster_threshold = 0.95
        value.min_speakers = None
        value.max_speakers = None
        value._numpy = SimpleNamespace(
            float32="float32",
            frombuffer=lambda *_a, **_k: [0.0],
        )
        value._engine = engine
        value.metadata = {}
        value._decode_chunk = lambda *_a, **_k: [0.0]
        return value

    monkeypatch.setattr(portable, "find_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(portable, "_probe_audio_duration", lambda *_args: 120.0)
    first_engine = FailingEngine()

    with pytest.raises(RuntimeError, match="interrupted"):
        adapter(first_engine).process(audio, checkpoint_path=checkpoint)

    assert first_engine.calls == 2
    assert checkpoint.is_file()
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert [value["index"] for value in payload["chunks"]] == [0]

    second_engine = CompletingEngine()
    resumed = adapter(second_engine)
    messages: list[str] = []
    turns = resumed.process(
        audio,
        progress=messages.append,
        checkpoint_path=checkpoint,
    )

    assert second_engine.calls == 1
    assert turns == [
        PortableSpeakerTurn(10.0, 12.0, "SPEAKER_C000_00"),
        PortableSpeakerTurn(65.0, 67.0, "SPEAKER_C001_01"),
    ]
    assert resumed.metadata["checkpoint_enabled"] is True
    assert resumed.metadata["checkpoint_chunks_reused"] == 1
    assert resumed.metadata["checkpoint_ignored"] is False
    assert "Reusing checkpointed fast speaker preview chunk 1/2" in messages
    assert not checkpoint.exists()


def test_chunk_checkpoint_is_not_reused_after_audio_changes(
    monkeypatch, tmp_path: Path
) -> None:
    audio = tmp_path / "radio.wav"
    audio.write_bytes(b"audio")
    checkpoint = tmp_path / "radio.sherpa-onnx.chunks.json"

    class Engine:
        def __init__(self, fail_on_second: bool = False) -> None:
            self.calls = 0
            self.fail_on_second = fail_on_second

        def process(self, _samples: object, callback) -> list[SimpleNamespace]:
            self.calls += 1
            callback(1, 1)
            if self.fail_on_second and self.calls == 2:
                raise RuntimeError("interrupted")
            return [SimpleNamespace(start=10.0, end=12.0, speaker=0)]

    def adapter(engine: object) -> SherpaOnnxDiarizer:
        value = object.__new__(SherpaOnnxDiarizer)
        value.chunk_seconds = 60
        value.overlap_seconds = 5.0
        value.cluster_threshold = 0.95
        value.min_speakers = None
        value.max_speakers = None
        value._numpy = SimpleNamespace(
            float32="float32",
            frombuffer=lambda *_a, **_k: [0.0],
        )
        value._engine = engine
        value.metadata = {}
        value._decode_chunk = lambda *_a, **_k: [0.0]
        return value

    monkeypatch.setattr(portable, "find_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(portable, "_probe_audio_duration", lambda *_args: 120.0)
    with pytest.raises(RuntimeError, match="interrupted"):
        adapter(Engine(fail_on_second=True)).process(
            audio,
            checkpoint_path=checkpoint,
        )

    audio.write_bytes(b"changed audio")
    replacement_engine = Engine()
    replacement = adapter(replacement_engine)
    replacement.process(audio, checkpoint_path=checkpoint)

    assert replacement_engine.calls == 2
    assert replacement.metadata["checkpoint_chunks_reused"] == 0
    assert replacement.metadata["checkpoint_ignored"] is True
    assert not checkpoint.exists()
