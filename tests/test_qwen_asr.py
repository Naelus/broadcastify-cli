from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import broadcastify_cli.qwen_asr as qwen_module
from broadcastify_cli.asr import prepare_asr_model
from broadcastify_cli.qwen_asr import (
    QWEN3_ASR_MANIFEST,
    QWEN3_ASR_MIN_HINT_SECONDS,
    QWEN3_ASR_MODEL,
    QWEN3_ASR_MODEL_DIRECTORY,
    QWEN3_ASR_REQUIRED_FILES,
    SherpaQwen3Asr,
    _SpeechSlice,
    _safe_extract_qwen_archive,
    _write_qwen_manifest,
    find_qwen3_asr_model,
    prepare_qwen3_asr_model,
)


def _write_model(path: Path) -> None:
    for relative in QWEN3_ASR_REQUIRED_FILES:
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(f"fixture:{relative}".encode())


def _write_archive(path: Path) -> None:
    with tarfile.open(path, "w:bz2") as package:
        for relative in QWEN3_ASR_REQUIRED_FILES:
            payload = f"fixture:{relative}".encode()
            info = tarfile.TarInfo(f"{QWEN3_ASR_MODEL_DIRECTORY}/{relative}")
            info.size = len(payload)
            package.addfile(info, io.BytesIO(payload))


def test_qwen_model_discovery_requires_every_runtime_file_and_finds_vad(
    tmp_path: Path,
) -> None:
    model = tmp_path / QWEN3_ASR_MODEL_DIRECTORY
    _write_model(model)
    vad = model / qwen_module.SILERO_VAD_FILENAME
    vad.write_bytes(b"vad")

    info = find_qwen3_asr_model(explicit_path=model)

    assert info is not None
    assert info.path == model.resolve()
    assert info.vad_path == vad.resolve()
    (model / "decoder.int8.onnx").unlink()
    assert find_qwen3_asr_model(explicit_path=model) is None


def test_managed_qwen_discovery_requires_matching_identity_manifest(
    monkeypatch, tmp_path: Path
) -> None:
    root = tmp_path / "managed"
    model = root / "qwen3-asr" / QWEN3_ASR_MODEL_DIRECTORY
    _write_model(model)
    (model / qwen_module.SILERO_VAD_FILENAME).write_bytes(b"vad")
    monkeypatch.setattr(qwen_module, "_default_model_root", lambda: root)

    assert find_qwen3_asr_model() is None
    _write_qwen_manifest(model)
    assert find_qwen3_asr_model() is not None

    manifest_path = model / QWEN3_ASR_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model"] = "different-model"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert find_qwen3_asr_model() is None


def test_qwen_archive_rejects_parent_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar.bz2"
    with tarfile.open(archive, "w:bz2") as package:
        payload = b"unsafe"
        info = tarfile.TarInfo("../outside")
        info.size = len(payload)
        package.addfile(info, io.BytesIO(payload))

    with pytest.raises(RuntimeError, match="unsafe path"):
        _safe_extract_qwen_archive(archive, tmp_path / "output")
    assert not (tmp_path / "outside").exists()


def test_qwen_preparation_verifies_and_atomically_installs_local_assets(
    monkeypatch, tmp_path: Path
) -> None:
    archive = tmp_path / "model.tar.bz2"
    vad = tmp_path / "vad.onnx"
    _write_archive(archive)
    vad.write_bytes(b"verified-vad")
    monkeypatch.setattr(qwen_module, "_default_model_root", lambda: tmp_path / "models")
    monkeypatch.setattr(qwen_module, "QWEN3_ASR_ARCHIVE_BYTES", archive.stat().st_size)
    monkeypatch.setattr(
        qwen_module,
        "QWEN3_ASR_ARCHIVE_SHA256",
        hashlib.sha256(archive.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(qwen_module, "SILERO_VAD_BYTES", vad.stat().st_size)
    monkeypatch.setattr(
        qwen_module,
        "SILERO_VAD_SHA256",
        hashlib.sha256(vad.read_bytes()).hexdigest(),
    )

    result = prepare_qwen3_asr_model(
        QWEN3_ASR_MODEL,
        archive_path=archive,
        vad_path=vad,
    )

    installed = Path(result["path"])
    manifest = json.loads(
        (installed / QWEN3_ASR_MANIFEST).read_text(encoding="utf-8")
    )
    assert result["ready"] is True
    assert result["reused"] is False
    assert installed.parent == tmp_path / "models" / "qwen3-asr"
    assert (installed / "decoder.int8.onnx").is_file()
    assert (installed / qwen_module.SILERO_VAD_FILENAME).read_bytes() == b"verified-vad"
    assert manifest["engine"] == "qwen3-asr"
    assert manifest["timestamp_contract"].startswith("speech-segment bounds")
    assert not list(installed.parent.glob(".qwen3-asr-install-*"))


def test_qwen_transcription_uses_supplied_audio_bounds_not_token_timestamps(
    monkeypatch, tmp_path: Path
) -> None:
    class FakeStream:
        def __init__(self) -> None:
            self.result = SimpleNamespace(text="")
            self.options: dict[str, str] = {}

        def set_option(self, name: str, value: str) -> None:
            self.options[name] = value

        def accept_waveform(self, sample_rate: int, samples: object) -> None:
            assert sample_rate == 16_000
            assert len(samples) == 16_000

    class FakeRecognizer:
        def __init__(self) -> None:
            self.streams: list[FakeStream] = []

        def create_stream(self) -> FakeStream:
            stream = FakeStream()
            self.streams.append(stream)
            return stream

        def decode_streams(self, streams: list[FakeStream]) -> None:
            for stream in streams:
                assert stream.options["language"] == "English"
                stream.result.text = "15 to 20 shots fired. Nothing seen."

    adapter = object.__new__(SherpaQwen3Asr)
    adapter.model_name = QWEN3_ASR_MODEL
    adapter.model_path = tmp_path / "model"
    adapter.vad_path = tmp_path / "silero_vad.onnx"
    adapter.backend = "sherpa-onnx CPU / test"
    adapter.batch_size = 8
    adapter._recognizer = FakeRecognizer()
    adapter._last_duration = 0.0
    adapter._probe_samples = None
    monkeypatch.setattr(
        adapter,
        "_iter_hint_slices",
        lambda _path, _hints: iter(
            [_SpeechSlice(12.5, 13.5, [0.0] * 16_000)]
        ),
    )

    result = adapter.transcribe(
        tmp_path / "radio.wav",
        segment_hints=[(12.5, 13.5)],
    )

    assert [(value.start, value.end, value.text) for value in result.segments] == [
        (12.5, 13.5, "15 to 20 shots fired. Nothing seen.")
    ]
    assert result.words == ()
    assert result.metadata["timestamp_source"] == "pyannote-exclusive-speaker-turns"
    assert result.metadata["word_timestamps"] is False
    assert (
        result.metadata["minimum_diarization_region_seconds"]
        == QWEN3_ASR_MIN_HINT_SECONDS
    )


def test_qwen_ignores_short_diarization_fragments_that_invite_filler() -> None:
    adapter = object.__new__(SherpaQwen3Asr)

    bounded = adapter._bounded_hints(
        [
            (1.0, 1.2),
            (2.0, 2.0 + QWEN3_ASR_MIN_HINT_SECONDS),
            (5.0, 6.5),
        ]
    )

    assert bounded == [
        (2 * 16_000, int((2.0 + QWEN3_ASR_MIN_HINT_SECONDS) * 16_000)),
        (5 * 16_000, 6.5 * 16_000),
    ]


def test_generic_model_preparation_routes_qwen_without_forwarding_a_token(
    monkeypatch, tmp_path: Path
) -> None:
    received: dict[str, object] = {}

    def fake_prepare(model_name: str, **kwargs: object) -> dict[str, object]:
        received["model"] = model_name
        received.update(kwargs)
        return {"ready": True, "path": str(tmp_path / "model")}

    monkeypatch.setattr(qwen_module, "prepare_qwen3_asr_model", fake_prepare)

    result = prepare_asr_model(
        {
            "model": QWEN3_ASR_MODEL,
            "asr_engine": "qwen3-asr",
            "device": "cpu",
            "huggingface_token": "not-needed-for-public-qwen",
        }
    )

    assert result["ready"] is True
    assert received["model"] == QWEN3_ASR_MODEL
    assert "huggingface_token" not in received
