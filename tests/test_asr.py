import hashlib
import io
import json
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import pytest

from broadcastify_cli.asr import (
    AsrDependencyError,
    OpenVinoWhisperAsr,
    WhisperCppAsr,
    WindowsMlWhisperAsr,
    find_windows_ml_model,
    normalize_asr_engine,
    prepare_whisper_cpp_model,
    prepare_windows_ml_model,
    windows_ml_model_info,
)


def _write_whisper_cpp_vad(path: Path) -> Path:
    vad = path / "ggml-silero-v6.2.0.bin"
    vad.write_bytes(b"vad")
    return vad


def test_engine_auto_selection_follows_requested_accelerator() -> None:
    assert normalize_asr_engine("auto", "cuda") == "faster-whisper"
    assert normalize_asr_engine("auto", "vulkan") == "whisper.cpp"
    assert normalize_asr_engine("auto", "metal") == "whisper.cpp"
    assert normalize_asr_engine("auto", "openvino-gpu") == "openvino"
    assert normalize_asr_engine("auto", "windows-ml") == "windows-ml"
    assert normalize_asr_engine("qwen3", "cpu") == "qwen3-asr"


def test_macos_auto_selects_detected_native_metal(monkeypatch) -> None:
    monkeypatch.setattr("broadcastify_cli.asr.sys.platform", "darwin")
    monkeypatch.setattr("broadcastify_cli.asr.find_whisper_cpp", lambda: "/opt/whisper-cli")
    monkeypatch.setattr(
        "broadcastify_cli.asr.whisper_cpp_backends", lambda _path: ["cpu", "metal"]
    )

    assert normalize_asr_engine("auto", "auto") == "whisper.cpp"


def test_whisper_cpp_accepts_native_metal_backend(tmp_path: Path) -> None:
    executable = tmp_path / "whisper-cli"
    executable.write_bytes(b"binary")
    (tmp_path / "libggml-metal.dylib").write_bytes(b"backend")
    model = tmp_path / "ggml-tiny.en-q5_1.bin"
    model.write_bytes(b"model")
    _write_whisper_cpp_vad(tmp_path)

    engine = WhisperCppAsr(
        "tiny", device="metal", executable=executable, model_path=model
    )

    assert engine.backend == "metal"
    assert engine.backends == ["cpu", "metal"]


def test_portable_model_aliases_accept_web_ui_english_suffix(tmp_path: Path) -> None:
    from broadcastify_cli.asr import whisper_cpp_model_filename

    assert whisper_cpp_model_filename("tiny.en") == "ggml-tiny.en-q5_1.bin"
    assert whisper_cpp_model_filename("medium.en") == "ggml-medium.en-q5_0.bin"


def _write_windows_ml_model(
    path: Path,
    *,
    hidden_size: int = 384,
    encoder_layers: int = 4,
    decoder_layers: int = 4,
    provider: str = "cpu",
) -> None:
    path.mkdir(parents=True)
    provider_options = [] if provider == "cpu" else [{provider: {}}]
    component = {
        "session_options": {"provider_options": provider_options},
        "hidden_size": hidden_size,
        "num_hidden_layers": encoder_layers,
    }
    (path / "genai_config.json").write_text(
        json.dumps(
            {
                "model": {
                    "encoder": component,
                    "decoder": {
                        **component,
                        "num_hidden_layers": decoder_layers,
                    },
                }
            }
        ),
        encoding="utf-8",
    )


def test_windows_ml_discovers_suffixed_managed_model_and_infers_identity(
    monkeypatch, tmp_path: Path
) -> None:
    model_root = tmp_path / "managed"
    model = model_root / "windowsml" / "whisper-tiny-fp32-cpu"
    _write_windows_ml_model(model)
    monkeypatch.setenv("BROADCASTIFY_MODEL_DIR", str(model_root))
    monkeypatch.delenv("WINDOWS_ML_WHISPER_MODEL_PATH", raising=False)
    monkeypatch.chdir(tmp_path)

    resolved = find_windows_ml_model("tiny.en")
    info = windows_ml_model_info(model)

    assert resolved == model.resolve()
    assert info is not None
    assert info.model == "tiny"
    assert info.provider == "cpu"
    assert info.precision == "fp32"


def test_windows_ml_rejects_explicit_model_identity_mismatch(tmp_path: Path) -> None:
    model = tmp_path / "whisper-tiny-fp32-cpu"
    _write_windows_ml_model(model)

    with pytest.raises(AsrDependencyError, match="contains tiny"):
        find_windows_ml_model("turbo", model)


def test_windows_ml_resumes_completed_audio_chunks_after_interruption(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "combined_90001_20260712.mp3"
    source.write_bytes(b"retained combined audio")
    helper = tmp_path / "BroadcastifyCli.WindowsML.exe"
    helper.write_bytes(b"helper")
    model = tmp_path / "whisper-base-fp32-cpu"
    _write_windows_ml_model(model)

    engine = WindowsMlWhisperAsr.__new__(WindowsMlWhisperAsr)
    engine.requested_model_name = "base"
    engine.model_name = "base"
    engine.model_source = "openai/whisper-base"
    engine.model_provider = "cpu"
    engine.model_precision = "fp32"
    engine.model_path = model
    engine.helper = str(helper)
    engine.chunk_seconds = 5
    engine.backend = "Windows ML test backend"
    chunks = [
        b"\x00\x00" * (engine.SAMPLE_RATE * 5),
        b"\x01\x00" * (engine.SAMPLE_RATE * 5),
    ]
    monkeypatch.setattr(engine, "_audio_chunks", lambda _source: iter(chunks))

    class FakeProcess:
        def __init__(self, responses: list[dict[str, object]]) -> None:
            self.stdin = io.StringIO()
            self.stdout = io.StringIO(
                "".join(json.dumps(value) + "\n" for value in responses)
            )
            self.return_code: int | None = None

        def poll(self) -> int | None:
            return self.return_code

        def kill(self) -> None:
            self.return_code = -9

        def wait(self, timeout: int | None = None) -> int:
            del timeout
            if self.return_code is None:
                self.return_code = 0
            return self.return_code

    processes = [
        FakeProcess([{"text": "first dispatch", "backend": "test backend"}]),
        FakeProcess([{"text": "second dispatch", "backend": "test backend"}]),
    ]
    monkeypatch.setattr(
        "broadcastify_cli.asr.subprocess.Popen",
        lambda *_args, **_kwargs: processes.pop(0),
    )
    progress: list[str] = []

    with pytest.raises(RuntimeError, match="helper stopped"):
        engine.transcribe(source, progress.append)

    checkpoint = engine._checkpoint_path(source)
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert len(payload["chunks"]) == 1
    assert payload["chunks"][0]["text"] == "first dispatch"

    result = engine.transcribe(source, progress.append)

    assert result.text == "first dispatch second dispatch"
    assert [value.text for value in result.segments] == [
        "first dispatch",
        "second dispatch",
    ]
    assert result.metadata["checkpoint_chunks_reused"] == 1
    assert result.metadata["checkpoint_retained_until_cache"] is True
    assert checkpoint.exists()
    engine.finalize_checkpoint(source)
    assert not checkpoint.exists()
    assert any(
        value == "Reusing Windows ML transcript checkpoint chunk 1"
        for value in progress
    )


def test_windows_ml_preparation_reuses_matching_managed_model(
    monkeypatch, tmp_path: Path
) -> None:
    model_root = tmp_path / "managed"
    model = model_root / "windowsml" / "whisper-tiny-fp32-cpu"
    _write_windows_ml_model(model)
    monkeypatch.setenv("BROADCASTIFY_MODEL_DIR", str(model_root))
    monkeypatch.delenv("WINDOWS_ML_WHISPER_MODEL_PATH", raising=False)
    monkeypatch.chdir(tmp_path)

    result = prepare_windows_ml_model("tiny")

    assert result["ready"] is True
    assert result["reused"] is True
    assert result["model"] == "tiny"
    assert result["path"] == str(model.resolve())


def test_windows_ml_fresh_public_build_disables_missing_token_requirement(
    monkeypatch, tmp_path: Path
) -> None:
    model_root = tmp_path / "managed"
    monkeypatch.setenv("BROADCASTIFY_MODEL_DIR", str(model_root))
    monkeypatch.delenv("WINDOWS_ML_WHISPER_MODEL_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    package = ModuleType("onnxruntime_genai")
    package.__path__ = []  # type: ignore[attr-defined]
    models = ModuleType("onnxruntime_genai.models")
    models.__path__ = []  # type: ignore[attr-defined]
    builder = ModuleType("onnxruntime_genai.models.builder")
    monkeypatch.setitem(sys.modules, "onnxruntime_genai", package)
    monkeypatch.setitem(sys.modules, "onnxruntime_genai.models", models)
    monkeypatch.setitem(sys.modules, "onnxruntime_genai.models.builder", builder)
    invoked: list[str] = []

    class FakeProcess:
        def __init__(self, arguments, **_kwargs) -> None:
            invoked.extend(arguments)
            output = Path(arguments[arguments.index("-o") + 1])
            _write_windows_ml_model(output)
            self.stdout = io.StringIO("builder complete\n")

        @staticmethod
        def wait() -> int:
            return 0

    monkeypatch.setattr("broadcastify_cli.asr.subprocess.Popen", FakeProcess)

    result = prepare_windows_ml_model("tiny")

    assert result["ready"] is True
    assert result["reused"] is False
    assert invoked[invoked.index("--extra_options") + 1] == "hf_token=false"
    manifest = json.loads(
        (
            Path(result["path"])
            / "broadcastify-model.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["model"] == "tiny"
    assert manifest["provider"] == "cpu"


def test_whisper_cpp_rejects_renamed_or_mismatched_explicit_model(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "whisper-cli"
    executable.write_bytes(b"binary")
    model = tmp_path / "ggml-base.en-q5_1.bin"
    model.write_bytes(b"model")

    with pytest.raises(AsrDependencyError, match="expects ggml-tiny"):
        WhisperCppAsr(
            "tiny",
            device="cpu",
            executable=executable,
            model_path=model,
        )


def test_whisper_cpp_requires_a_speech_detector(tmp_path: Path) -> None:
    executable = tmp_path / "whisper-cli"
    executable.write_bytes(b"binary")
    model = tmp_path / "ggml-tiny.en-q5_1.bin"
    model.write_bytes(b"model")

    with pytest.raises(AsrDependencyError, match="VAD model"):
        WhisperCppAsr(
            "tiny",
            device="cpu",
            executable=executable,
            model_path=model,
        )


def test_whisper_cpp_preparation_pins_and_verifies_vad(
    monkeypatch, tmp_path: Path
) -> None:
    import broadcastify_cli.asr as asr_module

    model_root = tmp_path / "managed"
    model_bytes = b"model"
    vad_bytes = b"verified-vad"
    calls: list[tuple[str, str, str | None]] = []
    monkeypatch.setenv("BROADCASTIFY_MODEL_DIR", str(model_root))
    monkeypatch.delenv("WHISPER_CPP_MODEL_PATH", raising=False)
    monkeypatch.delenv("WHISPER_CPP_VAD_MODEL_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(asr_module, "WHISPER_CPP_VAD_BYTES", len(vad_bytes))
    monkeypatch.setattr(
        asr_module,
        "WHISPER_CPP_VAD_SHA256",
        hashlib.sha256(vad_bytes).hexdigest(),
    )

    def fake_download(
        *,
        repo_id: str,
        filename: str,
        local_dir: Path,
        token=None,
        revision=None,
    ) -> str:
        calls.append((repo_id, filename, revision))
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(
            vad_bytes
            if filename == asr_module.WHISPER_CPP_VAD_FILENAME
            else model_bytes
        )
        return str(target)

    hub = ModuleType("huggingface_hub")
    hub.hf_hub_download = fake_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    result = prepare_whisper_cpp_model("tiny")
    reused = prepare_whisper_cpp_model("tiny")

    assert result["reused"] is False
    assert result["bytes"] == len(model_bytes) + len(vad_bytes)
    assert Path(result["vad_path"]).read_bytes() == vad_bytes
    assert calls[-1] == (
        asr_module.WHISPER_CPP_VAD_REPOSITORY,
        asr_module.WHISPER_CPP_VAD_FILENAME,
        asr_module.WHISPER_CPP_VAD_REVISION,
    )
    assert reused["reused"] is True


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        ("metal", "native macOS whisper.cpp build compiled with GGML_METAL=ON"),
        ("vulkan", "GGML_VULKAN=1"),
        ("cpu", "native whisper.cpp"),
    ],
)
def test_missing_whisper_cpp_gives_device_specific_setup_help(
    monkeypatch, tmp_path: Path, device: str, expected: str
) -> None:
    monkeypatch.delenv("WHISPER_CPP_CONTAINER_IMAGE", raising=False)

    with pytest.raises(RuntimeError, match=expected):
        WhisperCppAsr(
            "tiny",
            device=device,
            executable=tmp_path / "missing-whisper-cli",
        )


def test_whisper_cpp_json_is_normalized(monkeypatch, tmp_path: Path) -> None:
    executable = tmp_path / "whisper-cli.exe"
    executable.write_bytes(b"binary")
    (tmp_path / "ggml-vulkan.dll").write_bytes(b"backend")
    model = tmp_path / "ggml-large-v3-turbo-q5_0.bin"
    model.write_bytes(b"model")
    _write_whisper_cpp_vad(tmp_path)
    audio = tmp_path / "radio.wav"
    audio.write_bytes(b"audio")
    invoked_with: list[str] = []

    class FakeProcess:
        def __init__(self, arguments, **_kwargs) -> None:
            invoked_with.extend(arguments)
            output = Path(arguments[arguments.index("--output-file") + 1]).with_suffix(".json")
            output.write_text(
                json.dumps(
                    {
                        "systeminfo": "VULKAN = 1",
                        "result": {"language": "en"},
                        "transcription": [
                            {
                                "offsets": {"from": 1250, "to": 4500},
                                "text": " Unit responding.",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            self.stderr = io.StringIO("whisper_print_progress: progress = 100%\n")

        @staticmethod
        def wait() -> int:
            return 0

    monkeypatch.setattr("broadcastify_cli.asr.subprocess.Popen", FakeProcess)
    engine = WhisperCppAsr(
        "turbo", device="vulkan", executable=executable, model_path=model
    )
    messages: list[str] = []

    result = engine.transcribe(audio, progress=messages.append)

    assert result.engine == "whisper.cpp"
    assert result.backend == "vulkan"
    assert result.text == "Unit responding."
    assert result.segments[0].start == 1.25
    assert result.segments[0].end == 4.5
    assert "--vad" in invoked_with
    assert invoked_with[invoked_with.index("--max-context") + 1] == "0"
    assert (
        Path(invoked_with[invoked_with.index("--vad-model") + 1]).name
        == "ggml-silero-v6.2.0.bin"
    )
    assert result.metadata["vad_enabled"] is True
    assert "whisper.cpp transcription: 100%" in messages


def test_whisper_cpp_prepares_non_wav_audio(monkeypatch, tmp_path: Path) -> None:
    executable = tmp_path / "whisper-cli.exe"
    executable.write_bytes(b"binary")
    model = tmp_path / "ggml-tiny.en-q5_1.bin"
    model.write_bytes(b"model")
    _write_whisper_cpp_vad(tmp_path)
    audio = tmp_path / "radio.mp3"
    audio.write_bytes(b"compressed audio")
    invoked_with: list[str] = []

    def fake_convert(arguments, **_kwargs):
        Path(arguments[-1]).write_bytes(b"prepared wav")
        return SimpleNamespace(returncode=0, stderr="")

    class FakeProcess:
        def __init__(self, arguments, **_kwargs) -> None:
            invoked_with.extend(arguments)
            output = Path(arguments[arguments.index("--output-file") + 1]).with_suffix(
                ".json"
            )
            output.write_text(
                json.dumps(
                    {
                        "result": {"language": "en"},
                        "transcription": [
                            {
                                "offsets": {"from": 0, "to": 1000},
                                "text": " Dispatch.",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            self.stderr = io.StringIO("")

        @staticmethod
        def wait() -> int:
            return 0

    monkeypatch.setattr("broadcastify_cli.asr.find_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr("broadcastify_cli.asr.subprocess.run", fake_convert)
    monkeypatch.setattr("broadcastify_cli.asr.subprocess.Popen", FakeProcess)
    engine = WhisperCppAsr(
        "tiny", device="cpu", executable=executable, model_path=model
    )

    result = engine.transcribe(audio)

    prepared = Path(invoked_with[invoked_with.index("--file") + 1])
    assert prepared.name == "radio.whisper.cpp.wav"
    assert not prepared.exists()
    assert result.text == "Dispatch."


def test_whisper_cpp_container_is_rootless_offline_and_bind_limited(
    monkeypatch, tmp_path: Path
) -> None:
    model = tmp_path / "ggml-tiny.en-q5_1.bin"
    model.write_bytes(b"model")
    _write_whisper_cpp_vad(tmp_path)
    audio = tmp_path / "radio.wav"
    audio.write_bytes(b"audio")
    dri = tmp_path / "dri"
    dri.mkdir()
    (dri / "renderD128").write_bytes(b"device")
    invoked_with: list[str] = []

    monkeypatch.setattr(
        "broadcastify_cli.accelerators.shutil.which", lambda name: name
    )
    monkeypatch.setattr(
        "broadcastify_cli.accelerators.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout="sha256:test-image\n", stderr=""
        ),
    )

    class FakeContainer:
        def __init__(self, arguments, **_kwargs) -> None:
            invoked_with.extend(arguments)
            mounts = [
                arguments[index + 1]
                for index, value in enumerate(arguments[:-1])
                if value == "--mount"
            ]
            output_mount = next(value for value in mounts if "dst=/output" in value)
            output_directory = Path(
                output_mount.split("src=", 1)[1].split(",dst=/output", 1)[0]
            )
            (output_directory / "result.json").write_text(
                json.dumps(
                    {
                        "systeminfo": "VULKAN = 1",
                        "result": {"language": "en"},
                        "transcription": [
                            {
                                "offsets": {"from": 0, "to": 1000},
                                "text": " Vulkan ready.",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            self.stderr = io.StringIO(
                "ggml_vulkan: Found 1 Vulkan devices\n"
                "whisper_print_progress: progress = 131%\n"
            )

        @staticmethod
        def wait() -> int:
            return 0

    monkeypatch.setattr("broadcastify_cli.asr.subprocess.Popen", FakeContainer)
    engine = WhisperCppAsr(
        "tiny",
        device="vulkan",
        model_path=model,
        container_image="ghcr.io/ggml-org/whisper.cpp@sha256:test",
        container_runtime="docker",
        container_device=dri,
    )

    messages: list[str] = []
    result = engine.transcribe(audio, progress=messages.append)

    assert invoked_with[:3] == ["docker", "run", "--rm"]
    assert ["--network", "none"] == invoked_with[3:5]
    assert "--read-only" in invoked_with
    assert "no-new-privileges" in invoked_with
    assert ["--cap-drop", "ALL"] == invoked_with[
        invoked_with.index("--cap-drop") : invoked_with.index("--cap-drop") + 2
    ]
    assert str(dri) in invoked_with
    assert invoked_with[invoked_with.index("--entrypoint") + 1] == (
        "/app/build/bin/whisper-cli"
    )
    assert all("dst=/" in value for value in invoked_with if value.startswith("type=bind"))
    assert result.text == "Vulkan ready."
    assert result.backend == "vulkan (container)"
    assert result.metadata["runtime_evidence"] == [
        "ggml_vulkan: Found 1 Vulkan devices"
    ]
    assert "whisper.cpp transcription: 100%" in messages


def test_openvino_offsets_each_bounded_audio_chunk(monkeypatch, tmp_path: Path) -> None:
    class FakePipeline:
        calls = 0

        def generate(self, _samples, **_kwargs):
            self.calls += 1
            return SimpleNamespace(
                texts=[f"chunk {self.calls}"],
                words=[SimpleNamespace(start_ts=1.0, end_ts=2.0, word=" dispatch")],
                chunks=[SimpleNamespace(start_ts=1.0, end_ts=3.0, text="dispatch")],
            )

    engine = object.__new__(OpenVinoWhisperAsr)
    engine.device = "GPU"
    engine._pipeline = FakePipeline()
    engine.model_id = "test-model"
    engine.model_path = tmp_path / "model"
    engine.chunk_seconds = 300
    monkeypatch.setattr(
        engine,
        "_audio_chunks",
        lambda _path: iter([[0.0] * 16000, [0.0] * 32000]),
    )

    result = engine.transcribe(tmp_path / "audio.mp3")

    assert result.text == "chunk 1 chunk 2"
    assert [word.start for word in result.words] == [1.0, 2.0]
    assert [segment.end for segment in result.segments] == [3.0, 4.0]
    assert result.duration == 3.0


def test_openvino_retries_failed_accelerator_on_cpu(monkeypatch, tmp_path: Path) -> None:
    class FailingPipeline:
        @staticmethod
        def generate(_samples, **_kwargs):
            raise RuntimeError("GPU execution failed\ninternal detail")

    class CpuPipeline:
        @staticmethod
        def generate(_samples, **_kwargs):
            return SimpleNamespace(
                texts=["unit responding"],
                words=[],
                chunks=[],
            )

    engine = object.__new__(OpenVinoWhisperAsr)
    engine.device = "GPU"
    engine.backend = "OpenVINO GPU"
    engine._pipeline = FailingPipeline()
    engine._constructor_args = {"word_timestamps": True}
    engine.model_id = "test-model"
    engine.model_path = tmp_path / "model"
    engine.chunk_seconds = 300
    monkeypatch.setattr(engine, "_create_pipeline", lambda device: CpuPipeline())
    monkeypatch.setattr(engine, "_audio_chunks", lambda _path: iter([[0.0] * 16000]))
    messages: list[str] = []

    result = engine.transcribe(tmp_path / "audio.mp3", progress=messages.append)

    assert result.text == "unit responding"
    assert result.backend == "OpenVINO CPU (fallback from GPU)"
    assert result.metadata["fallback_reason"] == "GPU execution failed"
    assert result.metadata["fallback_stage"] == "generation"
    assert engine.device == "CPU"
    assert "OpenVINO GPU rejected this model; retrying on CPU" in messages


def test_openvino_retries_failed_accelerator_initialization_on_cpu(
    monkeypatch, tmp_path: Path
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    calls: list[tuple[str, dict[str, object]]] = []

    class FakePipeline:
        def __init__(self, _path: str, device: str, **kwargs: object) -> None:
            calls.append((device, kwargs))
            if device == "NPU":
                raise RuntimeError("NPU compilation failed\ninternal detail")

    monkeypatch.setitem(
        __import__("sys").modules,
        "openvino_genai",
        SimpleNamespace(WhisperPipeline=FakePipeline),
    )

    engine = OpenVinoWhisperAsr("tiny", device="NPU", model_path=model)

    assert calls == [
        ("NPU", {"word_timestamps": True}),
        ("CPU", {"word_timestamps": True}),
    ]
    assert engine.device == "CPU"
    assert engine.backend == "OpenVINO CPU (fallback from NPU)"
    assert engine._fallback_reason == "NPU compilation failed"
    assert engine._fallback_stage == "initialization"


def test_windows_ml_keeps_one_helper_alive_and_offsets_chunks(
    monkeypatch, tmp_path: Path
) -> None:
    audio = tmp_path / "radio.mp3"
    audio.write_bytes(b"audio")
    model = tmp_path / "model"
    model.mkdir()
    writes: list[str] = []

    class FakeInput:
        def write(self, value: str) -> None:
            writes.append(value)

        @staticmethod
        def flush() -> None:
            pass

        @staticmethod
        def close() -> None:
            pass

    class FakeOutput:
        def __init__(self) -> None:
            self.values = iter(
                [
                    '{"text":"unit responding"}\n',
                    '{"text":"scene secure"}\n',
                ]
            )

        def readline(self) -> str:
            return next(self.values, "")

    class FakeProcess:
        def __init__(self, arguments, **_kwargs) -> None:
            assert arguments[-1] == "--stream"
            self.stdin = FakeInput()
            self.stdout = FakeOutput()

        @staticmethod
        def wait(timeout=None) -> int:
            return 0

        @staticmethod
        def poll():
            return 0

        @staticmethod
        def kill() -> None:
            pass

    monkeypatch.setattr("broadcastify_cli.asr.subprocess.Popen", FakeProcess)
    engine = object.__new__(WindowsMlWhisperAsr)
    engine.helper = str(tmp_path / "helper.exe")
    engine.model_path = model
    engine.model_name = "tiny"
    engine.chunk_seconds = 28
    engine.backend = "Windows ML test"
    monkeypatch.setattr(
        engine,
        "_audio_chunks",
        lambda _path: iter([b"\0\0" * 16_000, b"\0\0" * 32_000]),
    )

    result = engine.transcribe(audio)

    assert result.text == "unit responding scene secure"
    assert [segment.start for segment in result.segments] == [0.0, 1.0]
    assert [segment.end for segment in result.segments] == [1.0, 3.0]
    assert result.duration == 3.0
    assert len([value for value in writes if '"path"' in value]) == 2
    assert writes[-1] == '{"command":"stop"}\n'
