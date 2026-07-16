import io
import json
from pathlib import Path
from types import SimpleNamespace

from broadcastify_cli.asr import (
    OpenVinoWhisperAsr,
    WhisperCppAsr,
    WindowsMlWhisperAsr,
    normalize_asr_engine,
)


def test_engine_auto_selection_follows_requested_accelerator() -> None:
    assert normalize_asr_engine("auto", "cuda") == "faster-whisper"
    assert normalize_asr_engine("auto", "vulkan") == "whisper.cpp"
    assert normalize_asr_engine("auto", "openvino-gpu") == "openvino"
    assert normalize_asr_engine("auto", "windows-ml") == "windows-ml"


def test_whisper_cpp_json_is_normalized(monkeypatch, tmp_path: Path) -> None:
    executable = tmp_path / "whisper-cli.exe"
    executable.write_bytes(b"binary")
    (tmp_path / "ggml-vulkan.dll").write_bytes(b"backend")
    model = tmp_path / "ggml-large-v3-turbo-q5_0.bin"
    model.write_bytes(b"model")
    audio = tmp_path / "radio.wav"
    audio.write_bytes(b"audio")

    class FakeProcess:
        def __init__(self, arguments, **_kwargs) -> None:
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
    assert "whisper.cpp transcription: 100%" in messages


def test_whisper_cpp_prepares_non_wav_audio(monkeypatch, tmp_path: Path) -> None:
    executable = tmp_path / "whisper-cli.exe"
    executable.write_bytes(b"binary")
    model = tmp_path / "ggml-tiny.en-q5_1.bin"
    model.write_bytes(b"model")
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
    assert engine.device == "CPU"
    assert "OpenVINO GPU rejected this model; retrying on CPU" in messages


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
