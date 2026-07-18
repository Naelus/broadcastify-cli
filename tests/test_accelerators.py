from pathlib import Path
from types import SimpleNamespace

from broadcastify_cli.accelerators import (
    collect_accelerator_diagnostics,
    find_whisper_cpp,
    huggingface_model_cached,
    inspect_llama_devices,
    whisper_cpp_backends,
)


def test_llama_device_output_is_normalized(monkeypatch) -> None:
    monkeypatch.setattr(
        "broadcastify_cli.accelerators.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout=(
                "Available devices:\n"
                "  Vulkan0: AMD Radeon RX 7900 XTX (24560 MiB, 23000 MiB free)\n"
                "  SYCL1: Intel Arc Graphics (8192 MiB, 7000 MiB free)\n"
            ),
            stderr="",
        ),
    )

    devices = inspect_llama_devices("llama-server.exe")

    assert devices == [
        {
            "id": "Vulkan0",
            "backend": "vulkan",
            "name": "AMD Radeon RX 7900 XTX",
            "memory_mib": 24560,
            "free_mib": 23000,
        },
        {
            "id": "SYCL1",
            "backend": "sycl",
            "name": "Intel Arc Graphics",
            "memory_mib": 8192,
            "free_mib": 7000,
        },
    ]


def test_whisper_cpp_backend_dlls_are_detected(tmp_path: Path) -> None:
    executable = tmp_path / "whisper-cli.exe"
    executable.write_bytes(b"binary")
    (tmp_path / "ggml-vulkan.dll").write_bytes(b"vulkan")
    (tmp_path / "ggml-sycl.dll").write_bytes(b"sycl")
    (tmp_path / "libggml-metal.dylib").write_bytes(b"metal")

    assert whisper_cpp_backends(executable) == ["cpu", "metal", "sycl", "vulkan"]


def test_whisper_cpp_finds_portable_local_build(monkeypatch, tmp_path: Path) -> None:
    executable = tmp_path / "tools" / "whisper.cpp" / "build" / "bin" / "whisper-cli"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"binary")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("WHISPER_CPP_PATH", raising=False)
    monkeypatch.setattr("broadcastify_cli.accelerators.shutil.which", lambda _name: None)

    assert find_whisper_cpp() == str(executable.resolve())


def test_huggingface_model_cache_detection_requires_expected_snapshot_file(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "hub"
    snapshot = (
        cache
        / "models--pyannote--speaker-diarization-community-1"
        / "snapshots"
        / "test-revision"
    )
    snapshot.mkdir(parents=True)

    assert not huggingface_model_cached(
        "pyannote/speaker-diarization-community-1",
        cache_root=cache,
    )

    (snapshot / "config.yaml").write_text("version: test\n", encoding="utf-8")

    assert huggingface_model_cached(
        "pyannote/speaker-diarization-community-1",
        cache_root=cache,
    )


def test_cached_pyannote_model_counts_as_configured_without_token(
    monkeypatch, tmp_path: Path
) -> None:
    snapshot = (
        tmp_path
        / "models--pyannote--speaker-diarization-community-1"
        / "snapshots"
        / "offline-revision"
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.yaml").write_text("version: test\n", encoding="utf-8")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("WHISPER_CPP_CONTAINER_IMAGE", raising=False)
    monkeypatch.setattr(
        "broadcastify_cli.accelerators._torch_diagnostics",
        lambda: {"installed": True, "cuda_available": False, "cuda_devices": []},
    )
    monkeypatch.setattr(
        "broadcastify_cli.accelerators._openvino_diagnostics",
        lambda: {"runtime_installed": False, "genai_installed": False, "devices": []},
    )
    monkeypatch.setattr(
        "broadcastify_cli.accelerators._onnx_diagnostics",
        lambda: {"installed": False},
    )
    monkeypatch.setattr(
        "broadcastify_cli.accelerators._windows_ml_diagnostics",
        lambda: {"runtime_ready": False, "decode_ready": False},
    )
    monkeypatch.setattr("broadcastify_cli.accelerators.find_whisper_cpp", lambda: None)
    monkeypatch.setattr(
        "broadcastify_cli.accelerators.inspect_llama_devices",
        lambda _path: [{"id": "CPU", "backend": "cpu", "name": "CPU"}],
    )
    monkeypatch.setattr("broadcastify_cli.accelerators.module_available", lambda _name: True)

    diagnostics = collect_accelerator_diagnostics("/bin/llama-server")
    speakers = diagnostics["speaker_labels"]
    cpu = next(value for value in diagnostics["profiles"] if value["id"] == "cpu")

    assert speakers["token_configured"] is False
    assert speakers["model_cache_detected"] is True
    assert speakers["access_configured"] is True
    assert speakers["configured"] is True
    assert cpu["configured"] is True
    assert cpu["ready"] is False
    assert cpu["diarization_ready"] is True
    assert "cached model" in cpu["diarization"]


def test_vulkan_profile_requires_vulkan_llama_backend(monkeypatch) -> None:
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "test-token")
    monkeypatch.delenv("WHISPER_CPP_CONTAINER_IMAGE", raising=False)
    monkeypatch.setattr(
        "broadcastify_cli.accelerators._torch_diagnostics",
        lambda: {"installed": True, "cuda_available": False, "cuda_devices": []},
    )
    monkeypatch.setattr(
        "broadcastify_cli.accelerators._openvino_diagnostics",
        lambda: {"runtime_installed": False, "genai_installed": False, "devices": []},
    )
    monkeypatch.setattr(
        "broadcastify_cli.accelerators._onnx_diagnostics", lambda: {"installed": False}
    )
    monkeypatch.setattr(
        "broadcastify_cli.accelerators._windows_ml_diagnostics",
        lambda: {"runtime_ready": False, "decode_ready": False},
    )
    monkeypatch.setattr("broadcastify_cli.accelerators.find_whisper_cpp", lambda: "/bin/whisper-cli")
    monkeypatch.setattr(
        "broadcastify_cli.accelerators.whisper_cpp_backends",
        lambda _path: ["cpu", "vulkan"],
    )
    monkeypatch.setattr(
        "broadcastify_cli.accelerators.inspect_llama_devices",
        lambda _path: [{"id": "CPU", "backend": "cpu", "name": "CPU"}],
    )
    monkeypatch.setattr("broadcastify_cli.accelerators.module_available", lambda _name: True)

    diagnostics = collect_accelerator_diagnostics("/bin/llama-server")
    vulkan = next(value for value in diagnostics["profiles"] if value["id"] == "vulkan")

    assert vulkan["ready"] is False
    assert vulkan["transcription_ready"] is True
    assert vulkan["diarization_ready"] is True
    assert vulkan["analysis_ready"] is False
    assert vulkan["analysis"] == "needs a Vulkan llama.cpp build"


def test_explicit_missing_container_does_not_fall_back_to_native_profile(
    monkeypatch,
) -> None:
    monkeypatch.setenv("WHISPER_CPP_CONTAINER_IMAGE", "local/missing-vulkan")
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "test-token")
    monkeypatch.setattr(
        "broadcastify_cli.accelerators._torch_diagnostics",
        lambda: {"installed": True, "cuda_available": False, "cuda_devices": []},
    )
    monkeypatch.setattr(
        "broadcastify_cli.accelerators._openvino_diagnostics",
        lambda: {"runtime_installed": False, "genai_installed": False, "devices": []},
    )
    monkeypatch.setattr(
        "broadcastify_cli.accelerators._onnx_diagnostics", lambda: {"installed": False}
    )
    monkeypatch.setattr(
        "broadcastify_cli.accelerators._windows_ml_diagnostics",
        lambda: {"runtime_ready": False, "decode_ready": False},
    )
    monkeypatch.setattr("broadcastify_cli.accelerators.find_whisper_cpp", lambda: "/bin/whisper-cli")
    monkeypatch.setattr(
        "broadcastify_cli.accelerators.whisper_cpp_backends",
        lambda _path: ["cpu", "vulkan"],
    )
    monkeypatch.setattr(
        "broadcastify_cli.accelerators.whisper_cpp_container_diagnostics",
        lambda: {
            "configured": True,
            "ready": False,
            "backend": "vulkan",
        },
    )
    monkeypatch.setattr(
        "broadcastify_cli.accelerators.inspect_llama_devices",
        lambda _path: [{"id": "Vulkan0", "backend": "vulkan", "name": "GPU"}],
    )
    monkeypatch.setattr("broadcastify_cli.accelerators.module_available", lambda _name: True)

    diagnostics = collect_accelerator_diagnostics("/bin/llama-server")
    vulkan = next(value for value in diagnostics["profiles"] if value["id"] == "vulkan")

    assert vulkan["ready"] is False
    assert vulkan["transcription"] == "needs a Vulkan whisper.cpp build"


def test_macos_profile_requires_and_reports_both_metal_engines(monkeypatch) -> None:
    monkeypatch.setattr("broadcastify_cli.accelerators.sys.platform", "darwin")
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "test-token")
    monkeypatch.delenv("WHISPER_CPP_CONTAINER_IMAGE", raising=False)
    monkeypatch.setattr(
        "broadcastify_cli.accelerators._torch_diagnostics",
        lambda: {"installed": True, "cuda_available": False, "cuda_devices": []},
    )
    monkeypatch.setattr(
        "broadcastify_cli.accelerators._openvino_diagnostics",
        lambda: {"runtime_installed": False, "genai_installed": False, "devices": []},
    )
    monkeypatch.setattr(
        "broadcastify_cli.accelerators._onnx_diagnostics", lambda: {"installed": False}
    )
    monkeypatch.setattr(
        "broadcastify_cli.accelerators._windows_ml_diagnostics",
        lambda: {"runtime_ready": False, "decode_ready": False},
    )
    monkeypatch.setattr(
        "broadcastify_cli.accelerators.find_whisper_cpp", lambda: "/opt/whisper-cli"
    )
    monkeypatch.setattr(
        "broadcastify_cli.accelerators.whisper_cpp_backends",
        lambda _path: ["cpu", "metal"],
    )
    monkeypatch.setattr(
        "broadcastify_cli.accelerators.inspect_llama_devices",
        lambda _path: [{"id": "Metal0", "backend": "metal", "name": "Apple GPU"}],
    )
    monkeypatch.setattr(
        "broadcastify_cli.accelerators.module_available", lambda _name: True
    )

    diagnostics = collect_accelerator_diagnostics("/opt/llama-server")
    automatic = next(value for value in diagnostics["profiles"] if value["id"] == "auto")
    metal = next(value for value in diagnostics["profiles"] if value["id"] == "metal")

    assert automatic["configured"] is True
    assert automatic["ready"] is False
    assert automatic["transcription_ready"] is True
    assert automatic["diarization_ready"] is True
    assert automatic["analysis_ready"] is True
    assert automatic["transcription"] == "whisper.cpp on Apple Metal"
    assert metal["configured"] is True
    assert metal["ready"] is False
    assert metal["analysis"] == "llama.cpp / Metal"
    assert len(metal["next_steps"]) == 3
