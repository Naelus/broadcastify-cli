from pathlib import Path
from types import SimpleNamespace

from broadcastify_cli.accelerators import (
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

    assert whisper_cpp_backends(executable) == ["cpu", "sycl", "vulkan"]
