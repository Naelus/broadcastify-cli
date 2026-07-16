from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def find_whisper_cpp() -> str | None:
    configured = os.getenv("WHISPER_CPP_PATH")
    if configured and Path(configured).is_file():
        return str(Path(configured).resolve())

    for name in ("whisper-cli", "whisper-cli.exe"):
        discovered = shutil.which(name)
        if discovered:
            return discovered

    candidates = [
        Path.cwd() / "tools" / "whisper.cpp" / "whisper-cli",
        Path.cwd() / "tools" / "whisper.cpp" / "build" / "bin" / "whisper-cli",
        Path.cwd() / "whisper.cpp" / "build" / "bin" / "whisper-cli",
        Path.cwd() / "tools" / "whisper.cpp" / "whisper-cli.exe",
        Path.cwd() / "whisper.cpp" / "build" / "bin" / "Release" / "whisper-cli.exe",
        Path.cwd() / "whisper.cpp" / "build" / "bin" / "whisper-cli.exe",
    ]
    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        package_root = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
        try:
            candidates.extend(package_root.glob("*whisper*/*/whisper-cli.exe"))
            candidates.extend(package_root.glob("*whisper*/whisper-cli.exe"))
        except OSError:
            pass
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    return None


def find_container_runtime(explicit: str | Path | None = None) -> str | None:
    configured = str(
        explicit
        or os.getenv("WHISPER_CPP_CONTAINER_RUNTIME")
        or os.getenv("BROADCASTIFY_CONTAINER_RUNTIME")
        or ""
    ).strip()
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file():
            return str(candidate.resolve())
        discovered = shutil.which(configured)
        return discovered
    for name in ("podman", "docker"):
        discovered = shutil.which(name)
        if discovered:
            return discovered
    return None


def whisper_cpp_container_diagnostics(
    *,
    image: str | None = None,
    runtime: str | Path | None = None,
    backend: str | None = None,
    device: str | Path | None = None,
) -> dict[str, Any]:
    """Inspect an explicitly configured image without pulling or starting it."""

    image_name = str(image or os.getenv("WHISPER_CPP_CONTAINER_IMAGE") or "").strip()
    backend_name = str(
        backend or os.getenv("WHISPER_CPP_CONTAINER_BACKEND") or "vulkan"
    ).strip().lower()
    runtime_path = find_container_runtime(runtime)
    device_path = Path(
        device or os.getenv("WHISPER_CPP_CONTAINER_DEVICE") or "/dev/dri"
    )
    result: dict[str, Any] = {
        "configured": bool(image_name),
        "image": image_name,
        "runtime": runtime_path,
        "backend": backend_name,
        "device": str(device_path),
        "image_present": False,
        "device_present": backend_name != "vulkan" or device_path.exists(),
        "ready": False,
    }
    if not image_name or not runtime_path:
        return result
    try:
        inspected = subprocess.run(
            [runtime_path, "image", "inspect", "--format", "{{.Id}}", image_name],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        result["image_present"] = inspected.returncode == 0
        if inspected.returncode == 0:
            result["image_id"] = inspected.stdout.strip()
        elif inspected.stderr.strip():
            result["error"] = inspected.stderr.strip().splitlines()[-1]
    except (OSError, subprocess.SubprocessError) as exc:
        result["error"] = str(exc)
    result["ready"] = bool(result["image_present"] and result["device_present"])
    return result


def find_windows_ml_helper() -> str | None:
    configured = os.getenv("WINDOWS_ML_HELPER_PATH")
    if configured and Path(configured).is_file():
        return str(Path(configured).resolve())
    candidates = [
        Path.cwd()
        / "BroadcastifyCli.WindowsML"
        / "bin"
        / "Release"
        / "net10.0-windows10.0.26100.0"
        / "win-x64"
        / "BroadcastifyCli.WindowsML.exe",
        Path.cwd()
        / "BroadcastifyCli.WindowsML"
        / "bin"
        / "Debug"
        / "net10.0-windows10.0.26100.0"
        / "win-x64"
        / "BroadcastifyCli.WindowsML.exe",
        Path(sys.executable).resolve().parent / "windowsml" / "BroadcastifyCli.WindowsML.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    return None


def find_windows_ml_model() -> str | None:
    configured = os.getenv("WINDOWS_ML_WHISPER_MODEL_PATH")
    if configured and (Path(configured) / "genai_config.json").is_file():
        return str(Path(configured).resolve())
    return None


def _windows_ml_diagnostics() -> dict[str, Any]:
    helper = find_windows_ml_helper()
    model = find_windows_ml_model()
    value: dict[str, Any] = {
        "helper": helper,
        "model": model,
        "runtime_ready": False,
        "decode_ready": False,
    }
    if not helper:
        return value
    arguments = [helper, "--probe"]
    if model:
        arguments.extend(["--model", model])
    try:
        result = subprocess.run(
            arguments,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        payload = json.loads(lines[-1]) if lines else {}
        value["runtime_ready"] = bool(payload.get("ready"))
        value["decode_ready"] = bool(payload.get("decode_ready"))
        value["backend"] = str(payload.get("backend") or "")
        if result.returncode != 0 or payload.get("error"):
            value["error"] = str(payload.get("error") or result.stderr.strip())
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        value["error"] = str(exc)
    return value


def whisper_cpp_backends(executable: str | Path | None) -> list[str]:
    if not executable:
        return []
    directory = Path(executable).resolve().parent
    names: set[str] = set()
    try:
        files = {item.name.lower() for item in directory.iterdir() if item.is_file()}
    except OSError:
        files = set()
    if any("vulkan" in name for name in files):
        names.add("vulkan")
    if any("cuda" in name or "cublas" in name for name in files):
        names.add("cuda")
    if any("hip" in name or "rocm" in name for name in files):
        names.add("hip")
    if any("sycl" in name for name in files):
        names.add("sycl")
    if any("openvino" in name for name in files):
        names.add("openvino")
    if any("metal" in name for name in files):
        names.add("metal")
    if sys.platform == "darwin" and "metal" not in names:
        try:
            linked = subprocess.run(
                ["otool", "-L", str(Path(executable).resolve())],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            if re.search(r"(?:Metal\.framework|ggml-metal)", linked.stdout, re.IGNORECASE):
                names.add("metal")
        except (OSError, subprocess.SubprocessError):
            pass
    # Every official build retains a CPU backend even when it also offloads.
    names.add("cpu")
    return sorted(names)


def inspect_llama_devices(executable: str | Path | None) -> list[dict[str, Any]]:
    if not executable:
        return []
    try:
        result = subprocess.run(
            [str(executable), "--list-devices"],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    output = "\n".join(value for value in (result.stdout, result.stderr) if value)
    devices: list[dict[str, Any]] = []
    pattern = re.compile(
        r"^\s*(?P<id>[A-Za-z][A-Za-z0-9_.-]*):\s*(?P<name>.*?)"
        r"(?:\s*\((?P<memory>\d+)\s*MiB(?:,\s*(?P<free>\d+)\s*MiB free)?\))?\s*$"
    )
    for line in output.splitlines():
        match = pattern.match(line)
        if not match or match.group("id").lower() == "available":
            continue
        identifier = match.group("id")
        backend_match = re.match(r"[A-Za-z]+", identifier)
        devices.append(
            {
                "id": identifier,
                "backend": backend_match.group(0).lower() if backend_match else "unknown",
                "name": match.group("name").strip(),
                "memory_mib": int(match.group("memory")) if match.group("memory") else None,
                "free_mib": int(match.group("free")) if match.group("free") else None,
            }
        )
    return devices


def _torch_diagnostics() -> dict[str, Any]:
    value: dict[str, Any] = {
        "installed": module_available("torch"),
        "cuda_available": False,
        "cuda_devices": [],
        "xpu_available": False,
        "xpu_devices": [],
    }
    if not value["installed"]:
        return value
    try:
        import torch

        value["version"] = str(torch.__version__)
        value["cuda_available"] = bool(torch.cuda.is_available())
        if value["cuda_available"]:
            value["cuda_devices"] = [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
        xpu = getattr(torch, "xpu", None)
        value["xpu_available"] = bool(xpu and xpu.is_available())
        if value["xpu_available"]:
            value["xpu_devices"] = [
                xpu.get_device_name(index) for index in range(xpu.device_count())
            ]
    except Exception as exc:  # Diagnostics must not make the app fail to start.
        value["error"] = str(exc)
    return value


def _openvino_diagnostics() -> dict[str, Any]:
    value: dict[str, Any] = {
        "runtime_installed": module_available("openvino"),
        "genai_installed": module_available("openvino_genai"),
        "devices": [],
        "device_details": [],
    }
    if not value["runtime_installed"]:
        return value
    try:
        import openvino as ov

        value["version"] = str(getattr(ov, "__version__", ""))
        core = ov.Core()
        value["devices"] = list(core.available_devices)
        for device in value["devices"]:
            try:
                name = str(core.get_property(device, "FULL_DEVICE_NAME"))
            except Exception:
                name = str(device)
            value["device_details"].append({"id": str(device), "name": name})
    except Exception as exc:
        value["error"] = str(exc)
    return value


def _onnx_diagnostics() -> dict[str, Any]:
    value: dict[str, Any] = {
        "installed": module_available("onnxruntime"),
        "providers": [],
        "ep_devices": [],
        "windows_ml_api_installed": module_available("winui3"),
    }
    if not value["installed"]:
        return value
    try:
        import onnxruntime as ort

        value["version"] = str(getattr(ort, "__version__", ""))
        value["providers"] = list(ort.get_available_providers())
        get_ep_devices = getattr(ort, "get_ep_devices", None)
        if get_ep_devices:
            for device in get_ep_devices():
                hardware = getattr(device, "hardware_device", None) or getattr(
                    device, "device", None
                )
                value["ep_devices"].append(
                    {
                        "name": str(device.ep_name),
                        "type": str(getattr(hardware, "type", "unknown")),
                    }
                )
    except Exception as exc:
        value["error"] = str(exc)
    return value


def _profile(
    identifier: str,
    name: str,
    ready: bool,
    transcription: str,
    diarization: str,
    analysis: str,
    note: str = "",
) -> dict[str, Any]:
    return {
        "id": identifier,
        "name": name,
        "ready": ready,
        "transcription": transcription,
        "diarization": diarization,
        "analysis": analysis,
        "note": note,
    }


def collect_accelerator_diagnostics(llama_server: str | Path | None) -> dict[str, Any]:
    torch = _torch_diagnostics()
    openvino = _openvino_diagnostics()
    onnx = _onnx_diagnostics()
    windows_ml = _windows_ml_diagnostics()
    whisper_executable = find_whisper_cpp()
    whisper_backends = whisper_cpp_backends(whisper_executable)
    whisper_container = whisper_cpp_container_diagnostics()
    llama_devices = inspect_llama_devices(llama_server)
    llama_backends = {str(value.get("backend") or "") for value in llama_devices}
    token_ready = bool(os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN"))
    pyannote_ready = module_available("pyannote.audio") and token_ready
    faster_whisper_ready = module_available("faster_whisper")

    cuda_ready = bool(torch["cuda_available"] and faster_whisper_ready)
    if whisper_container["configured"]:
        vulkan_asr = bool(
            whisper_container["ready"]
            and whisper_container["backend"] == "vulkan"
        )
    else:
        vulkan_asr = bool(whisper_executable and "vulkan" in whisper_backends)
    vulkan_llm = "vulkan" in llama_backends
    metal_asr = bool(whisper_executable and "metal" in whisper_backends)
    metal_llm = "metal" in llama_backends
    openvino_devices = [str(value).upper() for value in openvino.get("devices", [])]
    openvino_device_names = [
        f"{value.get('id')}: {value.get('name')}"
        for value in openvino.get("device_details", [])
    ]
    openvino_asr = bool(openvino["genai_installed"] and openvino_devices)
    windows_ml_runtime = bool(windows_ml["runtime_ready"])
    windows_ml_decode = bool(windows_ml["decode_ready"])
    cpu_ready = bool(faster_whisper_ready and module_available("torch"))
    cpu_diarization = "pyannote CPU ready" if pyannote_ready else "pyannote needs its package and HF token"
    llama_ready = bool(llama_server)

    if cuda_ready:
        automatic_ready = cuda_ready and pyannote_ready and llama_ready
        automatic_transcription = "faster-whisper on CUDA"
        automatic_diarization = "pyannote on CUDA" if pyannote_ready else cpu_diarization
        automatic_analysis = "llama.cpp " + next(iter(sorted(llama_backends)), "CPU")
        automatic_note = "Uses the validated Windows CUDA path and keeps per-stage fallbacks explicit."
    elif sys.platform == "darwin" and metal_asr and metal_llm:
        automatic_ready = pyannote_ready and llama_ready
        automatic_transcription = "whisper.cpp on Apple Metal"
        automatic_diarization = cpu_diarization
        automatic_analysis = "llama.cpp / Metal"
        automatic_note = "Uses native Apple Metal for transcription and analysis; diarization stays on CPU."
    elif sys.platform.startswith("linux") and vulkan_asr and vulkan_llm:
        automatic_ready = pyannote_ready and llama_ready
        automatic_transcription = "whisper.cpp on Vulkan"
        automatic_diarization = cpu_diarization
        automatic_analysis = "llama.cpp / Vulkan"
        automatic_note = "Uses detected Vulkan runtimes for transcription and analysis; diarization stays on CPU."
    else:
        automatic_ready = cpu_ready and pyannote_ready and llama_ready
        automatic_transcription = "faster-whisper on CPU" if cpu_ready else "CPU ASR dependencies unavailable"
        automatic_diarization = cpu_diarization
        automatic_analysis = "llama.cpp / CPU" if llama_ready else "llama.cpp missing"
        automatic_note = "Uses the dependable CPU fallback because no validated accelerator pair was detected."

    profiles = [
        _profile(
            "auto",
            "Automatic (recommended)",
            automatic_ready,
            automatic_transcription,
            automatic_diarization,
            automatic_analysis,
            automatic_note,
        ),
        _profile(
            "cuda",
            "NVIDIA CUDA",
            cuda_ready and pyannote_ready and llama_ready,
            "faster-whisper / CUDA" if cuda_ready else "CUDA ASR dependencies unavailable",
            "pyannote / CUDA" if torch["cuda_available"] and pyannote_ready else cpu_diarization,
            "llama.cpp auto-offload" if llama_ready else "llama.cpp missing",
        ),
        _profile(
            "vulkan",
            "Cross-vendor Vulkan",
            vulkan_asr and pyannote_ready and vulkan_llm,
            "whisper.cpp / Vulkan" if vulkan_asr else "needs a Vulkan whisper.cpp build",
            cpu_diarization,
            "llama.cpp / Vulkan" if vulkan_llm else "needs a Vulkan llama.cpp build",
            "Diarization intentionally falls back to CPU because pyannote has no Vulkan backend.",
        ),
        _profile(
            "openvino",
            "OpenVINO runtime",
            openvino_asr and pyannote_ready and llama_ready,
            (
                "OpenVINO Whisper / "
                + ", ".join(openvino_device_names or openvino_devices)
                + " (safe CPU fallback)"
                if openvino_asr
                else "needs the OpenVINO GenAI optional package"
            ),
            cpu_diarization,
            (
                "llama.cpp / SYCL"
                if "sycl" in llama_backends
                else "llama.cpp auto-offload or CPU"
            ),
            "Uses devices exposed by the installed OpenVINO runtime; unsupported model/device combinations retry on CPU.",
        ),
        _profile(
            "windowsml",
            "Windows ML",
            windows_ml_decode and pyannote_ready and llama_ready,
            (
                f"{windows_ml.get('backend') or 'Windows ML ONNX Whisper'} (validated model)"
                if windows_ml_decode
                else "runtime detected; configure and validate an ONNX Whisper model"
                if windows_ml_runtime
                else "needs the Windows ML helper and a compatible ONNX Whisper model"
            ),
            cpu_diarization,
            "llama.cpp auto-offload or CPU",
            (
                "The configured model passed an actual silent-audio decode self-test. "
                "Diarization uses the dependable CPU fallback."
                if windows_ml_decode
                else "A runtime-only probe is not enough; this profile remains unavailable until a model decode passes."
            ),
        ),
    ]
    if sys.platform == "darwin":
        profiles.append(
            _profile(
                "metal",
                "Apple Metal",
                metal_asr and pyannote_ready and metal_llm,
                "whisper.cpp / Metal" if metal_asr else "needs a native ggml-metal whisper.cpp build",
                cpu_diarization,
                "llama.cpp / Metal" if metal_llm else "needs a Metal llama.cpp build",
                "Apple GPU acceleration is native-only; pyannote diarization intentionally uses CPU.",
            )
        )
    profiles.append(
        _profile(
            "cpu",
            "CPU only",
            cpu_ready and pyannote_ready and llama_ready,
            "faster-whisper / INT8 CPU" if cpu_ready else "CPU ASR dependencies unavailable",
            cpu_diarization,
            "llama.cpp / CPU" if llama_ready else "llama.cpp missing",
            "Slowest but portable and a dependable fallback for every stage.",
        )
    )

    return {
        "python": sys.version.split()[0],
        "torch": torch,
        "openvino": openvino,
        "onnx": onnx,
        "windows_ml": windows_ml,
        "whisper_cpp": {
            "executable": whisper_executable,
            "backends": whisper_backends,
            "container": whisper_container,
        },
        "llama_cpp": {
            "executable": str(llama_server) if llama_server else None,
            "devices": llama_devices,
        },
        "profiles": profiles,
    }
