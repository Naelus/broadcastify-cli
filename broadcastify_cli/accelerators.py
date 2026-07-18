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


PYANNOTE_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"


def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def huggingface_model_cached(
    repo_id: str,
    *,
    cache_root: str | Path | None = None,
    required_files: tuple[str, ...] = ("config.yaml",),
) -> bool:
    """Cheaply detect a usable-looking Hub snapshot without loading a model.

    This is intentionally only configuration evidence. The explicit model
    self-test remains the execution proof and catches incomplete dependencies.
    """

    roots: list[Path] = []
    if cache_root is not None:
        roots.append(Path(cache_root).expanduser())
    else:
        configured = (
            os.getenv("HF_HUB_CACHE")
            or os.getenv("HUGGINGFACE_HUB_CACHE")
            or ""
        ).strip()
        if configured:
            roots.append(Path(configured).expanduser())
        hf_home = str(os.getenv("HF_HOME") or "").strip()
        if hf_home:
            roots.append(Path(hf_home).expanduser() / "hub")
        roots.append(Path.home() / ".cache" / "huggingface" / "hub")

    repository_directory = "models--" + repo_id.replace("/", "--")
    for root in dict.fromkeys(path.resolve() for path in roots):
        snapshots = root / repository_directory / "snapshots"
        try:
            candidates = [path for path in snapshots.iterdir() if path.is_dir()]
        except OSError:
            continue
        for snapshot in candidates:
            if all((snapshot / relative).is_file() for relative in required_files):
                return True
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
    local_app_data = os.getenv("LOCALAPPDATA")
    managed_root = (
        Path(local_app_data) / "Broadcastify Desktop" / "models"
        if local_app_data
        else Path.home() / ".cache" / "broadcastify-desktop" / "models"
    )
    roots = [
        managed_root / "windowsml",
        Path.cwd() / "models" / "windowsml",
        Path.cwd() / ".models" / "windowsml",
    ]
    for root in roots:
        try:
            candidates = sorted(root.glob("*/genai_config.json"))
        except OSError:
            continue
        if candidates:
            return str(candidates[0].parent.resolve())
    return None


_MODEL_PATH_UNSET = object()


def _windows_ml_diagnostics(
    model_path: str | Path | None | object = _MODEL_PATH_UNSET,
) -> dict[str, Any]:
    helper = find_windows_ml_helper()
    model = (
        find_windows_ml_model()
        if model_path is _MODEL_PATH_UNSET
        else str(model_path)
        if model_path
        else None
    )
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


def find_whisper_cpp_model_file() -> str | None:
    configured = os.getenv("WHISPER_CPP_MODEL_PATH")
    if configured and Path(configured).is_file():
        return str(Path(configured).resolve())
    local_app_data = os.getenv("LOCALAPPDATA")
    managed_root = (
        Path(local_app_data) / "Broadcastify Desktop" / "models"
        if local_app_data
        else Path.home() / ".cache" / "broadcastify-desktop" / "models"
    )
    roots = [
        managed_root / "whisper.cpp",
        Path.cwd() / "models",
        Path.cwd() / "whisper.cpp" / "models",
    ]
    for root in roots:
        try:
            candidates = sorted(
                path for path in root.glob("ggml-*.bin") if path.is_file()
            )
        except OSError:
            continue
        if candidates:
            return str(candidates[0].resolve())
    return None


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
    configured: bool,
    transcription: str,
    diarization: str,
    analysis: str,
    note: str = "",
    *,
    transcription_ready: bool | None = None,
    diarization_ready: bool | None = None,
    analysis_ready: bool | None = None,
    transcription_setup: str = "Configure a supported transcription engine, then run Test engine.",
    diarization_setup: str = "Configure pyannote Community-1, then run Test speakers.",
    analysis_setup: str = "Configure llama.cpp or another analysis provider, then run Test analysis.",
    transcription_action: dict[str, str] | None = None,
    diarization_action: dict[str, str] | None = None,
    analysis_action: dict[str, str] | None = None,
) -> dict[str, Any]:
    transcription_configured = (
        configured if transcription_ready is None else transcription_ready
    )
    diarization_configured = (
        configured if diarization_ready is None else diarization_ready
    )
    analysis_configured = configured if analysis_ready is None else analysis_ready
    next_steps = [
        (
            "Run Verify profile (or Test engine) to prove this exact engine, model, and device."
            if transcription_configured
            else transcription_setup
        ),
        (
            "Run Verify profile (or Test speakers) to prove the selected speaker-label engine."
            if diarization_configured
            else diarization_setup
        ),
        (
            "Run Verify profile (or Test analysis) to load the selected model and generate a local synthetic result."
            if analysis_configured
            else analysis_setup
        ),
    ]
    if not transcription_configured:
        next_action = transcription_action or {
            "stage": "transcription",
            "kind": "configure-transcription",
            "label": "Set up transcription",
            "message": transcription_setup,
        }
    elif not diarization_configured:
        next_action = diarization_action or {
            "stage": "diarization",
            "kind": "configure-speakers",
            "label": "Set up speaker labels",
            "message": diarization_setup,
        }
    elif not analysis_configured:
        next_action = analysis_action or {
            "stage": "analysis",
            "kind": "configure-analysis",
            "label": "Set up analysis",
            "message": analysis_setup,
        }
    else:
        next_action = {
            "stage": "profile",
            "kind": "verify-profile",
            "label": "Verify profile",
            "message": (
                "All three stages are configured. Run Verify profile to execute "
                "the selected models with generated local input."
            ),
        }
    return {
        "id": identifier,
        "name": name,
        # Diagnostics never execute every model stage, so a profile cannot be
        # called verified/ready from this cheap inspection alone.
        "ready": False,
        "configured": configured,
        "verified": False,
        "transcription_ready": transcription_configured,
        "diarization_ready": diarization_configured,
        "analysis_ready": analysis_configured,
        "transcription": transcription,
        "diarization": diarization,
        "analysis": analysis,
        "note": note,
        "next_steps": next_steps,
        "next_action": next_action,
    }


def collect_accelerator_diagnostics(
    llama_server: str | Path | None,
    *,
    selected_asr_engine: str | None = None,
    selected_whisper_model: str | Path | None = None,
    selected_windows_model: str | Path | None = None,
    selected_qwen_model: str | Path | None = None,
    selected_diarization_engine: str | None = None,
) -> dict[str, Any]:
    from .portable_diarization import (
        COMMUNITY_DIARIZATION_ENGINE,
        PORTABLE_DIARIZATION_ENGINE,
        normalize_diarization_engine,
        portable_diarization_diagnostics,
    )
    from .qwen_asr import qwen3_asr_diagnostics

    torch = _torch_diagnostics()
    openvino = _openvino_diagnostics()
    onnx = _onnx_diagnostics()
    windows_ml = (
        _windows_ml_diagnostics(selected_windows_model)
        if selected_asr_engine == "windows-ml"
        else _windows_ml_diagnostics()
    )
    qwen3_asr = qwen3_asr_diagnostics(
        selected_qwen_model if selected_asr_engine == "qwen3-asr" else None
    )
    whisper_executable = find_whisper_cpp()
    if selected_asr_engine == "whisper.cpp":
        whisper_model = (
            str(selected_whisper_model) if selected_whisper_model else None
        )
    else:
        whisper_model = find_whisper_cpp_model_file()
    whisper_backends = whisper_cpp_backends(whisper_executable)
    whisper_container = whisper_cpp_container_diagnostics()
    llama_devices = inspect_llama_devices(llama_server)
    llama_backends = {str(value.get("backend") or "") for value in llama_devices}
    token_ready = bool(os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN"))
    pyannote_cache_detected = huggingface_model_cached(PYANNOTE_DIARIZATION_MODEL)
    pyannote_package_installed = module_available("pyannote.audio")
    pyannote_access_configured = token_ready or pyannote_cache_detected
    pyannote_ready = pyannote_package_installed and pyannote_access_configured
    selected_speaker_engine = normalize_diarization_engine(
        selected_diarization_engine or COMMUNITY_DIARIZATION_ENGINE
    )
    portable_speakers = portable_diarization_diagnostics()
    faster_whisper_ready = module_available("faster_whisper")

    cuda_ready = bool(torch["cuda_available"] and faster_whisper_ready)
    if whisper_container["configured"]:
        vulkan_runtime = bool(
            whisper_container["ready"]
            and whisper_container["backend"] == "vulkan"
        )
        vulkan_asr = bool(
            vulkan_runtime and whisper_model
        )
    else:
        vulkan_runtime = bool(
            whisper_executable and "vulkan" in whisper_backends
        )
        vulkan_asr = bool(
            vulkan_runtime and whisper_model
        )
    vulkan_llm = "vulkan" in llama_backends
    metal_runtime = bool(whisper_executable and "metal" in whisper_backends)
    metal_asr = bool(metal_runtime and whisper_model)
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
    portable_ready = bool(portable_speakers["ready"])
    portable_diarization = (
        "sherpa-onnx CPU fast preview configured"
        if portable_ready
        else "sherpa-onnx runtime installed; pinned preview models need Test speakers"
        if portable_speakers["runtime_installed"]
        else "sherpa-onnx portable speaker runtime is not installed"
    )
    portable_diarization_setup = (
        'Install `pip install -e ".[portable-diarization]"`, select Fast '
        "portable preview, then run Test speakers."
        if not portable_speakers["runtime_installed"]
        else "Run Test speakers to checksum-verify the public segmentation and embedding models."
    )
    if pyannote_ready:
        access_source = (
            "cached model" if pyannote_cache_detected else "first-download token"
        )
        community_cpu_diarization = (
            f"pyannote Community-1 CPU configured ({access_source})"
        )
    elif pyannote_package_installed:
        community_cpu_diarization = (
            "pyannote installed; Community-1 needs a first-download token "
            "or complete cache"
        )
    else:
        community_cpu_diarization = "pyannote package is not installed"
    community_diarization_setup = (
        "Install the transcription optional dependencies to add pyannote Community-1."
        if not pyannote_package_installed
        else "Add a Hugging Face read token for the first Community-1 download, or restore its complete local cache."
    )
    selected_diarization_ready = (
        portable_ready
        if selected_speaker_engine == PORTABLE_DIARIZATION_ENGINE
        else pyannote_ready
    )
    llama_ready = bool(llama_server)

    if cuda_ready:
        automatic_ready = cuda_ready and pyannote_ready and llama_ready
        automatic_transcription = "faster-whisper on CUDA"
        automatic_diarization = (
            "pyannote on CUDA"
            if pyannote_ready
            else community_cpu_diarization
        )
        automatic_analysis = "llama.cpp " + next(iter(sorted(llama_backends)), "CPU")
        automatic_note = "Uses the validated Windows CUDA path and keeps per-stage fallbacks explicit."
    elif sys.platform == "darwin" and metal_asr and metal_llm:
        automatic_ready = pyannote_ready and llama_ready
        automatic_transcription = "whisper.cpp on Apple Metal"
        automatic_diarization = community_cpu_diarization
        automatic_analysis = "llama.cpp / Metal"
        automatic_note = "Uses native Apple Metal for transcription and analysis; diarization stays on CPU."
    elif sys.platform.startswith("linux") and vulkan_asr and vulkan_llm:
        automatic_ready = pyannote_ready and llama_ready
        automatic_transcription = "whisper.cpp on Vulkan"
        automatic_diarization = community_cpu_diarization
        automatic_analysis = "llama.cpp / Vulkan"
        automatic_note = "Uses detected Vulkan runtimes for transcription and analysis; diarization stays on CPU."
    else:
        automatic_ready = cpu_ready and pyannote_ready and llama_ready
        automatic_transcription = "faster-whisper on CPU" if cpu_ready else "CPU ASR dependencies unavailable"
        automatic_diarization = community_cpu_diarization
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
            transcription_ready=(
                cuda_ready
                if cuda_ready
                else metal_asr
                if sys.platform == "darwin" and metal_asr and metal_llm
                else vulkan_asr
                if sys.platform.startswith("linux") and vulkan_asr and vulkan_llm
                else cpu_ready
            ),
            diarization_ready=pyannote_ready,
            analysis_ready=llama_ready,
            diarization_setup=community_diarization_setup,
        ),
        _profile(
            "cuda",
            "NVIDIA CUDA",
            cuda_ready and pyannote_ready and llama_ready,
            "faster-whisper / CUDA" if cuda_ready else "CUDA ASR dependencies unavailable",
            (
                "pyannote / CUDA"
                if torch["cuda_available"] and pyannote_ready
                else community_cpu_diarization
            ),
            "llama.cpp auto-offload" if llama_ready else "llama.cpp missing",
            transcription_ready=cuda_ready,
            diarization_ready=bool(torch["cuda_available"] and pyannote_ready),
            analysis_ready=llama_ready,
            transcription_setup=(
                "Install faster-whisper plus a CUDA-enabled PyTorch build, then rerun the hardware check."
            ),
            diarization_setup=community_diarization_setup,
            analysis_setup="Install llama-server or configure LLAMA_SERVER_PATH.",
        ),
        _profile(
            "vulkan",
            "Cross-vendor Vulkan",
            vulkan_asr and portable_ready and vulkan_llm,
            (
                "whisper.cpp / Vulkan"
                if vulkan_asr
                else "Vulkan runtime detected; needs a matching GGML model"
                if vulkan_runtime
                else "needs a Vulkan whisper.cpp build"
            ),
            portable_diarization,
            "llama.cpp / Vulkan" if vulkan_llm else "needs a Vulkan llama.cpp build",
            "Fast portable speaker preview runs on CPU and remains upgradeable "
            "to Community-1 without repeating transcription.",
            transcription_ready=vulkan_asr,
            diarization_ready=portable_ready,
            analysis_ready=vulkan_llm,
            transcription_setup=(
                "Configure a Vulkan-enabled whisper-cli, then use Download & test model "
                "or set WHISPER_CPP_MODEL_PATH to a matching local GGML file."
            ),
            transcription_action=(
                {
                    "stage": "transcription",
                    "kind": "prepare-asr-model",
                    "label": "Download Vulkan model",
                    "message": (
                        "The Vulkan whisper.cpp runtime is detected. Download and "
                        "verify the selected GGML model, then run its local decode."
                    ),
                }
                if vulkan_runtime and not whisper_model
                else {
                    "stage": "transcription",
                    "kind": "configure-transcription",
                    "label": "Show Vulkan setup",
                    "message": (
                        "A Vulkan-enabled whisper-cli is not configured. Install or "
                        "build whisper.cpp with GGML_VULKAN=1, set WHISPER_CPP_PATH, "
                        "then refresh this check before downloading a model."
                    ),
                }
            ),
            diarization_setup=portable_diarization_setup,
            analysis_setup=(
                "Configure a Vulkan-enabled llama-server with LLAMA_SERVER_PATH; "
                "device inspection must report a Vulkan adapter."
            ),
        ),
        _profile(
            "openvino",
            "Intel OpenVINO",
            openvino_asr and portable_ready and llama_ready,
            (
                "OpenVINO Whisper / "
                + ", ".join(openvino_device_names or openvino_devices)
                + " (safe CPU fallback)"
                if openvino_asr
                else "needs the OpenVINO GenAI optional package"
            ),
            portable_diarization,
            (
                "llama.cpp / SYCL"
                if "sycl" in llama_backends
                else "llama.cpp auto-offload or CPU"
            ),
            "Uses devices exposed by the installed OpenVINO runtime; unsupported model/device combinations retry on CPU.",
            transcription_ready=openvino_asr,
            diarization_ready=portable_ready,
            analysis_ready=llama_ready,
            transcription_setup=(
                'Install this app\'s OpenVINO optional dependencies (`pip install -e ".[openvino]"`), '
                "then rerun the check and Test engine."
            ),
            transcription_action={
                "stage": "transcription",
                "kind": "configure-transcription",
                "label": "Show OpenVINO setup",
                "message": (
                    'Install the OpenVINO optional dependencies with `python -m pip '
                    'install -e ".[openvino]"`, then refresh the hardware check. '
                    "The first explicit engine test may acquire the selected model."
                ),
            },
            diarization_setup=portable_diarization_setup,
            analysis_setup="Install llama-server or configure LLAMA_SERVER_PATH.",
        ),
    ]
    if sys.platform == "win32":
        profiles.append(
            _profile(
                "windowsml",
                "Windows ML",
                windows_ml_decode and portable_ready and llama_ready,
                (
                    f"{windows_ml.get('backend') or 'Windows ML ONNX Whisper'} (validated model)"
                    if windows_ml_decode
                    else "runtime detected; configure and validate an ONNX Whisper model"
                    if windows_ml_runtime
                    else "needs the Windows ML helper and a compatible ONNX Whisper model"
                ),
                portable_diarization,
                "llama.cpp auto-offload or CPU" if llama_ready else "llama.cpp missing",
                (
                    "The configured model passed an actual silent-audio decode self-test. "
                    + "Fast portable speaker preview runs on CPU and remains "
                    "upgradeable to Community-1."
                    if windows_ml_decode
                    else "A runtime-only probe is not enough; this profile remains unavailable until a model decode passes."
                ),
                transcription_ready=windows_ml_decode,
                diarization_ready=portable_ready,
                analysis_ready=llama_ready,
                transcription_setup=(
                    "Use a build containing the Windows ML helper, then choose Build & test "
                    "model or set a compatible ONNX Whisper model path."
                ),
                transcription_action=(
                    {
                        "stage": "transcription",
                        "kind": "prepare-asr-model",
                        "label": "Build Windows ML model",
                        "message": (
                            "The Windows ML helper is ready, but the selected Whisper "
                            "graph has not passed its decode probe. Build and verify "
                            "the managed CPU graph now."
                        ),
                    }
                    if windows_ml_runtime and not windows_ml_decode
                    else {
                        "stage": "transcription",
                        "kind": "configure-transcription",
                        "label": "Show Windows ML setup",
                        "message": (
                            "Use the verified Windows publish that contains the Windows "
                            "ML helper, or configure WINDOWS_ML_HELPER_PATH, then refresh "
                            "this check before building a model."
                        ),
                    }
                ),
                diarization_setup=portable_diarization_setup,
                analysis_setup="Install llama-server or configure LLAMA_SERVER_PATH.",
            )
        )
    if sys.platform == "darwin":
        profiles.append(
            _profile(
                "metal",
                "Apple Metal",
                metal_asr and portable_ready and metal_llm,
                (
                    "whisper.cpp / Metal"
                    if metal_asr
                    else "Metal runtime detected; needs a matching GGML model"
                    if metal_runtime
                    else "needs a native ggml-metal whisper.cpp build"
                ),
                portable_diarization,
                "llama.cpp / Metal" if metal_llm else "needs a Metal llama.cpp build",
                "Fast portable speaker preview uses CPU and remains upgradeable "
                "to Community-1 without repeating transcription.",
                transcription_ready=metal_asr,
                diarization_ready=portable_ready,
                analysis_ready=metal_llm,
                transcription_setup=(
                    "Configure a native ggml-metal whisper-cli and local GGML model."
                ),
                transcription_action=(
                    {
                        "stage": "transcription",
                        "kind": "prepare-asr-model",
                        "label": "Download Metal model",
                        "message": (
                            "The native Metal whisper.cpp runtime is detected. Download "
                            "and verify the selected GGML model, then run its local decode."
                        ),
                    }
                    if metal_runtime and not whisper_model
                    else {
                        "stage": "transcription",
                        "kind": "configure-transcription",
                        "label": "Show Metal setup",
                        "message": (
                            "Configure a native whisper.cpp build with GGML_METAL=ON "
                            "and set WHISPER_CPP_PATH, then refresh this check."
                        ),
                    }
                ),
                diarization_setup=portable_diarization_setup,
                analysis_setup=(
                    "Configure a Metal-enabled llama-server; device inspection must report Metal."
                ),
            )
        )
    profiles.append(
        _profile(
            "qwen",
            "Fast CPU preview (Qwen3-ASR)",
            bool(qwen3_asr["ready"] and portable_ready and llama_ready),
            (
                "Qwen3-ASR 0.6B INT8 / sherpa-onnx CPU"
                if qwen3_asr["ready"]
                else "runtime detected; needs the verified Qwen3-ASR model and VAD"
                if qwen3_asr["runtime_installed"]
                else "needs the optional sherpa-onnx runtime and managed model"
            ),
            portable_diarization,
            "llama.cpp auto-offload or CPU" if llama_ready else "llama.cpp missing",
            (
                "Optional fast-CPU transcription candidate. Source-region timestamps "
                "are retained, but the current export has no word timestamps and does "
                "not replace the validated Whisper evidence default."
            ),
            transcription_ready=bool(qwen3_asr["ready"]),
            diarization_ready=portable_ready,
            analysis_ready=llama_ready,
            transcription_setup=(
                'Install `pip install -e ".[qwen]"`, then choose Download & test '
                "model to acquire the pinned INT8 export and Silero VAD."
            ),
            transcription_action=(
                {
                    "stage": "transcription",
                    "kind": "prepare-asr-model",
                    "label": "Download Qwen model",
                    "message": (
                        "sherpa-onnx is installed. Download and checksum-verify the "
                        "pinned Qwen3-ASR graph and Silero VAD, then run its CPU decode."
                    ),
                }
                if qwen3_asr["runtime_installed"] and not qwen3_asr["ready"]
                else {
                    "stage": "transcription",
                    "kind": "configure-transcription",
                    "label": "Show Qwen setup",
                    "message": (
                        'Install the fast-CPU runtime with `python -m pip install -e '
                        '".[qwen]"`, then refresh this check before downloading its model.'
                    ),
                }
            ),
            diarization_setup=portable_diarization_setup,
            analysis_setup="Install llama-server or configure LLAMA_SERVER_PATH.",
        )
    )
    profiles.append(
        _profile(
            "cpu",
            "CPU only",
            cpu_ready and pyannote_ready and llama_ready,
            "faster-whisper / INT8 CPU" if cpu_ready else "CPU ASR dependencies unavailable",
            community_cpu_diarization,
            "llama.cpp / CPU" if llama_ready else "llama.cpp missing",
            "Slowest but portable and a dependable fallback for every stage.",
            transcription_ready=cpu_ready,
            diarization_ready=pyannote_ready,
            analysis_ready=llama_ready,
            transcription_setup=(
                "Install the transcription optional dependencies to add faster-whisper and PyTorch."
            ),
            diarization_setup=community_diarization_setup,
            analysis_setup="Install llama-server or configure LLAMA_SERVER_PATH.",
        )
    )

    return {
        "python": sys.version.split()[0],
        "torch": torch,
        "openvino": openvino,
        "onnx": onnx,
        "windows_ml": windows_ml,
        "qwen3_asr": qwen3_asr,
        "speaker_labels": {
            "selected_engine": selected_speaker_engine,
            "package_installed": pyannote_package_installed,
            "token_configured": token_ready,
            "model_cache_detected": pyannote_cache_detected,
            "access_configured": pyannote_access_configured,
            "configured": selected_diarization_ready,
            "cuda_available": bool(torch["cuda_available"]),
            "community_1": {
                "package_installed": pyannote_package_installed,
                "access_configured": pyannote_access_configured,
                "ready": pyannote_ready,
            },
            "portable": portable_speakers,
        },
        "whisper_cpp": {
            "executable": whisper_executable,
            "model": whisper_model,
            "backends": whisper_backends,
            "container": whisper_container,
        },
        "llama_cpp": {
            "executable": str(llama_server) if llama_server else None,
            "devices": llama_devices,
        },
        "profiles": profiles,
    }
