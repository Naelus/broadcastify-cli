from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence


MANAGED_RUNTIME_ROOT_ENV = "BROADCASTIFY_MANAGED_RUNTIME_ROOT"
MANAGED_RUNTIME_MANIFEST_ENV = "BROADCASTIFY_MANAGED_RUNTIME_MANIFEST"
INSTALL_LOCK_STALE_SECONDS = 6 * 60 * 60
INSTALL_LOCK_INITIALIZATION_GRACE_SECONDS = 60


class ManagedRuntimeError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _configured_path(name: str) -> Path:
    value = str(os.getenv(name) or "").strip()
    if not value:
        raise ManagedRuntimeError(
            f"{name} was not supplied by the installed desktop application."
        )
    return Path(value).expanduser().resolve()


def _bounded_child(root: Path, child: Path) -> Path:
    resolved_root = root.resolve()
    resolved_child = child.resolve()
    try:
        resolved_child.relative_to(resolved_root)
    except ValueError as exc:
        raise ManagedRuntimeError(
            f"Managed runtime path escaped its storage root: {resolved_child}"
        ) from exc
    return resolved_child


def _safe_name(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-.")
    if not normalized:
        raise ManagedRuntimeError("The managed runtime revision is invalid.")
    return normalized


def load_bootstrap_manifest(path: str | Path | None = None) -> dict[str, Any]:
    manifest_path = (
        Path(path).expanduser().resolve()
        if path is not None
        else _configured_path(MANAGED_RUNTIME_MANIFEST_ENV)
    )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ManagedRuntimeError(
            "The packaged managed-runtime manifest is missing. Reinstall the app."
        ) from exc
    except json.JSONDecodeError as exc:
        raise ManagedRuntimeError(
            "The packaged managed-runtime manifest is invalid. Reinstall the app."
        ) from exc
    if not isinstance(payload, dict) or int(payload.get("schema_version", 0)) != 1:
        raise ManagedRuntimeError("Unsupported managed-runtime manifest schema.")
    if not isinstance(payload.get("profiles"), dict):
        raise ManagedRuntimeError("The managed-runtime manifest has no profiles.")
    payload["_path"] = str(manifest_path)
    return payload


def _profile_from_manifest(
    manifest: dict[str, Any], profile_id: str
) -> dict[str, Any]:
    profiles = manifest.get("profiles")
    profile = profiles.get(profile_id) if isinstance(profiles, dict) else None
    if not isinstance(profile, dict):
        raise ManagedRuntimeError(
            f"Managed runtime profile {profile_id!r} is not packaged by this build."
        )
    required = (
        "revision",
        "display_name",
        "python_version",
        "requirements_artifact",
        "packages",
    )
    if any(not profile.get(name) for name in required):
        raise ManagedRuntimeError(
            f"Managed runtime profile {profile_id!r} is incomplete."
        )
    if not isinstance(profile["packages"], list):
        raise ManagedRuntimeError(
            f"Managed runtime profile {profile_id!r} has invalid packages."
        )
    return profile


def _artifact_path(
    manifest: dict[str, Any], artifact_name: str
) -> tuple[Path, dict[str, Any]]:
    artifact = manifest.get(artifact_name)
    if not isinstance(artifact, dict):
        raise ManagedRuntimeError(
            f"The packaged {artifact_name.replace('_', ' ')} metadata is missing."
        )
    relative = str(artifact.get("path") or "").strip()
    expected_hash = str(artifact.get("sha256") or "").strip().lower()
    if not relative or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise ManagedRuntimeError(
            f"The packaged {artifact_name.replace('_', ' ')} metadata is invalid."
        )
    manifest_path = Path(str(manifest["_path"]))
    path = (manifest_path.parent / relative).resolve()
    try:
        path.relative_to(manifest_path.parent.resolve())
    except ValueError as exc:
        raise ManagedRuntimeError(
            f"The packaged {artifact_name.replace('_', ' ')} path is unsafe."
        ) from exc
    if not path.is_file():
        raise ManagedRuntimeError(
            f"The packaged {artifact_name.replace('_', ' ')} is missing. Reinstall the app."
        )
    if _sha256(path) != expected_hash:
        raise ManagedRuntimeError(
            f"The packaged {artifact_name.replace('_', ' ')} failed SHA-256 verification."
        )
    return path, artifact


def _runtime_paths(
    root: Path,
    profile_id: str,
    revision: str,
) -> dict[str, Path]:
    key = f"{_safe_name(profile_id)}-{_safe_name(revision)}"
    profiles = _bounded_child(root, root / "profiles")
    return {
        "root": root,
        "profiles": profiles,
        "final": _bounded_child(root, profiles / key),
        "partial": _bounded_child(root, profiles / f".{key}.partial"),
        "interpreters": _bounded_child(root, root / "interpreters"),
        "cache": _bounded_child(root, root / "cache"),
        "lock": _bounded_child(root, root / "install.lock"),
    }


def _receipt(path: Path) -> dict[str, Any] | None:
    receipt_path = path / "broadcastify-runtime.json"
    try:
        value = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def _directory_size_from_receipt(receipt: dict[str, Any] | None) -> int:
    if not receipt:
        return 0
    try:
        return max(0, int(receipt.get("installed_bytes", 0)))
    except (TypeError, ValueError):
        return 0


def _expected_artifact_hash(
    manifest: dict[str, Any], artifact_name: str
) -> str:
    artifact = manifest.get(artifact_name)
    if not isinstance(artifact, dict):
        return ""
    value = str(artifact.get("sha256") or "").strip().lower()
    return value if re.fullmatch(r"[0-9a-f]{64}", value) else ""


def managed_runtime_status(
    profile_id: str,
    *,
    root: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    runtime_root = (
        Path(root).expanduser().resolve()
        if root is not None
        else _configured_path(MANAGED_RUNTIME_ROOT_ENV)
    )
    manifest = load_bootstrap_manifest(manifest_path)
    profile = _profile_from_manifest(manifest, profile_id)
    paths = _runtime_paths(runtime_root, profile_id, str(profile["revision"]))
    receipt = _receipt(paths["final"])
    python_path = paths["final"] / "Scripts" / "python.exe"
    requirements_hash = _expected_artifact_hash(
        manifest, str(profile["requirements_artifact"])
    )
    ready = bool(
        receipt
        and receipt.get("profile") == profile_id
        and receipt.get("revision") == profile["revision"]
        and receipt.get("requirements_sha256") == requirements_hash
        and python_path.is_file()
    )
    partial = paths["partial"].is_dir()
    estimated = max(0, int(profile.get("estimated_installed_bytes", 0)))
    source_urls = [
        str(value)
        for value in profile.get("source_urls", [])
        if str(value).strip()
    ]
    licenses = [
        str(value)
        for value in profile.get("licenses", [])
        if str(value).strip()
    ]
    if ready:
        message = (
            f"{profile['display_name']} is installed. Restart the app to select "
            "it, then run Verify profile for execution proof."
        )
    elif partial:
        message = (
            "A partial managed-runtime installation is retained. Retry to reuse "
            "the verified download cache and continue safely."
        )
    else:
        message = (
            f"Install {profile['display_name']} explicitly before selecting this "
            "packaged profile. No Broadcastify request is made."
        )
    return {
        "profile": profile_id,
        "display_name": str(profile["display_name"]),
        "revision": str(profile["revision"]),
        "ready": ready,
        "partial": partial,
        "python_path": str(python_path) if ready else "",
        "storage_path": str(paths["final"]),
        "cache_path": str(paths["cache"]),
        "estimated_installed_bytes": estimated,
        "installed_bytes": _directory_size_from_receipt(receipt),
        "packages": [str(value) for value in profile["packages"]],
        "source_urls": source_urls,
        "licenses": licenses,
        "message": message,
        "installed_at": str(receipt.get("installed_at") or "") if receipt else "",
        "cuda_available_at_install": (
            bool(receipt.get("cuda_available")) if receipt else False
        ),
    }


@contextmanager
def _install_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor: int | None = None
    acquired = False
    try:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                age = max(0.0, time.time() - path.stat().st_mtime)
            except OSError:
                age = 0.0
            owner = _lock_owner_pid(path)
            if owner is not None and _process_is_running(owner):
                raise ManagedRuntimeError(
                    "A managed-runtime installation is already running."
                )
            if (
                owner is None
                and age < INSTALL_LOCK_INITIALIZATION_GRACE_SECONDS
            ):
                raise ManagedRuntimeError(
                    "A managed-runtime installation is already running."
                )
            if owner is None and age < INSTALL_LOCK_STALE_SECONDS:
                raise ManagedRuntimeError(
                    "The managed-runtime install lock is incomplete; retry after "
                    "its safety timeout or restart the app."
                )
            path.unlink(missing_ok=True)
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(descriptor, f"{os.getpid()}\n{_utc_now()}\n".encode("ascii"))
        os.close(descriptor)
        descriptor = None
        acquired = True
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if acquired:
            path.unlink(missing_ok=True)


def _lock_owner_pid(path: Path) -> int | None:
    try:
        first_line = path.read_text(encoding="ascii").splitlines()[0]
        value = int(first_line)
    except (FileNotFoundError, IndexError, OSError, UnicodeError, ValueError):
        return None
    return value if value > 0 else None


def _process_is_running(process_id: int) -> bool:
    if process_id == os.getpid():
        return True
    if sys.platform == "win32":
        # os.kill(pid, 0) is destructive on Windows: any non-console signal is
        # implemented with TerminateProcess. Query a handle and exit status
        # instead, without modifying the candidate process.
        import ctypes
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = (
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        )
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        )
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(
            process_query_limited_information,
            False,
            process_id,
        )
        if not handle:
            return ctypes.get_last_error() == 5
        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return True
            return exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _run_command(
    command: Sequence[str],
    *,
    environment: dict[str, str],
    on_line: Callable[[str], None],
) -> list[str]:
    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        creationflags=(
            subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        ),
    )
    lines: list[str] = []
    assert process.stdout is not None
    for raw in process.stdout:
        line = raw.strip()
        if line:
            lines.append(line)
            on_line(line)
    return_code = process.wait()
    if return_code != 0:
        detail = lines[-1] if lines else f"process exited with code {return_code}"
        raise ManagedRuntimeError(detail)
    return lines


def _find_managed_interpreter(directory: Path, version: str) -> Path:
    exact = sorted(directory.glob(f"cpython-{version}-windows-x86_64-none/python.exe"))
    candidates = exact or sorted(directory.glob("cpython-*/python.exe"))
    if not candidates:
        raise ManagedRuntimeError("uv did not produce the requested managed Python runtime.")
    return candidates[-1].resolve()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _directory_size(path: Path) -> int:
    total = 0
    for candidate in path.rglob("*"):
        try:
            if candidate.is_file():
                total += candidate.stat().st_size
        except OSError:
            continue
    return total


def install_managed_runtime(
    profile_id: str,
    *,
    emit_progress: Callable[[dict[str, Any]], None],
    root: str | Path | None = None,
    manifest_path: str | Path | None = None,
    runner: Callable[..., list[str]] = _run_command,
) -> dict[str, Any]:
    runtime_root = (
        Path(root).expanduser().resolve()
        if root is not None
        else _configured_path(MANAGED_RUNTIME_ROOT_ENV)
    )
    runtime_root.mkdir(parents=True, exist_ok=True)
    manifest = load_bootstrap_manifest(manifest_path)
    profile = _profile_from_manifest(manifest, profile_id)
    paths = _runtime_paths(runtime_root, profile_id, str(profile["revision"]))
    paths["profiles"].mkdir(parents=True, exist_ok=True)
    paths["cache"].mkdir(parents=True, exist_ok=True)
    paths["interpreters"].mkdir(parents=True, exist_ok=True)

    def progress(step: int, message: str) -> None:
        emit_progress(
            {
                "type": "progress",
                "phase": "managed_runtime",
                "current": step,
                "total": 5,
                "message": message[:1000],
            }
        )

    with _install_lock(paths["lock"]):
        current = managed_runtime_status(
            profile_id,
            root=runtime_root,
            manifest_path=manifest["_path"],
        )
        if current["ready"]:
            current["reused"] = True
            return current

        progress(1, "Verifying packaged bootstrap and application artifacts…")
        uv_path, uv_artifact = _artifact_path(manifest, "uv")
        app_wheel, app_artifact = _artifact_path(manifest, "app_wheel")
        requirements_name = str(profile["requirements_artifact"])
        requirements_path, requirements_artifact = _artifact_path(
            manifest, requirements_name
        )
        environment = os.environ.copy()
        environment.update(
            {
                "UV_CACHE_DIR": str(paths["cache"]),
                "UV_PYTHON_INSTALL_DIR": str(paths["interpreters"]),
                "UV_NO_CONFIG": "1",
                "UV_LINK_MODE": "copy",
                "NO_COLOR": "1",
            }
        )

        progress(2, f"Preparing isolated Python {profile['python_version']}…")
        runner(
            [
                str(uv_path),
                "python",
                "install",
                str(profile["python_version"]),
                "--install-dir",
                str(paths["interpreters"]),
                "--cache-dir",
                str(paths["cache"]),
                "--no-bin",
                "--no-registry",
                "--color",
                "never",
            ],
            environment=environment,
            on_line=lambda line: progress(2, line),
        )
        interpreter = _find_managed_interpreter(
            paths["interpreters"], str(profile["python_version"])
        )

        partial_python = paths["partial"] / "Scripts" / "python.exe"
        if not partial_python.is_file():
            progress(3, "Creating the persistent profile environment…")
            runner(
                [
                    str(uv_path),
                    "venv",
                    "--python",
                    str(interpreter),
                    "--allow-existing",
                    "--link-mode",
                    "copy",
                    "--no-project",
                    "--color",
                    "never",
                    str(paths["partial"]),
                ],
                environment=environment,
                on_line=lambda line: progress(3, line),
            )
        if not partial_python.is_file():
            raise ManagedRuntimeError("The managed profile Python executable is missing.")

        dependency_install_command = [
            str(uv_path),
            "pip",
            "install",
            "--python",
            str(partial_python),
            "--strict",
            "--only-binary",
            ":all:",
            "--link-mode",
            "copy",
            "--color",
            "never",
            "--requirements",
            str(requirements_path),
            "--require-hashes",
        ]
        progress(
            4,
            "Installing the checksum-locked Windows CUDA dependency set. "
            "Cancellation keeps the download cache and partial environment for "
            "a safe retry…",
        )
        runner(
            dependency_install_command,
            environment=environment,
            on_line=lambda line: progress(4, line),
        )
        runner(
            [
                str(uv_path),
                "pip",
                "install",
                "--python",
                str(partial_python),
                "--strict",
                "--only-binary",
                ":all:",
                "--no-deps",
                "--link-mode",
                "copy",
                "--color",
                "never",
                str(app_wheel),
            ],
            environment=environment,
            on_line=lambda line: progress(4, line),
        )

        verification_script = (
            "import importlib.metadata as m, json, warnings, torch; "
            "warnings.filterwarnings('ignore', category=UserWarning, "
            "module=r'pyannote\\.audio\\.core\\.io'); "
            "import faster_whisper, pyannote.audio, torchaudio; "
            "print(json.dumps({"
            "'torch':m.version('torch'),"
            "'torchaudio':m.version('torchaudio'),"
            "'faster_whisper':m.version('faster-whisper'),"
            "'pyannote_audio':m.version('pyannote.audio'),"
            "'cuda_available':bool(torch.cuda.is_available()),"
            "'cuda_version':str(torch.version.cuda or '')}))"
        )
        progress(5, "Verifying imports, package identity, and installed size…")
        lines = runner(
            [str(partial_python), "-c", verification_script],
            environment=environment,
            on_line=lambda line: progress(5, line),
        )
        try:
            verified = json.loads(lines[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            raise ManagedRuntimeError(
                "The managed CUDA environment did not return its package identity."
            ) from exc
        installed_bytes = _directory_size(paths["partial"])
        receipt = {
            "schema_version": 1,
            "profile": profile_id,
            "revision": str(profile["revision"]),
            "app_version": str(manifest.get("app_version") or ""),
            "installed_at": _utc_now(),
            "installed_bytes": installed_bytes,
            "python_version": str(profile["python_version"]),
            "packages": verified,
            "cuda_available": bool(verified.get("cuda_available")),
            "uv_version": str(uv_artifact.get("version") or ""),
            "uv_sha256": str(uv_artifact.get("sha256") or ""),
            "app_wheel_sha256": str(app_artifact.get("sha256") or ""),
            "requirements_sha256": str(
                requirements_artifact.get("sha256") or ""
            ),
            "manifest_sha256": _sha256(Path(str(manifest["_path"]))),
        }
        _write_json_atomic(
            paths["partial"] / "broadcastify-runtime.json", receipt
        )

        if paths["final"].exists():
            retained = _bounded_child(
                runtime_root,
                paths["profiles"]
                / f".{paths['final'].name}.replaced-{int(time.time())}",
            )
            os.replace(paths["final"], retained)
        os.replace(paths["partial"], paths["final"])

    result = managed_runtime_status(
        profile_id,
        root=runtime_root,
        manifest_path=manifest["_path"],
    )
    if not result["ready"]:
        raise ManagedRuntimeError("The managed runtime was not retained successfully.")
    result["reused"] = False
    return result
