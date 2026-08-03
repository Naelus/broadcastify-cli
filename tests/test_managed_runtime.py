import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from broadcastify_cli.managed_runtime import (
    ManagedRuntimeError,
    _install_lock,
    install_managed_runtime,
    managed_runtime_status,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bootstrap(tmp_path: Path) -> Path:
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    uv = bootstrap / "uv.exe"
    wheel = bootstrap / "broadcastify_cli-9.9.9-py3-none-any.whl"
    requirements = bootstrap / "windows-managed-cuda-lock.txt"
    uv.write_bytes(b"verified uv")
    wheel.write_bytes(b"verified app wheel")
    requirements.write_text(
        "torch @ https://packages.invalid/torch.whl#sha256=" + "a" * 64 + "\n",
        encoding="utf-8",
    )
    manifest = bootstrap / "managed-runtime.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "app_version": "9.9.9",
                "uv": {
                    "version": "0.12.1",
                    "path": uv.name,
                    "sha256": _hash(uv),
                },
                "app_wheel": {
                    "path": wheel.name,
                    "sha256": _hash(wheel),
                },
                "cuda_requirements": {
                    "path": requirements.name,
                    "sha256": _hash(requirements),
                },
                "profiles": {
                    "cuda": {
                        "revision": "cuda-test-r1",
                        "display_name": "CUDA test runtime",
                        "python_version": "3.12.10",
                        "requirements_artifact": "cuda_requirements",
                        "estimated_installed_bytes": 5_900_000_000,
                        "packages": [
                            "faster-whisper==1.2.1",
                            "pyannote.audio==4.0.7",
                            "torch==2.11.0",
                            "torchaudio==2.11.0",
                        ],
                        "source_urls": ["https://pypi.org/"],
                        "licenses": ["wheel metadata"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_managed_runtime_status_distinguishes_missing_partial_and_ready(
    tmp_path: Path,
) -> None:
    manifest = _bootstrap(tmp_path)
    root = tmp_path / "managed"

    missing = managed_runtime_status("cuda", root=root, manifest_path=manifest)
    assert missing["ready"] is False
    assert missing["partial"] is False
    assert missing["estimated_installed_bytes"] == 5_900_000_000
    assert missing["source_urls"] == ["https://pypi.org/"]

    partial = root / "profiles" / ".cuda-cuda-test-r1.partial"
    partial.mkdir(parents=True)
    resumable = managed_runtime_status("cuda", root=root, manifest_path=manifest)
    assert resumable["ready"] is False
    assert resumable["partial"] is True
    assert "retry" in resumable["message"].lower()

    final = root / "profiles" / "cuda-cuda-test-r1"
    (final / "Scripts").mkdir(parents=True)
    (final / "Scripts" / "python.exe").write_bytes(b"python")
    (final / "broadcastify-runtime.json").write_text(
        json.dumps(
            {
                "profile": "cuda",
                "revision": "cuda-test-r1",
                "installed_bytes": 1234,
                "installed_at": "2026-08-03T00:00:00+00:00",
                "requirements_sha256": _hash(
                    manifest.parent / "windows-managed-cuda-lock.txt"
                ),
            }
        ),
        encoding="utf-8",
    )
    ready = managed_runtime_status("cuda", root=root, manifest_path=manifest)
    assert ready["ready"] is True
    assert ready["installed_bytes"] == 1234
    assert Path(ready["python_path"]).name == "python.exe"
    assert Path(ready["python_path"]).parent.name == "Scripts"


def test_managed_runtime_install_is_binary_only_and_resumes_partial_work(
    tmp_path: Path,
) -> None:
    manifest = _bootstrap(tmp_path)
    root = tmp_path / "managed"
    commands: list[list[str]] = []
    progress: list[dict[str, Any]] = []
    fail_install = True

    def runner(
        command: list[str],
        *,
        environment: dict[str, str],
        on_line: Any,
    ) -> list[str]:
        nonlocal fail_install
        commands.append(list(command))
        assert environment["UV_NO_CONFIG"] == "1"
        if command[1:3] == ["python", "install"]:
            install_dir = Path(command[command.index("--install-dir") + 1])
            interpreter = (
                install_dir
                / "cpython-3.12.10-windows-x86_64-none"
                / "python.exe"
            )
            interpreter.parent.mkdir(parents=True, exist_ok=True)
            interpreter.write_bytes(b"python")
            return ["managed python ready"]
        if command[1] == "venv":
            python = Path(command[-1]) / "Scripts" / "python.exe"
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_bytes(b"python")
            return ["venv ready"]
        if command[1:3] == ["pip", "install"] and "--requirements" in command:
            if fail_install:
                fail_install = False
                raise ManagedRuntimeError("simulated cancelled transfer")
            return ["packages ready"]
        if command[1:3] == ["pip", "install"]:
            assert "--no-deps" in command
            return ["application ready"]
        return [
            json.dumps(
                {
                    "torch": "2.11.0+cu128",
                    "torchaudio": "2.11.0+cu128",
                    "faster_whisper": "1.2.1",
                    "pyannote_audio": "4.0.7",
                    "cuda_available": True,
                    "cuda_version": "12.8",
                }
            )
        ]

    with pytest.raises(ManagedRuntimeError, match="cancelled"):
        install_managed_runtime(
            "cuda",
            root=root,
            manifest_path=manifest,
            emit_progress=progress.append,
            runner=runner,
        )
    interrupted = managed_runtime_status(
        "cuda", root=root, manifest_path=manifest
    )
    assert interrupted["partial"] is True

    commands.clear()
    installed = install_managed_runtime(
        "cuda",
        root=root,
        manifest_path=manifest,
        emit_progress=progress.append,
        runner=runner,
    )
    assert installed["ready"] is True
    assert installed["reused"] is False
    assert not any(command[1] == "venv" for command in commands)
    pip_commands = [command for command in commands if command[1] == "pip"]
    assert len(pip_commands) == 2
    pip_command = pip_commands[0]
    assert pip_command[pip_command.index("--only-binary") + 1] == ":all:"
    assert "--require-hashes" in pip_command
    assert Path(pip_command[pip_command.index("--requirements") + 1]).name == (
        "windows-managed-cuda-lock.txt"
    )
    assert "--no-deps" in pip_commands[1]
    assert pip_commands[1][-1].endswith(".whl")
    assert progress[-1]["total"] == 5


def test_managed_runtime_rejects_corrupt_packaged_bootstrap(tmp_path: Path) -> None:
    manifest = _bootstrap(tmp_path)
    (manifest.parent / "uv.exe").write_bytes(b"tampered")

    with pytest.raises(ManagedRuntimeError, match="SHA-256"):
        install_managed_runtime(
            "cuda",
            root=tmp_path / "managed",
            manifest_path=manifest,
            emit_progress=lambda value: None,
        )


def test_live_install_lock_is_not_removed_by_a_competing_process(
    tmp_path: Path,
) -> None:
    manifest = _bootstrap(tmp_path)
    root = tmp_path / "managed"
    root.mkdir()
    lock = root / "install.lock"
    owner = f"{os.getpid()}\n2026-08-03T00:00:00+00:00\n"
    lock.write_text(owner, encoding="ascii")

    with pytest.raises(ManagedRuntimeError, match="already running"):
        install_managed_runtime(
            "cuda",
            root=root,
            manifest_path=manifest,
            emit_progress=lambda value: None,
        )
    assert lock.read_text(encoding="ascii") == owner


def test_cancelled_worker_lock_is_reclaimed_immediately(tmp_path: Path) -> None:
    lock = tmp_path / "install.lock"
    lock.write_text("2147483646\n2026-08-03T00:00:00+00:00\n", encoding="ascii")

    with _install_lock(lock):
        assert lock.read_text(encoding="ascii").splitlines()[0] == str(os.getpid())

    assert not lock.exists()


def test_ready_runtime_requires_the_current_dependency_lock(tmp_path: Path) -> None:
    manifest = _bootstrap(tmp_path)
    root = tmp_path / "managed"
    final = root / "profiles" / "cuda-cuda-test-r1"
    (final / "Scripts").mkdir(parents=True)
    (final / "Scripts" / "python.exe").write_bytes(b"python")
    (final / "broadcastify-runtime.json").write_text(
        json.dumps(
            {
                "profile": "cuda",
                "revision": "cuda-test-r1",
                "requirements_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )

    status = managed_runtime_status("cuda", root=root, manifest_path=manifest)

    assert status["ready"] is False
