from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from broadcastify_cli.linux_service import (
    LinuxServiceConfig,
    build_service_config,
    install_service,
    load_service_config,
    render_systemd_unit,
)


def _config(tmp_path: Path) -> LinuxServiceConfig:
    return build_service_config(
        python_executable=tmp_path / "venv with spaces" / "bin" / "python",
        working_dir=tmp_path / "Radio Archive 100%",
        output_dir=tmp_path / "Radio Archive 100%" / "archives",
        database_path=tmp_path / "Radio Archive 100%" / "archives" / "evidence.sqlite3",
        environment_file=tmp_path / "private settings" / ".env",
        port=18765,
    )


def test_systemd_unit_is_loopback_supervised_and_contains_no_secrets(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config_path = tmp_path / "config dir" / "service.json"
    unit = render_systemd_unit(config, config_path)

    assert "broadcastify_cli.linux_service serve" in unit
    assert "--config" in unit
    assert "Restart=on-failure" in unit
    assert "KillSignal=SIGINT" in unit
    assert "KillMode=control-group" in unit
    assert "NoNewPrivileges=yes" in unit
    assert "PrivateTmp=yes" in unit
    working_directory = str(tmp_path.resolve() / "Radio Archive 100%")
    escaped_working_directory = (
        working_directory.replace("\\", "\\\\").replace("%", "%%")
    )
    assert f"WorkingDirectory={escaped_working_directory}" in unit
    assert 'WorkingDirectory="' not in unit
    escaped_log_path = (
        str(tmp_path.resolve() / "Radio Archive 100%" / "radio-archive-web.log")
        .replace("\\", "\\\\")
        .replace("%", "%%")
    )
    assert f"StandardOutput=append:{escaped_log_path}" in unit
    assert f"StandardError=append:{escaped_log_path}" in unit
    assert "StandardOutput=journal" not in unit
    assert "0.0.0.0" not in unit
    assert "BROADCASTIFY_PASSWORD" not in unit
    assert "HUGGINGFACE_TOKEN" not in unit


def test_service_install_writes_config_unit_and_private_env_template(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config_path = tmp_path / "config" / "service.json"
    unit_path = tmp_path / "systemd" / "radio-archive-web.service"

    written_config, written_unit = install_service(
        config,
        config_path=config_path,
        unit_path=unit_path,
        manage_systemd=False,
        validate_runtime=False,
    )

    assert written_config == config_path.resolve()
    assert written_unit == unit_path.resolve()
    assert load_service_config(config_path) == config
    assert Path(config.environment_file).is_file()
    env_template = Path(config.environment_file).read_text(encoding="utf-8")
    assert "# BROADCASTIFY_PASSWORD=" in env_template
    assert "yourPassword" not in env_template
    assert json.loads(config_path.read_text(encoding="utf-8"))["host"] == "127.0.0.1"
    assert "ExecStart=" in unit_path.read_text(encoding="utf-8")


def test_service_install_refuses_to_replace_different_unit_without_force(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config_path = tmp_path / "config" / "service.json"
    unit_path = tmp_path / "systemd" / "radio-archive-web.service"
    unit_path.parent.mkdir(parents=True)
    unit_path.write_text("[Service]\nExecStart=/something/else\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--force"):
        install_service(
            config,
            config_path=config_path,
            unit_path=unit_path,
            manage_systemd=False,
            validate_runtime=False,
        )


def test_service_install_validates_the_selected_python_runtime(
    tmp_path: Path,
) -> None:
    config = build_service_config(
        python_executable=sys.executable,
        working_dir=tmp_path / "data",
        environment_file=tmp_path / "config" / ".env",
    )
    config_path = tmp_path / "config" / "service.json"
    unit_path = tmp_path / "systemd" / "radio-archive-web.service"

    install_service(
        config,
        config_path=config_path,
        unit_path=unit_path,
        manage_systemd=False,
    )

    assert load_service_config(config_path).python_executable == sys.executable


def test_service_install_rejects_a_missing_python_runtime(tmp_path: Path) -> None:
    config = build_service_config(
        python_executable=tmp_path / "missing-venv" / "bin" / "python",
        working_dir=tmp_path / "data",
    )

    with pytest.raises(ValueError, match="not runnable"):
        install_service(
            config,
            config_path=tmp_path / "config" / "service.json",
            unit_path=tmp_path / "systemd" / "radio-archive-web.service",
            manage_systemd=False,
        )


def test_service_config_accepts_explicit_trusted_lan_host(tmp_path: Path) -> None:
    value = {
        **json.loads(json.dumps(_config(tmp_path).__dict__)),
        "host": "0.0.0.0",
    }
    config = LinuxServiceConfig.from_mapping(value)

    assert config.host == "0.0.0.0"
    assert config.health_host == "127.0.0.1"
    assert config.access_scope == "trusted-lan"


def test_service_config_rejects_public_host(tmp_path: Path) -> None:
    value = {
        **json.loads(json.dumps(_config(tmp_path).__dict__)),
        "host": "8.8.8.8",
    }
    with pytest.raises(ValueError, match="public"):
        LinuxServiceConfig.from_mapping(value)


def test_service_config_preserves_virtual_environment_python_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    venv_python = tmp_path / "venv" / "bin" / "python"
    monkeypatch.setattr(
        "broadcastify_cli.linux_service.shutil.which",
        lambda value: str(venv_python) if value == "radio-python" else None,
    )

    config = build_service_config(
        python_executable="radio-python",
        working_dir=tmp_path / "data",
    )

    assert config.python_executable == str(venv_python)
