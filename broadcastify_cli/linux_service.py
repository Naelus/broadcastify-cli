from __future__ import annotations

import argparse
import http.client
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import webbrowser
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .web_app import (
    bind_is_loopback,
    bind_scope,
    format_web_url,
    run_web_app,
    validate_bind_host,
)


SERVICE_NAME = "radio-archive-web.service"
CONFIG_VERSION = 1
DEFAULT_PORT = 8765


def _xdg_directory(variable: str, fallback: Path) -> Path:
    configured = os.getenv(variable)
    return Path(configured).expanduser() if configured else fallback


def default_config_dir() -> Path:
    return _xdg_directory("XDG_CONFIG_HOME", Path.home() / ".config") / "radio-archive"


def default_data_dir() -> Path:
    return _xdg_directory("XDG_DATA_HOME", Path.home() / ".local" / "share") / "radio-archive"


def default_unit_dir() -> Path:
    return _xdg_directory("XDG_CONFIG_HOME", Path.home() / ".config") / "systemd" / "user"


@dataclass(frozen=True)
class LinuxServiceConfig:
    version: int
    python_executable: str
    working_dir: str
    output_dir: str
    database_path: str
    environment_file: str
    host: str = "127.0.0.1"
    port: int = DEFAULT_PORT

    def validate(self) -> None:
        if self.version != CONFIG_VERSION:
            raise ValueError(
                f"Unsupported service configuration version {self.version}; "
                f"expected {CONFIG_VERSION}."
            )
        if validate_bind_host(self.host) != self.host:
            raise ValueError("The Web service host may not contain surrounding whitespace.")
        if not 1 <= int(self.port) <= 65535:
            raise ValueError("The Web service port must be between 1 and 65535.")
        for label, raw_value in (
            ("Python executable", self.python_executable),
            ("working directory", self.working_dir),
            ("output directory", self.output_dir),
            ("database path", self.database_path),
            ("environment file", self.environment_file),
        ):
            if not raw_value or "\x00" in raw_value or "\n" in raw_value:
                raise ValueError(f"{label} is invalid.")
            if not Path(raw_value).is_absolute():
                raise ValueError(f"{label} must be an absolute path.")

    @property
    def url(self) -> str:
        return format_web_url(self.host, self.port)

    @property
    def health_host(self) -> str:
        if self.host == "0.0.0.0":
            return "127.0.0.1"
        if self.host == "::":
            return "::1"
        return self.host

    @property
    def access_scope(self) -> str:
        return bind_scope(self.host)

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "LinuxServiceConfig":
        expected = {
            "version",
            "python_executable",
            "working_dir",
            "output_dir",
            "database_path",
            "environment_file",
            "host",
            "port",
        }
        unknown = set(value) - expected
        if unknown:
            raise ValueError(
                "Unknown service configuration field(s): " + ", ".join(sorted(unknown))
            )
        try:
            config = cls(
                version=int(value["version"]),
                python_executable=str(value["python_executable"]),
                working_dir=str(value["working_dir"]),
                output_dir=str(value["output_dir"]),
                database_path=str(value["database_path"]),
                environment_file=str(value["environment_file"]),
                host=str(value.get("host") or "127.0.0.1"),
                port=int(value.get("port") or DEFAULT_PORT),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("The service configuration is incomplete or invalid.") from exc
        config.validate()
        return config


def build_service_config(
    *,
    python_executable: str | Path = sys.executable,
    working_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    database_path: str | Path | None = None,
    environment_file: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
) -> LinuxServiceConfig:
    executable_value = os.path.expanduser(os.fspath(python_executable))
    if not os.path.isabs(executable_value):
        discovered = shutil.which(executable_value)
        if discovered:
            executable_value = discovered
    executable = Path(os.path.abspath(executable_value))
    data = Path(working_dir or default_data_dir()).expanduser().resolve()
    output = Path(output_dir or data / "archives").expanduser().resolve()
    database = Path(
        database_path or output / "broadcastify-analysis.sqlite3"
    ).expanduser().resolve()
    environment = Path(
        environment_file or default_config_dir() / ".env"
    ).expanduser().resolve()
    config = LinuxServiceConfig(
        version=CONFIG_VERSION,
        # Do not resolve a venv's Python symlink to the system interpreter.
        python_executable=str(executable),
        working_dir=str(data),
        output_dir=str(output),
        database_path=str(database),
        environment_file=str(environment),
        host=validate_bind_host(host),
        port=int(port),
    )
    config.validate()
    return config


def _systemd_quote(value: str, *, command_argument: bool = False) -> str:
    if "\x00" in value or "\n" in value or "\r" in value:
        raise ValueError("systemd arguments may not contain NUL or line breaks.")
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("%", "%%")
    )
    if command_argument:
        # ExecStart performs environment expansion; $$ is its literal dollar.
        escaped = escaped.replace("$", "$$")
    return f'"{escaped}"'


def _systemd_path_value(value: str) -> str:
    if (
        "\x00" in value
        or "\n" in value
        or "\r" in value
        or value != value.strip()
    ):
        raise ValueError(
            "systemd directive paths may not contain NUL, line breaks, or "
            "leading/trailing whitespace."
        )
    # Path-valued directives do not use ExecStart's shell-like quote parser.
    # Spaces remain literal; backslashes and systemd specifier markers do not.
    return value.replace("\\", "\\\\").replace("%", "%%")


def render_systemd_unit(config: LinuxServiceConfig, config_path: Path) -> str:
    config.validate()
    config_path = config_path.expanduser().resolve()
    log_path = Path(config.working_dir) / "radio-archive-web.log"
    command = " ".join(
        [
            _systemd_quote(config.python_executable, command_argument=True),
            "-m",
            "broadcastify_cli.linux_service",
            "serve",
            "--config",
            _systemd_quote(str(config_path), command_argument=True),
        ]
    )
    return "\n".join(
        [
            "[Unit]",
            "Description=Radio Archive Intelligence local Web app",
            "Documentation=https://github.com/Naelus/broadcastify-cli",
            "",
            "[Service]",
            "Type=simple",
            f"WorkingDirectory={_systemd_path_value(config.working_dir)}",
            "Environment=PYTHONUNBUFFERED=1",
            f"ExecStart={command}",
            "Restart=on-failure",
            "RestartSec=5s",
            "KillMode=control-group",
            "KillSignal=SIGINT",
            "TimeoutStopSec=20s",
            "UMask=0077",
            "NoNewPrivileges=yes",
            "PrivateTmp=yes",
            "ProtectSystem=full",
            f"StandardOutput=append:{_systemd_path_value(str(log_path))}",
            f"StandardError=append:{_systemd_path_value(str(log_path))}",
            "SyslogIdentifier=radio-archive-web",
            "",
            "[Install]",
            "WantedBy=default.target",
            "",
        ]
    )


def _atomic_write(path: Path, content: str, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_service_config(path: Path, config: LinuxServiceConfig) -> None:
    config.validate()
    content = json.dumps(asdict(config), indent=2, sort_keys=True) + "\n"
    _atomic_write(path.expanduser().resolve(), content, 0o600)


def load_service_config(path: Path) -> LinuxServiceConfig:
    try:
        value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(
            f"No managed service configuration exists at {path}. Run install first."
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read service configuration at {path}.") from exc
    if not isinstance(value, dict):
        raise ValueError("The service configuration must contain one JSON object.")
    return LinuxServiceConfig.from_mapping(value)


ENVIRONMENT_TEMPLATE = """\
# Private settings for the managed Radio Archive Web service.
# Keep this file owner-readable only. Uncomment only the values you use.
#
# BROADCASTIFY_USERNAME=
# BROADCASTIFY_PASSWORD=
# HUGGINGFACE_TOKEN=
#
# WHISPER_CPP_PATH=
# WHISPER_CPP_MODEL_PATH=
# LLAMA_SERVER_PATH=
# ANALYSIS_PROVIDER=local
# ANALYSIS_MODEL=
"""


def _systemctl(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    executable = shutil.which("systemctl")
    if not executable:
        raise RuntimeError(
            "systemctl was not found. Use `broadcastify-web --open` in the "
            "foreground on Linux systems without systemd."
        )
    result = subprocess.run(
        [executable, "--user", *arguments],
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    if check and result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(
            f"systemctl --user {' '.join(arguments)} failed"
            + (f": {detail}" if detail else ".")
        )
    return result


def install_service(
    config: LinuxServiceConfig,
    *,
    config_path: Path,
    unit_path: Path,
    force: bool = False,
    start: bool = True,
    manage_systemd: bool = True,
    validate_runtime: bool = True,
) -> tuple[Path, Path]:
    config.validate()
    config_path = config_path.expanduser().resolve()
    unit_path = unit_path.expanduser().resolve()
    if unit_path.exists() and not force:
        existing = unit_path.read_text(encoding="utf-8")
        proposed = render_systemd_unit(config, config_path)
        if existing != proposed:
            raise FileExistsError(
                f"{unit_path} already exists with different settings. "
                "Re-run with --force to replace only this user service."
            )

    Path(config.working_dir).mkdir(parents=True, exist_ok=True)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.database_path).parent.mkdir(parents=True, exist_ok=True)
    if validate_runtime:
        executable = Path(config.python_executable)
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise ValueError(
                f"The configured Python executable is not runnable: {executable}"
            )
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        try:
            runtime_check = subprocess.run(
                [
                    str(executable),
                    "-c",
                    "import broadcastify_cli.linux_service",
                ],
                cwd=config.working_dir,
                env=environment,
                check=False,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired as exc:
            raise ValueError(
                "The configured Python runtime did not finish its package check "
                "within 30 seconds."
            ) from exc
        if runtime_check.returncode:
            detail = (runtime_check.stderr or runtime_check.stdout).strip().splitlines()
            suffix = f": {detail[-1]}" if detail else ""
            raise ValueError(
                "The configured Python runtime cannot import the installed "
                f"radio archive package{suffix}"
            )
    environment_path = Path(config.environment_file)
    environment_path.parent.mkdir(parents=True, exist_ok=True)
    if not environment_path.exists():
        _atomic_write(environment_path, ENVIRONMENT_TEMPLATE, 0o600)

    save_service_config(config_path, config)
    _atomic_write(unit_path, render_systemd_unit(config, config_path), 0o600)
    if manage_systemd:
        _systemctl("daemon-reload")
        if start:
            _systemctl("enable", "--now", SERVICE_NAME)
        else:
            _systemctl("enable", SERVICE_NAME)
    return config_path, unit_path


def health_ready(config: LinuxServiceConfig, timeout: float = 1.0) -> bool:
    connection = http.client.HTTPConnection(
        config.health_host,
        config.port,
        timeout=timeout,
    )
    try:
        connection.request("GET", "/health")
        response = connection.getresponse()
        body = response.read()
        if response.status != 200:
            return False
        value = json.loads(body)
        return value == {"status": "ok", "scope": config.access_scope}
    except (OSError, http.client.HTTPException, json.JSONDecodeError):
        return False
    finally:
        connection.close()


def wait_until_ready(config: LinuxServiceConfig, timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if health_ready(config):
            return True
        time.sleep(0.2)
    return health_ready(config)


def show_service_logs(
    config: LinuxServiceConfig,
    *,
    lines: int = 100,
    follow: bool = False,
) -> int:
    bounded_lines = max(1, min(int(lines), 10_000))
    log_path = Path(config.working_dir) / "radio-archive-web.log"
    if not log_path.is_file():
        print(
            f"No owner log exists yet at {log_path}. Start the service first.",
            file=sys.stderr,
        )
        return 1
    if follow:
        executable = shutil.which("tail")
        if not executable:
            raise RuntimeError("The `tail` command required for --follow was not found.")
        return subprocess.run(
            [executable, "--lines", str(bounded_lines), "--follow=name", str(log_path)],
            check=False,
        ).returncode
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        retained = deque(handle, maxlen=bounded_lines)
    for line in retained:
        print(line, end="" if line.endswith("\n") else "\n")
    return 0


def _require_linux_systemd() -> None:
    if platform.system() != "Linux":
        raise RuntimeError(
            "The managed service commands use Linux systemd. On Windows use the "
            "native WinUI app; on macOS run `broadcastify-web --open` for now."
        )


def _service_paths(arguments: argparse.Namespace) -> tuple[Path, Path]:
    config_dir = Path(arguments.config_dir).expanduser().resolve()
    unit_dir = Path(arguments.unit_dir).expanduser().resolve()
    return config_dir / "service.json", unit_dir / SERVICE_NAME


def _add_storage_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--working-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--database")
    parser.add_argument("--env-file")
    parser.add_argument("--python", default=sys.executable, dest="python_executable")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        type=validate_bind_host,
        help=(
            "Loopback or a trusted private/link-local interface address. "
            "0.0.0.0/:: listens on every interface."
        ),
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--config-dir", default=str(default_config_dir()))
    parser.add_argument("--unit-dir", default=str(default_unit_dir()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Install and manage the private Radio Archive Web app as a per-user "
            "Linux systemd service. Loopback is the default; trusted-LAN mode "
            "requires an explicit host."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    install = subparsers.add_parser(
        "install",
        help="Write, enable, and optionally start the per-user service.",
    )
    _add_storage_arguments(install)
    install.add_argument("--force", action="store_true")
    install.add_argument(
        "--no-start",
        action="store_true",
        help="Enable the service for future logins without starting it now.",
    )

    render = subparsers.add_parser(
        "print-unit",
        help="Print the exact systemd unit without changing the machine.",
    )
    _add_storage_arguments(render)

    for command in ("start", "restart", "status", "stop", "logs", "uninstall"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--config-dir", default=str(default_config_dir()))
        command_parser.add_argument("--unit-dir", default=str(default_unit_dir()))
        if command in {"start", "restart"}:
            command_parser.add_argument("--open", action="store_true", dest="open_browser")
        if command == "logs":
            command_parser.add_argument("--follow", action="store_true")
            command_parser.add_argument("--lines", type=int, default=100)

    serve = subparsers.add_parser("serve", help=argparse.SUPPRESS)
    serve.add_argument("--config", required=True)
    return parser


def _config_from_arguments(arguments: argparse.Namespace) -> LinuxServiceConfig:
    return build_service_config(
        python_executable=arguments.python_executable,
        working_dir=arguments.working_dir,
        output_dir=arguments.output_dir,
        database_path=arguments.database,
        environment_file=arguments.env_file,
        host=arguments.host,
        port=arguments.port,
    )


def _start_and_wait(config: LinuxServiceConfig, action: str, open_browser: bool) -> int:
    _systemctl(action, SERVICE_NAME)
    if not wait_until_ready(config):
        print(
            "The service did not become healthy within 15 seconds. "
            "Run `radio-archive-service logs` for the retained error.",
            file=sys.stderr,
        )
        return 1
    print(f"Radio Archive Intelligence is ready at {config.url}")
    if open_browser:
        webbrowser.open(config.url)
    return 0


def _main(arguments: argparse.Namespace) -> int:
    if arguments.command == "serve":
        config = load_service_config(Path(arguments.config))
        os.environ["BROADCASTIFY_ENV_FILE"] = config.environment_file
        return run_web_app(
            output_dir=config.output_dir,
            database_path=config.database_path,
            working_dir=config.working_dir,
            host=config.host,
            port=config.port,
        )

    config_path, unit_path = _service_paths(arguments)
    if arguments.command == "print-unit":
        config = _config_from_arguments(arguments)
        print(render_systemd_unit(config, config_path), end="")
        return 0

    _require_linux_systemd()
    if arguments.command == "install":
        config = _config_from_arguments(arguments)
        install_service(
            config,
            config_path=config_path,
            unit_path=unit_path,
            force=arguments.force,
            start=not arguments.no_start,
        )
        print(f"Installed {SERVICE_NAME} for this user.")
        print(f"Private settings template: {config.environment_file}")
        if not bind_is_loopback(config.host):
            print(
                "Trusted-LAN mode is active. Anyone who can reach this address "
                "can open retained transcripts and audio; do not port-forward it."
            )
        if arguments.no_start:
            print("The service is enabled and will start at the next user login.")
        else:
            if not wait_until_ready(config):
                print(
                    "The service was enabled but is not healthy yet. "
                    "Run `radio-archive-service logs` for details.",
                    file=sys.stderr,
                )
                return 1
            print(f"Radio Archive Intelligence is ready at {config.url}")
        return 0

    if arguments.command == "uninstall":
        _systemctl("disable", "--now", SERVICE_NAME, check=False)
        unit_path.unlink(missing_ok=True)
        _systemctl("daemon-reload")
        print(
            "Removed the per-user service. Archives, models, service settings, "
            "and the private .env were preserved."
        )
        return 0

    if arguments.command == "stop":
        _systemctl("stop", SERVICE_NAME)
        print(
            "Stopped the local Web service. Retained jobs remain resumable; "
            "because the unit stays enabled, it starts again with the next "
            "user-manager login."
        )
        return 0

    config = load_service_config(config_path)
    if arguments.command == "logs":
        return show_service_logs(
            config,
            lines=arguments.lines,
            follow=arguments.follow,
        )
    if arguments.command in {"start", "restart"}:
        return _start_and_wait(
            config,
            arguments.command,
            bool(arguments.open_browser),
        )
    if arguments.command == "status":
        result = _systemctl("is-active", SERVICE_NAME, check=False)
        active = result.returncode == 0 and result.stdout.strip() == "active"
        healthy = health_ready(config) if active else False
        print(f"Service: {'active' if active else result.stdout.strip() or 'inactive'}")
        print(f"Health: {'ready' if healthy else 'not ready'}")
        print(f"Address: {config.url}")
        return 0 if active and healthy else 1
    raise RuntimeError(f"Unsupported service command: {arguments.command}")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _main(build_parser().parse_args(argv))
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"radio-archive-service: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
