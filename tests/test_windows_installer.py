import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]


def _scan_release(
    stage: Path,
    *arguments: str,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "scan_public_release.py"),
            "--root",
            str(stage),
            *arguments,
        ],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )


def test_release_version_is_consistent_across_shipped_artifacts() -> None:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        version = tomllib.load(stream)["project"]["version"]
    parts = version.split(".")
    assert len(parts) in (3, 4) and all(part.isdigit() for part in parts)
    numeric_version = version if len(parts) == 4 else f"{version}.0"

    for project_path in (
        ROOT / "BroadcastifyCli.WinUI" / "BroadcastifyCli.WinUI.csproj",
        ROOT / "BroadcastifyCli.WindowsML" / "BroadcastifyCli.WindowsML.csproj",
    ):
        project = ElementTree.parse(project_path).getroot()
        values = {
            element.tag: element.text
            for element in project.iter()
            if element.tag in {"Version", "FileVersion"}
        }
        assert values == {"Version": version, "FileVersion": numeric_version}

    from broadcastify_cli import __version__

    assert __version__ == version
    installer = (ROOT / "installer" / "BroadcastifyDesktop.iss").read_text(
        encoding="utf-8"
    )
    definitions = dict(
        re.findall(
            r'^\s*#define\s+(\w+)\s+"([^"]+)"', installer, re.MULTILINE
        )
    )
    assert definitions["MyAppVersion"] == version
    assert definitions["MyAppVersionNumeric"] == numeric_version


def test_public_release_scanner_rejects_machine_paths_and_forbidden_files(
    tmp_path: Path,
) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "application.dll").write_bytes(
        b"prefix C:\\private\\source\\application.pdb suffix"
    )
    (stage / "settings.json").write_text("{}", encoding="utf-8")

    result = _scan_release(stage, "--forbid-path", r"C:\private\source")

    assert result.returncode == 1
    assert "application.dll: contains build path 1" in result.stderr
    assert "settings.json: forbidden release file" in result.stderr


def test_public_release_scanner_accepts_a_clean_stage(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "application.dll").write_bytes(b"portable release content")
    (stage / "build-manifest.json").write_text(
        '{"source_commit":"abc"}',
        encoding="utf-8",
    )

    result = _scan_release(stage, "--forbid-path", r"C:\private\source")

    assert result.returncode == 0
    assert "Public release scan passed: 2 files checked." in result.stdout


def test_public_release_scanner_ignores_generic_github_runner_profile(
    tmp_path: Path,
) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "uv.exe").write_bytes(
        b"third-party build metadata C:\\Users\\runneradmin\\work"
    )
    environment = os.environ.copy()
    environment.update(
        {
            "GITHUB_ACTIONS": "true",
            "RUNNER_ENVIRONMENT": "github-hosted",
        }
    )

    result = _scan_release(
        stage,
        "--forbid-path",
        r"D:\a\broadcastify-cli\broadcastify-cli",
        "--forbid-user-profile",
        r"C:\Users\runneradmin",
        environment=environment,
    )

    assert result.returncode == 0
    assert "Public release scan passed: 1 files checked." in result.stdout


def test_public_release_scanner_keeps_self_hosted_profile_protection(
    tmp_path: Path,
) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "application.dll").write_bytes(
        b"source C:\\Users\\runneradmin\\private\\application.pdb"
    )
    environment = os.environ.copy()
    environment.update(
        {
            "GITHUB_ACTIONS": "true",
            "RUNNER_ENVIRONMENT": "self-hosted",
        }
    )

    result = _scan_release(
        stage,
        "--forbid-user-profile",
        r"C:\Users\runneradmin",
        environment=environment,
    )

    assert result.returncode == 1
    assert "application.dll: contains build path 1" in result.stderr


def test_native_build_gate_runs_behavioral_suite_and_ui_e2e() -> None:
    project = ElementTree.parse(ROOT / "Directory.Build.targets").getroot()
    targets = {
        element.attrib["Name"]: element
        for element in project
        if element.tag == "Target"
    }

    regression = targets["RunProductRegressionGate"]
    assert regression.attrib["BeforeTargets"] == "PrepareForBuild"
    assert "run_product_regression_gate.ps1" in regression.find("Exec").attrib[
        "Command"
    ]

    ui_e2e = targets["RunNativeUiEndToEndGate"]
    assert ui_e2e.attrib["AfterTargets"] == "Build"
    assert "run_windows_ui_e2e.ps1" in ui_e2e.find("Exec").attrib["Command"]


def test_release_pipeline_exercises_installer_before_publish() -> None:
    workflow = (
        ROOT / ".github" / "workflows" / "windows-release.yml"
    ).read_text(encoding="utf-8")
    steps = re.findall(r"^\s+- name: (.+)$", workflow, re.MULTILINE)

    assert steps.index("Build installer") < steps.index(
        "Exercise installer lifecycle"
    )
    assert steps.index("Exercise installer lifecycle") < steps.index(
        "Upload workflow artifact"
    )
    assert steps.index("Exercise installer lifecycle") < steps.index(
        "Publish tagged release asset"
    )


def test_public_installer_build_wires_source_and_secret_guards() -> None:
    build = (
        ROOT / "scripts" / "build_windows_installer.ps1"
    ).read_text(encoding="utf-8")

    assert re.search(r"\$sourceCommit\s*=\s*Assert-CommittedBuildSource", build)
    assert re.search(r"\$scanArguments\s*=\s*@\(\s*\$publicReleaseScanner", build)
    assert re.search(r"&\s+\$builder\s+@scanArguments", build)
