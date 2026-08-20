import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_release_version_is_consistent_across_all_entry_points() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', pyproject, re.MULTILINE)
    assert match is not None
    version = match.group(1)
    version_parts = version.split(".")
    assert len(version_parts) in (3, 4)
    numeric = version if len(version_parts) == 4 else f"{version}.0"

    for project in (
        ROOT / "BroadcastifyCli.WinUI" / "BroadcastifyCli.WinUI.csproj",
        ROOT / "BroadcastifyCli.WindowsML" / "BroadcastifyCli.WindowsML.csproj",
    ):
        content = project.read_text(encoding="utf-8")
        assert f"<Version>{version}</Version>" in content
        assert f"<FileVersion>{numeric}</FileVersion>" in content

    package = (ROOT / "broadcastify_cli" / "__init__.py").read_text(
        encoding="utf-8"
    )
    installer = (
        ROOT / "installer" / "BroadcastifyDesktop.iss"
    ).read_text(encoding="utf-8")
    assert f'__version__ = "{version}"' in package
    assert f'#define MyAppVersion "{version}"' in installer
    assert f'#define MyAppVersionNumeric "{numeric}"' in installer


def test_installed_worker_uses_bundled_runtime_and_writable_data_root() -> None:
    worker = (
        ROOT / "BroadcastifyCli.WinUI" / "WorkerClient.cs"
    ).read_text(encoding="utf-8")
    main_window = (
        ROOT / "BroadcastifyCli.WinUI" / "MainWindow.xaml.cs"
    ).read_text(encoding="utf-8")
    diagnostics = (
        ROOT / "BroadcastifyCli.WinUI" / "AppDiagnostics.cs"
    ).read_text(encoding="utf-8")

    assert '"runtime",' in worker
    assert '"python.exe"' in worker
    assert "WorkingDirectory = IsBundledRuntime" in worker
    assert "? AppSettingsStore.LocalDataDirectory" in worker
    assert "WorkingDirectory = WorkingDirectory" in worker
    assert 'startInfo.Environment["PYTHONNOUSERSITE"] = "1"' in worker
    assert 'startInfo.Environment["PYTHONDONTWRITEBYTECODE"] = "1"' in worker
    assert 'startInfo.Environment["FFMPEG_PATH"] = ffmpeg' in worker
    assert 'startInfo.Environment["BROADCASTIFY_SECURE_ANALYSIS_DB"]' in worker
    assert 'startInfo.Environment["BROADCASTIFY_LIBRARY_ROOT"]' in worker
    assert "_worker.SetLibraryDirectory(OutputFolderBox.Text);" in main_window
    assert "_worker?.WorkingDirectory" in main_window
    assert "AppDiagnostics.FindPreviousLibraryDirectory(" in main_window
    assert "OutputDirectory = PersistedOutputDirectory()" in main_window
    assert "Storage follows the currently selected Library" in main_window
    assert 'const string marker = "Repository: "' in diagnostics
    assert '"broadcastify-analysis.sqlite3"' in diagnostics


def test_native_publish_restores_the_sibling_windows_ml_helper() -> None:
    project = (
        ROOT / "BroadcastifyCli.WinUI" / "BroadcastifyCli.WinUI.csproj"
    ).read_text(encoding="utf-8")

    assert 'Projects="$(WindowsMLProjectPath)"' in project
    assert 'Targets="Restore;Build"' in project


def test_installer_is_per_user_upgrade_safe_and_preserves_app_data() -> None:
    installer = (
        ROOT / "installer" / "BroadcastifyDesktop.iss"
    ).read_text(encoding="utf-8")

    assert "DefaultDirName={localappdata}\\Programs\\{#MyAppName}" in installer
    assert "PrivilegesRequired=lowest" in installer
    assert "AppId={{3B7D8D50-659D-4C72-9A84-0DA39472F8A3}" in installer
    assert "Flags: uninsneveruninstall" in installer
    assert 'Type: filesandordirs; Name: "{app}"' in installer
    assert "Create a &desktop shortcut" in installer
    assert "skipifsilent" in installer
    assert 'Type: filesandordirs; Name: "{app}\\runtime"' in installer
    assert 'Type: filesandordirs; Name: "{app}\\windowsml"' in installer
    assert 'Type: files; Name: "{app}\\*.pdb"' in installer
    assert "#if !FileExists(SourceDir + \"\\broadcastify-desktop.env\")" in installer
    assert 'Type: files; Name: "{app}\\broadcastify-desktop.env"' in installer
    assert "Keep scheduled feeds current" in installer
    assert "Start Broadcastify Desktop when I sign in (recommended)" in installer
    assert "StartupPage.Values[0] := True" in installer
    assert "if WizardSilent then" in installer
    assert "HasCommandLineFlag('ENABLESTARTUP')" in installer
    assert "HasCommandLineFlag('DISABLESTARTUP')" in installer
    assert "HasCommandLineFlag('LAUNCHAFTERINSTALL')" in installer
    assert 'Parameters: "--post-install --prompt-setup"' in installer
    assert '\'" --startup --prompt-setup\'' in installer


def test_public_installer_build_rejects_private_environment_and_pins_downloads() -> None:
    build = (
        ROOT / "scripts" / "build_windows_installer.ps1"
    ).read_text(encoding="utf-8")
    workflow = (
        ROOT / ".github" / "workflows" / "windows-release.yml"
    ).read_text(encoding="utf-8")
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    constraints = (
        ROOT / "installer" / "windows-runtime-constraints.txt"
    ).read_text(encoding="utf-8")
    cuda_lock = (
        ROOT / "installer" / "windows-managed-cuda-lock.txt"
    ).read_text(encoding="utf-8")

    assert "4ACBED6DD1C744B0376E3B1CF57CE906F9DC9E95E68824584C8099A63025A3C3" in build
    assert "DB580001CAA24AC104C8CB856CD113A87B0A443F7BDF47D8C12B1D740584A2EC" in build
    assert 'throw "A public installer stage contains broadcastify-desktop.env."' in build
    assert '"dist\\windows-private"' in build
    assert (
        "A private environment build cannot write to the public release directory."
        in build
    )
    assert "$normalizedOutput.StartsWith(" in build
    assert "$publicOutputPrefix" in build
    assert "does not match pyproject.toml version" in build
    assert "$repositoryRoot[windowsml,qwen,portable-diarization]" in build
    assert "--constraint $constraints" in build
    assert "$env:PYTHONDONTWRITEBYTECODE = \"1\"" in build
    assert '"-p:SelfContained=true"' in build
    assert '"-p:DebugType=None"' in build
    assert '"windowsml\\hostfxr.dll"' in build
    assert "tzdata>=2026.3,<2027" in project
    assert "tzdata==2026.3" in constraints
    assert '"numpy>=1.26,<2.5"' in project
    assert "numpy==2.4.3" in constraints
    assert '$uvVersion = "0.12.1"' in build
    assert "$cudaRequirementsSha256" in build
    assert 'requirements_artifact = "cuda_requirements"' in build
    assert "--no-build-isolation" not in build
    assert "torch_backend" not in build
    assert "torch @ https://download-r2.pytorch.org/whl/cu128/" in cuda_lock
    assert "torchaudio @ https://download-r2.pytorch.org/whl/cu128/" in cuda_lock
    assert "torchcodec==0.14.0" in cuda_lock
    assert "ZoneInfo('America/Chicago')" in build
    assert "$unexpectedPdbFiles.Count -gt 0" in build
    assert "dotnet clean $project -c Release" in build
    assert "scan_public_release.py" in build
    assert '"--forbid-path", $repositoryRoot' in build
    assert '"--forbid-user-profile", $env:USERPROFILE' in build
    assert "The public secret and forbidden-file scan failed." in build
    assert "Assert-CommittedBuildSource" in build
    guard = (ROOT / "scripts" / "assert_committed_build_source.ps1").read_text()
    targets = (ROOT / "Directory.Build.targets").read_text()
    assert "assert_committed_build_source.ps1" in build
    assert "status --porcelain --untracked-files=all" in guard
    assert "Commit every source, test, documentation, version" in guard
    assert "RequireCommittedBuildSource" in targets
    assert "BeforeTargets=\"PrepareForBuild\"" in targets
    assert '"-p:SourceRevisionId=$sourceCommit"' in build
    assert "source_commit = $sourceCommit" in build
    assert 'tags:' in workflow
    assert '"v*"' in workflow
    assert "gh release upload" in workflow
    assert "gh release create" in workflow


def test_public_release_scanner_rejects_machine_paths_and_forbidden_files(
    tmp_path: Path,
) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "application.dll").write_bytes(
        b"prefix C:\\private\\source\\application.pdb suffix"
    )
    (stage / "settings.json").write_text("{}", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "scan_public_release.py"),
            "--root",
            str(stage),
            "--forbid-path",
            r"C:\private\source",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

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

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "scan_public_release.py"),
            "--root",
            str(stage),
            "--forbid-path",
            r"C:\private\source",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

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

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "scan_public_release.py"),
            "--root",
            str(stage),
            "--forbid-path",
            r"D:\a\broadcastify-cli\broadcastify-cli",
            "--forbid-user-profile",
            r"C:\Users\runneradmin",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
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

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "scan_public_release.py"),
            "--root",
            str(stage),
            "--forbid-user-profile",
            r"C:\Users\runneradmin",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )

    assert result.returncode == 1
    assert "application.dll: contains build path 1" in result.stderr


def test_tagged_release_uses_the_shipped_windows_product_name() -> None:
    workflow = (
        ROOT / ".github" / "workflows" / "windows-release.yml"
    ).read_text(encoding="utf-8")

    assert '--title "Broadcastify Desktop $env:RELEASE_VERSION"' in workflow


def test_windows_release_workflow_isolates_and_validates_shell_inputs() -> None:
    workflow = (
        ROOT / ".github" / "workflows" / "windows-release.yml"
    ).read_text(encoding="utf-8")

    assert "REQUESTED_VERSION: ${{ inputs.version }}" in workflow
    assert "REF_NAME: ${{ github.ref_name }}" in workflow
    assert "REF_TYPE: ${{ github.ref_type }}" in workflow
    assert "$requested = $env:REQUESTED_VERSION" in workflow
    assert "Release tags must be v<major>.<minor>.<patch>[.<revision>]." in workflow
    assert "Release version must contain exactly three or four numeric components." in workflow
    assert "The requested version must match the pushed release tag." in workflow
    assert "RELEASE_VERSION: ${{ steps.version.outputs.value }}" in workflow
    assert "-Version $env:RELEASE_VERSION" in workflow
    assert "RELEASE_TAG: ${{ github.ref_name }}" in workflow
    assert "gh release view $env:RELEASE_TAG" in workflow
    assert "gh release upload $env:RELEASE_TAG" in workflow
    assert "gh release create $env:RELEASE_TAG" in workflow
    assert '"${{ inputs.version }}"' not in workflow
    assert '"${{ github.ref_name }}"' not in workflow
    assert '"${{ steps.version.outputs.value }}"' not in workflow


def test_windows_release_runs_the_guarded_installer_lifecycle_before_upload() -> None:
    workflow = (
        ROOT / ".github" / "workflows" / "windows-release.yml"
    ).read_text(encoding="utf-8")
    lifecycle = (
        ROOT / "scripts" / "run_windows_installer_lifecycle.ps1"
    ).read_text(encoding="utf-8")

    assert "run_windows_installer_lifecycle.ps1" in workflow
    assert "INSTALLER_PATH: ${{ github.workspace }}\\dist\\windows\\BroadcastifyDesktop-${{ steps.version.outputs.value }}-win-x64-setup.exe" in workflow
    assert "-Installer $env:INSTALLER_PATH" in workflow
    assert workflow.index("Exercise installer lifecycle") < workflow.index(
        "Upload workflow artifact"
    )
    assert "GITHUB_ACTIONS=true" in lifecycle
    assert "RUNNER_TEMP" in lifecycle
    assert "Assert-ContainedPath" in lifecycle
    assert "Start-Process" in lifecycle
    assert "-WindowStyle Hidden" in lifecycle
    assert "/DISABLESTARTUP" in lifecycle
    assert "/VERYSILENT" in lifecycle
    assert "runtime\\python\\python.exe" in lifecycle
    assert "run_windows_ui_e2e.ps1" in lifecycle
    assert "[System.IO.File]::WriteAllText" in lifecycle
    assert "[DateTime]::UtcNow.AddSeconds(15)" in lifecycle
    assert "same-version upgrade" in lifecycle
    assert "retained lifecycle sentinel" in lifecycle
    assert "Uninstall left the application directory behind" in lifecycle
    assert "Final uninstall did not preserve" in lifecycle
    assert "Remove-Item -LiteralPath $verifiedData -Recurse -Force" in lifecycle


def test_windows_shell_and_optional_helper_minimums_are_documented() -> None:
    shell = (
        ROOT / "BroadcastifyCli.WinUI" / "BroadcastifyCli.WinUI.csproj"
    ).read_text(encoding="utf-8")
    helper = (
        ROOT / "BroadcastifyCli.WindowsML" / "BroadcastifyCli.WindowsML.csproj"
    ).read_text(encoding="utf-8")
    setup = (ROOT / "docs" / "guides" / "windows-setup.md").read_text(
        encoding="utf-8"
    )

    assert "<TargetPlatformMinVersion>10.0.17763.0</TargetPlatformMinVersion>" in shell
    assert "<TargetPlatformMinVersion>10.0.26100.0</TargetPlatformMinVersion>" in helper
    assert "Windows 10 version 1809" in setup
    assert "Windows 11 24H2" in setup
    assert "optional Windows ML helper" in setup


def test_maintenance_update_uses_checkpointed_shutdown_without_force_kill() -> None:
    stop = (
        ROOT / "scripts" / "stop_windows_desktop_for_maintenance.ps1"
    ).read_text(encoding="utf-8")
    install = (
        ROOT / "scripts" / "install_windows_update.ps1"
    ).read_text(encoding="utf-8")

    assert "CloseMainWindow()" in stop
    assert "Get-CimInstance Win32_Process" in stop
    assert "broadcastify_cli.worker" in stop
    assert 'Name -ieq "python.exe"' in stop
    assert 'Name -ieq "pythonw.exe"' in stop
    assert "No process was force-killed; maintenance was refused." in stop
    assert "Stop-Process" not in stop
    assert "stop_windows_desktop_for_maintenance.ps1" in install
    assert '"/VERYSILENT"' in install
    assert '"/SUPPRESSMSGBOXES"' in install
    assert '"/NORESTART"' in install
    assert "[switch]$Restart" in install
    assert "if ($Restart)" in install
    assert "LAUNCHAFTERINSTALL" not in install


def test_every_desktop_build_requires_the_full_offline_product_gate() -> None:
    targets = (ROOT / "Directory.Build.targets").read_text(encoding="utf-8")
    gate = (ROOT / "scripts" / "run_product_regression_gate.ps1").read_text(
        encoding="utf-8"
    )
    workflow = (
        ROOT / ".github" / "workflows" / "windows-release.yml"
    ).read_text(encoding="utf-8")
    installer = (
        ROOT / "scripts" / "build_windows_installer.ps1"
    ).read_text(encoding="utf-8")
    ui_e2e = (
        ROOT / "scripts" / "run_windows_ui_e2e.ps1"
    ).read_text(encoding="utf-8")

    assert 'Name="RunProductRegressionGate"' in targets
    assert 'DependsOnTargets="RequireCommittedBuildSource"' in targets
    assert "scripts\\run_product_regression_gate.ps1" in targets
    assert 'Name="RunNativeUiEndToEndGate"' in targets
    assert 'AfterTargets="Build"' in targets
    assert "scripts\\run_windows_ui_e2e.ps1" in targets
    assert '$(TargetDir)$(AssemblyName).exe' in targets
    assert '$(TargetPath)&quot;' not in targets
    assert "tests\\e2e\\test_product_workflows.py" in gate
    assert '$arguments = @("-m", "pytest", "-q")' in gate
    assert "BROADCASTIFY_PRODUCT_REGRESSION_GATE" in gate
    assert "BROADCASTIFY_USERNAME" in gate
    assert "BROADCASTIFY_SECURE_PASSWORD" in gate
    assert "BROADCASTIFY_DESKTOP_TEST_DATA_ROOT" in ui_e2e
    assert "--ui-e2e-report" in ui_e2e
    assert "DesktopDockSide = \"right\"" in ui_e2e
    assert "DesktopDockSide = \"left\"" in ui_e2e
    assert "DesktopDockWidth = 640" in ui_e2e
    assert "BROADCASTIFY_DESKTOP_TEST_ABRUPT_EXIT" in ui_e2e
    assert 'BROADCASTIFY_DESKTOP_E2E_ISOLATED"] = "1"' in ui_e2e
    assert "Get-DesktopWorkAreaSignature" in ui_e2e
    assert "abrupt_cleanup = $abruptCleanupPassed" in ui_e2e
    assert "UIAutomation" not in ui_e2e
    assert "SendKeys" not in ui_e2e
    assert 'pip install --disable-pip-version-check -e ".[dev]"' in workflow
    assert "dotnet publish $project" in installer
    assert "ProductRegressionGate=false" not in installer
