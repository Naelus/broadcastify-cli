import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_release_version_is_consistent_across_all_entry_points() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', pyproject, re.MULTILINE)
    assert match is not None
    version = match.group(1)
    numeric = f"{version}.0"

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


def test_tagged_release_uses_the_shipped_windows_product_name() -> None:
    workflow = (
        ROOT / ".github" / "workflows" / "windows-release.yml"
    ).read_text(encoding="utf-8")

    assert '--title "Broadcastify Desktop ${{ steps.version.outputs.value }}"' in workflow


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
