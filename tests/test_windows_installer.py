from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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
    assert "_worker.SetLibraryDirectory(OutputFolderBox.Text);" in main_window
    assert "_worker?.WorkingDirectory" in main_window
    assert "AppDiagnostics.FindPreviousLibraryDirectory(" in main_window
    assert "OutputDirectory = PersistedOutputDirectory()" in main_window
    assert 'const string marker = "Repository: "' in diagnostics
    assert '"broadcastify-analysis.sqlite3"' in diagnostics


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
    assert "#if !FileExists(SourceDir + \"\\broadcastify-desktop.env\")" in installer
    assert 'Type: files; Name: "{app}\\broadcastify-desktop.env"' in installer


def test_public_installer_build_rejects_private_environment_and_pins_downloads() -> None:
    build = (
        ROOT / "scripts" / "build_windows_installer.ps1"
    ).read_text(encoding="utf-8")
    workflow = (
        ROOT / ".github" / "workflows" / "windows-release.yml"
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
    assert 'tags:' in workflow
    assert '"v*"' in workflow
    assert "gh release upload" in workflow
    assert "gh release create" in workflow
