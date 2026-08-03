from pathlib import Path
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
XAML_NAME = "{http://schemas.microsoft.com/winfx/2006/xaml}Name"


def test_area_assignment_brief_is_collapsed_in_native_and_web() -> None:
    root = ElementTree.parse(
        ROOT / "BroadcastifyCli.WinUI" / "MainWindow.xaml"
    ).getroot()
    native = next(
        element
        for element in root.iter()
        if element.tag.endswith("Expander")
        and element.attrib.get(XAML_NAME) == "AreaSummaryExpander"
    )

    assert native.attrib["Header"] == "Generated assignment brief"
    assert native.attrib["IsExpanded"] == "False"
    assert native.attrib["Visibility"] == "Collapsed"

    web = (ROOT / "broadcastify_cli" / "web_static" / "app.js").read_text(
        encoding="utf-8"
    )
    assert '<details class="area-narrative"><summary>Generated assignment brief</summary>' in web
    assert '<details class="area-narrative" open' not in web


def test_native_and_web_expose_preview_engine_and_accuracy_upgrade() -> None:
    root = ElementTree.parse(
        ROOT / "BroadcastifyCli.WinUI" / "MainWindow.xaml"
    ).getroot()
    names = {
        element.attrib.get(XAML_NAME): element
        for element in root.iter()
        if element.attrib.get(XAML_NAME)
    }

    engine = names["DiarizationEngineComboBox"]
    engine_tags = {
        item.attrib.get("Tag")
        for item in engine
        if item.tag.endswith("ComboBoxItem")
    }
    assert engine_tags == {"community-1", "sherpa-onnx"}
    assert names["LibrarySpeakerUpgradeButton"].attrib["Visibility"] == "Collapsed"
    assert (
        names["LibrarySpeakerUpgradeButton"].attrib["Click"]
        == "LibrarySpeakerUpgrade_Click"
    )

    web_html = (
        ROOT / "broadcastify_cli" / "web_static" / "index.html"
    ).read_text(encoding="utf-8")
    web_js = (
        ROOT / "broadcastify_cli" / "web_static" / "app.js"
    ).read_text(encoding="utf-8")
    assert 'id="settingDiarizationEngine"' in web_html
    assert '<option value="sherpa-onnx">Fast portable preview — CPU</option>' in web_html
    assert 'data-action="upgrade-speakers"' in web_js
    assert 'diarization_engine: "community-1"' in web_js
    assert "EnsureDiarizationSelectionCompatibility();" in (
        ROOT / "BroadcastifyCli.WinUI" / "MainWindow.xaml.cs"
    ).read_text(encoding="utf-8")
    assert (
        "Fast portable speaker preview runs on CPU; the speaker device was reset to CPU."
        in web_js
    )


def test_setup_copy_does_not_assume_every_transcription_engine_is_whisper() -> None:
    native = (
        ROOT / "BroadcastifyCli.WinUI" / "MainWindow.xaml"
    ).read_text(encoding="utf-8")
    web = (
        ROOT / "broadcastify_cli" / "web_static" / "app.js"
    ).read_text(encoding="utf-8")

    assert "Detecting the selected transcription engine and accelerator." in native
    assert "Detecting the selected Whisper engine and accelerator." not in native
    assert "choose a local transcription path" in web
    assert "choose a Whisper path" not in web


def test_native_and_web_expose_read_only_lan_archive_reuse() -> None:
    root = ElementTree.parse(
        ROOT / "BroadcastifyCli.WinUI" / "MainWindow.xaml"
    ).getroot()
    names = {
        element.attrib.get(XAML_NAME): element
        for element in root.iter()
        if element.attrib.get(XAML_NAME)
    }

    assert names["LanSyncToggle"].attrib["IsOn"] == "True"
    assert names["LanDiscoveryToggle"].attrib["IsOn"] == "True"
    assert names["LanShareToggle"].attrib["IsOn"] == "True"
    assert names["LanSharePortBox"].attrib["Value"] == "8766"

    native_worker = (
        ROOT / "BroadcastifyCli.WinUI" / "WorkerClient.cs"
    ).read_text(encoding="utf-8")
    assert '"-m", "broadcastify_cli.lan_node"' in native_worker
    assert '"--host", "0.0.0.0"' in native_worker

    web_html = (
        ROOT / "broadcastify_cli" / "web_static" / "index.html"
    ).read_text(encoding="utf-8")
    web_js = (
        ROOT / "broadcastify_cli" / "web_static" / "app.js"
    ).read_text(encoding="utf-8")
    assert 'id="settingLanSyncEnabled" type="checkbox" checked' in web_html
    assert 'id="settingLanDiscoveryEnabled" type="checkbox" checked' in web_html
    assert 'id="settingLanPeerUrls"' in web_html
    assert "lan_sync_enabled: Boolean(state.settings.lanSyncEnabled)" in web_js
    assert (
        "lan_discovery_enabled: Boolean(state.settings.lanDiscoveryEnabled)"
        in web_js
    )
    assert "lan_peer_urls: state.settings.lanPeerUrls" in web_js


def test_native_releases_media_handles_before_archive_mutation() -> None:
    native = (
        ROOT / "BroadcastifyCli.WinUI" / "MainWindow.xaml.cs"
    ).read_text(encoding="utf-8")

    assert "private async Task ReleaseMediaForArchiveMutationAsync()" in native
    assert native.count("await ReleaseMediaForArchiveMutationAsync();") == 2
    assert "_libraryMediaPlayer.Source = null;" in native
    assert "await Task.Delay(150);" in native


def test_native_visible_activity_log_is_bounded_without_truncating_disk_history() -> None:
    native = (
        ROOT / "BroadcastifyCli.WinUI" / "MainWindow.xaml.cs"
    ).read_text(encoding="utf-8")

    assert "MaximumVisibleActivityLogCharacters" in native
    assert "RetainedVisibleActivityLogCharacters" in native
    assert "Earlier activity remains available in the on-disk activity log" in native
    assert "_visibleActivityLog.Clear();" in native
    assert "AppDiagnostics.AppendActivity(message);" in native
    assert "LogBox.Text +=" not in native


def test_native_library_can_recheck_source_audio_for_complete_days() -> None:
    root = ElementTree.parse(
        ROOT / "BroadcastifyCli.WinUI" / "MainWindow.xaml"
    ).getroot()
    names = {
        element.attrib.get(XAML_NAME): element
        for element in root.iter()
        if element.attrib.get(XAML_NAME)
    }
    source_button = names["LibraryCheckSourceButton"]

    assert source_button.attrib["Content"] == "Check for new source audio"
    assert source_button.attrib["Click"] == "LibraryCheckSource_Click"

    native = (
        ROOT / "BroadcastifyCli.WinUI" / "MainWindow.xaml.cs"
    ).read_text(encoding="utf-8")
    assert "forceSourceCheck: true" in native
    assert "day.NeedsNetwork || forceSourceCheck" in native


def test_native_and_web_distinguish_working_storage_and_stale_transcripts() -> None:
    native_models = (
        ROOT / "BroadcastifyCli.WinUI" / "Models.cs"
    ).read_text(encoding="utf-8")
    native_window = (
        ROOT / "BroadcastifyCli.WinUI" / "MainWindow.xaml.cs"
    ).read_text(encoding="utf-8")
    web = (
        ROOT / "broadcastify_cli" / "web_static" / "app.js"
    ).read_text(encoding="utf-8")

    assert '"working_storage_bytes"' in native_models
    assert '"has_stale_transcript"' in native_models
    assert '"has_imported_transcript"' in native_models
    assert 'FormatBytes(StorageBytes, "retained")' in native_models
    assert 'FormatBytes(WorkingStorageBytes, "temporary")' in native_models
    assert "day.HasStaleTranscript" in native_window
    assert "day.HasImportedTranscript" in native_window
    assert "previous results are hidden" in native_window
    assert "function dayStorage(day)" in web
    assert "${bytes(working)} temporary" in web
    assert "day.has_stale_transcript" in web
    assert "day.has_imported_transcript" in web
    assert "previous transcript and its derived incidents" in web


def test_native_credentials_are_one_click_and_never_prefill_saved_secrets() -> None:
    root = ElementTree.parse(
        ROOT / "BroadcastifyCli.WinUI" / "MainWindow.xaml"
    ).getroot()
    names = {
        element.attrib.get(XAML_NAME): element
        for element in root.iter()
        if element.attrib.get(XAML_NAME)
    }

    assert names["CredentialsNavigationItem"].attrib["Tag"] == "credentials"
    assert names["HuggingFaceTokenBox"].attrib["PasswordRevealMode"] == "Peek"
    assert (
        names["SaveHuggingFaceTokenButton"].attrib["Click"]
        == "SaveHuggingFaceToken_Click"
    )

    native = (
        ROOT / "BroadcastifyCli.WinUI" / "MainWindow.xaml.cs"
    ).read_text(encoding="utf-8")
    worker = (
        ROOT / "BroadcastifyCli.WinUI" / "WorkerClient.cs"
    ).read_text(encoding="utf-8")
    assert "Password = saved?.Password" not in native
    assert "leave blank to reuse" in native
    assert 'startInfo.Environment["BROADCASTIFY_SECURE_PASSWORD"]' in worker
    assert 'startInfo.Environment["HUGGINGFACE_SECURE_TOKEN"]' in worker


def test_windows_startup_is_visible_configurable_and_recovery_aware() -> None:
    root = ElementTree.parse(
        ROOT / "BroadcastifyCli.WinUI" / "MainWindow.xaml"
    ).getroot()
    names = {
        element.attrib.get(XAML_NAME): element
        for element in root.iter()
        if element.attrib.get(XAML_NAME)
    }

    startup = names["StartWithWindowsToggle"]
    assert startup.attrib["Header"] == "Start Broadcastify Desktop when I sign in"
    assert startup.attrib["Toggled"] == "StartWithWindows_Toggled"
    assert startup.attrib["OnContent"] == "On — recommended"

    manager = (
        ROOT / "BroadcastifyCli.WinUI" / "WindowsStartupManager.cs"
    ).read_text(encoding="utf-8")
    app = (
        ROOT / "BroadcastifyCli.WinUI" / "App.xaml.cs"
    ).read_text(encoding="utf-8")
    window = (
        ROOT / "BroadcastifyCli.WinUI" / "MainWindow.xaml.cs"
    ).read_text(encoding="utf-8")
    worker = (
        ROOT / "BroadcastifyCli.WinUI" / "WorkerClient.cs"
    ).read_text(encoding="utf-8")

    assert r"Software\Microsoft\Windows\CurrentVersion\Run" in manager
    assert '" --startup --prompt-setup"' in manager
    assert 'commandLine.Contains("--startup")' in app
    assert 'commandLine.Contains("--prompt-setup")' in app
    assert "PromptForMissingAccountSetupAsync" in window
    assert "MissingUnattendedSetup" in window
    assert "_pauseScheduledJobsForSetup" in window
    assert "presenter.Minimize();" in window
    assert "Recovered {recovered} interrupted scheduled feed" in window
    assert 'Header = "Catch up from (optional)"' in window
    assert "BackfillStartDate = backfillPicker.Date" in window
    assert "result?.MissingDays.Count" in window
    assert 'Status = waitingForQuota' in window
    assert "Task<int> RecoverFeedSchedulesAsync" in worker
    assert window.index("await ApplyLaunchBehaviorAsync();") < window.index(
        "ConfigureFeedScheduleTimer();"
    )


def test_native_exposes_explicit_resumable_packaged_cuda_runtime() -> None:
    root = ElementTree.parse(
        ROOT / "BroadcastifyCli.WinUI" / "MainWindow.xaml"
    ).getroot()
    names = {
        element.attrib.get(XAML_NAME): element
        for element in root.iter()
        if element.attrib.get(XAML_NAME)
    }

    button = names["ManagedRuntimeInstallButton"]
    assert button.attrib["Click"] == "ManagedRuntimeInstall_Click"
    assert button.attrib["Content"] == "Check runtime"

    window = (
        ROOT / "BroadcastifyCli.WinUI" / "MainWindow.xaml.cs"
    ).read_text(encoding="utf-8")
    worker = (
        ROOT / "BroadcastifyCli.WinUI" / "WorkerClient.cs"
    ).read_text(encoding="utf-8")
    assert "Resume install" in window
    assert "No Broadcastify request is made" in window
    assert "process.Kill(entireProcessTree: true)" in worker
    assert '"BROADCASTIFY_MANAGED_RUNTIME_ROOT"' in worker
