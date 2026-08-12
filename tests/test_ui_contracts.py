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

    assert "private async Task ReleaseMediaForArchiveMutationAsync(" in native
    assert native.count("await ReleaseMediaForArchiveMutationAsync();") == 2
    assert "ReleaseMediaForArchiveMutationAsync(recreatePlayers: false)" in native
    assert "_libraryMediaPlayer.Source = null;" in native
    assert "ReleaseMediaPlayerInstances(recreate: recreatePlayers);" in native
    assert "ReleaseMediaPlayerInstances(recreate: false);" in native
    assert "LibraryAudioPlayer.SetMediaPlayer(null);" in native
    assert "_libraryMediaPlayer.Dispose();" in native
    assert "_libraryMediaPlayer = new MediaPlayer();" in native
    assert "await Task.Delay(500);" in native


def test_native_visible_activity_log_is_bounded_without_truncating_disk_history() -> None:
    native = (
        ROOT / "BroadcastifyCli.WinUI" / "MainWindow.xaml.cs"
    ).read_text(encoding="utf-8")
    worker = (
        ROOT / "BroadcastifyCli.WinUI" / "WorkerClient.cs"
    ).read_text(encoding="utf-8")

    assert "MaximumVisibleActivityLogCharacters" in native
    assert "RetainedVisibleActivityLogCharacters" in native
    assert "Earlier activity remains available in the on-disk activity log" in native
    assert "_visibleActivityLog.Clear();" in native
    assert "AppDiagnostics.AppendActivity(message);" in native
    assert "LogBox.Text +=" not in native
    assert "ConcurrentQueue<JsonElement> _pendingWorkerMessages" in native
    assert "ScheduleWorkerMessageDrain" in native
    assert "DrainWorkerMessages" in native
    assert "while (_pendingWorkerMessages.TryDequeue" in native
    assert "Task.Run(async () =>" in worker
    assert "ReadLineAsync(" in worker
    assert "ConfigureAwait(false)" in worker


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


def test_native_library_exposes_guarded_feed_delete_and_resume_all() -> None:
    root = ElementTree.parse(
        ROOT / "BroadcastifyCli.WinUI" / "MainWindow.xaml"
    ).getroot()
    names = {
        element.attrib.get(XAML_NAME): element
        for element in root.iter()
        if element.attrib.get(XAML_NAME)
    }

    assert names["ResumeAllLibraryButton"].attrib["Content"] == (
        "Resume / prioritize…"
    )
    assert names["ResumeAllLibraryButton"].attrib["Click"] == (
        "ResumeAllLibrary_Click"
    )
    assert names["CatchUpFeedButton"].attrib["Content"] == (
        "Catch up missing feed days…"
    )
    assert names["CatchUpFeedButton"].attrib["Click"] == "CatchUpFeed_Click"
    assert names["LibraryDeleteFeedButton"].attrib["Content"] == "Delete feed"
    assert names["LibraryDeleteFeedButton"].attrib["Click"] == (
        "LibraryDeleteFeed_Click"
    )
    assert names["LibraryActionInfoBar"].attrib["IsOpen"] == "False"

    native = (
        ROOT / "BroadcastifyCli.WinUI" / "MainWindow.xaml.cs"
    ).read_text(encoding="utf-8")
    worker = (
        ROOT / "BroadcastifyCli.WinUI" / "WorkerClient.cs"
    ).read_text(encoding="utf-8")
    assert "DefaultButton = ContentDialogButton.Close" in native
    assert "Finish retained local processing" in native
    assert "Check/download missing source audio now" in native
    assert "A chosen feed first" in native
    assert "value.NeedsLocalProcessing" in native
    assert "finishing its retained local stages now" in native
    assert "if (quota is null || !quota.Available)" in native
    assert "result?.DownloadLimited == true" in native
    assert "DownloadJobs = 1" in native
    assert "ShowLibraryCatchUpRangeAsync" in native
    assert "queues only absent or unfinished days" in native
    assert "RunLibraryResumePlanAsync" in native
    assert "automatically extends through the then-current day" in native
    assert "endPicker" not in native
    assert "SaveLibraryCatchUpAsync" in native
    assert "FinalizeLibraryCatchUpsAsync" in native
    assert 'LibraryActionInfoBar.Title = "Feed deleted"' in native
    assert 'Removed feed {result.FeedId} from the selected Library.' in native
    assert 'DaysDeleted:N0} local day(s)' not in native
    assert 'await ShowMessageAsync(\n                "Feed deleted"' not in native
    assert '"library-resume-plan"' in worker
    assert '"save-library-catch-up"' in worker
    assert '"finalize-library-catch-ups"' in worker
    assert '"--start-date"' in worker
    assert '"--through-current"' in worker
    assert "through_current = throughCurrent" in worker
    assert '"delete-library-feed"' in worker


def test_native_ui_smoke_launcher_isolated_from_real_user_data() -> None:
    settings = (
        ROOT / "BroadcastifyCli.WinUI" / "AppSettingsStore.cs"
    ).read_text()
    launcher = (
        ROOT / "scripts" / "launch_windows_ui_smoke.ps1"
    ).read_text()

    assert "BROADCASTIFY_DESKTOP_TEST_DATA_ROOT" in settings
    assert "!UsesTestDataRoot" in settings
    assert "broadcastify-desktop-ui-smoke-" in launcher
    assert "save-library-catch-up" in launcher
    assert "through_current = $true" in launcher
    assert "BROADCASTIFY_SECURE_ANALYSIS_DB" in launcher
    assert "System.Diagnostics.ProcessStartInfo" in launcher
    assert "UIAutomation" not in launcher
    assert "SendKeys" not in launcher


def test_native_window_exposes_real_appbar_docking_and_compact_layouts() -> None:
    root = ElementTree.parse(
        ROOT / "BroadcastifyCli.WinUI" / "MainWindow.xaml"
    ).getroot()
    names = {
        element.attrib.get(XAML_NAME): element
        for element in root.iter()
        if element.attrib.get(XAML_NAME)
    }

    assert names["DockButton"].attrib["Content"] == "Pin"
    assert names["DockButton"].attrib["Click"] == "DockButton_Click"
    assert names["DockButton"].attrib["RightTapped"] == "DockButton_RightTapped"
    assert names["DockLeftMenuItem"].attrib["Text"] == "Pin left"
    assert names["DockRightMenuItem"].attrib["Text"] == "Pin right"
    assert names["DockUnpinMenuItem"].attrib["Text"] == "Unpin"
    assert names["DockResizeHandle"].tag.endswith("DockResizeGrip")
    assert names["DockResizeHandle"].attrib["PointerPressed"] == (
        "DockResizeHandle_PointerPressed"
    )
    assert names["DockResizeHandle"].attrib["PointerMoved"] == (
        "DockResizeHandle_PointerMoved"
    )
    assert names["DockResizeHandle"].attrib["PointerReleased"] == (
        "DockResizeHandle_PointerReleased"
    )
    assert names["DockResizeHandle"].attrib["PointerCanceled"] == (
        "DockResizeHandle_PointerCanceled"
    )
    assert names["DockResizeHandle"].attrib["PointerCaptureLost"] == (
        "DockResizeHandle_PointerCaptureLost"
    )
    assert names["DockResizeHandle"].attrib["Visibility"] == "Collapsed"
    assert names["DockResizeHandle"].attrib["Width"] == "10"
    assert names["DockedFrameBorder"].attrib["Visibility"] == "Collapsed"
    assert names["DockedFrameBorder"].attrib["IsHitTestVisible"] == "False"
    assert names["DockedFrameBorder"].attrib["BorderThickness"] == "0"
    assert names["LibraryDetailTabView"].tag.endswith("TabView")
    assert names["ActivityLogExpander"].tag.endswith("Expander")
    assert names["HardwareProfilesExpander"].tag.endswith("Expander")

    for name in (
        "LibraryContentGrid",
        "ArchiveContentGrid",
        "ReviewContentGrid",
        "AreaFeedsGrid",
        "AreaStoryContentGrid",
        "PersistentStatusGrid",
    ):
        assert names[name].tag.endswith("Grid")

    docking = (
        ROOT / "BroadcastifyCli.WinUI" / "DesktopDockManager.cs"
    ).read_text(encoding="utf-8")
    native = (
        ROOT / "BroadcastifyCli.WinUI" / "MainWindow.xaml.cs"
    ).read_text(encoding="utf-8")
    settings = (
        ROOT / "BroadcastifyCli.WinUI" / "AppSettingsStore.cs"
    ).read_text(encoding="utf-8")
    policy = (
        ROOT / "BroadcastifyCli.WinUI" / "DesktopDockPolicy.cs"
    ).read_text(encoding="utf-8")
    grip = (
        ROOT / "BroadcastifyCli.WinUI" / "DockResizeGrip.cs"
    ).read_text(encoding="utf-8")
    ui_e2e = (
        ROOT / "BroadcastifyCli.WinUI" / "MainWindow.UiE2E.cs"
    ).read_text(encoding="utf-8")

    assert "SHAppBarMessage(AbmNew" in docking
    assert "SHAppBarMessage(AbmQueryPos" in docking
    assert "SHAppBarMessage(AbmSetPos" in docking
    assert "SHAppBarMessage(AbmRemove" in docking
    assert "SHAppBarMessage(AbmActivate" in docking
    assert "SHAppBarMessage(AbmWindowPosChanged" in docking
    assert 'RegisterWindowMessage("TaskbarCreated")' in docking
    assert "message == WmNcCalcSize && IsDocked" in docking
    assert "SetCapture(_windowHandle)" in docking
    assert "GetCursorPos" in docking
    assert "ReleaseCapture" in docking
    assert "RemoveAppBar();" in docking
    assert "EnumDisplayMonitors" in docking
    assert "GetDpiForMonitor" in docking
    assert "MonitorDeviceName" in docking
    assert "RestoreFloatingWindowForStartup" in docking
    assert "NormalizeFloatingBounds" in docking
    assert "MinimumWidthDips = 520" in policy
    assert "RecommendedWidthDips = 600" in policy
    assert "MaximumWidthDips = 960" in policy
    assert "availableWidth / 2" in policy
    assert "DipsToPixels" in policy
    assert "PixelsToDips" in policy
    assert "presenter.IsResizable = !pinned" in docking
    assert "presenter.IsMinimizable = !pinned" in docking
    assert "presenter.IsMaximizable = !pinned" in docking
    assert "InputSystemCursorShape.SizeWestEast" in grip

    assert "ApplyResponsiveLayout" in native
    assert "width < 1_100" in native
    assert "NavigationViewPaneDisplayMode.LeftMinimal" in native
    assert "Grid.SetRow(LibraryDetailBorder, 1)" in native
    assert "Grid.SetRow(ArchiveJobBorder, 1)" in native
    assert "Grid.SetRow(ReviewTabView, 1)" in native
    assert "shortCompact ? 170 : 280" in native
    assert "ReviewDaySelectorGrid.RowSpacing = shortCompact ? 4 : 10" in native
    assert 'AnalysisFeedCombo.Header = shortCompact ? null : "Feed"' in native
    assert "DailyReviewContentGrid.RowSpacing = shortCompact ? 4 : 9" in native
    assert "IncidentPlayer.Height = shortCompact ? 44 : 72" in native
    assert "Grid.SetRow(AreaQueueBorder, 1)" in native
    assert "Grid.SetRow(AreaStoryDetailBorder, 1)" in native
    assert "AreaStoriesContentGrid.RowSpacing = shortCompact ? 4 : 10" in native
    assert "AreaSummaryScroll.MaxHeight = shortCompact ? 64 : 150" in native
    assert "Started with Windows and kept the explicitly pinned status window visible" in native
    assert "presenter.SetBorderAndTitleBar(hasBorder: false, hasTitleBar: false)" in native
    assert "presenter.SetBorderAndTitleBar(hasBorder: true, hasTitleBar: false)" in native
    assert "DwmWindowCornerPreference" in native
    assert "DwmWindowBorderColor" in native
    assert "DwmCornerDoNotRound" in native
    assert "DwmColorNone" in native
    assert "DockedFrameBorder.BorderThickness" in native
    assert "ApplyDesktopDockActivationState" in native
    assert native.count("new ContentDialog") == 10
    assert native.count("Content = CreateDialog") == 10
    assert '"DialogScrollHost"' in native

    assert "public int Version { get; init; } = 9" in settings
    assert "public string DesktopDockSide" in settings
    assert "public double DesktopDockWidth" in settings
    assert "public string DesktopDockMonitor" in settings
    assert "public int? DesktopWindowX" in settings
    assert "public int? DesktopWindowY" in settings
    assert "public int? DesktopWindowWidth" in settings
    assert "public int? DesktopWindowHeight" in settings
    assert "public bool DesktopWindowMaximized" in settings
    assert "if (settings.Version < 8)" in settings
    assert "if (settings.Version < 9)" in settings
    assert 'new UiProbeSize("pinned-width", 600, 900, true)' in ui_e2e
    assert 'new UiProbeSize("minimum", 520, 640, true)' in ui_e2e
    assert 'new UiProbeSize("barely-overflowing", 680, 840, true)' in ui_e2e
    assert 'new UiProbeSize("compact-short", 720, 720, true)' in ui_e2e
    assert 'new UiProbeSize("medium", 960, 720, true)' in ui_e2e
    assert 'new UiProbeSize("threshold-below", 1099, 760, true)' in ui_e2e
    assert 'new UiProbeSize("threshold-above", 1101, 760, false)' in ui_e2e
    assert 'new UiProbeSize("reference", 1240, 900, false)' in ui_e2e
    assert 'new UiProbeSize("large", 1600, 1000, false)' in ui_e2e
    assert "ResizeForClientSizeAsync" in ui_e2e
    assert "Math.Abs(WindowRoot.ActualWidth - size.WidthDips) <= 3" in ui_e2e
    assert "ExerciseScrollAsync" in ui_e2e
    assert "ExerciseItemsScrollAsync" in ui_e2e
    assert "ExerciseTextBoxScrollAsync" in ui_e2e
    assert "RunUiDockingProbeAsync" in ui_e2e
    assert 'VerifyDockedSurfaceMatrixAsync("left")' in ui_e2e
    assert 'VerifyDockedSurfaceMatrixAsync("right")' in ui_e2e
    assert "VerifyDockedDialogAndFlyoutAsync" in ui_e2e
    assert "ExerciseInteractiveResizeForEndToEndTest" in ui_e2e
    assert "ReRegisterAfterShellRestartForEndToEndTest" in ui_e2e
    assert "CaptureAvailableMonitors" in ui_e2e
    assert "DwmCornerDoNotRound" in ui_e2e
    assert "DwmColorNone" in ui_e2e
    assert "OverlappedPresenterState.Maximized" in ui_e2e
    assert "VerifyMonthQuestionUiAsync" in ui_e2e
    assert "State the archive date and time for every event mentioned" in ui_e2e

    for element in root.iter():
        name = element.attrib.get(XAML_NAME)
        if name and (
            element.tag.endswith("ScrollViewer")
            or element.tag.endswith("ListView")
        ):
            assert name in ui_e2e, f"{name} is missing from the native UI E2E gate"
    for name in ("LibraryTranscriptPreviewText", "LogBox"):
        assert name in ui_e2e


def test_native_library_shows_feed_coverage_and_named_archive_chat() -> None:
    root = ElementTree.parse(
        ROOT / "BroadcastifyCli.WinUI" / "MainWindow.xaml"
    ).getroot()
    names = {
        element.attrib.get(XAML_NAME): element
        for element in root.iter()
        if element.attrib.get(XAML_NAME)
    }

    assert names["LibraryFeedCoverageList"].attrib["SelectionMode"] == "Single"
    assert names["AnalysisFeedCombo"].attrib["DisplayMemberPath"] == "FeedLabel"
    assert names["ArchiveChatList"].attrib["SelectionMode"] == "None"
    assert names["ArchiveChatScroll"].tag.endswith("ScrollViewer")
    assert names["ArchiveQuestionScopePanel"].tag.endswith("StackPanel")
    assert names["ArchiveQuestionStarterPanel"].tag.endswith("StackPanel")
    assert names["AskButton"].attrib["Content"] == "Send"
    assert names["UseQuestionMonthButton"].attrib["Content"] == "Use selected month"
    assert names["UseEntireQuestionFeedButton"].attrib["Content"] == (
        "Use entire downloaded feed"
    )
    assert names["CheckQuestionCoverageButton"].attrib["Content"] == "Check downloaded coverage"
    assert names["QuestionCoverageInfoBar"].attrib["IsOpen"] == "True"

    native = (
        ROOT / "BroadcastifyCli.WinUI" / "MainWindow.xaml.cs"
    ).read_text(encoding="utf-8")
    models = (
        ROOT / "BroadcastifyCli.WinUI" / "Models.cs"
    ).read_text(encoding="utf-8")
    assert "ShowResumeAllOptionsAsync" in native
    assert "HandleQuestionWorkerMessage" in native
    assert "ApplyQuestionMonthAsync" in native
    assert "ApplyEntireQuestionFeedAsync" in native
    assert "AskFeedHotspotsExample_Click" in native
    assert "State the archive date and time for every event mentioned" in native
    assert "Include exact archive dates and times for representative events" in native
    assert "RefreshQuestionCoverageAsync" in native
    clear_chat = native.split(
        "private void ClearArchiveChat_Click", maxsplit=1
    )[1].split("private void AskShotsExample_Click", maxsplit=1)[0]
    assert "ArchiveChatScroll.ChangeView(null, 0, null, true)" in clear_chat
    assert "QuestionBox.Focus" not in clear_chat
    assert "QuestionReadyDayCount" in native
    assert "_pipelineCancellation" in native
    assert "_questionCancellation" in native
    assert "_analysisOperationGate" in native
    assert "Analyze = false" in native
    assert "IsFeedBusy(selected.FeedId)" in native
    assert "IsFeedPipelineBusy(_selectedLibraryDay.FeedId)" in native
    assert "Clip playback and export are temporarily held" in native
    assert 'JsonPropertyName("history")' in models
    assert 'JsonPropertyName("output_dir")' in models
    assert 'JsonPropertyName("question_ready_dates")' in models
    assert 'JsonPropertyName("question_ready_ranges")' in models
    assert 'JsonPropertyName("coverage")' in models

    worker = (
        ROOT / "BroadcastifyCli.WinUI" / "WorkerClient.cs"
    ).read_text(encoding="utf-8")
    assert '"question-coverage"' in worker
    assert "GetArchiveQuestionCoverageAsync" in worker
    assert "GetEntireFeedQuestionCoverageAsync" in worker

    web_html = (
        ROOT / "broadcastify_cli" / "web_static" / "index.html"
    ).read_text(encoding="utf-8")
    web_script = (
        ROOT / "broadcastify_cli" / "web_static" / "app.js"
    ).read_text(encoding="utf-8")
    assert 'id="askMonth"' in web_html
    assert 'id="useAskMonthButton"' in web_html
    assert 'id="useAskEntireFeedButton"' in web_html
    assert 'id="askFeedHotspotsButton"' in web_html
    assert "coverage.question_ready_day_count" in web_script
    assert "The answer will report retained coverage gaps" in web_script
    assert "useEntireDownloadedFeed" in web_script
    assert "Include exact archive dates and times for representative events" in web_script


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
    credential_store = (
        ROOT / "BroadcastifyCli.WinUI" / "CredentialStore.cs"
    ).read_text(encoding="utf-8")
    assert "Password = saved?.Password" not in native
    assert "leave blank to reuse" in native
    assert 'startInfo.Environment["BROADCASTIFY_SECURE_PASSWORD"]' in worker
    assert 'startInfo.Environment["HUGGINGFACE_SECURE_TOKEN"]' in worker
    assert "ScopedResource" in credential_store
    assert "AppSettingsStore.TestDataRootEnvironment" in credential_store
    assert 'return $"{resource}.Test.{digest[..16]}"' in credential_store


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
    assert "RecurringCatchUp = recurringCatchUpBox.IsChecked" in window
    assert "suggestRecurringCatchUp: true" in window
    assert "result?.MissingDays.Count" in window
    assert 'Status = waitingForQuota' in window
    assert "Task<int> RecoverFeedSchedulesAsync" in worker
    assert window.index("await ApplyLaunchBehaviorAsync();") < window.index(
        "ConfigureFeedScheduleTimer();"
    )


def test_native_about_page_exposes_version_paths_and_safe_support_details() -> None:
    root = ElementTree.parse(
        ROOT / "BroadcastifyCli.WinUI" / "MainWindow.xaml"
    ).getroot()
    names = {
        element.attrib.get(XAML_NAME): element
        for element in root.iter()
        if element.attrib.get(XAML_NAME)
    }

    assert names["AboutNavigationItem"].attrib["Tag"] == "about"
    assert names["AboutVersionText"].attrib["Text"] == "Version —"
    assert names["AboutOpenLibraryButton"].attrib["Click"] == (
        "AboutOpenLibrary_Click"
    )
    assert names["AboutOpenDataButton"].attrib["Click"] == "AboutOpenData_Click"
    assert names["AboutCopySupportButton"].attrib["Click"] == (
        "AboutCopySupport_Click"
    )

    native = (
        ROOT / "BroadcastifyCli.WinUI" / "MainWindow.xaml.cs"
    ).read_text(encoding="utf-8")
    assert "AssemblyInformationalVersionAttribute" in native
    assert "AboutPage.Visibility = page == \"about\"" in native
    assert '"Activity log: {AppDiagnostics.ActivityLogPath}"' in native
    assert "Clipboard.SetContent(package);" in native
    assert "No credentials or tokens were included." in native


def test_native_library_launches_do_not_retain_winrt_folder_handles() -> None:
    native = (
        ROOT / "BroadcastifyCli.WinUI" / "MainWindow.xaml.cs"
    ).read_text(encoding="utf-8")

    assert "StorageFolder.GetFolderFromPathAsync" not in native
    assert "Launcher.LaunchFileAsync" not in native
    assert "UseShellExecute = true" in native
    assert "ReleaseMediaForArchiveMutationAsync(recreatePlayers: false)" in native
    assert "GC.WaitForPendingFinalizers();" in native
    assert "RestoreMediaPlayerInstances();" in native


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
