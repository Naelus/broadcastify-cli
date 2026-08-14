using System.Collections.Concurrent;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using Microsoft.UI;
using Microsoft.UI.Windowing;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Automation;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;
using Windows.ApplicationModel.DataTransfer;
using Windows.Graphics;
using Windows.Media.Core;
using Windows.Media.Playback;
using Windows.Security.Credentials;
using Windows.Storage;
using Windows.Storage.Pickers;
using WinRT.Interop;

namespace BroadcastifyCli.WinUI;

public sealed partial class MainWindow : Window
{
    private sealed record ResumeAllSelection(
        List<LibraryDay> Days,
        bool IncludeLocal,
        bool IncludeNetwork);

    private sealed record LibraryCatchUpRange(
        LibraryFeedCoverage Feed,
        DateTimeOffset StartDate,
        bool SaveForResume,
        bool CreateRecurringSchedule);

    private const string DefaultAnalysisModel = "ggml-org/gemma-4-12B-it-GGUF:Q4_0";
    private const int NearestAreaFeedShortcutCount = 3;
    private const int MaximumVisibleActivityLogCharacters = 24_000;
    private const int RetainedVisibleActivityLogCharacters = 16_000;
    private const string VisibleActivityLogTrimMarker =
        "[Earlier activity remains available in the on-disk activity log.]";
    private const int DwmWindowCornerPreference = 33;
    private const int DwmWindowBorderColor = 34;
    private const int DwmCornerDefault = 0;
    private const int DwmCornerDoNotRound = 1;
    private const int DwmColorDefault = -1;
    private const int DwmColorNone = -2;
    private readonly ObservableCollection<FeedSearchResult> _feeds = [];
    private readonly ObservableCollection<FeedSearchResult> _areaFeeds = [];
    private List<FeedSearchResult> _allAreaFeeds = [];
    private readonly HashSet<string> _selectedAreaFeedIds = new(StringComparer.Ordinal);
    private readonly ObservableCollection<AreaProfile> _areaProfiles = [];
    private readonly ObservableCollection<AreaStory> _areaStories = [];
    private readonly ObservableCollection<AnalysisDay> _analysisDays = [];
    private readonly ObservableCollection<IncidentRecord> _visibleIncidents = [];
    private readonly ObservableCollection<LibraryDay> _libraryDays = [];
    private readonly ObservableCollection<LibraryDay> _visibleLibraryDays = [];
    private readonly ObservableCollection<LibraryFeedCoverage> _libraryFeeds = [];
    private readonly ObservableCollection<ArchiveChatMessage> _archiveChatMessages = [];
    private readonly ObservableCollection<HardwareProfileStatus> _hardwareProfiles = [];
    private WorkerClient? _worker;
    private FeedSearchResult? _selectedFeed;
    private CancellationTokenSource? _operationCancellation;
    private CancellationTokenSource? _pipelineCancellation;
    private CancellationTokenSource? _questionCancellation;
    private bool _exclusiveBusy;
    private bool _exclusiveJobRunning;
    private bool _libraryMutationBusy;
    private readonly HashSet<string> _activePipelineFeedIds = new(StringComparer.Ordinal);
    private string _activeQuestionFeedId = "";
    private bool _syncingAnalysisFeedSelection;
    private DayReport? _currentReport;
    private MediaPlayer _incidentMediaPlayer = new();
    private MediaPlayer _areaStoryMediaPlayer = new();
    private MediaPlayer _libraryMediaPlayer = new();
    private bool _mediaPlayersAttached = true;
    private IncidentClip? _pendingIncidentClip;
    private LibraryDay? _selectedLibraryDay;
    private int _librarySelectionVersion;
    private bool _broadcastifyRateLimitObserved;
    private bool _loadingSettings = true;
    private Microsoft.UI.Dispatching.DispatcherQueueTimer? _settingsSaveTimer;
    private Microsoft.UI.Dispatching.DispatcherQueueTimer? _feedScheduleTimer;
    private Microsoft.UI.Dispatching.DispatcherQueueTimer? _systemActivityTimer;
    private bool _checkingFeedSchedule;
    private bool _refreshingSystemActivity;
    private string _lastAreaProfileName = "";
    private string _lastReviewFeedId = "";
    private string _lastReviewDate = "";
    private bool _refreshingAreaFeedSelection;
    private AreaCoverage _currentAreaCoverage = new();
    private bool _diagnosticsLoaded;
    private bool _archiveAccessConfigured;
    private bool _archiveAccessVerified;
    private bool _storageReady;
    private string _storageReadinessMessage = "Checking the archive library folder.";
    private bool _asrVerifiedThisSession;
    private string _asrVerificationMessage = "";
    private bool _diarizationVerifiedThisSession;
    private string _diarizationVerificationMessage = "";
    private bool? _analysisProviderReady;
    private bool _analysisProviderVerified;
    private string _analysisProviderReadinessMessage = "";
    private bool _analysisModelVerifiedThisSession;
    private string _analysisModelVerificationMessage = "";
    private bool _pyannotePackageInstalled;
    private bool _pyannoteAccessConfigured;
    private SavedSecret? _savedHuggingFaceToken;
    private bool _portableDiarizationRuntimeInstalled;
    private bool _portableDiarizationModelReady;
    private bool _huggingFaceTokenConfigured;
    private bool _cudaAvailable;
    private ProfileSetupAction? _profileRecoveryAction;
    private ManagedRuntimeStatus? _managedCudaRuntime;
    private int _diagnosticsLoadVersion;
    private string _configuredPythonRuntimePath = "";
    private bool _pythonRuntimeInputReady;
    private readonly bool _startupLaunch;
    private readonly bool _promptForSetup;
    private readonly string? _uiEndToEndReportPath;
    private bool _updatingStartupPreference;
    private bool _pauseScheduledJobsForSetup;
    private readonly StringBuilder _visibleActivityLog = new();
    private readonly ConcurrentQueue<JsonElement> _pendingWorkerMessages = new();
    private readonly SemaphoreSlim _analysisOperationGate = new(1, 1);
    private int _workerMessageDrainScheduled;
    private DesktopDockManager? _desktopDockManager;
    private DesktopDockSide _desktopDockSide;
    private double _desktopDockWidth = DesktopDockManager.RecommendedWidthDips;
    private string _desktopDockMonitor = "";
    private bool _desktopDockRestoreAttempted;
    private bool _desktopWindowFrameDocked;
    private bool _desktopWindowActive = true;
    private bool _suspendDesktopPlacementTracking;
    private int _desktopWindowCornerPreference = DwmCornerDefault;
    private int _desktopWindowBorderColor = DwmColorDefault;
    private bool? _compactLayoutApplied;
    private bool? _shortCompactLayoutApplied;

    public MainWindow(
        bool startupLaunch = false,
        bool promptForSetup = false,
        string? uiEndToEndReportPath = null)
    {
        _startupLaunch = startupLaunch;
        _promptForSetup = promptForSetup;
        _uiEndToEndReportPath = uiEndToEndReportPath;
        // Keep one startup snapshot. Processing-tab controls can be realized
        // after the window constructor, so the worker must not depend on a
        // second settings read or the current visual value of that tab.
        var startupSettings = AppSettingsStore.Load();
        InitializeComponent();
        WindowRoot.SizeChanged += WindowRoot_SizeChanged;
        if (AppWindow.Presenter is OverlappedPresenter initialPresenter)
        {
            initialPresenter.SetBorderAndTitleBar(
                hasBorder: true,
                hasTitleBar: false);
        }
        ExtendsContentIntoTitleBar = true;
        SetTitleBar(TitleBarDragRegion);
        Activated += MainWindow_Activated;
        var windowHandle = WindowNative.GetWindowHandle(this);
        var dpiScale = Math.Max(1.0, GetDpiForWindow(windowHandle) / 96.0);
        AppWindow.Resize(new SizeInt32((int)(1240 * dpiScale), (int)(900 * dpiScale)));
        _desktopDockSide = ParseDesktopDockSide(startupSettings.DesktopDockSide);
        _desktopDockWidth = Math.Max(
            DesktopDockManager.MinimumWidthDips,
            startupSettings.DesktopDockWidth);
        _desktopDockMonitor = startupSettings.DesktopDockMonitor ?? "";
        try
        {
            _desktopDockManager = new DesktopDockManager(
                windowHandle,
                AppWindow,
                DispatcherQueue,
                SavedFloatingBounds(startupSettings),
                startupSettings.DesktopWindowMaximized);
            _desktopDockManager.InteractiveResizeCompleted +=
                DesktopDockManager_InteractiveResizeCompleted;
            _desktopDockManager.RestoreFloatingWindowForStartup();
            AppWindow.Changed += AppWindow_Changed;
        }
        catch (Exception exception)
        {
            AppDiagnostics.AppendCrash(exception, "Desktop docking setup");
            _desktopDockSide = DesktopDockSide.None;
            DockButton.IsEnabled = false;
            ToolTipService.SetToolTip(
                DockButton,
                "Windows desktop docking is unavailable in this session.");
        }
        WindowRoot.Loaded += WindowRoot_Loaded;

        FeedResults.ItemsSource = _feeds;
        AreaFeedResults.ItemsSource = _areaFeeds;
        AreaProfileCombo.ItemsSource = _areaProfiles;
        AreaStoryList.ItemsSource = _areaStories;
        AnalysisDaysList.ItemsSource = _analysisDays;
        AnalysisFeedCombo.ItemsSource = _libraryFeeds;
        ArchiveChatList.ItemsSource = _archiveChatMessages;
        IncidentList.ItemsSource = _visibleIncidents;
        LibraryList.ItemsSource = _visibleLibraryDays;
        LibraryFeedCoverageList.ItemsSource = _libraryFeeds;
        HardwareProfileList.ItemsSource = _hardwareProfiles;
        IncidentPlayer.SetMediaPlayer(_incidentMediaPlayer);
        _incidentMediaPlayer.MediaOpened += IncidentMediaPlayer_MediaOpened;
        AreaStoryPlayer.SetMediaPlayer(_areaStoryMediaPlayer);
        _areaStoryMediaPlayer.MediaOpened += AreaStoryMediaPlayer_MediaOpened;
        LibraryAudioPlayer.SetMediaPlayer(_libraryMediaPlayer);
        CombineToggle.IsOn = true;
        CombineToggle.IsEnabled = false;
        KeepOriginalsToggle.IsEnabled = true;
        if (_uiEndToEndReportPath is null)
        {
            RefreshWindowsStartupUi();
        }
        else
        {
            StartWithWindowsToggle.IsOn = false;
            StartWithWindowsToggle.IsEnabled = false;
            WindowsStartupStatusText.Text =
                "Disabled inside the isolated UI end-to-end probe.";
        }
        LoadUserSettings(startupSettings);
        if (_uiEndToEndReportPath is null)
        {
            WireSettingsAutoSave();
        }
        var today = DateTimeOffset.Now;
        StartDatePicker.Date = today;
        EndDatePicker.Date = today;
        QuestionStartDatePicker.Date = today;
        QuestionEndDatePicker.Date = today;
        QuestionMonthPicker.Date = today;
        WeekEndingPicker.Date = today;
        AreaStartDatePicker.Date = today;
        AreaEndDatePicker.Date = today;
        TranscribeCheckBox.Checked += Transcription_Changed;
        TranscribeCheckBox.Unchecked += Transcription_Changed;
        DiarizeCheckBox.Checked += Diarization_Changed;
        DiarizeCheckBox.Unchecked += Diarization_Changed;
        Closed += MainWindow_Closed;
        Transcription_Changed(TranscribeCheckBox, new RoutedEventArgs());
        Diarization_Changed(DiarizeCheckBox, new RoutedEventArgs());

        if (_uiEndToEndReportPath is not null)
        {
            RefreshAboutPage();
            WindowRoot.Loaded += StartUiEndToEndProbe;
            return;
        }

        try
        {
            _worker = new WorkerClient(startupSettings.PythonRuntimePath);
            _worker.SetLibraryDirectory(OutputFolderBox.Text);
            PersistUserSettings(logFailure: true);
            AppendLog($"Worker: {_worker.PythonDisplayName}");
            PythonRuntimeStatusText.Text = _worker.PythonRuntimeWarning
                ?? $"Active worker: {_worker.PythonDisplayName}";
            if (_worker.PythonRuntimeWarning is not null)
            {
                AppendLog(_worker.PythonRuntimeWarning);
            }
            AppendLog(_worker.IsBundledRuntime
                ? $"Installed runtime: {_worker.RepositoryRoot}"
                : $"Repository: {_worker.RepositoryRoot}");
            AppendLog($"Working data: {_worker.WorkingDirectory}");
            AppendLog(_worker.HasBundledWindowsMlHelper
                ? "Windows ML helper: bundled runtime"
                : "Windows ML helper: repository/runtime discovery");
            _ = ConfigureLanSharingAsync();
            _ = InitializeAsync();
        }
        catch (Exception exception)
        {
            AppendLog(exception.Message);
            StatusText.Text = "Python setup required";
            SearchButton.IsEnabled = false;
            StartButton.IsEnabled = false;
            RefreshLibraryButton.IsEnabled = false;
        }
        RefreshAboutPage();
    }

    private async Task ConfigureLanSharingAsync()
    {
        if (_loadingSettings || _worker is null || LanShareInfoBar is null)
        {
            return;
        }
        try
        {
            var status = await _worker.ConfigureLanNodeAsync(
                LanShareToggle.IsOn,
                string.IsNullOrWhiteSpace(OutputFolderBox.Text)
                    ? "archives"
                    : OutputFolderBox.Text.Trim(),
                RequiredInteger(LanSharePortBox.Value, 8766));
            LanShareInfoBar.Severity = LanShareToggle.IsOn
                ? InfoBarSeverity.Success
                : InfoBarSeverity.Informational;
            LanShareInfoBar.Title = LanShareToggle.IsOn
                ? "LAN peer and queue producer are active"
                : "This Windows client is not seeding";
            LanShareInfoBar.Message = status;
        }
        catch (Exception exception)
        {
            LanShareInfoBar.Severity = InfoBarSeverity.Warning;
            LanShareInfoBar.Title = "LAN archive sharing needs attention";
            LanShareInfoBar.Message = exception.Message;
            AppendLog(exception.Message);
        }
    }

    private async Task InitializeAsync()
    {
        RefreshStorageReadiness();
        await RefreshArchiveQuotaStatusAsync();
        await RefreshManagedRuntimeStatusAsync();
        await TryAutoSignInAsync();
        await LoadDiagnosticsAndDaysAsync();
        await RefreshLibraryAsync();
        if (_worker is not null)
        {
            var recovered = await _worker.RecoverFeedSchedulesAsync(
                CancellationToken.None);
            if (recovered > 0)
            {
                AppendLog(
                    $"Requeued {recovered} scheduled feed "
                    + $"run{(recovered == 1 ? "" : "s")}; interrupted work and "
                    + "quota-paused local processing will resume after the safety delay.");
            }
        }
        await RefreshFeedScheduleStatusAsync();
        await ApplyLaunchBehaviorAsync();
        ConfigureFeedScheduleTimer();
        ConfigureSystemActivityTimer();
    }

    private void RefreshWindowsStartupUi()
    {
        if (StartWithWindowsToggle is null || WindowsStartupStatusText is null)
        {
            return;
        }
        _updatingStartupPreference = true;
        try
        {
            var enabled = WindowsStartupManager.IsEnabled();
            StartWithWindowsToggle.IsOn = enabled;
            WindowsStartupStatusText.Text = enabled
                ? "Enabled for this Windows account. Login launches are minimized after setup checks; interrupted schedules resume from retained work."
                : "Off. Scheduled jobs run only while you open Broadcastify Desktop yourself.";
        }
        catch (Exception exception)
        {
            StartWithWindowsToggle.IsOn = false;
            StartWithWindowsToggle.IsEnabled = false;
            WindowsStartupStatusText.Text =
                $"Windows startup status is unavailable: {exception.Message}";
        }
        finally
        {
            _updatingStartupPreference = false;
        }
    }

    private void StartWithWindows_Toggled(object sender, RoutedEventArgs e)
    {
        if (_updatingStartupPreference)
        {
            return;
        }
        try
        {
            WindowsStartupManager.SetEnabled(StartWithWindowsToggle.IsOn);
            WindowsStartupStatusText.Text = StartWithWindowsToggle.IsOn
                ? "Enabled for this Windows account. Broadcastify Desktop will start minimized after sign-in and keep scheduled feeds current."
                : "Off. Existing schedules remain saved and will run the next time the app is open.";
            AppendLog(
                StartWithWindowsToggle.IsOn
                    ? "Start with Windows enabled."
                    : "Start with Windows disabled.");
        }
        catch (Exception exception)
        {
            _updatingStartupPreference = true;
            StartWithWindowsToggle.IsOn = !StartWithWindowsToggle.IsOn;
            _updatingStartupPreference = false;
            WindowsStartupStatusText.Text =
                $"Windows could not save the startup preference: {exception.Message}";
        }
    }

    private async Task ApplyLaunchBehaviorAsync()
    {
        var shouldPrompt = _promptForSetup || _startupLaunch;
        var promptShown = false;
        if (shouldPrompt)
        {
            promptShown = await PromptForMissingAccountSetupAsync();
        }
        _pauseScheduledJobsForSetup = promptShown;

        if (!_startupLaunch)
        {
            return;
        }
        if (promptShown)
        {
            AppendLog(
                "Started with Windows; setup needs attention before unattended schedules can run reliably.");
            return;
        }

        if (_desktopDockManager?.IsDocked == true)
        {
            AppendLog(
                "Started with Windows and kept the explicitly pinned status window visible; saved schedules were checked.");
            return;
        }

        AppendLog(
            "Started with Windows; saved schedules were checked and the window was minimized.");
        DispatcherQueue.TryEnqueue(() =>
        {
            if (AppWindow.Presenter is OverlappedPresenter presenter)
            {
                presenter.Minimize();
            }
        });
    }

    private async Task<bool> PromptForMissingAccountSetupAsync()
    {
        var (missingBroadcastify, missingHuggingFace) =
            MissingUnattendedSetup();
        if (!missingBroadcastify && !missingHuggingFace)
        {
            return false;
        }

        var missing = new List<string>();
        if (missingBroadcastify)
        {
            missing.Add("a saved Broadcastify premium website login or session");
        }
        if (missingHuggingFace)
        {
            missing.Add("a Hugging Face read token for Community-1 speaker labels");
        }
        var message =
            "Automatic startup is ready, but unattended processing still needs "
            + string.Join(" and ", missing)
            + ". Open Credentials now to finish setup. Scheduled jobs will wait, and no archive request will be made by this prompt.";

        if (((FrameworkElement)Content).XamlRoot is null)
        {
            OpenCredentialSetup();
            return true;
        }
        var dialog = new ContentDialog
        {
            XamlRoot = ((FrameworkElement)Content).XamlRoot,
            Title = "Finish unattended setup",
            Content = CreateDialogTextContent(message),
            PrimaryButtonText = "Open credentials",
            CloseButtonText = "Later",
            DefaultButton = ContentDialogButton.Primary,
        };
        if (await dialog.ShowAsync() == ContentDialogResult.Primary)
        {
            OpenCredentialSetup();
        }
        return true;
    }

    private (bool Broadcastify, bool HuggingFace) MissingUnattendedSetup()
    {
        var missingBroadcastify = !_archiveAccessConfigured;
        var tokenAvailable = _huggingFaceTokenConfigured
            || _pyannoteAccessConfigured
            || !string.IsNullOrWhiteSpace(CurrentHuggingFaceToken())
            || SelectedHardwareProfile()?.DiarizationReady == true;
        var missingHuggingFace =
            SelectedComboValue(DiarizationEngineComboBox, "community-1")
                == "community-1"
            && !tokenAvailable;
        return (missingBroadcastify, missingHuggingFace);
    }

    private void OpenCredentialSetup()
    {
        RootNavigation.SelectedItem = CredentialsNavigationItem;
        ShowPage("settings");
        SettingsTabView.SelectedItem = AccountSettingsTab;
        RefreshHuggingFaceCredentialUi();
    }

    private void RootNavigation_SelectionChanged(
        NavigationView sender,
        NavigationViewSelectionChangedEventArgs args)
    {
        var page = args.IsSettingsSelected
            ? "settings"
            : (args.SelectedItemContainer as NavigationViewItem)?.Tag?.ToString() ?? "library";
        if (page == "credentials")
        {
            ShowPage("settings");
            SettingsTabView.SelectedItem = AccountSettingsTab;
            RefreshHuggingFaceCredentialUi();
            return;
        }
        ShowPage(page);
    }

    private void ShowPage(string page)
    {
        LibraryPage.Visibility = page == "library" ? Visibility.Visible : Visibility.Collapsed;
        SystemPage.Visibility = page == "system" ? Visibility.Visible : Visibility.Collapsed;
        ArchivePage.Visibility = page == "archive" ? Visibility.Visible : Visibility.Collapsed;
        ReviewPage.Visibility = page == "review" ? Visibility.Visible : Visibility.Collapsed;
        AreaPage.Visibility = page == "area" ? Visibility.Visible : Visibility.Collapsed;
        SettingsPage.Visibility = page == "settings" ? Visibility.Visible : Visibility.Collapsed;
        AboutPage.Visibility = page == "about" ? Visibility.Visible : Visibility.Collapsed;
        if (page == "archive")
        {
            _ = RefreshArchiveQuotaStatusAsync();
            _ = RefreshFeedScheduleStatusAsync();
        }
        if (page == "system")
        {
            _ = RefreshSystemActivityAsync();
        }
        if (page == "settings")
        {
            RefreshHuggingFaceCredentialUi();
        }
        if (page == "about")
        {
            RefreshAboutPage();
        }
    }

    private static string InstalledVersion()
    {
        var assembly = Assembly.GetExecutingAssembly();
        var informational = assembly
            .GetCustomAttribute<AssemblyInformationalVersionAttribute>()
            ?.InformationalVersion
            .Split('+', 2)[0];
        if (!string.IsNullOrWhiteSpace(informational))
        {
            return informational;
        }
        var version = assembly.GetName().Version;
        return version is null ? "Unknown" : version.ToString(3);
    }

    private static string InstalledFileVersion() =>
        Assembly.GetExecutingAssembly()
            .GetCustomAttribute<AssemblyFileVersionAttribute>()
            ?.Version
        ?? "Unknown";

    private void RefreshAboutPage()
    {
        if (AboutVersionText is null)
        {
            return;
        }
        var distribution = _worker?.IsBundledRuntime == true
            ? "Installed desktop runtime"
            : "Source/development runtime";
        AboutVersionText.Text = $"Version {InstalledVersion()}";
        AboutBuildText.Text =
            $"File {InstalledFileVersion()} · {RuntimeInformation.ProcessArchitecture} · {distribution}";
        AboutRuntimeText.Text =
            $"{RuntimeInformation.OSDescription} · .NET {Environment.Version}";
        AboutLibraryPathText.Text = PersistedOutputDirectory();
        AboutDataPathText.Text = AppSettingsStore.LocalDataDirectory;
    }

    private string BuildSupportDetails()
    {
        RefreshAboutPage();
        return string.Join(
            Environment.NewLine,
            "Broadcastify Desktop support details",
            $"Version: {InstalledVersion()}",
            $"File version: {InstalledFileVersion()}",
            $"Architecture: {RuntimeInformation.ProcessArchitecture}",
            $"Operating system: {RuntimeInformation.OSDescription}",
            $".NET runtime: {Environment.Version}",
            $"Distribution: {(_worker?.IsBundledRuntime == true ? "installed" : "source/development")}",
            $"Library: {PersistedOutputDirectory()}",
            $"App data: {AppSettingsStore.LocalDataDirectory}",
            $"Activity log: {AppDiagnostics.ActivityLogPath}");
    }

    private async Task OpenAboutFolderAsync(string title, string path)
    {
        try
        {
            var fullPath = Path.GetFullPath(path);
            Directory.CreateDirectory(fullPath);
            using var launched = Process.Start(new ProcessStartInfo
            {
                FileName = fullPath,
                UseShellExecute = true,
            });
            if (launched is null)
            {
                throw new InvalidOperationException("Windows could not open the folder.");
            }
            AboutActionStatusText.Text = $"Opened {fullPath}";
        }
        catch (Exception exception) when (
            exception is ArgumentException
                or IOException
                or NotSupportedException
                or UnauthorizedAccessException
                or InvalidOperationException)
        {
            AboutActionStatusText.Text = $"Could not open the folder: {exception.Message}";
            await ShowMessageAsync(title, exception.Message);
        }
    }

    private async void AboutOpenLibrary_Click(object sender, RoutedEventArgs e) =>
        await OpenAboutFolderAsync("Could not open the library", PersistedOutputDirectory());

    private async void AboutOpenData_Click(object sender, RoutedEventArgs e) =>
        await OpenAboutFolderAsync("Could not open app data", AppSettingsStore.LocalDataDirectory);

    private void AboutCopySupport_Click(object sender, RoutedEventArgs e)
    {
        try
        {
            var package = new DataPackage
            {
                RequestedOperation = DataPackageOperation.Copy,
            };
            package.SetText(BuildSupportDetails());
            Clipboard.SetContent(package);
            Clipboard.Flush();
            AboutActionStatusText.Text =
                "Copied version, runtime, and local paths. No credentials or tokens were included.";
        }
        catch (Exception exception)
        {
            AboutActionStatusText.Text = $"Could not copy support details: {exception.Message}";
        }
    }

    private async Task RefreshArchiveQuotaStatusAsync()
    {
        if (_worker is null || ArchiveQuotaInfoBar is null)
        {
            return;
        }
        try
        {
            IReadOnlyList<string> profileIds = _worker.AuthorizedAccountPoolEnabled
                ? _worker.AvailableAccountProfileIds()
                : new[] { "default" };
            var statuses = new List<ArchiveQuotaStatus>();
            foreach (var profileId in profileIds)
            {
                var profileStatus = await _worker.GetArchiveQuotaStatusAsync(
                    CancellationToken.None,
                    profileId);
                if (profileStatus is not null)
                {
                    statuses.Add(profileStatus);
                }
            }
            if (statuses.Count == 0)
            {
                throw new InvalidOperationException("The quota ledger returned no status.");
            }
            if (statuses.Count > 1)
            {
                var totalRemaining = statuses.Sum(value => value.Remaining);
                var totalAutomated = statuses.Sum(value => value.AutomatedLimit);
                var totalUsed = statuses.Sum(value => value.Used);
                var totalReserve = statuses.Sum(value => value.UserReserve);
                var nextPoolSlot = statuses
                    .Select(value => DateTimeOffset.TryParse(
                        value.NextRequestAt,
                        out var parsed)
                        ? parsed
                        : (DateTimeOffset?)null)
                    .Where(value => value is not null)
                    .OrderBy(value => value)
                    .FirstOrDefault();
                ArchiveQuotaInfoBar.Severity = statuses.Any(value => value.Available)
                    ? InfoBarSeverity.Success
                    : InfoBarSeverity.Warning;
                ArchiveQuotaInfoBar.Title = statuses.Any(value => value.Available)
                    ? $"{totalRemaining} of {totalAutomated} authorized pooled archive requests available"
                    : "All authorized account profiles are waiting";
                ArchiveQuotaInfoBar.Message =
                    $"{statuses.Count} profiles × each account's standard limit · {totalUsed} used in their independent rolling 24-hour windows · {totalReserve} total held for manual use. "
                    + "Downloads remain sequential and spaced."
                    + (nextPoolSlot is not null
                        ? $" Next pool slot: {nextPoolSlot.Value.ToLocalTime():g}."
                        : "");
                return;
            }
            var status = statuses[0];
            var instance = status.InstanceId.Length > 8
                ? status.InstanceId[..8]
                : status.InstanceId;
            ArchiveQuotaInfoBar.Severity = status.Available
                ? InfoBarSeverity.Success
                : InfoBarSeverity.Warning;
            ArchiveQuotaInfoBar.Title = status.Available
                ? $"{status.Remaining} of {status.AutomatedLimit} automated archive requests available"
                : "Archive requests are paused for this installation";
            var next = DateTimeOffset.TryParse(status.NextRequestAt, out var nextRequest)
                ? $" Next safe request: {nextRequest.ToLocalTime():g}."
                : "";
            ArchiveQuotaInfoBar.Message =
                $"Account {status.AccountProfileId} · rolling 24 hours · {status.Used} used · {status.UserReserve} held for manual use · instance {instance}."
                + next
                + (status.Blocked && !string.IsNullOrWhiteSpace(status.BlockedReason)
                    ? $" {status.BlockedReason}"
                    : " Cached audio and local processing do not use this budget.");
        }
        catch (Exception exception)
        {
            ArchiveQuotaInfoBar.Severity = InfoBarSeverity.Warning;
            ArchiveQuotaInfoBar.Title = "Archive budget status unavailable";
            ArchiveQuotaInfoBar.Message = exception.Message;
        }
    }

    private void ConfigureSystemActivityTimer()
    {
        if (_systemActivityTimer is not null)
        {
            return;
        }
        _systemActivityTimer = DispatcherQueue.CreateTimer();
        _systemActivityTimer.Interval = TimeSpan.FromSeconds(10);
        _systemActivityTimer.IsRepeating = true;
        _systemActivityTimer.Tick += async (_, _) =>
        {
            if (SystemPage.Visibility == Visibility.Visible)
            {
                await RefreshSystemActivityAsync();
            }
        };
        _systemActivityTimer.Start();
    }

    private async void RefreshSystemActivity_Click(object sender, RoutedEventArgs e) =>
        await RefreshSystemActivityAsync();

    private async Task RefreshSystemActivityAsync()
    {
        if (_worker is null
            || SystemActivityInfoBar is null
            || _refreshingSystemActivity)
        {
            return;
        }
        _refreshingSystemActivity = true;
        RefreshSystemActivityButton.IsEnabled = false;
        try
        {
            using var cancellation = new CancellationTokenSource(
                TimeSpan.FromSeconds(8));
            var status = await _worker.GetCoordinatedActivityStatusAsync(
                cancellation.Token);
            if (status is null)
            {
                throw new InvalidOperationException(
                    "The coordinated status worker returned no result.");
            }

            var quotaStatuses = new List<ArchiveQuotaStatus>();
            IReadOnlyList<string> profileIds = _worker.AuthorizedAccountPoolEnabled
                ? _worker.AvailableAccountProfileIds()
                : new[] { "default" };
            foreach (var profileId in profileIds)
            {
                var quota = await _worker.GetArchiveQuotaStatusAsync(
                    cancellation.Token,
                    profileId);
                if (quota is not null)
                {
                    quotaStatuses.Add(quota);
                }
            }
            var localSchedules = await _worker.ListFeedSchedulesAsync(
                cancellation.Token);

            var acquisition = status.Activity.Acquisition;
            if (acquisition is not null)
            {
                var node = CoordinatedNodeLabel(acquisition);
                var profile = AccountProfileFromQuotaScope(
                    acquisition.QuotaScope);
                SystemAcquisitionText.Text =
                    $"{node} is downloading feed {acquisition.FeedId} for {acquisition.ArchiveDate} using account profile {profile}.";
                var matchingQuota = quotaStatuses.FirstOrDefault(value =>
                    value.AccountProfileId.Equals(
                        profile,
                        StringComparison.OrdinalIgnoreCase));
                SystemAcquisitionDetailText.Text = matchingQuota is null
                    ? "The renewable global lease is active. Other nodes defer the same website stream and reuse its retained blocks."
                    : $"{matchingQuota.Used}/{matchingQuota.AutomatedLimit} automated requests used in this account's rolling window; {matchingQuota.Remaining} remain. Other nodes defer the same website stream and reuse its retained blocks.";
            }
            else
            {
                SystemAcquisitionText.Text =
                    "No coordinated website download is active at this moment.";
                SystemAcquisitionDetailText.Text =
                    "Saved schedules and retained missing days remain listed below. Cached and LAN-reused blocks do not consume an account allowance.";
            }

            var processing = status.Activity.Processing;
            SystemProcessingText.Text = processing.Count == 0
                ? "No coordinated model/day lease is active."
                : string.Join(
                    Environment.NewLine,
                    processing.Select(value =>
                        $"• {CoordinatedNodeLabel(value)} · feed {value.FeedId} · {value.ArchiveDate}"));

            SystemQuotaText.Text = quotaStatuses.Count == 0
                ? "No account allowance status was returned."
                : string.Join(
                    Environment.NewLine,
                    quotaStatuses.Select(value =>
                        $"• {value.AccountProfileId}: {value.Used}/{value.AutomatedLimit} used · {value.Remaining} remaining"
                        + (value.Blocked ? " · waiting for rolling release" : " · available")));

            var coordinatorSchedules = status.Scheduler.Schedules;
            var scheduleLines = new List<string>();
            var activeSchedule = status.Scheduler.Active;
            if (activeSchedule is not null)
            {
                var stage = string.IsNullOrWhiteSpace(activeSchedule.Stage)
                    ? activeSchedule.Phase
                    : activeSchedule.Stage;
                var archiveDate = string.IsNullOrWhiteSpace(activeSchedule.ArchiveDate)
                    ? ""
                    : $" · {activeSchedule.ArchiveDate}";
                var progress = activeSchedule.Total > 0
                    ? $" · {activeSchedule.Current}/{activeSchedule.Total}"
                    : "";
                scheduleLines.Add(
                    $"• Coordinator active · {activeSchedule.FeedName} ({activeSchedule.FeedId})"
                    + $" · {stage}{archiveDate}{progress}"
                    + $" · account {activeSchedule.AccountProfileId}");
            }
            scheduleLines.AddRange(localSchedules
                .Where(value => value.Enabled)
                .Select(value =>
                    $"• Windows · {value.FeedName} ({value.FeedId}) · {value.State}"));
            scheduleLines.AddRange(coordinatorSchedules
                .Where(value => value.Enabled)
                .Select(value =>
                    $"• Coordinator · {value.FeedName} ({value.FeedId}) · {value.State}"
                    + (string.IsNullOrWhiteSpace(value.Message)
                        ? ""
                        : $" · {value.Message}")));
            SystemSchedulesText.Text = scheduleLines.Count == 0
                ? "No enabled Windows or coordinator schedules were reported."
                : string.Join(Environment.NewLine, scheduleLines);

            var activeLabel = acquisition is not null
                ? $"Feed {acquisition.FeedId} acquisition is active"
                : processing.Count > 0
                    ? $"{processing.Count} model/day claim{(processing.Count == 1 ? " is" : "s are")} active"
                    : activeSchedule is not null
                        ? $"Feed {activeSchedule.FeedId} {activeSchedule.Phase} is active"
                    : "Coordinated workers are between active leases";
            SystemActivityInfoBar.Severity = status.Connected
                ? acquisition is not null
                    ? InfoBarSeverity.Success
                    : InfoBarSeverity.Informational
                : InfoBarSeverity.Warning;
            SystemActivityInfoBar.Title = status.Connected
                ? activeLabel
                : "The coordinated activity surface is unavailable";
            SystemActivityInfoBar.Message = status.Connected
                ? $"Connected to {status.CoordinatorUrl}. This view refreshes every 10 seconds while open."
                : status.Error;
        }
        catch (OperationCanceledException)
        {
            SystemActivityInfoBar.Severity = InfoBarSeverity.Warning;
            SystemActivityInfoBar.Title = "Coordinated activity check timed out";
            SystemActivityInfoBar.Message =
                "The existing workers were not stopped; use Refresh to check again.";
        }
        catch (Exception exception)
        {
            SystemActivityInfoBar.Severity = InfoBarSeverity.Warning;
            SystemActivityInfoBar.Title = "Coordinated activity status unavailable";
            SystemActivityInfoBar.Message = exception.Message;
        }
        finally
        {
            _refreshingSystemActivity = false;
            RefreshSystemActivityButton.IsEnabled = true;
        }
    }

    private static string AccountProfileFromQuotaScope(string quotaScope)
    {
        var separator = quotaScope.LastIndexOf('.');
        return separator >= 0 && separator + 1 < quotaScope.Length
            ? quotaScope[(separator + 1)..]
            : string.IsNullOrWhiteSpace(quotaScope)
                ? "default"
                : quotaScope;
    }

    private static string CoordinatedNodeLabel(CoordinatedWorkLease activity)
    {
        if (Uri.TryCreate(activity.ProducerUrl, UriKind.Absolute, out var uri)
            && !string.IsNullOrWhiteSpace(uri.Host))
        {
            return uri.Host;
        }
        return activity.OwnerNodeId.Length > 10
            ? $"node {activity.OwnerNodeId[..10]}"
            : $"node {activity.OwnerNodeId}";
    }

    private void NavigateTo(NavigationViewItem item)
    {
        RootNavigation.SelectedItem = item;
        ShowPage(item.Tag?.ToString() ?? "library");
    }

    private void HardwareProfile_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (_loadingSettings
            || HardwareProfileComboBox?.SelectedItem is not ComboBoxItem item)
        {
            return;
        }
        var profile = item.Tag?.ToString() ?? "auto";
        var selectedModel = SelectedComboValue(ModelComboBox, "turbo");
        var resetUnsupportedPortableModel = profile == "vulkan"
            && string.Equals(
                selectedModel,
                "distil-large-v3",
                StringComparison.OrdinalIgnoreCase);
        var useWindowsMlStarter = profile == "windowsml"
            && !string.Equals(selectedModel, "base", StringComparison.OrdinalIgnoreCase);
        var useQwenStarter = profile == "qwen"
            && !string.Equals(
                selectedModel,
                "qwen3-asr-0.6b-int8",
                StringComparison.OrdinalIgnoreCase);
        var resetQwenModel = profile != "qwen"
            && string.Equals(
                selectedModel,
                "qwen3-asr-0.6b-int8",
                StringComparison.OrdinalIgnoreCase);
        switch (profile)
        {
            case "cuda":
                SelectComboTag(AsrEngineComboBox, "faster-whisper");
                SelectComboTag(DeviceComboBox, "cuda");
                SelectComboTag(DiarizationEngineComboBox, "community-1");
                SelectComboTag(DiarizationDeviceComboBox, "cuda");
                break;
            case "vulkan":
                SelectComboTag(AsrEngineComboBox, "whisper.cpp");
                SelectComboTag(DeviceComboBox, "vulkan");
                SelectComboTag(DiarizationEngineComboBox, "sherpa-onnx");
                SelectComboTag(DiarizationDeviceComboBox, "cpu");
                break;
            case "openvino":
                SelectComboTag(AsrEngineComboBox, "openvino");
                SelectComboTag(DeviceComboBox, "openvino-auto");
                SelectComboTag(DiarizationEngineComboBox, "sherpa-onnx");
                SelectComboTag(DiarizationDeviceComboBox, "cpu");
                break;
            case "windowsml":
                SelectComboTag(AsrEngineComboBox, "windows-ml");
                SelectComboTag(DeviceComboBox, "auto");
                SelectComboTag(DiarizationEngineComboBox, "sherpa-onnx");
                SelectComboTag(DiarizationDeviceComboBox, "cpu");
                break;
            case "qwen":
                SelectComboTag(AsrEngineComboBox, "qwen3-asr");
                SelectComboTag(DeviceComboBox, "cpu");
                SelectComboTag(DiarizationEngineComboBox, "sherpa-onnx");
                SelectComboTag(DiarizationDeviceComboBox, "cpu");
                break;
            case "cpu":
                SelectComboTag(AsrEngineComboBox, "faster-whisper");
                SelectComboTag(DeviceComboBox, "cpu");
                SelectComboTag(DiarizationEngineComboBox, "community-1");
                SelectComboTag(DiarizationDeviceComboBox, "cpu");
                break;
            default:
                SelectComboTag(AsrEngineComboBox, "auto");
                SelectComboTag(DeviceComboBox, "auto");
                SelectComboTag(DiarizationEngineComboBox, "community-1");
                SelectComboTag(DiarizationDeviceComboBox, "auto");
                break;
        }
        if (resetUnsupportedPortableModel)
        {
            SelectComboValue(ModelComboBox, "turbo");
            StatusText.Text =
                "Vulkan defaults applied. Whisper was reset to turbo because whisper.cpp has no managed distil-large-v3 mapping.";
        }
        else if (useWindowsMlStarter)
        {
            SelectComboValue(ModelComboBox, "base");
            StatusText.Text =
                "Windows ML defaults applied with the radio-tested Base CPU starter. Tiny is faster, but it missed important words in retained scanner audio.";
        }
        else if (useQwenStarter)
        {
            SelectComboValue(ModelComboBox, "qwen3-asr-0.6b-int8");
            StatusText.Text =
                "Qwen3-ASR fast CPU preview selected. It retains speech-region timestamps but remains optional while longer quality gates continue.";
        }
        else if (resetQwenModel)
        {
            SelectComboValue(ModelComboBox, profile == "windowsml" ? "base" : "turbo");
        }
        SelectComboTag(AnalysisDeviceComboBox, profile == "cpu" ? "cpu" : "auto");
        _asrVerifiedThisSession = false;
        _diarizationVerifiedThisSession = false;
        UpdateAsrModelPreparationUi();
        ApplySelectedHardwareProfileDescription();
        UpdateSetupSummary();
        if (_worker is not null && _diagnosticsLoaded)
        {
            StatusText.Text = "Refreshing the selected profile’s engine checks…";
            _ = LoadDiagnosticsAndDaysAsync();
        }
    }

    private void LoadUserSettings(DesktopSettings settings)
    {
        _loadingSettings = true;
        _configuredPythonRuntimePath = settings.PythonRuntimePath.Trim();
        PythonRuntimePathBox.Text = _configuredPythonRuntimePath;
        SelectComboValue(HardwareProfileComboBox, settings.HardwareProfile);
        SelectComboValue(ModelComboBox, settings.WhisperModel);
        SelectComboValue(AsrEngineComboBox, settings.AsrEngine);
        SelectComboValue(DeviceComboBox, settings.TranscriptionDevice);
        SelectComboValue(DiarizationEngineComboBox, settings.DiarizationEngine);
        SelectComboValue(DiarizationDeviceComboBox, settings.DiarizationDevice);
        EnsureDiarizationSelectionCompatibility();
        AsrModelPathBox.Text = settings.AsrModelPath;
        var configuredOutputDirectory = string.IsNullOrWhiteSpace(settings.OutputDirectory)
            ? "archives"
            : settings.OutputDirectory;
        var migratedOutputDirectory = false;
        if (WorkerClient.BundledRuntimeAvailable
            && !Path.IsPathRooted(configuredOutputDirectory))
        {
            var currentCandidate = Path.GetFullPath(
                configuredOutputDirectory,
                AppSettingsStore.LocalDataDirectory);
            var previousLibrary = AppDiagnostics.FindPreviousLibraryDirectory(
                configuredOutputDirectory,
                currentCandidate);
            if (!string.IsNullOrWhiteSpace(previousLibrary))
            {
                configuredOutputDirectory = previousLibrary;
                migratedOutputDirectory = true;
            }
        }
        OutputFolderBox.Text = configuredOutputDirectory;
        GpuIndexBox.Value = Math.Clamp(settings.GpuIndex, 0, 15);
        BatchSizeBox.Value = Math.Clamp(settings.BatchSize, 1, 64);
        MinimumSpeakersBox.Value = Math.Clamp(settings.MinimumSpeakers, 0, 64);
        MaximumSpeakersBox.Value = Math.Clamp(settings.MaximumSpeakers, 0, 64);
        DownloadJobsBox.Value = Math.Clamp(settings.DownloadJobs, 1, 32);
        CombineToggle.IsOn = settings.Combine;
        KeepOriginalsToggle.IsOn = settings.KeepOriginals;
        TranscribeCheckBox.IsChecked = settings.Transcribe;
        DiarizeCheckBox.IsChecked = settings.Diarize;
        AnalyzeAfterJobCheckBox.IsChecked = settings.AnalyzeAfterJob;
        SelectComboValue(AnalysisProviderComboBox, settings.AnalysisProvider);
        AnalysisModelBox.Text = string.IsNullOrWhiteSpace(settings.AnalysisModel)
            ? settings.AnalysisProvider switch
            {
                "local" => DefaultAnalysisModel,
                "openai-responses" => "gpt-5.6-luna",
                _ => "",
            }
            : settings.AnalysisModel;
        SelectComboValue(AnalysisDeviceComboBox, settings.AnalysisDevice);
        AnalysisEndpointBox.Text = settings.AnalysisEndpoint;
        AnalysisApiKeyEnvironmentBox.Text = string.IsNullOrWhiteSpace(settings.AnalysisApiKeyEnvironment)
            ? "OPENAI_API_KEY"
            : settings.AnalysisApiKeyEnvironment;
        CodexCliPathBox.Text = settings.CodexCliPath;
        AllowExternalAnalysisToggle.IsOn = settings.AllowExternalAnalysis;
        LanSyncToggle.IsOn = settings.LanSyncEnabled;
        LanDiscoveryToggle.IsOn = settings.LanDiscoveryEnabled;
        LanPeerUrlsBox.Text = settings.LanPeerUrls;
        LanShareToggle.IsOn = settings.LanShareEnabled;
        LanSharePortBox.Value = Math.Clamp(settings.LanSharePort, 1024, 65535);
        var savedAnalysisKey = CredentialStore.TryLoadAnalysisKey();
        if (savedAnalysisKey is not null)
        {
            AnalysisApiKeyBox.Password = savedAnalysisKey.Secret;
        }
        RememberAnalysisApiKeyCheckBox.IsChecked =
            settings.RememberAnalysisApiKey && savedAnalysisKey is not null;
        _savedHuggingFaceToken = CredentialStore.TryLoadHuggingFaceToken();
        _lastAreaProfileName = settings.LastAreaProfileName;
        _lastReviewFeedId = settings.LastReviewFeedId;
        _lastReviewDate = settings.LastReviewDate;
        AnalysisFeedBox.Text = _lastReviewFeedId;
        _loadingSettings = false;
        if (migratedOutputDirectory)
        {
            PersistUserSettings();
            AppendLog(
                $"Reconnected the installed app to the existing library at "
                + $"{configuredOutputDirectory}.");
        }
        RefreshHuggingFaceCredentialUi();
        UpdateAnalysisProviderUi();
        UpdateAsrModelPreparationUi();
        UpdateSetupSummary();
    }

    private static DesktopDockSide ParseDesktopDockSide(string? value) =>
        value?.Trim().ToLowerInvariant() switch
        {
            "left" => DesktopDockSide.Left,
            "right" => DesktopDockSide.Right,
            _ => DesktopDockSide.None,
        };

    private static RectInt32? SavedFloatingBounds(DesktopSettings settings)
    {
        if (settings.DesktopWindowX is not int x
            || settings.DesktopWindowY is not int y
            || settings.DesktopWindowWidth is not int width
            || settings.DesktopWindowHeight is not int height
            || width <= 0
            || height <= 0)
        {
            return null;
        }
        return new RectInt32(x, y, width, height);
    }

    private string DesktopDockSideSetting =>
        (_desktopDockManager?.IsDocked == true
            ? _desktopDockManager.Side
            : _desktopDockSide) switch
        {
            DesktopDockSide.Left => "left",
            DesktopDockSide.Right => "right",
            _ => "none",
        };

    private void WindowRoot_Loaded(object sender, RoutedEventArgs e)
    {
        if (_desktopDockRestoreAttempted)
        {
            return;
        }
        _desktopDockRestoreAttempted = true;
        if (_desktopDockSide is DesktopDockSide.Left or DesktopDockSide.Right)
        {
            TryDockDesktop(_desktopDockSide, persist: false);
        }
        else
        {
            UpdateDesktopDockUi();
        }
        ApplyResponsiveLayout(WindowRoot.ActualWidth);
    }

    private void DockButton_Click(object sender, RoutedEventArgs e)
    {
        if (_desktopDockManager?.IsDocked == true)
        {
            UnpinDesktop();
            return;
        }

        PrepareDesktopDockMenu();
        DockMenuFlyout.ShowAt(DockButton);
    }

    private void DockButton_RightTapped(object sender, RightTappedRoutedEventArgs e)
    {
        PrepareDesktopDockMenu();
        DockMenuFlyout.ShowAt(DockButton);
        e.Handled = true;
    }

    private void DockLeft_Click(object sender, RoutedEventArgs e) =>
        TryDockDesktop(DesktopDockSide.Left);

    private void DockRight_Click(object sender, RoutedEventArgs e) =>
        TryDockDesktop(DesktopDockSide.Right);

    private void DockUnpin_Click(object sender, RoutedEventArgs e) =>
        UnpinDesktop();

    private bool TryDockDesktop(
        DesktopDockSide side,
        bool persist = true)
    {
        if (_desktopDockManager is null)
        {
            return false;
        }

        try
        {
            var requestedMonitor = _desktopDockManager.IsDocked
                || _desktopDockSide is DesktopDockSide.Left or DesktopDockSide.Right
                    ? _desktopDockMonitor
                    : null;
            _desktopDockManager.Dock(
                side,
                _desktopDockWidth,
                requestedMonitor);
            SetDesktopWindowFrame(docked: true);
            _desktopDockSide = side;
            _desktopDockWidth = _desktopDockManager.WidthDips;
            _desktopDockMonitor = _desktopDockManager.MonitorDeviceName;
            UpdateDesktopDockUi();
            ApplyResponsiveLayout(WindowRoot.ActualWidth);
            if (persist)
            {
                PersistUserSettings(logFailure: true);
            }
            AppendLog(
                $"Pinned to the {side.ToString().ToLowerInvariant()} desktop edge at "
                + $"{_desktopDockWidth:N0} logical pixels; Windows work area is reserved.");
            return true;
        }
        catch (Exception exception)
        {
            _suspendDesktopPlacementTracking = true;
            try
            {
                _desktopDockManager.Unpin(restoreFloatingWindow: false);
                SetDesktopWindowFrame(docked: false);
                _desktopDockManager.RestoreFloatingWindowForStartup();
            }
            finally
            {
                _suspendDesktopPlacementTracking = false;
            }
            _desktopDockSide = DesktopDockSide.None;
            UpdateDesktopDockUi();
            StatusText.Text = "Desktop docking unavailable";
            AppendLog($"Desktop docking: {exception.Message}");
            return false;
        }
    }

    private void UnpinDesktop()
    {
        if (_desktopDockManager?.IsDocked != true)
        {
            return;
        }

        _desktopDockWidth = _desktopDockManager.WidthDips;
        _desktopDockMonitor = _desktopDockManager.MonitorDeviceName;
        _suspendDesktopPlacementTracking = true;
        try
        {
            _desktopDockManager.Unpin(restoreFloatingWindow: false);
            SetDesktopWindowFrame(docked: false);
            _desktopDockManager.RestoreFloatingWindowForStartup();
        }
        finally
        {
            _suspendDesktopPlacementTracking = false;
        }
        _desktopDockSide = DesktopDockSide.None;
        UpdateDesktopDockUi();
        ApplyResponsiveLayout(WindowRoot.ActualWidth);
        QueueFloatingTitleBarReattachment();
        PersistUserSettings(logFailure: true);
        AppendLog("Unpinned from the desktop edge and restored the floating window.");
    }

    private void QueueFloatingTitleBarReattachment()
    {
        DispatcherQueue.TryEnqueue(() =>
        {
            if (_desktopDockManager?.IsDocked == true)
            {
                return;
            }

            AttachCustomTitleBar(resetFirst: true);
            _desktopDockManager?.RefreshFrameAfterPresenterChange();
            WindowRoot.UpdateLayout();
        });
    }

    private void PrepareDesktopDockMenu()
    {
        var side = _desktopDockManager?.Side ?? DesktopDockSide.None;
        DockLeftMenuItem.IsEnabled = side != DesktopDockSide.Left;
        DockRightMenuItem.IsEnabled = side != DesktopDockSide.Right;
        var pinned = _desktopDockManager?.IsDocked == true;
        DockMenuSeparator.Visibility = pinned ? Visibility.Visible : Visibility.Collapsed;
        DockUnpinMenuItem.Visibility = pinned ? Visibility.Visible : Visibility.Collapsed;
    }

    private void TitleBarPaneButton_Click(object sender, RoutedEventArgs e)
    {
        if (_desktopDockManager?.IsDocked != true)
        {
            return;
        }

        RootNavigation.IsPaneOpen = !RootNavigation.IsPaneOpen;
    }

    private void TitleBarMinimizeButton_Click(object sender, RoutedEventArgs e)
    {
        if (AppWindow.Presenter is OverlappedPresenter presenter
            && presenter.IsMinimizable)
        {
            presenter.Minimize();
        }
    }

    private void TitleBarMaximizeButton_Click(object sender, RoutedEventArgs e)
    {
        if (AppWindow.Presenter is not OverlappedPresenter presenter)
        {
            return;
        }

        if (presenter.State == OverlappedPresenterState.Maximized)
        {
            presenter.Restore();
        }
        else if (presenter.IsMaximizable)
        {
            presenter.Maximize();
        }
        UpdateTitleBarWindowControls();
    }

    private void TitleBarCloseButton_Click(object sender, RoutedEventArgs e) => Close();

    private void UpdateTitleBarWindowControls()
    {
        var maximized = AppWindow.Presenter is OverlappedPresenter presenter
            && presenter.State == OverlappedPresenterState.Maximized;
        TitleBarMaximizeIcon.Glyph = maximized ? "\uE923" : "\uE922";
        var action = maximized ? "Restore" : "Maximize";
        AutomationProperties.SetName(TitleBarMaximizeButton, action);
        ToolTipService.SetToolTip(TitleBarMaximizeButton, action);
    }

    private void UpdateDesktopDockUi()
    {
        var manager = _desktopDockManager;
        var pinned = manager?.IsDocked == true;
        var side = manager?.Side ?? DesktopDockSide.None;
        AppTitleBar.Visibility = Visibility.Visible;
        TitleBarPaneButton.Visibility = pinned
            ? Visibility.Visible
            : Visibility.Collapsed;
        TitleBarMinimizeButton.Visibility = pinned
            ? Visibility.Collapsed
            : Visibility.Visible;
        TitleBarMaximizeButton.Visibility = pinned
            ? Visibility.Collapsed
            : Visibility.Visible;
        TitleBarCloseButton.Visibility = Visibility.Visible;
        RootNavigation.IsPaneToggleButtonVisible = !pinned;
        DockButton.Content = pinned ? "Unpin" : "Pin";
        AutomationProperties.SetName(
            DockButton,
            pinned ? "Unpin from desktop" : "Pin to desktop");
        AutomationProperties.SetHelpText(
            DockButton,
            pinned
                ? "Left click to unpin. Right click to change the pinned side."
                : "Choose the left or right side and reserve desktop space.");
        ToolTipService.SetToolTip(
            DockButton,
            pinned
                ? "Unpin (right-click to change side)"
                : "Pin to the left or right desktop edge");
        DockResizeHandle.Visibility = pinned ? Visibility.Visible : Visibility.Collapsed;
        DockedFrameBorder.Visibility = pinned ? Visibility.Visible : Visibility.Collapsed;
        DockResizeHandle.HorizontalAlignment = side == DesktopDockSide.Left
            ? HorizontalAlignment.Right
            : HorizontalAlignment.Left;
        DockedFrameBorder.BorderThickness = side switch
        {
            DesktopDockSide.Left => new Thickness(0, 0, 1, 0),
            DesktopDockSide.Right => new Thickness(1, 0, 0, 0),
            _ => new Thickness(0),
        };
        RootNavigation.PaneDisplayMode = pinned
            ? NavigationViewPaneDisplayMode.LeftMinimal
            : NavigationViewPaneDisplayMode.Auto;
        if (pinned)
        {
            RootNavigation.IsPaneOpen = false;
            AppTitleSubtitle.Text =
                $"Pinned {side.ToString().ToLowerInvariant()} · drag the inner edge to resize";
        }
        else
        {
            AppTitleSubtitle.Text = "Local radio archive intelligence";
        }
        UpdateTitleBarWindowControls();
        ApplyDesktopDockActivationState(_desktopWindowActive);
        PrepareDesktopDockMenu();
    }

    private void DockResizeHandle_PointerPressed(
        object sender,
        PointerRoutedEventArgs e)
    {
        if (e.GetCurrentPoint(DockResizeHandle).Properties.IsLeftButtonPressed
            && _desktopDockManager?.BeginPointerResize() == true)
        {
            DockResizeHandle.CapturePointer(e.Pointer);
            e.Handled = true;
        }
    }

    private void DockResizeHandle_PointerMoved(
        object sender,
        PointerRoutedEventArgs e)
    {
        if (_desktopDockManager?.IsInteractiveResize != true)
        {
            return;
        }
        _desktopDockManager.ContinuePointerResize();
        _desktopDockWidth = _desktopDockManager.WidthDips;
        ApplyResponsiveLayout(WindowRoot.ActualWidth);
        e.Handled = true;
    }

    private void DockResizeHandle_PointerReleased(
        object sender,
        PointerRoutedEventArgs e)
    {
        if (_desktopDockManager?.IsInteractiveResize == true)
        {
            _desktopDockManager.CompletePointerResize();
            DockResizeHandle.ReleasePointerCapture(e.Pointer);
            e.Handled = true;
        }
    }

    private void DockResizeHandle_PointerCanceled(
        object sender,
        PointerRoutedEventArgs e) =>
        _desktopDockManager?.CompletePointerResize();

    private void DockResizeHandle_PointerCaptureLost(
        object sender,
        PointerRoutedEventArgs e) =>
        _desktopDockManager?.CompletePointerResize();

    private void DesktopDockManager_InteractiveResizeCompleted(double widthDips)
    {
        _desktopDockWidth = widthDips;
        _desktopDockMonitor = _desktopDockManager?.MonitorDeviceName ?? "";
        PersistUserSettings(logFailure: true);
        AppendLog($"Pinned width saved at {_desktopDockWidth:N0} logical pixels.");
    }

    private void MainWindow_Activated(
        object sender,
        WindowActivatedEventArgs args) =>
        ApplyDesktopDockActivationState(
            args.WindowActivationState != WindowActivationState.Deactivated);

    private void ApplyDesktopDockActivationState(bool active)
    {
        _desktopWindowActive = active;
        if (DockedFrameBorder is null || DockResizeIndicator is null)
        {
            return;
        }
        AppTitleBar.Opacity = active ? 1 : 0.82;
        DockedFrameBorder.Opacity = active ? 1 : 0.72;
        DockResizeIndicator.Opacity = active ? 0.58 : 0.35;
    }

    private void SetDesktopWindowFrame(bool docked)
    {
        if (AppWindow.Presenter is not OverlappedPresenter presenter)
        {
            throw new InvalidOperationException(
                "Desktop docking requires a normal desktop window.");
        }

        if (docked)
        {
            if (presenter.State != OverlappedPresenterState.Restored)
            {
                presenter.Restore();
            }
            presenter.SetBorderAndTitleBar(hasBorder: false, hasTitleBar: false);
        }
        else
        {
            presenter.SetBorderAndTitleBar(hasBorder: true, hasTitleBar: false);
        }

        // The visible chrome is ordinary fixed XAML content in either mode.
        // Resetting only re-registers its dedicated drag surface after a
        // presenter transition; it cannot remove the visible chrome row.
        AttachCustomTitleBar(resetFirst: !docked);

        var cornerPreference = docked
            ? DwmCornerDoNotRound
            : DwmCornerDefault;
        SetDwmWindowAttributeOrThrow(
            DwmWindowCornerPreference,
            cornerPreference);
        var borderColor = docked ? DwmColorNone : DwmColorDefault;
        SetDwmWindowAttributeOrThrow(DwmWindowBorderColor, borderColor);
        _desktopWindowCornerPreference = cornerPreference;
        _desktopWindowBorderColor = borderColor;
        _desktopWindowFrameDocked = docked;
        _desktopDockManager?.RefreshFrameAfterPresenterChange();
    }

    private void AttachCustomTitleBar(bool resetFirst)
    {
        if (resetFirst)
        {
            SetTitleBar(null);
            ExtendsContentIntoTitleBar = false;
        }

        ExtendsContentIntoTitleBar = true;
        SetTitleBar(TitleBarDragRegion);
        AppTitleBar.Visibility = Visibility.Visible;
        UpdateTitleBarWindowControls();
    }

    private void SetDwmWindowAttributeOrThrow(int attribute, int value)
    {
        var result = DwmSetWindowAttribute(
            WindowNative.GetWindowHandle(this),
            attribute,
            ref value,
            sizeof(int));
        if (result < 0)
        {
            Marshal.ThrowExceptionForHR(result);
        }
    }

    private bool TryReadDwmWindowAttribute(int attribute, out int value)
    {
        value = 0;
        return DwmGetWindowAttribute(
            WindowNative.GetWindowHandle(this),
            attribute,
            out value,
            sizeof(int)) == 0;
    }

    private void AppWindow_Changed(
        AppWindow sender,
        AppWindowChangedEventArgs args)
    {
        if (args.DidPresenterChange)
        {
            UpdateTitleBarWindowControls();
        }
        if (_desktopDockManager is null
            || _suspendDesktopPlacementTracking
            || _desktopDockManager.IsDocked
            || (!args.DidPositionChange
                && !args.DidSizeChange
                && !args.DidPresenterChange))
        {
            return;
        }
        var maximized = sender.Presenter is OverlappedPresenter presenter
            && presenter.State == OverlappedPresenterState.Maximized;
        _desktopDockManager.UpdateFloatingPlacement(
            new RectInt32(
                sender.Position.X,
                sender.Position.Y,
                sender.Size.Width,
                sender.Size.Height),
            maximized);
        ScheduleSettingsSave();
    }

    private void WindowRoot_SizeChanged(object sender, SizeChangedEventArgs e) =>
        ApplyResponsiveLayout(e.NewSize.Width);

    private void ApplyResponsiveLayout(double width)
    {
        // The navigation pane, page padding, and widest master/detail surface
        // need more than a nominal tablet width. Stack before either pane can
        // compress the active Review/Library controls beyond their content.
        var compact = width < 1_100 || _desktopDockManager?.IsDocked == true;
        var shortCompact = compact
            && WindowRoot.ActualHeight > 0
            && WindowRoot.ActualHeight < 820;
        if (_compactLayoutApplied == compact
            && _shortCompactLayoutApplied == shortCompact)
        {
            return;
        }
        var enteringCompact = _compactLayoutApplied != true && compact;
        _compactLayoutApplied = compact;
        _shortCompactLayoutApplied = shortCompact;

        var star = new GridLength(1, GridUnitType.Star);
        var zero = new GridLength(0);
        var pagePadding = compact
            ? new Thickness(14, 14, 14, 12)
            : new Thickness(26, 20, 26, 18);
        foreach (var page in new FrameworkElement[]
                 {
                     LibraryPage,
                     SystemPage,
                     ArchivePage,
                     ReviewPage,
                     AreaPage,
                     SettingsPage,
                     AboutPage,
                 })
        {
            if (page is Grid grid)
            {
                grid.Padding = pagePadding;
            }
        }

        if (compact)
        {
            RootNavigation.IsPaneOpen = false;
            if (enteringCompact)
            {
                LibraryCoverageExpander.IsExpanded = false;
            }
            LibraryStatsGrid.Visibility = shortCompact
                ? Visibility.Collapsed
                : Visibility.Visible;

            Grid.SetRow(LibraryHeaderActions, 1);
            Grid.SetColumn(LibraryHeaderActions, 0);
            LibraryHeaderActions.Orientation = Orientation.Vertical;
            LibraryHeaderActions.HorizontalAlignment = HorizontalAlignment.Stretch;
            LibraryHeaderGrid.ColumnDefinitions[1].Width = zero;

            for (var index = 0; index < LibraryStatsGrid.ColumnDefinitions.Count; index++)
            {
                LibraryStatsGrid.ColumnDefinitions[index].Width = index < 2 ? star : zero;
            }
            Grid.SetRow(LibraryFeedsStat, 0);
            Grid.SetColumn(LibraryFeedsStat, 0);
            Grid.SetRow(LibraryDaysStat, 0);
            Grid.SetColumn(LibraryDaysStat, 1);
            Grid.SetRow(LibraryBacklogStat, 1);
            Grid.SetColumn(LibraryBacklogStat, 0);
            Grid.SetRow(LibraryReadyStat, 1);
            Grid.SetColumn(LibraryReadyStat, 1);

            LibraryFilterGrid.ColumnDefinitions[1].Width = zero;
            Grid.SetRow(LibraryFilterCombo, 1);
            Grid.SetColumn(LibraryFilterCombo, 0);

            LibraryContentGrid.ColumnDefinitions[0].Width = star;
            LibraryContentGrid.ColumnDefinitions[0].MinWidth = 0;
            LibraryContentGrid.ColumnDefinitions[1].Width = zero;
            LibraryContentGrid.ColumnDefinitions[1].MinWidth = 0;
            LibraryContentGrid.RowDefinitions[0].Height = new GridLength(2, GridUnitType.Star);
            LibraryContentGrid.RowDefinitions[1].Height = new GridLength(3, GridUnitType.Star);
            Grid.SetRow(LibraryDetailBorder, 1);
            Grid.SetColumn(LibraryDetailBorder, 0);
            LibraryDetailBorder.BorderThickness = new Thickness(0, 1, 0, 0);

            ArchiveContentGrid.ColumnDefinitions[0].Width = star;
            ArchiveContentGrid.ColumnDefinitions[1].Width = zero;
            ArchiveContentGrid.RowDefinitions[0].Height = new GridLength(3, GridUnitType.Star);
            ArchiveContentGrid.RowDefinitions[1].Height = new GridLength(4, GridUnitType.Star);
            Grid.SetRow(ArchiveJobBorder, 1);
            Grid.SetColumn(ArchiveJobBorder, 0);

            ReviewContentGrid.ColumnDefinitions[0].Width = star;
            ReviewContentGrid.ColumnDefinitions[1].Width = zero;
            ReviewContentGrid.RowDefinitions[0].Height = new GridLength(
                shortCompact ? 170 : 280);
            ReviewContentGrid.RowDefinitions[1].Height = star;
            Grid.SetRow(ReviewTabView, 1);
            Grid.SetColumn(ReviewTabView, 0);
            ReviewDaySelectorGrid.RowSpacing = shortCompact ? 4 : 10;
            AnalysisFeedCombo.Header = shortCompact ? null : "Feed";
            DailyReviewContentGrid.RowSpacing = shortCompact ? 4 : 9;
            SummaryText.MaxHeight = shortCompact ? 40 : 126;
            PlaybackStatusText.MaxLines = shortCompact ? 1 : 0;
            IncidentPlayer.Height = shortCompact ? 44 : 72;

            WeekControls.Orientation = Orientation.Vertical;
            QuestionScopeActions.Orientation = Orientation.Vertical;
            QuestionStarterPrimary.Orientation = Orientation.Vertical;
            QuestionStarterSecondary.Orientation = Orientation.Vertical;
            QuestionSendGrid.ColumnDefinitions[1].Width = zero;
            Grid.SetRow(AskButton, 1);
            Grid.SetColumn(AskButton, 0);
            AskButton.HorizontalAlignment = HorizontalAlignment.Stretch;

            AreaFeedsGrid.ColumnDefinitions[0].Width = star;
            AreaFeedsGrid.ColumnDefinitions[1].Width = zero;
            AreaFeedsGrid.RowDefinitions[0].Height = star;
            AreaFeedsGrid.RowDefinitions[1].Height = GridLength.Auto;
            Grid.SetRow(AreaQueueBorder, 1);
            Grid.SetColumn(AreaQueueBorder, 0);

            AreaStoryContentGrid.ColumnDefinitions[0].Width = star;
            AreaStoryContentGrid.ColumnDefinitions[1].Width = zero;
            AreaStoryContentGrid.RowDefinitions[0].Height = star;
            AreaStoryContentGrid.RowDefinitions[1].Height = new GridLength(2, GridUnitType.Star);
            Grid.SetRow(AreaStoryDetailBorder, 1);
            Grid.SetColumn(AreaStoryDetailBorder, 0);
            AreaStoriesContentGrid.RowSpacing = shortCompact ? 4 : 10;
            AreaSummaryScroll.MaxHeight = shortCompact ? 64 : 150;

            RuntimeControls.Orientation = Orientation.Vertical;
            CredentialButtons.Orientation = Orientation.Vertical;
            CredentialLinks.Orientation = Orientation.Vertical;
            AboutActionButtons.Orientation = Orientation.Vertical;
            AboutHelpLinks.Orientation = Orientation.Vertical;

            PersistentStatusGrid.ColumnDefinitions[0].Width = star;
            PersistentStatusGrid.ColumnDefinitions[1].Width = GridLength.Auto;
            PersistentStatusGrid.ColumnDefinitions[2].Width = zero;
            Grid.SetRow(JobProgress, 1);
            Grid.SetColumn(JobProgress, 0);
            Grid.SetColumnSpan(JobProgress, 2);
            Grid.SetRow(CancelButton, 0);
            Grid.SetColumn(CancelButton, 1);
        }
        else
        {
            LibraryStatsGrid.Visibility = Visibility.Visible;
            Grid.SetRow(LibraryHeaderActions, 0);
            Grid.SetColumn(LibraryHeaderActions, 1);
            LibraryHeaderActions.Orientation = Orientation.Horizontal;
            LibraryHeaderActions.HorizontalAlignment = HorizontalAlignment.Left;
            LibraryHeaderGrid.ColumnDefinitions[1].Width = GridLength.Auto;

            foreach (var definition in LibraryStatsGrid.ColumnDefinitions)
            {
                definition.Width = star;
            }
            foreach (var value in new[]
                     {
                         LibraryFeedsStat,
                         LibraryDaysStat,
                         LibraryBacklogStat,
                         LibraryReadyStat,
                     })
            {
                Grid.SetRow(value, 0);
            }
            Grid.SetColumn(LibraryFeedsStat, 0);
            Grid.SetColumn(LibraryDaysStat, 1);
            Grid.SetColumn(LibraryBacklogStat, 2);
            Grid.SetColumn(LibraryReadyStat, 3);

            LibraryFilterGrid.ColumnDefinitions[1].Width = new GridLength(220);
            Grid.SetRow(LibraryFilterCombo, 0);
            Grid.SetColumn(LibraryFilterCombo, 1);

            LibraryContentGrid.ColumnDefinitions[0].Width = new GridLength(5, GridUnitType.Star);
            LibraryContentGrid.ColumnDefinitions[0].MinWidth = 330;
            LibraryContentGrid.ColumnDefinitions[1].Width = new GridLength(6, GridUnitType.Star);
            LibraryContentGrid.ColumnDefinitions[1].MinWidth = 390;
            LibraryContentGrid.RowDefinitions[0].Height = star;
            LibraryContentGrid.RowDefinitions[1].Height = zero;
            Grid.SetRow(LibraryDetailBorder, 0);
            Grid.SetColumn(LibraryDetailBorder, 1);
            LibraryDetailBorder.BorderThickness = new Thickness(1, 0, 0, 0);

            ArchiveContentGrid.ColumnDefinitions[0].Width = new GridLength(3, GridUnitType.Star);
            ArchiveContentGrid.ColumnDefinitions[1].Width = new GridLength(2, GridUnitType.Star);
            ArchiveContentGrid.RowDefinitions[0].Height = star;
            ArchiveContentGrid.RowDefinitions[1].Height = zero;
            Grid.SetRow(ArchiveJobBorder, 0);
            Grid.SetColumn(ArchiveJobBorder, 1);

            ReviewContentGrid.ColumnDefinitions[0].Width = new GridLength(310);
            ReviewContentGrid.ColumnDefinitions[1].Width = star;
            ReviewContentGrid.RowDefinitions[0].Height = star;
            ReviewContentGrid.RowDefinitions[1].Height = zero;
            Grid.SetRow(ReviewTabView, 0);
            Grid.SetColumn(ReviewTabView, 1);
            ReviewDaySelectorGrid.RowSpacing = 10;
            AnalysisFeedCombo.Header = "Feed";
            DailyReviewContentGrid.RowSpacing = 9;
            SummaryText.MaxHeight = 126;
            PlaybackStatusText.MaxLines = 0;
            IncidentPlayer.Height = 72;

            WeekControls.Orientation = Orientation.Horizontal;
            QuestionScopeActions.Orientation = Orientation.Horizontal;
            QuestionStarterPrimary.Orientation = Orientation.Horizontal;
            QuestionStarterSecondary.Orientation = Orientation.Horizontal;
            QuestionSendGrid.ColumnDefinitions[1].Width = GridLength.Auto;
            Grid.SetRow(AskButton, 0);
            Grid.SetColumn(AskButton, 1);
            AskButton.HorizontalAlignment = HorizontalAlignment.Left;

            AreaFeedsGrid.ColumnDefinitions[0].Width = new GridLength(3, GridUnitType.Star);
            AreaFeedsGrid.ColumnDefinitions[1].Width = new GridLength(2, GridUnitType.Star);
            AreaFeedsGrid.RowDefinitions[0].Height = star;
            AreaFeedsGrid.RowDefinitions[1].Height = zero;
            Grid.SetRow(AreaQueueBorder, 0);
            Grid.SetColumn(AreaQueueBorder, 1);

            AreaStoryContentGrid.ColumnDefinitions[0].Width = new GridLength(330);
            AreaStoryContentGrid.ColumnDefinitions[1].Width = star;
            AreaStoryContentGrid.RowDefinitions[0].Height = star;
            AreaStoryContentGrid.RowDefinitions[1].Height = zero;
            Grid.SetRow(AreaStoryDetailBorder, 0);
            Grid.SetColumn(AreaStoryDetailBorder, 1);
            AreaStoriesContentGrid.RowSpacing = 10;
            AreaSummaryScroll.MaxHeight = 150;

            RuntimeControls.Orientation = Orientation.Horizontal;
            CredentialButtons.Orientation = Orientation.Horizontal;
            CredentialLinks.Orientation = Orientation.Horizontal;
            AboutActionButtons.Orientation = Orientation.Horizontal;
            AboutHelpLinks.Orientation = Orientation.Horizontal;

            PersistentStatusGrid.ColumnDefinitions[0].Width = star;
            PersistentStatusGrid.ColumnDefinitions[1].Width = new GridLength(260);
            PersistentStatusGrid.ColumnDefinitions[2].Width = GridLength.Auto;
            Grid.SetRow(JobProgress, 0);
            Grid.SetColumn(JobProgress, 1);
            Grid.SetColumnSpan(JobProgress, 1);
            Grid.SetRow(CancelButton, 0);
            Grid.SetColumn(CancelButton, 2);
        }

        AppTitleSubtitle.Text = _desktopDockManager?.IsDocked == true
            ? $"Pinned {_desktopDockManager.Side.ToString().ToLowerInvariant()} · drag the inner edge to resize"
            : compact
                ? ""
                : "Local radio archive intelligence";
    }

    private void MainWindow_Closed(object sender, WindowEventArgs args)
    {
        _settingsSaveTimer?.Stop();
        _feedScheduleTimer?.Stop();
        _systemActivityTimer?.Stop();
        _pipelineCancellation?.Cancel();
        _operationCancellation?.Cancel();
        _questionCancellation?.Cancel();
        PersistUserSettings(logFailure: true);
        PersistAnalysisCredentialPreference();
        AppWindow.Changed -= AppWindow_Changed;
        Activated -= MainWindow_Activated;
        if (_desktopDockManager is not null)
        {
            _desktopDockManager.InteractiveResizeCompleted -=
                DesktopDockManager_InteractiveResizeCompleted;
        }
        _desktopDockManager?.Dispose();
        _worker?.StopLanNode();
        ReleaseMediaPlayerInstances(recreate: false);
    }

    private DesktopSettings CaptureUserSettings()
    {
        var floatingBounds = _desktopDockManager?.FloatingBounds
            ?? new RectInt32(
                AppWindow.Position.X,
                AppWindow.Position.Y,
                AppWindow.Size.Width,
                AppWindow.Size.Height);
        var floatingMaximized = _desktopDockManager?.RestoreMaximized
            ?? (AppWindow.Presenter is OverlappedPresenter presenter
                && presenter.State == OverlappedPresenterState.Maximized);
        return new DesktopSettings
        {
            PythonRuntimePath = _configuredPythonRuntimePath,
            HardwareProfile = SelectedComboValue(HardwareProfileComboBox, "auto"),
            WhisperModel = SelectedComboValue(ModelComboBox, "turbo"),
            AsrEngine = SelectedComboValue(AsrEngineComboBox, "auto"),
            TranscriptionDevice = SelectedComboValue(DeviceComboBox, "auto"),
            DiarizationEngine = SelectedComboValue(DiarizationEngineComboBox, "community-1"),
            DiarizationDevice = SelectedComboValue(DiarizationDeviceComboBox, "auto"),
            AsrModelPath = AsrModelPathBox.Text.Trim(),
            OutputDirectory = PersistedOutputDirectory(),
            GpuIndex = RequiredInteger(GpuIndexBox.Value, 0),
            BatchSize = RequiredInteger(BatchSizeBox.Value, 8),
            MinimumSpeakers = RequiredInteger(MinimumSpeakersBox.Value, 0),
            MaximumSpeakers = RequiredInteger(MaximumSpeakersBox.Value, 0),
            DownloadJobs = RequiredInteger(DownloadJobsBox.Value, 1),
            Combine = CombineToggle.IsOn,
            KeepOriginals = KeepOriginalsToggle.IsOn,
            Transcribe = TranscribeCheckBox.IsChecked == true,
            Diarize = DiarizeCheckBox.IsChecked == true,
            AnalyzeAfterJob = AnalyzeAfterJobCheckBox.IsChecked == true,
            AnalysisProvider = SelectedComboValue(AnalysisProviderComboBox, "local"),
            AnalysisModel = AnalysisModelBox.Text.Trim(),
            AnalysisDevice = SelectedComboValue(AnalysisDeviceComboBox, "auto"),
            AnalysisEndpoint = AnalysisEndpointBox.Text.Trim(),
            AnalysisApiKeyEnvironment = string.IsNullOrWhiteSpace(AnalysisApiKeyEnvironmentBox.Text)
                ? "OPENAI_API_KEY"
                : AnalysisApiKeyEnvironmentBox.Text.Trim(),
            CodexCliPath = CodexCliPathBox.Text.Trim(),
            AllowExternalAnalysis = AllowExternalAnalysisToggle.IsOn,
            RememberAnalysisApiKey = RememberAnalysisApiKeyCheckBox.IsChecked == true,
            LanSyncEnabled = LanSyncToggle.IsOn,
            LanDiscoveryEnabled = LanDiscoveryToggle.IsOn,
            LanPeerUrls = LanPeerUrlsBox.Text.Trim(),
            LanShareEnabled = LanShareToggle.IsOn,
            LanSharePort = RequiredInteger(LanSharePortBox.Value, 8766),
            LastAreaProfileName = _lastAreaProfileName,
            LastReviewFeedId = _lastReviewFeedId,
            LastReviewDate = _lastReviewDate,
            DesktopDockSide = DesktopDockSideSetting,
            DesktopDockWidth = _desktopDockManager?.WidthDips
                ?? _desktopDockWidth,
            DesktopDockMonitor = _desktopDockManager?.MonitorDeviceName
                ?? _desktopDockMonitor,
            DesktopWindowX = floatingBounds.X,
            DesktopWindowY = floatingBounds.Y,
            DesktopWindowWidth = floatingBounds.Width,
            DesktopWindowHeight = floatingBounds.Height,
            DesktopWindowMaximized = floatingMaximized,
        };
    }

    private string PersistedOutputDirectory()
    {
        var configured = string.IsNullOrWhiteSpace(OutputFolderBox?.Text)
            ? "archives"
            : OutputFolderBox.Text.Trim();
        try
        {
            if (Path.IsPathRooted(configured))
            {
                return Path.GetFullPath(configured);
            }
            if (_worker is null)
            {
                return configured;
            }
            return Path.GetFullPath(configured, _worker.WorkingDirectory);
        }
        catch (Exception exception) when (
            exception is ArgumentException
                or IOException
                or NotSupportedException)
        {
            // Preserve in-progress text without crashing autosave. Storage
            // validation will explain the invalid path before a job can run.
            return configured;
        }
    }

    private void PersistUserSettings(bool logFailure = false)
    {
        if (_loadingSettings)
        {
            return;
        }
        if (!AppSettingsStore.TrySave(CaptureUserSettings()) && logFailure)
        {
            AppendLog("Settings could not be saved to this Windows account.");
        }
    }

    private void PersistAnalysisCredentialPreference()
    {
        if (RememberAnalysisApiKeyCheckBox.IsChecked == true
            && !string.IsNullOrWhiteSpace(AnalysisApiKeyBox.Password))
        {
            CredentialStore.SaveAnalysisKey(AnalysisApiKeyBox.Password);
        }
        else if (RememberAnalysisApiKeyCheckBox.IsChecked != true)
        {
            CredentialStore.ClearAnalysisKey();
        }
    }

    private void WireSettingsAutoSave()
    {
        _settingsSaveTimer = DispatcherQueue.CreateTimer();
        _settingsSaveTimer.Interval = TimeSpan.FromMilliseconds(750);
        _settingsSaveTimer.IsRepeating = false;
        _settingsSaveTimer.Tick += async (_, _) =>
        {
            PersistUserSettings();
            await ConfigureLanSharingAsync();
        };

        foreach (var comboBox in new[]
                 {
                     HardwareProfileComboBox,
                     ModelComboBox,
                     AsrEngineComboBox,
                     DeviceComboBox,
                     DiarizationEngineComboBox,
                     DiarizationDeviceComboBox,
                     AnalysisProviderComboBox,
                     AnalysisDeviceComboBox,
                 })
        {
            comboBox.SelectionChanged += (_, _) => ScheduleSettingsSave();
        }
        foreach (var textBox in new[]
                 {
                     AsrModelPathBox,
                     OutputFolderBox,
                     AnalysisModelBox,
                     AnalysisEndpointBox,
                     AnalysisApiKeyEnvironmentBox,
                     CodexCliPathBox,
                     LanPeerUrlsBox,
                     AnalysisFeedBox,
                 })
        {
            textBox.TextChanged += (_, _) => ScheduleSettingsSave();
        }
        PythonRuntimePathBox.TextChanged += (_, _) =>
        {
            if (_loadingSettings || !_pythonRuntimeInputReady)
            {
                return;
            }
            _configuredPythonRuntimePath =
                PythonRuntimePathBox.Text.Trim();
            ScheduleSettingsSave();
        };
        foreach (var numberBox in new[]
                 {
                     GpuIndexBox,
                     BatchSizeBox,
                     MinimumSpeakersBox,
                     MaximumSpeakersBox,
                     DownloadJobsBox,
                     LanSharePortBox,
                 })
        {
            numberBox.ValueChanged += (_, _) => ScheduleSettingsSave();
        }
        foreach (var toggle in new[]
                 {
                     CombineToggle,
                     KeepOriginalsToggle,
                     AllowExternalAnalysisToggle,
                     LanSyncToggle,
                     LanDiscoveryToggle,
                     LanShareToggle,
                 })
        {
            toggle.Toggled += (_, _) => ScheduleSettingsSave();
        }
        foreach (var checkBox in new[]
                 {
                     TranscribeCheckBox,
                     DiarizeCheckBox,
                     AnalyzeAfterJobCheckBox,
                     RememberAnalysisApiKeyCheckBox,
                 })
        {
            checkBox.Checked += (_, _) => ScheduleSettingsSave();
            checkBox.Unchecked += (_, _) => ScheduleSettingsSave();
        }
        AnalysisModelBox.TextChanged += (_, _) => ResetAnalysisModelVerification();
        AnalysisDeviceComboBox.SelectionChanged += (_, _) => ResetAnalysisModelVerification();
        AnalysisApiKeyBox.PasswordChanged += (_, _) => ResetAnalysisModelVerification();
        AnalysisApiKeyEnvironmentBox.TextChanged += (_, _) => ResetAnalysisModelVerification();
        CodexCliPathBox.TextChanged += (_, _) => ResetAnalysisModelVerification();
        ModelComboBox.SelectionChanged += (_, _) =>
        {
            ResetAsrVerification();
            UpdateAsrModelPreparationUi();
        };
        AsrEngineComboBox.SelectionChanged += (_, _) =>
        {
            EnsureSelectedAsrModelCompatibility();
            ResetAsrVerification();
            UpdateAsrModelPreparationUi();
        };
        DeviceComboBox.SelectionChanged += (_, _) =>
        {
            ResetAsrVerification();
            UpdateAsrModelPreparationUi();
        };
        AsrModelPathBox.TextChanged += (_, _) => ResetAsrVerification();
        DiarizationEngineComboBox.SelectionChanged += (_, _) =>
        {
            EnsureDiarizationSelectionCompatibility();
            ResetDiarizationVerification();
        };
        DiarizationDeviceComboBox.SelectionChanged += (_, _) =>
        {
            EnsureDiarizationSelectionCompatibility();
            ResetDiarizationVerification();
        };
        GpuIndexBox.ValueChanged += (_, _) =>
        {
            ResetAsrVerification();
            ResetDiarizationVerification();
        };
        BatchSizeBox.ValueChanged += (_, _) =>
        {
            ResetAsrVerification();
            ResetDiarizationVerification();
        };
    }

    private void EnsureDiarizationSelectionCompatibility()
    {
        if (DiarizationEngineComboBox is null
            || DiarizationDeviceComboBox is null)
        {
            return;
        }
        if (SelectedComboValue(DiarizationEngineComboBox, "community-1")
                == "sherpa-onnx"
            && SelectedComboValue(DiarizationDeviceComboBox, "auto") == "cuda")
        {
            SelectComboTag(DiarizationDeviceComboBox, "cpu");
        }
    }

    private void ScheduleSettingsSave()
    {
        if (_loadingSettings || _settingsSaveTimer is null)
        {
            return;
        }
        _worker?.SetLibraryDirectory(OutputFolderBox.Text);
        _settingsSaveTimer.Stop();
        _settingsSaveTimer.Start();
    }

    private void ResetAnalysisModelVerification()
    {
        if (_loadingSettings)
        {
            return;
        }
        _analysisModelVerifiedThisSession = false;
        _analysisModelVerificationMessage = "";
        _profileRecoveryAction = null;
        if (AnalysisSelfTestInfoBar is not null)
        {
            AnalysisSelfTestInfoBar.Severity = InfoBarSeverity.Informational;
            AnalysisSelfTestInfoBar.Title = "Analysis model not tested for these settings";
            AnalysisSelfTestInfoBar.Message =
                "Run Test model after changing the provider, model, endpoint, credential, or device.";
        }
        UpdateSetupSummary();
    }

    private void ResetAsrVerification()
    {
        if (_loadingSettings)
        {
            return;
        }
        _asrVerifiedThisSession = false;
        _asrVerificationMessage = "";
        _profileRecoveryAction = null;
        if (AsrSelfTestInfoBar is not null)
        {
            AsrSelfTestInfoBar.Severity = InfoBarSeverity.Informational;
            AsrSelfTestInfoBar.Title = "Transcription not tested for these settings";
            AsrSelfTestInfoBar.Message =
                "Run Test engine after changing its engine, model, device, path, or batch.";
        }
        UpdateSetupSummary();
    }

    private void ResetDiarizationVerification()
    {
        if (_loadingSettings)
        {
            return;
        }
        _diarizationVerifiedThisSession = false;
        _diarizationVerificationMessage = "";
        _profileRecoveryAction = null;
        if (DiarizationSelfTestInfoBar is not null)
        {
            DiarizationSelfTestInfoBar.Severity = InfoBarSeverity.Informational;
            DiarizationSelfTestInfoBar.Title = "Speaker labels not tested for these settings";
            DiarizationSelfTestInfoBar.Message =
                "Run Test speakers after changing its engine, device, token, GPU, or batch.";
        }
        UpdateSetupSummary();
    }

    private void AnalysisProvider_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (_loadingSettings || AnalysisModelBox is null)
        {
            return;
        }
        var provider = SelectedComboValue(AnalysisProviderComboBox, "local");
        switch (provider)
        {
            case "openai-responses":
                if (string.IsNullOrWhiteSpace(AnalysisModelBox.Text)
                    || AnalysisModelBox.Text.StartsWith("ggml-org/", StringComparison.OrdinalIgnoreCase)
                    || AnalysisModelBox.Text == "local-model")
                {
                    AnalysisModelBox.Text = "gpt-5.6-luna";
                }
                if (string.IsNullOrWhiteSpace(AnalysisEndpointBox.Text))
                {
                    AnalysisEndpointBox.Text = "https://api.openai.com/v1";
                }
                break;
            case "openai-compatible":
                if (string.IsNullOrWhiteSpace(AnalysisModelBox.Text)
                    || AnalysisModelBox.Text.StartsWith("ggml-org/", StringComparison.OrdinalIgnoreCase)
                    || AnalysisModelBox.Text.StartsWith("gpt-", StringComparison.OrdinalIgnoreCase))
                {
                    AnalysisModelBox.Text = "local-model";
                }
                if (string.IsNullOrWhiteSpace(AnalysisEndpointBox.Text)
                    || AnalysisEndpointBox.Text.StartsWith("https://api.openai.com", StringComparison.OrdinalIgnoreCase))
                {
                    AnalysisEndpointBox.Text = "http://127.0.0.1:1234/v1";
                }
                break;
            case "codex-cli":
                if (AnalysisModelBox.Text.StartsWith("ggml-org/", StringComparison.OrdinalIgnoreCase)
                    || AnalysisModelBox.Text.StartsWith("gpt-", StringComparison.OrdinalIgnoreCase)
                    || AnalysisModelBox.Text == "local-model")
                {
                    AnalysisModelBox.Text = "";
                }
                AnalysisEndpointBox.Text = "";
                break;
            default:
                if (string.IsNullOrWhiteSpace(AnalysisModelBox.Text)
                    || AnalysisModelBox.Text.StartsWith("gpt-", StringComparison.OrdinalIgnoreCase)
                    || AnalysisModelBox.Text == "local-model")
                {
                    AnalysisModelBox.Text = DefaultAnalysisModel;
                }
                if (AnalysisEndpointBox.Text.StartsWith("https://api.openai.com", StringComparison.OrdinalIgnoreCase)
                    || AnalysisEndpointBox.Text == "http://127.0.0.1:1234/v1")
                {
                    AnalysisEndpointBox.Text = "";
                }
                break;
        }
        _analysisProviderReady = null;
        _analysisProviderVerified = false;
        _analysisProviderReadinessMessage = "";
        ResetAnalysisModelVerification();
        AnalysisProviderStatusText.Text = "Provider settings changed; run the check before a large job.";
        UpdateAnalysisProviderUi();
        UpdateSetupSummary();
    }

    private void AnalysisEndpoint_TextChanged(object sender, TextChangedEventArgs e)
    {
        if (!_loadingSettings)
        {
            _analysisProviderReady = null;
            _analysisProviderVerified = false;
            ResetAnalysisModelVerification();
            UpdateAnalysisProviderUi();
            UpdateSetupSummary();
        }
    }

    private void AllowExternalAnalysis_Toggled(object sender, RoutedEventArgs e)
    {
        _analysisProviderReady = null;
        _analysisProviderVerified = false;
        ResetAnalysisModelVerification();
        UpdateAnalysisProviderUi();
        UpdateSetupSummary();
    }

    private bool SelectedAnalysisProviderIsExternal()
    {
        var provider = SelectedComboValue(AnalysisProviderComboBox, "local");
        if (provider is "openai-responses" or "codex-cli")
        {
            return true;
        }
        if (provider != "openai-compatible")
        {
            return false;
        }
        return !Uri.TryCreate(AnalysisEndpointBox.Text.Trim(), UriKind.Absolute, out var endpoint)
            || !endpoint.IsLoopback;
    }

    private string SelectedAnalysisProviderDisplayName() =>
        SelectedComboValue(AnalysisProviderComboBox, "local") switch
        {
            "openai-responses" => "OpenAI",
            "openai-compatible" => "the compatible model endpoint",
            "codex-cli" => "Codex CLI",
            _ => "local Gemma",
        };

    private void UpdateAnalysisProviderUi()
    {
        if (AnalysisProviderInfoBar is null)
        {
            return;
        }
        var provider = SelectedComboValue(AnalysisProviderComboBox, "local");
        var usesApiKey = provider is "openai-responses" or "openai-compatible";
        var usesCodex = provider == "codex-cli";
        var external = SelectedAnalysisProviderIsExternal();
        AnalysisEndpointBox.IsEnabled = !usesCodex;
        AnalysisApiKeyBox.IsEnabled = usesApiKey;
        AnalysisApiKeyEnvironmentBox.IsEnabled = usesApiKey;
        RememberAnalysisApiKeyCheckBox.IsEnabled = usesApiKey;
        CodexCliPathBox.IsEnabled = usesCodex;
        AllowExternalAnalysisToggle.IsEnabled = external;
        if (!external)
        {
            AnalysisProviderInfoBar.Severity = InfoBarSeverity.Success;
            AnalysisProviderInfoBar.Title = provider == "local"
                ? "Private local analysis"
                : "Local compatible endpoint";
            AnalysisProviderInfoBar.Message = "Transcript text stays on this computer.";
        }
        else if (!AllowExternalAnalysisToggle.IsOn)
        {
            AnalysisProviderInfoBar.Severity = InfoBarSeverity.Warning;
            AnalysisProviderInfoBar.Title = "External analysis is off";
            AnalysisProviderInfoBar.Message =
                "This provider cannot receive transcript excerpts until you explicitly allow it.";
        }
        else
        {
            AnalysisProviderInfoBar.Severity = InfoBarSeverity.Warning;
            AnalysisProviderInfoBar.Title = "External transcript sharing allowed";
            AnalysisProviderInfoBar.Message =
                "Only analysis prompts are sent—not raw audio—but radio text may contain sensitive or unverified details.";
        }
    }

    private T ApplyAnalysisProvider<T>(T request) where T : AnalysisProviderRequest
    {
        var provider = SelectedComboValue(AnalysisProviderComboBox, "local");
        request.AnalysisProvider = provider;
        request.AnalysisModel = AnalysisModelBox.Text.Trim();
        request.AnalysisDevice = SelectedComboValue(AnalysisDeviceComboBox, "auto");
        request.AnalysisEndpoint = AnalysisEndpointBox.Text.Trim();
        request.AnalysisApiKey = provider is "openai-responses" or "openai-compatible"
            && !string.IsNullOrWhiteSpace(AnalysisApiKeyBox.Password)
                ? AnalysisApiKeyBox.Password
                : null;
        request.AnalysisApiKeyEnvironment = string.IsNullOrWhiteSpace(AnalysisApiKeyEnvironmentBox.Text)
            ? "OPENAI_API_KEY"
            : AnalysisApiKeyEnvironmentBox.Text.Trim();
        request.CodexCliPath = CodexCliPathBox.Text.Trim();
        request.AllowExternalAnalysis = SelectedAnalysisProviderIsExternal()
            && AllowExternalAnalysisToggle.IsOn;
        return request;
    }

    private async void CheckAnalysisProvider_Click(object sender, RoutedEventArgs e)
    {
        if (_worker is null)
        {
            return;
        }
        AnalysisProviderCheckButton.IsEnabled = false;
        AnalysisProviderStatusText.Text = "Checking configuration without sending transcript text…";
        try
        {
            using var cancellation = new CancellationTokenSource(TimeSpan.FromSeconds(20));
            var status = await _worker.GetAnalysisProviderDiagnosticsAsync(
                ApplyAnalysisProvider(new AnalysisProviderDiagnosticsRequest()),
                cancellation.Token);
            if (status is null)
            {
                AnalysisProviderStatusText.Text = "The provider check returned no status.";
                return;
            }
            var verification = status.Verified ? "Verified" : status.Ready ? "Configured" : "Setup needed";
            AnalysisProviderStatusText.Text = $"{verification}: {status.Message}";
            _analysisProviderReady = status.Ready;
            _analysisProviderVerified = status.Verified;
            _analysisProviderReadinessMessage = status.Message;
            UpdateSetupSummary();
        }
        catch (Exception exception)
        {
            AnalysisProviderStatusText.Text = exception.Message;
            _analysisProviderReady = false;
            _analysisProviderVerified = false;
            _analysisProviderReadinessMessage = exception.Message;
            UpdateSetupSummary();
        }
        finally
        {
            AnalysisProviderCheckButton.IsEnabled = true;
        }
    }

    private async void AnalysisSelfTest_Click(object sender, RoutedEventArgs e)
    {
        if (_worker is null || _operationCancellation is not null)
        {
            return;
        }
        _operationCancellation = new CancellationTokenSource();
        SetBusy(true, "Testing the selected analysis model…", jobRunning: true);
        JobProgress.IsIndeterminate = true;
        AnalysisSelfTestInfoBar.Severity = InfoBarSeverity.Informational;
        AnalysisSelfTestInfoBar.Title = "Loading and generating synthetic structured output";
        AnalysisSelfTestInfoBar.Message = SelectedAnalysisProviderIsExternal()
            ? "No archive text is sent, but the selected hosted provider may record minimal model usage."
            : "The selected local model may download once. No archive text or Broadcastify quota is used.";
        try
        {
            var status = await _worker.RunAnalysisSelfTestAsync(
                ApplyAnalysisProvider(new AnalysisSelfTestRequest()),
                HandleWorkerMessage,
                _operationCancellation.Token);
            if (status is null || !status.Ready || !status.Verified)
            {
                throw new InvalidOperationException(
                    "The analysis worker ended without a successful generation result.");
            }
            AnalysisSelfTestInfoBar.Severity = InfoBarSeverity.Success;
            AnalysisSelfTestInfoBar.Title = "Selected analysis model is ready";
            AnalysisSelfTestInfoBar.Message = status.Message;
            AnalysisProviderStatusText.Text = $"Verified: {status.Message}";
            _analysisProviderReady = true;
            _analysisModelVerifiedThisSession = true;
            _analysisModelVerificationMessage = status.Message;
            StatusText.Text = status.Message;
            UpdateSetupSummary();
        }
        catch (OperationCanceledException)
        {
            AnalysisSelfTestInfoBar.Severity = InfoBarSeverity.Warning;
            AnalysisSelfTestInfoBar.Title = "Analysis test cancelled";
            AnalysisSelfTestInfoBar.Message = "No archive work was changed.";
            _analysisModelVerifiedThisSession = false;
            StatusText.Text = "Analysis model test cancelled";
            UpdateSetupSummary();
        }
        catch (Exception exception)
        {
            AnalysisSelfTestInfoBar.Severity = InfoBarSeverity.Error;
            AnalysisSelfTestInfoBar.Title = "Selected analysis model needs setup";
            AnalysisSelfTestInfoBar.Message = exception.Message;
            AnalysisProviderStatusText.Text = exception.Message;
            _analysisModelVerifiedThisSession = false;
            _analysisModelVerificationMessage = exception.Message;
            StatusText.Text = exception.Message;
            UpdateSetupSummary();
            AppendLog(exception.Message);
        }
        finally
        {
            _operationCancellation.Dispose();
            _operationCancellation = null;
            JobProgress.IsIndeterminate = false;
            SetBusy(false);
        }
    }

    private void ClearAnalysisKey_Click(object sender, RoutedEventArgs e)
    {
        CredentialStore.ClearAnalysisKey();
        AnalysisApiKeyBox.Password = "";
        RememberAnalysisApiKeyCheckBox.IsChecked = false;
        AnalysisProviderStatusText.Text = "Saved analysis API key removed from Windows Credential Locker.";
    }

    private void ApplySelectedHardwareProfileDescription()
    {
        if (HardwareProfileDescriptionText is null
            || HardwareProfileComboBox?.SelectedItem is not ComboBoxItem item)
        {
            return;
        }
        var id = item.Tag?.ToString() ?? "auto";
        var profile = _hardwareProfiles.FirstOrDefault(value => value.Id == id);
        if (profile is null)
        {
            HardwareProfileDescriptionText.Text =
                "Run diagnostics to confirm the transcription, speaker-label, and analysis engines for this profile.";
            if (ProfileNextStepsInfoBar is not null)
            {
                ProfileNextStepsInfoBar.Severity = InfoBarSeverity.Informational;
                ProfileNextStepsInfoBar.Title = "Hardware check needed";
                ProfileNextStepsInfoBar.Message =
                    "Refresh the check to build a stage-by-stage setup plan for this profile.";
            }
            return;
        }
        HardwareProfileDescriptionText.Text =
            string.IsNullOrWhiteSpace(profile.Note)
                ? $"{profile.Name}: {profile.StateText.ToLowerInvariant()}."
                : profile.Note;
        UpdateProfileReadinessBanner(profile);
    }

    private bool SelectedProfileVerifiedThisSession() =>
        _asrVerifiedThisSession
        && _diarizationVerifiedThisSession
        && _analysisModelVerifiedThisSession;

    private void UpdateProfileReadinessBanner(HardwareProfileStatus? profile = null)
    {
        if (DiagnosticsInfoBar is null || ProfileNextStepsInfoBar is null)
        {
            return;
        }
        profile ??= SelectedHardwareProfile();
        if (profile is null)
        {
            return;
        }

        var verified = profile.Configured && SelectedProfileVerifiedThisSession();
        UpdateHardwareProfileVerification(profile.Id, verified);
        DiagnosticsInfoBar.Severity = verified
            ? InfoBarSeverity.Success
            : profile.Configured
                ? InfoBarSeverity.Informational
                : InfoBarSeverity.Warning;
        DiagnosticsInfoBar.Title = verified
            ? $"{profile.Name} is verified for this session"
            : profile.Configured
                ? $"{profile.Name} is detected"
                : $"{profile.Name} needs setup";
        DiagnosticsInfoBar.Message = verified
            ? $"Transcription, speaker labels, and analysis executed successfully. "
              + $"Transcription: {profile.Transcription} · Speakers: {profile.Diarization} · "
              + $"Analysis: {profile.Analysis}"
            : $"Transcription: {profile.Transcription} · Speakers: {profile.Diarization} · "
              + $"Analysis: {profile.Analysis}";

        ProfileNextStepsInfoBar.Severity = verified
            ? InfoBarSeverity.Success
            : profile.Configured
                ? InfoBarSeverity.Informational
                : InfoBarSeverity.Warning;
        ProfileNextStepsInfoBar.Title = verified
            ? "All three model stages executed"
            : profile.Configured
                ? "Detected, not yet execution-verified"
                : "Complete these setup steps";
        ProfileNextStepsInfoBar.Message = verified
            ? "This proof is intentionally session-specific; changing an engine, model, or device requires a retest."
            : profile.NextSteps.Count > 0
                ? string.Join(" ", profile.NextSteps.Select((step, index) => $"{index + 1}. {step}"))
                : "Run the stage tests before a long unattended job.";
        UpdateProfileNextAction(profile.NextAction);
    }

    private void UpdateProfileNextAction(ProfileSetupAction? detectedAction)
    {
        if (ProfileNextActionButton is null)
        {
            return;
        }
        var action = _profileRecoveryAction ?? detectedAction;
        var show = action is not null
            && !string.IsNullOrWhiteSpace(action.Kind)
            && action.Kind != "verify-profile";
        ProfileNextActionButton.Visibility = show
            ? Visibility.Visible
            : Visibility.Collapsed;
        if (!show || action is null)
        {
            ProfileNextActionButton.Tag = null;
            UpdateAsrModelPreparationUi();
            return;
        }
        ProfileNextActionButton.Content = string.IsNullOrWhiteSpace(action.Label)
            ? "Complete setup"
            : action.Label;
        ProfileNextActionButton.Tag = action;
        ToolTipService.SetToolTip(ProfileNextActionButton, action.Message);
        UpdateAsrModelPreparationUi();
    }

    private void UpdateHardwareProfileVerification(
        string selectedProfileId,
        bool selectedProfileVerified)
    {
        for (var index = 0; index < _hardwareProfiles.Count; index++)
        {
            var profile = _hardwareProfiles[index];
            var verified = selectedProfileVerified && profile.Id == selectedProfileId;
            if (profile.Verified != verified)
            {
                _hardwareProfiles[index] = profile with { Verified = verified };
            }
        }
    }

    private HardwareProfileStatus? SelectedHardwareProfile()
    {
        var id = (HardwareProfileComboBox?.SelectedItem as ComboBoxItem)?.Tag?.ToString()
            ?? "auto";
        return _hardwareProfiles.FirstOrDefault(value => value.Id == id);
    }

    private void ProfileNextAction_Click(object sender, RoutedEventArgs e)
    {
        var action = _profileRecoveryAction
            ?? SelectedHardwareProfile()?.NextAction
            ?? ProfileNextActionButton.Tag as ProfileSetupAction;
        if (action is null)
        {
            return;
        }
        StatusText.Text = action.Message;
        ProfileNextStepsInfoBar.Severity = InfoBarSeverity.Warning;
        ProfileNextStepsInfoBar.Title = action.Label;
        ProfileNextStepsInfoBar.Message = action.Message;
        switch (action.Kind)
        {
            case "verify-profile":
                ProfileSelfTest_Click(sender, e);
                break;
            case "prepare-asr-model":
                SettingsTabView.SelectedItem = ProcessingSettingsTab;
                AsrPrepare_Click(sender, e);
                break;
            case "test-transcription":
                SettingsTabView.SelectedItem = ProcessingSettingsTab;
                AsrSelfTest_Click(sender, e);
                break;
            case "configure-speakers":
                SettingsTabView.SelectedItem = AccountSettingsTab;
                HuggingFaceTokenBox.Focus(FocusState.Programmatic);
                break;
            case "configure-analysis":
                SettingsTabView.SelectedItem = AnalysisSettingsTab;
                AnalysisProviderComboBox.Focus(FocusState.Programmatic);
                break;
            case "install-managed-cuda-runtime":
                SettingsTabView.SelectedItem = ProcessingSettingsTab;
                ManagedRuntimeInstall_Click(sender, e);
                break;
            default:
                SettingsTabView.SelectedItem = ProcessingSettingsTab;
                AdvancedProcessingExpander.IsExpanded = true;
                AsrEngineComboBox.Focus(FocusState.Programmatic);
                break;
        }
    }

    private void RefreshStorageReadiness()
    {
        try
        {
            var configured = string.IsNullOrWhiteSpace(OutputFolderBox?.Text)
                ? "archives"
                : OutputFolderBox.Text.Trim();
            var baseDirectory = _worker?.WorkingDirectory ?? Environment.CurrentDirectory;
            var path = Path.GetFullPath(configured, baseDirectory);
            Directory.CreateDirectory(path);
            var probe = Path.Combine(path, $".radio-archive-write-{Guid.NewGuid():N}.tmp");
            using (new FileStream(
                       probe,
                       FileMode.CreateNew,
                       FileAccess.Write,
                       FileShare.None,
                       bufferSize: 1,
                       FileOptions.DeleteOnClose))
            {
            }
            if (File.Exists(probe))
            {
                File.Delete(probe);
            }
            _storageReady = true;
            _storageReadinessMessage = $"Writable local library: {path}";
        }
        catch (Exception exception) when (exception is IOException
                                          or UnauthorizedAccessException
                                          or ArgumentException
                                          or NotSupportedException)
        {
            _storageReady = false;
            _storageReadinessMessage = exception.Message;
        }
        UpdateSetupSummary();
    }

    private void UpdateSetupSummary()
    {
        if (SetupReadinessInfoBar is null
            || HardwareProfileComboBox is null
            || AnalysisProviderComboBox is null
            || DiarizationEngineComboBox is null
            || DiarizationDeviceComboBox is null
            || HuggingFaceTokenBox is null)
        {
            return;
        }

        var profile = SelectedHardwareProfile();
        var profileAction = _profileRecoveryAction ?? profile?.NextAction;
        var tokenAvailable = _huggingFaceTokenConfigured
            || !string.IsNullOrWhiteSpace(CurrentHuggingFaceToken());
        var selectedDiarizationDevice = SelectedComboValue(DiarizationDeviceComboBox, "auto");
        var selectedDiarizationEngine = SelectedComboValue(
            DiarizationEngineComboBox, "community-1");
        var typedTokenMakesDiarizationAvailable = _pyannotePackageInstalled
            && (_pyannoteAccessConfigured || tokenAvailable)
            && (selectedDiarizationDevice != "cuda" || _cudaAvailable);
        var selectedEngineDetected = selectedDiarizationEngine == "sherpa-onnx"
            ? _portableDiarizationRuntimeInstalled && _portableDiarizationModelReady
            : typedTokenMakesDiarizationAvailable;
        var transcriptionAvailable = _asrVerifiedThisSession
            || profile?.TranscriptionReady == true;
        var diarizationAvailable = _diarizationVerifiedThisSession
            || selectedEngineDetected;
        var provider = SelectedComboValue(AnalysisProviderComboBox, "local");
        var detectedLocalAnalysis = provider == "local" && profile?.AnalysisReady == true;
        var analysisAvailable = _analysisProviderReady ?? detectedLocalAnalysis;

        SetupAccountStateText.Text = _archiveAccessVerified
            ? "Verified"
            : _archiveAccessConfigured
                ? "Configured"
                : "Needs sign-in";
        SetupAccountDetailText.Text = _archiveAccessVerified
            ? "The premium website session was refreshed successfully in this app session."
            : _archiveAccessConfigured
                ? "A saved website session or secure login is available; the next archive request can refresh it."
                : "Sign in with a premium Broadcastify website account before acquiring archives.";
        SetupAccountActionButton.Content = _archiveAccessConfigured ? "Account" : "Sign in";

        SetupStorageStateText.Text = _storageReady ? "Ready" : "Needs attention";
        SetupStorageDetailText.Text = _storageReadinessMessage;

        SetupTranscriptionStateText.Text = _asrVerifiedThisSession
            ? "Verified"
            : transcriptionAvailable
                ? "Detected"
                : _diagnosticsLoaded
                    ? "Setup needed"
                    : "Checking";
        SetupTranscriptionDetailText.Text = _asrVerifiedThisSession
            ? _asrVerificationMessage
            : !transcriptionAvailable && profileAction?.Stage == "transcription"
                ? profileAction.Message
            : profile?.Transcription
              ?? "Run the local hardware check to choose a transcription path.";
        SetupTranscriptionActionButton.Content = _asrVerifiedThisSession
            ? "Retest"
            : transcriptionAvailable
                ? "Test"
                : profileAction?.Stage == "transcription"
                    ? profileAction.Label
                    : "Review";

        SetupDiarizationStateText.Text = _diarizationVerifiedThisSession
            ? "Verified"
            : diarizationAvailable
                ? "Configured"
                : _diagnosticsLoaded
                    ? "Setup needed"
                    : "Checking";
        SetupDiarizationDetailText.Text = _diarizationVerifiedThisSession
            ? _diarizationVerificationMessage
            : !diarizationAvailable && profileAction?.Stage == "diarization"
                ? profileAction.Message
            : profile?.Diarization
              ?? "Run the local hardware check to inspect the selected speaker-label engine.";
        SetupDiarizationActionButton.Content = _diarizationVerifiedThisSession
            ? "Retest"
            : diarizationAvailable
                ? "Test"
                : profileAction?.Stage == "diarization"
                    ? profileAction.Label
                    : "Configure";

        SetupAnalysisStateText.Text = _analysisModelVerifiedThisSession
            ? "Verified"
            : analysisAvailable
                ? "Configured"
                : _diagnosticsLoaded
                    ? "Check provider"
                    : "Checking";
        SetupAnalysisDetailText.Text = _analysisModelVerifiedThisSession
            ? _analysisModelVerificationMessage
            : !string.IsNullOrWhiteSpace(_analysisProviderReadinessMessage)
                ? _analysisProviderReadinessMessage
            : !analysisAvailable && profileAction?.Stage == "analysis"
                ? profileAction.Message
            : detectedLocalAnalysis
                ? profile?.Analysis ?? "Local llama.cpp is detected."
                : $"Check {SelectedAnalysisProviderDisplayName()} without sending transcript text.";
        SetupAnalysisActionButton.Content = _analysisModelVerifiedThisSession
            ? "Retest"
            : analysisAvailable
                ? "Test"
                : profileAction?.Stage == "analysis"
                    ? profileAction.Label
                    : "Check";

        var available = new[]
        {
            _archiveAccessConfigured,
            _storageReady,
            transcriptionAvailable,
            diarizationAvailable,
            analysisAvailable,
        }.Count(value => value);
        var verified = new[]
        {
            _archiveAccessVerified,
            _asrVerifiedThisSession,
            _diarizationVerifiedThisSession,
            _analysisModelVerifiedThisSession,
        }.Count(value => value);
        var processingVerified = SelectedProfileVerifiedThisSession();
        SetupReadinessProgress.Value = available;
        SetupReadinessCountText.Text =
            $"{available} of 5 setup steps available · {verified} execution "
            + $"check{(verified == 1 ? "" : "s")} verified this session";
        SetupReadinessInfoBar.Severity = !_diagnosticsLoaded
            ? InfoBarSeverity.Informational
            : available == 5 && processingVerified
                ? InfoBarSeverity.Success
                : available == 5
                    ? InfoBarSeverity.Informational
                    : InfoBarSeverity.Warning;
        SetupReadinessInfoBar.Title = !_diagnosticsLoaded
            ? "Checking this computer"
            : available == 5 && processingVerified
                ? "The processing pipeline is verified for this session"
                : available == 5
                ? "This computer is configured for the full pipeline"
                : $"{5 - available} setup step{(available == 4 ? "" : "s")} need attention";
        SetupReadinessInfoBar.Message = available == 5 && processingVerified
            ? "The selected transcription, speaker-label, and analysis models all executed successfully."
            : available == 5
            ? "Detection is complete. Use the test actions to prove model execution before a long unattended run."
            : "Open the item that needs attention; no Broadcastify quota is used by these setup checks.";
        UpdateProfileReadinessBanner(profile);
    }

    private void SetupAccount_Click(object sender, RoutedEventArgs e)
    {
        SettingsTabView.SelectedItem = AccountSettingsTab;
        SignIn_Click(sender, e);
    }

    private void SetupStorage_Click(object sender, RoutedEventArgs e)
    {
        SettingsTabView.SelectedItem = ProcessingSettingsTab;
        OutputFolderBox.Focus(FocusState.Programmatic);
    }

    private void SetupTranscription_Click(object sender, RoutedEventArgs e)
    {
        if (SelectedHardwareProfile()?.TranscriptionReady == true || _asrVerifiedThisSession)
        {
            AsrSelfTest_Click(sender, e);
            return;
        }
        var action = _profileRecoveryAction ?? SelectedHardwareProfile()?.NextAction;
        if (action?.Stage == "transcription")
        {
            ProfileNextAction_Click(sender, e);
            return;
        }
        SettingsTabView.SelectedItem = ProcessingSettingsTab;
        AsrEngineComboBox.Focus(FocusState.Programmatic);
    }

    private void SetupDiarization_Click(object sender, RoutedEventArgs e)
    {
        var selectedEngine = SelectedComboValue(
            DiarizationEngineComboBox, "community-1");
        var tokenAvailable = _huggingFaceTokenConfigured
            || !string.IsNullOrWhiteSpace(CurrentHuggingFaceToken());
        var canTest = selectedEngine == "sherpa-onnx"
            ? _portableDiarizationRuntimeInstalled
            : _pyannotePackageInstalled && (_pyannoteAccessConfigured || tokenAvailable);
        if (canTest)
        {
            DiarizationSelfTest_Click(sender, e);
            return;
        }
        var action = _profileRecoveryAction ?? SelectedHardwareProfile()?.NextAction;
        if (action?.Stage == "diarization")
        {
            ProfileNextAction_Click(sender, e);
            return;
        }
        SettingsTabView.SelectedItem = ProcessingSettingsTab;
        if (selectedEngine == "sherpa-onnx")
        {
            DiarizationEngineComboBox.Focus(FocusState.Programmatic);
        }
        else
        {
            SettingsTabView.SelectedItem = AccountSettingsTab;
            HuggingFaceTokenBox.Focus(FocusState.Programmatic);
        }
    }

    private void SetupAnalysis_Click(object sender, RoutedEventArgs e)
    {
        SettingsTabView.SelectedItem = AnalysisSettingsTab;
        var localDetected = SelectedComboValue(AnalysisProviderComboBox, "local") == "local"
            && SelectedHardwareProfile()?.AnalysisReady == true;
        if (_analysisModelVerifiedThisSession || _analysisProviderReady == true || localDetected)
        {
            AnalysisSelfTest_Click(sender, e);
            return;
        }
        var action = _profileRecoveryAction ?? SelectedHardwareProfile()?.NextAction;
        if (action?.Stage == "analysis")
        {
            ProfileNextAction_Click(sender, e);
            return;
        }
        CheckAnalysisProvider_Click(sender, e);
    }

    private void HuggingFaceToken_Changed(object sender, RoutedEventArgs e)
    {
        ResetAsrVerification();
        ResetDiarizationVerification();
        UpdateSetupSummary();
    }

    private string? CurrentHuggingFaceToken()
    {
        if (!string.IsNullOrWhiteSpace(HuggingFaceTokenBox?.Password))
        {
            return HuggingFaceTokenBox.Password.Trim();
        }
        return string.IsNullOrWhiteSpace(_savedHuggingFaceToken?.Secret)
            ? null
            : _savedHuggingFaceToken.Secret;
    }

    private void RefreshHuggingFaceCredentialUi()
    {
        if (HuggingFaceCredentialStatusText is null
            || HuggingFaceProcessingInfoBar is null)
        {
            return;
        }
        var saved = _savedHuggingFaceToken;
        var configured = saved is not null || _huggingFaceTokenConfigured;
        var preview = saved is null
            ? ""
            : CredentialStore.CreateSecretPreview(saved.Secret, 8);
        HuggingFaceCredentialStatusText.Text = saved is not null
            ? $"Saved read token {preview} in Windows Credential Locker for this Windows account."
            : _huggingFaceTokenConfigured
                ? "A Hugging Face token is configured in the private environment. Its value is not shown or copied into settings."
                : "No saved read token. Create one and accept the Community-1 terms before its first download.";
        ClearHuggingFaceTokenButton.IsEnabled = saved is not null;
        HuggingFaceProcessingInfoBar.Severity = configured
            ? InfoBarSeverity.Success
            : InfoBarSeverity.Warning;
        HuggingFaceProcessingInfoBar.Title = configured
            ? "Hugging Face model access configured"
            : "Gated model access needs a token";
        HuggingFaceProcessingInfoBar.Message = saved is not null
            ? $"{preview} will be supplied automatically to local model jobs."
            : _huggingFaceTokenConfigured
                ? "The private environment token will be supplied to local model jobs."
                : "Open Credentials to save a read token securely.";
    }

    private void ManageHuggingFaceToken_Click(object sender, RoutedEventArgs e)
    {
        RootNavigation.SelectedItem = CredentialsNavigationItem;
        ShowPage("settings");
        SettingsTabView.SelectedItem = AccountSettingsTab;
        RefreshHuggingFaceCredentialUi();
        HuggingFaceTokenBox.Focus(FocusState.Programmatic);
    }

    private async void SaveHuggingFaceToken_Click(object sender, RoutedEventArgs e)
    {
        var token = HuggingFaceTokenBox.Password.Trim();
        if (string.IsNullOrWhiteSpace(token))
        {
            await ShowMessageAsync(
                "Hugging Face token required",
                "Enter a read token to replace the saved token. Leaving the field blank keeps the current saved token.");
            return;
        }
        if (!token.StartsWith("hf_", StringComparison.Ordinal))
        {
            await ShowMessageAsync(
                "Check the token",
                "Hugging Face user access tokens begin with hf_. Use the linked token page to create a read token.");
            return;
        }
        CredentialStore.SaveHuggingFaceToken(token);
        _savedHuggingFaceToken = new SavedSecret(token);
        HuggingFaceTokenBox.Password = "";
        RefreshHuggingFaceCredentialUi();
        ResetAsrVerification();
        ResetDiarizationVerification();
        UpdateSetupSummary();
        StatusText.Text = "Hugging Face token saved securely";
    }

    private void ClearHuggingFaceToken_Click(object sender, RoutedEventArgs e)
    {
        CredentialStore.ClearHuggingFaceToken();
        _savedHuggingFaceToken = null;
        HuggingFaceTokenBox.Password = "";
        RefreshHuggingFaceCredentialUi();
        ResetAsrVerification();
        ResetDiarizationVerification();
        UpdateSetupSummary();
        StatusText.Text = "Saved Hugging Face token removed";
    }

    private async void RefreshDiagnostics_Click(object sender, RoutedEventArgs e)
    {
        DiagnosticsRefreshButton.IsEnabled = false;
        DiagnosticsInfoBar.Severity = InfoBarSeverity.Informational;
        DiagnosticsInfoBar.Title = "Checking local engines";
        DiagnosticsInfoBar.Message = "Inspecting transcription, speaker-label, and analysis backends…";
        try
        {
            await LoadDiagnosticsAndDaysAsync();
        }
        finally
        {
            DiagnosticsRefreshButton.IsEnabled = _worker is not null;
        }
    }

    private string EffectiveAsrEngineForPreparation()
    {
        var engine = SelectedComboValue(AsrEngineComboBox, "auto");
        if (engine != "auto")
        {
            return engine;
        }
        return SelectedComboValue(DeviceComboBox, "auto") switch
        {
            "vulkan" => "whisper.cpp",
            "windows-ml" => "windows-ml",
            _ => "faster-whisper",
        };
    }

    private void EnsureSelectedAsrModelCompatibility()
    {
        if (_loadingSettings || AsrEngineComboBox is null || ModelComboBox is null)
        {
            return;
        }
        var engine = SelectedComboValue(AsrEngineComboBox, "auto");
        var model = SelectedComboValue(ModelComboBox, "turbo");
        if (engine == "qwen3-asr"
            && !string.Equals(
                model,
                "qwen3-asr-0.6b-int8",
                StringComparison.OrdinalIgnoreCase))
        {
            SelectComboValue(ModelComboBox, "qwen3-asr-0.6b-int8");
        }
        else if (engine != "qwen3-asr"
                 && string.Equals(
                     model,
                     "qwen3-asr-0.6b-int8",
                     StringComparison.OrdinalIgnoreCase))
        {
            SelectComboValue(ModelComboBox, "turbo");
        }
    }

    private void UpdateAsrModelPreparationUi()
    {
        if (AsrPrepareButton is null)
        {
            return;
        }
        var engine = EffectiveAsrEngineForPreparation();
        var nextAction = _profileRecoveryAction ?? SelectedHardwareProfile()?.NextAction;
        var runtimeBlocked = nextAction?.Stage == "transcription"
            && nextAction.Kind == "configure-transcription";
        AsrPrepareButton.Visibility =
            !runtimeBlocked
            && engine is "windows-ml" or "whisper.cpp" or "qwen3-asr"
                ? Visibility.Visible
                : Visibility.Collapsed;
        AsrPrepareButton.IsEnabled = true;
        if (engine == "windows-ml")
        {
            AsrPrepareButton.Content = "Build & test model";
            ToolTipService.SetToolTip(
                AsrPrepareButton,
                "Explicitly build the selected ONNX Runtime GenAI CPU model, save its managed path, then prove a local decode.");
        }
        else if (engine == "whisper.cpp")
        {
            AsrPrepareButton.Content = "Download & test model";
            ToolTipService.SetToolTip(
                AsrPrepareButton,
                "Explicitly download the selected public GGML model, save its managed path, then prove the configured whisper.cpp runtime.");
        }
        else if (engine == "qwen3-asr")
        {
            AsrPrepareButton.Content = "Download & test model";
            ToolTipService.SetToolTip(
                AsrPrepareButton,
                "Explicitly download and checksum-verify the Qwen3-ASR 0.6B INT8 model plus Silero VAD, save their managed path, then prove local CPU execution.");
        }
        if (runtimeBlocked && nextAction is not null)
        {
            ToolTipService.SetToolTip(AsrPrepareButton, nextAction.Message);
        }
    }

    private AsrSelfTestRequest CreateAsrSelfTestRequest() =>
        new()
        {
            Model = SelectedComboValue(ModelComboBox, "turbo"),
            AsrEngine = SelectedComboValue(AsrEngineComboBox, "auto"),
            Device = SelectedComboValue(DeviceComboBox, "auto"),
            DeviceIndex = RequiredInteger(GpuIndexBox.Value, 0),
            AsrModelPath = string.IsNullOrWhiteSpace(AsrModelPathBox.Text)
                ? null
                : AsrModelPathBox.Text.Trim(),
            DiarizationEngine = SelectedComboValue(
                DiarizationEngineComboBox, "community-1"),
            DiarizationDevice = SelectedComboValue(
                DiarizationDeviceComboBox, "auto"),
            BatchSize = RequiredInteger(BatchSizeBox.Value, 8),
            HuggingFaceToken = CurrentHuggingFaceToken(),
        };

    private void ApplyAsrSelfTestResult(AsrSelfTestStatus result)
    {
        AsrSelfTestInfoBar.Severity = InfoBarSeverity.Success;
        AsrSelfTestInfoBar.Title = "Selected transcription engine executed locally";
        AsrSelfTestInfoBar.Message = string.IsNullOrWhiteSpace(result.FallbackReason)
            ? result.Message
            : $"{result.Message} Fallback during {result.FallbackStage}: {result.FallbackReason}";
        _asrVerifiedThisSession = true;
        _asrVerificationMessage = AsrSelfTestInfoBar.Message;
        _profileRecoveryAction = null;
        UpdateSetupSummary();
    }

    private async void AsrSelfTest_Click(object sender, RoutedEventArgs e)
    {
        if (_worker is null || _operationCancellation is not null)
        {
            return;
        }
        _operationCancellation = new CancellationTokenSource();
        SetBusy(true, "Testing the selected transcription engine…", jobRunning: true);
        JobProgress.IsIndeterminate = true;
        AsrSelfTestInfoBar.Severity = InfoBarSeverity.Informational;
        AsrSelfTestInfoBar.Title = "Loading and decoding local synthetic audio";
        AsrSelfTestInfoBar.Message =
            "A missing managed model may download once. Raw archive audio is not used.";
        try
        {
            var result = await _worker.RunAsrSelfTestAsync(
                CreateAsrSelfTestRequest(),
                HandleWorkerMessage,
                _operationCancellation.Token);
            if (result is null || !result.Ready)
            {
                throw new InvalidOperationException(
                    "The transcription worker ended without a successful self-test result.");
            }
            ApplyAsrSelfTestResult(result);
        }
        catch (OperationCanceledException)
        {
            AsrSelfTestInfoBar.Severity = InfoBarSeverity.Warning;
            AsrSelfTestInfoBar.Title = "Transcription test cancelled";
            AsrSelfTestInfoBar.Message = "No archive work was changed.";
            _asrVerifiedThisSession = false;
            UpdateSetupSummary();
        }
        catch (Exception exception)
        {
            AsrSelfTestInfoBar.Severity = InfoBarSeverity.Error;
            AsrSelfTestInfoBar.Title = "Selected transcription engine needs setup";
            AsrSelfTestInfoBar.Message = exception.Message;
            _asrVerifiedThisSession = false;
            _asrVerificationMessage = exception.Message;
            UpdateSetupSummary();
            AppendLog(exception.Message);
        }
        finally
        {
            _operationCancellation.Dispose();
            _operationCancellation = null;
            JobProgress.IsIndeterminate = false;
            SetBusy(false);
        }
    }

    private async void AsrPrepare_Click(object sender, RoutedEventArgs e)
    {
        if (_worker is null || _operationCancellation is not null)
        {
            return;
        }
        _operationCancellation = new CancellationTokenSource();
        SetBusy(true, "Preparing the selected transcription model…", jobRunning: true);
        JobProgress.IsIndeterminate = true;
        AsrSelfTestInfoBar.Severity = InfoBarSeverity.Informational;
        AsrSelfTestInfoBar.Title = "Preparing the selected model";
        AsrSelfTestInfoBar.Message =
            "This explicit action may download model files. It does not use archive audio or Broadcastify quota.";
        try
        {
            var prepared = await _worker.PrepareAsrModelAsync(
                CreateAsrSelfTestRequest(),
                HandleWorkerMessage,
                _operationCancellation.Token);
            if (prepared is null || !prepared.Ready || string.IsNullOrWhiteSpace(prepared.Path))
            {
                throw new InvalidOperationException(
                    "The model worker ended without returning a usable managed path.");
            }
            AsrModelPathBox.Text = prepared.Path;
            PersistUserSettings();
            AsrSelfTestInfoBar.Title = prepared.Reused
                ? "Matching model found; proving a decode"
                : "Model prepared; proving a decode";
            AsrSelfTestInfoBar.Message = prepared.Message;
            var verified = await _worker.RunAsrSelfTestAsync(
                CreateAsrSelfTestRequest(),
                HandleWorkerMessage,
                _operationCancellation.Token);
            if (verified is null || !verified.Ready)
            {
                throw new InvalidOperationException(
                    "The model was prepared, but its local decode did not return a successful result.");
            }
            ApplyAsrSelfTestResult(verified);
        }
        catch (OperationCanceledException)
        {
            AsrSelfTestInfoBar.Severity = InfoBarSeverity.Warning;
            AsrSelfTestInfoBar.Title = "Model preparation cancelled";
            AsrSelfTestInfoBar.Message =
                "No archive work was changed. Completed model-cache downloads remain reusable.";
            _asrVerifiedThisSession = false;
            UpdateSetupSummary();
        }
        catch (Exception exception)
        {
            AsrSelfTestInfoBar.Severity = InfoBarSeverity.Error;
            AsrSelfTestInfoBar.Title = "Selected model needs setup";
            AsrSelfTestInfoBar.Message = exception.Message;
            _asrVerifiedThisSession = false;
            _asrVerificationMessage = exception.Message;
            UpdateSetupSummary();
            AppendLog(exception.Message);
        }
        finally
        {
            _operationCancellation.Dispose();
            _operationCancellation = null;
            JobProgress.IsIndeterminate = false;
            SetBusy(false);
        }
    }

    private async void DiarizationSelfTest_Click(object sender, RoutedEventArgs e)
    {
        if (_worker is null || _operationCancellation is not null)
        {
            return;
        }
        _operationCancellation = new CancellationTokenSource();
        SetBusy(true, "Testing the selected speaker-label engine…", jobRunning: true);
        JobProgress.IsIndeterminate = true;
        DiarizationSelfTestInfoBar.Severity = InfoBarSeverity.Informational;
        DiarizationSelfTestInfoBar.Title = "Loading and executing the local speaker-label model";
        DiarizationSelfTestInfoBar.Message =
            "The first explicit test may download the selected pinned model. Archive audio and quota are not used.";
        try
        {
            var result = await _worker.RunDiarizationSelfTestAsync(
                new DiarizationSelfTestRequest
                {
                    DiarizationEngine = SelectedComboValue(
                        DiarizationEngineComboBox, "community-1"),
                    DiarizationDevice = SelectedComboValue(DiarizationDeviceComboBox, "auto"),
                    DeviceIndex = RequiredInteger(GpuIndexBox.Value, 0),
                    BatchSize = RequiredInteger(BatchSizeBox.Value, 8),
                    MinimumSpeakers = OptionalPositiveInteger(MinimumSpeakersBox.Value),
                    MaximumSpeakers = OptionalPositiveInteger(MaximumSpeakersBox.Value),
                    HuggingFaceToken = CurrentHuggingFaceToken(),
                },
                HandleWorkerMessage,
                _operationCancellation.Token);
            if (result is null || !result.Ready)
            {
                throw new InvalidOperationException(
                    "The speaker-label worker ended without a successful self-test result.");
            }
            DiarizationSelfTestInfoBar.Severity = InfoBarSeverity.Success;
            DiarizationSelfTestInfoBar.Title = "Selected speaker-label engine is ready";
            DiarizationSelfTestInfoBar.Message = result.Message;
            _diarizationVerifiedThisSession = true;
            _diarizationVerificationMessage = result.Message;
            _profileRecoveryAction = null;
            UpdateSetupSummary();
        }
        catch (OperationCanceledException)
        {
            DiarizationSelfTestInfoBar.Severity = InfoBarSeverity.Warning;
            DiarizationSelfTestInfoBar.Title = "Speaker-label test cancelled";
            DiarizationSelfTestInfoBar.Message = "No archive work was changed.";
            _diarizationVerifiedThisSession = false;
            UpdateSetupSummary();
        }
        catch (Exception exception)
        {
            DiarizationSelfTestInfoBar.Severity = InfoBarSeverity.Error;
            DiarizationSelfTestInfoBar.Title = "Selected speaker-label engine needs setup";
            DiarizationSelfTestInfoBar.Message = exception.Message;
            _diarizationVerifiedThisSession = false;
            _diarizationVerificationMessage = exception.Message;
            UpdateSetupSummary();
            AppendLog(exception.Message);
        }
        finally
        {
            _operationCancellation.Dispose();
            _operationCancellation = null;
            JobProgress.IsIndeterminate = false;
            SetBusy(false);
        }
    }

    private LocalProcessingRequest CreateProfileSelfTestRequest() =>
        ApplyAnalysisProvider(new LocalProcessingRequest
        {
            Model = SelectedComboValue(ModelComboBox, "turbo"),
            AsrEngine = SelectedComboValue(AsrEngineComboBox, "auto"),
            Device = SelectedComboValue(DeviceComboBox, "auto"),
            DeviceIndex = RequiredInteger(GpuIndexBox.Value, 0),
            AsrModelPath = string.IsNullOrWhiteSpace(AsrModelPathBox.Text)
                ? null
                : AsrModelPathBox.Text.Trim(),
            DiarizationEngine = SelectedComboValue(
                DiarizationEngineComboBox, "community-1"),
            DiarizationDevice = SelectedComboValue(DiarizationDeviceComboBox, "auto"),
            BatchSize = RequiredInteger(BatchSizeBox.Value, 8),
            HuggingFaceToken = CurrentHuggingFaceToken(),
        });

    private void ApplyProfileSelfTestResults(ProfileSelfTestStatus status)
    {
        if (status.Results.Transcription is { Ready: true } transcription)
        {
            _asrVerifiedThisSession = true;
            _asrVerificationMessage = transcription.Message;
            AsrSelfTestInfoBar.Severity = InfoBarSeverity.Success;
            AsrSelfTestInfoBar.Title = "Selected transcription engine executed locally";
            AsrSelfTestInfoBar.Message = string.IsNullOrWhiteSpace(transcription.FallbackReason)
                ? transcription.Message
                : $"{transcription.Message} Fallback during {transcription.FallbackStage}: "
                  + transcription.FallbackReason;
        }
        if (status.Results.Diarization is { Ready: true } diarization)
        {
            _diarizationVerifiedThisSession = true;
            _diarizationVerificationMessage = diarization.Message;
            DiarizationSelfTestInfoBar.Severity = InfoBarSeverity.Success;
            DiarizationSelfTestInfoBar.Title = "Selected speaker-label engine is ready";
            DiarizationSelfTestInfoBar.Message = diarization.Message;
        }
        if (status.Results.Analysis is { Ready: true, Verified: true } analysis)
        {
            _analysisProviderReady = true;
            _analysisProviderVerified = true;
            _analysisModelVerifiedThisSession = true;
            _analysisModelVerificationMessage = analysis.Message;
            AnalysisProviderStatusText.Text = $"Verified: {analysis.Message}";
            AnalysisSelfTestInfoBar.Severity = InfoBarSeverity.Success;
            AnalysisSelfTestInfoBar.Title = "Selected analysis model is ready";
            AnalysisSelfTestInfoBar.Message = analysis.Message;
        }
    }

    private void ApplyProfileSelfTestFailure(ProfileSelfTestStatus status)
    {
        _profileRecoveryAction = status.Recovery;
        switch (status.FailedStage)
        {
            case "transcription":
                AsrSelfTestInfoBar.Severity = InfoBarSeverity.Error;
                AsrSelfTestInfoBar.Title = "Selected transcription engine needs setup";
                AsrSelfTestInfoBar.Message = status.Message;
                break;
            case "diarization":
                DiarizationSelfTestInfoBar.Severity = InfoBarSeverity.Error;
                DiarizationSelfTestInfoBar.Title = "Selected speaker-label engine needs setup";
                DiarizationSelfTestInfoBar.Message = status.Message;
                break;
            case "analysis":
                AnalysisSelfTestInfoBar.Severity = InfoBarSeverity.Error;
                AnalysisSelfTestInfoBar.Title = "Selected analysis model needs setup";
                AnalysisSelfTestInfoBar.Message = status.Message;
                SettingsTabView.SelectedItem = AnalysisSettingsTab;
                break;
        }
    }

    private async void ProfileSelfTest_Click(object sender, RoutedEventArgs e)
    {
        if (_worker is null || _operationCancellation is not null)
        {
            return;
        }
        ResetAsrVerification();
        ResetDiarizationVerification();
        ResetAnalysisModelVerification();
        _operationCancellation = new CancellationTokenSource();
        SetBusy(true, "Verifying transcription, speaker labels, and analysis…", jobRunning: true);
        JobProgress.IsIndeterminate = true;
        SetupReadinessInfoBar.Severity = InfoBarSeverity.Informational;
        SetupReadinessInfoBar.Title = "Verifying all three model stages";
        SetupReadinessInfoBar.Message =
            "Generated input is used in sequence; archive audio and quota are not used.";
        ProfileNextStepsInfoBar.Severity = InfoBarSeverity.Informational;
        ProfileNextStepsInfoBar.Title = "Running all three execution checks";
        ProfileNextStepsInfoBar.Message =
            "The sequence uses generated input and stops at the first stage that needs setup. "
            + "Archive audio and quota are not used.";
        try
        {
            var status = await _worker.RunProfileSelfTestAsync(
                CreateProfileSelfTestRequest(),
                HandleWorkerMessage,
                _operationCancellation.Token);
            if (status is null)
            {
                throw new InvalidOperationException(
                    "The profile worker ended without a verification summary.");
            }
            ApplyProfileSelfTestResults(status);
            if (!status.Ready || !status.Verified)
            {
                ApplyProfileSelfTestFailure(status);
                ProfileNextStepsInfoBar.Severity = InfoBarSeverity.Warning;
                ProfileNextStepsInfoBar.Title = string.IsNullOrWhiteSpace(status.FailedStage)
                    ? "Selected profile needs setup"
                    : $"{status.FailedStage.Replace('_', ' ')} needs setup";
                ProfileNextStepsInfoBar.Message = status.Message;
                StatusText.Text = status.Message;
                AppendLog(status.Message);
                UpdateSetupSummary();
                SetupReadinessInfoBar.Severity = InfoBarSeverity.Warning;
                SetupReadinessInfoBar.Title = string.IsNullOrWhiteSpace(status.FailedStage)
                    ? "Profile verification needs attention"
                    : $"{status.FailedStage.Replace('_', ' ')} needs attention";
                SetupReadinessInfoBar.Message = status.Message;
                return;
            }
            _profileRecoveryAction = null;
            ProfileNextStepsInfoBar.Severity = InfoBarSeverity.Success;
            ProfileNextStepsInfoBar.Title = "Selected profile verified";
            ProfileNextStepsInfoBar.Message = status.Message;
            StatusText.Text = status.Message;
            AppendLog(status.Message);
            UpdateSetupSummary();
        }
        catch (OperationCanceledException)
        {
            ProfileNextStepsInfoBar.Severity = InfoBarSeverity.Warning;
            ProfileNextStepsInfoBar.Title = "Profile verification cancelled";
            ProfileNextStepsInfoBar.Message = "No archive work was changed.";
            StatusText.Text = "Profile verification cancelled";
            UpdateSetupSummary();
            SetupReadinessInfoBar.Severity = InfoBarSeverity.Warning;
            SetupReadinessInfoBar.Title = "Profile verification cancelled";
            SetupReadinessInfoBar.Message = "No archive work was changed.";
        }
        catch (Exception exception)
        {
            ProfileNextStepsInfoBar.Severity = InfoBarSeverity.Error;
            ProfileNextStepsInfoBar.Title = "Profile verification could not finish";
            ProfileNextStepsInfoBar.Message = exception.Message;
            StatusText.Text = exception.Message;
            AppendLog(exception.Message);
            UpdateSetupSummary();
            SetupReadinessInfoBar.Severity = InfoBarSeverity.Error;
            SetupReadinessInfoBar.Title = "Profile verification could not finish";
            SetupReadinessInfoBar.Message = exception.Message;
        }
        finally
        {
            _operationCancellation.Dispose();
            _operationCancellation = null;
            JobProgress.IsIndeterminate = false;
            SetBusy(false);
        }
    }

    private async void RefreshLibrary_Click(object sender, RoutedEventArgs e) =>
        await RefreshLibraryAsync();

    private DateTimeOffset DefaultCatchUpStart(LibraryFeedCoverage feed)
    {
        var candidates = new List<DateTimeOffset>();
        if (DateTimeOffset.TryParse(feed.TargetStartDate, out var scheduledStart))
        {
            candidates.Add(scheduledStart);
        }
        foreach (var day in _libraryDays.Where(value => value.FeedId == feed.FeedId))
        {
            if (DateTimeOffset.TryParse(day.ArchiveDate, out var retainedDate))
            {
                candidates.Add(retainedDate);
            }
        }
        var today = new DateTimeOffset(DateTime.Today);
        return candidates.Count > 0
            ? candidates.Min().Date > today.Date ? today : candidates.Min().Date
            : today.AddDays(-1);
    }

    private async Task<LibraryCatchUpRange?> ShowLibraryCatchUpRangeAsync()
    {
        var feeds = _libraryFeeds.ToList();
        if (feeds.Count == 0)
        {
            await ShowMessageAsync(
                "No Library feed available",
                "Retain or schedule at least one feed before evaluating a catch-up range.");
            return null;
        }
        var feedCombo = new ComboBox
        {
            Header = "Feed",
            HorizontalAlignment = HorizontalAlignment.Stretch,
            DisplayMemberPath = "FeedLabel",
            ItemsSource = feeds,
            SelectedItem = feeds.FirstOrDefault(value =>
                    value.FeedId == _selectedLibraryDay?.FeedId)
                ?? LibraryFeedCoverageList.SelectedItem as LibraryFeedCoverage
                ?? feeds[0],
        };
        var today = new DateTimeOffset(DateTime.Today);
        var startPicker = new DatePicker
        {
            Header = "Catch up from",
            MaxYear = today,
        };
        var saveForResumeBox = new CheckBox
        {
            Content = "Keep this start date resumable through today until caught up (recommended)",
            IsChecked = true,
        };
        var recurringScheduleBox = new CheckBox
        {
            Content = "Also create or update a daily recurring catch-up from this date",
            IsChecked = false,
        };
        void ApplyFeedDefault()
        {
            if (feedCombo.SelectedItem is LibraryFeedCoverage feed)
            {
                if (feed.CatchUpSaved
                    && DateTimeOffset.TryParse(feed.CatchUpStartDate, out var savedStart))
                {
                    startPicker.Date = savedStart;
                    saveForResumeBox.IsChecked = true;
                }
                else
                {
                    startPicker.Date = DefaultCatchUpStart(feed);
                    saveForResumeBox.IsChecked = true;
                }
            }
        }
        ApplyFeedDefault();
        feedCombo.SelectionChanged += (_, _) => ApplyFeedDefault();
        var content = new StackPanel { Spacing = 12, MaxWidth = 520 };
        content.Children.Add(new TextBlock
        {
            Text =
                "Choose an existing feed and the earliest date you want covered. The local planner checks every calendar day from that date through today, queues only absent or unfinished days, and skips days already complete. This evaluation does not contact Broadcastify.",
            TextWrapping = TextWrapping.Wrap,
        });
        content.Children.Add(feedCombo);
        content.Children.Add(startPicker);
        content.Children.Add(new TextBlock
        {
            Text = $"Through today · {today:yyyy-MM-dd}",
            FontWeight = Microsoft.UI.Text.FontWeights.SemiBold,
        });
        content.Children.Add(saveForResumeBox);
        content.Children.Add(recurringScheduleBox);
        content.Children.Add(new TextBlock
        {
            Text =
                "When started, retained local stages run first and missing downloads stay sequential behind the persistent rolling request guard. The resumable one-time boundary automatically extends through the then-current day and clears after every day through current is complete. A recurring catch-up keeps the boundary and rechecks it at the saved daily schedule time; the next dialog confirms processing and timing. Uncheck resumable intent and evaluate to clear an existing one-time catch-up.",
            TextWrapping = TextWrapping.Wrap,
            Foreground = (Brush)Application.Current.Resources[
                "TextFillColorSecondaryBrush"],
        });
        var dialog = new ContentDialog
        {
            XamlRoot = ((FrameworkElement)Content).XamlRoot,
            Title = "Catch up missing feed days",
            Content = CreateDialogScrollContent(content),
            PrimaryButtonText = "Find missing days",
            CloseButtonText = "Cancel",
            DefaultButton = ContentDialogButton.Close,
        };
        if (await dialog.ShowAsync() != ContentDialogResult.Primary)
        {
            return null;
        }
        if (feedCombo.SelectedItem is not LibraryFeedCoverage selectedFeed)
        {
            await ShowMessageAsync(
                "Feed required",
                "Choose an existing feed and a start date.");
            return null;
        }
        var startDate = startPicker.Date;
        if (startDate.Date > today.Date)
        {
            await ShowMessageAsync(
                "Invalid catch-up start date",
                "The catch-up start date cannot be in the future.");
            return null;
        }
        return new LibraryCatchUpRange(
            selectedFeed,
            startDate,
            saveForResumeBox.IsChecked == true,
            recurringScheduleBox.IsChecked == true);
    }

    private async void CatchUpFeed_Click(object sender, RoutedEventArgs e)
    {
        if (_worker is null)
        {
            return;
        }
        var range = await ShowLibraryCatchUpRangeAsync();
        if (range is null)
        {
            return;
        }
        if (!range.SaveForResume && range.Feed.CatchUpSaved)
        {
            try
            {
                using var cancellation = new CancellationTokenSource(TimeSpan.FromMinutes(1));
                await _worker.DeleteLibraryCatchUpAsync(
                    range.Feed.FeedId,
                    cancellation.Token);
            }
            catch (Exception exception)
            {
                await ShowErrorAsync(exception);
                return;
            }
        }
        if (range.CreateRecurringSchedule)
        {
            try
            {
                var existing = (await _worker.ListFeedSchedulesAsync(
                    CancellationToken.None))
                    .FirstOrDefault(value => value.FeedId == range.Feed.FeedId);
                if (!await ShowFeedScheduleEditorAsync(
                        range.Feed.FeedId,
                        range.Feed.FeedName,
                        existing,
                        range.StartDate,
                        suggestRecurringCatchUp: true))
                {
                    return;
                }
                await RefreshFeedScheduleStatusAsync();
            }
            catch (Exception exception)
            {
                await ShowErrorAsync(exception);
                return;
            }
        }
        LibraryResumePlan plan;
        try
        {
            using var cancellation = new CancellationTokenSource(TimeSpan.FromMinutes(2));
            plan = await _worker.GetLibraryResumePlanAsync(
                PersistedOutputDirectory(),
                cancellation.Token,
                range.Feed.FeedId,
                range.StartDate.ToString("yyyy-MM-dd"),
                throughCurrent: true);
        }
        catch (Exception exception)
        {
            await ShowErrorAsync(exception);
            return;
        }
        await RunLibraryResumePlanAsync(plan, range);
    }

    private async Task<ResumeAllSelection?> ShowResumeAllOptionsAsync(
        LibraryResumePlan plan)
    {
        var localBox = new CheckBox
        {
            Content = $"Finish retained local processing ({plan.LocalCount:N0} day(s))",
            IsChecked = plan.LocalCount > 0,
            IsEnabled = plan.LocalCount > 0,
        };
        var networkBox = new CheckBox
        {
            Content = $"Check/download missing source audio now ({plan.NetworkCount:N0} day(s))",
            IsChecked = plan.NetworkCount > 0,
            IsEnabled = plan.NetworkCount > 0,
        };
        var priorityCombo = new ComboBox
        {
            Header = "Order selected work by",
            HorizontalAlignment = HorizontalAlignment.Stretch,
            SelectedIndex = 0,
            Items =
            {
                new ComboBoxItem { Content = "Local work first (recommended)", Tag = "local-first" },
                new ComboBoxItem { Content = "A chosen feed first", Tag = "feed-first" },
                new ComboBoxItem { Content = "Newest days first", Tag = "newest" },
                new ComboBoxItem { Content = "Oldest days first", Tag = "oldest" },
            },
        };
        var candidateFeedIds = plan.Days
            .Select(value => value.FeedId)
            .ToHashSet(StringComparer.Ordinal);
        var priorityFeeds = plan.Feeds
            .Where(value => candidateFeedIds.Contains(value.FeedId))
            .ToList();
        var priorityFeedCombo = new ComboBox
        {
            Header = "Feed to put first",
            HorizontalAlignment = HorizontalAlignment.Stretch,
            DisplayMemberPath = "FeedLabel",
            ItemsSource = priorityFeeds,
            SelectedItem = priorityFeeds.FirstOrDefault(value =>
                    value.FeedId == _selectedLibraryDay?.FeedId)
                ?? priorityFeeds.FirstOrDefault(),
        };
        priorityFeedCombo.IsEnabled = false;
        priorityCombo.SelectionChanged += (_, _) =>
        {
            priorityFeedCombo.IsEnabled =
                (priorityCombo.SelectedItem as ComboBoxItem)?.Tag?.ToString()
                    == "feed-first";
        };

        var feedChecks = new Dictionary<string, CheckBox>(StringComparer.Ordinal);
        var feedsPanel = new StackPanel { Spacing = 6 };
        foreach (var feed in priorityFeeds)
        {
            var feedDays = plan.Days.Where(value => value.FeedId == feed.FeedId).ToList();
            var local = feedDays.Count(value => value.NeedsLocalProcessing);
            var network = feedDays.Count(value => value.NeedsNetwork);
            var check = new CheckBox
            {
                Content = $"{feed.FeedName} · {local:N0} local / {network:N0} source-network day(s)",
                IsChecked = true,
            };
            feedChecks[feed.FeedId] = check;
            feedsPanel.Children.Add(check);
        }
        var quotaText = plan.NetworkCount == 0
            ? "No source or archive request is needed by this plan."
            : plan.Quota.Available
                ? $"The persistent ledger currently has {plan.Quota.Remaining:N0} guarded archive request(s) available. Each missing media block—not each day—uses one."
                : "The rolling archive allowance is paused. Local work can still run; selected network work will stop before making a request.";
        var content = new StackPanel { Spacing = 12, MaxWidth = 560 };
        if (!string.IsNullOrWhiteSpace(plan.ScopeFeedId))
        {
            var scopedFeed = plan.Feeds.FirstOrDefault(value =>
                value.FeedId == plan.ScopeFeedId);
            content.Children.Add(new InfoBar
            {
                IsOpen = true,
                IsClosable = false,
                Severity = InfoBarSeverity.Informational,
                Title = $"Evaluated {scopedFeed?.TargetDayCount ?? 0:N0} calendar day(s)",
                Message =
                    $"{scopedFeed?.FeedName ?? $"Feed {plan.ScopeFeedId}"} · "
                    + $"{plan.ScopeStartDate} through {plan.ScopeEndDate} · "
                    + $"{scopedFeed?.ReadyDayCount ?? 0:N0} already ready · "
                    + $"{plan.Days.Count:N0} pending. Completed days are omitted from the work queue, not from evaluation.",
            });
        }
        content.Children.Add(new TextBlock
        {
            Text = $"Choose which feeds and work types to run. A day can appear in both counts when retained local stages are ready but today's source listing is due for refresh. Planning used only local state and did not contact Broadcastify. {quotaText}",
            TextWrapping = TextWrapping.Wrap,
        });
        content.Children.Add(localBox);
        content.Children.Add(networkBox);
        content.Children.Add(priorityCombo);
        content.Children.Add(priorityFeedCombo);
        content.Children.Add(new TextBlock
        {
            Text = "Feeds",
            FontWeight = Microsoft.UI.Text.FontWeights.SemiBold,
        });
        content.Children.Add(feedsPanel);
        var dialog = new ContentDialog
        {
            XamlRoot = ((FrameworkElement)Content).XamlRoot,
            Title = $"Resume and prioritize {plan.Days.Count:N0} pending day(s)",
            Content = CreateDialogScrollContent(content, 620),
            PrimaryButtonText = "Start selected work",
            CloseButtonText = "Cancel",
            DefaultButton = ContentDialogButton.Close,
        };
        if (await dialog.ShowAsync() != ContentDialogResult.Primary)
        {
            return null;
        }
        var selectedFeedIds = feedChecks
            .Where(value => value.Value.IsChecked == true)
            .Select(value => value.Key)
            .ToHashSet(StringComparer.Ordinal);
        var includeLocal = localBox.IsChecked == true;
        var includeNetwork = networkBox.IsChecked == true;
        var priorityMode =
            (priorityCombo.SelectedItem as ComboBoxItem)?.Tag?.ToString()
            ?? "local-first";
        var priorityFeedId =
            (priorityFeedCombo.SelectedItem as LibraryFeedCoverage)?.FeedId
            ?? "";
        IEnumerable<LibraryDay> selected = plan.Days
            .Where(value => selectedFeedIds.Contains(value.FeedId))
            .Where(value =>
                (includeLocal && value.NeedsLocalProcessing)
                || (includeNetwork && value.NeedsNetwork));
        selected = priorityMode switch
        {
            "feed-first" => selected
                .OrderBy(value => value.FeedId == priorityFeedId ? 0 : 1)
                .ThenBy(value => value.NeedsNetwork)
                .ThenByDescending(value => value.ArchiveDate),
            "newest" => selected
                .OrderByDescending(value => value.ArchiveDate)
                .ThenBy(value => value.FeedId),
            "oldest" => selected
                .OrderBy(value => value.ArchiveDate)
                .ThenBy(value => value.FeedId),
            _ => selected
                .OrderBy(value => !value.NeedsLocalProcessing)
                .ThenBy(value => value.NeedsNetwork)
                .ThenByDescending(value => value.ArchiveDate)
                .ThenBy(value => value.FeedId),
        };
        return new ResumeAllSelection(
            selected.ToList(),
            includeLocal,
            includeNetwork);
    }

    private async void ResumeAllLibrary_Click(object sender, RoutedEventArgs e)
    {
        if (_worker is null || _pipelineCancellation is not null)
        {
            return;
        }

        LibraryResumePlan plan;
        try
        {
            using var cancellation = new CancellationTokenSource(TimeSpan.FromMinutes(2));
            plan = await _worker.GetLibraryResumePlanAsync(
                PersistedOutputDirectory(),
                cancellation.Token);
        }
        catch (Exception exception)
        {
            await ShowErrorAsync(exception);
            return;
        }
        await RunLibraryResumePlanAsync(plan);
    }

    private async Task RunLibraryResumePlanAsync(
        LibraryResumePlan plan,
        LibraryCatchUpRange? catchUpRange = null)
    {
        var worker = _worker;
        if (worker is null)
        {
            return;
        }
        if (plan.Days.Count == 0)
        {
            if (catchUpRange?.Feed.CatchUpSaved == true)
            {
                try
                {
                    using var cancellation = new CancellationTokenSource(TimeSpan.FromMinutes(1));
                    await worker.FinalizeLibraryCatchUpsAsync(
                        PersistedOutputDirectory(),
                        cancellation.Token);
                    await RefreshLibraryAsync();
                }
                catch (Exception exception)
                {
                    AppendLog($"Catch-up finalization: {exception.Message}");
                }
            }
            await ShowMessageAsync(
                "Library is caught up",
                string.IsNullOrWhiteSpace(plan.ScopeFeedId)
                    ? "Every retained or scheduled feed-day is already ready to review."
                    : $"Feed {plan.ScopeFeedId} is already complete from "
                        + $"{plan.ScopeStartDate} through {plan.ScopeEndDate}.");
            return;
        }

        var selection = await ShowResumeAllOptionsAsync(plan);
        if (selection is null)
        {
            return;
        }
        var selectedDays = selection.Days;
        if (selectedDays.Count == 0)
        {
            await ShowMessageAsync(
                "Nothing selected",
                "Choose at least one feed and either local processing or guarded downloads.");
            return;
        }

        var minimumSpeakers = OptionalPositiveInteger(MinimumSpeakersBox.Value);
        var maximumSpeakers = OptionalPositiveInteger(MaximumSpeakersBox.Value);
        if (minimumSpeakers is not null
            && maximumSpeakers is not null
            && minimumSpeakers > maximumSpeakers)
        {
            await ShowMessageAsync(
                "Invalid speaker range",
                "Minimum speakers cannot exceed maximum speakers.");
            return;
        }
        if (catchUpRange?.SaveForResume == true)
        {
            try
            {
                using var cancellation = new CancellationTokenSource(TimeSpan.FromMinutes(1));
                await worker.SaveLibraryCatchUpAsync(
                    catchUpRange.Feed.FeedId,
                    catchUpRange.Feed.FeedName,
                    catchUpRange.StartDate.ToString("yyyy-MM-dd"),
                    throughCurrent: true,
                    cancellationToken: cancellation.Token);
                AppendLog(
                    $"Saved catch-up for {catchUpRange.Feed.FeedName}: "
                    + $"{catchUpRange.StartDate:yyyy-MM-dd} through today; "
                    + "the boundary will extend while work remains incomplete.");
            }
            catch (Exception exception)
            {
                await ShowErrorAsync(exception);
                return;
            }
        }

        if (_pipelineCancellation is not null)
        {
            var persisted = catchUpRange?.SaveForResume == true
                || catchUpRange?.CreateRecurringSchedule == true;
            await ShowMessageAsync(
                persisted
                    ? "Catch-up saved behind the active pipeline"
                    : "Background pipeline already running",
                persisted
                    ? "The selected full range is retained and will still extend through today. "
                        + "The current archive/transcription pipeline remains sequential; when it finishes, "
                        + "use Catch up missing feed days again or let its recurring schedule resume the saved range."
                    : "The range was evaluated without starting a concurrent download. "
                        + "Let the active pipeline finish, then start this catch-up again.");
            await RefreshLibraryAsync();
            return;
        }

        var pipeline = new CancellationTokenSource();
        _pipelineCancellation = pipeline;
        _activePipelineFeedIds.UnionWith(selectedDays.Select(value => value.FeedId));
        var attempted = 0;
        var pausedForQuota = false;
        SetPipelineBusy(true, "Resuming selected library work in the background…");
        JobProgress.IsIndeterminate = false;
        JobProgress.Maximum = Math.Max(1, selectedDays.Count);
        JobProgress.Value = 0;
        try
        {
            foreach (var day in selectedDays)
            {
                pipeline.Token.ThrowIfCancellationRequested();
                if (!DateTime.TryParse(day.ArchiveDate, out var archiveDate))
                {
                    throw new InvalidOperationException(
                        $"Could not parse library date {day.ArchiveDate}.");
                }
                var useNetwork = day.NeedsNetwork && selection.IncludeNetwork;
                StatusText.Text =
                    $"Resuming {attempted + 1:N0}/{selectedDays.Count:N0}: "
                    + $"{day.FeedName} · {day.ArchiveDate}";
                AppendLog(
                    $"Resume all: {day.FeedName} on {day.ArchiveDate} "
                    + (useNetwork ? "(guarded archive coverage)." : "(local only)."));
                var workDay = useNetwork
                    ? day
                    : day with { NeedsNetwork = false, SourceCheckDue = false };
                JobRunResult? result;
                if (useNetwork)
                {
                    var accountRun = await ContinueLibraryDayAcrossAccountsAsync(
                        workDay,
                        archiveDate.Date,
                        minimumSpeakers,
                        maximumSpeakers,
                        pipeline.Token);
                    result = accountRun.Result;
                    if (accountRun.WaitingForQuota)
                    {
                        pausedForQuota = true;
                        if (result is null
                            && selection.IncludeLocal
                            && day.NeedsLocalProcessing)
                        {
                            AppendLog(
                                $"Every eligible account is waiting for {day.FeedName} on "
                                + $"{day.ArchiveDate}; finishing its retained local stages now.");
                            result = await ContinueLibraryDayWorkAsync(
                                day with
                                {
                                    NeedsNetwork = false,
                                    SourceCheckDue = false,
                                },
                                archiveDate.Date,
                                minimumSpeakers,
                                maximumSpeakers,
                                forceAllStages: true,
                                cancellationToken: pipeline.Token);
                        }
                        else if (result is null)
                        {
                            AppendLog(
                                "Catch-up paused before the next network day because all "
                                + "authorized account ledgers are waiting for a safe request slot.");
                            break;
                        }
                    }
                }
                else
                {
                    result = await ContinueLibraryDayWorkAsync(
                        workDay,
                        archiveDate.Date,
                        minimumSpeakers,
                        maximumSpeakers,
                        forceAllStages: true,
                        cancellationToken: pipeline.Token);
                }
                attempted++;
                JobProgress.Maximum = Math.Max(1, selectedDays.Count);
                JobProgress.Value = attempted;
                if (result?.DownloadLimited == true)
                {
                    pausedForQuota = true;
                    AppendLog(
                        "Catch-up stopped after every eligible account reached its "
                        + "rolling request boundary; retained progress will be reused.");
                    break;
                }
            }
        }
        catch (OperationCanceledException)
        {
            StatusText.Text = "Resume all cancelled";
            AppendLog("Resume all cancelled; completed work and checkpoints remain saved.");
        }
        catch (Exception exception)
        {
            await ShowErrorAsync(exception);
        }
        finally
        {
            pipeline.Dispose();
            if (ReferenceEquals(_pipelineCancellation, pipeline))
            {
                _pipelineCancellation = null;
            }
            _activePipelineFeedIds.Clear();
            JobProgress.IsIndeterminate = false;
            SetPipelineBusy(false);
            try
            {
                using var finalization = new CancellationTokenSource(TimeSpan.FromMinutes(2));
                var completedCatchUps = await worker.FinalizeLibraryCatchUpsAsync(
                    PersistedOutputDirectory(),
                    finalization.Token);
                if (completedCatchUps.Count > 0)
                {
                    AppendLog(
                        "Completed saved catch-up range(s): "
                        + string.Join(", ", completedCatchUps));
                }
            }
            catch (Exception exception)
            {
                AppendLog($"Catch-up finalization: {exception.Message}");
            }
            await RefreshLibraryAsync();
            await RefreshAnalysisDaysAsync();
            await RefreshArchiveQuotaStatusAsync();
        }

        if (pausedForQuota)
        {
            await ShowMessageAsync(
                "Local work finished; archive work paused",
                $"Processed {attempted:N0} day(s). Remaining network work stayed "
                + "queued locally because the rolling request guard stopped it. "
                + "Use Resume all later; completed work will not repeat.");
        }
        else if (attempted == selectedDays.Count)
        {
            await ShowMessageAsync(
                "Resume all finished",
                $"Processed {attempted:N0} incomplete library day(s). Cached stages were reused.");
        }
    }

    private async Task RefreshLibraryAsync()
    {
        if (_worker is null)
        {
            return;
        }
        RefreshLibraryButton.IsEnabled = false;
        try
        {
            using var cancellation = new CancellationTokenSource(TimeSpan.FromMinutes(2));
            var result = await _worker.ListLibraryAsync(
                string.IsNullOrWhiteSpace(OutputFolderBox.Text) ? "archives" : OutputFolderBox.Text.Trim(),
                cancellation.Token);
            _libraryDays.Clear();
            foreach (var day in result.Days)
            {
                _libraryDays.Add(day);
            }
            _libraryFeeds.Clear();
            foreach (var feed in result.Feeds)
            {
                _libraryFeeds.Add(feed);
            }
            LibraryFeedCountText.Text = result.Summary.FeedCount.ToString("N0");
            LibraryDayCountText.Text = result.Summary.DayCount.ToString("N0");
            LibraryAttentionCountText.Text = result.Summary.BacklogCount.ToString("N0");
            LibraryCompleteCountText.Text = result.Summary.CompleteCount.ToString("N0");
            LibraryBacklogSummaryText.Text = result.Summary.BacklogCount == 0
                ? "Every configured target day and retained processing stage is caught up. Source availability is based on the last authenticated listing snapshot; today's snapshot is refreshed at most every 30 minutes when resumed."
                : $"{result.Summary.BacklogCount:N0} feed-day(s) need work: "
                    + $"{result.Summary.MissingDayCount:N0} scheduled day(s) are absent and "
                    + $"{result.Summary.NetworkDayCount:N0} day(s) need a guarded source check or download. Local processing never spends archive requests.";
            SyncAnalysisFeedSelection();
            ApplyLibraryFilter();
            UpdateCommandAvailability();
        }
        catch (Exception exception)
        {
            AppendLog($"Library: {exception.Message}");
            LibraryEmptyText.Text = $"The local library could not be loaded: {exception.Message}";
            LibraryEmptyText.Visibility = Visibility.Visible;
        }
        finally
        {
            UpdateCommandAvailability();
        }
    }

    private void LibraryFilter_Changed(AutoSuggestBox sender, AutoSuggestBoxTextChangedEventArgs args) =>
        ApplyLibraryFilter();

    private void LibraryFilterCombo_SelectionChanged(object sender, SelectionChangedEventArgs e) =>
        ApplyLibraryFilter();

    private void ApplyLibraryFilter()
    {
        if (LibrarySearchBox is null || LibraryFilterCombo is null || LibraryEmptyText is null)
        {
            return;
        }
        var preferredFeedId = _selectedLibraryDay?.FeedId;
        var preferredDate = _selectedLibraryDay?.ArchiveDate;
        var query = LibrarySearchBox.Text.Trim();
        var filter = (LibraryFilterCombo.SelectedItem as ComboBoxItem)?.Tag?.ToString() ?? "all";
        var values = _libraryDays.Where(day =>
            (string.IsNullOrWhiteSpace(query)
             || day.FeedName.Contains(query, StringComparison.OrdinalIgnoreCase)
             || day.FeedId.Contains(query, StringComparison.OrdinalIgnoreCase)
             || day.ArchiveDate.Contains(query, StringComparison.OrdinalIgnoreCase))
            && (filter switch
            {
                "attention" => !day.IsComplete,
                "complete" => day.IsComplete,
                "network" => day.NeedsNetwork || day.SourceCheckDue,
                _ => true,
            }));
        _visibleLibraryDays.Clear();
        foreach (var day in values)
        {
            _visibleLibraryDays.Add(day);
        }
        LibraryEmptyText.Text = _libraryDays.Count == 0
            ? "No retained archive days were found. Start a new archive job to build the local library."
            : "No local archive days match this view.";
        LibraryEmptyText.Visibility = _visibleLibraryDays.Count == 0
            ? Visibility.Visible
            : Visibility.Collapsed;
        if (_visibleLibraryDays.Count == 0)
        {
            LibraryList.SelectedItem = null;
            ShowLibraryDetails(null);
            return;
        }
        LibraryList.SelectedItem = _visibleLibraryDays.FirstOrDefault(day =>
                day.FeedId == preferredFeedId && day.ArchiveDate == preferredDate)
            ?? _visibleLibraryDays[0];
    }

    private async void LibraryList_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        var day = LibraryList.SelectedItem as LibraryDay;
        var selectionVersion = ShowLibraryDetails(day);
        if (day is not null && !IsFeedPipelineBusy(day.FeedId))
        {
            await LoadLibraryTranscriptPreviewAsync(day, selectionVersion);
        }
    }

    private int ShowLibraryDetails(LibraryDay? day)
    {
        _selectedLibraryDay = day;
        var selectionVersion = ++_librarySelectionVersion;
        _libraryMediaPlayer.Pause();
        _libraryMediaPlayer.Source = null;

        LibraryDetailEmptyPanel.Visibility = day is null ? Visibility.Visible : Visibility.Collapsed;
        LibraryDetailPanel.Visibility = day is null ? Visibility.Collapsed : Visibility.Visible;
        if (day is null)
        {
            LibraryTranscriptPreviewText.Text = "";
            LibrarySourceCoverageText.Text = "";
            UpdateCommandAvailability();
            return selectionVersion;
        }

        LibraryDetailTitleText.Text = day.FeedName;
        LibraryDetailSubtitleText.Text = $"Feed {day.FeedId} · {day.ArchiveDate} · {day.StorageSummary}";
        LibraryDetailInfoBar.Severity = day.SourceCheckDue
            ? InfoBarSeverity.Informational
            : day.IsComplete
                ? InfoBarSeverity.Success
                : day.NeedsNetwork
                ? InfoBarSeverity.Warning
                : InfoBarSeverity.Informational;
        LibraryDetailInfoBar.Title = day.Status;
        LibraryDetailInfoBar.Message =
            $"{day.StatusDetail.TrimEnd('.', ' ')}. Next: {day.NextStep}.";
        LibraryDetailProgress.Value = day.PipelinePercent;
        LibraryDetailStatusText.Text = day.PipelineSummary;
        LibrarySourceCoverageText.Text = day.SourceCoverageSummary;

        LibraryDownloadStageText.Text = day.HasCombined
            ? day.SourceSnapshotComplete
                ? $"✓  1. Archive audio — {day.RetainedSourceCount:N0}/{day.KnownSourceCount:N0} blocks retained from the last source check"
                    + (day.SourceCheckDue ? "; today's listing is due for refresh" : "")
                : day.RawFileCount > 0
                ? $"✓  1. Archive audio — {day.RawFileCount:N0} source segments retained"
                : "✓  1. Archive audio — verified in the combined recording"
            : day.RawFileCount > 0
                ? $"!  1. Archive audio — {day.RawFileCount:N0} segments retained; coverage needs verification"
                : "○  1. Archive audio — missing; Broadcastify access is required";
        LibraryCombineStageText.Text = day.HasCombined
            ? "✓  2. Combine — one continuous day timeline is ready"
            : "○  2. Combine — waits for complete archive coverage";
        LibraryTranscriptStageText.Text = day.HasTranscript
            ? day.HasImportedTranscript
                ? $"✓  3. Transcription — {day.SegmentCount:N0} timestamped segments"
                : "→  3. Transcription — current file awaits database import"
            : day.HasStaleTranscript
                ? "→  3. Transcription — combined audio changed; previous results are hidden"
            : day.HasCombined
                ? "→  3. Transcription — ready to run locally"
                : "○  3. Transcription — waits for combined audio";
        LibraryDiarizationStageText.Text = day.HasDiarization
            ? day.SpeakerUpgradeAvailable
                ? "✓  4. Speaker labels — fast preview attached; Community-1 accuracy upgrade is available"
                : "✓  4. Speaker labels — Community-1 accuracy labels are attached"
            : day.HasTranscript
                ? "→  4. Speaker labels — ready to add without repeating transcription"
                : "○  4. Speaker labels — waits for a transcript";
        LibraryAnalysisStageText.Text = day.HasAnalysis
            ? $"✓  5. Event analysis — {day.IncidentCount:N0} incidents saved"
            : day.HasTranscript && !day.HasImportedTranscript
                ? "→  5. Event analysis — import the current transcript first"
            : day.HasStaleAnalysis
                ? "→  5. Event analysis — saved results need current evidence rules"
            : day.HasTranscript
                ? "→  5. Event analysis — local classification and summary remain"
                : "○  5. Event analysis — waits for a transcript";
        LibraryPathText.Text = $"Local folder: {day.DayDirectory}";

        LibraryDetailPrimaryButton.Content = day.PrimaryButtonLabel;
        LibraryDetailPrimaryButton.IsEnabled = _worker is not null && _operationCancellation is null;
        LibraryDetailReviewButton.IsEnabled = day.CanOpenReview && _operationCancellation is null;
        LibraryDetailReviewButton.Visibility = day.CanOpenReview
            && day.PrimaryAction != "open_review"
                ? Visibility.Visible
                : Visibility.Collapsed;
        LibrarySpeakerUpgradeButton.IsEnabled = day.SpeakerUpgradeAvailable
            && _worker is not null
            && _operationCancellation is null;
        LibrarySpeakerUpgradeButton.Visibility = day.SpeakerUpgradeAvailable
            ? Visibility.Visible
            : Visibility.Collapsed;
        LibraryCheckSourceButton.IsEnabled =
            _worker is not null && _operationCancellation is null;
        LibraryDeleteFeedButton.IsEnabled =
            _worker is not null && _operationCancellation is null;
        LibraryOpenFolderButton.IsEnabled = Directory.Exists(day.DayDirectory);
        LibraryOpenTranscriptButton.IsEnabled =
            !IsFeedPipelineBusy(day.FeedId)
            && day.HasTranscript
            && File.Exists(day.TranscriptPath);

        if (IsFeedPipelineBusy(day.FeedId))
        {
            LibraryAudioStatusText.Text =
                "This feed is being updated by the background pipeline. Playback and transcript-file opening are temporarily held to avoid Windows file locks; other feeds and Review & Ask remain usable.";
            LibraryAudioPlayer.IsEnabled = false;
            LibraryTranscriptPreviewText.Text =
                "Transcript preview is paused while this feed's retained files are being updated.";
        }
        else if (File.Exists(day.CombinedPath))
        {
            LibraryAudioStatusText.Text = "Combined day recording — use the transport controls to listen locally.";
            LibraryAudioPlayer.IsEnabled = true;
            _libraryMediaPlayer.Source = MediaSource.CreateFromUri(
                new Uri(Path.GetFullPath(day.CombinedPath)));
        }
        else
        {
            LibraryAudioStatusText.Text = "No combined day recording is available yet.";
            LibraryAudioPlayer.IsEnabled = false;
        }
        if (!IsFeedPipelineBusy(day.FeedId))
        {
            LibraryTranscriptPreviewText.Text = day.HasStaleTranscript
                ? "The previous transcript is preserved but hidden because the combined recording changed. Finish this day locally to update it."
                : File.Exists(day.TranscriptPath)
                ? "Loading timestamped transcript preview…"
                : "No transcript is available yet. The Processing tab shows the next step.";
        }
        UpdateCommandAvailability();
        return selectionVersion;
    }

    private async Task LoadLibraryTranscriptPreviewAsync(LibraryDay day, int selectionVersion)
    {
        if (!day.HasTranscript || !File.Exists(day.TranscriptPath))
        {
            return;
        }
        try
        {
            var preview = await Task.Run(() => BuildTranscriptPreview(day.TranscriptPath));
            if (selectionVersion == _librarySelectionVersion)
            {
                LibraryTranscriptPreviewText.Text = preview;
            }
        }
        catch (Exception exception) when (exception is IOException or JsonException or UnauthorizedAccessException)
        {
            if (selectionVersion == _librarySelectionVersion)
            {
                LibraryTranscriptPreviewText.Text = $"The transcript preview could not be loaded: {exception.Message}";
            }
        }
    }

    private static string BuildTranscriptPreview(string transcriptPath)
    {
        using var stream = File.OpenRead(transcriptPath);
        using var document = JsonDocument.Parse(stream);
        if (!document.RootElement.TryGetProperty("segments", out var segments)
            || segments.ValueKind != JsonValueKind.Array)
        {
            return "This transcript has no timestamped segment list.";
        }

        const int maximumSegments = 120;
        const int maximumCharacters = 14_000;
        var builder = new StringBuilder();
        var shown = 0;
        var total = segments.GetArrayLength();
        foreach (var segment in segments.EnumerateArray())
        {
            var text = segment.TryGetProperty("text", out var textValue)
                ? (textValue.GetString() ?? "").Trim()
                : "";
            if (string.IsNullOrWhiteSpace(text))
            {
                continue;
            }
            var seconds = segment.TryGetProperty("start", out var startValue)
                && startValue.TryGetDouble(out var parsedStart)
                    ? parsedStart
                    : 0;
            var speaker = segment.TryGetProperty("speaker", out var speakerValue)
                ? speakerValue.GetString()
                : null;
            builder.Append('[').Append(FormatLibraryTimestamp(seconds)).Append("] ");
            if (!string.IsNullOrWhiteSpace(speaker))
            {
                builder.Append(speaker).Append(": ");
            }
            builder.AppendLine(text);
            shown++;
            if (shown >= maximumSegments || builder.Length >= maximumCharacters)
            {
                break;
            }
        }
        if (shown == 0)
        {
            return "This transcript contains no non-empty speech segments.";
        }
        if (shown < total)
        {
            builder.AppendLine().Append($"Previewing {shown:N0} of {total:N0} segments. Open the full transcript for the rest.");
        }
        return builder.ToString().TrimEnd();
    }

    private static string FormatLibraryTimestamp(double seconds)
    {
        var value = TimeSpan.FromSeconds(Math.Max(0, seconds));
        return value.TotalHours >= 1
            ? $"{(int)value.TotalHours:00}:{value.Minutes:00}:{value.Seconds:00}"
            : $"{value.Minutes:00}:{value.Seconds:00}";
    }

    private async void LibraryDetailPrimary_Click(object sender, RoutedEventArgs e)
    {
        if (_selectedLibraryDay is not null)
        {
            await ContinueLibraryDayAsync(_selectedLibraryDay);
        }
    }

    private async void LibraryDetailReview_Click(object sender, RoutedEventArgs e)
    {
        if (_selectedLibraryDay is not null && _selectedLibraryDay.CanOpenReview)
        {
            await OpenLibraryDayInReviewAsync(_selectedLibraryDay);
        }
    }

    private async void LibrarySpeakerUpgrade_Click(object sender, RoutedEventArgs e)
    {
        if (_selectedLibraryDay is not { SpeakerUpgradeAvailable: true } day)
        {
            return;
        }
        await ContinueLibraryDayAsync(day, diarizationEngineOverride: "community-1");
    }

    private async void LibraryFeedCoverage_SelectionChanged(
        object sender,
        SelectionChangedEventArgs e)
    {
        if (LibraryFeedCoverageList.SelectedItem is not LibraryFeedCoverage feed)
        {
            return;
        }
        var day = _visibleLibraryDays.FirstOrDefault(value => value.FeedId == feed.FeedId)
            ?? _libraryDays.FirstOrDefault(value => value.FeedId == feed.FeedId);
        if (day is not null)
        {
            LibraryList.SelectedItem = day;
        }
        else
        {
            LibraryList.SelectedItem = null;
            ShowLibraryDetails(null);
        }
        SelectReviewFeed(feed.FeedId, clearChatWhenChanged: false);
        await RefreshAnalysisDaysAsync();
    }

    private async void LibraryCheckSource_Click(object sender, RoutedEventArgs e)
    {
        if (_selectedLibraryDay is not null)
        {
            await ContinueLibraryDayAsync(
                _selectedLibraryDay,
                forceSourceCheck: true);
        }
    }

    private async void LibraryDeleteFeed_Click(object sender, RoutedEventArgs e)
    {
        if (_worker is null
            || _libraryMutationBusy
            || _selectedLibraryDay is not { } selected)
        {
            return;
        }
        if (IsFeedBusy(selected.FeedId))
        {
            await ShowMessageAsync(
                "Feed is currently in use",
                $"{selected.FeedName} has an active processing or archive-chat worker. "
                + "Let that feed finish or cancel it before deleting. Other feeds can still be deleted while background work continues.");
            return;
        }
        LibraryActionInfoBar.IsOpen = false;

        IReadOnlyList<FeedSchedule> schedules;
        try
        {
            using var cancellation = new CancellationTokenSource(TimeSpan.FromMinutes(1));
            schedules = await _worker.ListFeedSchedulesAsync(cancellation.Token);
        }
        catch (Exception exception)
        {
            await ShowErrorAsync(exception);
            return;
        }

        var feedDays = _libraryDays
            .Where(value => value.FeedId == selected.FeedId)
            .ToList();
        var retainedBytes = feedDays.Sum(value => value.StorageBytes);
        var workingBytes = feedDays.Sum(value => value.WorkingStorageBytes);
        var hasSchedule = schedules.Any(value => value.FeedId == selected.FeedId);
        var removeScheduleCheckBox = new CheckBox
        {
            Content = "Also remove this feed's scheduled download",
            IsChecked = true,
            Visibility = hasSchedule ? Visibility.Visible : Visibility.Collapsed,
        };
        var content = new StackPanel { Spacing = 10 };
        content.Children.Add(new TextBlock
        {
            Text =
                $"This permanently deletes {feedDays.Count:N0} local day(s) for "
                + $"{selected.FeedName} (feed {selected.FeedId}), including source "
                + "audio, combined recordings, transcripts, speaker labels, incidents, "
                + "summaries, and search indexes.",
            TextWrapping = TextWrapping.Wrap,
        });
        content.Children.Add(new TextBlock
        {
            Text =
                $"Approximate storage: {(retainedBytes + workingBytes) / 1_048_576d:0.#} MB. "
                + "Saved Area Watch profiles remain configured and may acquire this "
                + "feed again when you explicitly run them.",
            TextWrapping = TextWrapping.Wrap,
            Foreground = (Brush)Application.Current.Resources[
                "TextFillColorSecondaryBrush"],
        });
        content.Children.Add(removeScheduleCheckBox);
        var confirmation = new ContentDialog
        {
            XamlRoot = ((FrameworkElement)Content).XamlRoot,
            Title = $"Delete {selected.FeedName} from the Library?",
            Content = CreateDialogScrollContent(content),
            PrimaryButtonText = "Delete feed",
            CloseButtonText = "Cancel",
            DefaultButton = ContentDialogButton.Close,
        };
        if (await confirmation.ShowAsync() != ContentDialogResult.Primary)
        {
            return;
        }
        if (IsFeedBusy(selected.FeedId))
        {
            LibraryActionInfoBar.Severity = InfoBarSeverity.Warning;
            LibraryActionInfoBar.Title = "Feed became busy";
            LibraryActionInfoBar.Message =
                "A background worker began using this feed while the confirmation was open. Nothing was deleted; retry after that feed finishes.";
            LibraryActionInfoBar.IsOpen = true;
            return;
        }

        LibraryFeedDeleteResult? result = null;
        using var deletion = new CancellationTokenSource();
        _libraryMutationBusy = true;
        StatusText.Text = $"Deleting {selected.FeedName} from the local Library…";
        UpdateCommandAvailability();
        try
        {
            await ReleaseMediaForArchiveMutationAsync(recreatePlayers: false);
            result = await _worker.DeleteLibraryFeedAsync(
                PersistedOutputDirectory(),
                selected.FeedId,
                hasSchedule && removeScheduleCheckBox.IsChecked == true,
                deletion.Token);
            if (result is null)
            {
                throw new InvalidOperationException(
                    "The deletion worker returned no completion result.");
            }
            _selectedLibraryDay = null;
            AppendLog(
                $"Deleted feed {selected.FeedId}: {result.DaysDeleted:N0} day(s), "
                + $"{result.SegmentsDeleted:N0} transcript segment(s), "
                + $"{result.IncidentsDeleted:N0} incident(s), and "
                + $"{result.SchedulesDeleted:N0} schedule(s).");
            if (result.CleanupPending)
            {
                AppendLog(
                    "The feed was removed from the Library, but a detached cleanup "
                    + "folder remains for a later safe cleanup pass.");
            }
        }
        catch (OperationCanceledException)
        {
            StatusText.Text = "Feed deletion cancelled";
        }
        catch (Exception exception)
        {
            if (exception.Message.Contains(
                    "is still in use",
                    StringComparison.OrdinalIgnoreCase))
            {
                AppendLog($"Delete feed: {exception.Message}");
                LibraryActionInfoBar.Severity = InfoBarSeverity.Error;
                LibraryActionInfoBar.Title = "Feed is still in use";
                LibraryActionInfoBar.Message = exception.Message;
                LibraryActionInfoBar.IsOpen = true;
            }
            else
            {
                StatusText.Text = "Feed deletion failed";
                AppendLog($"ERROR: {exception.Message}");
                LibraryActionInfoBar.Severity = InfoBarSeverity.Error;
                LibraryActionInfoBar.Title = "Feed could not be deleted";
                LibraryActionInfoBar.Message = exception.Message;
                LibraryActionInfoBar.IsOpen = true;
            }
        }
        finally
        {
            RestoreMediaPlayerInstances();
            _libraryMutationBusy = false;
            UpdateCommandAvailability();
            await RefreshLibraryAsync();
            await RefreshAnalysisDaysAsync();
            await RefreshFeedScheduleStatusAsync();
        }

        if (result is not null)
        {
            StatusText.Text = $"Deleted feed {result.FeedId} from the Library";
            LibraryActionInfoBar.Severity = InfoBarSeverity.Success;
            LibraryActionInfoBar.Title = "Feed deleted";
            LibraryActionInfoBar.Message =
                $"Removed feed {result.FeedId} from the selected Library. "
                + (result.SchedulesDeleted > 0
                    ? "Its scheduled download was also removed. "
                    : "No scheduled download was removed. ")
                + (result.CatchUpsDeleted > 0
                    ? "Its saved catch-up range was also cleared."
                    : "No saved catch-up range was active.");
            LibraryActionInfoBar.IsOpen = true;
        }
    }

    private async Task RefreshManagedRuntimeStatusAsync()
    {
        if (_worker is null
            || ManagedRuntimeStateText is null
            || ManagedRuntimeDetailText is null
            || ManagedRuntimeSourceText is null
            || ManagedRuntimeInstallButton is null)
        {
            return;
        }
        if (!_worker.HasBundledManagedRuntime)
        {
            _managedCudaRuntime = null;
            ManagedRuntimeStateText.Text = "Not packaged";
            ManagedRuntimeDetailText.Text =
                "Managed setup is available in the installed Windows package.";
            ManagedRuntimeInstallButton.Content = "Unavailable";
            ManagedRuntimeInstallButton.IsEnabled = false;
            return;
        }
        try
        {
            _managedCudaRuntime = await _worker.GetManagedRuntimeStatusAsync(
                "cuda",
                CancellationToken.None);
            if (_managedCudaRuntime is null)
            {
                throw new InvalidOperationException(
                    "The managed runtime worker returned no status.");
            }
            ApplyManagedRuntimeStatus(_managedCudaRuntime);
        }
        catch (Exception exception)
        {
            _managedCudaRuntime = null;
            ManagedRuntimeStateText.Text = "Needs attention";
            ManagedRuntimeDetailText.Text = exception.Message;
            ManagedRuntimeInstallButton.Content = "Retry check";
            ManagedRuntimeInstallButton.IsEnabled = true;
            AppendLog($"Managed runtime: {exception.Message}");
        }
    }

    private void ApplyManagedRuntimeStatus(ManagedRuntimeStatus status)
    {
        _managedCudaRuntime = status;
        ManagedRuntimeStateText.Text = status.Ready
            ? "Installed"
            : status.Partial
                ? "Resumable"
                : "Optional";
        ManagedRuntimeDetailText.Text = status.Ready
            ? $"{status.Message} Installed size: {status.InstalledStorage}. "
              + $"Storage: {status.StoragePath}"
            : $"{status.Message} Allow about {status.EstimatedStorage}. "
              + $"Storage: {status.StoragePath}";
        var sources = status.SourceUrls.Count == 0
            ? "packaged manifest"
            : string.Join(", ", status.SourceUrls);
        var licenses = status.Licenses.Count == 0
            ? "package metadata"
            : string.Join("; ", status.Licenses);
        ManagedRuntimeSourceText.Text =
            $"Sources: {sources}. Licenses: {licenses}. "
            + "The packaged bootstrap and app wheel are SHA-256 verified before use.";
        ManagedRuntimeInstallButton.Content = status.Ready
            ? "Use runtime"
            : status.Partial
                ? "Resume install"
                : "Install runtime";
        ManagedRuntimeInstallButton.IsEnabled = _operationCancellation is null;
    }

    private async void ManagedRuntimeInstall_Click(
        object sender,
        RoutedEventArgs e)
    {
        if (_worker is null || _operationCancellation is not null)
        {
            return;
        }
        if (_managedCudaRuntime is null)
        {
            await RefreshManagedRuntimeStatusAsync();
        }
        var current = _managedCudaRuntime;
        if (current is null)
        {
            return;
        }
        if (current.Ready)
        {
            SelectManagedRuntime(current);
            await ShowMessageAsync(
                "Restart to use the managed runtime",
                "The packaged CUDA runtime is selected. Restart Broadcastify Desktop, "
                + "then use Verify profile to prove transcription, speaker labels, and "
                + "analysis on this GPU.");
            return;
        }

        var action = current.Partial ? "Resume" : "Install";
        var confirmation = new ContentDialog
        {
            XamlRoot = ((FrameworkElement)Content).XamlRoot,
            Title = $"{action} packaged CUDA runtime?",
            Content = CreateDialogTextContent(
                $"This explicit setup uses about {current.EstimatedStorage} in "
                + $"{current.StoragePath}. It downloads an isolated Python runtime and "
                + "pinned binary wheels from the sources shown on this page. Package "
                + "licenses are retained. Cancel stops the process and keeps its verified "
                + "cache so a later retry can resume. No Broadcastify request is made."),
            PrimaryButtonText = action,
            CloseButtonText = "Not now",
            DefaultButton = ContentDialogButton.Primary,
        };
        if (await confirmation.ShowAsync() != ContentDialogResult.Primary)
        {
            return;
        }

        _operationCancellation = new CancellationTokenSource();
        var activity = current.Partial ? "Resuming" : "Installing";
        SetBusy(
            true,
            $"{activity} the managed CUDA runtime…",
            jobRunning: true);
        JobProgress.IsIndeterminate = true;
        try
        {
            var installed = await _worker.InstallManagedRuntimeAsync(
                "cuda",
                HandleWorkerMessage,
                _operationCancellation.Token);
            if (installed is null || !installed.Ready)
            {
                throw new InvalidOperationException(
                    "The managed runtime installer ended without a verified environment.");
            }
            ApplyManagedRuntimeStatus(installed);
            SelectManagedRuntime(installed);
            StatusText.Text = "Managed CUDA runtime installed";
            AppendLog(
                $"Managed CUDA runtime retained at {installed.StoragePath}; "
                + "restart required before execution verification.");
            await ShowMessageAsync(
                "CUDA runtime installed",
                "Restart Broadcastify Desktop to activate the managed runtime. "
                + "After restart, choose NVIDIA CUDA and run Verify profile before "
                + "starting an unattended archive job.");
        }
        catch (OperationCanceledException)
        {
            StatusText.Text = "Managed runtime installation paused";
            AppendLog(
                "Managed runtime installation cancelled; partial work and the "
                + "verified download cache remain resumable.");
        }
        catch (Exception exception)
        {
            await ShowErrorAsync(exception);
        }
        finally
        {
            _operationCancellation.Dispose();
            _operationCancellation = null;
            JobProgress.IsIndeterminate = false;
            SetBusy(false);
            await RefreshManagedRuntimeStatusAsync();
        }
    }

    private void SelectManagedRuntime(ManagedRuntimeStatus status)
    {
        if (!status.Ready || string.IsNullOrWhiteSpace(status.PythonPath))
        {
            return;
        }
        _configuredPythonRuntimePath = status.PythonPath;
        PythonRuntimePathBox.Text = status.PythonPath;
        PythonRuntimeStatusText.Text =
            "Managed CUDA runtime selected. Restart the app to activate and verify it.";
        PersistUserSettings(logFailure: true);
    }

    private async void LibraryOpenFolder_Click(object sender, RoutedEventArgs e)
    {
        if (_selectedLibraryDay is null || !Directory.Exists(_selectedLibraryDay.DayDirectory))
        {
            return;
        }
        try
        {
            using var launched = Process.Start(new ProcessStartInfo
            {
                FileName = Path.GetFullPath(_selectedLibraryDay.DayDirectory),
                UseShellExecute = true,
            });
            if (launched is null)
            {
                throw new InvalidOperationException("Windows could not open the folder.");
            }
        }
        catch (Exception exception)
        {
            await ShowErrorAsync(exception);
        }
    }

    private async void LibraryOpenTranscript_Click(object sender, RoutedEventArgs e)
    {
        if (_selectedLibraryDay is null || !File.Exists(_selectedLibraryDay.TranscriptPath))
        {
            return;
        }
        if (IsFeedPipelineBusy(_selectedLibraryDay.FeedId))
        {
            LibraryDetailStatusText.Text =
                "This feed is being updated by the background pipeline. Its transcript will be available again when the current day finishes; other feeds remain usable.";
            return;
        }
        try
        {
            using var launched = Process.Start(new ProcessStartInfo
            {
                FileName = Path.GetFullPath(_selectedLibraryDay.TranscriptPath),
                UseShellExecute = true,
            });
            if (launched is null)
            {
                throw new InvalidOperationException("Windows could not open the transcript.");
            }
        }
        catch (Exception exception)
        {
            await ShowErrorAsync(exception);
        }
    }

    private async void LibraryPrimary_Click(object sender, RoutedEventArgs e)
    {
        if (sender is not FrameworkElement { DataContext: LibraryDay day } || _worker is null)
        {
            return;
        }
        await ContinueLibraryDayAsync(day);
    }

    private async Task ContinueLibraryDayAsync(
        LibraryDay day,
        string? diarizationEngineOverride = null,
        bool forceSourceCheck = false)
    {
        if (_worker is null || _pipelineCancellation is not null)
        {
            return;
        }
        if (day.PrimaryAction == "open_review"
            && string.IsNullOrWhiteSpace(diarizationEngineOverride)
            && !forceSourceCheck)
        {
            await OpenLibraryDayInReviewAsync(day);
            return;
        }
        if (!DateTime.TryParse(day.ArchiveDate, out var archiveDate))
        {
            await ShowMessageAsync("Invalid library date", $"Could not parse {day.ArchiveDate}.");
            return;
        }
        var minimumSpeakers = OptionalPositiveInteger(MinimumSpeakersBox.Value);
        var maximumSpeakers = OptionalPositiveInteger(MaximumSpeakersBox.Value);
        if (minimumSpeakers is not null && maximumSpeakers is not null && minimumSpeakers > maximumSpeakers)
        {
            await ShowMessageAsync("Invalid speaker range", "Minimum speakers cannot exceed maximum speakers.");
            return;
        }

        var pipeline = new CancellationTokenSource();
        _pipelineCancellation = pipeline;
        _activePipelineFeedIds.Add(day.FeedId);
        SetPipelineBusy(
            true,
            forceSourceCheck
                ? $"Checking source audio for {day.FeedName} on {day.ArchiveDate} in the background…"
                : $"Continuing {day.FeedName} for {day.ArchiveDate} in the background…");
        JobProgress.IsIndeterminate = true;
        try
        {
            await ContinueLibraryDayWorkAsync(
                day,
                archiveDate.Date,
                minimumSpeakers,
                maximumSpeakers,
                diarizationEngineOverride,
                forceSourceCheck,
                cancellationToken: pipeline.Token);
        }
        catch (OperationCanceledException)
        {
            StatusText.Text = "Cancelled";
            AppendLog("Library continuation cancelled.");
        }
        catch (Exception exception)
        {
            await ShowErrorAsync(exception);
        }
        finally
        {
            pipeline.Dispose();
            if (ReferenceEquals(_pipelineCancellation, pipeline))
            {
                _pipelineCancellation = null;
            }
            _activePipelineFeedIds.Clear();
            JobProgress.IsIndeterminate = false;
            SetPipelineBusy(false);
            await RefreshLibraryAsync();
            await RefreshArchiveQuotaStatusAsync();
        }
    }

    private async Task OpenLibraryDayInReviewAsync(LibraryDay day)
    {
        SelectReviewFeed(day.FeedId);
        NavigateTo(ReviewNavigationItem);
        await RefreshAnalysisDaysAsync();
        var match = _analysisDays.FirstOrDefault(value =>
            value.FeedId == day.FeedId && value.ArchiveDate == day.ArchiveDate);
        if (match is not null)
        {
            AnalysisDaysList.SelectedItem = match;
            await LoadDayReportAsync(match);
        }
    }

    private async void Search_Click(object sender, RoutedEventArgs e) => await SearchAsync();

    private async void Search_QuerySubmitted(AutoSuggestBox sender, AutoSuggestBoxQuerySubmittedEventArgs args) => await SearchAsync();

    private async Task SearchAsync()
    {
        if (_worker is null || string.IsNullOrWhiteSpace(SearchBox.Text))
        {
            return;
        }

        SetBusy(true, "Searching feeds…");
        try
        {
            using var cancellation = new CancellationTokenSource(TimeSpan.FromSeconds(30));
            var results = await _worker.SearchFeedsAsync(SearchBox.Text.Trim(), cancellation.Token);
            _feeds.Clear();
            foreach (var result in results)
            {
                _feeds.Add(result);
            }
            StatusText.Text = $"Found {results.Count} feed{(results.Count == 1 ? "" : "s")}";
            AppendLog($"Search returned {results.Count} feed(s).");
        }
        catch (Exception exception)
        {
            await ShowErrorAsync(exception);
        }
        finally
        {
            SetBusy(false);
        }
    }

    private void FeedResults_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        _selectedFeed = FeedResults.SelectedItem as FeedSearchResult;
        SelectedFeedText.Text = _selectedFeed is null
            ? "No feed selected"
            : $"Selected: {_selectedFeed.Name} · feed {_selectedFeed.FeedId}";
        if (_selectedFeed is not null)
        {
            SelectReviewFeed(_selectedFeed.FeedId);
        }
    }

    private string SelectedBroadcastifyProfileId()
    {
        return BroadcastifyAccountProfileBox.SelectedItem is ComboBoxItem item
            && item.Tag is string profileId
            ? profileId
            : "default";
    }

    private void BroadcastifyAccountProfile_SelectionChanged(
        object sender,
        SelectionChangedEventArgs e)
    {
        if (ClearSavedLoginButton is null)
        {
            return;
        }
        var profileId = SelectedBroadcastifyProfileId();
        ClearSavedLoginButton.IsEnabled = profileId != "__new__"
            && CredentialStore.TryLoadBroadcastifyProfile(profileId) is not null;
    }

    private void RefreshBroadcastifyProfileControls(string? selectedProfileId = null)
    {
        var selected = string.IsNullOrWhiteSpace(selectedProfileId)
            ? SelectedBroadcastifyProfileId()
            : selectedProfileId;
        var profiles = CredentialStore.ListBroadcastifyProfiles();
        BroadcastifyAccountProfileBox.Items.Clear();
        if (profiles.All(value => value.Id != "default"))
        {
            BroadcastifyAccountProfileBox.Items.Add(new ComboBoxItem
            {
                Content = "Primary account · environment or saved session",
                Tag = "default",
            });
        }
        foreach (var profile in profiles)
        {
            BroadcastifyAccountProfileBox.Items.Add(new ComboBoxItem
            {
                Content = $"{profile.Label} · {profile.Username}",
                Tag = profile.Id,
            });
        }
        if (_worker is not null)
        {
            foreach (var profileId in _worker.AvailableAccountProfileIds()
                         .Where(profileId => BroadcastifyAccountProfileBox.Items
                             .OfType<ComboBoxItem>()
                             .All(item => !string.Equals(
                                 item.Tag as string,
                                 profileId,
                                 StringComparison.OrdinalIgnoreCase))))
            {
                BroadcastifyAccountProfileBox.Items.Add(new ComboBoxItem
                {
                    Content = profileId == "default"
                        ? "Primary account · private environment or saved session"
                        : $"{profileId} · private environment or saved session",
                    Tag = profileId,
                });
            }
        }
        BroadcastifyAccountProfileBox.Items.Add(new ComboBoxItem
        {
            Content = "Add another authorized account…",
            Tag = "__new__",
        });
        BroadcastifyAccountProfileBox.SelectedItem =
            BroadcastifyAccountProfileBox.Items
                .OfType<ComboBoxItem>()
                .FirstOrDefault(value => string.Equals(
                    value.Tag as string,
                    selected,
                    StringComparison.OrdinalIgnoreCase))
            ?? BroadcastifyAccountProfileBox.Items.OfType<ComboBoxItem>().First();
        var selectedId = SelectedBroadcastifyProfileId();
        ClearSavedLoginButton.IsEnabled = selectedId != "__new__"
            && CredentialStore.TryLoadBroadcastifyProfile(selectedId) is not null;
    }

    private async void SignIn_Click(object sender, RoutedEventArgs e)
    {
        if (_worker is null)
        {
            return;
        }

        RefreshBroadcastifyProfileControls();
        var selectedProfileId = SelectedBroadcastifyProfileId();
        var addingProfile = selectedProfileId == "__new__";
        var saved = addingProfile
            ? null
            : CredentialStore.TryLoadBroadcastifyProfile(selectedProfileId);
        var labelBox = new TextBox
        {
            Header = "Account label",
            Text = saved?.Label ?? (addingProfile
                ? "Secondary account"
                : selectedProfileId == "default"
                    ? "Primary account"
                    : selectedProfileId),
            IsReadOnly = !addingProfile && selectedProfileId == "default",
        };
        var usernameBox = new TextBox
        {
            Header = "Broadcastify username",
            Text = saved?.Username ?? "",
        };
        var passwordBox = new PasswordBox
        {
            Header = "Password",
            PlaceholderText = saved is null
                ? "Enter your Broadcastify password"
                : $"Saved: {CredentialStore.CreateSecretPreview(saved.Password, 2)} — leave blank to reuse",
            PasswordRevealMode = PasswordRevealMode.Peek,
        };
        var rememberCheckBox = new CheckBox
        {
            Content = "Save this login securely in Windows Credential Locker and sign in automatically",
            IsChecked = true,
        };
        var errorText = new TextBlock
        {
            Foreground = new Microsoft.UI.Xaml.Media.SolidColorBrush(Colors.IndianRed),
            TextWrapping = TextWrapping.Wrap,
        };
        var fields = new StackPanel { Spacing = 12 };
        fields.Children.Add(new TextBlock
        {
            Text = "Credentials are sent only to Broadcastify. Windows Credential Locker encrypts each account separately; session cookies and rolling quota records use the same non-secret profile ID. Add profiles only when you have provider authorization for aggregate account capacity.",
            TextWrapping = TextWrapping.Wrap,
        });
        fields.Children.Add(labelBox);
        fields.Children.Add(usernameBox);
        fields.Children.Add(passwordBox);
        fields.Children.Add(rememberCheckBox);
        fields.Children.Add(errorText);

        var dialog = new ContentDialog
        {
            XamlRoot = ((FrameworkElement)Content).XamlRoot,
            Title = addingProfile
                ? "Add an authorized premium account"
                : $"Sign in · {labelBox.Text}",
            PrimaryButtonText = "Sign in",
            CloseButtonText = "Cancel",
            DefaultButton = ContentDialogButton.Primary,
            Content = CreateDialogScrollContent(fields),
        };

        dialog.PrimaryButtonClick += async (_, args) =>
        {
            var deferral = args.GetDeferral();
            try
            {
                var username = usernameBox.Text.Trim();
                var password = passwordBox.Password;
                if (string.IsNullOrEmpty(password)
                    && saved is not null
                    && username.Equals(saved.Username, StringComparison.OrdinalIgnoreCase))
                {
                    password = saved.Password;
                }
                if (string.IsNullOrWhiteSpace(username) || string.IsNullOrEmpty(password))
                {
                    args.Cancel = true;
                    errorText.Text = saved is null
                        ? "Enter both your username and password."
                        : "Enter a password, or keep the saved username to reuse the encrypted password.";
                    return;
                }
                var profileId = selectedProfileId;
                if (addingProfile)
                {
                    if (string.IsNullOrWhiteSpace(labelBox.Text))
                    {
                        args.Cancel = true;
                        errorText.Text = "Enter a short label for this account.";
                        return;
                    }
                    var baseId = CredentialStore.CreateProfileId(labelBox.Text);
                    profileId = baseId;
                    var existingIds = CredentialStore.ListBroadcastifyProfiles()
                        .Select(value => value.Id)
                        .ToHashSet(StringComparer.OrdinalIgnoreCase);
                    for (var suffix = 2; existingIds.Contains(profileId); suffix++)
                    {
                        profileId = $"{baseId}-{suffix}";
                    }
                }
                await _worker.AuthenticateAsync(
                    username,
                    password,
                    HandleWorkerMessage,
                    CancellationToken.None,
                    profileId);
                if (rememberCheckBox.IsChecked == true)
                {
                    CredentialStore.SaveBroadcastifyProfile(
                        profileId,
                        labelBox.Text,
                        username,
                        password);
                    SettingsAuthStatusText.Text =
                        $"Saved {labelBox.Text.Trim()} ({profileId}) for {username} with password "
                        + $"{CredentialStore.CreateSecretPreview(password, 2)} in Windows Credential Locker. "
                        + "Its session and request allowance are isolated from every other profile.";
                }
                else
                {
                    if (!addingProfile)
                    {
                        CredentialStore.ClearBroadcastifyProfile(profileId);
                    }
                    SettingsAuthStatusText.Text = "Signed in for this session; the username and password were not saved.";
                }
                passwordBox.Password = "";
                AuthInfoBar.Severity = InfoBarSeverity.Success;
                AuthInfoBar.Title = "Signed in";
                AuthInfoBar.Message = $"Premium archive session {profileId} is ready and uses its own cookie and rolling allowance.";
                _archiveAccessConfigured = true;
                _archiveAccessVerified = true;
                RefreshBroadcastifyProfileControls(profileId);
                UpdateSetupSummary();
                AppendLog($"Broadcastify sign-in succeeded for account profile {profileId}.");
            }
            catch (Exception exception)
            {
                args.Cancel = true;
                errorText.Text = exception.Message;
            }
            finally
            {
                deferral.Complete();
            }
        };

        await dialog.ShowAsync();
    }

    private async Task TryAutoSignInAsync()
    {
        if (_worker is null)
        {
            return;
        }
        RefreshBroadcastifyProfileControls("default");
        var profiles = CredentialStore.ListBroadcastifyProfiles();
        var saved = profiles.FirstOrDefault();
        if (saved is null)
        {
            _archiveAccessConfigured = _worker.HasConfiguredAccountCredentials;
            SettingsAuthStatusText.Text = _worker.HasConfiguredAccountCredentials
                ? "A private account environment or isolated saved session is available. Archive jobs can refresh the selected profile automatically."
                : "No Windows Credential Locker login is saved. An existing session cookie or repository .env can still provide access.";
            RefreshBroadcastifyProfileControls("default");
            UpdateSetupSummary();
            return;
        }
        _archiveAccessConfigured = true;
        RefreshBroadcastifyProfileControls(saved.Id);
        SettingsAuthStatusText.Text = $"Refreshing {saved.Label} for {saved.Username}…";
        try
        {
            using var cancellation = new CancellationTokenSource(TimeSpan.FromSeconds(45));
            await _worker.AuthenticateAsync(
                saved.Username,
                saved.Password,
                HandleWorkerMessage,
                cancellation.Token,
                saved.Id);
            AuthInfoBar.Severity = InfoBarSeverity.Success;
            AuthInfoBar.Title = "Signed in automatically";
            AuthInfoBar.Message = $"Windows Credential Locker refreshed {saved.Label}; other profiles remain isolated and will refresh when used.";
            SettingsAuthStatusText.Text =
                $"{profiles.Count} saved account profile{(profiles.Count == 1 ? "" : "s")}. "
                + $"Automatic sign-in is enabled for {saved.Username} with password "
                + $"{CredentialStore.CreateSecretPreview(saved.Password, 2)}. "
                + "Complete passwords remain in Windows Credential Locker."
                + (_worker.AuthorizedAccountPoolEnabled
                    ? " Written-authorization pooling is enabled locally."
                    : " Automatic multi-account pooling is off until locally authorized.");
            _archiveAccessVerified = true;
            UpdateSetupSummary();
        }
        catch (Exception exception)
        {
            AuthInfoBar.Severity = InfoBarSeverity.Warning;
            AuthInfoBar.Title = "Saved login needs attention";
            AuthInfoBar.Message = "Automatic sign-in failed. Use Sign in to replace the saved login.";
            SettingsAuthStatusText.Text = exception.Message;
            _archiveAccessVerified = false;
            UpdateSetupSummary();
            AppendLog($"Automatic sign-in: {exception.Message}");
        }
    }

    private void ClearSavedLogin_Click(object sender, RoutedEventArgs e)
    {
        var profileId = SelectedBroadcastifyProfileId();
        if (profileId == "__new__")
        {
            return;
        }
        CredentialStore.ClearBroadcastifyProfile(profileId);
        RefreshBroadcastifyProfileControls("default");
        SettingsAuthStatusText.Text =
            $"Account profile {profileId} was removed from Windows Credential Locker. Its current isolated session cookie and quota history were retained.";
        _archiveAccessConfigured = _archiveAccessVerified
            || (_worker?.HasConfiguredAccountCredentials ?? false);
        UpdateSetupSummary();
        AuthInfoBar.Severity = InfoBarSeverity.Informational;
        AuthInfoBar.Title = "Saved login removed";
        AuthInfoBar.Message = "You can continue with retained sessions or add the account again later.";
    }

    private async void Browse_Click(object sender, RoutedEventArgs e)
    {
        var picker = new FolderPicker();
        picker.FileTypeFilter.Add("*");
        InitializeWithWindow.Initialize(picker, WindowNative.GetWindowHandle(this));
        var folder = await picker.PickSingleFolderAsync();
        if (folder is not null)
        {
            OutputFolderBox.Text = folder.Path;
            RefreshStorageReadiness();
        }
    }

    private async Task<(JobRunResult? Result, bool WaitingForQuota)>
        ContinueLibraryDayAcrossAccountsAsync(
            LibraryDay day,
            DateTime archiveDate,
            int? minimumSpeakers,
            int? maximumSpeakers,
            CancellationToken cancellationToken)
    {
        if (_worker is null)
        {
            return (null, false);
        }
        IReadOnlyList<string> profileIds = _worker.AuthorizedAccountPoolEnabled
            ? _worker.AvailableAccountProfileIds()
            : new[] { "default" };
        var statuses = new List<ArchiveQuotaStatus>();
        foreach (var profileId in profileIds)
        {
            var status = await _worker.GetArchiveQuotaStatusAsync(
                cancellationToken,
                profileId);
            if (status is not null)
            {
                statuses.Add(status);
            }
        }
        var eligible = statuses
            .Where(value => value.Available)
            .OrderByDescending(value => value.Remaining)
            .ThenBy(value => value.AccountProfileId, StringComparer.OrdinalIgnoreCase)
            .ToList();
        if (eligible.Count == 0)
        {
            return (null, true);
        }
        JobRunResult? lastResult = null;
        Exception? lastAuthenticationFailure = null;
        foreach (var status in eligible)
        {
            try
            {
                AppendLog(
                    $"Catch-up is using account profile {status.AccountProfileId} "
                    + $"for {day.FeedName} on {day.ArchiveDate}; "
                    + $"{status.Remaining}/{status.AutomatedLimit} requests remain in its window.");
                lastResult = await ContinueLibraryDayWorkAsync(
                    day,
                    archiveDate,
                    minimumSpeakers,
                    maximumSpeakers,
                    forceAllStages: true,
                    accountProfileId: status.AccountProfileId,
                    cancellationToken: cancellationToken);
                if (lastResult?.DownloadLimited != true)
                {
                    return (lastResult, false);
                }
                AppendLog(
                    $"Account profile {status.AccountProfileId} reached its boundary; "
                    + "the same retained day will continue on the next eligible profile.");
            }
            catch (InvalidOperationException exception)
                when (_worker.AuthorizedAccountPoolEnabled
                    && IsAccountAuthenticationFailure(exception.Message))
            {
                lastAuthenticationFailure = exception;
                AppendLog(
                    $"Account profile {status.AccountProfileId} needs sign-in; "
                    + "trying the next authorized profile.");
            }
        }
        if (lastResult is null && lastAuthenticationFailure is not null)
        {
            throw lastAuthenticationFailure;
        }
        return (lastResult, true);
    }

    private async Task<JobRunResult?> ContinueLibraryDayWorkAsync(
        LibraryDay day,
        DateTime archiveDate,
        int? minimumSpeakers,
        int? maximumSpeakers,
        string? diarizationEngineOverride = null,
        bool forceSourceCheck = false,
        bool forceAllStages = false,
        string accountProfileId = "default",
        CancellationToken cancellationToken = default)
    {
        if (_worker is null)
        {
            return null;
        }
        if (day.NeedsNetwork || forceSourceCheck)
        {
            AppendLog(forceSourceCheck
                ? $"Checking feed {day.FeedId} on {day.ArchiveDate} for new source audio."
                : $"Resuming archive coverage for feed {day.FeedId} on {day.ArchiveDate}.");
            var request = CreateJobRequest(
                day.FeedId,
                archiveDate,
                archiveDate,
                minimumSpeakers,
                maximumSpeakers,
                string.Equals(
                    day.FeedName,
                    $"Feed {day.FeedId}",
                    StringComparison.Ordinal)
                    ? null
                    : day.FeedName);
            if (forceAllStages)
            {
                request = request with
                {
                    Combine = true,
                    Transcribe = true,
                    Diarize = true,
                    DownloadJobs = 1,
                };
            }
            return await RunAndAnalyzeJobAsync(
                request,
                cancellationToken,
                forceAllStages ? true : null,
                accountProfileId);
        }

        await _worker.ContinueLocalDayAsync(
            ApplyAnalysisProvider(new LocalProcessingRequest
            {
                FeedId = day.FeedId,
                ArchiveDate = day.ArchiveDate,
                OutputDirectory = PersistedOutputDirectory(),
                Model = SelectedComboValue(ModelComboBox, "turbo"),
                AsrEngine = SelectedComboValue(AsrEngineComboBox, "auto"),
                Device = SelectedComboValue(DeviceComboBox, "auto"),
                DeviceIndex = RequiredInteger(GpuIndexBox.Value, 0),
                AsrModelPath = string.IsNullOrWhiteSpace(AsrModelPathBox.Text)
                    ? null
                    : AsrModelPathBox.Text.Trim(),
                DiarizationEngine = diarizationEngineOverride
                    ?? SelectedComboValue(DiarizationEngineComboBox, "community-1"),
                DiarizationDevice = SelectedComboValue(
                    DiarizationDeviceComboBox, "auto"),
                BatchSize = RequiredInteger(BatchSizeBox.Value, 8),
                Diarize = true,
                // Keep model analysis as a separate process so archive chat can
                // use the shared local model while ASR/diarization is running.
                Analyze = false,
                MinimumSpeakers = minimumSpeakers,
                MaximumSpeakers = maximumSpeakers,
                HuggingFaceToken = CurrentHuggingFaceToken(),
            }),
            HandleWorkerMessage,
            cancellationToken);
        AppendLog(
            $"Waiting for the shared analysis slot for {day.FeedName} on {day.ArchiveDate}; archive chat and this pipeline will not load competing local models.");
        await _analysisOperationGate.WaitAsync(cancellationToken);
        DayReport? report;
        try
        {
            report = await _worker.AnalyzeDayAsync(
                ApplyAnalysisProvider(new AnalysisRequest
                {
                    FeedId = day.FeedId,
                    ArchiveDate = day.ArchiveDate,
                    OutputDirectory = PersistedOutputDirectory(),
                }),
                HandleWorkerMessage,
                cancellationToken);
        }
        finally
        {
            _analysisOperationGate.Release();
        }
        SelectReviewFeed(day.FeedId, clearChatWhenChanged: false);
        if (report is not null)
        {
            ApplyReport(report);
        }
        await RefreshAnalysisDaysAsync();
        return null;
    }

    private async void BrowsePythonRuntime_Click(
        object sender,
        RoutedEventArgs e)
    {
        var picker = new FileOpenPicker
        {
            SuggestedStartLocation = PickerLocationId.ComputerFolder,
        };
        picker.FileTypeFilter.Add(".exe");
        InitializeWithWindow.Initialize(
            picker,
            WindowNative.GetWindowHandle(this));
        var file = await picker.PickSingleFileAsync();
        if (file is null)
        {
            return;
        }
        _configuredPythonRuntimePath = file.Path.Trim();
        PythonRuntimePathBox.Text = _configuredPythonRuntimePath;
        PythonRuntimeStatusText.Text =
            "Saved. Restart the app to validate and use this Python runtime.";
        PersistUserSettings(logFailure: true);
    }

    private void ClearPythonRuntime_Click(object sender, RoutedEventArgs e)
    {
        _configuredPythonRuntimePath = "";
        PythonRuntimePathBox.Text = "";
        PythonRuntimeStatusText.Text =
            "Bundled portable Python will be used after the app restarts.";
        PersistUserSettings(logFailure: true);
    }

    private void PythonRuntimePathBox_Loaded(
        object sender,
        RoutedEventArgs e)
    {
        _pythonRuntimeInputReady = false;
        if (!string.Equals(
                PythonRuntimePathBox.Text,
                _configuredPythonRuntimePath,
                StringComparison.Ordinal))
        {
            PythonRuntimePathBox.Text = _configuredPythonRuntimePath;
        }
        _pythonRuntimeInputReady = true;
    }

    private void Combine_Toggled(object sender, RoutedEventArgs e)
    {
        KeepOriginalsToggle.IsEnabled = CombineToggle.IsOn;
        UpdateAnalyzeAfterJobState();
    }

    private void Transcription_Changed(object sender, RoutedEventArgs e)
    {
        var enabled = TranscribeCheckBox.IsChecked == true;
        DiarizeCheckBox.IsEnabled = enabled;
        if (!enabled)
        {
            DiarizeCheckBox.IsChecked = false;
        }
        UpdateAnalyzeAfterJobState();
    }

    private void Diarization_Changed(object sender, RoutedEventArgs e)
    {
        var enabled = DiarizeCheckBox.IsChecked == true;
        MinimumSpeakersBox.IsEnabled = enabled;
        MaximumSpeakersBox.IsEnabled = enabled;
        CombineToggle.IsOn = enabled || CombineToggle.IsOn;
        CombineToggle.IsEnabled = !enabled;
        KeepOriginalsToggle.IsEnabled = CombineToggle.IsOn;
        UpdateAnalyzeAfterJobState();
    }

    private void UpdateAnalyzeAfterJobState()
    {
        var enabled = TranscribeCheckBox.IsChecked == true && CombineToggle.IsOn;
        AnalyzeAfterJobCheckBox.IsEnabled = enabled;
        if (!enabled)
        {
            AnalyzeAfterJobCheckBox.IsChecked = false;
        }
    }

    private async void Start_Click(object sender, RoutedEventArgs e)
    {
        if (_worker is null || _pipelineCancellation is not null)
        {
            return;
        }
        if (_selectedFeed is null)
        {
            await ShowMessageAsync("Feed required", "Select a feed before starting the job.");
            return;
        }

        var startDate = StartDatePicker.Date.Date;
        var endDate = EndDatePicker.Date.Date;
        if (startDate > endDate)
        {
            await ShowMessageAsync("Invalid date range", "Start date must be on or before end date.");
            return;
        }

        var minimumSpeakers = OptionalPositiveInteger(MinimumSpeakersBox.Value);
        var maximumSpeakers = OptionalPositiveInteger(MaximumSpeakersBox.Value);
        if (minimumSpeakers is not null && maximumSpeakers is not null && minimumSpeakers > maximumSpeakers)
        {
            await ShowMessageAsync("Invalid speaker range", "Minimum speakers cannot exceed maximum speakers.");
            return;
        }

        var request = CreateJobRequest(
            _selectedFeed.FeedId, startDate, endDate, minimumSpeakers, maximumSpeakers,
            _selectedFeed.Name);

        var pipeline = new CancellationTokenSource();
        _pipelineCancellation = pipeline;
        _activePipelineFeedIds.Add(request.FeedId);
        SetPipelineBusy(true, "Starting job in the background…");
        JobProgress.IsIndeterminate = true;
        JobProgress.Value = 0;
        AppendLog($"Starting feed {request.FeedId}: {request.StartDate} through {request.EndDate}");
        try
        {
            await RunAndAnalyzeJobAsync(request, pipeline.Token);
        }
        catch (OperationCanceledException)
        {
            StatusText.Text = "Cancelled";
            AppendLog("Job cancelled.");
        }
        catch (Exception exception)
        {
            await ShowErrorAsync(exception);
        }
        finally
        {
            pipeline.Dispose();
            if (ReferenceEquals(_pipelineCancellation, pipeline))
            {
                _pipelineCancellation = null;
            }
            _activePipelineFeedIds.Clear();
            JobProgress.IsIndeterminate = false;
            SetPipelineBusy(false);
            await RefreshLibraryAsync();
        }
    }

    private async void ScheduleFeed_Click(object sender, RoutedEventArgs e)
    {
        if (_worker is null || _selectedFeed is null)
        {
            await ShowMessageAsync("Feed required", "Search for and select the feed to schedule.");
            return;
        }
        var existing = (await _worker.ListFeedSchedulesAsync(CancellationToken.None))
            .FirstOrDefault(value => value.FeedId == _selectedFeed.FeedId);
        if (await ShowFeedScheduleEditorAsync(
                _selectedFeed.FeedId,
                _selectedFeed.Name,
                existing))
        {
            await RefreshFeedScheduleStatusAsync();
            await RefreshLibraryAsync();
        }
    }

    private async Task<bool> ShowFeedScheduleEditorAsync(
        string feedId,
        string feedName,
        FeedSchedule? existing,
        DateTimeOffset? suggestedBackfillStart = null,
        bool suggestRecurringCatchUp = false)
    {
        if (_worker is null)
        {
            return false;
        }

        var today = DateTime.Today;
        var storedJob = existing?.Job ?? CreateJobRequest(
            feedId,
            today,
            today,
            OptionalPositiveInteger(MinimumSpeakersBox.Value),
            OptionalPositiveInteger(MaximumSpeakersBox.Value),
            feedName);
        var timePicker = new TimePicker
        {
            Header = "Run daily at this local time",
            Time = TimeSpan.TryParse(existing?.RunTimeLocal, out var savedTime)
                ? savedTime
                : new TimeSpan(2, 0, 0),
            MinuteIncrement = 5,
        };
        var lookbackBox = new NumberBox
        {
            Header = "Revisit the latest days (including today)",
            Minimum = 1,
            Maximum = 14,
            Value = existing?.LookbackDays ?? 2,
            SpinButtonPlacementMode = NumberBoxSpinButtonPlacementMode.Inline,
        };
        var backfillPicker = new CalendarDatePicker
        {
            Header = "Catch up from (optional)",
            PlaceholderText = "No historical catch-up",
            Date = DateTimeOffset.TryParse(
                existing?.BackfillStartDate,
                out var savedBackfillDate)
                ? suggestedBackfillStart ?? savedBackfillDate
                : suggestedBackfillStart,
            MaxDate = DateTimeOffset.Now,
        };
        var recurringCatchUpBox = new CheckBox
        {
            Content = "Keep this catch-up boundary and recheck it every day",
            IsChecked = suggestRecurringCatchUp || (existing?.RecurringCatchUp ?? false),
        };
        void UpdateRecurringCatchUpState()
        {
            recurringCatchUpBox.IsEnabled = backfillPicker.Date is not null;
            if (backfillPicker.Date is null)
            {
                recurringCatchUpBox.IsChecked = false;
            }
        }
        backfillPicker.DateChanged += (_, _) => UpdateRecurringCatchUpState();
        UpdateRecurringCatchUpState();
        var accountProfileBox = new ComboBox
        {
            Header = "Archive account",
            MinWidth = 320,
        };
        if (_worker.AuthorizedAccountPoolEnabled)
        {
            accountProfileBox.Items.Add(new ComboBoxItem
            {
                Content = "Automatic authorized pool · use the next eligible account",
                Tag = "automatic",
            });
        }
        foreach (var profile in CredentialStore.ListBroadcastifyProfiles())
        {
            accountProfileBox.Items.Add(new ComboBoxItem
            {
                Content = $"{profile.Label} · {profile.Username}",
                Tag = profile.Id,
            });
        }
        foreach (var profileId in _worker.AvailableAccountProfileIds()
                     .Where(profileId => accountProfileBox.Items
                         .OfType<ComboBoxItem>()
                         .All(item => !string.Equals(
                             item.Tag as string,
                             profileId,
                             StringComparison.OrdinalIgnoreCase))))
        {
            accountProfileBox.Items.Add(new ComboBoxItem
            {
                Content = $"{profileId} · private environment or saved session",
                Tag = profileId,
            });
        }
        if (!accountProfileBox.Items.OfType<ComboBoxItem>().Any(value =>
                string.Equals(
                    value.Tag as string,
                    "default",
                    StringComparison.OrdinalIgnoreCase)))
        {
            accountProfileBox.Items.Add(new ComboBoxItem
            {
                Content = "Primary account · environment or saved session",
                Tag = "default",
            });
        }
        var existingProfileId = existing?.AccountProfileId ?? (
            _worker.AuthorizedAccountPoolEnabled ? "automatic" : "default");
        accountProfileBox.SelectedItem = accountProfileBox.Items
            .OfType<ComboBoxItem>()
            .FirstOrDefault(value => string.Equals(
                value.Tag as string,
                existingProfileId,
                StringComparison.OrdinalIgnoreCase))
            ?? accountProfileBox.Items.OfType<ComboBoxItem>().First();
        var enabledBox = new CheckBox
        {
            Content = "Schedule enabled",
            IsChecked = existing?.Enabled ?? true,
        };
        var combineBox = new CheckBox
        {
            Content = "Create combined daily audio",
            IsChecked = storedJob.Combine,
        };
        var transcribeBox = new CheckBox
        {
            Content = "Transcribe locally",
            IsChecked = storedJob.Transcribe,
        };
        var diarizeBox = new CheckBox
        {
            Content = "Add speaker diarization",
            IsChecked = storedJob.Diarize,
        };
        var analyzeBox = new CheckBox
        {
            Content = "Extract incidents, summarize, and index",
            IsChecked = existing?.Analyze ?? true,
        };
        var refreshProfileBox = new CheckBox
        {
            Content = "Refresh advanced processing settings from the current Settings page",
            IsChecked = false,
            Visibility = existing is null ? Visibility.Collapsed : Visibility.Visible,
        };
        var profileSummary = new TextBlock
        {
            Text = $"Saved processing: {storedJob.AsrEngine} / {storedJob.Model} / {storedJob.Device}; "
                + $"speaker labels: {storedJob.DiarizationEngine} / {storedJob.DiarizationDevice}; "
                + $"analysis: {storedJob.AnalysisProvider} / "
                + $"{(string.IsNullOrWhiteSpace(storedJob.AnalysisModel) ? "default model" : storedJob.AnalysisModel)}.",
            TextWrapping = TextWrapping.Wrap,
            FontSize = 12,
            Opacity = 0.72,
        };

        void UpdateProcessingDependencies()
        {
            if (diarizeBox.IsChecked == true)
            {
                transcribeBox.IsChecked = true;
                combineBox.IsChecked = true;
            }
            diarizeBox.IsEnabled = transcribeBox.IsChecked == true;
            analyzeBox.IsEnabled =
                transcribeBox.IsChecked == true && combineBox.IsChecked == true;
            if (transcribeBox.IsChecked != true)
            {
                diarizeBox.IsChecked = false;
                analyzeBox.IsChecked = false;
            }
            if (combineBox.IsChecked != true)
            {
                diarizeBox.IsChecked = false;
                analyzeBox.IsChecked = false;
            }
        }

        combineBox.Checked += (_, _) => UpdateProcessingDependencies();
        combineBox.Unchecked += (_, _) => UpdateProcessingDependencies();
        transcribeBox.Checked += (_, _) => UpdateProcessingDependencies();
        transcribeBox.Unchecked += (_, _) => UpdateProcessingDependencies();
        diarizeBox.Checked += (_, _) => UpdateProcessingDependencies();
        diarizeBox.Unchecked += (_, _) => UpdateProcessingDependencies();
        UpdateProcessingDependencies();

        var explanation = new TextBlock
        {
            Text = "The schedule reuses retained work and waits for rolling request slots. "
                + "An optional catch-up date can clear after the first complete run or remain active for a recurring full-range gap check. "
                + "An authorized pool rotates only after an account is unavailable, with one spaced download at a time. "
                + "Changing only the time or processing stages keeps its saved model and hardware choices.",
            TextWrapping = TextWrapping.Wrap,
        };
        var content = new StackPanel
        {
            Spacing = 12,
            MaxWidth = 480,
        };
        content.Children.Add(explanation);
        content.Children.Add(timePicker);
        content.Children.Add(lookbackBox);
        content.Children.Add(backfillPicker);
        content.Children.Add(recurringCatchUpBox);
        content.Children.Add(accountProfileBox);
        content.Children.Add(enabledBox);
        content.Children.Add(new TextBlock
        {
            Text = "Processing",
            FontWeight = Microsoft.UI.Text.FontWeights.SemiBold,
        });
        content.Children.Add(combineBox);
        content.Children.Add(transcribeBox);
        content.Children.Add(diarizeBox);
        content.Children.Add(analyzeBox);
        content.Children.Add(profileSummary);
        content.Children.Add(refreshProfileBox);
        var dialog = new ContentDialog
        {
            XamlRoot = ((FrameworkElement)Content).XamlRoot,
            Title = $"{(existing is null ? "New" : "Edit")} daily schedule · {feedName}",
            PrimaryButtonText = "Save schedule",
            CloseButtonText = "Cancel",
            DefaultButton = ContentDialogButton.Primary,
            Content = CreateDialogScrollContent(content, 600),
        };
        if (await dialog.ShowAsync() != ContentDialogResult.Primary)
        {
            return false;
        }

        var request = refreshProfileBox.IsChecked == true
            ? CreateJobRequest(
                feedId,
                today,
                today,
                OptionalPositiveInteger(MinimumSpeakersBox.Value),
                OptionalPositiveInteger(MaximumSpeakersBox.Value),
                feedName)
            : storedJob with
            {
                FeedId = feedId,
                FeedName = feedName,
                StartDate = today.ToString("yyyy-MM-dd"),
                EndDate = today.ToString("yyyy-MM-dd"),
            };
        request = request with
        {
            // Storage follows the currently selected Library even when the
            // schedule keeps its saved model and accelerator profile.
            OutputDirectory = PersistedOutputDirectory(),
            Combine = combineBox.IsChecked == true,
            KeepOriginals = true,
            Transcribe = transcribeBox.IsChecked == true,
            Diarize = diarizeBox.IsChecked == true,
            DownloadJobs = 1,
        };
        var saved = await _worker.SaveFeedScheduleAsync(
            new FeedScheduleSaveRequest
            {
                FeedId = feedId,
                FeedName = feedName,
                RunTimeLocal = $"{timePicker.Time.Hours:00}:{timePicker.Time.Minutes:00}",
                LookbackDays = RequiredInteger(lookbackBox.Value, 2),
                BackfillStartDate = backfillPicker.Date?.ToString("yyyy-MM-dd") ?? "",
                RecurringCatchUp = recurringCatchUpBox.IsChecked == true,
                AccountProfileId =
                    (accountProfileBox.SelectedItem as ComboBoxItem)?.Tag as string
                    ?? "default",
                Job = request,
                Analyze = analyzeBox.IsChecked == true,
                Enabled = enabledBox.IsChecked == true,
            },
            CancellationToken.None);
        AppendLog(saved is null
            ? "The feed schedule returned no saved record."
            : $"Scheduled {saved.FeedName} {saved.ScheduleSummary}.");
        return saved is not null;
    }

    private async void ManageSchedules_Click(object sender, RoutedEventArgs e)
    {
        if (_worker is null)
        {
            return;
        }

        while (true)
        {
            var schedules = await _worker.ListFeedSchedulesAsync(CancellationToken.None);
            if (schedules.Count == 0)
            {
                await ShowMessageAsync(
                    "No feed schedules",
                    "Select a feed and choose Schedule this feed.");
                return;
            }

            FeedSchedule? editSchedule = null;
            FeedSchedule? removeSchedule = null;
            var list = new StackPanel { Spacing = 12 };
            ContentDialog? manager = null;
            foreach (var schedule in schedules)
            {
                var schedulePanel = new StackPanel
                {
                    Spacing = 8,
                    MaxWidth = 480,
                };
                var text = new TextBlock
                {
                    Text = $"{schedule.FeedName} · feed {schedule.FeedId}\n"
                        + $"{schedule.ScheduleSummary} · {(schedule.Enabled ? "enabled" : "paused")}\n"
                        + schedule.StateSummary,
                    TextWrapping = TextWrapping.Wrap,
                };
                var edit = new Button
                {
                    Content = "Edit",
                    IsEnabled = !string.Equals(
                        schedule.State,
                        "running",
                        StringComparison.OrdinalIgnoreCase),
                };
                edit.Click += (_, _) =>
                {
                    editSchedule = schedule;
                    manager?.Hide();
                };
                var remove = new Button { Content = "Remove" };
                remove.Click += (_, _) =>
                {
                    removeSchedule = schedule;
                    manager?.Hide();
                };
                var actions = new StackPanel
                {
                    Orientation = Orientation.Horizontal,
                    Spacing = 8,
                    HorizontalAlignment = HorizontalAlignment.Right,
                };
                actions.Children.Add(edit);
                actions.Children.Add(remove);
                schedulePanel.Children.Add(text);
                schedulePanel.Children.Add(actions);
                list.Children.Add(schedulePanel);
            }
            manager = new ContentDialog
            {
                XamlRoot = ((FrameworkElement)Content).XamlRoot,
                Title = "Manage feed schedules",
                Content = CreateDialogScrollContent(list, 520),
                CloseButtonText = "Done",
            };
            await manager.ShowAsync();

            if (editSchedule is not null)
            {
                await ShowFeedScheduleEditorAsync(
                    editSchedule.FeedId,
                    editSchedule.FeedName,
                    editSchedule);
                await RefreshFeedScheduleStatusAsync();
                await RefreshLibraryAsync();
                continue;
            }
            if (removeSchedule is not null)
            {
                var confirmation = new ContentDialog
                {
                    XamlRoot = ((FrameworkElement)Content).XamlRoot,
                    Title = $"Remove schedule · {removeSchedule.FeedName}",
                    Content = CreateDialogTextContent(
                        "Retained audio, transcripts, and analysis will not be deleted."),
                    PrimaryButtonText = "Remove schedule",
                    CloseButtonText = "Cancel",
                    DefaultButton = ContentDialogButton.Close,
                };
                if (await confirmation.ShowAsync() == ContentDialogResult.Primary)
                {
                    await _worker.DeleteFeedScheduleAsync(
                        removeSchedule.Id,
                        CancellationToken.None);
                    AppendLog($"Removed the schedule for {removeSchedule.FeedName}.");
                    await RefreshFeedScheduleStatusAsync();
                    await RefreshLibraryAsync();
                }
                continue;
            }
            break;
        }
    }

    private void ConfigureFeedScheduleTimer()
    {
        if (_feedScheduleTimer is not null)
        {
            return;
        }
        _feedScheduleTimer = DispatcherQueue.CreateTimer();
        _feedScheduleTimer.Interval = TimeSpan.FromMinutes(1);
        _feedScheduleTimer.IsRepeating = true;
        _feedScheduleTimer.Tick += async (_, _) => await CheckDueFeedScheduleAsync();
        _feedScheduleTimer.Start();
        _ = CheckDueFeedScheduleAsync();
    }

    private async Task RefreshFeedScheduleStatusAsync()
    {
        if (_worker is null || FeedScheduleInfoBar is null)
        {
            return;
        }
        try
        {
            var schedules = await _worker.ListFeedSchedulesAsync(CancellationToken.None);
            var active = schedules.Where(value => value.Enabled).ToList();
            var next = active
                .Select(value => DateTimeOffset.TryParse(value.NextRunAt, out var date) ? date : (DateTimeOffset?)null)
                .Where(value => value is not null)
                .OrderBy(value => value)
                .FirstOrDefault();
            FeedScheduleInfoBar.Severity = active.Count > 0
                ? InfoBarSeverity.Success
                : InfoBarSeverity.Informational;
            FeedScheduleInfoBar.Title = active.Count == 0
                ? "No feed schedules yet"
                : $"{active.Count} daily feed schedule{(active.Count == 1 ? "" : "s")} active";
            FeedScheduleInfoBar.Message = active.Count == 0
                ? "Select a search result and schedule that specific feed. Settings → Setup can keep the app available after Windows sign-in."
                : $"Next check: {(next?.ToLocalTime().ToString("g") ?? "when due")}. Jobs reuse cached work, wait for rolling quota slots, and can run after sign-in when Start with Windows is on.";
        }
        catch (Exception exception)
        {
            FeedScheduleInfoBar.Severity = InfoBarSeverity.Warning;
            FeedScheduleInfoBar.Title = "Schedule status unavailable";
            FeedScheduleInfoBar.Message = exception.Message;
        }
    }

    private async Task CheckDueFeedScheduleAsync()
    {
        if (_worker is null
            || _checkingFeedSchedule
            || _pipelineCancellation is not null
            || _exclusiveBusy)
        {
            return;
        }
        if (_pauseScheduledJobsForSetup)
        {
            var (missingBroadcastify, missingHuggingFace) =
                MissingUnattendedSetup();
            if (missingBroadcastify || missingHuggingFace)
            {
                return;
            }
            _pauseScheduledJobsForSetup = false;
            AppendLog(
                "Unattended setup is ready; saved feed schedules can run.");
        }
        _checkingFeedSchedule = true;
        FeedSchedule? schedule = null;
        var ownsOperation = false;
        try
        {
            schedule = await _worker.ClaimDueFeedScheduleAsync(CancellationToken.None);
            if (schedule is null)
            {
                return;
            }
            if (_pipelineCancellation is not null)
            {
                await _worker.FinishFeedScheduleAsync(
                    new FeedScheduleFinishRequest
                    {
                        ScheduleId = schedule.Id,
                        DueDate = schedule.DueDate,
                        Status = "deferred",
                        Message = "A user-started job was already active.",
                    },
                    CancellationToken.None);
                return;
            }
            var pipeline = new CancellationTokenSource();
            _pipelineCancellation = pipeline;
            _activePipelineFeedIds.Add(schedule.FeedId);
            ownsOperation = true;
            SetPipelineBusy(true, $"Scheduled feed running in the background: {schedule.FeedName}");
            JobProgress.IsIndeterminate = true;
            AppendLog($"Scheduled run starting for {schedule.FeedName} ({schedule.FeedId}).");
            var accountRun = await RunScheduledJobAcrossAccountsAsync(
                schedule,
                pipeline.Token);
            var result = accountRun.Result;
            var waitingForQuota = accountRun.WaitingForQuota;
            var incomplete = (result?.MissingDays.Count ?? 0) > 0
                || (result?.PendingProcessingDays.Count ?? 0) > 0;
            await _worker.FinishFeedScheduleAsync(
                new FeedScheduleFinishRequest
                {
                    ScheduleId = schedule.Id,
                    DueDate = schedule.DueDate,
                    Status = waitingForQuota
                        ? "waiting_quota"
                        : incomplete
                            ? "deferred"
                            : "complete",
                    Message = waitingForQuota
                        ? accountRun.Message
                        : incomplete
                            ? "Some archive days were deferred; retrying retained work shortly."
                            : "Scheduled feed run completed.",
                    NextRequestAt = waitingForQuota
                        ? accountRun.NextRequestAt
                        : "",
                },
                CancellationToken.None);
        }
        catch (OperationCanceledException)
        {
            if (schedule is not null)
            {
                await _worker.FinishFeedScheduleAsync(
                    new FeedScheduleFinishRequest
                    {
                        ScheduleId = schedule.Id,
                        DueDate = schedule.DueDate,
                        Status = "canceled",
                        Message = "Scheduled run canceled.",
                    },
                    CancellationToken.None);
            }
        }
        catch (Exception exception)
        {
            AppendLog($"Scheduled feed failed: {exception.Message}");
            if (schedule is not null)
            {
                await _worker.FinishFeedScheduleAsync(
                    new FeedScheduleFinishRequest
                    {
                        ScheduleId = schedule.Id,
                        DueDate = schedule.DueDate,
                        Status = "failed",
                        Message = exception.Message,
                    },
                    CancellationToken.None);
            }
        }
        finally
        {
            if (ownsOperation && _pipelineCancellation is not null)
            {
                _pipelineCancellation.Dispose();
                _pipelineCancellation = null;
                _activePipelineFeedIds.Clear();
                JobProgress.IsIndeterminate = false;
                SetPipelineBusy(false);
                await RefreshLibraryAsync();
                await RefreshArchiveQuotaStatusAsync();
            }
            _checkingFeedSchedule = false;
            await RefreshFeedScheduleStatusAsync();
        }
    }

    private async Task<(
        JobRunResult? Result,
        bool WaitingForQuota,
        string NextRequestAt,
        string Message)> RunScheduledJobAcrossAccountsAsync(
        FeedSchedule schedule,
        CancellationToken cancellationToken)
    {
        if (_worker is null)
        {
            return (null, false, "", "The worker is unavailable.");
        }
        var automatic = string.Equals(
            schedule.AccountProfileId,
            "automatic",
            StringComparison.OrdinalIgnoreCase);
        IReadOnlyList<string> profileIds = automatic && _worker.AuthorizedAccountPoolEnabled
            ? _worker.AvailableAccountProfileIds()
            : new[]
            {
                automatic ? "default" : schedule.AccountProfileId,
            };
        var pooledAcquisition = automatic
            && _worker.AuthorizedAccountPoolEnabled
            && profileIds.Count > 1;
        var acquisitionJob = pooledAcquisition
            ? schedule.Job with
            {
                Combine = false,
                Transcribe = false,
                Diarize = false,
            }
            : schedule.Job;
        var statuses = new List<ArchiveQuotaStatus>();
        foreach (var profileId in profileIds)
        {
            var status = await _worker.GetArchiveQuotaStatusAsync(
                cancellationToken,
                profileId);
            if (status is not null)
            {
                statuses.Add(status);
            }
        }
        var eligible = statuses
            .Where(value => value.Available)
            .OrderByDescending(value => value.Remaining)
            .ThenBy(value => value.AccountProfileId, StringComparer.OrdinalIgnoreCase)
            .ToList();
        JobRunResult? lastResult = null;
        Exception? lastFailure = null;
        string? processingProfileId = null;
        foreach (var status in eligible)
        {
            try
            {
                AppendLog(
                    $"Scheduled catch-up acquisition is using account profile {status.AccountProfileId} "
                    + $"({status.Remaining}/{status.AutomatedLimit} automated requests available)."
                );
                lastResult = await RunAndAnalyzeJobAsync(
                    acquisitionJob,
                    cancellationToken,
                    pooledAcquisition ? false : schedule.Analyze,
                    status.AccountProfileId);
                processingProfileId = status.AccountProfileId;
                if (lastResult?.DownloadLimited != true)
                {
                    if (!pooledAcquisition)
                    {
                        return (
                            lastResult,
                            false,
                            "",
                            $"Scheduled feed run completed with account profile {status.AccountProfileId}.");
                    }
                    break;
                }
                var refreshed = await _worker.GetArchiveQuotaStatusAsync(
                    CancellationToken.None,
                    status.AccountProfileId);
                if (refreshed is not null)
                {
                    statuses.RemoveAll(value => value.AccountProfileId.Equals(
                        refreshed.AccountProfileId,
                        StringComparison.OrdinalIgnoreCase));
                    statuses.Add(refreshed);
                }
                if (!automatic || !_worker.AuthorizedAccountPoolEnabled)
                {
                    break;
                }
                AppendLog(
                    $"Account profile {status.AccountProfileId} reached its rolling boundary; "
                    + "continuing retained missing days on the next eligible authorized profile.");
            }
            catch (InvalidOperationException exception)
                when (automatic
                    && _worker.AuthorizedAccountPoolEnabled
                    && IsAccountAuthenticationFailure(exception.Message))
            {
                lastFailure = exception;
                AppendLog(
                    $"Account profile {status.AccountProfileId} could not authenticate; "
                    + "trying the next authorized profile without discarding retained work.");
            }
        }
        if (pooledAcquisition && (lastResult is not null || eligible.Count == 0))
        {
            processingProfileId ??= profileIds[0];
            AppendLog(
                "Available archive acquisition turns are checkpointed; "
                + "continuing combination, transcription, speaker labels, and analysis locally "
                + "while another coordinated machine may take the next website turn.");
            lastResult = await RunAndAnalyzeJobAsync(
                schedule.Job,
                cancellationToken,
                schedule.Analyze,
                processingProfileId);
            if (lastResult?.DownloadLimited != true)
            {
                return (
                    lastResult,
                    false,
                    "",
                    $"Scheduled feed run completed after pooled acquisition with account profile {processingProfileId}.");
            }
            var refreshed = await _worker.GetArchiveQuotaStatusAsync(
                CancellationToken.None,
                processingProfileId);
            if (refreshed is not null)
            {
                statuses.RemoveAll(value => value.AccountProfileId.Equals(
                    refreshed.AccountProfileId,
                    StringComparison.OrdinalIgnoreCase));
                statuses.Add(refreshed);
            }
        }
        if (lastResult is null && lastFailure is not null && eligible.Count > 0)
        {
            throw lastFailure;
        }
        var nextRequestAt = statuses
            .Select(value => DateTimeOffset.TryParse(
                value.NextRequestAt,
                out var parsed)
                ? parsed
                : (DateTimeOffset?)null)
            .Where(value => value is not null)
            .OrderBy(value => value)
            .FirstOrDefault()
            ?.ToString("O") ?? "";
        var accountCount = profileIds.Count;
        var message = automatic && _worker.AuthorizedAccountPoolEnabled
            ? $"All {accountCount} authorized account profile{(accountCount == 1 ? "" : "s")} are waiting for their next rolling archive-request slot."
            : $"Account profile {profileIds[0]} is waiting for its next rolling archive-request slot.";
        return (lastResult, true, nextRequestAt, message);
    }

    private static bool IsAccountAuthenticationFailure(string message)
    {
        var normalized = message.ToLowerInvariant();
        return normalized.Contains("authentication")
            || normalized.Contains("sign in")
            || normalized.Contains("login")
            || normalized.Contains("credentials")
            || normalized.Contains("premium archive access");
    }

    private JobRequest CreateJobRequest(
        string feedId,
        DateTime startDate,
        DateTime endDate,
        int? minimumSpeakers,
        int? maximumSpeakers,
        string? feedName = null)
    {
        return ApplyAnalysisProvider(new JobRequest
        {
            FeedId = feedId,
            FeedName = feedName?.Trim() ?? "",
            StartDate = startDate.ToString("yyyy-MM-dd"),
            EndDate = endDate.ToString("yyyy-MM-dd"),
            OutputDirectory = PersistedOutputDirectory(),
            Combine = DiarizeCheckBox.IsChecked == true || CombineToggle.IsOn,
            KeepOriginals = KeepOriginalsToggle.IsOn,
            Transcribe = TranscribeCheckBox.IsChecked == true,
            Diarize = DiarizeCheckBox.IsChecked == true,
            Model = SelectedComboValue(ModelComboBox, "turbo"),
            AsrEngine = SelectedComboValue(AsrEngineComboBox, "auto"),
            Device = SelectedComboValue(DeviceComboBox, "auto"),
            DeviceIndex = RequiredInteger(GpuIndexBox.Value, 0),
            AsrModelPath = string.IsNullOrWhiteSpace(AsrModelPathBox.Text)
                ? null
                : AsrModelPathBox.Text.Trim(),
            DiarizationEngine = SelectedComboValue(
                DiarizationEngineComboBox, "community-1"),
            DiarizationDevice = SelectedComboValue(
                DiarizationDeviceComboBox, "auto"),
            DownloadJobs = _broadcastifyRateLimitObserved
                ? 1
                : RequiredInteger(DownloadJobsBox.Value, 1),
            BatchSize = RequiredInteger(BatchSizeBox.Value, 8),
            MinimumSpeakers = minimumSpeakers,
            MaximumSpeakers = maximumSpeakers,
            HuggingFaceToken = CurrentHuggingFaceToken(),
            LanSyncEnabled = LanSyncToggle.IsOn,
            LanDiscoveryEnabled = LanDiscoveryToggle.IsOn,
            LanPeerUrls = LanPeerUrlsBox.Text
                .Split(
                    new[] { '\r', '\n', ',', ';', ' ', '\t' },
                    StringSplitOptions.RemoveEmptyEntries
                        | StringSplitOptions.TrimEntries)
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToList(),
        });
    }

    private async Task<JobRunResult?> RunAndAnalyzeJobAsync(
        JobRequest request,
        CancellationToken cancellationToken,
        bool? analyzeOverride = null,
        string accountProfileId = "default")
    {
        if (_worker is null)
        {
            return null;
        }
        await ReleaseMediaForArchiveMutationAsync();
        // Make the local source-block node reachable before the worker decides
        // whether this PC can own the shared LAN acquisition lease.
        await ConfigureLanSharingAsync();
        var jobResult = await _worker.RunJobAsync(
            request,
            HandleWorkerMessage,
            cancellationToken,
            accountProfileId);
        await AnalyzeCompletedJobAsync(
            request,
            jobResult,
            cancellationToken,
            analyzeOverride);
        return jobResult;
    }

    private async Task ReleaseMediaForArchiveMutationAsync(
        bool recreatePlayers = true)
    {
        var hadMediaSource = _libraryMediaPlayer.Source is not null
            || _incidentMediaPlayer.Source is not null
            || _areaStoryMediaPlayer.Source is not null;
        _libraryMediaPlayer.Pause();
        _libraryMediaPlayer.Source = null;
        _incidentMediaPlayer.Pause();
        _incidentMediaPlayer.Source = null;
        _areaStoryMediaPlayer.Pause();
        _areaStoryMediaPlayer.Source = null;
        _pendingIncidentClip = null;
        // Setting Source to null is not sufficient on Windows: Media Foundation
        // can retain the underlying file handle through the MediaPlayerElement.
        // Detach and dispose every instance before a directory rename. Feed
        // deletion keeps them detached until the worker has finished; attaching
        // replacements too early can keep the previous native playback graph alive.
        ReleaseMediaPlayerInstances(recreate: recreatePlayers);
        // Older builds opened folders through WinRT StorageFolder objects. Their
        // native wrappers can outlive the local C# variable and keep a directory
        // handle until finalization. Folder and transcript launches now use the
        // Windows shell; collect any legacy wrappers before the mutation.
        GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true);
        GC.WaitForPendingFinalizers();
        GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true);
        if (hadMediaSource)
        {
            LibraryAudioStatusText.Text =
                "Playback released while the archive recording is updated.";
            // Media Foundation releases its Windows file handle asynchronously.
            // Give that close a bounded moment before FFmpeg publishes a refresh.
            await Task.Delay(500);
        }
    }

    private void ReleaseMediaPlayerInstances(bool recreate)
    {
        LibraryAudioPlayer.SetMediaPlayer(null);
        IncidentPlayer.SetMediaPlayer(null);
        AreaStoryPlayer.SetMediaPlayer(null);
        _incidentMediaPlayer.MediaOpened -= IncidentMediaPlayer_MediaOpened;
        _areaStoryMediaPlayer.MediaOpened -= AreaStoryMediaPlayer_MediaOpened;
        _libraryMediaPlayer.Dispose();
        _incidentMediaPlayer.Dispose();
        _areaStoryMediaPlayer.Dispose();
        _mediaPlayersAttached = false;
        if (!recreate)
        {
            return;
        }

        RestoreMediaPlayerInstances();
    }

    private void RestoreMediaPlayerInstances()
    {
        if (_mediaPlayersAttached)
        {
            return;
        }
        _libraryMediaPlayer = new MediaPlayer();
        _incidentMediaPlayer = new MediaPlayer();
        _areaStoryMediaPlayer = new MediaPlayer();
        _incidentMediaPlayer.MediaOpened += IncidentMediaPlayer_MediaOpened;
        _areaStoryMediaPlayer.MediaOpened += AreaStoryMediaPlayer_MediaOpened;
        LibraryAudioPlayer.SetMediaPlayer(_libraryMediaPlayer);
        IncidentPlayer.SetMediaPlayer(_incidentMediaPlayer);
        AreaStoryPlayer.SetMediaPlayer(_areaStoryMediaPlayer);
        _mediaPlayersAttached = true;
    }

    private async Task AnalyzeCompletedJobAsync(
        JobRequest request,
        JobRunResult? jobResult,
        CancellationToken cancellationToken,
        bool? analyzeOverride = null)
    {
        if (_worker is null)
        {
            return;
        }
        if (!request.Transcribe
            || !request.Combine
            || !(analyzeOverride ?? AnalyzeAfterJobCheckBox.IsChecked == true))
        {
            return;
        }

        DayReport? latestReport = null;
        var transcriptDays = jobResult?.Days
            .Where(day => day.Transcripts.Count > 0)
            .ToList() ?? [];
        foreach (var day in transcriptDays)
        {
            cancellationToken.ThrowIfCancellationRequested();
            AppendLog($"Waiting for the shared analysis slot for feed {request.FeedId} on {day.ArchiveDate}…");
            await _analysisOperationGate.WaitAsync(cancellationToken);
            try
            {
                AppendLog($"Analyzing feed {request.FeedId} for {day.ArchiveDate}…");
                try
                {
                    latestReport = await _worker.AnalyzeDayAsync(
                        ApplyAnalysisProvider(new AnalysisRequest
                        {
                            FeedId = request.FeedId,
                            ArchiveDate = day.ArchiveDate,
                            OutputDirectory = request.OutputDirectory,
                        }),
                        HandleWorkerMessage,
                        cancellationToken);
                }
                catch (Exception exception) when (exception is not OperationCanceledException)
                {
                    throw new InvalidOperationException(
                        $"Local incident analysis failed for feed {request.FeedId} on "
                        + $"{day.ArchiveDate}. Its downloads, combined audio, transcript, "
                        + "and diarization remain saved and will be reused when you retry. "
                        + exception.Message,
                        exception);
                }
            }
            finally
            {
                _analysisOperationGate.Release();
            }
        }
        if (transcriptDays.Count == 0)
        {
            AppendLog("No completed transcripts were available for analysis.");
        }
        SelectReviewFeed(request.FeedId, clearChatWhenChanged: false);
        if (latestReport is not null)
        {
            ApplyReport(latestReport);
        }
        await RefreshAnalysisDaysAsync();
    }

    private void Cancel_Click(object sender, RoutedEventArgs e)
    {
        _pipelineCancellation?.Cancel();
        _operationCancellation?.Cancel();
        _questionCancellation?.Cancel();
    }

    private async Task LoadDiagnosticsAndDaysAsync()
    {
        if (_worker is null)
        {
            return;
        }
        var loadVersion = ++_diagnosticsLoadVersion;
        try
        {
            using var cancellation = new CancellationTokenSource(TimeSpan.FromSeconds(30));
            var diagnostics = await _worker.GetDiagnosticsAsync(
                cancellation.Token,
                CreateAsrSelfTestRequest());
            if (loadVersion != _diagnosticsLoadVersion)
            {
                return;
            }
            if (diagnostics is JsonElement value)
            {
                var cuda = value.TryGetProperty("cuda_available", out var cudaValue)
                    && cudaValue.GetBoolean();
                _cudaAvailable = cuda;
                var gpu = value.TryGetProperty("cuda_devices", out var devices)
                    && devices.GetArrayLength() > 0
                    ? devices[0].GetString()
                    : null;
                var llamaReady = value.TryGetProperty("llama_server", out var llama)
                    && llama.ValueKind == JsonValueKind.String
                    && !string.IsNullOrWhiteSpace(llama.GetString());
                var tokenReady = value.TryGetProperty("huggingface_token_configured", out var token)
                    && token.GetBoolean();
                _huggingFaceTokenConfigured = tokenReady;
                RefreshHuggingFaceCredentialUi();
                var environmentCredentials = value.TryGetProperty(
                    "broadcastify_credentials_configured", out var credentials)
                    && credentials.GetBoolean();
                var savedSession = value.TryGetProperty("saved_session_available", out var session)
                    && session.GetBoolean();
                var environmentFile = value.TryGetProperty("environment_file", out var environment)
                    ? environment.GetString() ?? ""
                    : "";
                _hardwareProfiles.Clear();
                if (value.TryGetProperty("accelerators", out var acceleratorValue)
                    && acceleratorValue.TryGetProperty("profiles", out var profileValues))
                {
                    var profiles = profileValues.Deserialize<List<HardwareProfileStatus>>() ?? [];
                    foreach (var profile in profiles)
                    {
                        _hardwareProfiles.Add(profile);
                    }
                    if (acceleratorValue.TryGetProperty("speaker_labels", out var speakerLabels))
                    {
                        _pyannotePackageInstalled = speakerLabels.TryGetProperty(
                            "package_installed", out var packageInstalled)
                            && packageInstalled.GetBoolean();
                        _pyannoteAccessConfigured = speakerLabels.TryGetProperty(
                            "access_configured", out var accessConfigured)
                            && accessConfigured.GetBoolean();
                        if (speakerLabels.TryGetProperty(
                                "portable", out var portableSpeakers))
                        {
                            _portableDiarizationRuntimeInstalled =
                                portableSpeakers.TryGetProperty(
                                    "runtime_installed", out var runtimeInstalled)
                                && runtimeInstalled.GetBoolean();
                            _portableDiarizationModelReady =
                                portableSpeakers.TryGetProperty(
                                    "model_ready", out var modelReady)
                                && modelReady.GetBoolean();
                        }
                    }
                }
                var selectedProfileId =
                    (HardwareProfileComboBox.SelectedItem as ComboBoxItem)?.Tag?.ToString() ?? "auto";
                var selectedProfile = _hardwareProfiles.FirstOrDefault(
                    profile => profile.Id == selectedProfileId);
                if (selectedProfile is null)
                {
                    DiagnosticsInfoBar.Severity = InfoBarSeverity.Warning;
                    DiagnosticsInfoBar.Title = "Local processing needs setup";
                    DiagnosticsInfoBar.Message =
                        $"GPU: {(gpu ?? (cuda ? "CUDA available" : "not available"))} · "
                        + $"llama.cpp: {(llamaReady ? "detected" : "missing")} · "
                        + "speaker engines: "
                        + $"Community-1 {(_pyannoteAccessConfigured ? "configured" : "missing")}, "
                        + $"portable preview {(_portableDiarizationModelReady ? "ready" : _portableDiarizationRuntimeInstalled ? "runtime only" : "missing")}";
                }
                ApplySelectedHardwareProfileDescription();
                SettingsEnvironmentText.Text = string.IsNullOrWhiteSpace(environmentFile)
                    ? "Private build environment: no .env was loaded. Repository .env and private build output are both supported."
                    : $"Private environment loaded from {environmentFile}. "
                      + $"Broadcastify login: {(environmentCredentials ? "configured" : "missing")} · "
                      + $"saved session cookie: {(savedSession ? "available" : "not present")}.";
                if (CredentialStore.TryLoad() is null && (environmentCredentials || savedSession))
                {
                    AuthInfoBar.Severity = InfoBarSeverity.Success;
                    AuthInfoBar.Title = "Archive credentials configured";
                    AuthInfoBar.Message = environmentCredentials
                        ? "The private environment can refresh premium archive access automatically."
                        : "A saved Broadcastify session cookie is available.";
                }
                _archiveAccessConfigured = _archiveAccessConfigured
                    || CredentialStore.TryLoad() is not null
                    || environmentCredentials
                    || savedSession;
                _diagnosticsLoaded = true;
                RefreshStorageReadiness();
                UpdateSetupSummary();
            }
            await RefreshAnalysisDaysAsync();
            if (loadVersion != _diagnosticsLoadVersion)
            {
                return;
            }
            await RefreshAreaProfilesAsync();
            if (loadVersion == _diagnosticsLoadVersion
                && _operationCancellation is null)
            {
                StatusText.Text = "Ready";
            }
        }
        catch (Exception exception)
        {
            if (loadVersion != _diagnosticsLoadVersion)
            {
                return;
            }
            _diagnosticsLoaded = true;
            DiagnosticsInfoBar.Severity = InfoBarSeverity.Warning;
            DiagnosticsInfoBar.Title = "Diagnostics unavailable";
            DiagnosticsInfoBar.Message = exception.Message;
            AppendLog($"Diagnostics: {exception.Message}");
            UpdateSetupSummary();
        }
    }

    private void SyncAnalysisFeedSelection()
    {
        if (AnalysisFeedCombo is null || AnalysisFeedBox is null)
        {
            return;
        }
        var feedId = AnalysisFeedBox.Text.Trim();
        var match = _libraryFeeds.FirstOrDefault(value => value.FeedId == feedId)
            ?? _libraryFeeds.FirstOrDefault();
        _syncingAnalysisFeedSelection = true;
        try
        {
            AnalysisFeedCombo.SelectedItem = match;
            if (match is not null)
            {
                AnalysisFeedBox.Text = match.FeedId;
                ArchiveChatFeedText.Text = $"{match.FeedName} · feed {match.FeedId}";
            }
            else
            {
                ArchiveChatFeedText.Text = "No retained or scheduled feeds are available yet.";
            }
        }
        finally
        {
            _syncingAnalysisFeedSelection = false;
        }
    }

    private void SelectReviewFeed(
        string feedId,
        bool clearChatWhenChanged = true)
    {
        var normalized = feedId.Trim();
        if (string.IsNullOrWhiteSpace(normalized))
        {
            return;
        }
        var changed = !string.Equals(
            AnalysisFeedBox.Text.Trim(),
            normalized,
            StringComparison.Ordinal);
        AnalysisFeedBox.Text = normalized;
        SyncAnalysisFeedSelection();
        if (changed && clearChatWhenChanged)
        {
            _archiveChatMessages.Clear();
            ArchiveChatStatusText.Text = "New feed selected; start a new evidence chat.";
            ResetQuestionCoverage();
        }
    }

    private async void AnalysisFeedCombo_SelectionChanged(
        object sender,
        SelectionChangedEventArgs e)
    {
        if (_syncingAnalysisFeedSelection
            || AnalysisFeedCombo.SelectedItem is not LibraryFeedCoverage feed)
        {
            return;
        }
        var changed = !string.Equals(
            AnalysisFeedBox.Text.Trim(),
            feed.FeedId,
            StringComparison.Ordinal);
        AnalysisFeedBox.Text = feed.FeedId;
        ArchiveChatFeedText.Text = feed.FeedLabel;
        _lastReviewFeedId = feed.FeedId;
        ScheduleSettingsSave();
        if (changed)
        {
            _archiveChatMessages.Clear();
            ArchiveChatStatusText.Text = "New feed selected; start a new evidence chat.";
            ResetQuestionCoverage();
        }
        await RefreshAnalysisDaysAsync();
        UpdateCommandAvailability();
    }

    private async void RefreshAnalysis_Click(object sender, RoutedEventArgs e) =>
        await RefreshAnalysisDaysAsync();

    private async Task RefreshAnalysisDaysAsync()
    {
        if (_worker is null)
        {
            return;
        }
        using var cancellation = new CancellationTokenSource(TimeSpan.FromSeconds(30));
        var days = await _worker.ListAnalysisDaysAsync(
            string.IsNullOrWhiteSpace(AnalysisFeedBox.Text) ? null : AnalysisFeedBox.Text.Trim(),
            cancellation.Token);
        _analysisDays.Clear();
        foreach (var day in days)
        {
            _analysisDays.Add(day);
        }
        if (_analysisDays.Count > 0)
        {
            var preferred = _analysisDays.FirstOrDefault(value =>
                string.Equals(value.FeedId, _lastReviewFeedId, StringComparison.Ordinal)
                && string.Equals(value.ArchiveDate, _lastReviewDate, StringComparison.Ordinal));
            AnalysisDaysList.SelectedItem = preferred ?? _analysisDays[0];
        }
        else
        {
            _currentReport = null;
            _visibleIncidents.Clear();
            SummaryText.Text = "No analyzed days were found for this feed.";
        }
        AppendLog($"Loaded {days.Count} saved analysis day(s).");
    }

    private async void AnalysisDays_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (AnalysisDaysList.SelectedItem is AnalysisDay day)
        {
            _lastReviewFeedId = day.FeedId;
            _lastReviewDate = day.ArchiveDate;
            ScheduleSettingsSave();
            if (DateTimeOffset.TryParse(day.ArchiveDate, out var selectedDate))
            {
                QuestionStartDatePicker.Date = selectedDate;
                QuestionEndDatePicker.Date = selectedDate;
                WeekEndingPicker.Date = selectedDate;
            }
            await LoadDayReportAsync(day);
        }
    }

    private async Task LoadDayReportAsync(AnalysisDay day)
    {
        if (_worker is null)
        {
            return;
        }
        try
        {
            using var cancellation = new CancellationTokenSource(TimeSpan.FromSeconds(30));
            var report = await _worker.GetDayReportAsync(
                day.FeedId, day.ArchiveDate, cancellation.Token);
            if (report is not null)
            {
                ApplyReport(report);
            }
        }
        catch (Exception exception)
        {
            await ShowErrorAsync(exception);
        }
    }

    private void ApplyReport(DayReport report)
    {
        _incidentMediaPlayer.Pause();
        _incidentMediaPlayer.Source = null;
        _pendingIncidentClip = null;
        _currentReport = report;
        if (report.AnalysisUpdateRequired)
        {
            PlaybackStatusText.Text =
                "Saved incident claims are hidden until the retained transcript is reanalyzed.";
            SummaryText.Text =
                "Saved analysis no longer matches the retained audio, transcript, or current evidence rules. Finish the local day to rebuild it without downloading the archive again.";
        }
        else
        {
            PlaybackStatusText.Text = string.IsNullOrWhiteSpace(report.AudioPath)
                ? "Combined audio is unavailable for this saved day."
                : "Play or export an exact local evidence clip for an incident.";
            SummaryText.Text = string.IsNullOrWhiteSpace(report.Summary)
                ? "This day has not been summarized yet."
                : report.Summary;
        }
        ApplyIncidentFilter();
    }

    private void PriorityFilter_SelectionChanged(object sender, SelectionChangedEventArgs e) =>
        ApplyIncidentFilter();

    private void IncidentSearch_TextChanged(object sender, TextChangedEventArgs e) =>
        ApplyIncidentFilter();

    private void ApplyIncidentFilter()
    {
        _visibleIncidents.Clear();
        if (_currentReport is null)
        {
            return;
        }
        var minimumPriority = 3;
        if (PriorityFilterCombo.SelectedItem is ComboBoxItem item
            && int.TryParse(item.Tag?.ToString(), out var parsed))
        {
            minimumPriority = parsed;
        }
        var search = IncidentSearchBox?.Text.Trim() ?? "";
        if (!string.IsNullOrWhiteSpace(search))
        {
            minimumPriority = 1;
        }
        foreach (var incident in _currentReport.Incidents
                     .Where(value => value.Priority >= minimumPriority)
                     .Where(value =>
                         string.IsNullOrWhiteSpace(search)
                         || string.Join(
                                 " ",
                                 value.EventType,
                                 value.Title,
                                 value.Summary,
                                 value.Location ?? "",
                                 value.ArchiveTime)
                             .Contains(search, StringComparison.OrdinalIgnoreCase))
                     .OrderByDescending(value => value.Priority)
                     .ThenBy(value => value.ArchiveTime))
        {
            _visibleIncidents.Add(incident);
        }
    }

    private async void PlayIncident_Click(object sender, RoutedEventArgs e) =>
        await PlayIncidentAsync(sender, includeSurroundingContext: false);

    private async void PlayIncidentContext_Click(object sender, RoutedEventArgs e) =>
        await PlayIncidentAsync(sender, includeSurroundingContext: true);

    private async Task PlayIncidentAsync(object sender, bool includeSurroundingContext)
    {
        if (sender is not FrameworkElement { DataContext: IncidentRecord incident }
            || _worker is null)
        {
            return;
        }
        if (_currentReport is not null && IsFeedPipelineBusy(_currentReport.FeedId))
        {
            PlaybackStatusText.Text =
                "This feed is being updated by the background pipeline. Exact-clip playback is temporarily held to avoid locking its combined audio; other feeds remain playable.";
            return;
        }

        var button = sender as Button;
        if (button is not null)
        {
            button.IsEnabled = false;
        }
        try
        {
            var clip = await PrepareIncidentClipAsync(
                incident,
                includeSurroundingContext);
            if (clip is null)
            {
                return;
            }
            _pendingIncidentClip = clip;
            _incidentMediaPlayer.Pause();
            _incidentMediaPlayer.Source = MediaSource.CreateFromUri(
                new Uri(Path.GetFullPath(clip.Path)));
        }
        catch (Exception exception)
        {
            await ShowErrorAsync(exception);
        }
        finally
        {
            if (button is not null)
            {
                button.IsEnabled = true;
            }
        }
    }

    private void IncidentMediaPlayer_MediaOpened(MediaPlayer sender, object args)
    {
        DispatcherQueue.TryEnqueue(() =>
        {
            sender.Play();
            PlaybackStatusText.Text = _pendingIncidentClip is null
                ? "Playing the exact local evidence clip."
                : _pendingIncidentClip.ClipKind == "context"
                    ? $"Playing {_pendingIncidentClip.DurationSeconds / 60:0.0} minutes of surrounding radio traffic "
                      + $"for I{_pendingIncidentClip.IncidentId}; only the incident's exact clip is cited evidence."
                    : $"Playing I{_pendingIncidentClip.IncidentId} evidence from {_pendingIncidentClip.ArchiveTime} "
                      + $"({_pendingIncidentClip.DurationSeconds:0} seconds with compact context).";
        });
    }

    private async Task<IncidentClip?> PrepareIncidentClipAsync(
        IncidentRecord incident,
        bool includeSurroundingContext = false)
    {
        if (_worker is null)
        {
            return null;
        }
        if (_currentReport is not null && IsFeedPipelineBusy(_currentReport.FeedId))
        {
            throw new InvalidOperationException(
                "This feed is being updated by the background pipeline. Clip playback and export are temporarily held so the pipeline can replace its combined audio safely; other feeds remain usable.");
        }
        PlaybackStatusText.Text = includeSurroundingContext
            ? $"Preparing surrounding radio traffic for I{incident.Id}…"
            : $"Preparing the exact cited radio segment for I{incident.Id}…";
        using var cancellation = new CancellationTokenSource(TimeSpan.FromMinutes(2));
        var clip = await _worker.GetIncidentClipAsync(
            incident.Id,
            includeSurroundingContext,
            cancellation.Token);
        if (clip is null || string.IsNullOrWhiteSpace(clip.Path) || !File.Exists(clip.Path))
        {
            throw new FileNotFoundException(
                $"The local evidence clip for incident I{incident.Id} could not be created.");
        }
        return clip;
    }

    private async void ExportIncidentClip_Click(object sender, RoutedEventArgs e)
    {
        if (sender is not FrameworkElement { DataContext: IncidentRecord incident })
        {
            return;
        }
        var button = sender as Button;
        if (button is not null)
        {
            button.IsEnabled = false;
        }
        try
        {
            var clip = await PrepareIncidentClipAsync(incident);
            if (clip is null)
            {
                return;
            }
            if (await ExportClipAsync(
                    clip.Path,
                    $"feed-{clip.FeedId}_{clip.ArchiveDate}_I{clip.IncidentId}_evidence"))
            {
                PlaybackStatusText.Text = $"Exported the exact evidence clip for I{incident.Id}.";
            }
        }
        catch (Exception exception)
        {
            await ShowErrorAsync(exception);
        }
        finally
        {
            if (button is not null)
            {
                button.IsEnabled = true;
            }
        }
    }

    private void PlayStoryEvidence_Click(object sender, RoutedEventArgs e)
    {
        if (sender is not FrameworkElement { DataContext: AreaStoryReference reference })
        {
            return;
        }
        if (IsFeedPipelineBusy(reference.FeedId))
        {
            AreaPlaybackStatusText.Text =
                "This feed is being updated by the background pipeline, so its clip is temporarily held to avoid a Windows file lock.";
            return;
        }
        if (!reference.ClipAvailable || string.IsNullOrWhiteSpace(reference.ClipPath)
            || !File.Exists(reference.ClipPath))
        {
            AreaPlaybackStatusText.Text = "This evidence clip is unavailable. The retained transcript quote and source record remain visible.";
            return;
        }

        _areaStoryMediaPlayer.Pause();
        AreaPlaybackStatusText.Text = $"Loading {reference.FeedName} evidence from {reference.ArchiveTime}…";
        _areaStoryMediaPlayer.Source = MediaSource.CreateFromUri(
            new Uri(Path.GetFullPath(reference.ClipPath)));
    }

    private void AreaStoryMediaPlayer_MediaOpened(MediaPlayer sender, object args)
    {
        DispatcherQueue.TryEnqueue(() =>
        {
            sender.Play();
            AreaPlaybackStatusText.Text = "Playing the timestamped local evidence clip. Verify the ASR quote by ear before publishing.";
        });
    }

    private async void ExportStoryEvidence_Click(object sender, RoutedEventArgs e)
    {
        if (sender is not FrameworkElement { DataContext: AreaStoryReference reference }
            || !reference.ClipAvailable
            || string.IsNullOrWhiteSpace(reference.ClipPath)
            || !File.Exists(reference.ClipPath))
        {
            AreaPlaybackStatusText.Text = "This evidence clip is unavailable for export.";
            return;
        }
        try
        {
            if (await ExportClipAsync(
                    reference.ClipPath,
                    $"feed-{reference.FeedId}_I{reference.IncidentId}_evidence"))
            {
                AreaPlaybackStatusText.Text = $"Exported the evidence clip for incident I{reference.IncidentId}.";
            }
        }
        catch (Exception exception)
        {
            await ShowErrorAsync(exception);
        }
    }

    private async Task<bool> ExportClipAsync(string sourcePath, string suggestedName)
    {
        var picker = new FileSavePicker
        {
            SuggestedStartLocation = PickerLocationId.Downloads,
            SuggestedFileName = suggestedName,
        };
        picker.FileTypeChoices.Add("MP3 audio", [".mp3"]);
        InitializeWithWindow.Initialize(picker, WindowNative.GetWindowHandle(this));
        var destination = await picker.PickSaveFileAsync();
        if (destination is null)
        {
            return false;
        }
        var source = await StorageFile.GetFileFromPathAsync(Path.GetFullPath(sourcePath));
        await source.CopyAndReplaceAsync(destination);
        return true;
    }

    private async void AnalyzeSelected_Click(object sender, RoutedEventArgs e)
    {
        if (_worker is null || AnalysisDaysList.SelectedItem is not AnalysisDay day)
        {
            await ShowMessageAsync("Saved day required", "Select a saved transcript day first.");
            return;
        }
        _operationCancellation = new CancellationTokenSource();
        SetBusy(true, $"Analyzing {day.ArchiveDate}…", jobRunning: true);
        JobProgress.IsIndeterminate = true;
        try
        {
            var report = await _worker.AnalyzeDayAsync(
                ApplyAnalysisProvider(new AnalysisRequest
                {
                    FeedId = day.FeedId,
                    ArchiveDate = day.ArchiveDate,
                    OutputDirectory = string.IsNullOrWhiteSpace(OutputFolderBox.Text)
                        ? "archives"
                        : OutputFolderBox.Text.Trim(),
                }),
                HandleWorkerMessage,
                _operationCancellation.Token);
            if (report is not null)
            {
                ApplyReport(report);
            }
            await RefreshAnalysisDaysAsync();
        }
        catch (OperationCanceledException)
        {
            StatusText.Text = "Cancelled";
            AppendLog("Analysis cancelled.");
        }
        catch (Exception exception)
        {
            await ShowErrorAsync(exception);
        }
        finally
        {
            _operationCancellation.Dispose();
            _operationCancellation = null;
            JobProgress.IsIndeterminate = false;
            SetBusy(false);
            await RefreshLibraryAsync();
        }
    }

    private async void ReloadReport_Click(object sender, RoutedEventArgs e)
    {
        if (AnalysisDaysList.SelectedItem is AnalysisDay day)
        {
            await LoadDayReportAsync(day);
        }
    }

    private void QuestionRange_DateChanged(
        object sender,
        DatePickerValueChangedEventArgs e) =>
        ResetQuestionCoverage();

    private void ResetQuestionCoverage()
    {
        if (QuestionCoverageInfoBar is null)
        {
            return;
        }
        QuestionCoverageInfoBar.Severity = InfoBarSeverity.Informational;
        QuestionCoverageInfoBar.Title = "Downloaded coverage not checked";
        QuestionCoverageInfoBar.Message =
            "Check retained coverage before asking; this uses only local files and records.";
    }

    private void ShowQuestionCoverage(ArchiveQuestionCoverage coverage)
    {
        QuestionCoverageInfoBar.Severity = coverage.QuestionReadyDayCount == 0
            ? InfoBarSeverity.Warning
            : coverage.CompleteCoverage
                ? InfoBarSeverity.Success
                : InfoBarSeverity.Informational;
        QuestionCoverageInfoBar.Title =
            $"{coverage.QuestionReadyDayCount:N0}/{coverage.RequestedDayCount:N0} day(s) ready for questions";
        QuestionCoverageInfoBar.Message = coverage.DisplaySummary
            + " Answers use only question-ready dates and treat every other date as a coverage gap.";
    }

    private async Task<ArchiveQuestionCoverage?> RefreshQuestionCoverageAsync()
    {
        if (_worker is null)
        {
            return null;
        }
        var feedId = AnalysisFeedBox.Text.Trim();
        var startDate = QuestionStartDatePicker.Date.Date;
        var endDate = QuestionEndDatePicker.Date.Date;
        if (string.IsNullOrWhiteSpace(feedId))
        {
            ResetQuestionCoverage();
            QuestionCoverageInfoBar.Severity = InfoBarSeverity.Warning;
            QuestionCoverageInfoBar.Title = "Choose a feed";
            return null;
        }
        if (startDate > endDate)
        {
            ResetQuestionCoverage();
            QuestionCoverageInfoBar.Severity = InfoBarSeverity.Warning;
            QuestionCoverageInfoBar.Title = "Invalid date range";
            QuestionCoverageInfoBar.Message = "Start date must be on or before end date.";
            return null;
        }
        QuestionCoverageInfoBar.Severity = InfoBarSeverity.Informational;
        QuestionCoverageInfoBar.Title = "Checking retained coverage";
        QuestionCoverageInfoBar.Message =
            "Reading local files and evidence records; no archive request is made.";
        try
        {
            using var cancellation = new CancellationTokenSource(TimeSpan.FromMinutes(2));
            var coverage = await _worker.GetArchiveQuestionCoverageAsync(
                PersistedOutputDirectory(),
                feedId,
                startDate.ToString("yyyy-MM-dd"),
                endDate.ToString("yyyy-MM-dd"),
                cancellation.Token);
            if (coverage is null)
            {
                throw new InvalidOperationException(
                    "The worker did not return retained coverage.");
            }
            ShowQuestionCoverage(coverage);
            return coverage;
        }
        catch (Exception exception)
        {
            QuestionCoverageInfoBar.Severity = InfoBarSeverity.Warning;
            QuestionCoverageInfoBar.Title = "Coverage check failed";
            QuestionCoverageInfoBar.Message = exception.Message;
            AppendLog($"Question coverage: {exception.Message}");
            return null;
        }
    }

    private async Task<bool> ApplyQuestionMonthAsync()
    {
        var selected = QuestionMonthPicker.Date.Date;
        var monthStart = new DateTime(selected.Year, selected.Month, 1);
        var today = DateTime.Today;
        if (monthStart > today)
        {
            await ShowMessageAsync(
                "Future month unavailable",
                "Choose the current month or an earlier month with retained archive data.");
            return false;
        }
        var monthEnd = monthStart.AddMonths(1).AddDays(-1);
        if (monthEnd > today)
        {
            monthEnd = today;
        }
        QuestionStartDatePicker.Date = new DateTimeOffset(monthStart);
        QuestionEndDatePicker.Date = new DateTimeOffset(monthEnd);
        await RefreshQuestionCoverageAsync();
        return true;
    }

    private async void UseQuestionMonth_Click(object sender, RoutedEventArgs e) =>
        await ApplyQuestionMonthAsync();

    private async Task<bool> ApplyEntireQuestionFeedAsync()
    {
        if (_worker is null)
        {
            return false;
        }
        var feedId = AnalysisFeedBox.Text.Trim();
        if (string.IsNullOrWhiteSpace(feedId))
        {
            await ShowMessageAsync(
                "Feed required",
                "Choose a named feed before selecting its entire downloaded span.");
            return false;
        }
        QuestionCoverageInfoBar.Severity = InfoBarSeverity.Informational;
        QuestionCoverageInfoBar.Title = "Finding the retained feed span";
        QuestionCoverageInfoBar.Message =
            "Reading local files and evidence records; no archive request is made.";
        try
        {
            using var cancellation = new CancellationTokenSource(TimeSpan.FromMinutes(2));
            var coverage = await _worker.GetEntireFeedQuestionCoverageAsync(
                PersistedOutputDirectory(),
                feedId,
                cancellation.Token);
            if (coverage is null
                || !DateTimeOffset.TryParse(coverage.StartDate, out var startDate)
                || !DateTimeOffset.TryParse(coverage.EndDate, out var endDate))
            {
                throw new InvalidOperationException(
                    "The worker did not return a valid retained feed span.");
            }
            QuestionStartDatePicker.Date = startDate;
            QuestionEndDatePicker.Date = endDate;
            ShowQuestionCoverage(coverage);
            return true;
        }
        catch (Exception exception)
        {
            QuestionCoverageInfoBar.Severity = InfoBarSeverity.Warning;
            QuestionCoverageInfoBar.Title = "Entire-feed scope unavailable";
            QuestionCoverageInfoBar.Message = exception.Message;
            AppendLog($"Entire-feed question scope: {exception.Message}");
            return false;
        }
    }

    private async void UseEntireQuestionFeed_Click(object sender, RoutedEventArgs e) =>
        await ApplyEntireQuestionFeedAsync();

    private async void CheckQuestionCoverage_Click(object sender, RoutedEventArgs e) =>
        await RefreshQuestionCoverageAsync();

    private async void Ask_Click(object sender, RoutedEventArgs e)
    {
        if (_worker is null || _questionCancellation is not null)
        {
            return;
        }
        var feedId = AnalysisFeedBox.Text.Trim();
        var question = QuestionBox.Text.Trim();
        var startDate = QuestionStartDatePicker.Date.Date;
        var endDate = QuestionEndDatePicker.Date.Date;
        if (string.IsNullOrWhiteSpace(feedId) || string.IsNullOrWhiteSpace(question))
        {
            await ShowMessageAsync("Question incomplete", "Choose a named feed and enter a message.");
            return;
        }
        if (startDate > endDate)
        {
            await ShowMessageAsync("Invalid date range", "Start date must be on or before end date.");
            return;
        }
        var coverage = await RefreshQuestionCoverageAsync();
        if (coverage is null)
        {
            return;
        }
        if (coverage.QuestionReadyDayCount == 0)
        {
            await ShowMessageAsync(
                "No question-ready days",
                "This range has no current retained transcripts. Finish local processing for at least one downloaded day, then ask again.");
            return;
        }

        var history = _archiveChatMessages
            .TakeLast(8)
            .Select(value => new ArchiveConversationTurn
            {
                Role = value.Role,
                Content = value.Content,
            })
            .ToList();
        _archiveChatMessages.Add(new ArchiveChatMessage
        {
            Role = "user",
            Content = question,
        });
        QuestionBox.Text = "";
        ArchiveChatList.ScrollIntoView(_archiveChatMessages[^1]);
        var questionCancellation = new CancellationTokenSource();
        _questionCancellation = questionCancellation;
        _activeQuestionFeedId = feedId;
        ArchiveChatStatusText.Text =
            $"Retrieving evidence with {SelectedAnalysisProviderDisplayName()}… The archive/transcription pipeline continues independently.";
        UpdateCommandAvailability();
        var analysisSlotHeld = false;
        try
        {
            ArchiveChatStatusText.Text =
                "Waiting for the shared analysis slot; downloading/transcription continues while local model work is serialized.";
            await _analysisOperationGate.WaitAsync(questionCancellation.Token);
            analysisSlotHeld = true;
            ArchiveChatStatusText.Text =
                $"Retrieving evidence with {SelectedAnalysisProviderDisplayName()}…";
            var answer = await _worker.AskArchiveAsync(
                ApplyAnalysisProvider(new ArchiveQuestionRequest
                {
                    FeedId = feedId,
                    StartDate = startDate.ToString("yyyy-MM-dd"),
                    EndDate = endDate.ToString("yyyy-MM-dd"),
                    OutputDirectory = PersistedOutputDirectory(),
                    Question = question,
                    History = history,
                }),
                HandleQuestionWorkerMessage,
                questionCancellation.Token);
            _archiveChatMessages.Add(answer is null
                ? new ArchiveChatMessage
                {
                    Role = "assistant",
                    Content = "No answer was returned.",
                }
                : new ArchiveChatMessage
                {
                    Role = "assistant",
                    Content = answer.Answer,
                    EvidenceIds = answer.EvidenceIds,
                    Limitations = answer.Limitations,
                    Coverage = answer.Coverage,
                });
            if (answer is not null)
            {
                ShowQuestionCoverage(answer.Coverage);
            }
            ArchiveChatList.ScrollIntoView(_archiveChatMessages[^1]);
            ArchiveChatStatusText.Text =
                "Answer complete. Follow-up messages retain the recent chat context but must cite fresh archive evidence.";
        }
        catch (OperationCanceledException)
        {
            ArchiveChatStatusText.Text = "Question cancelled; the background archive pipeline was not stopped.";
        }
        catch (Exception exception)
        {
            ArchiveChatStatusText.Text = $"Question failed: {exception.Message}";
            AppendLog($"Archive chat: {exception.Message}");
        }
        finally
        {
            if (analysisSlotHeld)
            {
                _analysisOperationGate.Release();
            }
            questionCancellation.Dispose();
            if (ReferenceEquals(_questionCancellation, questionCancellation))
            {
                _questionCancellation = null;
            }
            _activeQuestionFeedId = "";
            UpdateCommandAvailability();
        }
    }

    private void ClearArchiveChat_Click(object sender, RoutedEventArgs e)
    {
        if (_questionCancellation is not null)
        {
            _questionCancellation.Cancel();
        }
        _archiveChatMessages.Clear();
        ArchiveChatStatusText.Text = "New chat ready. Saved evidence and prior Q&A audit records were not deleted.";
        ArchiveChatScroll.ChangeView(null, 0, null, true);
    }

    private void AskShotsExample_Click(object sender, RoutedEventArgs e)
    {
        QuestionBox.Text = "How many distinct reports of shots fired or gunfire were there in this period, and what evidence supports each one?";
        QuestionBox.Focus(FocusState.Programmatic);
    }

    private void AskCraziestExample_Click(object sender, RoutedEventArgs e)
    {
        QuestionBox.Text = "What were the most unusual or surprising reported events in this period? Separate verified radio reports from uncertain interpretation.";
        QuestionBox.Focus(FocusState.Programmatic);
    }

    private void AskImportantExample_Click(object sender, RoutedEventArgs e)
    {
        var end = QuestionEndDatePicker.Date.Date;
        QuestionStartDatePicker.Date = end.AddDays(-6);
        QuestionBox.Text = "What were the most important reported events in the past week, ranked by public-safety significance with citations and coverage gaps?";
        QuestionBox.Focus(FocusState.Programmatic);
    }

    private async void AskMonthExample_Click(object sender, RoutedEventArgs e)
    {
        if (!await ApplyQuestionMonthAsync())
        {
            return;
        }
        QuestionBox.Text =
            "Across the retained days in this month, what were the most important reported events and recurring patterns? State the archive date and time for every event mentioned, rank them by public-safety significance, cite the supporting evidence, and clearly separate coverage gaps from days with no supported reports.";
        QuestionBox.Focus(FocusState.Programmatic);
    }

    private async void AskFeedHotspotsExample_Click(object sender, RoutedEventArgs e)
    {
        if (!await ApplyEntireQuestionFeedAsync())
        {
            return;
        }
        QuestionBox.Text =
            "Across the entire downloaded feed, where and when do supported incident records cluster? Rank repeated extracted locations, categories, weekdays, and six-hour time windows using exact aggregate counts and citations. Include exact archive dates and times for representative events. Treat missing or unprocessed dates as coverage limits, and do not claim population-normalized crime rates or trends.";
        QuestionBox.Focus(FocusState.Programmatic);
    }

    private async void GenerateWeek_Click(object sender, RoutedEventArgs e)
    {
        if (_worker is null)
        {
            return;
        }
        var feedId = AnalysisFeedBox.Text.Trim();
        if (string.IsNullOrWhiteSpace(feedId))
        {
            await ShowMessageAsync("Feed required", "Enter a feed ID for the weekly brief.");
            return;
        }

        _operationCancellation = new CancellationTokenSource();
        var weekEnding = WeekEndingPicker.Date.Date.ToString("yyyy-MM-dd");
        SetBusy(true, $"Summarizing week ending {weekEnding}…", jobRunning: true);
        JobProgress.IsIndeterminate = true;
        WeekCoverageText.Text = "Loading seven-day evidence…";
        WeekSummaryText.Text = $"Working with {SelectedAnalysisProviderDisplayName()}…";
        try
        {
            var report = await _worker.SummarizeWeekAsync(
                ApplyAnalysisProvider(new WeeklySummaryRequest
                {
                    FeedId = feedId,
                    WeekEnding = weekEnding,
                }),
                HandleWorkerMessage,
                _operationCancellation.Token);
            if (report is null)
            {
                WeekCoverageText.Text = "No weekly report was returned.";
                WeekSummaryText.Text = "";
            }
            else
            {
                WeekCoverageText.Text = report.CoverageSummary
                    + (string.IsNullOrWhiteSpace(report.NotableRecordsSummary)
                        ? ""
                        : Environment.NewLine + report.NotableRecordsSummary);
                WeekSummaryText.Text = report.Summary;
            }
        }
        catch (OperationCanceledException)
        {
            WeekCoverageText.Text = "Weekly summary cancelled.";
            WeekSummaryText.Text = "";
        }
        catch (Exception exception)
        {
            await ShowErrorAsync(exception);
        }
        finally
        {
            _operationCancellation.Dispose();
            _operationCancellation = null;
            JobProgress.IsIndeterminate = false;
            SetBusy(false);
        }
    }

    private static List<string> ParseZipCodes(string value)
    {
        var zipCodes = value.Split(
                new[] { ',', ';', ' ', '\r', '\n', '\t' },
                StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Distinct()
            .ToList();
        if (zipCodes.Count == 0 || zipCodes.Any(zip => zip.Length != 5 || !zip.All(char.IsDigit)))
        {
            throw new FormatException("Enter one or more five-digit ZIP codes separated by commas or spaces.");
        }
        if (zipCodes.Count > 20)
        {
            throw new FormatException("Area searches are limited to 20 ZIP codes at a time.");
        }
        return zipCodes;
    }

    private void AreaCoverageMode_SelectionChanged(object sender, SelectionChangedEventArgs e)
        => UpdateAreaCoverageControls();

    private void UpdateAreaCoverageControls()
    {
        if (AreaRadiusBox is null || AreaMaxZipCodesBox is null || AreaZipCodesBox is null)
        {
            return;
        }
        var radiusMode = string.Equals(
            SelectedComboValue(AreaCoverageModeCombo, "radius"),
            "radius",
            StringComparison.OrdinalIgnoreCase);
        AreaRadiusBox.IsEnabled = radiusMode;
        AreaMaxZipCodesBox.IsEnabled = radiusMode;
        AreaZipCodesBox.Header = radiusMode ? "Center ZIP" : "ZIPs in priority order";
        AreaZipCodesBox.PlaceholderText = radiusMode ? "5-digit ZIP" : "Comma-separated ZIPs";
        if (DiscoverAreaFeedsButton is not null)
        {
            DiscoverAreaFeedsButton.Content = radiusMode ? "Discover nearest" : "Discover feeds";
        }
    }

    private void AreaPublicSafetyOnly_Changed(object sender, RoutedEventArgs e)
        => RefreshAreaFeedFilter();

    private HashSet<string> SelectedAreaFeedIds()
        => _selectedAreaFeedIds.ToHashSet(StringComparer.Ordinal);

    private List<FeedSearchResult> SelectedAreaFeedsInPriorityOrder()
        => _allAreaFeeds
            .Where(value => _selectedAreaFeedIds.Contains(value.FeedId))
            .ToList();

    private void RefreshAreaFeedFilter(IEnumerable<string>? preferredFeedIds = null)
    {
        if (AreaFeedResults is null)
        {
            return;
        }
        if (preferredFeedIds is not null)
        {
            var availableFeedIds = _allAreaFeeds
                .Select(value => value.FeedId)
                .ToHashSet(StringComparer.Ordinal);
            _selectedAreaFeedIds.Clear();
            foreach (var feedId in preferredFeedIds.Where(availableFeedIds.Contains))
            {
                _selectedAreaFeedIds.Add(feedId);
            }
        }
        var visible = AreaPublicSafetyOnlyCheckBox?.IsChecked == true
            ? _allAreaFeeds.Where(value => string.Equals(
                value.Genre, "Public Safety", StringComparison.OrdinalIgnoreCase))
            : _allAreaFeeds;
        _refreshingAreaFeedSelection = true;
        try
        {
            _areaFeeds.Clear();
            foreach (var feed in visible)
            {
                _areaFeeds.Add(feed);
            }
            AreaFeedResults.SelectedItems.Clear();
            foreach (var feed in _areaFeeds.Where(
                         value => _selectedAreaFeedIds.Contains(value.FeedId)))
            {
                AreaFeedResults.SelectedItems.Add(feed);
            }
        }
        finally
        {
            _refreshingAreaFeedSelection = false;
        }
        UpdateAreaFeedSelectionSummary();
    }

    private void AreaFeedResults_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (_refreshingAreaFeedSelection)
        {
            return;
        }
        foreach (var feed in e.RemovedItems.OfType<FeedSearchResult>())
        {
            _selectedAreaFeedIds.Remove(feed.FeedId);
        }
        foreach (var feed in e.AddedItems.OfType<FeedSearchResult>())
        {
            _selectedAreaFeedIds.Add(feed.FeedId);
        }
        UpdateAreaFeedSelectionSummary();
    }

    private void SelectNearestAreaFeeds_Click(object sender, RoutedEventArgs e)
    {
        _selectedAreaFeedIds.Clear();
        foreach (var feed in _areaFeeds.Take(NearestAreaFeedShortcutCount))
        {
            _selectedAreaFeedIds.Add(feed.FeedId);
        }
        RefreshAreaFeedFilter();
    }

    private void ClearAreaFeeds_Click(object sender, RoutedEventArgs e)
    {
        _selectedAreaFeedIds.Clear();
        RefreshAreaFeedFilter();
    }

    private void UpdateAreaFeedSelectionSummary()
    {
        if (AreaFeedSelectionText is null || SaveAreaProfileButton is null)
        {
            return;
        }
        var selectedCount = _selectedAreaFeedIds.Count;
        var visibleSelectedCount = AreaFeedResults?.SelectedItems.Count ?? 0;
        var visibleCount = _areaFeeds.Count;
        AreaFeedSelectionText.Text = selectedCount == 0
            ? $"{visibleCount} nearby feed{(visibleCount == 1 ? "" : "s")} shown; none selected. Choose individually or start with the nearest {Math.Min(NearestAreaFeedShortcutCount, visibleCount)}."
            : selectedCount == visibleSelectedCount
                ? $"{selectedCount} of {visibleCount} shown feed{(visibleCount == 1 ? "" : "s")} selected. Archive work will keep this nearest-first order."
                : $"{selectedCount} feeds selected; {visibleSelectedCount} of {visibleCount} currently shown. Hidden selections remain explicit until cleared.";
        SaveAreaProfileButton.Content = selectedCount == 0
            ? "Select feeds to save this area"
            : $"Save {selectedCount} selected feed{(selectedCount == 1 ? "" : "s")} nearest first";
        if (StatusText is not null && AreaPage?.Visibility == Visibility.Visible)
        {
            StatusText.Text = $"{selectedCount} selected · {visibleCount} nearby feed{(visibleCount == 1 ? "" : "s")} shown";
        }
    }

    private async void DiscoverAreaFeeds_Click(object sender, RoutedEventArgs e)
    {
        if (_worker is null)
        {
            return;
        }
        List<string> zipCodes;
        var radiusMode = string.Equals(
            SelectedComboValue(AreaCoverageModeCombo, "radius"),
            "radius",
            StringComparison.OrdinalIgnoreCase);
        try
        {
            zipCodes = ParseZipCodes(AreaZipCodesBox.Text);
            if (radiusMode && zipCodes.Count != 1)
            {
                throw new FormatException("Radius coverage needs one five-digit center ZIP.");
            }
        }
        catch (FormatException exception)
        {
            await ShowMessageAsync("ZIP codes required", exception.Message);
            return;
        }

        var radiusMiles = Math.Clamp(AreaRadiusBox.Value, 1, 100);
        var maxZipCodes = Math.Clamp(RequiredInteger(AreaMaxZipCodesBox.Value, 12), 1, 20);
        var selectedFeedIds = SelectedAreaFeedIds();
        SetBusy(
            true,
            radiusMode
                ? $"Discovering the nearest ZIP areas within {radiusMiles:0.#} miles…"
                : $"Searching {zipCodes.Count} ordered ZIP code{(zipCodes.Count == 1 ? "" : "s")}…");
        try
        {
            using var cancellation = new CancellationTokenSource(TimeSpan.FromMinutes(2));
            var response = await _worker.SearchAreaFeedsAsync(
                zipCodes,
                radiusMode ? zipCodes[0] : null,
                radiusMode ? radiusMiles : null,
                maxZipCodes,
                cancellation.Token);
            _currentAreaCoverage = response.Coverage;
            var results = response.Results;
            _allAreaFeeds = results.ToList();
            RefreshAreaFeedFilter(selectedFeedIds);
            var visibleCount = _areaFeeds.Count;
            var selectedCount = _selectedAreaFeedIds.Count;
            var searchedCount = _currentAreaCoverage.SearchedZipCodes.Count;
            AreaCoverageText.Text =
                $"Showing {visibleCount} of {results.Count} unique feeds from {searchedCount} ZIP area{(searchedCount == 1 ? "" : "s")}, already ordered nearest first. {selectedCount} prior selection{(selectedCount == 1 ? "" : "s")} preserved.";
            StatusText.Text = $"Found {results.Count} nearby feed{(results.Count == 1 ? "" : "s")} · {selectedCount} selected";
            AppendLog(
                $"Area search returned {results.Count} unique feed(s); showing {visibleCount} in quota priority order with {selectedCount} selected.");
        }
        catch (Exception exception)
        {
            await ShowErrorAsync(exception);
        }
        finally
        {
            SetBusy(false);
        }
    }

    private async Task RefreshAreaProfilesAsync(string? selectName = null)
    {
        if (_worker is null)
        {
            return;
        }
        selectName ??= (AreaProfileCombo.SelectedItem as AreaProfile)?.Name;
        selectName ??= _lastAreaProfileName;
        using var cancellation = new CancellationTokenSource(TimeSpan.FromSeconds(30));
        var profiles = await _worker.ListAreaProfilesAsync(cancellation.Token);
        _areaProfiles.Clear();
        foreach (var profile in profiles)
        {
            _areaProfiles.Add(profile);
        }
        if (!string.IsNullOrWhiteSpace(selectName))
        {
            AreaProfileCombo.SelectedItem = _areaProfiles.FirstOrDefault(
                value => string.Equals(value.Name, selectName, StringComparison.OrdinalIgnoreCase));
        }
        else if (_areaProfiles.Count > 0)
        {
            AreaProfileCombo.SelectedIndex = 0;
        }
    }

    private async void AreaProfile_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (AreaProfileCombo.SelectedItem is not AreaProfile profile)
        {
            return;
        }
        _lastAreaProfileName = profile.Name;
        ScheduleSettingsSave();
        AreaProfileNameBox.Text = profile.Name;
        _currentAreaCoverage = profile.Coverage with
        {
            SearchedZipCodes = profile.Coverage.SearchedZipCodes.Count > 0
                ? profile.Coverage.SearchedZipCodes
                : profile.ZipCodes.Select(value => new AreaZipCandidate { ZipCode = value }).ToList(),
        };
        SelectComboTag(AreaCoverageModeCombo, _currentAreaCoverage.Mode);
        AreaZipCodesBox.Text = string.Equals(
            _currentAreaCoverage.Mode, "radius", StringComparison.OrdinalIgnoreCase)
            ? (_currentAreaCoverage.CenterZip.Length > 0
                ? _currentAreaCoverage.CenterZip
                : profile.ZipCodes.FirstOrDefault() ?? "")
            : string.Join(", ", profile.ZipCodes);
        AreaRadiusBox.Value = _currentAreaCoverage.RadiusMiles ?? 25;
        AreaMaxZipCodesBox.Value = _currentAreaCoverage.MaxZipCodes > 0
            ? _currentAreaCoverage.MaxZipCodes
            : Math.Min(20, profile.ZipCodes.Count);
        UpdateAreaCoverageControls();
        _allAreaFeeds = profile.Feeds.ToList();
        RefreshAreaFeedFilter(profile.Feeds.Select(value => value.FeedId));
        AreaCoverageText.Text =
            $"Loaded {profile.DisplayName} for {profile.CoverageArea}. Selected feeds are explicit and can be changed before saving.";
        await LoadLatestAreaQueueAsync(profile.Name);
        await LoadLatestAreaDigestAsync(profile);
    }

    private async Task LoadLatestAreaQueueAsync(string profileName)
    {
        if (_worker is null)
        {
            return;
        }
        try
        {
            using var cancellation = new CancellationTokenSource(TimeSpan.FromSeconds(30));
            var runs = await _worker.ListAreaAcquisitionRunsAsync(profileName, cancellation.Token);
            var latest = runs.FirstOrDefault();
            AreaQueueStatusText.Text = latest is null
                ? "No retained queue for this profile yet."
                : $"{latest.Summary} · {latest.StartDate} through {latest.EndDate}"
                    + (string.IsNullOrWhiteSpace(latest.StopReason) ? "" : $" · {latest.StopReason}");
        }
        catch (Exception exception)
        {
            AreaQueueStatusText.Text = $"Queue history unavailable: {exception.Message}";
        }
    }

    private async Task LoadLatestAreaDigestAsync(AreaProfile profile)
    {
        if (_worker is null)
        {
            return;
        }
        try
        {
            using var cancellation = new CancellationTokenSource(TimeSpan.FromSeconds(30));
            var report = await _worker.GetLatestAreaDigestAsync(profile.Name, cancellation.Token);
            if (report is null)
            {
                _areaStories.Clear();
                ClearAreaStoryDetail(
                    "Older story claims are hidden until their source days use the current evidence rules.");
                SetAreaSummary("");
                AreaCoverageText.Text =
                    "No current evidence-gated area brief is saved. Reanalyze retained feed-days, then find story leads again.";
                return;
            }
            if (DateTimeOffset.TryParse(report.StartDate, out var startDate))
            {
                AreaStartDatePicker.Date = startDate;
            }
            if (DateTimeOffset.TryParse(report.EndDate, out var endDate))
            {
                AreaEndDatePicker.Date = endDate;
            }
            ApplyAreaDigest(report);
        }
        catch (Exception exception)
        {
            AppendLog($"Saved area brief: {exception.Message}");
        }
    }

    private void ApplyAreaDigest(AreaDigestReport report)
    {
        var selectedStoryId = (AreaStoryList.SelectedItem as AreaStory)?.StoryId;
        AreaCoverageText.Text = report.CoverageSummary;
        SetAreaSummary(report.Summary);
        _areaStories.Clear();
        foreach (var story in report.Stories)
        {
            _areaStories.Add(story);
        }
        var selectedStory = _areaStories.FirstOrDefault(
                value => string.Equals(value.StoryId, selectedStoryId, StringComparison.Ordinal))
            ?? _areaStories.FirstOrDefault();
        AreaStoryList.SelectedItem = selectedStory;
        if (selectedStory is null)
        {
            ClearAreaStoryDetail("No story evidence is available for this range.");
        }
        else
        {
            AreaStoryList.ScrollIntoView(selectedStory);
        }
    }

    private void SetAreaSummary(string value)
    {
        AreaSummaryText.Text = value;
        var hasSummary = !string.IsNullOrWhiteSpace(value);
        AreaSummaryExpander.Visibility = hasSummary ? Visibility.Visible : Visibility.Collapsed;
        AreaSummaryExpander.IsExpanded = false;
    }

    private void AreaStoryList_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (AreaStoryList.SelectedItem is AreaStory story)
        {
            ShowAreaStoryDetail(story);
        }
        else
        {
            ClearAreaStoryDetail(
                _areaStories.Count == 0
                    ? "No story evidence is available for this range."
                    : "Choose a ranked lead to inspect its evidence.");
        }
    }

    private void ShowAreaStoryDetail(AreaStory story)
    {
        _areaStoryMediaPlayer.Pause();
        _areaStoryMediaPlayer.Source = null;
        AreaStoryDetailEmptyText.Visibility = Visibility.Collapsed;
        AreaStoryDetailScroll.Visibility = Visibility.Visible;
        AreaStoryDetailScoreText.Text = story.ScoreSummary;
        AreaStoryDetailHeadlineText.Text = story.Headline;
        AreaStoryDetailTimePlaceText.Text = story.TimeAndPlace;
        AreaStoryDetailSummaryText.Text = story.Summary;
        AreaStoryDetailWhyText.Text = story.WhyInteresting;
        AreaStoryDetailEvidenceSummaryText.Text = story.EvidenceSummary;
        AreaStoryDetailAudienceText.Text = story.AudienceSummary;
        AreaStoryEvidenceList.ItemsSource = story.IncidentReferences;
        AreaPlaybackStatusText.Text = story.IncidentReferences.Any(value => value.ClipAvailable)
            ? "Choose a source clip to audit its transcript quote by ear."
            : "This lead has transcript evidence but no retained playable clip.";
        AreaStoryDetailScroll.ChangeView(null, 0, null, true);
    }

    private void ClearAreaStoryDetail(string message)
    {
        _areaStoryMediaPlayer.Pause();
        _areaStoryMediaPlayer.Source = null;
        AreaStoryEvidenceList.ItemsSource = null;
        AreaStoryDetailScroll.Visibility = Visibility.Collapsed;
        AreaStoryDetailEmptyText.Text = message;
        AreaStoryDetailEmptyText.Visibility = Visibility.Visible;
        AreaPlaybackStatusText.Text = message;
    }

    private async void SaveAreaProfile_Click(object sender, RoutedEventArgs e)
    {
        if (_worker is null)
        {
            return;
        }
        var selected = SelectedAreaFeedsInPriorityOrder();
        if (selected.Count == 0)
        {
            await ShowMessageAsync("Feeds required", "Select at least one discovered feed.");
            return;
        }
        if (string.IsNullOrWhiteSpace(AreaProfileNameBox.Text))
        {
            await ShowMessageAsync("Profile name required", "Give this area watch a newsroom-friendly name.");
            return;
        }
        List<string> zipCodes;
        try
        {
            zipCodes = _currentAreaCoverage.SearchedZipCodes
                .Select(value => value.ZipCode)
                .Where(value => !string.IsNullOrWhiteSpace(value))
                .Distinct()
                .ToList();
            if (zipCodes.Count == 0)
            {
                zipCodes = ParseZipCodes(AreaZipCodesBox.Text);
            }
        }
        catch (FormatException exception)
        {
            await ShowMessageAsync("ZIP codes required", exception.Message);
            return;
        }

        SetBusy(true, "Saving area profile…");
        try
        {
            using var cancellation = new CancellationTokenSource(TimeSpan.FromSeconds(30));
            var profile = await _worker.SaveAreaProfileAsync(
                new AreaProfileSaveRequest
                {
                    Name = AreaProfileNameBox.Text.Trim(),
                    ZipCodes = zipCodes,
                    Feeds = selected,
                    Coverage = _currentAreaCoverage,
                },
                HandleWorkerMessage,
                cancellation.Token);
            await RefreshAreaProfilesAsync(profile?.Name);
        }
        catch (Exception exception)
        {
            await ShowErrorAsync(exception);
        }
        finally
        {
            SetBusy(false);
        }
    }

    private async void ProcessAreaFeeds_Click(object sender, RoutedEventArgs e)
    {
        if (_worker is null || _pipelineCancellation is not null)
        {
            return;
        }
        if (AreaProfileCombo.SelectedItem is not AreaProfile profile)
        {
            await ShowMessageAsync(
                "Saved profile required",
                "Save the reviewed nearest-first feed selection before starting a resumable area queue.");
            return;
        }
        var selected = profile.Feeds;
        if (selected.Count == 0)
        {
            await ShowMessageAsync("Feeds required", "Select the feeds to archive and analyze.");
            return;
        }
        var startDate = AreaStartDatePicker.Date.Date;
        var endDate = AreaEndDatePicker.Date.Date;
        if (startDate > endDate)
        {
            await ShowMessageAsync("Invalid date range", "Start date must be on or before end date.");
            return;
        }
        var minimumSpeakers = OptionalPositiveInteger(MinimumSpeakersBox.Value);
        var maximumSpeakers = OptionalPositiveInteger(MaximumSpeakersBox.Value);
        if (minimumSpeakers is not null && maximumSpeakers is not null && minimumSpeakers > maximumSpeakers)
        {
            await ShowMessageAsync("Invalid speaker range", "Minimum speakers cannot exceed maximum speakers.");
            return;
        }

        var pipeline = new CancellationTokenSource();
        _pipelineCancellation = pipeline;
        _activePipelineFeedIds.UnionWith(selected.Select(value => value.FeedId));
        SetPipelineBusy(true, "Starting explicit multi-feed area job in the background…");
        JobProgress.IsIndeterminate = true;
        JobProgress.Value = 0;
        try
        {
            await ReleaseMediaForArchiveMutationAsync();
            await ConfigureLanSharingAsync();
            var baseRequest = CreateJobRequest(
                selected[0].FeedId, startDate, endDate, minimumSpeakers, maximumSpeakers,
                selected[0].Name);
            var result = await _worker.RunAreaAcquisitionAsync(
                new AreaAcquisitionRequest
                {
                    ProfileName = profile.Name,
                    FeedIds = selected.Select(value => value.FeedId).ToList(),
                    Job = baseRequest,
                },
                HandleWorkerMessage,
                pipeline.Token);
            if (result is null)
            {
                AreaCoverageText.Text = "The area worker returned no queue result.";
                return;
            }
            foreach (var feedResult in result.FeedResults)
            {
                pipeline.Token.ThrowIfCancellationRequested();
                await AnalyzeCompletedJobAsync(
                    baseRequest with { FeedId = feedResult.Feed.FeedId },
                    feedResult.Result,
                    pipeline.Token);
            }
            AreaCoverageText.Text = result.DownloadLimited
                ? $"Queue {result.Id} paused at the archive quota boundary. Resume later; the first incomplete feed stays next and lower-priority feeds made no requests."
                : $"Queue {result.Id} is {result.Status}. Generate the area brief to rank retained cross-feed leads.";
            AreaQueueStatusText.Text = result.Summary
                + (string.IsNullOrWhiteSpace(result.StopReason) ? "" : $" · {result.StopReason}");
        }
        catch (OperationCanceledException)
        {
            StatusText.Text = "Cancelled";
            AppendLog("Area job cancelled.");
        }
        catch (Exception exception)
        {
            await ShowErrorAsync(exception);
        }
        finally
        {
            pipeline.Dispose();
            if (ReferenceEquals(_pipelineCancellation, pipeline))
            {
                _pipelineCancellation = null;
            }
            _activePipelineFeedIds.Clear();
            JobProgress.IsIndeterminate = false;
            SetPipelineBusy(false);
            await RefreshLibraryAsync();
        }
    }

    private async void GenerateAreaDigest_Click(object sender, RoutedEventArgs e)
    {
        if (_worker is null || AreaProfileCombo.SelectedItem is not AreaProfile profile)
        {
            await ShowMessageAsync("Area profile required", "Save or select an area profile first.");
            return;
        }
        var startDate = AreaStartDatePicker.Date.Date;
        var endDate = AreaEndDatePicker.Date.Date;
        if (startDate > endDate)
        {
            await ShowMessageAsync("Invalid date range", "Start date must be on or before end date.");
            return;
        }

        _operationCancellation = new CancellationTokenSource();
        SetBusy(true, $"Ranking story leads for {profile.Name}…", jobRunning: true);
        JobProgress.IsIndeterminate = true;
        AreaCoverageText.Text =
            $"Loading saved incidents across selected feeds with {SelectedAnalysisProviderDisplayName()}…";
        SetAreaSummary("");
        _areaStoryMediaPlayer.Pause();
        _areaStoryMediaPlayer.Source = null;
        _areaStories.Clear();
        ClearAreaStoryDetail("Preparing story evidence packages…");
        try
        {
            var report = await _worker.SummarizeAreaAsync(
                ApplyAnalysisProvider(new AreaDigestRequest
                {
                    ProfileName = profile.Name,
                    StartDate = startDate.ToString("yyyy-MM-dd"),
                    EndDate = endDate.ToString("yyyy-MM-dd"),
                }),
                HandleWorkerMessage,
                _operationCancellation.Token);
            if (report is null)
            {
                AreaCoverageText.Text = "No area report was returned.";
                SetAreaSummary("");
                return;
            }
            ApplyAreaDigest(report);
        }
        catch (OperationCanceledException)
        {
            AreaCoverageText.Text = "Area digest cancelled.";
            SetAreaSummary("");
        }
        catch (Exception exception)
        {
            await ShowErrorAsync(exception);
        }
        finally
        {
            _operationCancellation.Dispose();
            _operationCancellation = null;
            JobProgress.IsIndeterminate = false;
            SetBusy(false);
        }
    }

    private void HandleQuestionWorkerMessage(JsonElement message)
    {
        var snapshot = message.Clone();
        var text = snapshot.TryGetProperty("message", out var messageValue)
            ? messageValue.GetString() ?? ""
            : "";
        if (string.IsNullOrWhiteSpace(text))
        {
            return;
        }
        AppDiagnostics.AppendActivity($"Archive chat: {text}");
        DispatcherQueue.TryEnqueue(() =>
        {
            ArchiveChatStatusText.Text = text;
            AppendVisibleLog($"Archive chat: {text}");
            RefreshVisibleLog();
        });
    }

    private void HandleWorkerMessage(JsonElement message)
    {
        var snapshot = message.Clone();
        var text = snapshot.TryGetProperty("message", out var messageValue)
            ? messageValue.GetString() ?? ""
            : "";
        if (!string.IsNullOrWhiteSpace(text))
        {
            // Preserve every worker event in the complete diagnostic log on
            // the output-reader thread. The UI only needs a bounded recent
            // window and must never backpressure a multi-day worker burst.
            AppDiagnostics.AppendActivity(text);
        }
        _pendingWorkerMessages.Enqueue(snapshot);
        ScheduleWorkerMessageDrain();
    }

    private void ScheduleWorkerMessageDrain()
    {
        if (Interlocked.CompareExchange(
                ref _workerMessageDrainScheduled,
                1,
                0) != 0)
        {
            return;
        }
        if (!DispatcherQueue.TryEnqueue(DrainWorkerMessages))
        {
            Interlocked.Exchange(ref _workerMessageDrainScheduled, 0);
        }
    }

    private void DrainWorkerMessages()
    {
        string? latestStatus = null;
        var visibleLogChanged = false;
        var sawProgress = false;
        var progressCurrent = 0;
        var progressTotal = 0;
        var sawComplete = false;
        while (_pendingWorkerMessages.TryDequeue(out var message))
        {
            var type = message.TryGetProperty("type", out var typeValue)
                ? typeValue.GetString()
                : "";
            var text = message.TryGetProperty("message", out var messageValue)
                ? messageValue.GetString() ?? ""
                : "";
            if (!string.IsNullOrWhiteSpace(text))
            {
                latestStatus = text;
                AppendVisibleLog(text);
                visibleLogChanged = true;
            }
            if (!_broadcastifyRateLimitObserved
                && (text.Contains("HTTP 429", StringComparison.OrdinalIgnoreCase)
                    || text.Contains("download quota", StringComparison.OrdinalIgnoreCase)
                    || text.Contains("quota is exhausted", StringComparison.OrdinalIgnoreCase)))
            {
                _broadcastifyRateLimitObserved = true;
                var protectionMessage =
                    "Rate-limit protection enabled: future archive jobs in this app session "
                    + "will start with one download at a time.";
                AppendVisibleLog(protectionMessage);
                AppDiagnostics.AppendActivity(protectionMessage);
                visibleLogChanged = true;
            }
            if (type == "progress")
            {
                sawProgress = true;
                progressCurrent = message.TryGetProperty(
                    "current", out var currentValue)
                    ? currentValue.GetInt32()
                    : 0;
                progressTotal = message.TryGetProperty(
                    "total", out var totalValue)
                    ? totalValue.GetInt32()
                    : 0;
            }
            else if (type == "complete")
            {
                sawComplete = true;
            }
        }

        if (latestStatus is not null)
        {
            StatusText.Text = latestStatus;
        }
        if (visibleLogChanged)
        {
            RefreshVisibleLog();
        }
        if (sawProgress)
        {
            JobProgress.IsIndeterminate = progressTotal <= 0;
            JobProgress.Maximum = Math.Max(1, progressTotal);
            JobProgress.Value = Math.Clamp(
                progressCurrent,
                0,
                Math.Max(1, progressTotal));
        }
        if (sawComplete)
        {
            JobProgress.IsIndeterminate = false;
            JobProgress.Value = JobProgress.Maximum;
        }

        Interlocked.Exchange(ref _workerMessageDrainScheduled, 0);
        if (!_pendingWorkerMessages.IsEmpty)
        {
            ScheduleWorkerMessageDrain();
        }
    }

    private void SetBusy(bool busy, string? status = null, bool jobRunning = false)
    {
        if (busy)
        {
            _settingsSaveTimer?.Stop();
            PersistUserSettings();
            PersistAnalysisCredentialPreference();
        }
        _exclusiveBusy = busy;
        _exclusiveJobRunning = busy && jobRunning;
        if (status is not null)
        {
            StatusText.Text = status;
        }
        UpdateCommandAvailability();
    }

    private void SetPipelineBusy(bool busy, string? status = null)
    {
        if (busy)
        {
            _settingsSaveTimer?.Stop();
            PersistUserSettings();
            PersistAnalysisCredentialPreference();
        }
        if (status is not null)
        {
            StatusText.Text = status;
        }
        if (_selectedLibraryDay is not null)
        {
            ShowLibraryDetails(_selectedLibraryDay);
        }
        UpdateCommandAvailability();
    }

    private bool IsFeedPipelineBusy(string feedId) =>
        _activePipelineFeedIds.Contains(feedId);

    private bool IsFeedBusy(string feedId) =>
        IsFeedPipelineBusy(feedId)
        || (!string.IsNullOrWhiteSpace(_activeQuestionFeedId)
            && string.Equals(
                _activeQuestionFeedId,
                feedId,
                StringComparison.Ordinal));

    private void UpdateCommandAvailability()
    {
        var ready = _worker is not null;
        var interactive = ready && !_exclusiveBusy;
        var pipelineIdle = _pipelineCancellation is null;
        var selectedDay = _selectedLibraryDay;

        SearchButton.IsEnabled = interactive;
        StartButton.IsEnabled = interactive && pipelineIdle;
        ScheduleFeedButton.IsEnabled = interactive;
        ManageSchedulesButton.IsEnabled = interactive;
        RefreshAnalysisButton.IsEnabled = interactive;
        AnalyzeSelectedButton.IsEnabled = interactive && pipelineIdle;
        ReloadReportButton.IsEnabled = interactive;
        AskButton.IsEnabled = interactive
            && _questionCancellation is null
            && !string.IsNullOrWhiteSpace(AnalysisFeedBox.Text);
        UseQuestionMonthButton.IsEnabled = interactive
            && _questionCancellation is null;
        UseEntireQuestionFeedButton.IsEnabled = interactive
            && _questionCancellation is null
            && !string.IsNullOrWhiteSpace(AnalysisFeedBox.Text);
        CheckQuestionCoverageButton.IsEnabled = interactive
            && _questionCancellation is null
            && !string.IsNullOrWhiteSpace(AnalysisFeedBox.Text);
        ClearArchiveChatButton.IsEnabled = interactive;
        GenerateWeekButton.IsEnabled = interactive
            && pipelineIdle
            && _questionCancellation is null;
        DiscoverAreaFeedsButton.IsEnabled = interactive;
        SaveAreaProfileButton.IsEnabled = interactive;
        ProcessAreaFeedsButton.IsEnabled = interactive && pipelineIdle;
        GenerateAreaDigestButton.IsEnabled = interactive
            && pipelineIdle
            && _questionCancellation is null;
        AnalysisProviderCheckButton.IsEnabled = interactive && pipelineIdle;
        AnalysisModelTestButton.IsEnabled = interactive && pipelineIdle;
        SetupProfileSelfTestButton.IsEnabled = interactive && pipelineIdle;
        ProfileSelfTestButton.IsEnabled = interactive && pipelineIdle;
        AsrPrepareButton.IsEnabled = interactive && pipelineIdle;
        AsrSelfTestButton.IsEnabled = interactive && pipelineIdle;
        DiarizationSelfTestButton.IsEnabled = interactive && pipelineIdle;
        ManagedRuntimeInstallButton.IsEnabled = interactive
            && pipelineIdle
            && _worker?.HasBundledManagedRuntime == true;
        SetupAccountActionButton.IsEnabled = interactive;
        SetupTranscriptionActionButton.IsEnabled = interactive && pipelineIdle;
        SetupDiarizationActionButton.IsEnabled = interactive && pipelineIdle;
        SetupAnalysisActionButton.IsEnabled = interactive && pipelineIdle;
        AreaProfileCombo.IsEnabled = interactive;
        RefreshLibraryButton.IsEnabled = interactive;
        CatchUpFeedButton.IsEnabled = ready
            && !_libraryMutationBusy
            && _libraryFeeds.Count > 0;
        ResumeAllLibraryButton.IsEnabled = interactive
            && pipelineIdle
            && _libraryFeeds.Any(value => value.BacklogCount > 0);
        LibraryList.IsEnabled = interactive && !_libraryMutationBusy;
        LibraryFeedCoverageList.IsEnabled = interactive && !_libraryMutationBusy;
        LibraryDetailPrimaryButton.IsEnabled = interactive
            && pipelineIdle
            && selectedDay is not null;
        LibraryDetailReviewButton.IsEnabled = interactive
            && selectedDay?.CanOpenReview == true;
        LibraryCheckSourceButton.IsEnabled = interactive
            && pipelineIdle
            && selectedDay is not null;
        LibraryDeleteFeedButton.IsEnabled = interactive
            && !_libraryMutationBusy
            && selectedDay is not null
            && !IsFeedBusy(selectedDay.FeedId);
        LibraryOpenFolderButton.IsEnabled = interactive
            && selectedDay is not null
            && Directory.Exists(selectedDay.DayDirectory);
        LibraryOpenTranscriptButton.IsEnabled = interactive
            && selectedDay is not null
            && !IsFeedPipelineBusy(selectedDay.FeedId)
            && File.Exists(selectedDay.TranscriptPath);
        var selectedAccountProfile = SelectedBroadcastifyProfileId();
        ClearSavedLoginButton.IsEnabled = interactive
            && selectedAccountProfile != "__new__"
            && CredentialStore.TryLoadBroadcastifyProfile(
                selectedAccountProfile) is not null;
        SaveHuggingFaceTokenButton.IsEnabled = interactive;
        ClearHuggingFaceTokenButton.IsEnabled = interactive
            && CredentialStore.TryLoadHuggingFaceToken() is not null;
        CancelButton.IsEnabled = _pipelineCancellation is not null
            || (_exclusiveBusy && _exclusiveJobRunning)
            || _questionCancellation is not null;
    }

    private void AppendLog(string message)
    {
        AppendVisibleLog(message);
        RefreshVisibleLog();
        AppDiagnostics.AppendActivity(message);
    }

    private void AppendVisibleLog(string message)
    {
        _visibleActivityLog.Append(
            $"[{DateTime.Now:HH:mm:ss}] {message}{Environment.NewLine}");
        if (_visibleActivityLog.Length > MaximumVisibleActivityLogCharacters)
        {
            var overflow = _visibleActivityLog.Length
                - RetainedVisibleActivityLogCharacters;
            var current = _visibleActivityLog.ToString();
            var lineBreak = current.IndexOf(
                Environment.NewLine,
                overflow,
                StringComparison.Ordinal);
            var retained = lineBreak >= 0
                ? current[(lineBreak + Environment.NewLine.Length)..]
                : current[^RetainedVisibleActivityLogCharacters..];
            _visibleActivityLog.Clear();
            _visibleActivityLog.Append(VisibleActivityLogTrimMarker);
            _visibleActivityLog.Append(Environment.NewLine);
            _visibleActivityLog.Append(retained);
        }
    }

    private void RefreshVisibleLog()
    {
        LogBox.Text = _visibleActivityLog.ToString();
        LogBox.Select(_visibleActivityLog.Length, 0);
    }

    private async Task ShowErrorAsync(Exception exception)
    {
        StatusText.Text = "Error";
        AppendLog($"ERROR: {exception.Message}");
        await ShowMessageAsync("Processing error", exception.Message);
    }

    private async Task ShowMessageAsync(string title, string message)
    {
        var dialog = new ContentDialog
        {
            XamlRoot = ((FrameworkElement)Content).XamlRoot,
            Title = title,
            Content = CreateDialogTextContent(message),
            CloseButtonText = "OK",
        };
        await dialog.ShowAsync();
    }

    private static ScrollViewer CreateDialogScrollContent(
        object content,
        double maximumHeight = 560)
    {
        var scrollViewer = new ScrollViewer
        {
            MaxHeight = maximumHeight,
            VerticalScrollMode = ScrollMode.Enabled,
            VerticalScrollBarVisibility = ScrollBarVisibility.Auto,
            HorizontalScrollMode = ScrollMode.Disabled,
            HorizontalScrollBarVisibility = ScrollBarVisibility.Disabled,
            HorizontalContentAlignment = HorizontalAlignment.Stretch,
            Content = content,
        };
        AutomationProperties.SetAutomationId(
            scrollViewer,
            "DialogScrollHost");
        return scrollViewer;
    }

    private static ScrollViewer CreateDialogTextContent(
        string message,
        double maximumHeight = 560) =>
        CreateDialogScrollContent(
            new TextBlock
            {
                Text = message,
                TextWrapping = TextWrapping.Wrap,
                IsTextSelectionEnabled = true,
            },
            maximumHeight);

    private static string SelectedComboValue(ComboBox? comboBox, string fallback)
    {
        if (comboBox?.SelectedItem is not ComboBoxItem item)
        {
            return fallback;
        }
        return item.Tag?.ToString() ?? item.Content?.ToString() ?? fallback;
    }

    private static void SelectComboTag(ComboBox? comboBox, string tag)
    {
        if (comboBox is null)
        {
            return;
        }
        foreach (var value in comboBox.Items.OfType<ComboBoxItem>())
        {
            if (string.Equals(value.Tag?.ToString(), tag, StringComparison.OrdinalIgnoreCase))
            {
                comboBox.SelectedItem = value;
                return;
            }
        }
    }

    private static void SelectComboValue(ComboBox? comboBox, string value)
    {
        if (comboBox is null)
        {
            return;
        }
        foreach (var item in comboBox.Items.OfType<ComboBoxItem>())
        {
            if (string.Equals(item.Tag?.ToString(), value, StringComparison.OrdinalIgnoreCase)
                || string.Equals(item.Content?.ToString(), value, StringComparison.OrdinalIgnoreCase))
            {
                comboBox.SelectedItem = item;
                return;
            }
        }
    }

    private static int RequiredInteger(double value, int fallback) =>
        double.IsNaN(value) ? fallback : Math.Max(0, Convert.ToInt32(value));

    private static int? OptionalPositiveInteger(double value) =>
        double.IsNaN(value) || value <= 0 ? null : Convert.ToInt32(value);

    [DllImport("user32.dll")]
    private static extern uint GetDpiForWindow(nint windowHandle);

    [DllImport("dwmapi.dll")]
    private static extern int DwmSetWindowAttribute(
        nint windowHandle,
        int attribute,
        ref int value,
        int valueSize);

    [DllImport("dwmapi.dll")]
    private static extern int DwmGetWindowAttribute(
        nint windowHandle,
        int attribute,
        out int value,
        int valueSize);
}
