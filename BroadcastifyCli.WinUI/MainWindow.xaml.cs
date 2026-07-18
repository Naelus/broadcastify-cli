using System.Collections.ObjectModel;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using Microsoft.UI;
using Microsoft.UI.Windowing;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Windows.Graphics;
using Windows.Media.Core;
using Windows.Media.Playback;
using Windows.Security.Credentials;
using Windows.Storage;
using Windows.Storage.Pickers;
using Windows.System;
using WinRT.Interop;

namespace BroadcastifyCli.WinUI;

public sealed partial class MainWindow : Window
{
    private const string DefaultAnalysisModel = "ggml-org/gemma-4-12B-it-GGUF:Q4_0";
    private const int NearestAreaFeedShortcutCount = 3;
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
    private readonly ObservableCollection<HardwareProfileStatus> _hardwareProfiles = [];
    private WorkerClient? _worker;
    private FeedSearchResult? _selectedFeed;
    private CancellationTokenSource? _operationCancellation;
    private DayReport? _currentReport;
    private readonly MediaPlayer _incidentMediaPlayer = new();
    private readonly MediaPlayer _areaStoryMediaPlayer = new();
    private readonly MediaPlayer _libraryMediaPlayer = new();
    private IncidentClip? _pendingIncidentClip;
    private LibraryDay? _selectedLibraryDay;
    private int _librarySelectionVersion;
    private bool _broadcastifyRateLimitObserved;
    private bool _loadingSettings = true;
    private Microsoft.UI.Dispatching.DispatcherQueueTimer? _settingsSaveTimer;
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
    private bool _huggingFaceTokenConfigured;
    private bool _cudaAvailable;
    private ProfileSetupAction? _profileRecoveryAction;

    public MainWindow()
    {
        InitializeComponent();
        ExtendsContentIntoTitleBar = true;
        SetTitleBar(AppTitleBar);
        var dpiScale = Math.Max(1.0, GetDpiForWindow(WindowNative.GetWindowHandle(this)) / 96.0);
        AppWindow.Resize(new SizeInt32((int)(1240 * dpiScale), (int)(900 * dpiScale)));

        FeedResults.ItemsSource = _feeds;
        AreaFeedResults.ItemsSource = _areaFeeds;
        AreaProfileCombo.ItemsSource = _areaProfiles;
        AreaStoryList.ItemsSource = _areaStories;
        AnalysisDaysList.ItemsSource = _analysisDays;
        IncidentList.ItemsSource = _visibleIncidents;
        LibraryList.ItemsSource = _visibleLibraryDays;
        HardwareProfileList.ItemsSource = _hardwareProfiles;
        IncidentPlayer.SetMediaPlayer(_incidentMediaPlayer);
        _incidentMediaPlayer.MediaOpened += IncidentMediaPlayer_MediaOpened;
        AreaStoryPlayer.SetMediaPlayer(_areaStoryMediaPlayer);
        _areaStoryMediaPlayer.MediaOpened += AreaStoryMediaPlayer_MediaOpened;
        LibraryAudioPlayer.SetMediaPlayer(_libraryMediaPlayer);
        CombineToggle.IsOn = true;
        CombineToggle.IsEnabled = false;
        KeepOriginalsToggle.IsEnabled = true;
        LoadUserSettings();
        WireSettingsAutoSave();
        var today = DateTimeOffset.Now;
        StartDatePicker.Date = today;
        EndDatePicker.Date = today;
        QuestionStartDatePicker.Date = today;
        QuestionEndDatePicker.Date = today;
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

        try
        {
            _worker = new WorkerClient();
            AppendLog($"Worker: {_worker.PythonDisplayName}");
            AppendLog($"Repository: {_worker.RepositoryRoot}");
            AppendLog(_worker.HasBundledWindowsMlHelper
                ? "Windows ML helper: bundled runtime"
                : "Windows ML helper: repository/runtime discovery");
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
    }

    private async Task InitializeAsync()
    {
        RefreshStorageReadiness();
        await TryAutoSignInAsync();
        await LoadDiagnosticsAndDaysAsync();
        await RefreshLibraryAsync();
    }

    private void RootNavigation_SelectionChanged(
        NavigationView sender,
        NavigationViewSelectionChangedEventArgs args)
    {
        var page = args.IsSettingsSelected
            ? "settings"
            : (args.SelectedItemContainer as NavigationViewItem)?.Tag?.ToString() ?? "library";
        ShowPage(page);
    }

    private void ShowPage(string page)
    {
        LibraryPage.Visibility = page == "library" ? Visibility.Visible : Visibility.Collapsed;
        ArchivePage.Visibility = page == "archive" ? Visibility.Visible : Visibility.Collapsed;
        ReviewPage.Visibility = page == "review" ? Visibility.Visible : Visibility.Collapsed;
        AreaPage.Visibility = page == "area" ? Visibility.Visible : Visibility.Collapsed;
        SettingsPage.Visibility = page == "settings" ? Visibility.Visible : Visibility.Collapsed;
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
                SelectComboTag(DiarizationDeviceComboBox, "cuda");
                break;
            case "vulkan":
                SelectComboTag(AsrEngineComboBox, "whisper.cpp");
                SelectComboTag(DeviceComboBox, "vulkan");
                SelectComboTag(DiarizationDeviceComboBox, "cpu");
                break;
            case "openvino":
                SelectComboTag(AsrEngineComboBox, "openvino");
                SelectComboTag(DeviceComboBox, "openvino-auto");
                SelectComboTag(DiarizationDeviceComboBox, "cpu");
                break;
            case "windowsml":
                SelectComboTag(AsrEngineComboBox, "windows-ml");
                SelectComboTag(DeviceComboBox, "auto");
                SelectComboTag(DiarizationDeviceComboBox, "cpu");
                break;
            case "qwen":
                SelectComboTag(AsrEngineComboBox, "qwen3-asr");
                SelectComboTag(DeviceComboBox, "cpu");
                SelectComboTag(DiarizationDeviceComboBox, "cpu");
                break;
            case "cpu":
                SelectComboTag(AsrEngineComboBox, "faster-whisper");
                SelectComboTag(DeviceComboBox, "cpu");
                SelectComboTag(DiarizationDeviceComboBox, "cpu");
                break;
            default:
                SelectComboTag(AsrEngineComboBox, "auto");
                SelectComboTag(DeviceComboBox, "auto");
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
    }

    private void LoadUserSettings()
    {
        _loadingSettings = true;
        var settings = AppSettingsStore.Load();
        SelectComboValue(HardwareProfileComboBox, settings.HardwareProfile);
        SelectComboValue(ModelComboBox, settings.WhisperModel);
        SelectComboValue(AsrEngineComboBox, settings.AsrEngine);
        SelectComboValue(DeviceComboBox, settings.TranscriptionDevice);
        SelectComboValue(DiarizationDeviceComboBox, settings.DiarizationDevice);
        AsrModelPathBox.Text = settings.AsrModelPath;
        OutputFolderBox.Text = string.IsNullOrWhiteSpace(settings.OutputDirectory)
            ? "archives"
            : settings.OutputDirectory;
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
        var savedAnalysisKey = CredentialStore.TryLoadAnalysisKey();
        if (savedAnalysisKey is not null)
        {
            AnalysisApiKeyBox.Password = savedAnalysisKey.Secret;
        }
        RememberAnalysisApiKeyCheckBox.IsChecked =
            settings.RememberAnalysisApiKey && savedAnalysisKey is not null;
        _lastAreaProfileName = settings.LastAreaProfileName;
        _lastReviewFeedId = settings.LastReviewFeedId;
        _lastReviewDate = settings.LastReviewDate;
        AnalysisFeedBox.Text = _lastReviewFeedId;
        _loadingSettings = false;
        UpdateAnalysisProviderUi();
        UpdateAsrModelPreparationUi();
        UpdateSetupSummary();
    }

    private void MainWindow_Closed(object sender, WindowEventArgs args)
    {
        _settingsSaveTimer?.Stop();
        PersistUserSettings(logFailure: true);
        PersistAnalysisCredentialPreference();
    }

    private DesktopSettings CaptureUserSettings() =>
        new()
        {
            HardwareProfile = SelectedComboValue(HardwareProfileComboBox, "auto"),
            WhisperModel = SelectedComboValue(ModelComboBox, "turbo"),
            AsrEngine = SelectedComboValue(AsrEngineComboBox, "auto"),
            TranscriptionDevice = SelectedComboValue(DeviceComboBox, "auto"),
            DiarizationDevice = SelectedComboValue(DiarizationDeviceComboBox, "auto"),
            AsrModelPath = AsrModelPathBox.Text.Trim(),
            OutputDirectory = string.IsNullOrWhiteSpace(OutputFolderBox.Text)
                ? "archives"
                : OutputFolderBox.Text.Trim(),
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
            LastAreaProfileName = _lastAreaProfileName,
            LastReviewFeedId = _lastReviewFeedId,
            LastReviewDate = _lastReviewDate,
        };

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
        _settingsSaveTimer.Tick += (_, _) => PersistUserSettings();

        foreach (var comboBox in new[]
                 {
                     HardwareProfileComboBox,
                     ModelComboBox,
                     AsrEngineComboBox,
                     DeviceComboBox,
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
                     AnalysisFeedBox,
                 })
        {
            textBox.TextChanged += (_, _) => ScheduleSettingsSave();
        }
        foreach (var numberBox in new[]
                 {
                     GpuIndexBox,
                     BatchSizeBox,
                     MinimumSpeakersBox,
                     MaximumSpeakersBox,
                     DownloadJobsBox,
                 })
        {
            numberBox.ValueChanged += (_, _) => ScheduleSettingsSave();
        }
        foreach (var toggle in new[]
                 {
                     CombineToggle,
                     KeepOriginalsToggle,
                     AllowExternalAnalysisToggle,
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
        DiarizationDeviceComboBox.SelectionChanged += (_, _) => ResetDiarizationVerification();
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

    private void ScheduleSettingsSave()
    {
        if (_loadingSettings || _settingsSaveTimer is null)
        {
            return;
        }
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
                "Run Test speakers after changing its device, token, GPU, or batch.";
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
                SettingsTabView.SelectedItem = ProcessingSettingsTab;
                HuggingFaceTokenBox.Focus(FocusState.Programmatic);
                break;
            case "configure-analysis":
                SettingsTabView.SelectedItem = AnalysisSettingsTab;
                AnalysisProviderComboBox.Focus(FocusState.Programmatic);
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
            var baseDirectory = _worker?.RepositoryRoot ?? Environment.CurrentDirectory;
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
            || DiarizationDeviceComboBox is null
            || HuggingFaceTokenBox is null)
        {
            return;
        }

        var profile = SelectedHardwareProfile();
        var profileAction = _profileRecoveryAction ?? profile?.NextAction;
        var tokenAvailable = _huggingFaceTokenConfigured
            || !string.IsNullOrWhiteSpace(HuggingFaceTokenBox?.Password);
        var selectedDiarizationDevice = SelectedComboValue(DiarizationDeviceComboBox, "auto");
        var typedTokenMakesDiarizationAvailable = _pyannotePackageInstalled
            && (_pyannoteAccessConfigured || tokenAvailable)
            && (selectedDiarizationDevice != "cuda" || _cudaAvailable);
        var transcriptionAvailable = _asrVerifiedThisSession
            || profile?.TranscriptionReady == true;
        var diarizationAvailable = _diarizationVerifiedThisSession
            || profile?.DiarizationReady == true
            || typedTokenMakesDiarizationAvailable;
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
              ?? "Run the local hardware check to inspect pyannote and its model token.";
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
        var tokenAvailable = _huggingFaceTokenConfigured
            || !string.IsNullOrWhiteSpace(HuggingFaceTokenBox.Password);
        if (_pyannotePackageInstalled && (_pyannoteAccessConfigured || tokenAvailable))
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
        HuggingFaceTokenBox.Focus(FocusState.Programmatic);
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
            BatchSize = RequiredInteger(BatchSizeBox.Value, 8),
            HuggingFaceToken = string.IsNullOrWhiteSpace(HuggingFaceTokenBox.Password)
                ? null
                : HuggingFaceTokenBox.Password,
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
            "The first explicit test may download the pyannote model. Archive audio and quota are not used.";
        try
        {
            var result = await _worker.RunDiarizationSelfTestAsync(
                new DiarizationSelfTestRequest
                {
                    DiarizationDevice = SelectedComboValue(DiarizationDeviceComboBox, "auto"),
                    DeviceIndex = RequiredInteger(GpuIndexBox.Value, 0),
                    BatchSize = RequiredInteger(BatchSizeBox.Value, 8),
                    HuggingFaceToken = string.IsNullOrWhiteSpace(HuggingFaceTokenBox.Password)
                        ? null
                        : HuggingFaceTokenBox.Password,
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
            DiarizationDevice = SelectedComboValue(DiarizationDeviceComboBox, "auto"),
            BatchSize = RequiredInteger(BatchSizeBox.Value, 8),
            HuggingFaceToken = string.IsNullOrWhiteSpace(HuggingFaceTokenBox.Password)
                ? null
                : HuggingFaceTokenBox.Password,
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
            LibraryFeedCountText.Text = result.Summary.FeedCount.ToString("N0");
            LibraryDayCountText.Text = result.Summary.DayCount.ToString("N0");
            LibraryAttentionCountText.Text = result.Summary.AttentionCount.ToString("N0");
            LibraryCompleteCountText.Text = result.Summary.CompleteCount.ToString("N0");
            ApplyLibraryFilter();
        }
        catch (Exception exception)
        {
            AppendLog($"Library: {exception.Message}");
            LibraryEmptyText.Text = $"The local library could not be loaded: {exception.Message}";
            LibraryEmptyText.Visibility = Visibility.Visible;
        }
        finally
        {
            RefreshLibraryButton.IsEnabled = _worker is not null && _operationCancellation is null;
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
                "network" => day.NeedsNetwork,
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
        if (day is not null)
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
            return selectionVersion;
        }

        LibraryDetailTitleText.Text = day.FeedName;
        LibraryDetailSubtitleText.Text = $"Feed {day.FeedId} · {day.ArchiveDate} · {day.StorageSummary}";
        LibraryDetailInfoBar.Severity = day.IsComplete
            ? InfoBarSeverity.Success
            : day.NeedsNetwork
                ? InfoBarSeverity.Warning
                : InfoBarSeverity.Informational;
        LibraryDetailInfoBar.Title = day.Status;
        LibraryDetailInfoBar.Message =
            $"{day.StatusDetail.TrimEnd('.', ' ')}. Next: {day.NextStep}.";
        LibraryDetailProgress.Value = day.PipelinePercent;
        LibraryDetailStatusText.Text = day.PipelineSummary;

        LibraryDownloadStageText.Text = day.HasCombined
            ? day.RawFileCount > 0
                ? $"✓  1. Archive audio — {day.RawFileCount:N0} source segments retained"
                : "✓  1. Archive audio — verified in the combined recording"
            : day.RawFileCount > 0
                ? $"!  1. Archive audio — {day.RawFileCount:N0} segments retained; coverage needs verification"
                : "○  1. Archive audio — missing; Broadcastify access is required";
        LibraryCombineStageText.Text = day.HasCombined
            ? "✓  2. Combine — one continuous day timeline is ready"
            : "○  2. Combine — waits for complete archive coverage";
        LibraryTranscriptStageText.Text = day.HasTranscript
            ? $"✓  3. Transcription — {day.SegmentCount:N0} timestamped segments"
            : day.HasCombined
                ? "→  3. Transcription — ready to run locally"
                : "○  3. Transcription — waits for combined audio";
        LibraryDiarizationStageText.Text = day.HasDiarization
            ? "✓  4. Speaker labels — diarization is attached"
            : day.HasTranscript
                ? "→  4. Speaker labels — ready to add without repeating transcription"
                : "○  4. Speaker labels — waits for a transcript";
        LibraryAnalysisStageText.Text = day.HasAnalysis
            ? $"✓  5. Event analysis — {day.IncidentCount:N0} incidents saved"
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
        LibraryOpenFolderButton.IsEnabled = Directory.Exists(day.DayDirectory);
        LibraryOpenTranscriptButton.IsEnabled = File.Exists(day.TranscriptPath);

        if (File.Exists(day.CombinedPath))
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
        LibraryTranscriptPreviewText.Text = File.Exists(day.TranscriptPath)
            ? "Loading timestamped transcript preview…"
            : "No transcript is available yet. The Processing tab shows the next step.";
        return selectionVersion;
    }

    private async Task LoadLibraryTranscriptPreviewAsync(LibraryDay day, int selectionVersion)
    {
        if (!File.Exists(day.TranscriptPath))
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

    private async void LibraryOpenFolder_Click(object sender, RoutedEventArgs e)
    {
        if (_selectedLibraryDay is null || !Directory.Exists(_selectedLibraryDay.DayDirectory))
        {
            return;
        }
        try
        {
            var folder = await StorageFolder.GetFolderFromPathAsync(
                Path.GetFullPath(_selectedLibraryDay.DayDirectory));
            await Launcher.LaunchFolderAsync(folder);
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
        try
        {
            var transcript = await StorageFile.GetFileFromPathAsync(
                Path.GetFullPath(_selectedLibraryDay.TranscriptPath));
            await Launcher.LaunchFileAsync(transcript);
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

    private async Task ContinueLibraryDayAsync(LibraryDay day)
    {
        if (_worker is null)
        {
            return;
        }
        if (day.PrimaryAction == "open_review")
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

        _operationCancellation = new CancellationTokenSource();
        SetBusy(true, $"Continuing {day.FeedName} for {day.ArchiveDate}…", jobRunning: true);
        JobProgress.IsIndeterminate = true;
        try
        {
            if (day.NeedsNetwork)
            {
                AppendLog($"Resuming archive coverage for feed {day.FeedId} on {day.ArchiveDate}.");
                await RunAndAnalyzeJobAsync(CreateJobRequest(
                    day.FeedId, archiveDate.Date, archiveDate.Date, minimumSpeakers, maximumSpeakers,
                    string.Equals(day.FeedName, $"Feed {day.FeedId}", StringComparison.Ordinal)
                        ? null
                        : day.FeedName));
            }
            else
            {
                var report = await _worker.ContinueLocalDayAsync(
                    ApplyAnalysisProvider(new LocalProcessingRequest
                    {
                        FeedId = day.FeedId,
                        ArchiveDate = day.ArchiveDate,
                        OutputDirectory = string.IsNullOrWhiteSpace(OutputFolderBox.Text) ? "archives" : OutputFolderBox.Text.Trim(),
                        Model = SelectedComboValue(ModelComboBox, "turbo"),
                        AsrEngine = SelectedComboValue(AsrEngineComboBox, "auto"),
                        Device = SelectedComboValue(DeviceComboBox, "auto"),
                        DeviceIndex = RequiredInteger(GpuIndexBox.Value, 0),
                        AsrModelPath = string.IsNullOrWhiteSpace(AsrModelPathBox.Text)
                            ? null
                            : AsrModelPathBox.Text.Trim(),
                        DiarizationDevice = SelectedComboValue(DiarizationDeviceComboBox, "auto"),
                        BatchSize = RequiredInteger(BatchSizeBox.Value, 8),
                        Diarize = true,
                        Analyze = true,
                        MinimumSpeakers = minimumSpeakers,
                        MaximumSpeakers = maximumSpeakers,
                        HuggingFaceToken = string.IsNullOrWhiteSpace(HuggingFaceTokenBox.Password)
                            ? null
                            : HuggingFaceTokenBox.Password,
                    }),
                    HandleWorkerMessage,
                    _operationCancellation.Token);
                AnalysisFeedBox.Text = day.FeedId;
                if (report is not null)
                {
                    ApplyReport(report);
                }
                await RefreshAnalysisDaysAsync();
            }
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
            _operationCancellation.Dispose();
            _operationCancellation = null;
            JobProgress.IsIndeterminate = false;
            SetBusy(false);
            await RefreshLibraryAsync();
        }
    }

    private async Task OpenLibraryDayInReviewAsync(LibraryDay day)
    {
        AnalysisFeedBox.Text = day.FeedId;
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
            AnalysisFeedBox.Text = _selectedFeed.FeedId;
        }
    }

    private async void SignIn_Click(object sender, RoutedEventArgs e)
    {
        if (_worker is null)
        {
            return;
        }

        var saved = CredentialStore.TryLoad();
        var usernameBox = new TextBox
        {
            Header = "Broadcastify username",
            Text = saved?.Username ?? "",
        };
        var passwordBox = new PasswordBox
        {
            Header = "Password",
            Password = saved?.Password ?? "",
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
            Text = "Credentials are sent only to Broadcastify. When saving is enabled, Windows Credential Locker encrypts the login for this Windows account so an expired session can refresh automatically.",
            TextWrapping = TextWrapping.Wrap,
        });
        fields.Children.Add(usernameBox);
        fields.Children.Add(passwordBox);
        fields.Children.Add(rememberCheckBox);
        fields.Children.Add(errorText);

        var dialog = new ContentDialog
        {
            XamlRoot = ((FrameworkElement)Content).XamlRoot,
            Title = "Sign in for premium archives",
            PrimaryButtonText = "Sign in",
            CloseButtonText = "Cancel",
            DefaultButton = ContentDialogButton.Primary,
            Content = fields,
        };

        dialog.PrimaryButtonClick += async (_, args) =>
        {
            var deferral = args.GetDeferral();
            try
            {
                if (string.IsNullOrWhiteSpace(usernameBox.Text) || string.IsNullOrEmpty(passwordBox.Password))
                {
                    args.Cancel = true;
                    errorText.Text = "Enter both your username and password.";
                    return;
                }
                await _worker.AuthenticateAsync(
                    usernameBox.Text.Trim(), passwordBox.Password, HandleWorkerMessage, CancellationToken.None);
                if (rememberCheckBox.IsChecked == true)
                {
                    CredentialStore.Save(usernameBox.Text.Trim(), passwordBox.Password);
                    SettingsAuthStatusText.Text =
                        $"Saved login for {usernameBox.Text.Trim()} in Windows Credential Locker. Automatic session refresh is enabled.";
                }
                else
                {
                    CredentialStore.Clear();
                    SettingsAuthStatusText.Text = "Signed in for this session; the username and password were not saved.";
                }
                passwordBox.Password = "";
                AuthInfoBar.Severity = InfoBarSeverity.Success;
                AuthInfoBar.Title = "Signed in";
                AuthInfoBar.Message = "Premium archive session is ready and can refresh automatically when a saved login is available.";
                _archiveAccessConfigured = true;
                _archiveAccessVerified = true;
                UpdateSetupSummary();
                AppendLog("Broadcastify sign-in succeeded.");
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
        var saved = CredentialStore.TryLoad();
        if (saved is null)
        {
            _archiveAccessConfigured = _worker.HasBundledEnvironment;
            SettingsAuthStatusText.Text = _worker.HasBundledEnvironment
                ? "A private bundled .env is available. Archive jobs can refresh the Broadcastify session automatically."
                : "No Windows Credential Locker login is saved. An existing session cookie or repository .env can still provide access.";
            ClearSavedLoginButton.IsEnabled = false;
            UpdateSetupSummary();
            return;
        }
        _archiveAccessConfigured = true;
        ClearSavedLoginButton.IsEnabled = true;
        SettingsAuthStatusText.Text = $"Refreshing the saved Broadcastify session for {saved.Username}…";
        try
        {
            using var cancellation = new CancellationTokenSource(TimeSpan.FromSeconds(45));
            await _worker.AuthenticateAsync(
                saved.Username,
                saved.Password,
                HandleWorkerMessage,
                cancellation.Token);
            AuthInfoBar.Severity = InfoBarSeverity.Success;
            AuthInfoBar.Title = "Signed in automatically";
            AuthInfoBar.Message = "Windows Credential Locker supplied the saved login and refreshed the premium session.";
            SettingsAuthStatusText.Text =
                $"Automatic sign-in is enabled for {saved.Username}. The password remains in Windows Credential Locker.";
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
        CredentialStore.Clear();
        ClearSavedLoginButton.IsEnabled = false;
        SettingsAuthStatusText.Text =
            "The saved username and password were removed from Windows Credential Locker. The current session cookie was left intact.";
        _archiveAccessConfigured = _archiveAccessVerified;
        UpdateSetupSummary();
        AuthInfoBar.Severity = InfoBarSeverity.Informational;
        AuthInfoBar.Title = "Saved login removed";
        AuthInfoBar.Message = "You can continue with the current session or sign in again later.";
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
        if (_worker is null)
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

        _operationCancellation = new CancellationTokenSource();
        SetBusy(true, "Starting job…", jobRunning: true);
        JobProgress.IsIndeterminate = true;
        JobProgress.Value = 0;
        AppendLog($"Starting feed {request.FeedId}: {request.StartDate} through {request.EndDate}");
        try
        {
            await RunAndAnalyzeJobAsync(request);
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
            _operationCancellation.Dispose();
            _operationCancellation = null;
            JobProgress.IsIndeterminate = false;
            SetBusy(false);
            await RefreshLibraryAsync();
        }
    }

    private JobRequest CreateJobRequest(
        string feedId,
        DateTime startDate,
        DateTime endDate,
        int? minimumSpeakers,
        int? maximumSpeakers,
        string? feedName = null) => new()
    {
        FeedId = feedId,
        FeedName = feedName?.Trim() ?? "",
        StartDate = startDate.ToString("yyyy-MM-dd"),
        EndDate = endDate.ToString("yyyy-MM-dd"),
        OutputDirectory = string.IsNullOrWhiteSpace(OutputFolderBox.Text) ? "archives" : OutputFolderBox.Text.Trim(),
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
        DiarizationDevice = SelectedComboValue(DiarizationDeviceComboBox, "auto"),
        DownloadJobs = _broadcastifyRateLimitObserved
            ? 1
            : RequiredInteger(DownloadJobsBox.Value, 1),
        BatchSize = RequiredInteger(BatchSizeBox.Value, 8),
        MinimumSpeakers = minimumSpeakers,
        MaximumSpeakers = maximumSpeakers,
        HuggingFaceToken = string.IsNullOrWhiteSpace(HuggingFaceTokenBox.Password)
            ? null
            : HuggingFaceTokenBox.Password,
    };

    private async Task<JobRunResult?> RunAndAnalyzeJobAsync(JobRequest request)
    {
        if (_worker is null || _operationCancellation is null)
        {
            return null;
        }
        var jobResult = await _worker.RunJobAsync(
            request, HandleWorkerMessage, _operationCancellation.Token);
        await AnalyzeCompletedJobAsync(request, jobResult);
        return jobResult;
    }

    private async Task AnalyzeCompletedJobAsync(JobRequest request, JobRunResult? jobResult)
    {
        if (_worker is null || _operationCancellation is null)
        {
            return;
        }
        if (!request.Transcribe || !request.Combine || AnalyzeAfterJobCheckBox.IsChecked != true)
        {
            return;
        }

        DayReport? latestReport = null;
        var transcriptDays = jobResult?.Days
            .Where(day => day.Transcripts.Count > 0)
            .ToList() ?? [];
        foreach (var day in transcriptDays)
        {
            _operationCancellation.Token.ThrowIfCancellationRequested();
            AppendLog($"Analyzing feed {request.FeedId} for {day.ArchiveDate}…");
            latestReport = await _worker.AnalyzeDayAsync(
                ApplyAnalysisProvider(new AnalysisRequest
                {
                    FeedId = request.FeedId,
                    ArchiveDate = day.ArchiveDate,
                    OutputDirectory = request.OutputDirectory,
                }),
                HandleWorkerMessage,
                _operationCancellation.Token);
        }
        if (transcriptDays.Count == 0)
        {
            AppendLog("No completed transcripts were available for analysis.");
        }
        AnalysisFeedBox.Text = request.FeedId;
        if (latestReport is not null)
        {
            ApplyReport(latestReport);
        }
        await RefreshAnalysisDaysAsync();
    }

    private void Cancel_Click(object sender, RoutedEventArgs e) => _operationCancellation?.Cancel();

    private async Task LoadDiagnosticsAndDaysAsync()
    {
        if (_worker is null)
        {
            return;
        }
        try
        {
            using var cancellation = new CancellationTokenSource(TimeSpan.FromSeconds(30));
            var diagnostics = await _worker.GetDiagnosticsAsync(
                cancellation.Token,
                CreateAsrSelfTestRequest());
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
                        + $"pyannote access: {(_pyannoteAccessConfigured ? "configured" : "missing")}";
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
            await RefreshAreaProfilesAsync();
        }
        catch (Exception exception)
        {
            _diagnosticsLoaded = true;
            DiagnosticsInfoBar.Severity = InfoBarSeverity.Warning;
            DiagnosticsInfoBar.Title = "Diagnostics unavailable";
            DiagnosticsInfoBar.Message = exception.Message;
            AppendLog($"Diagnostics: {exception.Message}");
            UpdateSetupSummary();
        }
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
                "Saved analysis predates the current evidence rules. Choose Analyze to rebuild it from the retained transcript without downloading the archive again.";
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

    private async void Ask_Click(object sender, RoutedEventArgs e)
    {
        if (_worker is null)
        {
            return;
        }
        var feedId = AnalysisFeedBox.Text.Trim();
        var question = QuestionBox.Text.Trim();
        var startDate = QuestionStartDatePicker.Date.Date;
        var endDate = QuestionEndDatePicker.Date.Date;
        if (string.IsNullOrWhiteSpace(feedId) || string.IsNullOrWhiteSpace(question))
        {
            await ShowMessageAsync("Question incomplete", "Enter a feed ID and a question.");
            return;
        }
        if (startDate > endDate)
        {
            await ShowMessageAsync("Invalid date range", "Start date must be on or before end date.");
            return;
        }

        _operationCancellation = new CancellationTokenSource();
        SetBusy(true, "Retrieving evidence…", jobRunning: true);
        JobProgress.IsIndeterminate = true;
        AnswerText.Text = $"Working with {SelectedAnalysisProviderDisplayName()}…";
        try
        {
            var answer = await _worker.AskArchiveAsync(
                ApplyAnalysisProvider(new ArchiveQuestionRequest
                {
                    FeedId = feedId,
                    StartDate = startDate.ToString("yyyy-MM-dd"),
                    EndDate = endDate.ToString("yyyy-MM-dd"),
                    Question = question,
                }),
                HandleWorkerMessage,
                _operationCancellation.Token);
            AnswerText.Text = answer is null
                ? "No answer was returned."
                : answer.Answer + (string.IsNullOrWhiteSpace(answer.LimitationsSummary)
                    ? ""
                    : Environment.NewLine + Environment.NewLine + answer.LimitationsSummary);
        }
        catch (OperationCanceledException)
        {
            AnswerText.Text = "Question cancelled.";
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
        AreaZipCodesBox.PlaceholderText = radiusMode ? "12345" : "12345, 12346, 12347";
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
                AreaSummaryText.Text = "";
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
        AreaSummaryText.Text = report.Summary;
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
        if (_worker is null)
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

        _operationCancellation = new CancellationTokenSource();
        SetBusy(true, "Starting explicit multi-feed area job…", jobRunning: true);
        JobProgress.IsIndeterminate = true;
        JobProgress.Value = 0;
        try
        {
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
                _operationCancellation.Token);
            if (result is null)
            {
                AreaCoverageText.Text = "The area worker returned no queue result.";
                return;
            }
            foreach (var feedResult in result.FeedResults)
            {
                _operationCancellation.Token.ThrowIfCancellationRequested();
                await AnalyzeCompletedJobAsync(
                    baseRequest with { FeedId = feedResult.Feed.FeedId },
                    feedResult.Result);
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
            _operationCancellation.Dispose();
            _operationCancellation = null;
            JobProgress.IsIndeterminate = false;
            SetBusy(false);
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
        AreaCoverageText.Text = "Loading saved incidents across selected feeds…";
        AreaSummaryText.Text = $"Working with {SelectedAnalysisProviderDisplayName()}…";
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
                AreaSummaryText.Text = "";
                return;
            }
            ApplyAreaDigest(report);
        }
        catch (OperationCanceledException)
        {
            AreaCoverageText.Text = "Area digest cancelled.";
            AreaSummaryText.Text = "";
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

    private void HandleWorkerMessage(JsonElement message)
    {
        DispatcherQueue.TryEnqueue(() =>
        {
            var type = message.TryGetProperty("type", out var typeValue) ? typeValue.GetString() : "";
            var text = message.TryGetProperty("message", out var messageValue) ? messageValue.GetString() ?? "" : "";
            if (!string.IsNullOrWhiteSpace(text))
            {
                StatusText.Text = text;
                AppendLog(text);
            }
            if (!_broadcastifyRateLimitObserved
                && (text.Contains("HTTP 429", StringComparison.OrdinalIgnoreCase)
                    || text.Contains("download quota", StringComparison.OrdinalIgnoreCase)
                    || text.Contains("quota is exhausted", StringComparison.OrdinalIgnoreCase)))
            {
                _broadcastifyRateLimitObserved = true;
                AppendLog(
                    "Rate-limit protection enabled: future archive jobs in this app session "
                    + "will start with one download at a time.");
            }
            if (type == "progress")
            {
                var current = message.TryGetProperty("current", out var currentValue) ? currentValue.GetInt32() : 0;
                var total = message.TryGetProperty("total", out var totalValue) ? totalValue.GetInt32() : 0;
                JobProgress.IsIndeterminate = total <= 0;
                JobProgress.Maximum = Math.Max(1, total);
                JobProgress.Value = Math.Clamp(current, 0, Math.Max(1, total));
            }
            else if (type == "complete")
            {
                JobProgress.IsIndeterminate = false;
                JobProgress.Value = JobProgress.Maximum;
            }
        });
    }

    private void SetBusy(bool busy, string? status = null, bool jobRunning = false)
    {
        if (busy)
        {
            _settingsSaveTimer?.Stop();
            PersistUserSettings();
            PersistAnalysisCredentialPreference();
        }
        SearchButton.IsEnabled = !busy && _worker is not null;
        StartButton.IsEnabled = !busy && _worker is not null;
        RefreshAnalysisButton.IsEnabled = !busy && _worker is not null;
        AnalyzeSelectedButton.IsEnabled = !busy && _worker is not null;
        ReloadReportButton.IsEnabled = !busy && _worker is not null;
        AskButton.IsEnabled = !busy && _worker is not null;
        GenerateWeekButton.IsEnabled = !busy && _worker is not null;
        DiscoverAreaFeedsButton.IsEnabled = !busy && _worker is not null;
        SaveAreaProfileButton.IsEnabled = !busy && _worker is not null;
        ProcessAreaFeedsButton.IsEnabled = !busy && _worker is not null;
        GenerateAreaDigestButton.IsEnabled = !busy && _worker is not null;
        AnalysisProviderCheckButton.IsEnabled = !busy && _worker is not null;
        AnalysisModelTestButton.IsEnabled = !busy && _worker is not null;
        SetupProfileSelfTestButton.IsEnabled = !busy && _worker is not null;
        ProfileSelfTestButton.IsEnabled = !busy && _worker is not null;
        AsrPrepareButton.IsEnabled = !busy && _worker is not null;
        AsrSelfTestButton.IsEnabled = !busy && _worker is not null;
        DiarizationSelfTestButton.IsEnabled = !busy && _worker is not null;
        SetupAccountActionButton.IsEnabled = !busy && _worker is not null;
        SetupTranscriptionActionButton.IsEnabled = !busy && _worker is not null;
        SetupDiarizationActionButton.IsEnabled = !busy && _worker is not null;
        SetupAnalysisActionButton.IsEnabled = !busy && _worker is not null;
        AreaProfileCombo.IsEnabled = !busy && _worker is not null;
        RefreshLibraryButton.IsEnabled = !busy && _worker is not null;
        LibraryList.IsEnabled = !busy && _worker is not null;
        LibraryDetailPrimaryButton.IsEnabled = !busy && _worker is not null
            && _selectedLibraryDay is not null;
        LibraryDetailReviewButton.IsEnabled = !busy
            && _selectedLibraryDay?.CanOpenReview == true;
        LibraryOpenFolderButton.IsEnabled = !busy
            && _selectedLibraryDay is not null
            && Directory.Exists(_selectedLibraryDay.DayDirectory);
        LibraryOpenTranscriptButton.IsEnabled = !busy
            && _selectedLibraryDay is not null
            && File.Exists(_selectedLibraryDay.TranscriptPath);
        ClearSavedLoginButton.IsEnabled = !busy && CredentialStore.TryLoad() is not null;
        CancelButton.IsEnabled = busy && jobRunning;
        if (status is not null)
        {
            StatusText.Text = status;
        }
    }

    private void AppendLog(string message)
    {
        LogBox.Text += $"[{DateTime.Now:HH:mm:ss}] {message}{Environment.NewLine}";
        LogBox.Select(LogBox.Text.Length, 0);
        AppDiagnostics.AppendActivity(message);
    }

    private async Task ShowErrorAsync(Exception exception)
    {
        StatusText.Text = "Error";
        AppendLog($"ERROR: {exception.Message}");
        await ShowMessageAsync("Broadcastify error", exception.Message);
    }

    private async Task ShowMessageAsync(string title, string message)
    {
        var dialog = new ContentDialog
        {
            XamlRoot = ((FrameworkElement)Content).XamlRoot,
            Title = title,
            Content = message,
            CloseButtonText = "OK",
        };
        await dialog.ShowAsync();
    }

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
}
