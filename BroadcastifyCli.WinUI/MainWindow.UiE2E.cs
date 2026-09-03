using System.Text;
using System.Text.Json;
using System.Runtime.InteropServices;
using Microsoft.UI.Windowing;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Automation;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;
using Windows.Graphics;

namespace BroadcastifyCli.WinUI;

public sealed partial class MainWindow
{
    private sealed record UiProbeSize(
        string Name,
        double WidthDips,
        double HeightDips,
        bool Compact);

    private async void StartUiEndToEndProbe(object sender, RoutedEventArgs e)
    {
        WindowRoot.Loaded -= StartUiEndToEndProbe;
        var reportPath = _uiEndToEndReportPath;
        if (string.IsNullOrWhiteSpace(reportPath))
        {
            return;
        }

        var report = new Dictionary<string, object?>
        {
            ["schema_version"] = 1,
            ["started_at_utc"] = DateTimeOffset.UtcNow,
            ["process_id"] = Environment.ProcessId,
            ["version"] = InstalledVersion(),
            ["test_data_root"] = AppSettingsStore.LocalDataDirectory,
        };
        if (string.Equals(
                Environment.GetEnvironmentVariable(
                    "BROADCASTIFY_DESKTOP_TEST_ABRUPT_EXIT"),
                "1",
                StringComparison.Ordinal))
        {
            var manager = _desktopDockManager
                ?? throw new InvalidOperationException(
                    "The abrupt-exit probe could not initialize desktop docking.");
            Require(
                manager.IsDocked,
                "The abrupt-exit probe did not restore its isolated dock preference.");
            await WaitForUiLayoutAsync(180);
            var snapshot = manager.CaptureSnapshot();
            Require(
                snapshot.AppBarRegistered,
                "The abrupt-exit probe did not register its AppBar.");
            report["abrupt_exit_ready"] = true;
            report["desktop_docking"] = new Dictionary<string, object?>
            {
                ["side"] = manager.Side.ToString().ToLowerInvariant(),
                ["window"] = RectToReport(snapshot.WindowBounds),
                ["work_area"] = RectToReport(snapshot.WorkArea),
            };
            report["passed"] = true;
            report["finished_at_utc"] = DateTimeOffset.UtcNow;
            await WriteUiProbeReportAsync(reportPath, report);
            Environment.Exit(73);
            return;
        }
        var passed = false;
        try
        {
            report["startup_docking"] = await VerifyStartupDockRestoreAsync();
            ConfigureUiEndToEndFixture();
            report["responsive_breakpoint"] = VerifyResponsiveBreakpointContract();
            report["layout_matrix"] = await RunUiLayoutMatrixAsync();
            report["desktop_docking"] = await RunUiDockingProbeAsync();
            report["passed"] = true;
            passed = true;
        }
        catch (Exception exception)
        {
            report["passed"] = false;
            report["error"] = exception.Message;
            report["exception"] = exception.ToString();
        }
        finally
        {
            if (_desktopDockManager?.IsDocked == true)
            {
                UnpinDesktop();
            }
            report["finished_at_utc"] = DateTimeOffset.UtcNow;
            try
            {
                await WriteUiProbeReportAsync(reportPath, report);
            }
            catch (Exception exception)
            {
                AppDiagnostics.AppendCrash(exception, "UI end-to-end report");
                passed = false;
            }
            Environment.ExitCode = passed ? 0 : 1;
            TitleBarCloseButton_Click(
                TitleBarCloseButton,
                new RoutedEventArgs());
        }
    }

    private async Task<Dictionary<string, object?>> VerifyStartupDockRestoreAsync()
    {
        var manager = _desktopDockManager
            ?? throw new InvalidOperationException(
                "The desktop docking manager was unavailable at startup.");
        Require(
            manager.IsDocked,
            "The isolated saved dock preference was not restored at startup.");
        await WaitForStableCustomTitleChromeAsync();
        var snapshot = manager.CaptureSnapshot();
        Require(
            manager.Side == DesktopDockSide.Right,
            "The isolated saved right-edge preference was not restored.");
        const double savedWidthDips = 640;
        var expectedWidthDips = DesktopDockPolicy.NormalizeWidthDips(
            savedWidthDips,
            DesktopDockPolicy.PixelsToDips(
                snapshot.MonitorBounds.Width,
                snapshot.Dpi));
        Require(
            Math.Abs(manager.WidthDips - expectedWidthDips) <= 8,
            "The isolated saved dock width was not restored within the "
                + "current display's safe width constraint.");
        Require(
            snapshot.AppBarRegistered,
            "Startup docking did not register its AppBar.");
        Require(
            snapshot.WorkArea.X + snapshot.WorkArea.Width
                <= snapshot.WindowBounds.X + 8,
            "Startup docking did not reserve the saved right edge.");
        Require(
            TitleBarPaneButton.Visibility == Visibility.Visible
                && TitleBarMinimizeButton.Visibility == Visibility.Collapsed
                && TitleBarMaximizeButton.Visibility == Visibility.Collapsed
                && TitleBarCloseButton.Visibility == Visibility.Visible,
            "Startup docking did not apply the pinned custom chrome controls.");
        VerifyCustomTitleChromeBounds("startup-pinned", pinned: true);
        var result = new Dictionary<string, object?>
        {
            ["side"] = manager.Side.ToString().ToLowerInvariant(),
            ["requested_width_dips"] = savedWidthDips,
            ["expected_width_dips"] = expectedWidthDips,
            ["width_dips"] = manager.WidthDips,
            ["monitor"] = snapshot.MonitorDeviceName,
            ["dpi"] = snapshot.Dpi,
            ["window"] = RectToReport(snapshot.WindowBounds),
            ["work_area"] = RectToReport(snapshot.WorkArea),
        };
        UnpinDesktop();
        await WaitForUiLayoutAsync(180);
        Require(
            !manager.IsDocked,
            "The startup restore probe could not return to floating mode.");
        VerifyFloatingTitleBarState("startup-unpin");
        return result;
    }

    private void ConfigureUiEndToEndFixture()
    {
        const string repeatedStatus =
            "Retained local work is safe. Download, combine, transcription, "
            + "speaker labeling, and analysis checkpoints remain independently resumable. ";
        // Keep the isolated fixture deterministic instead of allowing the
        // navigation refresh to replace it with this machine's live LAN state.
        _refreshingSystemActivity = true;
        _visibleLibraryDays.Clear();
        foreach (var index in Enumerable.Range(1, 48))
        {
            _visibleLibraryDays.Add(new LibraryDay
            {
                FeedId = "999991",
                FeedName = $"Synthetic retained feed {index:00}",
                ArchiveDate = $"2026-07-{((index - 1) % 28) + 1:00}",
                Status = index % 3 == 0 ? "Ready" : "Processing",
                NextStep = index % 3 == 0 ? "Review evidence" : "Resume local work",
                PipelinePercent = index % 3 == 0 ? 100 : 65,
                StorageBytes = index * 1_048_576L,
            });
        }
        _feeds.Clear();
        _areaFeeds.Clear();
        foreach (var index in Enumerable.Range(1, 42))
        {
            var feed = new FeedSearchResult
            {
                FeedId = $"99{index:0000}",
                Name = $"Synthetic agency feed {index:00}",
                Location = "Isolated test county",
                Description =
                    "Local-only feed search result used to verify list scrolling and bounds.",
                Genre = "Public Safety",
                Listeners = index,
                PriorityRank = index,
                NearestZipCode = "00000",
                DistanceMiles = index / 2.0,
                MatchedZipCodes = ["00000"],
            };
            _feeds.Add(feed);
            _areaFeeds.Add(feed);
        }
        _libraryFeeds.Clear();
        foreach (var index in Enumerable.Range(1, 30))
        {
            _libraryFeeds.Add(new LibraryFeedCoverage
            {
                FeedId = $"88{index:0000}",
                FeedName = $"Synthetic retained coverage {index:00}",
                CatchUpSaved = true,
                CatchUpThroughCurrent = true,
                CatchUpStartDate = "2026-06-01",
                CatchUpEndDate = "2026-08-12",
                TargetDayCount = 73,
                RetainedDayCount = 40 + index,
                ReadyDayCount = 30 + index,
                IncompleteDayCount = 4,
                MissingDayCount = Math.Max(0, 29 - index),
                LocalProcessingDayCount = 4,
                NetworkDayCount = Math.Max(0, 29 - index),
                BacklogCount = 8,
                ProgressPercent = Math.Min(100, 54 + index),
                Status = "Missing and unfinished days remain resumable",
                KnownSourceBlockCount = 240,
                RetainedSourceBlockCount = 180 + index,
                MissingSourceBlockCount = 60 - index,
                LastSourceCheckAt = "2026-08-12T14:30:10-05:00",
            });
        }
        _analysisDays.Clear();
        foreach (var index in Enumerable.Range(1, 36))
        {
            _analysisDays.Add(new AnalysisDay
            {
                FeedId = "999991",
                FeedName = $"Synthetic review day {index:00}",
                ArchiveDate = $"2026-06-{((index - 1) % 28) + 1:00}",
                SegmentCount = 240 + index,
                IncidentCount = index % 7,
                DurationSeconds = 86_400,
                HasDiarizationValue = 1,
                SpeakerCount = 5,
            });
        }
        _visibleIncidents.Clear();
        foreach (var index in Enumerable.Range(1, 40))
        {
            _visibleIncidents.Add(new IncidentRecord
            {
                Id = index,
                EventType = "synthetic_event",
                Title = $"Synthetic cited event {index:00}",
                Summary = "Review-required local fixture incident.",
                Location = "Test location",
                Priority = (index % 3) + 1,
                Confidence = 0.75,
                ArchiveTime = $"2026-07-{((index - 1) % 28) + 1:00} 14:{index % 60:00}:10",
                EvidenceQuote = "Synthetic transcript evidence.",
            });
        }
        _areaStories.Clear();
        foreach (var index in Enumerable.Range(1, 34))
        {
            _areaStories.Add(new AreaStory
            {
                StoryId = $"synthetic-{index}",
                Headline = $"Synthetic ranked assignment lead {index:00}",
                Summary = "Local-only story fixture.",
                EventType = "synthetic_event",
                Location = "Test location",
                FirstReported = $"2026-08-01 14:{index % 60:00}:10",
                NewsworthinessScore = 80 - (index % 20),
                InterestLevel = "Review",
                Priority = (index % 3) + 1,
                EvidenceClipCount = 1,
                QuoteCount = 1,
            });
        }
        _hardwareProfiles.Clear();
        foreach (var index in Enumerable.Range(1, 18))
        {
            _hardwareProfiles.Add(new HardwareProfileStatus
            {
                Id = $"synthetic-{index}",
                Name = $"Synthetic processing profile {index:00}",
                Configured = true,
                Transcription = "isolated",
                Diarization = "isolated",
                Analysis = "isolated",
                Note = "No model or provider is invoked by this fixture.",
            });
        }
        LibraryDetailEmptyPanel.Visibility = Visibility.Collapsed;
        LibraryDetailPanel.Visibility = Visibility.Visible;
        LibraryDetailTitleText.Text = "Synthetic UI fixture · feed 999991";
        LibraryDetailSubtitleText.Text = "2026-08-01 · Ready to review";
        LibraryDetailStatusText.Text = string.Concat(
            Enumerable.Repeat(repeatedStatus, 8));
        LibrarySourceCoverageText.Text = string.Concat(
            Enumerable.Repeat(
                "All retained blocks were provided by the isolated local fixture. ",
                8));
        LibraryDownloadStageText.Text = "✓ Download — retained local fixture";
        LibraryCombineStageText.Text = "✓ Combine — complete";
        LibraryTranscriptStageText.Text = "✓ Transcript — complete";
        LibraryDiarizationStageText.Text = "✓ Speakers — complete";
        LibraryAnalysisStageText.Text = "✓ Analysis — complete";
        LibraryPathText.Text = AppSettingsStore.LocalDataDirectory;
        LibraryDetailPrimaryButton.Content = "Open review";
        LibraryTranscriptPreviewText.Text = string.Join(
            Environment.NewLine,
            Enumerable.Range(1, 160).Select(index =>
                $"[00:{index / 60:00}:{index % 60:00}.000] SPEAKER_01: Synthetic transcript line {index}."));

        WeekSummaryText.Text = string.Join(
            Environment.NewLine,
            Enumerable.Range(1, 36).Select(index =>
                $"Day {index}: synthetic evidence line with an explicit archive date and time."));
        WeekCoverageText.Text = "7/7 isolated fixture days available.";

        _archiveChatMessages.Clear();
        _archiveChatMessages.Add(new ArchiveChatMessage
        {
            Role = "assistant",
            Content =
                "A reported event was retained [E1].\n\n"
                + "Cited event dates and times (archive time):\n"
                + string.Join(
                    "\n",
                    Enumerable.Range(0, 18).Select(index =>
                        $"- E{index + 1} — 2026-07-{index + 1:00} 14:{index:00}:10 — retrieved transcript evidence")),
            EvidenceIds = ["E1"],
            Coverage = new ArchiveQuestionCoverage
            {
                RequestedDayCount = 31,
                QuestionReadyDayCount = 18,
                AudioDayCount = 20,
                Summary = "18/31 days are question-ready; uncovered dates remain explicit gaps.",
            },
        });
        foreach (var index in Enumerable.Range(2, 20))
        {
            _archiveChatMessages.Add(new ArchiveChatMessage
            {
                Role = index % 2 == 0 ? "user" : "assistant",
                Content =
                    $"Synthetic bounded conversation turn {index}. "
                    + "Every retained event remains tied to an archive day and time.",
                EvidenceIds = index % 2 == 0 ? [] : [$"E{index}"],
            });
        }

        AreaSummaryExpander.Visibility = Visibility.Visible;
        AreaSummaryExpander.IsExpanded = true;
        AreaSummaryText.Text = string.Join(
            Environment.NewLine,
            Enumerable.Range(1, 24).Select(index =>
                $"Synthetic assignment lead {index} retains its source day, time, and limitations."));
        AreaStoryDetailEmptyText.Visibility = Visibility.Collapsed;
        AreaStoryDetailScroll.Visibility = Visibility.Visible;
        AreaStoryDetailHeadlineText.Text = "Synthetic evidence-backed assignment lead";
        AreaStoryDetailTimePlaceText.Text = "2026-08-01 14:30:10 archive time";
        AreaStoryDetailSummaryText.Text = string.Concat(
            Enumerable.Repeat(
                "This local-only fixture verifies scrolling and responsive layout without a provider request. ",
                24));
        AreaStoryDetailWhyText.Text = string.Concat(
            Enumerable.Repeat("Exact structured evidence remains review-required. ", 12));
        AreaStoryDetailEvidenceSummaryText.Text = "18 synthetic evidence references";
        AreaStoryDetailAudienceText.Text = string.Concat(
            Enumerable.Repeat("Coverage gaps are not inactivity. ", 12));

        _visibleActivityLog.Clear();
        foreach (var index in Enumerable.Range(1, 60))
        {
            _visibleActivityLog.AppendLine($"Isolated activity line {index}");
        }
        RefreshVisibleLog();
        ArchiveQuotaInfoBar.Title = "Synthetic rolling archive guard";
        ArchiveQuotaInfoBar.Message = string.Concat(
            Enumerable.Repeat(
                "No live request is made; cached and local work remain available while every quota boundary stays visible. ",
                16));
        FeedScheduleInfoBar.Title = "Synthetic recurring catch-up status";
        FeedScheduleInfoBar.Message = string.Concat(
            Enumerable.Repeat(
                "Missing and unfinished days remain sequential, resumable, and explicitly bounded through current. ",
                16));
        SystemActivityInfoBar.Title = "Synthetic coordinated activity";
        SystemActivityInfoBar.Message =
            "Isolated multi-node state verifies the pinned and floating activity surface.";
        SystemAcquisitionText.Text =
            "Synthetic optional follower is acquiring feed 999991 for 2026-08-01 with the secondary account profile.";
        SystemAcquisitionDetailText.Text = string.Concat(
            Enumerable.Repeat(
                "The global lease remains sequential and retained blocks are reusable. ",
                10));
        SystemProcessingText.Text = string.Join(
            Environment.NewLine,
            Enumerable.Range(1, 24).Select(index =>
                $"• synthetic-node-{index:00} · feed 999991 · 2026-07-{index:00}"));
        SystemQuotaText.Text = string.Join(
            Environment.NewLine,
            Enumerable.Range(1, 12).Select(index =>
                $"• synthetic-profile-{index:00}: {index}/240 used · {240 - index} remaining"));
        SystemSchedulesText.Text = string.Join(
            Environment.NewLine,
            Enumerable.Range(1, 32).Select(index =>
                $"• Coordinator · synthetic recurring feed {index:00} · retained catch-up active"));
        SetupStorageDetailText.Text = string.Join(
            Environment.NewLine,
            Enumerable.Range(1, 56).Select(index =>
                $"Synthetic setup readiness line {index}: retained data stays in the isolated fixture."));
        AnalysisProviderStatusText.Text = string.Join(
            Environment.NewLine,
            Enumerable.Range(1, 40).Select(index =>
                $"Synthetic provider contract line {index}: local fixture only; no model, credential, or network request."));
        SettingsEnvironmentText.Text = string.Join(
            Environment.NewLine,
            Enumerable.Range(1, 56).Select(index =>
                $"Synthetic credential safety line {index}: no secret or account value is loaded."));
        RefreshAboutPage();
        AboutActionStatusText.Text = string.Join(
            Environment.NewLine,
            Enumerable.Range(1, 72).Select(index =>
                $"Synthetic support detail {index}: no credential, token, transcript, or archive content."));
    }

    private async Task<List<Dictionary<string, object?>>> RunUiLayoutMatrixAsync()
    {
        var sizes = new[]
        {
            new UiProbeSize("pinned-width", 600, 900, true),
            new UiProbeSize("minimum", 520, 640, true),
            new UiProbeSize("barely-overflowing", 680, 840, true),
            new UiProbeSize("compact-short", 720, 720, true),
            new UiProbeSize("medium", 960, 720, true),
            new UiProbeSize("threshold-below", 1099, 760, true),
            new UiProbeSize("threshold-above", 1101, 760, false),
            new UiProbeSize("reference", 1240, 900, false),
            new UiProbeSize("large", 1600, 1000, false),
        };
        var results = new List<Dictionary<string, object?>>();
        foreach (var size in sizes)
        {
            results.Add(await VerifyUiLayoutAtSizeAsync(size));
        }
        return results;
    }

    private async Task<Dictionary<string, object?>> VerifyUiLayoutAtSizeAsync(
        UiProbeSize size)
    {
        var reachableSize = GetReachableClientSize(size);
        await ResizeForClientSizeAsync(
            reachableSize.WidthDips,
            reachableSize.HeightDips);
        Require(
            Math.Abs(WindowRoot.ActualWidth - reachableSize.WidthDips) <= 3
                && Math.Abs(WindowRoot.ActualHeight - reachableSize.HeightDips) <= 3,
            $"{size.Name}: targeted a reachable "
            + $"{reachableSize.WidthDips:N0}×{reachableSize.HeightDips:N0}-DIP client "
            + $"but rendered {WindowRoot.ActualWidth:N1}×{WindowRoot.ActualHeight:N1} DIP.");
        var renderedCompact = WindowRoot.ActualWidth < 1_100;
        Require(
            _compactLayoutApplied == renderedCompact,
            $"{size.Name}: compact layout state did not match the rendered width.");
        Require(
            reachableSize.WidthConstrained || _compactLayoutApplied == size.Compact,
            $"{size.Name}: an unconstrained client did not match the requested width contract.");

        var scrollResults = new Dictionary<string, double>();
        var listResults = new Dictionary<string, int>();

        NavigateTo(LibraryNavigationItem);
        await WaitForUiLayoutAsync();
        RequireOnlyPageVisible("library");
        RequireHorizontalBounds(
            size.Name,
            LibraryPage,
            LibraryHeaderActions,
            LibraryFilterGrid,
            LibraryContentBorder,
            PersistentStatusGrid);
        if (_compactLayoutApplied == true && WindowRoot.ActualHeight < 820)
        {
            Require(
                LibraryStatsGrid.Visibility == Visibility.Collapsed,
                $"{size.Name}: short compact layout did not preserve Library work space.");
        }
        else
        {
            RequireHorizontalBounds(size.Name, LibraryStatsGrid);
        }
        scrollResults["library_processing"] = await ExerciseScrollAsync(
            LibraryProcessingScroll,
            $"{size.Name}/library-processing");
        listResults["library_days"] = await ExerciseItemsScrollAsync(
            LibraryList,
            $"{size.Name}/library-days");
        LibraryDetailTabView.SelectedItem = LibraryTranscriptTab;
        await WaitForUiLayoutAsync(40);
        listResults["library_transcript"] = await ExerciseTextBoxScrollAsync(
            LibraryTranscriptPreviewText,
            $"{size.Name}/library-transcript");
        LibraryDetailTabView.SelectedItem = LibraryProcessingTab;
        LibraryCoverageExpander.IsExpanded = true;
        await WaitForUiLayoutAsync(40);
        listResults["library_feed_coverage"] = await ExerciseItemsScrollAsync(
            LibraryFeedCoverageList,
            $"{size.Name}/library-feed-coverage");
        LibraryCoverageExpander.IsExpanded = false;
        await WaitForUiLayoutAsync(40);

        NavigateTo(SystemNavigationItem);
        await WaitForUiLayoutAsync();
        RequireOnlyPageVisible("system");
        RequireHorizontalBounds(
            size.Name,
            SystemPage,
            RefreshSystemActivityButton,
            SystemActivityScroll,
            SystemActivityInfoBar,
            PersistentStatusGrid);
        scrollResults["system_activity"] = await ExerciseScrollAsync(
            SystemActivityScroll,
            $"{size.Name}/system-activity");

        NavigateTo(ArchiveNavigationItem);
        await WaitForUiLayoutAsync();
        RequireOnlyPageVisible("archive");
        RequireHorizontalBounds(
            size.Name,
            ArchivePage,
            ArchiveFeedSearchBorder,
            ArchiveJobBorder,
            StartButton,
            ScheduleFeedButton,
            PersistentStatusGrid);
        scrollResults["archive_job"] = await ExerciseScrollAsync(
            ArchiveJobScroll,
            $"{size.Name}/archive-job");
        listResults["archive_feed_results"] = await ExerciseItemsScrollAsync(
            FeedResults,
            $"{size.Name}/archive-feed-results");
        ActivityLogExpander.IsExpanded = true;
        await WaitForUiLayoutAsync(40);
        listResults["activity_log"] = await ExerciseTextBoxScrollAsync(
            LogBox,
            $"{size.Name}/activity-log");

        NavigateTo(ReviewNavigationItem);
        ReviewTabView.SelectedItem = DailyReviewTab;
        await WaitForUiLayoutAsync();
        RequireHorizontalBounds(
            size.Name,
            ReviewPage,
            ReviewTabView,
            AnalysisDaysList,
            IncidentList);
        listResults["review_days"] = await ExerciseItemsScrollAsync(
            AnalysisDaysList,
            $"{size.Name}/review-days");
        listResults["review_incidents"] = await ExerciseItemsScrollAsync(
            IncidentList,
            $"{size.Name}/review-incidents");
        ReviewTabView.SelectedItem = WeeklyBriefTab;
        await WaitForUiLayoutAsync();
        RequireOnlyPageVisible("review");
        RequireHorizontalBounds(
            size.Name,
            ReviewPage,
            ReviewDaySelectorBorder,
            ReviewTabView,
            WeekControls,
            GenerateWeekButton,
            PersistentStatusGrid);
        scrollResults["weekly_brief"] = await ExerciseScrollAsync(
            WeeklyBriefScroll,
            $"{size.Name}/weekly-brief");

        ReviewTabView.SelectedItem = ArchiveQuestionTab;
        await WaitForUiLayoutAsync();
        RequireHorizontalBounds(
            size.Name,
            ReviewPage,
            ReviewTabView,
            ArchiveQuestionScopePanel,
            QuestionStarterPrimary,
            QuestionStarterSecondary,
            QuestionSendGrid,
            AskButton);
        scrollResults["archive_chat"] = await ExerciseScrollAsync(
            ArchiveChatScroll,
            $"{size.Name}/archive-chat");
        listResults["archive_chat_messages"] = await ExerciseItemsScrollAsync(
            ArchiveChatList,
            $"{size.Name}/archive-chat-messages");
        if (size.Name == "pinned-width")
        {
            await VerifyMonthQuestionUiAsync();
        }

        NavigateTo(AreaNavigationItem);
        AreaTabView.SelectedItem = AreaFeedsTab;
        await WaitForUiLayoutAsync();
        RequireOnlyPageVisible("area");
        RequireHorizontalBounds(
            size.Name,
            AreaPage,
            AreaTabView,
            AreaFeedDiscoveryBorder,
            AreaQueueBorder,
            ProcessAreaFeedsButton,
            PersistentStatusGrid);
        listResults["area_feed_results"] = await ExerciseItemsScrollAsync(
            AreaFeedResults,
            $"{size.Name}/area-feed-results");
        AreaTabView.SelectedItem = AreaStoriesTab;
        await WaitForUiLayoutAsync();
        RequireHorizontalBounds(
            size.Name,
            AreaPage,
            AreaStoryContentGrid,
            AreaStoryListBorder,
            AreaStoryDetailBorder,
            AreaStoryDetailScroll);
        scrollResults["area_summary"] = await ExerciseScrollAsync(
            AreaSummaryScroll,
            $"{size.Name}/area-summary");
        scrollResults["area_story_detail"] = await ExerciseScrollAsync(
            AreaStoryDetailScroll,
            $"{size.Name}/area-story-detail");
        listResults["area_stories"] = await ExerciseItemsScrollAsync(
            AreaStoryList,
            $"{size.Name}/area-stories");

        ShowPage("settings");
        SettingsTabView.SelectedItem = SetupSettingsTab;
        await WaitForUiLayoutAsync();
        RequireOnlyPageVisible("settings");
        RequireHorizontalBounds(
            size.Name,
            SettingsPage,
            SettingsTabView,
            SetupSettingsScroll,
            SetupReadinessInfoBar,
            PersistentStatusGrid);
        scrollResults["settings_setup"] = await ExerciseScrollAsync(
            SetupSettingsScroll,
            $"{size.Name}/settings-setup");

        SettingsTabView.SelectedItem = ProcessingSettingsTab;
        await WaitForUiLayoutAsync();
        HardwareProfilesExpander.IsExpanded = true;
        Require(
            HardwareProfileList.Items.Count >= 2,
            $"{size.Name}/hardware-profiles: the fixture did not populate the profile list.");
        RequireHorizontalBounds(
            size.Name,
            SettingsPage,
            ProcessingSettingsScroll,
            DiagnosticsInfoBar,
            OutputFolderBox);
        scrollResults["settings_processing"] = await ExerciseScrollAsync(
            ProcessingSettingsScroll,
            $"{size.Name}/settings-processing");

        SettingsTabView.SelectedItem = AnalysisSettingsTab;
        await WaitForUiLayoutAsync();
        RequireHorizontalBounds(
            size.Name,
            SettingsPage,
            AnalysisSettingsScroll,
            AnalysisProviderComboBox,
            AnalysisModelBox);
        scrollResults["settings_analysis"] = await ExerciseScrollAsync(
            AnalysisSettingsScroll,
            $"{size.Name}/settings-analysis");

        SettingsTabView.SelectedItem = AccountSettingsTab;
        await WaitForUiLayoutAsync();
        RequireHorizontalBounds(
            size.Name,
            SettingsPage,
            CredentialsSettingsScroll,
            CredentialButtons,
            CredentialLinks);
        scrollResults["settings_credentials"] = await ExerciseScrollAsync(
            CredentialsSettingsScroll,
            $"{size.Name}/settings-credentials");

        NavigateTo(AboutNavigationItem);
        await WaitForUiLayoutAsync();
        RequireOnlyPageVisible("about");
        RequireHorizontalBounds(
            size.Name,
            AboutPage,
            AboutScroll,
            AboutVersionText,
            AboutActionButtons,
            AboutHelpLinks,
            PersistentStatusGrid);
        Require(
            AboutVersionText.Text.StartsWith("Version ", StringComparison.Ordinal),
            $"{size.Name}: About did not render an installed version.");
        scrollResults["about"] = await ExerciseScrollAsync(
            AboutScroll,
            $"{size.Name}/about");

        return new Dictionary<string, object?>
        {
            ["name"] = size.Name,
            ["requested_width_dips"] = size.WidthDips,
            ["requested_height_dips"] = size.HeightDips,
            ["target_width_dips"] = reachableSize.WidthDips,
            ["target_height_dips"] = reachableSize.HeightDips,
            ["display_constrained"] =
                reachableSize.WidthConstrained || reachableSize.HeightConstrained,
            ["width_constrained"] = reachableSize.WidthConstrained,
            ["height_constrained"] = reachableSize.HeightConstrained,
            ["rendered_width_dips"] = Math.Round(WindowRoot.ActualWidth, 1),
            ["rendered_height_dips"] = Math.Round(WindowRoot.ActualHeight, 1),
            ["compact"] = _compactLayoutApplied,
            ["scrollable_heights"] = scrollResults,
            ["scrollable_lists"] = listResults,
        };
    }

    private (
        double WidthDips,
        double HeightDips,
        bool WidthConstrained,
        bool HeightConstrained) GetReachableClientSize(UiProbeSize size)
    {
        var manager = _desktopDockManager
            ?? throw new InvalidOperationException(
                "The desktop docking manager was unavailable for display sizing.");
        var snapshot = manager.CaptureSnapshot();
        var dpiScale = Math.Max(1.0, snapshot.Dpi / 96.0);
        var nonClientWidthPixels = Math.Max(
            0,
            AppWindow.Size.Width
                - (int)Math.Round(WindowRoot.ActualWidth * dpiScale));
        var nonClientHeightPixels = Math.Max(
            0,
            AppWindow.Size.Height
                - (int)Math.Round(WindowRoot.ActualHeight * dpiScale));
        var maximumWidthDips = Math.Max(
            1,
            Math.Floor(
                Math.Max(1, snapshot.WorkArea.Width - nonClientWidthPixels)
                    / dpiScale));
        var maximumHeightDips = Math.Max(
            1,
            Math.Floor(
                Math.Max(1, snapshot.WorkArea.Height - nonClientHeightPixels)
                    / dpiScale));
        var targetWidthDips = Math.Min(size.WidthDips, maximumWidthDips);
        var targetHeightDips = Math.Min(size.HeightDips, maximumHeightDips);
        return (
            targetWidthDips,
            targetHeightDips,
            targetWidthDips < size.WidthDips - 1,
            targetHeightDips < size.HeightDips - 1);
    }

    private Dictionary<string, object> VerifyResponsiveBreakpointContract()
    {
        ApplyResponsiveLayout(1_099);
        Require(
            _compactLayoutApplied == true,
            "The responsive layout did not enter compact mode below its breakpoint.");
        ApplyResponsiveLayout(1_101);
        Require(
            _compactLayoutApplied == false,
            "The responsive layout did not leave compact mode above its breakpoint.");
        ApplyResponsiveLayout(WindowRoot.ActualWidth);
        return new Dictionary<string, object>
        {
            ["below_width_dips"] = 1_099,
            ["below_compact"] = true,
            ["above_width_dips"] = 1_101,
            ["above_compact"] = false,
        };
    }

    private async Task ResizeForClientSizeAsync(
        double widthDips,
        double heightDips)
    {
        var dpiScale = Math.Max(
            1.0,
            GetDpiForWindow(WinRT.Interop.WindowNative.GetWindowHandle(this)) / 96.0);
        for (var attempt = 0; attempt < 3; attempt++)
        {
            var widthDelta = widthDips - WindowRoot.ActualWidth;
            var heightDelta = heightDips - WindowRoot.ActualHeight;
            if (Math.Abs(widthDelta) <= 2 && Math.Abs(heightDelta) <= 2)
            {
                return;
            }
            AppWindow.Resize(new SizeInt32(
                Math.Max(
                    320,
                    AppWindow.Size.Width
                        + (int)Math.Round(widthDelta * dpiScale)),
                Math.Max(
                    320,
                    AppWindow.Size.Height
                        + (int)Math.Round(heightDelta * dpiScale))));
            await WaitForUiLayoutAsync(150);
        }
    }

    private async Task VerifyMonthQuestionUiAsync()
    {
        var selectedMonth = new DateTime(DateTime.Today.Year, DateTime.Today.Month, 1)
            .AddMonths(-1);
        QuestionMonthPicker.Date = new DateTimeOffset(selectedMonth);
        Require(
            await ApplyQuestionMonthAsync(),
            "The month scope action rejected a valid prior month.");
        Require(
            QuestionStartDatePicker.Date.Date == selectedMonth,
            "The month scope did not start on the first calendar day.");
        Require(
            QuestionEndDatePicker.Date.Date == selectedMonth.AddMonths(1).AddDays(-1),
            "The month scope did not end on the final calendar day.");
        AskMonthExample_Click(AskButton, new RoutedEventArgs());
        await WaitForUiLayoutAsync(80);
        Require(
            QuestionBox.Text.Contains(
                "State the archive date and time for every event mentioned",
                StringComparison.Ordinal),
            "The month summary starter omitted the required event day/time instruction.");
    }

    private async Task<Dictionary<string, object?>> RunUiDockingProbeAsync()
    {
        var manager = _desktopDockManager
            ?? throw new InvalidOperationException("The desktop docking manager was unavailable.");
        var baseline = manager.CaptureSnapshot();
        var monitorBaselines = manager.CaptureAvailableMonitors();
        var dpiPolicy = VerifyDpiAndWidthPolicy();
        VerifyFloatingTitleBarState("startup-floating");

        Require(
            TryDockDesktop(DesktopDockSide.Left, persist: false),
            "Pin-left failed.");
        await WaitForUiLayoutAsync(250);
        var left = manager.CaptureSnapshot();
        VerifyDockedFrameState(DesktopDockSide.Left, left, "left");
        var leftSurfaces = await VerifyDockedSurfaceMatrixAsync("left");
        var leftDialog = await VerifyDockedDialogAndFlyoutAsync("left");

        Require(
            TryDockDesktop(DesktopDockSide.Right, persist: false),
            "Pin-right failed.");
        await WaitForUiLayoutAsync(250);
        var rightBeforeResize = manager.CaptureSnapshot();
        VerifyDockedFrameState(
            DesktopDockSide.Right,
            rightBeforeResize,
            "right");
        var rightSurfaces = await VerifyDockedSurfaceMatrixAsync("right");
        var rightDialog = await VerifyDockedDialogAndFlyoutAsync("right");

        var rightMonitorWidthDips = DesktopDockPolicy.PixelsToDips(
            rightBeforeResize.MonitorBounds.Width,
            rightBeforeResize.Dpi);
        var resizeMinimumDips = DesktopDockPolicy.NormalizeWidthDips(
            DesktopDockPolicy.MinimumWidthDips,
            rightMonitorWidthDips);
        var resizeMaximumDips = DesktopDockPolicy.NormalizeWidthDips(
            DesktopDockPolicy.MaximumWidthDips,
            rightMonitorWidthDips);
        var hasResizeRange = resizeMaximumDips - resizeMinimumDips >= 16;
        var resizeTowardLarger = hasResizeRange
            && resizeMaximumDips - manager.WidthDips >= 16;
        var resizePointerDeltaPixels = resizeTowardLarger ? -64 : 64;
        var resize = manager.ExerciseInteractiveResizeForEndToEndTest(
            resizePointerDeltaPixels);
        await WaitForUiLayoutAsync(180);
        var rightAfterResize = resize.After;
        Require(
            !resize.During.AppBarRegistered
                && resize.During.InteractiveResize,
            "Interactive resize did not temporarily release the AppBar reservation.");
        var resizeWidthDeltaPixels = resize.During.WindowBounds.Width
            - resize.Before.WindowBounds.Width;
        if (hasResizeRange)
        {
            Require(
                Math.Abs(resizeWidthDeltaPixels) >= 16
                    && (resizeTowardLarger
                        ? resizeWidthDeltaPixels > 0
                        : resizeWidthDeltaPixels < 0),
                "Captured pointer resizing did not move toward the available legal bound.");
        }
        else
        {
            Require(
                Math.Abs(resizeWidthDeltaPixels) <= 8,
                "A fixed-width narrow display escaped its legal dock-width bound.");
        }
        Require(
            resize.After.AppBarRegistered
                && !resize.After.InteractiveResize,
            "Completing resize did not reclaim the AppBar reservation.");
        Require(
            rightAfterResize.WorkArea.X + rightAfterResize.WorkArea.Width
                <= rightAfterResize.WindowBounds.X + 8,
            "Completing resize did not restore right-edge work-area reservation.");

        _desktopDockWidth = manager.WidthDips;
        _desktopDockMonitor = manager.MonitorDeviceName;
        PersistUserSettings(logFailure: true);
        var persisted = AppSettingsStore.Load();
        Require(
            persisted.DesktopDockSide == "right"
                && Math.Abs(persisted.DesktopDockWidth - manager.WidthDips) <= 1
                && string.Equals(
                    persisted.DesktopDockMonitor,
                    manager.MonitorDeviceName,
                    StringComparison.OrdinalIgnoreCase),
            "Dock side, width, or monitor did not survive an atomic settings reload.");
        Require(
            persisted.DesktopWindowWidth is > 0
                && persisted.DesktopWindowHeight is > 0,
            "The floating bounds were not retained independently from docking.");

        manager.ReRegisterAfterShellRestartForEndToEndTest();
        await WaitForUiLayoutAsync(180);
        var shellRestart = manager.CaptureSnapshot();
        Require(
            shellRestart.AppBarRegistered
                && shellRestart.WorkArea.X + shellRestart.WorkArea.Width
                    <= shellRestart.WindowBounds.X + 8,
            "Shell re-registration did not restore right-edge reservation.");

        Dictionary<string, object?>? alternateMonitor = null;
        var alternate = monitorBaselines.FirstOrDefault(value =>
            !string.Equals(
                value.DeviceName,
                baseline.MonitorDeviceName,
                StringComparison.OrdinalIgnoreCase));
        if (!string.IsNullOrWhiteSpace(alternate.DeviceName))
        {
            _desktopDockMonitor = alternate.DeviceName;
            Require(
                TryDockDesktop(DesktopDockSide.Left, persist: false),
                "Docking on the alternate monitor failed.");
            await WaitForUiLayoutAsync(220);
            var alternateDock = manager.CaptureSnapshot();
            VerifyDockedFrameState(
                DesktopDockSide.Left,
                alternateDock,
                "alternate-monitor");
            Require(
                string.Equals(
                    alternateDock.MonitorDeviceName,
                    alternate.DeviceName,
                    StringComparison.OrdinalIgnoreCase),
                "The saved monitor identity did not select the alternate display.");
            Require(
                RectangleContains(
                    alternate.Bounds,
                    alternateDock.WindowBounds,
                    8),
                "The alternate-monitor AppBar extended off its target display.");
            alternateMonitor = new Dictionary<string, object?>
            {
                ["device"] = alternateDock.MonitorDeviceName,
                ["dpi"] = alternateDock.Dpi,
                ["window"] = RectToReport(alternateDock.WindowBounds),
                ["work_area"] = RectToReport(alternateDock.WorkArea),
            };
        }

        UnpinDesktop();
        await WaitForUiLayoutAsync(250);
        var restored = manager.CaptureSnapshot();
        Require(!manager.IsDocked, "Unpin left the AppBar registered.");
        Require(DockButton.Content?.ToString() == "Pin", "Unpin did not restore the Pin control.");
        Require(DockResizeHandle.Visibility == Visibility.Collapsed, "Unpin left the resize edge visible.");
        Require(
            DockedFrameBorder.Visibility == Visibility.Collapsed
                && !_desktopWindowFrameDocked
                && ExtendsContentIntoTitleBar
                && AppTitleBar.Visibility == Visibility.Visible
                && TitleBarPaneButton.Visibility == Visibility.Collapsed
                && RootNavigation.IsPaneToggleButtonVisible,
            "Unpin did not restore the floating title-bar and border mode.");
        VerifyFloatingTitleBarState("unpin");
        Require(
            TryReadDwmWindowAttribute(DwmWindowCornerPreference, out var restoredCorners)
                && restoredCorners == DwmCornerDefault
                && _desktopWindowCornerPreference == DwmCornerDefault,
            "Unpin did not restore the default DWM corner policy.");
        // Border color is settable on all supported Windows 11 builds, but
        // some builds reject attribute 34 in DwmGetWindowAttribute. The
        // checked setter is authoritative; a supported readback must agree.
        var restoredBorderReadbackSupported = TryReadDwmWindowAttribute(
            DwmWindowBorderColor,
            out var restoredBorder);
        Require(
            _desktopWindowBorderColor == DwmColorDefault
                && (!restoredBorderReadbackSupported
                    || restoredBorder == DwmColorDefault),
            "Unpin did not restore the default DWM outer-border policy.");
        Require(
            RectanglesApproximatelyEqual(baseline.WorkArea, restored.WorkArea, 8),
            "Unpin did not restore the monitor work area.");
        Require(
            RectanglesApproximatelyEqual(
                baseline.WindowBounds,
                restored.WindowBounds,
                12),
            "Unpin did not restore the prior floating bounds.");

        var presenter = AppWindow.Presenter as OverlappedPresenter
            ?? throw new InvalidOperationException(
                "The desktop presenter was not an overlapped window.");
        TitleBarMaximizeButton_Click(
            TitleBarMaximizeButton,
            new RoutedEventArgs());
        await WaitForUiLayoutAsync(220);
        Require(
            presenter.State == OverlappedPresenterState.Maximized
                && TitleBarMaximizeIcon.Glyph == "\uE923"
                && AutomationProperties.GetName(TitleBarMaximizeButton) == "Restore",
            "The custom maximize control could not enter its maximized state.");
        _desktopDockMonitor = baseline.MonitorDeviceName;
        Require(
            TryDockDesktop(DesktopDockSide.Left, persist: false),
            "Docking from a maximized window failed.");
        await WaitForUiLayoutAsync(220);
        Require(
            presenter.State == OverlappedPresenterState.Restored,
            "Docking did not normalize the maximized presenter.");
        Require(
            manager.RestoreMaximized,
            "Docking did not retain the prior maximized-state flag.");
        UnpinDesktop();
        await WaitForUiLayoutAsync(260);
        Require(
            presenter.State == OverlappedPresenterState.Maximized,
            "Unpin did not restore the prior maximized state "
            + $"(state={presenter.State}, saved={manager.RestoreMaximized}).");
        VerifyFloatingTitleBarState("maximized-unpin");
        TitleBarMaximizeButton_Click(
            TitleBarMaximizeButton,
            new RoutedEventArgs());
        await WaitForUiLayoutAsync(220);
        Require(
            presenter.State == OverlappedPresenterState.Restored
                && TitleBarMaximizeIcon.Glyph == "\uE922"
                && AutomationProperties.GetName(TitleBarMaximizeButton) == "Maximize",
            "The custom restore control did not return to floating state.");
        TitleBarMinimizeButton_Click(
            TitleBarMinimizeButton,
            new RoutedEventArgs());
        await WaitForUiLayoutAsync(180);
        Require(
            presenter.State == OverlappedPresenterState.Minimized,
            "The custom minimize control did not minimize the floating window.");
        presenter.Restore();
        await WaitForUiLayoutAsync(220);
        Require(
            presenter.State == OverlappedPresenterState.Restored,
            "The floating window did not recover after the custom minimize test.");
        VerifyFloatingTitleBarState("minimize-restore");
        var restoredFromMaximized = manager.CaptureSnapshot();
        Require(
            RectanglesApproximatelyEqual(
                baseline.WindowBounds,
                restoredFromMaximized.WindowBounds,
                12),
            "Restoring after maximized docking lost the prior floating bounds.");

        return new Dictionary<string, object?>
        {
            ["baseline_work_area"] = RectToReport(baseline.WorkArea),
            ["left_window"] = RectToReport(left.WindowBounds),
            ["left_work_area"] = RectToReport(left.WorkArea),
            ["right_window_before_resize"] = RectToReport(rightBeforeResize.WindowBounds),
            ["right_window_after_resize"] = RectToReport(rightAfterResize.WindowBounds),
            ["resize_minimum_dips"] = resizeMinimumDips,
            ["resize_maximum_dips"] = resizeMaximumDips,
            ["resize_movement_expected"] = hasResizeRange,
            ["resize_width_delta_pixels"] = resizeWidthDeltaPixels,
            ["right_work_area"] = RectToReport(rightAfterResize.WorkArea),
            ["shell_restart_work_area"] = RectToReport(shellRestart.WorkArea),
            ["restored_work_area"] = RectToReport(restored.WorkArea),
            ["restored_window"] = RectToReport(restored.WindowBounds),
            ["left_surfaces"] = leftSurfaces,
            ["right_surfaces"] = rightSurfaces,
            ["left_dialog"] = leftDialog,
            ["right_dialog"] = rightDialog,
            ["dpi_width_policy"] = dpiPolicy,
            ["monitors"] = monitorBaselines.Select(value =>
                new Dictionary<string, object?>
                {
                    ["device"] = value.DeviceName,
                    ["dpi"] = value.Dpi,
                    ["bounds"] = RectToReport(value.Bounds),
                    ["work_area"] = RectToReport(value.WorkArea),
                }).ToList(),
            ["alternate_monitor"] = alternateMonitor,
            ["maximized_restore"] = true,
            ["minimized_restore"] = true,
        };
    }

    private void VerifyFloatingTitleBarState(string label)
    {
        var presenter = AppWindow.Presenter as OverlappedPresenter
            ?? throw new InvalidOperationException(
                $"{label}: the floating window did not retain an overlapped presenter.");
        Require(
            presenter.HasBorder
                && !presenter.HasTitleBar
                && AppWindow.TitleBar.ExtendsContentIntoTitleBar
                && ExtendsContentIntoTitleBar,
            $"{label}: the floating custom-title-bar frame was not active.");
        Require(
            AppTitleBar.IsLoaded
                && AppTitleBar.XamlRoot is not null
                && AppTitleBar.Visibility == Visibility.Visible
                && AppTitleBar.Opacity > 0
                && AppTitleBar.IsHitTestVisible
                && AppTitleBar.ActualWidth >= 200
                && AppTitleBar.ActualHeight >= 47,
            $"{label}: the floating title bar was not rendered with usable bounds.");
        Require(
            TitleBarPaneButton.Visibility == Visibility.Collapsed
                && RootNavigation.IsPaneToggleButtonVisible
                && TitleBarMinimizeButton.Visibility == Visibility.Visible
                && TitleBarMaximizeButton.Visibility == Visibility.Visible
                && TitleBarCloseButton.Visibility == Visibility.Visible,
            $"{label}: the floating custom chrome showed the wrong controls.");
        VerifyCustomTitleChromeBounds(label, pinned: false);
    }

    private void VerifyCustomTitleChromeBounds(string label, bool pinned)
    {
        var titleRow = WindowRoot.RowDefinitions[0].Height;
        Require(
            HasUsableCustomTitleChromeBounds(),
            $"{label}: the fixed custom title chrome lost its reserved row or "
                + $"drag region (row={titleRow.Value:N1}/{titleRow.GridUnitType}, "
                + $"root={WindowRoot.ActualWidth:N1}x{WindowRoot.ActualHeight:N1}, "
                + $"title={AppTitleBar.ActualWidth:N1}x{AppTitleBar.ActualHeight:N1}, "
                + $"drag={TitleBarDragRegion.ActualWidth:N1}x"
                + $"{TitleBarDragRegion.ActualHeight:N1}, "
                + $"drag-loaded={TitleBarDragRegion.IsLoaded}).");
        var titleBarPoint = AppTitleBar
            .TransformToVisual(WindowRoot)
            .TransformPoint(default);
        var navigationPoint = RootNavigation
            .TransformToVisual(WindowRoot)
            .TransformPoint(default);
        Require(
            titleBarPoint.Y >= -1
                && titleBarPoint.Y + AppTitleBar.ActualHeight
                    <= WindowRoot.ActualHeight + 1
                && navigationPoint.Y
                    >= titleBarPoint.Y + AppTitleBar.ActualHeight - 1,
            $"{label}: custom title chrome was clipped or overlapped by navigation.");

        RequireTitleChromeControlHitTest(DockButton, $"{label}/pin");
        RequireTitleChromeControlHitTest(TitleBarCloseButton, $"{label}/close");
        if (pinned)
        {
            RequireTitleChromeControlHitTest(
                TitleBarPaneButton,
                $"{label}/navigation");
        }
        else
        {
            RequireTitleChromeControlHitTest(
                TitleBarMinimizeButton,
                $"{label}/minimize");
            RequireTitleChromeControlHitTest(
                TitleBarMaximizeButton,
                $"{label}/maximize");
        }
    }

    private void RequireTitleChromeControlHitTest(Control control, string label)
    {
        var controlPoint = control
            .TransformToVisual(WindowRoot)
            .TransformPoint(default);
        var controlCenter = new Windows.Foundation.Point(
            controlPoint.X + control.ActualWidth / 2,
            controlPoint.Y + control.ActualHeight / 2);
        var hitElements = VisualTreeHelper.FindElementsInHostCoordinates(
            controlCenter,
            WindowRoot);
        Require(
            control.Visibility == Visibility.Visible
                && control.IsEnabled
                && control.ActualWidth >= 32
                && control.ActualHeight >= 24
                && hitElements.Any(element =>
                    IsVisualDescendantOrSelf(element, control)),
            $"{label}: the custom title-chrome control was not visible and hittable.");
    }

    private void VerifyDockedFrameState(
        DesktopDockSide expectedSide,
        DesktopDockSnapshot snapshot,
        string label)
    {
        var manager = _desktopDockManager
            ?? throw new InvalidOperationException(
                "The desktop docking manager was unavailable.");
        Require(
            manager.Side == expectedSide,
            $"{label}: the requested dock side was not retained.");
        Require(
            snapshot.AppBarRegistered && !snapshot.InteractiveResize,
            $"{label}: the AppBar was not registered in its steady state.");
        Require(
            RectangleContains(snapshot.MonitorBounds, snapshot.WindowBounds, 8),
            $"{label}: the pinned window extended beyond its monitor.");
        Require(
            snapshot.WindowBounds.Height >= snapshot.WorkArea.Height - 8,
            $"{label}: the pinned window was not full-height within the monitor work area.");
        if (expectedSide == DesktopDockSide.Left)
        {
            Require(
                snapshot.WorkArea.X
                    >= snapshot.WindowBounds.X + snapshot.WindowBounds.Width - 8,
                $"{label}: the left AppBar did not reserve Windows work area.");
        }
        else
        {
            Require(
                snapshot.WorkArea.X + snapshot.WorkArea.Width
                    <= snapshot.WindowBounds.X + 8,
                $"{label}: the right AppBar did not reserve Windows work area.");
        }

        Require(
            _desktopWindowFrameDocked
                && ExtendsContentIntoTitleBar
                && AppTitleBar.Visibility == Visibility.Visible,
            $"{label}: the persistent XAML title bar was not active while pinned.");
        var presenter = AppWindow.Presenter as OverlappedPresenter
            ?? throw new InvalidOperationException(
                $"{label}: the pinned window did not retain an overlapped presenter.");
        Require(
            !presenter.IsResizable
                && !presenter.IsMinimizable
                && !presenter.IsMaximizable,
            $"{label}: OS resize/minimize/maximize commands remained enabled while pinned.");
        Require(
            DockButton.Content?.ToString() == "Unpin",
            $"{label}: the pinned control did not become Unpin.");
        Require(
            DockResizeHandle.Visibility == Visibility.Visible
                && DockedFrameBorder.Visibility == Visibility.Visible,
            $"{label}: the pinned inner edge or border was hidden.");
        Require(
            DockResizeHandle.ActualWidth >= 8,
            $"{label}: the resize pointer hit target was too narrow.");
        Require(
            DockResizeHandle is DockResizeGrip,
            $"{label}: the resize edge did not use the horizontal-resize cursor control.");
        Require(
            RootNavigation.PaneDisplayMode
                == NavigationViewPaneDisplayMode.LeftMinimal,
            $"{label}: pinned navigation did not enter minimal mode.");
        Require(
            TitleBarPaneButton.Visibility == Visibility.Visible
                && !RootNavigation.IsPaneToggleButtonVisible,
            $"{label}: the sidebar toggle was not moved into the pinned title bar.");
        Require(
            TitleBarMinimizeButton.Visibility == Visibility.Collapsed
                && TitleBarMaximizeButton.Visibility == Visibility.Collapsed
                && TitleBarCloseButton.Visibility == Visibility.Visible,
            $"{label}: pinned custom chrome exposed invalid window-state controls.");
        VerifyCustomTitleChromeBounds(label, pinned: true);
        var paneWasOpen = RootNavigation.IsPaneOpen;
        TitleBarPaneButton_Click(TitleBarPaneButton, new RoutedEventArgs());
        Require(
            RootNavigation.IsPaneOpen != paneWasOpen,
            $"{label}: the title-bar sidebar button did not toggle the navigation pane.");
        TitleBarPaneButton_Click(TitleBarPaneButton, new RoutedEventArgs());
        Require(
            RootNavigation.IsPaneOpen == paneWasOpen,
            $"{label}: the title-bar sidebar button did not restore the navigation pane state.");
        var handlePoint = DockResizeHandle
            .TransformToVisual(WindowRoot)
            .TransformPoint(default);
        if (expectedSide == DesktopDockSide.Left)
        {
            Require(
                DockResizeHandle.HorizontalAlignment == HorizontalAlignment.Right
                    && DockedFrameBorder.BorderThickness.Right == 1
                    && DockedFrameBorder.BorderThickness.Left == 0
                    && handlePoint.X + DockResizeHandle.ActualWidth
                        >= WindowRoot.ActualWidth - 2,
                $"{label}: the resize grip and single border were not on the inner right edge.");
        }
        else
        {
            Require(
                DockResizeHandle.HorizontalAlignment == HorizontalAlignment.Left
                    && DockedFrameBorder.BorderThickness.Left == 1
                    && DockedFrameBorder.BorderThickness.Right == 0
                    && handlePoint.X <= 2,
                $"{label}: the resize grip and single border were not on the inner left edge.");
        }

        Require(
            TryReadDwmWindowAttribute(DwmWindowCornerPreference, out var corners)
                && corners == DwmCornerDoNotRound
                && _desktopWindowCornerPreference == DwmCornerDoNotRound,
            $"{label}: DWM did not report square docked corners.");
        var borderReadbackSupported = TryReadDwmWindowAttribute(
            DwmWindowBorderColor,
            out var borderColor);
        Require(
            _desktopWindowBorderColor == DwmColorNone
                && (!borderReadbackSupported || borderColor == DwmColorNone),
            $"{label}: DWM did not accept or consistently report outer-border suppression.");

        ApplyDesktopDockActivationState(active: false);
        var inactiveOpacity = DockedFrameBorder.Opacity;
        ApplyDesktopDockActivationState(active: true);
        Require(
            inactiveOpacity < DockedFrameBorder.Opacity,
            $"{label}: active and inactive dock borders were visually indistinguishable.");
        foreach (var theme in new[] { ElementTheme.Light, ElementTheme.Dark })
        {
            WindowRoot.RequestedTheme = theme;
            WindowRoot.UpdateLayout();
            var borderBrush = DockedFrameBorder.BorderBrush as SolidColorBrush;
            Require(
                borderBrush is not null && borderBrush.Color.A > 0,
                $"{label}: the {theme.ToString().ToLowerInvariant()} dock border did not resolve to a visible brush.");
        }
        WindowRoot.RequestedTheme = ElementTheme.Default;
    }

    private async Task<Dictionary<string, object?>> VerifyDockedSurfaceMatrixAsync(
        string label)
    {
        StatusText.Text = $"Synthetic docked background checkpoint · {label}";
        JobProgress.IsIndeterminate = false;
        JobProgress.Value = label == "left" ? 37 : 63;
        CancelButton.IsEnabled = true;
        var expectedStatus = StatusText.Text;
        var expectedProgress = JobProgress.Value;
        var results = new Dictionary<string, object?>();

        NavigateTo(LibraryNavigationItem);
        await WaitForUiLayoutAsync();
        RequireOnlyPageVisible("library");
        RequireHorizontalBounds(
            $"docked-{label}",
            LibraryPage,
            LibraryHeaderActions,
            LibraryFilterGrid,
            LibraryContentBorder,
            PersistentStatusGrid);
        await RequireKeyboardFocusAsync(
            LibrarySearchBox,
            $"docked-{label}/library-focus");
        results["library_processing"] = await ExerciseScrollAsync(
            LibraryProcessingScroll,
            $"docked-{label}/library-processing");
        results["library_days"] = await ExerciseItemsScrollAsync(
            LibraryList,
            $"docked-{label}/library-days");
        LibraryDetailTabView.SelectedItem = LibraryTranscriptTab;
        await WaitForUiLayoutAsync(40);
        results["library_transcript"] = await ExerciseTextBoxScrollAsync(
            LibraryTranscriptPreviewText,
            $"docked-{label}/library-transcript");
        LibraryDetailTabView.SelectedItem = LibraryProcessingTab;
        LibraryCoverageExpander.IsExpanded = true;
        await WaitForUiLayoutAsync(40);
        results["library_feed_coverage"] = await ExerciseItemsScrollAsync(
            LibraryFeedCoverageList,
            $"docked-{label}/library-feed-coverage");
        LibraryCoverageExpander.IsExpanded = false;
        await WaitForUiLayoutAsync(40);

        NavigateTo(SystemNavigationItem);
        await WaitForUiLayoutAsync();
        RequireOnlyPageVisible("system");
        RequireHorizontalBounds(
            $"docked-{label}",
            SystemPage,
            RefreshSystemActivityButton,
            SystemActivityScroll,
            SystemActivityInfoBar,
            PersistentStatusGrid);
        results["system_activity"] = await ExerciseScrollAsync(
            SystemActivityScroll,
            $"docked-{label}/system-activity");

        NavigateTo(ArchiveNavigationItem);
        await WaitForUiLayoutAsync();
        RequireOnlyPageVisible("archive");
        RequireHorizontalBounds(
            $"docked-{label}",
            ArchivePage,
            ArchiveFeedSearchBorder,
            ArchiveJobBorder,
            PersistentStatusGrid);
        await RequireKeyboardFocusAsync(
            SearchBox,
            $"docked-{label}/archive-focus");
        results["archive_job"] = await ExerciseScrollAsync(
            ArchiveJobScroll,
            $"docked-{label}/archive-job");
        results["archive_feed_results"] = await ExerciseItemsScrollAsync(
            FeedResults,
            $"docked-{label}/archive-feed-results");
        ActivityLogExpander.IsExpanded = true;
        await WaitForUiLayoutAsync(40);
        results["activity_log"] = await ExerciseTextBoxScrollAsync(
            LogBox,
            $"docked-{label}/activity-log");

        NavigateTo(ReviewNavigationItem);
        ReviewTabView.SelectedItem = DailyReviewTab;
        await WaitForUiLayoutAsync();
        RequireOnlyPageVisible("review");
        await RequireKeyboardFocusAsync(
            AnalysisFeedCombo,
            $"docked-{label}/review-focus");
        results["review_days"] = await ExerciseItemsScrollAsync(
            AnalysisDaysList,
            $"docked-{label}/review-days");
        results["review_incidents"] = await ExerciseItemsScrollAsync(
            IncidentList,
            $"docked-{label}/review-incidents");
        ReviewTabView.SelectedItem = WeeklyBriefTab;
        await WaitForUiLayoutAsync(40);
        results["weekly_brief"] = await ExerciseScrollAsync(
            WeeklyBriefScroll,
            $"docked-{label}/weekly-brief");
        ReviewTabView.SelectedItem = ArchiveQuestionTab;
        await WaitForUiLayoutAsync(40);
        RequireHorizontalBounds(
            $"docked-{label}",
            ArchiveQuestionScopePanel,
            QuestionStarterPrimary,
            QuestionStarterSecondary,
            QuestionSendGrid,
            AskButton);
        results["archive_chat"] = await ExerciseScrollAsync(
            ArchiveChatScroll,
            $"docked-{label}/archive-chat");
        results["archive_chat_messages"] = await ExerciseItemsScrollAsync(
            ArchiveChatList,
            $"docked-{label}/archive-chat-messages");

        NavigateTo(AreaNavigationItem);
        AreaTabView.SelectedItem = AreaFeedsTab;
        await WaitForUiLayoutAsync();
        RequireOnlyPageVisible("area");
        await RequireKeyboardFocusAsync(
            AreaProfileNameBox,
            $"docked-{label}/area-focus");
        results["area_feed_results"] = await ExerciseItemsScrollAsync(
            AreaFeedResults,
            $"docked-{label}/area-feed-results");
        AreaTabView.SelectedItem = AreaStoriesTab;
        await WaitForUiLayoutAsync(40);
        RequireHorizontalBounds(
            $"docked-{label}",
            AreaStoryContentGrid,
            AreaStoryListBorder,
            AreaStoryDetailBorder);
        results["area_summary"] = await ExerciseScrollAsync(
            AreaSummaryScroll,
            $"docked-{label}/area-summary");
        results["area_story_detail"] = await ExerciseScrollAsync(
            AreaStoryDetailScroll,
            $"docked-{label}/area-story-detail");
        results["area_stories"] = await ExerciseItemsScrollAsync(
            AreaStoryList,
            $"docked-{label}/area-stories");

        ShowPage("settings");
        SettingsTabView.SelectedItem = SetupSettingsTab;
        await WaitForUiLayoutAsync();
        RequireOnlyPageVisible("settings");
        results["settings_setup"] = await ExerciseScrollAsync(
            SetupSettingsScroll,
            $"docked-{label}/settings-setup");
        SettingsTabView.SelectedItem = ProcessingSettingsTab;
        await WaitForUiLayoutAsync(40);
        await RequireKeyboardFocusAsync(
            HardwareProfileComboBox,
            $"docked-{label}/settings-focus");
        HardwareProfilesExpander.IsExpanded = true;
        Require(
            HardwareProfileList.Items.Count >= 2,
            $"docked-{label}/hardware-profiles: the fixture list was empty.");
        results["settings_processing"] = await ExerciseScrollAsync(
            ProcessingSettingsScroll,
            $"docked-{label}/settings-processing");
        SettingsTabView.SelectedItem = AnalysisSettingsTab;
        await WaitForUiLayoutAsync(40);
        results["settings_analysis"] = await ExerciseScrollAsync(
            AnalysisSettingsScroll,
            $"docked-{label}/settings-analysis");
        SettingsTabView.SelectedItem = AccountSettingsTab;
        await WaitForUiLayoutAsync(40);
        results["settings_credentials"] = await ExerciseScrollAsync(
            CredentialsSettingsScroll,
            $"docked-{label}/settings-credentials");

        NavigateTo(AboutNavigationItem);
        await WaitForUiLayoutAsync();
        RequireOnlyPageVisible("about");
        await RequireKeyboardFocusAsync(
            AboutOpenDataButton,
            $"docked-{label}/about-focus");
        results["about"] = await ExerciseScrollAsync(
            AboutScroll,
            $"docked-{label}/about");

        Require(
            StatusText.Text == expectedStatus
                && Math.Abs(JobProgress.Value - expectedProgress) < 0.01
                && CancelButton.IsEnabled,
            $"docked-{label}: page navigation lost the persistent background task state.");
        return results;
    }

    private async Task<Dictionary<string, object?>> VerifyDockedDialogAndFlyoutAsync(
        string label)
    {
        var flyoutOpened = false;
        DockMenuFlyout.Opened += (_, _) => flyoutOpened = true;
        PrepareDesktopDockMenu();
        DockMenuFlyout.ShowAt(DockButton);
        Require(
            await WaitForUiConditionAsync(() =>
                flyoutOpened
                    && DockUnpinMenuItem.Visibility == Visibility.Visible
                    && DockMenuSeparator.Visibility == Visibility.Visible),
            $"docked-{label}: the side/unpin flyout did not open with all pinned actions.");
        Require(
            AutomationProperties.GetAutomationId(DockLeftMenuItem)
                == "DockLeftMenuItem"
                && AutomationProperties.GetAutomationId(DockRightMenuItem)
                    == "DockRightMenuItem"
                && AutomationProperties.GetAutomationId(DockUnpinMenuItem)
                    == "DockUnpinMenuItem",
            $"docked-{label}: the dock flyout lost its automation identities.");
        var flyoutClosed = false;
        DockMenuFlyout.Closed += (_, _) => flyoutClosed = true;
        DockMenuFlyout.Hide();
        Require(
            await WaitForUiConditionAsync(() => flyoutClosed),
            $"docked-{label}: the side/unpin flyout did not close before the dialog opened.");

        var dialogContent = CreateDialogTextContent(
            string.Join(
                Environment.NewLine,
                Enumerable.Range(1, 80).Select(index =>
                    $"Synthetic safe dialog line {index}: retained work is unchanged.")),
            420);
        var dialog = new ContentDialog
        {
            XamlRoot = ((FrameworkElement)Content).XamlRoot,
            Title = $"Synthetic docked dialog · {label}",
            Content = dialogContent,
            PrimaryButtonText = "Continue",
            CloseButtonText = "Cancel",
            DefaultButton = ContentDialogButton.Close,
        };
        var operation = dialog.ShowAsync();
        var dialogContained = await WaitForDialogContainmentAsync(dialog);
        Require(
            dialogContained,
            $"docked-{label}: the dialog did not settle inside the pinned window "
                + $"(dialog {dialog.ActualWidth:F1}x{dialog.ActualHeight:F1}, "
                + $"window {WindowRoot.ActualWidth:F1}x{WindowRoot.ActualHeight:F1}).");
        var scrollable = await ExerciseScrollAsync(
            dialogContent,
            $"docked-{label}/dialog");
        var buttons = VisualTreeHelper
            .GetOpenPopupsForXamlRoot(dialog.XamlRoot)
            .Where(popup => popup.Child is not null)
            .SelectMany(popup => FindVisualDescendants<Button>(popup.Child!))
            .Where(button => button.Content?.ToString() is "Continue" or "Cancel")
            .ToList();
        Require(
            buttons.Count == 2
                && buttons.All(button =>
                    button.ActualWidth > 0 && button.ActualHeight > 0),
            $"docked-{label}: the dialog actions were not rendered and reachable.");
        RequireHorizontalBounds($"docked-{label}/dialog-actions", [.. buttons]);
        var dialogWidth = Math.Round(dialog.ActualWidth, 1);
        var dialogHeight = Math.Round(dialog.ActualHeight, 1);
        dialog.Hide();
        await operation;
        return new Dictionary<string, object?>
        {
            ["flyout_opened"] = flyoutOpened,
            ["dialog_width"] = dialogWidth,
            ["dialog_height"] = dialogHeight,
            ["dialog_scrollable_height"] = scrollable,
            ["dialog_buttons"] = buttons.Count,
        };
    }

    private async Task<bool> WaitForDialogContainmentAsync(ContentDialog dialog)
    {
        return await WaitForUiConditionAsync(() =>
        {
            dialog.UpdateLayout();
            return dialog.ActualWidth > 0
                && dialog.ActualWidth <= WindowRoot.ActualWidth + 2
                && dialog.ActualHeight <= WindowRoot.ActualHeight + 2;
        });
    }

    private async Task<bool> WaitForUiConditionAsync(Func<bool> condition)
    {
        const int attempts = 30;
        for (var attempt = 0; attempt < attempts; attempt++)
        {
            WindowRoot.UpdateLayout();
            if (condition())
            {
                return true;
            }

            await Task.Delay(10);
        }

        WindowRoot.UpdateLayout();
        return condition();
    }

    private static List<Dictionary<string, object>> VerifyDpiAndWidthPolicy()
    {
        var results = new List<Dictionary<string, object>>();
        foreach (var dpi in new uint[] { 96, 120, 144, 192 })
        {
            foreach (var width in new[]
                     {
                         DesktopDockPolicy.MinimumWidthDips,
                         DesktopDockPolicy.RecommendedWidthDips,
                         DesktopDockPolicy.MaximumWidthDips,
                     })
            {
                var pixels = DesktopDockPolicy.DipsToPixels(width, dpi);
                var roundTrip = DesktopDockPolicy.PixelsToDips(pixels, dpi);
                Require(
                    Math.Abs(roundTrip - width) <= 1,
                    $"Dock width lost logical size at {dpi} DPI.");
                results.Add(new Dictionary<string, object>
                {
                    ["dpi"] = dpi,
                    ["width_dips"] = width,
                    ["width_pixels"] = pixels,
                    ["round_trip_dips"] = roundTrip,
                });
            }
        }
        Require(
            DesktopDockPolicy.NormalizeWidthDips(double.NaN, 1_920)
                == DesktopDockPolicy.RecommendedWidthDips,
            "Invalid dock widths did not return to the recommended width.");
        Require(
            DesktopDockPolicy.NormalizeWidthDips(20_000, 1_200) <= 600,
            "Dock width was not constrained to half of a narrow display.");
        return results;
    }

    private async Task RequireKeyboardFocusAsync(
        Control control,
        string label)
    {
        Require(
            control.IsEnabled
                && control.Visibility == Visibility.Visible,
            $"{label}: the requested focus target was disabled or hidden.");
        control.StartBringIntoView(new BringIntoViewOptions
        {
            AnimationDesired = false,
        });
        await WaitForUiLayoutAsync(60);
        var windowHandle = WinRT.Interop.WindowNative.GetWindowHandle(this);
        Activate();
        var foregroundAccepted = SetForegroundWindow(windowHandle);
        SetActiveWindow(windowHandle);
        SetFocus(windowHandle);
        await WaitForUiLayoutAsync(35);
        SetFocus(windowHandle);
        var focusAccepted = control.Focus(FocusState.Keyboard)
            || control.Focus(FocusState.Programmatic);
        Require(
            focusAccepted,
            $"{label}: the primary control could not receive keyboard focus.");
        await WaitForUiLayoutAsync(20);
        var focused = FocusManager.GetFocusedElement(control.XamlRoot);
        Require(
            IsVisualDescendantOrSelf(focused as DependencyObject, control),
            $"{label}: keyboard focus was not retained on the requested control "
            + $"(state={control.FocusState}, focused={focused?.GetType().Name ?? "none"}, "
            + $"foreground={foregroundAccepted}).");
    }

    private static bool IsVisualDescendantOrSelf(
        DependencyObject? candidate,
        DependencyObject ancestor)
    {
        for (var current = candidate; current is not null;)
        {
            if (ReferenceEquals(current, ancestor))
            {
                return true;
            }
            current = VisualTreeHelper.GetParent(current);
        }
        return false;
    }

    private static IEnumerable<T> FindVisualDescendants<T>(DependencyObject root)
        where T : DependencyObject
    {
        for (var index = 0; index < VisualTreeHelper.GetChildrenCount(root); index++)
        {
            var child = VisualTreeHelper.GetChild(root, index);
            if (child is T match)
            {
                yield return match;
            }
            foreach (var descendant in FindVisualDescendants<T>(child))
            {
                yield return descendant;
            }
        }
    }

    private static bool RectangleContains(
        RectInt32 outer,
        RectInt32 inner,
        int tolerance) =>
        inner.X >= outer.X - tolerance
        && inner.Y >= outer.Y - tolerance
        && inner.X + inner.Width <= outer.X + outer.Width + tolerance
        && inner.Y + inner.Height <= outer.Y + outer.Height + tolerance;

    private async Task<double> ExerciseScrollAsync(
        ScrollViewer scrollViewer,
        string label)
    {
        scrollViewer.UpdateLayout();
        var scrollableHeight = scrollViewer.ScrollableHeight;
        Require(
            scrollViewer.ScrollableWidth <= 2,
            $"{label}: content introduced unintended horizontal scrolling.");
        Require(
            scrollableHeight > 1,
            $"{label}: content did not expose a vertical scroll range.");
        scrollViewer.ChangeView(null, scrollableHeight, null, true);
        Require(
            await WaitForScrollOffsetAsync(
                scrollViewer,
                offset => offset > Math.Min(1, scrollableHeight / 2)),
            $"{label}: scrolling to the end did not change the vertical offset.");
        scrollViewer.ChangeView(null, 0, null, true);
        Require(
            await WaitForScrollOffsetAsync(scrollViewer, offset => offset <= 1),
            $"{label}: scrolling back to the start did not restore the top.");
        return Math.Round(scrollableHeight, 1);
    }

    private static async Task<bool> WaitForScrollOffsetAsync(
        ScrollViewer scrollViewer,
        Func<double, bool> condition)
    {
        const int attempts = 20;
        for (var attempt = 0; attempt < attempts; attempt++)
        {
            scrollViewer.UpdateLayout();
            if (condition(scrollViewer.VerticalOffset))
            {
                return true;
            }

            await Task.Delay(10);
        }

        scrollViewer.UpdateLayout();
        return condition(scrollViewer.VerticalOffset);
    }

    private async Task<int> ExerciseItemsScrollAsync(
        ListView listView,
        string label)
    {
        Require(
            listView.Items.Count >= 2,
            $"{label}: the fixture did not provide enough items to scroll.");
        listView.UpdateLayout();
        var scrollViewer = FindVisualDescendant<ScrollViewer>(listView)
            ?? throw new InvalidOperationException(
                $"{label}: the rendered list had no scroll host.");
        Require(
            scrollViewer.ScrollableHeight > 1,
            $"{label}: the rendered list did not expose a vertical scroll range.");
        Require(
            scrollViewer.ScrollableWidth <= 2,
            $"{label}: the rendered list introduced unintended horizontal scrolling.");
        listView.ScrollIntoView(listView.Items[listView.Items.Count - 1]);
        await WaitForUiLayoutAsync(50);
        Require(
            scrollViewer.VerticalOffset > 1,
            $"{label}: scrolling to the final item did not move the list.");
        listView.ScrollIntoView(listView.Items[0]);
        await WaitForUiLayoutAsync(50);
        Require(
            scrollViewer.VerticalOffset <= 1,
            $"{label}: scrolling back to the first item did not restore the top.");
        return listView.Items.Count;
    }

    private async Task<int> ExerciseTextBoxScrollAsync(
        TextBox textBox,
        string label)
    {
        textBox.UpdateLayout();
        var scrollViewer = FindVisualDescendant<ScrollViewer>(textBox)
            ?? throw new InvalidOperationException(
                $"{label}: the rendered text box had no scroll host.");
        Require(
            scrollViewer.ScrollableHeight > 1,
            $"{label}: the text fixture did not expose a vertical scroll range.");
        Require(
            scrollViewer.ScrollableWidth <= 2,
            $"{label}: the text viewport introduced unintended horizontal scrolling.");
        scrollViewer.ChangeView(null, scrollViewer.ScrollableHeight, null, true);
        await WaitForUiLayoutAsync(40);
        Require(
            scrollViewer.VerticalOffset > 1,
            $"{label}: scrolling to the transcript end did not move the viewport.");
        scrollViewer.ChangeView(null, 0, null, true);
        await WaitForUiLayoutAsync(40);
        Require(
            scrollViewer.VerticalOffset <= 1,
            $"{label}: scrolling to the transcript start did not restore the top.");
        return textBox.Text.Count(character => character == '\n') + 1;
    }

    private static T? FindVisualDescendant<T>(DependencyObject root)
        where T : DependencyObject
    {
        for (var index = 0; index < VisualTreeHelper.GetChildrenCount(root); index++)
        {
            var child = VisualTreeHelper.GetChild(root, index);
            if (child is T match)
            {
                return match;
            }
            var descendant = FindVisualDescendant<T>(child);
            if (descendant is not null)
            {
                return descendant;
            }
        }
        return null;
    }

    private void RequireHorizontalBounds(
        string sizeName,
        params FrameworkElement[] elements)
    {
        foreach (var element in elements)
        {
            Require(
                element.Visibility == Visibility.Visible,
                $"{sizeName}/{element.Name}: expected the control to be visible.");
            element.UpdateLayout();
            Require(
                element.ActualWidth > 0 && element.ActualHeight > 0,
                $"{sizeName}/{element.Name}: the control had no rendered bounds.");
            var point = element.TransformToVisual(WindowRoot).TransformPoint(default);
            Require(
                point.X >= -2
                    && point.X + element.ActualWidth <= WindowRoot.ActualWidth + 2,
                $"{sizeName}/{element.Name}: horizontal bounds "
                + $"{point.X:N1}..{point.X + element.ActualWidth:N1} exceeded "
                + $"the {WindowRoot.ActualWidth:N1}-DIP window.");
        }
    }

    private void RequireOnlyPageVisible(string expected)
    {
        var pages = new Dictionary<string, FrameworkElement>
        {
            ["library"] = LibraryPage,
            ["system"] = SystemPage,
            ["archive"] = ArchivePage,
            ["review"] = ReviewPage,
            ["area"] = AreaPage,
            ["settings"] = SettingsPage,
            ["about"] = AboutPage,
        };
        var visible = pages
            .Where(value => value.Value.Visibility == Visibility.Visible)
            .Select(value => value.Key)
            .ToArray();
        Require(
            visible.SequenceEqual([expected]),
            $"Navigation expected only {expected}, but visible pages were {string.Join(", ", visible)}.");
    }

    private async Task WaitForUiLayoutAsync(int milliseconds = 110)
    {
        await Task.Delay(milliseconds);
        WindowRoot.UpdateLayout();
        await Task.Yield();
    }

    private bool HasUsableCustomTitleChromeBounds()
    {
        var titleRow = WindowRoot.RowDefinitions[0].Height;
        return titleRow.GridUnitType == GridUnitType.Pixel
            && Math.Abs(titleRow.Value - 48) < 0.01
            && AppTitleBar.ActualHeight >= 47
            && TitleBarDragRegion.IsLoaded
            && TitleBarDragRegion.ActualWidth >= 80
            // WinUI can report a requested 47-DIP child as 46.67 DIPs after
            // physical-pixel rounding on a scaled display.
            && TitleBarDragRegion.ActualHeight >= 46.5
            && AppTitleText.Text == "Broadcastify Desktop";
    }

    private async Task WaitForStableCustomTitleChromeAsync()
    {
        var deadline = DateTimeOffset.UtcNow.AddSeconds(3);
        var stableLayoutPasses = 0;
        while (DateTimeOffset.UtcNow < deadline)
        {
            WindowRoot.UpdateLayout();
            if (HasUsableCustomTitleChromeBounds())
            {
                stableLayoutPasses += 1;
                if (stableLayoutPasses >= 2)
                {
                    return;
                }
            }
            else
            {
                stableLayoutPasses = 0;
            }
            await Task.Delay(35);
            await Task.Yield();
        }
    }

    private static bool RectanglesApproximatelyEqual(
        RectInt32 left,
        RectInt32 right,
        int tolerance) =>
        Math.Abs(left.X - right.X) <= tolerance
        && Math.Abs(left.Y - right.Y) <= tolerance
        && Math.Abs(left.Width - right.Width) <= tolerance
        && Math.Abs(left.Height - right.Height) <= tolerance;

    private static Dictionary<string, int> RectToReport(RectInt32 value) =>
        new()
        {
            ["x"] = value.X,
            ["y"] = value.Y,
            ["width"] = value.Width,
            ["height"] = value.Height,
        };

    private static void Require(bool condition, string message)
    {
        if (!condition)
        {
            throw new InvalidOperationException(message);
        }
    }

    private static async Task WriteUiProbeReportAsync(
        string reportPath,
        Dictionary<string, object?> report)
    {
        var fullPath = Path.GetFullPath(reportPath);
        Directory.CreateDirectory(Path.GetDirectoryName(fullPath)!);
        var temporaryPath = fullPath + ".tmp";
        var json = JsonSerializer.Serialize(
            report,
            new JsonSerializerOptions { WriteIndented = true });
        await File.WriteAllTextAsync(
            temporaryPath,
            json,
            new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
        File.Move(temporaryPath, fullPath, overwrite: true);
    }

    [DllImport("user32.dll")]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool SetForegroundWindow(nint windowHandle);

    [DllImport("user32.dll")]
    private static extern nint SetActiveWindow(nint windowHandle);

    [DllImport("user32.dll")]
    private static extern nint SetFocus(nint windowHandle);
}
