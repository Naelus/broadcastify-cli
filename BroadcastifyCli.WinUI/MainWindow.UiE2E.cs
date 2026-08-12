using System.Text;
using System.Text.Json;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
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
        var passed = false;
        try
        {
            ConfigureUiEndToEndFixture();
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
            Close();
        }
    }

    private void ConfigureUiEndToEndFixture()
    {
        const string repeatedStatus =
            "Retained local work is safe. Download, combine, transcription, "
            + "speaker labeling, and analysis checkpoints remain independently resumable. ";
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

        LogBox.Text = string.Join(
            Environment.NewLine,
            Enumerable.Range(1, 60).Select(index => $"Isolated activity line {index}"));
        RefreshAboutPage();
    }

    private async Task<List<Dictionary<string, object?>>> RunUiLayoutMatrixAsync()
    {
        var sizes = new[]
        {
            new UiProbeSize("pinned-width", 600, 900, true),
            new UiProbeSize("compact-short", 720, 720, true),
            new UiProbeSize("medium", 960, 720, false),
            new UiProbeSize("reference", 1240, 900, false),
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
        var dpiScale = Math.Max(
            1.0,
            GetDpiForWindow(WinRT.Interop.WindowNative.GetWindowHandle(this)) / 96.0);
        AppWindow.Resize(new SizeInt32(
            (int)Math.Round(size.WidthDips * dpiScale),
            (int)Math.Round(size.HeightDips * dpiScale)));
        await WaitForUiLayoutAsync();
        Require(
            Math.Abs(WindowRoot.ActualWidth - size.WidthDips) <= 80,
            $"{size.Name}: requested width {size.WidthDips:N0} DIP but rendered {WindowRoot.ActualWidth:N1} DIP.");
        Require(
            _compactLayoutApplied == size.Compact,
            $"{size.Name}: compact layout state did not match the width contract.");

        var scrollResults = new Dictionary<string, double>();

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
        if (size.Compact && size.HeightDips < 820)
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

        NavigateTo(ReviewNavigationItem);
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
            ["rendered_width_dips"] = Math.Round(WindowRoot.ActualWidth, 1),
            ["rendered_height_dips"] = Math.Round(WindowRoot.ActualHeight, 1),
            ["compact"] = _compactLayoutApplied,
            ["scrollable_heights"] = scrollResults,
        };
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

        Require(
            TryDockDesktop(DesktopDockSide.Left, persist: false),
            "Pin-left failed.");
        await WaitForUiLayoutAsync(250);
        var left = manager.CaptureSnapshot();
        Require(manager.Side == DesktopDockSide.Left, "Pin-left did not retain its side.");
        Require(DockButton.Content?.ToString() == "Unpin", "Pinned control did not become Unpin.");
        Require(DockResizeHandle.Visibility == Visibility.Visible, "Pinned resize edge was hidden.");
        Require(
            RootNavigation.PaneDisplayMode == NavigationViewPaneDisplayMode.LeftMinimal,
            "Pinned navigation did not enter minimal mode.");
        Require(
            left.WorkArea.X >= left.WindowBounds.X + left.WindowBounds.Width - 8,
            "The left AppBar did not reserve Windows work area.");

        Require(
            TryDockDesktop(DesktopDockSide.Right, persist: false),
            "Pin-right failed.");
        await WaitForUiLayoutAsync(250);
        var rightBeforeResize = manager.CaptureSnapshot();
        Require(manager.Side == DesktopDockSide.Right, "Pin-right did not retain its side.");
        Require(
            rightBeforeResize.WorkArea.X + rightBeforeResize.WorkArea.Width
                <= rightBeforeResize.WindowBounds.X + 8,
            "The right AppBar did not reserve Windows work area.");

        var previousWidth = rightBeforeResize.WindowBounds.Width;
        manager.ChangeWidth(manager.WidthDips + 40);
        _desktopDockWidth = manager.WidthDips;
        await WaitForUiLayoutAsync(180);
        var rightAfterResize = manager.CaptureSnapshot();
        Require(
            rightAfterResize.WindowBounds.Width >= previousWidth + 20,
            "Dragging-equivalent dock resizing did not increase the reserved width.");

        UnpinDesktop();
        await WaitForUiLayoutAsync(250);
        var restored = manager.CaptureSnapshot();
        Require(!manager.IsDocked, "Unpin left the AppBar registered.");
        Require(DockButton.Content?.ToString() == "Pin", "Unpin did not restore the Pin control.");
        Require(DockResizeHandle.Visibility == Visibility.Collapsed, "Unpin left the resize edge visible.");
        Require(
            RectanglesApproximatelyEqual(baseline.WorkArea, restored.WorkArea, 8),
            "Unpin did not restore the monitor work area.");

        return new Dictionary<string, object?>
        {
            ["baseline_work_area"] = RectToReport(baseline.WorkArea),
            ["left_window"] = RectToReport(left.WindowBounds),
            ["left_work_area"] = RectToReport(left.WorkArea),
            ["right_window_before_resize"] = RectToReport(rightBeforeResize.WindowBounds),
            ["right_window_after_resize"] = RectToReport(rightAfterResize.WindowBounds),
            ["right_work_area"] = RectToReport(rightAfterResize.WorkArea),
            ["restored_work_area"] = RectToReport(restored.WorkArea),
        };
    }

    private async Task<double> ExerciseScrollAsync(
        ScrollViewer scrollViewer,
        string label)
    {
        scrollViewer.UpdateLayout();
        var scrollableHeight = scrollViewer.ScrollableHeight;
        Require(
            scrollableHeight > 1,
            $"{label}: content did not expose a vertical scroll range.");
        scrollViewer.ChangeView(null, scrollableHeight, null, true);
        await WaitForUiLayoutAsync(35);
        Require(
            scrollViewer.VerticalOffset > Math.Min(1, scrollableHeight / 2),
            $"{label}: scrolling to the end did not change the vertical offset.");
        scrollViewer.ChangeView(null, 0, null, true);
        await WaitForUiLayoutAsync(35);
        Require(
            scrollViewer.VerticalOffset <= 1,
            $"{label}: scrolling back to the start did not restore the top.");
        return Math.Round(scrollableHeight, 1);
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
}
