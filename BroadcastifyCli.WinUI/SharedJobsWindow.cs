using System.Text.Json;
using System.Globalization;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;

namespace BroadcastifyCli.WinUI;

public sealed partial class MainWindow
{
    private JsonElement _sharedJobs;
    private JsonElement _sharedWorkflowEta;
    private StackPanel? _sharedJobsPanel;
    private bool _refreshingSharedJobs;
    private bool _showingSharedJob;

    private static string SharedText(JsonElement row, string name) =>
        row.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.String
            ? value.GetString() ?? "" : "";

    private static int SharedCount(JsonElement row, string name) =>
        row.TryGetProperty(name, out var value) && value.TryGetInt32(out var count)
            ? count : 0;

    private async Task<bool> SharedJobsOwnProcessingAsync()
    {
        await RefreshSharedJobsAsync();
        return _sharedJobs.ValueKind == JsonValueKind.Array && _sharedJobs.EnumerateArray().Any(row =>
            SharedText(row, "status") == "paused" ||
            (SharedText(row, "owner") == "master" &&
             SharedText(row, "status") is "queued" or "running" or "pausing" or "canceling"));
    }

    private async Task RefreshSharedJobsAsync()
    {
        if (_worker is null || _refreshingSharedJobs) return;
        if (WorkerClient.IsPreview)
        {
            FullWorkflowEtaText.Text = "Full workflow (all retained work): Preview has no live estimate.";
            return;
        }
        _refreshingSharedJobs = true;
        try
        {
            var value = await _worker.SharedJobsRequestAsync("");
            _sharedJobs = value.GetProperty("jobs").Clone();
            _sharedWorkflowEta = value.TryGetProperty("workflow_eta", out var workflowEta)
                ? workflowEta.Clone() : default;
            var rows = _sharedJobs.EnumerateArray().ToList();
            var active = rows.FirstOrDefault(r => SharedText(r, "status") is "running" or "pausing" or "canceling");
            FullWorkflowEtaText.Text = _sharedWorkflowEta.ValueKind == JsonValueKind.Object
                ? SharedWorkflowEtaDescription(_sharedWorkflowEta)
                : "Full workflow (all retained work): Whole-backlog estimate unavailable.";
            if (_pipelineCancellation is null && active.ValueKind == JsonValueKind.Object)
            {
                _showingSharedJob = true;
                StatusText.Text = SharedJobDescription(active);
                JobTimingText.Text = "Shared Windows job — the same record and controls are available on the NAS.";
            }
            else if (_pipelineCancellation is null && _showingSharedJob)
            {
                _showingSharedJob = false;
                StatusText.Text = rows.Any(r => SharedText(r,"status") == "paused")
                    ? "Model work paused; saved work remains resumable. Downloads may continue."
                    : "No shared job is running. Open Jobs for queued work and completed history.";
                JobTimingText.Text = "";
            }
            SharedJobsButton.Content = $"Jobs ({rows.Count(r => SharedText(r, "status") is "queued" or "running" or "paused" or "waiting_quota")})";
            RenderSharedJobs();
        }
        catch (Exception exception) when (exception is HttpRequestException or TaskCanceledException or InvalidOperationException)
        {
            SharedJobsButton.Content = "Jobs";
            FullWorkflowEtaText.Text = "Full workflow (all retained work): Master disconnected; the live estimate is hidden. "
                + "Reconnect to refresh it.";
            if (_sharedJobsPanel is not null)
            {
                _sharedJobsPanel.Children.Clear();
                _sharedJobsPanel.Children.Add(new TextBlock { Text = $"Master jobs unavailable: {exception.Message}", TextWrapping = TextWrapping.Wrap });
            }
        }
        finally { _refreshingSharedJobs = false; }
    }

    private static string SharedJobDescription(JsonElement row)
    {
        var text = $"{SharedText(row, "status")} · {SharedText(row, "mode")} · feed {SharedText(row, "feed_id")} · {SharedText(row, "id")}";
        if (row.TryGetProperty("scope", out var scope))
            text += $" · {SharedText(scope, "start_date")} through {SharedText(scope, "end_date")}";
        if (row.TryGetProperty("scope_progress", out var progress)
            && progress.ValueKind == JsonValueKind.Object)
        {
            text += $"\nScope: {SharedCount(progress, "ready_days")}/{SharedCount(progress, "total_days")} ready"
                + $" · {SharedCount(progress, "source_days")} source"
                + $" · {SharedCount(progress, "processing_days")} processing"
                + $" · {SharedCount(progress, "unavailable_days")} unavailable"
                + $" · {SharedCount(progress, "awaiting_source_days")} awaiting source";
        }
        if (row.TryGetProperty("events", out var events) && events.ValueKind == JsonValueKind.Array)
        {
            var lastProgress = events.EnumerateArray().LastOrDefault(e =>
                SharedText(e, "type") is "progress" or "stage");
            if (lastProgress.ValueKind == JsonValueKind.Object) text += $"\n{SharedText(lastProgress, "message")}";
        }
        if (!string.IsNullOrWhiteSpace(SharedText(row, "error"))) text += $"\n{SharedText(row, "error")}";
        if (row.TryGetProperty("eta", out var eta) && eta.ValueKind == JsonValueKind.Object)
            text += $"\n{SharedJobEtaDescription(row)}";
        return text;
    }

    private static string SharedJobEtaDescription(JsonElement row)
    {
        if (!row.TryGetProperty("eta", out var eta) || eta.ValueKind != JsonValueKind.Object)
            return "Full workflow (requested scope): Estimate unavailable.";
        return FormatEtaDescription(eta, "Full workflow (requested scope)");
    }

    private static string SharedWorkflowEtaDescription(JsonElement eta)
    {
        var prefix = "Full workflow (all retained work)";
        if (eta.TryGetProperty("pending_days", out var pendingDays)
            && pendingDays.TryGetInt32(out var count))
            prefix += $" · {count:N0} feed-days unfinished";
        return FormatEtaDescription(eta, prefix);
    }

    private static string FormatEtaDescription(JsonElement eta, string prefix)
    {
        if (eta.ValueKind != JsonValueKind.Object)
            return $"{prefix}: Estimate unavailable.";
        var state = SharedText(eta, "state");
        var message = SharedText(eta, "message");
        if (DateTimeOffset.TryParse(SharedText(eta, "next_start_at"), out var scheduled))
            message += $" · scheduled {scheduled.ToLocalTime():g} local";
        if (state is "learning" or "waiting_source" or "scheduled" or "paused"
            && TryEtaSeconds(eta, "lower_seconds", out var measuredLower)
            && TryEtaSeconds(eta, "upper_seconds", out var measuredUpper) && measuredUpper > 0)
            message += $" · calibrated stages need {FormatEtaDuration(measuredLower)}–{FormatEtaDuration(measuredUpper)}; other work/waits excluded";
        if (state != "estimated")
            return string.IsNullOrWhiteSpace(message)
                ? $"{prefix}: {string.IsNullOrWhiteSpace(state) ? "Estimate unavailable" : state}."
                : $"{prefix}: {message}.";

        if (!DateTimeOffset.TryParse(SharedText(eta, "as_of"), CultureInfo.InvariantCulture,
                DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal, out var asOf)
            || Math.Abs((DateTimeOffset.UtcNow - asOf).TotalSeconds) > 120
            || !DateTimeOffset.TryParse(SharedText(eta, "finish_earliest"), CultureInfo.InvariantCulture,
                DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal, out var earliest)
            || !DateTimeOffset.TryParse(SharedText(eta, "finish_latest"), CultureInfo.InvariantCulture,
                DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal, out var latest)
            || latest < DateTimeOffset.UtcNow)
            return $"{prefix}: Estimate is stale; reconnect or refresh Jobs to update it.";

        if (!TryEtaSeconds(eta, "lower_seconds", out var lower)
            || !TryEtaSeconds(eta, "upper_seconds", out var upper))
            return $"{prefix}: Estimate is unavailable until remaining work is measured.";

        var text = $"{prefix}: estimated finish {earliest.ToLocalTime():g}–{latest.ToLocalTime():g} local"
            + $" · remaining {FormatEtaDuration(lower)}–{FormatEtaDuration(upper)}";
        if (TryEtaSeconds(eta, "queue_lower_seconds", out var queueLower)
            && TryEtaSeconds(eta, "queue_upper_seconds", out var queueUpper)
            && (queueLower > 0 || queueUpper > 0))
            text += $" · queue delay {FormatEtaDuration(queueLower)}–{FormatEtaDuration(queueUpper)}";
        if (eta.TryGetProperty("notes", out var notes) && notes.ValueKind == JsonValueKind.Array)
        {
            var noteText = string.Join("; ", notes.EnumerateArray()
                .Where(note => note.ValueKind == JsonValueKind.String)
                .Select(note => note.GetString())
                .Where(note => !string.IsNullOrWhiteSpace(note)));
            if (!string.IsNullOrWhiteSpace(noteText)) text += $" · provisional: {noteText}";
            else text += " · estimate is provisional";
        }
        else text += " · estimate is provisional";
        return text;
    }

    private static bool TryEtaSeconds(JsonElement eta, string name, out double seconds)
    {
        seconds = 0;
        return eta.TryGetProperty(name, out var value)
            && value.ValueKind == JsonValueKind.Number
            && value.TryGetDouble(out seconds)
            && double.IsFinite(seconds)
            && seconds >= 0;
    }

    private static string FormatEtaDuration(double seconds)
    {
        var duration = TimeSpan.FromSeconds(Math.Min(seconds, TimeSpan.MaxValue.TotalSeconds));
        if (duration.TotalDays >= 1)
            return $"{(int)duration.TotalDays}d {duration.Hours}h";
        if (duration.TotalHours >= 1)
            return $"{(int)duration.TotalHours}h {duration.Minutes}m";
        if (duration.TotalMinutes >= 1)
            return $"{(int)duration.TotalMinutes}m";
        return $"{Math.Max(0, (int)duration.TotalSeconds)}s";
    }

    private void RenderSharedJobs()
    {
        if (_sharedJobsPanel is null || _sharedJobs.ValueKind != JsonValueKind.Array) return;
        _sharedJobsPanel.Children.Clear();
        foreach (var row in _sharedJobs.EnumerateArray())
        {
            var jobId = SharedText(row, "id");
            var status = SharedText(row, "status");
            var group = new StackPanel { Spacing = 6, Margin = new Thickness(0, 4, 0, 12) };
            group.Children.Add(new TextBlock { Text = SharedJobDescription(row), TextWrapping = TextWrapping.Wrap });
            group.Children.Add(new TextBlock { Text = $"Last report: {SharedText(row, "updated_at")}", FontSize = 12 });
            var controls = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 8 };
            var actions = status is "paused" or "waiting_quota" or "failed" or "canceled"
                ? new[] { "resume", "cancel" } : status == "queued" ? new[] { "next", "pause", "cancel" }
                : status == "running" ? new[] { "pause", "cancel" } : Array.Empty<string>();
            foreach (var action in actions)
            {
                var button = new Button { Content = action == "next" ? "Run next" : action };
                button.Click += async (_, _) =>
                {
                    button.IsEnabled = false;
                    try { await _worker!.SharedJobsRequestAsync($"/{jobId}/{action}", new { }); await RefreshSharedJobsAsync(); }
                    catch (Exception exception)
                    {
                        group.Children.Add(new TextBlock { Text = exception.Message, TextWrapping = TextWrapping.Wrap });
                        button.IsEnabled = true;
                    }
                };
                controls.Children.Add(button);
            }
            group.Children.Add(controls);
            _sharedJobsPanel.Children.Add(group);
        }
        if (_sharedJobsPanel.Children.Count == 0)
            _sharedJobsPanel.Children.Add(new TextBlock { Text = "No shared jobs yet. New desktop and NAS jobs appear here." });
    }

    private async void SharedJobs_Click(object sender, RoutedEventArgs e)
    {
        _sharedJobsPanel = new StackPanel { Spacing = 8, MinWidth = 300, MaxWidth = 700 };
        var dialog = new ContentDialog
        {
            XamlRoot = ((FrameworkElement)Content).XamlRoot,
            Title = "Windows master jobs",
            Content = CreateDialogScrollContent(_sharedJobsPanel, 580),
            PrimaryButtonText = "New catch-up…", CloseButtonText = "Close",
        };
        await RefreshSharedJobsAsync();
        ContentDialogResult result;
        try { result = await dialog.ShowAsync(); }
        finally { _sharedJobsPanel = null; }
        if (result == ContentDialogResult.Primary) await QueueSharedJobAsync();
    }

    private async Task QueueSharedJobAsync()
    {
        if (_worker is null) return;
        try
        {
            var schedules = await _worker.ListFeedSchedulesAsync(CancellationToken.None);
            var feeds = new ComboBox { Header = "Feed", ItemsSource = schedules, DisplayMemberPath = "FeedName", SelectedIndex = 0 };
            var mode = new ComboBox { Header = "Work", ItemsSource = new[] { "Full workflow", "Download audio only" }, SelectedIndex = 0 };
            var start = new DatePicker { Header = "Start date", Date = DateTimeOffset.Now };
            var end = new DatePicker { Header = "End date", Date = DateTimeOffset.Now };
            var content = new StackPanel { Spacing = 10 };
            content.Children.Add(new TextBlock { Text = "Target: Windows master. Today is selected initially. This uses the feed's saved profile, reuses completed work, and queues behind active jobs.", TextWrapping = TextWrapping.Wrap });
            content.Children.Add(feeds); content.Children.Add(mode); content.Children.Add(start); content.Children.Add(end);
            var dialog = new ContentDialog { XamlRoot = ((FrameworkElement)Content).XamlRoot,
                Title = "Queue catch-up on Windows", Content = CreateDialogScrollContent(content, 580),
                PrimaryButtonText = "Queue on Windows", CloseButtonText = "Cancel", IsPrimaryButtonEnabled = schedules.Count > 0 };
            if (await dialog.ShowAsync() != ContentDialogResult.Primary || feeds.SelectedItem is not FeedSchedule feed) return;
            if (start.Date.Date > end.Date.Date || end.Date.Date > DateTime.Today)
                throw new InvalidOperationException("Choose an ordered date range ending no later than today.");
            var full = mode.SelectedIndex == 0;
            var response = await _worker.SharedJobsRequestAsync("", new { command = "run-scheduled", payload = new
            {
                account_profile_id = "automatic", use_saved_profile = true, analyze = full,
                job = new { feed_id = feed.FeedId, start_date = start.Date.ToString("yyyy-MM-dd"),
                    end_date = end.Date.ToString("yyyy-MM-dd"), combine = full, transcribe = full, diarize = full },
            }});
            var job = response.GetProperty("job");
            AppendLog($"Shared job {SharedText(job, "id")}: {SharedText(job, "status")} on Windows.");
            await RefreshSharedJobsAsync();
        }
        catch (Exception exception) { await ShowErrorAsync(exception); }
    }
}
