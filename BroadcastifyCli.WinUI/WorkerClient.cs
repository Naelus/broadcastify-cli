using System.Diagnostics;
using System.Text;
using System.Text.Json;

namespace BroadcastifyCli.WinUI;

internal sealed class WorkerClient
{
    private static readonly Encoding Utf8WithoutBom =
        new UTF8Encoding(encoderShouldEmitUTF8Identifier: false);

    private readonly PythonCommand _python;
    private readonly SemaphoreSlim _lanNodeGate = new(1, 1);
    private Process? _lanNodeProcess;
    private string _lanNodeConfiguration = "";
    private int _lanNodePort;
    private int _lanNodeShutdown;

    public string RepositoryRoot { get; }
    public string PythonDisplayName => _python.DisplayName;
    public string BundledEnvironmentPath { get; }
    public bool HasBundledEnvironment => File.Exists(BundledEnvironmentPath);
    public string BundledWindowsMlHelperPath { get; }
    public bool HasBundledWindowsMlHelper => File.Exists(BundledWindowsMlHelperPath);

    public WorkerClient()
    {
        RepositoryRoot = FindRepositoryRoot();
        _python = ResolvePython(RepositoryRoot);
        BundledEnvironmentPath = Path.Combine(AppContext.BaseDirectory, "broadcastify-desktop.env");
        BundledWindowsMlHelperPath = Path.Combine(
            AppContext.BaseDirectory,
            "windowsml",
            "BroadcastifyCli.WindowsML.exe");
    }

    public async Task<string> ConfigureLanNodeAsync(
        bool enabled,
        string outputDirectory,
        int port)
    {
        await _lanNodeGate.WaitAsync();
        try
        {
            if (Volatile.Read(ref _lanNodeShutdown) != 0)
            {
                StopLanNodeCore();
                return "LAN archive sharing stopped with the desktop app.";
            }
            if (!enabled)
            {
                StopLanNodeCore();
                return "LAN archive sharing is off.";
            }
            var output = Path.GetFullPath(
                string.IsNullOrWhiteSpace(outputDirectory)
                    ? Path.Combine(RepositoryRoot, "archives")
                    : Path.IsPathRooted(outputDirectory)
                        ? outputDirectory
                        : Path.Combine(RepositoryRoot, outputDirectory));
            var boundedPort = Math.Clamp(port, 1024, 65535);
            var configuration = $"{output}|{boundedPort}";
            if (_lanNodeProcess is { HasExited: false }
                && configuration.Equals(
                    _lanNodeConfiguration,
                    StringComparison.OrdinalIgnoreCase))
            {
                return $"Sharing original archive blocks and eligible for shared upstream leases on trusted LAN port {boundedPort}.";
            }

            StopLanNodeCore();
            var startInfo = CreateStartInfo(
                [
                    "-m", "broadcastify_cli.lan_node",
                    "--host", "0.0.0.0",
                    "--port", boundedPort.ToString(
                        System.Globalization.CultureInfo.InvariantCulture),
                    "--output-dir", output,
                ],
                redirectStreams: false);
            var process = new Process
            {
                StartInfo = startInfo,
                EnableRaisingEvents = true,
            };
            if (!process.Start())
            {
                process.Dispose();
                throw new InvalidOperationException(
                    "Unable to start the read-only LAN archive node.");
            }
            _lanNodeProcess = process;
            _lanNodeConfiguration = configuration;
            Volatile.Write(ref _lanNodePort, boundedPort);
            if (Volatile.Read(ref _lanNodeShutdown) != 0)
            {
                StopLanNodeCore();
                return "LAN archive sharing stopped with the desktop app.";
            }

            using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(1) };
            var health = new Uri($"http://127.0.0.1:{boundedPort}/health");
            for (var attempt = 0; attempt < 20; attempt++)
            {
                if (Volatile.Read(ref _lanNodeShutdown) != 0)
                {
                    StopLanNodeCore();
                    return "LAN archive sharing stopped with the desktop app.";
                }
                int? exitCode = null;
                try
                {
                    if (process.HasExited)
                    {
                        exitCode = process.ExitCode;
                    }
                }
                catch (InvalidOperationException)
                    when (Volatile.Read(ref _lanNodeShutdown) != 0)
                {
                    return "LAN archive sharing stopped with the desktop app.";
                }
                if (exitCode is not null)
                {
                    StopLanNodeCore();
                    throw new InvalidOperationException(
                        $"The LAN archive node exited during startup (code {exitCode}). "
                        + "Choose another port or review the local firewall/runtime.");
                }
                try
                {
                    using var response = await client.GetAsync(health);
                    if (response.IsSuccessStatusCode)
                    {
                        var body = await response.Content.ReadAsStringAsync();
                        if (body.Contains(
                            "radio-archive-lan/1",
                            StringComparison.Ordinal))
                        {
                            return $"Sharing original archive blocks and eligible for shared upstream leases on trusted LAN port {boundedPort}.";
                        }
                    }
                }
                catch (HttpRequestException)
                {
                    // The process is still starting.
                }
                catch (TaskCanceledException)
                {
                    // Retry within the bounded startup window.
                }
                await Task.Delay(150);
            }
            StopLanNodeCore();
            throw new InvalidOperationException(
                "The LAN archive node did not become healthy within three seconds.");
        }
        finally
        {
            _lanNodeGate.Release();
        }
    }

    public void StopLanNode()
    {
        // Window close is synchronous. Do not wait on an async startup
        // continuation that may need the UI thread; terminate the owned child
        // immediately and let an in-flight configuration unwind.
        Interlocked.Exchange(ref _lanNodeShutdown, 1);
        StopLanNodeCore();
    }

    private void StopLanNodeCore()
    {
        var process = Interlocked.Exchange(ref _lanNodeProcess, null);
        _lanNodeConfiguration = "";
        Volatile.Write(ref _lanNodePort, 0);
        if (process is null)
        {
            return;
        }
        try
        {
            if (!process.HasExited)
            {
                process.Kill(entireProcessTree: true);
                process.WaitForExit(2_000);
            }
        }
        catch (InvalidOperationException)
        {
            // It exited between the state check and stop request.
        }
        finally
        {
            process.Dispose();
        }
    }

    public async Task<IReadOnlyList<FeedSearchResult>> SearchFeedsAsync(
        string query,
        CancellationToken cancellationToken)
    {
        List<FeedSearchResult>? results = null;
        await RunWorkerAsync(
            ["-m", "broadcastify_cli.worker", "search", "--query", query],
            null,
            message =>
            {
                if (message.TryGetProperty("type", out var type) && type.GetString() == "result")
                {
                    results = message.GetProperty("results")
                        .Deserialize<List<FeedSearchResult>>(JsonOptions);
                }
            },
            cancellationToken);
        return results ?? [];
    }

    public async Task<AreaSearchResponse> SearchAreaFeedsAsync(
        IReadOnlyList<string> zipCodes,
        string? centerZip,
        double? radiusMiles,
        int maxZipCodes,
        CancellationToken cancellationToken)
    {
        List<FeedSearchResult>? results = null;
        var arguments = new List<string> { "-m", "broadcastify_cli.worker", "area-search" };
        if (!string.IsNullOrWhiteSpace(centerZip))
        {
            arguments.Add("--center-zip");
            arguments.Add(centerZip);
            arguments.Add("--radius-miles");
            arguments.Add((radiusMiles ?? 25).ToString(System.Globalization.CultureInfo.InvariantCulture));
            arguments.Add("--max-zip-codes");
            arguments.Add(maxZipCodes.ToString(System.Globalization.CultureInfo.InvariantCulture));
        }
        else
        {
            foreach (var zipCode in zipCodes)
            {
                arguments.Add("--zip");
                arguments.Add(zipCode);
            }
        }
        AreaCoverage coverage = new();
        await RunWorkerAsync(
            arguments,
            null,
            message =>
            {
                if (message.TryGetProperty("type", out var type) && type.GetString() == "area_search")
                {
                    results = message.GetProperty("results")
                        .Deserialize<List<FeedSearchResult>>(JsonOptions);
                    if (message.TryGetProperty("coverage", out var coverageValue))
                    {
                        coverage = coverageValue.Deserialize<AreaCoverage>(JsonOptions) ?? new();
                    }
                }
            },
            cancellationToken);
        return new AreaSearchResponse { Results = results ?? [], Coverage = coverage };
    }

    public async Task<IReadOnlyList<AreaProfile>> ListAreaProfilesAsync(
        CancellationToken cancellationToken)
    {
        List<AreaProfile>? profiles = null;
        await RunWorkerAsync(
            ["-m", "broadcastify_cli.worker", "area-profiles"],
            null,
            message =>
            {
                if (message.TryGetProperty("type", out var type) && type.GetString() == "area_profiles")
                {
                    profiles = message.GetProperty("profiles")
                        .Deserialize<List<AreaProfile>>(JsonOptions);
                }
            },
            cancellationToken);
        return profiles ?? [];
    }

    public async Task<AreaProfile?> SaveAreaProfileAsync(
        AreaProfileSaveRequest request,
        Action<JsonElement> onMessage,
        CancellationToken cancellationToken)
    {
        AreaProfile? profile = null;
        await RunWorkerAsync(
            ["-m", "broadcastify_cli.worker", "save-area-profile"],
            JsonSerializer.Serialize(request, JsonOptions),
            message =>
            {
                if (message.TryGetProperty("type", out var type)
                    && type.GetString() == "area_profile_saved")
                {
                    profile = message.GetProperty("profile").Deserialize<AreaProfile>(JsonOptions);
                }
                onMessage(message);
            },
            cancellationToken);
        return profile;
    }

    public Task AuthenticateAsync(
        string username,
        string password,
        Action<JsonElement> onMessage,
        CancellationToken cancellationToken)
    {
        var json = JsonSerializer.Serialize(new { username, password });
        return RunWorkerAsync(
            ["-m", "broadcastify_cli.worker", "authenticate"],
            json,
            onMessage,
            cancellationToken);
    }

    public async Task<JobRunResult?> RunJobAsync(
        JobRequest request,
        Action<JsonElement> onMessage,
        CancellationToken cancellationToken)
    {
        JobRunResult? result = null;
        var json = JsonSerializer.Serialize(request, JsonOptions);
        await RunWorkerAsync(
            ["-m", "broadcastify_cli.worker", "run"],
            json,
            message =>
            {
                if (message.TryGetProperty("type", out var type)
                    && type.GetString() == "complete"
                    && message.TryGetProperty("result", out var value))
                {
                    result = value.Deserialize<JobRunResult>(JsonOptions);
                }
                onMessage(message);
            },
            cancellationToken);
        return result;
    }

    public async Task<AreaAcquisitionResult?> RunAreaAcquisitionAsync(
        AreaAcquisitionRequest request,
        Action<JsonElement> onMessage,
        CancellationToken cancellationToken)
    {
        AreaAcquisitionResult? result = null;
        await RunWorkerAsync(
            ["-m", "broadcastify_cli.worker", "run-area"],
            JsonSerializer.Serialize(request, JsonOptions),
            message =>
            {
                if (message.TryGetProperty("type", out var type)
                    && type.GetString() == "area_complete"
                    && message.TryGetProperty("result", out var value))
                {
                    result = value.Deserialize<AreaAcquisitionResult>(JsonOptions);
                }
                onMessage(message);
            },
            cancellationToken);
        return result;
    }

    public async Task<IReadOnlyList<AreaAcquisitionResult>> ListAreaAcquisitionRunsAsync(
        string profileName,
        CancellationToken cancellationToken)
    {
        List<AreaAcquisitionResult>? results = null;
        await RunWorkerAsync(
            [
                "-m", "broadcastify_cli.worker", "area-runs",
                "--profile-name", profileName,
            ],
            null,
            message =>
            {
                if (message.TryGetProperty("type", out var type)
                    && type.GetString() == "area_runs")
                {
                    results = message.GetProperty("runs")
                        .Deserialize<List<AreaAcquisitionResult>>(JsonOptions);
                }
            },
            cancellationToken);
        return results ?? [];
    }

    public async Task<LibraryResponse> ListLibraryAsync(
        string outputDirectory,
        CancellationToken cancellationToken)
    {
        LibraryResponse? result = null;
        await RunWorkerAsync(
            [
                "-m", "broadcastify_cli.worker", "library",
                "--output-dir", string.IsNullOrWhiteSpace(outputDirectory) ? "archives" : outputDirectory,
            ],
            null,
            message =>
            {
                if (message.TryGetProperty("type", out var type)
                    && type.GetString() == "library_days")
                {
                    result = message.Deserialize<LibraryResponse>(JsonOptions);
                }
            },
            cancellationToken);
        return result ?? new LibraryResponse();
    }

    public async Task<DayReport?> ContinueLocalDayAsync(
        LocalProcessingRequest request,
        Action<JsonElement> onMessage,
        CancellationToken cancellationToken)
    {
        DayReport? report = null;
        await RunWorkerAsync(
            ["-m", "broadcastify_cli.worker", "continue-local"],
            JsonSerializer.Serialize(request, JsonOptions),
            message =>
            {
                if (message.TryGetProperty("type", out var type)
                    && type.GetString() == "local_complete"
                    && message.TryGetProperty("report", out var value)
                    && value.ValueKind == JsonValueKind.Object)
                {
                    report = value.Deserialize<DayReport>(JsonOptions);
                }
                onMessage(message);
            },
            cancellationToken);
        return report;
    }

    public async Task<IReadOnlyList<AnalysisDay>> ListAnalysisDaysAsync(
        string? feedId,
        CancellationToken cancellationToken)
    {
        List<AnalysisDay>? days = null;
        var arguments = new List<string> { "-m", "broadcastify_cli.worker", "analysis-days" };
        if (!string.IsNullOrWhiteSpace(feedId))
        {
            arguments.Add("--feed-id");
            arguments.Add(feedId.Trim());
        }
        await RunWorkerAsync(
            arguments,
            null,
            message =>
            {
                if (message.TryGetProperty("type", out var type) && type.GetString() == "analysis_days")
                {
                    days = message.GetProperty("days").Deserialize<List<AnalysisDay>>(JsonOptions);
                }
            },
            cancellationToken);
        return days ?? [];
    }

    public async Task<DayReport?> GetDayReportAsync(
        string feedId,
        string archiveDate,
        CancellationToken cancellationToken)
    {
        DayReport? report = null;
        await RunWorkerAsync(
            [
                "-m", "broadcastify_cli.worker", "report-day",
                "--feed-id", feedId,
                "--date", archiveDate,
            ],
            null,
            message =>
            {
                if (message.TryGetProperty("type", out var type) && type.GetString() == "day_report")
                {
                    report = message.GetProperty("report").Deserialize<DayReport>(JsonOptions);
                }
            },
            cancellationToken);
        return report;
    }

    public async Task<IncidentClip?> GetIncidentClipAsync(
        long incidentId,
        bool includeSurroundingContext,
        CancellationToken cancellationToken)
    {
        IncidentClip? clip = null;
        var arguments = new List<string>
        {
            "-m", "broadcastify_cli.worker", "incident-clip",
            "--incident-id", incidentId.ToString(),
        };
        if (includeSurroundingContext)
        {
            arguments.Add("--context");
        }
        await RunWorkerAsync(
            arguments,
            null,
            message =>
            {
                if (message.TryGetProperty("type", out var type)
                    && type.GetString() == "incident_clip")
                {
                    clip = message.GetProperty("clip").Deserialize<IncidentClip>(JsonOptions);
                }
            },
            cancellationToken);
        return clip;
    }

    public async Task<DayReport?> AnalyzeDayAsync(
        AnalysisRequest request,
        Action<JsonElement> onMessage,
        CancellationToken cancellationToken)
    {
        DayReport? report = null;
        var json = JsonSerializer.Serialize(request, JsonOptions);
        await RunWorkerAsync(
            ["-m", "broadcastify_cli.worker", "analyze-day"],
            json,
            message =>
            {
                if (message.TryGetProperty("type", out var type)
                    && type.GetString() == "analysis_complete")
                {
                    report = message.GetProperty("report").Deserialize<DayReport>(JsonOptions);
                }
                onMessage(message);
            },
            cancellationToken);
        return report;
    }

    public async Task<ArchiveAnswer?> AskArchiveAsync(
        ArchiveQuestionRequest request,
        Action<JsonElement> onMessage,
        CancellationToken cancellationToken)
    {
        ArchiveAnswer? answer = null;
        var json = JsonSerializer.Serialize(request, JsonOptions);
        await RunWorkerAsync(
            ["-m", "broadcastify_cli.worker", "ask"],
            json,
            message =>
            {
                if (message.TryGetProperty("type", out var type) && type.GetString() == "answer")
                {
                    answer = message.GetProperty("result").Deserialize<ArchiveAnswer>(JsonOptions);
                }
                onMessage(message);
            },
            cancellationToken);
        return answer;
    }

    public async Task<WeeklyReport?> SummarizeWeekAsync(
        WeeklySummaryRequest request,
        Action<JsonElement> onMessage,
        CancellationToken cancellationToken)
    {
        WeeklyReport? report = null;
        var json = JsonSerializer.Serialize(request, JsonOptions);
        await RunWorkerAsync(
            ["-m", "broadcastify_cli.worker", "summarize-week"],
            json,
            message =>
            {
                if (message.TryGetProperty("type", out var type)
                    && type.GetString() == "weekly_summary")
                {
                    report = message.GetProperty("result").Deserialize<WeeklyReport>(JsonOptions);
                }
                onMessage(message);
            },
            cancellationToken);
        return report;
    }

    public async Task<AreaDigestReport?> SummarizeAreaAsync(
        AreaDigestRequest request,
        Action<JsonElement> onMessage,
        CancellationToken cancellationToken)
    {
        AreaDigestReport? report = null;
        await RunWorkerAsync(
            ["-m", "broadcastify_cli.worker", "summarize-area"],
            JsonSerializer.Serialize(request, JsonOptions),
            message =>
            {
                if (message.TryGetProperty("type", out var type)
                    && type.GetString() == "area_digest")
                {
                    report = message.GetProperty("result").Deserialize<AreaDigestReport>(JsonOptions);
                }
                onMessage(message);
            },
            cancellationToken);
        return report;
    }

    public async Task<AreaDigestReport?> GetLatestAreaDigestAsync(
        string profileName,
        CancellationToken cancellationToken)
    {
        AreaDigestReport? report = null;
        await RunWorkerAsync(
            [
                "-m", "broadcastify_cli.worker", "saved-area-digest",
                "--profile-name", profileName,
            ],
            null,
            message =>
            {
                if (message.TryGetProperty("type", out var type)
                    && type.GetString() == "saved_area_digest"
                    && message.TryGetProperty("result", out var value)
                    && value.ValueKind == JsonValueKind.Object)
                {
                    report = value.Deserialize<AreaDigestReport>(JsonOptions);
                }
            },
            cancellationToken);
        return report;
    }

    public async Task<JsonElement?> GetDiagnosticsAsync(
        CancellationToken cancellationToken,
        AsrSelfTestRequest? asrRequest = null)
    {
        JsonElement? diagnostics = null;
        await RunWorkerAsync(
            [
                "-m",
                "broadcastify_cli.worker",
                asrRequest is null ? "diagnostics" : "diagnostics-selected",
            ],
            asrRequest is null ? null : JsonSerializer.Serialize(asrRequest, JsonOptions),
            message =>
            {
                if (message.TryGetProperty("type", out var type) && type.GetString() == "diagnostics")
                {
                    diagnostics = message.Clone();
                }
            },
            cancellationToken);
        return diagnostics;
    }

    public async Task<ArchiveQuotaStatus?> GetArchiveQuotaStatusAsync(
        CancellationToken cancellationToken)
    {
        ArchiveQuotaStatus? status = null;
        await RunWorkerAsync(
            ["-m", "broadcastify_cli.worker", "quota-status"],
            null,
            message =>
            {
                if (message.TryGetProperty("type", out var type)
                    && type.GetString() == "archive_quota_status"
                    && message.TryGetProperty("status", out var value))
                {
                    status = value.Deserialize<ArchiveQuotaStatus>(JsonOptions);
                }
            },
            cancellationToken);
        return status;
    }

    public async Task<IReadOnlyList<FeedSchedule>> ListFeedSchedulesAsync(
        CancellationToken cancellationToken)
    {
        List<FeedSchedule>? schedules = null;
        await RunWorkerAsync(
            ["-m", "broadcastify_cli.worker", "schedules"],
            null,
            message =>
            {
                if (message.TryGetProperty("type", out var type)
                    && type.GetString() == "feed_schedules")
                {
                    schedules = message.GetProperty("schedules")
                        .Deserialize<List<FeedSchedule>>(JsonOptions);
                }
            },
            cancellationToken);
        return schedules ?? [];
    }

    public async Task<FeedSchedule?> SaveFeedScheduleAsync(
        FeedScheduleSaveRequest request,
        CancellationToken cancellationToken)
    {
        FeedSchedule? schedule = null;
        await RunWorkerAsync(
            ["-m", "broadcastify_cli.worker", "save-schedule"],
            JsonSerializer.Serialize(request, JsonOptions),
            message =>
            {
                if (message.TryGetProperty("type", out var type)
                    && type.GetString() == "feed_schedule_saved")
                {
                    schedule = message.GetProperty("schedule")
                        .Deserialize<FeedSchedule>(JsonOptions);
                }
            },
            cancellationToken);
        return schedule;
    }

    public async Task<FeedSchedule?> ClaimDueFeedScheduleAsync(
        CancellationToken cancellationToken)
    {
        FeedSchedule? schedule = null;
        await RunWorkerAsync(
            ["-m", "broadcastify_cli.worker", "claim-due-schedule"],
            null,
            message =>
            {
                if (message.TryGetProperty("type", out var type)
                    && type.GetString() == "feed_schedule_claim"
                    && message.TryGetProperty("schedule", out var value)
                    && value.ValueKind != JsonValueKind.Null)
                {
                    schedule = value.Deserialize<FeedSchedule>(JsonOptions);
                }
            },
            cancellationToken);
        return schedule;
    }

    public async Task<FeedSchedule?> FinishFeedScheduleAsync(
        FeedScheduleFinishRequest request,
        CancellationToken cancellationToken)
    {
        FeedSchedule? schedule = null;
        await RunWorkerAsync(
            ["-m", "broadcastify_cli.worker", "finish-schedule"],
            JsonSerializer.Serialize(request, JsonOptions),
            message =>
            {
                if (message.TryGetProperty("type", out var type)
                    && type.GetString() == "feed_schedule_finished")
                {
                    schedule = message.GetProperty("schedule")
                        .Deserialize<FeedSchedule>(JsonOptions);
                }
            },
            cancellationToken);
        return schedule;
    }

    public async Task DeleteFeedScheduleAsync(
        long scheduleId,
        CancellationToken cancellationToken)
    {
        await RunWorkerAsync(
            ["-m", "broadcastify_cli.worker", "delete-schedule"],
            JsonSerializer.Serialize(new { schedule_id = scheduleId }, JsonOptions),
            _ => { },
            cancellationToken);
    }

    public async Task RecoverFeedSchedulesAsync(CancellationToken cancellationToken)
    {
        await RunWorkerAsync(
            ["-m", "broadcastify_cli.worker", "recover-schedules"],
            null,
            _ => { },
            cancellationToken);
    }

    public async Task<AsrSelfTestStatus?> RunAsrSelfTestAsync(
        AsrSelfTestRequest request,
        Action<JsonElement> onMessage,
        CancellationToken cancellationToken)
    {
        AsrSelfTestStatus? result = null;
        await RunWorkerAsync(
            ["-m", "broadcastify_cli.worker", "asr-self-test"],
            JsonSerializer.Serialize(request, JsonOptions),
            message =>
            {
                if (message.TryGetProperty("type", out var type)
                    && type.GetString() == "asr_self_test"
                    && message.TryGetProperty("result", out var value))
                {
                    result = value.Deserialize<AsrSelfTestStatus>(JsonOptions);
                }
                onMessage(message);
            },
            cancellationToken);
        return result;
    }

    public async Task<AsrModelPreparationStatus?> PrepareAsrModelAsync(
        AsrSelfTestRequest request,
        Action<JsonElement> onMessage,
        CancellationToken cancellationToken)
    {
        AsrModelPreparationStatus? result = null;
        await RunWorkerAsync(
            ["-m", "broadcastify_cli.worker", "prepare-asr-model"],
            JsonSerializer.Serialize(request, JsonOptions),
            message =>
            {
                if (message.TryGetProperty("type", out var type)
                    && type.GetString() == "asr_model_prepared"
                    && message.TryGetProperty("result", out var value))
                {
                    result = value.Deserialize<AsrModelPreparationStatus>(JsonOptions);
                }
                onMessage(message);
            },
            cancellationToken);
        return result;
    }

    public async Task<DiarizationSelfTestStatus?> RunDiarizationSelfTestAsync(
        DiarizationSelfTestRequest request,
        Action<JsonElement> onMessage,
        CancellationToken cancellationToken)
    {
        DiarizationSelfTestStatus? result = null;
        await RunWorkerAsync(
            ["-m", "broadcastify_cli.worker", "diarization-self-test"],
            JsonSerializer.Serialize(request, JsonOptions),
            message =>
            {
                if (message.TryGetProperty("type", out var type)
                    && type.GetString() == "diarization_self_test"
                    && message.TryGetProperty("result", out var value))
                {
                    result = value.Deserialize<DiarizationSelfTestStatus>(JsonOptions);
                }
                onMessage(message);
            },
            cancellationToken);
        return result;
    }

    public async Task<ProfileSelfTestStatus?> RunProfileSelfTestAsync(
        LocalProcessingRequest request,
        Action<JsonElement> onMessage,
        CancellationToken cancellationToken)
    {
        ProfileSelfTestStatus? result = null;
        await RunWorkerAsync(
            ["-m", "broadcastify_cli.worker", "profile-self-test"],
            JsonSerializer.Serialize(request, JsonOptions),
            message =>
            {
                if (message.TryGetProperty("type", out var type)
                    && type.GetString() == "profile_self_test"
                    && message.TryGetProperty("result", out var value))
                {
                    result = value.Deserialize<ProfileSelfTestStatus>(JsonOptions);
                }
                onMessage(message);
            },
            cancellationToken);
        return result;
    }

    public async Task<AnalysisProviderStatus?> GetAnalysisProviderDiagnosticsAsync(
        AnalysisProviderDiagnosticsRequest request,
        CancellationToken cancellationToken)
    {
        AnalysisProviderStatus? result = null;
        await RunWorkerAsync(
            ["-m", "broadcastify_cli.worker", "analysis-provider-diagnostics"],
            JsonSerializer.Serialize(request, JsonOptions),
            message =>
            {
                if (message.TryGetProperty("type", out var type)
                    && type.GetString() == "analysis_provider_diagnostics")
                {
                    result = message.GetProperty("result")
                        .Deserialize<AnalysisProviderStatus>(JsonOptions);
                }
            },
            cancellationToken);
        return result;
    }

    public async Task<AnalysisProviderStatus?> RunAnalysisSelfTestAsync(
        AnalysisSelfTestRequest request,
        Action<JsonElement> onMessage,
        CancellationToken cancellationToken)
    {
        AnalysisProviderStatus? result = null;
        await RunWorkerAsync(
            ["-m", "broadcastify_cli.worker", "analysis-self-test"],
            JsonSerializer.Serialize(request, JsonOptions),
            message =>
            {
                if (message.TryGetProperty("type", out var type)
                    && type.GetString() == "analysis_self_test"
                    && message.TryGetProperty("result", out var value))
                {
                    result = value.Deserialize<AnalysisProviderStatus>(JsonOptions);
                }
                onMessage(message);
            },
            cancellationToken);
        return result;
    }

    private async Task RunWorkerAsync(
        IReadOnlyList<string> arguments,
        string? stdin,
        Action<JsonElement> onMessage,
        CancellationToken cancellationToken,
        IReadOnlyDictionary<string, string>? environment = null)
    {
        using var process = new Process
        {
            StartInfo = CreateStartInfo(arguments, environment),
            EnableRaisingEvents = true,
        };
        if (!process.Start())
        {
            throw new InvalidOperationException("Unable to start the Python worker.");
        }

        using var registration = cancellationToken.Register(() =>
        {
            try
            {
                if (!process.HasExited)
                {
                    process.Kill(entireProcessTree: true);
                }
            }
            catch (InvalidOperationException)
            {
                // The worker exited between the check and kill.
            }
        });

        if (stdin is not null)
        {
            await process.StandardInput.WriteAsync(stdin.AsMemory(), cancellationToken);
        }
        process.StandardInput.Close();

        var stderrTask = process.StandardError.ReadToEndAsync(cancellationToken);
        string? workerError = null;
        while (await process.StandardOutput.ReadLineAsync(cancellationToken) is { } line)
        {
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }
            using var document = JsonDocument.Parse(line);
            var message = document.RootElement.Clone();
            if (message.TryGetProperty("type", out var type) && type.GetString() == "error")
            {
                workerError = message.TryGetProperty("message", out var value)
                    ? value.GetString()
                    : "Python worker failed.";
            }
            onMessage(message);
        }

        await process.WaitForExitAsync(cancellationToken);
        var stderr = await stderrTask;
        if (process.ExitCode != 0)
        {
            throw new InvalidOperationException(
                workerError ?? (string.IsNullOrWhiteSpace(stderr) ? "Python worker failed." : stderr.Trim()));
        }
    }

    private ProcessStartInfo CreateStartInfo(
        IReadOnlyList<string> arguments,
        IReadOnlyDictionary<string, string>? environment = null,
        bool redirectStreams = true)
    {
        var startInfo = new ProcessStartInfo
        {
            FileName = _python.FileName,
            WorkingDirectory = RepositoryRoot,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardInput = redirectStreams,
            RedirectStandardOutput = redirectStreams,
            RedirectStandardError = redirectStreams,
        };
        if (redirectStreams)
        {
            startInfo.StandardInputEncoding = Utf8WithoutBom;
            startInfo.StandardOutputEncoding = Utf8WithoutBom;
            startInfo.StandardErrorEncoding = Utf8WithoutBom;
        }
        startInfo.Environment["PYTHONIOENCODING"] = "utf-8";
        startInfo.Environment["PYTHONUTF8"] = "1";
        Directory.CreateDirectory(AppSettingsStore.LocalDataDirectory);
        startInfo.Environment["BROADCASTIFY_QUOTA_LEDGER"] = Path.Combine(
            AppSettingsStore.LocalDataDirectory,
            "archive-quota.sqlite3");
        foreach (var argument in _python.PrefixArguments.Concat(arguments))
        {
            startInfo.ArgumentList.Add(argument);
        }
        if (HasBundledEnvironment)
        {
            startInfo.Environment["BROADCASTIFY_ENV_FILE"] = BundledEnvironmentPath;
        }
        var lanNodePort = Volatile.Read(ref _lanNodePort);
        if (lanNodePort is >= 1024 and <= 65535)
        {
            startInfo.Environment["BROADCASTIFY_LAN_SELF_PORT"] =
                lanNodePort.ToString(
                    System.Globalization.CultureInfo.InvariantCulture);
        }
        if (HasBundledWindowsMlHelper
            && (!startInfo.Environment.TryGetValue("WINDOWS_ML_HELPER_PATH", out var configuredHelper)
                || string.IsNullOrWhiteSpace(configuredHelper)))
        {
            startInfo.Environment["WINDOWS_ML_HELPER_PATH"] = BundledWindowsMlHelperPath;
        }
        if (environment is not null)
        {
            foreach (var (name, value) in environment)
            {
                startInfo.Environment[name] = value;
            }
        }
        return startInfo;
    }

    private static string FindRepositoryRoot()
    {
        foreach (var startingPath in new[] { Environment.CurrentDirectory, AppContext.BaseDirectory })
        {
            var directory = new DirectoryInfo(startingPath);
            while (directory is not null)
            {
                if (File.Exists(Path.Combine(directory.FullName, "pyproject.toml")))
                {
                    return directory.FullName;
                }
                directory = directory.Parent;
            }
        }
        throw new DirectoryNotFoundException(
            "Could not find the broadcastify-cli repository. Start the app from the repository or its build output.");
    }

    private static PythonCommand ResolvePython(string repositoryRoot)
    {
        var configured = Environment.GetEnvironmentVariable("BROADCASTIFY_PYTHON");
        var candidates = new List<PythonCommand>();
        if (!string.IsNullOrWhiteSpace(configured))
        {
            candidates.Add(new PythonCommand(configured, [], configured));
        }
        candidates.Add(new PythonCommand(
            Path.Combine(repositoryRoot, ".venv", "Scripts", "python.exe"), [], ".venv Python"));
        candidates.Add(new PythonCommand("python.exe", [], "python.exe"));
        candidates.Add(new PythonCommand("py.exe", ["-3"], "py.exe -3"));

        foreach (var candidate in candidates)
        {
            if (CanRunPython(candidate, repositoryRoot))
            {
                return candidate;
            }
        }
        throw new FileNotFoundException(
            "No working Python installation was found. Recreate .venv or set BROADCASTIFY_PYTHON to python.exe.");
    }

    private static bool CanRunPython(PythonCommand candidate, string repositoryRoot)
    {
        try
        {
            using var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = candidate.FileName,
                    WorkingDirectory = repositoryRoot,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                },
            };
            foreach (var argument in candidate.PrefixArguments)
            {
                process.StartInfo.ArgumentList.Add(argument);
            }
            process.StartInfo.ArgumentList.Add("--version");
            return process.Start() && process.WaitForExit(3000) && process.ExitCode == 0;
        }
        catch (Exception exception) when (
            exception is System.ComponentModel.Win32Exception
                or InvalidOperationException
                or FileNotFoundException)
        {
            return false;
        }
    }

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
    };

    private sealed record PythonCommand(
        string FileName,
        IReadOnlyList<string> PrefixArguments,
        string DisplayName);
}
