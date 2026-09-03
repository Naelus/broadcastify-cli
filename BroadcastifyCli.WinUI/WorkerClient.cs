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
    private string _libraryDirectory = "";
    private int _lanNodePort;
    private int _lanNodeShutdown;

    public static string BundledPythonPath => Path.Combine(
        AppContext.BaseDirectory,
        "runtime",
        "python",
        "python.exe");
    public static bool BundledRuntimeAvailable =>
        File.Exists(BundledPythonPath);
    public string RepositoryRoot { get; }
    public string WorkingDirectory { get; }
    public bool IsBundledRuntime { get; }
    public string PythonDisplayName => _python.DisplayName;
    public string? PythonRuntimeWarning { get; }
    public string BundledEnvironmentPath { get; }
    public bool HasBundledEnvironment => File.Exists(BundledEnvironmentPath);
    public string BundledManagedRuntimeManifestPath { get; }
    public string BundledWorkerWheelPath { get; }
    public string ManagedRuntimeRoot { get; }
    public bool HasBundledManagedRuntime =>
        File.Exists(BundledManagedRuntimeManifestPath)
        && File.Exists(BundledWorkerWheelPath);
    public string BundledWindowsMlHelperPath { get; }
    public bool HasBundledWindowsMlHelper => File.Exists(BundledWindowsMlHelperPath);

    public WorkerClient(string? preferredPython = null)
    {
        var bundledPython = BundledPythonPath;
        IsBundledRuntime = File.Exists(bundledPython);
        RepositoryRoot = IsBundledRuntime
            ? AppContext.BaseDirectory.TrimEnd(
                Path.DirectorySeparatorChar,
                Path.AltDirectorySeparatorChar)
            : FindRepositoryRoot();
        WorkingDirectory = IsBundledRuntime
            ? AppSettingsStore.LocalDataDirectory
            : RepositoryRoot;
        Directory.CreateDirectory(WorkingDirectory);
        SetLibraryDirectory("archives");
        var bootstrapDirectory = Path.Combine(
            AppContext.BaseDirectory,
            "runtime",
            "bootstrap");
        BundledManagedRuntimeManifestPath = Path.Combine(
            bootstrapDirectory,
            "managed-runtime.json");
        BundledWorkerWheelPath = Directory.Exists(bootstrapDirectory)
            ? Directory.GetFiles(
                    bootstrapDirectory,
                    "broadcastify_cli-*.whl",
                    SearchOption.TopDirectoryOnly)
                .OrderByDescending(value => value, StringComparer.OrdinalIgnoreCase)
                .FirstOrDefault() ?? ""
            : "";
        ManagedRuntimeRoot = Path.Combine(
            AppSettingsStore.LocalDataDirectory,
            "managed-runtimes");
        var pythonResolution = ResolvePython(
            RepositoryRoot,
            WorkingDirectory,
            IsBundledRuntime ? bundledPython : null,
            preferredPython,
            BundledWorkerWheelPath);
        _python = pythonResolution.Command;
        PythonRuntimeWarning = pythonResolution.Warning;
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
                    ? Path.Combine(WorkingDirectory, "archives")
                    : Path.IsPathRooted(outputDirectory)
                        ? outputDirectory
                        : Path.Combine(WorkingDirectory, outputDirectory));
            var boundedPort = Math.Clamp(port, 1024, 65535);
            var configuration = $"{output}|{boundedPort}";
            if (_lanNodeProcess is { HasExited: false }
                && configuration.Equals(
                    _lanNodeConfiguration,
                    StringComparison.OrdinalIgnoreCase))
            {
                return $"Windows master is available for optional follower requests on trusted LAN port {boundedPort}.";
            }

            StopLanNodeCore();
            var startInfo = CreateStartInfo(
                [
                    "-m", "broadcastify_cli.lan_node",
                    "--host", "0.0.0.0",
                    "--port", boundedPort.ToString(
                        System.Globalization.CultureInfo.InvariantCulture),
                    "--output-dir", output,
                    "--role", "master",
                    "--database", Path.Combine(
                        output,
                        "broadcastify-analysis.sqlite3"),
                    "--working-dir", WorkingDirectory,
                    "--enable-job-relay",
                    "--no-background-sync",
                    "--parent-pid", Environment.ProcessId.ToString(
                        System.Globalization.CultureInfo.InvariantCulture),
                ],
                redirectStreams: false,
                coordinatorPort: boundedPort);
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
                            return $"Windows master is available for optional follower requests on trusted LAN port {boundedPort}.";
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

    public void SetLibraryDirectory(string outputDirectory)
    {
        var configured = string.IsNullOrWhiteSpace(outputDirectory)
            ? "archives"
            : outputDirectory.Trim();
        try
        {
            var resolved = Path.GetFullPath(
                Path.IsPathRooted(configured)
                    ? configured
                    : Path.Combine(WorkingDirectory, configured));
            Volatile.Write(ref _libraryDirectory, resolved);
        }
        catch (Exception exception) when (
            exception is ArgumentException
                or IOException
                or NotSupportedException)
        {
            // TextChanged fires while the user is still editing. Keep the last
            // valid library until the UI's storage validation reports the path.
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
        CancellationToken cancellationToken,
        string accountProfileId = "default")
    {
        var json = JsonSerializer.Serialize(new { username, password });
        return RunWorkerAsync(
            ["-m", "broadcastify_cli.worker", "authenticate"],
            json,
            onMessage,
            cancellationToken,
            AccountProfileEnvironment(accountProfileId));
    }

    public async Task<JobRunResult?> RunJobAsync(
        JobRequest request,
        Action<JsonElement> onMessage,
        CancellationToken cancellationToken,
        string accountProfileId = "default")
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
            cancellationToken,
            AccountProfileEnvironment(accountProfileId));
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

    public async Task<LibraryResumePlan> GetLibraryResumePlanAsync(
        string outputDirectory,
        CancellationToken cancellationToken,
        string feedId = "",
        string startDate = "",
        string endDate = "",
        bool throughCurrent = false)
    {
        LibraryResumePlan? result = null;
        var arguments = new List<string>
        {
            "-m", "broadcastify_cli.worker", "library-resume-plan",
            "--output-dir", string.IsNullOrWhiteSpace(outputDirectory)
                ? "archives"
                : outputDirectory,
        };
        if (!string.IsNullOrWhiteSpace(feedId))
        {
            arguments.AddRange(["--feed-id", feedId, "--start-date", startDate]);
            if (throughCurrent)
            {
                arguments.Add("--through-current");
            }
            else
            {
                arguments.AddRange(["--end-date", endDate]);
            }
        }
        await RunWorkerAsync(
            arguments,
            null,
            message =>
            {
                if (message.TryGetProperty("type", out var type)
                    && type.GetString() == "library_resume_plan")
                {
                    result = message.Deserialize<LibraryResumePlan>(JsonOptions);
                }
            },
            cancellationToken);
        return result ?? new LibraryResumePlan();
    }

    public async Task SaveLibraryCatchUpAsync(
        string feedId,
        string feedName,
        string startDate,
        bool throughCurrent,
        CancellationToken cancellationToken)
    {
        var saved = false;
        await RunWorkerAsync(
            ["-m", "broadcastify_cli.worker", "save-library-catch-up"],
            JsonSerializer.Serialize(
                new
                {
                    feed_id = feedId,
                    feed_name = feedName,
                    start_date = startDate,
                    through_current = throughCurrent,
                },
                JsonOptions),
            message =>
            {
                if (message.TryGetProperty("type", out var type)
                    && type.GetString() == "library_catch_up_saved")
                {
                    saved = true;
                }
            },
            cancellationToken);
        if (!saved)
        {
            throw new InvalidOperationException(
                "The worker did not confirm the saved catch-up range.");
        }
    }

    public async Task<bool> DeleteLibraryCatchUpAsync(
        string feedId,
        CancellationToken cancellationToken)
    {
        var deleted = false;
        await RunWorkerAsync(
            ["-m", "broadcastify_cli.worker", "delete-library-catch-up"],
            JsonSerializer.Serialize(new { feed_id = feedId }, JsonOptions),
            message =>
            {
                if (message.TryGetProperty("type", out var type)
                    && type.GetString() == "library_catch_up_deleted"
                    && message.TryGetProperty("deleted", out var value))
                {
                    deleted = value.GetBoolean();
                }
            },
            cancellationToken);
        return deleted;
    }

    public async Task<IReadOnlyList<string>> FinalizeLibraryCatchUpsAsync(
        string outputDirectory,
        CancellationToken cancellationToken)
    {
        List<string>? feedIds = null;
        await RunWorkerAsync(
            [
                "-m", "broadcastify_cli.worker", "finalize-library-catch-ups",
                "--output-dir", string.IsNullOrWhiteSpace(outputDirectory)
                    ? "archives"
                    : outputDirectory,
            ],
            null,
            message =>
            {
                if (message.TryGetProperty("type", out var type)
                    && type.GetString() == "library_catch_ups_finalized"
                    && message.TryGetProperty("feed_ids", out var value))
                {
                    feedIds = value.Deserialize<List<string>>(JsonOptions);
                }
            },
            cancellationToken);
        return feedIds ?? [];
    }

    public async Task<LibraryFeedDeleteResult?> DeleteLibraryFeedAsync(
        string outputDirectory,
        string feedId,
        bool removeSchedule,
        CancellationToken cancellationToken)
    {
        LibraryFeedDeleteResult? result = null;
        await RunWorkerAsync(
            ["-m", "broadcastify_cli.worker", "delete-library-feed"],
            JsonSerializer.Serialize(
                new
                {
                    output_dir = string.IsNullOrWhiteSpace(outputDirectory)
                        ? "archives"
                        : outputDirectory,
                    feed_id = feedId,
                    remove_schedule = removeSchedule,
                },
                JsonOptions),
            message =>
            {
                if (message.TryGetProperty("type", out var type)
                    && type.GetString() == "library_feed_deleted"
                    && message.TryGetProperty("result", out var value))
                {
                    result = value.Deserialize<LibraryFeedDeleteResult>(JsonOptions);
                }
            },
            cancellationToken);
        return result;
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

    public async Task<ArchiveQuestionCoverage?> GetArchiveQuestionCoverageAsync(
        string outputDirectory,
        string feedId,
        string startDate,
        string endDate,
        CancellationToken cancellationToken)
    {
        ArchiveQuestionCoverage? coverage = null;
        await RunWorkerAsync(
            [
                "-m", "broadcastify_cli.worker", "question-coverage",
                "--output-dir", string.IsNullOrWhiteSpace(outputDirectory)
                    ? "archives"
                    : outputDirectory,
                "--feed-id", feedId,
                "--start-date", startDate,
                "--end-date", endDate,
            ],
            null,
            message =>
            {
                if (message.TryGetProperty("type", out var type)
                    && type.GetString() == "question_coverage")
                {
                    coverage = message.GetProperty("coverage")
                        .Deserialize<ArchiveQuestionCoverage>(JsonOptions);
                }
            },
            cancellationToken);
        return coverage;
    }

    public async Task<ArchiveQuestionCoverage?> GetEntireFeedQuestionCoverageAsync(
        string outputDirectory,
        string feedId,
        CancellationToken cancellationToken)
    {
        ArchiveQuestionCoverage? coverage = null;
        await RunWorkerAsync(
            [
                "-m", "broadcastify_cli.worker", "question-coverage",
                "--output-dir", string.IsNullOrWhiteSpace(outputDirectory)
                    ? "archives"
                    : outputDirectory,
                "--feed-id", feedId,
                "--entire-feed",
            ],
            null,
            message =>
            {
                if (message.TryGetProperty("type", out var type)
                    && type.GetString() == "question_coverage")
                {
                    coverage = message.GetProperty("coverage")
                        .Deserialize<ArchiveQuestionCoverage>(JsonOptions);
                }
            },
            cancellationToken);
        return coverage;
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
        CancellationToken cancellationToken,
        string accountProfileId = "default")
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
            cancellationToken,
            AccountProfileEnvironment(accountProfileId));
        return status;
    }

    public async Task<CoordinatedActivityStatus?> GetCoordinatedActivityStatusAsync(
        CancellationToken cancellationToken,
        IReadOnlyList<string>? peerUrls = null,
        bool discoveryEnabled = true)
    {
        CoordinatedActivityStatus? status = null;
        await RunWorkerAsync(
            ["-m", "broadcastify_cli.worker", "coordinated-status"],
            JsonSerializer.Serialize(
                new
                {
                    peer_urls = peerUrls ?? [],
                    discovery_enabled = discoveryEnabled,
                },
                JsonOptions),
            message =>
            {
                if (message.TryGetProperty("type", out var type)
                    && type.GetString() == "coordinated_activity_status"
                    && message.TryGetProperty("status", out var value))
                {
                    status = value.Deserialize<CoordinatedActivityStatus>(JsonOptions);
                }
            },
            cancellationToken);
        return status;
    }

    public IReadOnlyList<string> AvailableAccountProfileIds()
    {
        var ids = CredentialStore.ListBroadcastifyProfiles()
            .Select(value => value.Id)
            .ToList();
        var configuredProfiles = ReadPrivateSetting(
            "BROADCASTIFY_ACCOUNT_PROFILES");
        if (!string.IsNullOrWhiteSpace(configuredProfiles))
        {
            foreach (var value in configuredProfiles.Split(
                         new[] { ',', ';', ' ', '\t', '\r', '\n' },
                         StringSplitOptions.RemoveEmptyEntries
                             | StringSplitOptions.TrimEntries))
            {
                var profileId = CredentialStore.NormalizeProfileId(value);
                if (PrivateAccountProfileConfigured(profileId)
                    || File.Exists(AccountCookiePath(profileId)))
                {
                    ids.Add(profileId);
                }
            }
        }
        if (!ids.Contains("default", StringComparer.OrdinalIgnoreCase))
        {
            ids.Insert(0, "default");
        }
        return ids.Distinct(StringComparer.OrdinalIgnoreCase).ToList();
    }

    public bool AuthorizedAccountPoolEnabled =>
        ReadPrivateBooleanSetting("BROADCASTIFY_AUTHORIZED_ACCOUNT_POOL");

    public bool HasConfiguredAccountCredentials =>
        CredentialStore.ListBroadcastifyProfiles().Count > 0
        || AvailableAccountProfileIds().Any(profileId =>
            PrivateAccountProfileConfigured(profileId)
            || File.Exists(AccountCookiePath(profileId)));

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

    public async Task<int> RecoverFeedSchedulesAsync(
        CancellationToken cancellationToken)
    {
        var recovered = 0;
        await RunWorkerAsync(
            ["-m", "broadcastify_cli.worker", "recover-schedules"],
            null,
            message =>
            {
                if (message.TryGetProperty("type", out var type)
                    && type.GetString() == "feed_schedules_recovered"
                    && message.TryGetProperty("recovered", out var value))
                {
                    recovered = value.GetInt32();
                }
            },
            cancellationToken);
        return recovered;
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

    public async Task<ManagedRuntimeStatus?> GetManagedRuntimeStatusAsync(
        string profile,
        CancellationToken cancellationToken)
    {
        ManagedRuntimeStatus? result = null;
        await RunWorkerAsync(
            [
                "-m", "broadcastify_cli.worker", "managed-runtime-status",
                "--profile", profile,
            ],
            null,
            message =>
            {
                if (message.TryGetProperty("type", out var type)
                    && type.GetString() == "managed_runtime_status"
                    && message.TryGetProperty("result", out var value))
                {
                    result = value.Deserialize<ManagedRuntimeStatus>(JsonOptions);
                }
            },
            cancellationToken);
        return result;
    }

    public async Task<ManagedRuntimeStatus?> InstallManagedRuntimeAsync(
        string profile,
        Action<JsonElement> onMessage,
        CancellationToken cancellationToken)
    {
        ManagedRuntimeStatus? result = null;
        await RunWorkerAsync(
            ["-m", "broadcastify_cli.worker", "install-managed-runtime"],
            JsonSerializer.Serialize(new { profile }, JsonOptions),
            message =>
            {
                if (message.TryGetProperty("type", out var type)
                    && type.GetString() == "managed_runtime_installed"
                    && message.TryGetProperty("result", out var value))
                {
                    result = value.Deserialize<ManagedRuntimeStatus>(JsonOptions);
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
            await process.StandardInput.WriteAsync(
                stdin.AsMemory(), cancellationToken).ConfigureAwait(false);
        }
        process.StandardInput.Close();

        var stderrTask = process.StandardError.ReadToEndAsync(cancellationToken);
        var workerError = await Task.Run(async () =>
        {
            string? reportedError = null;
            while (await process.StandardOutput.ReadLineAsync(
                       cancellationToken).ConfigureAwait(false) is { } line)
            {
                if (string.IsNullOrWhiteSpace(line))
                {
                    continue;
                }
                using var document = JsonDocument.Parse(line);
                var message = document.RootElement.Clone();
                if (message.TryGetProperty("type", out var type)
                    && type.GetString() == "error")
                {
                    reportedError = message.TryGetProperty(
                        "message", out var value)
                        ? value.GetString()
                        : "Python worker failed.";
                }
                onMessage(message);
            }
            return reportedError;
        }, CancellationToken.None).ConfigureAwait(false);

        await process.WaitForExitAsync(cancellationToken).ConfigureAwait(false);
        var stderr = await stderrTask.ConfigureAwait(false);
        if (process.ExitCode != 0)
        {
            throw new InvalidOperationException(
                workerError ?? (string.IsNullOrWhiteSpace(stderr) ? "Python worker failed." : stderr.Trim()));
        }
    }

    private ProcessStartInfo CreateStartInfo(
        IReadOnlyList<string> arguments,
        IReadOnlyDictionary<string, string>? environment = null,
        bool redirectStreams = true,
        int? coordinatorPort = null)
    {
        var startInfo = new ProcessStartInfo
        {
            FileName = _python.FileName,
            WorkingDirectory = WorkingDirectory,
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
        startInfo.Environment["BROADCASTIFY_DESKTOP_MASTER"] = "true";
        startInfo.Environment["BROADCASTIFY_LAN_ROLE"] = "master";
        var configuredLanNodePort = coordinatorPort ?? Volatile.Read(ref _lanNodePort);
        if (configuredLanNodePort is < 1024 or > 65535)
        {
            configuredLanNodePort = 8766;
        }
        var localLanNodeUrl = $"http://127.0.0.1:{configuredLanNodePort}";
        startInfo.Environment["BROADCASTIFY_LAN_MASTER_URL"] = localLanNodeUrl;
        startInfo.Environment["BROADCASTIFY_LAN_COORDINATOR"] = localLanNodeUrl;
        startInfo.Environment["BROADCASTIFY_LAN_QUOTA_COORDINATOR"] = localLanNodeUrl;
        if (IsBundledRuntime)
        {
            startInfo.Environment["PYTHONNOUSERSITE"] = "1";
            startInfo.Environment["PYTHONDONTWRITEBYTECODE"] = "1";
            var toolsDirectory = Path.Combine(
                AppContext.BaseDirectory,
                "runtime",
                "tools");
            var ffmpeg = Path.Combine(toolsDirectory, "ffmpeg.exe");
            if (File.Exists(ffmpeg))
            {
                var inheritedPath = startInfo.Environment.TryGetValue(
                    "PATH",
                    out var currentPath)
                    ? currentPath ?? ""
                    : "";
                startInfo.Environment["FFMPEG_PATH"] = ffmpeg;
                startInfo.Environment["PATH"] =
                    toolsDirectory + Path.PathSeparator
                    + inheritedPath;
            }
        }
        Directory.CreateDirectory(AppSettingsStore.LocalDataDirectory);
        if (HasBundledManagedRuntime)
        {
            startInfo.Environment["BROADCASTIFY_MANAGED_RUNTIME_ROOT"] =
                ManagedRuntimeRoot;
            startInfo.Environment["BROADCASTIFY_MANAGED_RUNTIME_MANIFEST"] =
                BundledManagedRuntimeManifestPath;
            var inheritedPythonPath = startInfo.Environment.TryGetValue(
                "PYTHONPATH",
                out var currentPythonPath)
                ? currentPythonPath ?? ""
                : "";
            startInfo.Environment["PYTHONPATH"] = string.IsNullOrWhiteSpace(
                inheritedPythonPath)
                ? BundledWorkerWheelPath
                : BundledWorkerWheelPath + Path.PathSeparator + inheritedPythonPath;
        }
        var libraryDirectory = Volatile.Read(ref _libraryDirectory);
        startInfo.Environment["BROADCASTIFY_QUOTA_LEDGER"] = Path.Combine(
            AppSettingsStore.LocalDataDirectory,
            "archive-quota.sqlite3");
        startInfo.Environment["BROADCASTIFY_LIBRARY_ROOT"] = libraryDirectory;
        startInfo.Environment["BROADCASTIFY_SECURE_ANALYSIS_DB"] = Path.Combine(
            libraryDirectory,
            "broadcastify-analysis.sqlite3");
        foreach (var argument in _python.PrefixArguments.Concat(arguments))
        {
            startInfo.ArgumentList.Add(argument);
        }
        if (HasBundledEnvironment)
        {
            startInfo.Environment["BROADCASTIFY_ENV_FILE"] = BundledEnvironmentPath;
        }
        var requestedProfileId = "default";
        if (environment is not null
            && environment.TryGetValue(
                "BROADCASTIFY_ACCOUNT_PROFILE",
                out var configuredProfileId)
            && !string.IsNullOrWhiteSpace(configuredProfileId))
        {
            requestedProfileId = CredentialStore.NormalizeProfileId(
                configuredProfileId);
        }
        var savedProfile = CredentialStore.TryLoadBroadcastifyProfile(
            requestedProfileId);
        startInfo.Environment.Remove("BROADCASTIFY_SECURE_USERNAME");
        startInfo.Environment.Remove("BROADCASTIFY_SECURE_PASSWORD");
        if (savedProfile is not null)
        {
            startInfo.Environment["BROADCASTIFY_SECURE_USERNAME"] =
                savedProfile.Username;
            startInfo.Environment["BROADCASTIFY_SECURE_PASSWORD"] =
                savedProfile.Password;
        }
        var savedHuggingFaceToken = CredentialStore.TryLoadHuggingFaceToken();
        if (savedHuggingFaceToken is not null)
        {
            startInfo.Environment["HUGGINGFACE_SECURE_TOKEN"] =
                savedHuggingFaceToken.Secret;
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

    private static IReadOnlyDictionary<string, string> AccountProfileEnvironment(
        string accountProfileId)
    {
        var profileId = CredentialStore.NormalizeProfileId(accountProfileId);
        return new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            ["BROADCASTIFY_ACCOUNT_PROFILE"] = profileId,
            ["BROADCASTIFY_COOKIE_PATH"] = AccountCookiePath(profileId),
            ["BROADCASTIFY_GLOBAL_REQUEST_SPACING_SECONDS"] = "5",
        };
    }

    private static string AccountCookiePath(string profileId)
    {
        return profileId == "default"
            ? Path.Combine(AppSettingsStore.LocalDataDirectory, "cookies.json")
            : Path.Combine(
                AppSettingsStore.LocalDataDirectory,
                "account-sessions",
                $"{profileId}.json");
    }

    private bool PrivateAccountProfileConfigured(string profileId)
    {
        if (profileId == "default")
        {
            var username = ReadPrivateSetting("BROADCASTIFY_USERNAME")
                ?? ReadPrivateSetting("USERNAME");
            var password = ReadPrivateSetting("BROADCASTIFY_PASSWORD")
                ?? ReadPrivateSetting("PASSWORD");
            return !string.IsNullOrWhiteSpace(username)
                && !string.IsNullOrWhiteSpace(password);
        }
        var suffix = new string(profileId
            .ToUpperInvariant()
            .Select(value => char.IsLetterOrDigit(value) ? value : '_')
            .ToArray());
        return !string.IsNullOrWhiteSpace(ReadPrivateSetting(
                $"BROADCASTIFY_ACCOUNT_{suffix}_USERNAME"))
            && !string.IsNullOrWhiteSpace(ReadPrivateSetting(
                $"BROADCASTIFY_ACCOUNT_{suffix}_PASSWORD"));
    }

    private bool ReadPrivateBooleanSetting(string name)
    {
        return TryBoolean(ReadPrivateSetting(name), out var enabled) && enabled;
    }

    private string? ReadPrivateSetting(string name)
    {
        var inherited = Environment.GetEnvironmentVariable(name);
        if (!string.IsNullOrWhiteSpace(inherited))
        {
            return inherited.Trim();
        }
        foreach (var path in new[]
        {
            Path.Combine(RepositoryRoot, ".env"),
            Path.Combine(RepositoryRoot, ".env.accounts"),
            Path.Combine(WorkingDirectory, ".env.accounts"),
            BundledEnvironmentPath,
        }.Distinct(StringComparer.OrdinalIgnoreCase))
        {
            if (!File.Exists(path))
            {
                continue;
            }
            try
            {
                foreach (var line in File.ReadLines(path))
                {
                    var trimmed = line.Trim();
                    if (trimmed.StartsWith('#') || !trimmed.Contains('='))
                    {
                        continue;
                    }
                    var separator = trimmed.IndexOf('=');
                    if (!trimmed[..separator].Trim().Equals(
                            name,
                            StringComparison.OrdinalIgnoreCase))
                    {
                        continue;
                    }
                    var value = trimmed[(separator + 1)..].Trim().Trim('"', '\'');
                    return value;
                }
            }
            catch (IOException)
            {
                // A worker can still use its default single account if a
                // private environment file is momentarily unavailable.
            }
        }
        return null;
    }

    private static bool TryBoolean(string? value, out bool result)
    {
        switch (value?.Trim().ToLowerInvariant())
        {
            case "1":
            case "true":
            case "yes":
            case "on":
                result = true;
                return true;
            case "0":
            case "false":
            case "no":
            case "off":
                result = false;
                return true;
            default:
                result = false;
                return false;
        }
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

    private static PythonResolution ResolvePython(
        string repositoryRoot,
        string workingDirectory,
        string? bundledPython,
        string? preferredPython,
        string? workerWheel)
    {
        var warnings = new List<string>();
        var configured = Environment.GetEnvironmentVariable(
            "BROADCASTIFY_PYTHON");
        var candidates = new List<PythonCommand>();
        AddConfiguredCandidate(
            candidates,
            warnings,
            preferredPython,
            workingDirectory,
            "saved Python runtime",
            workerWheel);
        AddConfiguredCandidate(
            candidates,
            warnings,
            configured,
            workingDirectory,
            "BROADCASTIFY_PYTHON",
            workerWheel);
        if (!string.IsNullOrWhiteSpace(bundledPython))
        {
            candidates.Add(new PythonCommand(
                bundledPython,
                [],
                "bundled Python 3.12"));
        }
        candidates.Add(new PythonCommand(
            Path.Combine(repositoryRoot, ".venv", "Scripts", "python.exe"), [], ".venv Python"));
        candidates.Add(new PythonCommand("python.exe", [], "python.exe"));
        candidates.Add(new PythonCommand("py.exe", ["-3"], "py.exe -3"));

        foreach (var candidate in candidates)
        {
            if (CanRunWorker(candidate, workingDirectory, workerWheel))
            {
                return new PythonResolution(
                    candidate,
                    warnings.Count == 0
                        ? null
                        : string.Join(" ", warnings));
            }
        }
        throw new FileNotFoundException(
            "No Python runtime capable of importing broadcastify_cli was found. "
            + "Use the bundled runtime, select a prepared Python environment in "
            + "Settings, or set BROADCASTIFY_PYTHON.");
    }

    private static void AddConfiguredCandidate(
        ICollection<PythonCommand> candidates,
        ICollection<string> warnings,
        string? value,
        string workingDirectory,
        string source,
        string? workerWheel)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return;
        }
        try
        {
            var configured = value.Trim();
            var path = Path.IsPathFullyQualified(configured)
                || configured.Contains(Path.DirectorySeparatorChar)
                || configured.Contains(Path.AltDirectorySeparatorChar)
                    ? Path.GetFullPath(configured, workingDirectory)
                    : configured;
            var candidate = new PythonCommand(
                path,
                [],
                $"{source}: {path}");
            if (CanRunWorker(candidate, workingDirectory, workerWheel))
            {
                candidates.Add(candidate);
            }
            else
            {
                warnings.Add(
                    $"The {source} could not import broadcastify_cli; "
                    + "the next compatible runtime was used.");
            }
        }
        catch (Exception exception) when (
            exception is ArgumentException
                or IOException
                or NotSupportedException)
        {
            warnings.Add(
                $"The {source} path is invalid; the next compatible runtime was used.");
        }
    }

    private static bool CanRunWorker(
        PythonCommand candidate,
        string workingDirectory,
        string? workerWheel)
    {
        try
        {
            using var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = candidate.FileName,
                    WorkingDirectory = workingDirectory,
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
            if (!string.IsNullOrWhiteSpace(workerWheel)
                && File.Exists(workerWheel))
            {
                process.StartInfo.Environment["PYTHONPATH"] = workerWheel;
            }
            process.StartInfo.ArgumentList.Add("-c");
            process.StartInfo.ArgumentList.Add("import broadcastify_cli");
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

    private sealed record PythonResolution(
        PythonCommand Command,
        string? Warning);
}
