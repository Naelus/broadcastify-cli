using System.Text.Json;

namespace BroadcastifyCli.WinUI;

internal sealed record DesktopSettings
{
    public int Version { get; init; } = 9;
    public string PythonRuntimePath { get; init; } = "";
    public string HardwareProfile { get; init; } = "auto";
    public string WhisperModel { get; init; } = "turbo";
    public string AsrEngine { get; init; } = "auto";
    public string TranscriptionDevice { get; init; } = "auto";
    public string DiarizationEngine { get; init; } = "community-1";
    public string DiarizationDevice { get; init; } = "auto";
    public string AsrModelPath { get; init; } = "";
    public string OutputDirectory { get; init; } = "archives";
    public int GpuIndex { get; init; }
    public int BatchSize { get; init; } = 8;
    public int MinimumSpeakers { get; init; } = 2;
    public int MaximumSpeakers { get; init; } = 8;
    public int DownloadJobs { get; init; } = 1;
    public bool Combine { get; init; } = true;
    public bool KeepOriginals { get; init; } = true;
    public bool Transcribe { get; init; } = true;
    public bool Diarize { get; init; } = true;
    public bool AnalyzeAfterJob { get; init; } = true;
    public string AnalysisProvider { get; init; } = "local";
    public string AnalysisModel { get; init; } = "ggml-org/gemma-4-12B-it-GGUF:Q4_0";
    public string AnalysisDevice { get; init; } = "auto";
    public string AnalysisEndpoint { get; init; } = "";
    public string AnalysisApiKeyEnvironment { get; init; } = "OPENAI_API_KEY";
    public string CodexCliPath { get; init; } = "";
    public bool AllowExternalAnalysis { get; init; }
    public bool RememberAnalysisApiKey { get; init; }
    public bool LanSyncEnabled { get; init; } = true;
    public bool LanDiscoveryEnabled { get; init; } = true;
    public string LanPeerUrls { get; init; } = "";
    public bool LanShareEnabled { get; init; } = true;
    public int LanSharePort { get; init; } = 8766;
    public string LastAreaProfileName { get; init; } = "";
    public string LastReviewFeedId { get; init; } = "";
    public string LastReviewDate { get; init; } = "";
    public string DesktopDockSide { get; init; } = "none";
    public double DesktopDockWidth { get; init; } =
        DesktopDockManager.RecommendedWidthDips;
    public string DesktopDockMonitor { get; init; } = "";
    public int? DesktopWindowX { get; init; }
    public int? DesktopWindowY { get; init; }
    public int? DesktopWindowWidth { get; init; }
    public int? DesktopWindowHeight { get; init; }
    public bool DesktopWindowMaximized { get; init; }
}

internal static class AppSettingsStore
{
    internal const string TestDataRootEnvironment =
        "BROADCASTIFY_DESKTOP_TEST_DATA_ROOT";

    private static readonly JsonSerializerOptions SerializerOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        WriteIndented = true,
    };

    internal static string LocalDataDirectory
    {
        get
        {
            var testRoot = Environment.GetEnvironmentVariable(
                TestDataRootEnvironment)?.Trim();
            if (!string.IsNullOrWhiteSpace(testRoot))
            {
                return Path.GetFullPath(testRoot);
            }
            var userProfile = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            return Path.Combine(userProfile, "AppData", "Local", "Broadcastify Desktop");
        }
    }

    private static bool UsesTestDataRoot => !string.IsNullOrWhiteSpace(
        Environment.GetEnvironmentVariable(TestDataRootEnvironment));

    internal static string SettingsPath => Path.Combine(LocalDataDirectory, "settings.json");

    public static DesktopSettings Load()
    {
        var legacyPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "Broadcastify Desktop",
            "settings.json");
        try
        {
            var sourcePath = File.Exists(SettingsPath)
                ? SettingsPath
                : !UsesTestDataRoot
                  && !Path.GetFullPath(legacyPath).Equals(
                    Path.GetFullPath(SettingsPath), StringComparison.OrdinalIgnoreCase)
                  && File.Exists(legacyPath)
                    ? legacyPath
                    : null;
            if (sourcePath is null)
            {
                return new DesktopSettings();
            }
            var settings = JsonSerializer.Deserialize<DesktopSettings>(
                File.ReadAllText(sourcePath), SerializerOptions) ?? new DesktopSettings();
            var needsSave = !sourcePath.Equals(
                SettingsPath,
                StringComparison.OrdinalIgnoreCase);
            if (settings.Version < 5)
            {
                // Version 5 makes a LAN-reuse client eligible to own the shared
                // upstream lease. Only original source blocks are served, and
                // the visible Settings toggle can still turn seeding back off.
                settings = settings with
                {
                    Version = 5,
                    LanShareEnabled = settings.LanSyncEnabled,
                };
                needsSave = true;
            }
            if (settings.Version < 6)
            {
                // Version 6 persists library locations as absolute paths so
                // moving from a source checkout to an installed app cannot
                // make an existing library appear empty.
                settings = settings with { Version = 6 };
                needsSave = true;
            }
            if (settings.Version < 7)
            {
                // Version 7 can select an external Python environment for
                // optional heavyweight accelerator dependencies while the
                // installer keeps its portable worker as the safe default.
                settings = settings with { Version = 7 };
                needsSave = true;
            }
            if (settings.Version < 8)
            {
                // Version 8 persists only the user's explicit left/right
                // AppBar choice and logical width. Retained archives and all
                // processing state remain in their existing locations.
                settings = settings with
                {
                    Version = 8,
                    DesktopDockSide = "none",
                    DesktopDockWidth = DesktopDockManager.RecommendedWidthDips,
                };
                needsSave = true;
            }
            if (settings.Version < 9)
            {
                // Version 9 retains the target monitor and the last floating
                // bounds independently from the pinned AppBar rectangle. An
                // upgrade never interprets a docked edge as a floating window.
                settings = settings with
                {
                    Version = 9,
                    DesktopDockMonitor = "",
                    DesktopWindowX = null,
                    DesktopWindowY = null,
                    DesktopWindowWidth = null,
                    DesktopWindowHeight = null,
                    DesktopWindowMaximized = false,
                };
                needsSave = true;
            }
            if (settings.DesktopDockSide is not ("none" or "left" or "right"))
            {
                settings = settings with { DesktopDockSide = "none" };
                needsSave = true;
            }
            if (!double.IsFinite(settings.DesktopDockWidth)
                || settings.DesktopDockWidth < DesktopDockManager.MinimumWidthDips
                || settings.DesktopDockWidth > DesktopDockManager.MaximumWidthDips)
            {
                settings = settings with
                {
                    DesktopDockWidth = double.IsFinite(settings.DesktopDockWidth)
                        ? Math.Clamp(
                            settings.DesktopDockWidth,
                            DesktopDockManager.MinimumWidthDips,
                            DesktopDockManager.MaximumWidthDips)
                        : DesktopDockManager.RecommendedWidthDips,
                };
                needsSave = true;
            }
            var hasCompleteWindowBounds =
                settings.DesktopWindowX is not null
                && settings.DesktopWindowY is not null
                && settings.DesktopWindowWidth is > 0
                && settings.DesktopWindowHeight is > 0;
            var hasAnyWindowBounds =
                settings.DesktopWindowX is not null
                || settings.DesktopWindowY is not null
                || settings.DesktopWindowWidth is not null
                || settings.DesktopWindowHeight is not null;
            if (hasAnyWindowBounds && !hasCompleteWindowBounds)
            {
                settings = settings with
                {
                    DesktopWindowX = null,
                    DesktopWindowY = null,
                    DesktopWindowWidth = null,
                    DesktopWindowHeight = null,
                    DesktopWindowMaximized = false,
                };
                needsSave = true;
            }
            var trimmedMonitor = (settings.DesktopDockMonitor ?? "").Trim();
            if (!string.Equals(
                    trimmedMonitor,
                    settings.DesktopDockMonitor,
                    StringComparison.Ordinal))
            {
                settings = settings with
                {
                    DesktopDockMonitor = trimmedMonitor,
                };
                needsSave = true;
            }
            if (needsSave)
            {
                TrySave(settings);
            }
            return settings;
        }
        catch (IOException)
        {
            return new DesktopSettings();
        }
        catch (JsonException)
        {
            return new DesktopSettings();
        }
        catch (UnauthorizedAccessException)
        {
            return new DesktopSettings();
        }
    }

    public static bool TrySave(DesktopSettings settings)
    {
        try
        {
            var directory = Path.GetDirectoryName(SettingsPath)!;
            Directory.CreateDirectory(directory);
            var temporaryPath = SettingsPath + ".tmp";
            var content = JsonSerializer.Serialize(settings, SerializerOptions);
            using (var stream = new FileStream(
                       temporaryPath,
                       FileMode.Create,
                       FileAccess.Write,
                       FileShare.None,
                       bufferSize: 4_096,
                       FileOptions.WriteThrough))
            using (var writer = new StreamWriter(stream))
            {
                writer.Write(content);
                writer.Flush();
                stream.Flush(flushToDisk: true);
            }
            File.Move(temporaryPath, SettingsPath, overwrite: true);
            return true;
        }
        catch (IOException)
        {
            return false;
        }
        catch (UnauthorizedAccessException)
        {
            return false;
        }
    }
}
