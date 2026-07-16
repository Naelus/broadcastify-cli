using System.Text.Json;

namespace BroadcastifyCli.WinUI;

internal sealed record DesktopSettings
{
    public int Version { get; init; } = 1;
    public string HardwareProfile { get; init; } = "auto";
    public string WhisperModel { get; init; } = "turbo";
    public string AsrEngine { get; init; } = "auto";
    public string TranscriptionDevice { get; init; } = "auto";
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
    public string AnalysisModel { get; init; } = "ggml-org/gemma-4-12B-it-GGUF:Q4_K_M";
    public string AnalysisEndpoint { get; init; } = "";
    public string AnalysisApiKeyEnvironment { get; init; } = "OPENAI_API_KEY";
    public string CodexCliPath { get; init; } = "";
    public bool AllowExternalAnalysis { get; init; }
    public bool RememberAnalysisApiKey { get; init; }
}

internal static class AppSettingsStore
{
    private static readonly JsonSerializerOptions SerializerOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        WriteIndented = true,
    };

    private static string SettingsPath => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "Broadcastify Desktop",
        "settings.json");

    public static DesktopSettings Load()
    {
        try
        {
            return File.Exists(SettingsPath)
                ? JsonSerializer.Deserialize<DesktopSettings>(
                    File.ReadAllText(SettingsPath), SerializerOptions) ?? new DesktopSettings()
                : new DesktopSettings();
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
            File.WriteAllText(
                temporaryPath,
                JsonSerializer.Serialize(settings, SerializerOptions));
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
