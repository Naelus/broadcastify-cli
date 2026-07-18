using System.Diagnostics;
using System.Text;

namespace BroadcastifyCli.WinUI;

internal static class AppDiagnostics
{
    private static readonly object LogLock = new();
    private const long MaximumActivityLogBytes = 5 * 1024 * 1024;

    internal static string ActivityLogPath =>
        Path.Combine(AppSettingsStore.LocalDataDirectory, "activity.log");

    internal static string CrashLogPath =>
        Path.Combine(AppSettingsStore.LocalDataDirectory, "crash.log");

    internal static void AppendActivity(string message)
    {
        try
        {
            lock (LogLock)
            {
                Directory.CreateDirectory(AppSettingsStore.LocalDataDirectory);
                RotateActivityLogIfNeeded();
                File.AppendAllText(
                    ActivityLogPath,
                    $"[{DateTimeOffset.Now:O}] {message}{Environment.NewLine}",
                    Encoding.UTF8);
            }
        }
        catch
        {
            // Diagnostics must never interrupt an archive or processing job.
        }
    }

    internal static void AppendCrash(Exception exception, string source)
    {
        try
        {
            lock (LogLock)
            {
                Directory.CreateDirectory(AppSettingsStore.LocalDataDirectory);
                var entry =
                    $"{Environment.NewLine}=== {DateTimeOffset.Now:O} ==={Environment.NewLine}"
                    + $"Source: {source}{Environment.NewLine}"
                    + $"Process: {Environment.ProcessPath} ({Environment.ProcessId}){Environment.NewLine}"
                    + $"Runtime: {Environment.Version}{Environment.NewLine}"
                    + exception
                    + Environment.NewLine;
                File.AppendAllText(CrashLogPath, entry, Encoding.UTF8);
                File.AppendAllText(
                    Path.Combine(Path.GetTempPath(), "BroadcastifyDesktop-startup.log"),
                    entry,
                    Encoding.UTF8);
            }
        }
        catch
        {
            // Never hide the original failure with a diagnostic failure.
        }
    }

    private static void RotateActivityLogIfNeeded()
    {
        var path = new FileInfo(ActivityLogPath);
        if (!path.Exists || path.Length < MaximumActivityLogBytes)
        {
            return;
        }
        var previous = Path.Combine(AppSettingsStore.LocalDataDirectory, "activity.previous.log");
        File.Move(ActivityLogPath, previous, overwrite: true);
    }
}
