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

    internal static string? FindPreviousLibraryDirectory(
        string relativeOutputDirectory,
        string currentCandidate)
    {
        if (string.IsNullOrWhiteSpace(relativeOutputDirectory)
            || Path.IsPathRooted(relativeOutputDirectory))
        {
            return null;
        }
        var logs = new[]
        {
            ActivityLogPath,
            Path.Combine(AppSettingsStore.LocalDataDirectory, "activity.previous.log"),
        };
        foreach (var log in logs)
        {
            string[] lines;
            try
            {
                if (!File.Exists(log))
                {
                    continue;
                }
                lines = File.ReadAllLines(log, Encoding.UTF8);
            }
            catch (Exception exception) when (
                exception is IOException
                    or UnauthorizedAccessException)
            {
                continue;
            }
            for (var index = lines.Length - 1; index >= 0; index--)
            {
                const string marker = "Repository: ";
                var markerIndex = lines[index].IndexOf(
                    marker,
                    StringComparison.Ordinal);
                if (markerIndex < 0)
                {
                    continue;
                }
                try
                {
                    var previousRoot = lines[index][
                        (markerIndex + marker.Length)..].Trim();
                    var candidate = Path.GetFullPath(
                        relativeOutputDirectory,
                        previousRoot);
                    if (candidate.Equals(
                        Path.GetFullPath(currentCandidate),
                        StringComparison.OrdinalIgnoreCase))
                    {
                        continue;
                    }
                    if (LooksLikeArchiveLibrary(candidate))
                    {
                        return candidate;
                    }
                }
                catch (Exception exception) when (
                    exception is ArgumentException
                        or IOException
                        or NotSupportedException
                        or UnauthorizedAccessException)
                {
                    // Ignore malformed or no-longer-accessible historical paths.
                }
            }
        }
        return null;
    }

    private static bool LooksLikeArchiveLibrary(string directory)
    {
        try
        {
            if (!Directory.Exists(directory))
            {
                return false;
            }
            if (File.Exists(Path.Combine(
                directory,
                "broadcastify-analysis.sqlite3")))
            {
                return true;
            }
            return Directory.EnumerateDirectories(directory)
                .Select(Path.GetFileName)
                .Any(name => !string.IsNullOrWhiteSpace(name)
                    && name.All(char.IsDigit));
        }
        catch (Exception exception) when (
            exception is IOException
                or UnauthorizedAccessException)
        {
            return false;
        }
    }

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
