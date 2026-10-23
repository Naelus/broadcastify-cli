using System.Diagnostics;

namespace BroadcastifyCli.WinUI;

internal sealed record PreviewLaunchResult(string DataRoot, int ProcessId);

internal static class PreviewLauncher
{
    private const string PreviewFolderName = "Broadcastify Desktop Preview";

    public static async Task<PreviewLaunchResult> LaunchAsync(
        string? configuredPython,
        CancellationToken cancellationToken = default)
    {
        var applicationDirectory = AppContext.BaseDirectory;
        var python = ResolvePython(applicationDirectory, configuredPython, out var sourceRoot);
        var ffmpeg = ResolveFfmpeg(applicationDirectory);
        var tempParent = Path.Combine(Path.GetTempPath(), PreviewFolderName);
        Directory.CreateDirectory(tempParent);
        var previewRoot = Path.Combine(tempParent, Guid.NewGuid().ToString("N"));
        var dataRoot = Path.Combine(previewRoot, "app-data");
        var libraryRoot = Path.Combine(previewRoot, "library");
        var database = Path.Combine(libraryRoot, "broadcastify-analysis.sqlite3");
        var quotaLedger = Path.Combine(dataRoot, "archive-quota.sqlite3");

        var seed = CreateIsolatedStartInfo(
            python,
            dataRoot,
            libraryRoot,
            database,
            quotaLedger,
            applicationDirectory,
            sourceRoot,
            preview: false);
        seed.ArgumentList.Add("-m");
        seed.ArgumentList.Add("broadcastify_cli.preview");
        seed.ArgumentList.Add("--root");
        seed.ArgumentList.Add(previewRoot);
        seed.ArgumentList.Add("--ffmpeg");
        seed.ArgumentList.Add(ffmpeg);
        seed.ArgumentList.Add("--python");
        seed.ArgumentList.Add(python);

        await RunSeederAsync(seed, cancellationToken).ConfigureAwait(false);

        var preview = CreateIsolatedStartInfo(
            Path.Combine(applicationDirectory, "Broadcastify Desktop.exe"),
            dataRoot,
            libraryRoot,
            database,
            quotaLedger,
            applicationDirectory,
            sourceRoot,
            preview: true);
        preview.ArgumentList.Add("--workflow-preview");
        using var process = Process.Start(preview)
            ?? throw new InvalidOperationException("The isolated preview window could not be started.");
        return new PreviewLaunchResult(dataRoot, process.Id);
    }

    private static async Task RunSeederAsync(
        ProcessStartInfo startInfo,
        CancellationToken cancellationToken)
    {
        using var process = Process.Start(startInfo)
            ?? throw new InvalidOperationException("The isolated preview sample data could not be prepared.");
        var standardOutput = process.StandardOutput.ReadToEndAsync(cancellationToken);
        var standardError = process.StandardError.ReadToEndAsync(cancellationToken);
        await process.WaitForExitAsync(cancellationToken).ConfigureAwait(false);
        var output = await standardOutput.ConfigureAwait(false);
        var error = await standardError.ConfigureAwait(false);
        if (process.ExitCode != 0)
        {
            var detail = string.Join(Environment.NewLine, new[] { error.Trim(), output.Trim() }
                .Where(value => value.Length > 0));
            throw new InvalidOperationException(
                "Could not prepare isolated preview data. "
                + (detail.Length == 0 ? "The bundled worker did not report a reason." : detail));
        }
    }

    private static ProcessStartInfo CreateIsolatedStartInfo(
        string executable,
        string dataRoot,
        string libraryRoot,
        string database,
        string quotaLedger,
        string applicationDirectory,
        string? sourceRoot,
        bool preview)
    {
        var startInfo = new ProcessStartInfo
        {
            FileName = executable,
            WorkingDirectory = sourceRoot ?? applicationDirectory,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = executable.EndsWith("python.exe", StringComparison.OrdinalIgnoreCase),
            RedirectStandardError = executable.EndsWith("python.exe", StringComparison.OrdinalIgnoreCase),
        };
        if (startInfo.RedirectStandardOutput)
        {
            startInfo.StandardOutputEncoding = new System.Text.UTF8Encoding(
                encoderShouldEmitUTF8Identifier: false);
            startInfo.StandardErrorEncoding = new System.Text.UTF8Encoding(
                encoderShouldEmitUTF8Identifier: false);
        }
        RemoveInheritedSecrets(startInfo.Environment);
        startInfo.Environment[AppSettingsStore.TestDataRootEnvironment] = dataRoot;
        startInfo.Environment["BROADCASTIFY_DESKTOP_E2E_ISOLATED"] = "1";
        startInfo.Environment["BROADCASTIFY_ENV_FILE"] = "";
        startInfo.Environment["BROADCASTIFY_LIBRARY_ROOT"] = libraryRoot;
        startInfo.Environment["BROADCASTIFY_SECURE_ANALYSIS_DB"] = database;
        startInfo.Environment["BROADCASTIFY_QUOTA_LEDGER"] = quotaLedger;
        startInfo.Environment["BROADCASTIFY_CREDENTIAL_STORE"] = Path.Combine(dataRoot, "credentials.enc");
        startInfo.Environment["PYTHONUTF8"] = "1";
        startInfo.Environment["PYTHONIOENCODING"] = "utf-8";
        startInfo.Environment["PYTHONNOUSERSITE"] = "1";
        startInfo.Environment["PYTHONDONTWRITEBYTECODE"] = "1";
        if (preview)
        {
            startInfo.Environment["BROADCASTIFY_DESKTOP_PREVIEW"] = "1";
        }
        var toolsDirectory = Path.Combine(applicationDirectory, "runtime", "tools");
        var ffmpegDirectory = File.Exists(Path.Combine(toolsDirectory, "ffmpeg.exe"))
            ? toolsDirectory
            : null;
        if (ffmpegDirectory is not null)
        {
            var inheritedPath = startInfo.Environment.TryGetValue("PATH", out var currentPath)
                ? currentPath ?? ""
                : "";
            startInfo.Environment["PATH"] = ffmpegDirectory + Path.PathSeparator + inheritedPath;
        }
        return startInfo;
    }

    private static void RemoveInheritedSecrets(IDictionary<string, string?> environment)
    {
        foreach (var name in environment.Keys.ToArray())
        {
            var upper = name.ToUpperInvariant();
            if (upper.StartsWith("BROADCASTIFY_", StringComparison.Ordinal)
                || upper.StartsWith("HUGGINGFACE_", StringComparison.Ordinal)
                || upper.StartsWith("HF_", StringComparison.Ordinal)
                || upper.StartsWith("OPENAI_", StringComparison.Ordinal)
                || upper.StartsWith("CODEX_", StringComparison.Ordinal)
                || upper.StartsWith("ANTHROPIC_", StringComparison.Ordinal)
                || upper.StartsWith("GEMINI_", StringComparison.Ordinal)
                || upper is "HTTP_PROXY" or "HTTPS_PROXY" or "ALL_PROXY" or "NO_PROXY"
                || upper.Contains("API_KEY", StringComparison.Ordinal)
                || upper.EndsWith("_TOKEN", StringComparison.Ordinal)
                || upper.EndsWith("_PASSWORD", StringComparison.Ordinal)
                || upper.EndsWith("_COOKIE", StringComparison.Ordinal)
                || upper.EndsWith("_SECRET", StringComparison.Ordinal)
                || upper is "PYTHONPATH" or "PYTHONHOME" or "PYTHONUSERBASE"
                or "FFMPEG_PATH" or "FFPROBE_PATH")
            {
                environment.Remove(name);
            }
        }
    }

    private static string ResolvePython(
        string applicationDirectory,
        string? configuredPython,
        out string? sourceRoot)
    {
        var bundledPython = Path.Combine(applicationDirectory, "runtime", "python", "python.exe");
        if (File.Exists(bundledPython))
        {
            sourceRoot = null;
            return bundledPython;
        }
        sourceRoot = FindSourceRoot(applicationDirectory);
        if (!string.IsNullOrWhiteSpace(configuredPython))
        {
            var configured = Path.IsPathFullyQualified(configuredPython)
                ? configuredPython
                : Path.GetFullPath(configuredPython, sourceRoot ?? applicationDirectory);
            if (File.Exists(configured))
            {
                return configured;
            }
        }
        var sourcePython = sourceRoot is null
            ? null
            : Path.Combine(sourceRoot, ".venv", "Scripts", "python.exe");
        if (sourcePython is not null && File.Exists(sourcePython))
        {
            return sourcePython;
        }
        throw new FileNotFoundException(
            "The preview requires the installed bundled Python runtime. A source build can use its configured worker runtime.");
    }

    private static string ResolveFfmpeg(string applicationDirectory)
    {
        var bundled = Path.Combine(applicationDirectory, "runtime", "tools", "ffmpeg.exe");
        if (File.Exists(bundled))
        {
            return bundled;
        }
        var configured = Environment.GetEnvironmentVariable("FFMPEG_PATH");
        if (!string.IsNullOrWhiteSpace(configured) && File.Exists(configured))
        {
            return Path.GetFullPath(configured);
        }
        var path = Environment.GetEnvironmentVariable("PATH") ?? "";
        foreach (var directory in path.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries))
        {
            var candidate = Path.Combine(directory.Trim('"'), OperatingSystem.IsWindows() ? "ffmpeg.exe" : "ffmpeg");
            if (File.Exists(candidate))
            {
                return candidate;
            }
        }
        throw new FileNotFoundException(
            "FFmpeg is required to create playable sample audio. The installed runtime normally includes it.");
    }

    private static string? FindSourceRoot(string applicationDirectory)
    {
        for (DirectoryInfo? directory = new(applicationDirectory);
             directory is not null;
             directory = directory.Parent)
        {
            if (File.Exists(Path.Combine(directory.FullName, "broadcastify_cli", "__init__.py")))
            {
                return directory.FullName;
            }
        }
        return null;
    }
}
