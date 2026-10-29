using System.ComponentModel;
using System.Diagnostics;

namespace BroadcastifyCli.WinUI;

internal static class DesktopProcessSupervisor
{
    internal static int Run(
        string executable,
        string arguments,
        TimeSpan retryDelay,
        int maxRestarts = 3)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(executable);
        ArgumentNullException.ThrowIfNull(arguments);
        if (retryDelay < TimeSpan.Zero)
        {
            throw new ArgumentOutOfRangeException(nameof(retryDelay));
        }
        if (maxRestarts < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxRestarts));
        }

        executable = Path.GetFullPath(executable);
        var startInfo = new ProcessStartInfo
        {
            FileName = executable,
            Arguments = arguments,
            UseShellExecute = false,
            CreateNoWindow = true,
            WorkingDirectory = Path.GetDirectoryName(executable)
                ?? Environment.CurrentDirectory,
        };

        var maxAttempts = maxRestarts + 1;
        var lastExitCode = 1;
        for (var attempt = 1; attempt <= maxAttempts; attempt++)
        {
            Process? child;
            try
            {
                child = Process.Start(startInfo);
                if (child is null)
                {
                    throw new InvalidOperationException(
                        "Windows did not return a process for the desktop session.");
                }
            }
            catch (Win32Exception exception)
            {
                lastExitCode = 1;
                AppDiagnostics.AppendLifecycle(
                    $"desktop supervisor attempt {attempt}/{maxAttempts} could not start the UI: {exception.Message}");
                if (attempt < maxAttempts)
                {
                    Thread.Sleep(retryDelay);
                }
                continue;
            }
            catch (InvalidOperationException exception)
            {
                lastExitCode = 1;
                AppDiagnostics.AppendLifecycle(
                    $"desktop supervisor attempt {attempt}/{maxAttempts} could not start the UI: {exception.Message}");
                if (attempt < maxAttempts)
                {
                    Thread.Sleep(retryDelay);
                }
                continue;
            }

            using (child)
            {
                var processId = child.Id;
                AppDiagnostics.AppendLifecycle(
                    $"desktop supervisor attempt {attempt}/{maxAttempts} started UI PID {processId}");
                child.WaitForExit();
                lastExitCode = child.ExitCode;
                if (lastExitCode == 0)
                {
                    AppDiagnostics.AppendLifecycle(
                        $"desktop supervisor UI PID {processId} exited normally");
                    return 0;
                }

                AppDiagnostics.AppendLifecycle(
                    $"desktop supervisor UI PID {processId} exited with code {lastExitCode} on attempt {attempt}/{maxAttempts}");
            }

            if (attempt < maxAttempts)
            {
                Thread.Sleep(retryDelay);
            }
        }

        return lastExitCode;
    }

    internal static int RunDesktop()
    {
        var executable = Environment.ProcessPath
            ?? throw new InvalidOperationException(
                "Windows could not determine the Broadcastify Desktop executable path.");
        return Run(
            executable,
            "--desktop-session --startup --prompt-setup",
            TimeSpan.FromMinutes(1),
            maxRestarts: 3);
    }
}
