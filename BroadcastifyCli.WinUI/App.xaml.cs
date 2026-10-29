using Microsoft.UI.Xaml;

namespace BroadcastifyCli.WinUI;

public partial class App : Application
{
    private Window? _window;

    public App()
    {
        InitializeComponent();
        UnhandledException += (_, eventArgs) =>
            AppDiagnostics.AppendCrash(eventArgs.Exception, "WinUI unhandled exception");
        AppDomain.CurrentDomain.UnhandledException += (_, eventArgs) =>
        {
            if (eventArgs.ExceptionObject is Exception exception)
            {
                AppDiagnostics.AppendCrash(exception, "AppDomain unhandled exception");
            }
        };
        TaskScheduler.UnobservedTaskException += (_, eventArgs) =>
            AppDiagnostics.AppendCrash(eventArgs.Exception, "Unobserved task exception");
    }

    protected override void OnLaunched(LaunchActivatedEventArgs args)
    {
        try
        {
            var arguments = Environment.GetCommandLineArgs().Skip(1).ToArray();
            var commandLine = arguments.ToHashSet(StringComparer.OrdinalIgnoreCase);
            var isolated = !string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable(
                AppSettingsStore.TestDataRootEnvironment));
            if (!isolated && arguments.Any(argument =>
                argument.Equals("--ui-e2e-report", StringComparison.OrdinalIgnoreCase)
                || argument.StartsWith("--ui-e2e-report=", StringComparison.OrdinalIgnoreCase)))
                throw new InvalidOperationException("The UI probe requires isolated desktop data.");
            var startupConfiguration = arguments.FirstOrDefault(argument =>
                argument.StartsWith("--configure-startup=", StringComparison.OrdinalIgnoreCase));
            if (startupConfiguration is not null)
            {
                if (isolated || Environment.GetEnvironmentVariable("BROADCASTIFY_DESKTOP_PREVIEW") == "1")
                    throw new InvalidOperationException("Isolated previews and probes cannot change Windows startup.");
                try
                {
                    WindowsStartupManager.Configure(startupConfiguration.Split('=', 2)[1]);
                }
                catch (Exception exception)
                {
                    AppDiagnostics.AppendCrash(exception, "Windows startup configuration");
                    Environment.ExitCode = 1;
                }
                Exit();
                return;
            }
            var workflowPreviewRequested = commandLine.Contains("--workflow-preview");
            var workflowPreviewEnabled = Environment.GetEnvironmentVariable(
                "BROADCASTIFY_DESKTOP_PREVIEW") == "1";
            if (workflowPreviewRequested != workflowPreviewEnabled)
            {
                throw new InvalidOperationException(
                    "An isolated workflow preview must be started by the preview launcher.");
            }
            if (workflowPreviewRequested)
            {
                ValidateWorkflowPreviewEnvironment();
            }
            // A scheduled host is independent of the installer/editor process
            // tree. Windows can restart a failed host without a new sign-in.
            if (!isolated && !workflowPreviewRequested)
            {
                try
                {
                    WindowsStartupManager.Configure("preserve");
                    if (WindowsStartupManager.IsEnabled()
                        && !commandLine.Contains("--scheduled-host"))
                    {
                        WindowsStartupManager.Start();
                        Exit();
                        return;
                    }
                }
                catch (Exception exception)
                {
                    // A disabled/unavailable Windows service must not prevent
                    // the user from opening the app and recovering work.
                    AppDiagnostics.AppendActivity($"Windows task recovery unavailable: {exception.Message}");
                }
            }
            AppDiagnostics.AppendLifecycle("started");
            string? uiEndToEndReportPath = null;
            for (var index = 0; index < arguments.Length; index++)
            {
                if (arguments[index].Equals(
                        "--ui-e2e-report",
                        StringComparison.OrdinalIgnoreCase)
                    && index + 1 < arguments.Length)
                {
                    uiEndToEndReportPath = Path.GetFullPath(arguments[++index]);
                }
                else if (arguments[index].StartsWith(
                             "--ui-e2e-report=",
                             StringComparison.OrdinalIgnoreCase))
                {
                    uiEndToEndReportPath = Path.GetFullPath(
                        arguments[index].Split('=', 2)[1]);
                }
            }
            if (uiEndToEndReportPath is not null
                && string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable(
                    AppSettingsStore.TestDataRootEnvironment)))
            {
                throw new InvalidOperationException(
                    "The UI end-to-end probe requires an isolated desktop test data root.");
            }
            _window = new MainWindow(
                startupLaunch: commandLine.Contains("--startup"),
                promptForSetup: commandLine.Contains("--prompt-setup"),
                uiEndToEndReportPath: uiEndToEndReportPath);
            _window.Activate();
        }
        catch (Exception exception)
        {
            AppDiagnostics.AppendCrash(exception, "Window startup");
            Environment.ExitCode = 1;
            throw;
        }
    }

    private static void ValidateWorkflowPreviewEnvironment()
    {
        var dataRoot = Environment.GetEnvironmentVariable(
            AppSettingsStore.TestDataRootEnvironment);
        var libraryRoot = Environment.GetEnvironmentVariable("BROADCASTIFY_LIBRARY_ROOT");
        var database = Environment.GetEnvironmentVariable("BROADCASTIFY_SECURE_ANALYSIS_DB");
        var quotaLedger = Environment.GetEnvironmentVariable("BROADCASTIFY_QUOTA_LEDGER");
        var credentials = Environment.GetEnvironmentVariable("BROADCASTIFY_CREDENTIAL_STORE");
        if (Environment.GetEnvironmentVariable("BROADCASTIFY_DESKTOP_E2E_ISOLATED") != "1"
            || string.IsNullOrWhiteSpace(dataRoot)
            || string.IsNullOrWhiteSpace(libraryRoot)
            || string.IsNullOrWhiteSpace(database)
            || string.IsNullOrWhiteSpace(quotaLedger)
            || string.IsNullOrWhiteSpace(credentials))
        {
            throw new InvalidOperationException(
                "The workflow preview requires isolated settings, library, quota, and credential paths.");
        }

        var resolvedRoot = Path.GetFullPath(dataRoot);
        var previewRoot = Directory.GetParent(resolvedRoot)?.FullName
            ?? throw new InvalidOperationException("The isolated preview data root has no parent directory.");
        RequireContainedPath(previewRoot, resolvedRoot);
        RequireContainedPath(previewRoot, libraryRoot);
        RequireContainedPath(previewRoot, database);
        RequireContainedPath(resolvedRoot, quotaLedger);
        RequireContainedPath(resolvedRoot, credentials);
        RequireContainedPath(resolvedRoot, AppSettingsStore.SettingsPath);
    }

    private static void RequireContainedPath(string root, string candidate)
    {
        var fullPath = Path.GetFullPath(candidate);
        var relative = Path.GetRelativePath(root, fullPath);
        if (relative == "."
            || Path.IsPathRooted(relative)
            || relative.Equals("..", StringComparison.Ordinal)
            || relative.StartsWith(".." + Path.DirectorySeparatorChar, StringComparison.Ordinal)
            || relative.StartsWith(".." + Path.AltDirectorySeparatorChar, StringComparison.Ordinal))
        {
            throw new InvalidOperationException(
                "Workflow preview paths must remain below their isolated data root.");
        }
    }
}
