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
            throw;
        }
    }
}
