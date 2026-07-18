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
            _window = new MainWindow();
            _window.Activate();
        }
        catch (Exception exception)
        {
            AppDiagnostics.AppendCrash(exception, "Window startup");
            throw;
        }
    }
}
