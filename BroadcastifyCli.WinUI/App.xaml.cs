using Microsoft.UI.Xaml;

namespace BroadcastifyCli.WinUI;

public partial class App : Application
{
    private Window? _window;

    public App()
    {
        InitializeComponent();
        UnhandledException += (_, eventArgs) => WriteStartupError(eventArgs.Exception);
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
            WriteStartupError(exception);
            throw;
        }
    }

    private static void WriteStartupError(Exception exception)
    {
        try
        {
            File.WriteAllText(
                Path.Combine(Path.GetTempPath(), "BroadcastifyDesktop-startup.log"),
                exception.ToString());
        }
        catch
        {
            // Never hide the original startup failure with a diagnostic failure.
        }
    }
}
