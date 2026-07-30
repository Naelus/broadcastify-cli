using Microsoft.Win32;

namespace BroadcastifyCli.WinUI;

internal static class WindowsStartupManager
{
    internal const string RunKeyPath =
        @"Software\Microsoft\Windows\CurrentVersion\Run";
    internal const string RunValueName = "Broadcastify Desktop";
    internal const string PreferenceKeyPath =
        @"Software\Radio Archive Project\Broadcastify Desktop";
    internal const string PreferenceValueName = "StartWithWindows";

    public static bool IsEnabled()
    {
        using var key = Registry.CurrentUser.OpenSubKey(RunKeyPath);
        return key?.GetValue(RunValueName) is string command
            && !string.IsNullOrWhiteSpace(command);
    }

    public static void SetEnabled(bool enabled)
    {
        if (enabled)
        {
            using var runKey = Registry.CurrentUser.CreateSubKey(
                RunKeyPath,
                writable: true) ?? throw new InvalidOperationException(
                "Windows could not open the current-user startup registry key.");
            runKey.SetValue(
                RunValueName,
                StartupCommand(),
                RegistryValueKind.String);
        }
        else
        {
            using var runKey = Registry.CurrentUser.OpenSubKey(
                RunKeyPath,
                writable: true);
            runKey?.DeleteValue(RunValueName, throwOnMissingValue: false);
        }

        using var preferenceKey = Registry.CurrentUser.CreateSubKey(
            PreferenceKeyPath,
            writable: true) ?? throw new InvalidOperationException(
            "Windows could not save the startup preference.");
        preferenceKey.SetValue(
            PreferenceValueName,
            enabled ? 1 : 0,
            RegistryValueKind.DWord);
    }

    internal static string StartupCommand(string? executablePath = null)
    {
        var path = executablePath ?? Environment.ProcessPath;
        if (string.IsNullOrWhiteSpace(path))
        {
            throw new InvalidOperationException(
                "Windows could not determine the Broadcastify Desktop executable path.");
        }
        return $"\"{Path.GetFullPath(path)}\" --startup --prompt-setup";
    }
}
