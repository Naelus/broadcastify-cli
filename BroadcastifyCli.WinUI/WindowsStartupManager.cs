using System.Runtime.InteropServices;
using System.Security.Principal;
using System.Diagnostics;
using Microsoft.Win32;

namespace BroadcastifyCli.WinUI;

internal static class WindowsStartupManager
{
    internal const string TaskName = "Broadcastify Desktop";
    internal const string RunKeyPath =
        @"Software\Microsoft\Windows\CurrentVersion\Run";
    internal const string RunValueName = "Broadcastify Desktop";
    internal const string PreferenceKeyPath =
        @"Software\Radio Archive Project\Broadcastify Desktop";
    internal const string PreferenceValueName = "StartWithWindows";

    private const int TaskCreateOrUpdate = 6;
    private const int TaskTriggerLogon = 9;
    private const int TaskActionExec = 0;
    private const int TaskLogonInteractiveToken = 3;
    private const int TaskRunLevelLeastPrivilege = 0;
    private const int TaskMultipleInstancesIgnoreNew = 2;
    private const int TaskPriorityNormal = 4;
    private const string StartupArguments =
        "--startup --prompt-setup --scheduled-host";

    public static bool IsEnabled()
    {
        using var service = new TaskSchedulerConnection();
        var task = FindTask(service.Folder);
        if (task is not null)
        {
            try
            {
                dynamic definition = task.Definition;
                dynamic settings = definition.Settings;
                return Convert.ToBoolean(settings.Enabled)
                    && HasCurrentUserLogonTrigger(definition);
            }
            finally
            {
                ReleaseComObject(task);
            }
        }

        return HasLegacyRunValue();
    }

    public static void SetEnabled(bool enabled) => Configure(enabled ? "enabled" : "disabled");

    internal static void Configure(string mode)
    {
        var normalizedMode = mode.Trim().ToLowerInvariant();
        if (normalizedMode == "remove")
        {
            RemoveTask();
            RemoveLegacyRunValue();
            return;
        }

        var enabled = normalizedMode switch
        {
            "enabled" => true,
            "disabled" => false,
            "preserve" => ResolveSavedPreference(),
            _ => throw new ArgumentOutOfRangeException(nameof(mode)),
        };

        RegisterOrUpdateTask(enabled);
        SavePreference(enabled);
        RemoveLegacyRunValue();
    }

    internal static void Start()
    {
        // Opening the app again should reveal the existing host rather than
        // merely submitting an ignored second task instance.
        foreach (var process in Process.GetProcessesByName(Path.GetFileNameWithoutExtension(Environment.ProcessPath)))
        {
            using (process)
            {
                try
                {
                    if (process.Id != Environment.ProcessId && process.MainWindowHandle != 0
                        && string.Equals(process.MainModule?.FileName, Environment.ProcessPath, StringComparison.OrdinalIgnoreCase))
                    {
                        ShowWindowAsync(process.MainWindowHandle, 9);
                        SetForegroundWindow(process.MainWindowHandle);
                        return;
                    }
                }
                catch (System.ComponentModel.Win32Exception) { /* Another user's process. */ }
                catch (InvalidOperationException) { /* Process exited during enumeration. */ }
            }
        }
        using var service = new TaskSchedulerConnection();
        var task = FindTask(service.Folder)
            ?? throw new InvalidOperationException(
                "The Broadcastify Desktop sign-in task is not registered.");
        try
        {
            task.Run(null);
        }
        finally
        {
            ReleaseComObject(task);
        }
    }

    internal static string StartupCommand(string? executablePath = null)
    {
        var path = executablePath ?? Environment.ProcessPath;
        if (string.IsNullOrWhiteSpace(path))
        {
            throw new InvalidOperationException(
                "Windows could not determine the Broadcastify Desktop executable path.");
        }
        return $"\"{Path.GetFullPath(path)}\" {StartupArguments}";
    }

    private static bool ResolveSavedPreference()
    {
        using var preferenceKey = Registry.CurrentUser.OpenSubKey(PreferenceKeyPath);
        var savedPreference = preferenceKey?.GetValue(PreferenceValueName);
        if (savedPreference is not null)
        {
            return Convert.ToInt32(savedPreference) != 0;
        }

        if (HasLegacyRunValue())
        {
            return true;
        }

        using var service = new TaskSchedulerConnection();
        var task = FindTask(service.Folder);
        if (task is null)
        {
            return false;
        }
        try
        {
            dynamic definition = task.Definition;
            dynamic settings = definition.Settings;
            return Convert.ToBoolean(settings.Enabled)
                && HasCurrentUserLogonTrigger(definition);
        }
        finally
        {
            ReleaseComObject(task);
        }
    }

    private static void RegisterOrUpdateTask(bool enabled)
    {
        var executable = Environment.ProcessPath;
        if (string.IsNullOrWhiteSpace(executable))
        {
            throw new InvalidOperationException(
                "Windows could not determine the Broadcastify Desktop executable path.");
        }
        executable = Path.GetFullPath(executable);
        var userSid = WindowsIdentity.GetCurrent().User?.Value
            ?? throw new InvalidOperationException(
                "Windows could not determine the current user's security identifier.");

        using var service = new TaskSchedulerConnection();
        var existingTask = FindTask(service.Folder);
        try
        {
            if (existingTask is not null
                && MatchesRequiredDefinition(
                    existingTask,
                    executable,
                    userSid,
                    enabled))
            {
                return;
            }

            dynamic definition = service.Service.NewTask(0);
            try
            {
                ConfigureDefinition(definition, executable, userSid, enabled);
                service.Folder.RegisterTaskDefinition(
                    TaskName,
                    definition,
                    TaskCreateOrUpdate,
                    userSid,
                    null,
                    TaskLogonInteractiveToken,
                    null);
            }
            finally
            {
                ReleaseComObject(definition);
            }
        }
        finally
        {
            if (existingTask is not null)
            {
                ReleaseComObject(existingTask);
            }
        }
    }

    private static void ConfigureDefinition(
        dynamic definition,
        string executable,
        string userSid,
        bool enabled)
    {
        dynamic registrationInfo = definition.RegistrationInfo;
        registrationInfo.Description =
            "Starts Broadcastify Desktop at sign-in and restarts it after failures.";

        dynamic principal = definition.Principal;
        principal.UserId = userSid;
        principal.LogonType = TaskLogonInteractiveToken;
        principal.RunLevel = TaskRunLevelLeastPrivilege;

        dynamic settings = definition.Settings;
        settings.Enabled = enabled;
        settings.AllowHardTerminate = true;
        settings.DisallowStartIfOnBatteries = false;
        settings.StopIfGoingOnBatteries = false;
        settings.ExecutionTimeLimit = "PT0S";
        settings.MultipleInstances = TaskMultipleInstancesIgnoreNew;
        settings.Priority = TaskPriorityNormal;
        settings.RestartCount = 3;
        settings.RestartInterval = "PT1M";

        dynamic triggers = definition.Triggers;
        if (enabled)
        {
            dynamic trigger = triggers.Create(TaskTriggerLogon);
            trigger.UserId = userSid;
        }

        dynamic actions = definition.Actions;
        dynamic action = actions.Create(TaskActionExec);
        action.Path = executable;
        action.Arguments = StartupArguments;
        action.WorkingDirectory = Path.GetDirectoryName(executable) ?? "";
    }

    private static bool MatchesRequiredDefinition(
        dynamic task,
        string executable,
        string userSid,
        bool enabled)
    {
        dynamic definition = task.Definition;
        dynamic principal = definition.Principal;
        dynamic settings = definition.Settings;
        dynamic triggers = definition.Triggers;
        dynamic actions = definition.Actions;
        if (!IsCurrentUser(Convert.ToString(principal.UserId), userSid)
            || Convert.ToInt32(principal.LogonType) != TaskLogonInteractiveToken
            || Convert.ToInt32(principal.RunLevel) != TaskRunLevelLeastPrivilege
            || Convert.ToBoolean(settings.Enabled) != enabled
            || !Convert.ToBoolean(settings.AllowHardTerminate)
            || Convert.ToBoolean(settings.DisallowStartIfOnBatteries)
            || Convert.ToBoolean(settings.StopIfGoingOnBatteries)
            || !string.Equals(
                Convert.ToString(settings.ExecutionTimeLimit),
                "PT0S",
                StringComparison.OrdinalIgnoreCase)
            || Convert.ToInt32(settings.MultipleInstances)
                != TaskMultipleInstancesIgnoreNew
            || Convert.ToInt32(settings.Priority) != TaskPriorityNormal
            || Convert.ToInt32(settings.RestartCount) != 3
            || !string.Equals(
                Convert.ToString(settings.RestartInterval),
                "PT1M",
                StringComparison.OrdinalIgnoreCase)
            || Convert.ToInt32(triggers.Count) != (enabled ? 1 : 0)
            || Convert.ToInt32(actions.Count) != 1)
        {
            return false;
        }

        dynamic action = actions.Item(1);
        var hasExpectedTrigger = !enabled;
        if (enabled)
        {
            dynamic trigger = triggers.Item(1);
            hasExpectedTrigger = Convert.ToInt32(trigger.Type) == TaskTriggerLogon
                && IsCurrentUser(Convert.ToString(trigger.UserId), userSid);
            ReleaseComObject(trigger);
        }
        var hasExpectedAction = Convert.ToInt32(action.Type) == TaskActionExec
            && string.Equals(
                Convert.ToString(action.Path),
                executable,
                StringComparison.OrdinalIgnoreCase)
            && string.Equals(
                Convert.ToString(action.Arguments),
                StartupArguments,
                StringComparison.Ordinal)
            && string.Equals(
                Convert.ToString(action.WorkingDirectory),
                Path.GetDirectoryName(executable),
                StringComparison.OrdinalIgnoreCase);
        ReleaseComObject(action);
        return hasExpectedTrigger && hasExpectedAction;
    }

    private static bool HasCurrentUserLogonTrigger(dynamic definition)
    {
        dynamic triggers = definition.Triggers;
        if (Convert.ToInt32(triggers.Count) == 0)
        {
            return false;
        }

        var userSid = WindowsIdentity.GetCurrent().User?.Value;
        if (userSid is null) return false;
        for (var index = 1; index <= Convert.ToInt32(triggers.Count); index++)
        {
            dynamic trigger = triggers.Item(index);
            var matches = Convert.ToInt32(trigger.Type) == TaskTriggerLogon
                && IsCurrentUser(Convert.ToString(trigger.UserId), userSid);
            ReleaseComObject(trigger);
            if (matches)
            {
                return true;
            }
        }
        return false;
    }

    private static bool IsCurrentUser(string? userIdentity, string currentUserSid)
    {
        if (string.IsNullOrWhiteSpace(userIdentity))
        {
            return false;
        }

        try
        {
            SecurityIdentifier userSid;
            try
            {
                userSid = new SecurityIdentifier(userIdentity);
            }
            catch (ArgumentException)
            {
                userSid = (SecurityIdentifier)new NTAccount(userIdentity)
                    .Translate(typeof(SecurityIdentifier));
            }
            return userSid.Value.Equals(
                currentUserSid,
                StringComparison.OrdinalIgnoreCase);
        }
        catch (IdentityNotMappedException)
        {
            return false;
        }
        catch (ArgumentException)
        {
            return false;
        }
    }

    private static dynamic? FindTask(dynamic folder)
    {
        dynamic tasks = folder.GetTasks(1);
        try
        {
            var count = Convert.ToInt32(tasks.Count);
            for (var index = 1; index <= count; index++)
            {
                dynamic task = tasks.Item(index);
                if (string.Equals(
                        Convert.ToString(task.Name),
                        TaskName,
                        StringComparison.OrdinalIgnoreCase))
                {
                    return task;
                }
                ReleaseComObject(task);
            }
            return null;
        }
        finally
        {
            ReleaseComObject(tasks);
        }
    }

    private static void RemoveTask()
    {
        using var service = new TaskSchedulerConnection();
        var task = FindTask(service.Folder);
        if (task is null)
        {
            return;
        }
        ReleaseComObject(task);
        service.Folder.DeleteTask(TaskName, 0);
    }

    private static bool HasLegacyRunValue()
    {
        using var runKey = Registry.CurrentUser.OpenSubKey(RunKeyPath);
        return runKey?.GetValue(RunValueName) is string command
            && !string.IsNullOrWhiteSpace(command);
    }

    private static void RemoveLegacyRunValue()
    {
        using var runKey = Registry.CurrentUser.OpenSubKey(RunKeyPath, writable: true);
        runKey?.DeleteValue(RunValueName, throwOnMissingValue: false);
    }

    private static void SavePreference(bool enabled)
    {
        using var preferenceKey = Registry.CurrentUser.CreateSubKey(
            PreferenceKeyPath,
            writable: true) ?? throw new InvalidOperationException(
            "Windows could not save the startup preference.");
        preferenceKey.SetValue(
            PreferenceValueName,
            enabled ? 1 : 0,
            RegistryValueKind.DWord);
    }

    private static void ReleaseComObject(object? value)
    {
        if (value is not null && Marshal.IsComObject(value))
        {
            Marshal.FinalReleaseComObject(value);
        }
    }

    private sealed class TaskSchedulerConnection : IDisposable
    {
        private readonly object _serviceObject;

        internal dynamic Service { get; }
        internal dynamic Folder { get; }

        internal TaskSchedulerConnection()
        {
            var serviceType = Type.GetTypeFromProgID("Schedule.Service", throwOnError: true)
                ?? throw new InvalidOperationException(
                    "Windows Task Scheduler is unavailable.");
            _serviceObject = Activator.CreateInstance(serviceType)
                ?? throw new InvalidOperationException(
                    "Windows could not create a Task Scheduler connection.");
            Service = _serviceObject;
            Service.Connect();
            Folder = Service.GetFolder("\\");
        }

        public void Dispose()
        {
            ReleaseComObject(Folder);
            ReleaseComObject(Service);
        }
    }

    [DllImport("user32.dll")]
    private static extern bool ShowWindowAsync(nint window, int command);
    [DllImport("user32.dll")]
    private static extern bool SetForegroundWindow(nint window);
}
