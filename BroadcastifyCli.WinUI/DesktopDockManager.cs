using System.ComponentModel;
using System.Runtime.InteropServices;
using Microsoft.UI.Dispatching;
using Microsoft.UI.Windowing;
using Windows.Graphics;

namespace BroadcastifyCli.WinUI;

internal enum DesktopDockSide
{
    None,
    Left,
    Right,
}

internal readonly record struct DesktopMonitorSnapshot(
    string DeviceName,
    RectInt32 Bounds,
    RectInt32 WorkArea,
    uint Dpi);

internal readonly record struct DesktopDockSnapshot(
    RectInt32 WindowBounds,
    RectInt32 MonitorBounds,
    RectInt32 WorkArea,
    string MonitorDeviceName,
    uint Dpi,
    bool AppBarRegistered,
    bool InteractiveResize);

internal readonly record struct DesktopDockResizeSnapshot(
    DesktopDockSnapshot Before,
    DesktopDockSnapshot During,
    DesktopDockSnapshot After);

/// <summary>
/// Registers the unpackaged desktop window as a Windows AppBar. Unlike merely
/// moving a window to an edge, an AppBar reserves work area so other windows do
/// not cover the pinned status surface.
/// </summary>
internal sealed class DesktopDockManager : IDisposable
{
    internal const double RecommendedWidthDips =
        DesktopDockPolicy.RecommendedWidthDips;
    internal const double MinimumWidthDips =
        DesktopDockPolicy.MinimumWidthDips;
    internal const double MaximumWidthDips =
        DesktopDockPolicy.MaximumWidthDips;

    private const uint AbmNew = 0x00000000;
    private const uint AbmRemove = 0x00000001;
    private const uint AbmQueryPos = 0x00000002;
    private const uint AbmSetPos = 0x00000003;
    private const uint AbmActivate = 0x00000006;
    private const uint AbmWindowPosChanged = 0x00000009;
    private const uint AbeLeft = 0;
    private const uint AbeRight = 2;
    private const uint AbnPosChanged = 0x00000001;
    private const uint MonitorDefaultToNearest = 0x00000002;
    private const uint WmDestroy = 0x0002;
    private const uint WmActivate = 0x0006;
    private const uint WmSettingChange = 0x001A;
    private const uint WmWindowPosChanged = 0x0047;
    private const uint WmDisplayChange = 0x007E;
    private const uint WmNcCalcSize = 0x0083;
    private const uint WmDpiChanged = 0x02E0;
    private const uint SwpNoSize = 0x0001;
    private const uint SwpNoMove = 0x0002;
    private const uint SwpNoZOrder = 0x0004;
    private const uint SwpNoActivate = 0x0010;
    private const uint SwpFrameChanged = 0x0020;
    private const int MonitorInfoDeviceNameLength = 32;

    private static readonly nuint SubclassId = 0x42524443; // "BRDC"

    private readonly nint _windowHandle;
    private readonly AppWindow _appWindow;
    private readonly DispatcherQueue _dispatcherQueue;
    private readonly uint _callbackMessage;
    private readonly uint _taskbarCreatedMessage;
    private readonly SubclassProcedure _subclassProcedure;
    private RectInt32 _floatingBounds;
    private bool _restoreMaximized;
    private bool _hasRetainedFloatingBounds;
    private bool _registered;
    private bool _disposed;
    private bool _positionUpdateQueued;
    private bool _applyingPosition;
    private bool _interactiveResize;
    private int _pointerResizeStartX;
    private NativeRectangle _pointerResizeStartBounds;
    private string _monitorDeviceName = "";

    internal DesktopDockManager(
        nint windowHandle,
        AppWindow appWindow,
        DispatcherQueue dispatcherQueue,
        RectInt32? savedFloatingBounds = null,
        bool restoreMaximized = false)
    {
        if (windowHandle == 0)
        {
            throw new ArgumentException(
                "A desktop window handle is required.",
                nameof(windowHandle));
        }

        _windowHandle = windowHandle;
        _appWindow = appWindow;
        _dispatcherQueue = dispatcherQueue;
        _floatingBounds = savedFloatingBounds is { } saved
            ? NormalizeFloatingBounds(saved)
            : new RectInt32(
                appWindow.Position.X,
                appWindow.Position.Y,
                appWindow.Size.Width,
                appWindow.Size.Height);
        _restoreMaximized = restoreMaximized;
        _hasRetainedFloatingBounds = savedFloatingBounds is not null;
        _callbackMessage = RegisterWindowMessage(
            $"BroadcastifyDesktop.AppBar.{Environment.ProcessId}");
        _taskbarCreatedMessage = RegisterWindowMessage("TaskbarCreated");
        if (_callbackMessage == 0 || _taskbarCreatedMessage == 0)
        {
            throw new Win32Exception(Marshal.GetLastWin32Error());
        }

        _subclassProcedure = WindowSubclassProcedure;
        if (!SetWindowSubclass(
                _windowHandle,
                _subclassProcedure,
                SubclassId,
                0))
        {
            throw new Win32Exception(Marshal.GetLastWin32Error());
        }
    }

    internal event Action<double>? InteractiveResizeCompleted;

    internal DesktopDockSide Side { get; private set; }

    internal double WidthDips { get; private set; } = RecommendedWidthDips;

    internal string MonitorDeviceName => _monitorDeviceName;

    internal bool IsDocked =>
        Side is DesktopDockSide.Left or DesktopDockSide.Right;

    internal bool IsInteractiveResize => _interactiveResize;

    internal RectInt32 FloatingBounds => _floatingBounds;

    internal bool RestoreMaximized => _restoreMaximized;

    internal DesktopDockSnapshot CaptureSnapshot()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var monitor = MonitorFromWindow(
            _windowHandle,
            MonitorDefaultToNearest);
        if (!TryCaptureMonitor(monitor, out var monitorSnapshot)
            || !TryGetWindowRectangle(out var windowRectangle))
        {
            throw new Win32Exception(Marshal.GetLastWin32Error());
        }

        return new DesktopDockSnapshot(
            ToRectInt32(windowRectangle),
            monitorSnapshot.Bounds,
            monitorSnapshot.WorkArea,
            monitorSnapshot.DeviceName,
            monitorSnapshot.Dpi,
            _registered,
            _interactiveResize);
    }

    internal IReadOnlyList<DesktopMonitorSnapshot> CaptureAvailableMonitors()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var monitors = new List<DesktopMonitorSnapshot>();
        MonitorEnumProcedure callback = (
            monitor,
            _,
            _,
            _) =>
        {
            if (TryCaptureMonitor(monitor, out var snapshot))
            {
                monitors.Add(snapshot);
            }
            return true;
        };
        if (!EnumDisplayMonitors(0, 0, callback, 0))
        {
            throw new Win32Exception(Marshal.GetLastWin32Error());
        }
        GC.KeepAlive(callback);
        return monitors;
    }

    internal void Dock(
        DesktopDockSide side,
        double widthDips,
        string? monitorDeviceName = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (side is not (DesktopDockSide.Left or DesktopDockSide.Right))
        {
            throw new ArgumentOutOfRangeException(nameof(side));
        }

        var wasDocked = IsDocked;
        var changingSide = wasDocked && Side != side;
        if (!wasDocked)
        {
            if (_appWindow.Presenter is OverlappedPresenter presenter)
            {
                var maximized =
                    presenter.State == OverlappedPresenterState.Maximized;
                if (!maximized || !_hasRetainedFloatingBounds)
                {
                    _floatingBounds = new RectInt32(
                        _appWindow.Position.X,
                        _appWindow.Position.Y,
                        _appWindow.Size.Width,
                        _appWindow.Size.Height);
                    _hasRetainedFloatingBounds = true;
                }
                _restoreMaximized = maximized;
                if (presenter.State != OverlappedPresenterState.Restored)
                {
                    presenter.Restore();
                }
            }
        }
        var requestedMonitor = string.IsNullOrWhiteSpace(monitorDeviceName)
            ? wasDocked
                ? _monitorDeviceName
                : null
            : monitorDeviceName;
        var monitor = ResolveMonitor(requestedMonitor);
        var changingMonitor = wasDocked
            && !string.Equals(
                _monitorDeviceName,
                monitor.DeviceName,
                StringComparison.OrdinalIgnoreCase);
        if (changingSide || changingMonitor)
        {
            RemoveAppBar();
        }
        Side = side;
        _monitorDeviceName = monitor.DeviceName;
        WidthDips = DesktopDockPolicy.NormalizeWidthDips(
            widthDips,
            DesktopDockPolicy.PixelsToDips(
                monitor.Bounds.Width,
                monitor.Dpi));
        RegisterAppBar();
        ApplyDockPosition();
        SetPinnedPresenterState(true);
        RefreshNonClientFrame();
    }

    internal void ChangeWidth(double widthDips)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var monitor = ResolveMonitor(_monitorDeviceName);
        WidthDips = DesktopDockPolicy.NormalizeWidthDips(
            widthDips,
            DesktopDockPolicy.PixelsToDips(
                monitor.Bounds.Width,
                monitor.Dpi));
        if (IsDocked && !_interactiveResize)
        {
            ApplyDockPosition();
        }
    }

    internal bool BeginPointerResize()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_interactiveResize
            || !IsDocked
            || !TryGetWindowRectangle(out var windowRectangle)
            || !GetCursorPos(out var pointer))
        {
            return false;
        }

        BeginInteractiveResize(windowRectangle, pointer.X, capturePointer: true);
        return true;
    }

    internal void ContinuePointerResize()
    {
        if (!_interactiveResize || !GetCursorPos(out var pointer))
        {
            return;
        }
        ApplyInteractiveResize(pointer.X);
    }

    internal void CompletePointerResize()
    {
        if (!_interactiveResize)
        {
            return;
        }
        CompleteInteractiveResize(releasePointerCapture: true);
    }

    internal DesktopDockResizeSnapshot ExerciseInteractiveResizeForEndToEndTest(
        int pointerDeltaPixels)
    {
        if (string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable(
                AppSettingsStore.TestDataRootEnvironment)))
        {
            throw new InvalidOperationException(
                "Interactive resize diagnostics require an isolated test data root.");
        }
        if (!IsDocked || !TryGetWindowRectangle(out var startBounds))
        {
            throw new InvalidOperationException(
                "The desktop must be pinned before exercising resize diagnostics.");
        }

        var before = CaptureSnapshot();
        BeginInteractiveResize(startBounds, 0, capturePointer: false);
        ApplyInteractiveResize(pointerDeltaPixels);
        var during = CaptureSnapshot();
        CompleteInteractiveResize(releasePointerCapture: false);
        return new DesktopDockResizeSnapshot(
            before,
            during,
            CaptureSnapshot());
    }

    internal void ReRegisterAfterShellRestartForEndToEndTest()
    {
        if (string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable(
                AppSettingsStore.TestDataRootEnvironment)))
        {
            throw new InvalidOperationException(
                "Shell-restart diagnostics require an isolated test data root.");
        }
        RemoveAppBar();
        RegisterAppBar();
        ApplyDockPosition();
    }

    internal void UpdateFloatingPlacement(
        RectInt32 bounds,
        bool maximized)
    {
        if (_disposed || IsDocked || _applyingPosition)
        {
            return;
        }
        if (!maximized && bounds.Width > 0 && bounds.Height > 0)
        {
            _floatingBounds = NormalizeFloatingBounds(bounds);
            _hasRetainedFloatingBounds = true;
        }
        _restoreMaximized = maximized;
    }

    internal void RestoreFloatingWindowForStartup()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (!_hasRetainedFloatingBounds)
        {
            return;
        }
        _applyingPosition = true;
        try
        {
            _appWindow.MoveAndResize(_floatingBounds);
            if (_restoreMaximized
                && _appWindow.Presenter is OverlappedPresenter presenter)
            {
                presenter.Maximize();
            }
        }
        finally
        {
            _applyingPosition = false;
        }
    }

    internal void Unpin(bool restoreFloatingWindow = true)
    {
        if (_disposed || !IsDocked)
        {
            return;
        }

        if (_interactiveResize)
        {
            CompleteInteractiveResize(releasePointerCapture: true);
        }
        RemoveAppBar();
        Side = DesktopDockSide.None;
        SetPinnedPresenterState(false);
        RefreshNonClientFrame();
        if (!restoreFloatingWindow)
        {
            return;
        }

        _applyingPosition = true;
        try
        {
            _appWindow.MoveAndResize(NormalizeFloatingBounds(_floatingBounds));
            if (_restoreMaximized
                && _appWindow.Presenter is OverlappedPresenter presenter)
            {
                presenter.Maximize();
            }
        }
        finally
        {
            _applyingPosition = false;
        }
    }

    private void RegisterAppBar()
    {
        if (_registered)
        {
            return;
        }

        var data = CreateAppBarData();
        if (SHAppBarMessage(AbmNew, ref data) == 0)
        {
            throw new Win32Exception(
                Marshal.GetLastWin32Error(),
                "Windows could not reserve desktop space for Broadcastify Desktop.");
        }
        _registered = true;
    }

    private void RemoveAppBar()
    {
        if (!_registered)
        {
            return;
        }

        var data = CreateAppBarData();
        SHAppBarMessage(AbmRemove, ref data);
        _registered = false;
    }

    private void ApplyDockPosition()
    {
        if (!_registered || !IsDocked || _applyingPosition || _interactiveResize)
        {
            return;
        }

        _applyingPosition = true;
        try
        {
            var monitor = ResolveMonitor(_monitorDeviceName);
            _monitorDeviceName = monitor.DeviceName;
            var monitorWidthDips = DesktopDockPolicy.PixelsToDips(
                monitor.Bounds.Width,
                monitor.Dpi);
            WidthDips = DesktopDockPolicy.NormalizeWidthDips(
                WidthDips,
                monitorWidthDips);
            var widthPixels = DesktopDockPolicy.DipsToPixels(
                WidthDips,
                monitor.Dpi);

            var data = CreateAppBarData();
            data.Edge = Side == DesktopDockSide.Left ? AbeLeft : AbeRight;
            data.Rectangle = ToNativeRectangle(monitor.Bounds);
            SHAppBarMessage(AbmQueryPos, ref data);
            if (Side == DesktopDockSide.Left)
            {
                data.Rectangle.Right = data.Rectangle.Left + widthPixels;
            }
            else
            {
                data.Rectangle.Left = data.Rectangle.Right - widthPixels;
            }

            if (SHAppBarMessage(AbmSetPos, ref data) == 0)
            {
                throw new Win32Exception(
                    Marshal.GetLastWin32Error(),
                    "Windows could not position the pinned Broadcastify Desktop window.");
            }

            var bounds = data.Rectangle;
            _appWindow.MoveAndResize(new RectInt32(
                bounds.Left,
                bounds.Top,
                Math.Max(1, bounds.Right - bounds.Left),
                Math.Max(1, bounds.Bottom - bounds.Top)));
            WidthDips = DesktopDockPolicy.PixelsToDips(
                bounds.Right - bounds.Left,
                monitor.Dpi);
        }
        finally
        {
            _applyingPosition = false;
        }
    }

    private void BeginInteractiveResize(
        NativeRectangle startBounds,
        int pointerX,
        bool capturePointer)
    {
        _pointerResizeStartBounds = startBounds;
        _pointerResizeStartX = pointerX;
        _interactiveResize = true;
        RemoveAppBar();
        if (capturePointer)
        {
            SetCapture(_windowHandle);
        }
    }

    private void ApplyInteractiveResize(int pointerX)
    {
        var monitor = ResolveMonitor(_monitorDeviceName);
        var initialWidth =
            _pointerResizeStartBounds.Right - _pointerResizeStartBounds.Left;
        var delta = pointerX - _pointerResizeStartX;
        var requestedPixels = Side == DesktopDockSide.Right
            ? initialWidth - delta
            : initialWidth + delta;
        WidthDips = DesktopDockPolicy.NormalizeWidthDips(
            DesktopDockPolicy.PixelsToDips(requestedPixels, monitor.Dpi),
            DesktopDockPolicy.PixelsToDips(
                monitor.Bounds.Width,
                monitor.Dpi));
        var widthPixels = DesktopDockPolicy.DipsToPixels(
            WidthDips,
            monitor.Dpi);
        var height = Math.Max(
            1,
            _pointerResizeStartBounds.Bottom - _pointerResizeStartBounds.Top);
        var bounds = Side == DesktopDockSide.Right
            ? new RectInt32(
                _pointerResizeStartBounds.Right - widthPixels,
                _pointerResizeStartBounds.Top,
                widthPixels,
                height)
            : new RectInt32(
                _pointerResizeStartBounds.Left,
                _pointerResizeStartBounds.Top,
                widthPixels,
                height);
        _applyingPosition = true;
        try
        {
            _appWindow.MoveAndResize(bounds);
        }
        finally
        {
            _applyingPosition = false;
        }
    }

    private void CompleteInteractiveResize(bool releasePointerCapture)
    {
        _interactiveResize = false;
        if (releasePointerCapture && GetCapture() == _windowHandle)
        {
            ReleaseCapture();
        }
        if (IsDocked)
        {
            RegisterAppBar();
            ApplyDockPosition();
            InteractiveResizeCompleted?.Invoke(WidthDips);
        }
    }

    private void SetPinnedPresenterState(bool pinned)
    {
        if (_appWindow.Presenter is not OverlappedPresenter presenter)
        {
            return;
        }

        presenter.IsResizable = !pinned;
        presenter.IsMinimizable = !pinned;
        presenter.IsMaximizable = !pinned;
    }

    private void RefreshNonClientFrame()
    {
        SetWindowPos(
            _windowHandle,
            0,
            0,
            0,
            0,
            0,
            SwpNoSize
                | SwpNoMove
                | SwpNoZOrder
                | SwpNoActivate
                | SwpFrameChanged);
    }

    private DesktopMonitorSnapshot ResolveMonitor(string? deviceName)
    {
        if (!string.IsNullOrWhiteSpace(deviceName))
        {
            foreach (var monitor in CaptureAvailableMonitors())
            {
                if (string.Equals(
                        monitor.DeviceName,
                        deviceName,
                        StringComparison.OrdinalIgnoreCase))
                {
                    return monitor;
                }
            }
        }

        var nearest = MonitorFromWindow(
            _windowHandle,
            MonitorDefaultToNearest);
        if (TryCaptureMonitor(nearest, out var fallback))
        {
            return fallback;
        }
        throw new Win32Exception(
            Marshal.GetLastWin32Error(),
            "Windows could not identify the monitor used for docking.");
    }

    private bool TryCaptureMonitor(
        nint monitor,
        out DesktopMonitorSnapshot snapshot)
    {
        var monitorInfo = new MonitorInfo
        {
            Size = (uint)Marshal.SizeOf<MonitorInfo>(),
            DeviceName = "",
        };
        if (monitor == 0 || !GetMonitorInfo(monitor, ref monitorInfo))
        {
            snapshot = default;
            return false;
        }
        snapshot = new DesktopMonitorSnapshot(
            monitorInfo.DeviceName ?? "",
            ToRectInt32(monitorInfo.Monitor),
            ToRectInt32(monitorInfo.WorkArea),
            GetMonitorDpi(monitor));
        return true;
    }

    private uint GetMonitorDpi(nint monitor)
    {
        if (GetDpiForMonitor(monitor, 0, out var dpiX, out _) == 0
            && dpiX > 0)
        {
            return Math.Max(96u, dpiX);
        }
        return Math.Max(96u, GetDpiForWindow(_windowHandle));
    }

    private RectInt32 NormalizeFloatingBounds(RectInt32 requested)
    {
        var native = ToNativeRectangle(requested);
        var monitor = MonitorFromRect(
            ref native,
            MonitorDefaultToNearest);
        if (!TryCaptureMonitor(monitor, out var snapshot))
        {
            return requested;
        }
        var width = Math.Min(
            Math.Max(480, requested.Width),
            snapshot.WorkArea.Width);
        var height = Math.Min(
            Math.Max(360, requested.Height),
            snapshot.WorkArea.Height);
        var maximumX = snapshot.WorkArea.X + snapshot.WorkArea.Width - width;
        var maximumY = snapshot.WorkArea.Y + snapshot.WorkArea.Height - height;
        return new RectInt32(
            Math.Clamp(requested.X, snapshot.WorkArea.X, maximumX),
            Math.Clamp(requested.Y, snapshot.WorkArea.Y, maximumY),
            width,
            height);
    }

    private bool TryGetWindowRectangle(out NativeRectangle rectangle)
    {
        rectangle = new NativeRectangle();
        return GetWindowRect(_windowHandle, ref rectangle);
    }

    private AppBarData CreateAppBarData() =>
        new()
        {
            Size = (uint)Marshal.SizeOf<AppBarData>(),
            WindowHandle = _windowHandle,
            CallbackMessage = _callbackMessage,
        };

    private nint WindowSubclassProcedure(
        nint windowHandle,
        uint message,
        nuint wParam,
        nint lParam,
        nuint subclassId,
        nuint referenceData)
    {
        try
        {
            if (message == WmNcCalcSize && IsDocked)
            {
                return 0;
            }
            if (message == _taskbarCreatedMessage)
            {
                _registered = false;
                QueueDockPositionUpdate(registerFirst: true);
            }
            else if (!_applyingPosition
                     && !_interactiveResize
                     && ((message == _callbackMessage
                          && (uint)wParam == AbnPosChanged)
                         || message is WmDisplayChange
                             or WmDpiChanged
                             or WmSettingChange))
            {
                QueueDockPositionUpdate(registerFirst: false);
            }
            else if (message == WmActivate && _registered)
            {
                var data = CreateAppBarData();
                data.Parameter = (wParam & 0xffff) == 0 ? 0 : 1;
                SHAppBarMessage(AbmActivate, ref data);
            }
            else if (message == WmWindowPosChanged
                     && _registered
                     && !_interactiveResize)
            {
                var data = CreateAppBarData();
                SHAppBarMessage(AbmWindowPosChanged, ref data);
            }
            else if (message == WmDestroy)
            {
                RemoveAppBar();
            }
        }
        catch (Exception exception)
        {
            AppDiagnostics.AppendCrash(exception, "Desktop docking window message");
        }

        return DefSubclassProc(
            windowHandle,
            message,
            wParam,
            lParam);
    }

    private void QueueDockPositionUpdate(bool registerFirst)
    {
        if (!IsDocked || _interactiveResize || _positionUpdateQueued)
        {
            return;
        }

        _positionUpdateQueued = true;
        if (!_dispatcherQueue.TryEnqueue(() =>
        {
            _positionUpdateQueued = false;
            if (_disposed || !IsDocked || _interactiveResize)
            {
                return;
            }
            try
            {
                if (registerFirst)
                {
                    RegisterAppBar();
                }
                ApplyDockPosition();
            }
            catch (Exception exception)
            {
                AppDiagnostics.AppendCrash(exception, "Desktop docking refresh");
            }
        }))
        {
            _positionUpdateQueued = false;
        }
    }

    private static RectInt32 ToRectInt32(NativeRectangle value) =>
        new(
            value.Left,
            value.Top,
            Math.Max(0, value.Right - value.Left),
            Math.Max(0, value.Bottom - value.Top));

    private static NativeRectangle ToNativeRectangle(RectInt32 value) =>
        new()
        {
            Left = value.X,
            Top = value.Y,
            Right = value.X + value.Width,
            Bottom = value.Y + value.Height,
        };

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        if (_interactiveResize)
        {
            CompleteInteractiveResize(releasePointerCapture: true);
        }
        RemoveAppBar();
        RemoveWindowSubclass(
            _windowHandle,
            _subclassProcedure,
            SubclassId);
        _disposed = true;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct NativeRectangle
    {
        internal int Left;
        internal int Top;
        internal int Right;
        internal int Bottom;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct NativePoint
    {
        internal int X;
        internal int Y;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct AppBarData
    {
        internal uint Size;
        internal nint WindowHandle;
        internal uint CallbackMessage;
        internal uint Edge;
        internal NativeRectangle Rectangle;
        internal nint Parameter;
    }

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    private struct MonitorInfo
    {
        internal uint Size;
        internal NativeRectangle Monitor;
        internal NativeRectangle WorkArea;
        internal uint Flags;

        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = MonitorInfoDeviceNameLength)]
        internal string DeviceName;
    }

    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    private delegate nint SubclassProcedure(
        nint windowHandle,
        uint message,
        nuint wParam,
        nint lParam,
        nuint subclassId,
        nuint referenceData);

    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private delegate bool MonitorEnumProcedure(
        nint monitor,
        nint deviceContext,
        nint monitorRectangle,
        nint data);

    [DllImport("shell32.dll", SetLastError = true)]
    private static extern nuint SHAppBarMessage(
        uint message,
        ref AppBarData data);

    [DllImport("user32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
    private static extern uint RegisterWindowMessage(string message);

    [DllImport("user32.dll")]
    private static extern nint MonitorFromWindow(nint windowHandle, uint flags);

    [DllImport("user32.dll")]
    private static extern nint MonitorFromRect(
        ref NativeRectangle rectangle,
        uint flags);

    [DllImport("user32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool EnumDisplayMonitors(
        nint deviceContext,
        nint clipRectangle,
        MonitorEnumProcedure callback,
        nint data);

    [DllImport("user32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool GetWindowRect(
        nint windowHandle,
        ref NativeRectangle rectangle);

    [DllImport(
        "user32.dll",
        EntryPoint = "GetMonitorInfoW",
        SetLastError = true,
        CharSet = CharSet.Unicode)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool GetMonitorInfo(
        nint monitor,
        ref MonitorInfo monitorInfo);

    [DllImport("user32.dll")]
    private static extern uint GetDpiForWindow(nint windowHandle);

    [DllImport("shcore.dll")]
    private static extern int GetDpiForMonitor(
        nint monitor,
        int dpiType,
        out uint dpiX,
        out uint dpiY);

    [DllImport("user32.dll")]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool GetCursorPos(out NativePoint point);

    [DllImport("user32.dll")]
    private static extern nint SetCapture(nint windowHandle);

    [DllImport("user32.dll")]
    private static extern nint GetCapture();

    [DllImport("user32.dll")]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool ReleaseCapture();

    [DllImport("user32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool SetWindowPos(
        nint windowHandle,
        nint insertAfter,
        int x,
        int y,
        int width,
        int height,
        uint flags);

    [DllImport("comctl32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool SetWindowSubclass(
        nint windowHandle,
        SubclassProcedure subclassProcedure,
        nuint subclassId,
        nuint referenceData);

    [DllImport("comctl32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool RemoveWindowSubclass(
        nint windowHandle,
        SubclassProcedure subclassProcedure,
        nuint subclassId);

    [DllImport("comctl32.dll")]
    private static extern nint DefSubclassProc(
        nint windowHandle,
        uint message,
        nuint wParam,
        nint lParam);
}
