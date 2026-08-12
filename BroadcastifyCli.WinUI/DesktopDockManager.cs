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

internal readonly record struct DesktopDockSnapshot(
    RectInt32 WindowBounds,
    RectInt32 MonitorBounds,
    RectInt32 WorkArea);

/// <summary>
/// Registers the unpackaged desktop window as a Windows AppBar. Unlike merely
/// moving a window to an edge, an AppBar reserves work area so other windows do
/// not cover the pinned status surface.
/// </summary>
internal sealed class DesktopDockManager : IDisposable
{
    internal const double RecommendedWidthDips = 600;
    internal const double MinimumWidthDips = 520;
    private const double MaximumMonitorWidthFraction = 0.72;

    private const uint AbmNew = 0x00000000;
    private const uint AbmRemove = 0x00000001;
    private const uint AbmQueryPos = 0x00000002;
    private const uint AbmSetPos = 0x00000003;
    private const uint AbeLeft = 0;
    private const uint AbeRight = 2;
    private const uint AbnPosChanged = 0x00000001;
    private const uint MonitorDefaultToNearest = 0x00000002;
    private const uint WmDisplayChange = 0x007E;
    private const uint WmDpiChanged = 0x02E0;
    private const uint WmSettingChange = 0x001A;

    private static readonly nuint SubclassId = 0x42524443; // "BRDC"

    private readonly nint _windowHandle;
    private readonly AppWindow _appWindow;
    private readonly DispatcherQueue _dispatcherQueue;
    private readonly uint _callbackMessage;
    private readonly uint _taskbarCreatedMessage;
    private readonly SubclassProcedure _subclassProcedure;
    private RectInt32 _floatingBounds;
    private bool _restoreMaximized;
    private bool _registered;
    private bool _disposed;
    private bool _positionUpdateQueued;
    private bool _applyingPosition;

    internal DesktopDockManager(
        nint windowHandle,
        AppWindow appWindow,
        DispatcherQueue dispatcherQueue)
    {
        if (windowHandle == 0)
        {
            throw new ArgumentException("A desktop window handle is required.", nameof(windowHandle));
        }

        _windowHandle = windowHandle;
        _appWindow = appWindow;
        _dispatcherQueue = dispatcherQueue;
        _floatingBounds = new RectInt32(
            appWindow.Position.X,
            appWindow.Position.Y,
            appWindow.Size.Width,
            appWindow.Size.Height);
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

    internal DesktopDockSide Side { get; private set; }

    internal double WidthDips { get; private set; } = RecommendedWidthDips;

    internal bool IsDocked => Side is DesktopDockSide.Left or DesktopDockSide.Right;

    internal DesktopDockSnapshot CaptureSnapshot()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var monitor = MonitorFromWindow(_windowHandle, MonitorDefaultToNearest);
        var monitorInfo = new MonitorInfo
        {
            Size = (uint)Marshal.SizeOf<MonitorInfo>(),
        };
        var windowRectangle = new NativeRectangle();
        if (monitor == 0
            || !GetMonitorInfo(monitor, ref monitorInfo)
            || !GetWindowRect(_windowHandle, ref windowRectangle))
        {
            throw new Win32Exception(Marshal.GetLastWin32Error());
        }

        return new DesktopDockSnapshot(
            ToRectInt32(windowRectangle),
            ToRectInt32(monitorInfo.Monitor),
            ToRectInt32(monitorInfo.WorkArea));
    }

    internal void Dock(DesktopDockSide side, double widthDips)
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
            _floatingBounds = new RectInt32(
                _appWindow.Position.X,
                _appWindow.Position.Y,
                _appWindow.Size.Width,
                _appWindow.Size.Height);
            if (_appWindow.Presenter is OverlappedPresenter presenter)
            {
                _restoreMaximized = presenter.State == OverlappedPresenterState.Maximized;
                if (_restoreMaximized)
                {
                    presenter.Restore();
                }
            }
        }
        else if (changingSide)
        {
            RemoveAppBar();
        }

        Side = side;
        WidthDips = NormalizeRequestedWidth(widthDips);
        RegisterAppBar();
        ApplyDockPosition();
        SetPinnedPresenterState(true);
    }

    internal void ChangeWidth(double widthDips)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        WidthDips = NormalizeRequestedWidth(widthDips);
        if (IsDocked)
        {
            ApplyDockPosition();
        }
    }

    internal void Unpin(bool restoreFloatingWindow = true)
    {
        if (_disposed || !IsDocked)
        {
            return;
        }

        RemoveAppBar();
        Side = DesktopDockSide.None;
        SetPinnedPresenterState(false);
        if (!restoreFloatingWindow)
        {
            return;
        }

        _appWindow.MoveAndResize(_floatingBounds);
        if (_restoreMaximized
            && _appWindow.Presenter is OverlappedPresenter presenter)
        {
            presenter.Maximize();
        }
        _restoreMaximized = false;
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
        if (!_registered || !IsDocked || _applyingPosition)
        {
            return;
        }

        _applyingPosition = true;
        try
        {
            ApplyDockPositionCore();
        }
        finally
        {
            _applyingPosition = false;
        }
    }

    private void ApplyDockPositionCore()
    {

        var monitor = MonitorFromWindow(_windowHandle, MonitorDefaultToNearest);
        var monitorInfo = new MonitorInfo
        {
            Size = (uint)Marshal.SizeOf<MonitorInfo>(),
        };
        if (monitor == 0 || !GetMonitorInfo(monitor, ref monitorInfo))
        {
            throw new Win32Exception(
                Marshal.GetLastWin32Error(),
                "Windows could not identify the monitor used for docking.");
        }

        var dpi = Math.Max(96u, GetDpiForWindow(_windowHandle));
        var monitorWidth = monitorInfo.Monitor.Right - monitorInfo.Monitor.Left;
        var minimumPixels = (int)Math.Round(MinimumWidthDips * dpi / 96.0);
        var requestedPixels = (int)Math.Round(WidthDips * dpi / 96.0);
        var maximumPixels = Math.Max(
            minimumPixels,
            (int)Math.Round(monitorWidth * MaximumMonitorWidthFraction));
        var widthPixels = Math.Clamp(requestedPixels, minimumPixels, maximumPixels);

        var data = CreateAppBarData();
        data.Edge = Side == DesktopDockSide.Left ? AbeLeft : AbeRight;
        data.Rectangle = monitorInfo.Monitor;
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
        WidthDips = Math.Round(
            (bounds.Right - bounds.Left) * 96.0 / dpi,
            1,
            MidpointRounding.AwayFromZero);
    }

    private void SetPinnedPresenterState(bool pinned)
    {
        if (_appWindow.Presenter is not OverlappedPresenter presenter)
        {
            return;
        }

        presenter.IsResizable = !pinned;
        presenter.IsMaximizable = !pinned;
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
        if (message == _taskbarCreatedMessage)
        {
            // Explorer restarts forget every registered AppBar. Re-register
            // this still-running window and reserve its edge again.
            _registered = false;
            QueueDockPositionUpdate(registerFirst: true);
        }
        else if (!_applyingPosition
                 && ((message == _callbackMessage && (uint)wParam == AbnPosChanged)
                     || message is WmDisplayChange or WmDpiChanged or WmSettingChange))
        {
            QueueDockPositionUpdate(registerFirst: false);
        }

        return DefSubclassProc(windowHandle, message, wParam, lParam);
    }

    private void QueueDockPositionUpdate(bool registerFirst)
    {
        if (!IsDocked || _positionUpdateQueued)
        {
            return;
        }

        _positionUpdateQueued = true;
        if (!_dispatcherQueue.TryEnqueue(() =>
        {
            _positionUpdateQueued = false;
            if (_disposed || !IsDocked)
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

    private static double NormalizeRequestedWidth(double widthDips) =>
        double.IsFinite(widthDips)
            ? Math.Max(MinimumWidthDips, widthDips)
            : RecommendedWidthDips;

    private static RectInt32 ToRectInt32(NativeRectangle value) =>
        new(
            value.Left,
            value.Top,
            Math.Max(0, value.Right - value.Left),
            Math.Max(0, value.Bottom - value.Top));

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        RemoveAppBar();
        RemoveWindowSubclass(_windowHandle, _subclassProcedure, SubclassId);
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
    private struct AppBarData
    {
        internal uint Size;
        internal nint WindowHandle;
        internal uint CallbackMessage;
        internal uint Edge;
        internal NativeRectangle Rectangle;
        internal nint Parameter;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct MonitorInfo
    {
        internal uint Size;
        internal NativeRectangle Monitor;
        internal NativeRectangle WorkArea;
        internal uint Flags;
    }

    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    private delegate nint SubclassProcedure(
        nint windowHandle,
        uint message,
        nuint wParam,
        nint lParam,
        nuint subclassId,
        nuint referenceData);

    [DllImport("shell32.dll", SetLastError = true)]
    private static extern nuint SHAppBarMessage(uint message, ref AppBarData data);

    [DllImport("user32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
    private static extern uint RegisterWindowMessage(string message);

    [DllImport("user32.dll")]
    private static extern nint MonitorFromWindow(nint windowHandle, uint flags);

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
    private static extern bool GetMonitorInfo(nint monitor, ref MonitorInfo monitorInfo);

    [DllImport("user32.dll")]
    private static extern uint GetDpiForWindow(nint windowHandle);

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
