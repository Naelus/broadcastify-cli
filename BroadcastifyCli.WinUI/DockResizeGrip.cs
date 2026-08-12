using Microsoft.UI.Input;
using Microsoft.UI.Xaml.Controls;

namespace BroadcastifyCli.WinUI;

public sealed class DockResizeGrip : ContentControl
{
    public DockResizeGrip()
    {
        ProtectedCursor = InputSystemCursor.Create(
            InputSystemCursorShape.SizeWestEast);
    }
}
