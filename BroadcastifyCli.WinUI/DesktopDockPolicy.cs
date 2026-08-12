namespace BroadcastifyCli.WinUI;

internal static class DesktopDockPolicy
{
    internal const double MinimumWidthDips = 520;
    internal const double RecommendedWidthDips = 600;
    internal const double MaximumWidthDips = 960;

    internal static double NormalizeWidthDips(
        double requestedWidthDips,
        double displayWidthDips)
    {
        var availableWidth = double.IsFinite(displayWidthDips)
            ? Math.Max(1, displayWidthDips)
            : RecommendedWidthDips * 2;
        var maximum = Math.Max(
            Math.Min(MinimumWidthDips, availableWidth),
            Math.Min(MaximumWidthDips, availableWidth / 2));
        var minimum = Math.Min(MinimumWidthDips, maximum);
        var requested = double.IsFinite(requestedWidthDips)
            ? requestedWidthDips
            : RecommendedWidthDips;
        return Math.Round(
            Math.Clamp(requested, minimum, maximum),
            1,
            MidpointRounding.AwayFromZero);
    }

    internal static int DipsToPixels(double value, uint dpi) =>
        Math.Max(
            1,
            (int)Math.Round(
                value * NormalizeDpi(dpi) / 96.0,
                MidpointRounding.AwayFromZero));

    internal static double PixelsToDips(int value, uint dpi) =>
        Math.Round(
            Math.Max(1, value) * 96.0 / NormalizeDpi(dpi),
            1,
            MidpointRounding.AwayFromZero);

    private static uint NormalizeDpi(uint dpi) => Math.Max(96u, dpi);
}
