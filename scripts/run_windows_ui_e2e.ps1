[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Executable,
    [int]$TimeoutSeconds = 240,
    [switch]$KeepArtifacts
)

$ErrorActionPreference = "Stop"
$Executable = [System.IO.Path]::GetFullPath($Executable)
if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
    throw "Broadcastify Desktop executable not found: $Executable"
}
if ($TimeoutSeconds -lt 30) {
    throw "TimeoutSeconds must be at least 30."
}

$temporaryRoot = [System.IO.Path]::GetFullPath(
    [System.IO.Path]::GetTempPath()
).TrimEnd([System.IO.Path]::DirectorySeparatorChar)
$sessionRoot = Join-Path $temporaryRoot (
    "broadcastify-desktop-ui-e2e-" + [Guid]::NewGuid().ToString("N")
)
$sessionRoot = [System.IO.Path]::GetFullPath($sessionRoot)
$expectedPrefix = $temporaryRoot + [System.IO.Path]::DirectorySeparatorChar
if (-not $sessionRoot.StartsWith(
        $expectedPrefix,
        [System.StringComparison]::OrdinalIgnoreCase
    ) -or -not ([System.IO.Path]::GetFileName($sessionRoot)).StartsWith(
        "broadcastify-desktop-ui-e2e-",
        [System.StringComparison]::Ordinal
    )) {
    throw "Refusing to create an E2E session outside the system temporary directory."
}

$dataRoot = Join-Path $sessionRoot "app-data"
$libraryRoot = Join-Path $sessionRoot "library"
$reportPath = Join-Path $sessionRoot "ui-e2e-report.json"
New-Item -ItemType Directory -Path $dataRoot, $libraryRoot -Force | Out-Null

$settings = [ordered]@{
    Version = 9
    OutputDirectory = $libraryRoot
    LanSyncEnabled = $false
    LanDiscoveryEnabled = $false
    LanShareEnabled = $false
    DesktopDockSide = "right"
    DesktopDockWidth = 640
}
$settings | ConvertTo-Json -Depth 4 |
    Set-Content -LiteralPath (Join-Path $dataRoot "settings.json") -Encoding UTF8

$process = $null
$passed = $false
$abruptCleanupPassed = $false
function Get-DesktopWorkAreaSignature {
    Add-Type -AssemblyName System.Windows.Forms
    return [string]::Join(
        "|",
        @([System.Windows.Forms.Screen]::AllScreens |
            Sort-Object DeviceName |
            ForEach-Object {
                $area = $_.WorkingArea
                "$($_.DeviceName):$($area.X),$($area.Y),$($area.Width),$($area.Height)"
            })
    )
}

function New-IsolatedProcessStartInfo {
    param(
        [Parameter(Mandatory = $true)][string]$DataDirectory,
        [Parameter(Mandatory = $true)][string]$LibraryDirectory,
        [Parameter(Mandatory = $true)][string]$Report,
        [switch]$AbruptExit
    )
    $info = [System.Diagnostics.ProcessStartInfo]::new()
    $info.FileName = $Executable
    $info.Arguments = "--ui-e2e-report=`"$Report`""
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $false
    $info.EnvironmentVariables[
        "BROADCASTIFY_DESKTOP_TEST_DATA_ROOT"
    ] = $DataDirectory
    $info.EnvironmentVariables["BROADCASTIFY_LIBRARY_ROOT"] = $LibraryDirectory
    $info.EnvironmentVariables[
        "BROADCASTIFY_SECURE_ANALYSIS_DB"
    ] = Join-Path $LibraryDirectory "broadcastify-analysis.sqlite3"
    $info.EnvironmentVariables[
        "BROADCASTIFY_QUOTA_LEDGER"
    ] = Join-Path $DataDirectory "archive-quota.sqlite3"
    if ($AbruptExit) {
        $info.EnvironmentVariables[
            "BROADCASTIFY_DESKTOP_TEST_ABRUPT_EXIT"
        ] = "1"
    }
    foreach ($name in @(
            "BROADCASTIFY_USERNAME",
            "BROADCASTIFY_PASSWORD",
            "BROADCASTIFY_SECURE_USERNAME",
            "BROADCASTIFY_SECURE_PASSWORD",
            "HUGGINGFACE_TOKEN",
            "HUGGINGFACE_SECURE_TOKEN",
            "HF_TOKEN",
            "OPENAI_API_KEY",
            "BROADCASTIFY_ANALYSIS_API_KEY",
            "BROADCASTIFY_CREDENTIAL_STORE",
            "BROADCASTIFY_ENV_FILE"
        )) {
        $info.EnvironmentVariables[$name] = ""
    }
    $info.EnvironmentVariables["BROADCASTIFY_DESKTOP_E2E_ISOLATED"] = "1"
    return $info
}

$startInfo = New-IsolatedProcessStartInfo `
    -DataDirectory $dataRoot `
    -LibraryDirectory $libraryRoot `
    -Report $reportPath
try {
    $process = [System.Diagnostics.Process]::Start($startInfo)
    if ($null -eq $process) {
        throw "The isolated Broadcastify Desktop E2E process did not start."
    }

    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    while (-not $process.WaitForExit(1000)) {
        if ([DateTime]::UtcNow -ge $deadline) {
            try {
                $process.Kill()
            }
            catch {
                # The process may have exited between the timeout and Kill.
            }
            throw "The UI E2E probe exceeded $TimeoutSeconds seconds. Artifacts: $sessionRoot"
        }
    }
    $process.WaitForExit()

    if (-not (Test-Path -LiteralPath $reportPath -PathType Leaf)) {
        throw (
            "The UI E2E process exited with code $($process.ExitCode) without a report. " +
            "Artifacts: $sessionRoot"
        )
    }
    $report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
    if ($report.passed -ne $true) {
        throw (
            "The UI E2E probe failed: $($report.error) " +
            "Report: $reportPath"
        )
    }
    if ($process.ExitCode -ne 0) {
        throw (
            "The UI E2E report passed but the process exited with code " +
            "$($process.ExitCode). Report: $reportPath"
        )
    }

    $beforeAbruptExit = Get-DesktopWorkAreaSignature
    $abruptDataRoot = Join-Path $sessionRoot "abrupt-app-data"
    $abruptLibraryRoot = Join-Path $sessionRoot "abrupt-library"
    $abruptReportPath = Join-Path $sessionRoot "abrupt-ui-e2e-report.json"
    New-Item -ItemType Directory `
        -Path $abruptDataRoot, $abruptLibraryRoot `
        -Force | Out-Null
    $abruptSettings = [ordered]@{
        Version = 9
        OutputDirectory = $abruptLibraryRoot
        LanSyncEnabled = $false
        LanDiscoveryEnabled = $false
        LanShareEnabled = $false
        DesktopDockSide = "left"
        DesktopDockWidth = 620
    }
    $abruptSettings | ConvertTo-Json -Depth 4 |
        Set-Content `
            -LiteralPath (Join-Path $abruptDataRoot "settings.json") `
            -Encoding UTF8
    $abruptStartInfo = New-IsolatedProcessStartInfo `
        -DataDirectory $abruptDataRoot `
        -LibraryDirectory $abruptLibraryRoot `
        -Report $abruptReportPath `
        -AbruptExit
    $abruptProcess = [System.Diagnostics.Process]::Start($abruptStartInfo)
    if ($null -eq $abruptProcess) {
        throw "The abrupt AppBar cleanup process did not start."
    }
    try {
        if (-not $abruptProcess.WaitForExit(30000)) {
            try { $abruptProcess.Kill() } catch {}
            throw "The abrupt AppBar cleanup process did not exit."
        }
        $abruptProcess.WaitForExit()
        if ($abruptProcess.ExitCode -ne 73) {
            throw (
                "The abrupt AppBar cleanup process exited with code " +
                "$($abruptProcess.ExitCode), expected 73."
            )
        }
        if (-not (Test-Path -LiteralPath $abruptReportPath -PathType Leaf)) {
            throw "The abrupt AppBar cleanup process did not write its ready report."
        }
        $abruptReport = Get-Content -LiteralPath $abruptReportPath -Raw |
            ConvertFrom-Json
        if ($abruptReport.passed -ne $true `
                -or $abruptReport.abrupt_exit_ready -ne $true) {
            throw "The abrupt AppBar cleanup process was not docked before exiting."
        }
    }
    finally {
        $abruptProcess.Dispose()
    }
    $cleanupDeadline = [DateTime]::UtcNow.AddSeconds(8)
    do {
        $afterAbruptExit = Get-DesktopWorkAreaSignature
        if ($afterAbruptExit -eq $beforeAbruptExit) {
            $abruptCleanupPassed = $true
            break
        }
        Start-Sleep -Milliseconds 100
    } while ([DateTime]::UtcNow -lt $cleanupDeadline)
    if (-not $abruptCleanupPassed) {
        throw (
            "Windows did not restore every monitor work area after abrupt exit. " +
            "Before: $beforeAbruptExit After: $afterAbruptExit"
        )
    }

    $passed = $true
    [pscustomobject]@{
        status = "passed"
        executable = $Executable
        version = $report.version
        layout_sizes = @($report.layout_matrix).Count
        docking = "startup, left, right, border, resize, monitor, unpin, abrupt cleanup"
        abrupt_cleanup = $abruptCleanupPassed
        report = if ($KeepArtifacts) { $reportPath } else { "cleaned" }
    }
}
finally {
    if ($null -ne $process) {
        $process.Dispose()
    }
    if ($passed -and -not $KeepArtifacts) {
        $resolvedSession = [System.IO.Path]::GetFullPath($sessionRoot)
        if ($resolvedSession.StartsWith(
                $expectedPrefix,
                [System.StringComparison]::OrdinalIgnoreCase
            ) -and ([System.IO.Path]::GetFileName($resolvedSession)).StartsWith(
                "broadcastify-desktop-ui-e2e-",
                [System.StringComparison]::Ordinal
            )) {
            Remove-Item -LiteralPath $resolvedSession -Recurse -Force
        }
        else {
            throw "Refusing to clean an unverified UI E2E path: $resolvedSession"
        }
    }
    elseif (-not $passed) {
        Write-Warning "UI E2E artifacts were retained at $sessionRoot"
    }
}
