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
    Version = 8
    OutputDirectory = $libraryRoot
    LanSyncEnabled = $false
    LanDiscoveryEnabled = $false
    LanShareEnabled = $false
    DesktopDockSide = "none"
    DesktopDockWidth = 600
}
$settings | ConvertTo-Json -Depth 4 |
    Set-Content -LiteralPath (Join-Path $dataRoot "settings.json") -Encoding UTF8

$startInfo = [System.Diagnostics.ProcessStartInfo]::new()
$startInfo.FileName = $Executable
$startInfo.Arguments = "--ui-e2e-report=`"$reportPath`""
$startInfo.UseShellExecute = $false
$startInfo.CreateNoWindow = $false
$startInfo.EnvironmentVariables[
    "BROADCASTIFY_DESKTOP_TEST_DATA_ROOT"
] = $dataRoot
$startInfo.EnvironmentVariables["BROADCASTIFY_LIBRARY_ROOT"] = $libraryRoot
$startInfo.EnvironmentVariables[
    "BROADCASTIFY_SECURE_ANALYSIS_DB"
] = Join-Path $libraryRoot "broadcastify-analysis.sqlite3"
$startInfo.EnvironmentVariables[
    "BROADCASTIFY_QUOTA_LEDGER"
] = Join-Path $dataRoot "archive-quota.sqlite3"
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
    $startInfo.EnvironmentVariables[$name] = ""
}

$process = $null
$passed = $false
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

    $passed = $true
    [pscustomobject]@{
        status = "passed"
        executable = $Executable
        version = $report.version
        layout_sizes = @($report.layout_matrix).Count
        docking = "left, right, resize, unpin"
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
