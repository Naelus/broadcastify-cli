[CmdletBinding()]
param(
    [string]$Executable = "",
    [string]$SessionRoot = "",
    [string]$FeedId = "999991"
)

$ErrorActionPreference = "Stop"
if (-not $FeedId -or $FeedId -notmatch '^\d+$') {
    throw "FeedId must contain only digits."
}
if ([string]::IsNullOrWhiteSpace($Executable)) {
    $Executable = Join-Path $env:LOCALAPPDATA (
        "Programs\Broadcastify Desktop\Broadcastify Desktop.exe"
    )
}
$Executable = [System.IO.Path]::GetFullPath($Executable)
if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
    throw "Installed desktop executable not found: $Executable"
}
if ([string]::IsNullOrWhiteSpace($SessionRoot)) {
    $SessionRoot = Join-Path (
        [System.IO.Path]::GetTempPath()
    ) ("broadcastify-desktop-ui-smoke-" + [Guid]::NewGuid().ToString("N"))
}
$SessionRoot = [System.IO.Path]::GetFullPath($SessionRoot)
if (Test-Path -LiteralPath $SessionRoot) {
    throw "The isolated UI smoke root already exists: $SessionRoot"
}

$dataRoot = Join-Path $SessionRoot "app-data"
$libraryRoot = Join-Path $SessionRoot "library"
$today = [DateTime]::Today
$startDate = $today.AddDays(-9)
$dayRoot = Join-Path (
    (Join-Path $libraryRoot $FeedId)
) $startDate.ToString("yyyyMMdd")
New-Item -ItemType Directory -Path $dataRoot, $dayRoot -Force | Out-Null

$settings = [ordered]@{
    Version = 9
    OutputDirectory = $libraryRoot
    LanSyncEnabled = $false
    LanDiscoveryEnabled = $false
    LanShareEnabled = $false
    LastReviewFeedId = $FeedId
    LastReviewDate = $startDate.ToString("yyyy-MM-dd")
}
$settings | ConvertTo-Json -Depth 3 |
    Set-Content -LiteralPath (Join-Path $dataRoot "settings.json") -Encoding UTF8

$installRoot = Split-Path -Parent $Executable
$python = Join-Path $installRoot "runtime\python\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Installed portable Python runtime not found: $python"
}
$database = Join-Path $libraryRoot "broadcastify-analysis.sqlite3"
$ledger = Join-Path $dataRoot "archive-quota.sqlite3"
$payload = [ordered]@{
    feed_id = $FeedId
    feed_name = "Isolated UI smoke feed"
    start_date = $startDate.ToString("yyyy-MM-dd")
    through_current = $true
} | ConvertTo-Json -Compress

$savedEnvironment = @{}
$workerEnvironment = [ordered]@{
    BROADCASTIFY_DESKTOP_TEST_DATA_ROOT = $dataRoot
    BROADCASTIFY_LIBRARY_ROOT = $libraryRoot
    BROADCASTIFY_SECURE_ANALYSIS_DB = $database
    BROADCASTIFY_QUOTA_LEDGER = $ledger
}
try {
    foreach ($entry in $workerEnvironment.GetEnumerator()) {
        $savedEnvironment[$entry.Key] = [Environment]::GetEnvironmentVariable(
            $entry.Key,
            [EnvironmentVariableTarget]::Process
        )
        [Environment]::SetEnvironmentVariable(
            $entry.Key,
            $entry.Value,
            [EnvironmentVariableTarget]::Process
        )
    }
    $workerOutput = $payload |
        & $python -m broadcastify_cli.worker save-library-catch-up 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the isolated saved catch-up fixture: $workerOutput"
    }
}
finally {
    foreach ($entry in $savedEnvironment.GetEnumerator()) {
        [Environment]::SetEnvironmentVariable(
            $entry.Key,
            $entry.Value,
            [EnvironmentVariableTarget]::Process
        )
    }
}

$startInfo = [System.Diagnostics.ProcessStartInfo]::new()
$startInfo.FileName = $Executable
$startInfo.UseShellExecute = $false
foreach ($entry in $workerEnvironment.GetEnumerator()) {
    $startInfo.Environment[$entry.Key] = $entry.Value
}
$process = [System.Diagnostics.Process]::Start($startInfo)
if ($null -eq $process) {
    throw "The isolated desktop smoke process did not start."
}

$session = [ordered]@{
    schema_version = 1
    process_id = $process.Id
    executable = $Executable
    session_root = $SessionRoot
    data_root = $dataRoot
    library_root = $libraryRoot
    database = $database
    feed_id = $FeedId
    catch_up_start_date = $startDate.ToString("yyyy-MM-dd")
    catch_up_end_date = $today.ToString("yyyy-MM-dd")
    catch_up_through_current = $true
}
$session | ConvertTo-Json -Depth 3 |
    Set-Content -LiteralPath (Join-Path $SessionRoot "smoke-session.json") -Encoding UTF8
$session | ConvertTo-Json -Depth 3
