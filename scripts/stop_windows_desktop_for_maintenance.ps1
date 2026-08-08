param(
    [ValidateRange(5, 600)]
    [int]$ShutdownTimeoutSeconds = 90
)

$ErrorActionPreference = "Stop"
$appExecutableName = "Broadcastify Desktop.exe"
$workerCommandPattern = "*broadcastify_cli.worker*"

$initialSnapshot = @(Get-CimInstance Win32_Process)
$appProcesses = @($initialSnapshot | Where-Object { $_.Name -ieq $appExecutableName })
$trackedProcessIds = [System.Collections.Generic.HashSet[int]]::new()

foreach ($appProcess in $appProcesses) {
    [void]$trackedProcessIds.Add([int]$appProcess.ProcessId)
}

do {
    $addedDescendant = $false
    foreach ($candidate in $initialSnapshot) {
        if ($trackedProcessIds.Contains([int]$candidate.ParentProcessId) -and
            $trackedProcessIds.Add([int]$candidate.ProcessId)) {
            $addedDescendant = $true
        }
    }
} while ($addedDescendant)

foreach ($appProcess in $appProcesses) {
    $process = Get-Process -Id $appProcess.ProcessId -ErrorAction SilentlyContinue
    if ($null -ne $process -and -not $process.HasExited) {
        [void]$process.CloseMainWindow()
    }
}

$deadline = [DateTime]::UtcNow.AddSeconds($ShutdownTimeoutSeconds)
$remaining = @()
do {
    $snapshot = @(Get-CimInstance Win32_Process)

    # Continue following descendants even if the UI process exits first.
    do {
        $addedDescendant = $false
        foreach ($candidate in $snapshot) {
            if ($trackedProcessIds.Contains([int]$candidate.ParentProcessId) -and
                $trackedProcessIds.Add([int]$candidate.ProcessId)) {
                $addedDescendant = $true
            }
        }
    } while ($addedDescendant)

    $remaining = @($snapshot | Where-Object {
        $trackedProcessIds.Contains([int]$_.ProcessId) -or
        $_.Name -ieq $appExecutableName -or
        ($_.CommandLine -and $_.CommandLine -like $workerCommandPattern)
    })

    if ($remaining.Count -eq 0) {
        break
    }

    Start-Sleep -Milliseconds 250
} while ([DateTime]::UtcNow -lt $deadline)

if ($remaining.Count -gt 0) {
    $details = ($remaining | ForEach-Object {
        "{0} (PID {1})" -f $_.Name, $_.ProcessId
    }) -join ", "
    throw (
        "Broadcastify Desktop did not finish its checkpointed shutdown within " +
        "$ShutdownTimeoutSeconds seconds. Still running: $details. " +
        "No process was force-killed; maintenance was refused."
    )
}

[pscustomobject]@{
    RequestedClose = $appProcesses.Count -gt 0
    ClosedAppCount = $appProcesses.Count
    TrackedProcessCount = $trackedProcessIds.Count
    CleanShutdown = $true
}
