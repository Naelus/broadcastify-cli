[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$Executable)
$ErrorActionPreference='Stop'
if ($env:GITHUB_ACTIONS -ine 'true') { throw 'This probe changes startup registration and is restricted to the isolated CI account.' }
$Executable=[IO.Path]::GetFullPath($Executable)
function Configure-Startup([string]$Mode) {
  $p=Start-Process -FilePath $Executable -ArgumentList "--configure-startup=$Mode" -WindowStyle Hidden -PassThru -Wait
  if ($p.ExitCode -ne 0) { throw "Startup configuration $Mode failed: $($p.ExitCode)" }
}
$service=New-Object -ComObject Schedule.Service
$service.Connect()
$folder=$service.GetFolder('\')
$probeName='Broadcastify recovery probe '+[guid]::NewGuid().ToString('N')
$probeRoot=Join-Path $env:RUNNER_TEMP $probeName
New-Item -ItemType Directory -Path $probeRoot | Out-Null
try {
  Configure-Startup enabled
  $actual=$folder.GetTask('Broadcastify Desktop')
  $definition=$actual.Definition
  if ($definition.Principal.LogonType -ne 3 -or $definition.Principal.RunLevel -ne 0) { throw 'Recovery must run as the signed-in user without elevation/password.' }
  if ($definition.Settings.ExecutionTimeLimit -ne 'PT0S') { throw 'Long catch-up must have no task time limit.' }
  if ($definition.Settings.Priority -ne 4) { throw 'Interactive catch-up must retain normal I/O priority.' }
  if ($definition.Settings.RestartCount -ne 3 -or $definition.Settings.RestartInterval -ne 'PT1M') { throw 'Crash recovery policy is missing.' }
  $legacyRun=[Microsoft.Win32.Registry]::CurrentUser.OpenSubKey('Software\Microsoft\Windows\CurrentVersion\Run')
  try {
    if ($legacyRun -and $legacyRun.GetValue('Broadcastify Desktop')) { throw 'Legacy startup would launch a duplicate.' }
  } finally { if ($legacyRun) { $legacyRun.Dispose() } }
  Configure-Startup disabled
  if ($folder.GetTask('Broadcastify Desktop').Enabled) { throw 'Disabling startup left recovery enabled.' }
  Configure-Startup preserve
  if ($folder.GetTask('Broadcastify Desktop').Enabled) { throw 'Upgrade did not preserve disabled startup.' }
  Configure-Startup remove
  # Windows accepted RestartOnFailure but did not retry a successfully started
  # process exiting nonzero in our live checks. Exercise the explicit shipped
  # supervisor with real disposable OS processes and no production backoff.
  $oldRoot=$env:BROADCASTIFY_DESKTOP_TEST_DATA_ROOT
  $oldIsolated=$env:BROADCASTIFY_DESKTOP_E2E_ISOLATED
  try {
    $env:BROADCASTIFY_DESKTOP_TEST_DATA_ROOT=$probeRoot
    $env:BROADCASTIFY_DESKTOP_E2E_ISOLATED='1'
    $process=Start-Process -FilePath $Executable -ArgumentList '--verify-desktop-recovery' -WindowStyle Hidden -Wait -PassThru
    if ($process.ExitCode -ne 0) { throw 'Desktop supervisor behavior probe failed.' }
    $report=Get-Content (Join-Path $probeRoot 'desktop-recovery-report.json') -Raw | ConvertFrom-Json
    if (!$report.passed) { throw 'Desktop supervisor behavior report did not pass.' }
    $report | ConvertTo-Json
  } finally {
    $env:BROADCASTIFY_DESKTOP_TEST_DATA_ROOT=$oldRoot
    $env:BROADCASTIFY_DESKTOP_E2E_ISOLATED=$oldIsolated
  }
  Write-Output 'PASS: failed desktop recovered; normal close stayed closed; failure retries bounded; startup opt-out preserved.'
}
finally {
  Configure-Startup remove
}
