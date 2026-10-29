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
$probeScript=Join-Path $probeRoot 'exit.ps1'
$record=Join-Path $probeRoot 'runs.txt'
@'
param([string]$Record)
Add-Content -LiteralPath $Record -Value $PID
if (@(Get-Content -LiteralPath $Record).Count -eq 1) { exit 73 }
exit 0
'@ | Set-Content -LiteralPath $probeScript
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
  # Exercise the shipped scheduler policy with a disposable executable instead
  # of any archive/model worker. First exit fails; its retry exits normally.
  $definition.Triggers.Clear()
  $definition.Actions.Clear()
  $action=$definition.Actions.Create(0)
  $action.Path=(Get-Command powershell.exe).Source
  $action.Arguments='-NoProfile -NonInteractive -WindowStyle Hidden -File "'+$probeScript+'" -Record "'+$record+'"'
  $action.WorkingDirectory=$probeRoot
  $probe=$folder.RegisterTaskDefinition($probeName,$definition,6,$null,$null,3,$null)
  $null=$probe.Run($null)
  $deadline=[datetime]::UtcNow.AddSeconds(100)
  do {
    $runs=@(Get-Content -LiteralPath $record -ErrorAction SilentlyContinue)
    if ($runs.Count -ge 2) { break }
    Start-Sleep -Milliseconds 500
  } while ([datetime]::UtcNow -lt $deadline)
  if ($runs.Count -ne 2) { throw 'Windows did not restart the failed independent task.' }
  # The minute is the behavior under test: normal close must not retry again.
  $deadline=[datetime]::UtcNow.AddSeconds(65)
  do {
    if (@(Get-Content -LiteralPath $record).Count -ne 2) { throw 'Normal close unexpectedly restarted the app.' }
    Start-Sleep -Milliseconds 500
  } while ([datetime]::UtcNow -lt $deadline)
  if ($folder.GetTask($probeName).LastTaskResult -ne 0) { throw 'The recovered process did not finish normally.' }
  Write-Output 'PASS: failed detached task restarted; normal close stayed closed; startup opt-out survived migration.'
}
finally {
  try { $folder.DeleteTask($probeName,0) } catch {}
  Configure-Startup remove
}
