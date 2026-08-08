param(
    [Parameter(Mandatory = $true)]
    [string]$Installer,

    [ValidateRange(5, 600)]
    [int]$ShutdownTimeoutSeconds = 90,

    [switch]$Restart
)

$ErrorActionPreference = "Stop"
$installerPath = (Resolve-Path -LiteralPath $Installer).Path
if (-not (Test-Path -LiteralPath $installerPath -PathType Leaf)) {
    throw "Installer is not a file: $installerPath"
}

$stopScript = Join-Path $PSScriptRoot "stop_windows_desktop_for_maintenance.ps1"
$shutdown = & $stopScript -ShutdownTimeoutSeconds $ShutdownTimeoutSeconds

$installerArguments = @(
    "/VERYSILENT",
    "/SUPPRESSMSGBOXES",
    "/NORESTART",
    "/SP-"
)
$install = Start-Process `
    -FilePath $installerPath `
    -ArgumentList $installerArguments `
    -Wait `
    -PassThru `
    -WindowStyle Hidden
if ($install.ExitCode -ne 0) {
    throw "Broadcastify Desktop installer exited with code $($install.ExitCode)."
}

$installedExecutable = Join-Path `
    $env:LOCALAPPDATA `
    "Programs\Broadcastify Desktop\Broadcastify Desktop.exe"
if (-not (Test-Path -LiteralPath $installedExecutable -PathType Leaf)) {
    throw "The installer completed, but the installed executable was not found."
}

$installedVersion = (Get-Item -LiteralPath $installedExecutable).VersionInfo.ProductVersion
$restartedApp = $false
if ($Restart) {
    Start-Process -FilePath $installedExecutable
    $restartedApp = $true
}

[pscustomobject]@{
    Installer = $installerPath
    InstalledExecutable = $installedExecutable
    InstalledVersion = $installedVersion
    CleanShutdown = $shutdown.CleanShutdown
    Restarted = $restartedApp
}
