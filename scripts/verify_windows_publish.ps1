[CmdletBinding()]
param(
    [string]$OutputDirectory = "",
    [switch]$BundleLocalEnv
)

$ErrorActionPreference = "Stop"
$repositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$project = Join-Path $repositoryRoot "BroadcastifyCli.WinUI\BroadcastifyCli.WinUI.csproj"
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $flavor = if ($BundleLocalEnv) { "private" } else { "normal" }
    $OutputDirectory = Join-Path $repositoryRoot ".tmp\verified-publish-$flavor"
}
$publishDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
$bundleValue = if ($BundleLocalEnv) { "true" } else { "false" }

$arguments = @(
    "publish",
    $project,
    "-c", "Release",
    "-r", "win-x64",
    "-o", $publishDirectory,
    "-p:BundleLocalEnv=$bundleValue"
)
& dotnet @arguments
if ($LASTEXITCODE -ne 0) {
    throw "dotnet publish failed with exit code $LASTEXITCODE."
}

$required = @(
    "Broadcastify Desktop.exe",
    "Broadcastify Desktop.pri",
    "App.xbf",
    "MainWindow.xbf",
    "windowsml\BroadcastifyCli.WindowsML.exe",
    "windowsml\BroadcastifyCli.WindowsML.runtimeconfig.json",
    "windowsml\onnxruntime-genai.dll"
)
foreach ($relativePath in $required) {
    $candidate = Join-Path $publishDirectory $relativePath
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        throw "Published runtime is missing $relativePath."
    }
}

$publishedEnvironment = Join-Path $publishDirectory "broadcastify-desktop.env"
if ($BundleLocalEnv) {
    $sourceEnvironment = Join-Path $repositoryRoot ".env"
    if (-not (Test-Path -LiteralPath $sourceEnvironment -PathType Leaf)) {
        throw "BundleLocalEnv was requested, but the ignored repository .env is missing."
    }
    if (-not (Test-Path -LiteralPath $publishedEnvironment -PathType Leaf)) {
        throw "The private publish is missing broadcastify-desktop.env."
    }
    $sourceHash = (Get-FileHash -LiteralPath $sourceEnvironment -Algorithm SHA256).Hash
    $publishedHash = (Get-FileHash -LiteralPath $publishedEnvironment -Algorithm SHA256).Hash
    if ($sourceHash -ne $publishedHash) {
        throw "The private publish environment does not match the ignored source file."
    }
}
elseif (Test-Path -LiteralPath $publishedEnvironment) {
    throw "A normal publish contains the private environment file."
}

$helper = Join-Path $publishDirectory "windowsml\BroadcastifyCli.WindowsML.exe"
$probeText = & $helper --probe
if ($LASTEXITCODE -ne 0) {
    throw "The published Windows ML helper probe failed with exit code $LASTEXITCODE."
}
$probe = $probeText | ConvertFrom-Json
if ($probe.ready -ne $true -or $probe.architecture -ne "x64") {
    throw "The published Windows ML helper did not report an x64 ready runtime."
}

[pscustomobject]@{
    publish_directory = $publishDirectory
    private_environment = [bool]$BundleLocalEnv
    desktop_executable = (Join-Path $publishDirectory "Broadcastify Desktop.exe")
    windows_ml_helper = $helper
    windows_ml_probe_ready = [bool]$probe.ready
    windows_ml_decode_ready = [bool]$probe.decode_ready
    source_tree_required = $true
}
