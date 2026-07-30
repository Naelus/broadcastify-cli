[CmdletBinding()]
param(
    [string]$OutputDirectory = "",
    [string]$Version = "",
    [string]$BuildPython = "",
    [string]$InnoCompiler = "",
    [string]$CacheDirectory = "",
    [switch]$BundleLocalEnv,
    [switch]$SkipInstaller
)

$ErrorActionPreference = "Stop"
$repositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$project = Join-Path $repositoryRoot "BroadcastifyCli.WinUI\BroadcastifyCli.WinUI.csproj"
$innoScript = Join-Path $repositoryRoot "installer\BroadcastifyDesktop.iss"
$constraints = Join-Path $repositoryRoot "installer\windows-runtime-constraints.txt"
$pythonVersion = "3.12.10"
$pythonSha256 = "4ACBED6DD1C744B0376E3B1CF57CE906F9DC9E95E68824584C8099A63025A3C3"
$pythonUrl = "https://www.python.org/ftp/python/$pythonVersion/python-$pythonVersion-embed-amd64.zip"
$ffmpegVersion = "8.1.2"
$ffmpegSha256 = "DB580001CAA24AC104C8CB856CD113A87B0A443F7BDF47D8C12B1D740584A2EC"
$ffmpegUrl = "https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-$ffmpegVersion-essentials_build.zip"

function Assert-ChildPath {
    param(
        [Parameter(Mandatory = $true)][string]$Parent,
        [Parameter(Mandatory = $true)][string]$Child
    )
    $resolvedParent = [System.IO.Path]::GetFullPath($Parent).TrimEnd(
        [System.IO.Path]::DirectorySeparatorChar,
        [System.IO.Path]::AltDirectorySeparatorChar
    ) + [System.IO.Path]::DirectorySeparatorChar
    $resolvedChild = [System.IO.Path]::GetFullPath($Child)
    if (-not $resolvedChild.StartsWith(
        $resolvedParent,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Refusing to mutate a path outside $resolvedParent`: $resolvedChild"
    }
    return $resolvedChild
}

function Get-ProjectVersion {
    $pyproject = Get-Content -LiteralPath (Join-Path $repositoryRoot "pyproject.toml") -Raw
    $match = [regex]::Match(
        $pyproject,
        '(?ms)^\[project\].*?^version\s*=\s*"(?<version>\d+\.\d+\.\d+(?:\.\d+)?)"'
    )
    if (-not $match.Success) {
        throw "Unable to read [project].version from pyproject.toml."
    }
    return $match.Groups["version"].Value
}

function Get-NumericVersion {
    param([Parameter(Mandatory = $true)][string]$Value)
    $parts = @($Value.Split("."))
    while ($parts.Count -lt 4) {
        $parts += "0"
    }
    return ($parts[0..3] -join ".")
}

function Get-VerifiedDownload {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter(Mandatory = $true)][string]$Destination,
        [Parameter(Mandatory = $true)][string]$Sha256
    )
    if (-not (Test-Path -LiteralPath $Destination -PathType Leaf)) {
        Invoke-WebRequest -UseBasicParsing -Uri $Uri -OutFile $Destination
    }
    $actual = (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash
    if ($actual -ne $Sha256) {
        throw "Checksum verification failed for $Destination."
    }
    return $Destination
}

function Resolve-BuildPython {
    if (-not [string]::IsNullOrWhiteSpace($BuildPython)) {
        return [System.IO.Path]::GetFullPath($BuildPython)
    }
    $venvPython = Join-Path $repositoryRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
        return $venvPython
    }
    $command = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }
    throw "A Python 3.12 build interpreter is required."
}

function Resolve-InnoCompiler {
    if (-not [string]::IsNullOrWhiteSpace($InnoCompiler)) {
        return [System.IO.Path]::GetFullPath($InnoCompiler)
    }
    $command = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }
    $candidates = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 7\ISCC.exe"),
        (Join-Path $env:ProgramFiles "Inno Setup 7\ISCC.exe"),
        (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 7\ISCC.exe")
    ) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    $found = $candidates | Where-Object {
        Test-Path -LiteralPath $_ -PathType Leaf
    } | Select-Object -First 1
    if ($found) {
        return [System.IO.Path]::GetFullPath($found)
    }
    throw "ISCC.exe was not found. Run scripts\install_inno_setup.ps1 first."
}

$projectVersion = Get-ProjectVersion
if ([string]::IsNullOrWhiteSpace($Version)) {
    $Version = $projectVersion
}
elseif ($Version -ne $projectVersion) {
    throw "Requested version $Version does not match pyproject.toml version $projectVersion."
}
if ($Version -notmatch '^\d+\.\d+\.\d+(?:\.\d+)?$') {
    throw "Version must contain three or four numeric components."
}
$numericVersion = Get-NumericVersion $Version
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $repositoryRoot "dist\windows"
}
if ([string]::IsNullOrWhiteSpace($CacheDirectory)) {
    $CacheDirectory = Join-Path $repositoryRoot ".tmp\installer-cache"
}
$output = [System.IO.Path]::GetFullPath($OutputDirectory)
$cache = [System.IO.Path]::GetFullPath($CacheDirectory)
$stageRoot = Assert-ChildPath `
    (Join-Path $repositoryRoot ".tmp") `
    (Join-Path $repositoryRoot ".tmp\windows-installer\$Version")
$application = Join-Path $stageRoot "app"
$pythonRoot = Join-Path $application "runtime\python"
$sitePackages = Join-Path $pythonRoot "Lib\site-packages"
$toolsRoot = Join-Path $application "runtime\tools"

if (Test-Path -LiteralPath $stageRoot) {
    Remove-Item -LiteralPath $stageRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $application, $output, $cache | Out-Null

$bundleValue = if ($BundleLocalEnv) { "true" } else { "false" }
& dotnet publish $project `
    -c Release `
    -r win-x64 `
    -o $application `
    "-p:BundleLocalEnv=$bundleValue" `
    "-p:Version=$Version"
if ($LASTEXITCODE -ne 0) {
    throw "dotnet publish failed with exit code $LASTEXITCODE."
}

$requiredDesktopFiles = @(
    "Broadcastify Desktop.exe",
    "Broadcastify Desktop.pri",
    "App.xbf",
    "MainWindow.xbf",
    "windowsml\BroadcastifyCli.WindowsML.exe"
)
foreach ($relativePath in $requiredDesktopFiles) {
    if (-not (Test-Path -LiteralPath (Join-Path $application $relativePath) -PathType Leaf)) {
        throw "The native publish is missing $relativePath."
    }
}

$privateEnvironment = Join-Path $application "broadcastify-desktop.env"
if ($BundleLocalEnv) {
    $sourceEnvironment = Join-Path $repositoryRoot ".env"
    if (-not (Test-Path -LiteralPath $sourceEnvironment -PathType Leaf)) {
        throw "BundleLocalEnv was requested, but the ignored repository .env is missing."
    }
    if (-not (Test-Path -LiteralPath $privateEnvironment -PathType Leaf)) {
        throw "The private application stage does not contain broadcastify-desktop.env."
    }
    if ((Get-FileHash $sourceEnvironment -Algorithm SHA256).Hash -ne
        (Get-FileHash $privateEnvironment -Algorithm SHA256).Hash) {
        throw "The private application stage does not contain the expected ignored .env."
    }
}
elseif (Test-Path -LiteralPath $privateEnvironment) {
    throw "A public installer stage contains broadcastify-desktop.env."
}

$pythonArchive = Get-VerifiedDownload `
    -Uri $pythonUrl `
    -Destination (Join-Path $cache "python-$pythonVersion-embed-amd64.zip") `
    -Sha256 $pythonSha256
New-Item -ItemType Directory -Force -Path $pythonRoot, $sitePackages | Out-Null
Expand-Archive -LiteralPath $pythonArchive -DestinationPath $pythonRoot -Force
@(
    "python312.zip",
    ".",
    "Lib\site-packages",
    "import site"
) | Set-Content -LiteralPath (Join-Path $pythonRoot "python312._pth") -Encoding ASCII

$builder = Resolve-BuildPython
$builderVersion = & $builder -c "import sys; print('.'.join(map(str, sys.version_info[:3])))"
if ($LASTEXITCODE -ne 0 -or $builderVersion -notmatch '^3\.12\.') {
    throw "The build interpreter must be Python 3.12; found $builderVersion."
}
$packageTarget = "$repositoryRoot[windowsml,qwen,portable-diarization]"
& $builder -m pip install `
    --disable-pip-version-check `
    --no-input `
    --ignore-installed `
    --upgrade `
    --no-compile `
    --constraint $constraints `
    --target $sitePackages `
    $packageTarget
if ($LASTEXITCODE -ne 0) {
    throw "Installing the portable Python worker runtime failed with exit code $LASTEXITCODE."
}

Get-ChildItem -LiteralPath $sitePackages -Filter direct_url.json -Recurse -File |
    Remove-Item -Force
$prunableRuntimeTests = @(
    (Join-Path $sitePackages "onnx\backend\test"),
    (Join-Path $sitePackages "onnx\test"),
    (Join-Path $sitePackages "numpy\tests")
)
foreach ($runtimeTestPath in $prunableRuntimeTests) {
    if (Test-Path -LiteralPath $runtimeTestPath) {
        $verifiedRuntimeTestPath = Assert-ChildPath $sitePackages $runtimeTestPath
        Remove-Item -LiteralPath $verifiedRuntimeTestPath -Recurse -Force
    }
}
Get-ChildItem -LiteralPath $sitePackages -Filter __pycache__ -Recurse -Directory |
    Sort-Object FullName -Descending |
    Remove-Item -Recurse -Force

$ffmpegArchive = Get-VerifiedDownload `
    -Uri $ffmpegUrl `
    -Destination (Join-Path $cache "ffmpeg-$ffmpegVersion-essentials_build.zip") `
    -Sha256 $ffmpegSha256
$ffmpegExtract = Join-Path $stageRoot "ffmpeg"
New-Item -ItemType Directory -Force -Path $ffmpegExtract, $toolsRoot | Out-Null
Expand-Archive -LiteralPath $ffmpegArchive -DestinationPath $ffmpegExtract -Force
$ffmpeg = Get-ChildItem -LiteralPath $ffmpegExtract -Filter ffmpeg.exe -Recurse -File |
    Select-Object -First 1
$ffprobe = Get-ChildItem -LiteralPath $ffmpegExtract -Filter ffprobe.exe -Recurse -File |
    Select-Object -First 1
if (-not $ffmpeg -or -not $ffprobe) {
    throw "The verified FFmpeg archive did not contain ffmpeg.exe and ffprobe.exe."
}
Copy-Item -LiteralPath $ffmpeg.FullName -Destination (Join-Path $toolsRoot "ffmpeg.exe")
Copy-Item -LiteralPath $ffprobe.FullName -Destination (Join-Path $toolsRoot "ffprobe.exe")

Copy-Item -LiteralPath (Join-Path $repositoryRoot "LICENSE") -Destination $application
Copy-Item -LiteralPath (
    Join-Path $repositoryRoot "BroadcastifyCli.WinUI\Assets\BroadcastifyDesktop.ico"
) -Destination (Join-Path $application "BroadcastifyDesktop.ico")
@"
Bundled runtime notices
=======================

Python ${pythonVersion}: https://www.python.org/
FFmpeg ${ffmpegVersion} essentials build: https://www.gyan.dev/ffmpeg/builds/
FFmpeg source revision: https://github.com/FFmpeg/FFmpeg/commit/38b88335f9

FFmpeg's bundled Windows essentials build is GPLv3. The application's GPLv3
license is included as LICENSE. Python package license files remain alongside
their installed package metadata under runtime\python\Lib\site-packages.
"@ | Set-Content -LiteralPath (Join-Path $application "THIRD-PARTY-NOTICES.txt") -Encoding UTF8

$embeddedPython = Join-Path $pythonRoot "python.exe"
$oldNoUserSite = $env:PYTHONNOUSERSITE
$oldNoBytecode = $env:PYTHONDONTWRITEBYTECODE
$oldFfmpeg = $env:FFMPEG_PATH
try {
    $env:PYTHONNOUSERSITE = "1"
    $env:PYTHONDONTWRITEBYTECODE = "1"
    $env:FFMPEG_PATH = Join-Path $toolsRoot "ffmpeg.exe"
    & $embeddedPython -B -c (
        "import broadcastify_cli, cryptography, requests, sherpa_onnx; " +
        "import onnxruntime_genai; print('portable-runtime-ready')"
    )
    if ($LASTEXITCODE -ne 0) {
        throw "The embedded Python import smoke test failed with exit code $LASTEXITCODE."
    }
    & (Join-Path $toolsRoot "ffmpeg.exe") -hide_banner -version | Select-Object -First 1
    if ($LASTEXITCODE -ne 0) {
        throw "The bundled FFmpeg smoke test failed with exit code $LASTEXITCODE."
    }
}
finally {
    $env:PYTHONNOUSERSITE = $oldNoUserSite
    $env:PYTHONDONTWRITEBYTECODE = $oldNoBytecode
    $env:FFMPEG_PATH = $oldFfmpeg
}
Get-ChildItem -LiteralPath $sitePackages -Filter __pycache__ -Recurse -Directory |
    Sort-Object FullName -Descending |
    Remove-Item -Recurse -Force

$manifest = [ordered]@{
    schema_version = 1
    app_version = $Version
    architecture = "win-x64"
    python = [ordered]@{
        version = $pythonVersion
        archive_sha256 = $pythonSha256
        constraints_sha256 = (
            Get-FileHash -LiteralPath $constraints -Algorithm SHA256
        ).Hash
    }
    ffmpeg = [ordered]@{
        version = $ffmpegVersion
        archive_sha256 = $ffmpegSha256
        source_revision = "38b88335f9"
    }
    private_environment = [bool]$BundleLocalEnv
    built_utc = [DateTime]::UtcNow.ToString("o")
}
$manifest | ConvertTo-Json -Depth 5 |
    Set-Content -LiteralPath (Join-Path $application "build-manifest.json") -Encoding UTF8

if ($SkipInstaller) {
    [pscustomobject]@{
        version = $Version
        application_directory = $application
        installer = $null
        private_environment = [bool]$BundleLocalEnv
        embedded_python = $embeddedPython
        bundled_ffmpeg = (Join-Path $toolsRoot "ffmpeg.exe")
    }
    return
}

$compiler = Resolve-InnoCompiler
& $compiler `
    "/DMyAppVersion=$Version" `
    "/DMyAppVersionNumeric=$numericVersion" `
    "/DSourceDir=$application" `
    "/DOutputDir=$output" `
    $innoScript
if ($LASTEXITCODE -ne 0) {
    throw "Inno Setup compilation failed with exit code $LASTEXITCODE."
}
$installer = Join-Path $output "BroadcastifyDesktop-$Version-win-x64-setup.exe"
if (-not (Test-Path -LiteralPath $installer -PathType Leaf)) {
    throw "Inno Setup did not produce the expected installer: $installer"
}

[pscustomobject]@{
    version = $Version
    application_directory = $application
    installer = $installer
    installer_sha256 = (Get-FileHash -LiteralPath $installer -Algorithm SHA256).Hash
    installer_bytes = (Get-Item -LiteralPath $installer).Length
    private_environment = [bool]$BundleLocalEnv
    embedded_python = $embeddedPython
    bundled_ffmpeg = (Join-Path $toolsRoot "ffmpeg.exe")
}
