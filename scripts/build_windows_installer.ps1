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
$cudaRequirementsSource = Join-Path $repositoryRoot "installer\windows-managed-cuda-lock.txt"
$publicReleaseScanner = Join-Path $repositoryRoot "scripts\scan_public_release.py"
$packagingPreflight = Join-Path $repositoryRoot "scripts\verify_windows_packaging_preflight.ps1"
$pythonVersion = "3.12.10"
$pythonSha256 = "4ACBED6DD1C744B0376E3B1CF57CE906F9DC9E95E68824584C8099A63025A3C3"
$pythonUrl = "https://www.python.org/ftp/python/$pythonVersion/python-$pythonVersion-embed-amd64.zip"
$ffmpegVersion = "8.1.2"
$ffmpegSha256 = "DB580001CAA24AC104C8CB856CD113A87B0A443F7BDF47D8C12B1D740584A2EC"
$ffmpegUrl = "https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-$ffmpegVersion-essentials_build.zip"
$uvVersion = "0.12.1"
$uvWheelSha256 = "BD02F2DA212E6A983115DC64A6FC94E9256C2D60E056D6B669DE0A6025AAEC05"
$uvWheelUrl = "https://files.pythonhosted.org/packages/0d/a4/467c99c76fefa8b1259a1d382a5e49f73068f38a2d58db401504a783ed2c/uv-0.12.1-py3-none-win_amd64.whl"

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

function Assert-CommittedBuildSource {
    $guard = Join-Path $repositoryRoot "scripts\assert_committed_build_source.ps1"
    $head = (& $guard -RepositoryRoot $repositoryRoot).Trim()
    if ($LASTEXITCODE -ne 0 -or $head -notmatch '^[0-9a-f]{40}$') {
        throw "The committed-source build guard failed."
    }
    return $head
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

$sourceCommit = Assert-CommittedBuildSource
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
    $defaultOutputDirectory = if ($BundleLocalEnv) {
        "dist\windows-private"
    }
    else {
        "dist\windows"
    }
    $OutputDirectory = Join-Path $repositoryRoot $defaultOutputDirectory
}
if ([string]::IsNullOrWhiteSpace($CacheDirectory)) {
    $CacheDirectory = Join-Path $repositoryRoot ".tmp\installer-cache"
}
$output = [System.IO.Path]::GetFullPath($OutputDirectory)
$publicOutput = [System.IO.Path]::GetFullPath(
    (Join-Path $repositoryRoot "dist\windows")
).TrimEnd(
    [System.IO.Path]::DirectorySeparatorChar,
    [System.IO.Path]::AltDirectorySeparatorChar
)
$normalizedOutput = $output.TrimEnd(
    [System.IO.Path]::DirectorySeparatorChar,
    [System.IO.Path]::AltDirectorySeparatorChar
)
$publicOutputPrefix = $publicOutput + [System.IO.Path]::DirectorySeparatorChar
if ($BundleLocalEnv -and (
        $normalizedOutput.Equals(
            $publicOutput,
            [StringComparison]::OrdinalIgnoreCase
        ) -or
        $normalizedOutput.StartsWith(
            $publicOutputPrefix,
            [StringComparison]::OrdinalIgnoreCase
        ))) {
    throw "A private environment build cannot write to the public release directory. Use dist\windows-private or another explicitly private location."
}
$builder = Resolve-BuildPython
$compiler = ""
if (-not $SkipInstaller) {
    $compiler = Resolve-InnoCompiler
}
$preflightResult = & $packagingPreflight `
    -RepositoryRoot $repositoryRoot `
    -Version $Version `
    -SourceCommit $sourceCommit `
    -BuildPython $builder `
    -InnoCompiler $compiler `
    -SkipInstaller:$SkipInstaller
Write-Host (
    "Packaging preflight passed under PowerShell " +
    "$($preflightResult.powershell) with Python $($preflightResult.python)."
)
$cache = [System.IO.Path]::GetFullPath($CacheDirectory)
$stageRoot = Assert-ChildPath `
    (Join-Path $repositoryRoot ".tmp") `
    (Join-Path $repositoryRoot ".tmp\windows-installer\$Version")
$application = Join-Path $stageRoot "app"
$pythonRoot = Join-Path $application "runtime\python"
$sitePackages = Join-Path $pythonRoot "Lib\site-packages"
$toolsRoot = Join-Path $application "runtime\tools"
$bootstrapRoot = Join-Path $application "runtime\bootstrap"

if (Test-Path -LiteralPath $stageRoot) {
    Remove-Item -LiteralPath $stageRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $application, $output, $cache | Out-Null

$bundleValue = if ($BundleLocalEnv) { "true" } else { "false" }
& dotnet clean $project -c Release
if ($LASTEXITCODE -ne 0) {
    throw "Cleaning stale native build outputs failed with exit code $LASTEXITCODE."
}
& dotnet publish $project `
    -c Release `
    -r win-x64 `
    -o $application `
    "-p:BundleLocalEnv=$bundleValue" `
    "-p:Version=$Version" `
    "-p:FileVersion=$numericVersion" `
    "-p:AssemblyVersion=$numericVersion" `
    "-p:SourceRevisionId=$sourceCommit" `
    "-p:SelfContained=true" `
    "-p:DebugType=None" `
    "-p:DebugSymbols=false"
if ($LASTEXITCODE -ne 0) {
    throw "dotnet publish failed with exit code $LASTEXITCODE."
}

$requiredDesktopFiles = @(
    "Broadcastify Desktop.exe",
    "Broadcastify Desktop.pri",
    "App.xbf",
    "MainWindow.xbf",
    "hostfxr.dll",
    "hostpolicy.dll",
    "coreclr.dll",
    "System.Private.CoreLib.dll",
    "windowsml\BroadcastifyCli.WindowsML.exe",
    "windowsml\hostfxr.dll",
    "windowsml\hostpolicy.dll",
    "windowsml\coreclr.dll",
    "windowsml\System.Private.CoreLib.dll"
)
foreach ($relativePath in $requiredDesktopFiles) {
    if (-not (Test-Path -LiteralPath (Join-Path $application $relativePath) -PathType Leaf)) {
        throw "The native publish is missing $relativePath."
    }
}

$nativeVersionFiles = @(
    (Join-Path $application "Broadcastify Desktop.exe"),
    (Join-Path $application "windowsml\BroadcastifyCli.WindowsML.exe")
)
foreach ($nativeVersionFile in $nativeVersionFiles) {
    $versionInfo = (Get-Item -LiteralPath $nativeVersionFile).VersionInfo
    if ($versionInfo.FileVersion -ne $numericVersion) {
        throw (
            "The native publish has stale FileVersion $($versionInfo.FileVersion) " +
            "in $nativeVersionFile; expected $numericVersion."
        )
    }
    if (-not $versionInfo.ProductVersion.StartsWith(
            $Version,
            [StringComparison]::Ordinal
        )) {
        throw (
            "The native publish has stale ProductVersion $($versionInfo.ProductVersion) " +
            "in $nativeVersionFile; expected $Version."
        )
    }
    if ($versionInfo.ProductVersion.IndexOf(
            $sourceCommit.Substring(0, 7),
            [StringComparison]::OrdinalIgnoreCase
        ) -lt 0) {
        throw (
            "The native publish ProductVersion is not bound to source commit " +
            "$sourceCommit in $nativeVersionFile."
        )
    }
}

$unexpectedPdbFiles = @(
    Get-ChildItem -LiteralPath $application -Filter *.pdb -Recurse -File
)
if ($unexpectedPdbFiles.Count -gt 0) {
    throw (
        "The public application stage contains debug symbols that can expose " +
        "local build paths: " +
        (($unexpectedPdbFiles | ForEach-Object FullName) -join ", ")
    )
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
$generatedLauncherDirectory = Join-Path $sitePackages "bin"
if (Test-Path -LiteralPath $generatedLauncherDirectory) {
    # pip's --target launchers contain an absolute shebang to the build
    # interpreter. The desktop invokes every worker through the bundled
    # python.exe and module name, so these non-portable stubs are unnecessary.
    $verifiedLauncherDirectory = Assert-ChildPath `
        $sitePackages `
        $generatedLauncherDirectory
    Remove-Item -LiteralPath $verifiedLauncherDirectory -Recurse -Force
}
$prunableRuntimeTests = @(
    Get-ChildItem -LiteralPath $sitePackages -Directory -Recurse |
        Where-Object { $_.Name -eq "test" -or $_.Name -eq "tests" } |
        Sort-Object FullName -Descending
)
$prunedRuntimeTestBytes = [int64](
    $prunableRuntimeTests |
        ForEach-Object {
            (Get-ChildItem -LiteralPath $_.FullName -File -Recurse |
                Measure-Object Length -Sum).Sum
        } |
        Measure-Object -Sum
).Sum
foreach ($runtimeTestPath in $prunableRuntimeTests) {
    $verifiedRuntimeTestPath = Assert-ChildPath `
        $sitePackages `
        $runtimeTestPath.FullName
    Remove-Item -LiteralPath $verifiedRuntimeTestPath -Recurse -Force
}
$remainingRuntimeTests = @(
    Get-ChildItem -LiteralPath $sitePackages -Directory -Recurse |
        Where-Object { $_.Name -eq "test" -or $_.Name -eq "tests" }
)
if ($remainingRuntimeTests.Count -gt 0) {
    throw "The portable runtime still contains third-party test trees."
}
Get-ChildItem -LiteralPath $sitePackages -Filter __pycache__ -Recurse -Directory |
    Sort-Object FullName -Descending |
    Remove-Item -Recurse -Force

if (Test-Path -LiteralPath $generatedLauncherDirectory) {
    throw "The portable runtime still contains non-portable pip launchers."
}

New-Item -ItemType Directory -Force -Path $bootstrapRoot | Out-Null
& $builder -m pip wheel `
    --disable-pip-version-check `
    --no-input `
    --no-deps `
    --wheel-dir $bootstrapRoot `
    $repositoryRoot
if ($LASTEXITCODE -ne 0) {
    throw "Building the managed-runtime application wheel failed with exit code $LASTEXITCODE."
}
$appWheels = @(
    Get-ChildItem -LiteralPath $bootstrapRoot `
        -Filter "broadcastify_cli-$Version-*.whl" -File
)
if ($appWheels.Count -ne 1) {
    throw "Expected one managed-runtime application wheel; found $($appWheels.Count)."
}
$appWheel = $appWheels[0]
$appWheelSha256 = (Get-FileHash -LiteralPath $appWheel.FullName -Algorithm SHA256).Hash

if (-not (Test-Path -LiteralPath $cudaRequirementsSource -PathType Leaf)) {
    throw "The checksum-locked Windows CUDA requirements file is missing."
}
$packagedCudaRequirements = Join-Path $bootstrapRoot "windows-managed-cuda-lock.txt"
Copy-Item -LiteralPath $cudaRequirementsSource -Destination $packagedCudaRequirements
$cudaRequirementsSha256 = (
    Get-FileHash -LiteralPath $packagedCudaRequirements -Algorithm SHA256
).Hash

$uvWheel = Get-VerifiedDownload `
    -Uri $uvWheelUrl `
    -Destination (Join-Path $cache "uv-$uvVersion-py3-none-win_amd64.zip") `
    -Sha256 $uvWheelSha256
$uvExtract = Join-Path $stageRoot "uv"
New-Item -ItemType Directory -Force -Path $uvExtract | Out-Null
Expand-Archive -LiteralPath $uvWheel -DestinationPath $uvExtract -Force
$uvExecutables = @(
    Get-ChildItem -LiteralPath $uvExtract -Filter uv.exe -Recurse -File
)
if ($uvExecutables.Count -ne 1) {
    throw "Expected one uv.exe in the verified wheel; found $($uvExecutables.Count)."
}
$uvExecutable = $uvExecutables[0]
$packagedUv = Join-Path $bootstrapRoot "uv.exe"
Copy-Item -LiteralPath $uvExecutable.FullName -Destination $packagedUv
$uvExecutableSha256 = (Get-FileHash -LiteralPath $packagedUv -Algorithm SHA256).Hash
$uvLicenses = @(
    Get-ChildItem -LiteralPath $uvExtract -Filter "LICENSE-*" -Recurse -File
)
if ($uvLicenses.Count -lt 2) {
    throw "The verified uv wheel did not contain its Apache-2.0 and MIT licenses."
}
foreach ($license in $uvLicenses) {
    Copy-Item -LiteralPath $license.FullName -Destination (
        Join-Path $bootstrapRoot "uv-$($license.Name)"
    )
}

$managedRuntimeManifest = [ordered]@{
    schema_version = 1
    app_version = $Version
    uv = [ordered]@{
        version = $uvVersion
        path = "uv.exe"
        sha256 = $uvExecutableSha256.ToLowerInvariant()
        source_url = $uvWheelUrl
        source_wheel_sha256 = $uvWheelSha256.ToLowerInvariant()
        licenses = @("Apache-2.0", "MIT")
    }
    app_wheel = [ordered]@{
        path = $appWheel.Name
        sha256 = $appWheelSha256.ToLowerInvariant()
    }
    cuda_requirements = [ordered]@{
        path = "windows-managed-cuda-lock.txt"
        sha256 = $cudaRequirementsSha256.ToLowerInvariant()
    }
    profiles = [ordered]@{
        cuda = [ordered]@{
            revision = "cuda-cu128-py312-r1"
            display_name = "NVIDIA CUDA transcription and speaker labels"
            python_version = $pythonVersion
            requirements_artifact = "cuda_requirements"
            estimated_installed_bytes = 5900000000
            packages = @(
                "faster-whisper==1.2.1",
                "pyannote.audio==4.0.7",
                "torch==2.11.0",
                "torchaudio==2.11.0"
            )
            source_urls = @(
                "https://pypi.org/",
                "https://download.pytorch.org/whl/cu128",
                "https://github.com/astral-sh/uv"
            )
            licenses = @(
                "Package licenses are retained in each installed wheel's metadata.",
                "uv: Apache-2.0 OR MIT",
                "PyTorch: BSD-3-Clause",
                "faster-whisper: MIT",
                "pyannote.audio: MIT"
            )
        }
    }
}
$managedRuntimeManifest | ConvertTo-Json -Depth 8 |
    Set-Content -LiteralPath (
        Join-Path $bootstrapRoot "managed-runtime.json"
    ) -Encoding UTF8

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
uv ${uvVersion}: https://github.com/astral-sh/uv (Apache-2.0 OR MIT)

FFmpeg's bundled Windows essentials build is GPLv3. The application's GPLv3
license is included as LICENSE. Python package license files remain alongside
their installed package metadata under runtime\python\Lib\site-packages.
"@ | Set-Content -LiteralPath (Join-Path $application "THIRD-PARTY-NOTICES.txt") -Encoding UTF8

$embeddedPython = Join-Path $pythonRoot "python.exe"
$oldNoUserSite = $env:PYTHONNOUSERSITE
$oldNoBytecode = $env:PYTHONDONTWRITEBYTECODE
$oldFfmpeg = $env:FFMPEG_PATH
$runtimeSmokeRoot = Assert-ChildPath `
    $stageRoot `
    (Join-Path $stageRoot "runtime-smoke")
$isolatedRuntimeEnvironment = @(
    "BROADCASTIFY_USERNAME",
    "BROADCASTIFY_PASSWORD",
    "BROADCASTIFY_SECURE_USERNAME",
    "BROADCASTIFY_SECURE_PASSWORD",
    "HUGGINGFACE_TOKEN",
    "HUGGINGFACE_SECURE_TOKEN",
    "HF_TOKEN",
    "OPENAI_API_KEY",
    "BROADCASTIFY_ANALYSIS_API_KEY",
    "BROADCASTIFY_ANALYSIS_DB",
    "BROADCASTIFY_SECURE_ANALYSIS_DB",
    "BROADCASTIFY_LIBRARY_ROOT",
    "BROADCASTIFY_QUOTA_LEDGER",
    "BROADCASTIFY_CREDENTIAL_STORE",
    "BROADCASTIFY_ENV_FILE"
)
$savedRuntimeEnvironment = @{}
try {
    $env:PYTHONNOUSERSITE = "1"
    $env:PYTHONDONTWRITEBYTECODE = "1"
    $env:FFMPEG_PATH = Join-Path $toolsRoot "ffmpeg.exe"
    foreach ($name in $isolatedRuntimeEnvironment) {
        $savedRuntimeEnvironment[$name] = [Environment]::GetEnvironmentVariable(
            $name,
            [EnvironmentVariableTarget]::Process
        )
        [Environment]::SetEnvironmentVariable(
            $name,
            $null,
            [EnvironmentVariableTarget]::Process
        )
    }
    New-Item -ItemType Directory -Force -Path $runtimeSmokeRoot | Out-Null
    $env:BROADCASTIFY_LIBRARY_ROOT = Join-Path $runtimeSmokeRoot "archives"
    $env:BROADCASTIFY_SECURE_ANALYSIS_DB = Join-Path $runtimeSmokeRoot "analysis.sqlite3"
    $env:BROADCASTIFY_QUOTA_LEDGER = Join-Path $runtimeSmokeRoot "quota.sqlite3"
    $env:BROADCASTIFY_CREDENTIAL_STORE = Join-Path $runtimeSmokeRoot "credentials.enc"
    & $embeddedPython -B -c (
        "import broadcastify_cli, cryptography, requests, sherpa_onnx; " +
        "import onnxruntime_genai; from zoneinfo import ZoneInfo; " +
        "ZoneInfo('America/Chicago'); print('portable-runtime-ready')"
    )
    if ($LASTEXITCODE -ne 0) {
        throw "The embedded Python import smoke test failed with exit code $LASTEXITCODE."
    }
    & (Join-Path $toolsRoot "ffmpeg.exe") -hide_banner -version | Select-Object -First 1
    if ($LASTEXITCODE -ne 0) {
        throw "The bundled FFmpeg smoke test failed with exit code $LASTEXITCODE."
    }
    Push-Location $runtimeSmokeRoot
    try {
        $runtimeDiagnostics = @(
            & $embeddedPython -B -m broadcastify_cli.worker diagnostics
        )
        if ($LASTEXITCODE -ne 0 -or
            -not ($runtimeDiagnostics -match '"type"\s*:\s*"diagnostics"')) {
            throw "The packaged worker diagnostics smoke test failed."
        }
    }
    finally {
        Pop-Location
    }
}
finally {
    $env:PYTHONNOUSERSITE = $oldNoUserSite
    $env:PYTHONDONTWRITEBYTECODE = $oldNoBytecode
    $env:FFMPEG_PATH = $oldFfmpeg
    foreach ($entry in $savedRuntimeEnvironment.GetEnumerator()) {
        [Environment]::SetEnvironmentVariable(
            $entry.Key,
            $entry.Value,
            [EnvironmentVariableTarget]::Process
        )
    }
    if (Test-Path -LiteralPath $runtimeSmokeRoot) {
        Remove-Item -LiteralPath $runtimeSmokeRoot -Recurse -Force
    }
}
Get-ChildItem -LiteralPath $sitePackages -Filter __pycache__ -Recurse -Directory |
    Sort-Object FullName -Descending |
    Remove-Item -Recurse -Force

$manifest = [ordered]@{
    schema_version = 1
    app_version = $Version
    source_commit = $sourceCommit
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
    managed_runtime = [ordered]@{
        uv_version = $uvVersion
        uv_executable_sha256 = $uvExecutableSha256
        uv_source_wheel_sha256 = $uvWheelSha256
        app_wheel_sha256 = $appWheelSha256
        cuda_requirements_sha256 = $cudaRequirementsSha256
        profiles = @("cuda")
    }
    runtime_pruning = [ordered]@{
        test_tree_count = $prunableRuntimeTests.Count
        bytes = $prunedRuntimeTestBytes
    }
    private_environment = [bool]$BundleLocalEnv
    built_utc = [DateTime]::UtcNow.ToString("o")
}
$manifest | ConvertTo-Json -Depth 5 |
    Set-Content -LiteralPath (Join-Path $application "build-manifest.json") -Encoding UTF8

if (-not $BundleLocalEnv) {
    $scanArguments = @(
        $publicReleaseScanner,
        "--root", $application,
        "--forbid-path", $repositoryRoot
    )
    if (-not [string]::IsNullOrWhiteSpace($env:USERPROFILE)) {
        $scanArguments += @("--forbid-user-profile", $env:USERPROFILE)
    }
    & $builder @scanArguments
    if ($LASTEXITCODE -ne 0) {
        throw "The public secret and forbidden-file scan failed."
    }
}

if ($SkipInstaller) {
    [pscustomobject]@{
        version = $Version
        source_commit = $sourceCommit
        application_directory = $application
        installer = $null
        private_environment = [bool]$BundleLocalEnv
        embedded_python = $embeddedPython
        bundled_ffmpeg = (Join-Path $toolsRoot "ffmpeg.exe")
    }
    return
}

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
    source_commit = $sourceCommit
    application_directory = $application
    installer = $installer
    installer_sha256 = (Get-FileHash -LiteralPath $installer -Algorithm SHA256).Hash
    installer_bytes = (Get-Item -LiteralPath $installer).Length
    private_environment = [bool]$BundleLocalEnv
    embedded_python = $embeddedPython
    bundled_ffmpeg = (Join-Path $toolsRoot "ffmpeg.exe")
    pruned_test_tree_count = $prunableRuntimeTests.Count
    pruned_test_bytes = $prunedRuntimeTestBytes
}
