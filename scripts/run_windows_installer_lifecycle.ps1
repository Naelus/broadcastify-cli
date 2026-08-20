[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Installer
)

$ErrorActionPreference = "Stop"
if ($env:GITHUB_ACTIONS -ine "true") {
    throw "The installer lifecycle probe is CI-only and requires GITHUB_ACTIONS=true."
}
if ([string]::IsNullOrWhiteSpace($env:RUNNER_TEMP)) {
    throw "RUNNER_TEMP is required for the isolated installer lifecycle probe."
}
if ([string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
    throw "LOCALAPPDATA is required for the retained product-data lifecycle check."
}

function Assert-ContainedPath {
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
        throw "Refusing to use a path outside $resolvedParent`: $resolvedChild"
    }
    return $resolvedChild
}

function Invoke-HiddenProcess {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList,
        [Parameter(Mandatory = $true)][string]$Description
    )
    $process = Start-Process `
        -FilePath $FilePath `
        -ArgumentList $ArgumentList `
        -Wait `
        -PassThru `
        -WindowStyle Hidden
    if ($process.ExitCode -ne 0) {
        throw "$Description failed with exit code $($process.ExitCode)."
    }
}

$repositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$installerPath = [System.IO.Path]::GetFullPath($Installer)
if (-not (Test-Path -LiteralPath $installerPath -PathType Leaf)) {
    throw "Generated installer was not found: $installerPath"
}

$runnerTemp = [System.IO.Path]::GetFullPath($env:RUNNER_TEMP)
if (-not (Test-Path -LiteralPath $runnerTemp -PathType Container)) {
    throw "RUNNER_TEMP is not a directory: $runnerTemp"
}
$sessionRoot = Assert-ContainedPath `
    $runnerTemp `
    (Join-Path $runnerTemp (
        "broadcastify-desktop-installer-lifecycle-" +
        [Guid]::NewGuid().ToString("N")
    ))
$installDirectory = Assert-ContainedPath `
    $sessionRoot `
    (Join-Path $sessionRoot "app")
New-Item -ItemType Directory -Path $installDirectory -Force | Out-Null

$productDataRoot = [System.IO.Path]::GetFullPath(
    (Join-Path $env:LOCALAPPDATA "Broadcastify Desktop")
)
$testDataDirectory = Assert-ContainedPath `
    $productDataRoot `
    (Join-Path $productDataRoot "ci-installer-lifecycle")
if (Test-Path -LiteralPath $testDataDirectory) {
    throw "Refusing to overwrite pre-existing lifecycle data: $testDataDirectory"
}
$sentinelPath = Join-Path $testDataDirectory "sentinel.txt"
$sentinelValue = "Broadcastify Desktop CI lifecycle sentinel"
New-Item -ItemType Directory -Path $testDataDirectory -Force | Out-Null
[System.IO.File]::WriteAllText(
    $sentinelPath,
    $sentinelValue,
    [System.Text.UTF8Encoding]::new($false)
)

$desktopExecutable = Join-Path $installDirectory "Broadcastify Desktop.exe"
$bundledPython = Join-Path $installDirectory "runtime\python\python.exe"
$uninstaller = Join-Path $installDirectory "unins000.exe"
$installerArguments = @(
    "/VERYSILENT",
    "/SUPPRESSMSGBOXES",
    "/NORESTART",
    "/SP-",
    "/DISABLESTARTUP",
    "/DIR=`"$installDirectory`""
)
$uninstallerArguments = @(
    "/VERYSILENT",
    "/SUPPRESSMSGBOXES",
    "/NORESTART",
    "/SP-"
)

function Install-Product {
    Invoke-HiddenProcess `
        -FilePath $installerPath `
        -ArgumentList $installerArguments `
        -Description "Silent installer run"
}

function Verify-InstalledRuntime {
    foreach ($path in @($desktopExecutable, $bundledPython)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Installed runtime is missing $path."
        }
    }
}

function Uninstall-Product {
    if (-not (Test-Path -LiteralPath $uninstaller -PathType Leaf)) {
        throw "Installed uninstaller was not found: $uninstaller"
    }
    Invoke-HiddenProcess `
        -FilePath $uninstaller `
        -ArgumentList $uninstallerArguments `
        -Description "Silent uninstall run"
    $deadline = [DateTime]::UtcNow.AddSeconds(15)
    while ((Test-Path -LiteralPath $installDirectory) `
            -and [DateTime]::UtcNow -lt $deadline) {
        Start-Sleep -Milliseconds 250
    }
}

$passed = $false
$uiE2e = Join-Path $repositoryRoot "scripts\run_windows_ui_e2e.ps1"
try {
    Install-Product
    Verify-InstalledRuntime
    & $uiE2e -Executable $desktopExecutable
    if (-not $?) {
        throw "The installed native UI E2E probe failed."
    }

    $beforeUpgrade = Get-Content -LiteralPath $sentinelPath -Raw
    Install-Product
    Verify-InstalledRuntime
    if ((Get-Content -LiteralPath $sentinelPath -Raw) -ne $beforeUpgrade) {
        throw "The retained lifecycle sentinel changed during same-version upgrade."
    }

    Uninstall-Product
    if (Test-Path -LiteralPath $installDirectory) {
        throw "Uninstall left the application directory behind: $installDirectory"
    }
    if (-not (Test-Path -LiteralPath $sentinelPath -PathType Leaf)) {
        throw "Uninstall removed the retained product-data sentinel."
    }

    Install-Product
    Verify-InstalledRuntime
    Uninstall-Product
    if (Test-Path -LiteralPath $installDirectory) {
        throw "Final uninstall left the application directory behind: $installDirectory"
    }
    if ((Get-Content -LiteralPath $sentinelPath -Raw) -ne $sentinelValue) {
        throw "Final uninstall did not preserve the retained product-data sentinel."
    }

    $passed = $true
    [pscustomobject]@{
        status = "passed"
        installer = $installerPath
        install_directory = $installDirectory
        retained_data_directory = $testDataDirectory
        startup = "explicitly disabled"
        native_ui_e2e = "passed"
        upgrade = "preserved sentinel"
        uninstall = "removed app and preserved data"
        reinstall = "passed"
    }
}
finally {
    if ($passed) {
        $verifiedData = Assert-ContainedPath $productDataRoot $testDataDirectory
        if (Test-Path -LiteralPath $verifiedData) {
            Remove-Item -LiteralPath $verifiedData -Recurse -Force
        }
        $verifiedSession = Assert-ContainedPath $runnerTemp $sessionRoot
        if (Test-Path -LiteralPath $verifiedSession) {
            Remove-Item -LiteralPath $verifiedSession -Recurse -Force
        }
    }
    else {
        Write-Warning "Installer lifecycle artifacts were retained for CI diagnosis: $sessionRoot"
    }
}
