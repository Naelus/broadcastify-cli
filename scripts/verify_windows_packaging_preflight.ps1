[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$RepositoryRoot,
    [Parameter(Mandatory = $true)][string]$Version,
    [Parameter(Mandatory = $true)][string]$SourceCommit,
    [Parameter(Mandatory = $true)][string]$BuildPython,
    [string]$InnoCompiler = "",
    [switch]$SkipInstaller
)

$ErrorActionPreference = "Stop"
$root = [System.IO.Path]::GetFullPath($RepositoryRoot)
$builder = [System.IO.Path]::GetFullPath($BuildPython)

if (-not (Test-Path -LiteralPath $root -PathType Container)) {
    throw "The packaging preflight repository root does not exist: $root"
}
if ($Version -notmatch '^\d+\.\d+\.\d+(?:\.\d+)?$') {
    throw "The packaging version must contain three or four numeric components."
}
if ($SourceCommit -notmatch '^[0-9a-f]{40}$') {
    throw "The packaging source revision must be a full lowercase Git commit."
}
if (-not (Test-Path -LiteralPath $builder -PathType Leaf)) {
    throw "The packaging Python interpreter was not found: $builder"
}
if (-not $SkipInstaller -and
    (-not (Test-Path -LiteralPath $InnoCompiler -PathType Leaf))) {
    throw "The Inno Setup compiler was not found: $InnoCompiler"
}
if ($null -eq (Get-Command dotnet.exe -ErrorAction SilentlyContinue)) {
    throw "dotnet.exe is required to build the Windows application."
}

$builderVersion = (& $builder -c "import sys; print('.'.join(map(str, sys.version_info[:3])))").Trim()
if ($LASTEXITCODE -ne 0 -or $builderVersion -notmatch '^3\.12\.') {
    throw "The build interpreter must be Python 3.12; found $builderVersion."
}

$pyproject = Get-Content -LiteralPath (Join-Path $root "pyproject.toml") -Raw
$projectVersion = [regex]::Match(
    $pyproject,
    '(?ms)^\[project\].*?^version\s*=\s*"(?<version>\d+\.\d+\.\d+(?:\.\d+)?)"'
).Groups["version"].Value
if ($projectVersion -ne $Version) {
    throw "Packaging version $Version does not match pyproject.toml version $projectVersion."
}

$parts = @($Version.Split("."))
while ($parts.Count -lt 4) {
    $parts += "0"
}
$numericVersion = $parts[0..3] -join "."
$versionContracts = @(
    @{
        Path = Join-Path $root "BroadcastifyCli.WinUI\BroadcastifyCli.WinUI.csproj"
        Patterns = @(
            "<Version>$([regex]::Escape($Version))</Version>",
            "<FileVersion>$([regex]::Escape($numericVersion))</FileVersion>"
        )
    },
    @{
        Path = Join-Path $root "BroadcastifyCli.WindowsML\BroadcastifyCli.WindowsML.csproj"
        Patterns = @(
            "<Version>$([regex]::Escape($Version))</Version>",
            "<FileVersion>$([regex]::Escape($numericVersion))</FileVersion>"
        )
    },
    @{
        Path = Join-Path $root "broadcastify_cli\__init__.py"
        Patterns = @("__version__\s*=\s*`"$([regex]::Escape($Version))`"")
    },
    @{
        Path = Join-Path $root "installer\BroadcastifyDesktop.iss"
        Patterns = @(
            "#define\s+MyAppVersion\s+`"$([regex]::Escape($Version))`"",
            "#define\s+MyAppVersionNumeric\s+`"$([regex]::Escape($numericVersion))`""
        )
    }
)
foreach ($contract in $versionContracts) {
    $content = Get-Content -LiteralPath $contract.Path -Raw
    foreach ($pattern in $contract.Patterns) {
        if ($content -notmatch $pattern) {
            throw "Release metadata is inconsistent in $($contract.Path)."
        }
    }
}

# Exercise the exact overloads used for post-publish ProductVersion checks.
$shortCommit = $SourceCommit.Substring(0, 7)
$syntheticProductVersion = "$Version+$shortCommit"
if (-not $syntheticProductVersion.StartsWith(
        $Version,
        [StringComparison]::Ordinal
    ) -or $syntheticProductVersion.IndexOf(
        $shortCommit,
        [StringComparison]::OrdinalIgnoreCase
    ) -lt 0) {
    throw "The current PowerShell runtime cannot perform native version checks."
}

$packagingScripts = @(
    "assert_committed_build_source.ps1",
    "build_windows_installer.ps1",
    "install_inno_setup.ps1",
    "install_windows_update.ps1",
    "run_product_regression_gate.ps1",
    "run_windows_installer_lifecycle.ps1",
    "run_windows_ui_e2e.ps1",
    "stop_windows_desktop_for_maintenance.ps1",
    "verify_windows_packaging_preflight.ps1",
    "verify_windows_publish.ps1"
)
foreach ($name in $packagingScripts) {
    $path = Join-Path $root "scripts\$name"
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "A required packaging helper is missing: $name"
    }
    $tokens = $null
    $parseErrors = $null
    [System.Management.Automation.Language.Parser]::ParseFile(
        $path,
        [ref]$tokens,
        [ref]$parseErrors
    ) | Out-Null
    if ($parseErrors.Count -gt 0) {
        $details = $parseErrors | ForEach-Object { $_.Message }
        throw "Packaging helper $name is not valid PowerShell: $($details -join '; ')"
    }
}

[pscustomobject]@{
    status = "passed"
    version = $Version
    source_commit = $SourceCommit
    powershell = $PSVersionTable.PSVersion.ToString()
    python = $builderVersion
    installer = -not [bool]$SkipInstaller
}
