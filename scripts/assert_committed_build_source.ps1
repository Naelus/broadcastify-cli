[CmdletBinding()]
param(
    [string]$RepositoryRoot = "",
    [switch]$Quiet
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
    $RepositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
}
else {
    $RepositoryRoot = [System.IO.Path]::GetFullPath($RepositoryRoot)
}

$git = Get-Command git.exe -ErrorAction SilentlyContinue
if (-not $git) {
    throw "Git is required to bind this build to a committed source revision."
}

$head = (& $git.Source -C $RepositoryRoot rev-parse --verify HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $head -notmatch '^[0-9a-f]{40}$') {
    throw "Unable to resolve the committed Git source revision."
}

$changes = @(
    & $git.Source -C $RepositoryRoot status --porcelain --untracked-files=all
)
if ($LASTEXITCODE -ne 0) {
    throw "Unable to verify the Git worktree before building."
}
if ($changes.Count -gt 0) {
    throw (
        "Commit every source, test, documentation, version, and release " +
        "change before building. Versioned builds must map to exactly one " +
        "Git commit.`n" + ($changes -join "`n")
    )
}

if (-not $Quiet) {
    $head
}
