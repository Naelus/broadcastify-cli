[CmdletBinding()]
param(
    [string]$Python = "",
    [switch]$Focused
)

$ErrorActionPreference = "Stop"
$repositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
if ([string]::IsNullOrWhiteSpace($Python)) {
    $venvPython = Join-Path $repositoryRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
        $Python = $venvPython
    }
    else {
        $command = Get-Command python.exe -ErrorAction SilentlyContinue
        if ($null -eq $command) {
            throw "Python is required to run the product regression gate."
        }
        $Python = $command.Source
    }
}
$Python = [System.IO.Path]::GetFullPath($Python)
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "The product regression Python interpreter was not found: $Python"
}

$arguments = @(
    "-m", "pytest", "-q",
    "--durations=10", "--durations-min=0.25"
)
if ($Focused) {
    $arguments += @(
        "tests\e2e\test_product_workflows.py"
    )
}

# A product build must never inherit personal credentials into its proof run.
# Individual tests create only temporary credentials, libraries, and ledgers.
$isolatedEnvironment = @(
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
$savedEnvironment = @{}
$exitCode = 1
Push-Location $repositoryRoot
try {
    foreach ($name in $isolatedEnvironment) {
        $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable(
            $name,
            [EnvironmentVariableTarget]::Process
        )
        [Environment]::SetEnvironmentVariable(
            $name,
            $null,
            [EnvironmentVariableTarget]::Process
        )
    }
    $savedEnvironment["BROADCASTIFY_PRODUCT_REGRESSION_GATE"] = (
        [Environment]::GetEnvironmentVariable(
            "BROADCASTIFY_PRODUCT_REGRESSION_GATE",
            [EnvironmentVariableTarget]::Process
        )
    )
    [Environment]::SetEnvironmentVariable(
        "BROADCASTIFY_PRODUCT_REGRESSION_GATE",
        "1",
        [EnvironmentVariableTarget]::Process
    )
    & $Python @arguments
    $exitCode = $LASTEXITCODE
}
finally {
    foreach ($entry in $savedEnvironment.GetEnumerator()) {
        [Environment]::SetEnvironmentVariable(
            $entry.Key,
            $entry.Value,
            [EnvironmentVariableTarget]::Process
        )
    }
    Pop-Location
}

if ($exitCode -ne 0) {
    throw "The product regression gate failed with exit code $exitCode. The native build was not allowed to start."
}

[pscustomobject]@{
    status = "passed"
    scope = if ($Focused) { "recent product workflows" } else { "full offline suite" }
    python = $Python
}
