[CmdletBinding()]
param(
    [string]$CacheDirectory = "",
    [switch]$MachineWide
)

$ErrorActionPreference = "Stop"
$version = "7.0.2"
$expectedSha256 = "5AD54CA3DEF786F8F4212552E54CC6D8D61329E2D24A1CFEE0571D42C2684FF1"
$downloadUrl = "https://github.com/jrsoftware/issrc/releases/download/is-7_0_2/innosetup-7.0.2-x64.exe"
$repositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
if ([string]::IsNullOrWhiteSpace($CacheDirectory)) {
    $CacheDirectory = Join-Path $repositoryRoot ".tmp\installer-cache"
}
$cache = [System.IO.Path]::GetFullPath($CacheDirectory)
New-Item -ItemType Directory -Force -Path $cache | Out-Null
$installer = Join-Path $cache "innosetup-$version-x64.exe"

if (-not (Test-Path -LiteralPath $installer -PathType Leaf)) {
    Invoke-WebRequest -UseBasicParsing -Uri $downloadUrl -OutFile $installer
}
$actualHash = (Get-FileHash -LiteralPath $installer -Algorithm SHA256).Hash
if ($actualHash -ne $expectedSha256) {
    throw "The Inno Setup installer checksum did not match the pinned $version release."
}
$signature = Get-AuthenticodeSignature -LiteralPath $installer
if ($signature.Status -ne [System.Management.Automation.SignatureStatus]::Valid -or
    $signature.SignerCertificate.Subject -notlike "CN=Pyrsys B.V.*") {
    throw "The Inno Setup installer does not have the expected valid Pyrsys B.V. signature."
}

$scopeArgument = if ($MachineWide) { "/ALLUSERS" } else { "/CURRENTUSER" }
$process = Start-Process -FilePath $installer -ArgumentList @(
    "/VERYSILENT",
    "/SUPPRESSMSGBOXES",
    "/NORESTART",
    "/SP-",
    $scopeArgument
) -Wait -PassThru -WindowStyle Hidden
if ($process.ExitCode -ne 0) {
    throw "Inno Setup installation failed with exit code $($process.ExitCode)."
}

$candidates = @(
    (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 7\ISCC.exe"),
    (Join-Path $env:ProgramFiles "Inno Setup 7\ISCC.exe"),
    (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 7\ISCC.exe")
) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
$compiler = $candidates | Where-Object {
    Test-Path -LiteralPath $_ -PathType Leaf
} | Select-Object -First 1
if (-not $compiler) {
    throw "Inno Setup installed, but ISCC.exe was not found."
}

[pscustomobject]@{
    version = $version
    compiler = [System.IO.Path]::GetFullPath($compiler)
    installer_sha256 = $actualHash
    scope = if ($MachineWide) { "all-users" } else { "current-user" }
}
