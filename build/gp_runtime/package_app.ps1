param(
    [string]$Name = "DeepvacAIAdvisor",
    [switch]$Clean,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$env:PYTHONNOUSERSITE = "1"

function Invoke-Native {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Exe,
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Args
    )

    & $Exe @Args
    if ($LASTEXITCODE -ne 0) {
        throw "$Exe $($Args -join ' ') failed with exit code $LASTEXITCODE"
    }
}

$Python = (Get-Command python).Source
Write-Host "Using Python: $Python"

if ($Clean) {
    Remove-Item -Recurse -Force ".\dist", ".\build", ".\$Name.spec" -ErrorAction SilentlyContinue
}

try {
    Invoke-Native $Python -s -m pip --version
} catch {
    Write-Host "pip is not available in this interpreter; bootstrapping with ensurepip."
    Invoke-Native $Python -s -m ensurepip --upgrade
}

if (-not $SkipInstall) {
    Invoke-Native $Python -s -m pip install -r .\requirements_runtime.txt
}

try {
    Invoke-Native $Python -s -c "import PyInstaller"
} catch {
    Write-Host "PyInstaller is not installed; installing it now."
    Invoke-Native $Python -s -m pip install pyinstaller
}

Invoke-Native $Python -s -m PyInstaller `
    --noconfirm `
    --clean `
    --onedir `
    --name $Name `
    --console `
    --add-data "model;model" `
    --add-data "data;data" `
    --add-data "tcp;tcp" `
    --add-data "utils;utils" `
    --exclude-module matplotlib `
    --exclude-module matplotlib.backends `
    --exclude-module seaborn `
    --collect-submodules sklearn `
    --collect-submodules torch `
    .\live_gp_controller.py

$ExePath = Join-Path $Root "dist\$Name\$Name.exe"
if (-not (Test-Path $ExePath)) {
    throw "PyInstaller finished but the expected executable was not found: $ExePath"
}

Write-Host ""
Write-Host "Built app folder:"
Write-Host "  $Root\dist\$Name"
Write-Host ""
Write-Host "Run:"
Write-Host "  .\dist\$Name\$Name.exe --dry-run --duration-s 120"
