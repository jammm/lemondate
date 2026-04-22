#Requires -Version 7.0

param(
    [string] $GfxIndex = "gfx120X-all",
    [string] $PythonExe = "python"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Venv     = Join-Path $RepoRoot "venv"
$Reqs     = Join-Path $PSScriptRoot "requirements.txt"

Write-Host "[ptt/install] RepoRoot: $RepoRoot"
Write-Host "[ptt/install] Venv:     $Venv"
Write-Host "[ptt/install] GfxIndex: $GfxIndex"

if (-not (Test-Path $Venv)) {
    Write-Host "[ptt/install] Creating venv..."
    & $PythonExe -m venv $Venv
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
}

$VenvPy  = Join-Path $Venv "Scripts\python.exe"
$VenvPip = Join-Path $Venv "Scripts\pip.exe"

if (-not (Test-Path $VenvPy)) {
    throw "venv python not found at $VenvPy"
}

Write-Host "[ptt/install] Upgrading pip..."
& $VenvPy -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }

Write-Host "[ptt/install] Installing torch + rocm[libraries,devel] from TheRock..."
& $VenvPip install --index-url "https://rocm.nightlies.amd.com/v2/$GfxIndex/" torch "rocm[libraries,devel]"
if ($LASTEXITCODE -ne 0) { throw "torch+rocm install failed" }

Write-Host "[ptt/install] Installing ptt requirements..."
& $VenvPip install -r $Reqs
if ($LASTEXITCODE -ne 0) { throw "requirements install failed" }

Write-Host "[ptt/install] Initialising ROCm SDK (rocm-sdk init)..."
$RocmInit = Join-Path $Venv "Scripts\rocm-sdk.exe"
if (Test-Path $RocmInit) {
    & $RocmInit init
    if ($LASTEXITCODE -ne 0) { Write-Warning "rocm-sdk init failed (non-fatal; continuing)" }
} else {
    Write-Warning "rocm-sdk CLI not found; skipping init."
}

Write-Host ""
Write-Host "[ptt/install] Done. Venv ready at $Venv."
Write-Host "[ptt/install] Test: `"$VenvPy`" `"$PSScriptRoot\ptt_daemon.py`""
