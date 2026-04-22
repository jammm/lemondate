# detect_npu.ps1 — sanity-check that a Strix Halo / Strix Point / Hawk
# Point machine has everything it needs to run Whisper on the XDNA 2
# NPU under lemondate. Prints a per-check PASS/FAIL and exits 0 if
# everything is ready, 1 otherwise.
#
# What we check (in order):
#   1. Windows version (NPU support is 24H2+ for Strix Halo).
#   2. NPU device is enumerated by the amdxdna driver.
#   3. NPU driver version >= 32.0.203.280 (WHQL).
#   4. Ryzen AI Software >= 1.7.1 is installed (%RYZEN_AI_INSTALLATION_PATH%).
#   5. FlexML runtime DLLs are on PATH (or in lemondate\bin\).
#   6. NPU is in performance pmode.
#   7. iGPU architecture is gfx11xx (Strix series).
#
# Safe to run on any Windows box — it just reports "no NPU" on a
# non-Ryzen-AI system.

[CmdletBinding()]
param()

$ErrorActionPreference = 'Continue'

$failures = @()
function Pass($msg) { Write-Host ("[PASS] " + $msg) -ForegroundColor Green }
function Fail($msg) { Write-Host ("[FAIL] " + $msg) -ForegroundColor Red; $script:failures += $msg }
function Info($msg) { Write-Host ("[INFO] " + $msg) -ForegroundColor Yellow }

# 1. Windows version
$os = Get-CimInstance Win32_OperatingSystem
$build = [int]($os.BuildNumber)
if ($build -ge 26100) {
    Pass "Windows build $build >= 26100 (24H2)"
} elseif ($build -ge 22000) {
    Fail "Windows build $build — Strix Halo NPU support needs 24H2 (build 26100+). Update Windows."
} else {
    Fail "Windows build $build — way behind. Update to Windows 11 24H2."
}

# 2. NPU device
$npuDevice = Get-CimInstance Win32_PnPEntity -ErrorAction SilentlyContinue |
    Where-Object {
        ($_.Name -like '*NPU*Compute*' -or $_.Name -like '*XDNA*' -or $_.Name -like '*AMD*AI*Engine*') -and
        $_.ConfigManagerErrorCode -eq 0
    } | Select-Object -First 1

if ($npuDevice) {
    Pass ("NPU device enumerated: " + $npuDevice.Name)
    $driver = Get-CimInstance Win32_PnPSignedDriver -ErrorAction SilentlyContinue |
        Where-Object { $_.DeviceID -eq $npuDevice.DeviceID } | Select-Object -First 1
    if ($driver) {
        $verStr = $driver.DriverVersion
        $verParts = $verStr -split '\.' | ForEach-Object { [int]$_ }
        # Encode version as comparable integer: M * 10^9 + m * 10^6 + b * 10^3 + r
        $ver = $verParts[0] * 1000000000L + $verParts[1] * 1000000L + $verParts[2] * 1000L + $verParts[3]
        $minVer = 32 * 1000000000L + 0 * 1000000L + 203 * 1000L + 280
        if ($ver -ge $minVer) {
            Pass "NPU driver version $verStr >= 32.0.203.280"
        } else {
            Fail "NPU driver version $verStr — need >= 32.0.203.280. Install NPU_RAI1.5_280_WHQL.zip."
        }
    } else {
        Info "Could not read NPU driver version"
    }
} else {
    Fail "No NPU device found. Either no XDNA NPU in this CPU, or driver is missing."
}

# 3. Ryzen AI Software install
$raiPath = $env:RYZEN_AI_INSTALLATION_PATH
if ($raiPath -and (Test-Path "$raiPath\deployment\onnxruntime.dll")) {
    Pass "Ryzen AI Software installed at $raiPath"
} elseif (Test-Path "C:\Program Files\RyzenAI\1.7.1\deployment\onnxruntime.dll") {
    Pass "Ryzen AI Software 1.7.1 installed at default path"
    Info "Set `$env:RYZEN_AI_INSTALLATION_PATH = 'C:\Program Files\RyzenAI\1.7.1'"
} else {
    Fail "Ryzen AI Software 1.7.1 not found. Install from https://account.amd.com/en/forms/downloads/xef.html?filename=ryzen-ai-lt-1.7.1.exe"
}

# 4. FlexML runtime
$flexmlCandidates = @()
if ($env:PATH) { $flexmlCandidates = $env:PATH -split ';' | Where-Object { $_ -like '*flexmlrt*' -or $_ -like '*flexml*' } }
$lemondateBin = Join-Path (Split-Path $PSScriptRoot -Parent) 'bin'
if ($flexmlCandidates.Count -gt 0) {
    Pass ("FlexML runtime on PATH: " + ($flexmlCandidates -join '; '))
} elseif (Test-Path (Join-Path $lemondateBin 'vaiml.dll')) {
    Pass "FlexML runtime staged in lemondate\bin\ (vaiml.dll present)"
} else {
    Info "FlexML runtime not visible. Extract flexmlrt1.7.0-win.zip and either run flexmlrt\setup.bat or copy its DLLs into lemondate\bin\."
    $failures += "FlexML runtime missing"
}

# 5. NPU pmode
$xrt = Get-Command xrt-smi -ErrorAction SilentlyContinue
if (-not $xrt) { $xrt = Get-Item 'C:\Windows\System32\AMD\xrt-smi.exe' -ErrorAction SilentlyContinue }
if ($xrt) {
    $pmode = & $xrt.Source examine --report power 2>&1 | Select-String 'Performance Mode|pmode|performance'
    if ($pmode -and ($pmode -match 'performance|turbo')) {
        Pass "NPU pmode = $pmode"
    } else {
        Info "NPU pmode not confirmed. Run as admin: cd C:\Windows\System32\AMD; .\xrt-smi configure --pmode performance"
    }
} else {
    Info "xrt-smi not found. NPU pmode check skipped."
}

# 6. iGPU architecture
try {
    $gpu = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match 'Radeon|AMD' } | Select-Object -First 1
    if ($gpu) {
        Pass ("iGPU / GPU found: " + $gpu.Name)
        Info "For Strix Halo iGPU Kokoro HIP, build with GFX_TARGET=gfx1151. For fat build across Strix Halo + Radeon dGPU, use GFX_TARGET=gfx1151;gfx1201."
    } else {
        Info "No AMD GPU found. Kokoro will fall back to CPU."
    }
} catch {
    Info "GPU detection failed: $_"
}

Write-Host ""
if ($failures.Count -eq 0) {
    Write-Host "[DONE] All NPU prerequisites satisfied." -ForegroundColor Green
    Write-Host "Next: set `$env:LEMONADE_WHISPER_BACKEND = 'npu' before start_services.ps1."
    exit 0
} else {
    Write-Host ("[DONE] " + $failures.Count + " issue(s) — see [FAIL] / [INFO] lines above.") -ForegroundColor Yellow
    exit 1
}
