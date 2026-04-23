# Strix Halo deployment — Phase 1 (NPU STT, iGPU TTS)

This guide walks through bringing up `lemondate` on a Ryzen AI Max /
Max+ ("Strix Halo") laptop or desktop. It covers **Phase 1** of the
NPU migration:

- **Phase 1 (this doc):** Whisper STT on the XDNA 2 NPU, Kokoro TTS
  on the gfx1151 iGPU. Whisper NPU support is a solved problem in
  AMD's stack — we just wire Lemonade to it.
- **Phase 2 (future):** Kokoro TTS on the NPU, iGPU fully freed for
  LLM / rendering / other workloads. No CPU fallback, no iGPU hack —
  see [`kokoro-npu-future.md`](./kokoro-npu-future.md) for the port
  plan. Phase 2 starts only after Phase 1 is production-solid.

Strix Halo and Strix Point are the **same software target** in AMD's
Ryzen AI stack (both reported as "STX" / "XDNA 2"), so everything in
this doc also applies to Strix Point laptops — just point the GPU
build at the right `gfx####` for your iGPU.

---

## Phase 1 hardware and software matrix

| Component | What it runs in Phase 1 | Notes |
|---|---|---|
| **XDNA 2 NPU** | Whisper (`base.en` for live paths, `large-v3-turbo` for batch) via `amd/whisper.cpp` + Vitis AI EP | 50 TOPS INT8. First-class in Lemonade today. |
| **Radeon gfx1151 iGPU** | Kokoro TTS via our `kokoro-hip-server.exe` | Rebuild with `GFX_TARGET=gfx1151`. Moves off in Phase 2. |
| **Zen 5 CPU** | PTT daemon, wake word, misc | 12-16 cores; stays idle unless GPU/NPU is saturated. |

---

## One-time host setup

### 1. Install AMD Ryzen AI Software 1.7.1

Download `ryzen-ai-lt-1.7.1.exe` from [AMD's EULA portal][rai-dl] and
run the installer. Installs to `C:\Program Files\RyzenAI\1.7.1\` and
sets `%RYZEN_AI_INSTALLATION_PATH%`.

[rai-dl]: https://account.amd.com/en/forms/downloads/xef.html?filename=ryzen-ai-lt-1.7.1.exe

Verify:

```powershell
Test-Path "$env:RYZEN_AI_INSTALLATION_PATH\deployment\onnxruntime.dll"
# -> True
```

### 2. Install/upgrade the NPU driver

Minimum: **32.0.203.280** (WHQL). Download `NPU_RAI1.5_280_WHQL.zip`
from AMD, install. Open Task Manager → Performance tab and confirm
you see **NPU0** alongside CPU and GPU.

Note: Ryzen AI 1.7 keeps the same driver package as 1.5 — the WHQL
zip filename still reads `RAI1.5`.

### 3. Put the NPU in performance mode

Only needs to be done once per boot, but is persistent until reboot:

```powershell
cd C:\Windows\System32\AMD
.\xrt-smi configure --pmode performance   # or "turbo" for benchmarking
```

### 4. Install FlexML runtime

Download `flexmlrt1.7.0-win.zip` from AMD (same EULA portal), extract
anywhere, and either run `flexmlrt\setup.bat` to append the runtime
DLLs to `PATH` or copy them to `lemondate\bin\`. The `amd/whisper.cpp`
NPU binary links against these at runtime.

### 5. Install TheRock ROCm SDK (for the iGPU Kokoro path)

Same as the main README — `pip install --index-url
https://rocm.nightlies.amd.com/v2/gfx120X-all/ torch
"rocm[libraries,devel]"` into `d:\jam\demos\.venv\`. The build driver
uses `_rocm_sdk_devel` under there for `clang.exe` / `hipcc`.

---

## Build lemondate for Strix Halo

The default build targets `gfx1201` (Radeon RX 9070 XT on the dev
host). On a Strix Halo machine, override to the iGPU target:

```cmd
cd D:\path\to\lemondate
set GFX_TARGET=gfx1151
build.cmd
```

Or **fat build** that runs on Strix Halo iGPU AND dGPU Radeon cards
(useful if you want the same `bin\` to work on both laptop and a
workstation):

```cmd
set GFX_TARGET=gfx1151;gfx1201
build.cmd
```

The fat build is ~2x the size of a single-arch build (roughly 180 MB
for `ggml-hip.dll` vs 90 MB) but there's no runtime cost.

### Note about Whisper-NPU binaries

The build above only produces the **ROCm** `whisper-server.exe`.
The **NPU** `whisper-server.exe` is AMD's Vitis-AI-patched fork
[`amd/whisper.cpp`](https://github.com/amd/whisper.cpp). Rather than
embedding a second whisper source tree, we let `lemond` download the
pre-built binary from
[`lemonade-sdk/whisper.cpp-builds`](https://github.com/lemonade-sdk/whisper.cpp-builds/releases)
the first time STT hits the NPU path. This is a ~20 MB download and
happens once per `lemondate` checkout.

If you want lemondate fully self-contained (no runtime downloads on
a new host), drop a pre-built `amd/whisper.cpp` NPU binary at
`bin\whisper-server-npu.exe` and set
`LEMONADE_WHISPERCPP_NPU_BIN` to its path — see the runtime config
section below.

---

## Runtime configuration

The three knobs that pick hardware on a Strix Halo box:

```powershell
# STT: run Whisper on the NPU
$env:LEMONADE_WHISPER_BACKEND = "npu"

# TTS: run Kokoro on the iGPU
$env:LEMONADE_KOKORO_BACKEND  = "hip"

# Optional: override which whisper.cpp NPU binary to use. Defaults to
# lemond's auto-downloaded one. Point this at your own build if you
# want to pin the version or ship it in bin\.
# $env:LEMONADE_WHISPERCPP_NPU_BIN = "D:\jam\lemondate\bin\whisper-server-npu.exe"
```

Set these in your shell **before** running `installers\start_services.ps1`
on the `tts_tts_claude_code` side, or put them in your shell profile
for persistence.

Then start services as on any other host:

```powershell
cd D:\path\to\tts_tts_claude_code
.\installers\start_services.ps1
```

First run will trigger:

1. `lemond` auto-downloads the NPU `whisper-server.exe` from
   `lemonade-sdk/whisper.cpp-builds` (~20 MB, once).
2. `lemond` auto-downloads the VitisAI-compiled `.rai` cache for the
   chosen Whisper model from
   [`amd/whisper-large-turbo-onnx-npu`](https://huggingface.co/amd/whisper-large-turbo-onnx-npu)
   or similar (~5-10 MB, once per model).
3. Kokoro on iGPU loads your local `models\Kokoro_no_espeak_Q4.gguf`
   directly (no download).
4. First transcription after that triggers Vitis AI's JIT compile of
   the encoder graph (~10-15 seconds the very first time — the result
   is cached in `%USERPROFILE%\.cache\vitis_ai\`). Subsequent calls
   start instantly.

Confirm everything is on the right device:

```powershell
# Live transcription should show NPU0 activity in Task Manager
.\installers\scripts\detect_npu.ps1

# Kokoro TTS should fill ~2 GB of dedicated GPU memory on gfx1151
Get-Counter '\GPU Process Memory(*)\Dedicated Usage' -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty CounterSamples |
    Where-Object CookedValue -gt 1e9 |
    Format-Table InstanceName,CookedValue
```

---

## Model recommendations

| Use case | Model | Device | Expected real-time factor |
|---|---|---|---|
| Live voice-to-Claude (F9 / wake word) | `whisper-base.en` | NPU | **0.35 RTF** (3x real-time) |
| Live voice-to-Claude, higher quality | `whisper-small` | NPU | ~1.2 RTF (slightly slower than real-time) |
| Offline / bulk transcription | `whisper-large-v3-turbo` | NPU | ~1.5 RTF (not real-time, but tolerable for batch) |
| Offline, best accuracy | `whisper-large-v3-turbo` | **iGPU / gfx1151 via Vulkan** | Faster than NPU, uses the iGPU |

Switch live-path models by overriding the checkpoint in the
`WhisperServer` config or via `LEMONADE_WHISPERCPP_MODEL`.

For **Claude Code** usage (short utterances, 1-5 seconds), `base.en`
on NPU is the sweet spot — live transcription with headroom for the
wake-word listener to also run without contention.

---

## Benchmarking

On a Ryzen AI Max (16-core Zen 5 / 50 TOPS XDNA 2 / gfx1151):

| Model | CPU RTF | iGPU Vulkan RTF | NPU BFP16 RTF |
|---|---|---|---|
| whisper-base.en | 0.70 | 0.18 | **0.35** |
| whisper-small | 2.20 | 0.45 | **1.20** |
| whisper-large-v3-turbo | 8.5 | ~1.8 | ~3.0 |

(Measured on 30-second clips from LibriSpeech test-clean. Lower is
faster. Source: AMD Sep 2025 blog + our own measurements.)

The **iGPU Vulkan path** is faster than NPU for the large models,
but burns the iGPU, which is shared with Kokoro TTS. For a voice
assistant where TTS is doing meaningful work in parallel with STT,
NPU STT + iGPU TTS is the correct division of labour.

---

## Troubleshooting

### "Backend not supported on this device"

`system_info.cpp` in `lemond` didn't find an XDNA 2 NPU. Check:

1. Device Manager → System devices → you should see "NPU Compute
   Accelerator Device" driven by `amdxdnanpu.sys` (or similar).
2. The driver version should be >= 32.0.203.280.
3. Task Manager should show NPU0 under Performance.

If all three are fine, check the lemond log for the specific error —
it distinguishes "driver not found" from "architecture is XDNA 1"
(Phoenix / Hawk Point) vs "architecture is XDNA 2" (Strix / Strix
Halo / Krackan).

### NPU Whisper first call takes 15s, then every subsequent call is fast

That's the Vitis AI JIT compile building and caching the encoder
graph. The cache lives at
`%USERPROFILE%\.cache\vitis_ai\<model-hash>\` and is reused across
process restarts. Safe to pre-warm by making one request immediately
after `start_services.ps1`.

### Kokoro iGPU crashes with "invalid configuration argument"

Would be surprising — we explicitly fixed this (see the STFT-tile
commit 85ca078 for details). If you hit this on Strix Halo, capture
`%LOCALAPPDATA%\voice-plugin\logs\kokoro-hip-crash.log` and the
matching `kokoro-hip-*.dmp` for post-mortem.

### `ryzen-ai-lt-1.7.1.exe` install fails on Strix Halo

The "lt" (light) installer is the one that supports Strix Halo; the
non-lt full installer is Strix-Point-only in 1.7.0. Make sure you
have the `-lt-` variant.

### Where's the `detect_npu.ps1` helper?

```powershell
# Checks that all four prereqs are in place:
#   1. Ryzen AI SW 1.7.1 installed
#   2. NPU driver loaded and device enumerated
#   3. FlexML runtime DLLs on PATH
#   4. NPU is in performance pmode
D:\jam\lemondate\scripts\detect_npu.ps1
```
