# lemondate

Monorepo consolidating the HIP/ROCm-accelerated voice stack used by the [tts_tts_claude_code](https://github.com/jammm/tts_tts_claude_code) Claude-Code plugin. Everything the voice stack needs is embedded as first-class source here; the only external dependency is [TheRock](https://github.com/ROCm/TheRock)'s ROCm SDK.

```
lemondate/
|-- CMakeLists.txt            # top-level (builds ggml + ttscpp + whisper + kokoro-hip-server)
|-- build.cmd                 # one-shot builder (Windows)
|-- src/
|   |-- lemond/               # lemonade HTTP orchestrator (C++)
|   |-- ggml/                 # unified ggml (llama.cpp base + our kcpp dirtypatch ops + CUDA kernels)
|   |-- ttscpp/               # Kokoro TTS engine (HIP, from koboldcpp/otherarch/ttscpp + our patches)
|   |-- whisper/              # whisper.cpp + whisper-server.exe (HIP)
|   `-- kokoro-hip-server/    # new cpp-httplib server exposing /v1/audio/speech (powered by ttscpp)
|-- ptt/                      # Python PTT / wake-word daemon + Python TTS servers
|-- assets/                   # Kokoro voice packs, phonemizer data
`-- bin/                      # staged build output (lemond.exe / whisper-server.exe / kokoro-hip-server.exe)
```

## Build

Prereqs:
- Windows 11 (tested on gfx1201 / RX 9070 XT; gfx1151 / Strix Halo iGPU)
- VS 2022 x64 developer environment
- Python 3.12 venv with TheRock ROCm SDK. Use the pip index for your
  GPU — each ships matching hipBLASLt / rocBLAS kernel libraries:

  | GPU | pip index |
  |---|---|
  | RX 9070 / XT (gfx1201) | `https://rocm.nightlies.amd.com/v2/gfx120X-all/` |
  | Strix Halo iGPU (gfx1151) | `https://rocm.nightlies.amd.com/v2/gfx1151/` |
  | RX 7900 XTX (gfx1100) | `https://rocm.nightlies.amd.com/v2/gfx110X-all/` |

  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install --index-url <your-index> torch "rocm[libraries,devel]"
  rocm-sdk init
  ```

Then:

```powershell
.\build.cmd
```

Artefacts land in `bin\`:
- `lemond.exe` / `lemonade.exe` — HTTP orchestrator
- `whisper-server.exe` — ROCm STT (spawned by lemond on demand)
- `kokoro-hip-server.exe` — HIP Kokoro TTS (spawned by lemond on demand)

ROCm runtime DLLs (hipblas, rocblas, rocfft, hiprtc, etc.) are **not**
copied into `bin\` — they resolve at runtime from the venv's ROCm SDK
via `rocm-sdk path --root`/bin on PATH. The `run_lemond.ps1` shim
sets this up automatically.

To re-target a different GPU:

```powershell
$env:GFX_TARGET = "gfx1151"          # Strix Halo iGPU
.\build.cmd
```

## PTT / wake-word daemon

Python daemon in [ptt/](ptt/). Provisions its own venv inside `lemondate/`:

```powershell
.\ptt\install.ps1
```

See [ptt/README.md](ptt/README.md) for runtime details.

## Strix Halo / XDNA 2 NPU deployment

Validated on Ryzen AI Max+ 395 (Strix Halo): Whisper Large-v3-Turbo
on the XDNA 2 NPU at **RTF 0.105** (10x real-time), Kokoro TTS on
the gfx1151 iGPU at **0.41s warm**.

- **Phase 1 (working):** Whisper STT on the **XDNA 2 NPU** (via
  AMD's `amd/whisper.cpp` VitisAI fork), Kokoro TTS on the
  **gfx1151 iGPU**. Full setup in [docs/strix-halo.md](docs/strix-halo.md).
- **Phase 2 (blocked):** Kokoro TTS on the NPU is not possible with
  Ryzen AI SDK 1.7.1 — the VitisAI EP only supports CNN INT8 on STX
  hardware. The compiler produces valid AIE2P code but the runtime
  rejects the HW context. Details in [docs/kokoro-npu-future.md](docs/kokoro-npu-future.md).

Quick readiness check on a Strix Halo host:

```powershell
.\scripts\detect_npu.ps1
```

For standalone PTT testing (no lemond needed — just whisper-server +
PTT daemon):

```powershell
$env:TRANSCRIBE_ENDPOINT = "http://127.0.0.1:13305/inference"
$env:PTT_REQUIRE_APPS = "any"   # types into any focused window
```

## Consumers

- [jammm/tts_tts_claude_code](https://github.com/jammm/tts_tts_claude_code) -- the Claude Code plugin, installer, and service shims. Its installer takes `-LemondatePath` pointing at this repo's built install tree and wires up the F9 hotkey + wake word + Stop-hook TTS.

## Credits / upstream

lemondate is a vendored fusion of:
- [lemonade-sdk/lemonade](https://github.com/lemonade-sdk/lemonade) (HTTP orchestrator + model manager)
- [ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp) (STT)
- [LostRuins/koboldcpp](https://github.com/LostRuins/koboldcpp) (the `otherarch/ttscpp/` subtree and our gfx1201-hip patches to make Kokoro run on AMD GPUs)
- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (shared ggml tensor library)

Our additions on top:
- Kokoro-on-HIP correctness fixes: 128-byte tensor alignment, `reciprocal` kernel rewrite, staging-buffer `set_inputs`, shared backend instance.
- Fused `snake_1d` megakernel + CUDA kernels for the `ttscpp` custom ops (mod / cumsum_tts / ttsround / reciprocal / upscale_linear / conv_transpose_1d_tts / stft / istft) so the Kokoro graph runs entirely on the GPU.
- STFT/ISTFT/CONCAT kernel grid.y tiling to handle TTS output >13.6s without crashing HIP's 65535-block-per-dim limit.
- `ROCBLAS_USE_HIPBLASLT=1` baked into the server startup so F32 matmuls go through hipBLASLt (7x speedup on gfx1201).
- SIGABRT / SIGSEGV / SEH crash handlers + MiniDumpWriteDump for post-mortem debugging.
- 65k-entry `kokoro_ipa.embd` phonemizer override dictionary (from koboldcpp) + `/phonemize` debug endpoint for triaging mispronunciations.
