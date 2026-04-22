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
- Windows 11 (tested on gfx1201 / RX 9070 XT; gfx1151 / Strix Halo supported)
- VS 2022 x64 developer environment
- Python 3.12 venv with TheRock ROCm SDK:
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install --index-url https://rocm.nightlies.amd.com/v2/gfx120X-all/ torch "rocm[libraries,devel]"
  rocm-sdk init
  ```
  For Strix Halo, swap `gfx120X-all` for `gfx1151` in the index URL.

Then:

```powershell
.\build.cmd
```

Artefacts land in `bin\`:
- `lemond.exe` / `lemonade.exe` -- HTTP orchestrator
- `whisper-server.exe` -- ROCm STT (spawned by lemond on demand)
- `kokoro-hip-server.exe` -- HIP Kokoro TTS (spawned by lemond on demand)

To re-target a different GPU:

```powershell
$env:GFX_TARGET = "gfx1151"          # Strix Halo iGPU
# or "gfx1151;gfx1201" for a fat binary
.\build.cmd
```

## PTT / wake-word daemon

Python daemon in [ptt/](ptt/). Provisions its own venv inside `lemondate/`:

```powershell
.\ptt\install.ps1
```

See [ptt/README.md](ptt/README.md) for runtime details.

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
- `ROCBLAS_USE_HIPBLASLT=1` baked into the server startup so F32 matmuls go through hipBLASLt (7x speedup on gfx1201).
