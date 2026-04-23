# Phase 2 — Kokoro TTS on the Strix Halo XDNA 2 NPU

**Status (2026-04-23): not supported yet. Phase 1 (Whisper STT on
NPU, see [strix-halo.md](./strix-halo.md)) ships first. Only once
that is solid do we start this work.**

Phase 2 goal: full NPU offload of Kokoro-82M. No CPU fallback on
hot paths, no iGPU co-execution — a single NPU-resident graph end
to end. Strix Halo's iGPU then stays completely free for whatever
LLM / rendering / other work wants it.

This doc captures the research and the port plan.

---

## Why Kokoro-NPU isn't shipping today

Unlike Whisper, Kokoro has **zero upstream precedent** on XDNA 2:

- AMD has no `amd/Kokoro-*` HF repo and no `RyzenAI-SW/Demos/TTS/`
  example.
- Microsoft's Windows ML samples don't cover it.
- Community: one 7-star hobby repo
  (`magicunicorn/kokoro-npu-quantized`) targets Phoenix (XDNA 1) via
  an unofficial MLIR-AIE2 runtime. Not Strix-Halo-compatible, not
  using the Vitis AI EP, not production-grade.
- Lemonade's own Kokoro backend is CPU-only:
  `config.json` has `"kokoro": { "cpu_bin": "builtin" }` (compare to
  `whispercpp` which has both `cpu_bin` and `npu_bin`).

Kokoro-82M is StyleTTS2 distilled — encoder + prosody predictor +
ISTFTNet decoder. The iSTFTNet decoder's `ConvTranspose1d` + custom
`STFT`/`iSTFT` tail is the awkward bit: it's perfectly fine on CPU
and CUDA, but it broke DirectML for other teams
([hexgrad/kokoro#79][hxg-79]) and nobody's certified it on Vitis AI
EP yet.

On paper the op coverage should work. Every op Kokoro uses is in
AMD's Vitis AI 1.7 BF16 support table ([ops_support][rai-ops]):

| Op | Used by Kokoro | Vitis AI BF16 support? |
|---|---|---|
| `Conv1d` | text encoder | **Yes** |
| `LSTM` | prosody predictor | **Yes** |
| `ConvTranspose1d` | custom_stft.py | **Yes** (BF16 / A16W8 / A8W8) |
| `STFT` | spectrogram | **Yes** (BF16 only) |
| Reshape / Transpose / Einsum | decoder tail | **Yes** |

But "on paper" is the operative phrase. No one has run this exact
graph through Vitis AI's graph partitioner yet.

[hxg-79]: https://github.com/hexgrad/kokoro/issues/79
[rai-ops]: https://ryzenai.docs.amd.com/en/latest/ops_support.html

---

## Concrete port plan (3-5 weeks)

### Week 1 — static-shape ONNX export

The stock `torch.onnx.export` of Kokoro produces a dynamic-shape
graph (phoneme count, style dim, output length are all symbolic).
Vitis AI's graph compiler needs **static shapes**. Pick a fixed
working envelope:

- phoneme context = 128 tokens (covers ~90% of Claude Code replies)
- style embedding = 256 dims
- audio output = 5s maximum (120 000 samples at 24 kHz)

Export separate static-shape models for:
- encoder + predictor (dynamic over phoneme count, capped at 128)
- decoder / iSTFTNet (dynamic over predicted frame count, capped at
  audio output / hop)

Starter scripts to fork:
[`adrianlyjak/kokoro-onnx-export`](https://github.com/adrianlyjak/kokoro-onnx-export)
— has a good per-node impact-analysis tool (`trial-quantization`)
that's useful for step 2.

### Week 2 — Quark BFP16 quantization

Run each static ONNX through AMD Quark 0.11.1 with `BFP16Spec`
using a ~50-utterance calibration set (phonemized English from
Librispeech or similar). Use `Quark Auto-Search` to pick per-node
strategies.

Inspect `vitisai_ep_report.json` to see op partition between NPU
and CPU fallback. Target: **>90% of ops on NPU**.

If any op falls back, iterate:

- Small cleanup passes with ONNX Simplifier / ONNX Optimizer.
- Graph surgery to replace the unsupported op with an equivalent
  one that's supported. Common one: `torch.fft.rfft` lowers to
  ops the EP doesn't fuse; replacing with manual DFT math can fix
  it.
- If an op genuinely can't run on NPU, keep it on CPU via EP
  partitioning — acceptable as long as the per-request overhead
  stays under ~30ms.

### Week 3 — quality validation

Compare waveform output against CPU reference on a 50-utterance
test set. Measure:

- **PESQ** (should be >3.5 vs CPU BFP16 oracle)
- **Mean Opinion Score** via a small listener panel (or CER via
  Whisper round-trip — if Whisper on the CPU-Kokoro output transcribes
  identically to Whisper on the NPU-Kokoro output, that's a strong
  signal)

Re-quantize any problem ops at higher precision (A16W8 → BF16)
as needed.

### Week 4 — productionize

- Pre-computed `.rai` cache per voice (Kokoro has ~60 voices, so
  this is ~60 small cache files). Ship them as a HF repo.
- Warm-up strategy: pre-load the first voice on service start so
  the first user-visible TTS call doesn't eat the JIT.
- Batch-1 latency tuning: Kokoro is always batch-1 in our use case,
  which exercises the EP's small-batch path. Verify no unexpected
  overhead vs batched inference.

### Week 5 — ship and monitor

- Add `LEMONADE_KOKORO_BACKEND=npu` wiring in `run_lemond.ps1.tmpl`
  mirroring the existing `hip`/`cpu` paths.
- Add `kokoro-npu-server.exe` to `lemondate/bin/` that loads the
  BFP16 ONNX models + `.rai` cache + pre-baked voice embeddings.
- Document the setup in `docs/strix-halo.md` alongside the Whisper
  NPU section.

---

## When does Phase 2 start?

Phase 2 starts only once Phase 1 is production-solid:

1. `LEMONADE_WHISPER_BACKEND=npu` yields real-time `base.en` on
   Strix Halo with no crashes over a multi-hour session.
2. First-call JIT compile is cached and reproducible across service
   restarts.
3. Wake-word + F9 PTT + Stop-hook transcription all route through
   the NPU path without regressions vs the current ROCm path.

Until those three are green, Kokoro stays on the HIP iGPU path and
this doc is planning material, not a task list.

---

## Full NPU offload only — no partial port, no iGPU fallback

The architectural directive is explicit: Kokoro must run **entirely
on the NPU** once Phase 2 ships. No "NPU encoder + CPU decoder" hack,
no iGPU-only HIP path sitting behind an env var. The iGPU is reserved
for whatever the user wants to do next (LLM inference, game, etc.),
not for the voice stack.

That means every op in Kokoro's graph must run on the XDNA 2 NPU
via the Vitis AI EP, or there's a graph-level replacement that does.
The known awkward bits (`ConvTranspose1d`, `STFT`/`iSTFT`) all have
Vitis AI BF16 support on paper, so this is a tractable engineering
problem — just not a cheap one.

If we hit a genuine op gap that can't be lowered or replaced, we
stop and reopen the design rather than silently fall back. Falling
back to CPU/iGPU after announcing "NPU TTS" is worse than shipping
nothing.

---

## Useful links

- Ryzen AI Vitis AI EP docs:
  <https://ryzenai.docs.amd.com/en/latest/modelrun.html>
- Quark quantizer:
  <https://quark.docs.amd.com/latest/onnx/tutorial_bfp16_quantization.html>
- AMD Whisper fork (template for Kokoro fork):
  <https://github.com/amd/whisper.cpp>
- AMD RyzenAI-SW demos:
  <https://github.com/amd/RyzenAI-SW>
- Kokoro ONNX export tooling:
  <https://github.com/adrianlyjak/kokoro-onnx-export>
- Kokoro canonical CPU ONNX:
  <https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX>
