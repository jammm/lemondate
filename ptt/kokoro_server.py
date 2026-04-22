"""Persistent Kokoro TTS service running on the AMD GPU (ROCm/HIP).

Loads ``hexgrad/Kokoro-82M`` onto the first ``torch.cuda`` device at startup
(first call JIT-compiles ~15 s of MIOpen kernels; we pre-warm to amortize).
Exposes an OpenAI-compatible ``POST /api/v1/audio/speech`` endpoint, plus a
``GET /api/v1/health`` probe.

Drop-in for the Lemonade CPU Kokoro backend — ``speak.py`` just points at
``http://localhost:13306`` instead of Lemonade's ``13305``.

Env overrides:
    KOKORO_PORT   default 13306
    KOKORO_LANG   default "a" (American English)
    KOKORO_VOICE  default "af_heart"
    KOKORO_DEVICE default "cuda" (set "cpu" to bypass the GPU)
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import queue
import threading
import time
from concurrent.futures import Future

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

# Silence MIOpen's "workspace=0" warnings during JIT; they're informational.
os.environ.setdefault("MIOPEN_LOG_LEVEL", "0")

from kokoro import KPipeline  # noqa: E402 (after env var)

# Keep cuDNN/MIOpen enabled — the aten LSTM fallback produces numerically
# degraded audio on ROCm 7.2 (tinnitus-like high-frequency artifacts at
# utterance starts, raspy timbre on voiced sustain). MIOpen's LSTM path is
# slightly slower but preserves the model's original precision.
# Set KOKORO_DISABLE_CUDNN=1 if you want to experiment.
if os.environ.get("KOKORO_DISABLE_CUDNN", "0") == "1":
    torch.backends.cudnn.enabled = False

# Use full float32 for matmul (default is "highest"; confirm explicitly to
# defeat any global config that would downgrade precision for RNN-adjacent
# operations).
torch.set_float32_matmul_precision("highest")

# Dynamo's default cache_size_limit=8 is too small for us: a Claude reply
# paragraph comfortably hits 10-15 distinct sentence-length shapes, and the
# startup warmup sweep adds another ~20. Once the cache fills, Dynamo evicts
# compiled graphs for never-seen shapes and falls back to slow eager
# per-op dispatch (~2.3 s vs ~0.3 s for a cached shape). Each unique shape
# only costs a one-time ~1-2 s compile, so keeping a couple hundred of them
# around is cheap memory-wise and eliminates the "new sentence thrashes
# back to eager" cliff. 256 is comfortably above any realistic phoneme-
# length distribution (Kokoro's max context is 512 tokens).
torch._dynamo.config.cache_size_limit = int(
    os.environ.get("KOKORO_DYNAMO_CACHE", "256")
)

log = logging.getLogger("kokoro_server")

PORT = int(os.environ.get("KOKORO_PORT", "13306"))
LANG_CODE = os.environ.get("KOKORO_LANG", "a")
DEFAULT_VOICE = os.environ.get("KOKORO_VOICE", "af_heart")
DEVICE = os.environ.get("KOKORO_DEVICE", "cuda")
SAMPLE_RATE = 24_000
# hexgrad/Kokoro-82M peaks at ~0.19 for normal speech; Lemonade's Kokoros
# ONNX export peaks at ~0.48 for the same text. Match that with a peak
# normalize so the user doesn't need to crank the volume to compare.
TARGET_PEAK = float(os.environ.get("KOKORO_TARGET_PEAK", "0.5"))

# Mixed-precision opt-in. fp32 is the default because for a 82M-param
# LSTM-heavy model at batch=1, the autocast cast-op overhead outweighs
# fp16's arithmetic savings on ROCm gfx1201. Try KOKORO_DTYPE=bfloat16 or
# float16 if you want to experiment.
_AUTOCAST_DTYPES = {
    "float32":  None,
    "fp32":     None,
    "float16":  torch.float16,
    "fp16":     torch.float16,
    "half":     torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16":     torch.bfloat16,
}
_dtype_key = os.environ.get("KOKORO_DTYPE", "float32").lower()
AUTOCAST_DTYPE = _AUTOCAST_DTYPES.get(_dtype_key, None)


class SpeechRequest(BaseModel):
    """Mirrors OpenAI's /audio/speech body so existing clients (Lemonade's
    callers, our speak.py) work unchanged."""

    model: str | None = None
    input: str
    voice: str = DEFAULT_VOICE
    response_format: str = "wav"
    speed: float = 1.0


# One short text prompt is enough to warm the g2p pipeline, voice pack
# loading, and the first Dynamo trace. Further text-based warmup would be
# minutes-per-shape and a waste — we seed the remaining shape buckets via
# direct synthetic forward_with_tokens calls in _shape_warm() below.
_TEXT_WARMUP = ("Hi.",)


def _parse_shape_range(spec: str) -> list[int]:
    """Parse a comma-separated sweep spec like "5-150:2,150-260:4".

    Each piece is `lo-hi:step`. Returns a deduplicated sorted list of ints.
    Empty spec disables synthetic warmup."""
    out: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        rng, _, step_s = chunk.partition(":")
        lo_s, _, hi_s = rng.partition("-")
        lo, hi = int(lo_s), int(hi_s)
        step = int(step_s) if step_s else 1
        out.update(range(lo, hi + 1, step))
    return sorted(out)


# Default sweep covers every realistic single-sentence phoneme length.
# Kokoro wraps input_ids as [0, *ids, 0] so the seq lens here match
# the tensor shape the compiled forward sees (phoneme count + 2).
#
# Cost profile (measured on 9070 XT / gfx1201, backend=eager):
#   L=6-20:   ~0.1-0.9 s each — trivial
#   L=20-60:  ~0.7-2   s each — main production band for Claude replies
#   L=60-80:  ~2-3     s each — one long sentence
#   L=80+:    up to 30 s      — big paragraph chunks. DELIBERATELY SKIPPED.
#
# KPipeline chunks text at 510 phonemes max, so L can technically reach
# ~512. In practice Claude replies split on newlines and sentence
# boundaries before we hit the TTS, so typical L is 15-70 with a long
# tail of 70-120 for run-on sentences. Sweeping up to 80 covers >95 %
# of real traffic and keeps warmup under ~60 s; sentences past L=80
# pay a one-time ~5-20 s compile on first hit, which a streaming client
# can hide behind playback of earlier chunks.
_DEFAULT_SHAPE_SWEEP = "6-80:2"


class KokoroService:
    def __init__(self) -> None:
        device = DEVICE
        if device == "cuda" and not torch.cuda.is_available():
            log.warning("CUDA/HIP not available; falling back to CPU")
            device = "cpu"
        self._device = device
        device_name = (
            torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
        )
        log.info("Loading Kokoro on %s (lang=%s)", device_name, LANG_CODE)
        t0 = time.monotonic()
        self._pipeline = KPipeline(lang_code=LANG_CODE, device=device)
        log.info("Kokoro load: %.1f s", time.monotonic() - t0)
        self._lock = threading.Lock()

        # torch.compile on ROCm 7.2 / gfx1201 / triton-windows 3.6:
        #
        #   backend="eager"      — Dynamo graph capture + eager kernels.
        #                          Produces bit-correct audio (corr>0.9998 vs
        #                          pure eager) and ~3x faster than eager
        #                          thanks to removing per-op Python overhead.
        #                          This is the default when KOKORO_COMPILE=1.
        #   backend="aot_eager"  — also correct, ~2.8x speedup. Slower compile.
        #   backend="inductor"   — CURRENTLY BROKEN on this stack. Compiled
        #                          waveform decorrelates from eager
        #                          (peak ratio ~0.5, waveform corr <0.05)
        #                          while STFT-mag stays ~0.5 correlated —
        #                          sounds like garbled speech at half
        #                          volume. The bisection narrowed this down
        #                          to Inductor's Triton codegen; Dynamo
        #                          graph capture and AOT decomposition are
        #                          both fine. Inductor is opt-in via
        #                          KOKORO_COMPILE_BACKEND=inductor.
        #
        # See tmp/compile_bisect.py for the reproducer.
        _compile_raw = os.environ.get("KOKORO_COMPILE", "1").lower()
        _enable = _compile_raw not in ("0", "off", "false", "no", "")
        _backend = os.environ.get("KOKORO_COMPILE_BACKEND", "eager")
        if device == "cuda" and _enable:
            log.info("torch.compile(dynamic=True, backend=%s) on KModel.forward_with_tokens...", _backend)
            try:
                kw = dict(dynamic=True, fullgraph=False)
                if _backend == "inductor":
                    kw["mode"] = _compile_raw if _compile_raw not in ("1", "true", "yes", "on") else "default"
                else:
                    kw["backend"] = _backend
                # Compile forward_with_tokens (takes an already-built
                # input_ids LongTensor) rather than forward (takes a Python
                # `phonemes: str`). Otherwise Dynamo guards on the exact
                # phoneme STRING value every call, hits cache_size_limit=8
                # after the 8th distinct text, and falls back to eager —
                # observed as the 10-line poem taking 70 s via repeated
                # per-string recompiles. The string→tensor conversion stays
                # in the un-compiled Python wrapper (`forward`), so Dynamo
                # never sees it.
                #
                # mark_dynamic runs on the OUTSIDE (before compile) so the
                # seq-length dim is treated as fully dynamic from first
                # trace, rather than specializing on the first observed size.
                compiled_fwt = torch.compile(
                    self._pipeline.model.forward_with_tokens, **kw,
                )

                def _marked_fwt(input_ids, ref_s, speed=1):
                    if isinstance(input_ids, torch.Tensor) and input_ids.dim() >= 2:
                        try:
                            torch._dynamo.mark_dynamic(input_ids, 1)
                        except Exception:
                            pass
                    return compiled_fwt(input_ids, ref_s, speed)

                self._pipeline.model.forward_with_tokens = _marked_fwt
            except Exception:
                log.exception("torch.compile failed; falling back to eager")

        # Single-threaded inference worker so torch.compile's CUDA graphs
        # stay valid. FastAPI's threadpool would otherwise invalidate them
        # on every thread hop.
        self._job_queue: queue.Queue[tuple[str, str, Future]] = queue.Queue()
        self._worker = threading.Thread(
            target=self._worker_loop, name="kokoro-worker", daemon=True
        )
        self._worker.start()
        self._warm()

    @property
    def device(self) -> str:
        return self._device

    def _warm(self) -> None:
        """Two-phase warmup:

        1. One real text synthesis to amortize g2p + voice pack load +
           first Dynamo trace.
        2. Synthetic direct-to-`forward_with_tokens` calls at every seq
           length we'd expect in production. Bypassing g2p cuts warmup
           time roughly in half per shape and covers 3-4x more shapes.
        """
        total = time.monotonic()
        for i, text in enumerate(_TEXT_WARMUP, 1):
            t0 = time.monotonic()
            try:
                self.synthesize(text, voice=DEFAULT_VOICE)
                log.info("text-warm [%d/%d] %6.0f ms : %s",
                         i, len(_TEXT_WARMUP),
                         (time.monotonic() - t0) * 1000, text[:60])
            except Exception:
                log.exception("text-warm failed (server will still start)")
                break

        self._shape_warm()
        log.info("warmup complete: %.1f s total", time.monotonic() - total)

    def _shape_warm(self) -> None:
        spec = os.environ.get("KOKORO_SHAPE_WARMUP", _DEFAULT_SHAPE_SWEEP)
        try:
            seq_lens = _parse_shape_range(spec)
        except Exception:
            log.exception("couldn't parse KOKORO_SHAPE_WARMUP=%r; skipping", spec)
            return
        if not seq_lens or self._device != "cuda":
            return

        try:
            pack = self._pipeline.load_voice(DEFAULT_VOICE).to(self._device)
        except Exception:
            log.exception("shape-warm: couldn't load voice pack; skipping")
            return

        # Route through the same worker thread that real requests use so any
        # per-thread state (CUDA stream affinity, torch._dynamo tls, MIOpen
        # handles) matches production. We pass a Callable job to the queue
        # via the existing sentinel path.
        fut: Future[dict] = Future()
        self._job_queue.put(("__shape_warm__", (seq_lens, pack), fut))
        try:
            info = fut.result()
        except Exception:
            log.exception("shape-warm aborted")
            return
        ms = info["per_shape_ms"]
        if ms:
            srt = sorted(ms)
            log.info(
                "shape-warm: %d shapes in %.1f s (min=%.0f med=%.0f max=%.0f ms)",
                len(ms), info["elapsed"], srt[0], srt[len(srt) // 2], srt[-1],
            )

    def _worker_loop(self) -> None:
        """Consume (text, voice, future) jobs from the queue, run the
        Kokoro pipeline on this thread, set the future's result.

        Pinning inference to this thread keeps torch.compile's cached
        CUDA graphs valid across requests.
        """
        while True:
            text, voice, fut = self._job_queue.get()
            try:
                if text == "__shape_warm__":
                    seq_lens, pack = voice  # tuple smuggled as "voice"
                    fut.set_result(self._run_shape_warm(seq_lens, pack))
                    continue
                t0 = time.monotonic()
                wav = self._synthesize_on_worker(text, voice)
                dt = (time.monotonic() - t0) * 1000
                log.info("synth %.0f ms (%d chars -> %d wav bytes)",
                         dt, len(text), len(wav))
                fut.set_result(wav)
            except BaseException as exc:  # propagate to the caller
                fut.set_exception(exc)

    def _run_shape_warm(self, seq_lens, pack):
        """Worker-thread body for _shape_warm.

        Direct calls to `forward_with_tokens` at every requested seq length,
        using a pool of random (but valid) phoneme token ids for each call.
        The random-token mix matters a lot here: `forward_with_tokens`
        computes `pred_dur` from the input, and several downstream tensors
        have shapes that are functions of `pred_dur.sum()`. Using a
        constant fill id gives an artificially uniform pred_dur and seeds
        MIOpen's kernel cache for only ONE (input_len, output_len) pair
        per L — real text hits many more, which means cold-first-call
        latency stays high despite the seq-dim guards being primed.
        Randomising the tokens spreads the pred_dur.sum distribution so
        MIOpen ends up caching kernels for the same (roughly-gaussian)
        output-length band that real requests will hit.
        """
        model = self._pipeline.model
        device = self._device
        pack_rows = pack.shape[0]
        fwt = model.forward_with_tokens
        per_shape_ms: list[float] = []
        t_sweep = time.monotonic()
        n = len(seq_lens)

        vocab_ids = sorted({v for v in model.vocab.values() if v is not None and v != 0})
        if not vocab_ids:
            vocab_ids = [4]
        vocab_tensor = torch.tensor(vocab_ids, dtype=torch.long, device=device)

        # Seed a generator so the sweep is deterministic across restarts
        # (useful for comparing warmup behavior between configs).
        gen = torch.Generator(device=device).manual_seed(1729)

        for i, L in enumerate(seq_lens, 1):
            idx = torch.randint(
                0, len(vocab_ids), (1, L), dtype=torch.long,
                device=device, generator=gen,
            )
            ids = vocab_tensor[idx.squeeze(0)].unsqueeze(0)
            ids[:, 0] = 0
            ids[:, -1] = 0
            ref_s = pack[min(max(L - 3, 0), pack_rows - 1)]
            if self._device == "cuda" and AUTOCAST_DTYPE is not None:
                ctx = torch.autocast(device_type="cuda", dtype=AUTOCAST_DTYPE)
            else:
                ctx = contextlib.nullcontext()
            t0 = time.monotonic()
            try:
                with ctx:
                    audio, _ = fwt(ids, ref_s, 1)
                    if isinstance(audio, torch.Tensor):
                        audio.detach().cpu()
            except Exception as exc:
                log.warning("shape-warm L=%d failed: %s", L, exc)
                continue
            dt = (time.monotonic() - t0) * 1000
            per_shape_ms.append(dt)
            log.info("shape-warm [%d/%d] L=%d %6.0f ms", i, n, L, dt)
        return {
            "elapsed": time.monotonic() - t_sweep,
            "per_shape_ms": per_shape_ms,
        }

    def _synthesize_on_worker(self, text: str, voice: str) -> bytes:
        if not text.strip():
            return b""
        if self._device == "cuda" and AUTOCAST_DTYPE is not None:
            ctx = torch.autocast(device_type="cuda", dtype=AUTOCAST_DTYPE)
        else:
            ctx = contextlib.nullcontext()
        with ctx:
            gen = self._pipeline(text, voice=voice)
            chunks: list[np.ndarray] = []
            for _gs, _ps, audio in gen:
                if audio is None:
                    continue
                if isinstance(audio, torch.Tensor):
                    audio = audio.detach().cpu().numpy()
                chunks.append(np.asarray(audio, dtype=np.float32))
        if not chunks:
            return b""
        joined = np.concatenate(chunks)
        # Peak-normalize so clients don't have to crank the volume. The
        # raw PyTorch checkpoint outputs ~2.5x quieter than Lemonade's
        # Kokoros ONNX export for the same text.
        peak = float(np.max(np.abs(joined)))
        if TARGET_PEAK > 0 and peak > 0:
            joined = joined * (TARGET_PEAK / peak)
        buf = io.BytesIO()
        sf.write(buf, joined, SAMPLE_RATE, subtype="PCM_16", format="WAV")
        return buf.getvalue()

    def synthesize(self, text: str, *, voice: str) -> bytes:
        fut: Future[bytes] = Future()
        self._job_queue.put((text, voice, fut))
        return fut.result()


@contextlib.asynccontextmanager
async def _lifespan(_app: FastAPI):
    global service
    service = KokoroService()
    yield
    # No teardown — process exit reclaims VRAM


app = FastAPI(title="Voice Plugin Kokoro Server (ROCm)", lifespan=_lifespan)
service: KokoroService | None = None


@app.get("/api/v1/health")
def health():
    if service is None:
        return {"status": "loading"}
    return {
        "status": "ok",
        "model": "kokoro-v1",
        "device": service.device,
        "port": PORT,
    }


@app.post("/api/v1/audio/speech")
def audio_speech(req: SpeechRequest):
    if service is None:
        raise HTTPException(status_code=503, detail="Model still loading")
    wav = service.synthesize(req.input, voice=req.voice)
    if not wav:
        raise HTTPException(status_code=400, detail="Empty input")
    return Response(content=wav, media_type="audio/wav")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")


if __name__ == "__main__":
    main()
