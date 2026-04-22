"""Persistent F5-TTS service running on the AMD GPU (ROCm/HIP).

F5-TTS is a flow-matching DiT (Diffusion Transformer) model. Pure eager
PyTorch on ROCm — no ``torch.compile``, no Dynamo guards, no per-shape
recompile spikes. Each generation is ``nfe_step`` transformer forwards
through mel-frames; first call is as fast as the thousandth.

Exposes the same ``POST /api/v1/audio/speech`` + ``GET /api/v1/health``
shape as kokoro_server.py so speak.py and the test harnesses work
unchanged — just point them at ``http://127.0.0.1:13307`` instead of
``:13306`` via ``TTS_URL`` / ``F5_PORT``.

Env overrides:
    F5_PORT        default 13307
    F5_DEVICE      default "cuda" (set "cpu" to skip the GPU)
    F5_NFE         default 32 — flow-matching step count. Higher is
                   cleaner; 16 is ~1.5x faster and still decent, 8 is
                   visibly noisy.
    F5_CFG         default 2.0 — classifier-free guidance strength
    F5_SPEED       default 1.15 — playback speed multiplier applied at
                   synthesis time (not a post-process resample). 1.0 is
                   the reference voice's natural pace; 1.1-1.2 feels
                   more conversational without hurting intelligibility.
    F5_TAIL_PAD_MS default 180 — silent milliseconds appended after
                   each generation so short utterances don't feel
                   clipped. F5's duration predictor occasionally
                   under-allocates final phonemes; a small pad hides
                   that cleanly.
    F5_REF_AUDIO   default uses the package's bundled basic_ref_en.wav
                   (a female narrator voice). Override to clone a
                   specific voice; audio should be 10-20 s of clean
                   mono speech, ideally 24 kHz.
    F5_REF_TEXT    transcript of F5_REF_AUDIO. Auto-transcribed by F5
                   if empty, but an accurate explicit transcript
                   produces noticeably better output prosody.
    F5_TARGET_PEAK default 0.5 — peak-normalize output to this level.
                   Set 0 to disable.
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
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

os.environ.setdefault("MIOPEN_LOG_LEVEL", "0")
# MIOpen autotuning on gfx1151 (Strix Halo) adds 30+ s per new conv shape
# on first launch. FAST mode uses heuristics and keeps cold start bounded.
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")

# TheRock PyTorch on Windows omits most of torch.distributed. F5-TTS pulls
# in encodec which imports torch.distributed.ReduceOp at module load. Stub
# it to an Enum so the import chain succeeds — the all_reduce wrapper that
# references it is only called during distributed training.
if not hasattr(torch.distributed, "ReduceOp"):
    import enum as _enum

    class _ReduceOpStub(_enum.Enum):
        SUM = 0
        PRODUCT = 1
        MIN = 2
        MAX = 3
        BAND = 4
        BOR = 5
        BXOR = 6
        AVG = 7
        PREMUL_SUM = 8

    torch.distributed.ReduceOp = _ReduceOpStub  # type: ignore[attr-defined]

# torchaudio 2.11 on Windows delegates load()/save() to torchcodec, which
# needs FFmpeg DLLs on PATH (torchcodec doesn't ship them). soundfile is
# already a dep and handles the WAV/FLAC that F5-TTS wants. Patch at
# module scope so the swap happens before f5_tts imports torchaudio.
import torchaudio as _torchaudio  # noqa: E402


def _load_via_soundfile(path, *args, **kwargs):
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    else:
        data = data.T
    return torch.from_numpy(np.ascontiguousarray(data)), sr


def _save_via_soundfile(path, wav, sample_rate, *args, **kwargs):
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu().numpy()
    if wav.ndim == 2:
        wav = wav.T
    sf.write(str(path), wav, sample_rate)


_torchaudio.load = _load_via_soundfile  # type: ignore[assignment]
_torchaudio.save = _save_via_soundfile  # type: ignore[assignment]

from f5_tts.api import F5TTS  # noqa: E402


log = logging.getLogger("f5_tts_server")

PORT = int(os.environ.get("F5_PORT", "13307"))
DEVICE = os.environ.get("F5_DEVICE", "cuda")
NFE = int(os.environ.get("F5_NFE", "32"))
CFG = float(os.environ.get("F5_CFG", "2.0"))
SPEED = float(os.environ.get("F5_SPEED", "1.15"))
TAIL_PAD_MS = int(os.environ.get("F5_TAIL_PAD_MS", "180"))
TARGET_PEAK = float(os.environ.get("F5_TARGET_PEAK", "0.5"))


def _default_ref_audio() -> str:
    import f5_tts
    return str(Path(f5_tts.__path__[0]) / "infer" / "examples" / "basic" / "basic_ref_en.wav")


REF_AUDIO = os.environ.get("F5_REF_AUDIO") or _default_ref_audio()
# The bundled basic_ref_en.wav is "Some call me nature, others call me mother
# nature." Provide that by default so the first call doesn't wait on
# auto-transcription.
REF_TEXT = os.environ.get(
    "F5_REF_TEXT",
    "Some call me nature, others call me mother nature.",
)


class SpeechRequest(BaseModel):
    model: str | None = None
    input: str
    voice: str | None = None
    response_format: str = "wav"
    # speed=None means "use the server-configured F5_SPEED". Clients
    # that want the reference voice's natural pace pass speed=1.0.
    speed: float | None = None


class F5Service:
    def __init__(self) -> None:
        device = DEVICE
        if device == "cuda" and not torch.cuda.is_available():
            log.warning("CUDA/HIP not available; falling back to CPU")
            device = "cpu"
        self._device = device
        device_name = (
            torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
        )
        log.info(
            "Loading F5-TTS on %s (nfe=%d, cfg=%.1f, speed=%.2f, tail_pad=%d ms, ref=%s)",
            device_name, NFE, CFG, SPEED, TAIL_PAD_MS, Path(REF_AUDIO).name,
        )
        t0 = time.monotonic()
        self._tts = F5TTS(device=device)
        log.info("F5-TTS load: %.1f s", time.monotonic() - t0)

        # Single-threaded inference worker. F5-TTS doesn't rely on CUDA
        # graphs, so threading-wise it would work fine across threads,
        # but pinning to one thread keeps a stable observable model of
        # GPU utilization and mirrors kokoro_server's pattern.
        self._job_queue: queue.Queue[tuple[str, float, Future]] = queue.Queue()
        self._worker = threading.Thread(
            target=self._worker_loop, name="f5-worker", daemon=True,
        )
        self._worker.start()
        self._warm()

    @property
    def device(self) -> str:
        return self._device

    def _warm(self) -> None:
        """One short generation to pay any one-time init costs (weight
        layout, kernel selection, first MIOpen handle). F5-TTS does not
        specialize on sequence length so we don't need a length sweep."""
        t0 = time.monotonic()
        try:
            self.synthesize("Hello.", speed=1.0)
            log.info("warmup %.0f ms", (time.monotonic() - t0) * 1000)
        except Exception:
            log.exception("warmup failed (server will still start)")

    def _worker_loop(self) -> None:
        while True:
            text, speed, fut = self._job_queue.get()
            try:
                t0 = time.monotonic()
                wav = self._synthesize_on_worker(text, speed)
                dt = (time.monotonic() - t0) * 1000
                log.info(
                    "synth %.0f ms (%d chars -> %d wav bytes)",
                    dt, len(text), len(wav),
                )
                fut.set_result(wav)
            except BaseException as exc:
                fut.set_exception(exc)

    def _synthesize_on_worker(self, text: str, speed: float) -> bytes:
        if not text.strip():
            return b""
        # Append a trailing space: F5's duration estimator is derived
        # from phoneme counts, and without a post-terminal token it
        # sometimes truncates the final consonant. A trailing space
        # gives the model an extra frame to land on.
        wav, sr, _spec = self._tts.infer(
            REF_AUDIO,
            REF_TEXT,
            text.rstrip() + " ",
            nfe_step=NFE,
            cfg_strength=CFG,
            speed=speed,
            remove_silence=False,
            show_info=lambda *a, **k: None,
        )
        audio = np.asarray(wav, dtype=np.float32).squeeze()
        # Belt-and-suspenders tail pad for the rare case the trailing
        # space isn't enough.
        if TAIL_PAD_MS > 0 and audio.size:
            pad = np.zeros(int(sr * TAIL_PAD_MS / 1000), dtype=np.float32)
            audio = np.concatenate([audio, pad])
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if TARGET_PEAK > 0 and peak > 0:
            audio = audio * (TARGET_PEAK / peak)
        buf = io.BytesIO()
        sf.write(buf, audio, sr, subtype="PCM_16", format="WAV")
        return buf.getvalue()

    def synthesize(self, text: str, *, speed: float | None = None, voice: str | None = None) -> bytes:
        # voice is accepted for API parity with kokoro_server but ignored:
        # F5-TTS clones the REF_AUDIO voice regardless. speed=None means
        # use the server-configured default (F5_SPEED). Callers that want
        # the reference voice's natural pace pass speed=1.0 explicitly.
        fut: Future[bytes] = Future()
        self._job_queue.put((text, SPEED if speed is None else float(speed), fut))
        return fut.result()


@contextlib.asynccontextmanager
async def _lifespan(_app: FastAPI):
    global service
    service = F5Service()
    yield


app = FastAPI(title="Voice Plugin F5-TTS Server (ROCm)", lifespan=_lifespan)
service: F5Service | None = None


@app.get("/api/v1/health")
def health():
    if service is None:
        return {"status": "loading"}
    return {
        "status": "ok",
        "model": "f5-tts-v1-base",
        "device": service.device,
        "port": PORT,
        "nfe": NFE,
        "cfg": CFG,
        "speed": SPEED,
        "tail_pad_ms": TAIL_PAD_MS,
    }


@app.post("/api/v1/audio/speech")
def audio_speech(req: SpeechRequest):
    if service is None:
        raise HTTPException(status_code=503, detail="Model still loading")
    wav = service.synthesize(req.input, speed=req.speed, voice=req.voice)
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
