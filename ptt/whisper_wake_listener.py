"""Whisper-based wake listener.

Reuses the same Whisper STT endpoint that handles F9 transcriptions and
detects the wake phrase after the fact. Lets us support arbitrary wake
phrases (e.g. "hey halo") without training a custom keyword-spotter.

Control flow:

    idle
      └─▶ mic sample RMS > EOU_ENERGY_THRESHOLD
            └─▶ recorder.start(source="whisper_wake", vad_endpoint=True)
                  └─▶ recorder captures until EOU_SILENCE_MS of silence
                        └─▶ recorder POSTs clip to Whisper
                              └─▶ if transcription matches WAKE_PHRASE:
                                    strip phrase, type remainder
                                  else:
                                    silent drop

The recorder does the heavy lifting (VAD-endpointed capture, Whisper
POST, focus gate, typing, auto-submit). We just watch audio energy and
flip the recorder on.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np
import sounddevice as sd

from . import config

log = logging.getLogger(__name__)


class WhisperWakeListener:
    """Energy-gated wake detector. Monitors the mic, and on each burst
    of speech kicks off a recorder capture with source="whisper_wake" —
    the recorder then transcribes via Lemonade/Whisper and types the
    result only if the transcript starts with WAKE_PHRASE."""

    def __init__(self, recorder, stop_event: threading.Event):
        self.recorder = recorder
        self.stop_event = stop_event
        self._thread: threading.Thread | None = None
        self._last_fire = 0.0

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="whisper-wake-listener", daemon=True,
        )
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(self) -> None:
        log.info(
            "Whisper wake listener ready (phrase=%r, model=%s)",
            config.WAKE_PHRASE, config.WHISPER_MODEL,
        )
        try:
            with sd.InputStream(
                samplerate=config.SAMPLE_RATE,
                channels=config.CHANNELS,
                dtype=config.DTYPE,
                blocksize=config.CHUNK_SAMPLES,
            ) as stream:
                while not self.stop_event.is_set():
                    frame, overflowed = stream.read(config.CHUNK_SAMPLES)
                    if overflowed:
                        log.debug("whisper wake stream overflow")
                    frame = frame.flatten()

                    # If a recording is already in progress (this
                    # listener kicked one, or F9 is held), just feed it.
                    if self.recorder.is_recording():
                        self.recorder.feed(frame)
                        continue

                    # Don't react while Kokoro/F5 is playing back — the
                    # speaker audio would loop back through the mic and
                    # trigger us on the TTS reading its own previous
                    # reply.
                    if config.tts_is_active():
                        continue

                    # Cooldown after a fire (wake-match OR not) to avoid
                    # immediately re-triggering on the tail of the same
                    # utterance.
                    if (time.monotonic() - self._last_fire) < config.WAKE_COOLDOWN_SECONDS:
                        continue

                    # Energy gate: only kick off an expensive recording
                    # when we're actually hearing sound. EOU_ENERGY_-
                    # THRESHOLD (int16 RMS) is tuned by the existing
                    # VAD-endpoint path so reusing it keeps the knob
                    # count small.
                    rms = _rms_int16(frame)
                    if rms < config.EOU_ENERGY_THRESHOLD:
                        continue

                    # Energy crossed threshold — start recording. The
                    # recorder's VAD endpoint will stop it on silence.
                    # recorder.start handles re-entry, so if something
                    # else just started it we'll quietly lose this
                    # frame.
                    if not self.recorder.start(source="whisper_wake", vad_endpoint=True):
                        continue
                    # Seed this frame into the buffer so the first sound
                    # isn't lost.
                    self.recorder.feed(frame)
                    self._last_fire = time.monotonic()
        except Exception:
            log.exception("Whisper wake listener crashed")


def _rms_int16(frame: np.ndarray) -> float:
    if frame.size == 0:
        return 0.0
    x = frame.astype(np.float32)
    return float(np.sqrt(np.mean(x * x)))
