"""Shared Recorder: accepts frames from the F9 PTT or Whisper-wake path,
encodes to WAV, POSTs to Lemonade /audio/transcriptions, types the
result via pyautogui.

One mutex guards state so the two activation paths can't collide.

source values the recorder understands:
    "ptt"           — F9 push-to-talk. Always types the transcription.
    "whisper_wake"  — WhisperWakeListener kicked us off on an energy
                      burst. The transcription is only typed if it
                      matches WAKE_PHRASE at the start; otherwise
                      silently dropped (wasn't addressed to us).
"""

from __future__ import annotations

import io
import logging
import re
import threading
import time

import numpy as np
import pyautogui
import requests
import soundfile as sf

from . import config
from .window_check import describe_current_focus, focus_passes_gate

log = logging.getLogger(__name__)

# Compiled once. Used by the whisper_wake source to gate typing — if
# the transcription doesn't match this at the start, the utterance
# wasn't addressed to us and we silently drop.
_WAKE_RE = re.compile(config.WAKE_PHRASE, re.IGNORECASE)


class Recorder:
    """Thread-safe recorder driven by external frame sources.

    F9 path:   start(source="ptt"), feed(frame)*, stop()
    Wake path: start(source="wake", prebuffer=..., vad_endpoint=True), feed(frame)*
               -> auto-stops on silence (energy VAD) or EOU_MAX_RECORDING_MS
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._recording = False
        self._source: str | None = None
        self._buffer: list[np.ndarray] = []
        self._started_at = 0.0
        self._last_voice_at = 0.0
        self._vad_endpoint = False

    def is_recording(self) -> bool:
        return self._recording

    def start(
        self,
        *,
        source: str,
        prebuffer: np.ndarray | None = None,
        vad_endpoint: bool = False,
    ) -> bool:
        with self._lock:
            if self._recording:
                log.debug("start ignored, already recording from %s", self._source)
                return False
            self._recording = True
            self._source = source
            self._buffer = []
            if prebuffer is not None and prebuffer.size > 0:
                self._buffer.append(prebuffer.astype(np.int16, copy=False))
            self._started_at = time.monotonic()
            self._last_voice_at = self._started_at
            self._vad_endpoint = vad_endpoint
            log.info("recording started (source=%s, vad=%s)", source, vad_endpoint)
            return True

    def feed(self, frame: np.ndarray) -> None:
        """Append a frame. Triggers auto-stop for wake-word path when silence
        has persisted past EOU_SILENCE_MS or the total duration exceeds
        EOU_MAX_RECORDING_MS."""
        with self._lock:
            if not self._recording:
                return
            self._buffer.append(frame.astype(np.int16, copy=False))

            if self._vad_endpoint:
                rms = _rms_int16(frame)
                now = time.monotonic()
                if rms >= config.EOU_ENERGY_THRESHOLD:
                    self._last_voice_at = now
                silence_ms = (now - self._last_voice_at) * 1000
                total_ms = (now - self._started_at) * 1000
                if silence_ms >= config.EOU_SILENCE_MS or total_ms >= config.EOU_MAX_RECORDING_MS:
                    log.info(
                        "VAD endpoint (silence=%.0fms, total=%.0fms)",
                        silence_ms,
                        total_ms,
                    )
                    threading.Thread(
                        target=self._finish_locked_release,
                        name="recorder-finish",
                        daemon=True,
                    ).start()

    def _finish_locked_release(self) -> None:
        # Drop the lock before calling stop() to avoid nested-acquire.
        self.stop()

    def stop(self) -> None:
        with self._lock:
            if not self._recording:
                return
            buf = self._buffer
            self._buffer = []
            self._recording = False
            source = self._source
            self._source = None
            self._vad_endpoint = False
        if not buf:
            log.info("recording stopped (source=%s, empty buffer)", source)
            return
        audio = np.concatenate(buf).astype(np.int16, copy=False)
        duration_ms = (audio.size / config.SAMPLE_RATE) * 1000
        log.info(
            "recording stopped (source=%s, samples=%d, %.0f ms)",
            source,
            audio.size,
            duration_ms,
        )
        if duration_ms < config.MIN_CLIP_MS:
            log.info("skipping transcribe, clip too short (%.0f ms < %d)",
                     duration_ms, config.MIN_CLIP_MS)
            return
        # Pad still-short clips with silence on both sides so Whisper's
        # 30 s attention window gets something structurally resembling a
        # normal utterance. Without this, 0.8-1.5 s clips routinely
        # decode to stock phrases like "Thank you." instead of the
        # actual words.
        if duration_ms < config.MIN_PAD_MS:
            pad_samples = int(
                (config.MIN_PAD_MS - duration_ms) * config.SAMPLE_RATE / 1000 / 2
            )
            silence = np.zeros(pad_samples, dtype=np.int16)
            audio = np.concatenate([silence, audio, silence])
        try:
            text = _transcribe(audio)
        except Exception:
            log.exception("transcribe failed")
            return
        text = (text or "").strip()
        if not text:
            log.info("empty transcription, nothing to type")
            return
        if source == "whisper_wake":
            m = _WAKE_RE.search(text)
            if not m or m.start() > 8:
                # No wake phrase near the start — this utterance wasn't
                # for us. Silent drop (info-level log for debugging).
                log.info("whisper wake: no match in %r, dropping", text[:80])
                return
            # Short utterances sometimes make Whisper repeat the same
            # sentence with slightly different punctuation/wording, so
            # the raw "after the first match" slice can end up like
            # "what's the time?\n Hey Halo, what's the time?". Split on
            # newline first and keep only the first line, then strip
            # leading punctuation that the wake phrase left behind.
            command = text[m.end():]
            command = command.split("\n", 1)[0]
            command = command.lstrip(" ,.:;!?-").strip()
            if not command:
                log.info("whisper wake: matched %r but no command after", text)
                return
            log.info("whisper wake fired: %r -> command %r", text[:60], command)
            text = command
        if not focus_passes_gate():
            # Guard against focus drifting during Whisper's RTT (~100-300 ms).
            # Better to drop the transcript than type it into a random app.
            log.info("dropping transcript, focus shifted to %s: %r",
                     describe_current_focus(), text)
            return
        log.info("typing: %r (auto_submit=%s)", text, config.PTT_AUTO_SUBMIT)
        try:
            # Trailing space then Enter: the space gives apps with
            # trailing-newline sensitivity (e.g. shells with
            # completion menus) a chance to flush the autocomplete
            # list before the submit arrives. Skipped when
            # PTT_AUTO_SUBMIT=0 so the user can review before sending.
            pyautogui.typewrite(text + " ", interval=0.005)
            if config.PTT_AUTO_SUBMIT:
                time.sleep(0.05)
                pyautogui.press("enter")
        except Exception:
            log.exception("typewrite failed")


def _rms_int16(frame: np.ndarray) -> float:
    if frame.size == 0:
        return 0.0
    x = frame.astype(np.float32)
    return float(np.sqrt(np.mean(x * x)))


def _transcribe(audio: np.ndarray) -> str:
    buf = io.BytesIO()
    sf.write(buf, audio, config.SAMPLE_RATE, subtype="PCM_16", format="WAV")
    buf.seek(0)
    t0 = time.monotonic()
    data = {"model": config.WHISPER_MODEL}
    if config.WHISPER_LANGUAGE:
        data["language"] = config.WHISPER_LANGUAGE
    if config.WHISPER_PROMPT:
        data["prompt"] = config.WHISPER_PROMPT
    r = requests.post(
        config.TRANSCRIBE_ENDPOINT,
        files={"file": ("clip.wav", buf, "audio/wav")},
        data=data,
        timeout=60,
    )
    r.raise_for_status()
    latency_ms = (time.monotonic() - t0) * 1000
    payload = r.json()
    text = payload.get("text", "")
    log.info("whisper: %.0f ms -> %r", latency_ms, text)
    return text
