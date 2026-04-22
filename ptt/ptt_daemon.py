"""PTT daemon entry point.

Holds the process alive. Spawns:
  - pynput global keyboard listener (F9 keydown/keyup for push-to-talk)
  - WhisperWakeListener background thread — energy-gated capture +
    Whisper transcription, matches WAKE_PHRASE against the transcript

Both activation paths funnel into the same Recorder, which talks to the
Lemonade Whisper endpoint and types the result via pyautogui.

Run directly during development:
    python -m ptt.ptt_daemon

After install_windows.ps1 copies this tree into %LOCALAPPDATA%\\voice-plugin\\,
Task Scheduler (optional) or start_services.ps1 launches it.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time

import numpy as np
import sounddevice as sd
from pynput import keyboard

from . import config
from .recorder import Recorder
from .whisper_wake_listener import WhisperWakeListener
from .window_check import describe_current_focus, focus_passes_gate

log = logging.getLogger("ptt")


class PTTKeyHandler:
    """Global F9 press-and-hold.

    If a WhisperWakeListener is running, PTT piggybacks on its mic
    stream (the listener's main loop already feeds the recorder when
    `recorder.is_recording()` is true) and seeds the recorder with the
    listener's rolling prebuffer on keydown. This catches audio spoken
    just before the key actually landed, and avoids opening a second
    concurrent WASAPI InputStream on the same device (two streams on
    the same mic produced duplicate/interleaved frames and garbled
    transcripts).

    If no wake listener is available (--no-wake), PTT falls back to
    opening its own InputStream while held.
    """

    def __init__(
        self,
        recorder: Recorder,
        stop_event: threading.Event,
        wake_listener: "WhisperWakeListener | None" = None,
    ):
        self.recorder = recorder
        self.stop_event = stop_event
        self._wake_listener = wake_listener
        self._held = False
        self._owns_stream = False
        self._stream: sd.InputStream | None = None
        self._target_key = _parse_hotkey(config.PTT_HOTKEY)

    def on_press(self, key) -> None:
        if self._is_target(key) and not self._held:
            self._held = True
            if not focus_passes_gate():
                log.info("PTT suppressed: focus is %s", describe_current_focus())
                return

            prebuffer = None
            if self._wake_listener is not None:
                prebuffer = self._wake_listener.get_prebuffer()
                # Prevent wake path from firing on PTT tail audio after
                # release — it shares the same mic stream.
                self._wake_listener.note_external_fire()

            if not self.recorder.start(
                source="ptt",
                vad_endpoint=False,
                prebuffer=prebuffer,
            ):
                return

            # Wake listener's main loop will feed subsequent frames
            # while recorder.is_recording(). Only open a dedicated
            # stream when it isn't running.
            if self._wake_listener is None:
                self._open_stream()

    def on_release(self, key) -> None:
        if self._is_target(key) and self._held:
            self._held = False
            self._close_stream()
            self.recorder.stop()
        if _is_quit_combo(key):
            log.info("Esc+Esc: requesting shutdown")
            self.stop_event.set()

    def _is_target(self, key) -> bool:
        return key == self._target_key

    def _open_stream(self) -> None:
        def _callback(indata, frames, time_info, status):
            if status:
                log.debug("ptt stream status: %s", status)
            if frames == 0:
                return
            self.recorder.feed(np.asarray(indata, dtype=np.int16).flatten())

        try:
            self._stream = sd.InputStream(
                samplerate=config.SAMPLE_RATE,
                channels=config.CHANNELS,
                dtype=config.DTYPE,
                blocksize=config.CHUNK_SAMPLES,
                callback=_callback,
            )
            self._stream.start()
            self._owns_stream = True
        except Exception:
            log.exception("Failed to open PTT input stream")
            self.recorder.stop()

    def _close_stream(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                log.exception("Error closing PTT stream")
            finally:
                self._stream = None
                self._owns_stream = False


def _parse_hotkey(name: str):
    """Map a friendly config name like 'f9' to a pynput Key value."""
    name = (name or "").strip().lower()
    if hasattr(keyboard.Key, name):
        return getattr(keyboard.Key, name)
    return keyboard.KeyCode.from_char(name)


def _is_quit_combo(key) -> bool:
    # Reserve nothing by default. If you want a kill switch, wire it here.
    return False


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Voice PTT + wake-word daemon")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--no-wake", action="store_true", help="Disable wake-word listener")
    parser.add_argument("--no-ptt", action="store_true", help="Disable F9 push-to-talk")
    args = parser.parse_args(argv)

    _setup_logging(args.verbose)
    log.info("Starting PTT daemon. Lemonade=%s, Whisper=%s",
             config.LEMONADE_URL, config.WHISPER_MODEL)
    log.info("Hotkey=%s, WakePhrase=%r", config.PTT_HOTKEY, config.WAKE_PHRASE)

    stop_event = threading.Event()
    recorder = Recorder()

    wake: WhisperWakeListener | None = None
    if not args.no_wake:
        wake = WhisperWakeListener(recorder, stop_event)
        wake.start()

    key_listener: keyboard.Listener | None = None
    if not args.no_ptt:
        handler = PTTKeyHandler(recorder, stop_event, wake_listener=wake)
        key_listener = keyboard.Listener(on_press=handler.on_press, on_release=handler.on_release)
        key_listener.start()
        src = "shared wake-listener stream" if wake is not None else "own stream"
        log.info("F9 push-to-talk armed (audio: %s, prebuffer=%d ms)",
                 src, config.PTT_PREBUFFER_MS if wake is not None else 0)

    def _shutdown(*_):
        log.info("Shutdown requested")
        stop_event.set()

    try:
        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)
    except (AttributeError, ValueError):
        pass

    try:
        while not stop_event.is_set():
            time.sleep(0.25)
    except KeyboardInterrupt:
        stop_event.set()

    log.info("Stopping...")
    if key_listener is not None:
        key_listener.stop()
    if wake is not None:
        wake.join(timeout=3.0)
    recorder.stop()
    log.info("Stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
