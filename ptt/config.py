"""Runtime configuration for the PTT daemon.

All paths and tunables live here so install_windows.ps1 can patch
LEMONADE_URL or TTS_LOCK_PATH without touching code.
"""

from __future__ import annotations

import os
from pathlib import Path

# 127.0.0.1 not localhost — see speak.py for the Windows IPv6-fallback note.
LEMONADE_URL = os.environ.get("LEMONADE_URL", "http://127.0.0.1:13305")
TRANSCRIBE_ENDPOINT = f"{LEMONADE_URL}/api/v1/audio/transcriptions"

# Whisper-Large-v3-Turbo: ~800 M params distilled from Large-v3. Near-Large
# accuracy at ~Medium speed. Whisper-Small mishears technical English
# ("what's the current time" -> "watch the word") at a rate that's
# unworkable for dictating Claude Code prompts. Override to a smaller
# variant (Whisper-Small / Whisper-Base) if you want faster STT at the
# cost of accuracy.
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "Whisper-Large-v3-Turbo")
# Force-English decoding. Leave WHISPER_PROMPT empty by default: on
# short or quiet clips Whisper will echo the prompt into the transcript
# (hallucination), which tanks wake-phrase matching because the
# transcription becomes the context sentence instead of whatever the
# user said. Set your own prompt if you have clean audio and want the
# domain bias.
WHISPER_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "en")
WHISPER_PROMPT = os.environ.get("WHISPER_PROMPT", "")

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
CHUNK_SAMPLES = 1280  # 80 ms at 16 kHz — convenient fixed frame size

PTT_HOTKEY = os.environ.get("PTT_HOTKEY", "f9")

# After typing the transcription, also press Enter so the prompt gets
# submitted automatically. Set PTT_AUTO_SUBMIT=0 if you want to review
# the text before sending.
PTT_AUTO_SUBMIT = os.environ.get("PTT_AUTO_SUBMIT", "1").lower() not in (
    "0", "off", "false", "no", "",
)

# Wake-word detection works by reusing Whisper itself — every energy-gated
# speech burst gets transcribed, and if the transcription matches
# WAKE_PHRASE at the start, we strip the match and type the remainder
# (otherwise silently drop). No keyword-spotter dependency, and changing
# the wake phrase is a one-env-var change rather than training a custom
# model.
WAKE_PHRASE = os.environ.get(
    "WAKE_PHRASE",
    # "hey halo" / "hi halo" + common Whisper mishearings of each. The
    # prefix is REQUIRED so bare "halo ..." (a sentence someone happens
    # to start with the word halo) doesn't trigger us.
    r"^\s*(?:hey|hi|high|he)[,\s]+"
    r"(?:halo|hello|hallo|hailo|halow|haloo|hollow)"
    r"[\s,.:;!?-]*",
)
WAKE_COOLDOWN_SECONDS = 2.0

# End-of-utterance detection for wake-word recordings. Once EOU_SILENCE_MS
# of below-threshold audio has accrued, we stop capturing and transcribe.
# Also gates the Whisper-wake listener: if a frame's RMS is below
# EOU_ENERGY_THRESHOLD while idle, we stay idle (we're not hearing speech
# yet, no point recording).
EOU_SILENCE_MS = 800
EOU_MAX_RECORDING_MS = 10_000
# int16 RMS. Tunable vs room noise — higher rejects more background,
# lower picks up quieter speech. 350 is the middle ground between
# catching soft "hey halo" and not firing on ambient HVAC / keyboard
# clicks (each of which Whisper then hallucinates into stock phrases
# like "Thank you." or "All right.").
EOU_ENERGY_THRESHOLD = int(os.environ.get("EOU_ENERGY_THRESHOLD", "350"))

# Whisper hallucinates badly on clips under ~1 s — 300-700 ms of audio
# typically decodes to one of a few stock phrases ("Thank you.", "All
# right.", "Bye.", "You."). Drop anything shorter and pad still-short
# clips up to MIN_PAD_MS total by wrapping them in silence, which
# regularizes Whisper's 30 s attention window for short inputs.
MIN_CLIP_MS = int(os.environ.get("MIN_CLIP_MS", "700"))
MIN_PAD_MS = int(os.environ.get("MIN_PAD_MS", "2000"))


# Cross-process signal: speak.py touches this file while Kokoro is
# playing so the wake listener stays quiet. Prevents the TTS reading a
# reply back from re-triggering the wake path on itself.
def _default_lock_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    if base:
        return Path(base) / "voice-plugin"
    return Path.home() / ".voice-plugin"


TTS_LOCK_PATH = Path(os.environ.get("TTS_LOCK_PATH", str(_default_lock_dir() / "tts_active.lock")))


def tts_is_active() -> bool:
    return TTS_LOCK_PATH.exists()
