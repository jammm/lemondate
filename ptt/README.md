# ptt — PTT + wake-word daemon

F9 push-to-talk plus an always-listening **"hey halo"** wake-phrase, both
routed to lemond's HTTP API for STT (and, optionally, to the legacy
local TTS servers bundled here for A/B comparison).

## Purpose

Two activation paths, one pipeline:

- **F9 hold-to-talk** (`pynput` global hotkey) — press and hold F9, speak,
  release; the captured WAV is POSTed to
  `http://127.0.0.1:13305/api/v1/audio/transcriptions` and the
  transcription is typed into the focused window via `pyautogui`.
- **"hey halo ..." wake phrase** — an energy-gated listener feeds the
  mic into the same recorder whenever RMS crosses a threshold. The
  clip is transcribed by Whisper on lemond, and if the transcript
  starts with the configured wake phrase the remainder is typed into
  the focused window. Everything before the wake phrase — and any
  utterance that doesn't match it — is silently dropped.

Both paths are gated by `window_check.focus_passes_gate()`: audio only
gets processed when a recognised terminal/editor host is in the
foreground with `claude.exe` alive somewhere on the system. This keeps
STT from typing into whatever random app the user has focused.

## Launched by

The consumer repo (`tts_tts_claude_code`) ships an installer shim
(`installers\run_ptt.ps1.tmpl`) which is substituted at install time to
point at this repo's `venv\Scripts\python.exe` and `ptt\ptt_daemon.py`.
**End users don't invoke this directly.**

## Architecture

| File | Role |
|---|---|
| `ptt_daemon.py` | Entry point. F9 hotkey + wake-word state machine, signal handling. |
| `recorder.py` | Mic capture, WAV encoding, HTTP POST to Whisper, `pyautogui` typing. Shared by both paths. |
| `whisper_wake_listener.py` | Always-on energy-gated listener that fires the recorder on speech. |
| `window_check.py` | Focus gate — only lets audio through when claude.exe is running and the foreground window is a known terminal/editor. Pure `ctypes` + `psutil`, no pywin32. |
| `config.py` | Env-var-driven config (`LEMONADE_URL`, `WAKE_PHRASE`, `PTT_HOTKEY`, energy/duration thresholds, etc.). |
| `f5_tts_server.py` | **Legacy.** Standalone Python F5-TTS HTTP server for the A/B path. Not the default TTS backend — lemond's in-process TTS is. |
| `kokoro_server.py` | **Legacy.** Standalone Python Kokoro HTTP server for the A/B path. Same — not default. |
| `__init__.py` | Package marker + docstring. |

The `f5_tts_server.py` / `kokoro_server.py` scripts are kept because
they're the reference implementations the lemond in-process TTS was
ported from, and are still useful for apples-to-apples comparisons
against the C++ backend. They pull in `torch`, `f5-tts`, `kokoro`,
`fastapi`, and `uvicorn`, which is most of what bloats
`requirements.txt` — the PTT daemon itself only needs `requests`,
`pynput`, `pyautogui`, `numpy`, `sounddevice`, `soundfile`, and
`psutil`.

## Install

`install.ps1` builds a venv at `<repo>\venv\` (one level above this
folder) and pip-installs torch plus `rocm[libraries,devel]` from
TheRock's nightly index, followed by the rest of `requirements.txt`:

```powershell
cd d:\jam\lemondate

# Default: gfx120X-all (covers RX 9070 XT / gfx1201 and other RDNA 4).
.\ptt\install.ps1

# Strix Halo APUs:
.\ptt\install.ps1 -GfxIndex gfx1151

# Use a specific Python:
.\ptt\install.ps1 -PythonExe C:\Python312\python.exe
```

Each pip step fails fast; `install.ps1` has `$ErrorActionPreference =
"Stop"` and checks `$LASTEXITCODE` after every install.

## Run standalone (for testing)

```powershell
d:\jam\lemondate\venv\Scripts\python.exe d:\jam\lemondate\ptt\ptt_daemon.py
```

Useful flags:

- `-v` / `--verbose` — debug logging (gate decisions, VAD frames, etc.)
- `--no-wake` — disable the "hey halo" listener, keep F9 only
- `--no-ptt` — disable F9, keep only the wake listener

Common env overrides (see `config.py` for the full list):

| Var | Default | Effect |
|---|---|---|
| `LEMONADE_URL` | `http://127.0.0.1:13305` | Base URL for Whisper endpoint. |
| `PTT_HOTKEY` | `f9` | Any `pynput.keyboard.Key` name. |
| `PTT_AUTO_SUBMIT` | `1` | `0` to skip the trailing Enter after typing. |
| `PTT_REQUIRE_APPS` | `claude.exe` | Comma list, or `any` to disable the focus gate. |
| `WAKE_PHRASE` | regex for `hey/hi halo` + mishearings | Change to retrain the wake word. |
| `WHISPER_MODEL` | `Whisper-Large-v3-Turbo` | Name passed to lemond's transcription endpoint. |
| `EOU_ENERGY_THRESHOLD` | `350` (int16 RMS) | Lower = more sensitive to quiet speech / background. |
