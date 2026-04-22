"""Voice PTT + wake-word daemon for Claude Code on Windows.

Two activation paths, one pipeline:
  F9 (hold-to-talk, pynput)            -> Recorder.start/stop
  "hey halo ..." (Whisper wake)        -> Recorder.start/stop (VAD-endpointed)

Both routes POST the captured WAV to Lemonade's /api/v1/audio/transcriptions
endpoint (Whisper-Large-v3-Turbo by default) and pyautogui.typewrite the
result into the focused window.
"""
