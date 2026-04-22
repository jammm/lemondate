"""Foreground-window gate for the PTT daemon.

Both F9 and the wake-word listener consult ``focus_passes_gate()`` before
recording. Returns True when either:

1. ``claude.exe`` is a strict descendant of the foreground window's
   process — works when you're running claude in a shell that IS the
   foreground window (e.g. Cursor's integrated terminal, a plain PowerShell
   window), OR
2. The foreground window is a known terminal / editor host AND any
   ``claude.exe`` is alive anywhere on the system. Windows Terminal in
   particular breaks the direct-descendant chain because ConPTY spawns
   shells as children of ``OpenConsole.exe`` rather than of
   ``WindowsTerminal.exe``.

Rationale: STT that types into random windows is a footgun. Gating on Claude
Code's presence ensures transcriptions land in the terminal where Claude is
running, not in the browser or Slack that happens to be focused.

Override via env:
  ``PTT_REQUIRE_APPS="claude.exe,codex.exe"`` — comma-separated allow-list.
  ``PTT_REQUIRE_APPS="any"`` — disable the gate (back to "types anywhere").
"""

from __future__ import annotations

import ctypes
import logging
import os
from ctypes import wintypes
from typing import Iterable

import psutil

log = logging.getLogger(__name__)

_DEFAULT_REQUIRED_APPS = ("claude.exe",)

# Terminal / IDE hosts whose presence in the foreground counts as "user is
# probably interacting with a shell". On Windows Terminal the foreground
# window is WindowsTerminal.exe but child shells actually live under
# OpenConsole.exe, so a strict descendant check misses them. For these hosts
# we allow the gate to pass as long as SOME required-app process exists on
# the system.
_KNOWN_TERMINAL_HOSTS = frozenset({
    "windowsterminal.exe",
    "openconsole.exe",
    "conhost.exe",
    "cmd.exe",
    "powershell.exe",
    "pwsh.exe",
    "cursor.exe",
    "code.exe",
    "alacritty.exe",
    "wezterm.exe",
    "wezterm-gui.exe",
    "mintty.exe",
})

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_user32.GetForegroundWindow.restype = wintypes.HWND
_user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
_user32.GetWindowThreadProcessId.restype = wintypes.DWORD


def _required_apps() -> tuple[str, ...]:
    raw = os.environ.get("PTT_REQUIRE_APPS")
    if not raw:
        return _DEFAULT_REQUIRED_APPS
    return tuple(name.strip().lower() for name in raw.split(",") if name.strip())


def _gate_disabled(apps: Iterable[str]) -> bool:
    return any(a in ("*", "any", "none", "off") for a in apps)


def _foreground_pid() -> int:
    hwnd = _user32.GetForegroundWindow()
    if not hwnd:
        return 0
    pid = wintypes.DWORD()
    _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return int(pid.value)


def _is_descendant_of(pid: int, ancestor_pid: int) -> bool:
    try:
        p: psutil.Process | None = psutil.Process(pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False
    try:
        while p is not None:
            if p.pid == ancestor_pid:
                return True
            p = p.parent()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return False


def _foreground_process_name(fg_pid: int) -> str:
    try:
        return psutil.Process(fg_pid).name().lower()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return ""


def focus_passes_gate() -> bool:
    """Either ``claude.exe`` descends directly from the foreground window,
    or the foreground is a known terminal/editor host and some
    ``claude.exe`` is alive somewhere on the system."""
    apps = _required_apps()
    if not apps or _gate_disabled(apps):
        return True
    fg_pid = _foreground_pid()
    if fg_pid == 0:
        log.debug("focus gate: no foreground window")
        return False

    apps_set = {a.lower() for a in apps}
    claude_processes = [
        p for p in psutil.process_iter(["name", "pid"])
        if (p.info.get("name") or "").lower() in apps_set
    ]
    if not claude_processes:
        return False

    # Strict: any claude.exe descended from the foreground window?
    for proc in claude_processes:
        if _is_descendant_of(proc.pid, fg_pid):
            return True

    # Permissive: foreground is a known terminal/editor host AND claude
    # runs somewhere. This rescues Windows Terminal, whose ConPTY model
    # breaks the direct descendant chain.
    if _foreground_process_name(fg_pid) in _KNOWN_TERMINAL_HOSTS:
        return True

    return False


def describe_current_focus() -> str:
    """For logging: which process owns the current foreground window?"""
    pid = _foreground_pid()
    if pid == 0:
        return "<none>"
    try:
        p = psutil.Process(pid)
        return f"{p.name()} (pid={pid})"
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return f"<pid={pid}>"
