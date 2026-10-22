"""Global hotkey via RegisterHotKey on a dedicated message-loop thread."""

from __future__ import annotations

import ctypes
import threading
from ctypes import wintypes
from typing import Callable

MOD = {"alt": 0x0001, "ctrl": 0x0002, "shift": 0x0004, "win": 0x0008}
MOD_NOREPEAT = 0x4000
WM_HOTKEY, WM_QUIT = 0x0312, 0x0012
_NAMED_VK = {f"f{i}": 0x6F + i for i in range(1, 13)} | {"space": 0x20, "pause": 0x13}


class HotkeyError(ValueError):
    pass


def parse(spec: str) -> tuple[int, int]:
    """'ctrl+alt+h' -> (modifiers, virtual-key)."""
    parts = [p.strip().lower() for p in spec.split("+") if p.strip()]
    if len(parts) < 2:
        raise HotkeyError(f"hotkey {spec!r} needs at least one modifier and a key")
    *mods, key = parts
    m = 0
    for name in mods:
        if name not in MOD:
            raise HotkeyError(f"unknown modifier {name!r} in {spec!r}")
        m |= MOD[name]
    if len(key) == 1 and key.isalnum():
        vk = ord(key.upper())
    elif key in _NAMED_VK:
        vk = _NAMED_VK[key]
    else:
        raise HotkeyError(f"unsupported key {key!r} in {spec!r}")
    return m, vk


class GlobalHotkey:
    def __init__(self, spec: str, callback: Callable[[], None]) -> None:
        self.mods, self.vk = parse(spec)
        self.spec, self.callback = spec, callback
        self._u = ctypes.WinDLL("user32", use_last_error=True)
        self._thread: threading.Thread | None = None
        self._tid = 0
        self._ready = threading.Event()
        self._error: OSError | None = None

    def _run(self) -> None:
        self._tid = ctypes.WinDLL("kernel32").GetCurrentThreadId()
        if not self._u.RegisterHotKey(None, 1, self.mods | MOD_NOREPEAT, self.vk):
            self._error = ctypes.WinError(ctypes.get_last_error())
            self._ready.set()
            return
        self._ready.set()
        msg = wintypes.MSG()
        try:
            while self._u.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                if msg.message == WM_HOTKEY:
                    self.callback()
        finally:
            self._u.UnregisterHotKey(None, 1)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="hotkey", daemon=True)
        self._thread.start()
        self._ready.wait(5)
        if self._error is not None:
            raise HotkeyError(f"could not register {self.spec!r} (already used by another app?): {self._error}")

    def stop(self) -> None:
        if self._thread and self._thread.is_alive():
            self._u.PostThreadMessageW(self._tid, WM_QUIT, 0, 0)
            self._thread.join(2)
