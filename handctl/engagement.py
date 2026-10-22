"""Decides whether hand gestures are allowed to act right now.

The policy (`EngagementGate`) is pure and testable; `InputActivityHook` feeds it from global
low-level keyboard/mouse hooks. Events we inject ourselves via SendInput carry the
LLKHF_INJECTED / LLMHF_INJECTED flag and are ignored, otherwise our own cursor moves would
disengage the controller.
"""

from __future__ import annotations

import ctypes
import threading
import time
from ctypes import wintypes

from .config import EngagementCfg


class EngagementGate:
    def __init__(self, cfg: EngagementCfg) -> None:
        self.cfg = cfg
        self._last_physical = -1e9
        self._lock = threading.Lock()
        self.paused = False

    def note_keyboard(self, t: float) -> None:
        if self.cfg.disengage_on_keyboard:
            with self._lock:
                self._last_physical = max(self._last_physical, t)

    def note_mouse(self, t: float) -> None:
        if self.cfg.disengage_on_mouse:
            with self._lock:
                self._last_physical = max(self._last_physical, t)

    def engaged(self, t: float, score: float | None) -> bool:
        if self.paused or score is None or score < self.cfg.min_confidence:
            return False
        with self._lock:
            return (t - self._last_physical) * 1000 >= self.cfg.reengage_after_ms


# --- Win32 low-level hooks -------------------------------------------------------------
WH_KEYBOARD_LL, WH_MOUSE_LL = 13, 14
LLKHF_INJECTED, LLMHF_INJECTED = 0x10, 0x01
WM_QUIT = 0x0012

_LRESULT = ctypes.c_ssize_t
_HOOKPROC = ctypes.WINFUNCTYPE(_LRESULT, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)


class _KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [("vkCode", wintypes.DWORD), ("scanCode", wintypes.DWORD), ("flags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class _MSLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [("pt", wintypes.POINT), ("mouseData", wintypes.DWORD), ("flags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class InputActivityHook:
    """Runs WH_KEYBOARD_LL / WH_MOUSE_LL on a dedicated message-loop thread."""

    def __init__(self, gate: EngagementGate, clock=time.monotonic) -> None:
        self.gate, self.clock = gate, clock
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._user32.CallNextHookEx.argtypes = [wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM]
        self._user32.CallNextHookEx.restype = _LRESULT
        self._user32.SetWindowsHookExW.argtypes = [ctypes.c_int, _HOOKPROC, wintypes.HINSTANCE, wintypes.DWORD]
        self._user32.SetWindowsHookExW.restype = wintypes.HHOOK
        self._thread: threading.Thread | None = None
        self._thread_id = 0
        self._ready = threading.Event()
        self._error: OSError | None = None
        # Keep references: if the callbacks are garbage-collected Windows calls freed memory.
        self._kb_proc = _HOOKPROC(self._on_kb)
        self._ms_proc = _HOOKPROC(self._on_mouse)

    def _on_kb(self, code, wparam, lparam):
        if code >= 0:
            info = ctypes.cast(lparam, ctypes.POINTER(_KBDLLHOOKSTRUCT)).contents
            if not info.flags & LLKHF_INJECTED:
                self.gate.note_keyboard(self.clock())
        return self._user32.CallNextHookEx(None, code, wparam, lparam)

    def _on_mouse(self, code, wparam, lparam):
        if code >= 0:
            info = ctypes.cast(lparam, ctypes.POINTER(_MSLLHOOKSTRUCT)).contents
            if not info.flags & LLMHF_INJECTED:
                self.gate.note_mouse(self.clock())
        return self._user32.CallNextHookEx(None, code, wparam, lparam)

    def _run(self) -> None:
        kernel32 = ctypes.WinDLL("kernel32")
        self._thread_id = kernel32.GetCurrentThreadId()
        hooks = []
        for kind, proc in ((WH_KEYBOARD_LL, self._kb_proc), (WH_MOUSE_LL, self._ms_proc)):
            h = self._user32.SetWindowsHookExW(kind, proc, None, 0)
            if not h:
                self._error = ctypes.WinError(ctypes.get_last_error())
                break
            hooks.append(h)
        self._ready.set()
        if self._error is None:
            msg = wintypes.MSG()
            while self._user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                self._user32.TranslateMessage(ctypes.byref(msg))
                self._user32.DispatchMessageW(ctypes.byref(msg))
        for h in hooks:
            self._user32.UnhookWindowsHookEx(h)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="input-hooks", daemon=True)
        self._thread.start()
        self._ready.wait(5)
        if self._error is not None:
            raise self._error

    def stop(self) -> None:
        if self._thread and self._thread.is_alive():
            self._user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
            self._thread.join(2)
