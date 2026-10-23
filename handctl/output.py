"""Turns Actions into real Windows input via SendInput.

Cursor moves are sent as absolute virtual-desktop coordinates computed from the current
cursor position, so Windows' own "enhance pointer precision" curve is not applied on top of
ours. Sub-pixel motion and fractional wheel deltas are accumulated so slow, smooth gestures
aren't rounded away.
"""

from __future__ import annotations

import ctypes
import math
import threading
import time
from ctypes import wintypes

from .actions import Action

INPUT_MOUSE = 0
MOUSEEVENTF_MOVE, MOUSEEVENTF_ABSOLUTE, MOUSEEVENTF_VIRTUALDESK = 0x0001, 0x8000, 0x4000
MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP = 0x0002, 0x0004
MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP = 0x0008, 0x0010
MOUSEEVENTF_WHEEL, MOUSEEVENTF_HWHEEL = 0x0800, 0x1000
SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN, SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN = 76, 77, 78, 79
ES_DISPLAY_REQUIRED, ES_SYSTEM_REQUIRED = 0x00000002, 0x00000001

_BUTTONS = {"left": (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP),
            "right": (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP)}


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class _INPUTUNION(ctypes.Union):
    # Sized to the largest member (KEYBDINPUT/HARDWAREINPUT are smaller than MOUSEINPUT).
    _fields_ = [("mi", _MOUSEINPUT)]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


class OutputError(OSError):
    pass


GLIDE_TICK_S = 0.008                    # 125 Hz
_DPI_AWARE_V2 = ctypes.c_void_p(-4)


def _make_dpi_aware(user32) -> None:
    """Without this, on Kayden's two-monitor setup GetCursorPos (scaled) and absolute SendInput
    (physical) disagree and small moves drift: +43 px came out as +13. Must run before any
    window is created; harmless if the process is already aware."""
    try:
        user32.SetProcessDpiAwarenessContext(_DPI_AWARE_V2)
    except AttributeError:          # pre-1703 Windows
        user32.SetProcessDPIAware()


class WindowsOutput:
    """`smooth_ms` > 0: cursor moves are queued and a 125 Hz thread glides the cursor toward the
    target (exponential, time constant smooth_ms), so ~20 fps camera updates don't show up as
    20 visible jumps per second. Clicks flush the queue first so they land where intended."""

    def __init__(self, smooth_ms: float = 0.0) -> None:
        self._u = ctypes.WinDLL("user32", use_last_error=True)
        _make_dpi_aware(self._u)
        self._k = ctypes.WinDLL("kernel32")
        self._fx = self._fy = 0.0          # sub-pixel remainder
        self._wh = self._wv = 0.0          # fractional wheel remainder
        self._held: set[str] = set()
        self._lock = threading.Lock()
        self._pending = [0.0, 0.0]
        self._tau = smooth_ms / 1000.0
        self._stop = threading.Event()
        self._glide: threading.Thread | None = None
        self.glide_errors = 0
        if self._tau > 0:
            self._glide = threading.Thread(target=self._glide_loop, name="cursor-glide", daemon=True)
            self._glide.start()

    def _glide_loop(self) -> None:
        last = time.perf_counter()
        while not self._stop.wait(GLIDE_TICK_S):
            now = time.perf_counter()
            frac = 1.0 - math.exp(-(now - last) / self._tau)
            last = now
            with self._lock:
                dx, dy = self._pending[0] * frac, self._pending[1] * frac
                if abs(self._pending[0]) + abs(self._pending[1]) < 0.5:
                    # Tail end: hand over the remainder in one go (sub-pixel accounting in _move
                    # keeps the fraction), so no motion is ever dropped.
                    dx, dy = self._pending
                    if not (dx or dy):
                        continue
                self._pending[0] -= dx
                self._pending[1] -= dy
                try:
                    self._move(dx, dy)
                except OutputError:
                    self.glide_errors += 1      # elevated window focused; keep gliding later

    def _flush(self) -> None:
        with self._lock:
            dx, dy = self._pending
            self._pending = [0.0, 0.0]
            if dx or dy:
                self._move(dx, dy)

    def close(self) -> None:
        self._stop.set()
        if self._glide is not None:
            self._glide.join(1)
        self._flush()

    def _send(self, flags: int, dx: int = 0, dy: int = 0, data: int = 0) -> None:
        inp = _INPUT(type=INPUT_MOUSE, u=_INPUTUNION(mi=_MOUSEINPUT(dx, dy, data & 0xFFFFFFFF, flags, 0, 0)))
        if self._u.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT)) != 1:
            raise OutputError(ctypes.get_last_error(), "SendInput was blocked (UIPI or secure desktop?)")

    def _move(self, dx: float, dy: float) -> None:
        self._fx += dx
        self._fy += dy
        ix, iy = int(self._fx), int(self._fy)
        if ix == 0 and iy == 0:
            return
        self._fx -= ix
        self._fy -= iy
        pt = wintypes.POINT()
        if not self._u.GetCursorPos(ctypes.byref(pt)):
            raise OutputError(ctypes.get_last_error(), "GetCursorPos failed")
        vx, vy = (self._u.GetSystemMetrics(i) for i in (SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN))
        vw, vh = (self._u.GetSystemMetrics(i) for i in (SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN))
        x = min(max(pt.x + ix, vx), vx + vw - 1)
        y = min(max(pt.y + iy, vy), vy + vh - 1)
        # Absolute coordinates are 0..65535 across the whole virtual desktop.
        ax = round((x - vx) * 65535 / max(1, vw - 1))
        ay = round((y - vy) * 65535 / max(1, vh - 1))
        self._send(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK, ax, ay)

    def _scroll(self, h: float, v: float) -> None:
        self._wh += h
        self._wv += v
        iv, ih = int(self._wv), int(self._wh)
        if iv:
            self._wv -= iv
            self._send(MOUSEEVENTF_WHEEL, data=iv)
        if ih:
            self._wh -= ih
            self._send(MOUSEEVENTF_HWHEEL, data=ih)

    def apply(self, action: Action) -> None:
        k = action.kind
        if k == "move":
            if self._tau > 0:
                with self._lock:
                    self._pending[0] += action.dx
                    self._pending[1] += action.dy
            else:
                self._move(action.dx, action.dy)
            return
        if k in ("down", "up", "click"):
            self._flush()
        if k == "scroll":
            self._scroll(action.dx, action.dy)
        elif k in ("down", "up", "click"):
            down, up = _BUTTONS[action.button]
            if k in ("down", "click"):
                self._send(down)
                self._held.add(action.button)
            if k in ("up", "click"):
                self._send(up)
                self._held.discard(action.button)
        elif k == "wake":
            # Nudge the display awake without moving the cursor.
            self._k.SetThreadExecutionState(ES_DISPLAY_REQUIRED | ES_SYSTEM_REQUIRED)
        else:
            raise ValueError(f"unknown action {k!r}")

    def release_all(self) -> None:
        """Never leave a button held down (called on pause, exit and crash)."""
        for b in list(self._held):
            try:
                self._send(_BUTTONS[b][1])
            finally:
                self._held.discard(b)
