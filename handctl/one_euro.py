"""One Euro filter (Casiez, Roussel & Vogel, CHI 2012), vectorised.

Adaptive low-pass: the cutoff rises with signal speed, so a resting hand is heavily
smoothed (no cursor jitter) while fast motion passes with little lag.
"""

from __future__ import annotations

import math

import numpy as np


def _alpha(cutoff_hz: float, dt: float) -> float:
    tau = 1.0 / (2 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    def __init__(self, min_cutoff_hz: float, beta: float, d_cutoff_hz: float) -> None:
        if min_cutoff_hz <= 0 or d_cutoff_hz <= 0 or beta < 0:
            raise ValueError("cutoffs must be > 0 and beta >= 0")
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff_hz, beta, d_cutoff_hz
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None
        self._t: float | None = None

    def reset(self) -> None:
        self._x = self._dx = self._t = None

    def __call__(self, x: np.ndarray, t: float, speed_scale: float = 1.0) -> np.ndarray:
        """Filter sample `x` at time `t`. Speed is divided by `speed_scale` before applying beta,
        so a pixel-space signal can use a beta tuned in hand lengths (pass hand size in px)."""
        if speed_scale <= 0:
            raise ValueError("speed_scale must be > 0")
        x = np.asarray(x, dtype=float)
        if self._x is None or self._t is None:
            self._x, self._dx, self._t = x.copy(), np.zeros_like(x), t
            return x.copy()
        dt = t - self._t
        if dt <= 0:
            # Duplicate/out-of-order timestamp: keep the previous estimate rather than divide by zero.
            return self._x.copy()
        dx = (x - self._x) / dt
        self._dx = self._dx + _alpha(self.d_cutoff, dt) * (dx - self._dx)
        cutoff = self.min_cutoff + self.beta * np.abs(self._dx) / speed_scale
        tau = 1.0 / (2 * np.pi * cutoff)
        a = 1.0 / (1.0 + tau / dt)
        self._x = self._x + a * (x - self._x)
        self._t = t
        return self._x.copy()
