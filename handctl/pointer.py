"""Whole-hand relative cursor, trackpad style.

Motion is measured in hand lengths (palm displacement / hand size in pixels), so the same
physical gesture moves the cursor the same distance whether the hand is 30 cm or 80 cm from
the camera. A soft deadzone kills tracker jitter without making slow moves steppy, and an
acceleration curve lets small wrist-resting motions cross the whole screen.
"""

from __future__ import annotations

import numpy as np

from .config import PointerCfg


class Pointer:
    def __init__(self, cfg: PointerCfg) -> None:
        self.cfg = cfg
        self._prev_px: np.ndarray | None = None
        self._prev_hand: float = 0.0
        self._prev_t = 0.0
        self.last_step = 0.0     # hand lengths moved in the last frame (used by drag detection)
        self.last_speed = 0.0    # hand lengths / s

    def reset(self) -> None:
        self._prev_px = None
        self.last_step = self.last_speed = 0.0

    def update(self, palm_px: np.ndarray, hand_px: float, t: float, frozen: bool) -> tuple[float, float]:
        """Return the cursor move in screen pixels for this frame (0, 0 when frozen)."""
        if hand_px <= 0:
            raise ValueError("hand_px must be > 0")
        if self._prev_px is None:
            self._prev_px, self._prev_hand, self._prev_t = palm_px.copy(), hand_px, t
            return 0.0, 0.0
        step = (palm_px - self._prev_px) / (0.5 * (hand_px + self._prev_hand))
        dt = t - self._prev_t
        self._prev_px, self._prev_hand, self._prev_t = palm_px.copy(), hand_px, t
        mag = float(np.linalg.norm(step))
        self.last_step = mag
        self.last_speed = mag / dt if dt > 0 else 0.0
        # Always track position while frozen so the cursor doesn't jump when released.
        if frozen or mag <= self.cfg.deadzone:
            return 0.0, 0.0
        step = step * (mag - self.cfg.deadzone) / mag
        c = self.cfg
        boost = max(1.0, self.last_speed / c.accel_ref_hands_per_s) ** (c.acceleration - 1.0)
        dx, dy = step * c.pixels_per_hand * boost
        return (-dx if c.mirror else dx), dy
