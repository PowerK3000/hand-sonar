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
    """Leash, not per-frame deadzone: live data showed the palm centre jitters a median 2.8% of
    hand length per frame at ~20 fps, so 80% of frames beat a per-frame deadzone and leaked
    10-15 px twitches. The anchor now stays put until the palm leaves `deadzone` (hand lengths)
    around it, then is dragged along: jitter inside the zone never reaches the cursor."""

    def __init__(self, cfg: PointerCfg) -> None:
        self.cfg = cfg
        self._anchor: np.ndarray | None = None   # leashed palm position, hand units relative to entry
        self._pos = np.zeros(2)
        self._prev_px: np.ndarray | None = None
        self._prev_hand: float = 0.0
        self._prev_size: float = 0.0
        self._prev_t = 0.0
        self.last_step = 0.0     # hand lengths moved in the last frame (used by drag detection)
        self.last_speed = 0.0    # hand lengths / s

    def reset(self) -> None:
        self._prev_px = None
        self._anchor = None
        self.last_step = self.last_speed = 0.0

    def update(self, palm_px: np.ndarray, hand_px: float, t: float, frozen: bool,
               palm_size_px: float = 0.0) -> tuple[float, float]:
        """Return the cursor move in screen pixels for this frame (0, 0 when frozen)."""
        if hand_px <= 0:
            raise ValueError("hand_px must be > 0")
        if self._prev_px is None:
            self._prev_px, self._prev_hand, self._prev_t, self._prev_size = palm_px.copy(), hand_px, t, palm_size_px
            return 0.0, 0.0
        raw_step = (palm_px - self._prev_px) / (0.5 * (hand_px + self._prev_hand))
        if self.cfg.vertical_source == "height" and palm_size_px > 0 and self._prev_size > 0:
            # Raising the hand toward an overhead camera makes the palm look bigger: map that to
            # cursor-up. In hand-length-equivalent units so leash/acceleration treat it like x.
            raw_step = np.array([raw_step[0], -np.log(palm_size_px / self._prev_size) * self.cfg.height_gain])
        dt = t - self._prev_t
        self._prev_px, self._prev_hand, self._prev_t, self._prev_size = palm_px.copy(), hand_px, t, palm_size_px
        mag = float(np.linalg.norm(raw_step))
        self.last_step = mag
        self.last_speed = mag / dt if dt > 0 else 0.0
        # Track the palm in hand units (integrated steps, so changing distance doesn't shift it).
        self._pos = (self._pos if self._anchor is not None else np.zeros(2)) + raw_step
        if self._anchor is None:
            self._anchor = self._pos.copy()
            return 0.0, 0.0
        d = self._pos - self._anchor
        dist = float(np.linalg.norm(d))
        if dist <= self.cfg.deadzone:
            return 0.0, 0.0
        step = d * (dist - self.cfg.deadzone) / dist
        self._anchor = self._anchor + step
        # Still track while frozen (pinch, scroll) so releasing doesn't jump the cursor.
        if frozen:
            return 0.0, 0.0
        c = self.cfg
        boost = max(1.0, self.last_speed / c.accel_ref_hands_per_s) ** (c.acceleration - 1.0)
        dx, dy = step * c.pixels_per_hand * boost
        return (-dx if c.mirror else dx), dy
