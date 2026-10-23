"""Landmark stabiliser: stops keypoints jittering in place while keeping real motion.

Measured on Kayden's rest recordings: raw MediaPipe keypoints move ~2.5 px per frame at rest
(on a ~100-145 px hand), fingertips spike to ~15 px; 3D keypoints ~2 mm per frame.

Two stages per keypoint, in hand-size units so behaviour doesn't depend on camera distance:
1. One Euro filter: adaptive low-pass (heavy smoothing when slow, light when fast).
2. Leash (deadband): the output point stays exactly still while the filtered point remains
   within `radius` of it, and is dragged along by the excess once it leaves. Jitter inside the
   radius never reaches the output; a real movement passes through minus a fixed small offset,
   which recovers as soon as the motion stops.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .hand import MIDDLE_MCP, WRIST, HandFrame
from .one_euro import OneEuroFilter


@dataclass(frozen=True)
class StabilizerCfg:
    # Chosen by tools/stabilizer_eval.py on Kayden's rest + tap recordings:
    # rest jitter ~half of raw, tap contrast ~96% of raw.
    min_cutoff_hz: float = 2.0
    beta: float = 10.0                # per hand-length/s; high = taps pass with little lag
    d_cutoff_hz: float = 1.0
    leash_2d: float = 0.03            # hand lengths (~3-4 px on Kayden's hand)
    leash_3d: float = 0.03            # hand lengths (~3 mm)


def _leash(held: np.ndarray, x: np.ndarray, radius: float) -> np.ndarray:
    """Per-point deadband: move `held` toward `x` only by the distance beyond `radius`."""
    d = x - held
    dist = np.linalg.norm(d, axis=-1, keepdims=True)
    excess = np.maximum(dist - radius, 0.0)
    return held + d * np.where(dist > 1e-12, excess / np.maximum(dist, 1e-12), 0.0)


class LandmarkStabilizer:
    def __init__(self, cfg: StabilizerCfg = StabilizerCfg()) -> None:
        self.cfg = cfg
        self._f2 = OneEuroFilter(cfg.min_cutoff_hz, cfg.beta, cfg.d_cutoff_hz)
        self._f3 = OneEuroFilter(cfg.min_cutoff_hz, cfg.beta, cfg.d_cutoff_hz)
        self._held2: np.ndarray | None = None
        self._held3: np.ndarray | None = None
        self._last_t: float | None = None

    def reset(self) -> None:
        self._f2.reset(); self._f3.reset()
        self._held2 = self._held3 = None
        self._last_t = None

    def __call__(self, frame: HandFrame | None) -> HandFrame | None:
        if frame is None:
            self.reset()
            return None
        if self._last_t is not None and frame.t - self._last_t > 0.25:
            self.reset()                      # hand re-entered: don't smooth across the gap
        self._last_t = frame.t
        hand_px = float(np.linalg.norm(frame.image[MIDDLE_MCP] - frame.image[WRIST])) or 1.0
        hand_m = float(np.linalg.norm(frame.world[MIDDLE_MCP] - frame.world[WRIST])) or 1.0
        # Filter in hand units: normalise, filter, scale back.
        img = self._f2(frame.image.ravel() / hand_px, frame.t).reshape(21, 2) * hand_px
        wld = self._f3(frame.world.ravel() / hand_m, frame.t).reshape(21, 3) * hand_m
        if self._held2 is None:
            self._held2, self._held3 = img, wld
        else:
            self._held2 = _leash(self._held2, img, self.cfg.leash_2d * hand_px)
            self._held3 = _leash(self._held3, wld, self.cfg.leash_3d * hand_m)
        return replace(frame, image=self._held2.copy(), world=self._held3.copy())
