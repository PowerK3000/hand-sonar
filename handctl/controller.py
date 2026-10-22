"""Per-frame pipeline: hand frame -> features -> filtering -> gestures/pointer -> Actions.

Independent of the camera and of Windows, so the whole behaviour can be driven by synthetic
hands in tests.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .actions import Action
from .config import Config
from .engagement import EngagementGate
from .features import DegenerateHand, HandFeatures, compute
from .gestures import LeftPinch, RightPinch, Scroll, Wake
from .hand import HandFrame
from .one_euro import OneEuroFilter
from .pointer import Pointer

# A gap longer than this restarts filters and the pointer, so a hand re-entering the frame
# doesn't produce one huge jump from where it left.
TRACK_GAP_S = 0.25


class Controller:
    def __init__(self, cfg: Config, gate: EngagementGate) -> None:
        self.cfg, self.gate = cfg, gate
        fc = cfg.filter
        self._shape_filter = OneEuroFilter(fc.min_cutoff_hz, fc.beta, fc.d_cutoff_hz)
        self._palm_filter = OneEuroFilter(fc.min_cutoff_hz, fc.beta, fc.d_cutoff_hz)
        self.pointer = Pointer(cfg.pointer)
        self.left = LeftPinch(cfg)
        self.right = RightPinch(cfg)
        self.scroll = Scroll(cfg)
        self.wake = Wake(cfg)
        self._last_t: float | None = None
        self.last_features: HandFeatures | None = None
        self.dropped_degenerate = 0

    def _filter(self, f: HandFeatures) -> HandFeatures:
        names = list(f.straightness)
        shape = np.concatenate([[f.pinch_index, f.pinch_middle],
                                [f.straightness[n] for n in names], f.index_local])
        s = self._shape_filter(shape, f.t)
        palm_hand = self._palm_filter(np.append(f.palm_px, f.hand_px), f.t, speed_scale=f.hand_px)
        return replace(f, pinch_index=float(s[0]), pinch_middle=float(s[1]),
                       straightness={n: float(v) for n, v in zip(names, s[2:2 + len(names)])},
                       index_local=s[2 + len(names):], palm_px=palm_hand[:2], hand_px=float(palm_hand[2]))

    def _reset_tracking(self) -> None:
        self._shape_filter.reset()
        self._palm_filter.reset()
        self.pointer.reset()

    def step(self, frame: HandFrame | None, t: float) -> list[Action]:
        """Advance one camera frame (frame=None when no hand is visible)."""
        f: HandFeatures | None = None
        if frame is not None:
            try:
                raw = compute(frame)
            except DegenerateHand:
                self.dropped_degenerate += 1
                raw = None
            if raw is not None:
                if self._last_t is None or raw.t - self._last_t > TRACK_GAP_S:
                    self._reset_tracking()
                f = self._filter(raw)
                self._last_t = raw.t
        self.last_features = f
        engaged = self.gate.engaged(t, f.score if f is not None else None)

        actions: list[Action] = []
        actions += self.left.update(f, engaged, self.pointer.last_step, other_busy=self.right.busy)
        actions += self.right.update(f, engaged, other_busy=self.left.busy)
        actions += self.scroll.update(f, engaged, t, blocked=self.left.busy or self.right.busy)

        if f is None:
            self.pointer.reset()
        else:
            frozen = (not engaged or self.left.freezes_pointer(f) or self.right.busy
                      or self.scroll.active)
            dx, dy = self.pointer.update(f.palm_px, f.hand_px, f.t, frozen)
            if dx or dy:
                actions.append(Action("move", dx=dx, dy=dy))
        actions += self.wake.update(f, engaged, self.pointer.last_speed)
        return actions

    def release(self) -> list[Action]:
        """Drop everything held (pause/exit)."""
        self.scroll.stop()
        self.right.pressed = False
        return self.left.release()
