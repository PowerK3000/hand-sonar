"""Per-frame pipeline: hand frame -> features -> filtering -> gestures/pointer -> Actions.

Independent of the camera and of Windows, so the whole behaviour can be driven by synthetic
hands in tests.
"""

from __future__ import annotations

from collections import deque
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
from .tap_model import TapDetector
from .lift_detector import LiftDetector

# A gap longer than this restarts filters and the pointer, so a hand re-entering the frame
# doesn't produce one huge jump from where it left.
TRACK_GAP_S = 0.25
# MediaPipe's left/right label flickers frame to frame on one hand (75 flips in 2.5 min on
# Kayden's first live test); the label only changes after this many consecutive frames.
HANDEDNESS_SWITCH_FRAMES = 15


class HandednessLatch:
    def __init__(self) -> None:
        self.current: str | None = None
        self._streak = 0

    def reset(self) -> None:
        self.current, self._streak = None, 0

    def __call__(self, label: str) -> str:
        if self.current is None or label == self.current:
            self.current, self._streak = label, 0
        else:
            self._streak += 1
            if self._streak >= HANDEDNESS_SWITCH_FRAMES:
                self.current, self._streak = label, 0
        return self.current


class Controller:
    def __init__(self, cfg: Config, gate: EngagementGate, taps: TapDetector | None = None) -> None:
        if cfg.taps.mode == "model" and taps is None:
            raise ValueError("[taps] mode = model needs a TapDetector (train one with handctl.train_taps)")
        self.cfg, self.gate = cfg, gate
        if cfg.taps.mode == "model":
            self.taps = taps
        elif cfg.taps.mode == "lift":
            self.taps = LiftDetector()
        else:
            self.taps = None
        self._moves: deque[tuple[float, float, float]] = deque()   # (t, dx, dy) for click rewind
        fc = cfg.filter
        self._shape_filter = OneEuroFilter(fc.min_cutoff_hz, fc.beta, fc.d_cutoff_hz)
        self._palm_filter = OneEuroFilter(fc.palm_min_cutoff_hz, fc.palm_beta, fc.d_cutoff_hz)
        self._size_filter = OneEuroFilter(fc.palm_min_cutoff_hz, fc.palm_beta, fc.d_cutoff_hz)
        self.pointer = Pointer(cfg.pointer)
        self.left = LeftPinch(cfg)
        self.right = RightPinch(cfg)
        self.scroll = Scroll(cfg)
        self.wake = Wake(cfg)
        self._handedness = HandednessLatch()
        self._last_t: float | None = None
        self.last_features: HandFeatures | None = None
        self.dropped_degenerate = 0
        self.clutched = False
        self._open_since: float | None = None

    def _filter(self, f: HandFeatures) -> HandFeatures:
        names = list(f.straightness)
        shape = np.concatenate([[f.pinch_index, f.pinch_middle, f.pinch_index_2d, f.pinch_middle_2d],
                                [f.straightness[n] for n in names], f.index_local])
        s = self._shape_filter(shape, f.t)
        return replace(f, pinch_index=float(s[0]), pinch_middle=float(s[1]),
                       pinch_index_2d=float(s[2]), pinch_middle_2d=float(s[3]),
                       straightness={n: float(v) for n, v in zip(names, s[4:4 + len(names)])},
                       index_local=s[4 + len(names):])

    def _filter_palm(self, f: HandFeatures) -> HandFeatures:
        """Cursor path: palm position/size get their own (heavier) smoothing."""
        palm_hand = self._palm_filter(np.append(f.palm_px, f.hand_px), f.t, speed_scale=f.hand_px)
        # Log palm size is dimensionless (0.1 = 10% bigger), so it filters in its own units.
        log_size = self._size_filter(np.array([np.log(max(f.palm_size_px, 1e-6))]), f.t)[0]
        return replace(f, palm_px=palm_hand[:2], hand_px=float(palm_hand[2]), palm_size_px=float(np.exp(log_size)))

    def _reset_tracking(self) -> None:
        self._shape_filter.reset()
        self._palm_filter.reset()
        self._size_filter.reset()
        self.pointer.reset()

    def step(self, frame: HandFrame | None, t: float, pointer_frame: HandFrame | None = None) -> list[Action]:
        """Advance one camera frame (frame=None when no hand is visible).

        `pointer_frame`: optionally a stabilised copy of the same frame for the cursor only.
        Gestures must read the raw landmarks: the cursor stabiliser smoothed quick taps away.
        """
        f: HandFeatures | None = None
        if frame is not None:
            if self._last_t is not None and frame.t - self._last_t > TRACK_GAP_S:
                self._handedness.reset()
            side = self._handedness(frame.handedness)
            if side != frame.handedness:
                frame = replace(frame, handedness=side)
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
        c = self.cfg.clutch
        was_clutched = self.clutched
        if not c.enabled or f is None:
            self.clutched, self._open_since = False, None
        else:
            curl = max(f.straightness.values())
            if not self.clutched:
                self.clutched = curl <= c.max_straightness
            elif curl > c.release_straightness:
                # Sticky: leave the clutch only once the hand has been clearly open for a moment,
                # or a flickering fist reading lets cursor motion through.
                self._open_since = self._open_since if self._open_since is not None else t
                if t - self._open_since >= c.release_s:
                    self.clutched, self._open_since = False, None
            else:
                self._open_since = None
        # A fist brings the thumb against the index: don't let that read as a pinch or scroll.
        gestures_on = engaged and not self.clutched
        clutch_rewind = self._rewind(t, c.rewind_s) if (self.clutched and not was_clutched) else []

        # Closing the hand into a fist moves the palm before the fist is recognised: undo that.
        actions: list[Action] = list(clutch_rewind)
        if self.taps is not None:
            tapped = self.taps.update(frame if f is not None else None)
            if tapped and gestures_on and not self.scroll.active:
                actions += self._rewind(t)
                actions.append(Action("click", tapped))
        else:
            hand_speed = self.pointer.last_speed
            actions += self.left.update(f, gestures_on, other_busy=self.right.busy, hand_speed=hand_speed)
            actions += self.right.update(f, gestures_on, other_busy=self.left.busy, hand_speed=hand_speed)
        # Scroll keeps its momentum through a clutch (a fist can't start a scroll: it needs the index out).
        actions += self.scroll.update(f, engaged, t, blocked=self.left.busy or self.right.busy)

        pf = None
        if f is not None:
            src = raw
            if pointer_frame is not None:
                try:
                    src = compute(replace(pointer_frame, handedness=f.handedness))
                except DegenerateHand:
                    pass
            pf = self._filter_palm(src)
        if f is None:
            self.pointer.reset()
        else:
            frozen = not engaged or self.clutched or self.right.busy or self.scroll.active
            if self.taps is None:
                frozen = frozen or self.left.freezes_pointer(f)
            dx, dy = self.pointer.update(pf.palm_px, pf.hand_px, pf.t, frozen, pf.palm_size_px)
            if dx or dy:
                actions.append(Action("move", dx=dx, dy=dy))
                self._moves.append((t, dx, dy))
        while self._moves and t - self._moves[0][0] > max(self.cfg.taps.rewind_s, self.cfg.clutch.rewind_s):
            self._moves.popleft()
        actions += self.wake.update(f, engaged, self.pointer.last_speed)
        return actions

    def _rewind(self, t: float, window_s: float | None = None) -> list[Action]:
        """Undo cursor motion from the last rewind_s: the tap began before the model saw it,
        and the hand drifts while tapping, so click where the cursor was when the tap started."""
        w = self.cfg.taps.rewind_s if window_s is None else window_s
        dx = sum(m[1] for m in self._moves if t - m[0] <= w)
        dy = sum(m[2] for m in self._moves if t - m[0] <= w)
        self._moves.clear()
        return [Action("move", dx=-dx, dy=-dy)] if (dx or dy) else []

    def release(self) -> list[Action]:
        """Drop everything held (pause/exit)."""
        self.scroll.stop()
        return self.left.release() + self.right.release()
