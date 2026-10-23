"""Gesture state machines: filtered hand features in, Actions out.

Each machine is deterministic given (features, engaged, t), so behaviour is testable without
a camera. Every machine releases anything it holds (mouse button, momentum) the moment the
hand disengages or disappears; a stuck mouse button is the worst failure an input device has.
"""

from __future__ import annotations

import numpy as np

from .actions import Action
from .config import Config
from .features import HandFeatures


def index_pointing(f: HandFeatures, cfg: Config) -> bool:
    s, p = f.straightness, cfg.poses
    return (s["index"] >= p.extended_min and s["middle"] <= p.curled_max
            and s["ring"] <= p.curled_max and s["pinky"] <= p.curled_max)


def open_palm(f: HandFeatures, cfg: Config) -> bool:
    """Fingers straight AND fanned out: a relaxed resting hand is straight-ish but never spread."""
    return (all(v >= cfg.poses.extended_min for v in f.straightness.values())
            and f.spread_deg >= cfg.wake.min_spread_deg)


class LeftPinch:
    """Thumb-index works like a real left mouse button: pressed the moment the fingers touch,
    released when they part. Clicking used to wait for the release (to tell clicks from drags),
    which made every click feel late. Drag is now simply: touch, move, release. The cursor stays
    frozen while pressed until the hand clearly moves, so a click can't become a micro-drag."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.state = "idle"          # idle | pressed (cursor frozen) | dragging (cursor follows)
        self._origin: np.ndarray | None = None  # palm position (px) when the fingers touched
        # A touch only counts after the fingers were seen apart while engaged: a pinch that began
        # while paused (typing, clutch, hand out of view) must not press when control resumes.
        self._armed = False

    @property
    def busy(self) -> bool:
        return self.state != "idle"

    def freezes_pointer(self, f: HandFeatures | None) -> bool:
        return self.state == "pressed"

    def release(self) -> list[Action]:
        held = self.state != "idle"
        self.state, self._armed = "idle", False
        return [Action("up", "left")] if held else []

    def update(self, f: HandFeatures | None, engaged: bool, other_busy: bool) -> list[Action]:
        if f is None or not engaged:
            return self.release()
        c, d = self.cfg.click, f.pinch_index
        if self.state == "idle":
            if d > c.pinch_open:
                self._armed = True
            # Claim the pinch only if the thumb is nearer the index than the middle finger.
            if self._armed and d < c.pinch_close and d < f.pinch_middle and not other_busy:
                self.state, self._origin = "pressed", f.palm_px.copy()
                return [Action("down", "left")]
            return []
        if d > c.pinch_open:
            self.state = "idle"
            return [Action("up", "left")]
        if self.state == "pressed":
            # Net displacement since the touch, not summed jitter.
            moved = float(np.linalg.norm(f.palm_px - self._origin)) / f.hand_px
            if moved >= self.cfg.drag.start_move:
                self.state = "dragging"
        return []


class RightPinch:
    """Thumb-middle works like a real right mouse button: down on touch, up on release."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.pressed = False
        self._armed = False          # see LeftPinch._armed

    @property
    def busy(self) -> bool:
        return self.pressed

    def release(self) -> list[Action]:
        held, self.pressed, self._armed = self.pressed, False, False
        return [Action("up", "right")] if held else []

    def update(self, f: HandFeatures | None, engaged: bool, other_busy: bool) -> list[Action]:
        if f is None or not engaged:
            return self.release()
        c, d = self.cfg.right_click, f.pinch_middle
        if not self.pressed:
            if d > c.pinch_open:
                self._armed = True
            if self._armed and d < c.pinch_close and d < f.pinch_index and not other_busy:
                self.pressed = True
                return [Action("down", "right")]
            return []
        if d > c.pinch_open:
            self.pressed = False
            return [Action("up", "right")]
        return []


class Scroll:
    """Index finger alone slides; content follows the finger (iOS 'natural'); flicks coast."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.active = False
        self._prev: np.ndarray | None = None
        self._prev_t = 0.0
        self._vel = np.zeros(2)       # wheel units/s (h, v)
        self._coast_t: float | None = None
        self._pose_since: float | None = None

    def stop(self) -> None:
        self.active, self._prev, self._vel, self._coast_t = False, None, np.zeros(2), None

    def _finger_to_wheel(self, delta: np.ndarray, handedness: str) -> np.ndarray:
        s = self.cfg.scroll
        up = delta[1] if s.vertical_axis == "palm_forward" else delta[2]
        # Palm-frame x points to the thumb: screen-left for a right hand, screen-right for a left.
        right = -delta[0] if handedness == "right" else delta[0]
        sign = -1.0 if s.natural else 1.0
        h = sign * right * s.wheel_per_hand if s.horizontal else 0.0
        return np.array([h, sign * up * s.wheel_per_hand])

    def update(self, f: HandFeatures | None, engaged: bool, t: float, blocked: bool) -> list[Action]:
        s = self.cfg.scroll
        if not engaged:
            self.stop()
            return []
        pose = f is not None and not blocked and index_pointing(f, self.cfg) \
            and f.pinch_index > self.cfg.click.pinch_open
        if pose:
            if self._pose_since is None:
                self._pose_since = f.t
            if not self.active and (f.t - self._pose_since) * 1000 < s.hold_ms:
                return []
        else:
            self._pose_since = None
        if pose:
            self._coast_t = None
            if not self.active or self._prev is None:
                self.active, self._prev, self._prev_t = True, f.index_local.copy(), f.t
                self._vel = np.zeros(2)
                return []
            dt = f.t - self._prev_t
            wheel = self._finger_to_wheel(f.index_local - self._prev, f.handedness)
            self._prev, self._prev_t = f.index_local.copy(), f.t
            if dt > 0:
                self._vel = 0.6 * self._vel + 0.4 * (wheel / dt)
            return [Action("scroll", dx=wheel[0], dy=wheel[1])] if np.any(wheel) else []
        # Pose ended: optionally coast.
        if self.active:
            self.active, self._prev = False, None
            if s.momentum and np.linalg.norm(self._vel) >= s.momentum_min_wheel_per_s:
                self._coast_t = t
            else:
                self._vel = np.zeros(2)
        if self._coast_t is None:
            return []
        if f is not None and f.pinch_index < self.cfg.click.pinch_open:
            self.stop()  # grabbing to click stops a coasting page, like touching a scrolling phone
            return []
        dt = t - self._coast_t
        self._coast_t = t
        if dt <= 0:
            return []
        self._vel = self._vel * (s.momentum_decay_per_s ** dt)
        if np.linalg.norm(self._vel) < s.momentum_min_wheel_per_s:
            self.stop()
            return []
        step = self._vel * dt
        return [Action("scroll", dx=step[0], dy=step[1])]


class Wake:
    """Open palm held still -> one wake event; re-arms when the pose breaks."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._since: float | None = None
        self._fired = False

    def update(self, f: HandFeatures | None, engaged: bool, palm_speed: float) -> list[Action]:
        w = self.cfg.wake
        if not w.enabled or f is None or not engaged or not open_palm(f, self.cfg) or palm_speed > w.max_speed_hands_per_s:
            self._since, self._fired = None, False
            return []
        if self._since is None:
            self._since = f.t
        if not self._fired and (f.t - self._since) * 1000 >= w.hold_ms:
            self._fired = True
            return [Action("wake")]
        return []
