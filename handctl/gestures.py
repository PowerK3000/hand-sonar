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


class Touch:
    """Deliberate thumb-finger touch detector shared by both buttons.

    A touch = the gap was clearly open for `arm_s`, then snapped closed within `snap_s` while the
    hand was roughly still. On Kayden's front-camera recording a relaxed thumb drifts across the
    index in the image (false clicks); deliberate taps snap closed from open.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._open_since: float | None = None
        self._last_open: float | None = None

    def reset(self) -> None:
        self._open_since = self._last_open = None

    def gap(self, f: HandFeatures, own: str) -> tuple[float, float]:
        """(this finger's gap, the other finger's gap) in the configured metric."""
        if self.cfg.metric == "2d":
            pair = (f.pinch_index_2d, f.pinch_middle_2d)
        else:
            pair = (f.pinch_index, f.pinch_middle)
        return pair if own == "index" else (pair[1], pair[0])

    def touched(self, f: HandFeatures, own: str, hand_speed: float) -> bool:
        c = self.cfg
        d, other = self.gap(f, own)
        if d > c.pinch_open:
            if self._open_since is None:
                self._open_since = f.t
            self._last_open = f.t
            return False
        if self._last_open is None or f.t - self._last_open > c.snap_s:
            self._open_since = None          # drifted closed slowly: not a tap; must reopen
            return False
        armed = self._open_since is not None and self._last_open - self._open_since >= c.arm_s
        if armed and d < c.pinch_close and d < other and hand_speed <= c.still_max:
            self.reset()
            return True
        return False

    def opened(self, f: HandFeatures, own: str) -> bool:
        return self.gap(f, own)[0] > self.cfg.pinch_open


class LeftPinch:
    """Thumb-index works like a real left mouse button: pressed the moment a deliberate touch is
    seen, released when the fingers part. Drag = touch, move, release. The cursor stays frozen
    while pressed until the hand clearly moves, so a click can't become a micro-drag."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.state = "idle"          # idle | pressed (cursor frozen) | dragging (cursor follows)
        self._origin: np.ndarray | None = None
        self._t_press = 0.0
        self.touch = Touch(cfg.click)

    @property
    def busy(self) -> bool:
        return self.state != "idle"

    def freezes_pointer(self, f: HandFeatures | None) -> bool:
        return self.state == "pressed"

    def release(self) -> list[Action]:
        held = self.state != "idle"
        self.state = "idle"
        self.touch.reset()                   # after any interruption the fingers must reopen first
        return [Action("up", "left")] if held else []

    def update(self, f: HandFeatures | None, engaged: bool, other_busy: bool,
               hand_speed: float = 0.0) -> list[Action]:
        if f is None or not engaged:
            return self.release()
        if self.state == "idle":
            if self.touch.touched(f, "index", hand_speed) and not other_busy:
                self.state, self._origin, self._t_press = "pressed", f.palm_px.copy(), f.t
                return [Action("down", "left")]
            return []
        if self.touch.opened(f, "index"):
            self.state = "idle"
            self.touch.touched(f, "index", hand_speed)   # starts the open timer for the next tap
            return [Action("up", "left")]
        if self.state == "pressed":
            moved = float(np.linalg.norm(f.palm_px - self._origin)) / f.hand_px
            if moved >= self.cfg.drag.start_move:
                self.state = "dragging"
            elif f.t - self._t_press > self.cfg.drag.static_hold_max_s:
                # Held without moving: almost always a thumb resting on the index, not intent.
                # Live, one such press stayed down ~30 s (a stuck button selects/drags things).
                return self.release()
        return []


class RightPinch:
    """Thumb-middle works like a real right mouse button: down on a deliberate touch, up on release."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.pressed = False
        self._t_press = 0.0
        self.touch = Touch(cfg.right_click)

    @property
    def busy(self) -> bool:
        return self.pressed

    def release(self) -> list[Action]:
        held, self.pressed = self.pressed, False
        self.touch.reset()
        return [Action("up", "right")] if held else []

    def update(self, f: HandFeatures | None, engaged: bool, other_busy: bool,
               hand_speed: float = 0.0) -> list[Action]:
        if f is None or not engaged:
            return self.release()
        if not self.pressed:
            if self.touch.touched(f, "middle", hand_speed) and not other_busy:
                self.pressed, self._t_press = True, f.t
                return [Action("down", "right")]
            return []
        if f.t - self._t_press > self.cfg.drag.static_hold_max_s:
            return self.release()
        if self.touch.opened(f, "middle"):
            self.pressed = False
            self.touch.touched(f, "middle", hand_speed)
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
