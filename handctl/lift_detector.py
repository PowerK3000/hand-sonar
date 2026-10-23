"""Finger-lift clicks: index lift = left click, middle lift = right click.

Chosen by data (tools/gesture_snr.py, session cand1): seen from an overhead camera, a fingertip
lifting off the desk moves ~8.7x the rest jitter, versus ~3.6x for a thumb-index tap whose
contact is hidden. Training-free and relative to the hand's own recent pose, so posture changes
between sittings don't matter.

Signal per fingertip: position relative to the palm centroid in palm-size units (no rotation
alignment: aligning by a short palm vector swings distant fingertips with its angular noise).
A lift = the finger's displacement from its own baseline (median of the previous ~1 s) exceeds
`on_ratio` x that finger's typical jitter AND dominates the other fingers, then returns.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .hand import HandFrame

PALM = (0, 5, 9, 13, 17)
_PAIRS = [(a, b) for i, a in enumerate(PALM) for b in PALM[i + 1:]]
TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
BUTTON_FINGER = {"left": "index", "right": "middle"}


@dataclass(frozen=True)
class LiftCfg:
    # Tuned by tools/lift_eval.py: cand1 index lifts 7/14, ~0.7 false/min across rest/typing/
    # cursor/face in two other sittings (0 during rest).
    span_s: float = 0.25           # motion measured as position change over this span
    on: float = 0.8                # outward stroke size (palm sizes) to start
    back_ratio: float = 0.6        # return stroke must be >= this fraction of the outward one...
    back_cos: float = -0.3         # ...and point back the way it came (cosine with the outward stroke)
    max_dur_s: float = 0.7         # out-and-back within this time = a click
    lead: float = 1.2              # clicking finger must move more than every other non-thumb finger
    refractory_s: float = 0.8
    finger_to_hand: float = 2.0    # the finger's stroke must be >= this many times the hand's own motion.
                                   # Measured (cand1 vs t1/new-sitting): lifts ~1.9 finger vs ~0.4 hand;
                                   # cursor sweeps / face reaches ~1 hand motion for any finger motion.
    hand_abort: float = 1.0        # hand moving more than this mid-click cancels it


# Stillness is judged from points an index/middle lift doesn't move: lifting the index flexes its
# own knuckle, which made every real lift look like whole-hand motion.
ANCHOR = (0, 13, 17)
_ANCHOR_PAIRS = [(0, 13), (0, 17), (13, 17)]


def palm_state(frame: HandFrame) -> tuple[np.ndarray, float] | None:
    im = frame.image
    size = float(np.mean([np.linalg.norm(im[a] - im[b]) for a, b in _ANCHOR_PAIRS]))
    return (im[list(ANCHOR)].mean(axis=0), size) if size >= 5.0 else None


def tip_vector(frame: HandFrame) -> np.ndarray | None:
    im = frame.image
    size = float(np.mean([np.linalg.norm(im[a] - im[b]) for a, b in _PAIRS]))
    if size < 5.0:
        return None
    c = im[list(PALM)].mean(axis=0)
    return np.stack([(im[k] - c) / size for k in TIPS.values()])     # (5, 2)


class LiftDetector:
    """A click = a fingertip flicking out and back (relative to the palm). Measured as short-span
    strokes, so it needs no long baseline: live data showed back-to-back lifts contaminate any
    1 s baseline, and neighbouring fingers move along (mechanical coupling), so the clicking
    finger only has to lead, not dominate."""

    def __init__(self, cfg: LiftCfg = LiftCfg()) -> None:
        self.cfg = cfg
        self._hist: deque[tuple[float, np.ndarray]] = deque()
        self._out: tuple[float, str, np.ndarray] | None = None   # (time, button, outward stroke)
        self._last_fire = -1e9
        self.last_stroke: dict[str, float] = {k: 0.0 for k in TIPS}

    def reset(self) -> None:
        self._hist.clear()
        self._out = None

    def _stroke(self, t: float, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
        past = [(p, pc, ps) for tt, p, pc, ps in self._hist if tt <= t - self.cfg.span_s]
        return None if not past else (v - past[-1][0], past[-1][1], past[-1][2])

    def update(self, frame: HandFrame | None) -> str | None:
        if frame is None:
            self.reset()
            return None
        v = tip_vector(frame)
        ps = palm_state(frame)
        if v is None or ps is None:
            return None
        t, c = frame.t, self.cfg
        prev = self._stroke(t, v)
        self._hist.append((t, v, ps[0], ps[1]))
        while self._hist and self._hist[0][0] < t - c.span_s - 0.3:
            self._hist.popleft()
        if prev is None:
            return None
        stroke, palm_then, size_then = prev
        palm_move = np.linalg.norm(ps[0] - palm_then) / ps[1] + abs(np.log(ps[1] / size_then))
        if palm_move > c.hand_abort:
            self._out = None                # whole hand moving: not a click
            return None
        mag = np.linalg.norm(stroke, axis=1)
        self.last_stroke = dict(zip(TIPS, map(float, mag)))
        names = list(TIPS)
        if self._out is not None and t - self._out[0] > c.max_dur_s:
            self._out = None
        if self._out is None:
            if t - self._last_fire < c.refractory_s:
                return None
            for button, finger in BUTTON_FINGER.items():
                i = names.index(finger)
                others = [mag[j] for j, n in enumerate(names) if n not in (finger, "thumb")]
                if mag[i] > c.on and mag[i] > c.lead * max(others) and mag[i] >= c.finger_to_hand * palm_move:
                    self._out = (t, button, stroke[i].copy())
                    break
            return None
        t0, button, out = self._out
        i = names.index(BUTTON_FINGER[button])
        if t - t0 < c.span_s * 0.75:
            return None                     # still inside the outward stroke
        back = stroke[i]
        nb, no = np.linalg.norm(back), np.linalg.norm(out)
        if nb >= c.back_ratio * no and np.dot(back, out) / (nb * no + 1e-9) <= c.back_cos:
            self._out = None
            self._last_fire = t
            return button
        return None
