"""Whole-hand pose descriptors for the personal motion model.

Per frame: 2D keypoints in a hand-aligned image frame (the camera sees these reliably),
3D keypoints in the anatomical palm frame, 15 joint flexion angles, and three fingertip gaps.
Windows of W frames are summarised (mean/min/max/last-first) so the model sees *motion*,
not a single pose: a tap is recognisable by how the visible hand moves even when the thumb
tip itself is hidden.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .features import DegenerateHand, palm_frame
from .hand import FINGERS, INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP, THUMB_TIP, WRIST, HandFrame

FEATURE_VERSION = 2
WINDOW_S = 0.4                          # motion window the model sees
WINDOW_POINTS = 10                      # resampled to this many evenly spaced points
MAX_GAP_S = 0.15                        # a window with a longer tracking gap is not trusted
# Kept for callers that size buffers in frames; time-based windows replaced frame counts
# because live fps (10-25) differed from recording fps (~26) and broke the model.
WINDOW_FRAMES = WINDOW_POINTS
_CHAINS = [(1, 2, 3, 4)] + list(FINGERS.values())


def frame_vector(world: np.ndarray, image: np.ndarray, side: str) -> np.ndarray:
    size = np.linalg.norm(world[MIDDLE_MCP] - world[WRIST])
    if size < 0.02:
        raise DegenerateHand("hand too small")
    origin, rot = palm_frame(world, side)
    w3 = ((world - origin) @ rot.T / size).ravel()
    up = image[MIDDLE_MCP] - image[WRIST]
    scale = np.linalg.norm(up)
    if scale < 1e-6:
        raise DegenerateHand("degenerate image palm")
    c, s = up / scale
    r2 = np.array([[c, s], [-s, c]])    # rotate so wrist -> middle knuckle is +x
    w2 = ((image - image[WRIST]) @ r2.T / scale).ravel()
    angles = []
    for ch in _CHAINS:
        pts = [world[WRIST]] + [world[i] for i in ch]
        for a, b, cc in zip(pts, pts[1:], pts[2:]):
            v1, v2 = b - a, cc - b
            cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
            angles.append(np.arccos(np.clip(cosang, -1.0, 1.0)))
    gap = lambda i, j: np.linalg.norm(image[i] - image[j]) / scale
    extra = [gap(THUMB_TIP, INDEX_TIP), gap(THUMB_TIP, MIDDLE_TIP), gap(INDEX_TIP, MIDDLE_TIP)]
    return np.concatenate([w2, w3, angles, extra])


def summarise(win: np.ndarray) -> np.ndarray:
    return np.concatenate([win.mean(0), win.min(0), win.max(0), win[-1] - win[0]])


def resample(ts: np.ndarray, X: np.ndarray, t_end: float) -> np.ndarray | None:
    """Linearly resample frames covering [t_end - WINDOW_S, t_end] to WINDOW_POINTS rows."""
    m = (ts >= t_end - WINDOW_S - MAX_GAP_S) & (ts <= t_end)
    tt, xx = ts[m], X[m]
    if len(tt) < 3 or tt[0] > t_end - WINDOW_S + 1e-6 or np.max(np.diff(tt)) > MAX_GAP_S:
        return None
    grid = np.linspace(t_end - WINDOW_S, t_end, WINDOW_POINTS)
    # Vectorised linear interpolation of all columns at once (per-column np.interp cost ~8 ms/frame).
    hi = np.clip(np.searchsorted(tt, grid), 1, len(tt) - 1)
    lo = hi - 1
    w = ((grid - tt[lo]) / np.maximum(tt[hi] - tt[lo], 1e-9))[:, None]
    return xx[lo] * (1 - w) + xx[hi] * w


def windows(ts: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Summaries for every frame that ends a complete window -> (end_times, features)."""
    out_t, out_f = [], []
    for i, t_end in enumerate(ts):
        win = resample(ts[:i + 1], X[:i + 1], t_end)
        if win is not None:
            out_t.append(t_end)
            out_f.append(summarise(win))
    return np.array(out_t), np.array(out_f)


class WindowBuffer:
    """Live counterpart of `windows`: push frames, get the current window summary."""

    def __init__(self) -> None:
        self._t: deque[float] = deque()
        self._x: deque[np.ndarray] = deque()

    def reset(self) -> None:
        self._t.clear()
        self._x.clear()

    def push(self, frame: HandFrame) -> np.ndarray | None:
        try:
            v = frame_vector(frame.world, frame.image, frame.handedness)
        except DegenerateHand:
            return None
        self._t.append(frame.t)
        self._x.append(v)
        while self._t and self._t[0] < frame.t - WINDOW_S - MAX_GAP_S - 0.05:
            self._t.popleft()
            self._x.popleft()
        win = resample(np.array(self._t), np.array(self._x), frame.t)
        return None if win is None else summarise(win)
