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

FEATURE_VERSION = 1
WINDOW_FRAMES = 10                      # ~0.4 s at 25 fps
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


def windows(X: np.ndarray, w: int = WINDOW_FRAMES) -> np.ndarray:
    """Row i summarises frames i .. i+w-1 (aligned to the window's last frame)."""
    return np.array([summarise(X[i - w + 1:i + 1]) for i in range(w - 1, len(X))])


class WindowBuffer:
    """Live counterpart of `windows`: push frames, get the current window summary."""

    def __init__(self, w: int = WINDOW_FRAMES) -> None:
        self._buf: deque[np.ndarray] = deque(maxlen=w)

    def reset(self) -> None:
        self._buf.clear()

    def push(self, frame: HandFrame) -> np.ndarray | None:
        try:
            self._buf.append(frame_vector(frame.world, frame.image, frame.handedness))
        except DegenerateHand:
            return None
        if len(self._buf) < self._buf.maxlen:
            return None
        return summarise(np.array(self._buf))
