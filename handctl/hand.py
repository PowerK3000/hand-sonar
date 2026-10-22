"""Tracker-independent hand frame: the only thing downstream code sees from the camera."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# MediaPipe 21-point hand topology.
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGERS = {
    "index": (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
}
N_LANDMARKS = 21


@dataclass(frozen=True)
class HandFrame:
    """One tracked hand at time `t` (seconds, monotonic).

    world: (21, 3) metric 3D keypoints (any origin/orientation; features are invariant to both).
    image: (21, 2) keypoints in camera pixels (used only for whole-hand cursor motion).
    """
    t: float
    world: np.ndarray
    image: np.ndarray
    handedness: str
    score: float

    def __post_init__(self) -> None:
        if self.world.shape != (N_LANDMARKS, 3):
            raise ValueError(f"world must be (21, 3), got {self.world.shape}")
        if self.image.shape != (N_LANDMARKS, 2):
            raise ValueError(f"image must be (21, 2), got {self.image.shape}")
        if self.handedness not in ("left", "right"):
            raise ValueError(f"handedness must be left/right, got {self.handedness!r}")
        if not np.all(np.isfinite(self.world)) or not np.all(np.isfinite(self.image)):
            raise ValueError("non-finite landmark coordinates")
