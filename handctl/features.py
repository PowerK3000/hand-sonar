"""View-invariant hand descriptors.

Everything gesture logic needs is expressed in a hand-local frame and in units of hand
size, so a pinch measures the same whether the camera sees the hand from the front, the
side or above, near or far. Only the whole-hand image position (for the cursor) is
view-dependent, by design.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hand import (FINGERS, INDEX_MCP, INDEX_PIP, INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP, PINKY_MCP, PINKY_PIP, RING_MCP,
                   THUMB_TIP, WRIST, HandFrame)

PALM = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)
_MIN_SIZE_M = 0.02  # anything smaller is a tracking glitch, not a hand


class DegenerateHand(ValueError):
    """Landmarks too collapsed to define a hand frame (tracker glitch)."""


@dataclass(frozen=True)
class HandFeatures:
    t: float
    score: float
    handedness: str
    size_m: float                    # wrist -> middle knuckle, metres
    pinch_index: float               # thumb tip <-> index tip, hand lengths
    pinch_middle: float              # thumb tip <-> middle tip, hand lengths
    straightness: dict[str, float]   # per finger, ~1 straight, ~0.5-0.7 curled
    spread_deg: float                # angle between index and pinky in the palm plane (fingers fanned out)
    index_local: np.ndarray          # index tip in palm frame (x: thumb side, y: forward, z: dorsal), hand lengths
    palm_px: np.ndarray              # palm centre in camera pixels
    hand_px: float                   # hand length in camera pixels at the hand's current pose
    palm_size_px: float = 0.0        # mean distance between palm keypoints in the image: grows as the hand rises toward the camera
    pinch_index_2d: float = 1.0      # thumb tip <-> index tip in the image, in image hand lengths (wrist->middle knuckle)
    pinch_middle_2d: float = 1.0


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise DegenerateHand("zero-length axis")
    return v / n


def palm_frame(world: np.ndarray, handedness: str) -> tuple[np.ndarray, np.ndarray]:
    """Origin and rotation (rows = x, y, z axes) of the anatomical palm frame.

    x points toward the thumb side, y toward the fingers, z out of the back of the hand,
    for both hands, so left- and right-hand gestures produce the same numbers.
    """
    origin = world[list(PALM)].mean(axis=0)
    y = _unit(world[MIDDLE_MCP] - world[WRIST])
    across = world[INDEX_MCP] - world[PINKY_MCP]
    x = _unit(across - np.dot(across, y) * y)
    z = np.cross(x, y)
    # x×y points out of the back of a right hand; a left hand is its mirror image, so flip.
    if handedness == "left":
        z = -z
    return origin, np.stack([x, y, z])


def finger_straightness(world: np.ndarray, chain: tuple[int, int, int, int]) -> float:
    mcp, pip, dip, tip = (world[i] for i in chain)
    bone = np.linalg.norm(pip - mcp) + np.linalg.norm(dip - pip) + np.linalg.norm(tip - dip)
    if bone < 1e-9:
        raise DegenerateHand("zero-length finger")
    return float(np.linalg.norm(tip - mcp) / bone)


def finger_spread_deg(world: np.ndarray, rot: np.ndarray) -> float:
    """Fan-out angle between index and pinky, measured in the palm plane.

    Uses the first bone (knuckle -> middle joint) only: whole-finger vectors of curled fingers
    project to near-zero length in the palm plane and give meaningless angles.
    """
    a = (rot @ (world[INDEX_PIP] - world[INDEX_MCP]))[:2]
    b = (rot @ (world[PINKY_PIP] - world[PINKY_MCP]))[:2]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))))


def compute(frame: HandFrame) -> HandFeatures:
    w = frame.world
    size = float(np.linalg.norm(w[MIDDLE_MCP] - w[WRIST]))
    if size < _MIN_SIZE_M:
        raise DegenerateHand(f"hand size {size:.3f} m below {_MIN_SIZE_M} m")
    origin, rot = palm_frame(w, frame.handedness)
    index_local = rot @ (w[INDEX_TIP] - origin) / size

    # Pixels per metre from all palm-point pairs: averaging over many 3D directions keeps the
    # estimate stable when the hand tilts and one segment is foreshortened.
    img = frame.image
    pairs = [(a, b) for i, a in enumerate(PALM) for b in PALM[i + 1:]]
    d_img = sum(np.linalg.norm(img[a] - img[b]) for a, b in pairs)
    d_world = sum(np.linalg.norm(w[a] - w[b]) for a, b in pairs)
    hand_px = size * d_img / d_world
    palm_size_px = d_img / len(pairs)
    hand_img = float(np.linalg.norm(img[MIDDLE_MCP] - img[WRIST])) or 1.0

    return HandFeatures(
        t=frame.t,
        score=frame.score,
        handedness=frame.handedness,
        size_m=size,
        pinch_index=float(np.linalg.norm(w[THUMB_TIP] - w[INDEX_TIP]) / size),
        pinch_middle=float(np.linalg.norm(w[THUMB_TIP] - w[MIDDLE_TIP]) / size),
        straightness={name: finger_straightness(w, chain) for name, chain in FINGERS.items()},
        spread_deg=finger_spread_deg(w, rot),
        index_local=index_local,
        palm_px=img[list(PALM)].mean(axis=0),
        hand_px=float(hand_px),
        palm_size_px=float(palm_size_px),
        pinch_index_2d=float(np.linalg.norm(img[THUMB_TIP] - img[INDEX_TIP]) / hand_img),
        pinch_middle_2d=float(np.linalg.norm(img[THUMB_TIP] - img[MIDDLE_TIP]) / hand_img),
    )
