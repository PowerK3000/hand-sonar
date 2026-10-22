"""Synthetic hands for tests: anatomically proportioned, posable, viewable from any angle.

Built in the anatomical frame (x: thumb side, y: toward fingers, z: back of hand), metres,
then rotated/translated into "camera" space and projected through a pinhole camera.
"""

from __future__ import annotations

import numpy as np

from handctl.hand import HandFrame

# (x, y) of each MCP knuckle and (proximal, middle, distal) bone lengths.
_FINGERS = {
    "index": ((0.025, 0.088), (0.040, 0.024, 0.020), 5),
    "middle": ((0.000, 0.090), (0.045, 0.028, 0.021), 9),
    "ring": ((-0.021, 0.085), (0.042, 0.026, 0.020), 13),
    "pinky": ((-0.040, 0.078), (0.032, 0.019, 0.018), 17),
}
_FLEX_MAX = np.radians([70, 95, 65])  # MCP, PIP, DIP flexion at full curl
_THUMB_CMC = np.array([0.022, 0.025, -0.012])
_THUMB_BONES = (0.040, 0.032, 0.028)

FOCAL_PX, CX, CY = 600.0, 320.0, 240.0


_SPREAD_WEIGHT = {"index": 1.0, "middle": 0.0, "ring": -0.7, "pinky": -1.4}


def _finger(mcp_xy, bones, curl, lateral):
    p = np.array([mcp_xy[0], mcp_xy[1], 0.0])
    pts, phi = [p], 0.0
    for length, flex in zip(bones, _FLEX_MAX * curl):
        phi += flex
        # Curls toward the palm (-z); `lateral` fans the finger toward the thumb (+x) or away.
        p = p + length * np.array([np.sin(lateral) * np.cos(phi), np.cos(lateral) * np.cos(phi), -np.sin(phi)])
        pts.append(p)
    return pts


def anatomical_hand(curl: dict[str, float] | None = None, thumb_to: str | None = None,
                    gap: float = 0.0, spread_deg: float = 0.0) -> np.ndarray:
    """(21, 3) right hand in the anatomical frame.

    curl: per finger 0 (straight) .. 1 (fist). spread_deg: index-to-pinky fan angle. thumb_to: 'index'/'middle' puts the thumb tip
    `gap` metres from that fingertip (0 = touching); None = relaxed thumb.
    """
    curl = {"index": 0.1, "middle": 0.1, "ring": 0.1, "pinky": 0.1, **(curl or {})}
    pts = np.zeros((21, 3))
    for name, (mcp_xy, bones, base) in _FINGERS.items():
        pts[base:base + 4] = _finger(mcp_xy, bones, curl[name], np.radians(spread_deg) * _SPREAD_WEIGHT[name] / 2.4)
    pts[1] = _THUMB_CMC
    if thumb_to is None:
        d = np.array([0.75, 0.6, -0.25])
        d /= np.linalg.norm(d)
        tip = _THUMB_CMC + d * sum(_THUMB_BONES)
    else:
        target = pts[{"index": 8, "middle": 12}[thumb_to]]
        away = target - _THUMB_CMC
        tip = target - away / np.linalg.norm(away) * gap
    # Thumb joints along a slightly bowed path from CMC to tip.
    bow = np.array([0.01, 0.0, -0.01])
    for i, f in zip((2, 3, 4), (0.4, 0.72, 1.0)):
        pts[i] = _THUMB_CMC + f * (tip - _THUMB_CMC) + bow * np.sin(np.pi * f)
    return pts


def rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy, sy, cp, sp, cr, sr = (np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch),
                              np.cos(roll), np.sin(roll))
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


def frame(t: float, anatomical: np.ndarray, rot: np.ndarray | None = None,
          position: tuple[float, float, float] = (0.0, 0.0, 0.5), scale: float = 1.0,
          handedness: str = "right", score: float = 0.95) -> HandFrame:
    pts = anatomical * scale
    if handedness == "left":
        pts = pts * np.array([-1.0, 1.0, 1.0])  # mirror image of the right hand
    rot = np.eye(3) if rot is None else rot
    cam = pts @ rot.T + np.asarray(position)
    if np.any(cam[:, 2] <= 0.05):
        raise ValueError("hand behind or touching the camera")
    image = np.stack([FOCAL_PX * cam[:, 0] / cam[:, 2] + CX, FOCAL_PX * cam[:, 1] / cam[:, 2] + CY], axis=1)
    # MediaPipe world landmarks are hand-centred; mimic that so tests can't lean on absolute position.
    world = cam - cam.mean(axis=0)
    return HandFrame(t=t, world=world, image=image, handedness=handedness, score=score)
