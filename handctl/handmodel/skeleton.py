"""Personal kinematic hand model: shape (Kayden's proportions) + pose (joint angles) -> 21 joints.

Anatomical local frame, matching handctl.features.palm_frame: origin at the wrist, x toward the
thumb side, y toward the fingers, z out of the back of the hand. Flexion bends toward the palm
(-z). Forward kinematics is batched over poses, because the fitter's finite-difference Jacobian
evaluates ~30 perturbed poses at once and per-pose Python loops would be the bottleneck.

Pose vector layout (N_PARAMS = 27):
  0:3   global rotation (axis-angle, local -> camera)
  3:6   global translation (camera coordinates, metres; wrist position)
  6:22  index, middle, ring, pinky x (abduction, MCP flex, PIP flex, DIP flex)
  22:27 thumb (azimuth, elevation, roll, MCP flex, IP flex)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FINGER_NAMES = ("index", "middle", "ring", "pinky")
FINGER_ROOTS = (5, 9, 13, 17)
N_PARAMS = 27
SL_ROT, SL_TRANS = slice(0, 3), slice(3, 6)
FINGER_SLICES = {n: slice(6 + 4 * i, 10 + 4 * i) for i, n in enumerate(FINGER_NAMES)}
SL_THUMB = slice(22, 27)

# Joint limits in radians (generous: a limit that is too tight forces wrong fits).
_FINGER_LO = [-0.35, -0.55, 0.0, -0.1]
_FINGER_HI = [0.35, 1.65, 1.95, 1.45]
_THUMB_LO = [-0.4, -0.8, -1.2, -0.4, -0.5]
_THUMB_HI = [1.7, 1.3, 1.2, 1.3, 1.5]


def bounds() -> tuple[np.ndarray, np.ndarray]:
    lo = np.full(N_PARAMS, -np.inf)
    hi = np.full(N_PARAMS, np.inf)
    lo[SL_ROT], hi[SL_ROT] = -2 * np.pi, 2 * np.pi
    lo[SL_TRANS], hi[SL_TRANS] = [-2.0, -2.0, 0.08], [2.0, 2.0, 3.0]   # in front of the camera
    for sl in FINGER_SLICES.values():
        lo[sl], hi[sl] = _FINGER_LO, _FINGER_HI
    lo[SL_THUMB], hi[SL_THUMB] = _THUMB_LO, _THUMB_HI
    return lo, hi


@dataclass(frozen=True)
class HandShape:
    """Kayden's proportions. Lengths in metres; absolute scale is weakly observable from a single
    camera, the ratios are what make the model personal."""
    palm: np.ndarray            # (5, 3) thumb CMC, index/middle/ring/pinky MCP, wrist at origin
    finger_bones: np.ndarray    # (4, 3) proximal, middle, distal length per finger
    finger_splay: np.ndarray    # (4,) rest direction of each finger in the palm plane (rad, + = toward thumb)
    thumb_bones: np.ndarray     # (3,) metacarpal (CMC->MCP), proximal, distal

    def __post_init__(self) -> None:
        for k in ("palm", "finger_bones", "finger_splay", "thumb_bones"):
            object.__setattr__(self, k, np.ascontiguousarray(getattr(self, k), dtype=float))
        if self.palm.shape != (5, 3) or self.finger_bones.shape != (4, 3) or self.finger_splay.shape != (4,) \
                or self.thumb_bones.shape != (3,):
            raise ValueError("bad HandShape array shapes")
        if np.any(self.finger_bones <= 0) or np.any(self.thumb_bones <= 0):
            raise ValueError("bone lengths must be positive")

    def to_dict(self) -> dict:
        return {k: getattr(self, k).tolist() for k in ("palm", "finger_bones", "finger_splay", "thumb_bones")}

    @classmethod
    def from_dict(cls, d: dict) -> "HandShape":
        return cls(**{k: np.asarray(d[k], dtype=float) for k in ("palm", "finger_bones", "finger_splay", "thumb_bones")})


def default_shape() -> HandShape:
    """Average adult hand; the starting point that calibration personalises."""
    return HandShape(
        palm=np.array([[0.022, 0.025, -0.012], [0.025, 0.088, 0.0], [0.0, 0.090, 0.0],
                       [-0.021, 0.085, 0.0], [-0.040, 0.078, 0.0]]),
        finger_bones=np.array([[0.040, 0.024, 0.020], [0.045, 0.028, 0.021],
                               [0.042, 0.026, 0.020], [0.032, 0.019, 0.018]]),
        finger_splay=np.array([0.05, 0.0, -0.05, -0.12]),
        thumb_bones=np.array([0.040, 0.032, 0.028]),
    )


def rodrigues(r: np.ndarray) -> np.ndarray:
    """(B, 3) axis-angle -> (B, 3, 3) rotation matrices."""
    theta = np.linalg.norm(r, axis=1)
    k = r / np.maximum(theta, 1e-12)[:, None]
    K = np.zeros((len(r), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    s, c = np.sin(theta)[:, None, None], np.cos(theta)[:, None, None]
    return np.eye(3)[None] + s * K + (1 - c) * (K @ K)


def local_joints(shape: HandShape, pose: np.ndarray) -> np.ndarray:
    """(B, 27) poses -> (B, 21, 3) joints in the anatomical local frame (compiled fast path)."""
    pose = np.ascontiguousarray(np.atleast_2d(pose), dtype=float)
    if _fast is not None:
        return _fast.local_batch(shape.palm, shape.finger_bones, shape.finger_splay, shape.thumb_bones, pose)
    return local_joints_reference(shape, pose)


def local_joints_reference(shape: HandShape, pose: np.ndarray) -> np.ndarray:
    """NumPy reference implementation (tests check the compiled version against this)."""
    pose = np.atleast_2d(pose)
    B = len(pose)
    J = np.zeros((B, 21, 3))
    J[:, 1] = shape.palm[0]
    for i, (name, root) in enumerate(zip(FINGER_NAMES, FINGER_ROOTS)):
        abd, f1, f2, f3 = pose[:, FINGER_SLICES[name]].T
        lat = shape.finger_splay[i] + abd
        p = np.broadcast_to(shape.palm[i + 1], (B, 3)).copy()
        J[:, root] = p
        phi = np.zeros(B)
        for k, (flex, length) in enumerate(zip((f1, f2, f3), shape.finger_bones[i])):
            phi = phi + flex
            d = np.stack([np.sin(lat) * np.cos(phi), np.cos(lat) * np.cos(phi), -np.sin(phi)], axis=1)
            p = p + length * d
            J[:, root + 1 + k] = p
    # Thumb: metacarpal direction from azimuth (toward +x from +y) and elevation (toward -z,
    # i.e. palmar); MCP and IP flex about an axis perpendicular to the bone, rolled by `roll`
    # (opposition), so flexion sweeps the tip across the palm toward the fingers.
    az, el, roll, fm, fi = pose[:, SL_THUMB].T
    u = np.stack([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), -np.sin(el)], axis=1)
    ref = np.broadcast_to(np.array([0.0, 0.0, 1.0]), (B, 3))
    a0 = np.cross(u, ref)
    a0 /= np.maximum(np.linalg.norm(a0, axis=1, keepdims=True), 1e-9)
    b0 = np.cross(a0, u)
    axis = np.cos(roll)[:, None] * a0 + np.sin(roll)[:, None] * b0      # flexion axis
    p = J[:, 1] + shape.thumb_bones[0] * u
    J[:, 2] = p
    d = u
    for k, (flex, length) in enumerate(zip((fm, fi), shape.thumb_bones[1:])):
        d = _rotate(d, axis, flex)
        p = p + length * d
        J[:, 3 + k] = p
    return J


def _rotate(v: np.ndarray, axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Rodrigues rotation of vectors v (B,3) about unit axes (B,3) by angles (B,)."""
    c, s = np.cos(angle)[:, None], np.sin(angle)[:, None]
    return v * c + np.cross(axis, v) * s + axis * (np.sum(axis * v, axis=1, keepdims=True)) * (1 - c)


def camera_joints(shape: HandShape, pose: np.ndarray) -> np.ndarray:
    """(B, 27) -> (B, 21, 3) joints in camera coordinates."""
    pose = np.ascontiguousarray(np.atleast_2d(pose), dtype=float)
    if _fast is not None:
        return _fast.camera_batch(shape.palm, shape.finger_bones, shape.finger_splay, shape.thumb_bones, pose)
    return camera_joints_reference(shape, pose)


def camera_joints_reference(shape: HandShape, pose: np.ndarray) -> np.ndarray:
    pose = np.atleast_2d(pose)
    R = rodrigues(pose[:, SL_ROT])
    return np.einsum("bij,bkj->bki", R, local_joints_reference(shape, pose)) + pose[:, None, SL_TRANS]


try:  # compiled kernels are an optimisation, never a hard dependency
    from . import fast_fk as _fast
except ImportError:  # pragma: no cover
    _fast = None


def rest_pose(distance_m: float = 0.5) -> np.ndarray:
    p = np.zeros(N_PARAMS)
    p[SL_TRANS] = [0.0, 0.0, distance_m]
    for sl in FINGER_SLICES.values():
        p[sl] = [0.0, 0.2, 0.3, 0.2]
    p[SL_THUMB] = [0.9, 0.3, 0.0, 0.2, 0.2]
    return p
