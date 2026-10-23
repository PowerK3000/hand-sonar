"""Per-frame model fitting: find the pose of Kayden's skeleton that best explains the frame.

Cost terms (all scaled to "1 = one standard deviation of expected error"):
- 2D reprojection of all 21 joints vs MediaPipe's 2D keypoints (the reliable signal), robust
  loss so a single wrong keypoint (an occluded thumb tip guessed badly) can't drag the fit.
- Weak 3D prior toward MediaPipe's hand-centred world keypoints: breaks the toward/away
  ambiguity of a bone seen from one camera.
- Temporal prior toward the previous frame's pose: smoothing, and continuity through occlusion.
- Mild anatomical coupling (DIP flexion ~ 2/3 PIP flexion).
- Hard joint limits as solver bounds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from . import skeleton as sk
from .skeleton import HandShape


@dataclass(frozen=True)
class Camera:
    width: int = 1280
    height: int = 720
    # Typical webcam horizontal FOV ~70 deg -> f ~ 914 px at 1280 wide. Exact intrinsics barely
    # matter for a hand-sized object at desk distance; consistency matters.
    focal_px: float = 914.0

    def project(self, P: np.ndarray) -> np.ndarray:
        """(..., 3) camera points -> (..., 2) pixels."""
        z = np.maximum(P[..., 2:3], 1e-3)
        return self.focal_px * P[..., :2] / z + np.array([self.width / 2, self.height / 2])


@dataclass(frozen=True)
class FitWeights:
    sigma_px: float = 4.0            # 2D keypoint noise
    sigma_mp3d_m: float = 0.02       # MediaPipe 3D is a guess: weak
    sigma_angle_step: float = 0.25   # rad per frame
    sigma_rot_step: float = 0.15
    sigma_trans_step_m: float = 0.03
    sigma_coupling: float = 0.35
    robust_scale: float = 2.0        # soft_l1 kicks in beyond ~2 sigma


@dataclass(frozen=True)
class FitResult:
    pose: np.ndarray                 # (27,)
    joints: np.ndarray               # (21, 3) camera coordinates
    local: np.ndarray                # (21, 3) anatomical local frame
    rms_px: float
    nfev: int


def _coupling(pose: np.ndarray) -> np.ndarray:
    """(B, 27) -> (B, 4) DIP - 2/3 PIP per finger."""
    out = []
    for sl in sk.FINGER_SLICES.values():
        f = pose[:, sl]
        out.append(f[:, 3] - 0.67 * f[:, 2])
    return np.stack(out, axis=1)


class PoseFitter:
    def __init__(self, shape: HandShape, camera: Camera = Camera(), weights: FitWeights = FitWeights(),
                 max_nfev: int = 8) -> None:
        self.shape, self.camera, self.w, self.max_nfev = shape, camera, weights, max_nfev
        self.lo, self.hi = sk.bounds()
        self._prev: np.ndarray | None = None
        step = np.full(sk.N_PARAMS, weights.sigma_angle_step)
        step[sk.SL_ROT] = weights.sigma_rot_step
        step[sk.SL_TRANS] = weights.sigma_trans_step_m
        self._step_sigma = step

    def reset(self) -> None:
        self._prev = None

    # -- residuals (batched) ------------------------------------------------------------
    def _residuals(self, poses: np.ndarray, obs2d: np.ndarray, mp3d: np.ndarray,
                   prev: np.ndarray | None) -> np.ndarray:
        w = self.w
        J = sk.camera_joints(self.shape, poses)                       # (B,21,3)
        r2 = ((self.camera.project(J) - obs2d) / w.sigma_px).reshape(len(poses), -1)
        Jc = J - J.mean(axis=1, keepdims=True)
        r3 = ((Jc - mp3d) / w.sigma_mp3d_m).reshape(len(poses), -1)
        parts = [r2, r3, _coupling(poses) / w.sigma_coupling]
        if prev is not None:
            parts.append((poses - prev) / self._step_sigma)
        return np.concatenate(parts, axis=1)

    def _init_pose(self, obs2d: np.ndarray, mp3d: np.ndarray) -> np.ndarray:
        """Rough global pose from scratch: orientation from MediaPipe's 3D palm, distance from
        apparent palm size, position from the 2D wrist."""
        from ..features import palm_frame
        p = sk.rest_pose()
        _, rot = palm_frame(mp3d, "right")
        R = rot.T                                                      # local -> camera
        angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        if angle > 1e-6:
            axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(angle))
            p[sk.SL_ROT] = axis * angle
        palm_px = np.linalg.norm(obs2d[9] - obs2d[0])
        palm_m = np.linalg.norm(mp3d[9] - mp3d[0])
        z = self.camera.focal_px * palm_m / max(palm_px, 1.0)
        c = np.array([self.camera.width / 2, self.camera.height / 2])
        xy = (obs2d[0] - c) * z / self.camera.focal_px
        p[sk.SL_TRANS] = [xy[0], xy[1], np.clip(z, 0.15, 2.5)]
        return np.clip(p, self.lo + 1e-6, self.hi - 1e-6)

    def fit(self, obs2d: np.ndarray, mp3d: np.ndarray, handedness: str = "right") -> FitResult:
        """obs2d: (21,2) pixels. mp3d: (21,3) MediaPipe world (hand-centred, camera-aligned)."""
        if handedness == "left":
            # The model is a right hand; mirror the observation horizontally (and MediaPipe 3D x)
            # so a left hand fits the same skeleton, then mirror the result back.
            obs2d = obs2d.copy(); obs2d[:, 0] = self.camera.width - obs2d[:, 0]
            mp3d = mp3d * np.array([-1.0, 1.0, 1.0])
        mp3d = mp3d - mp3d.mean(axis=0)
        prev = self._prev
        x0 = self._init_pose(obs2d, mp3d) if prev is None else np.clip(prev, self.lo + 1e-9, self.hi - 1e-9)
        eps = 1e-4

        def fun(x):
            return self._residuals(x[None], obs2d, mp3d, prev)[0]

        def jac(x):
            batch = np.vstack([x[None], x[None] + eps * np.eye(sk.N_PARAMS)])
            R = self._residuals(batch, obs2d, mp3d, prev)
            return ((R[1:] - R[0]) / eps).T

        res = least_squares(fun, x0, jac=jac, bounds=(self.lo, self.hi), loss="soft_l1",
                            f_scale=self.w.robust_scale, max_nfev=self.max_nfev if prev is not None else 40,
                            method="trf", x_scale="jac")
        pose = res.x
        self._prev = pose.copy()
        J = sk.camera_joints(self.shape, pose[None])[0]
        L = sk.local_joints(self.shape, pose[None])[0]
        if handedness == "left":
            J = J.copy(); J[:, 0] = -J[:, 0]
        rms = float(np.sqrt(np.mean(np.sum((self.camera.project(sk.camera_joints(self.shape, pose[None])[0]) - obs2d) ** 2, axis=1))))
        return FitResult(pose=pose, joints=J, local=L, rms_px=rms, nfev=int(res.nfev))
