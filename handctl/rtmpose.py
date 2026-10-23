"""RTMPose-m hand keypoints (OpenMMLab, Apache-2.0) on the GPU via ONNX Runtime + DirectML.

Model: rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256 (download.openmmlab.com, SHA-256 of the
SDK zip 45c9e1aa...a56d). Same 21-keypoint order as MediaPipe. Input: a 256x256 crop around the
hand (box padded 1.25x), ImageNet mean/std, RGB. Output: SimCC x/y distributions (512 bins = 2 bins
per input pixel); argmax gives the keypoint, its peak value the confidence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

INPUT = 256
SIMCC_SPLIT = 2.0
PADDING = 1.25
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


class RTMPoseError(RuntimeError):
    pass


class RTMPoseHand:
    def __init__(self, model_path: Path, prefer_gpu: bool = True) -> None:
        if not model_path.is_file():
            raise RTMPoseError(f"RTMPose model not found at {model_path}")
        import onnxruntime as ort
        providers = (["DmlExecutionProvider"] if prefer_gpu else []) + ["CPUExecutionProvider"]
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)
        self.provider = self.session.get_providers()[0]
        self._input = self.session.get_inputs()[0].name

    @staticmethod
    def box_from_keypoints(kp: np.ndarray) -> tuple[np.ndarray, float]:
        """Square crop (centre, side) around keypoints, padded like the model was trained."""
        lo, hi = kp.min(axis=0), kp.max(axis=0)
        return (lo + hi) / 2, float(max(hi - lo) * PADDING)

    def infer(self, bgr: np.ndarray, center: np.ndarray, side: float) -> tuple[np.ndarray, np.ndarray]:
        """Keypoints (21,2) in full-image pixels and per-keypoint confidence (21,)."""
        import cv2
        if side < 8:
            raise RTMPoseError("hand box too small")
        s = INPUT / side
        # Affine: image -> crop (scale about the centre, then shift to the crop centre).
        M = np.array([[s, 0, INPUT / 2 - s * center[0]], [0, s, INPUT / 2 - s * center[1]]], dtype=np.float32)
        crop = cv2.warpAffine(bgr, M, (INPUT, INPUT), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        rgb = crop[:, :, ::-1].astype(np.float32)
        x = ((rgb - MEAN) / STD).transpose(2, 0, 1)[None]
        sx, sy = self.session.run(None, {self._input: np.ascontiguousarray(x)})
        ix, iy = sx[0].argmax(axis=1), sy[0].argmax(axis=1)
        conf = np.minimum(sx[0].max(axis=1), sy[0].max(axis=1))
        kp_crop = np.stack([ix, iy], axis=1).astype(np.float32) / SIMCC_SPLIT
        kp = (kp_crop - np.array([INPUT / 2, INPUT / 2])) / s + center
        return kp, conf
