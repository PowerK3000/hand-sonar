"""Webcam capture + MediaPipe hand landmarks -> HandFrame.

Capture runs on its own thread and keeps only the newest frame: when processing falls behind,
stale frames are dropped instead of queued, so latency never grows.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np

from .hand import HandFrame

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")


class TrackerError(RuntimeError):
    pass


class Camera:
    def __init__(self, index: int, width: int = 1280, height: int = 720, fps: int = 30) -> None:
        # Skip MSMF's hardware colour-transform probe: it adds seconds to camera open time.
        os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
        import cv2  # imported lazily so tests and tooling don't need OpenCV loaded
        self._cv2 = cv2
        # Media Foundation, not DirectShow: on Kayden's Sonix 0C45:64AB webcam DirectShow falls
        # back to uncompressed YUY2 (8 fps at 720p); MSMF negotiates MJPG and delivers ~29 fps.
        self._cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
        if not self._cap.isOpened():
            raise TrackerError(f"cannot open camera {index} (plugged in? used by another app?)")
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # before the size
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        self._latest: tuple[float, np.ndarray] | None = None
        self._cond = threading.Condition()
        self._running = True
        self.read_failures = 0
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            ok, bgr = self._cap.read()
            t = time.monotonic()
            if not ok:
                self.read_failures += 1
                time.sleep(0.05)
                continue
            with self._cond:
                self._latest = (t, bgr)
                self._cond.notify()

    def latest(self, timeout: float = 1.0) -> tuple[float, np.ndarray]:
        """Newest frame (consumed). Raises TrackerError if the camera stops delivering."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._latest is not None or not self._running, timeout):
                raise TrackerError(f"no camera frame for {timeout:.1f} s ({self.read_failures} read failures)")
            if self._latest is None:
                raise TrackerError("camera closed")
            frame, self._latest = self._latest, None
            return frame

    def close(self) -> None:
        self._running = False
        with self._cond:
            self._cond.notify_all()
        self._thread.join(2)
        self._cap.release()


class HandTracker:
    """Wraps MediaPipe HandLandmarker (VIDEO mode) and picks the configured hand."""

    def __init__(self, model_path: Path, hand: str = "any", num_hands: int = 1,
                 process_width: int | None = None) -> None:
        if not model_path.is_file():
            raise TrackerError(f"hand model not found at {model_path}. Download it from {MODEL_URL}")
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
        self._mp = mp
        opts = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO, num_hands=num_hands,
            min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5)
        self._lm = HandLandmarker.create_from_options(opts)
        self.hand = hand
        self.process_width = process_width
        self._last_ms = -1

    def process(self, t: float, bgr: np.ndarray) -> HandFrame | None:
        # Landmarks come back normalised, so they're reported in full-frame pixels regardless
        # of the downscale used for tracking.
        h, w = bgr.shape[:2]
        if self.process_width and bgr.shape[1] > self.process_width:
            import cv2
            f = self.process_width / bgr.shape[1]
            bgr = cv2.resize(bgr, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        # MediaPipe requires strictly increasing timestamps in VIDEO mode.
        ms = max(int(t * 1000), self._last_ms + 1)
        self._last_ms = ms
        res = self._lm.detect_for_video(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb), ms)
        best: HandFrame | None = None
        for img_lms, world_lms, cats in zip(res.hand_landmarks, res.hand_world_landmarks, res.handedness):
            cat = cats[0]
            # MediaPipe labels handedness as if the image were mirrored (selfie view). We feed the
            # raw, unmirrored camera image, so the label is swapped.
            side = "left" if cat.category_name == "Right" else "right"
            if self.hand != "any" and side != self.hand:
                continue
            frame = HandFrame(
                t=t,
                world=np.array([[p.x, p.y, p.z] for p in world_lms], dtype=float),
                image=np.array([[p.x * w, p.y * h] for p in img_lms], dtype=float),
                handedness=side, score=float(cat.score))
            if best is None or frame.score > best.score:
                best = frame
        return best

    def close(self) -> None:
        self._lm.close()
