"""Personal tap detector: per-frame class probabilities from a trained model, turned into click
events with per-button thresholds and a refractory period (one tap can't double-click)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .hand import HandFrame
from .pose_features import FEATURE_VERSION, WindowBuffer

CLASSES = ("none", "left", "right")


class TapModelError(RuntimeError):
    pass


class SoftVote:
    """Average of member models' class probabilities. Members disagree usefully: boosted trees
    were best at left taps, the linear/MLP models at right taps (tools/model_bench.py)."""

    def __init__(self, members) -> None:
        self.members = members

    def fit(self, X, y) -> "SoftVote":
        for m in self.members:
            m.fit(X, y)
        return self

    def predict_proba(self, X):
        return sum(m.predict_proba(X) for m in self.members) / len(self.members)


@dataclass(frozen=True)
class TapModelInfo:
    feature_version: int
    window_frames: int
    trained_on: list[str]
    n_left: int
    n_right: int


def load_bundle(path: Path):
    import joblib
    if not path.is_file():
        raise TapModelError(f"tap model not found at {path}; train it with `python -m handctl.train_taps`")
    bundle = joblib.load(path)
    info: TapModelInfo = bundle["info"]
    if info.feature_version != FEATURE_VERSION:
        raise TapModelError(f"tap model built for feature v{info.feature_version}, code is v{FEATURE_VERSION}; retrain")
    return bundle["model"], info


class TapDetector:
    def __init__(self, model, info: TapModelInfo, left_threshold: float, right_threshold: float,
                 refractory_s: float) -> None:
        self.model, self.info = model, info
        self.thresholds = {"left": left_threshold, "right": right_threshold}
        self.refractory_s = refractory_s
        self._buf = WindowBuffer()
        self._last_fire = -1e9
        self._armed = {"left": True, "right": True}
        self.last_probs: dict[str, float] = {"left": 0.0, "right": 0.0}

    def reset(self) -> None:
        self._buf.reset()
        self._armed = {"left": True, "right": True}
        self.last_probs = {"left": 0.0, "right": 0.0}

    def update(self, frame: HandFrame | None) -> str | None:
        """Feed one frame; returns 'left'/'right' when a tap is recognised."""
        if frame is None:
            self.reset()
            return None
        x = self._buf.push(frame)
        if x is None:
            return None
        p = self.model.predict_proba(x[None, :])[0]
        probs = {"left": float(p[1]), "right": float(p[2])}
        self.last_probs = probs
        fired = None
        for button in ("left", "right"):
            over = probs[button] >= self.thresholds[button]
            # Fire on the rising edge only, then require the probability to drop before re-arming.
            if over and self._armed[button] and frame.t - self._last_fire >= self.refractory_s:
                if fired is None or probs[button] > probs[fired]:
                    fired = button
            if not over:
                self._armed[button] = True
        if fired:
            self._armed[fired] = False
            self._last_fire = frame.t
        return fired


def from_config(cfg) -> "TapDetector | None":
    """Detector for [taps] mode = model, else None."""
    from . import paths
    if cfg.taps.mode != "model":
        return None
    # One prediction per camera frame: thread fan-out costs more than it saves (42 -> 26 ms).
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    model, info = load_bundle(paths.resolve(cfg.taps.model_path))
    return TapDetector(model, info, cfg.taps.left_threshold, cfg.taps.right_threshold, cfg.taps.refractory_s)
