"""Train the personal tap model from training-protocol recordings.

    python -m handctl.train_taps data/recordings/t1 [more session dirs...] [--out models/tap_model.joblib]

Labels: tap time = beat + the session's median reaction latency per step (beat-locked labels
proved far less noisy than "closest thumb-finger approach", which wanders when the thumb is
occluded). Held-out evaluation lives in tools/train_eval.py; this fits the final model on
everything given.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import paths
from .controller import HandednessLatch
from .features import DegenerateHand
from .pose_features import FEATURE_VERSION, WINDOW_FRAMES, frame_vector, windows
from .tap_model import TapModelInfo

TAP_CLASSES = {"click": 1, "right_click": 2}
# Drag steps are excluded: a drag starts with exactly the motion of a tap.
NEGATIVE_STEPS = {"rest", "typing", "cursor", "face", "mouse", "press_index", "press_middle"}
POS_BEFORE_S, POS_AFTER_S = 0.08, 0.12     # label window around the tap moment
NEAR_S = 0.6                                # unlabeled frames this close to a tap are dropped


def load_step(path: Path):
    d = np.load(path, allow_pickle=False)
    latch = HandednessLatch()
    t, X = [], []
    for i, ti in enumerate(d["t"]):
        if not d["present"][i]:
            latch.reset()
            continue
        try:
            X.append(frame_vector(d["world"][i], d["image"][i], latch(str(d["handedness"][i]))))
            t.append(float(ti))
        except DegenerateHand:
            pass
    return str(d["label"]), np.array(t), np.array(X), np.array(d["beats"])


def reaction_latency(label: str, t: np.ndarray, X: np.ndarray, beats: np.ndarray) -> float:
    """Median beat -> closest-approach delay: robust to the per-tap noise of the 2D minimum."""
    col = -3 if label == "click" else -2
    lat = []
    for b in beats:
        m = (t >= b) & (t < b + 0.9)
        if m.any():
            idx = np.flatnonzero(m)
            lat.append(t[idx[np.argmin(X[idx, col])]] - b)
    if not lat:
        raise ValueError(f"no taps found in step {label}")
    return float(np.median(lat))


def build(session_dirs: list[Path]):
    """Returns per-step tuples (label, tw, F, y, taps)."""
    out = []
    for d in session_dirs:
        for p in sorted(d.glob("*.npz")):
            if p.name.endswith(".tmp.npz"):
                continue
            label, t, X, beats = load_step(p)
            if label not in TAP_CLASSES and label not in NEGATIVE_STEPS:
                continue
            if len(X) < 3 * WINDOW_FRAMES:
                continue
            F = windows(X)
            tw = t[WINDOW_FRAMES - 1:]
            y = np.zeros(len(F), int)
            taps: list[float] = []
            if label in TAP_CLASSES:
                lat = reaction_latency(label, t, X, beats)
                taps = [b + lat for b in beats]
                near = np.zeros(len(F), bool)
                for tap in taps:
                    y[(tw >= tap - POS_BEFORE_S) & (tw <= tap + POS_AFTER_S)] = TAP_CLASSES[label]
                    near |= np.abs(tw - tap) < NEAR_S
                keep = (y > 0) | ~near
                F, y, tw = F[keep], y[keep], tw[keep]
            out.append((label, tw, F, y, taps))
    return out


def make_model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, class_weight="balanced",
                                          random_state=0)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sessions", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=paths.ROOT / "models" / "tap_model.joblib")
    args = p.parse_args(argv)
    missing = [s for s in args.sessions if not s.is_dir()]
    if missing:
        p.error(f"not a directory: {missing}")
    steps = build(args.sessions)
    if not any(lbl in TAP_CLASSES for lbl, *_ in steps):
        print("error: no click/right_click steps found (record with --protocol train)", file=sys.stderr)
        return 2
    F = np.vstack([s[2] for s in steps])
    y = np.concatenate([s[3] for s in steps])
    n_left, n_right = int(sum(len(s[4]) for s in steps if s[0] == "click")), \
        int(sum(len(s[4]) for s in steps if s[0] == "right_click"))
    import joblib
    model = make_model().fit(F, y)
    info = TapModelInfo(FEATURE_VERSION, WINDOW_FRAMES, [str(s) for s in args.sessions], n_left, n_right)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    joblib.dump({"model": model, "info": info}, tmp)
    tmp.replace(args.out)
    print(f"trained on {len(y)} windows ({n_left} left taps, {n_right} right taps) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
