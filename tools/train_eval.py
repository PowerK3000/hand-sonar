"""Train/evaluate the personal motion model on a training session.

Time-ordered split per step (first part train, last part test: no neighbouring-frame leakage),
event-level scoring, and a head-to-head of the two candidate click gesture sets.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from handctl.hand import INDEX_TIP, MIDDLE_TIP
from tools.seq_experiment import W, events, frame_vector, load_step, windows

GESTURE_SETS = {
    "thumb taps": {"click": 1, "right_click": 2},
    "finger presses": {"press_index": 1, "press_middle": 2},
}
NEGATIVE = {"rest", "typing", "cursor", "face"}
TRAIN_FRAC = 0.6
# Column offsets inside frame_vector: 42 2D values first, index tip = landmark 8, middle = 12.
TIP2D = {INDEX_TIP: slice(16, 18), MIDDLE_TIP: slice(24, 26)}


def event_times(label, t, X, beats):
    """When did the gesture happen after each beat? Thumb taps: closest thumb-finger 2D approach.
    Presses: largest fingertip excursion from where it rested just before the beat."""
    out = []
    for b in beats:
        m = (t >= b) & (t < b + 0.9)
        if not m.any():
            continue
        idx = np.flatnonzero(m)
        if label in ("click", "right_click", "drag"):
            col = -3 if label in ("click", "drag") else -2
            out.append(t[idx[np.argmin(X[idx, col])]])
        else:
            tip = TIP2D[INDEX_TIP if label == "press_index" else MIDDLE_TIP]
            pre = (t >= b - 0.4) & (t < b)
            if not pre.any():
                continue
            base = np.median(X[pre][:, tip], axis=0)
            dist = np.linalg.norm(X[idx][:, tip] - base, axis=1)
            out.append(t[idx[np.argmax(dist)]])
    return out


def prepare(paths):
    steps = {}
    for p in paths:
        label, t, X, beats, _ = load_step(p)
        if len(X) < 3 * W:
            continue
        steps[label] = (t, X, beats, event_times(label, t, X, beats) if len(beats) else [])
    return steps


def split_xy(steps, classes, part):
    Fs, ys, metas = [], [], []
    for label, (t, X, beats, evts) in steps.items():
        F = windows(X)
        tw = t[W - 1:]
        cut = tw[0] + TRAIN_FRAC * (tw[-1] - tw[0])
        sel = tw < cut if part == "train" else tw >= cut + 0.5   # 0.5 s guard band
        cls = classes.get(label, 0)
        y = np.zeros(len(F), int)
        near = np.zeros(len(F), bool)
        for e in evts:
            if cls:
                y[(tw >= e - 0.04) & (tw <= e + 0.12)] = cls
            near |= np.abs(tw - e) < 0.6
        keep = sel & ((y > 0) | ~near) if part == "train" else sel
        if not keep.any():
            continue
        Fs.append(F[keep]); ys.append(y[keep])
        metas.append((label, tw[keep], [e for e in evts if (e >= cut + 0.5 if part == "test" else e < cut)], cls))
    return Fs, ys, metas


def evaluate(steps, set_name, classes):
    Ftr, ytr, _ = split_xy(steps, classes, "train")
    Fte, _, meta = split_xy(steps, classes, "test")
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, class_weight="balanced",
                                         random_state=0).fit(np.vstack(Ftr), np.concatenate(ytr))
    print(f"\n== {set_name}: trained on {sum(len(y) for y in ytr)} windows "
          f"({sum((np.concatenate(ytr) == 1))} left / {sum((np.concatenate(ytr) == 2))} right frames)")
    for thr in (0.5, 0.65, 0.8, 0.9):
        hits = {1: 0, 2: 0}; total = {1: 0, 2: 0}; wrong = 0; false_g = 0; false_n = 0; neg_s = 0.0
        false_by = {}
        for F, (label, tw, evts, cls) in zip(Fte, meta):
            P = clf.predict_proba(F)
            for c in (1, 2):
                for e in events(tw, P[:, c], thr):
                    near = any(abs(e - x) < 0.35 for x in evts)
                    if cls == c and near:
                        hits[c] += 1
                    elif cls and near:
                        wrong += 1
                    else:
                        false_by[label] = false_by.get(label, 0) + 1
                        if label in NEGATIVE:
                            false_n += 1
                        else:
                            false_g += 1
            if cls:
                total[cls] += len(evts)
            if label in NEGATIVE:
                neg_s += tw[-1] - tw[0]
        print(f"thr {thr:.2f}: left {hits[1]}/{total[1]}  right {hits[2]}/{total[2]}  wrong-button {wrong}  "
              f"false in gesture steps {false_g}  false in non-gesture steps {false_n} "
              f"({false_n / max(neg_s, 1) * 60:.1f}/min)  by step {false_by}")
    return clf


if __name__ == "__main__":
    base = Path(sys.argv[1] if len(sys.argv) > 1 else "data/recordings/t1")
    steps = prepare(sorted(base.glob("*.npz")))
    for name, classes in GESTURE_SETS.items():
        evaluate(steps, name, classes)
