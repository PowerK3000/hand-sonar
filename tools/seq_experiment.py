"""Feasibility test: can a personal model recognise taps from whole-hand MOTION over time,
when thumb-finger distance alone can't?

Leave-one-session-out: train on one session, test on the other. Reports event-level hits,
misses and false triggers per minute, so the answer is directly comparable to the
threshold detector (which was net negative).
"""
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from handctl.features import DegenerateHand, palm_frame
from handctl.hand import FINGERS, INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP, THUMB_TIP, WRIST
from handctl.controller import HandednessLatch

W = 10  # frames per window (~0.4 s at 25 fps)
CLASSES = ["none", "left", "right"]
TAP_STEPS = {"click": ("left", INDEX_TIP), "drag": ("left", INDEX_TIP), "right_click": ("right", MIDDLE_TIP)}
NEG_STEPS = {"rest", "typing", "mouse", "cursor", "scroll_up", "scroll_down", "flick", "wake", "face"}


def frame_vector(world, image, side):
    """View-normalised pose: 2D keypoints in a hand-aligned image frame + 3D in the palm frame
    + per-joint flexion angles."""
    size = np.linalg.norm(world[MIDDLE_MCP] - world[WRIST])
    if size < 0.02:
        raise DegenerateHand("tiny")
    origin, rot = palm_frame(world, side)
    w3 = ((world - origin) @ rot.T / size).ravel()
    up = image[MIDDLE_MCP] - image[WRIST]
    scale = np.linalg.norm(up)
    if scale < 1e-6:
        raise DegenerateHand("flat")
    c, s = up / scale
    r2 = np.array([[c, s], [-s, c]])                        # rotate so wrist->middle knuckle is +x
    w2 = ((image - image[WRIST]) @ r2.T / scale).ravel()
    angles = []
    chains = [(1, 2, 3, 4)] + list(FINGERS.values())
    for ch in chains:
        pts = [world[WRIST]] + [world[i] for i in ch]
        for a, b, cc in zip(pts, pts[1:], pts[2:]):
            v1, v2 = b - a, cc - b
            angles.append(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9), -1, 1)))
    d2 = lambda i, j: np.linalg.norm(image[i] - image[j]) / scale
    extra = [d2(THUMB_TIP, INDEX_TIP), d2(THUMB_TIP, MIDDLE_TIP), d2(INDEX_TIP, MIDDLE_TIP)]
    return np.concatenate([w2, w3, angles, extra])


def load_step(path):
    d = np.load(path, allow_pickle=False)
    latch = HandednessLatch()
    t, X = [], []
    for i, ti in enumerate(d["t"]):
        if not d["present"][i]:
            continue
        try:
            X.append(frame_vector(d["world"][i], d["image"][i], latch(str(d["handedness"][i]))))
            t.append(float(ti))
        except DegenerateHand:
            pass
    return str(d["label"]), np.array(t), np.array(X), np.array(d["beats"]), d


def windows(X):
    """Per frame i (i >= W-1): summary of frames i-W+1..i -> mean, min, max, last-first."""
    out = []
    for i in range(W - 1, len(X)):
        win = X[i - W + 1:i + 1]
        out.append(np.concatenate([win.mean(0), win.min(0), win.max(0), win[-1] - win[0]]))
    return np.array(out)


def tap_centres(label, t, X, beats):
    """Tap time per beat: frame of minimum 2D thumb-finger distance in [beat, beat+0.9]."""
    cls, tip = TAP_STEPS[label]
    col = -3 if tip == INDEX_TIP else -2
    centres = []
    for b in beats:
        m = (t >= b) & (t < b + 0.9)
        if m.any():
            idx = np.flatnonzero(m)
            centres.append(t[idx[np.argmin(X[idx, col])]])
    return cls, centres


def build(paths):
    feats, labels, meta = [], [], []
    for p in paths:
        label, t, X, beats, _ = load_step(p)
        if len(X) < W + 5 or (label not in TAP_STEPS and label not in NEG_STEPS):
            continue
        F = windows(X)
        tw = t[W - 1:]
        y = np.zeros(len(F), int)
        centres = []
        if label in TAP_STEPS:
            cls, centres = tap_centres(label, t, X, beats)
            near = np.zeros(len(F), bool)
            for c in centres:
                y[(tw >= c - 0.04) & (tw <= c + 0.12)] = CLASSES.index(cls)
                near |= np.abs(tw - c) < 0.6
            keep = (y > 0) | ~near                     # drop ambiguous frames around taps
            F, y, tw = F[keep], y[keep], tw[keep]
        feats.append(F); labels.append(y)
        meta.append((label, tw, centres, CLASSES.index(TAP_STEPS[label][0]) if label in TAP_STEPS else 0))
    return feats, labels, meta


def events(tw, prob, thr, gap=0.35):
    ev, last = [], -1e9
    above = prob >= thr
    i = 0
    while i < len(prob):
        if above[i]:
            j = i
            while j + 1 < len(prob) and above[j + 1]:
                j += 1
            peak = i + int(np.argmax(prob[i:j + 1]))
            if tw[peak] - last > gap:
                ev.append(tw[peak]); last = tw[peak]
            i = j + 1
        else:
            i += 1
    return ev


def evaluate(train_paths, test_paths, name):
    Ftr, ytr, _ = build(train_paths)
    Fte, yte, meta = build(test_paths)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, class_weight="balanced", random_state=0)
    clf.fit(np.vstack(Ftr), np.concatenate(ytr))
    print(f"\n== {name}: train {sum(len(y) for y in ytr)} windows, test {sum(len(y) for y in yte)}")
    for thr in (0.5, 0.7, 0.85):
        hit = {1: 0, 2: 0}; total = {1: 0, 2: 0}; false = 0; neg_time = 0.0; wrong = 0
        for F, (label, tw, centres, want) in zip(Fte, meta):
            P = clf.predict_proba(F)
            for c in (1, 2):
                for e in events(tw, P[:, c], thr):
                    if want == c and any(abs(e - x) < 0.35 for x in centres):
                        hit[c] += 1
                    elif want and any(abs(e - x) < 0.35 for x in centres):
                        wrong += 1                        # right step, wrong button
                    else:
                        false += 1
            if want:
                total[want] += len(centres)
            neg_time += (tw[-1] - tw[0]) if len(tw) else 0
        print(f"thr {thr}: left {hit[1]}/{total[1]}  right {hit[2]}/{total[2]}  wrong-button {wrong}  "
              f"false {false} ({false / max(neg_time, 1) * 60:.1f}/min of recorded time)")


if __name__ == "__main__":
    s1 = sorted(Path("data/recordings/s1").glob("*.npz"))
    s2 = sorted(Path("data/recordings/s2").glob("*.npz"))
    neg_train = [p for p in s1 if any(k in p.name for k in ("rest", "typing", "mouse", "face"))]
    neg_test = [p for p in s1 if any(k in p.name for k in ("cursor", "scroll", "flick", "wake"))]
    s1_taps = [p for p in s1 if any(k in p.name for k in ("click", "drag"))]
    evaluate(s1, s2, "train s1 -> test s2")
    evaluate(s2 + neg_train, s1_taps + neg_test, "train s2+s1 rest/typing/mouse/face -> test s1 taps + cursor/scroll/flick/wake")
