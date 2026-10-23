"""Leave-one-sitting-out tap evaluation using PERSONAL-MODEL features (fitted joint angles +
model-space fingertip gaps) instead of raw keypoints. Fitted poses are cached per step so the
expensive fit runs once.

    python -m tools.model_features_eval
"""
import json
from pathlib import Path

import numpy as np

from handctl import evaluate_taps as ev
from handctl import train_taps as tt
from handctl.controller import HandednessLatch
from handctl.handmodel import calibrate
from handctl.handmodel.fitter import PoseFitter
from handctl.pose_features import windows

SHAPE_PATH = Path("models/hand_shape.json")
FIT_VERSION = 2   # bump when the fitter changes; cache dirs are per shape+fitter, never deleted
CACHE = Path("data/fits")


def cache_dir() -> Path:
    import hashlib
    h = hashlib.sha1(SHAPE_PATH.read_bytes()).hexdigest()[:10]
    return CACHE / f"v{FIT_VERSION}-{h}"
TIPS = (8, 12, 16, 20)


def fit_step(path: Path, shape, cam):
    out = cache_dir() / path.parent.name / path.name
    if out.is_file():
        z = np.load(out)
        return z["t"], z["F"]
    z = np.load(path, allow_pickle=False)
    fitter, latch, last = PoseFitter(shape, cam), HandednessLatch(), None
    t, F = [], []
    for i in range(len(z["t"])):
        if not z["present"][i]:
            fitter.reset(); latch.reset(); continue
        ti = float(z["t"][i])
        if last is not None and ti - last > 0.25:
            fitter.reset()
        last = ti
        im, w = z["image"][i], z["world"][i]
        r = fitter.fit(im, w, latch(str(z["handedness"][i])))
        size = np.linalg.norm(r.local[9] - r.local[0])
        gaps = [np.linalg.norm(r.local[4] - r.local[k]) / size for k in TIPS]
        ref = np.linalg.norm(im[9] - im[0])
        gaps2 = [np.linalg.norm(im[4] - im[k]) / ref for k in TIPS]
        F.append(np.concatenate([r.pose[6:], gaps, gaps2, [r.rms_px / 10]]))
        t.append(ti)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, t=np.array(t), F=np.array(F))
    return np.array(t), np.array(F)


def build(session: Path, shape, cam):
    steps = []
    for p in sorted(session.glob("*.npz")):
        if p.name.endswith(".tmp.npz"):
            continue
        z = np.load(p, allow_pickle=False)
        label = str(z["label"])
        if label not in tt.TAP_CLASSES and label not in tt.NEGATIVE_STEPS:
            continue
        t, X = fit_step(p, shape, cam)
        if len(X) < 30:
            continue
        tw, F = windows(t, X)
        y = np.zeros(len(F), int)
        taps = []
        if label in tt.TAP_CLASSES:
            # Same beat-locked labels as train_taps (latency from the raw recording).
            _, traw, Xraw, beats = tt.load_step(p)
            lat = tt.reaction_latency(label, traw, Xraw, beats)
            taps = [b + lat for b in beats]
            near = np.zeros(len(F), bool)
            for tap in taps:
                y[(tw >= tap - tt.POS_BEFORE_S) & (tw <= tap + tt.POS_AFTER_S)] = tt.TAP_CLASSES[label]
                near |= np.abs(tw - tap) < tt.NEAR_S
            keep = (y > 0) | ~near
            F, y, tw = F[keep], y[keep], tw[keep]
        steps.append((label, tw, F, y, taps))
    return steps


def main():
    shape = calibrate.load(SHAPE_PATH)
    sessions = ev.sessions_with_taps(Path("data/recordings"))
    cam = calibrate.load_frames([sessions[0]])[1]
    data = {s: build(s, shape, cam) for s in sessions}
    grid = [(tl, tr) for tl in (0.4, 0.5, 0.6, 0.7, 0.8) for tr in (0.3, 0.4, 0.5, 0.6, 0.7)]
    results = {g: [] for g in grid}
    for held in sessions:
        train = [st for s, steps in data.items() if s != held for st in steps]
        X = np.vstack([st[2] for st in train]); y = np.concatenate([st[3] for st in train])
        model = tt.make_model().fit(X, y)
        for g in grid:
            results[g].append(ev.score(model, data[held], *g))
        best = max(grid, key=lambda g: sum(results[g][-1]["hits"].values()) - 2 * (results[g][-1]["false"] + results[g][-1]["extra"] + results[g][-1]["wrong"]))
        r = results[best][-1]
        print(f"held out {held.name}: L{best[0]} R{best[1]} left {r['hits'][1]}/{r['total'][1]} right {r['hits'][2]}/{r['total'][2]} "
              f"extra {r['extra']} wrong {r['wrong']} false {r['false']} ({r['false_per_min']:.1f}/min)")
    agg = {g: (sum(sum(r["hits"].values()) for r in results[g]), sum(r["false"] + r["extra"] + r["wrong"] for r in results[g])) for g in grid}
    g = max(grid, key=lambda k: agg[k][0] - 2 * agg[k][1])
    tot = sum(sum(r["total"].values()) for r in results[g])
    print(f"overall best L{g[0]} R{g[1]}: {agg[g][0]}/{tot} taps, {agg[g][1]} false/stray   (raw-keypoint model was 21/128, 25)")
    ok = [g for g in grid if all(ev.passes(r) for r in results[g])]
    print("SHIP BAR:", f"PASS at {ok[0]}" if ok else "not yet")


if __name__ == "__main__":
    main()
