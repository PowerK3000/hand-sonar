"""Training-free tap detection: a tap is a quick dip in the thumb-fingertip gap relative to the
hand's own recent baseline, followed by a return. Relative-to-self is the point: every learned
model failed across sittings because posture shifts absolute values.

Uses the cached personal-model fits (tools/model_features_eval.py). Picks parameters by the
WORST sitting, not the average, so the result has to generalise.
"""
import itertools
from pathlib import Path

import numpy as np

from handctl import evaluate_taps as ev
from handctl import train_taps as tt
from handctl.handmodel import calibrate
from tools.model_features_eval import fit_step

# Columns in the cached per-frame vectors.
MODEL_GAP = {"left": 21, "right": 22}    # thumb tip -> index / middle tip, personal model 3D
IMG_GAP = {"left": 25, "right": 26}      # same in 2D image (hand-length units)
NEG = {"rest", "typing", "cursor", "face", "mouse"}


def metric(F, which, kind):
    m3, m2 = F[:, MODEL_GAP[which]], F[:, IMG_GAP[which]]
    if kind == "model":
        return m3
    if kind == "image":
        return m2
    return 0.5 * (m3 / np.median(m3) + m2 / np.median(m2))      # combined, each self-normalised


def detect(t, mL, mR, p):
    """Returns [(time, button)]. Fires when the gap returns after a short dip below baseline."""
    out, state, t_on, btn = [], "idle", 0.0, None
    last_fire = -1e9
    for i in range(len(t)):
        win = (t >= t[i] - p["base_s"]) & (t < t[i] - 0.25)
        if win.sum() < 4:
            continue
        bL, bR = np.median(mL[win]), np.median(mR[win])
        dL, dR = (bL - mL[i]) / bL, (bR - mR[i]) / bR
        if state == "idle":
            if max(dL, dR) > p["dip_on"] and t[i] - last_fire > p["refractory"]:
                btn = "left" if dL - dR > p["margin"] else "right" if dR - dL > p["margin"] else None
                if btn:
                    state, t_on = "down", t[i]
        else:
            d = dL if btn == "left" else dR
            if d < p["dip_off"]:
                if t[i] - t_on <= p["max_dur"]:
                    out.append((t[i], btn)); last_fire = t[i]
                state = "idle"
            elif t[i] - t_on > p["max_dur"]:
                state = "hold"       # held too long: not a tap; wait for release
                if d < p["dip_off"]:
                    state = "idle"
        if state == "hold":
            d = dL if btn == "left" else dR
            if d < p["dip_off"]:
                state = "idle"
    return out


def score_sitting(steps, p, kind):
    hits = {"left": 0, "right": 0}; total = {"left": 0, "right": 0}
    wrong = extra = false = 0; neg_s = 0.0
    for label, t, F, taps in steps:
        evs = detect(t, metric(F, "left", kind), metric(F, "right", kind), p)
        want = {"click": "left", "right_click": "right"}.get(label)
        used = set()
        for e, b in evs:
            near = [i for i, x in enumerate(taps) if -0.1 <= e - x <= 0.7 and i not in used]
            if want and near and b == want:
                used.add(near[0]); hits[b] += 1
            elif want and near:
                wrong += 1
            elif want:
                extra += 1
            else:
                false += 1
        if want:
            total[want] += len(taps)
        elif label in NEG:
            neg_s += t[-1] - t[0]
    return {"hits": hits, "total": total, "wrong": wrong, "extra": extra, "false": false,
            "false_per_min": false / max(neg_s, 1) * 60}


def load_all():
    shape = calibrate.load(Path("models/hand_shape.json"))
    data = {}
    for s in ev.sessions_with_taps(Path("data/recordings")):
        cam = calibrate.load_frames([s], stride=1000)[1]
        steps = []
        for p in sorted(s.glob("*.npz")):
            if p.name.endswith(".tmp.npz"):
                continue
            label = str(np.load(p, allow_pickle=False)["label"])
            if label not in ("click", "right_click") and label not in NEG:
                continue
            t, F = fit_step(p, shape, cam)
            taps = []
            if label in ("click", "right_click"):
                _, traw, Xraw, beats = tt.load_step(p)
                lat = tt.reaction_latency(label, traw, Xraw, beats)
                taps = [b + lat for b in beats]
            steps.append((label, t, F, taps))
        data[s.name] = steps
    return data


def summary(r):
    return (f"L {r['hits']['left']}/{r['total']['left']} R {r['hits']['right']}/{r['total']['right']} "
            f"wrong {r['wrong']} extra {r['extra']} false {r['false']} ({r['false_per_min']:.1f}/min)")


def main():
    data = load_all()
    print("sittings:", {k: len(v) for k, v in data.items()})
    grid = dict(kind=["model", "image", "combo"], dip_on=[0.25, 0.35, 0.5], dip_off=[0.1, 0.18],
                margin=[0.05, 0.15], max_dur=[0.6], base_s=[1.2], refractory=[0.8])
    best = []
    for combo in itertools.product(*grid.values()):
        p = dict(zip(grid, combo))
        res = {k: score_sitting(v, p, p["kind"]) for k, v in data.items()}
        # Worst-sitting score: recall per button minus penalty for errors, minimised over sittings.
        def s(r):
            rec = [r["hits"][b] / r["total"][b] for b in ("left", "right") if r["total"][b]]
            n = sum(r["total"].values()) or 1
            return min(rec) - 2 * (r["wrong"] + r["extra"]) / n - 0.2 * r["false_per_min"]
        best.append((min(s(r) for r in res.values()), p, res))
    best.sort(key=lambda x: -x[0])
    for sc, p, res in best[:3]:
        print(f"\nworst-sitting score {sc:.2f}  {p}")
        for k, r in res.items():
            print(f"   {k:18} {summary(r)}")


if __name__ == "__main__":
    main()
