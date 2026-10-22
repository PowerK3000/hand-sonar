"""Grid-search pinch/tap detector designs over labelled recordings.

Scores each design by hits on beat-driven steps and false events on everything else, so the
choice of metric and thresholds comes from Kayden's data instead of guesses.
"""
import itertools
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from handctl.hand import INDEX_TIP, MIDDLE_TIP, THUMB_TIP
from handctl.one_euro import OneEuroFilter
from tools.inspect_rec import load

STEPS_EXPECT = {"click": "L", "right_click": "R", "drag": "D"}
NON_COMMAND = ("rest", "mouse", "cursor", "scroll_up", "scroll_down", "flick", "wake", "face")


def series(path):
    d, rows = load(path)
    t, ti, tm = [], {"2d": [], "3d": []}, {"2d": [], "3d": []}
    for i, (ti_, f) in enumerate(rows):
        if f is None:
            continue
        w, im = d["world"][i], d["image"][i]
        t.append(ti_)
        for tip, out in ((INDEX_TIP, ti), (MIDDLE_TIP, tm)):
            out["3d"].append(np.linalg.norm(w[THUMB_TIP] - w[tip]) / f.size_m)
            out["2d"].append(np.linalg.norm(im[THUMB_TIP] - im[tip]) / f.hand_px)
    return str(d["label"]), np.array(t), {k: np.array(v) for k, v in ti.items()}, \
        {k: np.array(v) for k, v in tm.items()}, np.array(d["beats"])


def metric(m, kind):
    if kind == "2d":
        return m["2d"]
    if kind == "3d":
        return m["3d"]
    if kind == "min":
        return np.minimum(m["2d"], m["3d"])
    return 0.5 * (m["2d"] + m["3d"])


def smooth(t, x, cutoff):
    f = OneEuroFilter(cutoff, 1.0, 1.0)
    return np.array([f(np.array([v]), ti)[0] for v, ti in zip(x, t)])


def detect(t, a, b, p):
    """Events from finger metric a (this finger) and b (the other finger)."""
    close, open_, base_min, base_s, max_tap, drag_after, margin = (
        p["close"], p["open"], p["base_min"], p["base_s"], p["max_tap"], p["drag_after"], p["margin"])
    events, state, t0 = [], "idle", 0.0
    for i in range(len(t)):
        if state == "idle":
            prior = (t >= t[i] - base_s) & (t < t[i] - 0.05)
            base = np.median(a[prior]) if prior.any() else 0.0
            if a[i] < close and base > base_min and a[i] < b[i] - margin:
                state, t0 = "pressed", t[i]
        elif state == "pressed":
            if a[i] > open_:
                if t[i] - t0 <= max_tap:
                    events.append((t0, "tap"))
                state = "idle"
            elif drag_after and t[i] - t0 > drag_after:
                events.append((t0, "hold"))
                state = "held"
        elif a[i] > open_:
            state = "idle"
    return events


def score(recs, p):
    hits, beats_total, false = Counter(), Counter(), Counter()
    for label, t, ti, tm, beats in recs:
        a = smooth(t, metric(ti, p["kind"]), p["cutoff"])
        b = smooth(t, metric(tm, p["kind"]), p["cutoff"])
        evs = [(t0, "L" if k == "tap" else "D") for t0, k in detect(t, a, b, p)]
        evs += [(t0, "R") for t0, k in detect(t, b, a, p) if k == "tap"]
        want = STEPS_EXPECT.get(label)
        used = set()
        for t0, e in sorted(evs):
            ok = False
            if e == want:
                for bi, bt in enumerate(beats):
                    if bi not in used and -0.1 <= t0 - bt <= 1.2:
                        used.add(bi); ok = True; break
            if ok:
                hits[label] += 1
            else:
                false[(label, e)] += 1
        if want:
            beats_total[label] += len(beats)
    return hits, beats_total, false


def main(base: Path):
    recs = []
    for f in sorted(base.glob("*.npz")):
        rec = series(f)
        if rec[0] in STEPS_EXPECT or rec[0] in NON_COMMAND:
            recs.append(rec)
    grid = {
        "kind": ["2d", "min", "mean"],
        "close": [0.06, 0.09, 0.12],
        "open": [0.16, 0.22],
        "base_min": [0.12, 0.18],
        "base_s": [0.4],
        "max_tap": [0.45],
        "drag_after": [0.5],
        "margin": [0.0, 0.08],
        "cutoff": [3.0, 8.0],
    }
    results = []
    for combo in itertools.product(*grid.values()):
        p = dict(zip(grid, combo))
        if p["open"] <= p["close"]:
            continue
        hits, beats, false = score(recs, p)
        tot_hits = sum(hits.values()); tot_beats = sum(beats.values()); tot_false = sum(false.values())
        results.append((tot_hits - 2 * tot_false, tot_hits, tot_beats, tot_false, p, hits, beats, false))
    results.sort(key=lambda r: -r[0])
    for s, h, bt, fl, p, hits, beats, false in results[:8]:
        print(f"score {s:4d}  hits {h}/{bt}  false {fl}  {p}")
        print("   per step:", {k: f"{hits[k]}/{beats[k]}" for k in beats}, "false:", dict(false))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
