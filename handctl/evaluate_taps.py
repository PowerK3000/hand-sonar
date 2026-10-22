"""Leave-one-sitting-out evaluation of the personal tap model: the only honest test of whether
it generalises to a sitting it has never seen.

    python -m handctl.evaluate_taps [session dirs...]   (default: every recorded session with taps)

For each sitting: train on all the others, test on it, sweep thresholds. Prints the SHIP BAR
verdict; the live controller should only use thresholds that pass it on every sitting.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import paths
from .train_taps import TAP_CLASSES, build, make_model

SHIP_RECALL = 0.90          # per button
SHIP_FALSE_PER_MIN = 0.5    # in non-gesture activity (rest/typing/cursor/face)
SHIP_EXTRA_RATIO = 0.10     # stray detections inside tap steps, relative to taps
MATCH_S = 0.4
REFRACTORY_S = 0.6


def detect(tw: np.ndarray, p: np.ndarray, thr: float) -> list[float]:
    """Rising-edge events with refractory, mirroring TapDetector."""
    out, last, armed = [], -1e9, True
    for t, v in zip(tw, p):
        if v >= thr and armed and t - last >= REFRACTORY_S:
            out.append(t); last = t; armed = False
        elif v < thr:
            armed = True
    return out


def score(model, steps, thr_left: float, thr_right: float) -> dict:
    hits = {1: 0, 2: 0}; total = {1: 0, 2: 0}; extra = wrong = false = 0; neg_s = 0.0
    for label, tw, F, y, taps in steps:
        if not len(F):
            continue
        P = model.predict_proba(F)
        cls = TAP_CLASSES.get(label, 0)
        used: set[int] = set()
        for c, thr in ((1, thr_left), (2, thr_right)):
            for e in detect(tw, P[:, c], thr):
                near = [i for i, x in enumerate(taps) if abs(e - x) < MATCH_S and i not in used]
                if cls == c and near:
                    used.add(near[0]); hits[c] += 1
                elif cls and near:
                    wrong += 1
                elif cls:
                    extra += 1
                else:
                    false += 1
        if cls:
            total[cls] += len(taps)
        else:
            neg_s += float(tw[-1] - tw[0])
    return {"hits": hits, "total": total, "extra": extra, "wrong": wrong, "false": false,
            "false_per_min": false / max(neg_s, 1.0) * 60, "neg_s": neg_s}


def passes(r: dict) -> bool:
    taps = r["total"][1] + r["total"][2]
    return (all(r["total"][c] and r["hits"][c] / r["total"][c] >= SHIP_RECALL for c in (1, 2))
            and r["false_per_min"] <= SHIP_FALSE_PER_MIN and (r["extra"] + r["wrong"]) <= SHIP_EXTRA_RATIO * taps)


def sessions_with_taps(root: Path) -> list[Path]:
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        names = {f.name.split("_", 1)[-1].removesuffix(".npz") for f in d.glob("*.npz")}
        if {"click", "right_click"} <= names:
            out.append(d)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sessions", nargs="*", type=Path)
    args = p.parse_args(argv)
    sessions = args.sessions or sessions_with_taps(paths.RECORDINGS)
    if len(sessions) < 2:
        print(f"need at least 2 sittings with click+right_click steps, found {[s.name for s in sessions]}. "
              "Record more with record_session.bat", file=sys.stderr)
        return 2
    data = {s: build([s]) for s in sessions}
    grid = [(tl, tr) for tl in (0.4, 0.5, 0.6, 0.7, 0.8) for tr in (0.3, 0.4, 0.5, 0.6, 0.7)]
    results = {g: [] for g in grid}
    for held in sessions:
        train = [st for s, steps in data.items() if s != held for st in steps]
        X = np.vstack([st[2] for st in train]); y = np.concatenate([st[3] for st in train])
        model = make_model().fit(X, y)
        print(f"\n== held out {held.name} (trained on {len(sessions) - 1} other sittings, {len(y)} windows)")
        for g in grid:
            results[g].append(score(model, data[held], *g))
        best = max(grid, key=lambda g: sum(results[g][-1]["hits"].values())
                   - 2 * (results[g][-1]["false"] + results[g][-1]["extra"] + results[g][-1]["wrong"]))
        r = results[best][-1]
        print(f"   best thresholds L{best[0]} R{best[1]}: left {r['hits'][1]}/{r['total'][1]}  right {r['hits'][2]}/{r['total'][2]}  "
              f"extra {r['extra']}  wrong-button {r['wrong']}  false {r['false']} ({r['false_per_min']:.1f}/min)")
    ok = [g for g in grid if all(passes(r) for r in results[g])]
    print("\nSHIP BAR (every held-out sitting): recall >= 90% per button, <= 0.5 false/min, <= 10% stray")
    if ok:
        print(f"PASS at thresholds {ok[0]} (and {len(ok) - 1} other combos). Put them in [taps] and go live.")
    else:
        agg = {g: (sum(sum(r["hits"].values()) for r in results[g]), sum(r["false"] + r["extra"] + r["wrong"] for r in results[g]))
               for g in grid}
        g = max(grid, key=lambda k: agg[k][0] - 2 * agg[k][1])
        tot = sum(sum(r["total"].values()) for r in results[g])
        print(f"NOT YET. Best overall L{g[0]} R{g[1]}: {agg[g][0]}/{tot} taps, {agg[g][1]} false/stray. "
              f"More sittings needed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
