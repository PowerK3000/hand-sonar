"""How well does a pinch metric separate 'tap happened' (after a beat) from everything else?"""
import sys
from pathlib import Path

import numpy as np

from tools.inspect_rec import load
from handctl.hand import THUMB_TIP, INDEX_TIP, MIDDLE_TIP


def metrics(d, rows, finger_tip):
    t, m3, m2 = [], [], []
    for i, (ti, f) in enumerate(rows):
        if f is None:
            continue
        t.append(ti)
        m3.append(np.linalg.norm(d["world"][i][THUMB_TIP] - d["world"][i][finger_tip]) / f.size_m)
        m2.append(np.linalg.norm(d["image"][i][THUMB_TIP] - d["image"][i][finger_tip]) / f.hand_px)
    return np.array(t), np.array(m3), np.array(m2)


def report(path, finger_tip, label):
    d, rows = load(path)
    t, m3, m2 = metrics(d, rows, finger_tip)
    beats = np.array(d["beats"])
    for name, m in (("3D world", m3), ("2D image", m2)):
        mins = []
        for b in beats:
            w = (t >= b + 0.0) & (t < b + 0.9)
            if w.any():
                mins.append(m[w].min())
        inbeat = np.zeros_like(t, bool)
        for b in beats:
            inbeat |= (t >= b - 0.1) & (t < b + 1.0)
        rest = m[~inbeat]
        print(f"{label:<12} {name}: per-beat min {np.round(sorted(mins), 2)}  | between beats p1 {np.percentile(rest, 1):.2f} p5 {np.percentile(rest, 5):.2f} p50 {np.median(rest):.2f}")


if __name__ == "__main__":
    base = Path(sys.argv[1])
    report(base / "04_click.npz", INDEX_TIP, "click/index")
    report(base / "05_right_click.npz", MIDDLE_TIP, "right/middle")
    report(base / "05_right_click.npz", INDEX_TIP, "right/index")
    for nm in ("00_rest", "01_typing", "03_cursor", "11_face"):
        d, rows = load(base / f"{nm}.npz")
        t, m3, m2 = metrics(d, rows, INDEX_TIP)
        print(f"{nm:<12} index 3D p1 {np.percentile(m3, 1):.2f} min {m3.min():.2f} | 2D p1 {np.percentile(m2, 1):.2f} min {m2.min():.2f}")
