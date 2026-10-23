"""Score the finger-lift detector: hits on index_lift beats, false fires everywhere else."""
import itertools
import sys
from pathlib import Path

import numpy as np

from handctl.hand import HandFrame
from handctl.lift_detector import LiftCfg, LiftDetector

POS = {"cand_index_lift": "left", "cand_middle_lift": "right", "lift_index": "left", "lift_middle": "right"}


def run(path, cfg):
    z = dict(np.load(path, allow_pickle=False))  # load once: indexing an NpzFile re-decompresses the whole array
    det, fires = LiftDetector(cfg), []
    for i in range(len(z["t"])):
        f = None
        if z["present"][i]:
            f = HandFrame(t=float(z["t"][i]), world=z["world"][i], image=z["image"][i],
                          handedness="right", score=float(z["score"][i]))
        b = det.update(f)
        if b:
            fires.append((float(z["t"][i]), b))
    return str(z["label"]), fires, list(z["beats"]), float(z["t"][-1] - z["t"][0])


def evaluate(files, cfg, verbose=False):
    hits = total = wrong = false = 0
    neg_s = 0.0
    rows = []
    for p in files:
        label, fires, beats, dur = run(p, cfg)
        want = POS.get(label)
        used, h, w, fl = set(), 0, 0, 0
        for t, b in fires:
            near = [k for k, x in enumerate(beats) if 0.0 <= t - x <= 1.3 and k not in used]
            if want and near and b == want:
                used.add(near[0]); h += 1
            elif want and near:
                w += 1
            else:
                fl += 1
        if want:
            hits += h; total += len(beats); wrong += w; false += fl
        else:
            false += fl; neg_s += dur
        rows.append(f"   {p.parent.name}/{label:<18} fires {len(fires):3d}  " + (f"hits {h}/{len(beats)} wrong {w} stray {fl}" if want else f"false {fl}"))
    if verbose:
        print("\n".join(rows))
    return hits, total, wrong, false, neg_s


if __name__ == "__main__":
    files = [Path(p) for p in sys.argv[1:]]
    grid = dict(on=[0.6, 0.8], back_ratio=[0.4, 0.6], lead=[1.2], span_s=[0.25, 0.35], finger_to_hand=[2.0],
                max_dur_s=[0.7, 1.0, 1.3], back_cos=[-0.3, 0.0])
    best = []
    for combo in itertools.product(*grid.values()):
        cfg = LiftCfg(**dict(zip(grid, combo)))
        h, tot, w, fl, neg_s = evaluate(files, cfg)
        best.append((h - 3 * (w + fl), h, tot, w, fl, cfg))
    best.sort(key=lambda r: -r[0])
    for s, h, tot, w, fl, cfg in best[:3]:
        print(f"on {cfg.on} back {cfg.back_ratio} lead {cfg.lead} span {cfg.span_s} ratio {cfg.finger_to_hand} dur {cfg.max_dur_s} cos {cfg.back_cos}: hits {h}/{tot}, wrong {w}, false {fl}")
    print("\nbest config per step:")
    evaluate(files, best[0][5], verbose=True)
