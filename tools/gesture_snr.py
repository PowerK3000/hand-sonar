"""Which candidate gesture stands out most clearly from rest jitter, as seen by this camera?

For each beat: the largest displacement of any fingertip (hand-aligned 2D, hand-length units)
from where it rested just before the beat. Noise: the same statistic over random windows of
the rest step. SNR = median tap displacement / 95th percentile of rest. Also reports which
fingertip moved, so two gestures can be checked for being distinguishable.
"""
import sys
from pathlib import Path

import numpy as np

from handctl.controller import HandednessLatch
from handctl.features import DegenerateHand
from handctl.pose_features import frame_vector

TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}


def tips2d(path):
    z = np.load(path, allow_pickle=False)
    latch = HandednessLatch()
    t, P = [], []
    for i in range(len(z["t"])):
        if not z["present"][i]:
            latch.reset(); continue
        try:
            v = frame_vector(z["world"][i], z["image"][i], latch(str(z["handedness"][i])))
        except DegenerateHand:
            continue
        w2 = v[:42].reshape(21, 2)
        P.append(np.stack([w2[k] for k in TIPS.values()]))
        t.append(float(z["t"][i]))
    return str(z["label"]), np.array(t), np.array(P), np.array(z["beats"])


def window_disp(t, P, start, pre=0.35, dur=0.9):
    pre_m = (t >= start - pre) & (t < start)
    m = (t >= start) & (t < start + dur)
    if pre_m.sum() < 3 or m.sum() < 3:
        return None
    base = np.median(P[pre_m], axis=0)
    d = np.linalg.norm(P[m] - base, axis=2)          # (frames, tips)
    return d.max(axis=0)                               # per tip


def main(session):
    files = {tips2d(p)[0]: p for p in sorted(Path(session).glob("*.npz"))}
    _, t, P, _ = tips2d(files["rest"])
    rng = np.random.default_rng(0)
    rest = [window_disp(t, P, s) for s in rng.uniform(t[0] + 0.5, t[-1] - 1.0, 200)]
    rest = np.array([r for r in rest if r is not None])
    noise = np.percentile(rest, 95, axis=0)
    print("rest jitter p95 per tip (hand lengths):", dict(zip(TIPS, np.round(noise, 3))))
    rows = []
    for label, path in files.items():
        if not label.startswith("cand_"):
            continue
        _, t, P, beats = tips2d(path)
        taps = np.array([d for d in (window_disp(t, P, b) for b in beats) if d is not None])
        snr = np.median(taps, axis=0) / noise
        best = int(np.argmax(snr))
        # Reliability: fraction of taps whose best-tip motion clears 1.5x the rest p95.
        hit = float(np.mean(taps[:, best] > 1.5 * noise[best]))
        rows.append((snr[best], label, list(TIPS)[best], hit, snr))
    print(f"\n{'gesture':<18} {'best tip':<8} {'SNR':>5} {'taps clearly visible':>21}   SNR per tip " + " ".join(f"{k:>6}" for k in TIPS))
    for snr, label, tip, hit, all_snr in sorted(rows, reverse=True):
        print(f"{label[5:]:<18} {tip:<8} {snr:5.1f} {hit:>20.0%}   {'':11}" + " ".join(f"{v:6.1f}" for v in all_snr))


if __name__ == "__main__":
    main(sys.argv[1])
