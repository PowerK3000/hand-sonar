"""Jitter at rest vs tap preservation for stabiliser settings, on real recordings."""
import itertools
import numpy as np
from handctl.hand import HandFrame
from handctl.controller import HandednessLatch
from handctl.stabilize import LandmarkStabilizer, StabilizerCfg

REST = ["data/recordings/t1/05_rest.npz", "data/recordings/t-20261022-1743/02_rest.npz"]
TAPS = [("data/recordings/t1/00_click.npz", 8), ("data/recordings/t-20261022-1743/00_click.npz", 8),
        ("data/recordings/t1/01_right_click.npz", 12), ("data/recordings/t-20261022-1743/01_right_click.npz", 12)]


def frames(path, cfg):
    z = np.load(path); st = LandmarkStabilizer(cfg) if cfg else None; latch = HandednessLatch()
    for i in range(len(z["t"])):
        if not z["present"][i]:
            if st: st(None)
            continue
        f = HandFrame(t=float(z["t"][i]), world=z["world"][i], image=z["image"][i],
                      handedness=latch(str(z["handedness"][i])), score=float(z["score"][i]))
        yield (st(f) if st else f), z["beats"]


def jitter(cfg):
    out = []
    for p in REST:
        ims = np.array([f.image for f, _ in frames(p, cfg)])
        rel = ims - ims[:, [0, 5, 9, 13, 17]].mean(1, keepdims=True)
        hand = np.median(np.linalg.norm(ims[:, 9] - ims[:, 0], axis=1))
        out.append(np.median(np.linalg.norm(np.diff(rel, axis=0), axis=2)) / hand)
    return float(np.mean(out))


def tap_contrast(cfg):
    """Median over taps of (gap before tap - minimum gap during tap), in hand lengths."""
    vals = []
    for p, tip in TAPS:
        F = list(frames(p, cfg)); t = np.array([f.t for f, _ in F]); beats = F[0][1]
        gap = np.array([np.linalg.norm(f.image[4] - f.image[tip]) / np.linalg.norm(f.image[9] - f.image[0]) for f, _ in F])
        for b in beats:
            pre, dur = (t >= b - 0.35) & (t < b), (t >= b) & (t < b + 0.9)
            if pre.any() and dur.any():
                vals.append(np.median(gap[pre]) - gap[dur].min())
    return float(np.median(vals))


base_j, base_c = jitter(None), tap_contrast(None)
print(f"raw:             jitter {base_j*100:.2f}% hand/frame   tap contrast {base_c:.3f} hand")
for beta, leash in itertools.product((1.0, 4.0, 10.0), (0.0, 0.02, 0.04)):
    cfg = StabilizerCfg(beta=beta, leash_2d=leash, leash_3d=leash)
    j, c = jitter(cfg), tap_contrast(cfg)
    print(f"beta {beta:4} leash {leash:.2f}: jitter {j*100:.2f}% ({j/base_j:.0%} of raw)   tap contrast {c:.3f} ({c/base_c:.0%} of raw)")
