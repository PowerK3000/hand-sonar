"""Feasibility: does a PERSONAL hand model (Kayden's own bone lengths) recover thumb-finger
contact that MediaPipe's generic 3D misses?

Method (Taylor 2000, known-length reconstruction under weak perspective):
1. Per frame, camera scale s from the palm: least-squares fit of 2D palm points to MediaPipe's
   (camera-aligned) 3D palm.
2. Personal bone lengths: a bone's projected length l2d/s is largest when it lies parallel to the
   image plane, so a high percentile over many hand orientations estimates its true length.
3. Reconstruct depth bone by bone: |dz| = sqrt(L^2 - (l2d/s)^2); sign from MediaPipe's z guess.
4. Compare thumb-index tip distance (personal 3D) at real taps vs everything else.
"""
import sys
from pathlib import Path

import numpy as np

from handctl.hand import INDEX_MCP, INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP, PINKY_MCP, RING_MCP, THUMB_TIP, WRIST

PALM = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
PARENT = {1: 0, 2: 1, 3: 2, 4: 3, 5: 0, 6: 5, 7: 6, 8: 7, 9: 0, 10: 9, 11: 10, 12: 11,
          13: 0, 14: 13, 15: 14, 16: 15, 17: 0, 18: 17, 19: 18, 20: 19}
BONE_PERCENTILE = 95


def palm_scale(world, image):
    W = world[PALM, :2] - world[PALM, :2].mean(0)
    I = image[PALM] - image[PALM].mean(0)
    return float((W * I).sum() / max((W * W).sum(), 1e-12))     # px per metre


def frames(paths):
    for p in paths:
        d = dict(np.load(p, allow_pickle=False))  # load once: indexing an NpzFile re-decompresses the whole array
        for i in np.flatnonzero(d["present"]):
            yield str(d["label"]), float(d["t"][i]), d["world"][i], d["image"][i], d["beats"]


def check_alignment(paths):
    """MediaPipe world axes should match image axes (x right, y down); verify on real data."""
    cx = cy = 0.0
    n = 0
    for _, _, w, im, _ in frames(paths):
        for c, p in PARENT.items():
            dw, di = w[c] - w[p], im[c] - im[p]
            cx += dw[0] * di[0]; cy += dw[1] * di[1]; n += 1
        if n > 20000:
            break
    return cx > 0, cy > 0


def bone_lengths(paths):
    proj = {c: [] for c in PARENT}
    for _, _, w, im, _ in frames(paths):
        s = palm_scale(w, im)
        if s <= 0:
            continue
        for c, p in PARENT.items():
            proj[c].append(np.linalg.norm(im[c] - im[p]) / s)
    return {c: float(np.percentile(v, BONE_PERCENTILE)) for c, v in proj.items()}


def reconstruct(w, im, L):
    s = palm_scale(w, im)
    P = np.zeros((21, 3))
    P[:, :2] = (im - im[WRIST]) / s
    for c in range(1, 21):             # parents always have lower indices
        p = PARENT[c]
        l2 = np.linalg.norm(P[c, :2] - P[p, :2])
        dz = np.sqrt(max(L[c] ** 2 - l2 ** 2, 0.0))
        sign = np.sign(w[c, 2] - w[p, 2]) or 1.0
        P[c, 2] = P[p, 2] + sign * dz
    return P


def separability(name, step_files, L, size_ref):
    for f in step_files:
        d = dict(np.load(f, allow_pickle=False))  # load once: indexing an NpzFile re-decompresses the whole array
        label = str(d["label"])
        tip = INDEX_TIP if label == "click" else MIDDLE_TIP
        t, gen, per, two = [], [], [], []
        for i in np.flatnonzero(d["present"]):
            w, im = d["world"][i], d["image"][i]
            size = np.linalg.norm(w[MIDDLE_MCP] - w[WRIST])
            P = reconstruct(w, im, L)
            t.append(d["t"][i])
            gen.append(np.linalg.norm(w[THUMB_TIP] - w[tip]) / size)
            per.append(np.linalg.norm(P[THUMB_TIP] - P[tip]) / size_ref)
            two.append(np.linalg.norm(P[THUMB_TIP, :2] - P[tip, :2]) / size_ref)
        t = np.array(t)
        beats = np.array(d["beats"])
        inb = np.zeros(len(t), bool)
        for b in beats:
            inb |= (t >= b - 0.1) & (t < b + 1.0)
        print(f"\n{name} {label} ({len(beats)} taps)")
        for nm, m in (("MediaPipe generic 3D", np.array(gen)), ("personal-model 3D", np.array(per)),
                      ("2D only", np.array(two))):
            mins = np.array([m[(t >= b) & (t < b + 0.9)].min() for b in beats if ((t >= b) & (t < b + 0.9)).any()])
            rest = m[~inb]
            # Fraction of taps whose minimum is below the 5th percentile of non-tap frames: higher = separable.
            sep = float(np.mean(mins < np.percentile(rest, 5)))
            print(f"  {nm:22} tap minima median {np.median(mins):.2f} | non-tap p5 {np.percentile(rest, 5):.2f} "
                  f"p50 {np.median(rest):.2f} | taps below non-tap p5: {sep:.0%}")


def main(session):
    files = sorted(Path(session).glob("*.npz"))
    ax, ay = check_alignment(files)
    print(f"world/image axis agreement: x {ax}, y {ay}")
    L = bone_lengths(files)
    size_ref = L[MIDDLE_MCP]          # wrist -> middle knuckle, personal
    print("personal bone lengths (mm):", {c: round(v * 1000, 1) for c, v in L.items()})
    separability(Path(session).name, [f for f in files if f.name.split("_", 1)[1] in ("click.npz", "right_click.npz")],
                 L, size_ref)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/recordings/t1")
