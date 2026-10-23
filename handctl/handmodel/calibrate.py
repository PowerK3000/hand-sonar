"""Learn Kayden's hand proportions (HandShape) from many frames of his hand.

1. Initial shape from MediaPipe's metric world keypoints: palm template and bone lengths are
   medians over all frames in the anatomical palm frame (robust to per-frame glitches).
2. Alternating refinement on a diverse subset of frames:
     a) fit each frame's pose with the current shape;
     b) re-solve the shape with poses fixed, minimising 2D reprojection error across all frames,
        regularised toward the initial shape (the camera can't see absolute scale).
   Different frames see different bones edge-on vs face-on, so together they pin down true
   proportions that no single frame reveals: the "scan from every angle" idea.

    python -m handctl.handmodel.calibrate data/recordings/<session> [more...] [--out models/hand_shape.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from ..features import DegenerateHand, palm_frame
from . import skeleton as sk
from .fitter import Camera, PoseFitter
from .skeleton import HandShape

PALM_IDS = [1, 5, 9, 13, 17]
SHAPE_PRIOR_REL = 0.25          # bone lengths may deviate ~25% from the anatomical average
PALM_PRIOR_REL = 0.8            # palm landmark layout: nearly free
SPLAY_PRIOR_RAD = 0.4


def initial_shape(frames: list[tuple[np.ndarray, np.ndarray, str]]) -> HandShape:
    """Anatomical average hand scaled to Kayden's overall hand size.

    MediaPipe's world keypoints are learned landmarks, not anatomical joints: on Kayden's scan
    their proximal phalanges came out ~24 mm (real ~40), the palm half as wide, pinky splay 37 deg.
    Starting calibration from those proportions anchored it to the wrong hand (fit rms 15-20 px
    vs ~10 px for a generic hand). So only the overall size is taken from MediaPipe; proportions
    come from 2D refinement.
    """
    sizes = [np.linalg.norm(w[9] - w[0]) for w, _, _ in frames]
    sizes = [x for x in sizes if x > 0.02]
    if len(sizes) < 20:
        raise ValueError(f"need at least 20 usable frames, got {len(sizes)}")
    base = sk.default_shape()
    k = float(np.median(sizes)) / float(np.linalg.norm(base.palm[2]))
    return HandShape(palm=base.palm * k, finger_bones=base.finger_bones * k,
                     finger_splay=base.finger_splay.copy(), thumb_bones=base.thumb_bones * k)


def _pack(s: HandShape) -> np.ndarray:
    return np.concatenate([s.palm.ravel(), s.finger_bones.ravel(), s.finger_splay, s.thumb_bones])


def _unpack(x: np.ndarray) -> HandShape:
    return HandShape(palm=x[:15].reshape(5, 3), finger_bones=np.abs(x[15:27]).reshape(4, 3),
                     finger_splay=x[27:31], thumb_bones=np.abs(x[31:34]))


def diverse_subset(frames, n: int) -> list:
    """Spread the sample over hand orientations: frames binned by palm normal direction."""
    if len(frames) <= n:
        return list(frames)
    keys = []
    for world, _, side in frames:
        try:
            _, rot = palm_frame(world, side)
            keys.append(tuple(np.round(rot[2] * 2).astype(int)))
        except DegenerateHand:
            keys.append(None)
    bins: dict = {}
    for f, k in zip(frames, keys):
        if k is not None:
            bins.setdefault(k, []).append(f)
    out, rng = [], np.random.default_rng(0)
    while len(out) < n and any(bins.values()):
        for k in list(bins):
            if bins[k]:
                out.append(bins[k].pop(rng.integers(len(bins[k]))))
                if len(out) >= n:
                    break
    return out


def refine(shape0: HandShape, frames, camera: Camera, rounds: int = 5, n: int = 150,
           log=print) -> HandShape:
    sample = diverse_subset(frames, n)
    x0 = _pack(shape0)
    # The tracker's keypoints are learned landmarks (e.g. a palm ~half as wide as anatomical
    # knuckles), so the palm template and splay must be free to move to where *MediaPipe*
    # puts them; bone lengths stay closer to anatomy.
    prior_sigma = np.maximum(np.abs(x0) * SHAPE_PRIOR_REL, 0.004)
    prior_sigma[:15] = np.maximum(np.abs(x0[:15]) * PALM_PRIOR_REL, 0.01)
    prior_sigma[27:31] = SPLAY_PRIOR_RAD
    shape = shape0
    for rnd in range(rounds):
        poses, rms = [], []
        for world, image, side in sample:
            fitter = PoseFitter(shape, camera)
            r = fitter.fit(image, world, side)
            poses.append((r.pose, image, world, side))
            rms.append(r.rms_px)
        log(f"  round {rnd + 1}: pose-fit rms {np.median(rms):.2f} px (median over {len(sample)} frames)")

        def residuals(x):
            s = _unpack(x)
            out = [(x - x0) / prior_sigma]
            for pose, image, world, side in poses:
                J = sk.camera_joints(s, pose[None])[0]
                if side == "left":
                    image = image.copy(); image[:, 0] = camera.width - image[:, 0]
                out.append(((camera.project(J) - image) / 4.0).ravel())
            return np.concatenate(out)
        res = least_squares(residuals, _pack(shape), loss="soft_l1", f_scale=2.0, max_nfev=30)
        shape = _unpack(res.x)
    return shape


def load_frames(session_dirs: list[Path], stride: int = 2) -> tuple[list, Camera]:
    frames, cam = [], Camera()
    for d in session_dirs:
        for p in sorted(d.glob("*.npz")):
            if p.name.endswith(".tmp.npz"):
                continue
            z = dict(np.load(p, allow_pickle=False))  # load once: indexing an NpzFile re-decompresses the whole array
            meta = json.loads(str(z["meta"]))
            if meta.get("camera"):
                cam = Camera(width=int(meta["camera"][0]), height=int(meta["camera"][1]))
            from ..controller import HandednessLatch
            latch = HandednessLatch()       # MediaPipe's label flickers; a wrong label mirrors the fit
            for i in range(len(z["t"])):
                if not z["present"][i]:
                    latch.reset()
                    continue
                side = latch(str(z["handedness"][i]))
                if i % stride == 0:
                    frames.append((z["world"][i], z["image"][i], side))
    return frames, cam


def save(shape: HandShape, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(shape.to_dict(), indent=2))
    tmp.replace(path)


def load(path: Path) -> HandShape:
    return HandShape.from_dict(json.loads(path.read_text()))


def main(argv: list[str] | None = None) -> int:
    from .. import paths
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sessions", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=paths.ROOT / "models" / "hand_shape.json")
    p.add_argument("--frames", type=int, default=150)
    args = p.parse_args(argv)
    frames, cam = load_frames(args.sessions)
    print(f"{len(frames)} frames, camera {cam.width}x{cam.height}")
    try:
        s0 = initial_shape(frames)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    s = refine(s0, frames, cam, n=args.frames)
    save(s, args.out)
    print("finger bones (mm):", np.round(s.finger_bones * 1000, 1).tolist())
    print("thumb bones (mm): ", np.round(s.thumb_bones * 1000, 1).tolist())
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
