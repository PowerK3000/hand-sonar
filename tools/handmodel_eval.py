"""Real-data test of the personal hand model: does the fitted skeleton separate Kayden's taps
from non-taps better than MediaPipe's generic 3D and plain 2D?"""
import sys
import time
from pathlib import Path

import numpy as np

from handctl.controller import HandednessLatch
from handctl.handmodel import calibrate
from handctl.handmodel.fitter import PoseFitter

STEPS = {"click": 8, "right_click": 12}


def run_step(path, shape, cam):
    z = dict(np.load(path, allow_pickle=False))  # load once: indexing an NpzFile re-decompresses the whole array
    label = str(z["label"])
    tip = STEPS[label]
    fitter = PoseFitter(shape, cam)
    latch = HandednessLatch()
    t, gen, model, two, rms = [], [], [], [], []
    last_t = None
    for i in range(len(z["t"])):
        if not z["present"][i]:
            fitter.reset(); latch.reset(); continue
        ti = float(z["t"][i])
        if last_t is not None and ti - last_t > 0.25:
            fitter.reset()
        last_t = ti
        w, im = z["world"][i], z["image"][i]
        side = latch(str(z["handedness"][i]))
        r = fitter.fit(im, w, side)
        size = np.linalg.norm(w[9] - w[0])
        msize = np.linalg.norm(r.joints[9] - r.joints[0])
        t.append(ti)
        gen.append(np.linalg.norm(w[4] - w[tip]) / size)
        model.append(np.linalg.norm(r.joints[4] - r.joints[tip]) / msize)
        two.append(np.linalg.norm(im[4] - im[tip]) / np.linalg.norm(im[9] - im[0]))
        rms.append(r.rms_px)
    return label, np.array(t), np.array(z["beats"]), {"MediaPipe 3D": np.array(gen), "personal model 3D": np.array(model), "2D": np.array(two)}, np.array(rms)


def report(label, t, beats, metrics, rms, name):
    inb = np.zeros(len(t), bool)
    for b in beats:
        inb |= (t >= b - 0.1) & (t < b + 1.0)
    print(f"\n{name} {label}: {len(beats)} taps, fit rms median {np.median(rms):.1f} px")
    for k, m in metrics.items():
        mins = np.array([m[(t >= b) & (t < b + 0.9)].min() for b in beats if ((t >= b) & (t < b + 0.9)).any()])
        rest = m[~inb]
        sep = float(np.mean(mins < np.percentile(rest, 5)))
        print(f"  {k:18} tap minima median {np.median(mins):.2f} | non-tap p5 {np.percentile(rest, 5):.2f} p50 {np.median(rest):.2f} | taps below non-tap p5: {sep:.0%}")


def main(calib_session, test_sessions):
    if calib_session == "--saved":
        shape = calibrate.load(Path("models/hand_shape.json"))
        cam = calibrate.load_frames([Path(test_sessions[0])])[1]
        print("using saved shape models/hand_shape.json")
        for s in test_sessions:
            for f in sorted(Path(s).glob("*.npz")):
                if f.name.split("_", 1)[1].removesuffix(".npz") in STEPS:
                    report(*run_step(f, shape, cam), Path(s).name)
        return
    frames, cam = calibrate.load_frames([Path(calib_session)])
    t0 = time.time()
    shape = calibrate.refine(calibrate.initial_shape(frames), frames, cam, rounds=3, n=150)
    calibrate.save(shape, Path("models/hand_shape.json"))
    print(f"calibrated on {calib_session} in {time.time() - t0:.0f}s; finger bones mm {np.round(shape.finger_bones*1000,1).tolist()} thumb {np.round(shape.thumb_bones*1000,1).tolist()}")
    for s in test_sessions:
        for f in sorted(Path(s).glob("*.npz")):
            if f.name.split("_", 1)[1].removesuffix(".npz") in STEPS:
                report(*run_step(f, shape, cam), Path(s).name)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
