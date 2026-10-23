"""Direction check: which hand motion should drive each cursor axis for this camera placement?

    python -m handctl.axis_check

Prompts (in a camera window): slowly move your hand to the RIGHT, then slowly RAISE it. For each
phase it measures how much each candidate signal changed (image x, image y, apparent size) and
picks the strongest, with its sign, then rewrites the four axis lines in config/gestures.toml.
"""

from __future__ import annotations

import re
import sys
import time

import numpy as np

from . import config, paths
from .features import DegenerateHand, compute
from .tracker import Camera, HandTracker, TrackerError

PHASES = [("right", "Slowly move your hand to the RIGHT (keep it at the same height)"),
          ("raise", "Slowly RAISE your hand (keep it in the same left/right spot)")]
PHASE_S = 4.0
MIN_CLARITY = 1.5      # winner must beat the runner-up by this factor, or we don't trust it
# Per-signal noise (rest jitter, same units as `choose` compares): apparent-size "height" is ~4x
# noisier than image position once scaled by height_gain, so it must win by signal-to-noise,
# not raw size. Picking it on raw size mapped left/right to the noisiest signal (2026-10-22).
NOISE = {"image_x": 1.0, "image_y": 1.0, "height": 4.0}


def measure(camera, tracker, cv2, prompt: str) -> dict[str, float] | None:
    t_end = time.monotonic() + 3 + PHASE_S
    xs, ys, ls, hp = [], [], [], []
    while (left := t_end - time.monotonic()) > 0:
        t, bgr = camera.latest()
        frame = tracker.process(t, bgr)
        recording = left <= PHASE_S
        if frame is not None and recording:
            try:
                f = compute(frame)
                xs.append(f.palm_px[0]); ys.append(f.palm_px[1]); ls.append(np.log(f.palm_size_px)); hp.append(f.hand_px)
            except DegenerateHand:
                pass
        img = bgr.copy()
        cv2.rectangle(img, (0, 0), (img.shape[1], 60), (0, 0, 0), -1)
        msg = prompt if recording else f"GET READY {left - PHASE_S:.0f}: {prompt}"
        cv2.putText(img, msg, (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("handctl direction check (Esc = abort)", img)
        if cv2.waitKey(1) == 27:
            return None
    if len(xs) < 10:
        return {}
    h = float(np.median(hp))
    k = len(xs) // 5 or 1
    first, last = slice(0, k), slice(-k, None)
    # Net change from the start to the end of the phase, in hand-length-equivalent units.
    return {"image_x": (np.median(xs[last]) - np.median(xs[first])) / h,
            "image_y": (np.median(ys[last]) - np.median(ys[first])) / h,
            "height": (np.median(ls[last]) - np.median(ls[first]))}


def choose(change: dict[str, float], height_gain: float, exclude: str | None = None) -> tuple[str, float, float]:
    scaled = {k: v * (height_gain if k == "height" else 1.0) / NOISE[k] for k, v in change.items() if k != exclude}
    ranked = sorted(scaled.items(), key=lambda kv: -abs(kv[1]))
    (src, val), (_, runner) = ranked[0], ranked[1]
    clarity = abs(val) / max(abs(runner), 1e-6)
    return src, float(np.sign(val) or 1.0), clarity


def write_config(h_src: str, h_sign: float, v_src: str, v_sign: float) -> None:
    text = paths.CONFIG.read_text()
    for key, val in (("horizontal_source", f'"{h_src}"'), ("horizontal_sign", f"{h_sign:.1f}"),
                     ("vertical_source", f'"{v_src}"'), ("vertical_sign", f"{v_sign:.1f}")):
        text, n = re.subn(rf"^{key} = .*$", f"{key} = {val}", text, flags=re.M)
        if n != 1:
            raise RuntimeError(f"could not find '{key}' in {paths.CONFIG}")
    paths.CONFIG.write_text(text)
    config.load(paths.CONFIG)            # validate what we wrote


def main() -> int:
    import cv2
    cfg = config.load(paths.CONFIG)
    try:
        tracker = HandTracker(paths.resolve(cfg.app.model_path), cfg.app.hand, cfg.app.num_hands, cfg.app.process_width)
        camera = Camera(cfg.app.camera_index)
    except TrackerError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    try:
        results = {}
        for name, prompt in PHASES:
            r = measure(camera, tracker, cv2, prompt)
            if r is None:
                print("aborted")
                return 1
            if not r:
                print(f"error: hand not tracked during '{name}' phase; check the camera view", file=sys.stderr)
                return 2
            results[name] = r
    finally:
        camera.close(); tracker.close(); cv2.destroyAllWindows()
    g = cfg.pointer.height_gain
    print("measured changes:", {p: {k: round(v, 3) for k, v in r.items()} for p, r in results.items()})
    h_src, h_sign, h_clear = choose(results["right"], g)
    v_src, v_sign_raw, v_clear = choose(results["raise"], g, exclude=h_src)
    v_sign = -v_sign_raw            # raising must move the cursor UP (negative screen y)
    print(f"horizontal <- {h_src} (sign {h_sign:+.0f}, clarity {h_clear:.1f}x); "
          f"vertical <- {v_src} (sign {v_sign:+.0f}, clarity {v_clear:.1f}x)")
    if min(h_clear, v_clear) < MIN_CLARITY:
        print("not clear enough to trust (movement too small or diagonal); config unchanged. Try again, "
              "moving more and in one direction only.", file=sys.stderr)
        return 1
    write_config(h_src, h_sign, v_src, v_sign)
    print(f"updated {paths.CONFIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
