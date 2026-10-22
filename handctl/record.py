"""Guided recording of labelled hand data for tuning.

    python -m handctl.record --name session1 [--position "desk, right of keyboard"]

Shows a live preview with landmarks and step-by-step prompts. Timed gestures follow an on-screen
beat, so replay knows how many events to expect. Each step is saved as soon as it finishes
(crash-safe): data/recordings/<name>/<NN>_<label>.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config, paths
from .tracker import Camera, HandTracker, TrackerError


@dataclass(frozen=True)
class Step:
    label: str
    seconds: float
    prompt: str
    beat_s: float = 0.0          # >0: one gesture per beat, flashed on screen


PROTOCOL = [
    Step("rest", 10, "Rest your hand naturally on the desk. Don't do anything."),
    Step("typing", 15, "Type normally (anything)."),
    Step("mouse", 10, "Use your real mouse normally."),
    Step("cursor", 15, "Move your whole hand around slowly, then faster, as if steering the cursor."),
    Step("click", 18, "On each BEAT: quick tap thumb to INDEX tip.", 1.5),
    Step("right_click", 18, "On each BEAT: quick tap thumb to MIDDLE tip.", 1.5),
    Step("drag", 20, "On each BEAT: pinch thumb+index, slide hand sideways, release.", 2.5),
    Step("scroll_up", 15, "Index out, others curled. On each BEAT: slide fingertip UP (forward).", 1.5),
    Step("scroll_down", 15, "Index out, others curled. On each BEAT: slide fingertip DOWN (back).", 1.5),
    Step("flick", 15, "On each BEAT: quick flick of the index finger, like flicking a phone.", 2.0),
    Step("wake", 12, "On each BEAT: spread your fingers wide and hold still ~1 s.", 4.0),
    Step("face", 12, "Do normal non-command stuff: touch face, scratch head, grab a drink."),
]

# Training set for the personal motion model: many fast reps, camera untouched, plus
# non-command activity so the model learns what not to fire on (~7.5 min).
TRAIN_PROTOCOL = [
    Step("click", 60, "On each BEAT: quick tap thumb to INDEX tip. Hand resting where you'd use it.", 1.4),
    Step("right_click", 60, "On each BEAT: quick tap thumb to MIDDLE tip.", 1.4),
    Step("press_index", 45, "On each BEAT: press INDEX fingertip down a little, like clicking a mouse button.", 1.4),
    Step("press_middle", 45, "On each BEAT: press MIDDLE fingertip down, like a right mouse button.", 1.4),
    Step("drag", 40, "On each BEAT: pinch thumb+index, slide hand sideways, release.", 2.5),
    Step("rest", 45, "Rest your hand naturally. Small fidgets are fine."),
    Step("typing", 45, "Type normally (anything)."),
    Step("cursor", 45, "Move your whole hand around as if steering the cursor. No taps."),
    Step("face", 40, "Normal non-command stuff: touch face, scratch head, grab a drink, reach around."),
]
PROTOCOLS = {"tuning": PROTOCOL, "train": TRAIN_PROTOCOL}


def _draw(cv2, img, frame, step: Step, remaining: float, beat_on: bool, idx: int, total: int):
    if frame is not None:
        for x, y in frame.image.astype(int):
            cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)
    cv2.rectangle(img, (0, 0), (img.shape[1], 70), (0, 0, 0), -1)
    cv2.putText(img, f"[{idx + 1}/{total}] {step.label}  {remaining:4.1f}s", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, step.prompt, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    if beat_on:
        cv2.circle(img, (img.shape[1] - 50, 120), 35, (0, 0, 255), -1)
        cv2.putText(img, "BEAT", (img.shape[1] - 82, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if frame is None:
        cv2.putText(img, "NO HAND", (10, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


def _save(path: Path, step: Step, rows: list, beats: list[float], meta: dict) -> None:
    present = np.array([r[1] is not None for r in rows], dtype=bool)
    world = np.full((len(rows), 21, 3), np.nan)
    image = np.full((len(rows), 21, 2), np.nan)
    hand = np.array(["" if r[1] is None else r[1].handedness for r in rows])
    score = np.array([0.0 if r[1] is None else r[1].score for r in rows])
    for i, (_, f) in enumerate(rows):
        if f is not None:
            world[i], image[i] = f.world, f.image
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, t=np.array([r[0] for r in rows]), present=present, world=world, image=image,
                        handedness=hand, score=score, beats=np.array(beats), label=step.label,
                        meta=json.dumps(meta))
    tmp.replace(path)  # atomic: a crash mid-write never leaves a corrupt recording


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--position", default="", help="free-text note: where the hand/camera is")
    p.add_argument("--only", nargs="*", help="record just these step labels")
    p.add_argument("--protocol", choices=sorted(PROTOCOLS), default="tuning")
    args = p.parse_args(argv)
    import cv2
    cfg = config.load(paths.CONFIG)
    protocol = PROTOCOLS[args.protocol]
    steps = [s for s in protocol if not args.only or s.label in args.only]
    if not steps:
        p.error(f"no steps match {args.only}; labels: {[s.label for s in protocol]}")
    try:
        tracker = HandTracker(paths.resolve(cfg.app.model_path), "any", cfg.app.num_hands,
                              cfg.app.process_width)
        camera = Camera(cfg.app.camera_index)
    except TrackerError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    out_dir = paths.RECORDINGS / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"position": args.position, "camera": [camera.width, camera.height, camera.fps]}
    win = "handctl recorder (Esc = abort, Space = skip step)"
    try:
        for idx, step in enumerate(steps):
            # 3 s get-ready countdown with live preview.
            t_ready = time.monotonic() + 3
            while (left := t_ready - time.monotonic()) > 0:
                t, bgr = camera.latest()
                img = bgr.copy()
                _draw(cv2, img, tracker.process(t, bgr), step, left, False, idx, len(steps))
                cv2.putText(img, f"GET READY {left:.0f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow(win, img)
                if cv2.waitKey(1) == 27:
                    return 1
            rows, beats = [], []
            t0 = time.monotonic()
            next_beat = t0 + step.beat_s if step.beat_s else None
            skipped = False
            while (el := time.monotonic() - t0) < step.seconds:
                t, bgr = camera.latest()
                frame = tracker.process(t, bgr)
                rows.append((t, frame))
                beat_on = False
                if next_beat is not None:
                    if t >= next_beat:
                        beats.append(next_beat)
                        next_beat += step.beat_s
                    beat_on = bool(beats) and t - beats[-1] < 0.25
                img = bgr.copy()
                _draw(cv2, img, frame, step, step.seconds - el, beat_on, idx, len(steps))
                cv2.imshow(win, img)
                key = cv2.waitKey(1)
                if key == 27:
                    return 1
                if key == 32:
                    skipped = True
                    break
            if skipped:
                continue
            path = out_dir / f"{idx:02d}_{step.label}.npz"
            _save(path, step, rows, beats, meta)
            seen = sum(r[1] is not None for r in rows) / max(1, len(rows))
            print(f"saved {path.name}: {len(rows)} frames, hand visible {seen:.0%}, {len(beats)} beats")
    finally:
        camera.close()
        tracker.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
