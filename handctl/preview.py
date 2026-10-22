"""Dry-run live preview: camera + skeleton + gesture readouts + the actions the controller
WOULD send. Sends no input. Logs per-frame features and actions for offline tuning.

    python -m handctl.preview [--seconds 120]
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import deque

from . import config, paths
from .controller import Controller
from .engagement import EngagementGate, InputActivityHook
from .gestures import index_pointing, open_palm
from .tracker import Camera, HandTracker, TrackerError

BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
         (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seconds", type=float, default=0, help="auto-close after N seconds (0 = until Esc)")
    args = p.parse_args(argv)
    import cv2
    cfg = config.load(paths.CONFIG)
    gate = EngagementGate(cfg.engagement)
    ctl = Controller(cfg, gate)
    hooks = InputActivityHook(gate)
    try:
        tracker = HandTracker(paths.resolve(cfg.app.model_path), cfg.app.hand, cfg.app.num_hands,
                              cfg.app.process_width)
        camera = Camera(cfg.app.camera_index)
    except TrackerError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    hooks.start()
    paths.LOGS.mkdir(parents=True, exist_ok=True)
    log_path = paths.LOGS / f"preview-{time.strftime('%Y%m%d-%H%M%S')}.csv"
    recent: deque[tuple[float, str]] = deque(maxlen=6)
    cursor = [0.0, 0.0]
    frames, t_start = deque(maxlen=30), time.monotonic()
    win = "handctl preview (DRY RUN - no input sent)  Esc = quit"
    try:
        with open(log_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t", "hand", "score", "engaged", "pinch_i", "pinch_m", "s_index", "s_middle", "s_ring",
                        "s_pinky", "spread", "idx_x", "idx_y", "idx_z", "palm_x", "palm_y", "hand_px", "actions"])
            while not args.seconds or time.monotonic() - t_start < args.seconds:
                t, bgr = camera.latest(timeout=2.0)
                frame = tracker.process(t, bgr)
                now = time.monotonic()
                actions = ctl.step(frame, now)
                frames.append(now)
                f = ctl.last_features
                engaged = gate.engaged(now, f.score if f else None)
                names = []
                for a in actions:
                    if a.kind == "move":
                        cursor[0] += a.dx
                        cursor[1] += a.dy
                        continue
                    label = f"{a.kind} {a.button or ''}".strip()
                    if a.kind == "scroll":
                        label = f"scroll {'down' if a.dy < 0 else 'up'} {abs(a.dy):.0f}"
                    names.append(label)
                    if a.kind != "scroll" or not recent or not recent[-1][1].startswith("scroll"):
                        recent.append((now, label))
                if f is not None:
                    s = f.straightness
                    w.writerow([f"{now:.3f}", f.handedness, f"{f.score:.2f}", int(engaged), f"{f.pinch_index:.3f}",
                                f"{f.pinch_middle:.3f}", *(f"{s[k]:.3f}" for k in ("index", "middle", "ring", "pinky")),
                                f"{f.spread_deg:.1f}", *(f"{v:.3f}" for v in f.index_local),
                                f"{f.palm_px[0]:.1f}", f"{f.palm_px[1]:.1f}", f"{f.hand_px:.1f}", "|".join(names)])
                img = bgr.copy()
                if frame is not None:
                    pts = frame.image.astype(int)
                    for a, b in BONES:
                        cv2.line(img, tuple(pts[a]), tuple(pts[b]), (0, 200, 0), 2)
                    for x, y in pts:
                        cv2.circle(img, (int(x), int(y)), 3, (0, 255, 255), -1)
                fps = (len(frames) - 1) / (frames[-1] - frames[0]) if len(frames) > 1 else 0
                lines = [f"{fps:4.1f} fps   {'ENGAGED' if engaged else 'disengaged'}   "
                         f"virtual cursor {cursor[0]:+.0f},{cursor[1]:+.0f}"]
                if f is not None:
                    pose = "POINT(scroll)" if index_pointing(f, cfg) else "OPEN" if open_palm(f, cfg) else "-"
                    lines.append(f"{f.handedness} {f.score:.2f}  pinch idx {f.pinch_index:.2f} mid {f.pinch_middle:.2f}"
                                 f"  (close<{cfg.click.pinch_close} open>{cfg.click.pinch_open})")
                    lines.append("straight " + " ".join(f"{k[0]}{v:.2f}" for k, v in f.straightness.items())
                                 + f"  spread {f.spread_deg:.0f}deg  pose {pose}")
                else:
                    lines.append("no hand")
                cv2.rectangle(img, (0, 0), (img.shape[1], 24 * len(lines) + 10), (0, 0, 0), -1)
                for i, line in enumerate(lines):
                    cv2.putText(img, line, (10, 24 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                for i, (ts, label) in enumerate(reversed(recent)):
                    alpha = max(0.3, 1 - (now - ts) / 4)
                    cv2.putText(img, f"WOULD: {label}", (10, img.shape[0] - 20 - 28 * i), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, int(255 * alpha), 255), 2)
                cv2.imshow(win, img)
                if cv2.waitKey(1) == 27:
                    break
    finally:
        camera.close()
        tracker.close()
        hooks.stop()
        cv2.destroyAllWindows()
    print(f"log: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
