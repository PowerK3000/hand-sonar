"""MediaPipe vs RTMPose on the SAME live frames: rest jitter, tap contrast, speed.

10 s hold still, then 10 s of left-click taps (prompted in the window).
"""
import time

import cv2
import numpy as np

from handctl import config, paths
from handctl.rtmpose import RTMPoseHand
from handctl.tracker import Camera, HandTracker

PALM = [0, 5, 9, 13, 17]


def rel(kp):
    return (kp - kp[PALM].mean(0)) / np.linalg.norm(kp[9] - kp[0])


def main():
    cfg = config.load(paths.CONFIG)
    mp = HandTracker(paths.resolve(cfg.app.model_path), "any", 1, cfg.app.process_width)
    rt = RTMPoseHand(paths.ROOT / "models" / "rtmpose_hand" / "end2end.onnx")
    cam = Camera(cfg.app.camera_index)
    print("rtmpose provider:", rt.provider)
    rows, t_mp, t_rt = [], [], []
    t0 = time.monotonic()
    prev_kp = None
    try:
        while (el := time.monotonic() - t0) < 23:
            t, bgr = cam.latest()
            a = time.perf_counter(); f = mp.process(t, bgr); b = time.perf_counter()
            t_mp.append(b - a)
            phase = "get ready" if el < 3 else "HOLD STILL" if el < 13 else "TAP thumb-index"
            if f is not None:
                box_src = f.image          # fair test: crop where MediaPipe found the hand (no self-feedback)
                c, side = RTMPoseHand.box_from_keypoints(box_src)
                a = time.perf_counter(); kp, conf = rt.infer(bgr, c, side); t_rt.append(time.perf_counter() - a)
                prev_kp = kp if conf.mean() > 0.3 else None
                if el >= 3:
                    rows.append((el, phase, f.image.copy(), kp.copy(), float(conf.mean())))
                for x, y in f.image.astype(int):
                    cv2.circle(bgr, (int(x), int(y)), 3, (0, 255, 0), -1)
                for x, y in kp.astype(int):
                    cv2.circle(bgr, (int(x), int(y)), 3, (0, 0, 255), -1)
            else:
                prev_kp = None
            cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 50), (0, 0, 0), -1)
            cv2.putText(bgr, f"{phase}  {max(0, (13 if el < 13 else 23) - el):.0f}s   green=MediaPipe red=RTMPose",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("tracker comparison", bgr)
            cv2.waitKey(1)
    finally:
        cam.close(); mp.close(); cv2.destroyAllWindows()
    print(f"speed: MediaPipe {np.mean(t_mp) * 1000:.1f} ms/frame, RTMPose {np.mean(t_rt) * 1000:.1f} ms/frame")
    for name, idx in (("MediaPipe", 2), ("RTMPose", 3)):
        still = [r[idx] for r in rows if r[1] == "HOLD STILL"]
        taps = [(r[0], r[idx]) for r in rows if r[1].startswith("TAP")]
        if len(still) < 10 or len(taps) < 10:
            print(name, "not enough frames"); continue
        R = np.array([rel(k) for k in still])
        jit = np.median(np.linalg.norm(np.diff(R, axis=0), axis=2), axis=0)
        gap_still = np.array([np.linalg.norm(k[4] - k[8]) / np.linalg.norm(k[9] - k[0]) for k in still])
        gap_tap = np.array([np.linalg.norm(k[4] - k[8]) / np.linalg.norm(k[9] - k[0]) for _, k in taps])
        print(f"{name:9}: rest jitter median {np.median(jit):.4f} (fingertips {np.median(jit[[4, 8, 12, 16, 20]]):.4f}) hand-lengths/frame | "
              f"thumb-index gap: still p5 {np.percentile(gap_still, 5):.2f}, taps min-p5 {np.percentile(gap_tap, 5):.2f}, taps median {np.median(gap_tap):.2f}")
    print("mean RTMPose confidence %.2f" % np.mean([r[4] for r in rows]))


if __name__ == "__main__":
    main()
