# hand-sonar

Turns an ordinary webcam into a real Windows input device: cursor, click, right-click, drag and scroll from a resting hand, with no theatrical poses.

The landmark model is reused (MediaPipe Hands). Everything above it is built from scratch, because that's where existing projects stop at demos: view-invariant hand features, One Euro filtering plus a jitter stabiliser, engagement gating so typing doesn't fire commands, a 27-DoF personal kinematic hand model with calibration, and personal tap classifiers.

The repo also has an earlier acoustic and RF sensing exploration (`hand_sonar/`: ultrasonic tone probes, BLE and Wi-Fi RSSI probes), which gives the project its name.

## How it was built: measure, then decide

The full lab notebook is [`docs/webcam-design.md`](docs/webcam-design.md), negative results included. Highlights:

- **Thumb-to-finger taps aren't observable from an overhead camera.** At real taps, MediaPipe 3D never shows contact (0.21–0.40 hand lengths). Any detector setting that catches taps fires more false clicks than hits. The fix is camera placement, not thresholds.
- **Personal hand model.** Scan calibration takes thumb-to-middle tap separability from 45% to 98% within one sitting. Cross-sitting generalisation stays the binding constraint, and is reported as such: 37/128 taps with 12 false, against 21/128 with 25 false for raw keypoints. Not shipped, because it didn't meet the bar.
- **GPU model trial.** RTMPose-m on a GTX 1060 versus MediaPipe on CPU: 27 ms vs 30 ms per frame, rest jitter 0.0112 vs 0.0089 hand lengths per frame. Not adopted.
- **Baseline comparison.** Run against a popular open-source project with identical tracking. The differences come from the smoothing chain and the engagement design.

## Layout

| Path | What |
|---|---|
| `handctl/` | The input device: tracker, features, filters, gestures, controller, engagement, output (SendInput), tray app, autostart |
| `handctl/handmodel/` | Kinematic skeleton, fitter, calibration, Numba forward kinematics |
| `tools/` | Evaluation and experiment scripts behind every number in the design doc |
| `tests/` | Synthetic-hand unit tests |
| `config/gestures.toml` | Gesture map and all thresholds |
| `hand_sonar/` | Earlier acoustic / BLE / Wi-Fi sensing probes |

## Run

```
pip install -r requirements.txt
python -m pytest tests
python -m handctl.app        # tray app; Ctrl+Alt+H pauses
```

The hand landmark model (`models/hand_landmarker.task`) is downloaded separately; its URL is in `handctl/tracker.py`.

Windows only (SendInput and global input hooks).
