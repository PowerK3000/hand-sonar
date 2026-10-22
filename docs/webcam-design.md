# Webcam hand control: design (2026-10-22)

Goal: a real input device, not a demo. Hands rest on or near the desk, work at off angles,
and the system infers intent instead of requiring theatrical poses.

## Build vs reuse
- **Reuse** the hand-landmark model (MediaPipe Hands: 21 3D keypoints/hand, ~30+ fps on CPU).
  Training a competitive detector from scratch needs tens of thousands of annotated hand images;
  we would end up with a worse version of the same component.
- **Build from scratch** everything above the landmarks. That is where existing projects are
  demos: they map one pose to one action with fixed thresholds, so they jitter, misfire while
  typing, and break when the hand isn't facing the camera.

## Architecture
1. `camera` — capture thread, fixed exposure where supported, frame timestamps, drop-old-frames.
2. `landmarks` — MediaPipe wrapper; outputs world-space 3D keypoints + per-hand confidence.
3. `features` — view-invariant hand descriptors: joint flexion angles, fingertip distances and
   thumb-to-finger contact metrics normalised by hand size, expressed in a hand-local frame
   (palm-plane basis), so the same gesture gives the same features from any camera angle.
4. `filter` — One Euro filter per keypoint (low jitter at rest, low lag when moving fast).
5. `intent` — personalised temporal classifier trained on Kayden's own recordings
   (labelled sessions at several angles/positions), plus explicit "not a command" classes
   (typing, reaching for mouse, resting, scratching face). Outputs class + confidence.
6. `engagement` — gate against accidental input: control only while engaged (deliberate
   pose or dwell), auto-disengage on keyboard/mouse activity (global input hooks) and when
   confidence drops.
7. `pointer` — relative (trackpad-style) cursor from small fingertip motion with an
   acceleration curve and clutching, so a few centimetres of hand travel covers the screen
   with the wrist resting.
8. `actions` — config-mapped outputs: SendInput click/right-click/scroll/drag, wake; optional
   events to the ipad-pointer overlay.
9. `app` — tray icon, pause hotkey, run at login, crash-safe restart.

## Ergonomics ("no gorilla arm")
- Camera on top of the monitor, tilted down to see the desk/keyboard area where hands rest.
- Gesture map (agreed 2026-10-22, full parameters in `config/gestures.toml`):
  | Gesture | Action |
  |---|---|
  | thumb-to-index tap | left click |
  | thumb-to-middle tap | right click |
  | thumb-to-index pinch, hold, move, release | drag (e.g. browser tabs) |
  | index finger alone slides up/down/sideways | scroll, iPhone "natural" direction, with flick momentum |
  | whole hand moves (palm centre) | cursor, relative + accelerated |
  | open palm held still | wake / show overlay |
- Scroll vs cursor are told apart by *what* moves: index tip moving relative to a still palm
  = scroll; the palm itself moving = cursor.

## Honest limits
- A single camera cannot see fingers hidden behind the palm (edge-on hand); temporal tracking
  and confidence gating soften this but cannot remove it.
- Low light hurts; a lamp near the desk matters more than camera resolution.

## Evaluation (data first)
Record labelled sessions at 3+ hand positions/angles; report per-gesture precision/recall and
false triggers per hour of normal typing. Ship only gestures that beat agreed thresholds.

## Status (2026-10-22, end of day)
Built and tested without a camera (42 tests on synthetic hands, plus live checks of SendInput,
input hooks and the hotkey on this PC):
`handctl/` config, features, one_euro, gestures, pointer, controller, engagement, output,
tracker, hotkey, app (tray), autostart, record, replay.

### When the webcam arrives
1. ~~Download the hand model~~ done 2026-10-22: `models/hand_landmarker.task`, 7,819,105 bytes,
   sha256 fbc2a300...607cde1 (from the URL in `handctl/tracker.py`; models/ is gitignored).
2. Mount the camera on the monitor, tilted down at the desk.
3. `python -m handctl.record --name s1 --position "front"` (repeat at 2-3 positions/angles).
4. `python -m handctl.replay data/recordings/s1` -> tune `config/gestures.toml` until hits are
   high and false triggers in rest/typing/mouse/cursor/face are ~zero.
5. `python -m handctl.app` (tray icon, Ctrl+Alt+H pauses), then
   `python -m handctl.autostart enable` once it's trusted.

### Unverified until real data
- MediaPipe handedness label swap (assumed: unmirrored feed swaps Left/Right).
- Scroll axis sign (`vertical_axis`), all thresholds.
- ipad-pointer integration not started (it should keep working unchanged since we drive the
  real cursor via SendInput).

## Findings from live data (2026-10-22)
- Webcam: Sonix 0C45:64AB, 720p30 via MSMF only. Tracking 25 fps (num_hands=1, 640 px).
- Session s1 (full protocol) and s2 (click/right/drag after re-aiming + light):
  **thumb-to-finger taps are not observable from an above/behind-the-hand camera.**
  MediaPipe 3D never shows contact at real taps (s2: 0.21-0.40 hand lengths at every tap);
  2D image overlap does (0.02-0.04) but the same overlap happens between taps (p5 0.04).
  Best tap detector on s2 + s1 negatives: any setting that catches taps produces more false
  clicks than hits. Thresholds can't fix missing signal.
- Options: (a) change click to a gesture visible from above (fingertip "press" like a mouse
  button), (b) camera from the side of the hand so the thumb gap is in profile, (c) palm
  facing the camera (rejected: gorilla arm).
