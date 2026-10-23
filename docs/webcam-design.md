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

## Motion-model experiment (Kayden's idea: interpret the whole hand over time)
`tools/seq_experiment.py`: per-frame view-normalised pose (2D hand-aligned keypoints, 3D palm
frame, 15 joint flexion angles) -> 0.4 s window summaries -> gradient-boosted classifier.
- Train s1 -> test s2 (camera was re-aimed between them): 0/29 taps. Doesn't transfer across
  camera placements.
- Pooled s1+s2, grouped 5-fold CV (3 s blocks): per-frame recall left 50% (precision 62%),
  right 49% (42%), vs base rates of 3.4% / 1.8%. **There is real signal in whole-hand motion**;
  44 taps is just far too little data, and the camera must stay put.
- Next: fixed camera, ~40 taps/min per gesture for a few minutes, negatives, leave-one-session-out.

## Live result of the personal tap model (2026-10-22)
- t1 training session (7.5 min, 84 taps): held-out last-40% of the SAME session scored
  left 12/16, right 16/17, 0 false - but that split shares posture/lighting/position.
- Live dry run (new sitting, 16 fps): 0 clicks; max confidence 0.28 left / 0.33 right.
- Replaying t1 through the exact live detector (26 and 13 fps) fires strongly, so the live
  path is correct: **the model does not generalise across sittings** (same as s1 -> s2).
- Wake (spread palm) fired 14x falsely in the live run: not usable as-is.
- Next options: (a) camera beside the hand so thumb contact is visible (geometry works
  without ML); (b) several training sessions across sittings/positions, evaluated
  leave-one-session-out, which is the only honest generalisation test.

## Personal-hand-model experiment (Kayden's "scan my hand, model it" idea)
`tools/personal_model_experiment.py` on t1: personal bone lengths (95th pct of projected
length over many orientations) + per-bone depth from known length (Taylor reconstruction).
- Separability (taps below the non-tap 5th percentile):
  click: generic 3D 36%, personal 3D 17%, 2D 40%; right: generic 45%, personal 71%, **2D 95%**.
- The simple closed-form version is worse for left taps: 2D keypoint noise of a few px on
  ~30 px bones blows up in sqrt(L^2 - l^2) depth, and depth signs from MediaPipe are noisy.
  A full kinematic fit (personal skeleton + joint limits + temporal prior) is the proper
  version; multi-day build, uncertain payoff for left taps.
- Strong side result: **thumb-to-middle contact is 95% separable in plain 2D** from this
  camera. Thumb-to-index is not (the thumb naturally lies across the index from above).

## Full personal hand model (built 2026-10-22, Kayden's choice)
`handctl/handmodel/`: kinematic skeleton (27 DoF, joint limits), per-frame fitter (robust 2D +
weak MediaPipe-3D prior + temporal prior + coupling, warm-started), calibration (MediaPipe-median
init + alternating pose/shape refinement), Numba FK (8.7 ms/frame). Scan protocol:
`scan_hand.bat` (record ~75 s rotation/curl/thumb-touch scan, then calibrate).
- Synthetic: occluded pinch that MediaPipe-style 3D puts 43.5 mm apart -> model 4.8 mm.
  Calibration halves proportion error vs a generic hand.
- Real, single-frame tap separability: s2 left click (not the calibration sitting)
  MediaPipe 9% -> model 82%; but s2 right 64% -> 36%, t1 left 36% -> 29%. Calibrated from tap
  sessions, not a scan: proportions look off (index proximal 30 mm), calib rms ~12 px.
- Leave-one-sitting-out tap model on model features: 14/128 hits, 9 false/stray
  (raw keypoints: 21/128, 25). Fewer false triggers, not more hits.
- **Binding constraint is data**: 3 sittings, two with 11 taps per class. No feature set can
  generalise across sittings from that. Plan: one hand scan (`scan_hand.bat`), then 3-4 short
  sittings on different days (`record_session.bat`), re-run `python -m handctl.evaluate_taps`
  and `python -m tools.model_features_eval`. Live integration of the fitter waits for a win.

### Scan calibration (scan1, 2026-10-22)
- Bug found: MediaPipe world keypoints are learned landmarks with non-anatomical proportions
  (proximal phalanges ~24 mm, palm half as wide). Calibration was initialised from them and
  anchored within 15%, giving 15-20 px fits. Now: anatomical average scaled to Kayden's hand
  size, palm layout nearly free (it must match where MediaPipe puts landmarks), 5 rounds.
  Result: index 41/22/17 mm, middle 42/27/19, thumb metacarpal 41 mm; scan fit ~9.9 px.
- Tap separability with the scan-calibrated model (taps below non-tap p5):
  t1 right 98% (MP 45%, 2D 86%); t1 click 36% (= MP); s2 click 55% (MP 9%, 2D 64%);
  s2 right 36% (MP 64%).
- Next fix: curled ring/pinky misfit 33-39 px during taps (top view: curled vs straight-away
  ambiguity -> local minimum). Add per-finger multi-hypothesis re-initialisation when a
  finger's 2D residual is high.
- Cross-sitting tap model on scan-calibrated fitted features (tools/model_features_eval.py):
  **37/128 taps, 12 false/stray** vs raw keypoints 21/128, 25 (s1 left 10/11, t1 left 25/42
  trained on only 22 left taps). ~1.8x hits, half the false triggers; ship bar not met.
  Right-click doesn't transfer yet. Bottleneck remains data: 3 sittings, 2 of them tiny.

### Jitter stabiliser + 4th sitting (2026-10-22)
- Rest jitter, raw MediaPipe: ~2.1% of hand length per frame (2.3-2.5 px), fingertips spiking
  to ~15 px; 3D ~2 mm/frame. `handctl/stabilize.py` (One Euro in hand units + per-point leash):
  beta 10, leash 0.03 -> rest jitter ~45% of raw, tap motion ~96% preserved (tools/stabilizer_eval.py).
- Cross-sitting tap model, 4 sittings (192 taps):
  unstabilised 55/192, 15 false/stray; stabilised 65/192, 28 false/stray. New sitting
  (t-20261022-1743, hand 50% larger in frame, different position) 0/32 either way.
- Conclusion so far: overhead-camera resting-hand taps do not generalise across sittings at
  this data scale, even with personal model + stabilisation.

## Decision point (2026-10-22, ~4 h of live tuning)
Overhead camera + resting hand is the root limit: thumb contact hidden, height only as noisy
apparent size, small foreshortened hand. Live: height-based vertical brought jitter back; lift
clicks fired 8x on a still hand. Every tweak traded jitter vs lag vs recall.
Strong fix = camera placement, not code: webcam low on the desk, 30-50 cm from the resting hand,
viewing it from the front/side. Then up/down = image y (raise/lower hand), side = image x,
clicks = thumb-finger pinch seen in profile, with the stabiliser/leash/glide kept on top.
Config switched to vertical_source = "image_y", taps mode = "pinch" for that setup.
