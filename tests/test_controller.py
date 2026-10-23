from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from handctl import config
from handctl.controller import Controller
from handctl.engagement import EngagementGate
from tests import synth

CFG = config.load(Path(__file__).resolve().parents[1] / "config" / "gestures.toml")
CFG = replace(CFG, taps=replace(CFG.taps, mode="pinch"),    # threshold-pinch behaviour under test
              wake=replace(CFG.wake, enabled=True))
FPS = 60.0
POINT = {"index": 0.1, "middle": 0.95, "ring": 0.95, "pinky": 0.95}   # index out, rest curled
OPEN = {"index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 0.0}


class Rig:
    def __init__(self, cfg=CFG, rot=None):
        self.gate = EngagementGate(cfg.engagement)
        self.ctl = Controller(cfg, self.gate)
        self.t = 0.0
        self.rot = rot
        self.log = []

    def run(self, seconds, hand=None, position=(0, 0, 0.5), end_position=None, **hand_kw):
        """Feed `seconds` of frames. hand=None -> no hand visible. Position may glide linearly."""
        n = max(1, round(seconds * FPS))
        anat = synth.anatomical_hand(**hand_kw) if hand is None and hand_kw else hand
        out = []
        for i in range(n):
            self.t += 1 / FPS
            if anat is None:
                frame = None
            else:
                p = np.asarray(position, float)
                if end_position is not None:
                    p = p + (np.asarray(end_position, float) - p) * (i + 1) / n
                frame = synth.frame(self.t, anat, rot=self.rot, position=tuple(p))
            out += self.ctl.step(frame, self.t)
        self.log += out
        return out


def kinds(actions):
    return [(a.kind, a.button) for a in actions if a.kind not in ("move", "scroll")]


def test_quick_tap_is_one_click_and_cursor_stays_put():
    r = Rig()
    r.run(0.3, thumb_to=None)
    acts = r.run(0.12, thumb_to="index") + r.run(0.3, thumb_to=None)
    assert kinds(acts) == [("click", "left")]
    assert not [a for a in acts if a.kind == "move"]


def test_hold_becomes_drag_and_releases():
    r = Rig()
    r.run(0.3, thumb_to=None)
    acts = r.run(0.5, thumb_to="index") + r.run(0.3, thumb_to=None)
    assert kinds(acts) == [("down", "left"), ("up", "left")]


def test_pinch_and_move_drags_with_cursor_motion():
    r = Rig()
    r.run(0.3, thumb_to=None)
    acts = r.run(0.6, thumb_to="index", position=(0, 0, 0.5), end_position=(0.08, 0, 0.5))
    acts += r.run(0.3, thumb_to=None, position=(0.08, 0, 0.5))
    k = kinds(acts)
    assert k == [("down", "left"), ("up", "left")]
    down_i = next(i for i, a in enumerate(acts) if a.kind == "down")
    up_i = next(i for i, a in enumerate(acts) if a.kind == "up")
    moved = sum(a.dx for a in acts[down_i:up_i] if a.kind == "move")
    assert abs(moved) > 100


def test_right_click_does_not_fire_left():
    r = Rig()
    r.run(0.3, thumb_to=None)
    acts = r.run(0.15, thumb_to="middle") + r.run(0.3, thumb_to=None)
    assert kinds(acts) == [("click", "right")]


def test_long_middle_pinch_cancels_right_click():
    r = Rig()
    r.run(0.3, thumb_to=None)
    acts = r.run(1.0, thumb_to="middle") + r.run(0.3, thumb_to=None)
    assert kinds(acts) == []


def test_hysteresis_no_double_click_when_hovering_near_threshold():
    r = Rig()
    r.run(0.3, thumb_to=None)
    acts = r.run(0.1, thumb_to="index", gap=0.0)
    # Hover between close (0.18) and open (0.28) hand lengths, jittering.
    for g in (0.019, 0.017, 0.020, 0.016, 0.021, 0.018):
        acts += r.run(1 / FPS, thumb_to="index", gap=g)
    acts += r.run(0.3, thumb_to=None)
    assert kinds(acts) == [("click", "left")]


def _finger_slide(r, start_curl, end_curl, seconds):
    acts = []
    n = round(seconds * FPS)
    for i in range(n):
        c = start_curl + (end_curl - start_curl) * (i + 1) / n
        acts += r.run(1 / FPS, curl={**POINT, "index": c})
    return acts


def test_scroll_is_natural_like_iphone():
    r = Rig()
    r.run(0.3, curl={**POINT, "index": 0.15})
    # Straightening the index pushes the tip forward ("up"): content follows -> page scrolls down.
    acts = _finger_slide(r, 0.15, 0.0, 0.3)
    v = sum(a.dy for a in acts if a.kind == "scroll")
    assert v < -50
    assert not [a for a in acts if a.kind == "move"]


def test_scroll_direction_flips_when_natural_disabled():
    cfg = replace(CFG, scroll=replace(CFG.scroll, natural=False))
    r = Rig(cfg)
    r.run(0.3, curl={**POINT, "index": 0.15})
    acts = _finger_slide(r, 0.15, 0.0, 0.3)
    assert sum(a.dy for a in acts if a.kind == "scroll") > 50


def test_flick_coasts_then_stops():
    r = Rig()
    r.run(0.3, curl={**POINT, "index": 0.15})
    _finger_slide(r, 0.15, 0.0, 0.1)                  # fast flick
    coast = r.run(2.0, curl={"index": 0.95, "middle": 0.95, "ring": 0.95, "pinky": 0.95})  # fist ends the pose
    steps = [a.dy for a in coast if a.kind == "scroll"]
    assert len(steps) > 5
    assert abs(steps[-1]) < abs(steps[0])
    assert r.ctl.scroll._coast_t is None              # momentum finished on its own


def test_keyboard_mid_drag_releases_button_and_blocks():
    r = Rig()
    r.run(0.3, thumb_to=None)
    acts = r.run(0.5, thumb_to="index")
    assert kinds(acts) == [("down", "left")]
    r.gate.note_keyboard(r.t)
    acts = r.run(0.2, thumb_to="index")
    assert kinds(acts) == [("up", "left")]
    acts = r.run(0.15, thumb_to=None) + r.run(0.12, thumb_to="index")
    assert kinds(acts) == []                           # still inside the 400 ms re-engage window


def test_hand_lost_mid_drag_releases_button():
    r = Rig()
    r.run(0.3, thumb_to=None)
    r.run(0.5, thumb_to="index")
    acts = r.run(0.1, hand=None)
    assert kinds(acts) == [("up", "left")]


def test_cursor_mirrors_and_is_distance_invariant():
    def travel(depth):
        r = Rig()
        r.run(0.3, thumb_to=None, position=(0, 0, depth))
        # Move the hand half a hand length (0.045 m) to the camera's right.
        acts = r.run(0.5, thumb_to=None, position=(0, 0, depth), end_position=(0.045, 0, depth))
        acts += r.run(0.5, thumb_to=None, position=(0.045, 0, depth))
        return sum(a.dx for a in acts if a.kind == "move")
    near, far = travel(0.4), travel(0.8)
    assert near < -100                                 # mirrored: camera-right = screen-left
    assert far == pytest.approx(near, rel=0.15)


def test_tracker_jitter_does_not_move_cursor():
    r = Rig()
    rng = np.random.default_rng(3)
    acts = []
    for _ in range(120):
        jitter = rng.normal(0, 0.0005, 3)               # ~0.5 mm tracker noise
        acts += r.run(1 / FPS, thumb_to=None, position=tuple(np.array([0, 0, 0.5]) + jitter))
    moved = sum(abs(a.dx) + abs(a.dy) for a in acts if a.kind == "move")
    assert moved < 15


@pytest.mark.parametrize("angles", [(0.0, 1.2, 0.0), (0.9, 0.0, 1.1), (2.5, -0.7, 0.4)])
def test_click_works_from_off_angles(angles):
    r = Rig(rot=synth.rotation(*angles))
    r.run(0.3, thumb_to=None)
    acts = r.run(0.12, thumb_to="index") + r.run(0.3, thumb_to=None)
    assert kinds(acts) == [("click", "left")]


def test_spread_palm_still_wakes_once():
    r = Rig()
    acts = r.run(2.0, curl=OPEN, spread_deg=40)
    assert kinds(acts) == [("wake", None)]


def test_relaxed_resting_hand_never_wakes():
    r = Rig()
    assert kinds(r.run(3.0, curl=OPEN)) == []


def test_low_confidence_frames_do_nothing():
    r = Rig()
    hand = synth.anatomical_hand(thumb_to="index")
    out = []
    for _ in range(40):
        r.t += 1 / FPS
        out += r.ctl.step(synth.frame(r.t, hand, score=0.3), r.t)
    assert kinds(out) == []


def test_paused_releases_and_ignores():
    r = Rig()
    r.run(0.3, thumb_to=None)
    r.run(0.5, thumb_to="index")
    r.gate.paused = True
    acts = r.run(0.2, thumb_to="index")
    assert kinds(acts) == [("up", "left")]
    assert kinds(r.run(0.5, thumb_to=None) + r.run(0.12, thumb_to="index")) == []


def test_tap_with_tracker_jitter_is_click_not_drag():
    # Live bug: summed frame-to-frame jitter crossed the drag-move threshold during real taps.
    r = Rig()
    rng = np.random.default_rng(5)
    r.run(0.3, thumb_to=None)
    acts = []
    for _ in range(round(0.25 * FPS)):
        jitter = tuple(np.array([0, 0, 0.5]) + rng.normal(0, 0.002, 3))
        acts += r.run(1 / FPS, thumb_to="index", position=jitter)
    acts += r.run(0.3, thumb_to=None)
    assert kinds(acts) == [("click", "left")]


def test_flickering_handedness_label_is_latched():
    # Live bug: MediaPipe flipped left/right 75 times on one hand, flipping scroll direction.
    r = Rig()
    hand = synth.anatomical_hand(curl={**POINT, "index": 0.15})
    out = []
    for i in range(60):
        r.t += 1 / FPS
        f = synth.frame(r.t, hand)
        if i % 3 == 2:  # every third frame mislabelled; the first frame is correct
            f = replace(f, handedness="left")
        out += r.ctl.step(f, r.t)
        assert r.ctl.last_features.handedness == "right"


def test_right_click_curl_does_not_scroll():
    r = Rig()
    r.run(0.3, curl={**POINT, "middle": 0.1}, thumb_to=None)
    acts = r.run(0.15, curl={**POINT, "middle": 0.6}, thumb_to="middle") + r.run(0.3, thumb_to=None)
    assert not [a for a in acts if a.kind == "scroll"]


def test_wake_disabled_never_fires():
    cfg = replace(CFG, wake=replace(CFG.wake, enabled=False))
    r = Rig(cfg)
    assert kinds(r.run(2.0, curl=OPEN, spread_deg=40)) == []


def test_cursor_leash_swallows_jitter_but_passes_real_motion():
    r = Rig()
    rng = np.random.default_rng(9)
    r.run(0.3, thumb_to=None)
    jit = []
    for _ in range(90):                                   # ~1.5 hand-% jitter, like the live log
        jit += r.run(1 / FPS, thumb_to=None, position=tuple(np.array([0, 0, 0.5]) + rng.normal(0, 0.0015, 3)))
    assert sum(abs(a.dx) + abs(a.dy) for a in jit if a.kind == "move") < 5
    acts = r.run(0.5, thumb_to=None, position=(0, 0, 0.5), end_position=(0.06, 0, 0.5))
    assert abs(sum(a.dx for a in acts if a.kind == "move")) > 100
