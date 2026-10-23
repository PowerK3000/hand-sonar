import numpy as np

from handctl.lift_detector import LiftDetector
from tests import synth

FPS = 25
BASE = {"index": 0.25, "middle": 0.25, "ring": 0.3, "pinky": 0.3}


def run(sequence, rot=None):
    """sequence: list of (seconds, curl overrides, position)."""
    det, out, t = LiftDetector(), [], 0.0
    for seconds, curl, pos in sequence:
        hand = synth.anatomical_hand(curl={**BASE, **curl})
        for _ in range(round(seconds * FPS)):
            t += 1 / FPS
            b = det.update(synth.frame(t, hand, rot=rot, position=pos))
            if b:
                out.append(b)
    return out


VIEW = synth.rotation(0.0, 0.0, -1.1)        # camera above and in front, like Kayden's setup
STILL = (0.0, 0.0, 0.5)


def flick(finger):
    return [(1.0, {}, STILL), (0.12, {finger: 0.75}, STILL), (0.12, {finger: 1.0}, STILL),
            (0.12, {finger: 0.6}, STILL), (0.8, {}, STILL)]


def test_index_lift_is_left_click():
    assert run(flick("index"), VIEW) == ["left"]


def test_middle_lift_is_right_click():
    assert run(flick("middle"), VIEW) == ["right"]


def test_resting_hand_never_clicks():
    assert run([(4.0, {}, STILL)], VIEW) == []


def test_whole_hand_moving_does_not_click():
    seq = [(1.0, {}, STILL)]
    for k in range(20):                       # sweeping the hand around like steering the cursor
        seq.append((0.05, {}, (0.03 * np.sin(k / 2), 0.02 * np.cos(k / 3), 0.5 - 0.01 * np.sin(k / 4))))
    assert run(seq, VIEW) == []
