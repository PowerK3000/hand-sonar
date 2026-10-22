import numpy as np
import pytest

from handctl import features
from handctl.hand import HandFrame
from tests import synth

RNG = np.random.default_rng(7)
ANGLES = [synth.rotation(*RNG.uniform(-np.pi, np.pi, 3)) for _ in range(25)]


@pytest.mark.parametrize("handedness", ["right", "left"])
def test_features_identical_from_any_view_angle(handedness):
    hand = synth.anatomical_hand(curl={"middle": 0.9, "ring": 0.9, "pinky": 0.9}, thumb_to="index", gap=0.01)
    ref = features.compute(synth.frame(0, hand, handedness=handedness))
    for rot in ANGLES:
        f = features.compute(synth.frame(0, hand, rot=rot, scale=1.15, handedness=handedness))
        assert f.pinch_index == pytest.approx(ref.pinch_index, abs=1e-9)
        assert f.pinch_middle == pytest.approx(ref.pinch_middle, abs=1e-9)
        for k in ref.straightness:
            assert f.straightness[k] == pytest.approx(ref.straightness[k], abs=1e-9)
        np.testing.assert_allclose(f.index_local, ref.index_local, atol=1e-9)


def test_left_and_right_hands_agree():
    hand = synth.anatomical_hand(curl={"index": 0.3})
    r = features.compute(synth.frame(0, hand, handedness="right"))
    l = features.compute(synth.frame(0, hand, handedness="left"))
    np.testing.assert_allclose(r.index_local, l.index_local, atol=1e-9)


def test_palm_frame_axes_are_anatomical():
    hand = synth.anatomical_hand(curl={"index": 0.0})
    f = features.compute(synth.frame(0, hand))
    # Straight index finger points forward (+y), sits on the thumb side (+x), in the palm plane.
    assert f.index_local[1] > 1.0
    assert f.index_local[0] > 0.1
    assert abs(f.index_local[2]) < 0.05
    curled = features.compute(synth.frame(0, synth.anatomical_hand(curl={"index": 1.0})))
    assert curled.index_local[2] < -0.3  # curls toward the palm side (-z)


def test_pinch_and_straightness_ranges():
    touching = features.compute(synth.frame(0, synth.anatomical_hand(thumb_to="index")))
    apart = features.compute(synth.frame(0, synth.anatomical_hand()))
    assert touching.pinch_index < 0.05
    assert apart.pinch_index > 0.4
    straight = features.compute(synth.frame(0, synth.anatomical_hand(curl={"ring": 0.0})))
    fist = features.compute(synth.frame(0, synth.anatomical_hand(curl={"ring": 1.0})))
    assert straight.straightness["ring"] > 0.95
    assert fist.straightness["ring"] < 0.7


def test_hand_px_tracks_distance_not_tilt():
    hand = synth.anatomical_hand()
    near = features.compute(synth.frame(0, hand, position=(0, 0, 0.4)))
    far = features.compute(synth.frame(0, hand, position=(0, 0, 0.8)))
    assert near.hand_px / far.hand_px == pytest.approx(2.0, rel=0.05)
    tilted = features.compute(synth.frame(0, hand, rot=synth.rotation(0, 0.6, 0.4), position=(0, 0, 0.4)))
    assert tilted.hand_px == pytest.approx(near.hand_px, rel=0.25)


def test_degenerate_hand_rejected():
    world = np.zeros((21, 3))
    frame = HandFrame(t=0, world=world, image=np.zeros((21, 2)), handedness="right", score=1)
    with pytest.raises(features.DegenerateHand):
        features.compute(frame)


def test_handframe_validates_shapes():
    with pytest.raises(ValueError):
        HandFrame(t=0, world=np.zeros((20, 3)), image=np.zeros((21, 2)), handedness="right", score=1)
    with pytest.raises(ValueError):
        HandFrame(t=0, world=np.zeros((21, 3)), image=np.zeros((21, 2)), handedness="up", score=1)
