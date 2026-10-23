import time

import numpy as np
import pytest
from scipy.optimize import least_squares

from handctl.handmodel import skeleton as sk
from handctl.handmodel.fitter import Camera, PoseFitter

SHAPE = sk.default_shape()
CAM = Camera()
RNG = np.random.default_rng(11)


def random_pose(rng=RNG, distance=0.5):
    lo, hi = sk.bounds()
    p = sk.rest_pose(distance)
    for sl in list(sk.FINGER_SLICES.values()) + [sk.SL_THUMB]:
        p[sl] = rng.uniform(lo[sl] * 0.7, hi[sl] * 0.7)
    p[sk.SL_ROT] = rng.normal(0, 0.6, 3)
    p[sk.SL_TRANS] = [rng.uniform(-0.1, 0.1), rng.uniform(-0.05, 0.05), distance]
    return p


def test_bone_lengths_preserved_for_any_pose():
    poses = np.stack([random_pose() for _ in range(50)])
    J = sk.camera_joints(SHAPE, poses)
    for i, root in enumerate(sk.FINGER_ROOTS):
        for k in range(3):
            d = np.linalg.norm(J[:, root + k + 1] - J[:, root + k], axis=1)
            np.testing.assert_allclose(d, SHAPE.finger_bones[i, k], rtol=1e-9)
    chain = [1, 2, 3, 4]
    for k in range(3):
        d = np.linalg.norm(J[:, chain[k + 1]] - J[:, chain[k]], axis=1)
        np.testing.assert_allclose(d, SHAPE.thumb_bones[k], rtol=1e-9)


def test_batched_fk_matches_single():
    poses = np.stack([random_pose() for _ in range(5)])
    batch = sk.camera_joints(SHAPE, poses)
    for i in range(5):
        np.testing.assert_allclose(batch[i], sk.camera_joints(SHAPE, poses[i])[0])


def test_flexion_curls_toward_palm():
    p = sk.rest_pose()
    p[sk.FINGER_SLICES["index"]] = [0, 0, 0, 0]
    straight = sk.local_joints(SHAPE, p)[0]
    p[sk.FINGER_SLICES["index"]] = [0, 1.2, 1.4, 0.9]
    curled = sk.local_joints(SHAPE, p)[0]
    assert straight[8, 1] > curled[8, 1]          # tip pulled back
    assert curled[8, 2] < -0.02                   # toward the palm side


@pytest.mark.parametrize("target", [8, 12])
def test_thumb_can_reach_index_and_middle_tips_within_limits(target):
    """A pinch must be representable, or the fitter can never explain one."""
    lo, hi = sk.bounds()

    def gap(x):
        p = sk.rest_pose(); p[sk.SL_THUMB] = x[:5]
        fs = sk.FINGER_SLICES["index" if target == 8 else "middle"]
        p[fs] = x[5:9]
        J = sk.local_joints(SHAPE, p)[0]
        return J[4] - J[target]
    x0 = np.concatenate([sk.rest_pose()[sk.SL_THUMB], [0, 0.6, 0.6, 0.3]])
    fs = sk.FINGER_SLICES["index"]
    b = (np.concatenate([lo[sk.SL_THUMB], lo[fs]]), np.concatenate([hi[sk.SL_THUMB], hi[fs]]))
    res = least_squares(gap, x0, bounds=b)
    assert np.linalg.norm(res.fun) < 0.004        # within 4 mm


def test_fitter_recovers_pose_from_noisy_2d():
    true = random_pose(np.random.default_rng(3))
    J = sk.camera_joints(SHAPE, true)[0]
    rng = np.random.default_rng(4)
    obs = CAM.project(J) + rng.normal(0, 2.0, (21, 2))
    mp3d = (J - J.mean(0)) + rng.normal(0, 0.012, (21, 3))           # noisy "MediaPipe" 3D
    fitter = PoseFitter(SHAPE, CAM)
    for _ in range(4):                                                  # a few frames of the same pose
        r = fitter.fit(obs, mp3d)
    err = np.linalg.norm((r.joints - r.joints.mean(0)) - (J - J.mean(0)), axis=1)
    assert r.rms_px < 4.0
    assert np.median(err) < 0.008


def _pinch_pose(rot):
    """A real thumb-index contact, solved within joint limits."""
    lo, hi = sk.bounds()
    base = sk.rest_pose(); base[sk.SL_ROT] = rot
    fs = sk.FINGER_SLICES["index"]

    def gap(x):
        p = base.copy(); p[sk.SL_THUMB] = x[:5]; p[fs] = x[5:]
        J = sk.local_joints(SHAPE, p)[0]
        return J[4] - J[8]
    x0 = np.concatenate([base[sk.SL_THUMB], [0, 0.6, 0.6, 0.3]])
    b = (np.concatenate([lo[sk.SL_THUMB], lo[fs]]), np.concatenate([hi[sk.SL_THUMB], hi[fs]]))
    x = least_squares(gap, x0, bounds=b).x
    base[sk.SL_THUMB], base[fs] = x[:5], x[5:]
    return base


def test_model_beats_raw_3d_when_thumb_tip_is_badly_guessed():
    """The failure seen live: 2D shows contact, the generic 3D guess puts the thumb tip 4 cm away."""
    p = _pinch_pose([1.2, 0.0, 0.0])                # camera looking down at the back of the hand
    J = sk.camera_joints(SHAPE, p)[0]
    true_gap = np.linalg.norm(J[4] - J[8])
    assert true_gap < 0.005
    rng = np.random.default_rng(5)
    obs = CAM.project(J) + rng.normal(0, 1.5, (21, 2))
    mp3d = J - J.mean(0) + rng.normal(0, 0.004, (21, 3))
    mp3d[4] += np.array([0.0, 0.0, 0.04])            # generic model's depth error on the thumb tip
    raw_gap = np.linalg.norm(mp3d[4] - mp3d[8])
    fitter = PoseFitter(SHAPE, CAM)
    for _ in range(4):
        r = fitter.fit(obs, mp3d)
    fit_gap = np.linalg.norm(r.joints[4] - r.joints[8])
    print(f"true {true_gap*1000:.1f} mm, MediaPipe-style {raw_gap*1000:.1f} mm, model {fit_gap*1000:.1f} mm")
    assert raw_gap > 0.03
    assert fit_gap < 0.015


def test_tracking_speed_budget():
    true = random_pose(np.random.default_rng(8))
    J = sk.camera_joints(SHAPE, true)[0]
    obs, mp3d = CAM.project(J), J - J.mean(0)
    fitter = PoseFitter(SHAPE, CAM)
    fitter.fit(obs, mp3d)
    t0 = time.perf_counter()
    for _ in range(20):
        fitter.fit(obs + RNG.normal(0, 1, obs.shape), mp3d)
    per_frame_ms = (time.perf_counter() - t0) / 20 * 1000
    print(f"fit {per_frame_ms:.1f} ms/frame")
    assert per_frame_ms < 40


def test_calibration_learns_true_proportions_from_generic_guesses():
    """2D comes from Kayden's true hand; the '3D guesses' come from a generic hand (like
    MediaPipe). Calibration must move the proportions toward the truth."""
    from handctl.handmodel import calibrate
    true = sk.HandShape(palm=SHAPE.palm * 1.0,
                        finger_bones=SHAPE.finger_bones * np.array([[1.18], [1.0], [0.9], [1.1]]),
                        finger_splay=SHAPE.finger_splay,
                        thumb_bones=SHAPE.thumb_bones * np.array([1.15, 0.9, 1.1]))
    rng = np.random.default_rng(21)
    frames = []
    for _ in range(80):
        pose = random_pose(rng)
        J_true = sk.camera_joints(true, pose)[0]
        J_gen = sk.camera_joints(SHAPE, pose)[0]
        obs = CAM.project(J_true) + rng.normal(0, 1.5, (21, 2))
        mp3d = (J_gen - J_gen.mean(0)) + rng.normal(0, 0.003, (21, 3))
        frames.append((mp3d, obs, "right"))
    s0 = calibrate.initial_shape(frames)
    s1 = calibrate.refine(s0, frames, CAM, rounds=2, n=60, log=lambda *_: None)

    def ratio_err(s):
        b = np.concatenate([s.finger_bones.ravel(), s.thumb_bones])
        t = np.concatenate([true.finger_bones.ravel(), true.thumb_bones])
        return np.abs(b / b.sum() - t / t.sum()).sum()
    print(f"ratio error before {ratio_err(s0):.4f} after {ratio_err(s1):.4f}")
    assert ratio_err(s1) < 0.6 * ratio_err(s0)
