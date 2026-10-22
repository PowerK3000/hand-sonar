from dataclasses import replace
from pathlib import Path

import numpy as np

from handctl import config
from handctl.controller import Controller
from handctl.engagement import EngagementGate
from handctl.pose_features import FEATURE_VERSION, WINDOW_FRAMES
from handctl.tap_model import TapDetector, TapModelInfo
from tests import synth

CFG = config.load(Path(__file__).resolve().parents[1] / "config" / "gestures.toml")
INFO = TapModelInfo(FEATURE_VERSION, WINDOW_FRAMES, ["test"], 1, 1)
WARM = 10  # at 25 fps the first full 0.4 s window ends on frame index 10


class ScriptedModel:
    """predict_proba returns the next row of a script (none, left, right)."""

    def __init__(self, rows):
        self.rows = list(rows)

    def predict_proba(self, x):
        return np.array([self.rows.pop(0) if self.rows else [1.0, 0.0, 0.0]])


def frames(n, start=0.0, fps=25, position=(0, 0, 0.5)):
    hand = synth.anatomical_hand()
    return [synth.frame(start + i / fps, hand, position=position) for i in range(n)]


def test_detector_fires_once_per_rising_edge_with_refractory():
    script = [[1, 0, 0]] * 3 + [[0.1, 0.9, 0]] * 4 + [[1, 0, 0]] * 3 + [[0.1, 0.9, 0]] * 2
    det = TapDetector(ScriptedModel(script), INFO, 0.7, 0.5, refractory_s=0.6)
    fired = [det.update(f) for f in frames(WARM + len(script))]
    hits = [x for x in fired if x]
    assert hits == ["left"]                     # second burst is inside the 0.6 s refractory


def test_detector_per_button_thresholds():
    script = [[0.4, 0.0, 0.6]]
    det = TapDetector(ScriptedModel(script), INFO, 0.7, 0.5, refractory_s=0.6)
    fired = [det.update(f) for f in frames(WARM + 1)]
    assert fired[-1] == "right"


def test_controller_model_mode_clicks_and_rewinds_drift():
    cfg = replace(CFG, taps=replace(CFG.taps, rewind_s=0.3))
    script = [[1, 0, 0]] * 20 + [[0, 1, 0]]
    det = TapDetector(ScriptedModel(script), INFO, 0.7, 0.5, refractory_s=0.6)
    ctl = Controller(cfg, EngagementGate(cfg.engagement), det)
    out = []
    hand = synth.anatomical_hand()
    for i in range(WARM + 21):
        t = i / 25
        x = 0.0 if i < 25 else 0.003 * (i - 24)            # drift in the 0.2 s before recognition
        out += ctl.step(synth.frame(t, hand, position=(x, 0, 0.5)), t)
    clicks = [i for i, a in enumerate(out) if a.kind == "click"]
    assert [out[i].button for i in clicks] == ["left"]
    moved_before = sum(a.dx for a in out[:clicks[0]] if a.kind == "move")
    assert abs(moved_before) < 1.0                          # drift undone before the click


def test_model_mode_requires_detector():
    import pytest
    with pytest.raises(ValueError):
        Controller(CFG, EngagementGate(CFG.engagement))
