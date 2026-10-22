from dataclasses import replace

from handctl import config, paths, replay
from handctl.record import Step, _save
from tests import synth

FPS = 60


def _record(tmp_path, label, script, beats):
    """script: list of (seconds, hand_kwargs or None)."""
    rows, t = [], 0.0
    for seconds, kw in script:
        anat = None if kw is None else synth.anatomical_hand(**kw)
        for _ in range(round(seconds * FPS)):
            t += 1 / FPS
            rows.append((t, None if anat is None else synth.frame(t, anat)))
    path = tmp_path / f"00_{label}.npz"
    _save(path, Step(label, t, "test", 1.5), rows, beats, {"position": "synthetic"})
    return path


def test_replay_counts_hits_and_false_triggers(tmp_path):
    cfg = config.load(paths.CONFIG)
    cfg = replace(cfg, taps=replace(cfg.taps, mode="pinch"))
    script, beats = [(0.5, {})], []
    for k in range(4):
        beats.append(0.5 + k * 1.5)
        script += [(0.12, {"thumb_to": "index"}), (1.38, {})]
    path = _record(tmp_path, "click", script, beats)
    r = replay.replay_file(path, cfg)
    assert (r["hits"], r["beats"], r["false"]) == (4, 4, {})
    assert r["visible"] == 1.0


def test_replay_flags_clicks_during_non_command_step(tmp_path):
    cfg = config.load(paths.CONFIG)
    cfg = replace(cfg, taps=replace(cfg.taps, mode="pinch"))
    path = _record(tmp_path, "rest", [(0.5, {}), (0.12, {"thumb_to": "index"}), (1.0, {}), (0.5, None)], [])
    r = replay.replay_file(path, cfg)
    assert r["false"] == {("click", "left"): 1}
    assert r["visible"] < 1.0
