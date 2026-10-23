import copy
import tomllib
from pathlib import Path

import numpy as np
import pytest

from handctl import config
from handctl.one_euro import OneEuroFilter

CFG_PATH = Path(__file__).resolve().parents[1] / "config" / "gestures.toml"


def test_one_euro_suppresses_jitter_at_rest():
    f = OneEuroFilter(1.2, 1.0, 1.0)
    rng = np.random.default_rng(0)
    noisy = rng.normal(0, 0.01, 300)
    out = np.array([f(np.array([x]), i / 60)[0] for i, x in enumerate(noisy)])
    assert out[60:].std() < noisy[60:].std() / 3


def test_one_euro_follows_fast_motion_with_little_lag():
    f = OneEuroFilter(1.2, 1.0, 1.0)
    t = np.arange(120) / 60
    ramp = 3.0 * t                                   # 3 hand lengths/s
    out = np.array([f(np.array([x]), ti)[0] for x, ti in zip(ramp, t)])
    lag_s = (ramp[-1] - out[-1]) / 3.0
    assert lag_s < 0.05


def test_one_euro_rejects_bad_params_and_handles_duplicate_time():
    with pytest.raises(ValueError):
        OneEuroFilter(0, 1, 1)
    f = OneEuroFilter(1, 1, 1)
    f(np.array([1.0]), 0.0)
    assert f(np.array([5.0]), 0.0)[0] == 1.0         # same timestamp: no division by zero


def _raw():
    with open(CFG_PATH, "rb") as fh:
        return tomllib.load(fh)


def test_shipped_config_is_valid():
    config.load(CFG_PATH)


@pytest.mark.parametrize("mutate, msg", [
    (lambda r: r["gestures"]["click"].update(pinch_close=0.3), "hysteresis"),
    (lambda r: r["gestures"]["scroll"].update(vertical_axis="sideways"), "vertical_axis"),
    (lambda r: r["pointer"].update(typo_key=1), "unknown keys"),
    (lambda r: r["app"].pop("camera_index"), "missing keys"),
    (lambda r: r["pointer"].update(horizontal_sign=2.0), "must be 1 or -1"),
    (lambda r: r["engagement"].update(disengage_on_keyboard="yes"), "must be bool"),
    (lambda r: r["pointer"].update(acceleration=True), "must be float"),
    (lambda r: r.pop("engagement"), "missing section"),
])
def test_bad_config_fails_loudly(mutate, msg):
    raw = copy.deepcopy(_raw())
    mutate(raw)
    with pytest.raises(config.ConfigError, match=msg):
        config.parse(raw)


def test_missing_file_is_config_error(tmp_path):
    with pytest.raises(config.ConfigError, match="not found"):
        config.load(tmp_path / "nope.toml")
