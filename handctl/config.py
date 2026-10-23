"""Typed, validated view of config/gestures.toml.

Validation happens once at load so a typo fails loudly at startup instead of
silently producing a controller that never clicks.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class AppCfg:
    camera_index: int
    model_path: str
    hand: str
    num_hands: int
    process_width: int
    pause_hotkey: str


@dataclass(frozen=True)
class TapsCfg:
    mode: str
    model_path: str
    left_threshold: float
    right_threshold: float
    refractory_s: float
    rewind_s: float


@dataclass(frozen=True)
class EngagementCfg:
    disengage_on_keyboard: bool
    disengage_on_mouse: bool
    reengage_after_ms: float
    min_confidence: float


@dataclass(frozen=True)
class FilterCfg:
    min_cutoff_hz: float
    beta: float
    d_cutoff_hz: float
    palm_min_cutoff_hz: float
    palm_beta: float


@dataclass(frozen=True)
class PosesCfg:
    extended_min: float
    curled_max: float


@dataclass(frozen=True)
class PointerCfg:
    smooth_ms: float
    horizontal_source: str
    horizontal_sign: float
    vertical_source: str
    vertical_sign: float
    height_gain: float
    pixels_per_hand: float
    acceleration: float
    accel_ref_hands_per_s: float
    deadzone: float


@dataclass(frozen=True)
class PinchCfg:
    metric: str               # "2d": thumb-finger gap in the image; "3d": MediaPipe world (overhead-camera era)
    pinch_close: float
    pinch_open: float
    arm_s: float              # fingers must be clearly apart this long before a touch counts
    snap_s: float             # ...and close within this time of last being apart (a tap snaps; a resting thumb drifts)
    still_max: float          # hand speed (hand lengths/s) above which new touches are ignored


@dataclass(frozen=True)
class DragCfg:
    start_move: float
    static_hold_max_s: float


@dataclass(frozen=True)
class ScrollCfg:
    hold_ms: float
    natural: bool
    vertical_axis: str
    horizontal: bool
    wheel_per_hand: float
    momentum: bool
    momentum_decay_per_s: float
    momentum_min_wheel_per_s: float


@dataclass(frozen=True)
class ClutchCfg:
    enabled: bool
    max_straightness: float
    release_straightness: float
    release_s: float
    rewind_s: float


@dataclass(frozen=True)
class WakeCfg:
    enabled: bool
    hold_ms: float
    max_speed_hands_per_s: float
    min_spread_deg: float


@dataclass(frozen=True)
class Config:
    app: AppCfg
    taps: TapsCfg
    engagement: EngagementCfg
    filter: FilterCfg
    poses: PosesCfg
    pointer: PointerCfg
    click: PinchCfg
    right_click: PinchCfg
    drag: DragCfg
    scroll: ScrollCfg
    clutch: ClutchCfg
    wake: WakeCfg


def _section(raw: dict, path: str) -> dict:
    node = raw
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            raise ConfigError(f"missing section [{path}]")
        node = node[key]
    return node


def _build(cls, raw: dict, path: str, drop: tuple[str, ...] = ("action", "pose")):
    sec = {k: v for k, v in _section(raw, path).items() if k not in drop}
    fields = cls.__dataclass_fields__
    unknown = set(sec) - set(fields)
    missing = set(fields) - set(sec)
    if unknown:
        raise ConfigError(f"[{path}] unknown keys: {sorted(unknown)}")
    if missing:
        raise ConfigError(f"[{path}] missing keys: {sorted(missing)}")
    for name, f in fields.items():
        want = {"float": (int, float), "int": int, "bool": bool, "str": str}[f.type]
        val = sec[name]
        # bool is a subclass of int; reject it where a number is expected.
        if (f.type in ("float", "int") and isinstance(val, bool)) or not isinstance(val, want):
            raise ConfigError(f"[{path}] {name} must be {f.type}, got {val!r}")
        if f.type == "float":
            sec[name] = float(val)
    return cls(**sec)


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def validate(c: Config) -> Config:
    _check(c.app.hand in ("left", "right", "any"), "[app] hand must be left/right/any")
    _check(c.app.camera_index >= 0, "[app] camera_index must be >= 0")
    _check(c.app.num_hands in (1, 2), "[app] num_hands must be 1 or 2")
    _check(160 <= c.app.process_width <= 1920, "[app] process_width must be in [160, 1920]")
    _check(c.taps.mode in ("lift", "model", "pinch"), "[taps] mode must be lift, model or pinch")
    _check(0 < c.taps.left_threshold < 1 and 0 < c.taps.right_threshold < 1, "[taps] thresholds must be in (0,1)")
    _check(c.taps.refractory_s >= 0 and 0 <= c.taps.rewind_s <= 1, "[taps] refractory_s >= 0, rewind_s in [0,1]")
    _check(0 <= c.engagement.min_confidence <= 1, "[engagement] min_confidence must be in [0,1]")
    _check(c.engagement.reengage_after_ms >= 0, "[engagement] reengage_after_ms must be >= 0")
    _check(c.filter.min_cutoff_hz > 0 and c.filter.d_cutoff_hz > 0 and c.filter.beta >= 0,
           "[filter] cutoffs must be > 0 and beta >= 0")
    _check(0 < c.poses.curled_max < c.poses.extended_min <= 1,
           "[poses] need 0 < curled_max < extended_min <= 1")
    for axis in ("horizontal", "vertical"):
        _check(getattr(c.pointer, f"{axis}_source") in ("image_x", "image_y", "height"),
               f"[pointer] {axis}_source must be image_x, image_y or height")
        _check(getattr(c.pointer, f"{axis}_sign") in (-1.0, 1.0), f"[pointer] {axis}_sign must be 1 or -1")
    _check(c.pointer.horizontal_source != c.pointer.vertical_source, "[pointer] axes need different sources")
    _check(c.pointer.height_gain > 0, "[pointer] height_gain must be > 0")
    _check(c.pointer.pixels_per_hand > 0 and c.pointer.acceleration >= 1
           and c.pointer.accel_ref_hands_per_s > 0 and c.pointer.deadzone >= 0,
           "[pointer] values out of range")
    for name, p in (("click", c.click), ("right_click", c.right_click)):
        _check(0 < p.pinch_close < p.pinch_open,
               f"[gestures.{name}] need 0 < pinch_close < pinch_open (hysteresis)")
        _check(p.metric in ("2d", "3d"), f"[gestures.{name}] metric must be 2d or 3d")
        _check(p.arm_s >= 0 and p.snap_s > 0 and p.still_max > 0, f"[gestures.{name}] timing values out of range")
    _check(0 < c.clutch.max_straightness < c.clutch.release_straightness < 1,
           "[gestures.clutch] need 0 < max_straightness < release_straightness < 1")
    _check(c.clutch.release_s >= 0 and 0 <= c.clutch.rewind_s <= 1, "[gestures.clutch] timing out of range")
    _check(c.drag.start_move > 0 and c.drag.static_hold_max_s > 0, "[gestures.drag] values must be > 0")
    _check(c.scroll.vertical_axis in ("palm_forward", "palm_normal"),
           "[gestures.scroll] vertical_axis must be palm_forward or palm_normal")
    _check(c.scroll.wheel_per_hand > 0, "[gestures.scroll] wheel_per_hand must be > 0")
    _check(0 < c.scroll.momentum_decay_per_s < 1, "[gestures.scroll] momentum_decay_per_s must be in (0,1)")
    _check(c.wake.hold_ms > 0 and c.wake.max_speed_hands_per_s > 0 and 0 < c.wake.min_spread_deg < 90,
           "[gestures.wake] values out of range")
    return c


def parse(raw: dict) -> Config:
    return validate(Config(
        app=_build(AppCfg, raw, "app"),
        taps=_build(TapsCfg, raw, "taps", drop=()),
        engagement=_build(EngagementCfg, raw, "engagement"),
        filter=_build(FilterCfg, raw, "filter"),
        poses=_build(PosesCfg, raw, "poses"),
        pointer=_build(PointerCfg, raw, "pointer"),
        click=_build(PinchCfg, raw, "gestures.click"),
        right_click=_build(PinchCfg, raw, "gestures.right_click"),
        drag=_build(DragCfg, raw, "gestures.drag"),
        scroll=_build(ScrollCfg, raw, "gestures.scroll"),
        clutch=_build(ClutchCfg, raw, "gestures.clutch"),
        wake=_build(WakeCfg, raw, "gestures.wake"),
    ))


def load(path: Path) -> Config:
    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except FileNotFoundError as e:
        raise ConfigError(f"config file not found: {path}") from e
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: {e}") from e
    return parse(raw)
