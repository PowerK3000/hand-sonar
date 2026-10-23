"""Score the controller against labelled recordings (offline tuning, no camera needed).

    python -m handctl.replay data/recordings/session1 [--config other.toml]

For beat-driven steps, an action counts as a hit if it lands within a window after a beat;
everything else is a false trigger. Non-command steps (rest, typing, mouse, cursor, face) should
produce no clicks/drags/scrolls at all.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from . import config, paths
from .controller import Controller
from .engagement import EngagementGate
from .hand import HandFrame
from .tap_model import from_config as taps_from_config

EXPECT = {"click": ("click", "left"), "right_click": ("click", "right"), "drag": ("down", "left"),
          "scroll_up": ("scroll", "down"), "scroll_down": ("scroll", "up"),  # natural: finger up = page down
          "flick": ("scroll", None), "wake": ("wake", None)}
HIT_WINDOW_S = 1.2


def _event(a) -> tuple[str, str | None] | None:
    if a.kind == "scroll":
        if abs(a.dy) < 1e-9:
            return None
        return ("scroll", "down" if a.dy < 0 else "up")
    if a.kind in ("click", "down", "wake"):
        return (a.kind, a.button)
    return None


def replay_file(path: Path, cfg: config.Config) -> dict:
    d = dict(np.load(path, allow_pickle=False))  # load once: indexing an NpzFile re-decompresses the whole array
    label = str(d["label"])
    gate = EngagementGate(cfg.engagement)
    ctl = Controller(cfg, gate, taps_from_config(cfg))
    events: list[tuple[float, tuple]] = []
    last_scroll: tuple[float, tuple] | None = None
    for i, t in enumerate(d["t"]):
        t = float(t)
        frame = None
        if d["present"][i]:
            frame = HandFrame(t=t, world=d["world"][i], image=d["image"][i],
                              handedness=str(d["handedness"][i]), score=float(d["score"][i]))
        for a in ctl.step(frame, t):
            ev = _event(a)
            if ev is None:
                continue
            if ev[0] == "scroll":
                # A continuous scroll is one event per burst (gaps < 0.3 s join the burst).
                burst = last_scroll is not None and last_scroll[1] == ev and t - last_scroll[0] < 0.3
                last_scroll = (t, ev)
                if burst:
                    continue
            events.append((t, ev))
    beats = list(d["beats"])
    want = EXPECT.get(label)
    hits = 0
    false = Counter()
    used: set[int] = set()
    for t, ev in events:
        match = want is not None and (ev == want or (want[1] is None and ev[0] == want[0]))
        if match:
            for bi, b in enumerate(beats):
                if bi not in used and 0 <= t - b <= HIT_WINDOW_S:
                    used.add(bi)
                    hits += 1
                    break
            else:
                false[ev] += 1
        else:
            false[ev] += 1
    return {"file": path.name, "label": label, "beats": len(beats), "hits": hits,
            "false": dict(false), "frames": len(d["t"]), "visible": float(np.mean(d["present"]))}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recording", type=Path, help="a session directory or a single .npz")
    p.add_argument("--config", type=Path, default=paths.CONFIG)
    args = p.parse_args(argv)
    try:
        cfg = config.load(args.config)
    except config.ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    files = sorted(args.recording.glob("*.npz")) if args.recording.is_dir() else [args.recording]
    files = [f for f in files if not f.name.endswith(".tmp.npz")]
    if not files:
        print(f"no recordings in {args.recording}", file=sys.stderr)
        return 2
    print(f"{'step':<12} {'visible':>7} {'hits':>9} {'false triggers'}")
    for f in files:
        r = replay_file(f, cfg)
        hits = f"{r['hits']}/{r['beats']}" if r["beats"] else "-"
        false = ", ".join(f"{k[0]}{'-' + k[1] if k[1] else ''} x{v}" for k, v in r["false"].items()) or "none"
        print(f"{r['label']:<12} {r['visible']:>7.0%} {hits:>9} {false}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
