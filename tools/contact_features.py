"""Compare candidate features at real tap moments vs incidental thumb-near-finger moments."""
import sys
from pathlib import Path

import numpy as np

from tools.inspect_rec import load
from handctl.hand import (THUMB_TIP, THUMB_IP, INDEX_TIP, INDEX_DIP, INDEX_PIP, MIDDLE_TIP, MIDDLE_DIP)


def feats(d, i, f):
    w, im = d["world"][i], d["image"][i]
    s3 = lambda a, b: np.linalg.norm(w[a] - w[b]) / f.size_m
    s2 = lambda a, b: np.linalg.norm(im[a] - im[b]) / f.hand_px
    return {
        "ti3": s3(THUMB_TIP, INDEX_TIP), "ti2": s2(THUMB_TIP, INDEX_TIP),
        "tm3": s3(THUMB_TIP, MIDDLE_TIP), "tm2": s2(THUMB_TIP, MIDDLE_TIP),
        "tidip3": s3(THUMB_TIP, INDEX_DIP), "tipip3": s3(THUMB_TIP, INDEX_PIP),
        "tmdip3": s3(THUMB_TIP, MIDDLE_DIP),
        "im3": s3(INDEX_TIP, MIDDLE_TIP),
        "s_idx": f.straightness["index"], "s_mid": f.straightness["middle"],
        "s_ring": f.straightness["ring"],
    }


def contact_frames(path, key):
    """Per beat: the frame with the smallest `key` distance within 0..0.9 s after the beat."""
    d, rows = load(path)
    out = []
    for b in d["beats"]:
        best = None
        for i, (t, f) in enumerate(rows):
            if f is None or not (b <= t < b + 0.9):
                continue
            x = feats(d, i, f)
            if best is None or x[key] < best[key]:
                best = x
        if best:
            out.append(best)
    return out


def incidental(path, thresh=0.12):
    d, rows = load(path)
    out = []
    for i, (t, f) in enumerate(rows):
        if f is None:
            continue
        x = feats(d, i, f)
        if min(x["ti3"], x["tm3"]) < thresh or min(x["ti2"], x["tm2"]) < 0.05:
            out.append(x)
    return out


def summarize(name, xs):
    if not xs:
        print(f"{name:<22} (none)"); return
    keys = xs[0].keys()
    print(f"{name:<22} n={len(xs):<4}" + " ".join(f"{k}={np.median([x[k] for x in xs]):.2f}" for k in keys))


if __name__ == "__main__":
    base = Path(sys.argv[1])
    summarize("CLICK taps", contact_frames(base / "04_click.npz", "ti2"))
    summarize("RIGHT taps", contact_frames(base / "05_right_click.npz", "tm2"))
    summarize("DRAG pinches", contact_frames(base / "06_drag.npz", "ti2"))
    for nm in ("00_rest", "02_mouse", "03_cursor", "07_scroll_up", "08_scroll_down", "09_flick", "11_face", "01_typing"):
        summarize(f"incidental {nm[3:]}", incidental(base / f"{nm}.npz"))
