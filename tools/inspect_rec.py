"""Print per-beat feature traces for a recording step (tuning aid)."""
import sys
from pathlib import Path

import numpy as np

from handctl import features
from handctl.controller import HandednessLatch
from handctl.hand import HandFrame


def load(path):
    d = dict(np.load(path, allow_pickle=False))  # load once: indexing an NpzFile re-decompresses the whole array
    latch = HandednessLatch()
    out = []
    for i, t in enumerate(d["t"]):
        if not d["present"][i]:
            out.append((float(t), None)); continue
        fr = HandFrame(t=float(t), world=d["world"][i], image=d["image"][i],
                       handedness=latch(str(d["handedness"][i])), score=float(d["score"][i]))
        try:
            out.append((float(t), features.compute(fr)))
        except features.DegenerateHand:
            out.append((float(t), None))
    return d, out


def main(path, keys):
    d, rows = load(path)
    t0 = rows[0][0]
    beats = list(d["beats"])
    print(path.name, "beats at", [round(b - t0, 1) for b in beats])
    for t, f in rows:
        if f is None:
            print(f"{t - t0:5.2f}  --"); continue
        mark = " <BEAT" if any(0 <= t - b < 1 / 25 for b in beats) else ""
        vals = []
        for k in keys:
            if k.startswith("s_"):
                vals.append(f"{k}={f.straightness[k[2:]]:.2f}")
            elif k == "idx":
                vals.append("idx=" + ",".join(f"{v:+.2f}" for v in f.index_local))
            else:
                vals.append(f"{k}={getattr(f, k):.2f}")
        print(f"{t - t0:5.2f}  " + " ".join(vals) + mark)


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2].split(","))
