"""Pick the tap model by held-out accuracy AND single-frame latency (live loop budget)."""
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from handctl import train_taps as tt
from tools.seq_experiment import events

CANDIDATES = {
    "hgb400": lambda: HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, class_weight="balanced", random_state=0),
    "hgb120": lambda: HistGradientBoostingClassifier(max_iter=120, learning_rate=0.12, max_depth=6, class_weight="balanced", random_state=0),
    "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000)),
    "rf100": lambda: RandomForestClassifier(n_estimators=100, max_depth=14, class_weight="balanced_subsample", n_jobs=-1, random_state=0),
    "et200": lambda: ExtraTreesClassifier(n_estimators=200, max_depth=16, class_weight="balanced_subsample", n_jobs=-1, random_state=0),
    "mlp": lambda: make_pipeline(StandardScaler(), MLPClassifier((64,), alpha=1e-2, max_iter=600, early_stopping=True, random_state=0)),
}


def split(steps, part, frac=0.6):
    out = []
    for label, tw, F, y, taps in steps:
        cut = tw[0] + frac * (tw[-1] - tw[0])
        m = tw < cut if part == "train" else tw >= cut + 0.5
        out.append((label, tw[m], F[m], y[m], [x for x in taps if (x < cut if part == "train" else x >= cut + 0.5)]))
    return out


def score(model, test, thr_l, thr_r):
    hits = {1: 0, 2: 0}; total = {1: 0, 2: 0}; extra = 0; neg = 0; neg_s = 0.0
    for label, tw, F, y, taps in test:
        if not len(F):
            continue
        P = model.predict_proba(F)
        cls = tt.TAP_CLASSES.get(label, 0)
        for c, thr in ((1, thr_l), (2, thr_r)):
            used = set()
            for e in events(tw, P[:, c], thr, gap=0.6):
                j = [i for i, x in enumerate(taps) if abs(e - x) < 0.4 and i not in used]
                if cls == c and j:
                    used.add(j[0]); hits[c] += 1
                elif label in tt.TAP_CLASSES:
                    extra += 1
                else:
                    neg += 1
        if cls:
            total[cls] += len(taps)
        else:
            neg_s += tw[-1] - tw[0]
    return hits, total, extra, neg, neg_s


def main(base):
    steps = tt.build([Path(base)])
    train, test = split(steps, "train"), split(steps, "test")
    Xtr = np.vstack([s[2] for s in train]); ytr = np.concatenate([s[3] for s in train])
    probe = test[0][2][:1]
    for name, make in CANDIDATES.items():
        m = make().fit(Xtr, ytr)
        with threadpool_limits(1):
            m.predict_proba(probe); t0 = time.perf_counter()
            for _ in range(50):
                m.predict_proba(probe)
            ms = (time.perf_counter() - t0) / 50 * 1000
        best = None
        for tl in (0.5, 0.7, 0.85):
            for tr in (0.4, 0.5, 0.7):
                h, tot, extra, neg, neg_s = score(m, test, tl, tr)
                s = h[1] + h[2] - 2 * (extra + neg)
                if best is None or s > best[0]:
                    best = (s, tl, tr, h, tot, extra, neg, neg_s)
        s, tl, tr, h, tot, extra, neg, neg_s = best
        print(f"{name:7} {ms:6.2f} ms/frame | best thr L{tl} R{tr}: left {h[1]}/{tot[1]} right {h[2]}/{tot[2]} "
              f"extra {extra} false-in-nongesture {neg} ({neg / max(neg_s, 1) * 60:.1f}/min)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/recordings/t1")
