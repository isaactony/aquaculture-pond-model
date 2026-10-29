"""
Compare feature sets on three things: test-like CV, how easily train and test can be told apart (adversarial
AUC), and what each model says about the test set (share called ponds, share of uncertain predictions).
Similar CV scores with different test pond shares is the sign that the train/test shift changes decisions.

    ./run.sh src/compare_features.py
"""
import os
import sys

import lightgbm as lgb
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from adversarial import adv_auc
from cv import LGB, N_ROUNDS, load, masked_copies, run
from features import OPTICAL, build, cube, masks

def is_summary(c):
    return "__" in c or c.startswith(("n_", "frac_", "first"))

def main():
    tr, te = load()
    x, y = cube(tr), tr["label"].to_numpy()
    ts, to = masks(te)
    Xte = build(cube(te)); allc = list(Xte.columns)
    raw_opt = [c for c in allc if c.split("_")[0] in OPTICAL]
    sets = {
        "all": allc,
        "no raw optical": [c for c in allc if c not in raw_opt],
        "summary stats (v1)": [c for c in allc if is_summary(c)],
        "index + SAR summaries": [c for c in allc if c not in raw_opt and is_summary(c)],
    }
    xm, _, _ = masked_copies(x, y, ts, to, 1, np.random.default_rng(0))
    Xm = build(xm)
    xa, ya, _ = masked_copies(x, y, ts, to, 10, np.random.default_rng(42))
    Xa = build(xa)
    for name, cols in sets.items():
        res, _ = run("aug", x, y, ts, to, cols=cols, verbose=False)
        p = lgb.train(LGB, lgb.Dataset(Xa[cols], ya), N_ROUNDS).predict(Xte[cols])
        auc = adv_auc(Xm[cols], Xte[cols], "", show=0, verbose=False)
        print(f"{name:<24} n={len(cols):<4} CV score {res['masked (test-like)']['score']:.4f}  "
              f"adversarial AUC {auc:.3f}  test pond share {np.mean(p >= 0.5):.3f}  "
              f"uncertain (0.1-0.9) {np.mean((p > 0.1) & (p < 0.9)):.3f}", flush=True)

if __name__ == "__main__":
    main()
