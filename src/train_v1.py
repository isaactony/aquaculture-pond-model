"""
v1: LightGBM on season-summary features, trained on training pixels masked with real test gaps.

Training pixels have all 12 months; test pixels only have a 4-6 month window. Each training pixel is copied 10
times, each copy with the missing-month pattern of a randomly chosen test pixel, so the model learns to classify
from partial windows. Only the summary features (per-band and per-index statistics over the months present) are
used: they score almost the same in CV as adding the 12 monthly columns (0.979 vs 0.980) and are less tied to the
calendar of the training year.

    ./run.sh src/train_v1.py
"""
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from cv import LGB, N_ROUNDS, load, masked_copies, run
from features import build, cube, masks

SEEDS = (42, 7, 2026)
K = 10

def summary_cols(X):
    return [c for c in X.columns if "__" in c or c.startswith(("n_", "frac_", "first"))]

def main(out_dir="submissions/v1"):
    os.makedirs(out_dir, exist_ok=True)
    tr, te = load()
    x, y = cube(tr), tr["label"].to_numpy()
    ts, to = masks(te)
    Xte = build(cube(te)); cols = summary_cols(Xte)

    # cross-validation on test-like masked validation pixels (same protocol as src/cv.py)
    res, _ = run("aug", x, y, ts, to, k=K, cols=cols)

    preds = []
    for s in SEEDS:
        xa, ya, _ = masked_copies(x, y, ts, to, K, np.random.default_rng(s))
        b = lgb.train({**LGB, "seed": s}, lgb.Dataset(build(xa)[cols], ya), N_ROUNDS)
        preds.append(b.predict(Xte[cols]))
    p = np.mean(preds, axis=0)
    sub = pd.DataFrame({"ID": te["ID"], "TargetF1": (p >= 0.5).astype(int), "TargetRAUC": p})
    sub.to_csv(os.path.join(out_dir, "submission_v1.csv"), index=False)
    report = {"cv": res, "n_features": len(cols), "test_pond_share": round(float((p >= 0.5).mean()), 3),
              "test_uncertain_share_0.1_0.9": round(float(((p > 0.1) & (p < 0.9)).mean()), 3)}
    with open(os.path.join(out_dir, "cv_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
