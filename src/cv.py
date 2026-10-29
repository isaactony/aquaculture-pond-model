"""
Cross-validation that mimics the test set: test pixels only have a 4-6 month window (plus cloud gaps), while
training pixels have all 12 months. Validation pixels are therefore scored under missing-month masks copied from
real test rows (several masks per pixel), and every copy of a pixel stays in the same fold.

Score = 0.6 x F1 (at the fixed 0.5 cut-off) + 0.4 x ROC-AUC, as on the leaderboard.

    ./run.sh src/cv.py full aug aug_full
"""
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(__file__))
from features import apply_masks, build, cube, masks

N_VAL_MASKS = 6
LGB = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_child_samples=40, feature_fraction=0.5,
           bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=10, seed=42)
N_ROUNDS = 600

def lb(f1, auc):
    return 0.6 * f1 + 0.4 * auc

def score(y, p):
    f1, auc = f1_score(y, (p >= 0.5).astype(int)), roc_auc_score(y, p)
    return dict(f1=round(f1, 4), auc=round(auc, 4), score=round(lb(f1, auc), 4), mean_p=round(float(p.mean()), 3))

def load(data_dir="data"):
    tr = pd.read_csv(os.path.join(data_dir, "Train.csv"))
    te = pd.read_csv(os.path.join(data_dir, "Test.csv"))
    return tr, te

def masked_copies(x, y, test_sar, test_opt, k, rng, keep_full=False):
    """k copies of every pixel, each with a randomly drawn real test mask. Returns cube, labels, source row index."""
    n = len(x)
    pick = rng.integers(0, len(test_sar), size=(n, k)).ravel()
    xs = np.repeat(x, k, axis=0)
    xs = apply_masks(xs, test_sar[pick], test_opt[pick])
    ys, src = np.repeat(y, k), np.repeat(np.arange(n), k)
    if keep_full:
        xs, ys, src = np.concatenate([xs, x]), np.concatenate([ys, y]), np.concatenate([src, np.arange(n)])
    return xs, ys, src

def run(mode, x, y, test_sar, test_opt, k=10, params=None, rounds=N_ROUNDS, cols=None, verbose=True):
    """
    5-fold CV. mode: "full" trains on complete pixels, "aug" on k masked copies, "aug_full" on both.
    cols restricts the feature columns (default: all). Returns scores and out-of-fold test-like predictions.
    """
    prm = {**LGB, **(params or {})}
    feats = (lambda c: build(c)[cols]) if cols is not None else build
    rng = np.random.default_rng(42)
    oof_masked = np.zeros(len(y) * N_VAL_MASKS); oof_full = np.zeros(len(y))
    yv_all = np.repeat(y, N_VAL_MASKS); t0 = time.time()
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=42).split(x, y):
        if mode == "full":
            xt, yt = x[tr], y[tr]
        else:
            xt, yt, _ = masked_copies(x[tr], y[tr], test_sar, test_opt, k, rng, keep_full=(mode == "aug_full"))
        b = lgb.train(prm, lgb.Dataset(feats(xt), yt), rounds)
        xv, _, _ = masked_copies(x[va], y[va], test_sar, test_opt, N_VAL_MASKS, np.random.default_rng(7))
        idx = (va[:, None] * N_VAL_MASKS + np.arange(N_VAL_MASKS)).ravel()
        oof_masked[idx] = b.predict(feats(xv))
        oof_full[va] = b.predict(feats(x[va]))
    res = {"masked (test-like)": score(yv_all, oof_masked), "full 12 months": score(y, oof_full)}
    if verbose:
        print(f"{mode:<10} k={k:<3} test-like {res['masked (test-like)']}  full-year {res['full 12 months']} "
              f"({time.time() - t0:.0f}s)", flush=True)
    return res, oof_masked

def main():
    tr, te = load()
    x, y = cube(tr), tr["label"].to_numpy()
    test_sar, test_opt = masks(te)
    for mode in sys.argv[1:]:
        run(mode, x, y, test_sar, test_opt)

if __name__ == "__main__":
    main()
