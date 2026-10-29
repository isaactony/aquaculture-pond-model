"""
Adversarial validation: can a model tell training pixels (masked with real test gaps) from test pixels?
AUC near 0.5 means the gaps are the only difference; higher means the year/period itself shifted the values.

    ./run.sh src/adversarial.py
"""
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(__file__))
from cv import load, masked_copies
from features import build, cube, masks

def adv_auc(Xa, Xb, label, show=12, verbose=True):
    """Out-of-fold AUC of a model separating Xa (0) from Xb (1); prints the top `show` features by gain."""
    X = pd.concat([Xa, Xb], ignore_index=True); z = np.r_[np.zeros(len(Xa)), np.ones(len(Xb))]
    p = np.zeros(len(z)); imp = 0
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(X, z):
        b = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=15, verbose=-1, seed=0, num_threads=10),
                      lgb.Dataset(X.iloc[tr], z[tr]), 300)
        p[va] = b.predict(X.iloc[va]); imp = imp + b.feature_importance("gain")
    auc = roc_auc_score(z, p)
    if verbose:
        print(f"{label}: adversarial AUC {auc:.3f}")
        if show:
            print((pd.Series(imp, index=X.columns) / imp.sum()).sort_values(ascending=False).head(show).round(3).to_string())
    return auc

def main():
    tr, te = load()
    x, y = cube(tr), tr["label"].to_numpy()
    ts, to = masks(te)
    xm, ym, _ = masked_copies(x, y, ts, to, 1, np.random.default_rng(0))
    Xtr, Xte = build(xm), build(cube(te))
    adv_auc(Xtr, Xte, "all features")
    stats = [c for c in Xtr.columns if "__" in c]
    adv_auc(Xtr[stats], Xte[stats], "summary stats only")
    adv_auc(Xtr[ym == 1], Xte, "training ponds only vs test", show=6)
    adv_auc(Xtr[ym == 0], Xte, "training non-ponds only vs test", show=6)

if __name__ == "__main__":
    main()
