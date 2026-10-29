"""
Figure for the README: the seasonal signature of ponds vs other land in the training year (monthly median and
interquartile range of radar VV backscatter and of the NDWI water index).

    .venv/bin/python src/plot_signature.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from features import BANDS, cube, indices

POND, OTHER = "#2a78d6", "#eb6834"          # categorical slots 1-2, validated for colour-blind separation
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e6e5e0", "#fcfcfb"
MONTHS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

def panel(ax, series, title, unit):
    for name, (v, color) in series.items():
        q1, med, q3 = np.nanpercentile(v, [25, 50, 75], axis=0)
        ax.fill_between(range(12), q1, q3, color=color, alpha=0.15, linewidth=0)
        ax.plot(range(12), med, color=color, linewidth=2, marker="o", markersize=4,
                markeredgecolor=SURFACE, markeredgewidth=1)
        ax.annotate(name, (11, med[-1]), xytext=(8, 0), textcoords="offset points", va="center",
                    color=INK, fontsize=9)
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=8)
    ax.set_ylabel(unit, color=MUTED, fontsize=9)
    ax.set_xticks(range(12), MONTHS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.axvspan(1.5, 8.5, color=GRID, alpha=0.35, linewidth=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_facecolor(SURFACE)

def main(out="figures/seasonal_signature.png"):
    tr = pd.read_csv("data/Train.csv")
    x, y = cube(tr), tr["label"].to_numpy()
    ix, names = indices(x)
    vv, ndwi = x[:, :, BANDS.index("VV")], ix[:, :, names.index("ndwi")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), facecolor=SURFACE)
    panel(axes[0], {"Ponds": (vv[y == 1], POND), "Other land": (vv[y == 0], OTHER)},
          "Radar backscatter (VV) drops while ponds hold water", "dB")
    panel(axes[1], {"Ponds": (ndwi[y == 1], POND), "Other land": (ndwi[y == 0], OTHER)},
          "Water index (NDWI) is high from March to November", "NDWI")
    fig.text(0.01, 0.01, "Training pixels, monthly median with interquartile band. Shaded months: March-September.",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 0.97, 1))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print("saved", out)

if __name__ == "__main__":
    main()
