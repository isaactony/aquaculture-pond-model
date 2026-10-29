"""
Features for the GeoAI Aquaculture Pond challenge.

Each row is a 10 m pixel with 12 monthly composites (01 = January ... 12 = December) of Sentinel-1 radar (VV, VH,
in dB) and Sentinel-2 optical bands. Missing months are -9999. Training rows are complete; test rows only cover a
4-6 month window, with extra optical gaps (clouds). So every feature here must work with whatever months exist.
"""
import warnings

import numpy as np
import pandas as pd

MONTHS = [f"{i:02d}" for i in range(1, 13)]
SAR = ["VV", "VH"]
OPTICAL = ["blue", "green", "red", "re1", "re2", "re3", "nir", "nira", "swir1", "swir2"]
BANDS = SAR + OPTICAL
WET = [2, 3, 4, 5, 6, 7, 8]          # March-September (0-based): ponds hold water in the training data

def cube(df):
    """(rows, 12 months, bands) array with NaN for missing values."""
    x = np.stack([df[[f"{b}_{m}" for m in MONTHS]].to_numpy(float) for b in BANDS], axis=2)
    x[x == -9999] = np.nan
    return x

def masks(df):
    """Boolean availability masks per month for SAR and optical, shape (rows, 12)."""
    return (df[[f"VV_{m}" for m in MONTHS]].to_numpy() != -9999,
            df[[f"blue_{m}" for m in MONTHS]].to_numpy() != -9999)

def apply_masks(x, sar_mask, opt_mask):
    """Blank out months in a cube, as in a test row."""
    x = x.copy()
    x[:, :, :2][~sar_mask] = np.nan
    x[:, :, 2:][~opt_mask] = np.nan
    return x

def _nd(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return (a - b) / (a + b)

def indices(x):
    """Monthly spectral indices, shape (rows, 12, k)."""
    g = {b: x[:, :, i] for i, b in enumerate(BANDS)}
    idx = {
        "ndvi": _nd(g["nir"], g["red"]),
        "ndwi": _nd(g["green"], g["nir"]),          # open water is positive
        "mndwi": _nd(g["green"], g["swir1"]),
        "ndmi": _nd(g["nir"], g["swir1"]),
        "awei": 4 * (g["green"] - g["swir1"]) - (0.25 * g["nir"] + 2.75 * g["swir2"]),
        "vv_minus_vh": g["VV"] - g["VH"],
        "sar_sum": g["VV"] + g["VH"],
    }
    return np.stack(list(idx.values()), axis=2), list(idx)

def _stats(v, name, out):
    """Summaries over whichever months are present (all-NaN rows give NaN)."""
    out[f"{name}__mean"] = np.nanmean(v, axis=1)
    out[f"{name}__min"] = np.nanmin(v, axis=1)
    out[f"{name}__max"] = np.nanmax(v, axis=1)
    out[f"{name}__med"] = np.nanmedian(v, axis=1)
    out[f"{name}__std"] = np.nanstd(v, axis=1)
    out[f"{name}__wet_mean"] = np.nanmean(v[:, WET], axis=1)

def build(x):
    """Feature frame from a (rows, 12, bands) cube that may contain NaN months."""
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ix, ix_names = indices(x)
    allv = np.concatenate([x, ix], axis=2)
    names = BANDS + ix_names
    out = {}
    for j, n in enumerate(names):
        for m in range(12):
            out[f"{n}_{MONTHS[m]}"] = allv[:, m, j]
        _stats(allv[:, :, j], n, out)
    sar_ok, opt_ok = ~np.isnan(x[:, :, 0]), ~np.isnan(x[:, :, 2])
    out["n_sar_months"] = sar_ok.sum(1)
    out["n_opt_months"] = opt_ok.sum(1)
    first = np.where(sar_ok.any(1), sar_ok.argmax(1), -1)
    out["first_sar_month"] = first
    out["n_water_months"] = (ix[:, :, ix_names.index("mndwi")] > 0).sum(1)
    out["frac_water_months"] = out["n_water_months"] / np.maximum(out["n_opt_months"], 1)
    out["n_dark_sar_months"] = (x[:, :, 0] < -18).sum(1)
    out["frac_dark_sar_months"] = out["n_dark_sar_months"] / np.maximum(out["n_sar_months"], 1)
    return pd.DataFrame(out).replace([np.inf, -np.inf], np.nan)
