#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================================
# Title: Advance warning of gamma-ray blazar flares from
#        Fermi-LAT light curves: a strictly causal
#        machine-learning backtest
# Authors: Zahir Shah,
#          Sikandar Akbar
#
# Code name: code_final_upload.py
# Language: Python 3
# License: MIT License (see LICENSE)
# Tested under: Python 3.x, Ubuntu Linux
#
# Description of input data:
#   Fermi-LAT daily light-curve CSV files from the Fermi-LAT Light
#   Curve Repository (LCR; Abdollahi et al. 2023), one file per
#   source, columns include Julian Date, TS, Photon Flux
#   [0.1-100 GeV] (photons cm^-2 s^-1), Photon Flux Error, Photon
#   Index, Photon Index Error. See README.md for exact expected
#   filenames.
#
# Description of output data:
#   Bayesian-Blocks flare diagnostics (PNG), rolling-window
#   feature/label tables (CSV), model ROC/PR/reliability curves
#   (PNG), probability timelines (PNG, CSV), lead-time tables
#   (CSV), and a cross-model comparison summary (CSV). Written to
#   pdf_flare_outputs_v3/ by default.
#
# System requirements: see requirements.txt for exact package
#   versions used to generate the results reported in the
#   accompanying manuscript (Table 3).
#
# Calls to external routines: astropy (bayesian_blocks,
#   LombScargle), scikit-learn (LogisticRegression,
#   RandomForestClassifier, etc.), scipy.stats, optionally
#   statsmodels (Lilliefors test) and xgboost if installed.
#
# Additional comments: XGBoost and the dual-threshold alert
#   system in this script are retained for exploratory reference
#   and are not part of the analysis reported in the manuscript;
#   only LR, PLR, and RF results (Table 3) are discussed there.
# ============================================================

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import time as _time
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from astropy.stats import bayesian_blocks
from astropy.timeseries import LombScargle

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                              f1_score, fbeta_score, brier_score_loss,
                              roc_curve, auc as sk_auc,
                              precision_recall_curve,
                              average_precision_score, accuracy_score)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from scipy.stats import skew, kurtosis, kstest, norm
try:
    from statsmodels.stats.diagnostic import lilliefors as sm_lilliefors
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
# NOTE: XGBoost and the dual-threshold alert system (Section 8 below) are
# retained here for exploratory reference. They are not part of the
# analysis reported in Shah & Akbar (2026); only LR, PLR, and RF results
# (Table 3) are discussed in the manuscript.
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from sklearn.metrics import matthews_corrcoef
    HAS_MCC = True
except ImportError:
    HAS_MCC = False


# ============================================================
# 0) SETTINGS
# ============================================================

# --- Input file(s) ---
# Accepts either Fermi-LAT ASCII (.txt) or daily CSV (.csv) files.
# CSV files are auto-detected by extension; no column-index settings needed.
LC_FILE  = "4FGL_J1048.4+7143_daily_3_23_2026.csv"   # primary / target source

# Multi-source: add extra training sources here (set to {} for single-source)
# NOTE: extra sources must have SIMILAR variability (peak/quiescent ratio) to the
# target.  Sources with much higher variability suppress target probabilities.
EXTRA_TRAIN_FILES = {"J1512": "4FGL_J1512.8-0906_daily_3_24_2026.csv",
"J1224":"4FGL_J1224.9+2122_daily_3_24_2026.csv",
"J0904":"4FGL_J0904.9-5734_daily_3_23_2026.csv",
"J1256":"4FGL_J1256.1-0547_daily_3_23_2026.csv",
"J2253":"4FGL_J2253.9+1609_daily_3_23_2026.csv",
"J0538":"4FGL_J0538.8-4405_daily_3_24_2026.csv",
"J0739":"4FGL_J0739.2+0137_daily_3_24_2026.csv",
"J1159":"4FGL_J1159.5+2914_daily_3_24_2026.csv",
"J0730":"4FGL_J0730.3-1141_daily_3_24_2026.csv",
"J1310":"4FGL_J1310.5+3221_daily_3_24_2026.csv",
"J1443":"4FGL_J1443.9+2501_daily_3_24_2026.csv",
"J1522":"4FGL_J1522.1+3144_daily_3_24_2026.csv",
"J0403":"4FGL_J0403.9-3605_daily_3_24_2026.csv",
}

# Column indices — used only for ASCII (.txt) files (ignored for CSV)
TIME_COL     = 0
TS_COL       = 1
FLUX_COL     = 2
FLUXERR_COL  = 3
INDEX_COL    = 4
INDEXERR_COL = 5

# TS quality cut (Fermi-LAT ~2 sigma detection threshold)
TS_MIN               = 4.0
REMOVE_NEGATIVE_FLUX = True
FLUX_MAX             = 1e-4   # remove outlier bins above this flux [ph cm-2 s-1]

REQUIRE_MINUIT_CONVERGENCE = True
INDEX_MIN = -5.0   # exclusive lower bound on "Photon Index" (LCR sign convention)
INDEX_MAX = 0.0    # exclusive upper bound on "Photon Index"

FLAG_LOW_EXPOSURE_ARTIFACTS = True
LOW_EXPOSURE_TS_MAX    = 15.0
LOW_EXPOSURE_FLUX_RATIO = 10.0

AUTO_CONVERT_JD_TO_MJD = True

OUTDIR = Path("pdf_flare_outputs_v3")
OUTDIR.mkdir(parents=True, exist_ok=True)
SHOW_PLOTS = False

# --- Rolling window ---
WINDOW_DAYS          = 365.0
STEP_DAYS            = 7.0
MIN_POINTS_IN_WINDOW = 30   
RECENT_DELTA_DAYS    = 90.0

# --- Bayesian Blocks ---
BB_FITNESS              = "measures"
BB_P0                   = 0.05
FLARE_BLOCK_PERCENTILE  = 70.0
MIN_FLARE_BLOCK_DAYS    = 9.0
BB_N_SIGMA_FLARE        = 3.0        
MERGE_FLARE_GAP_DAYS    = 9.0        

SIGMA_FLOOR_MODE  = "fraction_of_median_flux"
SIGMA_FLOOR_FRAC  = 0.10
SIGMA_FLOOR_ABS   = 0.0

# --- Source normalisation ---
F_QUIESCENT_PERCENTILE = 30.0        

# --- Forecast ---
HORIZON_DAYS         = 90.0
BACKTEST_CUTOFF_MJD  = 60000.0
MIN_OVERLAP_DAYS     = 3.0           
TRIGGER_HORIZON_DAYS = 45.0          

# --- Threshold optimisation ---
PROBA_THRESHOLD_DEFAULT = 0.50
THRESH_GRID             = np.linspace(0.05, 0.95, 91)   
THRESHOLD_FBETA         = 2.0                           
TRIGGER_THRESHOLD_FBETA = 0.5                           
TRIGGER_MIN_PRECISION   = 0.25
TRIGGER_MAX_ALERT_FRACTION = 0.20
WATCH_THRESHOLD_MODE    = "source_local_pooled"
WATCH_THRESHOLD_TIE_TOL = 0.0

# --- Calibration ---
USE_CALIBRATION_TAIL       = True
CALIBRATION_TAIL_FRAC      = 0.20
CALIBRATION_METHOD         = "sigmoid"
CALIBRATION_TAIL_MIN_FLARES = 2
CALIBRATION_TAIL_MIN_FRAC  = 0.20   

# --- PLR feature selection ---
PLR_SELECT_K = 50   

# --- Models to run ---
MODELS_TO_RUN = ["logreg", "polylogreg", "rf"]
if HAS_XGB:
    MODELS_TO_RUN.append("xgb")
MODELS_TO_RUN.append("ensemble")

# --- RF config ---
RF_N_ESTIMATORS      = 200
RF_MIN_SAMPLES_LEAF  = 5
RF_MAX_FEATURES      = "sqrt"

# --- XGB config ---
XGB_N_ESTIMATORS  = 500
XGB_MAX_DEPTH     = 4
XGB_LEARNING_RATE = 0.05

# --- Reliability curve ---
N_BINS_RELIABILITY = 8

# --- Timeline + smoothing ---
PROBA_VIS_LINE_DEFAULT = 0.50
SEQ_SMOOTH_WIN    = 7          
LEAD_USE_SMOOTHED = False      

# --- Lead-time ---
LEAD_LOOKBACK_DAYS    = HORIZON_DAYS     
LONG_LOOKBACK_DAYS    = 180.0            
NEAR_ONSET_WINDOW_DAYS = 30.0
MAX_LEAD_DAYS_WATCH    = HORIZON_DAYS    
MAX_LEAD_DAYS_TRIGGER  = TRIGGER_HORIZON_DAYS

EXPORT_PROBA_CSV   = True
EXPORT_LEADTIME_CSV = True

# --- Dual-threshold alerts ---
DO_DUAL_THRESHOLD_ALERTS = True
WATCHLIST_MIN_RECALL     = 0.85
WATCHLIST_MAX_FPR        = 0.40
MAX_TRIGGER_FRACTION     = 0.25
MAX_WATCHONLY_FRACTION   = 0.40
DUAL_MIN_GAP             = 0.05

# Dual-threshold scoring weights (exposed for tuning)
DUAL_W_WATCH_RECALL       = 1.0
DUAL_W_WATCH_PRECISION    = 0.2
DUAL_W_TRIGGER_PRECISION  = 1.0
DUAL_W_TRIGGER_RECALL     = 0.2
DUAL_W_PENALTY_TRIG_FRAC  = 0.5
DUAL_W_PENALTY_WATCH_FRAC = 0.2

# --- Alert persistence ---
WATCH_MIN_CONSEC    = 2
TRIGGER_MIN_CONSEC  = 2
COOLDOWN_WINDOWS    = 5

# --- Purged walk-forward ---
WF_TRAIN_MIN  = 120
WF_TEST_CHUNK = 90
WF_STEP       = 90
PURGE_LEN     = 6

# --- Permutation + bootstrap ---
TEST_BLOCK_LEN  = 9.           #may be short recomm>13
N_PERM_TEST     = 1000
N_BOOT_TEST     = 500
RANDOM_SEED     = 0

# --- Manuscript figure style ---
FIG_DPI                = 300

BB_FIG_WIDTH           = 6.2
BB_PANEL_HEIGHT        = 2.9

METRIC_FIGSIZE         = (4.2, 4.0)
METRIC_LABEL_FS        = 15
METRIC_TICK_FS         = 13
METRIC_LEGEND_FS       = 10
METRIC_TITLE_FS        = 12

TIMELINE_FIG_WIDTH     = 7.3
TIMELINE_PANEL_HEIGHT  = 3.25
TIMELINE_LABEL_FS      = 17
TIMELINE_TICK_FS       = 14
TIMELINE_LEGEND_FS     = 10
TIMELINE_TITLE_FS      = 13

BB_LABEL_FS            = 15
BB_TICK_FS             = 13
BB_LEGEND_FS           = 10
BB_TITLE_FS            = 13

WATCH_COLOR            = "#0b5fa5"
TRIGGER_COLOR          = "#9b1d20"
WATCH_MARKER_COLOR     = "#178a3d"
FLARE_FILL_COLOR       = "#f18f84"
TEST_SHADE_COLOR       = "#4c5d73"
GUIDE_COLOR            = "#6f6f6f"


def _style_axes(ax, tick_fs=METRIC_TICK_FS, spine_lw=1.1):
    ax.tick_params(axis="both", labelsize=tick_fs, width=spine_lw * 0.9)
    for spine in ax.spines.values():
        spine.set_linewidth(spine_lw)


# ============================================================
# 1) I/O
# ============================================================

def _try_float(x):
    """Convert string to float. Returns None for NaN, inf, or unparseable."""
    try:
        v = float(x)
        return None if (np.isnan(v) or np.isinf(v)) else v
    except Exception:
        return None


def read_lightcurve_ascii(filename,
                          time_col=0, ts_col=1,
                          flux_col=2, fluxerr_col=3,
                          index_col=None, indexerr_col=None,
                          comment_chars=("#", "!", "%"),
                          ts_min=4.0,
                          remove_negative_flux=True):
    """
    Read a Fermi-LAT ASCII light curve with TS-based quality filtering.

    Returns dict with keys: t, flux, flux_err, ts, index, index_err
    All arrays have the same length. Index is NaN where unavailable.
    """
    filename = Path(filename)
    t_list, ts_list, F_list, Fe_list = [], [], [], []
    idx_list, idxe_list = [], []

    with filename.open("r") as fh:
        for line in fh:
            line = line.strip()
            if (not line) or any(line.startswith(c) for c in comment_chars):
                continue
            parts = line.split()

            # Determine minimum required column index
            needed = [time_col, flux_col, fluxerr_col]
            if ts_col is not None:
                needed.append(ts_col)
            if index_col is not None:
                needed += [index_col, indexerr_col]
            if max(needed) >= len(parts):
                continue

            vt  = _try_float(parts[time_col])
            vF  = _try_float(parts[flux_col])
            vE  = _try_float(parts[fluxerr_col])
            vts = _try_float(parts[ts_col]) if ts_col is not None else None

            if vt is None or vF is None or vE is None:
                continue

            # TS quality cut — reject upper limits
            if ts_min is not None and (vts is None or vts < ts_min):
                continue

            # Reject non-physical flux values
            if remove_negative_flux and vF <= 0:
                continue

            t_list.append(vt)
            F_list.append(vF)
            Fe_list.append(vE)
            ts_list.append(vts if vts is not None else np.nan)

            # Spectral index — always append NaN if unavailable
            if index_col is not None and indexerr_col is not None:
                vi  = _try_float(parts[index_col])   if index_col  < len(parts) else None
                vie = _try_float(parts[indexerr_col]) if indexerr_col < len(parts) else None
                idx_list.append(np.nan  if vi  is None else vi)
                idxe_list.append(np.nan if vie is None else vie)
            else:
                idx_list.append(np.nan)
                idxe_list.append(np.nan)

    t   = np.asarray(t_list,   float)
    F   = np.asarray(F_list,   float)
    Fe  = np.asarray(Fe_list,  float)
    ts  = np.asarray(ts_list,  float)
    idx = np.asarray(idx_list, float)
    ide = np.asarray(idxe_list, float)

    order = np.argsort(t)
    t, F, Fe, ts = t[order], F[order], Fe[order], ts[order]
    idx, ide = idx[order], ide[order]

    out = {"t": t, "flux": F, "flux_err": Fe, "ts": ts,
           "index": idx, "index_err": ide}

    # Time-axis auto-detection: JD → MJD
    if AUTO_CONVERT_JD_TO_MJD:
        med = float(np.nanmedian(out["t"]))
        if 2.45e6 < med < 2.47e6:
            out["t"] = out["t"] - 2400000.5
            print("  [IO] JD detected → converted to MJD (JD - 2400000.5)")
        elif med > 1e8:
            print("  [IO] WARNING: Time axis looks like Fermi MET (seconds). "
                  "Convert to MJD manually before running.")
        # else: already MJD

    return out


def read_lightcurve_csv(filename,
                        ts_min=4.0,
                        remove_negative_flux=True,
                        require_minuit_convergence=True,
                        index_min=-5.0,
                        index_max=0.0,
                        verbose=True):

    def _csv_float(val):
        v = val.strip().lstrip("<").strip()
        if v in ("", "-", "nan", "NaN", "N/A"):
            return None
        try:
            f = float(v)
            return None if (np.isnan(f) or np.isinf(f)) else f
        except ValueError:
            return None

    t_list, ts_list, F_list, Fe_list, idx_list, idxe_list = [], [], [], [], [], []

    n_total = n_nonconvergent = n_index_range = 0

    with open(filename, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            n_total += 1
            vt  = _csv_float(row.get("Julian Date", ""))
            vts = _csv_float(row.get("TS", ""))
            vF  = _csv_float(row.get(
                "Photon Flux [0.1-100 GeV](photons cm-2 s-1)", ""))
            vE  = _csv_float(row.get(
                "Photon Flux Error(photons cm-2 s-1)", ""))
            vi  = _csv_float(row.get("Photon Index", ""))
            vie = _csv_float(row.get("Photon Index Error", ""))
            vmr = _csv_float(row.get("MINUIT Return Code", ""))

            if vt is None or vF is None:
                continue

            # TS quality cut
            if ts_min is not None and (vts is None or vts < ts_min):
                continue

            # Reject non-physical / upper-limit flux
            if remove_negative_flux and vF <= 0:
                continue

            # flux_err must be a real measurement (not upper-limit row)
            if vE is None:
                continue

            # LCR fit-convergence cut (MINUIT Return Code != 0 -> fit failed)
            if require_minuit_convergence and vmr is not None and vmr != 0:
                n_nonconvergent += 1
                continue

            # LCR index-range cut: reject a fitted index pegged at its bound
            if vi is not None and not (index_min < vi < index_max):
                n_index_range += 1
                continue

            t_list.append(vt)
            F_list.append(vF)
            Fe_list.append(vE)
            ts_list.append(vts if vts is not None else np.nan)
            idx_list.append(vi  if vi  is not None else np.nan)
            idxe_list.append(vie if vie is not None else np.nan)

    if verbose and (n_nonconvergent or n_index_range):
        print(f"  [IO] {Path(filename).name}: removed {n_nonconvergent} "
              f"non-convergent bin(s) (MINUIT Return Code != 0), "
              f"{n_index_range} bin(s) with Photon Index outside "
              f"({index_min}, {index_max})")

    t   = np.asarray(t_list,   float)
    F   = np.asarray(F_list,   float)
    Fe  = np.asarray(Fe_list,  float)
    ts  = np.asarray(ts_list,  float)
    idx = np.asarray(idx_list, float)
    ide = np.asarray(idxe_list, float)

    order = np.argsort(t)
    t, F, Fe, ts = t[order], F[order], Fe[order], ts[order]
    idx, ide = idx[order], ide[order]

    out = {"t": t, "flux": F, "flux_err": Fe, "ts": ts,
           "index": idx, "index_err": ide}

    # JD → MJD conversion (same logic as ASCII reader)
    if AUTO_CONVERT_JD_TO_MJD and len(t):
        med = float(np.nanmedian(out["t"]))
        if 2.45e6 < med < 2.47e6:
            out["t"] = out["t"] - 2400000.5
            print("  [IO] JD detected → converted to MJD (JD - 2400000.5)")
        elif med > 1e8:
            print("  [IO] WARNING: Time axis looks like Fermi MET (seconds). "
                  "Convert to MJD manually.")

    return out


def load_lightcurve(filename,
                    time_col=0, ts_col=1,
                    flux_col=2, fluxerr_col=3,
                    index_col=4, indexerr_col=5,
                    ts_min=4.0,
                    remove_negative_flux=True):
                    
    if str(filename).lower().endswith(".csv"):
        d = read_lightcurve_csv(
            filename,
            ts_min=ts_min,
            remove_negative_flux=remove_negative_flux,
            require_minuit_convergence=REQUIRE_MINUIT_CONVERGENCE,
            index_min=INDEX_MIN,
            index_max=INDEX_MAX,
        )
    else:
        d = read_lightcurve_ascii(
            filename,
            time_col=time_col, ts_col=ts_col,
            flux_col=flux_col, fluxerr_col=fluxerr_col,
            index_col=index_col, indexerr_col=indexerr_col,
            ts_min=ts_min,
            remove_negative_flux=remove_negative_flux,
        )

    # Outlier cut: remove bins with flux above FLUX_MAX
    if FLUX_MAX is not None:
        mask = d["flux"] <= FLUX_MAX
        n_removed = int(np.sum(~mask))
        if n_removed:
            print(f"  [IO] FLUX_MAX={FLUX_MAX:.1e}: removed {n_removed} "
                  f"outlier bin(s) (flux > {FLUX_MAX:.1e})")
        d = {k: v[mask] for k, v in d.items()}

    # LCR low-exposure artifact cut: marginal-TS bins reporting flux far
    # above the source's own typical level (see Section on quality cuts).
    if FLAG_LOW_EXPOSURE_ARTIFACTS and len(d["flux"]):
        med_flux = np.nanmedian(d["flux"])
        low_exp = ((d["ts"] < LOW_EXPOSURE_TS_MAX) &
                   (d["flux"] > LOW_EXPOSURE_FLUX_RATIO * med_flux))
        n_removed = int(np.sum(low_exp))
        if n_removed:
            print(f"  [IO] LOW-EXPOSURE ARTIFACT: removed {n_removed} bin(s) "
                  f"(TS<{LOW_EXPOSURE_TS_MAX:.0f} & flux>{LOW_EXPOSURE_FLUX_RATIO:.0f}x "
                  f"source median={med_flux:.2e})")
        d = {k: v[~low_exp] for k, v in d.items()}

    return d


def sanity_check(t, f, fe, ts=None, n_first=3):
    """Extended sanity check: basic stats, gaps, cadence, TS distribution."""
    def stats(x):
        fin = x[np.isfinite(x)]
        if len(fin) == 0:
            return np.nan, np.nan, np.nan
        return float(np.nanmin(fin)), float(np.nanmedian(fin)), float(np.nanmax(fin))

    tmin, tmed, tmax = stats(t)
    fmin, fmed, fmax = stats(f)
    emin, emed, emax = stats(fe)
    dt = np.diff(t[np.isfinite(t)])

    print("  " + "─" * 50)
    print(f"  Bins loaded         : {len(t)}")
    print(f"  Time span (MJD)     : {tmin:.2f} → {tmax:.2f}  "
          f"({tmax - tmin:.0f} d = {(tmax - tmin)/365.25:.1f} yr)")
    print(f"  Median cadence      : {np.nanmedian(dt):.2f} days")
    print(f"  Longest gap         : {np.nanmax(dt):.1f} days")
    print(f"  Flux  min/med/max   : {fmin:.3e} / {fmed:.3e} / {fmax:.3e}")
    print(f"  Ferr  min/med/max   : {emin:.3e} / {emed:.3e} / {emax:.3e}")
    print(f"  Peak / quiescent    : {fmax/fmed:.1f}×")
    if ts is not None and np.any(np.isfinite(ts)):
        tsf = ts[np.isfinite(ts)]
        nd  = int(np.sum(tsf >= 4))
        print(f"  TS≥4 detections     : {nd}/{len(tsf)}  "
              f"({100*nd/len(tsf):.1f}%)")
    print(f"  First {n_first} rows (MJD / flux / ferr):")
    for i in range(min(n_first, len(t))):
        print(f"    {t[i]:.3f}   {f[i]:.3e}   {fe[i]:.3e}")
    print("  " + "─" * 50)


# ============================================================
# 2) BAYESIAN BLOCKS → FLARE INTERVALS / ONSETS
# ============================================================

def normalise_flux_to_quiescent(flux, flux_err, quiescent_percentile=30.0):
    """
    Divide flux by the quiescent median so all sources are on the same
    scale (quiescent ≈ 1.0).  Returns (flux_norm, flux_err_norm, F_q).
    """
    threshold = np.nanpercentile(flux, quiescent_percentile)
    F_q = np.nanmedian(flux[flux <= threshold])
    if not (np.isfinite(F_q) and F_q > 0):
        F_q = np.nanmedian(flux)
    return flux / F_q, flux_err / F_q, float(F_q)


def merge_adjacent_flare_intervals(flare_intervals, merge_gap_days=9.0):
    """Merge flare intervals separated by less than merge_gap_days."""
    if len(flare_intervals) == 0:
        return np.zeros((0, 2), float)
    fi = np.asarray(flare_intervals, float)
    order = np.argsort(fi[:, 0])
    fi = fi[order]
    merged = [list(fi[0])]
    for s, e in fi[1:]:
        if s - merged[-1][1] <= float(merge_gap_days):
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return np.array(merged, float)


def _block_has_gap(edges, b, time, gap_threshold):
    """True if block b spans a data gap larger than gap_threshold."""
    m = (time >= edges[b]) & (time < edges[b + 1])
    if np.sum(m) < 2:
        return True
    return bool(np.any(np.diff(time[m]) > gap_threshold))


def bayes_blocks_flares(time, flux, flux_err,
                        p0=0.05, flare_percentile=70.0,
                        min_block_days=9.0,
                        n_sigma_flare=3.0):
    time  = np.asarray(time,     float)
    flux  = np.asarray(flux,     float)
    sigma = np.asarray(flux_err, float)

    # --- clean sigma ---
    good = np.isfinite(sigma) & (sigma > 0)
    if not np.any(good):
        sigma = np.full_like(flux, np.nanstd(flux))
    else:
        sigma = np.where(good, sigma, np.nanmedian(sigma[good]))

    # sigma floor
    if SIGMA_FLOOR_MODE == "fraction_of_median_flux":
        floor = SIGMA_FLOOR_FRAC * np.nanmedian(np.abs(flux))
    else:
        floor = SIGMA_FLOOR_ABS
    if np.isfinite(floor) and floor > 0:
        sigma = np.maximum(sigma, floor)

    # gap threshold for flagging
    dt_median = np.nanmedian(np.diff(time))
    gap_threshold = 3.0 * dt_median

    edges = bayesian_blocks(time, flux, sigma=sigma,
                            fitness=BB_FITNESS, p0=p0)

    block_idx  = np.digitize(time, edges) - 1
    n_blocks   = len(edges) - 1
    block_means  = np.full(n_blocks, np.nan, float)
    block_sigmas = np.full(n_blocks, np.nan, float)
    block_durs   = np.diff(edges)
    has_gap      = np.zeros(n_blocks, bool)

    for b in range(n_blocks):
        m = (block_idx == b)
        if np.any(m):
            fw = flux[m]
            sw = sigma[m]
            # inverse-variance weighted mean
            w = 1.0 / (sw ** 2)
            wsum = np.sum(w)
            if wsum > 0:
                block_means[b]  = float(np.sum(w * fw) / wsum)
                block_sigmas[b] = float(1.0 / np.sqrt(wsum))
        has_gap[b] = _block_has_gap(edges, b, time, gap_threshold)

    # --- dual threshold: percentile + absolute ---
    thr_pct = np.nanpercentile(block_means, flare_percentile)

    quiescent_mask = block_means <= np.nanpercentile(block_means, 40)
    if np.sum(quiescent_mask) >= 3:
        mu_q  = float(np.nanmean(block_means[quiescent_mask]))
        sig_q = float(np.nanstd(block_means[quiescent_mask]))
        thr_abs = mu_q + n_sigma_flare * sig_q
    else:
        thr_abs = thr_pct

    thr = max(thr_pct, thr_abs)

    is_flare_block = ((block_means >= thr)
                      & (block_durs >= min_block_days)
                      & (~has_gap))

    flare_intervals = np.array(
        [(edges[b], edges[b + 1])
         for b in range(n_blocks) if is_flare_block[b]], float
    )
    if flare_intervals.shape == (0,):
        flare_intervals = np.zeros((0, 2), float)

    flare_onsets = []
    for b in range(n_blocks):
        if is_flare_block[b] and (b == 0 or not is_flare_block[b - 1]):
            flare_onsets.append(edges[b])
    flare_onsets = np.asarray(flare_onsets, float)

    return (edges, block_means, block_sigmas, is_flare_block,
            flare_onsets, flare_intervals, thr)


def save_bb_diagnostics(edges, block_means, is_flare_block, thr,
                        tag="full", time=None, flux=None):
    """Two-panel BB diagnostic: (1) raw LC + shaded blocks, (2) block means."""
    try:
        mids    = 0.5 * (edges[:-1] + edges[1:])
        flare_m = np.asarray(is_flare_block, bool)
        n_panels = 2 if (time is not None and flux is not None) else 1
        fig, axes = plt.subplots(n_panels, 1,
                                 figsize=(BB_FIG_WIDTH,
                                          BB_PANEL_HEIGHT * n_panels),
                                 sharex=True)
        if n_panels == 1:
            axes = [axes]

        if n_panels == 2:
            ax0 = axes[0]
            ax0.plot(time, flux, ".", color="0.15", ms=2.4, alpha=0.50,
                     label="Flux")
            for b in range(len(block_means)):
                col = FLARE_FILL_COLOR if is_flare_block[b] else WATCH_COLOR
                x0 = (edges[b] - edges[0]) / (edges[-1] - edges[0] + 1e-10)
                x1 = (edges[b+1] - edges[0]) / (edges[-1] - edges[0] + 1e-10)
                ax0.axhspan(0, block_means[b] if np.isfinite(block_means[b]) else 0,
                            xmin=x0, xmax=x1, alpha=0.18, color=col)
            ax0.axhline(thr, color=TRIGGER_COLOR, ls="--", lw=2.0,
                        label=f"Flare thr = {thr:.2e}")
            ax0.set_ylabel("Flux (ph cm⁻² s⁻¹)", fontsize=BB_LABEL_FS)
            ax0.legend(fontsize=BB_LEGEND_FS, loc="upper left",
                       frameon=True, framealpha=0.95,
                       borderpad=0.35, handlelength=2.4)
            ax0.set_title(f"Bayesian Blocks — {tag}", fontsize=BB_TITLE_FS)
            _style_axes(ax0, tick_fs=BB_TICK_FS)

        ax1 = axes[-1]
        ax1.plot(mids, block_means, ".-", color="0.10", ms=6, lw=1.8)
        ax1.axhline(thr, color=TRIGGER_COLOR, ls="--", lw=2.2,
                    label=f"thr = {thr:.2e}")
        if np.any(flare_m):
            ax1.plot(mids[flare_m], block_means[flare_m],
                     "o", ms=8, color=TRIGGER_COLOR, mec="white", mew=0.8,
                     label="flare blocks")
        ax1.set_xlabel("Block mid-time (MJD)", fontsize=BB_LABEL_FS)
        ax1.set_ylabel("Block mean flux", fontsize=BB_LABEL_FS)
        ax1.legend(fontsize=BB_LEGEND_FS, frameon=True, framealpha=0.95,
                   borderpad=0.35, handlelength=2.4, loc="upper left")
        ax1.grid(True, alpha=0.20)
        _style_axes(ax1, tick_fs=BB_TICK_FS)

        plt.tight_layout(h_pad=0.9)
        outpng = OUTDIR / f"bb_block_means_{tag}.png"
        plt.savefig(outpng, dpi=FIG_DPI, bbox_inches="tight")
        if SHOW_PLOTS:
            plt.show()
        plt.close(fig)
        print(f"  [BB] Diagnostics → {outpng}")
    except Exception as exc:
        print(f"  [BB] Diagnostic save skipped: {exc}")


# ============================================================
# 3) FEATURES
# ============================================================

def gini_coefficient(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x >= 0)]
    if x.size == 0 or np.sum(x) == 0:
        return np.nan
    x = np.sort(x)
    n = x.size
    cumx = np.cumsum(x)
    g = (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
    return float(np.clip(g, 0.0, 1.0))


def fractional_variability(flux, flux_err):
    """Fvar = sqrt(S^2 - median(sigma^2)) / mean(F)  (Vaughan+2003)."""
    f = np.asarray(flux, float)
    e = np.asarray(flux_err, float)
    m = np.isfinite(f) & np.isfinite(e)
    f, e = f[m], e[m]
    if f.size < 8 or np.mean(f) <= 0:
        return np.nan
    s2   = np.var(f, ddof=1)
    err2 = float(np.median(e ** 2))      # robust: median not mean
    val  = s2 - err2
    if val <= 0:
        return 0.0
    return float(np.sqrt(val) / np.mean(f))


def normalised_excess_variance(flux, flux_err):
    """sigma^2_NXS = (S^2 - median(sigma^2)) / mean(F)^2  — dimensionless."""
    f = np.asarray(flux, float)
    e = np.asarray(flux_err, float)
    m = np.isfinite(f) & np.isfinite(e)
    f, e = f[m], e[m]
    if f.size < 8 or np.mean(f) <= 0:
        return np.nan
    s2   = np.var(f, ddof=1)
    err2 = float(np.median(e ** 2))
    return float((s2 - err2) / np.mean(f) ** 2)


def fit_lognormal_params(flux):
    f = np.asarray(flux, float)
    f = f[np.isfinite(f) & (f > 0)]
    if f.size < 10:
        return np.nan, np.nan
    z = np.log(f)
    return float(np.mean(z)), float(np.std(z, ddof=1))


def ks_lognormal_distance(flux, mu_ln, sig_ln):
    """
    KS distance from log-normal fit.
    Uses Lilliefors test (correct for self-fitted parameters) when
    statsmodels is available; falls back to standard KS otherwise.
    """
    f = np.asarray(flux, float)
    f = f[np.isfinite(f) & (f > 0)]
    if (f.size < 10 or not np.isfinite(mu_ln)
            or not np.isfinite(sig_ln) or sig_ln <= 0):
        return np.nan, np.nan
    z = np.log(f)
    if HAS_STATSMODELS:
        try:
            stat, pval = sm_lilliefors(z, dist='norm')
            return float(stat), float(pval)
        except Exception:
            pass
    stat = float(kstest(z, lambda x: norm.cdf(x, loc=mu_ln,
                                               scale=sig_ln)).statistic)
    return stat, np.nan


def safe_lin_slope(t, y):
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    if t.size < 3:
        return np.nan
    tt    = t - np.mean(t)
    denom = np.sum(tt ** 2)
    if denom <= 0:
        return np.nan
    return float(np.sum(tt * (y - np.mean(y))) / denom)


def recent_delta_mean(time, series, t0, t1,
                      delta_days=90.0, min_points=15):
    a0, a1 = t0, min(t0 + delta_days, t1)
    b0, b1 = max(t1 - delta_days, t0), t1
    mA = (time >= a0) & (time <= a1) & np.isfinite(series)
    mB = (time >= b0) & (time <= b1) & np.isfinite(series)
    if np.sum(mA) < min_points or np.sum(mB) < min_points:
        return np.nan
    return float(np.mean(series[mB]) - np.mean(series[mA]))


def trailing_window_slope(time, series, t1, delta_days=30.0, min_points=7):
    t0 = max(float(np.nanmin(time)), float(t1) - float(delta_days))
    m = (time >= t0) & (time <= t1) & np.isfinite(series)
    if np.sum(m) < min_points:
        return np.nan
    return safe_lin_slope(time[m], np.asarray(series, float)[m])


def trailing_fraction_above(time, flux, t1, delta_days=30.0,
                            threshold=1.0, min_points=7):
    t0 = max(float(np.nanmin(time)), float(t1) - float(delta_days))
    m = (time >= t0) & (time <= t1) & np.isfinite(flux)
    if np.sum(m) < min_points:
        return np.nan
    return float(np.mean(np.asarray(flux, float)[m] >= float(threshold)))


def lomb_scargle_features(time, flux, flux_err, t0, t1,
                           min_period=10.0, max_period=180.0):
    """Peak LS power, peak period, false alarm probability."""
    m = (time >= t0) & (time <= t1) & np.isfinite(flux)
    tw, fw, ew = time[m], flux[m], flux_err[m]
    if tw.size < 20:
        return np.nan, np.nan, np.nan
    try:
        ls    = LombScargle(tw, fw, ew)
        freqs = np.linspace(1.0 / max_period, 1.0 / min_period, 500)
        power = ls.power(freqs)
        pk    = float(np.nanmax(power))
        prd   = float(1.0 / freqs[np.argmax(power)])
        fap   = float(ls.false_alarm_probability(pk))
        return pk, prd, fap
    except Exception:
        return np.nan, np.nan, np.nan


def structure_function_slope(time, flux, t0, t1, n_lags=15):
    """
    Slope of log10(SF) vs log10(lag) — variability power-law index.
    SF(tau) = <(F(t+tau)-F(t))^2>.
    """
    m = (time >= t0) & (time <= t1) & np.isfinite(flux)
    tw, fw = time[m], flux[m]
    if tw.size < 15:
        return np.nan
    lags    = np.geomspace(3.0, (t1 - t0) / 3.0, n_lags)
    sf_vals = []
    lag_use = []
    for lag in lags:
        diffs = []
        for i in range(len(tw)):
            jm = np.abs(tw - tw[i] - lag) < (lag * 0.3)
            if np.any(jm):
                diffs.extend((fw[jm] - fw[i]) ** 2)
        if len(diffs) >= 5:
            sf_vals.append(float(np.mean(diffs)))
            lag_use.append(lag)
    if len(sf_vals) < 5:
        return np.nan
    return safe_lin_slope(np.log10(lag_use),
                          np.log10(np.maximum(sf_vals, 1e-60)))


def time_since_last_flare(T_end_val, flare_onsets_train):
    """Days since last known (TRAIN) flare onset before T_end_val."""
    past = flare_onsets_train[flare_onsets_train < float(T_end_val)]
    if len(past) == 0:
        return np.nan
    return float(T_end_val - past[-1])


def _subwindow_edges(t0, t1, n_sub=4):
    t0, t1 = float(t0), float(t1)
    if t1 <= t0:
        return []
    edges = np.linspace(t0, t1, int(n_sub) + 1)
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def _subwindow_stat(time, flux, flux_err, a, b, min_pts=15):
    m = (time >= a) & (time <= b) & np.isfinite(flux) & np.isfinite(flux_err)
    if np.sum(m) < min_pts:
        return np.nan, np.nan, np.nan
    fw, ew   = flux[m], flux_err[m]
    _mu, slg = fit_lognormal_params(fw)
    return float(np.mean(fw)), float(fractional_variability(fw, ew)), float(slg)


def subwindow_trend_features(time, flux, flux_err, t0, t1,
                              n_sub=4, min_pts_sub=15,
                              F_quiescent=1.0):
    segs = _subwindow_edges(t0, t1, n_sub=n_sub)
    if len(segs) < 3:
        return np.nan, np.nan, np.nan
    mids, means, fvars, sigs = [], [], [], []
    for a, b in segs:
        mn, fv, sg = _subwindow_stat(time, flux, flux_err, a, b,
                                     min_pts=min_pts_sub)
        mids.append(0.5 * (a + b))
        means.append(mn)
        fvars.append(fv)
        sigs.append(sg)
    mids  = np.asarray(mids,  float)
    means = np.asarray(means, float)
    fvars = np.asarray(fvars, float)
    sigs  = np.asarray(sigs,  float)
    Fq    = max(float(F_quiescent), 1e-30)
    return (
        safe_lin_slope(mids, means / Fq),   # normalised slope
        safe_lin_slope(mids, fvars),
        safe_lin_slope(mids, sigs),
    )


def window_features(time, flux, flux_err, t0, t1,
                    index=None,
                    min_points=60,
                    recent_days=90.0,
                    global_tail_threshold=None,
                    F_quiescent=1.0,
                    idx_quiescent=np.nan,
                    flare_onsets_train=None):
    """
    Compute all variability features for a single rolling window [t0, t1].
    Source-dependent features are normalised by F_quiescent / idx_quiescent
    so they are comparable across sources.
    """
    m = (time >= t0) & (time <= t1) & np.isfinite(flux) & np.isfinite(flux_err)
    if np.sum(m) < min_points:
        return None
    tw = time[m];  fw = flux[m];  ew = flux_err[m]
    Fq = max(float(F_quiescent), 1e-30)

    mu_ln, sig_ln = fit_lognormal_params(fw)
    ks_stat, ks_pval = ks_lognormal_distance(fw, mu_ln, sig_ln)

    feat = {}

    # --- basic distribution (normalised) ---
    # cadence_days replaces raw npts — makes the sampling density explicit
    # and cadence-invariant for subsequent features
    feat["cadence_days"] = float(t1 - t0) / max(fw.size, 1)
    feat["mean_norm"] = float(np.mean(fw)) / Fq
    feat["std_norm"]  = (float(np.std(fw, ddof=1))
                         if fw.size >= 10 else np.nan) / Fq
    feat["mu_ln_norm"] = (mu_ln - np.log(Fq)) if np.isfinite(mu_ln) else np.nan
    feat["sig_ln"]     = sig_ln   # dimensionless — no normalisation needed

    MIN_HM = 30   # minimum points for higher moments
    feat["skew"] = float(skew(fw, bias=False))     if fw.size >= MIN_HM else np.nan
    feat["kurt"] = float(kurtosis(fw, fisher=True,
                                  bias=False))      if fw.size >= MIN_HM else np.nan

    # --- tail features ---
    mu = float(np.mean(fw))
    sd = float(np.std(fw, ddof=1)) if fw.size >= 10 else np.nan

    # global training tail (leakage-free)
    if global_tail_threshold is not None and np.isfinite(global_tail_threshold):
        feat["tail_frac_global_p95"] = float(np.mean(fw >= global_tail_threshold))
    else:
        feat["tail_frac_global_p95"] = np.nan

    # excess above mean+2sig relative to Gaussian expectation
    if np.isfinite(sd) and sd > 0:
        obs_tail = float(np.mean(fw >= (mu + 2.0 * sd)))
        feat["excess_tail_2sig"] = obs_tail - (1.0 - norm.cdf(2.0))
    else:
        feat["excess_tail_2sig"] = np.nan

    # p95/median ratio — heavy-tail proxy
    if fw.size >= 10:
        feat["p95_over_median"] = float(
            np.nanpercentile(fw, 95.0) / (np.nanpercentile(fw, 50.0) + 1e-30))
        feat["p99_over_mean"]   = float(
            np.nanpercentile(fw, 99.0) / (mu + 1e-30))
    else:
        feat["p95_over_median"] = np.nan
        feat["p99_over_mean"]   = np.nan

    # IQR normalised by median
    if fw.size >= 10:
        feat["iqr_norm"] = float(
            (np.nanpercentile(fw, 75.0) - np.nanpercentile(fw, 25.0))
            / (np.nanpercentile(fw, 50.0) + 1e-30))
    else:
        feat["iqr_norm"] = np.nan

    feat["max_over_mean"] = float(np.nanmax(fw) / (mu + 1e-30))

    # --- variability amplitude ---
    feat["gini"]  = gini_coefficient(fw)
    feat["fvar"]  = fractional_variability(fw, ew)
    feat["nxs"]   = normalised_excess_variance(fw, ew)   # signed sigma^2_NXS
    feat["ks_lognorm"] = ks_stat
    feat["ks_pval"]    = ks_pval

    # --- trend features (normalised) ---
    feat["slope_flux_norm"]  = safe_lin_slope(tw, fw / Fq)
    feat["slope_logflux"]    = safe_lin_slope(tw, np.log(np.maximum(fw, 1e-30)))
    feat["delta90_mean_norm"] = recent_delta_mean(time, flux, t0, t1,
                                                  delta_days=recent_days,
                                                  min_points=7) / Fq
    feat["delta45_mean_norm"] = recent_delta_mean(time, flux, t0, t1,
                                                  delta_days=45.0,
                                                  min_points=7) / Fq
    feat["delta30_mean_norm"] = recent_delta_mean(time, flux, t0, t1,
                                                  delta_days=30.0,
                                                  min_points=7) / Fq
    feat["slope30_flux_norm"] = trailing_window_slope(
        time, flux / Fq, t1, delta_days=30.0, min_points=7)
    feat["slope45_flux_norm"] = trailing_window_slope(
        time, flux / Fq, t1, delta_days=45.0, min_points=7)
    feat["frac30_above_1p5q"] = trailing_fraction_above(
        time, flux, t1, delta_days=30.0, threshold=1.5 * Fq, min_points=7)
    feat["frac45_above_1p5q"] = trailing_fraction_above(
        time, flux, t1, delta_days=45.0, threshold=1.5 * Fq, min_points=7)

    slope_mean_sub, slope_fvar, slope_sig = subwindow_trend_features(
        time=time, flux=flux, flux_err=flux_err,
        t0=t0, t1=t1, n_sub=4, min_pts_sub=7,
        F_quiescent=Fq
    )
    feat["slope_mean_sub"] = slope_mean_sub
    feat["slope_fvar"]     = slope_fvar
    feat["slope_sig_ln"]   = slope_sig

    # --- Lomb-Scargle QPO features ---
    ls_pk, ls_prd, ls_fap = lomb_scargle_features(time, flux, flux_err,
                                                   t0, t1)
    feat["ls_peak_power"]  = ls_pk
    feat["ls_peak_period"] = ls_prd
    feat["ls_fap"]         = ls_fap

    # --- Structure function slope ---
    feat["sf_slope"] = structure_function_slope(time, flux, t0, t1)

    # --- time since last flare ---
    if flare_onsets_train is not None and len(flare_onsets_train) > 0:
        feat["days_since_flare"] = time_since_last_flare(
            t1, flare_onsets_train)
    else:
        feat["days_since_flare"] = np.nan

    # --- spectral index features (normalised by quiescent) ---
    if index is not None and len(index) == len(time):
        iw = index[m]
        ni = int(np.sum(np.isfinite(iw)))
        Gq = float(idx_quiescent) if np.isfinite(idx_quiescent) else np.nan
        feat["mean_idx"]           = float(np.nanmean(iw))  if ni > 5 else np.nan
        feat["delta_idx_from_mean"] = (float(np.nanmean(iw)) - Gq) if (ni > 5 and np.isfinite(Gq)) else np.nan
        feat["std_idx"]            = float(np.nanstd(iw, ddof=1)) if ni > 5 else np.nan
        feat["slope_idx"]          = safe_lin_slope(tw, iw)
        feat["delta90_idx"]        = recent_delta_mean(time, index, t0, t1,
                                                       delta_days=recent_days,
                                                       min_points=7)
        feat["delta45_idx"]        = recent_delta_mean(time, index, t0, t1,
                                                       delta_days=45.0,
                                                       min_points=7)
        feat["delta30_idx"]        = recent_delta_mean(time, index, t0, t1,
                                                       delta_days=30.0,
                                                       min_points=7)
    return feat


def build_windows(time, flux, flux_err, index,
                  window_days=365.0, step_days=7.0, min_points=60,
                  global_tail_threshold=None,
                  F_quiescent=1.0,
                  idx_quiescent=np.nan,
                  flare_onsets_train=None):
    tmin, tmax = float(time.min()), float(time.max())
    ends  = np.arange(tmin + window_days, tmax, step_days)
    feats = []
    T_end = []

    for Te in ends:
        t0, t1 = Te - window_days, Te
        feat = window_features(
            time, flux, flux_err, t0, t1,
            index=index,
            min_points=min_points,
            recent_days=RECENT_DELTA_DAYS,
            global_tail_threshold=global_tail_threshold,
            F_quiescent=F_quiescent,
            idx_quiescent=idx_quiescent,
            flare_onsets_train=flare_onsets_train,
        )
        if feat is None:
            continue
        feats.append(feat)
        T_end.append(Te)

    if not feats:
        raise RuntimeError(
            "No windows created. Reduce MIN_POINTS_IN_WINDOW or adjust WINDOW/STEP.")

    keys = list(feats[0].keys())
    X    = np.array([[f.get(k, np.nan) for k in keys] for f in feats], float)
    return X, np.array(T_end, float), keys


def median_impute_with_train(X, train_mask):
    """
    Impute NaN using TRAIN-set medians only (no leakage from test).
    Returns imputed X and the per-feature train medians.
    """
    X2      = np.asarray(X, float).copy()
    medians = np.nanmedian(X2[train_mask], axis=0)
    for j in range(X2.shape[1]):
        bad = ~np.isfinite(X2[:, j])
        fill = medians[j] if np.isfinite(medians[j]) else 0.0
        X2[bad, j] = fill
    return X2, medians


def label_windows(T_end, flare_intervals, horizon_days=90.0,
                  min_overlap_days=3.0):
    """
    Y = 1 if any flare interval overlaps (T_end, T_end+horizon] by
    at least min_overlap_days.
    """
    Y = np.zeros(len(T_end), dtype=int)
    if len(flare_intervals) == 0:
        return Y
    for i, Te in enumerate(T_end):
        h0, h1 = Te, Te + horizon_days
        for s, e in flare_intervals:
            overlap = min(e, h1) - max(s, h0)
            if overlap >= min_overlap_days:
                Y[i] = 1
                break
    return Y


def label_windows_to_onset(T_end, flare_onsets, flare_intervals=None,
                           horizon_days=45.0, min_lead_days=0.0,
                           exclude_active_flare=True):
    """
    Trigger label: Y = 1 only if a flare onset occurs within the next
    horizon_days, measured from T_end. Optionally exclude windows that are
    already inside a flare interval, so TRIGGER focuses on pre-onset warning.
    """
    T_end = np.asarray(T_end, float)
    flare_onsets = np.asarray(flare_onsets, float)
    intervals = np.asarray(flare_intervals, float) \
        if flare_intervals is not None else np.zeros((0, 2), float)

    Y = np.zeros(len(T_end), dtype=int)
    if flare_onsets.size == 0:
        return Y

    for i, Te in enumerate(T_end):
        if exclude_active_flare and len(intervals) > 0:
            in_flare = any((float(s) <= Te < float(e)) for s, e in intervals)
            if in_flare:
                continue

        dt = flare_onsets - float(Te)
        if np.any((dt > float(min_lead_days)) & (dt <= float(horizon_days))):
            Y[i] = 1
    return Y


def print_feature_correlations(X, keys, threshold=0.90):
    """Print highly correlated feature pairs."""
    corr = np.corrcoef(X.T)
    pairs = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if np.isfinite(corr[i, j]) and abs(corr[i, j]) >= threshold:
                pairs.append((keys[i], keys[j], corr[i, j]))
    if pairs:
        print(f"  Highly correlated pairs (|r|≥{threshold}):")
        for a, b, r in sorted(pairs, key=lambda x: -abs(x[2]))[:10]:
            print(f"    {a:30s} ↔ {b:30s}  r={r:.3f}")
    else:
        print(f"  No pairs with |r|≥{threshold}")


# ============================================================
# 4) MODELS
# ============================================================

def _make_sklearn_version():
    import sklearn
    parts = sklearn.__version__.split(".")
    return tuple(int(x) for x in parts[:2])


def make_model(name: str, n_pos=None, n_neg=None):
    name = name.lower().strip()

    if name == "logreg":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                solver="lbfgs",
                max_iter=5000,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                C=1.0,
            )),
        ])

    if name == "polylogreg":
        return Pipeline([
            ("poly",   PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_classif, k=PLR_SELECT_K)),
            ("clf",    LogisticRegression(
                solver="lbfgs",
                max_iter=5000,
                class_weight="balanced",
                C=0.05,
                random_state=RANDOM_SEED,
            )),
        ])

    if name == "rf":
        return RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            max_features=RF_MAX_FEATURES,
            class_weight="balanced",
            oob_score=True,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )

    if name == "xgb":
        if not HAS_XGB:
            raise ValueError("XGBoost not installed.")
        scale_pos = float(n_neg / n_pos) if (n_pos and n_neg) else 1.0
        return XGBClassifier(
            n_estimators=XGB_N_ESTIMATORS,
            max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            eval_metric="logloss",
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbosity=0,
        )

    if name == "ensemble":
        # built separately after individual models are fitted
        raise ValueError("Ensemble must be constructed externally.")

    raise ValueError(f"Unknown model: {name}")


# ============================================================
# 5) OOF + CALIBRATION
# ============================================================

def compute_tail_frac(y_train, min_flares_in_tail=2, default_frac=0.20,
                      min_frac=0.20):
    """Find smallest tail fraction containing >= min_flares_in_tail flares."""
    n = len(y_train)
    for frac in np.arange(min_frac, 0.61, 0.05):
        n_tail = int(frac * n)
        if np.sum(y_train[n - n_tail:]) >= min_flares_in_tail:
            return float(frac)
    return float(default_frac)


def oof_probs_walk_forward(model_name, X, y,
                           min_train=None, step=20, val_size=20,
                           require_two_classes=True,
                           purge_len=0,
                           n_pos=None, n_neg=None):
    X = np.asarray(X, float)
    y = np.asarray(y).astype(int)
    n = len(y)
    if min_train is None:
        min_train = max(60, int(0.30 * n))

    p_oof = np.full(n, np.nan, float)
    i = int(min_train)
    while i < n:
        j         = min(n, i + int(val_size))
        train_end = max(0, i - int(purge_len))
        if train_end < 10:
            i += int(step); continue
        y_fit = y[:train_end]
        if require_two_classes and len(np.unique(y_fit)) < 2:
            i += int(step); continue
        try:
            m = make_model(model_name, n_pos=n_pos, n_neg=n_neg)
            m.fit(X[:train_end], y_fit)
            p_oof[i:j] = m.predict_proba(X[i:j])[:, 1]
        except Exception:
            pass
        i += int(step)

    return p_oof if np.any(np.isfinite(p_oof)) else None


def _cal_classifier(base_model, method="sigmoid"):
    skv = _make_sklearn_version()
    try:
        from sklearn.frozen import FrozenEstimator
        return CalibratedClassifierCV(
            estimator=FrozenEstimator(base_model),
            method=method,
        )
    except Exception:
        kw = "estimator" if skv >= (1, 2) else "base_estimator"
        return CalibratedClassifierCV(**{kw: base_model},
                                      method=method, cv="prefit")


def fit_tail_calibrated_prefit(base_model, X_fit, y_fit,
                               X_cal, y_cal, method="sigmoid"):
    base_model.fit(X_fit, y_fit)
    cal = _cal_classifier(base_model, method=method)
    cal.fit(X_cal, y_cal)
    return cal


def fit_maybe_calibrated_and_oof(model_name, X_train, y_train,
                                  use_calibration_tail=True,
                                  tail_frac=0.20):
    X_train = np.asarray(X_train, float)
    y_train = np.asarray(y_train).astype(int)
    n       = len(y_train)
    n_pos   = int(np.sum(y_train))
    n_neg   = n - n_pos
    y_tail  = None
    p_tail  = None
    min_tr  = max(60, int(0.30 * n))

    if not use_calibration_tail:
        base = make_model(model_name, n_pos=n_pos, n_neg=n_neg)
        base.fit(X_train, y_train)
        p_oof   = oof_probs_walk_forward(
            model_name, X_train, y_train,
            min_train=min_tr, step=20, val_size=20,
            require_two_classes=True, purge_len=PURGE_LEN,
            n_pos=n_pos, n_neg=n_neg)
        oof_idx = np.arange(n, dtype=int)
        return base, False, p_oof, oof_idx, "none", y_tail, p_tail

    n_tail = int(np.clip(int(tail_frac * n), 20, n - 1))
    if n_tail <= 0 or n_tail >= n:
        base = make_model(model_name, n_pos=n_pos, n_neg=n_neg)
        base.fit(X_train, y_train)
        p_oof   = oof_probs_walk_forward(
            model_name, X_train, y_train,
            min_train=min_tr, step=20, val_size=20,
            require_two_classes=True, purge_len=PURGE_LEN,
            n_pos=n_pos, n_neg=n_neg)
        oof_idx = np.arange(n, dtype=int)
        return base, False, p_oof, oof_idx, "tail_invalid", y_tail, p_tail

    fit_n   = n - n_tail
    X_fit   = X_train[:fit_n];  y_fit = y_train[:fit_n]
    X_cal   = X_train[fit_n:];  y_cal = y_train[fit_n:]
    oof_idx = np.arange(fit_n, dtype=int)
    used    = f"tail({n_tail}/{n})"

    p_oof = oof_probs_walk_forward(
        model_name, X_fit, y_fit,
        min_train=max(60, int(0.30 * fit_n)),
        step=20, val_size=20,
        require_two_classes=True, purge_len=PURGE_LEN,
        n_pos=int(np.sum(y_fit)), n_neg=fit_n - int(np.sum(y_fit)))
    if p_oof is None:
        p_oof = oof_probs_walk_forward(
            model_name, X_fit, y_fit,
            min_train=max(30, int(0.15 * fit_n)),
            step=20, val_size=20,
            require_two_classes=False, purge_len=0,
            n_pos=int(np.sum(y_fit)), n_neg=fit_n - int(np.sum(y_fit)))

    if len(np.unique(y_fit)) < 2 or len(np.unique(y_cal)) < 2:
        base = make_model(model_name, n_pos=n_pos, n_neg=n_neg)
        base.fit(X_train, y_train)
        return base, False, p_oof, oof_idx, "tail_single_class", y_tail, p_tail

    base = make_model(model_name, n_pos=n_pos, n_neg=n_neg)
    try:
        fitted  = fit_tail_calibrated_prefit(
            base, X_fit, y_fit, X_cal, y_cal,
            method=CALIBRATION_METHOD)
        did_cal = True
        y_tail  = y_cal
        p_tail  = fitted.predict_proba(X_cal)[:, 1]
    except Exception as exc:
        print(f"  [CAL] {model_name}: calibration failed ({exc}) → uncalibrated")
        base = make_model(model_name, n_pos=n_pos, n_neg=n_neg)
        base.fit(X_train, y_train)
        fitted  = base
        did_cal = False
        used    = "tail_failed"
        y_tail  = y_cal
        try:
            p_tail = fitted.predict_proba(X_cal)[:, 1]
        except Exception:
            p_tail  = None
            y_tail  = None

    return fitted, did_cal, p_oof, oof_idx, used, y_tail, p_tail


# ============================================================
# 6) THRESHOLD OPTIMISATION + METRICS
# ============================================================

def optimize_threshold_grid(y_true, p, grid, beta=2.0,
                            prefer_higher_within=0.0):
    """Find threshold maximising F-beta score (recall-weighted)."""
    y_true = np.asarray(y_true).astype(int)
    p      = np.asarray(p, float)
    m      = np.isfinite(p) & np.isfinite(y_true)
    if np.sum(m) < 10 or len(np.unique(y_true[m])) < 2:
        return PROBA_THRESHOLD_DEFAULT, {"note": "insufficient data/classes"}

    rows = []
    for thr in grid:
        yhat = (p[m] >= thr).astype(int)
        fb   = fbeta_score(y_true[m], yhat, beta=beta, zero_division=0)
        f1v  = f1_score(y_true[m], yhat, zero_division=0)
        rows.append({
            "thr":       float(thr),
            "fbeta":     float(fb),
            "f1":        float(f1v),
            "precision": float(precision_score(y_true[m], yhat,
                                               zero_division=0)),
            "recall":    float(recall_score(y_true[m], yhat,
                                            zero_division=0)),
        })

    best_f = max(r["fbeta"] for r in rows)
    tol = max(0.0, float(prefer_higher_within))
    if tol > 0:
        keep = [r for r in rows if r["fbeta"] >= (best_f - tol)]
        best = max(keep, key=lambda r: (r["thr"], r["precision"], r["recall"]))
    else:
        best = max(rows, key=lambda r: r["fbeta"])
    return best["thr"], best


def assemble_causal_train_scores(y_train, p_oof, p_tail=None):
    y_train = np.asarray(y_train).astype(int)
    out = np.full(len(y_train), np.nan, float)

    if p_oof is not None:
        p_oof = np.asarray(p_oof, float)
        n_fit = min(len(out), len(p_oof))
        out[:n_fit] = p_oof[:n_fit]

    if p_tail is not None:
        p_tail = np.asarray(p_tail, float)
        n_tail = min(len(out), len(p_tail))
        out[len(out) - n_tail:] = p_tail[-n_tail:]

    return out


def build_source_local_causal_scores(all_data, model_name, label_key,
                                     train_end_boundary,
                                     use_calibration_tail=True):
    y_parts = []
    p_parts = []
    rows = []

    for src_name, d in all_data.items():
        if "X" not in d or label_key not in d or "T_end" not in d:
            continue

        tr_mask = (np.asarray(d["T_end"], float) <= float(train_end_boundary))
        if not np.any(tr_mask):
            continue

        X_src = np.asarray(d["X"][tr_mask], float)
        y_src = np.asarray(d[label_key][tr_mask]).astype(int)
        X_src, _ = median_impute_with_train(
            X_src, np.ones(len(X_src), dtype=bool))

        tail_frac = compute_tail_frac(
            y_src, min_flares_in_tail=CALIBRATION_TAIL_MIN_FLARES,
            min_frac=CALIBRATION_TAIL_MIN_FRAC)
        (_, _, p_oof_src, _,
         used_src, _, p_tail_src) = fit_maybe_calibrated_and_oof(
            model_name, X_src, y_src,
            use_calibration_tail=use_calibration_tail,
            tail_frac=tail_frac,
        )
        p_src = assemble_causal_train_scores(y_src, p_oof_src, p_tail_src)
        valid = int(np.sum(np.isfinite(p_src)))

        rows.append({
            "source": src_name,
            "n_train": int(len(y_src)),
            "n_pos": int(np.sum(y_src)),
            "n_valid_scores": valid,
            "calibration": str(used_src),
        })
        if valid == 0:
            continue

        y_parts.append(y_src)
        p_parts.append(p_src)

    if not y_parts:
        return None, None, rows

    return np.concatenate(y_parts), np.concatenate(p_parts), rows


def optimize_trigger_threshold_grid(y_true, p, grid, beta=0.5,
                                    min_precision=0.25,
                                    max_alert_fraction=0.20):
    """
    Trigger threshold search on the onset-based labels.
    We favour precision over recall, but avoid thresholds so extreme that the
    trigger never fires.
    """
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p, float)
    m = np.isfinite(p) & np.isfinite(y_true)
    if np.sum(m) < 10 or len(np.unique(y_true[m])) < 2:
        return PROBA_THRESHOLD_DEFAULT, {"note": "insufficient data/classes"}

    yt = y_true[m]
    pp = p[m]
    best = None
    fallback = None

    for thr in grid:
        yhat = (pp >= thr).astype(int)
        prec = float(precision_score(yt, yhat, zero_division=0))
        rec = float(recall_score(yt, yhat, zero_division=0))
        frac_alert = float(np.mean(yhat))
        fb = float(fbeta_score(yt, yhat, beta=beta, zero_division=0))
        info = {
            "thr": float(thr),
            "fbeta": fb,
            "precision": prec,
            "recall": rec,
            "alert_fraction": frac_alert,
        }

        if fallback is None or (info["precision"], info["recall"], info["fbeta"]) > \
                (fallback["precision"], fallback["recall"], fallback["fbeta"]):
            fallback = info

        if prec < float(min_precision):
            continue
        if max_alert_fraction is not None and frac_alert > float(max_alert_fraction):
            continue

        score = fb + 0.10 * prec - 0.10 * frac_alert
        info["score"] = float(score)
        if best is None or score > best["score"]:
            best = info

    if best is not None:
        return best["thr"], best
    if fallback is not None:
        fallback["note"] = "fallback_no_threshold_met_constraints"
        return fallback["thr"], fallback
    return PROBA_THRESHOLD_DEFAULT, {"note": "no_valid_threshold"}


def eval_at_threshold(y_true, p, thr):
    """Full confusion-matrix metrics at a given threshold."""
    y_true = np.asarray(y_true).astype(int)
    p      = np.asarray(p, float)
    m      = np.isfinite(p) & np.isfinite(y_true)
    if np.sum(m) < 5 or len(np.unique(y_true[m])) < 2:
        return None
    yhat = (p[m] >= thr).astype(int)
    yt   = y_true[m]

    tp = int(np.sum((yhat == 1) & (yt == 1)))
    fp = int(np.sum((yhat == 1) & (yt == 0)))
    tn = int(np.sum((yhat == 0) & (yt == 0)))
    fn = int(np.sum((yhat == 0) & (yt == 1)))

    prec  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr   = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    npv   = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1v   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0
    f2v   = 5*prec*rec / (4*prec+rec) if (4*prec+rec) > 0 else 0.0
    denom = np.sqrt(float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    mcc   = float((tp*tn - fp*fn) / denom) if denom > 0 else 0.0

    return {
        "threshold": float(thr), "precision": float(prec),
        "recall":    float(rec),  "fpr":  float(fpr),
        "npv":  float(npv),       "f1":   float(f1v),
        "f2":   float(f2v),       "mcc":  float(mcc),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "n":  int(np.sum(m)), "pos": int(np.sum(yt == 1)),
    }


# ============================================================
# 7) RELIABILITY CURVE + TIMELINE UTILITIES
# ============================================================

def plot_reliability_curve(y_true, p, outpath, title, n_bins=8):
    y_true = np.asarray(y_true).astype(int)
    p      = np.asarray(p, float)
    m      = np.isfinite(p) & np.isfinite(y_true)
    if np.sum(m) < 10 or len(np.unique(y_true[m])) < 2:
        print(f"  [CAL] Reliability curve skipped: {outpath}")
        return None

    n_bins = min(int(n_bins), max(3, int(np.sum(m)) // 8))
    frac_pos, mean_pred = calibration_curve(y_true[m], p[m],
                                            n_bins=n_bins,
                                            strategy="quantile")
    brier     = float(brier_score_loss(y_true[m], p[m]))
    base_rate = float(np.mean(y_true[m]))
    bs_clim   = base_rate * (1 - base_rate)
    bss       = float(1.0 - brier / bs_clim) if bs_clim > 0 else np.nan

    # binomial error bars
    n_per_bin  = max(1, int(np.sum(m)) // n_bins)
    sigma_bins = np.sqrt(np.maximum(frac_pos * (1 - frac_pos), 0) / n_per_bin)

    fig, ax = plt.subplots(figsize=METRIC_FIGSIZE)
    ax.plot([0, 1], [0, 1], color="0.10", ls="--", lw=1.8,
            label="Ideal")
    ax.errorbar(mean_pred, frac_pos, yerr=sigma_bins,
                fmt="o-", color=WATCH_COLOR, lw=2.3, ms=7,
                mec="white", mew=0.9, capsize=4.5, elinewidth=1.6,
                label=f"{title}  BSS={bss:.3f}")
    ax.set_xlabel("Mean predicted probability", fontsize=METRIC_LABEL_FS)
    ax.set_ylabel("Observed positive fraction", fontsize=METRIC_LABEL_FS)
    ax.set_title(f"Reliability — {title}", fontsize=METRIC_TITLE_FS, pad=9)
    ax.text(0.98, 0.04,
            f"Brier={brier:.4f}\nBSS={bss:.3f}\nbase={base_rate:.3f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9.0, color="0.20",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                      edgecolor="0.80", alpha=0.92))
    ax.grid(True, alpha=0.20)
    ax.legend(fontsize=METRIC_LEGEND_FS, frameon=True, framealpha=0.95,
              loc="upper left", borderpad=0.35, handlelength=2.4)
    _style_axes(ax, tick_fs=METRIC_TICK_FS)
    plt.tight_layout()
    plt.savefig(outpath, dpi=FIG_DPI, bbox_inches="tight")
    if SHOW_PLOTS: plt.show()
    plt.close(fig)
    print(f"  Reliability curve → {outpath}")
    return brier, bss


def plot_roc_curve(y_true, p, outpath, title, model_name,
                   thr_watch=None, thr_trigger=None):
    fpr_arr, tpr_arr, thresholds = roc_curve(y_true, p)
    roc_auc_val = float(sk_auc(fpr_arr, tpr_arr))

    fig, ax = plt.subplots(figsize=METRIC_FIGSIZE)
    ax.plot(fpr_arr, tpr_arr, color=WATCH_COLOR, lw=2.6,
            label=f"{model_name}  AUC={roc_auc_val:.3f}")
    ax.plot([0, 1], [0, 1], color="0.10", ls="--", lw=1.5)

    for thr_a, col, lbl in [
            (thr_watch,   "#d97706", f"WATCH  P≥{thr_watch:.2f}"   if thr_watch   else None),
            (thr_trigger, TRIGGER_COLOR, f"TRIGGER P≥{thr_trigger:.2f}" if thr_trigger else None),
    ]:
        if thr_a is not None and np.isfinite(thr_a):
            idx = np.argmin(np.abs(thresholds - thr_a))
            ax.scatter(fpr_arr[idx], tpr_arr[idx], s=110,
                       color=col, edgecolors="white", linewidths=0.9,
                       zorder=5, label=lbl)

    ax.set_xlabel("False Positive Rate", fontsize=METRIC_LABEL_FS)
    ax.set_ylabel("True Positive Rate", fontsize=METRIC_LABEL_FS)
    ax.set_title(f"ROC — {title}", fontsize=METRIC_TITLE_FS)
    ax.legend(fontsize=METRIC_LEGEND_FS, frameon=True, framealpha=0.95,
              loc="lower right", borderpad=0.35, handlelength=2.4)
    ax.grid(True, alpha=0.20)
    _style_axes(ax, tick_fs=METRIC_TICK_FS)
    plt.tight_layout()
    plt.savefig(outpath, dpi=FIG_DPI, bbox_inches="tight")
    if SHOW_PLOTS: plt.show()
    plt.close(fig)
    print(f"  ROC curve → {outpath}")
    return roc_auc_val


def plot_pr_curve(y_true, p, outpath, title, model_name,
                  thr_watch=None, thr_trigger=None):
    prec_arr, rec_arr, thresholds = precision_recall_curve(y_true, p)
    ap         = float(average_precision_score(y_true, p))
    base_rate  = float(np.mean(y_true))

    fig, ax = plt.subplots(figsize=METRIC_FIGSIZE)
    ax.plot(rec_arr, prec_arr, color=WATCH_COLOR, lw=2.6,
            label=f"{model_name}  AP={ap:.3f}")
    ax.axhline(base_rate, color="0.10", ls="--", lw=1.5,
               label=f"Random (base={base_rate:.2f})")

    for thr_a, col, lbl in [
            (thr_watch,   "#d97706", f"WATCH  P≥{thr_watch:.2f}"   if thr_watch   else None),
            (thr_trigger, TRIGGER_COLOR, f"TRIGGER P≥{thr_trigger:.2f}" if thr_trigger else None),
    ]:
        if thr_a is not None and np.isfinite(thr_a):
            idx = np.argmin(np.abs(thresholds - thr_a))
            if idx < len(rec_arr):
                ax.scatter(rec_arr[idx], prec_arr[idx], s=110,
                           color=col, edgecolors="white", linewidths=0.9,
                           zorder=5, label=lbl)

    ax.set_xlabel("Recall", fontsize=METRIC_LABEL_FS)
    ax.set_ylabel("Precision", fontsize=METRIC_LABEL_FS)
    ax.set_title(f"Precision-Recall — {title}", fontsize=METRIC_TITLE_FS)
    ax.legend(fontsize=METRIC_LEGEND_FS, frameon=True, framealpha=0.95,
              loc="lower left", borderpad=0.35, handlelength=2.4)
    ax.grid(True, alpha=0.20)
    _style_axes(ax, tick_fs=METRIC_TICK_FS)
    plt.tight_layout()
    plt.savefig(outpath, dpi=FIG_DPI, bbox_inches="tight")
    if SHOW_PLOTS: plt.show()
    plt.close(fig)
    print(f"  PR curve → {outpath}")
    return ap


def moving_average_causal(x, win=7):
    """Causal (trailing) moving average — no future look-ahead."""
    x = np.asarray(x, float)
    y = np.full_like(x, np.nan)
    for i in range(len(x)):
        a   = max(0, i - win + 1)
        seg = x[a:i + 1]
        m   = np.isfinite(seg)
        if np.any(m):
            y[i] = float(np.mean(seg[m]))
    return y


ONSET_COLOR = "#2b2b2b"   # neutral, distinct from TRIGGER_COLOR so onset
                          # markers don't blend with the TRIGGER curve/threshold


def plot_probability_timeline(T_end, p_series, y_series,
                               flare_onsets, flare_intervals,
                               outpath, title,
                               thr_single=None,
                               thr_watch=None, thr_trigger=None,
                               extra_hline=None,
                               shade_test_region=True,
                               test_boundary_mjd=None,
                               t_lc=None, flux_lc=None,
                               lead_time_rows=None,
                               p_trigger_series=None,
                               y_trigger_series=None):
    """Plot the flux and forecast-probability timeline.

    Three separate panels (flux / WATCH / TRIGGER) are used whenever a
    TRIGGER series is supplied, so the two scores, their thresholds, and
    the onset markers no longer compete visually in a single panel. If no
    TRIGGER series is given (e.g., the WATCH-only ensemble), a two-panel
    flux/WATCH layout is used instead.
    """
    T_end         = np.asarray(T_end,     float)
    p_series      = np.asarray(p_series,  float)
    p_trigger_series = (np.asarray(p_trigger_series, float)
                        if p_trigger_series is not None else None)
    flare_onsets  = np.asarray(flare_onsets, float) \
                    if flare_onsets  is not None else np.array([])
    flare_intervals = np.asarray(flare_intervals, float) \
                      if flare_intervals is not None else np.zeros((0, 2))

    has_flux    = (t_lc is not None and flux_lc is not None)
    has_trigger = (p_trigger_series is not None)
    n_panels    = int(has_flux) + 1 + int(has_trigger)

    fig, axes = plt.subplots(n_panels, 1,
                             figsize=(TIMELINE_FIG_WIDTH,
                                      TIMELINE_PANEL_HEIGHT * n_panels),
                             sharex=True)
    if n_panels == 1:
        axes = [axes]
    idx = 0

    # ─── flux panel ───
    if has_flux:
        ax0 = axes[idx]; idx += 1
        ax0.plot(t_lc, flux_lc, ".", color="0.10", ms=2.6, alpha=0.55,
                 label="Fermi flux")
        for k, (s, e) in enumerate(flare_intervals):
            ax0.axvspan(s, e, alpha=0.18, color=FLARE_FILL_COLOR,
                        label="Flare block" if k == 0 else "")
        if test_boundary_mjd is not None and np.isfinite(test_boundary_mjd):
            ax0.axvline(test_boundary_mjd, color=WATCH_COLOR, ls="--",
                        lw=2.0, label="Train/Test")
        ax0.set_ylabel("Flux (ph cm⁻² s⁻¹)", fontsize=TIMELINE_LABEL_FS)
        ax0.legend(fontsize=TIMELINE_LEGEND_FS, loc="upper left",
                   frameon=True, framealpha=0.95,
                   borderpad=0.35, handlelength=2.2)
        ax0.set_title(title, fontsize=TIMELINE_TITLE_FS)
        _style_axes(ax0, tick_fs=TIMELINE_TICK_FS, spine_lw=1.2)

    def _shade_common(ax):
        if shade_test_region and test_boundary_mjd is not None \
                and np.isfinite(test_boundary_mjd):
            ax.axvspan(float(test_boundary_mjd), float(np.nanmax(T_end)),
                       alpha=0.08, color=TEST_SHADE_COLOR, label="TEST")
        for k, (s, e) in enumerate(flare_intervals):
            ax.axvspan(s, e, alpha=0.10, color=FLARE_FILL_COLOR,
                       label="Flare interval" if k == 0 else "")

    def _onsets(ax, labeled):
        for k, t0 in enumerate(flare_onsets):
            ax.axvline(float(t0), ls="--", color=ONSET_COLOR, lw=1.4,
                       label=("Onset" if (labeled and k == 0) else ""))

    def _legend(ax, loc, anchor=None):
        kw = dict(fontsize=TIMELINE_LEGEND_FS, ncol=3, frameon=True,
                  framealpha=0.95, borderpad=0.35, handlelength=2.2,
                  columnspacing=1.1, labelspacing=0.35)
        if anchor is not None:
            ax.legend(loc=loc, bbox_to_anchor=anchor, **kw)
        else:
            ax.legend(loc=loc, **kw)

    # ─── WATCH panel ───
    ax_w = axes[idx]; idx += 1
    _shade_common(ax_w)
    mf = np.isfinite(p_series)
    ax_w.plot(T_end[mf], p_series[mf], color=WATCH_COLOR, lw=2.6,
              label=f"WATCH ({HORIZON_DAYS:.0f} d)")
    if y_series is not None:
        ys = np.asarray(y_series, int)
        mp = (ys == 1) & mf
        if np.any(mp):
            ax_w.scatter(T_end[mp], p_series[mp], s=42,
                         color=WATCH_MARKER_COLOR, edgecolors="white",
                         linewidths=0.8, zorder=4, label="WATCH label")
    _onsets(ax_w, labeled=True)
    if thr_watch is not None and np.isfinite(thr_watch):
        ax_w.axhline(thr_watch, color="#d97706", ls=":", lw=2.0,
                     label=f"WATCH thr={thr_watch:.2f}")
    if thr_single is not None and np.isfinite(thr_single):
        ax_w.axhline(thr_single, color=GUIDE_COLOR, ls=":", lw=1.5,
                     label=f"P={thr_single:.2f}")
    if extra_hline is not None and np.isfinite(extra_hline):
        ax_w.axhline(extra_hline, color=GUIDE_COLOR, ls="-.", lw=1.2)
    ax_w.set_ylim(-0.05, 1.10)
    ax_w.set_ylabel("WATCH probability" if has_trigger
                     else "Forecast probability", fontsize=TIMELINE_LABEL_FS)
    ax_w.grid(True, alpha=0.18)
    _style_axes(ax_w, tick_fs=TIMELINE_TICK_FS, spine_lw=1.2)
    if not has_flux:
        ax_w.set_title(title, fontsize=TIMELINE_TITLE_FS)
    _legend(ax_w, "lower center" if has_flux else "upper center",
            anchor=(0.5, 1.01) if has_flux else None)

    # ─── TRIGGER panel (only when a TRIGGER series is supplied) ───
    if has_trigger:
        ax_t = axes[idx]; idx += 1
        _shade_common(ax_t)
        mt = np.isfinite(p_trigger_series)
        ax_t.plot(T_end[mt], p_trigger_series[mt], color=TRIGGER_COLOR,
                  lw=2.6, label=f"TRIGGER ({TRIGGER_HORIZON_DAYS:.0f} d)")
        if y_trigger_series is not None:
            yt = np.asarray(y_trigger_series, int)
            mp_t = (yt == 1) & mt
            if np.any(mp_t):
                ax_t.scatter(T_end[mp_t], p_trigger_series[mp_t], s=48,
                             facecolors="none", edgecolors=TRIGGER_COLOR,
                             lw=1.5, zorder=4, label="TRIGGER label")
        _onsets(ax_t, labeled=True)
        if thr_trigger is not None and np.isfinite(thr_trigger):
            ax_t.axhline(thr_trigger, color=TRIGGER_COLOR, ls="--", lw=2.0,
                         label=f"TRIGGER thr={thr_trigger:.2f}")
        if thr_single is not None and np.isfinite(thr_single):
            ax_t.axhline(thr_single, color=GUIDE_COLOR, ls=":", lw=1.5)
        if extra_hline is not None and np.isfinite(extra_hline):
            ax_t.axhline(extra_hline, color=GUIDE_COLOR, ls="-.", lw=1.2)
        ax_t.set_ylim(-0.05, 1.10)
        ax_t.set_ylabel("TRIGGER probability", fontsize=TIMELINE_LABEL_FS)
        ax_t.grid(True, alpha=0.18)
        _style_axes(ax_t, tick_fs=TIMELINE_TICK_FS, spine_lw=1.2)
        _legend(ax_t, "lower center", anchor=(0.5, 1.01))

    # lead-time annotations go on the last panel (TRIGGER if present, else WATCH)
    ax_last = axes[-1]
    if lead_time_rows:
        for row in lead_time_rows:
            tc    = row.get("t_cross",   np.nan)
            onset = row.get("onset",     np.nan)
            lead  = row.get("lead_days", np.nan)
            if np.isfinite(tc) and np.isfinite(lead):
                ax_last.annotate(f"{lead:.0f}d",
                            xy=(onset, 1.0), xytext=(tc, 0.93),
                            fontsize=11, color="0.35",
                            arrowprops=dict(arrowstyle="->", color="0.45",
                                            lw=1.0))

    ax_last.set_xlabel("Window end T_end (MJD)", fontsize=TIMELINE_LABEL_FS)

    plt.tight_layout(h_pad=1.2)
    plt.savefig(outpath, dpi=FIG_DPI, bbox_inches="tight")
    if SHOW_PLOTS: plt.show()
    plt.close(fig)
    print(f"  Timeline → {outpath}")


def save_timeline_probabilities_csv(outpath, T_end, y_watch, y_trigger,
                                    split_is_test,
                                    p_watch_raw, p_watch_smooth,
                                    p_trigger_raw=None, p_trigger_smooth=None,
                                    thr_watch=None, thr_trigger=None):
    header = ["T_end_mjd", "Y_watch_eval", "Y_trigger_eval", "is_test",
              "proba_watch_raw", "proba_watch_smooth",
              "proba_trigger_raw", "proba_trigger_smooth",
              "state_normal", "state_watch", "state_trigger"]
    with open(outpath, "w") as fh:
        fh.write(",".join(header) + "\n")
        n = len(T_end)
        if p_trigger_raw is None:
            p_trigger_raw = np.full(n, np.nan)
        if p_trigger_smooth is None:
            p_trigger_smooth = np.full(n, np.nan)

        for t_v, yw_v, yt_v, it_v, pw_v, pws_v, pt_v, pts_v in zip(
                T_end, y_watch, y_trigger, split_is_test,
                p_watch_raw, p_watch_smooth,
                p_trigger_raw, p_trigger_smooth):
            is_trig = 0
            is_watch = 0
            is_norm = 1
            if thr_trigger is not None and np.isfinite(thr_trigger) and np.isfinite(pt_v):
                is_trig = int(pt_v >= thr_trigger)
            if (not is_trig and thr_watch is not None and np.isfinite(thr_watch)
                    and np.isfinite(pw_v)):
                is_watch = int(pw_v >= thr_watch)
            if is_trig or is_watch:
                is_norm = 0
            fh.write(
                f"{float(t_v):.8g},{int(yw_v)},{int(yt_v)},{int(it_v)},"
                f"{'nan' if np.isnan(pw_v)  else f'{float(pw_v):.8g}'},"
                f"{'nan' if np.isnan(pws_v) else f'{float(pws_v):.8g}'},"
                f"{'nan' if np.isnan(pt_v)  else f'{float(pt_v):.8g}'},"
                f"{'nan' if np.isnan(pts_v) else f'{float(pts_v):.8g}'},"
                f"{is_norm},{is_watch},{is_trig}\n"
            )
    print(f"  Timeline CSV → {outpath}")

def export_training_dataset_csv(all_data, keys_ref, train_end_boundary, outdir):
    """
    Export the full per-window feature/label table for every source, for
    Zenodo/data-availability archiving. One row per rolling window.
    """
    outpath = Path(outdir) / "training_dataset_all_sources.csv"
    header = (["source", "T_end_mjd", "split"]
              + list(keys_ref)
              + ["Y_watch", "Y_trigger"])

    with open(outpath, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for src_name, d in all_data.items():
            if "X" not in d or "Y_watch" not in d:
                continue
            X_s      = d["X"]
            T_end_s  = d["T_end"]
            Yw_s     = d["Y_watch"]
            Yt_s     = d["Y_trigger"]
            for i in range(len(T_end_s)):
                split = "TRAIN" if T_end_s[i] <= train_end_boundary else "TEST"
                row = ([src_name, f"{T_end_s[i]:.6f}", split]
                       + [f"{v:.8g}" for v in X_s[i]]
                       + [int(Yw_s[i]), int(Yt_s[i])])
                writer.writerow(row)

    print(f"  Training dataset CSV → {outpath}")
    return outpath
# ============================================================
# 8) DUAL-THRESHOLD GRID SEARCH
# ============================================================

def dual_threshold_grid_search(y_true, p, grid=None,
                                min_watch_recall=0.85,
                                max_watch_fpr=0.40,
                                min_trigger_precision=0.80,
                                max_trigger_fraction=None,
                                max_watch_only_fraction=None,
                                min_gap=0.05,
                                w_watch_recall=1.0,
                                w_watch_precision=0.2,
                                w_trigger_precision=1.0,
                                w_trigger_recall=0.2,
                                w_penalty_trigger_frac=0.5,
                                w_penalty_watch_frac=0.2):
    y_true = np.asarray(y_true).astype(int)
    p      = np.asarray(p, float)
    m      = np.isfinite(p) & np.isfinite(y_true)
    y  = y_true[m];  pp = p[m]

    if y.size < 10 or len(np.unique(y)) < 2:
        return None, None, {"note": "insufficient data"}, []

    if grid is None:
        grid = THRESH_GRID
    grid = np.asarray(grid, float)

    n     = y.size
    n_pos = int(np.sum(y == 1))
    n_neg = n - n_pos
    rows  = [];  best = None;  best_score = -np.inf

    for thr_w in grid:
        watch_pred = (pp >= thr_w).astype(int)
        tp_w = int(np.sum((watch_pred==1)&(y==1)))
        fp_w = int(np.sum((watch_pred==1)&(y==0)))
        tn_w = int(np.sum((watch_pred==0)&(y==0)))
        fn_w = int(np.sum((watch_pred==0)&(y==1)))
        w_rec = tp_w/(tp_w+fn_w) if (tp_w+fn_w)>0 else 0.0
        w_pre = tp_w/(tp_w+fp_w) if (tp_w+fp_w)>0 else 0.0
        w_fpr = fp_w/(fp_w+tn_w) if (fp_w+tn_w)>0 else 0.0
        if w_rec < min_watch_recall or w_fpr > max_watch_fpr:
            continue

        for thr_t in grid:
            if thr_t < (thr_w + min_gap):
                continue
            trig_pred = (pp >= thr_t).astype(int)
            tp_t = int(np.sum((trig_pred==1)&(y==1)))
            fp_t = int(np.sum((trig_pred==1)&(y==0)))
            tn_t = int(np.sum((trig_pred==0)&(y==0)))
            fn_t = int(np.sum((trig_pred==0)&(y==1)))
            t_rec = tp_t/(tp_t+fn_t) if (tp_t+fn_t)>0 else 0.0
            t_pre = tp_t/(tp_t+fp_t) if (tp_t+fp_t)>0 else 0.0
            if t_pre < min_trigger_precision:
                continue

            f_trig  = float(np.mean(pp >= thr_t))
            f_watch = float(np.mean((pp >= thr_w) & (pp < thr_t)))
            f_norm  = float(np.mean(pp < thr_w))

            if max_trigger_fraction and f_trig > max_trigger_fraction:
                continue
            if max_watch_only_fraction and f_watch > max_watch_only_fraction:
                continue

            score = (w_watch_recall      * w_rec
                   + w_watch_precision   * w_pre
                   + w_trigger_precision * t_pre
                   + w_trigger_recall    * t_rec
                   - w_penalty_trigger_frac * f_trig
                   - w_penalty_watch_frac   * f_watch)

            row = {
                "thr_watch": float(thr_w), "thr_trigger": float(thr_t),
                "watch_recall": w_rec, "watch_precision": w_pre,
                "watch_fpr": w_fpr, "trigger_recall": t_rec,
                "trigger_precision": t_pre,
                "frac_normal": f_norm, "frac_watch": f_watch,
                "frac_trigger": f_trig, "score": score,
                "n": n, "pos": n_pos,
            }
            rows.append(row)
            if score > best_score:
                best_score = score;  best = row

    rows_sorted = sorted(rows, key=lambda r: r["score"], reverse=True)
    if best is None:
        return (None, None,
                {"note": "no pair satisfied constraints", "n": n, "pos": n_pos},
                rows_sorted)
    return best["thr_watch"], best["thr_trigger"], best, rows_sorted


# ============================================================
# 9) STATISTICAL TESTS
# ============================================================

def estimate_label_acf_length(y, step_days=7.0):
    """Lag at which label ACF first drops below 1/e (in days)."""
    y = np.asarray(y, float) - np.mean(y)
    acf = np.correlate(y, y, mode='full')
    acf = acf[len(acf)//2:]
    if acf[0] == 0:
        return np.nan
    acf = acf / acf[0]
    below = np.where(acf < np.exp(-1))[0]
    if len(below) == 0:
        return float(len(y) * step_days)
    return float(below[0] * step_days)


def purged_walk_forward_auc(model_name, X_train, y_train,
                             train_min=80, test_chunk=45,
                             step=15, purge_len=6):
    X_train = np.asarray(X_train, float)
    y_train = np.asarray(y_train).astype(int)
    n       = len(y_train)
    n_pos   = int(np.sum(y_train))
    n_neg   = n - n_pos
    p_oof   = np.full(n, np.nan, float)
    aucs    = [];  folds_used = 0

    i = int(train_min)
    while i < n:
        j         = min(n, i + int(test_chunk))
        train_end = max(0, i - int(purge_len))
        if train_end < 20:
            i += int(step);  continue
        y_fit = y_train[:train_end]
        if len(np.unique(y_fit)) < 2:
            i += int(step);  continue
        y_val = y_train[i:j]
        try:
            mdl = make_model(model_name, n_pos=n_pos, n_neg=n_neg)
            mdl.fit(X_train[:train_end], y_fit)
            pv  = mdl.predict_proba(X_train[i:j])[:, 1]
            p_oof[i:j] = pv
            if len(np.unique(y_val)) == 2:
                aucs.append(float(roc_auc_score(y_val, pv)))
            folds_used += 1
        except Exception:
            pass
        i += int(step)

    m_ok   = np.isfinite(p_oof)
    pooled = np.nan
    if np.sum(m_ok) > 10 and len(np.unique(y_train[m_ok])) == 2:
        pooled = float(roc_auc_score(y_train[m_ok], p_oof[m_ok]))

    return aucs, pooled, p_oof, folds_used


def _block_shuffle_labels(y, block_len, rng):
    y = np.asarray(y).astype(int)
    n = len(y)
    L = max(1, int(block_len))
    blocks = [y[s:min(n, s+L)] for s in range(0, n, L)]
    rng.shuffle(blocks)
    return np.concatenate(blocks)[:n]


def permutation_test_auc_blockshuffle(y_true, p, n_perm=5000,
                                       block_len=9, seed=0):
    y_true = np.asarray(y_true).astype(int)
    p      = np.asarray(p, float)
    m      = np.isfinite(p) & np.isfinite(y_true)
    y  = y_true[m];  pp = p[m]
    if y.size < 10 or len(np.unique(y)) < 2:
        return np.nan, {"note": "insufficient test data"}

    auc_obs = float(roc_auc_score(y, pp))
    rng     = np.random.default_rng(int(seed))
    count   = 0;  used = 0

    for _ in range(int(n_perm)):
        y_perm = _block_shuffle_labels(y, block_len, rng)
        if len(np.unique(y_perm)) < 2:
            continue
        if float(roc_auc_score(y_perm, pp)) >= auc_obs:
            count += 1
        used += 1

    if used < 0.80 * n_perm:
        print(f"  [PERM] WARNING: only {used}/{n_perm} valid permutations "
              f"— p-value may be unreliable.")
    pval = (count + 1) / (used + 1) if used > 0 else np.nan
    return pval, {"auc_obs": auc_obs, "n_perm_used": used,
                  "block_len": int(block_len)}


def block_bootstrap_auc_ci(y_true, p, block_len=9, n_boot=3000,
                            seed=0, alpha=0.05):
    """Circular block bootstrap for dependence-aware AUC CI."""
    y_true = np.asarray(y_true).astype(int)
    p      = np.asarray(p, float)
    m      = np.isfinite(p) & np.isfinite(y_true)
    y  = y_true[m];  pp = p[m];  n = len(y)
    if n < 10 or len(np.unique(y)) < 2:
        return (np.nan, np.nan, np.nan), {"note": "insufficient test data"}

    auc_obs = float(roc_auc_score(y, pp))
    rng     = np.random.default_rng(int(seed))
    L       = max(1, int(block_len))
    aucs    = []

    for _ in range(int(n_boot)):
        idxs = []
        while len(idxs) < n:
            start = int(rng.integers(0, n))   # circular start
            block = [(start + k) % n for k in range(L)]
            idxs.extend(block)
        idxs = np.asarray(idxs[:n], int)
        yb   = y[idxs];  pb = pp[idxs]
        if len(np.unique(yb)) < 2:
            continue
        aucs.append(float(roc_auc_score(yb, pb)))

    if len(aucs) < 50:
        return (auc_obs, np.nan, np.nan), {"note": "too few valid bootstraps"}

    lo = float(np.quantile(aucs, alpha / 2))
    hi = float(np.quantile(aucs, 1 - alpha / 2))
    return (auc_obs, lo, hi), {"n_boot_valid": len(aucs),
                               "block_len": int(block_len)}


def block_length_sensitivity(y_true, p, block_lens=(5, 9, 15),
                              n_boot=500, seed=0):
    """Run bootstrap CI for multiple block lengths as a sensitivity check."""
    print("  [BLOCK SENSITIVITY]")
    for bl in block_lens:
        (_, lo, hi), _ = block_bootstrap_auc_ci(y_true, p,
                                                 block_len=bl,
                                                 n_boot=n_boot,
                                                 seed=seed)
        print(f"    block_len={bl:2d}: 95% CI = [{lo:.3f}, {hi:.3f}]")


def mcnemar_model_comparison(y_true, pred_a, pred_b):
    """McNemar test: are model A and B errors significantly different?"""
    b = int(np.sum((pred_a == y_true) & (pred_b != y_true)))
    c = int(np.sum((pred_a != y_true) & (pred_b == y_true)))
    n = b + c
    if n == 0:
        return np.nan, b, c
    if n < 25:
        # exact binomial
        from scipy.stats import binom_test
        pval = float(binom_test(min(b, c), n, 0.5, alternative="two-sided"))
    else:
        # chi-square approximation
        pval = float(1.0 - __import__("scipy").stats.chi2.cdf(
            (abs(b - c) - 1)**2 / (b + c), df=1))
    return pval, b, c


def diebold_mariano_test(y_true, p_a, p_b, lags=None):
    """
    DM test: does model A have lower Brier score than model B?
    H0: no difference. Returns (stat, pval).
    """
    from scipy.stats import norm as sp_norm
    y  = np.asarray(y_true, float)
    pa = np.asarray(p_a,    float)
    pb = np.asarray(p_b,    float)
    m  = np.isfinite(y) & np.isfinite(pa) & np.isfinite(pb)
    y, pa, pb = y[m], pa[m], pb[m]
    d     = (pa - y)**2 - (pb - y)**2
    n     = len(d)
    d_bar = np.mean(d)
    if lags is None:
        lags = int(np.ceil(4 * (n / 100) ** (2 / 9)))
    gamma_0 = np.var(d, ddof=1)
    cov_sum = sum(
        (1 - k/(lags+1)) * np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        for k in range(1, lags + 1)
    )
    lr_var  = gamma_0 + 2 * cov_sum
    if lr_var <= 0:
        return np.nan, np.nan
    dm_stat = d_bar / np.sqrt(lr_var / n)
    pval    = float(2 * (1 - sp_norm.cdf(abs(dm_stat))))
    return float(dm_stat), pval


def bootstrap_lead_time_ci(lead_days, n_boot=2000, seed=0, alpha=0.05):
    """Bootstrap CI on median lead time."""
    lead = np.asarray(lead_days, float)
    lead = lead[np.isfinite(lead)]
    if len(lead) < 2:
        return float(np.nanmedian(lead_days)), np.nan, np.nan
    rng  = np.random.default_rng(seed)
    meds = [float(np.median(rng.choice(lead, size=len(lead), replace=True)))
            for _ in range(n_boot)]
    return (float(np.median(lead)),
            float(np.quantile(meds, alpha / 2)),
            float(np.quantile(meds, 1 - alpha / 2)))


# ============================================================
# 10) LEAD-TIME ANALYSIS
# ============================================================

def _find_runs(mask):
    runs = [];  n = len(mask);  i = 0
    while i < n:
        if mask[i]:
            j = i + 1
            while j < n and mask[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def compute_lead_times_to_onsets(T_end, proba_series, flare_onsets,
                                  threshold, lookback_days=90.0,
                                  mode="latest", min_consecutive=1,
                                  cooldown_windows=1, max_lead_days=None,
                                  allowed_window_mask=None,
                                  onset_min=None, onset_max=None):
    T_end        = np.asarray(T_end,       float)
    p            = np.asarray(proba_series, float)
    flare_onsets = np.asarray(flare_onsets, float)

    if allowed_window_mask is None:
        allowed_window_mask = np.ones_like(T_end, dtype=bool)
    else:
        allowed_window_mask = np.asarray(allowed_window_mask, bool)

    step_days    = float(np.nanmedian(np.diff(T_end))) if len(T_end) > 2 else np.nan
    thr          = float(threshold)
    rows         = []

    def cap_mask(times, onset):
        if max_lead_days is None or not np.isfinite(max_lead_days):
            return np.ones_like(times, dtype=bool)
        return (float(onset) - times) <= float(max_lead_days)

    def nan_row(onset):
        rows.append({"onset": float(onset), "t_cross": np.nan,
                     "lead_days": np.nan, "p_at_cross": np.nan,
                     "mode": str(mode), "threshold": thr,
                     "lookback_days": float(lookback_days),
                     "min_consecutive": int(min_consecutive),
                     "cooldown_windows": int(cooldown_windows),
                     "step_days_est": float(step_days)})

    for onset in flare_onsets:
        onset = float(onset)
        if onset_min is not None and onset < float(onset_min): continue
        if onset_max is not None and onset > float(onset_max): continue

        t0   = onset - float(lookback_days)
        mwin = ((T_end >= t0) & (T_end <= onset)
                & np.isfinite(p) & allowed_window_mask)
        if not np.any(mwin):
            nan_row(onset); continue

        idxs = np.where(mwin)[0]
        mask = (p[idxs] >= thr)
        runs = [(a, b) for a, b in _find_runs(mask)
                if (b - a) >= int(min_consecutive)]
        if not runs:
            nan_row(onset); continue

        def run_times(run):
            ii = idxs[run[0]:run[1]]
            return T_end[ii], p[ii]

        def pick(run, which):
            times, probs = run_times(run)
            ok = cap_mask(times, onset)
            if not np.any(ok):
                return np.nan, np.nan
            j = np.where(ok)[0][0 if which == "earliest" else -1]
            return float(times[j]), float(probs[j])

        t_cross = p_cross = np.nan

        if mode == "earliest":
            for r in runs:
                tc, pc = pick(r, "earliest")
                if np.isfinite(tc):
                    t_cross, p_cross = tc, pc;  break

        elif mode == "latest":
            for r in runs[::-1]:
                tc, pc = pick(r, "latest")
                if np.isfinite(tc):
                    t_cross, p_cross = tc, pc;  break

        elif mode == "episode_start":
            anchor = None
            for r in runs[::-1]:
                tc_l, _ = pick(r, "latest")
                if np.isfinite(tc_l):
                    anchor = r;  break
            if anchor is not None:
                a_a, b_a   = anchor
                k          = a_a - 1
                cool       = 0
                start_idx  = a_a
                while k >= 0:
                    if mask[k]:
                        cool = 0;  start_idx = k
                    else:
                        cool += 1
                        if cool >= cooldown_windows:
                            break
                    k -= 1
                ep_ii  = idxs[start_idx:b_a]
                ep_t   = T_end[ep_ii]
                ep_p   = p[ep_ii]
                ok_ep  = cap_mask(ep_t, onset)
                if np.any(ok_ep):
                    j       = np.where(ok_ep)[0][0]
                    t_cross = float(ep_t[j])
                    p_cross = float(ep_p[j])

        lead = float(onset - t_cross) if np.isfinite(t_cross) else np.nan
        rows.append({"onset": onset, "t_cross": t_cross,
                     "lead_days": lead, "p_at_cross": p_cross,
                     "mode": mode, "threshold": thr,
                     "lookback_days": float(lookback_days),
                     "min_consecutive": int(min_consecutive),
                     "cooldown_windows": int(cooldown_windows),
                     "step_days_est": float(step_days)})
    return rows


def print_lead_time_summary(rows, threshold, lookback_days, tag=""):
    leads  = np.array([r.get("lead_days", np.nan) for r in rows], float)
    onsets = np.array([r.get("onset",     np.nan) for r in rows], float)
    tc_arr = np.array([r.get("t_cross",   np.nan) for r in rows], float)
    pc_arr = np.array([r.get("p_at_cross",np.nan) for r in rows], float)
    m      = np.isfinite(leads)

    print(f"\n  ── LEAD-TIME {tag} (thr={threshold:.2f}, lb={lookback_days:.0f}d) ──")
    print(f"  Onsets: {len(rows)}  |  with crossing: {int(np.sum(m))}")
    if np.sum(m) == 0:
        print("  No valid crossings."); return

    med, lo_ci, hi_ci = bootstrap_lead_time_ci(leads[m])
    print(f"  Lead (days): median={med:.1f} 95%CI=[{lo_ci:.1f},{hi_ci:.1f}]  "
          f"mean={np.mean(leads[m]):.1f}  "
          f"min={np.min(leads[m]):.1f}  max={np.max(leads[m]):.1f}")
    for i in range(len(rows)):
        if np.isfinite(leads[i]):
            print(f"    onset={onsets[i]:.1f}  t_cross={tc_arr[i]:.1f}  "
                  f"lead={leads[i]:.1f}d  p={pc_arr[i]:.3f}")


def save_lead_time_csv(outpath, rows):
    header = ["onset_mjd", "t_cross_mjd", "lead_days", "p_at_cross",
              "mode", "threshold", "lookback_days",
              "min_consecutive", "cooldown_windows", "step_days_est"]
    with open(outpath, "w") as fh:
        fh.write(",".join(header) + "\n")
        for r in rows:
            def _v(k):
                v = r.get(k, np.nan)
                return "" if (isinstance(v, float) and np.isnan(v)) else str(v)
            fh.write(
                f"{_v('onset')},{_v('t_cross')},{_v('lead_days')},"
                f"{_v('p_at_cross')},{_v('mode')},{_v('threshold')},"
                f"{_v('lookback_days')},{_v('min_consecutive')},"
                f"{_v('cooldown_windows')},{_v('step_days_est')}\n"
            )
    print(f"  Lead-time CSV → {outpath}")


def plot_lead_time_distribution(rows_watch, rows_trigger, outpath, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, rows, label, col in [
            (axes[0], rows_watch,   "WATCH",   "orange"),
            (axes[1], rows_trigger, "TRIGGER", "red"),
    ]:
        leads = np.array([r.get("lead_days", np.nan) for r in rows], float)
        leads = leads[np.isfinite(leads)]
        if len(leads) == 0:
            ax.text(0.5, 0.5, "No detections", ha="center", va="center",
                    transform=ax.transAxes)
        else:
            ax.hist(leads, bins=min(10, len(leads)),
                    color=col, alpha=0.7, edgecolor="k")
            ax.axvline(np.median(leads), color="k", ls="--", lw=1.5,
                       label=f"Median={np.median(leads):.0f}d")
            ax.axvline(np.mean(leads), color="gray", ls=":", lw=1.2,
                       label=f"Mean={np.mean(leads):.0f}d")
            ax.legend(fontsize=9)
        ax.set_xlabel("Lead time (days)", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(f"{label} lead times | {model_name}", fontsize=10)
        ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    if SHOW_PLOTS: plt.show()
    plt.close()
    print(f"  Lead-time distribution → {outpath}")


def print_detection_table(rows_watch, rows_trigger, model_name):
    all_onsets = sorted(set(
        [r["onset"] for r in rows_watch] +
        [r["onset"] for r in rows_trigger]
        if rows_watch or rows_trigger else []
    ))
    if not all_onsets:
        return
    wd = {r["onset"]: r for r in rows_watch}
    td = {r["onset"]: r for r in rows_trigger}
    print(f"\n  {'='*62}")
    print(f"  PER-FLARE DETECTION TABLE — {model_name}")
    print(f"  {'Onset (MJD)':<14} {'WATCH lead (d)':<18} {'TRIGGER lead (d)'}")
    print(f"  {'='*62}")
    nw = nt = 0
    for onset in all_onsets:
        wl = wd.get(onset, {}).get("lead_days", np.nan)
        tl = td.get(onset, {}).get("lead_days", np.nan)
        ws = f"{wl:.1f}" if np.isfinite(wl) else "NOT DETECTED"
        ts = f"{tl:.1f}" if np.isfinite(tl) else "NOT DETECTED"
        if np.isfinite(wl): nw += 1
        if np.isfinite(tl): nt += 1
        print(f"  {onset:<14.1f} {ws:<18} {ts}")
    n = len(all_onsets)
    print(f"  {'='*62}")
    print(f"  WATCH   detected {nw}/{n} ({100*nw/n:.0f}%)  |  "
          f"TRIGGER detected {nt}/{n} ({100*nt/n:.0f}%)")


# ============================================================
# 11) MAIN
# ============================================================

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    t_wall = _time.time()

    # ================================================================
    # STEP 1: LOAD ALL SOURCES
    # ================================================================
    print("\n" + "="*62)
    print("STEP 1 — LOADING LIGHT CURVES")
    print("="*62)

    source_files = {"target": LC_FILE}
    source_files.update(EXTRA_TRAIN_FILES)

    all_data = {}
    for src_name, src_file in source_files.items():
        print(f"\n[{src_name}]  {src_file}")
        try:
            d = load_lightcurve(
                src_file,
                time_col=TIME_COL, ts_col=TS_COL,
                flux_col=FLUX_COL, fluxerr_col=FLUXERR_COL,
                index_col=INDEX_COL, indexerr_col=INDEXERR_COL,
                ts_min=TS_MIN,
                remove_negative_flux=REMOVE_NEGATIVE_FLUX,
            )
            print(f"  Loaded {len(d['t'])} bins  "
                  f"[{d['t'].min():.1f}–{d['t'].max():.1f} MJD]")
            sanity_check(d["t"], d["flux"], d["flux_err"], ts=d.get("ts"))
            n_idx = int(np.sum(np.isfinite(d["index"])))
            print(f"  Spectral index valid: {n_idx}/{len(d['index'])} "
                  f"({100*n_idx/max(len(d['index']),1):.0f}%)")
            all_data[src_name] = d
        except Exception as exc:
            print(f"  ERROR: {exc} — skipping {src_name}")

    if "target" not in all_data:
        raise RuntimeError("Target source failed to load. Exiting.")

    # ================================================================
    # STEP 2: PER-SOURCE BB + FEATURE EXTRACTION
    # ================================================================
    print("\n" + "="*62)
    print("STEP 2 — BAYESIAN BLOCKS + FEATURES")
    print("="*62)

    keys_ref = None   # reference feature column order

    for src_name, d in all_data.items():
        print(f"\n--- {src_name} ---")
        t_s  = d["t"];   f_s  = d["flux"]
        fe_s = d["flux_err"];  idx_s = d["index"]

        m_tr_s = (t_s <= BACKTEST_CUTOFF_MJD)
        f_tr_s = f_s[m_tr_s]

        # per-source quiescent level (from TRAIN data only)
        F_q_s = np.nanpercentile(
            f_tr_s[np.isfinite(f_tr_s)], F_QUIESCENT_PERCENTILE)
        if not (np.isfinite(F_q_s) and F_q_s > 0):
            F_q_s = float(np.nanmedian(f_tr_s[np.isfinite(f_tr_s)]))

        idx_q_s = float(np.nanmedian(idx_s[m_tr_s & np.isfinite(idx_s)])) \
                  if np.any(np.isfinite(idx_s[m_tr_s])) else np.nan

        global_tail_s = np.nanpercentile(
            f_tr_s[np.isfinite(f_tr_s)], 95.0)

        print(f"  F_quiescent = {F_q_s:.3e}  "
              f"Γ_quiescent = {idx_q_s:.2f}  "
              f"tail_p95 = {global_tail_s:.3e}")

        # ---- FULL BB ----
        (efull, bmfull, bsfull, isfl_full,
         onsets_full, intv_full, thr_full) = bayes_blocks_flares(
            t_s, f_s, fe_s, p0=BB_P0,
            flare_percentile=FLARE_BLOCK_PERCENTILE,
            min_block_days=MIN_FLARE_BLOCK_DAYS,
            n_sigma_flare=BB_N_SIGMA_FLARE)
        if len(intv_full) > 0:
            intv_full = merge_adjacent_flare_intervals(
                intv_full, MERGE_FLARE_GAP_DAYS)
        # recompute onsets from merged intervals
        onsets_full = np.array([iv[0] for iv in intv_full], float) \
                      if len(intv_full) > 0 else np.array([], float)

        print(f"  FULL BB: {len(intv_full)} flare intervals  "
              f"thr={thr_full:.3e}")
        for i_fl, (s, e) in enumerate(intv_full):
            print(f"    #{i_fl+1}: MJD {s:.1f}→{e:.1f}  ({e-s:.0f}d)")
        save_bb_diagnostics(efull, bmfull, isfl_full, thr_full,
                            tag=f"{src_name}_full", time=t_s, flux=f_s)

        # ---- TRAIN BB ----
        t_tr_s  = t_s[m_tr_s];  fe_tr_s = fe_s[m_tr_s]
        f_tr_arr = f_s[m_tr_s]
        (etr, bmtr, bstr, isfl_tr,
         onsets_tr, intv_tr, thr_tr) = bayes_blocks_flares(
            t_tr_s, f_tr_arr, fe_tr_s, p0=BB_P0,
            flare_percentile=FLARE_BLOCK_PERCENTILE,
            min_block_days=MIN_FLARE_BLOCK_DAYS,
            n_sigma_flare=BB_N_SIGMA_FLARE)
        if len(intv_tr) > 0:
            intv_tr = merge_adjacent_flare_intervals(
                intv_tr, MERGE_FLARE_GAP_DAYS)
        onsets_tr = np.array([iv[0] for iv in intv_tr], float) \
                    if len(intv_tr) > 0 else np.array([], float)

        print(f"  TRAIN BB: {len(intv_tr)} flare intervals  "
              f"thr={thr_tr:.3e}")
        save_bb_diagnostics(etr, bmtr, isfl_tr, thr_tr,
                            tag=f"{src_name}_train",
                            time=t_tr_s, flux=f_tr_arr)

        # ---- BUILD WINDOWS ----
        X_s, T_end_s, keys_s = build_windows(
            time=t_s, flux=f_s, flux_err=fe_s, index=idx_s,
            window_days=WINDOW_DAYS, step_days=STEP_DAYS,
            min_points=MIN_POINTS_IN_WINDOW,
            global_tail_threshold=global_tail_s,
            F_quiescent=F_q_s,
            idx_quiescent=idx_q_s,
            flare_onsets_train=onsets_tr,
        )

        if keys_ref is None:
            keys_ref = keys_s
        elif keys_s != keys_ref:
            raise RuntimeError(
                f"Feature key mismatch for '{src_name}'. "
                f"Expected {keys_ref}, got {keys_s}")

        # store everything
        d.update({
            "X": X_s, "T_end": T_end_s,
            "flare_intervals_full":  intv_full,
            "flare_onsets_full":     onsets_full,
            "flare_intervals_train": intv_tr,
            "flare_onsets_train":    onsets_tr,
            "F_quiescent":  F_q_s,
            "idx_quiescent": idx_q_s,
            "global_tail":  global_tail_s,
        })

        print(f"  Windows: {len(T_end_s)}  "
              f"features: {X_s.shape[1]}")

    # ================================================================
    # STEP 3: STACK TRAINING DATA + IMPUTATION
    # ================================================================
    print("\n" + "="*62)
    print("STEP 3 — STACKING + IMPUTATION")
    print("="*62)

    TRAIN_END_BOUNDARY = float(BACKTEST_CUTOFF_MJD - HORIZON_DAYS)

    X_train_list = []
    y_watch_train_list = []
    y_trigger_train_list = []

    for src_name, d in all_data.items():
        T_end_s  = d["T_end"]
        X_s      = d["X"]
        intv_tr  = d["flare_intervals_train"]
        onsets_tr = d["flare_onsets_train"]

        tr_mask_s = (T_end_s <= TRAIN_END_BOUNDARY)
        Y_watch_s = label_windows(
            T_end_s, intv_tr,
            horizon_days=HORIZON_DAYS,
            min_overlap_days=MIN_OVERLAP_DAYS)
        Y_trigger_s = label_windows_to_onset(
            T_end_s, onsets_tr, flare_intervals=intv_tr,
            horizon_days=TRIGGER_HORIZON_DAYS,
            min_lead_days=0.0,
            exclude_active_flare=True)
        n_pos_watch_s = int(np.sum(Y_watch_s[tr_mask_s]))
        n_pos_trigger_s = int(np.sum(Y_trigger_s[tr_mask_s]))
        print(f"  [{src_name}] TRAIN windows={tr_mask_s.sum()}  "
              f"WATCH pos={n_pos_watch_s}  "
              f"TRIGGER pos={n_pos_trigger_s}")

        X_train_list.append(X_s[tr_mask_s])
        y_watch_train_list.append(Y_watch_s[tr_mask_s])
        y_trigger_train_list.append(Y_trigger_s[tr_mask_s])
        d["Y_watch"] = Y_watch_s
        d["Y_trigger"] = Y_trigger_s

    X_raw = np.vstack(X_train_list)
    y_watch_raw = np.concatenate(y_watch_train_list)
    y_trigger_raw = np.concatenate(y_trigger_train_list)
    n_tr  = len(y_watch_raw)
    n_pos_watch = int(np.sum(y_watch_raw))
    n_neg_watch = n_tr - n_pos_watch
    n_pos_trigger = int(np.sum(y_trigger_raw))
    n_neg_trigger = n_tr - n_pos_trigger

    print(f"\n  Combined TRAIN: {n_tr} windows  features={X_raw.shape[1]}")
    print(f"  WATCH balance: {n_pos_watch} pos / {n_neg_watch} neg  "
          f"ratio {n_neg_watch/max(n_pos_watch,1):.1f}:1")
    print(f"  TRIGGER balance: {n_pos_trigger} pos / {n_neg_trigger} neg  "
          f"ratio {n_neg_trigger/max(n_pos_trigger,1):.1f}:1")
    if n_neg_watch / max(n_pos_watch, 1) > 10:
        print("  [WARN] Severe WATCH imbalance (>10:1).")
    if n_neg_trigger / max(n_pos_trigger, 1) > 10:
        print("  [WARN] Severe TRIGGER imbalance (>10:1).")

    # --- impute using TRAIN-only combined medians ---
    # build a full TRAIN mask for imputation
    train_mask_combined = np.ones(n_tr, dtype=bool)   # everything in X_raw IS train
    X_train_all, train_medians = median_impute_with_train(X_raw, train_mask_combined)

    # --- label ACF check: verify block length is appropriate ---
    acf_days = estimate_label_acf_length(y_watch_raw, step_days=STEP_DAYS)
    recommended_block = max(3, int(np.ceil(acf_days / STEP_DAYS)))
    print(f"\n  Label ACF decorrelation length ≈ {acf_days:.0f} days  "
          f"→ recommended TEST_BLOCK_LEN ≥ {recommended_block} windows")
    if TEST_BLOCK_LEN < recommended_block:
        print(f"  [WARN] TEST_BLOCK_LEN={TEST_BLOCK_LEN} may be too short. "
              f"Consider increasing to {recommended_block}.")

    # --- feature correlation report ---

    # --- export derived training dataset (features + labels) for archiving ---
    export_training_dataset_csv(all_data, keys_ref, TRAIN_END_BOUNDARY, OUTDIR)
    print("\n  Feature correlations (|r|≥0.90):")
    print_feature_correlations(X_train_all, keys_ref, threshold=0.90)

    # ================================================================
    # STEP 4: TEST SET FROM TARGET SOURCE
    # ================================================================
    print("\n" + "="*62)
    print("STEP 4 — TEST SET (target source)")
    print("="*62)

    d_target      = all_data["target"]
    T_end_target  = d_target["T_end"]
    X_target_raw  = d_target["X"].copy()

    # Apply TRAIN medians to target features
    for j in range(X_target_raw.shape[1]):
        bad  = ~np.isfinite(X_target_raw[:, j])
        fill = train_medians[j] if np.isfinite(train_medians[j]) else 0.0
        X_target_raw[bad, j] = fill

    test_mask_target  = (T_end_target > TRAIN_END_BOUNDARY)
    X_test_target     = X_target_raw[test_mask_target] \
                        if np.any(test_mask_target) else None

    Y_full_target_watch = label_windows(
        T_end_target,
        d_target["flare_intervals_full"],
        horizon_days=HORIZON_DAYS,
        min_overlap_days=MIN_OVERLAP_DAYS)
    Y_full_target_trigger = label_windows_to_onset(
        T_end_target,
        d_target["flare_onsets_full"],
        flare_intervals=d_target["flare_intervals_full"],
        horizon_days=TRIGGER_HORIZON_DAYS,
        min_lead_days=0.0,
        exclude_active_flare=True)

    y_test_eval_watch = Y_full_target_watch[test_mask_target] \
        if np.any(test_mask_target) else None
    y_test_eval_trigger = Y_full_target_trigger[test_mask_target] \
        if np.any(test_mask_target) else None
    is_test_window = test_mask_target.astype(int)

    if np.any(test_mask_target):
        TEST_T0 = float(np.min(T_end_target[test_mask_target]))
        TEST_T1 = float(np.max(T_end_target[test_mask_target]))
        onsets_full_t = d_target["flare_onsets_full"]
        flare_onsets_test = onsets_full_t[
            (onsets_full_t >= TEST_T0) &
            (onsets_full_t <= TEST_T1 + HORIZON_DAYS)]
        print(f"  TEST T_end span: {TEST_T0:.1f}–{TEST_T1:.1f} MJD")
        print(f"  TEST windows: {int(test_mask_target.sum())}  "
              f"flare onsets in/near test: {len(flare_onsets_test)}")
        if y_test_eval_watch is not None:
            print(f"  TEST WATCH labels: {int(np.sum(y_test_eval_watch))} pos / "
                  f"{int(np.sum(y_test_eval_watch==0))} neg")
        if y_test_eval_trigger is not None:
            print(f"  TEST TRIGGER labels: {int(np.sum(y_test_eval_trigger))} pos / "
                  f"{int(np.sum(y_test_eval_trigger==0))} neg")
    else:
        TEST_T0 = TEST_T1 = np.nan
        flare_onsets_test = np.array([], float)
        print("  [WARN] No TEST windows.")

    # ================================================================
    # STEP 5: MODEL LOOP
    # ================================================================
    print("\n" + "="*62)
    print("STEP 5 — MODELS")
    print("="*62)

    model_results = {}
    fitted_models = {}   # WATCH models for ensemble

    for model_name in MODELS_TO_RUN:
        if model_name == "ensemble":
            continue   # built after individual models

        print(f"\n{'━'*20} {model_name.upper()} {'━'*20}")

        # ---- WATCH model: plain training accuracy on the full TRAIN set ----
        # Fit an uncalibrated copy of the model on all of X_train_all/y_watch_raw
        # (not just the pre-calibration-tail portion used below) and score it on
        # that same TRAIN set. This is a fit diagnostic only -- it reuses TRAIN
        # data for both fitting and scoring, so it is not a held-out metric and
        # is not used anywhere in threshold selection, calibration, or reported
        # held-out performance.
        n_pos_watch = int(np.sum(y_watch_raw))
        n_neg_watch = int(len(y_watch_raw) - n_pos_watch)
        train_acc_model = make_model(model_name, n_pos=n_pos_watch, n_neg=n_neg_watch)
        train_acc_model.fit(X_train_all, y_watch_raw)
        train_accuracy = float(accuracy_score(
            y_watch_raw, train_acc_model.predict(X_train_all)))
        print(f"  [TRAIN ACC] {model_name}: {train_accuracy:.3f} "
              f"(fit+scored on the same {len(y_watch_raw)} TRAIN windows)")

        # ---- WATCH model: purged walk-forward AUC on combined TRAIN ----
        wf_aucs, wf_pooled, wf_p_oof, wf_folds = purged_walk_forward_auc(
            model_name, X_train_all, y_watch_raw,
            train_min=WF_TRAIN_MIN,
            test_chunk=WF_TEST_CHUNK,
            step=WF_STEP,
            purge_len=PURGE_LEN,
        )
        if len(wf_aucs) > 0:
            print(f"\n  [WF] folds={wf_folds}  "
                  f"fold AUC: {np.mean(wf_aucs):.3f}±{np.std(wf_aucs):.3f}  "
                  f"[{np.min(wf_aucs):.3f},{np.max(wf_aucs):.3f}]")
            print(f"  [WF] pooled OOF AUC = {wf_pooled:.3f}")
        else:
            print("  [WF] Not enough valid folds.")
            wf_pooled = np.nan

        # ---- WATCH model fit + calibration ----
        tail_frac_watch = compute_tail_frac(
            y_watch_raw,
            min_flares_in_tail=CALIBRATION_TAIL_MIN_FLARES,
            min_frac=CALIBRATION_TAIL_MIN_FRAC)
        (fitted_watch, did_cal_watch, p_oof_watch, _,
         used_watch, y_tail_watch, p_tail_watch) = fit_maybe_calibrated_and_oof(
            model_name, X_train_all, y_watch_raw,
            use_calibration_tail=USE_CALIBRATION_TAIL,
            tail_frac=tail_frac_watch,
        )
        print(f"  [WATCH CAL] calibrated={did_cal_watch}  splits={used_watch}")

        if model_name == "rf" and hasattr(fitted_watch, "oob_score_"):
            print(f"  [WATCH RF] OOB score = {fitted_watch.oob_score_:.3f}")

        p_train_watch = assemble_causal_train_scores(
            y_watch_raw, p_oof_watch, p_tail_watch)
        thr_opt = float(PROBA_THRESHOLD_DEFAULT)
        thr_watch = float(PROBA_THRESHOLD_DEFAULT)
        watch_thr_y = y_watch_raw
        watch_thr_p = p_train_watch
        watch_thr_mode = "stacked_global"
        watch_thr_rows = []
        if WATCH_THRESHOLD_MODE == "source_local_pooled":
            y_local, p_local, local_rows = build_source_local_causal_scores(
                all_data, model_name, "Y_watch",
                TRAIN_END_BOUNDARY,
                use_calibration_tail=USE_CALIBRATION_TAIL,
            )
            if y_local is not None and p_local is not None:
                watch_thr_y = y_local
                watch_thr_p = p_local
                watch_thr_mode = "source_local_pooled"
                watch_thr_rows = local_rows
            else:
                print("  [WATCH THRESH] source-local scores unavailable → "
                      "falling back to stacked TRAIN scores.")

        if np.sum(np.isfinite(watch_thr_p)) > 10:
            thr_opt, thr_info = optimize_threshold_grid(
                watch_thr_y, watch_thr_p, THRESH_GRID,
                beta=THRESHOLD_FBETA,
                prefer_higher_within=WATCH_THRESHOLD_TIE_TOL,
            )
            thr_watch = float(thr_opt)
            print(f"  [WATCH THRESH] mode={watch_thr_mode}  "
                  f"opt={thr_opt:.2f}  "
                  f"F{THRESHOLD_FBETA}={thr_info.get('fbeta',np.nan):.3f}  "
                  f"P={thr_info.get('precision',np.nan):.3f}  "
                  f"R={thr_info.get('recall',np.nan):.3f}")
            if watch_thr_rows:
                kept = sum(1 for r in watch_thr_rows if r["n_valid_scores"] > 0)
                print(f"    pooled source-local scores from {kept}/"
                      f"{len(watch_thr_rows)} sources")
        else:
            print("  [WATCH THRESH] Skipped: insufficient causal TRAIN scores.")

        # ---- TRIGGER model fit + calibration ----
        tail_frac_trigger = compute_tail_frac(
            y_trigger_raw,
            min_flares_in_tail=CALIBRATION_TAIL_MIN_FLARES,
            min_frac=CALIBRATION_TAIL_MIN_FRAC)
        (fitted_trigger, did_cal_trigger, p_oof_trigger, _,
         used_trigger, y_tail_trigger, p_tail_trigger) = fit_maybe_calibrated_and_oof(
            model_name, X_train_all, y_trigger_raw,
            use_calibration_tail=USE_CALIBRATION_TAIL,
            tail_frac=tail_frac_trigger,
        )
        print(f"  [TRIGGER CAL] calibrated={did_cal_trigger}  splits={used_trigger}")

        if model_name == "rf" and hasattr(fitted_trigger, "oob_score_"):
            print(f"  [TRIGGER RF] OOB score = {fitted_trigger.oob_score_:.3f}")

        p_train_trigger = assemble_causal_train_scores(
            y_trigger_raw, p_oof_trigger, p_tail_trigger)
        thr_trigger = float(PROBA_THRESHOLD_DEFAULT)
        trigger_thr_info = {}
        if np.sum(np.isfinite(p_train_trigger)) > 10:
            thr_trigger, trigger_thr_info = optimize_trigger_threshold_grid(
                y_trigger_raw, p_train_trigger, THRESH_GRID,
                beta=TRIGGER_THRESHOLD_FBETA,
                min_precision=TRIGGER_MIN_PRECISION,
                max_alert_fraction=TRIGGER_MAX_ALERT_FRACTION)
            print(f"  [TRIGGER THRESH] opt={thr_trigger:.2f}  "
                  f"F{TRIGGER_THRESHOLD_FBETA}={trigger_thr_info.get('fbeta',np.nan):.3f}  "
                  f"P={trigger_thr_info.get('precision',np.nan):.3f}  "
                  f"R={trigger_thr_info.get('recall',np.nan):.3f}  "
                  f"alert_frac={trigger_thr_info.get('alert_fraction',np.nan):.3f}")
        else:
            print("  [TRIGGER THRESH] Skipped: insufficient causal TRAIN scores.")

        # ---- TEST predictions ----
        p_test_watch = None
        p_test_trigger = None
        auc_test = np.nan
        pval = np.nan
        ci_lo = ci_hi = np.nan
        bss = np.nan
        ap_val = np.nan
        auc_trigger = np.nan
        ap_trigger = np.nan

        if X_test_target is not None:
            p_test_watch = fitted_watch.predict_proba(X_test_target)[:, 1]
            p_test_trigger = fitted_trigger.predict_proba(X_test_target)[:, 1]

            if (y_test_eval_watch is not None
                    and len(np.unique(y_test_eval_watch)) == 2
                    and np.sum(np.isfinite(p_test_watch)) > 10):

                auc_test = float(roc_auc_score(y_test_eval_watch, p_test_watch))
                print(f"\n  [WATCH TEST] AUC = {auc_test:.3f}  "
                      f"n={len(p_test_watch)}  pos={int(np.sum(y_test_eval_watch))}")

                # permutation test
                pval, pinfo = permutation_test_auc_blockshuffle(
                    y_test_eval_watch, p_test_watch,
                    n_perm=N_PERM_TEST,
                    block_len=TEST_BLOCK_LEN,
                    seed=RANDOM_SEED)
                print(f"  [PERM] p={pval:.4g}  "
                      f"n_used={pinfo.get('n_perm_used','?')}")

                # bootstrap CI + block length sensitivity
                (_, ci_lo, ci_hi), binfo = block_bootstrap_auc_ci(
                    y_test_eval_watch, p_test_watch,
                    block_len=TEST_BLOCK_LEN,
                    n_boot=N_BOOT_TEST,
                    seed=RANDOM_SEED)
                print(f"  [BOOT] 95% CI=[{ci_lo:.3f},{ci_hi:.3f}]  "
                      f"n_valid={binfo.get('n_boot_valid','?')}")

                block_length_sensitivity(y_test_eval_watch, p_test_watch,
                                         block_lens=(5, 9, 15),
                                         n_boot=200)

                # Brier Skill Score
                br = float(brier_score_loss(y_test_eval_watch, p_test_watch))
                bs_clim = float(np.mean(y_test_eval_watch)) * (
                    1 - float(np.mean(y_test_eval_watch)))
                bss = float(1 - br / bs_clim) if bs_clim > 0 else np.nan
                print(f"  [BSS] Brier={br:.4f}  BSS={bss:.4f}")

                # extended metrics at optimal threshold
                met = eval_at_threshold(y_test_eval_watch, p_test_watch, thr_opt)
                if met:
                    print(f"  [METRICS @{thr_opt:.2f}]  "
                          f"P={met['precision']:.3f}  R={met['recall']:.3f}  "
                          f"F1={met['f1']:.3f}  F2={met['f2']:.3f}  "
                          f"MCC={met['mcc']:.3f}  FPR={met['fpr']:.3f}  "
                          f"NPV={met['npv']:.3f}  "
                          f"TP={met['tp']} FP={met['fp']} "
                          f"TN={met['tn']} FN={met['fn']}")

                # ROC + PR
                plot_roc_curve(
                    y_test_eval_watch, p_test_watch,
                    outpath=OUTDIR / f"roc_{model_name}.png",
                    title=model_name, model_name=model_name)
                ap_val = plot_pr_curve(
                    y_test_eval_watch, p_test_watch,
                    outpath=OUTDIR / f"pr_{model_name}.png",
                    title=model_name, model_name=model_name)
                print(f"  [PR] Average Precision = {ap_val:.3f}")

                # Reliability
                plot_reliability_curve(
                    y_test_eval_watch, p_test_watch,
                    outpath=OUTDIR / f"reliability_{model_name}.png",
                    title=f"Reliability | {model_name}",
                    n_bins=N_BINS_RELIABILITY)

            if (y_test_eval_trigger is not None
                    and len(np.unique(y_test_eval_trigger)) == 2
                    and np.sum(np.isfinite(p_test_trigger)) > 10):
                auc_trigger = float(roc_auc_score(y_test_eval_trigger, p_test_trigger))
                ap_trigger = float(average_precision_score(
                    y_test_eval_trigger, p_test_trigger))
                print(f"  [TRIGGER TEST] AUC={auc_trigger:.3f}  AP={ap_trigger:.3f}  "
                      f"pos={int(np.sum(y_test_eval_trigger))}")
                plot_roc_curve(
                    y_test_eval_trigger, p_test_trigger,
                    outpath=OUTDIR / f"roc_trigger_{model_name}.png",
                    title=f"{model_name} trigger", model_name=f"{model_name}-trigger")
                plot_pr_curve(
                    y_test_eval_trigger, p_test_trigger,
                    outpath=OUTDIR / f"pr_trigger_{model_name}.png",
                    title=f"{model_name} trigger", model_name=f"{model_name}-trigger")

        # ---- TEST alert metrics ----
        if p_test_watch is not None and y_test_eval_watch is not None and np.isfinite(thr_watch):
            met_watch = eval_at_threshold(y_test_eval_watch, p_test_watch, thr_watch)
            if met_watch:
                print(f"  [TEST WATCH @{thr_watch:.2f}]  "
                      f"P={met_watch['precision']:.3f}  "
                      f"R={met_watch['recall']:.3f}  "
                      f"F2={met_watch['f2']:.3f}  "
                      f"MCC={met_watch['mcc']:.3f}")
        else:
            met_watch = None

        if p_test_trigger is not None and y_test_eval_trigger is not None and np.isfinite(thr_trigger):
            met_trigger = eval_at_threshold(
                y_test_eval_trigger, p_test_trigger, thr_trigger)
            if met_trigger:
                print(f"  [TEST TRIGGER @{thr_trigger:.2f}]  "
                      f"P={met_trigger['precision']:.3f}  "
                      f"R={met_trigger['recall']:.3f}  "
                      f"F1={met_trigger['f1']:.3f}  "
                      f"MCC={met_trigger['mcc']:.3f}")
        else:
            met_trigger = None

        if p_test_watch is not None and p_test_trigger is not None:
            is_trig = (np.asarray(p_test_trigger, float) >= float(thr_trigger)) \
                if np.isfinite(thr_trigger) else np.zeros(len(p_test_trigger), dtype=bool)
            is_watch = (np.asarray(p_test_watch, float) >= float(thr_watch)) \
                if np.isfinite(thr_watch) else np.zeros(len(p_test_watch), dtype=bool)
            is_watch = is_watch & (~is_trig)
            f_trig = float(np.mean(is_trig))
            f_watch = float(np.mean(is_watch))
            f_norm = float(np.mean((~is_trig) & (~is_watch)))
            print(f"  [TEST STATES] normal={f_norm:.3f}  "
                  f"watch={f_watch:.3f}  trigger={f_trig:.3f}  "
                  f"n={len(is_trig)}")

        # ---- Full timeline probabilities (target source) ----
        p_all_watch = fitted_watch.predict_proba(X_target_raw)[:, 1]
        p_all_watch_smooth = moving_average_causal(p_all_watch, win=SEQ_SMOOTH_WIN)
        p_all_trigger = fitted_trigger.predict_proba(X_target_raw)[:, 1]
        p_all_trigger_smooth = moving_average_causal(
            p_all_trigger, win=SEQ_SMOOTH_WIN)

        # ---- Timeline plot ----
        plot_probability_timeline(
            T_end=T_end_target,
            p_series=p_all_watch_smooth,
            y_series=Y_full_target_watch,
            flare_onsets=d_target["flare_onsets_full"],
            flare_intervals=d_target["flare_intervals_full"],
            outpath=OUTDIR / f"timeline_{model_name}.png",
            title=(f"WATCH/TRIGGER forecast | {model_name} | "
                   f"W={WINDOW_DAYS:.0f}d S={STEP_DAYS:.0f}d"),
            thr_watch=thr_watch,
            thr_trigger=thr_trigger,
            extra_hline=PROBA_VIS_LINE_DEFAULT,
            shade_test_region=True,
            test_boundary_mjd=TRAIN_END_BOUNDARY,
            t_lc=d_target["t"], flux_lc=d_target["flux"],
            p_trigger_series=p_all_trigger_smooth,
            y_trigger_series=Y_full_target_trigger,
        )

        # ---- CSV export ----
        if EXPORT_PROBA_CSV:
            save_timeline_probabilities_csv(
                outpath=OUTDIR / f"timeline_{model_name}.csv",
                T_end=T_end_target,
                y_watch=Y_full_target_watch,
                y_trigger=Y_full_target_trigger,
                split_is_test=is_test_window,
                p_watch_raw=p_all_watch,
                p_watch_smooth=p_all_watch_smooth,
                p_trigger_raw=p_all_trigger,
                p_trigger_smooth=p_all_trigger_smooth,
                thr_watch=thr_watch,
                thr_trigger=thr_trigger,
            )

        # ---- Lead times ----
        rows_w_latest = []
        rows_t_latest = []

        if EXPORT_LEADTIME_CSV and np.any(test_mask_target):
            for alert, thr_a, consec, cap, onset_cap, series_for_lead in [
                ("watch", thr_watch, WATCH_MIN_CONSEC,
                 MAX_LEAD_DAYS_WATCH, HORIZON_DAYS, p_all_watch),
                ("trigger", thr_trigger, TRIGGER_MIN_CONSEC,
                 MAX_LEAD_DAYS_TRIGGER, TRIGGER_HORIZON_DAYS, p_all_trigger),
            ]:
                if not np.isfinite(thr_a):
                    continue
                for lookback in (LEAD_LOOKBACK_DAYS, LONG_LOOKBACK_DAYS, NEAR_ONSET_WINDOW_DAYS):
                    for mode in ("latest", "episode_start", "earliest"):
                        rows = compute_lead_times_to_onsets(
                            T_end=T_end_target,
                            proba_series=series_for_lead,
                            flare_onsets=flare_onsets_test,
                            threshold=float(thr_a),
                            lookback_days=float(lookback),
                            mode=mode,
                            min_consecutive=int(consec),
                            cooldown_windows=int(COOLDOWN_WINDOWS),
                            max_lead_days=float(cap),
                            allowed_window_mask=test_mask_target,
                            onset_min=TEST_T0,
                            onset_max=TEST_T1 + float(onset_cap),
                        )
                        print_lead_time_summary(
                            rows, threshold=thr_a, lookback_days=lookback,
                            tag=f"{alert.upper()} mode={mode} lb={lookback:.0f}d")
                        save_lead_time_csv(
                            OUTDIR / f"lead_{model_name}_{alert}_{mode}"
                                     f"_lb{int(lookback)}d.csv",
                            rows)
                        if mode == "latest" and lookback == LEAD_LOOKBACK_DAYS:
                            if alert == "watch":
                                rows_w_latest = rows
                            else:
                                rows_t_latest = rows

            # Lead-time distribution + detection table
            if rows_w_latest or rows_t_latest:
                plot_lead_time_distribution(
                    rows_watch=rows_w_latest,
                    rows_trigger=rows_t_latest,
                    outpath=OUTDIR / f"lead_dist_{model_name}.png",
                    model_name=model_name)
                print_detection_table(rows_w_latest, rows_t_latest, model_name)

        # ---- Store summary ----
        model_results[model_name] = {
            "wf_pooled":    float(wf_pooled)  if np.isfinite(wf_pooled) else np.nan,
            "auc_test":     float(auc_test),
            "ci_lo":        float(ci_lo),
            "ci_hi":        float(ci_hi),
            "pval":         float(pval),
            "bss":          float(bss),
            "ap":           float(ap_val),
            "trigger_auc":  float(auc_trigger),
            "trigger_ap":   float(ap_trigger),
            "thr_opt":      float(thr_opt),
            "thr_watch":    float(thr_watch),
            "thr_trigger":  float(thr_trigger),
            "train_accuracy": train_accuracy,
        }
        fitted_models[model_name] = fitted_watch

    # ---- Soft-voting ensemble ----
    if "ensemble" in MODELS_TO_RUN and len(fitted_models) >= 2:
        print(f"\n{'━'*20} ENSEMBLE {'━'*20}")
        try:
            base_names = [n for n in fitted_models]
            estimators = [(n, fitted_models[n]) for n in base_names]
            ensemble   = VotingClassifier(estimators=estimators,
                                          voting="soft")
            ensemble.estimators_       = [fitted_models[n] for n in base_names]
            ensemble.le_               = None
            ensemble.classes_          = np.array([0, 1])
            ensemble.estimators_       = [fitted_models[n] for n in base_names]
            ensemble.named_estimators_ = {n: fitted_models[n] for n in base_names}

            p_ens_test = None
            if X_test_target is not None:
                # average probabilities manually
                prob_stack = np.column_stack([
                    fitted_models[n].predict_proba(X_test_target)[:, 1]
                    for n in base_names])
                p_ens_test = np.mean(prob_stack, axis=1)

                if (y_test_eval_watch is not None
                        and len(np.unique(y_test_eval_watch)) == 2):
                    auc_ens = float(roc_auc_score(y_test_eval_watch, p_ens_test))
                    print(f"  [ENSEMBLE] AUC={auc_ens:.3f}")
                    plot_roc_curve(
                        y_test_eval_watch, p_ens_test,
                        outpath=OUTDIR / "roc_ensemble.png",
                        title="Ensemble", model_name="ensemble")

            # full timeline
            prob_stack_all = np.column_stack([
                fitted_models[n].predict_proba(X_target_raw)[:, 1]
                for n in base_names])
            p_ens_all    = np.mean(prob_stack_all, axis=1)
            p_ens_smooth = moving_average_causal(p_ens_all, win=SEQ_SMOOTH_WIN)

            plot_probability_timeline(
                T_end=T_end_target,
                p_series=p_ens_smooth,
                y_series=Y_full_target_watch,
                flare_onsets=d_target["flare_onsets_full"],
                flare_intervals=d_target["flare_intervals_full"],
                outpath=OUTDIR / "timeline_ensemble.png",
                title=f"Ensemble | W={WINDOW_DAYS:.0f}d S={STEP_DAYS:.0f}d",
                thr_watch=np.nanmean([model_results[n]["thr_watch"]
                                      for n in base_names]),
                thr_trigger=np.nanmean([model_results[n]["thr_trigger"]
                                        for n in base_names]),
                shade_test_region=True,
                test_boundary_mjd=TRAIN_END_BOUNDARY,
                t_lc=d_target["t"], flux_lc=d_target["flux"],
            )
        except Exception as exc:
            print(f"  [ENSEMBLE] Failed: {exc}")

    # ---- Pairwise model comparison ----
    if len(fitted_models) >= 2:
        print(f"\n{'─'*50}")
        print("PAIRWISE MODEL COMPARISON (Diebold-Mariano)")
        names = list(fitted_models.keys())
        if X_test_target is not None and y_test_eval_watch is not None:
            probs = {n: fitted_models[n].predict_proba(X_test_target)[:, 1]
                     for n in names}
            for i in range(len(names)):
                for j in range(i+1, len(names)):
                    na, nb = names[i], names[j]
                    dm, pv = diebold_mariano_test(y_test_eval_watch,
                                                  probs[na], probs[nb])
                    print(f"  {na} vs {nb}: DM={dm:.3f}  p={pv:.4g} "
                          f"({'A better' if dm<0 else 'B better'} if significant)")

    # ================================================================
    # STEP 6: CROSS-MODEL SUMMARY
    # ================================================================
    print("\n" + "="*62)
    print("MODEL COMPARISON SUMMARY")
    print("="*62)
    print(f"  {'Model':<14} {'WF_AUC':>8} {'Test_AUC':>10} "
          f"{'TrigAUC':>8} {'TrigAP':>8} "
          f"{'95%CI':>16} {'p-val':>8} {'BSS':>7} {'AP':>7}")
    print("  " + "─"*82)
    summary_rows = []
    for name, res in model_results.items():
        ci_str = (f"[{res['ci_lo']:.3f},{res['ci_hi']:.3f}]"
                  if np.isfinite(res['ci_lo']) else "   [n/a]  ")
        print(f"  {name:<14} {res['wf_pooled']:>8.3f} "
              f"{res['auc_test']:>10.3f} "
              f"{res['trigger_auc']:>8.3f} "
              f"{res['trigger_ap']:>8.3f} "
              f"{ci_str:>16} "
              f"{res['pval']:>8.4g} "
              f"{res['bss']:>7.3f} "
              f"{res['ap']:>7.3f}")
        summary_rows.append({"model": name, **res})

    # save summary CSV
    summary_path = OUTDIR / "model_comparison_summary.csv"
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with open(summary_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\n  Summary CSV → {summary_path}")

    elapsed = _time.time() - t_wall
    print(f"\nDONE in {elapsed:.1f}s.  Outputs in: {OUTDIR.resolve()}\n")


# ============================================================
# CSV LIGHT-CURVE READER  (Fermi-LAT daily CSV format)
# ============================================================

def _parse_csv_float(val: str):
    """Return float from a CSV cell; handles upper-limits ('<…') and missing ('-')."""
    v = val.strip().lstrip("<").strip()
    if v in ("", "-", "nan", "NaN"):
        return float("nan")
    try:
        return float(v)
    except ValueError:
        return float("nan")


def read_fermi_csv(filepath: str):
    """
    Read a Fermi-LAT daily light-curve CSV file and return arrays for
    the columns:  time (Julian Date), ts, flux, flux_err, index, index_err.

    Upper-limit flux values (prefixed with '<') and missing entries ('-')
    are converted to NaN.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file.

    Returns
    -------
    dict with keys:
        't'         – Julian Date (float array)
        'ts'        – Test Statistic
        'flux'      – Photon Flux  [photons cm-2 s-1]
        'flux_err'  – Photon Flux Error
        'index'     – Photon Index  (NaN where unavailable)
        'index_err' – Photon Index Error (NaN where unavailable)
    """
    t_list, ts_list, f_list, fe_list, i_list, ie_list = [], [], [], [], [], []

    with open(filepath, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            t_list.append (_parse_csv_float(row["Julian Date"]))
            ts_list.append(_parse_csv_float(row["TS"]))
            f_list.append (_parse_csv_float(
                row["Photon Flux [0.1-100 GeV](photons cm-2 s-1)"]))
            fe_list.append(_parse_csv_float(
                row["Photon Flux Error(photons cm-2 s-1)"]))
            i_list.append (_parse_csv_float(row["Photon Index"]))
            ie_list.append(_parse_csv_float(row["Photon Index Error"]))

    return {
        "t":         np.asarray(t_list,  float),
        "ts":        np.asarray(ts_list, float),
        "flux":      np.asarray(f_list,  float),
        "flux_err":  np.asarray(fe_list, float),
        "index":     np.asarray(i_list,  float),
        "index_err": np.asarray(ie_list, float),
    }


def print_fermi_csv(filepath: str):
    """Read a Fermi-LAT daily CSV and print the key columns to stdout."""
    data = read_fermi_csv(filepath)
    print(f"\n=== {filepath} ===")
    print(f"{'time(JD)':<14} {'ts':>10} {'flux':>14} "
          f"{'flux_err':>14} {'index':>10} {'index_err':>10}")
    print("-" * 78)
    for t, ts, fl, fe, idx, idxe in zip(
            data["t"], data["ts"], data["flux"], data["flux_err"],
            data["index"], data["index_err"]):
        print(f"{t:<14.1f} {ts:>10.2f} {fl:>14.4e} "
              f"{fe:>14.4e} {idx:>10.3f} {idxe:>10.3f}")


if __name__ == "__main__" and False:   # set False → True to run standalone
    import glob as _glob
    for _f in sorted(_glob.glob("4FGL_*.csv")):
        print_fermi_csv(_f)
