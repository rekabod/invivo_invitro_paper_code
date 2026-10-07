"""
Comprehensive electrophysiology statistics for in vivo / in vitro comparisons.

Purpose
-------
Compute and compare electrophysiological metrics (firing rates, PLV,
population coupling, dimensionality) across states (awake, sleep, vitro)
and cell types (RS-PC, IB-PC, IN, FS). Implements statistical tests,
visualization, and Excel export with publication-ready formatting.

Key features
------------
- State-by-state metric summaries (mean, median, SEM, 95% CI)
- Mann-Whitney U tests with FDR correction
- Publication-ready Excel export with conditional formatting
- Color-coded significance highlighting

Usage
-----
    python -m stats.ep_stat_new

Dependencies
------------
    numpy, pandas, scipy, statsmodels, matplotlib, seaborn, openpyxl
"""
import os

import re
import glob
import math
import warnings
from itertools import combinations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from scipy import stats
try:
    import statsmodels.formula.api as smf
except Exception:
    smf = None

from invivo_invitro_ep_core import *

warnings.filterwarnings("ignore", category=RuntimeWarning)
# prefer no background grids by default; individual plots can enable grids if needed
sns.set(style="white")
plt.rcParams['axes.grid'] = False
plt.rcParams["figure.dpi"] = 140


# =============================================================================
# Color system (2026 scheme requested)
# =============================================================================

COLORS_STATE = {
    "awake": "#B0B0B0",   # medium grey
    "sleep": "#6E6E6E",   # darker grey
    "vitro": "#111111",   # near-black
}

COLORS_HISTOLOGY_STATE = {
    "vivo": COLORS_STATE["awake"],
    "vitro": COLORS_STATE["vitro"],
}

COLORS_LAYER = {
    "supra": "#E96D6D",   # pink
    "gran":  "#AD4498",   # purple
    "infra": "#4D1F59",   # green
    "other": "#9E9E9E",
}

# RS-PC and IB-PC both light brown, per request
COLORS_CELLTYPE = {
    "RS-PC": "#D81B60",  # magenta
    "IB-PC": "#FB8C00",  # orange
    "IN": "#1E88E5",     # blue
    "FS": "#00897B",     # teal
    "UC": "#9E9E9E",
    "other": "#9E9E9E",
}
CELLTYPE_MARKERS = {"RS-PC": "^", "IB-PC": "^", "IN": "o", "FS": "o"}
CELLTYPE_FILLED = {"RS-PC": False, "IB-PC": False, "IN": False, "FS": False}

COLORS_EI = {
    "Exc": "#E53935",   # red
    "Inh": "#1565C0",   # blue (distinct from bare IN blue)
    "other": "#9E9E9E",
}

CELLTYPE_FILL_LIGHT = {
    "RS-PC": "#FCE4EC",
    "IB-PC": "#FFF3E0",
    "IN": "#E3F2FD",
    "FS": "#E0F2F1",
}

STATE_FILL_LIGHT = {
    "awake": "#F0F0F0",
    "sleep": "#D9D9D9",
    "vitro": "#BFBFBF",
}

BAND_FILL_LIGHT = {
    "delta": "#DCE6F1",
    "theta": "#E2F0D9",
    "low_gamma": "#FCE4D6",
    "high_gamma": "#E4DFEC",
}

# Histology stain colors (from histo_plot_example.py)
COLOR_NEUN = "#3a86ff"  # NeuN blue
COLOR_PV = "#ff006e"    # PV pink

CELLTYPE_ORDER = ["RS-PC", "IB-PC", "IN", "FS", "UC", "other"]
EI_ORDER = ["Exc", "Inh", "other"]
STATE_ORDER = ["awake", "sleep", "vitro"]
HISTOLOGY_PANEL_METRICS = [
    "pv_density_cells_mm2",
    "pv_coverage_percent",
    "neun_density_cells_mm2",
    "neun_coverage_percent",
    "pv_neun_density_ratio",
    "pv_neun_coverage_ratio",
]
EPHYS_MATRIX_METRICS = [
    "firing_rate_Hz",
    "max_firing_rate_10s_Hz",
    "max_firing_rate_1s_Hz",
    "burstiness_percent",
    "bursti_2",
    "isi_cv",
    "ref_viol",
    "halfwidth1_ms",
    "halfwidth2_ms",
    "peak_to_trough_delay_ms",
    "population_coupling",
    "coupling_strength_low_gamma",
    "coupling_strength_high_gamma",
    "mean_plv_low_gamma",
    "mean_plv_high_gamma",
    "max_plv_sig_low_gamma",
    "max_plv_sig_high_gamma",
    "plv_low_gamma_strength",
    "plv_high_gamma_strength",
    "frsim_strength",
    "dimensionality_pc1_variance",
    "dimensionality_participation_ratio",
    "dimensionality_eigenspectrum_entropy",
    "dimensionality_eigenspectrum_entropy_normalized",
    # Added delta/theta
    "mean_plv_delta",
    "max_plv_delta",
    "mean_plv_sig_delta",
    "max_plv_sig_delta",
    "resultant_length_sig_delta",
    "mean_plv_theta",
    "max_plv_theta",
    "mean_plv_sig_theta",
    "max_plv_sig_theta",
    "resultant_length_sig_theta",
]

HISTOLOGY_MIXEDLM_METRICS = [
    "neun_density_cells_mm2",
    "pv_density_cells_mm2",
    "pv_coverage_percent",
    "pv_neun_density_ratio",
]

EPHYS_MIXEDLM_METRICS = [
    "firing_rate_Hz",
    "max_firing_rate_1s_Hz",
    "max_firing_rate_10s_Hz",
    "burstiness_percent",
    "bursti_2",
    "isi_cv",
    "population_coupling",
    "halfwidth1_ms",
    "halfwidth2_ms",
    "mean_plv_sig_delta",
    "mean_plv_sig_theta",
    "mean_plv_sig_low_gamma",
    "mean_plv_sig_high_gamma",
    "resultant_length_sig_delta",
    "resultant_length_sig_theta",
    "resultant_length_sig_low_gamma",
    "resultant_length_sig_high_gamma",
]

MIXEDLM_CELLCLASS_ORDER = ["all", "Exc", "Inh", "FS"]
MIXEDLM_STATE_ORDER = ["vivo", "vitro"]

HISTOLOGY_MIXEDLM_LABELS = {
    "neun_density_cells_mm2": "NeuN density",
    "pv_density_cells_mm2": "PV density",
    "pv_coverage_percent": "PV coverage",
    "pv_neun_density_ratio": "PV/NeuN ratio",
}

EPHYS_MIXEDLM_LABELS = {
    "firing_rate_Hz": "Firing rate",
    "max_firing_rate_1s_Hz": "Max firing (1 s)",
    "max_firing_rate_10s_Hz": "Max firing (10 s)",
    "burstiness_percent": "Burstiness",
    "bursti_2": "Burst index B2",
    "isi_cv": "ISI CV",
    "population_coupling": "Population coupling",
    "halfwidth1_ms": "Halfwidth 1",
    "halfwidth2_ms": "Halfwidth 2",
    "mean_plv_sig_delta": "Mean sig PLV delta",
    "mean_plv_sig_theta": "Mean sig PLV theta",
    "mean_plv_sig_low_gamma": "Mean sig PLV low gamma",
    "mean_plv_sig_high_gamma": "Mean sig PLV high gamma",
    "resultant_length_sig_delta": "Resultant length delta",
    "resultant_length_sig_theta": "Resultant length theta",
    "resultant_length_sig_low_gamma": "Resultant length low gamma",
    "resultant_length_sig_high_gamma": "Resultant length high gamma",
}


# =============================================================================
# Small utilities
# =============================================================================

def _to_windows_abs_path(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return None
    return str(p).strip().replace("/", "\\")

def _safe_float(x):
    try:
        if x is None:
            return np.nan
        if isinstance(x, str) and x.strip() == "":
            return np.nan
        return float(x)
    except Exception:
        return np.nan

def _as_text_id(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s

def state_letter_to_state(s_letter: str) -> str:
    """
    User confirmed: state letter is a/s (and v for vitro in practice).
    """
    if s_letter is None or (isinstance(s_letter, float) and np.isnan(s_letter)):
        return "other"
    s = str(s_letter).strip().lower()
    if s == "a":
        return "awake"
    if s == "s":
        return "sleep"
    if s == "v":
        return "vitro"
    # fallback to existing "state" column if present elsewhere
    return "other"

def make_cell_uid(recording_id, state, unit_id):
    """Build fallback per-unit cell identifier used when no checked prefix exists."""
    state_letter = str(state).strip().lower()[:1] if pd.notna(state) and str(state).strip() else "u"
    return f"{_as_text_id(recording_id)}_{state_letter}{int(unit_id)}"

def make_cell_uid_from_prefix(cell_uid_prefix, unit_id):
    """Build canonical per-unit identifier from checked cell_uid_prefix + unit id."""
    return f"{_as_text_id(cell_uid_prefix)}{int(unit_id)}"

def derive_ei_class(celltype_norm: str) -> str:
    s = str(celltype_norm).strip()
    if s in ("RS-PC", "IB-PC"):
        return "Exc"
    if s in ("IN", "FS"):
        return "Inh"
    return "other"

def normalize_layer(layer) -> str:
    if layer is None or (isinstance(layer, float) and np.isnan(layer)):
        return "other"
    s = str(layer).strip().lower()
    if s.startswith("sup"):
        return "supra"
    if s.startswith("gra") or s.startswith("gran"):
        return "gran"
    if s.startswith("inf"):
        return "infra"
    return "other"

def normalize_state(state) -> str:
    if state is None or (isinstance(state, float) and np.isnan(state)):
        return "other"
    s = str(state).strip().lower()
    if s in ("a", "awake", "wake"):
        return "awake"
    if s in ("s", "sleep"):
        return "sleep"
    if s in ("v", "vitro", "in_vitro", "invitro"):
        return "vitro"
    return "other"

def assign_layer_from_channel_state(channel, state_norm) -> str:
    """
    Layer mapping requested by user.
    Awake/Sleep channels: 1-8 supra, 9-13 gran, >=14 infra.
    Vitro channels: 4-11 supra, 12-16 gran, >=17 infra.
    """
    ch = _safe_float(channel)
    if not np.isfinite(ch):
        return "other"
    ch = int(ch)
    s = normalize_state(state_norm)

    if s in ("awake", "sleep"):
        if 1 <= ch <= 8:
            return "supra"
        if 9 <= ch <= 13:
            return "gran"
        if ch >= 14:
            return "infra"
        return "other"

    if s == "vitro":
        if 4 <= ch <= 11:
            return "supra"
        if 12 <= ch <= 16:
            return "gran"
        if ch >= 17:
            return "infra"
        return "other"

    return "other"

def normalize_celltype(celltype) -> str:
    if celltype is None or (isinstance(celltype, float) and np.isnan(celltype)):
        return "other"
    s = str(celltype).strip().lower()
    # normalize a few common variants
    if s in ("rs", "rs-pc", "rs_pc", "regular", "regularly spiking", "regularly_spiking", "pc_rs", "pyramidal_rs"):
        return "RS-PC"
    if s in ("ib", "ib-pc", "ib_pc", "burst", "bursting", "pc_ib", "pyramidal_ib"):
        return "IB-PC"
    if s in ("in", "interneuron", "int"):
        return "IN"
    if s in ("fs", "fast_spiking", "fast spiking", "fast-spiking"):
        return "FS"
    # 'ax' / 'axonic' variants are no longer used; map them to 'other'
    if s in ("ax", "axonic", "axonics", "axo-axonic", "axoaxonic", "chandelier"):
        return "other"
    if s in ("unclassified", "unknown"):
        return "UC"
    return s

def infer_patient_from_recording_id(recording_id: str) -> str:
    """
    Heuristic: patient/case prefix like epi29, epi32, epi45, oiti40, etc.
    """
    if not isinstance(recording_id, str):
        return "unknown"
    m = re.match(r"^([a-zA-Z]+[0-9]+)", recording_id)
    return m.group(1) if m else "unknown"

def load_checked_recording_manifest(path_csv: str) -> pd.DataFrame:
    df = pd.read_csv(_to_windows_abs_path(path_csv)).copy()
    if df.empty:
        return pd.DataFrame(columns=["recording_id", "patient", "state_norm", "state_letter", "base_prefix", "cell_uid_prefix", "n_units"])

    for col in ["recording_id", "patient", "state_norm", "state_letter", "base_prefix", "cell_uid_prefix"]:
        if col not in df.columns:
            df[col] = ""

    df["recording_id"] = df["recording_id"].map(_as_text_id)
    df["patient"] = df["patient"].astype(str).str.strip().str.lower()
    df["state_norm"] = df["state_norm"].map(normalize_state)
    df["state_letter"] = df["state_letter"].astype(str).str.strip().str.lower().str[:1]
    missing_state_letter = df["state_letter"].eq("") | df["state_letter"].isna()
    df.loc[missing_state_letter, "state_letter"] = df.loc[missing_state_letter, "state_norm"].astype(str).str[:1]
    df["base_prefix"] = df["base_prefix"].map(_to_windows_abs_path)
    df["cell_uid_prefix"] = df["cell_uid_prefix"].astype(str).str.strip()

    missing_prefix = df["cell_uid_prefix"].eq("") | df["cell_uid_prefix"].isna()
    df.loc[missing_prefix, "cell_uid_prefix"] = df.loc[missing_prefix, "recording_id"].astype(str) + "_" + df.loc[missing_prefix, "state_letter"].astype(str)

    if "n_units" in df.columns:
        df["n_units"] = pd.to_numeric(df["n_units"], errors="coerce")
    else:
        df["n_units"] = np.nan

    return df[["recording_id", "patient", "state_norm", "state_letter", "base_prefix", "cell_uid_prefix", "n_units"]].drop_duplicates(subset=["recording_id"], keep="first")

def apply_checked_recording_manifest(master: pd.DataFrame, checked_manifest: Optional[pd.DataFrame]) -> pd.DataFrame:
    if checked_manifest is None or checked_manifest.empty or master is None or master.empty:
        return master

    out = master.copy()
    out["recording_id"] = out["recording_id"].map(_as_text_id)
    checked = checked_manifest.drop_duplicates(subset=["recording_id"], keep="first").copy()

    out = out.merge(
        checked[["recording_id", "patient", "state_norm", "state_letter", "base_prefix", "cell_uid_prefix"]],
        on="recording_id",
        how="left",
        suffixes=("", "_checked"),
    )

    for col in ["patient", "state_norm", "state_letter", "base_prefix"]:
        c_checked = f"{col}_checked"
        if c_checked in out.columns:
            if col == "state_norm":
                out[col] = out[c_checked].combine_first(out[col]).map(normalize_state)
            elif col == "patient":
                out[col] = out[c_checked].combine_first(out[col]).astype(str).str.strip().str.lower()
            else:
                out[col] = out[c_checked].combine_first(out[col])
            out = out.drop(columns=[c_checked])

    if "cell_uid" not in out.columns:
        out["cell_uid"] = np.nan

    valid_prefix = out["cell_uid_prefix"].notna() & out["cell_uid_prefix"].astype(str).str.strip().ne("")
    valid_unit = pd.to_numeric(out.get("unit_id", pd.Series(index=out.index)), errors="coerce").notna()
    use_prefix = valid_prefix & valid_unit
    out.loc[use_prefix, "cell_uid"] = out.loc[use_prefix].apply(
        lambda r: make_cell_uid_from_prefix(r["cell_uid_prefix"], r["unit_id"]),
        axis=1,
    )

    missing_cell_uid = out["cell_uid"].isna() | (out["cell_uid"].astype(str).str.strip() == "")
    if missing_cell_uid.any() and {"recording_id", "state_norm", "unit_id"}.issubset(out.columns):
        out.loc[missing_cell_uid, "cell_uid"] = out.loc[missing_cell_uid].apply(
            lambda r: make_cell_uid(r["recording_id"], r["state_norm"], r["unit_id"]),
            axis=1,
        )

    return out

def fdr_bh(pvals: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg FDR correction. Returns q-values.
    """
    pvals = np.asarray(pvals, dtype=float)
    n = pvals.size
    out = np.full(n, np.nan, dtype=float)
    ok = np.isfinite(pvals)
    p = pvals[ok]
    if p.size == 0:
        return out
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * (p.size / (np.arange(1, p.size + 1)))
    q = np.minimum.accumulate(q[::-1])[::-1]
    # place back
    tmp = np.full(p.size, np.nan)
    tmp[order] = np.clip(q, 0, 1)
    out[ok] = tmp
    return out

def cliffs_delta(x, y) -> float:
    """
    Effect size for two independent samples (robust/nonparametric).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return np.nan
    # O(n*m) is fine for typical unit counts; can optimize later
    gt = 0
    lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)
    return float((gt - lt) / (x.size * y.size))

def paired_rank_biserial(x, y) -> float:
    """
    Effect size for Wilcoxon signed-rank: rank-biserial correlation.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if x.size == 0:
        return np.nan
    d = y - x
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    Wpos = np.sum(ranks[d > 0])
    Wneg = np.sum(ranks[d < 0])
    return float((Wpos - Wneg) / (Wpos + Wneg))

def circular_mean(phases, weights=None) -> Tuple[float, float]:
    phases = np.asarray(phases, dtype=float)
    mask = np.isfinite(phases)
    phases = phases[mask]
    if weights is not None:
        weights = np.asarray(weights, dtype=float)[mask]
        weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1.0)
    if phases.size == 0:
        return np.nan, np.nan
    vec = np.average(np.exp(1j * phases), weights=weights) if weights is not None else np.mean(np.exp(1j * phases))
    return float(np.angle(vec)), float(np.abs(vec))

def circular_distance(a, b) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return np.nan
    return float(np.angle(np.exp(1j * (a - b))))

def stars(p):
    if not np.isfinite(p):
        return ""
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "n.s."

def add_sig_bracket(ax, x1, x2, y, text, dy=0.01, lw=1.0, fontsize: int = 9):
    """Draw a compact significance bracket between x1 and x2 at height y on axis `ax`."""
    ax.plot([x1, x1, x2, x2], [y, y + dy, y + dy, y], color="black", lw=lw, clip_on=False)
    ax.text((x1 + x2) / 2, y + dy, text, ha="center", va="bottom", fontsize=fontsize)

def save_figure(fig, out_png: str, also_svg: bool = True):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=220)
    if also_svg:
        root, _ = os.path.splitext(out_png)
        fig.savefig(root + ".svg")

def concat_nonempty(frames: List[pd.DataFrame]) -> pd.DataFrame:
    frames = [x for x in frames if isinstance(x, pd.DataFrame) and not x.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def pairwise_mannwhitney_table(df: pd.DataFrame, metric: str, group_col: str, groups: List[str], min_n: int = 3) -> pd.DataFrame:
    rows = []
    d = df.copy()
    if metric not in d.columns or group_col not in d.columns:
        return pd.DataFrame()
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    for i, ga in enumerate(groups):
        for gb in groups[i + 1:]:
            a = d[d[group_col] == ga][metric].dropna().astype(float)
            b = d[d[group_col] == gb][metric].dropna().astype(float)
            if len(a) < min_n or len(b) < min_n:
                continue
            try:
                stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            except Exception:
                stat, p = np.nan, np.nan
            rows.append({
                "metric": metric,
                "group_col": group_col,
                "group_a": ga,
                "group_b": gb,
                "n_a": int(len(a)),
                "n_b": int(len(b)),
                "median_a": float(np.nanmedian(a)),  # Kept for reference
                "median_b": float(np.nanmedian(b)),  # Kept for reference
                "mean_a": float(np.nanmean(a)) if len(a) else np.nan,  # Primary
                "mean_b": float(np.nanmean(b)) if len(b) else np.nan,  # Primary
                "effect_cliffs_delta": cliffs_delta(a, b),
                "test": "MannWhitneyU",
                "stat": float(stat) if np.isfinite(stat) else np.nan,
                "p": float(p) if np.isfinite(p) else np.nan,
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_fdr"] = fdr_bh(out["p"].values)
    return out

def _sig_text_from_row(row: pd.Series) -> str:
    """Return significance marker; skip 'n.s.' (only mark significant results)."""
    q = float(row.get("q_fdr", row.get("p", np.nan)))
    if np.isfinite(q):
        sig = stars(q)
        if sig == "n.s.":
            return ""  # skip non-significant
        return f"{sig} q={q:.2g}"
    sig = stars(row.get("p", np.nan))
    return "" if sig == "n.s." else sig

def annotate_pairwise_hue_brackets(
    ax,
    df: pd.DataFrame,
    x: str,
    y: str,
    order: List[str],
    hue: str,
    hue_order: List[str],
    min_n: int = 3,
    x_offset: float = 0.12,
    y_step_frac: float = 0.07,
    bracket_frac: float = 0.02,
):
    if df is None or df.empty or hue is None or not order or not hue_order:
        return
    vals = pd.to_numeric(df[y], errors="coerce")
    vals = vals[np.isfinite(vals)]
    if vals.empty:
        return
    ymin, ymax = float(vals.min()), float(vals.max())
    span = ymax - ymin if ymax > ymin else max(abs(ymax), 1.0)
    for xi, xval in enumerate(order):
        sub = df[df[x] == xval].copy()
        if sub.empty:
            continue
        present = [h for h in hue_order if h in set(sub[hue].dropna().tolist())]
        if len(present) < 2:
            continue
        table = pairwise_mannwhitney_table(sub, y, hue, present, min_n=min_n)
        if table.empty:
            continue
        sub_vals = pd.to_numeric(sub[y], errors="coerce")
        sub_vals = sub_vals[np.isfinite(sub_vals)]
        if sub_vals.empty:
            continue
        local_y = float(sub_vals.max()) + 0.08 * span
        step = y_step_frac * span
        offsets = np.linspace(-x_offset, x_offset, len(present)) if len(present) > 1 else np.array([0.0])
        xpos = {h: xi + offsets[idx] for idx, h in enumerate(present)}
        for k, (a, b) in enumerate(combinations(present, 2)):
            row = table[((table["group_a"] == a) & (table["group_b"] == b)) | ((table["group_a"] == b) & (table["group_b"] == a))]
            if row.empty:
                continue
            sig_text = _sig_text_from_row(row.iloc[0])
            if not sig_text:  # skip if no significance
                continue
            add_sig_bracket(ax, xpos[a], xpos[b], local_y + k * step, sig_text, dy=bracket_frac * span)

def annotate_pairwise_x_brackets(
    ax,
    df: pd.DataFrame,
    x: str,
    y: str,
    order: List[str],
    min_n: int = 3,
    x_offset: float = 0.12,
    y_step_frac: float = 0.07,
    bracket_frac: float = 0.02,
):
    if df is None or df.empty or not order:
        return
    vals = pd.to_numeric(df[y], errors="coerce")
    vals = vals[np.isfinite(vals)]
    if vals.empty:
        return
    table = pairwise_mannwhitney_table(df, y, x, order, min_n=min_n)
    if table.empty:
        return
    ymin, ymax = float(vals.min()), float(vals.max())
    span = ymax - ymin if ymax > ymin else max(abs(ymax), 1.0)
    base_y = ymax + 0.08 * span
    step = y_step_frac * span
    for k, (a, b) in enumerate(combinations(order, 2)):
        row = table[((table["group_a"] == a) & (table["group_b"] == b)) | ((table["group_a"] == b) & (table["group_b"] == a))]
        if row.empty:
            continue
        add_sig_bracket(ax, order.index(a), order.index(b), base_y + k * step, _sig_text_from_row(row.iloc[0]), dy=bracket_frac * span)

def annotate_same_hue_across_x_brackets(
    ax,
    df: pd.DataFrame,
    x: str,
    y: str,
    order: List[str],
    hue: str,
    hue_order: List[str],
    min_n: int = 3,
    y_step_frac: float = 0.06,
    bracket_frac: float = 0.02,
):
    """
    Add significance brackets comparing x-groups within each hue level.
    Example: awake vs sleep within IN, RS-PC, etc.
    """
    if df is None or df.empty or not order or not hue_order:
        return
    vals = pd.to_numeric(df[y], errors="coerce")
    vals = vals[np.isfinite(vals)]
    if vals.empty:
        return
    ymin, ymax = float(vals.min()), float(vals.max())
    span = ymax - ymin if ymax > ymin else max(abs(ymax), 1.0)
    base_y = ymax + 0.08 * span
    step = y_step_frac * span

    width = 0.72
    offsets = np.linspace(-width / 2 + width / (2 * max(len(hue_order), 1)), width / 2 - width / (2 * max(len(hue_order), 1)), len(hue_order)) if len(hue_order) > 1 else np.array([0.0])

    level = 0
    for h_idx, hval in enumerate(hue_order):
        sub = df[df[hue] == hval].copy()
        if sub.empty:
            continue
        present_x = [xx for xx in order if xx in set(sub[x].dropna().tolist())]
        if len(present_x) < 2:
            continue
        table = pairwise_mannwhitney_table(sub, y, x, present_x, min_n=min_n)
        if table.empty:
            continue
        for a, b in combinations(present_x, 2):
            row = table[((table["group_a"] == a) & (table["group_b"] == b)) | ((table["group_a"] == b) & (table["group_b"] == a))]
            if row.empty:
                continue
            sig_text = _sig_text_from_row(row.iloc[0])
            if not sig_text:
                continue
            xa = order.index(a) + offsets[h_idx]
            xb = order.index(b) + offsets[h_idx]
            add_sig_bracket(ax, xa, xb, base_y + level * step, sig_text, dy=bracket_frac * span)
            level += 1

def filter_true_connections(conn: pd.DataFrame) -> pd.DataFrame:
    """Filter connections to only true/verified connections (strict)."""
    if conn is None or conn.empty:
        return pd.DataFrame()
    c = conn.copy()
    # Strictly require true_connection='yes' if available
    if "true_connection" in c.columns:
        truth = c["true_connection"].astype(str).str.strip().str.lower()
        return c[truth.isin(["yes", "y", "true", "1"])].copy()
    # Fallback to is_true_connection boolean
    if "is_true_connection" in c.columns:
        return c[c["is_true_connection"].astype(bool)].copy()
    # Otherwise return empty (do not use putative/inferred as true)
    return pd.DataFrame()


def filter_outliers_iqr(series: pd.Series, factor: float = 1.5) -> pd.Series:
    """
    Return boolean mask marking non-outlier entries using IQR rule.
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return pd.Series([False] * len(series), index=series.index)
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    low = q1 - factor * iqr
    high = q3 + factor * iqr
    return series.apply(lambda v: pd.isna(v) and False or (v >= low and v <= high))


def compute_and_export_halfwidth_stats(master: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """
    Compute halfwidth statistics (halfwidth1_ms, halfwidth2_ms) for RS-PC, IB-PC, IN across states.
    Exports pairwise test results to CSV in out_dir and returns the table.
    """
    rows = []
    metrics = [m for m in ["halfwidth1_ms", "halfwidth2_ms"] if m in master.columns]
    ct_list = ["RS-PC", "IB-PC", "IN", "FS"]
    for metric in metrics:
        for ct in ct_list:
            sub = master[(master["celltype_norm"] == ct) & (master["state_norm"].isin(STATE_ORDER))].copy()
            if sub.empty:
                continue
            sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
            # remove outliers per celltype-state
            sub = sub[sub[metric].notna()]
            if sub.empty:
                continue
            # compute per-state groups
            groups = {}
            for st in STATE_ORDER:
                g = sub[sub["state_norm"] == st][metric].dropna().astype(float)
                # filter outliers within group
                if g.empty:
                    groups[st] = g
                    continue
                mask = filter_outliers_iqr(g, factor=1.5)
                groups[st] = g[mask.values]
            # pairwise tests
            for a, b in combinations(STATE_ORDER, 2):
                a_vals = groups.get(a, pd.Series(dtype=float))
                b_vals = groups.get(b, pd.Series(dtype=float))
                if len(a_vals) < 3 or len(b_vals) < 3:
                    continue
                try:
                    stat, p = stats.mannwhitneyu(a_vals, b_vals, alternative="two-sided")
                except Exception:
                    stat, p = np.nan, np.nan
                rows.append({
                    "metric": metric,
                    "celltype": ct,
                    "group_a": a,
                    "group_b": b,
                    "n_a": int(len(a_vals)),
                    "n_b": int(len(b_vals)),
                    "mean_a": float(np.nanmean(a_vals)) if len(a_vals) else np.nan,
                    "mean_b": float(np.nanmean(b_vals)) if len(b_vals) else np.nan,
                    "test": "MannWhitneyU",
                    "stat": float(stat) if np.isfinite(stat) else np.nan,
                    "p": float(p) if np.isfinite(p) else np.nan,
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_fdr"] = fdr_bh(out["p"].values)
        os.makedirs(out_dir, exist_ok=True)
        out.to_csv(os.path.join(out_dir, "halfwidth_pairwise_stats.csv"), index=False)
    return out


def overlay_means_on_categorical(ax, df: pd.DataFrame, x: str, y: str, order: List[str], hue: Optional[str] = None, hue_order: Optional[List[str]] = None):
    """Deprecated: no longer used. Kept for compatibility if called elsewhere."""
    pass


def compute_band_layer_coupling_stats(master: pd.DataFrame, out_dir: str, bands: List[str] = ["delta", "theta"]) -> pd.DataFrame:
    """
    Compute statistics for significantly coupled cells (n_spikes_phase_sig_{band} > 0) per layer.
    Exports CSV summary of pairwise findings across layers and celltypes.
    """
    rows = []
    for band in bands:
        ncol = f"n_spikes_phase_sig_{band}"
        pref = f"preferred_phase_sig_rad_{band}"
        reslen = f"resultant_length_sig_{band}"
        if ncol not in master.columns:
            continue
        df = master.copy()
        df[ncol] = pd.to_numeric(df[ncol], errors="coerce").fillna(0).astype(float)
        df = df[df[ncol] > 0]
        if df.empty:
            continue
        for layer in sorted(df.get("layer_norm", pd.Series()).dropna().unique()):
            sub = df[df["layer_norm"] == layer]
            if sub.empty:
                continue
            # overall by celltype
            for ct in ["RS-PC", "IB-PC", "IN", "FS"]:
                sct = sub[sub["celltype_norm"] == ct]
                if sct.empty:
                    continue
                vals = pd.to_numeric(sct.get(reslen, pd.Series(dtype=float)), errors="coerce").dropna().astype(float)
                rows.append({"band": band, "layer": layer, "celltype": ct, "n_units": int(len(vals)), "mean_resultant_length": float(np.nanmean(vals)) if len(vals) else np.nan})
    out = pd.DataFrame(rows)
    if not out.empty:
        os.makedirs(out_dir, exist_ok=True)
        out.to_csv(os.path.join(out_dir, "band_layer_coupling_summary.csv"), index=False)
    return out


def collate_significant_findings(out_dir: str) -> pd.DataFrame:
    """
    Collate known exported significant findings CSVs into one summary file.
    """
    srcs = [
        os.path.join(out_dir, "halfwidth_pairwise_stats.csv"),
        os.path.join(out_dir, "band_layer_coupling_summary.csv"),
        os.path.join(out_dir, "stats_population_preferred_phase.csv"),
    ]
    frames = []
    for p in srcs:
        if os.path.exists(p):
            try:
                frames.append(pd.read_csv(p))
            except Exception:
                continue
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    out_csv = os.path.join(out_dir, "summary_significant_findings.csv")
    os.makedirs(out_dir, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out


# =============================================================================
# Option B: load per-recording outputs if available
# =============================================================================

def recording_output_dir(base_prefix: str, out_dir: Optional[str] = None) -> str:
    """
    Your older pipeline saved outputs in: out_dir/recording_id/
    where recording_id == basename(base_prefix).
    """
    base_prefix = _to_windows_abs_path(base_prefix)
    rec_id = os.path.basename(str(base_prefix))
    if out_dir is None:
        out_dir = os.path.dirname(base_prefix)
    return os.path.join(_to_windows_abs_path(out_dir), rec_id)

def try_load_recording_tables(base_prefix: str, out_dir: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    rec_out = recording_output_dir(base_prefix, out_dir=out_dir)
    rec_id = os.path.basename(str(base_prefix))
    tables = {}

    # Core files produced by pipeline
    candidates = {
        "unit_stats": os.path.join(rec_out, f"{rec_id}_unit_stats.csv"),
        "plv_epochs": os.path.join(rec_out, f"{rec_id}_plv_epochs.csv"),
        "population_coupling": os.path.join(rec_out, f"{rec_id}_population_coupling.csv"),
        "connections": os.path.join(rec_out, f"{rec_id}_connections_putative.csv"),
        "plv_low_nodes": os.path.join(rec_out, f"{rec_id}_plv_graph_low_gamma_nodes.csv"),
        "plv_high_nodes": os.path.join(rec_out, f"{rec_id}_plv_graph_high_gamma_nodes.csv"),
        "popc_scalar_nodes": os.path.join(rec_out, f"{rec_id}_popcoupling_graph_scalar_nodes.csv"),
        "frsim_nodes": os.path.join(rec_out, f"{rec_id}_popcoupling_graph_frsim_nodes.csv"),
    }

    for k, p in candidates.items():
        if os.path.exists(p):
            tables[k] = pd.read_csv(p)

    return tables

def build_master_database_from_recording_outputs(
    outputs_root: str,
    out_csv: Optional[str] = None,
    units_manifest_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build one master units database from per-recording *_analysis.xlsx files.

    Expected layout:
      outputs_root/<recording_id>/<recording_id>_analysis.xlsx
    """
    outputs_root = _to_windows_abs_path(outputs_root)
    if outputs_root is None:
        raise ValueError("outputs_root is required")

    pattern = os.path.join(outputs_root, "*", "*_analysis.xlsx")
    excel_paths = sorted(glob.glob(pattern))
    if not excel_paths:
        raise FileNotFoundError(f"No *_analysis.xlsx files found under: {outputs_root}")

    manifest = None
    if units_manifest_csv and os.path.exists(_to_windows_abs_path(units_manifest_csv)):
        manifest = pd.read_csv(_to_windows_abs_path(units_manifest_csv)).copy()
        if "file" in manifest.columns:
            manifest["file"] = manifest["file"].map(_to_windows_abs_path)
            manifest["recording_id"] = manifest["file"].map(lambda p: os.path.basename(str(p)[:-4]) if isinstance(p, str) and p.lower().endswith(".bin") else os.path.basename(str(p)))
        if "recording_id" in manifest.columns:
            manifest["recording_id"] = manifest["recording_id"].map(_as_text_id)
        if "unit_id" in manifest.columns:
            manifest["unit_id"] = pd.to_numeric(manifest["unit_id"], errors="coerce")

    rows = []
    for xp in excel_paths:
        rec_id = os.path.basename(xp).replace("_analysis.xlsx", "")
        rec_out = os.path.dirname(xp)

        # Attempt to load a sensible units table from the workbook.
        units = None
        try:
            # prefer an explicit 'units' sheet
            xls = pd.ExcelFile(xp)
            if 'units' in [s.lower() for s in xls.sheet_names]:
                units = pd.read_excel(xp, sheet_name='units')
            else:
                # pick the first sheet that contains a unit identifier column
                for s in xls.sheet_names:
                    df_s = pd.read_excel(xp, sheet_name=s)
                    cols = set([c.lower() for c in df_s.columns])
                    if {'unit_id', 'cell_uid'}.intersection(cols):
                        units = df_s
                        break
                if units is None:
                    # fallback: take first sheet
                    units = pd.read_excel(xp, sheet_name=0)
        except Exception:
            units = None

        # If workbook didn't contain units, try per-recording CSVs produced by older pipeline
        if units is None or (isinstance(units, pd.DataFrame) and units.shape[0] == 0):
            csv_alt = os.path.join(rec_out, f"{rec_id}_unit_stats.csv")
            if os.path.exists(csv_alt):
                try:
                    units = pd.read_csv(csv_alt)
                except Exception:
                    units = None

        if units is None or (isinstance(units, pd.DataFrame) and units.shape[0] == 0):
            # nothing to add for this recording; continue to next
            continue

        units = units.copy()
        if "recording_id" not in units.columns:
            units["recording_id"] = rec_id
        units["recording_id"] = units["recording_id"].map(_as_text_id)
        if "base_prefix" not in units.columns:
            units["base_prefix"] = os.path.join(outputs_root, rec_id, rec_id)

        units["source_analysis_xlsx"] = xp
        units["source_output_dir"] = rec_out

        if "state_letter" not in units.columns:
            if "state" in units.columns:
                units["state_letter"] = units["state"].map(lambda s: (str(s).strip().lower()[:1] if pd.notna(s) and str(s).strip() else ""))

        if "state" not in units.columns and "state_letter" in units.columns:
            units["state"] = units["state_letter"].map(state_letter_to_state)

        if manifest is not None and {"recording_id", "unit_id", "state"}.issubset(manifest.columns):
            join_cols = [c for c in ["recording_id", "unit_id"] if c in units.columns]
            if len(join_cols) == 2:
                mf = manifest[["recording_id", "unit_id", "state"]].copy()
                mf["recording_id"] = mf["recording_id"].map(_as_text_id)
                mf["unit_id"] = pd.to_numeric(mf["unit_id"], errors="coerce")
                units["recording_id"] = units["recording_id"].map(_as_text_id)
                units["unit_id"] = pd.to_numeric(units["unit_id"], errors="coerce")
                units = units.merge(mf, on=["recording_id", "unit_id"], how="left", suffixes=("", "_manifest"))
                if "state_manifest" in units.columns:
                    units["state"] = units["state"].combine_first(units["state_manifest"])
                    units = units.drop(columns=["state_manifest"])

        rows.append(units)

    # filter valid dataframes
    valid_rows = [r for r in rows if isinstance(r, pd.DataFrame) and r.shape[0] > 0]
    if not valid_rows:
        raise ValueError(f"No unit rows found in analysis workbooks under: {outputs_root}. Checked files: {excel_paths}")

    out = pd.concat(valid_rows, ignore_index=True, sort=False)

    out["state_norm"] = out.get("state", pd.Series(index=out.index, dtype=object)).map(normalize_state)
    if "state_letter" in out.columns:
        mask_other = out["state_norm"] == "other"
        out.loc[mask_other, "state_norm"] = out.loc[mask_other, "state_letter"].map(state_letter_to_state)

    if "channel" in out.columns:
        out["layer_norm"] = [assign_layer_from_channel_state(ch, st) for ch, st in zip(out["channel"], out["state_norm"])]
    elif "layer" in out.columns:
        out["layer_norm"] = out["layer"].map(normalize_layer)
    else:
        out["layer_norm"] = "other"

    if "patient" not in out.columns:
        out["patient"] = out["recording_id"].astype(str).apply(infer_patient_from_recording_id)
    out["patient"] = out["patient"].astype(str)

    if "cell_uid" not in out.columns:
        out["cell_uid"] = np.nan
    missing_cell_uid = out["cell_uid"].isna() | (out["cell_uid"].astype(str).str.strip() == "")
    if missing_cell_uid.any() and {"recording_id", "state_norm", "unit_id"}.issubset(out.columns):
        out.loc[missing_cell_uid, "cell_uid"] = out.loc[missing_cell_uid].apply(
            lambda r: make_cell_uid(r["recording_id"], r["state_norm"], r["unit_id"]),
            axis=1,
        )

    if out_csv:
        out_csv = _to_windows_abs_path(out_csv)
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        out.to_csv(out_csv, index=False)

    return out

def build_connections_table_from_recording_outputs(outputs_root: str, out_csv: Optional[str] = None) -> pd.DataFrame:
    """
    Collect per-recording putative connections tables into one master table.

    Expected layout:
      outputs_root/<recording_id>/<recording_id>_connections_putative.csv
    """
    outputs_root = _to_windows_abs_path(outputs_root)
    if outputs_root is None:
        raise ValueError("outputs_root is required")

    pattern = os.path.join(outputs_root, "*", "*_connections_putative.csv")
    csv_paths = sorted(glob.glob(pattern))
    if not csv_paths:
        return pd.DataFrame()

    rows = []
    for cp in csv_paths:
        rec_id = os.path.basename(cp).replace("_connections_putative.csv", "")
        try:
            conn = pd.read_csv(cp)
        except Exception:
            continue
        if conn.empty:
            continue
        conn = conn.copy()
        conn["recording_id"] = _as_text_id(rec_id)
        conn["source_connections_csv"] = cp
        for c in ["unit_i", "unit_j", "unit_i_channel", "unit_j_channel"]:
            if c in conn.columns:
                conn[c] = pd.to_numeric(conn[c], errors="coerce")
        rows.append(conn)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    if out_csv:
        out_csv = _to_windows_abs_path(out_csv)
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        out.to_csv(out_csv, index=False)
    return out

def build_recording_manifest(master: pd.DataFrame) -> pd.DataFrame:
    """Build a recording-level patient/state map for downstream joins and plots."""
    if master is None or master.empty:
        return pd.DataFrame(columns=["recording_id", "patient", "state_norm", "state_letter", "base_prefix", "cell_uid_prefix", "n_units"])

    cols = [c for c in ["recording_id", "patient", "state_norm", "state_letter", "base_prefix", "cell_uid_prefix", "unit_id"] if c in master.columns]
    if not cols:
        return pd.DataFrame(columns=["recording_id", "patient", "state_norm", "state_letter", "base_prefix", "cell_uid_prefix", "n_units"])

    df = master[cols].copy()
    df["recording_id"] = df["recording_id"].map(_as_text_id)
    if "patient" not in df.columns:
        df["patient"] = df["recording_id"].astype(str).apply(infer_patient_from_recording_id)
    df["patient"] = df["patient"].astype(str)
    if "state_norm" not in df.columns:
        df["state_norm"] = "other"
    if "state_letter" not in df.columns:
        df["state_letter"] = df["state_norm"].map(lambda s: str(s).strip().lower()[:1] if pd.notna(s) and str(s).strip() else "u")
    if "base_prefix" not in df.columns:
        df["base_prefix"] = ""
    if "cell_uid_prefix" not in df.columns:
        df["cell_uid_prefix"] = ""
    missing_prefix = df["cell_uid_prefix"].isna() | (df["cell_uid_prefix"].astype(str).str.strip() == "")
    df.loc[missing_prefix, "cell_uid_prefix"] = df.loc[missing_prefix, "recording_id"].astype(str) + "_" + df.loc[missing_prefix, "state_letter"].astype(str)

    return (
        df.groupby(["recording_id", "patient", "state_norm", "state_letter", "base_prefix", "cell_uid_prefix"], dropna=False)
          .size()
          .reset_index(name="n_units")
          .sort_values(["patient", "recording_id", "state_norm"], kind="stable")
    )


# =============================================================================
# Data loading / tidying
# =============================================================================

@dataclass
class Paths:
    database_csv: str
    graph_metrics_csv: Optional[str] = None
    connections_csv: Optional[str] = None
    histology_csv: Optional[str] = None
    matched_units_csv: Optional[str] = None
    checked_recordings_csv: Optional[str] = None
    outputs_root: Optional[str] = None
    units_manifest_csv: Optional[str] = None
    build_database_from_outputs: bool = False
    out_dir: Optional[str] = None
    results_dir: str = "stats_out_light"

def load_master_database(path_csv: str) -> pd.DataFrame:
    df = pd.read_csv(_to_windows_abs_path(path_csv))
    if "patient" not in df.columns and "animal_id" in df.columns:
        df = df.rename(columns={"animal_id": "patient"})
    # Standard columns (best-effort)
    if "state" in df.columns:
        df["state_norm"] = df["state"].apply(normalize_state)
        if "state_letter" in df.columns:
            m = df["state_norm"] == "other"
            df.loc[m, "state_norm"] = df.loc[m, "state_letter"].apply(state_letter_to_state)
        if "state_letter" in df.columns:
            m = df["state_norm"] == "other"
            df.loc[m, "state_norm"] = df.loc[m, "state_letter"].apply(state_letter_to_state)
    elif "state_letter" in df.columns:
        df["state_norm"] = df["state_letter"].apply(state_letter_to_state)
    else:
        df["state_norm"] = "other"

    if "channel" in df.columns:
        df["layer_norm"] = [assign_layer_from_channel_state(ch, st) for ch, st in zip(df["channel"], df["state_norm"])]
    elif "layer" in df.columns:
        df["layer_norm"] = df["layer"].apply(normalize_layer)
    else:
        df["layer_norm"] = "other"

    if "celltype" in df.columns:
        df["celltype_norm"] = df["celltype"].apply(normalize_celltype)
    elif "celltype_norm" in df.columns:
        df["celltype_norm"] = df["celltype_norm"].apply(normalize_celltype)
    else:
        df["celltype_norm"] = "other"

    if "patient" not in df.columns:
        df["patient"] = df["recording_id"].astype(str).apply(infer_patient_from_recording_id)
    df["patient"] = df["patient"].astype(str).str.strip().str.lower()

    if "state_letter" not in df.columns:
        df["state_letter"] = df["state_norm"].map(lambda s: str(s).strip().lower()[:1] if pd.notna(s) and str(s).strip() else "u")
    if "cell_uid_prefix" not in df.columns:
        df["cell_uid_prefix"] = df["recording_id"].astype(str).map(_as_text_id) + "_" + df["state_letter"].astype(str)

    if "cellclass_ei" not in df.columns:
        df["cellclass_ei"] = df["celltype_norm"].map(derive_ei_class)

    # ensure numeric for common metrics used in stats
    numeric_cols = [
        "firing_rate_Hz", "max_firing_rate_10s_Hz", "max_firing_rate_1s_Hz", "burstiness_percent", "bursti_2",
        "autocorr_evenness", "amplitude_uV", "population_coupling",
        "halfwidth1_ms", "halfwidth2_ms", "peak_to_trough_delay_ms", "isi_cv", "ref_viol",
        "coupling_strength_low_gamma", "coupling_strength_high_gamma",
        # graph node metrics (if present)
        "plv_low_gamma_strength", "plv_low_gamma_clustering", "plv_low_gamma_pagerank", "plv_low_gamma_betweenness", "plv_low_gamma_closeness", "plv_low_gamma_local_efficiency", "plv_low_gamma_degree", "plv_low_gamma_hubness_z",
        "plv_high_gamma_strength", "plv_high_gamma_clustering", "plv_high_gamma_pagerank", "plv_high_gamma_betweenness", "plv_high_gamma_closeness", "plv_high_gamma_local_efficiency", "plv_high_gamma_degree", "plv_high_gamma_hubness_z",
        "popc_scalar_strength", "popc_scalar_clustering", "popc_scalar_pagerank", "popc_scalar_betweenness", "popc_scalar_closeness", "popc_scalar_local_efficiency", "popc_scalar_degree", "popc_scalar_hubness_z",
        "frsim_strength", "frsim_clustering", "frsim_pagerank", "frsim_betweenness", "frsim_closeness", "frsim_local_efficiency", "frsim_degree", "frsim_hubness_z",
        # Added delta/theta
        "mean_plv_delta", "max_plv_delta", "mean_plv_sig_delta", "max_plv_sig_delta", "resultant_length_sig_delta",
        "mean_plv_theta", "max_plv_theta", "mean_plv_sig_theta", "max_plv_sig_theta", "resultant_length_sig_theta",
    ]
    for band in ["delta", "theta", "low_gamma", "high_gamma"]:
        numeric_cols += [
            f"n_spikes_phase_all_{band}", f"n_spikes_phase_sig_{band}",
            f"preferred_phase_all_rad_{band}", f"preferred_phase_all_deg_{band}",
            f"resultant_length_all_{band}", f"circular_std_all_rad_{band}", f"rayleigh_p_all_{band}",
            f"preferred_phase_sig_rad_{band}", f"preferred_phase_sig_deg_{band}",
            f"resultant_length_sig_{band}", f"circular_std_sig_rad_{band}", f"rayleigh_p_sig_{band}",
            f"n_coupling_sig_epochs_{band}",
            f"best_rayleigh_p_sig_{band}", f"best_p_perm_sig_{band}",
        ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = df[c].apply(_safe_float)
    # Delta analyses are sleep-only by design.
    if "state_norm" in df.columns:
        non_sleep = df["state_norm"] != "sleep"
        for c in [k for k in df.columns if "delta" in str(k).lower()]:
            df.loc[non_sleep, c] = np.nan

    return df

def load_connections_database(path_csv: str, master: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    conn = pd.read_csv(_to_windows_abs_path(path_csv))
    if conn.empty:
        return conn
    conn["recording_id"] = conn["recording_id"].map(_as_text_id)
    if "patient" not in conn.columns:
        conn["patient"] = conn["recording_id"].astype(str).apply(infer_patient_from_recording_id)
    if "state_norm" in conn.columns:
        conn["state_norm"] = conn["state_norm"].map(normalize_state)
    for c in ["unit_i", "unit_j", "unit_i_channel", "unit_j_channel", "peak_latency_ms", "peak_z", "trough_z", "baseline", "baseline_std", "peak_value", "trough_value"]:
        if c in conn.columns:
            conn[c] = pd.to_numeric(conn[c], errors="coerce")

    if master is not None and not master.empty and {"recording_id", "unit_id"}.issubset(master.columns):
        meta_cols = [c for c in ["recording_id", "unit_id", "cell_uid", "state_norm", "layer_norm", "celltype_norm", "patient"] if c in master.columns]
        meta = master[meta_cols].drop_duplicates(["recording_id", "unit_id"]).copy()
        left = meta.add_prefix("unit_i_").rename(columns={"unit_i_recording_id": "recording_id", "unit_i_unit_id": "unit_i"})
        right = meta.add_prefix("unit_j_").rename(columns={"unit_j_recording_id": "recording_id", "unit_j_unit_id": "unit_j"})
        missing_left = "unit_i_layer_norm" not in conn.columns or "unit_i_celltype_norm" not in conn.columns
        missing_right = "unit_j_layer_norm" not in conn.columns or "unit_j_celltype_norm" not in conn.columns
        if missing_left:
            conn = conn.merge(left, on=["recording_id", "unit_i"], how="left", suffixes=("", "_meta_i"))
        if missing_right:
            conn = conn.merge(right, on=["recording_id", "unit_j"], how="left", suffixes=("", "_meta_j"))

    if "state_norm" not in conn.columns and "unit_i_state_norm" in conn.columns:
        conn["state_norm"] = conn["unit_i_state_norm"].map(normalize_state)
    if "layer_pair" not in conn.columns and {"unit_i_layer_norm", "unit_j_layer_norm"}.issubset(conn.columns):
        conn["layer_pair"] = conn["unit_i_layer_norm"].astype(str) + "->" + conn["unit_j_layer_norm"].astype(str)
        conn["same_layer"] = conn["unit_i_layer_norm"].astype(str) == conn["unit_j_layer_norm"].astype(str)
    for c in ["unit_i_celltype_norm", "unit_j_celltype_norm", "celltype_norm"]:
        if c in conn.columns:
            conn[c] = conn[c].map(normalize_celltype)
    if "celltype_pair" not in conn.columns and {"unit_i_celltype_norm", "unit_j_celltype_norm"}.issubset(conn.columns):
        conn["celltype_pair"] = conn["unit_i_celltype_norm"].astype(str) + "->" + conn["unit_j_celltype_norm"].astype(str)
    if "is_putative_connection" not in conn.columns and "inferred_type" in conn.columns:
        conn["is_putative_connection"] = conn["inferred_type"].astype(str).str.strip().str.lower() == "putative_monosynaptic_exc"
    if "true_connection" in conn.columns:
        truth = conn["true_connection"].astype(str).str.strip().str.lower()
        conn["is_true_connection"] = truth.isin(["yes", "y", "true", "1"])
    else:
        conn["is_true_connection"] = conn.get("is_putative_connection", False)
    return conn

def load_histology_database(path_csv: str) -> pd.DataFrame:
    h = pd.read_csv(_to_windows_abs_path(path_csv))
    if h.empty:
        return h
    if "state" in h.columns:
        h["histology_state"] = h["state"].astype(str).str.strip().str.lower()
    elif "condition" in h.columns:
        h["histology_state"] = h["condition"].astype(str).str.strip().str.lower()
    else:
        h["histology_state"] = "unknown"
    h["histology_state"] = h["histology_state"].replace({"in_vitro": "vitro", "invitro": "vitro", "in_vivo": "vivo", "invivo": "vivo"})
    h["staining"] = h.get("staining", pd.Series(index=h.index, dtype=object)).astype(str).str.strip().str.upper()
    h["patient"] = h.get("patient", pd.Series(index=h.index, dtype=object)).astype(str).str.strip()
    h["layer_norm"] = h.get("layer", pd.Series(index=h.index, dtype=object)).map(normalize_layer)
    for c in ["cell_number", "density (cell/mm^2)", "coverage (%)"]:
        if c in h.columns:
            h[c] = pd.to_numeric(h[c], errors="coerce")
    return h

def load_matched_units(path_csv: str) -> pd.DataFrame:
    df = pd.read_csv(_to_windows_abs_path(path_csv))
    # long format: one official_unit -> multiple state entries
    long = []
    for _, r in df.iterrows():
        off = r.get("official_unit", None)
        for col in ["sleep", "awake1", "awake2", "awake3", "awake4"]:
            v = r.get(col, None)
            if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == "":
                continue
            state = "sleep" if col == "sleep" else "awake"
            long.append({"official_unit": off, "entry": str(v).strip(), "state_norm": state})
    return pd.DataFrame(long)

def attach_matched_ids(master: pd.DataFrame, matched_long: pd.DataFrame) -> pd.DataFrame:
    """
    Map matched entries to master rows.
    We try to join on either:
      - cell_uid exactly equals 'entry'
      - OR recording_id + unit_id pattern if entry encodes that way
    """
    out = master.copy()

    if "official_unit" not in out.columns:
        out["official_unit"] = np.nan

    def _norm_uid(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        s = str(v).strip().lower()
        if s == "":
            return np.nan
        return re.sub(r"\s+", "", s)

    ml = matched_long.copy()
    ml["_entry_norm"] = ml["entry"].apply(_norm_uid)

    if "cell_uid" in out.columns:
        out["_cell_uid_norm"] = out["cell_uid"].apply(_norm_uid)
    else:
        out["_cell_uid_norm"] = np.nan

    # 1) direct match on normalized cell_uid strings
    m1 = ml[["official_unit", "_entry_norm"]].dropna(subset=["_entry_norm"]).drop_duplicates()
    out = out.merge(m1, left_on="_cell_uid_norm", right_on="_entry_norm", how="left", suffixes=("", "_m1"))
    if "official_unit_m1" in out.columns:
        out["official_unit"] = out["official_unit"].combine_first(out["official_unit_m1"])
        out = out.drop(columns=["official_unit_m1"])
    if "_entry_norm" in out.columns:
        out = out.drop(columns=["_entry_norm"])

    # 2) entries that look like '<recording_id>_<a|s|v><unit>'
    def parse_entry(e: str):
        mm = re.match(r"^(.*)_([asv])(\d+)$", str(e).strip().lower())
        if not mm:
            return None
        return mm.group(1), mm.group(2), int(mm.group(3))

    parsed = []
    for _, r in ml.iterrows():
        p = parse_entry(r["entry"])
        if p is None:
            continue
        rid, st_letter, uid = p
        parsed.append({"official_unit": r.get("official_unit", np.nan), "recording_id": rid, "state_letter": st_letter, "unit_id": uid})

    if parsed and {"recording_id", "unit_id"}.issubset(out.columns):
        parsed_df = pd.DataFrame(parsed)
        tmp = out.merge(parsed_df, on=["recording_id", "unit_id"], how="left", suffixes=("", "_parsed"))
        if "official_unit_parsed" in tmp.columns:
            out["official_unit"] = out["official_unit"].combine_first(tmp["official_unit_parsed"])

    if "cell_uid" not in out.columns:
        out["cell_uid"] = np.nan

    missing_cell_uid = out["cell_uid"].isna() | (out["cell_uid"].astype(str).str.strip() == "")
    has_prefix = "cell_uid_prefix" in out.columns
    if missing_cell_uid.any() and has_prefix:
        valid_prefix = out["cell_uid_prefix"].notna() & out["cell_uid_prefix"].astype(str).str.strip().ne("")
        valid_unit = pd.to_numeric(out.get("unit_id", pd.Series(index=out.index)), errors="coerce").notna()
        use_prefix = missing_cell_uid & valid_prefix & valid_unit
        out.loc[use_prefix, "cell_uid"] = out.loc[use_prefix].apply(
            lambda r: make_cell_uid_from_prefix(r["cell_uid_prefix"], r["unit_id"]),
            axis=1,
        )

    if "_cell_uid_norm" in out.columns:
        out = out.drop(columns=["_cell_uid_norm"])

    return out

def attach_matched_ids_from_checked_table(master: pd.DataFrame, matched_csv: str) -> pd.DataFrame:
    """
    Robust mapper for *_matched_units_checked.csv where sleep/awake* columns contain
    canonical cell_uids. This avoids regex parsing edge-cases and guarantees official_unit assignment.
    """
    out = master.copy()
    if "cell_uid" not in out.columns:
        return out
    out["cell_uid"] = out["cell_uid"].astype(str).str.strip()
    if not os.path.exists(_to_windows_abs_path(matched_csv)):
        return out

    raw = pd.read_csv(_to_windows_abs_path(matched_csv)).copy()
    if raw.empty or "official_unit" not in raw.columns:
        return out
    awake_cols = [c for c in raw.columns if str(c).lower().startswith("awake")]
    map_rows = []
    for _, r in raw.iterrows():
        off = r.get("official_unit", None)
        if off is None or (isinstance(off, float) and np.isnan(off)) or str(off).strip() == "":
            continue
        sl = r.get("sleep", None)
        if sl is not None and not (isinstance(sl, float) and np.isnan(sl)) and str(sl).strip() != "":
            map_rows.append({"cell_uid": str(sl).strip(), "official_unit": str(off).strip(), "state_norm": "sleep"})
        for ac in awake_cols:
            av = r.get(ac, None)
            if av is None or (isinstance(av, float) and np.isnan(av)) or str(av).strip() == "":
                continue
            map_rows.append({"cell_uid": str(av).strip(), "official_unit": str(off).strip(), "state_norm": "awake"})

    if not map_rows:
        return out
    mdf = pd.DataFrame(map_rows).drop_duplicates(subset=["cell_uid", "state_norm"], keep="first")
    out = out.merge(mdf[["cell_uid", "official_unit"]], on="cell_uid", how="left", suffixes=("", "_checked"))
    if "official_unit" in out.columns:
        out["official_unit"] = out["official_unit"].combine_first(out.get("official_unit_checked"))
    else:
        out["official_unit"] = out.get("official_unit_checked")
    if "official_unit_checked" in out.columns:
        out = out.drop(columns=["official_unit_checked"])
    return out


# =============================================================================
# Table builders (units / graph metrics / coupling)
# =============================================================================

GRAPH_PREFIXES = ["plv_low_gamma", "plv_high_gamma", "popc_scalar", "frsim"]
GRAPH_METRICS = ["strength", "clustering", "pagerank", "betweenness", "closeness", "local_efficiency", "degree", "hubness_z"]
def build_units_table(master: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "recording_id", "base_prefix", "cell_uid", "unit_id", "channel",
        "state_norm", "layer_norm", "celltype_norm", "cellclass_ei", "patient",
        "firing_rate_Hz", "max_firing_rate_10s_Hz", "max_firing_rate_1s_Hz", "burstiness_percent", "bursti_2",
        "autocorr_evenness", "amplitude_uV", "population_coupling",
        "coupling_strength_low_gamma", "coupling_strength_high_gamma",
        "official_unit",
    ]
    cols = [c for c in keep if c in master.columns]
    return master[cols].copy()

def build_graph_table(master: pd.DataFrame) -> pd.DataFrame:
    cols = ["recording_id", "cell_uid", "unit_id", "state_norm", "layer_norm", "celltype_norm", "cellclass_ei", "patient", "official_unit"]
    for pref in GRAPH_PREFIXES:
        for m in GRAPH_METRICS:
            c = f"{pref}_{m}"
            if c in master.columns:
                cols.append(c)
    cols = list(dict.fromkeys([c for c in cols if c in master.columns]))
    return master[cols].copy()

def build_coupling_table(master: pd.DataFrame) -> pd.DataFrame:
    cols = ["recording_id", "cell_uid", "unit_id", "state_norm", "layer_norm", "celltype_norm", "cellclass_ei", "patient", "official_unit"]
    # add per-band PLV summary columns if present
    for band in ["delta", "theta", "low_gamma", "high_gamma"]:
        for c in [
            f"n_epochs_{band}",
            f"n_spikes_phase_all_{band}",
            f"n_spikes_phase_sig_{band}",
            f"mean_plv_{band}",
            f"max_plv_{band}",
            f"mean_plv_sig_{band}",
            f"max_plv_sig_{band}",
            f"preferred_phase_all_rad_{band}",
            f"preferred_phase_all_deg_{band}",
            f"resultant_length_all_{band}",
            f"circular_std_all_rad_{band}",
            f"rayleigh_p_all_{band}",
            f"preferred_phase_sig_rad_{band}",
            f"preferred_phase_sig_deg_{band}",
            f"resultant_length_sig_{band}",
            f"circular_std_sig_rad_{band}",
            f"rayleigh_p_sig_{band}",
            f"n_coupling_sig_epochs_{band}",
            f"best_rayleigh_p_sig_{band}",
            f"best_p_perm_sig_{band}",
        ]:
            if c in master.columns:
                cols.append(c)

    cols += [c for c in ["coupling_strength_low_gamma", "coupling_strength_high_gamma", "population_coupling"] if c in master.columns]
    cols = list(dict.fromkeys(cols))
    return master[cols].copy()


def export_coupling_significance_summary(master: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    bands = [b for b in ["delta", "theta", "low_gamma", "high_gamma"] if f"n_coupling_sig_epochs_{b}" in master.columns or f"mean_plv_sig_{b}" in master.columns]
    celltypes = [c for c in ["RS-PC", "IB-PC", "IN", "FS"] if c in set(master.get("celltype_norm", pd.Series(dtype=object)).dropna())]
    if not bands or not celltypes:
        return pd.DataFrame()

    df = master.copy()
    df["state_norm"] = df.get("state_norm", pd.Series(index=df.index, dtype=object)).map(normalize_state)
    df["celltype_norm"] = df.get("celltype_norm", pd.Series(index=df.index, dtype=object)).map(normalize_celltype)
    df = df[df["state_norm"].isin(STATE_ORDER) & df["celltype_norm"].isin(celltypes)].copy()
    if df.empty:
        return pd.DataFrame()

    def _build_summary(group_cols: List[str]) -> pd.DataFrame:
        rows = []
        if group_cols == ["celltype_norm"]:
            group_values = [(ct,) for ct in celltypes]
        else:
            group_values = [(state, ct) for state in STATE_ORDER for ct in celltypes]

        for group_vals in group_values:
            if group_cols == ["celltype_norm"]:
                sub = df[df["celltype_norm"] == group_vals[0]].copy()
                row = {"celltype_norm": group_vals[0]}
            else:
                state, ct = group_vals
                sub = df[(df["state_norm"] == state) & (df["celltype_norm"] == ct)].copy()
                row = {"state_norm": state, "celltype_norm": ct}

            row["total_cells"] = int(len(sub))
            for band in bands:
                sig_col = f"n_coupling_sig_epochs_{band}" if f"n_coupling_sig_epochs_{band}" in sub.columns else f"mean_plv_sig_{band}"
                if sig_col not in sub.columns:
                    row[f"{band}_sig_cells"] = 0
                    row[f"{band}_sig_pct"] = np.nan
                    row[f"{band}_sig_epochs_sum"] = 0
                    continue
                sig_vals = pd.to_numeric(sub[sig_col], errors="coerce").fillna(0)
                sig_mask = sig_vals > 0
                row[f"{band}_sig_cells"] = int(sig_mask.sum())
                row[f"{band}_sig_pct"] = float(sig_mask.mean()) if len(sub) else np.nan
                row[f"{band}_sig_epochs_sum"] = int(sig_vals[sig_mask].sum())
            rows.append(row)
        return pd.DataFrame(rows)

    by_celltype = _build_summary(["celltype_norm"])
    by_state_celltype = _build_summary(["state_norm", "celltype_norm"])

    by_celltype.to_csv(os.path.join(out_dir, "table_coupling_significance_by_celltype.csv"), index=False)
    by_state_celltype.to_csv(os.path.join(out_dir, "table_coupling_significance_by_state_celltype.csv"), index=False)

    xlsx_path = os.path.join(out_dir, "table_coupling_significance_summary.xlsx")
    try:
        with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
            workbook = writer.book

            def _fmt(bg: str, bold: bool = False, num_format: Optional[str] = None):
                params = {"bg_color": bg, "bold": bold, "border": 1, "align": "center", "valign": "vcenter"}
                if num_format:
                    params["num_format"] = num_format
                return workbook.add_format(params)

            def _write_sheet(df_sheet: pd.DataFrame, sheet_name: str):
                worksheet = workbook.add_worksheet(sheet_name)
                writer.sheets[sheet_name] = worksheet
                worksheet.freeze_panes(1, 0)
                if len(df_sheet.columns) > 0:
                    worksheet.autofilter(0, 0, max(len(df_sheet), 1), len(df_sheet.columns) - 1)

                for col_idx, col in enumerate(df_sheet.columns):
                    band_key = next((b for b in bands if b in col), None)
                    if col == "celltype_norm":
                        bg = "#D9EAF7"
                    elif col == "state_norm":
                        bg = "#EDEDED"
                    elif band_key is not None:
                        bg = BAND_FILL_LIGHT.get(band_key, "#F2F2F2")
                    else:
                        bg = "#F2F2F2"
                    worksheet.write(0, col_idx, col, _fmt(bg, bold=True))

                for row_idx, (_, row) in enumerate(df_sheet.iterrows(), start=1):
                    row_fill = CELLTYPE_FILL_LIGHT.get(str(row.get("celltype_norm", "")), "#FFFFFF")
                    state_fill = STATE_FILL_LIGHT.get(str(row.get("state_norm", "")), row_fill)
                    for col_idx, col in enumerate(df_sheet.columns):
                        val = row[col]
                        bg = state_fill if col == "state_norm" else row_fill
                        if col.endswith("_pct"):
                            fmt = _fmt(bg, num_format="0.0%")
                        elif isinstance(val, (int, np.integer)):
                            fmt = _fmt(bg, num_format="0")
                        elif isinstance(val, (float, np.floating)):
                            fmt = _fmt(bg, num_format="0.000")
                        else:
                            fmt = _fmt(bg)
                        if pd.isna(val):
                            worksheet.write_blank(row_idx, col_idx, None, fmt)
                        elif isinstance(val, (int, np.integer)):
                            worksheet.write_number(row_idx, col_idx, int(val), fmt)
                        elif isinstance(val, (float, np.floating)):
                            worksheet.write_number(row_idx, col_idx, float(val), fmt)
                        else:
                            worksheet.write(row_idx, col_idx, str(val), fmt)

                worksheet.set_column(0, 0, 14)
                if "state_norm" in df_sheet.columns:
                    worksheet.set_column(df_sheet.columns.get_loc("state_norm"), df_sheet.columns.get_loc("state_norm"), 12)
                if "celltype_norm" in df_sheet.columns:
                    worksheet.set_column(df_sheet.columns.get_loc("celltype_norm"), df_sheet.columns.get_loc("celltype_norm"), 12)
                for col_idx, col in enumerate(df_sheet.columns):
                    if col.endswith("_pct"):
                        worksheet.set_column(col_idx, col_idx, 11)
                    elif col.endswith("_cells") or col.endswith("_sum") or col == "total_cells":
                        worksheet.set_column(col_idx, col_idx, 12)

            _write_sheet(by_celltype, "celltype_overall")
            _write_sheet(by_state_celltype, "state_by_celltype")
    except Exception:
        pass

    return by_state_celltype


# =============================================================================
# NEW: population preferred-phase and histology/ephys integration
# =============================================================================

def save_population_phase_polar_plots(master: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """
    Population polar histograms of preferred phase for significantly coupled cells.
    Uses unit-level preferred_phase_sig_rad_* columns written by invivo_invitro_outputs.py.
    """
    rows = []
    phase_dir = os.path.join(out_dir, "population_polar_plots")
    os.makedirs(phase_dir, exist_ok=True)
    band_states = {
        "delta": ["sleep"],
        "theta": STATE_ORDER,
        "low_gamma": STATE_ORDER,
        "high_gamma": STATE_ORDER,
    }

    def _first_existing(candidates: List[str]) -> Optional[str]:
        for candidate in candidates:
            if candidate in master.columns:
                return candidate
        return None

    for band in ["low_gamma", "high_gamma", "delta", "theta"]:
        phase_col = _first_existing([
            f"preferred_phase_sig_rad_{band}",
            f"preferred_phase_all_rad_{band}",
            f"preferred_phase_rad_{band}",
        ])
        weight_col = _first_existing([
            f"resultant_length_sig_{band}",
            f"resultant_length_all_{band}",
        ])
        n_col = _first_existing([
            f"n_spikes_phase_sig_{band}",
            f"n_spikes_phase_all_{band}",
            f"n_epochs_{band}",
        ])
        if phase_col is None:
            continue

        df = master.copy()
        df[phase_col] = pd.to_numeric(df[phase_col], errors="coerce")
        if weight_col is not None:
            df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce")
        else:
            weight_col = "__phase_weight__"
            df[weight_col] = 1.0
        if n_col is not None:
            df[n_col] = pd.to_numeric(df[n_col], errors="coerce")
            df = df[df[n_col].fillna(0) > 0]

        df = df[np.isfinite(df[phase_col])]
        # Fallback: if the significance filter emptied the frame, keep all finite phases.
        if df.empty:
            df = master.copy()
            df[phase_col] = pd.to_numeric(df[phase_col], errors="coerce")
            if weight_col not in df.columns:
                df[weight_col] = 1.0
            else:
                df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce")
            df = df[np.isfinite(df[phase_col])]
        if df.empty:
            continue

        # Shared radial limit for all polar plots per band so panels are comparable.
        band_bins = np.linspace(-np.pi, np.pi, 25)
        band_counts, _ = np.histogram(df[phase_col].values.astype(float), bins=band_bins)
        band_rmax = max(float(np.nanmax(band_counts)) if band_counts.size else 1.0, 1.0)

        state_subset = [s for s in band_states[band] if s in set(df["state_norm"])]
        if state_subset:
            fig, axs = plt.subplots(1, len(state_subset), figsize=(4.3 * len(state_subset), 4.2), subplot_kw={"projection": "polar"}, squeeze=False)
            axs = axs.ravel()
            for ax, state in zip(axs, state_subset):
                sub = df[df["state_norm"] == state].copy()
                weights = pd.to_numeric(sub[weight_col], errors="coerce").fillna(0).values
                _plot_phase_hist_ax(ax, sub[phase_col].values, weights, title=state, color=COLORS_STATE.get(state, "#777777"), weight_darkness=True, rmax=band_rmax, log_scale=False)
                pref, r = circular_mean(sub[phase_col].values, weights)
                rows.append({"band": band, "grouping": "state", "group": state, "n_units": int(len(sub)), "preferred_phase_rad": pref, "preferred_phase_deg": np.degrees(pref) % 360 if np.isfinite(pref) else np.nan, "resultant_length": r})
            fig.suptitle(f"Population preferred phase, {band}", y=1.05)
            fig.tight_layout()
            save_figure(fig, os.path.join(phase_dir, f"population_preferred_phase__{band}__by_state.png"))
            plt.close(fig)
            # Strength legend (alpha-coded PLV strength)
            fig_cb, ax_cb = plt.subplots(figsize=(3.2, 1.0))
            ax_cb.axis("off")
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list("strength_alpha", [(1, 1, 1, 0.2), matplotlib.colors.to_rgba("#333333", 1.0)])
            sm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0, vmax=1), cmap=cmap)
            sm.set_array([])
            cbar = fig_cb.colorbar(sm, ax=ax_cb, orientation="horizontal", fraction=0.8, pad=0.2)
            cbar.set_label("Normalized PLV strength (bar intensity)")
            fig_cb.tight_layout()
            save_figure(fig_cb, os.path.join(phase_dir, f"population_preferred_phase__{band}__strength_legend.png"))
            plt.close(fig_cb)

            fig, axs = plt.subplots(1, len(state_subset), figsize=(4.3 * len(state_subset), 4.2), subplot_kw={"projection": "polar"}, squeeze=False)
            axs = axs.ravel()
            for ax, state in zip(axs, state_subset):
                sub = df[df["state_norm"] == state].copy()
                weights = pd.to_numeric(sub[weight_col], errors="coerce").fillna(0).values
                _plot_phase_hist_ax(ax, sub[phase_col].values, weights, title=f"{state} (log)", color=COLORS_STATE.get(state, "#777777"), weight_darkness=True, rmax=band_rmax, log_scale=True)
            fig.suptitle(f"Population preferred phase, {band} (log radius)", y=1.05)
            fig.tight_layout()
            save_figure(fig, os.path.join(phase_dir, f"population_preferred_phase__{band}__by_state_log.png"))
            plt.close(fig)

        present_cells = [c for c in CELLTYPE_ORDER if c in set(df.get("celltype_norm", pd.Series(dtype=object)).dropna())]
        if present_cells:
            ncols = min(3, len(present_cells))
            nrows = int(math.ceil(len(present_cells) / ncols))
            fig, axs = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.0 * nrows), subplot_kw={"projection": "polar"})
            axs = np.atleast_1d(axs).ravel()
            for ax, ct in zip(axs, present_cells):
                sub = df[df["celltype_norm"] == ct].copy()
                weights = pd.to_numeric(sub[weight_col], errors="coerce").fillna(0).values
                _plot_phase_hist_ax(ax, sub[phase_col].values, weights, title=ct, color=COLORS_CELLTYPE.get(ct, "#777777"), weight_darkness=True, rmax=None, log_scale=False)
                pref, r = circular_mean(sub[phase_col].values, weights)
                rows.append({"band": band, "grouping": "celltype", "group": ct, "n_units": int(len(sub)), "preferred_phase_rad": pref, "preferred_phase_deg": np.degrees(pref) % 360 if np.isfinite(pref) else np.nan, "resultant_length": r})
            for ax in axs[len(present_cells):]:
                ax.axis("off")
            fig.suptitle(f"Population preferred phase by cell type: {band}", y=1.02)
            fig.tight_layout()
            save_figure(fig, os.path.join(phase_dir, f"population_preferred_phase__{band}__by_celltype.png"))
            plt.close(fig)

        present_ei = [c for c in EI_ORDER if c in set(df.get("cellclass_ei", pd.Series(dtype=object)).dropna())]
        if present_ei:
            fig, axs = plt.subplots(1, len(present_ei), figsize=(4.3 * len(present_ei), 4.2), subplot_kw={"projection": "polar"}, squeeze=False)
            axs = axs.ravel()
            for ax, ei in zip(axs, present_ei):
                sub = df[df.get("cellclass_ei") == ei].copy()
                weights = pd.to_numeric(sub[weight_col], errors="coerce").fillna(0).values
                _plot_phase_hist_ax(ax, sub[phase_col].values, weights, title=ei, color=COLORS_EI.get(ei, "#777777"), weight_darkness=True, rmax=band_rmax, log_scale=False)
                pref, r = circular_mean(sub[phase_col].values, weights)
                rows.append({"band": band, "grouping": "ei", "group": ei, "n_units": int(len(sub)), "preferred_phase_rad": pref, "preferred_phase_deg": np.degrees(pref) % 360 if np.isfinite(pref) else np.nan, "resultant_length": r})
            fig.suptitle(f"Population preferred phase by EI class: {band}", y=1.02)
            fig.tight_layout()
            save_figure(fig, os.path.join(phase_dir, f"population_preferred_phase__{band}__by_ei.png"))
            plt.close(fig)

        present_cells = [c for c in ["RS-PC", "IB-PC", "IN", "FS"] if c in set(df.get("celltype_norm", pd.Series(dtype=object)).dropna())]
        if present_cells:
            state_rows = [s for s in band_states[band] if s in set(df["state_norm"])]
            if not state_rows:
                state_rows = [s for s in STATE_ORDER if s in set(df["state_norm"])]
            fig, axs = plt.subplots(len(state_rows), len(present_cells), figsize=(4.0 * len(present_cells), 3.8 * len(state_rows)), subplot_kw={"projection": "polar"}, squeeze=False)
            for r, state in enumerate(state_rows):
                for cidx, ct in enumerate(present_cells):
                    sub = df[(df["state_norm"] == state) & (df["celltype_norm"] == ct)].copy()
                    weights = pd.to_numeric(sub[weight_col], errors="coerce").fillna(0).values
                    _plot_phase_hist_ax(axs[r, cidx], sub[phase_col].values, weights, title=f"{state} {ct}", color=COLORS_CELLTYPE.get(ct, "#777777"), weight_darkness=True, rmax=None, log_scale=False)
            fig.suptitle(f"Preferred phase by state and celltype: {band}", y=1.01)
            fig.tight_layout()
            save_figure(fig, os.path.join(phase_dir, f"population_preferred_phase__{band}__state_by_celltype.png"))
            plt.close(fig)

            # shared-axis overlay on one polar axis (same magnitude only here)
            fig, ax = plt.subplots(figsize=(5.2, 4.8), subplot_kw={"projection": "polar"})
            bins = np.linspace(-np.pi, np.pi, 25)
            overlay_cts = [c for c in ["RS-PC", "IB-PC", "IN", "FS"] if c in set(df.get("celltype_norm", pd.Series(dtype=object)).dropna())]
            rmax_overlay = 1.0
            cache = {}
            for ct in overlay_cts:
                ph = pd.to_numeric(df[df["celltype_norm"] == ct][phase_col], errors="coerce").dropna().values
                cts, edges = np.histogram(ph, bins=bins)
                cache[ct] = (cts, edges)
                if cts.size:
                    rmax_overlay = max(rmax_overlay, float(np.nanmax(cts)))
            for ct in overlay_cts:
                cts, edges = cache[ct]
                col = COLORS_CELLTYPE.get(ct, "#777777")
                ax.bar(edges[:-1], cts, width=np.diff(edges), align="edge", color=col, edgecolor=col, alpha=0.28, linewidth=0.7, label=ct)
            ax.set_theta_zero_location("E")
            ax.set_theta_direction(1)
            ax.set_ylim(0, rmax_overlay)
            ax.set_title(f"Preferred phase overlay (shared scale): {band}")
            ax.legend(title="celltype", bbox_to_anchor=(1.25, 1.1), loc="upper right", frameon=True)
            save_figure(fig, os.path.join(phase_dir, f"population_preferred_phase__{band}__overlay_shared_scale.png"))
            plt.close(fig)

        plot_phase_sine_map(df, phase_col, weight_col, band, phase_dir)

    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_csv(os.path.join(out_dir, "stats_population_preferred_phase.csv"), index=False)
    return out


def build_coupling_statistical_reporting(master: pd.DataFrame, stats_dict: Dict[str, pd.DataFrame], out_dir: str) -> pd.DataFrame:
    """
    Publication-style exploratory table:
    rows by state/frequency; columns with ratio+n and text summaries of significant
    differences for significantly-coupled-only and for all cells.
    """
    rows = []
    bands_by_state = {
        "sleep": ["delta", "theta", "low_gamma", "high_gamma"],
        "awake": ["theta", "low_gamma", "high_gamma"],
        "vitro": ["theta", "low_gamma", "high_gamma"],
    }
    groupings = [("layer_norm", ["supra", "gran", "infra"]), ("cellclass_ei", ["Exc", "Inh"]), ("celltype_norm", ["RS-PC", "IB-PC", "IN", "FS"])]
    metrics_focus = [m for m in ["firing_rate_Hz", "max_firing_rate_1s_Hz", "burstiness_percent", "population_coupling", "mean_plv_theta", "mean_plv_low_gamma", "mean_plv_high_gamma", "halfwidth1_ms", "halfwidth2_ms"] if m in master.columns]

    def _sig_text(df_stats: pd.DataFrame, metric_subset: List[str]) -> str:
        if df_stats is None or df_stats.empty:
            return ""
        use = df_stats.copy()
        use["q_eff"] = pd.to_numeric(use["q_fdr"] if "q_fdr" in use.columns else use.get("p", np.nan), errors="coerce")
        use = use[np.isfinite(use["q_eff"]) & (use["q_eff"] < 0.05)]
        if use.empty:
            return "none"
        if metric_subset:
            use = use[use.get("metric", pd.Series("", index=use.index)).isin(metric_subset)]
            if use.empty:
                return "none"
        out = []
        for _, r in use.sort_values("q_eff").head(8).iterrows():
            ga = r.get("group_a", "")
            gb = r.get("group_b", "")
            m = r.get("metric", "")
            q = r.get("q_eff", np.nan)
            out.append(f"{m}:{ga} vs {gb} (q={q:.2g})")
        return "; ".join(out) if out else "none"

    for state, bands in bands_by_state.items():
        state_df = master[master["state_norm"] == state].copy()
        if state_df.empty:
            continue
        for band in bands:
            sig_col = f"mean_plv_sig_{band}" if f"mean_plv_sig_{band}" in state_df.columns else None
            if sig_col is None:
                continue
            state_df[sig_col] = pd.to_numeric(state_df[sig_col], errors="coerce")
            state_df["__is_sig__"] = np.isfinite(state_df[sig_col]) & (state_df[sig_col] > 0)
            sig_only = state_df[state_df["__is_sig__"]].copy()
            all_n = int(len(state_df))
            sig_n = int(len(sig_only))
            ratio = (sig_n / all_n) if all_n > 0 else np.nan

            row = {
                "state": state,
                "frequency_band": band,
                "total_cells_n": all_n,
                "sig_cells_n": sig_n,
                "sig_ratio": ratio,
            }
            # group ratios
            for gcol, gorder in groupings:
                for g in gorder:
                    sub_all = state_df[state_df[gcol] == g]
                    if len(sub_all) == 0:
                        row[f"{gcol}__{g}__ratio_sig"] = np.nan
                        row[f"{gcol}__{g}__n"] = 0
                    else:
                        row[f"{gcol}__{g}__ratio_sig"] = float((sub_all["__is_sig__"]).mean())
                        row[f"{gcol}__{g}__n"] = int(len(sub_all))

            # significant differences text blocks
            sig_stats_rows = []
            all_stats_rows = []
            for metric in metrics_focus:
                if metric not in state_df.columns:
                    continue
                for gcol, _gorder in groupings:
                    s1 = compare_groups_unmatched(sig_only, metric, group_col=gcol, groups=tuple(sig_only[gcol].dropna().unique()[:2])) if len(sig_only[gcol].dropna().unique()) >= 2 else pd.DataFrame()
                    s2 = compare_groups_unmatched(state_df, metric, group_col=gcol, groups=tuple(state_df[gcol].dropna().unique()[:2])) if len(state_df[gcol].dropna().unique()) >= 2 else pd.DataFrame()
                    if not s1.empty:
                        sig_stats_rows.append(s1)
                    if not s2.empty:
                        all_stats_rows.append(s2)
            sig_stats_df = concat_nonempty(sig_stats_rows)
            all_stats_df = concat_nonempty(all_stats_rows)
            row["sig_only_differences_text"] = _sig_text(sig_stats_df, metrics_focus)
            row["all_cells_differences_text"] = _sig_text(all_stats_df, metrics_focus)
            rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_csv(os.path.join(out_dir, "table_coupling_statistical_reporting.csv"), index=False)
    return out

def _plot_phase_hist_ax(ax, phases, weights=None, title="", color="#4C78A8", weight_darkness=False, rmax: Optional[float] = None, log_scale: bool = False):
    phases = np.asarray(phases, dtype=float)
    mask = np.isfinite(phases)
    phases = phases[mask]
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        weights = weights[mask] if weights.size == mask.size else None
    if phases.size == 0:
        ax.set_title(f"{title}\nn=0")
        return
    bins = np.linspace(-np.pi, np.pi, 25)
    counts, edges = np.histogram(phases, bins=bins)
    if log_scale:
        counts = np.log10(counts + 1.0)
    if weight_darkness and weights is not None:
        wsum, _ = np.histogram(phases, bins=bins, weights=np.nan_to_num(weights, nan=0.0))
        mean_w = np.divide(wsum, counts, out=np.zeros_like(wsum, dtype=float), where=counts > 0)
        norm = mean_w / np.nanmax(mean_w) if np.nanmax(mean_w) > 0 else mean_w
        colors = [matplotlib.colors.to_rgba(color, 0.25 + 0.65 * v) for v in norm]
        ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color=colors, edgecolor="white")
    else:
        ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color=color, alpha=0.62, edgecolor="white")
    pref, r = circular_mean(phases, weights)
    if np.isfinite(pref):
        ymax = max(float(np.max(counts)), 1.0)
        ax.plot([pref, pref], [0, ymax], color="black", lw=2.1)
        ax.scatter([pref], [ymax], s=32, color="black", zorder=5)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    if rmax is not None and np.isfinite(rmax):
        ax.set_ylim(0, np.log10(rmax + 1.0) if log_scale else rmax)
    ax.set_title(f"{title}\nn={phases.size}, R={r:.2f}", fontsize=10)

def plot_phase_sine_map(df: pd.DataFrame, phase_col: str, weight_col: str, band: str, out_dir: str):
    if df.empty:
        return
    fig, axs = plt.subplots(len(STATE_ORDER), 1, figsize=(11.0, 2.8 * len(STATE_ORDER)), sharex=True)
    axs = np.atleast_1d(axs)
    x = np.linspace(0, 360, 361)
    y = np.sin(np.deg2rad(x))
    for ax, state in zip(axs, STATE_ORDER):
        ax.plot(x, y, color="#222222", lw=1.2)
        sub = df[df["state_norm"] == state].copy()
        for ct in CELLTYPE_ORDER:
            ss = sub[sub["celltype_norm"] == ct]
            if ss.empty:
                continue
            deg = (np.degrees(ss[phase_col].astype(float)) % 360.0).values
            inten = pd.to_numeric(ss[weight_col], errors="coerce").fillna(0).values
            sizes = 25 + 90 * (inten / max(np.nanmax(inten), 1e-12))
            alphas = 0.20 + 0.75 * (inten / max(np.nanmax(inten), 1e-12))
            colors = [matplotlib.colors.to_rgba(COLORS_CELLTYPE.get(ct, "#777777"), float(alpha)) for alpha in alphas]
            ax.scatter(deg, np.sin(np.deg2rad(deg)), s=sizes, color=colors, edgecolor="black", linewidth=0.25, label=ct)
        phases = (np.degrees(sub[phase_col].astype(float)) % 360.0).values
        if len(phases) >= 5:
            counts, edges = np.histogram(phases, bins=np.arange(0, 361, 30))
            if counts.max() > 0:
                thr = np.percentile(counts[counts > 0], 75)
                for i, count in enumerate(counts):
                    if count >= thr and count > 0:
                        ax.plot([edges[i], edges[i + 1]], [1.12, 1.12], color="#111111", lw=3.0, solid_capstyle="butt")
        ax.set_ylim(-1.25, 1.25)
        ax.set_ylabel(state)
        ax.spines["left"].set_visible(False)
        ax.set_yticks([-1, 0, 1])
    handles, labels = axs[0].get_legend_handles_labels()
    uniq = {}
    for h, lab in zip(handles, labels):
        if lab not in uniq:
            uniq[lab] = h
    if uniq:
        axs[0].legend(uniq.values(), uniq.keys(), title="celltype", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    axs[-1].set_xlabel("preferred phase (degrees)")
    fig.suptitle(f"Significantly coupled cells on phase sine wave: {band}", y=1.01)
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, f"phase_sine_map__{band}.png"))
    plt.close(fig)

def build_histo_ephys_tables(master: pd.DataFrame, histology: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if histology is None or histology.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    h = histology.copy()
    hist_summary = (
        h.groupby(["staining", "patient", "histology_state", "layer_norm"], dropna=False)
        .agg(
            histology_n_samples=("cell_number", "count"),
            mean_cell_number=("cell_number", "mean"),
            mean_density_cells_mm2=("density (cell/mm^2)", "mean"),
            mean_coverage_percent=("coverage (%)", "mean"),
        )
        .reset_index()
    )

    e = master.copy()
    e["histology_state"] = e["state_norm"].map(lambda s: "vitro" if s == "vitro" else "vivo")
    celltype_series = e["celltype_norm"].astype(str) if "celltype_norm" in e.columns else pd.Series("", index=e.index)
    in_like = celltype_series.isin(["IN", "FS", "interneuron"])
    e["ephys_cell_group"] = np.where(in_like, "inh_like", "all_units")

    metrics = [m for m in [
        "firing_rate_Hz", "max_firing_rate_1s_Hz", "population_coupling", "coupling_strength_low_gamma", "coupling_strength_high_gamma",
        "plv_low_gamma_strength", "plv_high_gamma_strength", "frsim_strength", "frsim_clustering",
        # Added delta/theta
        "mean_plv_delta", "max_plv_delta", "mean_plv_sig_delta", "max_plv_sig_delta", "resultant_length_sig_delta",
        "mean_plv_theta", "max_plv_theta", "mean_plv_sig_theta", "max_plv_sig_theta", "resultant_length_sig_theta",
    ] if m in e.columns]

    group_rows = []
    for label, sub in [("all_cells", e), ("inh_like", e[in_like]), ("fs_inh", e[in_like])]:
        if sub.empty:
            continue
        agg = sub.groupby(["patient", "histology_state", "state_norm", "layer_norm"], dropna=False).agg(
            n_recorded_units=("unit_id", "count"),
            **{f"mean_{m}": (m, "mean") for m in metrics},
            **{f"median_{m}": (m, "median") for m in metrics},
        ).reset_index()
        agg["ephys_cell_group"] = label
        group_rows.append(agg)
    ephys_summary = pd.concat(group_rows, ignore_index=True, sort=False) if group_rows else pd.DataFrame()

    pv = hist_summary[hist_summary["staining"] == "PV"].rename(columns={
        "mean_density_cells_mm2": "pv_density_cells_mm2",
        "mean_coverage_percent": "pv_coverage_percent",
        "mean_cell_number": "pv_cell_number",
    })
    neun = hist_summary[hist_summary["staining"] == "NEUN"].rename(columns={
        "mean_density_cells_mm2": "neun_density_cells_mm2",
        "mean_coverage_percent": "neun_coverage_percent",
        "mean_cell_number": "neun_cell_number",
    })
    hist_wide = pv.merge(
        neun[["patient", "histology_state", "layer_norm", "neun_density_cells_mm2", "neun_coverage_percent", "neun_cell_number"]],
        on=["patient", "histology_state", "layer_norm"],
        how="outer",
    )
    hist_wide["pv_neun_density_ratio"] = hist_wide["pv_density_cells_mm2"] / hist_wide["neun_density_cells_mm2"]
    hist_wide["pv_neun_coverage_ratio"] = hist_wide["pv_coverage_percent"] / hist_wide["neun_coverage_percent"]
    hist_wide["pv_fraction_of_neun_density"] = hist_wide["pv_density_cells_mm2"] / hist_wide["neun_density_cells_mm2"]

    merged = ephys_summary.merge(hist_wide, on=["patient", "histology_state", "layer_norm"], how="left")
    if "n_recorded_units" in merged.columns:
        merged["extrapolated_pv_units_from_density_ratio"] = merged["n_recorded_units"] * merged["pv_neun_density_ratio"]
        merged["recorded_units_per_neun_density"] = merged["n_recorded_units"] / merged["neun_density_cells_mm2"]
        merged["recorded_units_per_pv_density"] = merged["n_recorded_units"] / merged["pv_density_cells_mm2"]
    return hist_summary, ephys_summary, merged

def build_histo_ephys_nonaveraged(master: pd.DataFrame, histology: pd.DataFrame) -> pd.DataFrame:
    if histology is None or histology.empty:
        return pd.DataFrame()
    hist = histology.copy()
    pv = hist[hist["staining"] == "PV"].rename(columns={"density (cell/mm^2)": "pv_density_cells_mm2", "coverage (%)": "pv_coverage_percent", "cell_number": "pv_cell_number"})
    neun = hist[hist["staining"] == "NEUN"].rename(columns={"density (cell/mm^2)": "neun_density_cells_mm2", "coverage (%)": "neun_coverage_percent", "cell_number": "neun_cell_number"})
    keys = ["patient", "histology_state", "layer_norm"]
    rows = []
    for key, pvg in pv.groupby(keys, dropna=False):
        neung = neun
        for k, val in zip(keys, key if isinstance(key, tuple) else (key,)):
            neung = neung[neung[k] == val]
        if neung.empty:
            continue
        for _, pr in pvg.iterrows():
            for _, nr in neung.iterrows():
                rows.append({
                    **dict(zip(keys, key if isinstance(key, tuple) else (key,))),
                    "pv_density_cells_mm2": pr.get("pv_density_cells_mm2", np.nan),
                    "pv_coverage_percent": pr.get("pv_coverage_percent", np.nan),
                    "pv_cell_number": pr.get("pv_cell_number", np.nan),
                    "neun_density_cells_mm2": nr.get("neun_density_cells_mm2", np.nan),
                    "neun_coverage_percent": nr.get("neun_coverage_percent", np.nan),
                    "neun_cell_number": nr.get("neun_cell_number", np.nan),
                })
    hist_pairs = pd.DataFrame(rows)
    if hist_pairs.empty:
        return pd.DataFrame()
    hist_pairs["pv_neun_density_ratio"] = hist_pairs["pv_density_cells_mm2"] / hist_pairs["neun_density_cells_mm2"]
    hist_pairs["pv_neun_coverage_ratio"] = hist_pairs["pv_coverage_percent"] / hist_pairs["neun_coverage_percent"]

    _hs, ephys_summary, _merged = build_histo_ephys_tables(master, histology)
    if ephys_summary.empty:
        return pd.DataFrame()
    return ephys_summary.merge(hist_pairs, on=["patient", "histology_state", "layer_norm"], how="inner")


def _zscore_safe(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").astype(float)
    mu = vals.mean(skipna=True)
    sd = vals.std(skipna=True)
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(np.nan, index=vals.index)
    return (vals - mu) / sd


def _prepare_histology_wide_for_mixedlm(histology: pd.DataFrame) -> pd.DataFrame:
    if histology is None or histology.empty:
        return pd.DataFrame()

    h = histology.copy()
    h["staining"] = h.get("staining", pd.Series(index=h.index, dtype=object)).astype(str).str.strip().str.upper()
    h["patient"] = h.get("patient", pd.Series(index=h.index, dtype=object)).astype(str).str.strip()
    h["histology_state"] = h.get("histology_state", pd.Series(index=h.index, dtype=object)).astype(str).str.strip().str.lower()
    h["layer_norm"] = h.get("layer_norm", pd.Series(index=h.index, dtype=object)).map(normalize_layer)

    for col in ["density (cell/mm^2)", "coverage (%)"]:
        if col in h.columns:
            h[col] = pd.to_numeric(h[col], errors="coerce")

    grouped = (
        h.groupby(["staining", "patient", "histology_state", "layer_norm"], dropna=False)
         .agg(
             mean_density_cells_mm2=("density (cell/mm^2)", "mean"),
             mean_coverage_percent=("coverage (%)", "mean"),
         )
         .reset_index()
    )
    if grouped.empty:
        return pd.DataFrame()

    pv = grouped[grouped["staining"] == "PV"].rename(
        columns={
            "mean_density_cells_mm2": "pv_density_cells_mm2",
            "mean_coverage_percent": "pv_coverage_percent",
        }
    )
    neun = grouped[grouped["staining"] == "NEUN"].rename(
        columns={
            "mean_density_cells_mm2": "neun_density_cells_mm2",
            "mean_coverage_percent": "neun_coverage_percent",
        }
    )

    out = pv.merge(
        neun[["patient", "histology_state", "layer_norm", "neun_density_cells_mm2", "neun_coverage_percent"]],
        on=["patient", "histology_state", "layer_norm"],
        how="outer",
    )
    out["pv_neun_density_ratio"] = out["pv_density_cells_mm2"] / out["neun_density_cells_mm2"]
    out = out.dropna(subset=["patient", "histology_state", "layer_norm"]).copy()
    return out


def build_histo_ephys_unit_association_table(master: pd.DataFrame, histology: pd.DataFrame) -> pd.DataFrame:
    if master is None or master.empty or histology is None or histology.empty:
        return pd.DataFrame()

    hist_wide = _prepare_histology_wide_for_mixedlm(histology)
    if hist_wide.empty:
        return pd.DataFrame()

    m = master.copy()
    m["patient"] = m.get("patient", pd.Series(index=m.index, dtype=object)).astype(str).str.strip()
    m["state_norm"] = m.get("state_norm", pd.Series(index=m.index, dtype=object)).map(normalize_state)
    m["layer_norm"] = m.get("layer_norm", pd.Series(index=m.index, dtype=object)).map(normalize_layer)
    m["celltype_norm"] = m.get("celltype_norm", pd.Series(index=m.index, dtype=object)).map(normalize_celltype)
    m["histology_state"] = np.where(m["state_norm"] == "vitro", "vitro", "vivo")

    m["cellclass_mixedlm"] = "other"
    m.loc[m["celltype_norm"].isin(["RS-PC", "IB-PC"]), "cellclass_mixedlm"] = "Exc"
    m.loc[m["celltype_norm"].isin(["IN", "FS"]), "cellclass_mixedlm"] = "Inh"
    m.loc[m["celltype_norm"] == "FS", "cellclass_mixedlm"] = "FS"

    ephys_metrics = [metric for metric in EPHYS_MIXEDLM_METRICS if metric in m.columns]
    if not ephys_metrics:
        return pd.DataFrame()

    keep = ["patient", "histology_state", "state_norm", "layer_norm", "cellclass_mixedlm"] + ephys_metrics
    unit = m[keep].copy()

    all_view = unit.copy()
    all_view["cellclass_mixedlm"] = "all"
    classes_view = unit[unit["cellclass_mixedlm"].isin(["Exc", "Inh", "FS"])].copy()
    unit_stack = pd.concat([all_view, classes_view], ignore_index=True)
    if unit_stack.empty:
        return pd.DataFrame()

    e_long = unit_stack.melt(
        id_vars=["patient", "histology_state", "state_norm", "layer_norm", "cellclass_mixedlm"],
        value_vars=ephys_metrics,
        var_name="ephys_metric",
        value_name="ephys_value",
    )
    e_long["ephys_value"] = pd.to_numeric(e_long["ephys_value"], errors="coerce")

    hist_metrics = [metric for metric in HISTOLOGY_MIXEDLM_METRICS if metric in hist_wide.columns]
    if not hist_metrics:
        return pd.DataFrame()

    h_long = hist_wide.melt(
        id_vars=["patient", "histology_state", "layer_norm"],
        value_vars=hist_metrics,
        var_name="histology_metric",
        value_name="histology_value",
    )
    h_long["histology_value"] = pd.to_numeric(h_long["histology_value"], errors="coerce")

    merged = h_long.merge(e_long, on=["patient", "histology_state", "layer_norm"], how="inner")
    merged = merged[
        merged["histology_state"].isin(MIXEDLM_STATE_ORDER)
        & merged["cellclass_mixedlm"].isin(MIXEDLM_CELLCLASS_ORDER)
        & np.isfinite(merged["histology_value"]) & np.isfinite(merged["ephys_value"])
    ].copy()
    return merged


def fit_histo_ephys_mixed_association(sub: pd.DataFrame) -> Dict[str, object]:
    sub = sub.copy()
    sub["hist_z"] = _zscore_safe(sub["histology_value"])
    sub["ephys_z"] = _zscore_safe(sub["ephys_value"])
    sub = sub[np.isfinite(sub["hist_z"]) & np.isfinite(sub["ephys_z"])].copy()

    if len(sub) < 6 or sub["patient"].nunique() < 3:
        return {"beta_std": np.nan, "p_value": np.nan, "method": "insufficient_data", "converged": False}

    terms = ["hist_z"]
    if sub["layer_norm"].nunique(dropna=True) > 1:
        terms.append("C(layer_norm)")
    if sub["state_norm"].nunique(dropna=True) > 1:
        terms.append("C(state_norm)")
    formula = "ephys_z ~ " + " + ".join(terms)

    if smf is None:
        return {"beta_std": np.nan, "p_value": np.nan, "method": "statsmodels_unavailable", "converged": False}

    try:
        model = smf.mixedlm(formula, sub, groups=sub["patient"], re_formula="1")
        result = model.fit(reml=False, method="lbfgs", maxiter=250, disp=False)
        return {
            "beta_std": float(result.params.get("hist_z", np.nan)),
            "p_value": float(result.pvalues.get("hist_z", np.nan)),
            "method": "mixedlm",
            "converged": bool(getattr(result, "converged", True)),
        }
    except Exception:
        try:
            ols = smf.ols(formula, sub).fit(cov_type="cluster", cov_kwds={"groups": sub["patient"]})
            return {
                "beta_std": float(ols.params.get("hist_z", np.nan)),
                "p_value": float(ols.pvalues.get("hist_z", np.nan)),
                "method": "ols_cluster",
                "converged": True,
            }
        except Exception:
            return {"beta_std": np.nan, "p_value": np.nan, "method": "failed", "converged": False}


def run_histo_ephys_mixedlm_associations(master: pd.DataFrame, histology: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    merged = build_histo_ephys_unit_association_table(master, histology)
    if merged.empty:
        return pd.DataFrame()

    rows = []
    for hist_state in MIXEDLM_STATE_ORDER:
        for cellclass in MIXEDLM_CELLCLASS_ORDER:
            sub_cell = merged[(merged["histology_state"] == hist_state) & (merged["cellclass_mixedlm"] == cellclass)].copy()
            if sub_cell.empty:
                continue

            for hmet in [m for m in HISTOLOGY_MIXEDLM_METRICS if m in set(sub_cell["histology_metric"])]:
                for emet in [m for m in EPHYS_MIXEDLM_METRICS if m in set(sub_cell["ephys_metric"])]:
                    sub = sub_cell[(sub_cell["histology_metric"] == hmet) & (sub_cell["ephys_metric"] == emet)].copy()
                    fit = fit_histo_ephys_mixed_association(sub)
                    if not np.isfinite(fit.get("beta_std", np.nan)):
                        continue
                    rows.append(
                        {
                            "histology_state": hist_state,
                            "cellclass_ei": cellclass,
                            "histology_metric": hmet,
                            "ephys_metric": emet,
                            "beta_std": fit["beta_std"],
                            "p": fit["p_value"],
                            "method": fit["method"],
                            "converged": fit["converged"],
                            "n_units": int(len(sub)),
                            "n_patients": int(sub["patient"].nunique()),
                            "n_layers": int(sub["layer_norm"].nunique()),
                            "n_states": int(sub["state_norm"].nunique()),
                        }
                    )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["q_fdr"] = fdr_bh(pd.to_numeric(out["p"], errors="coerce").values)
    out = out.sort_values(["q_fdr", "p"], na_position="last")
    out.to_csv(os.path.join(out_dir, "stats_histo_ephys_mixedlm.csv"), index=False)

    for hist_state in MIXEDLM_STATE_ORDER:
        for cellclass in MIXEDLM_CELLCLASS_ORDER:
            sub = out[(out["histology_state"] == hist_state) & (out["cellclass_ei"] == cellclass)].copy()
            sub.to_csv(os.path.join(out_dir, f"stats_histo_ephys_mixedlm__{hist_state}__{cellclass}.csv"), index=False)

    return out


def plot_histo_ephys_mixedlm_matrices(mixed_df: pd.DataFrame, out_dir: str) -> None:
    if mixed_df is None or mixed_df.empty:
        return

    plot_dir = os.path.join(out_dir, "histo_ephys_mixedlm_matrices")
    os.makedirs(plot_dir, exist_ok=True)

    for hist_state in MIXEDLM_STATE_ORDER:
        for cellclass in MIXEDLM_CELLCLASS_ORDER:
            sub = mixed_df[(mixed_df["histology_state"] == hist_state) & (mixed_df["cellclass_ei"] == cellclass)].copy()
            if sub.empty:
                continue

            mat = sub.pivot_table(index="ephys_metric", columns="histology_metric", values="beta_std", aggfunc="mean")
            qmat = sub.pivot_table(index="ephys_metric", columns="histology_metric", values="q_fdr", aggfunc="mean")

            hist_order = [h for h in HISTOLOGY_MIXEDLM_METRICS if h in mat.columns]
            ephys_order = [e for e in EPHYS_MIXEDLM_METRICS if e in mat.index]
            mat = mat.reindex(index=ephys_order, columns=hist_order)
            qmat = qmat.reindex(index=ephys_order, columns=hist_order)
            mat = mat.dropna(axis=0, how="all").dropna(axis=1, how="all")
            qmat = qmat.reindex(index=mat.index, columns=mat.columns)
            if mat.empty:
                continue

            display = mat.rename(index=EPHYS_MIXEDLM_LABELS, columns=HISTOLOGY_MIXEDLM_LABELS)
            qdisp = qmat.rename(index=EPHYS_MIXEDLM_LABELS, columns=HISTOLOGY_MIXEDLM_LABELS)

            annot = pd.DataFrame("", index=display.index, columns=display.columns)
            for ridx in display.index:
                for cidx in display.columns:
                    beta = display.loc[ridx, cidx]
                    q = qdisp.loc[ridx, cidx]
                    if pd.notna(beta):
                        mark = "*" if pd.notna(q) and q < 0.05 else ""
                        annot.loc[ridx, cidx] = f"{mark}{beta:.2f}"

            fig, ax = plt.subplots(figsize=(1.7 * len(display.columns) + 3.8, 0.65 * len(display.index) + 2.9))
            sns.heatmap(
                display,
                vmin=-1,
                vmax=1,
                center=0,
                cmap="vlag",
                linewidths=0.55,
                linecolor="white",
                annot=annot.values,
                fmt="",
                cbar_kws={"label": "Std. beta"},
                ax=ax,
            )
            title_state = "in vivo" if hist_state == "vivo" else "in vitro"
            ax.set_title(f"Histology mixed-effects matrix: {title_state}, {cellclass}")
            ax.set_xlabel("histology")
            ax.set_ylabel("ephys")
            fig.tight_layout()
            save_figure(fig, os.path.join(plot_dir, f"histo_ephys_mixedlm_matrix__{hist_state}__{cellclass}.png"))
            plt.close(fig)

def correlate_histo_ephys(merged: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    if merged is None or merged.empty:
        return pd.DataFrame()
    plot_dir = os.path.join(out_dir, "histo_ephys_correlation_plots")
    os.makedirs(plot_dir, exist_ok=True)

    hist_metrics = [c for c in ["pv_density_cells_mm2", "pv_coverage_percent", "neun_density_cells_mm2", "neun_coverage_percent", "pv_neun_density_ratio", "pv_neun_coverage_ratio"] if c in merged.columns]
    ephys_metrics = [c for c in merged.columns if c.startswith("mean_") and any(k in c for k in ["firing_rate", "population_coupling", "coupling_strength", "plv_", "frsim"]) ]
    rows = []

    # record patients that have both in vivo and in vitro histology and ephys
    try:
        pat_states = merged.groupby("patient")["histology_state"].apply(lambda x: set([s for s in x.dropna().astype(str).str.strip().str.lower()]))
        patients_with_both = [p for p, s in pat_states.items() if {"vivo", "vitro"}.issubset(s)]
        if patients_with_both:
            pd.DataFrame({"patient": patients_with_both}).to_csv(os.path.join(out_dir, "patients_with_both_vivo_vitro_histology_and_ephys.csv"), index=False)
    except Exception:
        patients_with_both = []

    for cell_group in sorted(merged.get("ephys_cell_group", pd.Series(["all_cells"]) ).dropna().unique()):
        for e_state in sorted(merged.get("state_norm", pd.Series(dtype=object)).dropna().unique()):
            base = merged[(merged["ephys_cell_group"] == cell_group) & (merged["state_norm"] == e_state)].copy()
            if base.empty:
                continue
            for hist_state in base.get("histology_state", pd.Series(dtype=object)).dropna().unique():
                sub_base = base[base["histology_state"].astype(str).str.strip().str.lower() == str(hist_state).strip().lower()].copy()
                if sub_base.empty:
                    continue
                for hmet in hist_metrics:
                    for emet in ephys_metrics:
                        sub = sub_base[["patient", "layer_norm", hmet, emet]].copy()
                        sub[hmet] = pd.to_numeric(sub[hmet], errors="coerce")
                        sub[emet] = pd.to_numeric(sub[emet], errors="coerce")
                        sub = sub[np.isfinite(sub[hmet]) & np.isfinite(sub[emet])]
                        if len(sub) < 3:
                            continue
                        rho, p = stats.spearmanr(sub[hmet], sub[emet])
                        pear_r, pear_p = stats.pearsonr(sub[hmet], sub[emet]) if len(sub) >= 3 else (np.nan, np.nan)
                        rows.append({
                            "ephys_cell_group": cell_group,
                            "state_norm": e_state,
                            "histology_state": hist_state,
                            "histology_metric": hmet,
                            "ephys_metric": emet,
                            "relation_focus": (
                                "neun_vs_all_cells" if (str(hmet).startswith("neun_") and str(cell_group) == "all_cells") else
                                "pv_vs_fs_inh" if (str(hmet).startswith("pv_") and str(cell_group) in ["fs_inh", "inh_like"]) else
                                "other"
                            ),
                            "n_patient_layer_points": int(len(sub)),
                            "n_patients": int(sub["patient"].nunique()),
                            "spearman_rho": float(rho),
                            "spearman_p": float(p),
                            "pearson_r": float(pear_r),
                            "pearson_p": float(pear_p),
                        })

    stats_df = pd.DataFrame(rows)
    if not stats_df.empty and "spearman_p" in stats_df.columns:
        stats_df["spearman_q_fdr"] = fdr_bh(stats_df["spearman_p"].values)
        if "pearson_p" in stats_df.columns:
            stats_df["pearson_q_fdr"] = fdr_bh(stats_df["pearson_p"].values)
        stats_df = stats_df.sort_values("spearman_q_fdr", na_position="last")
        stats_df.to_csv(os.path.join(out_dir, "stats_histo_ephys_correlations.csv"), index=False)
        focus = stats_df[stats_df["relation_focus"].isin(["neun_vs_all_cells", "pv_vs_fs_inh"])].copy()
        if not focus.empty:
            focus.to_csv(os.path.join(out_dir, "stats_histo_ephys_focus_neun_all_pv_fsinh.csv"), index=False)
        plot_histo_ephys_correlation_matrices(stats_df, out_dir)
        for _, r in stats_df.head(24).iterrows():
            _plot_histo_ephys_scatter(merged, r, plot_dir)
    return stats_df

def export_histo_ephys_mean_tables(merged: pd.DataFrame, out_dir: str) -> None:
    if merged is None or merged.empty:
        return
    d = merged.copy()
    hist_metrics = [c for c in ["pv_density_cells_mm2", "pv_coverage_percent", "neun_density_cells_mm2", "neun_coverage_percent", "pv_neun_density_ratio", "pv_neun_coverage_ratio"] if c in d.columns]
    ephys_metrics = [c for c in d.columns if c.startswith("mean_")]
    if not hist_metrics or not ephys_metrics:
        return
    mean_tbl = d.groupby(["ephys_cell_group", "state_norm", "histology_state", "layer_norm"], dropna=False)[hist_metrics + ephys_metrics].mean(numeric_only=True).reset_index()
    mean_tbl.to_csv(os.path.join(out_dir, "table_histo_ephys_group_means.csv"), index=False)

def compute_bare_histology_stats(histology: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    if histology is None or histology.empty:
        return pd.DataFrame()
    h = histology.copy()
    metrics = [c for c in ["density (cell/mm^2)", "coverage (%)", "cell_number"] if c in h.columns]
    rows = []
    for stain in sorted(h.get("staining", pd.Series(dtype=object)).dropna().astype(str).str.upper().unique()):
        hs = h[h["staining"].astype(str).str.upper() == stain].copy()
        for metric in metrics:
            hs[metric] = pd.to_numeric(hs[metric], errors="coerce")
            sub = hs[np.isfinite(hs[metric]) & hs["histology_state"].isin(["vivo", "vitro"]) & hs["layer_norm"].isin(["supra", "infra"])]
            if sub.empty:
                continue
            # vivo vs vitro
            try:
                a = sub[sub["histology_state"] == "vivo"][metric].dropna().values
                b = sub[sub["histology_state"] == "vitro"][metric].dropna().values
                if len(a) >= 2 and len(b) >= 2:
                    stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                    rows.append({"staining": stain, "metric": metric, "comparison": "vivo_vs_vitro", "n_a": len(a), "n_b": len(b), "mean_a": float(np.nanmean(a)), "mean_b": float(np.nanmean(b)), "median_a": float(np.nanmedian(a)), "median_b": float(np.nanmedian(b)), "p": float(p), "effect_cliffs_delta": cliffs_delta(a, b)})
            except Exception:
                pass
            # supra vs infra
            try:
                a = sub[sub["layer_norm"] == "supra"][metric].dropna().values
                b = sub[sub["layer_norm"] == "infra"][metric].dropna().values
                if len(a) >= 2 and len(b) >= 2:
                    stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                    rows.append({"staining": stain, "metric": metric, "comparison": "supra_vs_infra", "n_a": len(a), "n_b": len(b), "mean_a": float(np.nanmean(a)), "mean_b": float(np.nanmean(b)), "median_a": float(np.nanmedian(a)), "median_b": float(np.nanmedian(b)), "p": float(p), "effect_cliffs_delta": cliffs_delta(a, b)})
            except Exception:
                pass
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_fdr"] = fdr_bh(out["p"].values)
        out.to_csv(os.path.join(out_dir, "stats_bare_histology.csv"), index=False)
        summary = h.groupby(["staining", "histology_state", "layer_norm"], dropna=False)[metrics].mean(numeric_only=True).reset_index()
        summary.to_csv(os.path.join(out_dir, "table_bare_histology_means.csv"), index=False)
    return out

def plot_burstiness_variants(master: pd.DataFrame, out_dir: str) -> None:
    plot_dir = os.path.join(out_dir, "burstiness_plots")
    os.makedirs(plot_dir, exist_ok=True)
    metrics = [m for m in ["burstiness_percent", "bursti_2"] if m in master.columns]
    if not metrics:
        return

    for metric in metrics:
        df = master[master["state_norm"].isin(STATE_ORDER) & master["celltype_norm"].isin(["RS-PC", "IB-PC", "IN", "FS"])].copy()
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
        df = df[np.isfinite(df[metric])]
        if df.empty:
            continue

        # linear
        boxstrip_plot(
            df=df,
            metric=metric,
            x="state_norm",
            hue="celltype_norm",
            out_png=os.path.join(plot_dir, f"{metric}__linear.png"),
            title=f"{metric.replace('_', ' ').title()} (linear)",
            order=STATE_ORDER,
            hue_order=["RS-PC", "IB-PC", "IN", "FS"],
        )

        # log-scale view
        fig, ax = plt.subplots(figsize=(10.4, 6.0))
        sns.boxplot(data=df, x="state_norm", y=metric, hue="celltype_norm", order=STATE_ORDER, hue_order=["RS-PC", "IB-PC", "IN", "FS"], palette=COLORS_CELLTYPE, showfliers=False, ax=ax)
        sns.stripplot(data=df, x="state_norm", y=metric, hue="celltype_norm", order=STATE_ORDER, hue_order=["RS-PC", "IB-PC", "IN", "FS"], dodge=True, palette=COLORS_CELLTYPE, alpha=0.35, size=2.5, linewidth=0.25, edgecolor="black", ax=ax)
        ymin = max(float(df[metric][df[metric] > 0].min()) if np.any(df[metric] > 0) else 1e-3, 1e-3)
        ymax = float(df[metric].max())
        if ymax > ymin:
            ax.set_yscale("log")
            ax.set_ylim(ymin, ymax * 1.12)
        handles, labels = ax.get_legend_handles_labels()
        uniq = {}
        for h, lab in zip(handles, labels):
            if lab not in uniq:
                uniq[lab] = h
        ax.legend(uniq.values(), uniq.keys(), title="celltype", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
        ax.set_title(f"{metric.replace('_', ' ').title()} (log scale)")
        ax.set_xlabel("")
        fig.tight_layout()
        save_figure(fig, os.path.join(plot_dir, f"{metric}__log.png"))
        plt.close(fig)

        # broken-axis view (linear) with fixed break for burstiness_percent
        if metric == "burstiness_percent":
            low_max = 20.0
            high_min = 40.0
            ymax = float(np.nanmax(df[metric])) if len(df) else np.nan
            if np.isfinite(ymax) and ymax > high_min:
                fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10.4, 7.0), sharex=True, gridspec_kw={"height_ratios": [1, 2]})
                for ax in [ax_top, ax_bot]:
                    sns.boxplot(data=df, x="state_norm", y=metric, hue="celltype_norm", order=STATE_ORDER, hue_order=["RS-PC", "IB-PC", "IN", "FS"], palette=COLORS_CELLTYPE, showfliers=False, ax=ax)
                    sns.stripplot(data=df, x="state_norm", y=metric, hue="celltype_norm", order=STATE_ORDER, hue_order=["RS-PC", "IB-PC", "IN", "FS"], dodge=True, palette=COLORS_CELLTYPE, alpha=0.32, size=2.3, linewidth=0.25, edgecolor="black", ax=ax)
                ax_bot.set_ylim(max(0.0, float(np.nanmin(df[metric])) * 0.98 if np.isfinite(np.nanmin(df[metric])) else 0.0), low_max)
                ax_top.set_ylim(high_min, ymax * 1.03)
                ax_top.spines["bottom"].set_visible(False)
                ax_bot.spines["top"].set_visible(False)
                ax_top.tick_params(labeltop=False)
                ax_bot.xaxis.tick_bottom()
                d = .006
                kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
                ax_top.plot((-d, +d), (-d, +d), **kwargs)
                ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
                kwargs.update(transform=ax_bot.transAxes)
                ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)
                ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
                if ax_top.legend_:
                    ax_top.legend_.remove()
                handles, labels = ax_bot.get_legend_handles_labels()
                uniq = {}
                for h, lab in zip(handles, labels):
                    if lab not in uniq:
                        uniq[lab] = h
                ax_bot.legend(uniq.values(), uniq.keys(), title="celltype", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
                fig.suptitle(f"{metric.replace('_', ' ').title()} (broken linear: 20-40)", y=0.99)
                fig.tight_layout()
                save_figure(fig, os.path.join(plot_dir, f"{metric}__broken_linear_20_40.png"))
                plt.close(fig)

        # percentile-based broken-axis view for remaining metrics
        p90 = float(np.nanpercentile(df[metric], 90))
        p99 = float(np.nanpercentile(df[metric], 99))
        if np.isfinite(p90) and np.isfinite(p99) and p99 > p90:
            fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10.4, 7.0), sharex=True, gridspec_kw={"height_ratios": [1, 2]})
            for ax in [ax_top, ax_bot]:
                sns.boxplot(data=df, x="state_norm", y=metric, hue="celltype_norm", order=STATE_ORDER, hue_order=["RS-PC", "IB-PC", "IN", "FS"], palette=COLORS_CELLTYPE, showfliers=False, ax=ax)
                sns.stripplot(data=df, x="state_norm", y=metric, hue="celltype_norm", order=STATE_ORDER, hue_order=["RS-PC", "IB-PC", "IN", "FS"], dodge=True, palette=COLORS_CELLTYPE, alpha=0.32, size=2.3, linewidth=0.25, edgecolor="black", ax=ax)
            ax_bot.set_ylim(float(np.nanmin(df[metric])) * 0.98 if np.isfinite(np.nanmin(df[metric])) else 0.0, p90)
            ax_top.set_ylim(p90, p99 * 1.03)
            ax_top.spines["bottom"].set_visible(False)
            ax_bot.spines["top"].set_visible(False)
            ax_top.tick_params(labeltop=False)
            ax_bot.xaxis.tick_bottom()
            d = .006
            kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
            ax_top.plot((-d, +d), (-d, +d), **kwargs)
            ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
            kwargs.update(transform=ax_bot.transAxes)
            ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)
            ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
            if ax_top.legend_:
                ax_top.legend_.remove()
            handles, labels = ax_bot.get_legend_handles_labels()
            uniq = {}
            for h, lab in zip(handles, labels):
                if lab not in uniq:
                    uniq[lab] = h
            ax_bot.legend(uniq.values(), uniq.keys(), title="celltype", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
            fig.suptitle(f"{metric.replace('_', ' ').title()} (broken axis)", y=0.99)
            fig.tight_layout()
            save_figure(fig, os.path.join(plot_dir, f"{metric}__broken_axis.png"))
            plt.close(fig)

def plot_halfwidth_focus_panels(master: pd.DataFrame, out_dir: str) -> None:
    plot_dir = os.path.join(out_dir, "halfwidth_diagnostics")
    os.makedirs(plot_dir, exist_ok=True)
    cts = ["RS-PC", "IB-PC", "IN", "FS"]
    for metric in [m for m in ["halfwidth1_ms", "halfwidth2_ms"] if m in master.columns]:
        df = master[master["state_norm"].isin(STATE_ORDER) & master["celltype_norm"].isin(cts)].copy()
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
        df = df[np.isfinite(df[metric])]
        if df.empty:
            continue
        fig, ax = plt.subplots(figsize=(10.0, 5.8))
        sns.boxplot(data=df, x="state_norm", y=metric, hue="celltype_norm", order=STATE_ORDER, hue_order=cts, palette=COLORS_CELLTYPE, showfliers=False, ax=ax)
        sns.stripplot(data=df, x="state_norm", y=metric, hue="celltype_norm", order=STATE_ORDER, hue_order=cts, dodge=True, palette=COLORS_CELLTYPE, alpha=0.34, size=2.5, linewidth=0.25, edgecolor="black", ax=ax)
        overlay_mean_sem_markers(ax, df, "state_norm", metric, STATE_ORDER, hue="celltype_norm", hue_order=cts)
        ax.set_ylim(0.0, 1.5)
        handles, labels = ax.get_legend_handles_labels()
        uniq = {}
        for h, lab in zip(handles, labels):
            if lab not in uniq:
                uniq[lab] = h
        ax.legend(uniq.values(), uniq.keys(), title="celltype", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
        ax.set_title(f"{metric.replace('_', ' ').title()} focus window (0-1.5 ms)")
        ax.set_xlabel("")
        fig.tight_layout()
        save_figure(fig, os.path.join(plot_dir, f"publication_{metric}__focus_0_1p5ms.png"))
        plt.close(fig)

def plot_coupling_significance_panels(master: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    plot_dir = os.path.join(out_dir, "coupling_significance_plots")
    os.makedirs(plot_dir, exist_ok=True)
    rows = []

    bands = ["delta", "theta", "low_gamma", "high_gamma"]
    band_metrics = []
    for b in bands:
        m = f"n_coupling_sig_epochs_{b}"
        if m in master.columns:
            band_metrics.append((b, m))
    if not band_metrics:
        return pd.DataFrame()

    df0 = master.copy()
    if "cellclass_ei" not in df0.columns and "celltype_norm" in df0.columns:
        df0["cellclass_ei"] = df0["celltype_norm"].map(derive_ei_class)

    for band, metric in band_metrics:
        df = df0[df0["state_norm"].isin(STATE_ORDER)].copy()
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
        df = df[np.isfinite(df[metric])]
        if df.empty:
            continue

        # by state and layer
        layers = [l for l in ["supra", "gran", "infra"] if l in set(df["layer_norm"]) ]
        if layers:
            boxstrip_plot(
                df=df[df["layer_norm"].isin(layers)],
                metric=metric,
                x="state_norm",
                hue="layer_norm",
                out_png=os.path.join(plot_dir, f"{metric}__state_by_layer.png"),
                title=f"Significant coupling epochs ({band}) by state and layer",
                order=STATE_ORDER,
                hue_order=layers,
            )

        # by state and fine celltype
        cts = [c for c in ["RS-PC", "IB-PC", "IN", "FS"] if c in set(df["celltype_norm"]) ]
        if cts:
            boxstrip_plot(
                df=df[df["celltype_norm"].isin(cts)],
                metric=metric,
                x="state_norm",
                hue="celltype_norm",
                out_png=os.path.join(plot_dir, f"{metric}__state_by_celltype.png"),
                title=f"Significant coupling epochs ({band}) by state and celltype",
                order=STATE_ORDER,
                hue_order=cts,
            )

        # by state and EI mode
        eis = [c for c in ["Exc", "Inh"] if c in set(df["cellclass_ei"]) ]
        if eis:
            boxstrip_plot(
                df=df[df["cellclass_ei"].isin(eis)],
                metric=metric,
                x="state_norm",
                hue="cellclass_ei",
                out_png=os.path.join(plot_dir, f"{metric}__state_by_ei.png"),
                title=f"Significant coupling epochs ({band}) by state and EI class",
                order=STATE_ORDER,
                hue_order=eis,
            )

        rows.append(compare_groups_unmatched(df, metric, groups=("awake", "sleep"), stratify=["layer_norm"]))
        rows.append(compare_groups_unmatched(df, metric, groups=("awake", "sleep"), stratify=["celltype_norm"]))
        rows.append(compare_groups_unmatched(df, metric, groups=("awake", "sleep"), stratify=["cellclass_ei"]))
        rows.append(compare_states_omnibus(df, metric, stratify=["layer_norm"]))
        rows.append(compare_states_omnibus(df, metric, stratify=["celltype_norm"]))
        rows.append(compare_states_omnibus(df, metric, stratify=["cellclass_ei"]))

    out = concat_nonempty(rows)
    if not out.empty:
        out.to_csv(os.path.join(out_dir, "stats_coupling_significance_panels.csv"), index=False)
    return out

def plot_histo_ephys_correlation_matrices(stats_df: pd.DataFrame, out_dir: str) -> None:
    if stats_df is None or stats_df.empty:
        return
    plot_dir = os.path.join(out_dir, "histo_ephys_correlation_matrices")
    os.makedirs(plot_dir, exist_ok=True)

    hist_order = [h for h in ["neun_density_cells_mm2", "neun_coverage_percent", "pv_density_cells_mm2", "pv_coverage_percent", "pv_neun_density_ratio", "pv_neun_coverage_ratio"] if h in set(stats_df["histology_metric"])]
    ephys_order = [f"mean_{m}" for m in EPHYS_MATRIX_METRICS if f"mean_{m}" in set(stats_df["ephys_metric"])]
    if not hist_order or not ephys_order:
        return

    for cell_group in sorted(stats_df["ephys_cell_group"].dropna().unique()):
        for state in [s for s in STATE_ORDER if s in set(stats_df["state_norm"])]:
            sub = stats_df[(stats_df["ephys_cell_group"] == cell_group) & (stats_df["state_norm"] == state)].copy()
            if sub.empty:
                continue
            mat = sub.pivot_table(index="ephys_metric", columns="histology_metric", values="spearman_rho", aggfunc="mean")
            qmat = sub.pivot_table(index="ephys_metric", columns="histology_metric", values="spearman_q_fdr", aggfunc="mean")
            mat = mat.reindex(index=ephys_order, columns=hist_order)
            qmat = qmat.reindex(index=mat.index, columns=mat.columns)
            mat = mat.dropna(axis=0, how="all").dropna(axis=1, how="all")
            qmat = qmat.reindex(index=mat.index, columns=mat.columns)
            if mat.empty:
                continue
            fig, ax = plt.subplots(figsize=(1.15 * len(mat.columns) + 5.0, 0.42 * len(mat.index) + 2.6))
            sns.heatmap(mat, vmin=-1, vmax=1, cmap="vlag", center=0, annot=True, fmt=".2f", linewidths=0.5, linecolor="white", cbar_kws={"label": "Spearman rho"}, ax=ax)
            ax.set_title(f"Histology/ephys correlation matrix: {cell_group}, {state}")
            ax.set_xlabel("histology")
            ax.set_ylabel("ephys")
            fig.tight_layout()
            save_figure(fig, os.path.join(plot_dir, f"histo_ephys_corr_matrix__{cell_group}__{state}.png"))
            plt.close(fig)

def _plot_histo_ephys_scatter(merged: pd.DataFrame, row: pd.Series, plot_dir: str):
    cell_group = row["ephys_cell_group"]
    state = row["state_norm"]
    hmet = row["histology_metric"]
    emet = row["ephys_metric"]
    sub = merged[(merged["ephys_cell_group"] == cell_group) & (merged["state_norm"] == state)].copy()
    sub[hmet] = pd.to_numeric(sub[hmet], errors="coerce")
    sub[emet] = pd.to_numeric(sub[emet], errors="coerce")
    sub = sub[np.isfinite(sub[hmet]) & np.isfinite(sub[emet])]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    sns.regplot(data=sub, x=hmet, y=emet, ax=ax, scatter=False, color="#333333", ci=95)
    sns.scatterplot(data=sub, x=hmet, y=emet, hue="layer_norm", style="patient", ax=ax, s=70, edgecolor="black")
    q = row.get("spearman_q_fdr", np.nan)
    ax.set_title(f"{cell_group}, {state}: {emet}\nvs {hmet}; rho={row['spearman_rho']:.2f}, q={q:.2g} {stars(q)}")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    fig.tight_layout()
    fname = f"histo_corr__{cell_group}__{state}__{hmet}__{emet}".replace("/", "_").replace(" ", "_").replace("%", "pct")
    save_figure(fig, os.path.join(plot_dir, f"{fname}.png"))
    plt.close(fig)

def plot_histology_summary(hist_summary: pd.DataFrame, out_dir: str) -> None:
    if hist_summary is None or hist_summary.empty:
        return
    plot_dir = os.path.join(out_dir, "histology_plots")
    os.makedirs(plot_dir, exist_ok=True)
    for metric in ["mean_density_cells_mm2", "mean_coverage_percent"]:
        for stain in sorted(hist_summary["staining"].dropna().unique()):
            sub = hist_summary[hist_summary["staining"] == stain].copy()
            if sub.empty or metric not in sub.columns:
                continue
            sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
            sub = sub[np.isfinite(sub[metric])]
            sub = sub[sub["layer_norm"].notna() & sub["histology_state"].notna()]
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(9.5, 5.8))
            sub = sub[sub["layer_norm"].isin(["supra", "infra"])]
            valid_order = [x for x in ["supra", "infra"] if x in set(sub["layer_norm"])]
            hist_hue_order = [x for x in ["vivo", "vitro"] if x in set(sub["histology_state"])]
            # Stain-specific palette
            stain_color = COLOR_NEUN if stain == "NEUN" else COLOR_PV
            stain_palette = {state: stain_color for state in hist_hue_order}
            if sub.groupby(["layer_norm", "histology_state"])[metric].size().max() >= 2:
                sns.boxplot(data=sub, x="layer_norm", y=metric, hue="histology_state", ax=ax, showfliers=False, order=valid_order, hue_order=hist_hue_order, palette=COLORS_HISTOLOGY_STATE)
            sns.stripplot(data=sub, x="layer_norm", y=metric, hue="histology_state", ax=ax, dodge=True, color="black", alpha=0.45, order=valid_order, hue_order=hist_hue_order)
            handles, labels = ax.get_legend_handles_labels()
            n = len(sub["histology_state"].dropna().unique())
            ax.legend(handles[:n], labels[:n], title="", frameon=True, loc="upper right")
            ylabel = "density (cells/mm²)" if metric == "mean_density_cells_mm2" else "coverage (%)"
            ax.set_title(f"{stain} by layer", fontsize=11)
            ax.set_xlabel("layer")
            ax.set_ylabel(ylabel)
            annotate_pairwise_hue_brackets(ax, sub, "layer_norm", metric, valid_order, "histology_state", hist_hue_order, min_n=2)
            fig.tight_layout()
            save_figure(fig, os.path.join(plot_dir, f"histology__{stain}__{metric}.png"))
            plt.close(fig)
    plot_histology_publication_panels(hist_summary, out_dir)

def plot_histology_publication_panels(hist_summary: pd.DataFrame, out_dir: str) -> None:
    if hist_summary is None or hist_summary.empty:
        return
    plot_dir = os.path.join(out_dir, "histology_plots")
    os.makedirs(plot_dir, exist_ok=True)
    stains = [s for s in ["NEUN", "PV"] if s in set(hist_summary["staining"])]
    metrics = [("mean_density_cells_mm2", "Density (cells/mm²)"), ("mean_coverage_percent", "Coverage (%)")]
    if not stains:
        return
    fig, axs = plt.subplots(len(metrics), len(stains), figsize=(5.1 * len(stains), 4.2 * len(metrics)), squeeze=False)
    for r, (metric, ylabel) in enumerate(metrics):
        for c, stain in enumerate(stains):
            ax = axs[r, c]
            sub = hist_summary[hist_summary["staining"] == stain].copy()
            if sub.empty or metric not in sub.columns:
                ax.axis("off")
                continue
            sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
            sub = sub[np.isfinite(sub[metric])]
            sub = sub[sub["layer_norm"].isin(["supra", "infra"])]
            valid_order = [x for x in ["supra", "infra"] if x in set(sub["layer_norm"])]
            hue_order = [x for x in ["vivo", "vitro"] if x in set(sub["histology_state"])]
            sns.boxplot(data=sub, x="layer_norm", y=metric, hue="histology_state", order=valid_order, hue_order=hue_order, palette=COLORS_HISTOLOGY_STATE, showfliers=False, ax=ax)
            sns.stripplot(data=sub, x="layer_norm", y=metric, hue="histology_state", order=valid_order, hue_order=hue_order, dodge=True, color="black", alpha=0.45, size=4, ax=ax)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[:len(hue_order)], labels[:len(hue_order)], title="", frameon=True, loc="upper right")
            ax.set_title(f"{stain}", fontsize=11)
            ax.set_xlabel("layer")
            ax.set_ylabel(ylabel)
            annotate_pairwise_hue_brackets(ax, sub, "layer_norm", metric, valid_order, "histology_state", hue_order, min_n=2)
    fig.suptitle("Histology summary: density and coverage by layer", y=1.01, fontsize=12)
    fig.tight_layout()
    save_figure(fig, os.path.join(plot_dir, "histology_publication_panel_density_coverage.png"))
    plt.close(fig)

def plot_histology_half_violins(histology: pd.DataFrame, out_dir: str) -> None:
    if histology is None or histology.empty:
        return
    plot_dir = os.path.join(out_dir, "histology_plots")
    os.makedirs(plot_dir, exist_ok=True)

    h = histology.copy()
    h["histology_state"] = h.get("histology_state", pd.Series(index=h.index, dtype=object)).astype(str).str.strip().str.lower()
    h["layer_norm"] = h.get("layer_norm", pd.Series(index=h.index, dtype=object)).astype(str).str.strip().str.lower()
    h["staining"] = h.get("staining", pd.Series(index=h.index, dtype=object)).astype(str).str.strip().str.upper()
    h = h[h["layer_norm"].isin(["supra", "infra"])]
    h = h[h["histology_state"].isin(["vivo", "vitro"])]
    if h.empty:
        return

    def _save(fig, name):
        fig.tight_layout()
        save_figure(fig, os.path.join(plot_dir, name))
        plt.close(fig)

    for metric, ylabel in [("density (cell/mm^2)", "Density (cells/mm²)"), ("coverage (%)", "Coverage (%)")]:
        if metric not in h.columns:
            continue
        sub = h.copy()
        sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
        sub = sub[np.isfinite(sub[metric]) & sub["staining"].isin(["PV", "NEUN"])]
        if sub.empty:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
        for ax, layer in zip(axes, ["supra", "infra"]):
            layer_df = sub[sub["layer_norm"] == layer].copy()
            if layer_df.empty:
                ax.axis("off")
                continue
            hue_order = [v for v in ["vivo", "vitro"] if v in set(layer_df["histology_state"])]
            sns.violinplot(data=layer_df, x="staining", y=metric, hue="histology_state", order=["PV", "NEUN"], hue_order=hue_order, split=True, inner="quart", cut=0, palette=COLORS_HISTOLOGY_STATE, ax=ax)
            sns.stripplot(data=layer_df, x="staining", y=metric, hue="histology_state", order=["PV", "NEUN"], hue_order=hue_order, dodge=True, palette=COLORS_HISTOLOGY_STATE, alpha=0.35, size=3.5, linewidth=0.25, edgecolor="black", ax=ax)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[:len(hue_order)], labels[:len(hue_order)], title="", frameon=True, loc="upper right")
            ax.set_title(layer)
            ax.set_xlabel("stain")
            ax.set_ylabel(ylabel)
            annotate_pairwise_hue_brackets(ax, layer_df, "staining", metric, ["PV", "NEUN"], "histology_state", hue_order, min_n=2)
        fig.suptitle(f"Histology {ylabel.lower()} by stain and layer", y=1.02)
        _save(fig, f"histology_half_violin__{metric.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')}.png")

    wide = h.pivot_table(index=["patient", "histology_state", "layer_norm"], columns="staining", values=["density (cell/mm^2)", "coverage (%)"], aggfunc="mean")
    if wide.empty:
        return
    wide.columns = [f"{a}__{b}" for a, b in wide.columns.to_flat_index()]
    wide = wide.reset_index()
    if "density (cell/mm^2)__PV" in wide.columns and "density (cell/mm^2)__NEUN" in wide.columns:
        wide["pv_neun_density_ratio"] = wide["density (cell/mm^2)__PV"] / wide["density (cell/mm^2)__NEUN"]
    if "coverage (%)__PV" in wide.columns and "coverage (%)__NEUN" in wide.columns:
        wide["pv_neun_coverage_ratio"] = wide["coverage (%)__PV"] / wide["coverage (%)__NEUN"]

    for metric, ylabel in [("pv_neun_density_ratio", "PV / NeuN density ratio"), ("pv_neun_coverage_ratio", "PV / NeuN coverage ratio")]:
        if metric not in wide.columns:
            continue
        sub = wide.copy()
        sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
        sub = sub[np.isfinite(sub[metric])]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(7.6, 4.9))
        hue_order = [v for v in ["vivo", "vitro"] if v in set(sub["histology_state"])]
        sns.violinplot(data=sub, x="layer_norm", y=metric, hue="histology_state", order=["supra", "infra"], hue_order=hue_order, split=True, inner="quart", cut=0, palette=COLORS_HISTOLOGY_STATE, ax=ax)
        sns.stripplot(data=sub, x="layer_norm", y=metric, hue="histology_state", order=["supra", "infra"], hue_order=hue_order, dodge=True, palette=COLORS_HISTOLOGY_STATE, alpha=0.35, size=3.5, linewidth=0.25, edgecolor="black", ax=ax)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:len(hue_order)], labels[:len(hue_order)], title="", frameon=True, loc="upper right")
        ax.set_title(ylabel)
        ax.set_xlabel("layer")
        ax.set_ylabel(ylabel)
        annotate_pairwise_hue_brackets(ax, sub, "layer_norm", metric, ["supra", "infra"], "histology_state", hue_order, min_n=2)
        _save(fig, f"histology_half_violin__{metric}.png")

def plot_celltype_metric_panels(master: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    plot_dir = os.path.join(out_dir, "celltype_plots")
    os.makedirs(plot_dir, exist_ok=True)
    metrics = [m for m in [
        "firing_rate_Hz", "max_firing_rate_1s_Hz", "burstiness_percent", "bursti_2", "isi_cv", "ref_viol",
        "halfwidth1_ms", "halfwidth2_ms", "peak_to_trough_delay_ms",
        "population_coupling", "coupling_strength_low_gamma", "coupling_strength_high_gamma",
        "mean_plv_low_gamma", "mean_plv_high_gamma", "plv_low_gamma_strength", "plv_high_gamma_strength",
        # Added delta/theta
        "mean_plv_delta", "max_plv_delta", "mean_plv_sig_delta", "max_plv_sig_delta", "resultant_length_sig_delta",
        "mean_plv_theta", "max_plv_theta", "mean_plv_sig_theta", "max_plv_sig_theta", "resultant_length_sig_theta",
    ] if m in master.columns]
    rows = []
    for metric in metrics:
        df = master[master["state_norm"].isin(STATE_ORDER)].copy()
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
        df = df[np.isfinite(df[metric]) & df["celltype_norm"].isin(CELLTYPE_ORDER)]
        if df.empty:
            continue

        rows.append(pairwise_mannwhitney_table(df, metric, "celltype_norm", [c for c in CELLTYPE_ORDER if c in set(df["celltype_norm"])]))
        for state in STATE_ORDER:
            sub = df[df["state_norm"] == state]
            if not sub.empty:
                t = pairwise_mannwhitney_table(sub, metric, "celltype_norm", [c for c in CELLTYPE_ORDER if c in set(sub["celltype_norm"])])
                if not t.empty:
                    t["state_norm"] = state
                    rows.append(t)

        fig, ax = plt.subplots(figsize=(10.8, 6.4))
        hue_order = [c for c in ["RS-PC", "IB-PC", "IN", "FS"] if c in set(df["celltype_norm"])]
        sns.boxplot(
            data=df, x="state_norm", y=metric, order=STATE_ORDER,
            color="#E5E7EB", showfliers=False, width=0.62, ax=ax,
        )
        # custom point style per celltype
        x_map = {v: i for i, v in enumerate(STATE_ORDER)}
        rng = np.random.default_rng(7)
        for ct in hue_order:
            sub = df[df["celltype_norm"] == ct].copy()
            xs = sub["state_norm"].map(x_map).astype(float).values
            ys = pd.to_numeric(sub[metric], errors="coerce").values
            m = np.isfinite(xs) & np.isfinite(ys)
            xs = xs[m] + rng.uniform(-0.13, 0.13, size=np.sum(m))
            ys = ys[m]
            if len(ys) == 0:
                continue
            marker = CELLTYPE_MARKERS.get(ct, "o")
            face = COLORS_CELLTYPE.get(ct, "#777777") if CELLTYPE_FILLED.get(ct, True) else "none"
            edge = COLORS_CELLTYPE.get(ct, "#777777")
            ax.scatter(xs, ys, marker=marker, s=28, facecolors=face, edgecolors=edge, linewidths=0.95, alpha=0.72, label=ct, zorder=4)
        handles, labels = ax.get_legend_handles_labels()
        uniq = {}
        for h, lab in zip(handles, labels):
            if lab not in uniq:
                uniq[lab] = h
        ax.legend(uniq.values(), uniq.keys(), title="celltype", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
        ax.set_title(f"{metric.replace('_', ' ').title()} by celltype and state")
        ax.set_xlabel("")
        ax.set_ylabel(metric.replace("_", " ").title())
        overlay_mean_sem_markers(ax, df, "state_norm", metric, STATE_ORDER, hue="celltype_norm", hue_order=[c for c in CELLTYPE_ORDER if c in set(df["celltype_norm"])])
        annotate_pairwise_hue_brackets(ax, df, "state_norm", metric, STATE_ORDER, "celltype_norm", [c for c in CELLTYPE_ORDER if c in set(df["celltype_norm"])], min_n=3)
        annotate_same_hue_across_x_brackets(ax, df, "state_norm", metric, STATE_ORDER, "celltype_norm", [c for c in CELLTYPE_ORDER if c in set(df["celltype_norm"])], min_n=3)
        fig.tight_layout()
        save_figure(fig, os.path.join(plot_dir, f"celltype_state__{metric}.png"))
        plt.close(fig)

    stats_df = concat_nonempty(rows)
    if not stats_df.empty:
        stats_df.to_csv(os.path.join(out_dir, "stats_celltype_pairwise_metrics.csv"), index=False)
    return stats_df

def plot_ei_metric_panels(master: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    plot_dir = os.path.join(out_dir, "cellclass_ei_plots")
    os.makedirs(plot_dir, exist_ok=True)
    metrics = [m for m in [
        "firing_rate_Hz", "max_firing_rate_1s_Hz", "burstiness_percent", "bursti_2",
        "halfwidth1_ms", "halfwidth2_ms", "peak_to_trough_delay_ms",
        "population_coupling", "mean_plv_delta", "mean_plv_theta", "mean_plv_low_gamma", "mean_plv_high_gamma",
    ] if m in master.columns]
    rows = []

    df0 = master.copy()
    if "cellclass_ei" not in df0.columns and "celltype_norm" in df0.columns:
        df0["cellclass_ei"] = df0["celltype_norm"].map(derive_ei_class)

    for metric in metrics:
        df = df0[df0["state_norm"].isin(STATE_ORDER) & df0["cellclass_ei"].isin(EI_ORDER)].copy()
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
        df = df[np.isfinite(df[metric])]
        if df.empty:
            continue

        rows.append(pairwise_mannwhitney_table(df, metric, "cellclass_ei", [c for c in EI_ORDER if c in set(df["cellclass_ei"])]))
        for state in STATE_ORDER:
            sub = df[df["state_norm"] == state]
            if sub.empty:
                continue
            t = pairwise_mannwhitney_table(sub, metric, "cellclass_ei", [c for c in EI_ORDER if c in set(sub["cellclass_ei"])])
            if not t.empty:
                t["state_norm"] = state
                rows.append(t)

        fig, ax = plt.subplots(figsize=(9.8, 5.8))
        hue_order = [c for c in EI_ORDER if c in set(df["cellclass_ei"])]
        sns.boxplot(data=df, x="state_norm", y=metric, hue="cellclass_ei", order=STATE_ORDER, hue_order=hue_order, palette=COLORS_EI, showfliers=False, ax=ax)
        sns.stripplot(data=df, x="state_norm", y=metric, hue="cellclass_ei", order=STATE_ORDER, hue_order=hue_order, dodge=True, palette=COLORS_EI, alpha=0.35, size=2.6, linewidth=0.25, edgecolor="black", ax=ax)
        overlay_mean_sem_markers(ax, df, "state_norm", metric, STATE_ORDER, hue="cellclass_ei", hue_order=hue_order)
        annotate_pairwise_hue_brackets(ax, df, "state_norm", metric, STATE_ORDER, "cellclass_ei", hue_order, min_n=3)
        annotate_same_hue_across_x_brackets(ax, df, "state_norm", metric, STATE_ORDER, "cellclass_ei", hue_order, min_n=3)
        handles, labels = ax.get_legend_handles_labels()
        uniq = {}
        for h, lab in zip(handles, labels):
            if lab not in uniq:
                uniq[lab] = h
        ax.legend(uniq.values(), uniq.keys(), title="class", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
        ax.set_title(f"{metric.replace('_', ' ').title()} by Exc/Inh and state")
        ax.set_xlabel("")
        fig.tight_layout()
        save_figure(fig, os.path.join(plot_dir, f"ei_state__{metric}.png"))
        plt.close(fig)

    out = concat_nonempty(rows)
    if not out.empty:
        out.to_csv(os.path.join(out_dir, "stats_cellclass_ei_pairwise_metrics.csv"), index=False)
    return out

def plot_halfwidth_diagnostics(master: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    plot_dir = os.path.join(out_dir, "halfwidth_diagnostics")
    os.makedirs(plot_dir, exist_ok=True)
    metrics = [m for m in ["halfwidth1_ms", "halfwidth2_ms", "peak_to_trough_delay_ms", "amplitude_uV", "firing_rate_Hz"] if m in master.columns]
    rows = []
    for metric in metrics:
        df = master[master["state_norm"].isin(STATE_ORDER)].copy()
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
        df = df[np.isfinite(df[metric])]
        if df.empty:
            continue
        rows.append(pairwise_mannwhitney_table(df, metric, "state_norm", STATE_ORDER))

        fig, ax = plt.subplots(figsize=(8.6, 5.8))
        sns.boxplot(data=df, x="state_norm", y=metric, order=STATE_ORDER, palette=COLORS_STATE, showfliers=False, ax=ax)
        sns.stripplot(data=df, x="state_norm", y=metric, order=STATE_ORDER, palette=COLORS_STATE, alpha=0.38, size=3, linewidth=0.25, edgecolor="black", ax=ax)
        overlay_mean_sem_markers(ax, df, "state_norm", metric, STATE_ORDER)
        _add_pairwise_sig_from_table(ax, pairwise_mannwhitney_table(df, metric, "state_norm", STATE_ORDER), STATE_ORDER, df[metric].values)
        ax.set_title(f"{metric.replace('_', ' ').title()}: awake -> sleep -> vitro")
        ax.set_xlabel("")
        fig.tight_layout()
        save_figure(fig, os.path.join(plot_dir, f"halfwidth_check__{metric}.png"))
        plt.close(fig)

    out = concat_nonempty(rows)
    if not out.empty:
        out.to_csv(os.path.join(out_dir, "stats_halfwidth_state_pairwise.csv"), index=False)
    return out

def plot_halfwidth_celltype_state_publication(master: pd.DataFrame, out_dir: str) -> None:
    plot_dir = os.path.join(out_dir, "halfwidth_diagnostics")
    os.makedirs(plot_dir, exist_ok=True)
    cts = ["RS-PC", "IB-PC", "IN", "FS"]
    for metric in ["halfwidth1_ms", "halfwidth2_ms"]:
        if metric not in master.columns:
            continue
        df = master[master["state_norm"].isin(STATE_ORDER) & master["celltype_norm"].isin(cts)].copy()
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
        df = df[np.isfinite(df[metric])]
        if df.empty:
            continue
        fig, ax = plt.subplots(figsize=(9.8, 5.8))
        sns.boxplot(data=df, x="state_norm", y=metric, hue="celltype_norm", order=STATE_ORDER, hue_order=cts, palette=COLORS_CELLTYPE, showfliers=False, ax=ax)
        sns.stripplot(data=df, x="state_norm", y=metric, hue="celltype_norm", order=STATE_ORDER, hue_order=cts, dodge=True, palette=COLORS_CELLTYPE, alpha=0.35, size=2.6, linewidth=0.25, edgecolor="black", ax=ax)
        overlay_mean_sem_markers(ax, df, "state_norm", metric, STATE_ORDER, hue="celltype_norm", hue_order=cts)
        clip_axis_to_quantiles(ax, df[metric], lo=0.02, hi=0.97, pad=0.08)
        annotate_group_ns(ax, df, "state_norm", metric, STATE_ORDER, hue="celltype_norm", hue_order=cts)
        annotate_pairwise_hue_brackets(ax, df, "state_norm", metric, STATE_ORDER, "celltype_norm", cts, min_n=3)
        handles, labels = ax.get_legend_handles_labels()
        uniq = {}
        for h, lab in zip(handles, labels):
            if lab not in uniq:
                uniq[lab] = h
        ax.legend(uniq.values(), uniq.keys(), title="celltype", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
        ax.set_title(f"{metric.replace('_', ' ').title()}: awake, sleep, in vitro by celltype")
        ax.set_xlabel("")
        fig.tight_layout()
        save_figure(fig, os.path.join(plot_dir, f"publication_halfwidth_by_state_celltype__{metric}.png"))
        plt.close(fig)

def build_and_plot_celltype_composition(master: pd.DataFrame, out_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    plot_dir = os.path.join(out_dir, "celltype_composition")
    os.makedirs(plot_dir, exist_ok=True)
    df = master[master["state_norm"].isin(STATE_ORDER) & master["celltype_norm"].isin(CELLTYPE_ORDER)].copy()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    by_state = df.groupby(["state_norm", "celltype_norm"], dropna=False).size().reset_index(name="n_units")
    by_state["total_in_state"] = by_state.groupby("state_norm")["n_units"].transform("sum")
    by_state["proportion"] = by_state["n_units"] / by_state["total_in_state"]
    by_layer = df.groupby(["state_norm", "layer_norm", "celltype_norm"], dropna=False).size().reset_index(name="n_units")
    by_layer["total_in_state_layer"] = by_layer.groupby(["state_norm", "layer_norm"])["n_units"].transform("sum")
    by_layer["proportion"] = by_layer["n_units"] / by_layer["total_in_state_layer"]
    by_state.to_csv(os.path.join(out_dir, "table_celltype_proportions_by_state.csv"), index=False)
    by_layer.to_csv(os.path.join(out_dir, "table_celltype_proportions_by_state_layer.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    piv = by_state.pivot(index="state_norm", columns="celltype_norm", values="proportion").reindex(STATE_ORDER).fillna(0)
    piv = piv[[c for c in CELLTYPE_ORDER if c in piv.columns]]
    bottom = np.zeros(len(piv))
    for ct in piv.columns:
        vals = piv[ct].values
        ax.bar(piv.index, vals, bottom=bottom, color=COLORS_CELLTYPE.get(ct, "#999999"), label=ct)
        bottom += vals
    ax.set_ylim(0, 1)
    ax.set_ylabel("proportion of recorded units")
    ax.set_xlabel("")
    ax.set_title("Celltype composition by state")
    totals = by_state.groupby("state_norm")["n_units"].sum().to_dict()
    for i, state in enumerate(piv.index.tolist()):
        ax.text(i, -0.06, f"n={int(totals.get(state, 0))}", ha="center", va="top", fontsize=9, color="#333333", clip_on=False)
    ax.legend(title="celltype", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    fig.tight_layout()
    save_figure(fig, os.path.join(plot_dir, "celltype_proportion_by_state.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    layer_plot = by_layer[by_layer["layer_norm"] != "other"].copy()
    sns.barplot(data=layer_plot, x="layer_norm", y="proportion", hue="celltype_norm", order=["supra", "gran", "infra"], hue_order=[c for c in CELLTYPE_ORDER if c in set(layer_plot["celltype_norm"])], palette=COLORS_CELLTYPE, ax=ax)
    ax.set_title("Celltype composition by layer")
    ax.set_xlabel("")
    ax.set_ylabel("proportion within state-layer")
    ax.legend(title="celltype", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    fig.tight_layout()
    save_figure(fig, os.path.join(plot_dir, "celltype_proportion_by_layer.png"))
    plt.close(fig)
    return by_state, by_layer

def _add_pairwise_sig_from_table(ax, stat_df: pd.DataFrame, order: List[str], values):
    if stat_df is None or stat_df.empty:
        return
    vals = pd.to_numeric(pd.Series(values), errors="coerce")
    vals = vals[np.isfinite(vals)]
    if vals.empty:
        return
    ymin, ymax = float(vals.min()), float(vals.max())
    span = ymax - ymin if ymax > ymin else max(abs(ymax), 1.0)
    y = ymax + 0.08 * span
    step = 0.10 * span
    for k, (a, b) in enumerate(combinations(order, 2)):
        r = stat_df[((stat_df["group_a"] == a) & (stat_df["group_b"] == b)) | ((stat_df["group_a"] == b) & (stat_df["group_b"] == a))]
        if r.empty:
            continue
        add_sig_bracket(ax, order.index(a), order.index(b), y + k * step, _sig_text_from_row(r.iloc[0]), dy=0.025 * span)

def clip_axis_to_quantiles(ax, values, lo=0.01, hi=0.99, pad=0.08):
    vals = pd.to_numeric(pd.Series(values), errors="coerce")
    vals = vals[np.isfinite(vals)]
    if len(vals) < 5:
        return
    qlo, qhi = vals.quantile(lo), vals.quantile(hi)
    if not np.isfinite(qlo) or not np.isfinite(qhi) or qhi <= qlo:
        return
    span = qhi - qlo
    ax.set_ylim(qlo - pad * span, qhi + pad * span)

def clip_axis_with_sig_headroom(ax, values, n_brackets: int = 0, lo=0.01, hi=0.99):
    """
    Keep central data visible for zero-heavy distributions while reserving compact space for significance brackets.
    """
    vals = pd.to_numeric(pd.Series(values), errors="coerce")
    vals = vals[np.isfinite(vals)]
    if len(vals) < 5:
        return
    qlo, qhi = vals.quantile(lo), vals.quantile(hi)
    if not np.isfinite(qlo) or not np.isfinite(qhi) or qhi <= qlo:
        return
    span = qhi - qlo
    head = (0.10 + 0.05 * max(0, n_brackets)) * span
    foot = 0.06 * span
    ax.set_ylim(qlo - foot, qhi + head)

def overlay_mean_sem_markers(ax, df: pd.DataFrame, x: str, y: str, order: List[str], hue: Optional[str] = None, hue_order: Optional[List[str]] = None):
    # SEM/SE whiskers and mean marker overlays disabled per publication style request.
    return

def annotate_group_ns(ax, df: pd.DataFrame, x: str, y: str, order: List[str], hue: Optional[str] = None, hue_order: Optional[List[str]] = None):
    ymax = ax.get_ylim()[1]
    ymin = ax.get_ylim()[0]
    ytxt = ymin + 0.02 * (ymax - ymin)
    if hue is None:
        for i, g in enumerate(order or []):
            n = int(df[df[x] == g][y].notna().sum())
            if n:
                ax.text(i, ytxt, f"n={n}", ha="center", va="bottom", fontsize=8, color="#333333")
    else:
        hs = hue_order or sorted(df[hue].dropna().unique())
        width = 0.72
        for i, g in enumerate(order or []):
            for j, h in enumerate(hs):
                n = int(df[(df[x] == g) & (df[hue] == h)][y].notna().sum())
                if n:
                    xpos = i - width / 2 + width * (j + 0.5) / max(len(hs), 1)
                    ax.text(xpos, ytxt, f"{n}", ha="center", va="bottom", fontsize=7, color="#333333")

def build_connection_density_table(conn: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    if conn is None or conn.empty:
        return pd.DataFrame()
    c_all = conn.copy()
    c = filter_true_connections(c_all)
    if c.empty:
        return pd.DataFrame()

    rows = []
    group_cols = [col for col in ["recording_id", "patient", "state_norm"] if col in c_all.columns]
    tested = c_all.groupby(group_cols, dropna=False).size().reset_index(name="n_pairs_tested") if group_cols else pd.DataFrame({"n_pairs_tested": [len(c_all)]})
    true_counts = c.groupby(group_cols, dropna=False).size().reset_index(name="n_true_connections") if group_cols else pd.DataFrame({"n_true_connections": [len(c)]})
    merged = tested.merge(true_counts, on=group_cols, how="left") if group_cols else tested.assign(n_true_connections=len(c))
    merged["n_true_connections"] = merged["n_true_connections"].fillna(0).astype(int)
    merged["true_connection_fraction_tested"] = merged["n_true_connections"] / merged["n_pairs_tested"]
    for keys, sub in c.groupby(group_cols, dropna=False):
        key_vals = keys if isinstance(keys, tuple) else (keys,)
        key = dict(zip(group_cols, key_vals))
        row = merged
        for col, val in key.items():
            row = row[row[col] == val]
        row = row.iloc[0].to_dict() if len(row) else {**key}
        if "inferred_type" in sub.columns:
            counts = sub["inferred_type"].astype(str).str.lower().value_counts()
            for label, count in counts.items():
                row[f"n_{label}"] = int(count)
        rows.append(row)
    return pd.DataFrame(rows)

def summarize_connection_latency(conn: pd.DataFrame, group_cols: Optional[List[str]] = None) -> pd.DataFrame:
    if conn is None or conn.empty or "peak_latency_ms" not in conn.columns:
        return pd.DataFrame()
    group_cols = [c for c in (group_cols or ["state_norm", "inferred_type", "same_layer"]) if c in conn.columns]
    c = filter_true_connections(conn)
    if c.empty:
        return pd.DataFrame()
    c["peak_latency_ms"] = pd.to_numeric(c["peak_latency_ms"], errors="coerce")
    rows = []
    for keys, sub in c.groupby(group_cols, dropna=False) if group_cols else [((), c)]:
        key_vals = keys if isinstance(keys, tuple) else (keys,)
        vals = sub["peak_latency_ms"].dropna().astype(float)
        if vals.empty:
            continue
        rows.append({
            **dict(zip(group_cols, key_vals)),
            "n": int(vals.size),
            "median_peak_latency_ms": float(vals.median()),
            "mean_peak_latency_ms": float(vals.mean()),
            "iqr_peak_latency_ms": float(vals.quantile(0.75) - vals.quantile(0.25)),
        })
    return pd.DataFrame(rows)

def unit_connection_degree(conn: pd.DataFrame) -> pd.DataFrame:
    if conn is None or conn.empty:
        return pd.DataFrame()
    c = filter_true_connections(conn)
    if c.empty:
        return pd.DataFrame()
    rows = []
    for side in ["i", "j"]:
        unit_col = f"unit_{side}"
        if unit_col not in c.columns:
            continue
        tmp = c[["recording_id", unit_col]].rename(columns={unit_col: "unit_id"}).copy()
        rows.append(tmp)
    if not rows:
        return pd.DataFrame()
    deg = pd.concat(rows, ignore_index=True)
    return deg.groupby(["recording_id", "unit_id"]).size().reset_index(name="putative_connection_degree")

def graph_connection_agreement(master: pd.DataFrame, conn: pd.DataFrame) -> pd.DataFrame:
    deg = unit_connection_degree(conn)
    if deg.empty:
        return pd.DataFrame()
    m = master.merge(deg, on=["recording_id", "unit_id"], how="left")
    m["putative_connection_degree"] = m["putative_connection_degree"].fillna(0)
    metrics = [c for c in DEFAULT_METRICS_GRAPH if c in m.columns]
    rows = []
    for metric in metrics:
        sub = m[["putative_connection_degree", metric]].copy()
        sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
        sub = sub[np.isfinite(sub[metric])]
        if len(sub) < 5:
            continue
        rho, p = stats.spearmanr(sub["putative_connection_degree"], sub[metric], nan_policy="omit")
        rows.append({
            "metric": metric,
            "n_units": int(len(sub)),
            "spearman_rho_with_putative_connection_degree": float(rho),
            "p": float(p),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_fdr"] = fdr_bh(out["p"].values)
    return out

def plot_putative_connection_summary(conn: pd.DataFrame, out_dir: str) -> None:
    if conn is None or conn.empty:
        return
    plot_dir = os.path.join(out_dir, "putative_connection_plots")
    os.makedirs(plot_dir, exist_ok=True)
    c = conn.copy()
    if "is_putative_connection" not in c.columns:
        c["is_putative_connection"] = c.get("inferred_type", "none").astype(str).str.strip().str.lower() == "putative_monosynaptic_exc"
    if "is_true_connection" in c.columns:
        c = c[c["is_true_connection"]].copy()
    else:
        c = c[c["is_putative_connection"]].copy()
    if c.empty:
        return
    c["connection_label"] = c.get("inferred_type", "true").astype(str)

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    rate = conn.copy()
    if "is_true_connection" in rate.columns:
        rate["connection_for_rate"] = rate["is_true_connection"]
    else:
        rate["connection_for_rate"] = rate.get("is_putative_connection", False)
    rate = rate.groupby("state_norm")["connection_for_rate"].mean().reindex(STATE_ORDER).dropna().reset_index()
    if not rate.empty:
        sns.barplot(data=rate, x="state_norm", y="connection_for_rate", order=[s for s in STATE_ORDER if s in set(rate["state_norm"])], palette=COLORS_STATE, ax=ax)
        ax.set_ylabel("fraction of tested pairs")
        ax.set_xlabel("")
        ax.set_title("True connection rate by state")
        raw_rate = conn.copy()
        raw_rate["connection_for_rate"] = pd.to_numeric(raw_rate.get("is_true_connection", raw_rate.get("is_putative_connection", False)), errors="coerce")
        raw_rate = raw_rate[np.isfinite(raw_rate["connection_for_rate"])]
        raw_state_order = [s for s in STATE_ORDER if s in set(raw_rate["state_norm"])]
        if len(raw_state_order) >= 2:
            ax.set_ylim(0, max(1.08, float(rate["connection_for_rate"].max()) + 0.18))
            _add_pairwise_sig_from_table(ax, pairwise_mannwhitney_table(raw_rate, "connection_for_rate", "state_norm", raw_state_order), raw_state_order, raw_rate["connection_for_rate"].values)
        save_figure(fig, os.path.join(plot_dir, "connection_rate_by_state.png"))
    plt.close(fig)

    put = c.copy()
    if put.empty:
        return
    for col, fname, title in [
        ("layer_pair", "connections_by_layer_pair.png", "True connections by layer pair"),
        ("celltype_pair", "connections_by_celltype_pair.png", "True connections by celltype pair"),
        ("inferred_type", "connections_by_type.png", "True connection type"),
    ]:
        if col not in put.columns:
            continue
        top = put[col].astype(str).value_counts().head(16).reset_index()
        top.columns = [col, "n"]
        fig, ax = plt.subplots(figsize=(9.5, max(4.8, 0.35 * len(top) + 2)))
        sns.barplot(data=top, y=col, x="n", color="#4E79A7", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("n true connections")
        ax.set_ylabel("")
        for p in ax.patches:
            ax.text(p.get_width() + 0.2, p.get_y() + p.get_height() / 2, f"{int(p.get_width())}", va="center", fontsize=9)
        fig.tight_layout()
        save_figure(fig, os.path.join(plot_dir, fname))
        plt.close(fig)

    if {"peak_latency_ms", "state_norm"}.issubset(put.columns):
        put["peak_latency_ms"] = pd.to_numeric(put["peak_latency_ms"], errors="coerce")
        put = put[np.isfinite(put["peak_latency_ms"])]
        if not put.empty:
            fig, ax = plt.subplots(figsize=(8.6, 5.5))
            sns.boxplot(data=put, x="state_norm", y="peak_latency_ms", order=STATE_ORDER, palette=COLORS_STATE, showfliers=False, ax=ax)
            sns.stripplot(data=put, x="state_norm", y="peak_latency_ms", order=STATE_ORDER, palette=COLORS_STATE, alpha=0.35, size=3, linewidth=0.25, edgecolor="black", ax=ax)
            clip_axis_to_quantiles(ax, put["peak_latency_ms"])
            annotate_group_ns(ax, put, "state_norm", "peak_latency_ms", STATE_ORDER)
            stat_df = pairwise_mannwhitney_table(put, "peak_latency_ms", "state_norm", STATE_ORDER)
            _add_pairwise_sig_from_table(ax, stat_df, STATE_ORDER, put["peak_latency_ms"].values)
            ax.set_title("True connection latency by state")
            ax.set_xlabel("")
            fig.tight_layout()
            save_figure(fig, os.path.join(plot_dir, "connection_latency_by_state.png"))
            plt.close(fig)

def plot_graph_metric_cellclass_differences(master: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    plot_dir = os.path.join(out_dir, "graph_cellclass_plots")
    os.makedirs(plot_dir, exist_ok=True)
    metrics = [m for m in DEFAULT_METRICS_GRAPH if m in master.columns]
    rows = []
    cts = ["RS-PC", "IB-PC", "IN", "FS"]
    for metric in metrics:
        df = master[master["state_norm"].isin(STATE_ORDER) & master["celltype_norm"].isin(cts)].copy()
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
        df = df[np.isfinite(df[metric])]
        if df.empty:
            continue
        for state in STATE_ORDER:
            sub = df[df["state_norm"] == state]
            t = pairwise_mannwhitney_table(sub, metric, "celltype_norm", [c for c in cts if c in set(sub["celltype_norm"])], min_n=3)
            if not t.empty:
                t["state_norm"] = state
                rows.append(t)
        if any(k in metric for k in ["strength", "pagerank", "clustering"]):
            fig, ax = plt.subplots(figsize=(9.8, 5.8))
            sns.boxplot(data=df, x="state_norm", y=metric, hue="celltype_norm", order=STATE_ORDER, hue_order=cts, palette=COLORS_CELLTYPE, showfliers=False, ax=ax)
            sns.stripplot(data=df, x="state_norm", y=metric, hue="celltype_norm", order=STATE_ORDER, hue_order=cts, dodge=True, palette=COLORS_CELLTYPE, alpha=0.30, size=2.4, linewidth=0.25, edgecolor="black", ax=ax)
            clip_axis_to_quantiles(ax, df[metric], lo=0.02, hi=0.98)
            annotate_group_ns(ax, df, "state_norm", metric, STATE_ORDER, hue="celltype_norm", hue_order=cts)
            annotate_pairwise_hue_brackets(ax, df, "state_norm", metric, STATE_ORDER, "celltype_norm", cts, min_n=3)
            handles, labels = ax.get_legend_handles_labels()
            uniq = {}
            for h, lab in zip(handles, labels):
                if lab not in uniq:
                    uniq[lab] = h
            ax.legend(uniq.values(), uniq.keys(), title="celltype", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
            ax.set_title(f"{metric.replace('_', ' ').title()}: graph position by cell class")
            ax.set_xlabel("")
            fig.tight_layout()
            save_figure(fig, os.path.join(plot_dir, f"graph_cellclass__{metric}.png"))
            plt.close(fig)
    out = concat_nonempty(rows)
    if not out.empty:
        out.to_csv(os.path.join(out_dir, "stats_graph_metric_cellclass_pairwise.csv"), index=False)
    return out

def melt_metrics_long(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    base_cols = [c for c in ["recording_id", "cell_uid", "unit_id", "state_norm", "layer_norm", "celltype_norm", "patient", "official_unit"] if c in df.columns]
    melted = df.melt(id_vars=base_cols, value_vars=[m for m in metrics if m in df.columns],
                     var_name="metric", value_name="value")
    melted["value"] = pd.to_numeric(melted["value"], errors="coerce")
    return melted


# =============================================================================
# Stats engines
# =============================================================================

def compare_groups_unmatched(df: pd.DataFrame,
                            metric: str,
                            group_col: str = "state_norm",
                            groups: Tuple[str, str] = ("awake", "sleep"),
                            stratify: Optional[List[str]] = None,
                            min_n: int = 5,
                            test: str = "mannwhitney") -> pd.DataFrame:
    stratify = stratify or []
    rows = []

    for keys, sub in df.groupby(stratify) if stratify else [((), df)]:
        if stratify:
            key_dict = dict(zip(stratify, keys if isinstance(keys, tuple) else (keys,)))
        else:
            key_dict = {}

        a = sub[sub[group_col] == groups[0]][metric].astype(float)
        b = sub[sub[group_col] == groups[1]][metric].astype(float)
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]

        if a.size < min_n or b.size < min_n:
            continue

        if test == "ttest":
            stat, p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
            test_name = "Welch_ttest"
        else:
            stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            test_name = "MannWhitneyU"

        rows.append({
            **key_dict,
            "metric": metric,
            "group_a": groups[0],
            "group_b": groups[1],
            "n_a": int(a.size),
            "n_b": int(b.size),
            "median_a": float(np.nanmedian(a)),  # Kept for reference
            "median_b": float(np.nanmedian(b)),  # Kept for reference
            "mean_a": float(np.nanmean(a)) if len(a) else np.nan,  # Primary
            "mean_b": float(np.nanmean(b)) if len(b) else np.nan,  # Primary
            "effect_cliffs_delta": cliffs_delta(a, b),
            "test": test_name,
            "stat": float(stat),
            "p": float(p),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_fdr"] = fdr_bh(out["p"].values)
    return out

def compare_awake_sleep_matched(master: pd.DataFrame,
                               metric: str,
                               id_col: str = "official_unit",
                               min_pairs: int = 3,
                               stratify: Optional[List[str]] = None) -> pd.DataFrame:
    stratify = stratify or []
    rows = []

    base = master.copy()
    base = base[np.isfinite(base[metric].astype(float))]

    # only units with official_unit
    base = base[base[id_col].notna()]

    for keys, sub in base.groupby(stratify) if stratify else [((), base)]:
        if stratify:
            key_dict = dict(zip(stratify, keys if isinstance(keys, tuple) else (keys,)))
        else:
            key_dict = {}

        piv = sub.pivot_table(index=id_col, columns="state_norm", values=metric, aggfunc="mean")
        if "awake" not in piv.columns or "sleep" not in piv.columns:
            continue
        x = piv["sleep"].values.astype(float)
        y = piv["awake"].values.astype(float)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]; y = y[mask]

        if x.size < min_pairs:
            continue

        # Wilcoxon is robust for skewed metrics
        try:
            stat, p = stats.wilcoxon(y, x, zero_method="wilcox", alternative="two-sided")
            test_name = "Wilcoxon_signed_rank"
        except Exception:
            stat, p = np.nan, np.nan
            test_name = "Wilcoxon_signed_rank"

        rows.append({
            **key_dict,
            "metric": metric,
            "n_pairs": int(x.size),
            "median_sleep": float(np.nanmedian(x)),
            "median_awake": float(np.nanmedian(y)),
            "mean_sleep": float(np.nanmean(x)),  # Primary
            "mean_awake": float(np.nanmean(y)),  # Primary
            "effect_rank_biserial": paired_rank_biserial(x, y),
            "test": test_name,
            "stat": float(stat) if np.isfinite(stat) else np.nan,
            "p": float(p) if np.isfinite(p) else np.nan,
            "n_official_units": int(piv.shape[0]),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_fdr"] = fdr_bh(out["p"].values)
    return out

def compare_states_omnibus(
    df: pd.DataFrame,
    metric: str,
    group_col: str = "state_norm",
    groups: Tuple[str, str, str] = ("awake", "sleep", "vitro"),
    stratify: Optional[List[str]] = None,
    min_n: int = 5,
) -> pd.DataFrame:
    stratify = stratify or []
    rows = []

    for keys, sub in df.groupby(stratify) if stratify else [((), df)]:
        key_dict = dict(zip(stratify, keys if isinstance(keys, tuple) else (keys,))) if stratify else {}

        gvals = []
        gnames = []
        for g in groups:
            x = pd.to_numeric(sub[sub[group_col] == g][metric], errors="coerce")
            x = x[np.isfinite(x)]
            if x.size >= min_n:
                gvals.append(x.values)
                gnames.append(g)

        if len(gvals) < 3:
            continue

        H, p = stats.kruskal(*gvals)
        n_total = int(sum(len(x) for x in gvals))
        k = len(gvals)
        eps2 = (H - k + 1) / (n_total - k) if n_total > k else np.nan

        row = {
            **key_dict,
            "metric": metric,
            "test": "KruskalWallis",
            "stat": float(H),
            "p": float(p),
            "effect_epsilon2": float(eps2) if np.isfinite(eps2) else np.nan,
            "groups": "|".join(gnames),
            "n_total": n_total,
        }
        for g, arr in zip(gnames, gvals):
            row[f"n_{g}"] = int(len(arr))
            row[f"median_{g}"] = float(np.nanmedian(arr))
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_fdr"] = fdr_bh(out["p"].values)
    return out

# =============================================================================
# Plotting
# =============================================================================

def set_plot_theme():
    sns.set_theme(style="white", context="talk", font_scale=0.9)
    plt.rcParams.update({
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.titleweight": "bold",
        "axes.grid": False,
    })

def palette_for_hue(hue: str):
    if hue == "state_norm":
        return COLORS_STATE
    if hue == "layer_norm":
        return COLORS_LAYER
    if hue == "celltype_norm":
        return COLORS_CELLTYPE
    if hue == "cellclass_ei":
        return COLORS_EI
    if hue == "histology_state":
        return COLORS_HISTOLOGY_STATE
    return None

def boxstrip_plot(df: pd.DataFrame,
                  metric: str,
                  x: str,
                  hue: Optional[str],
                  out_png: str,
                  title: str,
                  order: Optional[List[str]] = None,
                  hue_order: Optional[List[str]] = None,
                  annotate_sig: Optional[pd.DataFrame] = None,
                  sig_groups: Tuple[str, str] = ("awake", "sleep")):
    set_plot_theme()
    df = df.copy()
    if metric not in df.columns or x not in df.columns:
        return
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    keep_cols = [x, metric] + ([hue] if hue else [])
    df = df.dropna(subset=keep_cols)
    df = df[np.isfinite(df[metric])]
    if df.empty:
        return

    if order:
        order = [v for v in order if v in set(df[x])]
        if not order:
            return
    if hue and hue_order:
        hue_order = [v for v in hue_order if v in set(df[hue])]
        if not hue_order:
            hue_order = None

    fig, ax = plt.subplots(figsize=(9.5, 6))

    pal = palette_for_hue(hue) if hue else None

    enough_for_boxes = df.groupby(([x] + ([hue] if hue else [])), dropna=False)[metric].size().max() >= 2
    if enough_for_boxes:
        sns.boxplot(
            data=df, x=x, y=metric, hue=hue,
            order=order, hue_order=hue_order,
            palette=pal, ax=ax, showfliers=False, width=0.65
        )
    sns.stripplot(
        data=df, x=x, y=metric, hue=hue,
        order=order, hue_order=hue_order,
        dodge=True if hue else False,
        palette=pal, ax=ax, size=4, alpha=0.55, linewidth=0.3, edgecolor="black"
    )

    # de-duplicate legend (since box + strip add)
    if hue:
        handles, labels = ax.get_legend_handles_labels()
        uniq = {}
        for h, lab in zip(handles, labels):
            if lab not in uniq:
                uniq[lab] = h
        ax.legend(uniq.values(), uniq.keys(), title=hue, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    else:
        ax.legend_.remove() if ax.legend_ else None

    ax.set_title(title)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_xlabel(x.replace("_", " ").title())
    # Reserve minimal headroom for pairwise significance marks without letting outliers dominate.
    n_pairs = len(list(combinations(order, 2))) if order else 0
    clip_axis_with_sig_headroom(ax, df[metric].values, n_brackets=n_pairs, lo=0.02, hi=0.98)
    overlay_mean_sem_markers(ax, df, x, metric, order or sorted(df[x].dropna().unique()), hue=hue, hue_order=hue_order)
    annotate_group_ns(ax, df, x, metric, order or sorted(df[x].dropna().unique()), hue=hue, hue_order=hue_order)

    if hue is None and order and len(order) >= 2:
        stat_df = pairwise_mannwhitney_table(df, metric, x, order)
        _add_pairwise_sig_from_table(ax, stat_df, order, df[metric].values)
    elif hue is not None and order and hue_order:
        annotate_pairwise_hue_brackets(ax, df, x, metric, order, hue, hue_order)
        annotate_same_hue_across_x_brackets(ax, df, x, metric, order, hue, hue_order)

    if annotate_sig is not None and not annotate_sig.empty and (hue is None) and order and len(order) >= 2:
        # pick the row that matches metric
        row = annotate_sig[annotate_sig["metric"] == metric]
        if len(row):
            p = float(row.iloc[0]["q_fdr"]) if "q_fdr" in row.columns else float(row.iloc[0]["p"])
            text = f"{stars(p)} (q={p:.2g})"
            ymax = np.nanmax(df[metric].values.astype(float))
            ymin = np.nanmin(df[metric].values.astype(float))
            y = ymax + 0.08 * (ymax - ymin + 1e-12)
            add_sig_bracket(ax, 0, 1, y, text, dy=0.03 * (ymax - ymin + 1e-12))

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    save_figure(fig, out_png)
    plt.close(fig)

def paired_plot(master: pd.DataFrame,
                metric: str,
                out_png: str,
                title: str,
                id_col: str = "official_unit",
                facet_by: Optional[str] = None):
    """
    Paired sleep->awake lines for matched units.
    """
    set_plot_theme()

    df = master.copy()
    df = df[df["cellclass_ei"].isin(["Exc", "Inh"])]
    if id_col not in df.columns or metric not in df.columns:
        return  # Skip if required columns do not exist
    df = df[df[id_col].notna()]
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df[np.isfinite(df[metric])]
    if df.empty:
        return  # Skip if no valid matched data

    # pivot to pairs
    try:
        piv = df.pivot_table(index=[id_col] + ([facet_by] if facet_by else []),
                             columns="state_norm", values=metric, aggfunc="mean").reset_index()
    except KeyError:
        return  # Skip if pivot fails (e.g., missing columns)

    if "sleep" not in piv.columns or "awake" not in piv.columns:
        return

    if facet_by:
        facets = sorted([x for x in piv[facet_by].dropna().unique().tolist()])
        n = len(facets)
        fig, axs = plt.subplots(1, max(1, n), figsize=(5.2 * max(1, n), 5.8), sharey=True)
        axs = np.atleast_1d(axs)
        for ax, f in zip(axs, facets):
            sub = piv[piv[facet_by] == f]
            _paired_plot_ax(ax, sub, metric, subtitle=f"{facet_by}={f}")
        fig.suptitle(title, y=1.02)
        fig.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(6.2, 6.2))
        _paired_plot_ax(ax, piv, metric, subtitle=None)
        ax.set_title(title)
        fig.tight_layout()

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    save_figure(fig, out_png)
    plt.close(fig)

def _paired_plot_ax(ax, piv, metric, subtitle=None):
    x0, x1 = 0, 1
    sleep = piv["sleep"].values.astype(float)
    awake = piv["awake"].values.astype(float)
    mask = np.isfinite(sleep) & np.isfinite(awake)
    sleep = sleep[mask]; awake = awake[mask]
    for s, a in zip(sleep, awake):
        ax.plot([x0, x1], [s, a], color="black", alpha=0.25, lw=1)
    ax.scatter(np.full_like(sleep, x0), sleep, color=COLORS_STATE["sleep"], s=35, edgecolor="k", linewidth=0.3, alpha=0.9, label="sleep")
    ax.scatter(np.full_like(awake, x1), awake, color=COLORS_STATE["awake"], s=35, edgecolor="k", linewidth=0.3, alpha=0.9, label="awake")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["sleep", "awake"])
    ax.set_xlabel("")
    ax.set_ylabel(metric.replace("_", " ").title())
    if subtitle:
        ax.set_title(subtitle, fontsize=12)
    # stats annotation
    if sleep.size >= 5:
        try:
            stat, p = stats.wilcoxon(awake, sleep, zero_method="wilcox")
        except Exception:
            p = np.nan
        ymax = np.nanmax(np.r_[sleep, awake])
        ymin = np.nanmin(np.r_[sleep, awake])
        span = ymax - ymin if np.isfinite(ymax) and np.isfinite(ymin) and ymax > ymin else max(abs(ymax), 1.0)
        add_sig_bracket(ax, 0, 1, ymax + 0.08 * span, f"{stars(p)} p={p:.2g}", dy=0.03 * span)

def plot_matched_awake_sleep_panels(master: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    plot_dir = os.path.join(out_dir, "matched_awake_sleep_plots")
    os.makedirs(plot_dir, exist_ok=True)
    rows = []

    metrics = [m for m in [
        "firing_rate_Hz", "max_firing_rate_1s_Hz", "burstiness_percent", "bursti_2",
        "population_coupling", "coupling_strength_low_gamma", "coupling_strength_high_gamma",
        "mean_plv_delta", "mean_plv_theta", "mean_plv_low_gamma", "mean_plv_high_gamma",
    ] if m in master.columns]

    df = master.copy()
    if "official_unit" not in df.columns:
        return pd.DataFrame()
    df = df[df["official_unit"].notna()].copy()
    if df.empty:
        return pd.DataFrame()

    for metric in metrics:
        paired_plot(df, metric, os.path.join(plot_dir, f"matched__{metric}__global.png"), title=f"Matched awake vs sleep: {metric}", id_col="official_unit", facet_by=None)
        if "celltype_norm" in df.columns:
            paired_plot(df, metric, os.path.join(plot_dir, f"matched__{metric}__by_celltype.png"), title=f"Matched awake vs sleep by celltype: {metric}", id_col="official_unit", facet_by="celltype_norm")
            rows.append(compare_awake_sleep_matched(df, metric, stratify=["celltype_norm"], min_pairs=2))
        if "cellclass_ei" in df.columns:
            paired_plot(df, metric, os.path.join(plot_dir, f"matched__{metric}__by_ei.png"), title=f"Matched awake vs sleep by EI class: {metric}", id_col="official_unit", facet_by="cellclass_ei")
            rows.append(compare_awake_sleep_matched(df, metric, stratify=["cellclass_ei"], min_pairs=2))
        if "layer_norm" in df.columns:
            rows.append(compare_awake_sleep_matched(df, metric, stratify=["layer_norm"], min_pairs=2))
        rows.append(compare_awake_sleep_matched(df, metric, min_pairs=2))

    out = concat_nonempty(rows)
    if not out.empty:
        out.to_csv(os.path.join(out_dir, "stats_matched_awake_sleep_panels.csv"), index=False)
    return out


# =============================================================================
# Main runner (produces tables + plots)
# =============================================================================

DEFAULT_METRICS_UNITS = [
    "firing_rate_Hz",
    "max_firing_rate_10s_Hz",
    "max_firing_rate_1s_Hz",
    "burstiness_percent",
    "bursti_2",
    "autocorr_evenness",
    "amplitude_uV",
    "population_coupling",
    "halfwidth1_ms",
    "halfwidth2_ms",
    "peak_to_trough_delay_ms",
    "isi_cv",
    "ref_viol",
    "dimensionality_pc1_variance",
    "dimensionality_participation_ratio",
    "dimensionality_eigenspectrum_entropy",
    "dimensionality_eigenspectrum_entropy_normalized",
]

DEFAULT_METRICS_GRAPH = [
    "plv_low_gamma_strength", "plv_low_gamma_clustering", "plv_low_gamma_pagerank", "plv_low_gamma_betweenness", "plv_low_gamma_closeness", "plv_low_gamma_local_efficiency", "plv_low_gamma_degree", "plv_low_gamma_hubness_z",
    "plv_high_gamma_strength", "plv_high_gamma_clustering", "plv_high_gamma_pagerank", "plv_high_gamma_betweenness", "plv_high_gamma_closeness", "plv_high_gamma_local_efficiency", "plv_high_gamma_degree", "plv_high_gamma_hubness_z",
    "popc_scalar_strength", "popc_scalar_clustering", "popc_scalar_pagerank", "popc_scalar_betweenness", "popc_scalar_closeness", "popc_scalar_local_efficiency", "popc_scalar_degree", "popc_scalar_hubness_z",
    "frsim_strength", "frsim_clustering", "frsim_pagerank", "frsim_betweenness", "frsim_closeness", "frsim_local_efficiency", "frsim_degree", "frsim_hubness_z",
]

DEFAULT_METRICS_COUPLING = [
    "mean_plv_low_gamma", "max_plv_low_gamma", "mean_plv_sig_low_gamma", "max_plv_sig_low_gamma", "n_coupling_sig_epochs_low_gamma",
    "mean_plv_high_gamma", "max_plv_high_gamma", "mean_plv_sig_high_gamma", "max_plv_sig_high_gamma", "n_coupling_sig_epochs_high_gamma",
    "resultant_length_all_low_gamma", "resultant_length_sig_low_gamma", "circular_std_all_rad_low_gamma", "circular_std_sig_rad_low_gamma",
    "resultant_length_all_high_gamma", "resultant_length_sig_high_gamma", "circular_std_all_rad_high_gamma", "circular_std_sig_rad_high_gamma",
    # Added delta/theta
    "mean_plv_delta", "max_plv_delta", "mean_plv_sig_delta", "max_plv_sig_delta", "n_coupling_sig_epochs_delta",
    "mean_plv_theta", "max_plv_theta", "mean_plv_sig_theta", "max_plv_sig_theta", "n_coupling_sig_epochs_theta",
    "resultant_length_all_delta", "resultant_length_sig_delta", "circular_std_all_rad_delta", "circular_std_sig_rad_delta",
    "resultant_length_all_theta", "resultant_length_sig_theta", "circular_std_all_rad_theta", "circular_std_sig_rad_theta",
]

class DataLoader:
    def __init__(self, paths: Paths):
        self.paths = paths
    
    def load_master_database(self) -> pd.DataFrame:
        if self.paths.build_database_from_outputs:
            master = build_master_database_from_recording_outputs(
                outputs_root=self.paths.outputs_root,
                out_csv=self.paths.database_csv,
                units_manifest_csv=self.paths.units_manifest_csv,
            )
        else:
            master = load_master_database(self.paths.database_csv)

        checked = None
        if self.paths.checked_recordings_csv and os.path.exists(_to_windows_abs_path(self.paths.checked_recordings_csv)):
            checked = load_checked_recording_manifest(self.paths.checked_recordings_csv)
        master = apply_checked_recording_manifest(master, checked)
        if "celltype_norm" in master.columns:
            master["celltype_norm"] = master["celltype_norm"].map(normalize_celltype)
        if "cellclass_ei" not in master.columns:
            master["cellclass_ei"] = master.get("celltype_norm", pd.Series(index=master.index, dtype=object)).map(derive_ei_class)
        return master
    
    def load_connections_database(self, master: pd.DataFrame) -> pd.DataFrame:
        return load_connections_database(self.paths.connections_csv, master=master)
    
    def load_histology_database(self) -> pd.DataFrame:
        return load_histology_database(self.paths.histology_csv)
    
    def build_tables(self, master: pd.DataFrame, connections: pd.DataFrame, histology: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        tables = {
            "units": build_units_table(master),
            "graph": build_graph_table(master),
            "coupling": build_coupling_table(master),
            "recordings": build_recording_manifest(master),
            "connections": connections if not connections.empty else pd.DataFrame(),
            "histology": histology if not histology.empty else pd.DataFrame(),
        }
        if not histology.empty:
            hist_summary, ephys_summary, histo_ephys = build_histo_ephys_tables(master, histology)
            tables["hist_summary"] = hist_summary
            tables["ephys_summary"] = ephys_summary
            tables["histo_ephys"] = histo_ephys
        return tables

class StatsEngine:
    def __init__(self, master: pd.DataFrame, connections: pd.DataFrame, histology: pd.DataFrame):
        self.master = master
        self.connections = connections
        self.histology = histology
    
    def run_unmatched_comparisons(self) -> pd.DataFrame:
        rows = []
        for metric in DEFAULT_METRICS_UNITS + DEFAULT_METRICS_GRAPH + DEFAULT_METRICS_COUPLING:
            if metric not in self.master.columns:
                continue
            rows.extend([
                compare_groups_unmatched(self.master, metric, groups=("awake", "sleep")),
                compare_groups_unmatched(self.master, metric, groups=("awake", "vitro"), stratify=["layer_norm"]),
                compare_groups_unmatched(self.master, metric, groups=("sleep", "vitro"), stratify=["layer_norm"]),
            ])
        df = concat_nonempty(rows)
        if not df.empty:
            df["q_fdr"] = fdr_bh(df["p"].values)
        return df
    
    def run_matched_comparisons(self) -> pd.DataFrame:
        rows = []
        for metric in DEFAULT_METRICS_UNITS + DEFAULT_METRICS_GRAPH + DEFAULT_METRICS_COUPLING:
            if metric not in self.master.columns:
                continue
            rows.append(compare_awake_sleep_matched(self.master, metric, min_pairs=2))
            if "celltype_norm" in self.master.columns:
                rows.append(compare_awake_sleep_matched(self.master, metric, stratify=["celltype_norm"], min_pairs=2))
            if "cellclass_ei" in self.master.columns:
                rows.append(compare_awake_sleep_matched(self.master, metric, stratify=["cellclass_ei"], min_pairs=2))
            if "layer_norm" in self.master.columns:
                rows.append(compare_awake_sleep_matched(self.master, metric, stratify=["layer_norm"], min_pairs=2))
        df = concat_nonempty(rows)
        if not df.empty:
            df["q_fdr"] = fdr_bh(df["p"].values)
        return df
    
    def run_omnibus_tests(self) -> pd.DataFrame:
        rows = []
        for metric in DEFAULT_METRICS_UNITS + DEFAULT_METRICS_GRAPH + DEFAULT_METRICS_COUPLING:
            if metric not in self.master.columns:
                continue
            rows.append(compare_states_omnibus(self.master, metric))
        df = concat_nonempty(rows)
        if not df.empty:
            df["q_fdr"] = fdr_bh(df["p"].values)
        return df
    
    def correlate_histo_ephys(self) -> pd.DataFrame:
        if self.histology.empty:
            return pd.DataFrame()
        merged = build_histo_ephys_nonaveraged(self.master, self.histology)
        if merged.empty:
            return pd.DataFrame()

        hist_metrics = ["pv_density_cells_mm2", "pv_coverage_percent", "neun_density_cells_mm2", "neun_coverage_percent"]  # NeuN included
        ephys_metrics = [c for c in merged.columns if c.startswith("mean_")]
        rows = []

        # detect patients with both in vivo and in vitro histology and ephys
        try:
            pat_states = merged.groupby("patient")["histology_state"].apply(lambda x: set([s for s in x.dropna().astype(str).str.strip().str.lower()]))
            patients_with_both = [p for p, s in pat_states.items() if {"vivo", "vitro"}.issubset(s)]
            if patients_with_both:
                outp = os.path.join(self.out_dir if hasattr(self, 'out_dir') else ".", "patients_with_both_vivo_vitro_histology_and_ephys.csv")
                pd.DataFrame({"patient": patients_with_both}).to_csv(outp, index=False)
        except Exception:
            patients_with_both = []

        for cell_group in merged.get("ephys_cell_group", pd.Series(["all_units"]) ).dropna().unique():
            for e_state in merged.get("state_norm", pd.Series(dtype=object)).dropna().unique():
                base = merged[(merged["ephys_cell_group"] == cell_group) & (merged["state_norm"] == e_state)].copy()
                if base.empty:
                    continue
                for hist_state in base.get("histology_state", pd.Series(dtype=object)).dropna().unique():
                    sub_base = base[base["histology_state"].astype(str).str.strip().str.lower() == str(hist_state).strip().lower()].copy()
                    if sub_base.empty:
                        continue
                    for hmet in hist_metrics:
                        for emet in ephys_metrics:
                            sub = sub_base[["patient", "layer_norm", hmet, emet]].copy()
                            sub[hmet] = pd.to_numeric(sub[hmet], errors="coerce")
                            sub[emet] = pd.to_numeric(sub[emet], errors="coerce")
                            sub = sub[np.isfinite(sub[hmet]) & np.isfinite(sub[emet])]
                            if len(sub) < 3:
                                continue
                            rho, p = stats.spearmanr(sub[hmet], sub[emet])
                            rows.append({
                                "cell_group": cell_group,
                                "state": e_state,
                                "histology_state": hist_state,
                                "hist_metric": hmet,
                                "ephys_metric": emet,
                                "n": int(len(sub)),
                                "n_patients": int(sub["patient"].nunique()),
                                "rho": float(rho),
                                "p": float(p)
                            })

        df = pd.DataFrame(rows)
        if not df.empty and "p" in df.columns:
            df["q_fdr"] = fdr_bh(df["p"].values)
        return df

class Plotter:
    def __init__(self, master: pd.DataFrame, out_dir: str):
        self.master = master
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
    
    def plot_half_violins_histology(self):
        if not hasattr(self.master, 'histology') or self.master.histology.empty:
            return
        hist = self.master.histology
        metrics = [("pv_density_cells_mm2", "PV Density (cells/mm²)"), ("neun_density_cells_mm2", "NeuN Density (cells/mm²)"),
                   ("pv_coverage_percent", "PV Coverage (%)"), ("neun_coverage_percent", "NeuN Coverage (%)")]
        for metric, ylabel in metrics:
            if metric not in hist.columns:
                continue
            sub = hist[hist["layer_norm"].isin(["supra", "infra"]) & hist["histology_state"].isin(["vivo", "vitro"])]
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.violinplot(data=sub, x="layer_norm", y=metric, hue="histology_state", split=True, inner="quart", palette=COLORS_HISTOLOGY_STATE, ax=ax)
            ax.set_title(f"{ylabel} by Layer and State", fontsize=14, fontweight="bold")
            ax.set_xlabel("Layer")
            ax.set_ylabel(ylabel)
            plt.tight_layout()
            save_figure(fig, os.path.join(self.out_dir, f"histology_half_violin_{metric}.png"))
    
    def plot_boxstrip_with_significance(self, metric: str, x: str, hue: Optional[str], title: str, order: List[str], hue_order: List[str], stat_df: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.boxplot(data=self.master, x=x, y=metric, hue=hue, order=order, hue_order=hue_order, palette=COLORS_STATE if hue == "state_norm" else None, showfliers=False, ax=ax)
        sns.stripplot(data=self.master, x=x, y=metric, hue=hue, order=order, hue_order=hue_order, dodge=True, alpha=0.6, ax=ax)
        # Annotate only significant (q_fdr < 0.05), including cross-layer for same celltype
        sig = stat_df[stat_df["q_fdr"] < 0.05]
        for _, row in sig.iterrows():
            # Add brackets for pairwise (expand logic for cross-groups)
            pass  # Implement based on annotate_pairwise functions
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel(x.replace("_", " ").title())
        ax.set_ylabel(metric.replace("_", " ").title())
        plt.tight_layout()
        save_figure(fig, os.path.join(self.out_dir, f"{metric}_{x}.png"))

class ResultsExporter:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
    
    def export_tables(self, tables: Dict[str, pd.DataFrame]):
        for name, df in tables.items():
            df.to_csv(os.path.join(self.out_dir, f"table_{name}.csv"), index=False)
    
    def export_stats(self, stats: Dict[str, pd.DataFrame]):
        for name, df in stats.items():
            df.to_csv(os.path.join(self.out_dir, f"stats_{name}.csv"), index=False)
        # Excel exports (color-coded summary + significant-only)
        try:
            xlsx_all = os.path.join(self.out_dir, "stats_reporting_colored.xlsx")
            xlsx_sig = os.path.join(self.out_dir, "stats_significant_only_colored.xlsx")
            with pd.ExcelWriter(xlsx_all, engine="xlsxwriter") as w_all, pd.ExcelWriter(xlsx_sig, engine="xlsxwriter") as w_sig:
                for name, df in stats.items():
                    if df is None or df.empty:
                        continue
                    sheet = str(name)[:31]
                    df.to_excel(w_all, sheet_name=sheet, index=False)
                    if "q_fdr" in df.columns:
                        sig = df[pd.to_numeric(df["q_fdr"], errors="coerce") < 0.05].copy()
                    elif "p" in df.columns:
                        sig = df[pd.to_numeric(df["p"], errors="coerce") < 0.05].copy()
                    else:
                        sig = pd.DataFrame()
                    if not sig.empty:
                        sig.to_excel(w_sig, sheet_name=sheet, index=False)
        except Exception:
            pass
    
    def create_significant_summary(self, stats: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        sig_rows = []
        for name, df in stats.items():
            if df is None or df.empty or "q_fdr" not in df.columns:
                continue
            sig = df[df["q_fdr"] < 0.05].copy()
            if sig.empty:
                continue
            sig["analysis_type"] = name
            sig_rows.append(sig)
        summary = concat_nonempty(sig_rows)
        if summary.empty:
            summary.to_csv(os.path.join(self.out_dir, "significant_changes_summary.csv"), index=False)
            return summary

        def metric_family(metric: str) -> str:
            m = str(metric).lower()
            if "dimensionality_" in m:
                return "dimensionality"
            if "plv_" in m or "coupling" in m or "phase" in m:
                return "coupling_phase"
            if "halfwidth" in m or "isi" in m or "burst" in m or "firing" in m:
                return "spike_waveform"
            if "pagerank" in m or "clustering" in m or "hubness" in m or "strength" in m or "degree" in m:
                return "graph"
            return "other"

        summary["metric"] = summary.get("metric", pd.Series(index=summary.index, dtype=object)).astype(str)
        summary["metric_family"] = summary["metric"].map(metric_family)
        summary["effect_size"] = summary.get("effect_cliffs_delta", summary.get("effect_rank_biserial", np.nan))
        summary["comparison"] = summary.get("group_a", "").astype(str) + "_vs_" + summary.get("group_b", "").astype(str)
        summary["stratifier"] = ""
        for c in ["state_norm", "layer_norm", "celltype_norm", "cellclass_ei", "histology_state"]:
            if c in summary.columns:
                summary["stratifier"] = np.where(summary["stratifier"].eq(""), c + "=" + summary[c].astype(str), summary["stratifier"] + "; " + c + "=" + summary[c].astype(str))
        keep_cols = [c for c in [
            "analysis_type", "metric_family", "metric", "comparison", "stratifier", "q_fdr", "p", "test", "effect_size",
            "n_a", "n_b", "n_pairs", "n_total", "n_patient_layer_points", "n_patients"
        ] if c in summary.columns]
        summary_struct = summary[keep_cols].sort_values(["analysis_type", "metric_family", "q_fdr"], na_position="last")
        summary_struct.to_csv(os.path.join(self.out_dir, "significant_changes_summary_structured.csv"), index=False)
        summary.to_csv(os.path.join(self.out_dir, "significant_changes_summary.csv"), index=False)
        return summary_struct
    
    def export_key_results_text(self, summary: pd.DataFrame):
        with open(os.path.join(self.out_dir, "key_results_summary.txt"), "w") as f:
            f.write("Key Significant Findings\n")
            f.write("=" * 30 + "\n")
            for _, row in summary.head(20).iterrows():
                f.write(f"- {row.get('metric', 'N/A')}: {row.get('group_a', '')} vs {row.get('group_b', '')}, q={row['q_fdr']:.3g}\n")

class StatsRunner:
    def __init__(self, paths: Paths):
        self.paths = paths
        self.loader = DataLoader(paths)
        self.exporter = ResultsExporter(paths.results_dir)
    
    def run(self):
        master = self.loader.load_master_database()
        # Attach matched IDs from external table only when official_unit is missing.
        if ("official_unit" not in master.columns or master["official_unit"].isna().all()) and self.paths.matched_units_csv and os.path.exists(_to_windows_abs_path(self.paths.matched_units_csv)):
            matched_long = load_matched_units(self.paths.matched_units_csv)
            master = attach_matched_ids(master, matched_long)
            master = attach_matched_ids_from_checked_table(master, self.paths.matched_units_csv)
        connections = self.loader.load_connections_database(master)
        histology = self.loader.load_histology_database()
        tables = self.loader.build_tables(master, connections, histology)
        
        stats_engine = StatsEngine(master, connections, histology)
        stats = {
            "unmatched": stats_engine.run_unmatched_comparisons(),
            "matched": stats_engine.run_matched_comparisons(),
            "omnibus": stats_engine.run_omnibus_tests(),
        }
        stats["histo_ephys_mixedlm"] = run_histo_ephys_mixedlm_associations(master, histology, self.paths.results_dir)
        stats["coupling_statistical_reporting"] = build_coupling_statistical_reporting(master, stats, self.paths.results_dir)
        
        plot_root = os.path.join(self.paths.results_dir, "plots")
        plotter = Plotter(master, plot_root)
        plotter.plot_half_violins_histology()
        if "hist_summary" in tables and not tables["hist_summary"].empty:
            plot_histology_summary(tables["hist_summary"], plot_root)
        plot_celltype_metric_panels(master, plot_root)
        plot_ei_metric_panels(master, plot_root)
        # removed per request
        plot_halfwidth_celltype_state_publication(master, plot_root)
        plot_halfwidth_focus_panels(master, plot_root)
        plot_burstiness_variants(master, plot_root)
        plot_coupling_significance_panels(master, plot_root)
        export_coupling_significance_summary(master, self.paths.results_dir)
        build_and_plot_celltype_composition(master, plot_root)
        save_population_phase_polar_plots(master, plot_root)
        plot_matched_awake_sleep_panels(master, plot_root)
        plot_histology_half_violins(histology, plot_root)
        plot_graph_metric_cellclass_differences(master, plot_root)
        plot_histo_ephys_mixedlm_matrices(stats.get("histo_ephys_mixedlm", pd.DataFrame()), plot_root)
        if "histo_ephys" in tables and isinstance(tables["histo_ephys"], pd.DataFrame) and not tables["histo_ephys"].empty:
            export_histo_ephys_mean_tables(tables["histo_ephys"], self.paths.results_dir)
        if histology is not None and not histology.empty:
            compute_bare_histology_stats(histology, self.paths.results_dir)
        if connections is not None and not connections.empty:
            plot_putative_connection_summary(connections, plot_root)

        recording_manifest = build_recording_manifest(master)
        if not recording_manifest.empty:
            recording_manifest.to_csv(os.path.join(self.paths.results_dir, "recording_manifest.csv"), index=False)
            if self.paths.out_dir:
                recording_manifest.to_csv(os.path.join(self.paths.out_dir, "recording_manifest.csv"), index=False)
        
        self.exporter.export_tables(tables)
        self.exporter.export_stats(stats)
        summary = self.exporter.create_significant_summary(stats)
        self.exporter.export_key_results_text(summary)

if __name__ == "__main__":
    paths = Paths(
        database_csv=r"E:/in_vivo_in_vitro/stat/statbase/vivo_vitro_database_from_outputs_checked.csv",
        connections_csv=r"E:/in_vivo_in_vitro/stat/statbase/connections_master_checked.csv",
        histology_csv=r"E:/in_vivo_in_vitro/stat/histo_stat/in_vivo_in_vitro_pv_neun_histology.csv",
        matched_units_csv=r"E:/in_vivo_in_vitro/stat/statbase/vivo_sleep_awake_matched_units_checked.csv",
        checked_recordings_csv=r"E:/in_vivo_in_vitro/stat/statbase/table_recordings_checked.csv",
        outputs_root=r"E:/in_vivo_in_vitro/stat",
        units_manifest_csv=r"E:/in_vivo_in_vitro/stat/vivo_vitro_units.csv",
        build_database_from_outputs=False,
        out_dir=None,
        results_dir="stats_06_02",
    )
    runner = StatsRunner(paths)
    runner.run()
