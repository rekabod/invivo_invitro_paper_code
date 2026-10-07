"""
Histology + electrophysiology analysis pipeline.

Purpose
-------
Analyze ROI-level histology measurements and patient/layer/state/cellclass
electrophysiology summaries in a statistically cautious way.

Key ideas
---------
1) Histology rows are ROI-level observations.
2) Primary inference should be at patient-aggregated or patient×layer-aggregated level.
3) ROI-level plots are fine for visualization.
4) Correlations should be computed mainly on aggregated observations.
5) If you want ROI-level inference, use mixed-effects models with patient random effects.

Inputs
------
Histology CSV expected columns (best effort):
- state: vivo / vitro
- staining: PV / NeuN
- patient
- layer: supra / infra
- density (cell/mm^2)
- coverage (%)

Ephys CSV / master table expected best-effort columns:
- patient
- state_norm or state
- layer_norm or layer
- celltype_norm / celltype / cellclass_ei
- firing_rate_Hz
- max_firing_rate_1s_Hz
- burstiness_percent
- isi_cv or ISI-related metric
- mean_plv_delta, mean_plv_theta, mean_plv_low_gamma, mean_plv_high_gamma
- population_coupling
- optional other metrics

Outputs
-------
- QC tables
- histology summary tables
- histology comparison tables
- ephys summary tables
- histology–ephys correlation tables
- figures
"""

from __future__ import annotations

import os
import re
import math
import warnings
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore", category=RuntimeWarning)

sns.set_theme(style="white", context="paper", font_scale=1.05)
plt.rcParams["axes.grid"] = False
plt.rcParams["figure.dpi"] = 140


# =============================================================================
# Config
# =============================================================================

STATE_ORDER = ["vivo", "vitro"]
LAYER_ORDER = ["supra", "infra"]
STAIN_ORDER = ["PV", "NEUN"]

CELLCLASS_ORDER = ["all", "Exc", "Inh", "FS"]

HISTOLOGY_METRICS = [
    "density (cell/mm^2)",
    "coverage (%)",
    "pv_neun_density_ratio",
]

EPHYS_METRICS = [
    "firing_rate_Hz",
    "max_firing_rate_1s_Hz",
    "max_firing_rate_10s_Hz",
    "burstiness_percent",
    "bursti_2",
    "isi_lt20ms",
    "population_coupling",
    "mean_plv_sig_delta",
    "mean_plv_sig_theta",
    "mean_plv_sig_low_gamma",
    "mean_plv_sig_high_gamma",
    "resultant_length_sig_delta",
    "resultant_length_sig_theta",
    "resultant_length_sig_low_gamma",
    "resultant_length_sig_high_gamma",
]

HISTOLOGY_LABELS = {
    "PV_density": "PV density",
    "NEUN_density": "NeuN density",
    "PV_coverage": "PV coverage",
    "pv_neun_density_ratio": "PV/NeuN ratio",
}

EPHYS_LABELS = {
    "firing_rate_Hz": "Firing rate",
    "max_firing_rate_1s_Hz": "Max firing rate (1 s)",
    "max_firing_rate_10s_Hz": "Max firing rate (10 s)",
    "burstiness_percent": "Burstiness (%)",
    "bursti_2": "Burst index (B2)",
    "isi_lt20ms": "ISI < 20 ms",
    "population_coupling": "Population coupling",
    "mean_plv_sig_delta": "Mean sig. PLV, delta",
    "mean_plv_sig_theta": "Mean sig. PLV, theta",
    "mean_plv_sig_low_gamma": "Mean sig. PLV, low gamma",
    "mean_plv_sig_high_gamma": "Mean sig. PLV, high gamma",
    "resultant_length_sig_delta": "Resultant length, delta",
    "resultant_length_sig_theta": "Resultant length, theta",
    "resultant_length_sig_low_gamma": "Resultant length, low gamma",
    "resultant_length_sig_high_gamma": "Resultant length, high gamma",
}

STATE_LABELS = {"vivo": "In vivo", "vitro": "In vitro"}
CELLCLASS_LABELS = {
    "all": "All cells",
    "Exc": "Excitatory cells",
    "Inh": "Inhibitory cells",
    "FS": "FS cells",
}


# =============================================================================
# Utilities
# =============================================================================

def _safe_float(x):
    try:
        if x is None:
            return np.nan
        if isinstance(x, str) and x.strip() == "":
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def _as_text(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def normalize_state(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "other"
    s = str(x).strip().lower()
    if s in ("vivo", "in_vivo", "invivo"):
        return "vivo"
    if s in ("vitro", "in_vitro", "invitro"):
        return "vitro"
    return "other"


def normalize_layer(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "other"
    s = str(x).strip().lower()
    if s.startswith("sup"):
        return "supra"
    if s.startswith("inf"):
        return "infra"
    return "other"


def normalize_stain(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "other"
    s = str(x).strip().lower()
    if s in ("pv", "pval", "parvalbumin"):
        return "PV"
    if s in ("neun", "neu-n", "neuronal nuclei"):
        return "NEUN"
    return str(x).strip().upper()


def normalize_celltype(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "other"
    s = str(x).strip().lower()
    if s in ("rs", "rs-pc", "rs_pc", "regular spiking", "regularly spiking"):
        return "RS-PC"
    if s in ("ib", "ib-pc", "ib_pc", "burst", "bursting"):
        return "IB-PC"
    if s in ("in", "interneuron", "int"):
        return "IN"
    if s in ("fs", "fast spiking", "fast_spiking", "fast-spiking"):
        return "FS"
    if s in ("exc", "excitatory"):
        return "Exc"
    if s in ("inh", "inhibitory"):
        return "Inh"
    return str(x).strip()


def derive_cellclass_ei(celltype: str) -> str:
    s = str(celltype).strip()
    if s in ("RS-PC", "IB-PC"):
        return "Exc"
    if s in ("IN",):
        return "Inh"
    if s in ("FS",):
        return "FS"
    return "other"


def fdr_bh(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    out = np.full_like(pvals, np.nan, dtype=float)
    ok = np.isfinite(pvals)
    p = pvals[ok]
    if p.size == 0:
        return out
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * (p.size / np.arange(1, p.size + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    tmp = np.full(p.size, np.nan)
    tmp[order] = q
    out[ok] = tmp
    return out


def cliffs_delta(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return np.nan
    gt = 0
    lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)
    return float((gt - lt) / (x.size * y.size))


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


def save_figure(fig, out_png: str, also_svg: bool = True):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    if also_svg:
        root, _ = os.path.splitext(out_png)
        fig.savefig(root + ".svg", bbox_inches="tight")


def circular_mean(phases, weights=None) -> Tuple[float, float]:
    phases = np.asarray(phases, dtype=float)
    mask = np.isfinite(phases)
    phases = phases[mask]
    if phases.size == 0:
        return np.nan, np.nan
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.size == mask.size:
            weights = weights[mask]
        else:
            weights = None
    vec = np.average(np.exp(1j * phases), weights=weights) if weights is not None else np.mean(np.exp(1j * phases))
    return float(np.angle(vec)), float(np.abs(vec))


def is_positive_metric(series: pd.Series) -> bool:
    s = pd.to_numeric(series, errors="coerce")
    s = s[np.isfinite(s)]
    return bool((s > 0).all()) if len(s) else False


def zscore_safe(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    mu = float(np.nanmean(s))
    sd = float(np.nanstd(s, ddof=0))
    if not np.isfinite(sd) or sd == 0:
        return s * np.nan
    return (s - mu) / sd


def pretty_label(name: str) -> str:
    if name in HISTOLOGY_LABELS:
        return HISTOLOGY_LABELS[name]
    if name in EPHYS_LABELS:
        return EPHYS_LABELS[name]
    if name in STATE_LABELS:
        return STATE_LABELS[name]
    if name in CELLCLASS_LABELS:
        return CELLCLASS_LABELS[name]
    if name in {"supra", "infra"}:
        return str(name).capitalize()
    return str(name).replace("_", " ").strip().title()


# =============================================================================
# Data loading
# =============================================================================

@dataclass
class Paths:
    histology_csv: str
    ephys_csv: Optional[str] = None
    out_dir: str = "histology_ephys_out"
    ephys_is_master_table: bool = True


def load_histology(path_csv: str) -> pd.DataFrame:
    df = pd.read_csv(path_csv).copy()
    if df.empty:
        return df

    # normalize columns
    rename_map = {}
    for c in df.columns:
        cc = str(c).strip()
        if cc != c:
            rename_map[c] = cc
    if rename_map:
        df = df.rename(columns=rename_map)

    if "state" not in df.columns and "condition" in df.columns:
        df["state"] = df["condition"]
    if "layer" not in df.columns and "region" in df.columns:
        df["layer"] = df["region"]

    df["state_norm"] = df.get("state", pd.Series(index=df.index, dtype=object)).map(normalize_state)
    df["layer_norm"] = df.get("layer", pd.Series(index=df.index, dtype=object)).map(normalize_layer)
    df["staining_norm"] = df.get("staining", pd.Series(index=df.index, dtype=object)).map(normalize_stain)
    df["patient"] = df.get("patient", pd.Series(index=df.index, dtype=object)).astype(str).str.strip()

    # numeric
    for c in ["density (cell/mm^2)", "coverage (%)"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def load_ephys(path_csv: str) -> pd.DataFrame:
    df = pd.read_csv(path_csv).copy()
    if df.empty:
        return df

    rename_map = {}
    for c in df.columns:
        cc = str(c).strip()
        if cc != c:
            rename_map[c] = cc
    if rename_map:
        df = df.rename(columns=rename_map)

    if "state_norm" not in df.columns and "state" in df.columns:
        df["state_norm"] = df["state"].map(normalize_state)
    elif "state_norm" in df.columns:
        df["state_norm"] = df["state_norm"].map(normalize_state)
    else:
        df["state_norm"] = "other"

    if "layer_norm" not in df.columns and "layer" in df.columns:
        df["layer_norm"] = df["layer"].map(normalize_layer)
    elif "layer_norm" in df.columns:
        df["layer_norm"] = df["layer_norm"].map(normalize_layer)
    else:
        df["layer_norm"] = "other"

    if "celltype_norm" not in df.columns:
        if "celltype" in df.columns:
            df["celltype_norm"] = df["celltype"].map(normalize_celltype)
        elif "cellclass_ei" in df.columns:
            df["celltype_norm"] = df["cellclass_ei"].astype(str)
        else:
            df["celltype_norm"] = "other"
    else:
        df["celltype_norm"] = df["celltype_norm"].map(normalize_celltype)

    if "cellclass_ei" not in df.columns:
        df["cellclass_ei"] = df["celltype_norm"].map(derive_cellclass_ei)

    if "patient" in df.columns:
        df["patient"] = df["patient"].astype(str).str.strip()
    else:
        df["patient"] = ""

    # numeric columns
    for c in df.columns:
        if any(k in c.lower() for k in ["hz", "burst", "isi", "plv", "coupling", "ratio", "strength", "delay", "width", "entropy", "variance"]):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # add ISI<20ms if available as derived metric
    if "isi_lt20ms" not in df.columns:
        for candidate in ["isi_20ms", "isi_lt_20ms", "isi20_ms", "n_isi_lt20ms"]:
            if candidate in df.columns:
                df["isi_lt20ms"] = pd.to_numeric(df[candidate], errors="coerce")
                break

    return df


# =============================================================================
# Aggregation helpers
# =============================================================================

def patient_level_histology_summary(hist: pd.DataFrame) -> pd.DataFrame:
    if hist is None or hist.empty:
        return pd.DataFrame()
    cols = [c for c in ["patient", "state_norm", "layer_norm", "staining_norm"] if c in hist.columns]
    if len(cols) < 4:
        return pd.DataFrame()

    d = hist.copy()
    d["density (cell/mm^2)"] = pd.to_numeric(d.get("density (cell/mm^2)"), errors="coerce")
    d["coverage (%)"] = pd.to_numeric(d.get("coverage (%)"), errors="coerce")

    agg = (
        d.groupby(cols, dropna=False)
         .agg(
             n_roi=("patient", "size"),
             mean_density=("density (cell/mm^2)", "mean"),
             median_density=("density (cell/mm^2)", "median"),
             sd_density=("density (cell/mm^2)", "std"),
             mean_coverage=("coverage (%)", "mean"),
             median_coverage=("coverage (%)", "median"),
             sd_coverage=("coverage (%)", "std"),
         )
         .reset_index()
    )
    return agg


def patient_level_ephys_summary(ephys: pd.DataFrame) -> pd.DataFrame:
    if ephys is None or ephys.empty:
        return pd.DataFrame()
    cols = [c for c in ["patient", "state_norm", "layer_norm", "cellclass_ei"] if c in ephys.columns]
    if len(cols) < 4:
        return pd.DataFrame()

    d = ephys.copy()
    for c in d.columns:
        if c in ["patient", "state_norm", "layer_norm", "cellclass_ei", "celltype_norm"]:
            continue
        d[c] = pd.to_numeric(d[c], errors="coerce")

    metrics = [m for m in EPHYS_METRICS if m in d.columns]
    if not metrics:
        return pd.DataFrame()

    agg = (
        d.groupby(cols, dropna=False)
         .agg(**{f"mean_{m}": (m, "mean") for m in metrics},
              **{f"median_{m}": (m, "median") for m in metrics},
              n_cells=("patient", "size"),
              **{f"n_{m}": (m, lambda x: int(pd.to_numeric(x, errors="coerce").notna().sum())) for m in metrics})
         .reset_index()
    )
    return _attach_ephys_fraction_metrics(agg)


def make_ratio_table(hist: pd.DataFrame) -> pd.DataFrame:
    if hist is None or hist.empty:
        return pd.DataFrame()
    d = hist.copy()
    d["density (cell/mm^2)"] = pd.to_numeric(d.get("density (cell/mm^2)"), errors="coerce")
    piv = d.pivot_table(
        index=["patient", "state_norm", "layer_norm"],
        columns="staining_norm",
        values="density (cell/mm^2)",
        aggfunc="mean",
    )
    out = piv.reset_index()
    if "PV" in out.columns and "NEUN" in out.columns:
        out["pv_neun_density_ratio"] = out["PV"] / out["NEUN"]
    return out


def build_histology_long_table(hist: pd.DataFrame) -> pd.DataFrame:
    """
    Patient/state/layer histology table in long form.
    """
    if hist is None or hist.empty:
        return pd.DataFrame()

    hist_summary = patient_level_histology_summary(hist)
    ratio_tbl = make_ratio_table(hist)

    rows = []
    if not hist_summary.empty:
        for stain, metric_prefix in [("PV", "PV"), ("NEUN", "NEUN")]:
            sub = hist_summary[hist_summary["staining_norm"] == stain].copy()
            if sub.empty:
                continue
            for metric_name, col in [
                ("density (cell/mm^2)", "mean_density"),
                ("coverage (%)", "mean_coverage"),
            ]:
                if col not in sub.columns:
                    continue
                tmp = sub[["patient", "state_norm", "layer_norm"]].copy()
                tmp["histology_metric"] = f"{metric_prefix}_{metric_name}"
                tmp["histology_value"] = pd.to_numeric(sub[col], errors="coerce").values
                rows.append(tmp)

    if not ratio_tbl.empty and "pv_neun_density_ratio" in ratio_tbl.columns:
        tmp = ratio_tbl[["patient", "state_norm", "layer_norm"]].copy()
        tmp["histology_metric"] = "pv_neun_density_ratio"
        tmp["histology_value"] = pd.to_numeric(ratio_tbl["pv_neun_density_ratio"], errors="coerce")
        rows.append(tmp)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(subset=["patient", "state_norm", "layer_norm", "histology_value"])
    return out


def build_ephys_unit_long_table(ephys: pd.DataFrame) -> pd.DataFrame:
    """
    Single-unit ephys table in long form.
    """
    if ephys is None or ephys.empty:
        return pd.DataFrame()

    d = ephys.copy()
    if "cellclass_ei" not in d.columns and "celltype_norm" in d.columns:
        d["cellclass_ei"] = d["celltype_norm"].map(derive_cellclass_ei)

    rows = []
    all_view = d.copy()
    all_view["cellclass_ei"] = "all"
    for frame in [all_view, d]:
        for metric in EPHYS_METRICS:
            if metric not in frame.columns:
                continue
            tmp = frame[["patient", "state_norm", "layer_norm", "cellclass_ei"]].copy()
            tmp["ephys_metric"] = metric
            tmp["ephys_value"] = pd.to_numeric(frame[metric], errors="coerce").values
            rows.append(tmp)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(subset=["patient", "state_norm", "layer_norm", "cellclass_ei", "ephys_value"])
    return out


def fit_mixed_association(sub: pd.DataFrame) -> Dict[str, object]:
    """
    Fit a patient-random-intercept mixed model:
        ephys_z ~ hist_z + C(layer_norm)
    Falls back to clustered OLS if the mixed model fails.
    """
    sub = sub.copy()
    sub["hist_z"] = zscore_safe(sub["histology_value"])
    sub["ephys_z"] = zscore_safe(sub["ephys_value"])
    sub = sub[np.isfinite(sub["hist_z"]) & np.isfinite(sub["ephys_z"])].copy()

    if len(sub) < 5 or sub["patient"].nunique() < 3:
        return {
            "beta": np.nan,
            "p_value": np.nan,
            "method": "insufficient_data",
            "converged": False,
        }

    has_layer = sub["layer_norm"].nunique(dropna=True) > 1
    formula = "ephys_z ~ hist_z + C(layer_norm)" if has_layer else "ephys_z ~ hist_z"

    try:
        model = smf.mixedlm(formula, sub, groups=sub["patient"], re_formula="1")
        result = model.fit(reml=False, method="lbfgs", maxiter=200, disp=False)
        beta = float(result.params.get("hist_z", np.nan))
        p_value = float(result.pvalues.get("hist_z", np.nan))
        return {
            "beta": beta,
            "p_value": p_value,
            "method": "mixedlm",
            "converged": bool(getattr(result, "converged", True)),
        }
    except Exception:
        try:
            ols = smf.ols(formula, sub).fit(
                cov_type="cluster",
                cov_kwds={"groups": sub["patient"]},
            )
            beta = float(ols.params.get("hist_z", np.nan))
            p_value = float(ols.pvalues.get("hist_z", np.nan))
            return {
                "beta": beta,
                "p_value": p_value,
                "method": "ols_cluster",
                "converged": True,
            }
        except Exception:
            return {
                "beta": np.nan,
                "p_value": np.nan,
                "method": "failed",
                "converged": False,
            }


def run_unit_level_mixed_models(hist: pd.DataFrame, ephys: pd.DataFrame, out_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Build 8 state x cellclass unit-level histology/ephys association matrices.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_tables = {}

    hist_long = build_histology_long_table(hist)
    ephys_long = build_ephys_unit_long_table(ephys)
    if hist_long.empty or ephys_long.empty:
        return out_tables

    merged = hist_long.merge(ephys_long, on=["patient", "state_norm", "layer_norm"], how="inner")
    if merged.empty:
        return out_tables

    rows = []
    states_present = [s for s in STATE_ORDER if s in set(merged["state_norm"])]
    cellclasses_present = [c for c in CELLCLASS_ORDER if c in set(merged["cellclass_ei"])]
    state_iter = [s for s in STATE_ORDER if s in states_present]
    cellclass_iter = [c for c in CELLCLASS_ORDER if c in cellclasses_present]

    for state in state_iter:
        for cellclass in cellclass_iter:
            sub_cell = merged[(merged["state_norm"] == state) & (merged["cellclass_ei"] == cellclass)].copy()
            if sub_cell.empty:
                continue
            for hmet in sorted(sub_cell["histology_metric"].dropna().unique()):
                for emet in sorted(sub_cell["ephys_metric"].dropna().unique()):
                    sub = sub_cell[
                        (sub_cell["histology_metric"] == hmet) &
                        (sub_cell["ephys_metric"] == emet)
                    ].copy()
                    if sub.empty:
                        continue
                    fit = fit_mixed_association(sub)
                    if not np.isfinite(fit.get("beta", np.nan)):
                        continue
                    rows.append({
                        "state_norm": state,
                        "cellclass_ei": cellclass,
                        "histology_metric": hmet,
                        "histology_label": pretty_label(hmet),
                        "ephys_metric": emet,
                        "ephys_label": pretty_label(emet),
                        "beta_std": fit["beta"],
                        "p_value": fit["p_value"],
                        "method": fit["method"],
                        "converged": fit["converged"],
                        "n_units": int(len(sub)),
                        "n_patients": int(sub["patient"].nunique()),
                        "n_layers": int(sub["layer_norm"].nunique()),
                    })

    out = pd.DataFrame(rows)
    if out.empty:
        return out_tables

    out["q_value"] = fdr_bh(out["p_value"].values)
    out = out.sort_values(["q_value", "p_value"], na_position="last")
    out.to_csv(os.path.join(out_dir, "unit_level_mixedlm_associations.csv"), index=False)

    # one table per state x cellclass
    for state in STATE_ORDER:
        for cellclass in CELLCLASS_ORDER:
            sub = out[(out["state_norm"] == state) & (out["cellclass_ei"] == cellclass)].copy()
            fname = f"mixedlm__{state}__{cellclass}.csv"
            sub.to_csv(os.path.join(out_dir, fname), index=False)
            out_tables[f"mixedlm__{state}__{cellclass}"] = sub

    out_tables["mixedlm_all"] = out
    return out_tables


def _attach_ephys_fraction_metrics(ephys_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-patient/state/layer cell-count totals and fractions.
    """
    if ephys_summary is None or ephys_summary.empty or "n_cells" not in ephys_summary.columns:
        return ephys_summary

    d = ephys_summary.copy()
    totals = (
        d.groupby(["patient", "state_norm", "layer_norm"], dropna=False)["n_cells"]
         .sum()
         .reset_index(name="n_cells_total")
    )
    d = d.merge(totals, on=["patient", "state_norm", "layer_norm"], how="left")
    d["fraction_of_all_cells"] = d["n_cells"] / d["n_cells_total"]
    return d


# =============================================================================
# Statistics
# =============================================================================

def pairwise_mannwhitney(df: pd.DataFrame, metric: str, group_col: str, groups: List[str], min_n: int = 3) -> pd.DataFrame:
    rows = []
    if metric not in df.columns or group_col not in df.columns:
        return pd.DataFrame()
    d = df.copy()
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
                "mean_a": float(np.nanmean(a)),
                "mean_b": float(np.nanmean(b)),
                "median_a": float(np.nanmedian(a)),
                "median_b": float(np.nanmedian(b)),
                "effect_cliffs_delta": cliffs_delta(a, b),
                "test": "MannWhitneyU",
                "stat": float(stat) if np.isfinite(stat) else np.nan,
                "p": float(p) if np.isfinite(p) else np.nan,
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_fdr"] = fdr_bh(out["p"].values)
    return out


def compare_states_within_group(df: pd.DataFrame, metric: str, group_col: Optional[str] = None,
                                min_n: int = 3) -> pd.DataFrame:
    """
    Compare vivo vs vitro. If group_col is provided, do it within each subgroup.
    """
    rows = []
    if metric not in df.columns:
        return pd.DataFrame()
    d = df.copy()
    d[metric] = pd.to_numeric(d[metric], errors="coerce")

    if group_col is None:
        group_iter = [((), d)]
        strat_cols = []
    else:
        strat_cols = [group_col]
        group_iter = d.groupby(group_col, dropna=False)

    for keys, sub in group_iter:
        key_dict = {}
        if group_col is not None:
            key_dict[group_col] = keys if not isinstance(keys, tuple) else keys[0]

        a = sub[sub["state_norm"] == "vivo"][metric].dropna().astype(float)
        b = sub[sub["state_norm"] == "vitro"][metric].dropna().astype(float)
        if len(a) < min_n or len(b) < min_n:
            continue
        try:
            stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        except Exception:
            stat, p = np.nan, np.nan
        rows.append({
            **key_dict,
            "metric": metric,
            "comparison": "vivo_vs_vitro",
            "n_vivo": int(len(a)),
            "n_vitro": int(len(b)),
            "mean_vivo": float(np.nanmean(a)),
            "mean_vitro": float(np.nanmean(b)),
            "median_vivo": float(np.nanmedian(a)),
            "median_vitro": float(np.nanmedian(b)),
            "effect_cliffs_delta": cliffs_delta(a, b),
            "test": "MannWhitneyU",
            "stat": float(stat) if np.isfinite(stat) else np.nan,
            "p": float(p) if np.isfinite(p) else np.nan,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_fdr"] = fdr_bh(out["p"].values)
    return out


def correlation_table(df: pd.DataFrame, x_col: str, y_col: str,
                      stratify_cols: List[str], min_n: int = 5) -> pd.DataFrame:
    rows = []
    if x_col not in df.columns or y_col not in df.columns:
        return pd.DataFrame()

    use = df.copy()
    use[x_col] = pd.to_numeric(use[x_col], errors="coerce")
    use[y_col] = pd.to_numeric(use[y_col], errors="coerce")

    if stratify_cols:
        groups = use.groupby(stratify_cols, dropna=False)
    else:
        groups = [((), use)]

    for keys, sub in groups:
        key_dict = {}
        if stratify_cols:
            if not isinstance(keys, tuple):
                keys = (keys,)
            key_dict = dict(zip(stratify_cols, keys))

        sub = sub[np.isfinite(sub[x_col]) & np.isfinite(sub[y_col])]
        if len(sub) < min_n:
            continue
        try:
            rho, p = stats.spearmanr(sub[x_col], sub[y_col])
        except Exception:
            rho, p = np.nan, np.nan
        rows.append({
            **key_dict,
            "x": x_col,
            "y": y_col,
            "n_pairs": int(len(sub)),
            "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
            "spearman_p": float(p) if np.isfinite(p) else np.nan,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["spearman_q_fdr"] = fdr_bh(out["spearman_p"].values)
    return out


def build_correlation_grid(
    hist_summary: pd.DataFrame,
    ephys_summary: pd.DataFrame,
    hist_metrics: List[str],
    ephys_metrics: List[str],
    min_n: int = 5,
) -> pd.DataFrame:
    """
    Correlate patient-aggregated histology vs patient-aggregated ephys.
    Match on patient, state, layer.
    """
    if hist_summary.empty or ephys_summary.empty:
        return pd.DataFrame()

    # melt ephys summary into long form
    id_cols = [c for c in ["patient", "state_norm", "layer_norm", "cellclass_ei"] if c in ephys_summary.columns]
    long_rows = []
    for metric in ephys_metrics:
        col = f"mean_{metric}"
        if col not in ephys_summary.columns:
            continue
        tmp = ephys_summary[id_cols + [col]].copy()
        tmp = tmp.rename(columns={col: "ephys_value"})
        tmp["ephys_metric"] = metric
        long_rows.append(tmp)
    if not long_rows:
        return pd.DataFrame()
    e_long = pd.concat(long_rows, ignore_index=True)

    # hist long
    h = hist_summary.copy()
    if "staining_norm" not in h.columns:
        return pd.DataFrame()

    h["density (cell/mm^2)"] = pd.to_numeric(h.get("density (cell/mm^2)"), errors="coerce")
    h["coverage (%)"] = pd.to_numeric(h.get("coverage (%)"), errors="coerce")
    h["pv_neun_density_ratio"] = pd.to_numeric(h.get("pv_neun_density_ratio"), errors="coerce")

    hist_long_frames = []
    if "PV" in set(h["staining_norm"]):
        for metric in ["density (cell/mm^2)", "coverage (%)"]:
            if metric in h.columns:
                t = h[h["staining_norm"] == "PV"][id_cols[:3] if len(id_cols) >= 3 else ["patient", "state_norm", "layer_norm"]].copy()
                t["histology_metric"] = f"PV_{metric}"
                t["histology_value"] = pd.to_numeric(h[h["staining_norm"] == "PV"][metric], errors="coerce").values
                hist_long_frames.append(t)
    if "NEUN" in set(h["staining_norm"]):
        for metric in ["density (cell/mm^2)"]:
            if metric in h.columns:
                t = h[h["staining_norm"] == "NEUN"][id_cols[:3] if len(id_cols) >= 3 else ["patient", "state_norm", "layer_norm"]].copy()
                t["histology_metric"] = f"NEUN_{metric}"
                t["histology_value"] = pd.to_numeric(h[h["staining_norm"] == "NEUN"][metric], errors="coerce").values
                hist_long_frames.append(t)

    # ratio table
    if "pv_neun_density_ratio" in h.columns:
        t = h[id_cols[:3] if len(id_cols) >= 3 else ["patient", "state_norm", "layer_norm"]].copy()
        t["histology_metric"] = "pv_neun_density_ratio"
        t["histology_value"] = pd.to_numeric(h["pv_neun_density_ratio"], errors="coerce")
        hist_long_frames.append(t)

    if not hist_long_frames:
        return pd.DataFrame()

    h_long = pd.concat(hist_long_frames, ignore_index=True)
    h_long = h_long.dropna(subset=["patient", "state_norm", "layer_norm", "histology_value"])
    e_long = e_long.dropna(subset=["patient", "state_norm", "layer_norm", "ephys_value"])

    merged = h_long.merge(e_long, on=["patient", "state_norm", "layer_norm"], how="inner")
    if merged.empty:
        return pd.DataFrame()

    rows = []
    for hmet in merged["histology_metric"].dropna().unique():
        for emet in merged["ephys_metric"].dropna().unique():
            for state in [s for s in STATE_ORDER if s in set(merged["state_norm"])]:
                for layer in [l for l in LAYER_ORDER if l in set(merged["layer_norm"])]:
                    sub = merged[
                        (merged["histology_metric"] == hmet) &
                        (merged["ephys_metric"] == emet) &
                        (merged["state_norm"] == state) &
                        (merged["layer_norm"] == layer)
                    ].copy()
                    if len(sub) < min_n:
                        continue
                    try:
                        rho, p = stats.spearmanr(sub["histology_value"], sub["ephys_value"])
                    except Exception:
                        rho, p = np.nan, np.nan
                    rows.append({
                        "state_norm": state,
                        "layer_norm": layer,
                        "histology_metric": hmet,
                        "ephys_metric": emet,
                        "n_pairs": int(len(sub)),
                        "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                        "spearman_p": float(p) if np.isfinite(p) else np.nan,
                    })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["spearman_q_fdr"] = fdr_bh(out["spearman_p"].values)
    return out


# =============================================================================
# Plotting
# =============================================================================

COLORS_STATE = {
    "vivo": "#111111",
    "vitro": "#A8A8A8",
}
COLORS_LAYER = {
    "supra": "#C85C64",
    "infra": "#54428E",
}
COLORS_STAIN = {
    "PV": "#B23A48",
    "NEUN": "#2F6FB1",
}
COLORS_CELLCLASS = {
    "all": "#666666",
    "Exc": "#D1495B",
    "Inh": "#277DA1",
    "FS": "#2A9D8F",
}


def set_theme():
    sns.set_theme(style="white", context="paper", font_scale=1.0)
    plt.rcParams.update({
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.grid": False,
        "font.family": "DejaVu Sans",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def add_sig_bracket(ax, x1, x2, y, text, dy=0.02, fontsize=9):
    ax.plot([x1, x1, x2, x2], [y, y + dy, y + dy, y], color="black", lw=1.0, clip_on=False)
    ax.text((x1 + x2) / 2, y + dy, text, ha="center", va="bottom", fontsize=fontsize)


def annotate_pairwise(ax, df: pd.DataFrame, x: str, y: str, order: List[str],
                      hue: Optional[str] = None, hue_order: Optional[List[str]] = None,
                      min_n: int = 3):
    if df is None or df.empty or x not in df.columns or y not in df.columns:
        return
    vals = pd.to_numeric(df[y], errors="coerce")
    vals = vals[np.isfinite(vals)]
    if vals.empty:
        return
    ymin, ymax = float(vals.min()), float(vals.max())
    span = ymax - ymin if ymax > ymin else max(abs(ymax), 1.0)

    if hue is None:
        table = pairwise_mannwhitney(df, y, x, order, min_n=min_n)
        if table.empty:
            return
        base_y = ymax + 0.08 * span
        step = 0.07 * span
        for k, (a, b) in enumerate(combinations(order, 2)):
            row = table[((table["group_a"] == a) & (table["group_b"] == b)) |
                        ((table["group_a"] == b) & (table["group_b"] == a))]
            if row.empty:
                continue
            p = float(row.iloc[0]["q_fdr"]) if "q_fdr" in row.columns else float(row.iloc[0]["p"])
            txt = stars(p)
            if txt == "n.s.":
                continue
            add_sig_bracket(ax, order.index(a), order.index(b), base_y + k * step, f"{txt} q={p:.2g}")
    else:
        # compare hue levels within each x group
        if hue_order is None:
            hue_order = [h for h in df[hue].dropna().unique().tolist()]
        width = 0.72
        offsets = np.linspace(-width / 4, width / 4, len(hue_order)) if len(hue_order) > 1 else np.array([0.0])
        for xi, xv in enumerate(order):
            sub = df[df[x] == xv]
            if sub.empty:
                continue
            present = [h for h in hue_order if h in set(sub[hue].dropna().tolist())]
            if len(present) < 2:
                continue
            table = pairwise_mannwhitney(sub, y, hue, present, min_n=min_n)
            if table.empty:
                continue
            local_base = float(pd.to_numeric(sub[y], errors="coerce").max()) + 0.08 * span
            step = 0.07 * span
            for k, (a, b) in enumerate(combinations(present, 2)):
                row = table[((table["group_a"] == a) & (table["group_b"] == b)) |
                            ((table["group_a"] == b) & (table["group_b"] == a))]
                if row.empty:
                    continue
                p = float(row.iloc[0]["q_fdr"]) if "q_fdr" in row.columns else float(row.iloc[0]["p"])
                txt = stars(p)
                if txt == "n.s.":
                    continue
                xa = xi + offsets[present.index(a)]
                xb = xi + offsets[present.index(b)]
                add_sig_bracket(ax, xa, xb, local_base + k * step, f"{txt} q={p:.2g}")


def boxstrip_plot(df: pd.DataFrame, metric: str, x: str, hue: Optional[str], out_png: str,
                  title: str, order: Optional[List[str]] = None, hue_order: Optional[List[str]] = None):
    set_theme()
    d = df.copy()
    if metric not in d.columns or x not in d.columns:
        return
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    keep = [x, metric] + ([hue] if hue else [])
    d = d.dropna(subset=keep)
    d = d[np.isfinite(d[metric])]
    if d.empty:
        return

    if order is not None:
        order = [o for o in order if o in set(d[x])]
        if not order:
            return

    if hue is not None and hue_order is not None:
        hue_order = [h for h in hue_order if h in set(d[hue])]
        if not hue_order:
            hue_order = None

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    pal = None
    if hue == "state_norm":
        pal = COLORS_STATE
    elif hue == "layer_norm":
        pal = COLORS_LAYER
    elif hue == "staining_norm":
        pal = COLORS_STAIN
    elif hue == "cellclass_ei":
        pal = COLORS_CELLCLASS

    sns.boxplot(data=d, x=x, y=metric, hue=hue, order=order, hue_order=hue_order,
                palette=pal, showfliers=False, width=0.65, ax=ax)
    sns.stripplot(data=d, x=x, y=metric, hue=hue, order=order, hue_order=hue_order,
                  dodge=True if hue else False, palette=pal, alpha=0.55, size=3.5,
                  linewidth=0.25, edgecolor="black", ax=ax)

    if hue:
        handles, labels = ax.get_legend_handles_labels()
        uniq = {}
        for h, lab in zip(handles, labels):
            if lab not in uniq:
                uniq[lab] = h
        ax.legend(uniq.values(), uniq.keys(), title=hue, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    else:
        if ax.legend_:
            ax.legend_.remove()

    ax.set_title(title)
    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel(metric.replace("_", " ").title())

    annotate_pairwise(ax, d, x=x, y=metric, order=order or sorted(d[x].dropna().unique()),
                      hue=hue, hue_order=hue_order, min_n=3)

    fig.tight_layout()
    save_figure(fig, out_png)
    plt.close(fig)


def scatter_corr_plot(df: pd.DataFrame, x: str, y: str, hue: Optional[str],
                      out_png: str, title: str):
    set_theme()
    d = df.copy()
    d[x] = pd.to_numeric(d[x], errors="coerce")
    d[y] = pd.to_numeric(d[y], errors="coerce")
    d = d[np.isfinite(d[x]) & np.isfinite(d[y])]
    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    if hue and hue in d.columns:
        sns.scatterplot(data=d, x=x, y=y, hue=hue, ax=ax, s=55, edgecolor="black", linewidth=0.25)
    else:
        sns.scatterplot(data=d, x=x, y=y, ax=ax, s=55, edgecolor="black", linewidth=0.25)

    sns.regplot(data=d, x=x, y=y, scatter=False, ax=ax, color="#222222", ci=95)
    rho, p = stats.spearmanr(d[x], d[y])
    ax.set_title(f"{title}\nSpearman rho={rho:.2f}, p={p:.2g}, n={len(d)}")
    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel(y.replace("_", " ").title())
    if hue and hue in d.columns:
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    fig.tight_layout()
    save_figure(fig, out_png)
    plt.close(fig)


# =============================================================================
# Main analysis routines
# =============================================================================

def run_histology_stats(hist: pd.DataFrame, out_dir: str) -> Dict[str, pd.DataFrame]:
    os.makedirs(out_dir, exist_ok=True)
    out_tables = {}

    # histology QC / summary
    qc = (
        hist.groupby(["patient", "state_norm", "layer_norm", "staining_norm"], dropna=False)
            .agg(
                n_roi=("staining_norm", "size"),
                density_mean=("density (cell/mm^2)", "mean"),
                density_median=("density (cell/mm^2)", "median"),
                coverage_mean=("coverage (%)", "mean"),
                coverage_median=("coverage (%)", "median"),
            )
            .reset_index()
    )
    qc.to_csv(os.path.join(out_dir, "histology_qc_patient_layer_summary.csv"), index=False)
    out_tables["histology_qc"] = qc

    # patient-level aggregation for inference
    hist_agg = patient_level_histology_summary(hist)
    if not hist_agg.empty:
        hist_agg.to_csv(os.path.join(out_dir, "histology_patient_level_summary.csv"), index=False)
    out_tables["histology_agg"] = hist_agg

    # ratio table
    ratio_tbl = make_ratio_table(hist)
    if not ratio_tbl.empty:
        ratio_tbl.to_csv(os.path.join(out_dir, "histology_ratio_table.csv"), index=False)
    out_tables["histology_ratio"] = ratio_tbl

    # state comparisons within each stain
    comp_rows = []
    for stain in [s for s in STAIN_ORDER if s in set(hist["staining_norm"])]:
        sub = hist[hist["staining_norm"] == stain].copy()
        for metric in ["density (cell/mm^2)", "coverage (%)"]:
            if metric in sub.columns:
                t = compare_states_within_group(sub, metric=metric, group_col="layer_norm", min_n=3)
                if not t.empty:
                    t["staining_norm"] = stain
                    comp_rows.append(t)
                # overall state comparison
                t2 = compare_states_within_group(sub, metric=metric, group_col=None, min_n=3)
                if not t2.empty:
                    t2["staining_norm"] = stain
                    comp_rows.append(t2)

    comp = pd.concat(comp_rows, ignore_index=True) if comp_rows else pd.DataFrame()
    if not comp.empty:
        comp.to_csv(os.path.join(out_dir, "histology_state_tests.csv"), index=False)
    out_tables["histology_comparisons"] = comp

    # plots
    plot_dir = os.path.join(out_dir, "plots_histology")
    os.makedirs(plot_dir, exist_ok=True)

    for metric in ["density (cell/mm^2)", "coverage (%)"]:
        if metric not in hist.columns:
            continue
        # state by layer, separate stain panels
        for stain in [s for s in STAIN_ORDER if s in set(hist["staining_norm"])]:
            sub = hist[hist["staining_norm"] == stain].copy()
            if sub.empty:
                continue
            boxstrip_plot(
                df=sub,
                metric=metric,
                x="state_norm",
                hue="layer_norm",
                out_png=os.path.join(plot_dir, f"histology_{stain}_{metric.replace(' ', '_').replace('/', '_')}_state_by_layer.png"),
                title=f"{stain}: {metric} by state and layer",
                order=STATE_ORDER,
                hue_order=[l for l in LAYER_ORDER if l in set(sub["layer_norm"])],
            )

            boxstrip_plot(
                df=sub,
                metric=metric,
                x="layer_norm",
                hue="state_norm",
                out_png=os.path.join(plot_dir, f"histology_{stain}_{metric.replace(' ', '_').replace('/', '_')}_layer_by_state.png"),
                title=f"{stain}: {metric} by layer and state",
                order=LAYER_ORDER,
                hue_order=STATE_ORDER,
            )

    # ratio plots
    if not ratio_tbl.empty and "pv_neun_density_ratio" in ratio_tbl.columns:
        boxstrip_plot(
            df=ratio_tbl,
            metric="pv_neun_density_ratio",
            x="state_norm",
            hue="layer_norm",
            out_png=os.path.join(plot_dir, "histology_pv_neun_density_ratio_state_by_layer.png"),
            title="PV / NeuN density ratio by state and layer",
            order=STATE_ORDER,
            hue_order=LAYER_ORDER,
        )
        boxstrip_plot(
            df=ratio_tbl,
            metric="pv_neun_density_ratio",
            x="layer_norm",
            hue="state_norm",
            out_png=os.path.join(plot_dir, "histology_pv_neun_density_ratio_layer_by_state.png"),
            title="PV / NeuN density ratio by layer and state",
            order=LAYER_ORDER,
            hue_order=STATE_ORDER,
        )

    return out_tables


def run_ephys_summary(ephys: pd.DataFrame, out_dir: str) -> Dict[str, pd.DataFrame]:
    os.makedirs(out_dir, exist_ok=True)
    out_tables = {}

    if ephys is None or ephys.empty:
        return out_tables

    # summary per patient/state/layer/cellclass
    e_agg = patient_level_ephys_summary(ephys)
    if not e_agg.empty:
        e_agg.to_csv(os.path.join(out_dir, "ephys_patient_level_summary.csv"), index=False)
    out_tables["ephys_agg"] = e_agg

    # simple QC by class
    qc = (
        ephys.groupby(["patient", "state_norm", "layer_norm", "cellclass_ei"], dropna=False)
             .size()
             .reset_index(name="n_rows")
    )
    qc.to_csv(os.path.join(out_dir, "ephys_qc_patient_layer_class.csv"), index=False)
    out_tables["ephys_qc"] = qc

    return out_tables


def run_correlation_analysis(hist: pd.DataFrame, ephys: pd.DataFrame, out_dir: str) -> Dict[str, pd.DataFrame]:
    os.makedirs(out_dir, exist_ok=True)
    out_tables = {}

    hist_summary = patient_level_histology_summary(hist)
    ephys_summary = patient_level_ephys_summary(ephys)

    if hist_summary.empty or ephys_summary.empty:
        return out_tables

    # correlations on patient-aggregated data
    # create histology long
    hist_long_rows = []
    for stain in [s for s in STAIN_ORDER if s in set(hist_summary["staining_norm"])]:
        sub = hist_summary[hist_summary["staining_norm"] == stain].copy()
        if sub.empty:
            continue
        for metric in ["density (cell/mm^2)", "coverage (%)"]:
            if metric in sub.columns:
                tmp = sub[["patient", "state_norm", "layer_norm"]].copy()
                tmp["histology_metric"] = f"{stain}_{metric}"
                tmp["histology_value"] = pd.to_numeric(sub[metric], errors="coerce").values
                hist_long_rows.append(tmp)

    ratio_tbl = make_ratio_table(hist)
    if not ratio_tbl.empty and "pv_neun_density_ratio" in ratio_tbl.columns:
        tmp = ratio_tbl[["patient", "state_norm", "layer_norm"]].copy()
        tmp["histology_metric"] = "pv_neun_density_ratio"
        tmp["histology_value"] = pd.to_numeric(ratio_tbl["pv_neun_density_ratio"], errors="coerce")
        hist_long_rows.append(tmp)

    hist_long = pd.concat(hist_long_rows, ignore_index=True) if hist_long_rows else pd.DataFrame()
    if hist_long.empty:
        return out_tables

    # ephys long by cellclass
    ephys_metric_cols = [c for c in ephys_summary.columns if c.startswith("mean_")]
    ephys_long_rows = []
    for col in ephys_metric_cols:
        base_metric = col.replace("mean_", "")
        if base_metric not in EPHYS_METRICS:
            continue
        tmp = ephys_summary[["patient", "state_norm", "layer_norm", "cellclass_ei"]].copy()
        tmp["ephys_metric"] = base_metric
        tmp["ephys_value"] = pd.to_numeric(ephys_summary[col], errors="coerce").values
        ephys_long_rows.append(tmp)

    if "fraction_of_all_cells" in ephys_summary.columns:
        frac_map = {
            "Exc": "Exc_fraction_of_all_cells",
            "Inh": "Inh_fraction_of_all_cells",
            "FS": "FS_fraction_of_all_cells",
            "all": "all_fraction_of_all_cells",
        }
        for class_name, metric_name in frac_map.items():
            tmp = ephys_summary[ephys_summary["cellclass_ei"] == class_name][
                ["patient", "state_norm", "layer_norm", "cellclass_ei", "fraction_of_all_cells"]
            ].copy()
            if tmp.empty:
                continue
            tmp["ephys_metric"] = metric_name
            tmp["ephys_value"] = pd.to_numeric(tmp["fraction_of_all_cells"], errors="coerce")
            ephys_long_rows.append(tmp)

    ephys_long = pd.concat(ephys_long_rows, ignore_index=True) if ephys_long_rows else pd.DataFrame()
    if ephys_long.empty:
        return out_tables

    merged = hist_long.merge(ephys_long, on=["patient", "state_norm", "layer_norm"], how="inner")
    merged = merged[np.isfinite(merged["histology_value"]) & np.isfinite(merged["ephys_value"])]
    if merged.empty:
        return out_tables

    rows = []
    for hmet in sorted(merged["histology_metric"].dropna().unique()):
        for emet in sorted(merged["ephys_metric"].dropna().unique()):
            for state in [s for s in STATE_ORDER if s in set(merged["state_norm"])]:
                for layer in [l for l in LAYER_ORDER if l in set(merged["layer_norm"])]:
                    for cellclass in [c for c in CELLCLASS_ORDER if c in set(merged["cellclass_ei"])]:
                        sub = merged[
                            (merged["histology_metric"] == hmet) &
                            (merged["ephys_metric"] == emet) &
                            (merged["state_norm"] == state) &
                            (merged["layer_norm"] == layer) &
                            (merged["cellclass_ei"] == cellclass)
                        ].copy()
                        if len(sub) < 5:
                            continue
                        try:
                            rho, p = stats.spearmanr(sub["histology_value"], sub["ephys_value"])
                        except Exception:
                            rho, p = np.nan, np.nan
                        rows.append({
                            "state_norm": state,
                            "layer_norm": layer,
                            "cellclass_ei": cellclass,
                            "histology_metric": hmet,
                            "ephys_metric": emet,
                            "n_pairs": int(len(sub)),
                            "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                            "spearman_p": float(p) if np.isfinite(p) else np.nan,
                        })

    corr = pd.DataFrame(rows)
    if not corr.empty:
        corr["spearman_q_fdr"] = fdr_bh(corr["spearman_p"].values)
        corr = corr.sort_values(["spearman_q_fdr", "spearman_p"], na_position="last")
        corr.to_csv(os.path.join(out_dir, "histology_ephys_correlations.csv"), index=False)

        sig = corr[corr["spearman_q_fdr"] < 0.05].copy()
        sig.to_csv(os.path.join(out_dir, "histology_ephys_correlations_significant.csv"), index=False)

        out_tables["correlations"] = corr
        out_tables["correlations_sig"] = sig

    return out_tables


def plot_correlation_grid(corr: pd.DataFrame, out_dir: str):
    if corr is None or corr.empty:
        return
    plot_dir = os.path.join(out_dir, "plots_correlations")
    os.makedirs(plot_dir, exist_ok=True)

    # one plot per significant / top correlation row
    top = corr.sort_values(["spearman_q_fdr", "spearman_p"], na_position="last").head(36)
    for _, r in top.iterrows():
        # this script only has summary tables, so scatter plots of full raw data are not generated here.
        # For a true scatter plot you’d use the merged table before summary.
        pass

    # save matrix-like summary as heatmaps
    for cellclass in [c for c in CELLCLASS_ORDER if c in set(corr["cellclass_ei"])]:
        for state in [s for s in STATE_ORDER if s in set(corr["state_norm"])]:
            for layer in [l for l in LAYER_ORDER if l in set(corr["layer_norm"])]:
                sub = corr[
                    (corr["cellclass_ei"] == cellclass) &
                    (corr["state_norm"] == state) &
                    (corr["layer_norm"] == layer)
                ].copy()
                if sub.empty:
                    continue
                mat = sub.pivot_table(index="ephys_metric", columns="histology_metric", values="spearman_rho", aggfunc="mean")
                if mat.empty:
                    continue
                fig, ax = plt.subplots(figsize=(1.2 * len(mat.columns) + 5, 0.45 * len(mat.index) + 3))
                sns.heatmap(mat, vmin=-1, vmax=1, center=0, cmap="vlag", annot=True, fmt=".3f",
                            linewidths=0.5, linecolor="white", cbar_kws={"label": "Spearman rho"}, ax=ax)
                ax.set_title(f"Histology–ephys correlation matrix\n{cellclass}, {state}, {layer}")
                ax.set_xlabel("histology metric")
                ax.set_ylabel("ephys metric")
                fig.tight_layout()
                save_figure(fig, os.path.join(plot_dir, f"corr_matrix__{cellclass}__{state}__{layer}.png"))
                plt.close(fig)


def plot_mixedlm_grid(mixed: pd.DataFrame, out_dir: str):
    if mixed is None or mixed.empty:
        return
    plot_dir = os.path.join(out_dir, "plots_mixedlm")
    os.makedirs(plot_dir, exist_ok=True)

    for state in [s for s in STATE_ORDER if s in set(mixed["state_norm"])]:
        for cellclass in [c for c in CELLCLASS_ORDER if c in set(mixed["cellclass_ei"])]:
            sub = mixed[(mixed["state_norm"] == state) & (mixed["cellclass_ei"] == cellclass)].copy()
            if sub.empty:
                continue

            mat = sub.pivot_table(index="ephys_metric", columns="histology_metric", values="beta_std", aggfunc="mean")
            qmat = sub.pivot_table(index="ephys_metric", columns="histology_metric", values="q_value", aggfunc="mean")
            nmat = sub.pivot_table(index="ephys_metric", columns="histology_metric", values="n_units", aggfunc="mean")
            if mat.empty:
                continue

            hist_order = [h for h in ["NEUN_density", "PV_density", "PV_coverage", "pv_neun_density_ratio"] if h in mat.columns]
            ephys_order = [m for m in EPHYS_METRICS if m in mat.index]
            mat = mat.reindex(index=ephys_order, columns=hist_order)
            qmat = qmat.reindex(index=ephys_order, columns=hist_order)
            nmat = nmat.reindex(index=ephys_order, columns=hist_order)
            mat = mat.rename(index=EPHYS_LABELS, columns=HISTOLOGY_LABELS)
            qmat = qmat.rename(index=EPHYS_LABELS, columns=HISTOLOGY_LABELS)
            nmat = nmat.rename(index=EPHYS_LABELS, columns=HISTOLOGY_LABELS)

            annot = pd.DataFrame("", index=mat.index, columns=mat.columns)
            for i in mat.index:
                for j in mat.columns:
                    beta = mat.loc[i, j]
                    q = qmat.loc[i, j]
                    n = nmat.loc[i, j]
                    if pd.notna(beta):
                        txt = f"{beta:.2f}"
                        if pd.notna(q) and q < 0.05:
                            txt = "*" + txt
                        annot.loc[i, j] = txt

            fig, ax = plt.subplots(figsize=(1.7 * len(mat.columns) + 3.4, 0.72 * len(mat.index) + 2.6))
            sns.heatmap(
                mat,
                vmin=-1, vmax=1, center=0, cmap="vlag",
                linewidths=0.6, linecolor="white",
                annot=annot.values,
                fmt="",
                cbar_kws={"label": "Std. beta"},
                ax=ax,
            )
            ax.tick_params(axis="both", labelsize=10)
            ax.set_title(f"{STATE_LABELS.get(state, state)} | {pretty_label(cellclass)}", fontsize=13, pad=12)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.collections[0].colorbar.ax.tick_params(labelsize=9)
            fig.tight_layout()
            save_figure(fig, os.path.join(plot_dir, f"mixedlm_matrix__{state}__{cellclass}.png"))
            plt.close(fig)


def plot_targeted_patient_pairs(hist: pd.DataFrame, out_dir: str, patients: Optional[List[str]] = None):
    """
    Publication-style paired plots for patients with both vivo and vitro histology.
    """
    if hist is None or hist.empty:
        return

    if patients is None:
        patients = ["epi29", "epi45"]

    hist_long = build_histology_long_table(hist)
    if hist_long.empty:
        return

    hist_long = hist_long[hist_long["patient"].isin(patients)].copy()
    if hist_long.empty:
        return

    # Keep only the requested histology readouts.
    hist_long["histology_metric"] = hist_long["histology_metric"].replace({
        "PV_density": "PV_density",
        "NEUN_density": "NEUN_density",
        "PV_coverage": "PV_coverage",
        "pv_neun_density_ratio": "pv_neun_density_ratio",
    })

    pair_metrics = ["NEUN_density", "PV_density", "PV_coverage", "pv_neun_density_ratio"]
    hist_long = hist_long[hist_long["histology_metric"].isin(pair_metrics)].copy()
    if hist_long.empty:
        return

    out_dir = os.path.join(out_dir, "plots_targeted_pairs")
    os.makedirs(out_dir, exist_ok=True)

    # export the paired values
    hist_long.to_csv(os.path.join(out_dir, "paired_epi29_epi45_histology_values.csv"), index=False)

    layers = [l for l in LAYER_ORDER if l in set(hist_long["layer_norm"])]
    if not layers:
        return

    patient_colors = {
        "epi29": "#111111",
        "epi45": "#3A3A3A",
    }

    fig, axes = plt.subplots(
        nrows=len(layers),
        ncols=len(pair_metrics),
        figsize=(4.1 * len(pair_metrics), 3.2 * len(layers)),
        sharex=True,
        sharey=False,
    )
    if len(layers) == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, layer in enumerate(layers):
        layer_df = hist_long[hist_long["layer_norm"] == layer].copy()
        for c, metric in enumerate(pair_metrics):
            ax = axes[r, c]
            sub = layer_df[layer_df["histology_metric"] == metric].copy()
            if sub.empty:
                ax.axis("off")
                continue

            x_positions = {s: i for i, s in enumerate(STATE_ORDER)}
            for patient in patients:
                psub = sub[sub["patient"] == patient].copy()
                vals = []
                xs = []
                for state in STATE_ORDER:
                    state_vals = psub[psub["state_norm"] == state]["histology_value"].dropna().values
                    if len(state_vals):
                        xs.append(x_positions[state])
                        vals.append(float(np.nanmean(state_vals)))
                if len(xs) < 2:
                    continue
                ax.plot(xs, vals, color=patient_colors.get(patient, "#444444"), lw=1.6, alpha=0.9, zorder=2)
                for state, x in x_positions.items():
                    state_vals = psub[psub["state_norm"] == state]["histology_value"].dropna().values
                    if not len(state_vals):
                        continue
                    ax.scatter(
                        x,
                        float(np.nanmean(state_vals)),
                        s=42,
                        facecolor=COLORS_STATE[state],
                        edgecolor=patient_colors.get(patient, "#444444"),
                        linewidth=0.8,
                        zorder=3,
                    )

            if r == 0:
                ax.set_title(pretty_label(metric), fontsize=13, pad=10)
            if c == 0:
                ax.set_ylabel(f"{pretty_label(layer)}\nvalue", fontsize=11)
            else:
                ax.set_ylabel("")

            ax.set_xticks([0, 1])
            ax.set_xticklabels([STATE_LABELS["vivo"], STATE_LABELS["vitro"]], fontsize=11)
            ax.tick_params(axis="y", labelsize=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(False)

    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS_STATE["vivo"], markeredgecolor="#111111", markersize=7, label=STATE_LABELS["vivo"]),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS_STATE["vitro"], markeredgecolor="#111111", markersize=7, label=STATE_LABELS["vitro"]),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02), fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig, os.path.join(out_dir, "paired_epi29_epi45_histology_summary.png"))
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def run_pipeline(paths: Paths):
    os.makedirs(paths.out_dir, exist_ok=True)
    set_theme()

    hist = load_histology(paths.histology_csv)
    if hist.empty:
        raise ValueError("Histology table is empty.")

    hist["staining_norm"] = hist["staining_norm"].replace({"OTHER": "other"})
    if "density (cell/mm^2)" in hist.columns:
        hist["density (cell/mm^2)"] = pd.to_numeric(hist["density (cell/mm^2)"], errors="coerce")
    if "coverage (%)" in hist.columns:
        hist["coverage (%)"] = pd.to_numeric(hist["coverage (%)"], errors="coerce")

    # qc histology input
    qc_dir = os.path.join(paths.out_dir, "01_qc")
    os.makedirs(qc_dir, exist_ok=True)
    hist.to_csv(os.path.join(qc_dir, "histology_cleaned.csv"), index=False)

    hist_tables = run_histology_stats(hist, os.path.join(paths.out_dir, "02_histology"))

    ephys = pd.DataFrame()
    if paths.ephys_csv and os.path.exists(paths.ephys_csv):
        ephys = load_ephys(paths.ephys_csv)
        ephys.to_csv(os.path.join(qc_dir, "ephys_cleaned.csv"), index=False)
        ephys_tables = run_ephys_summary(ephys, os.path.join(paths.out_dir, "03_ephys"))
    else:
        ephys_tables = {}

    corr_tables = {}
    if not ephys.empty:
        corr_tables = run_correlation_analysis(hist, ephys, os.path.join(paths.out_dir, "04_correlations"))

    mixed_tables = {}
    if not ephys.empty:
        mixed_tables = run_unit_level_mixed_models(hist, ephys, os.path.join(paths.out_dir, "05_mixedlm"))
        if "mixedlm_all" in mixed_tables and not mixed_tables["mixedlm_all"].empty:
            plot_mixedlm_grid(mixed_tables["mixedlm_all"], os.path.join(paths.out_dir, "05_mixedlm"))
    plot_targeted_patient_pairs(hist, os.path.join(paths.out_dir, "06_targeted_pairs"))

    # optional: create a human-readable summary
    summary_path = os.path.join(paths.out_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Histology + ephys analysis summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Histology rows: {len(hist)}\n")
        f.write(f"Patients: {hist['patient'].nunique() if 'patient' in hist.columns else 'NA'}\n")
        f.write(f"States: {', '.join(sorted(hist['state_norm'].dropna().unique()))}\n")
        f.write(f"Layers: {', '.join(sorted(hist['layer_norm'].dropna().unique()))}\n")
        f.write(f"Stains: {', '.join(sorted(hist['staining_norm'].dropna().unique()))}\n\n")
        f.write("Histology metrics used: density, coverage, PV/NeuN ratio\n")
        f.write("Ephys metrics used: firing/ISI/PLV/coupling plus class fractions\n\n")
        f.write(f"Mixed-effects tables: {len([k for k in mixed_tables if k.startswith('mixedlm__')])}\n\n")

        if "histology_comparisons" in hist_tables and not hist_tables["histology_comparisons"].empty:
            sig = hist_tables["histology_comparisons"].copy()
            if "q_fdr" in sig.columns:
                sig = sig[sig["q_fdr"] < 0.05]
            f.write("Significant histology comparisons:\n")
            if sig.empty:
                f.write("  none\n")
            else:
                for _, r in sig.head(25).iterrows():
                    f.write(f"  - {r.get('staining_norm', '')} {r.get('metric', '')}: {r.get('comparison', 'state comparison')} q={r.get('q_fdr', np.nan):.3g}\n")
            f.write("\n")

        if "correlations" in corr_tables and not corr_tables["correlations"].empty:
            sig = corr_tables["correlations"]
            sig = sig[sig["spearman_q_fdr"] < 0.05] if "spearman_q_fdr" in sig.columns else sig
            f.write("Significant histology–ephys correlations:\n")
            if sig.empty:
                f.write("  none\n")
            else:
                for _, r in sig.head(40).iterrows():
                    f.write(
                        f"  - {r['histology_metric']} vs {r['ephys_metric']} "
                        f"({r['cellclass_ei']}, {r['state_norm']}, {r['layer_norm']}): "
                        f"rho={r['spearman_rho']:.2f}, q={r['spearman_q_fdr']:.3g}\n"
                    )
        if "mixedlm_all" in mixed_tables and not mixed_tables["mixedlm_all"].empty:
            sig = mixed_tables["mixedlm_all"]
            sig = sig[sig["q_value"] < 0.05] if "q_value" in sig.columns else sig
            f.write("\nSignificant mixed-effects associations:\n")
            if sig.empty:
                f.write("  none\n")
            else:
                for _, r in sig.head(40).iterrows():
                    f.write(
                        f"  - {r['histology_metric']} vs {r['ephys_metric']} "
                        f"({r['cellclass_ei']}, {r['state_norm']}): "
                        f"beta={r['beta_std']:.3f}, q={r.get('q_value', np.nan):.3g}, "
                        f"n_units={r['n_units']}, n_patients={r['n_patients']}\n"
                    )

    # save combined tables
    if hist_tables:
        for name, tbl in hist_tables.items():
            if isinstance(tbl, pd.DataFrame):
                tbl.to_csv(os.path.join(paths.out_dir, f"table_{name}.csv"), index=False)

    if ephys_tables:
        for name, tbl in ephys_tables.items():
            if isinstance(tbl, pd.DataFrame):
                tbl.to_csv(os.path.join(paths.out_dir, f"table_{name}.csv"), index=False)

    if corr_tables:
        for name, tbl in corr_tables.items():
            if isinstance(tbl, pd.DataFrame):
                tbl.to_csv(os.path.join(paths.out_dir, f"table_{name}.csv"), index=False)

    return {
        "histology": hist_tables,
        "ephys": ephys_tables,
        "correlations": corr_tables,
        "mixedlm": mixed_tables,
    }


if __name__ == "__main__":
    # Edit paths here
    paths = Paths(
        histology_csv="E:/in_vivo_in_vitro/stat/statbase/in_vivo_in_vitro_pv_neun_histology.csv",
        ephys_csv="E:/in_vivo_in_vitro/stat/statbase/vivo_vitro_database_from_outputs_checked.csv",  # set to your ephys master CSV
        out_dir="E:/in_vivo_in_vitro/stat/histology_ephys_out",
    )
    run_pipeline(paths)
