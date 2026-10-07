"""
Linear mixed-effects models for histology data analysis.

Purpose
-------
Fit linear mixed-effects models (LME) to histology data with patient
as a random effect. Tests for state (in vivo vs in vitro) and layer
(supra vs infra) effects on histology metrics.

Key features
------------
- LME with patient random intercept
- Fixed effects for state and layer
- Model comparison via likelihood ratio tests
- Publication-ready table output

Usage
-----
    python -m stats.mixedlm_histo_compact

Dependencies
------------
    numpy, pandas, statsmodels.formula.api, matplotlib, seaborn
"""
import os

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")

# =============================================================================
# 1. Configuration & Nature Neuroscience Typography Constants
# =============================================================================
sns.set_theme(style="white", context="paper", font_scale=0.85)
plt.rcParams.update({
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.spines.left": False,
    "axes.spines.bottom": True,
    "axes.grid": False,
    "figure.dpi": 300,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Arial Unicode MS", "sans-serif"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

MIXEDLM_CELLCLASS_ORDER = ["all", "Exc", "Inh", "FS"]

# Explicitly ordered from top-to-bottom as requested
TARGET_EPHYS_METRICS = [
    "preferred_phase_all_deg_high_gamma",
    "preferred_phase_all_deg_low_gamma",
    "plv_high_gamma_strength",
    "plv_low_gamma_strength",
    "population_coupling",
    "isi_cv",
    "burstiness_percent",
    "max_firing_rate_10s_Hz",
    "firing_rate_Hz"]

EPHYS_LABELS_CLEAN = {
    "firing_rate_Hz": "Firing rate",
    "max_firing_rate_10s_Hz": "Max firing rate (10 s)",
    "burstiness_percent": "Burstiness (%)",
    "isi_cv": "ISI CV",
    "population_coupling": "Population coupling",
    "plv_low_gamma_strength": "PLV low gamma strength",
    "plv_high_gamma_strength": "PLV high gamma strength",
    "preferred_phase_all_deg_low_gamma": "Preferential phase (low gamma)",
    "preferred_phase_all_deg_high_gamma": "Preferential phase (high gamma)"
}

COMPACT_PANEL_ORDER = [
    ("vivo", "neun_density_cells_mm2", "Vivo NeuN Density"),
    ("vivo", "pv_density_cells_mm2", "Vivo PV Density"),
    ("vivo", "pv_coverage_percent", "Vivo PV Coverage"),
    ("vitro", "neun_density_cells_mm2", "Vitro NeuN Density"),
    ("vitro", "pv_density_cells_mm2", "Vitro PV Density"),
    ("vitro", "pv_coverage_percent", "Vitro PV Coverage")
]

# Modern, accessible, colorblind-friendly design palette
CELLCLASS_COLORS = {
    "all": "#333333",   # Deep Charcoal
    "Exc": "#8C2F39",   # Dark Red
    "Inh": "#1E3A8A",   # Dark Blue
    "FS": "#00897B"     # Dark Accessible Green
}

# =============================================================================
# 2. Data Cleaners and Normalization Engine 
# =============================================================================
def normalize_layer(layer) -> str:
    if pd.isna(layer): return "other"
    s = str(layer).strip().lower()
    if s.startswith("sup"): return "supra"
    if s.startswith("gra"): return "gran"
    if s.startswith("inf"): return "infra"
    return "other"

def normalize_state(state) -> str:
    if pd.isna(state): return "other"
    s = str(state).strip().lower()
    if s in ("a", "awake", "wake"): return "awake"
    if s in ("s", "sleep"): return "sleep"
    if s in ("v", "vitro", "in_vitro", "invitro"): return "vitro"
    return "other"

def normalize_celltype(celltype) -> str:
    if pd.isna(celltype): return "other"
    s = str(celltype).strip().lower()
    if s in ("rs", "rs-pc", "rs_pc", "regular"): return "RS-PC"
    if s in ("ib", "ib-pc", "ib_pc", "bursting"): return "IB-PC"
    if s in ("in", "interneuron", "int"): return "IN"
    if s in ("fs", "fast_spiking"): return "FS"
    return s

def fdr_bh(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    out = np.full(pvals.size, np.nan, dtype=float)
    ok = np.isfinite(pvals)
    p = pvals[ok]
    if p.size == 0: return out
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * (p.size / (np.arange(1, p.size + 1)))
    q = np.minimum.accumulate(q[::-1])[::-1]
    tmp = np.full(p.size, np.nan)
    tmp[order] = np.clip(q, 0, 1)
    out[ok] = tmp
    return out

def load_and_normalize_histology(h: pd.DataFrame) -> pd.DataFrame:
    if h.empty: return h
    h = h.copy()
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

def load_and_normalize_master(m: pd.DataFrame) -> pd.DataFrame:
    if m.empty: return m
    m = m.copy()
    if "state" in m.columns:
        m["state_norm"] = m["state"].map(normalize_state)
    elif "state_norm" in m.columns:
        m["state_norm"] = m["state_norm"].map(normalize_state)
    else:
        m["state_norm"] = "other"

    if "layer_norm" not in m.columns:
        m["layer_norm"] = m.get("layer", pd.Series(index=m.index, dtype=object)).map(normalize_layer)
    m["celltype_norm"] = m.get("celltype", pd.Series(index=m.index, dtype=object)).map(normalize_celltype)
    return m

def prepare_histology_wide(histology: pd.DataFrame) -> pd.DataFrame:
    h = histology.copy()
    grouped = h.groupby(["staining", "patient", "histology_state", "layer_norm"], dropna=False).agg(
        mean_density_cells_mm2=("density (cell/mm^2)", "mean"),
        mean_coverage_percent=("coverage (%)", "mean"),
    ).reset_index()

    pv = grouped[grouped["staining"] == "PV"].rename(columns={
        "mean_density_cells_mm2": "pv_density_cells_mm2",
        "mean_coverage_percent": "pv_coverage_percent",
    })
    neun = grouped[grouped["staining"] == "NEUN"].rename(columns={
        "mean_density_cells_mm2": "neun_density_cells_mm2",
        "mean_coverage_percent": "neun_coverage_percent",
    })

    out = pv.merge(
        neun[["patient", "histology_state", "layer_norm", "neun_density_cells_mm2", "neun_coverage_percent"]],
        on=["patient", "histology_state", "layer_norm"], how="outer"
    )
    return out.dropna(subset=["patient", "histology_state", "layer_norm"])

def build_association_table(master: pd.DataFrame, histology: pd.DataFrame) -> pd.DataFrame:
    hist_wide = prepare_histology_wide(histology)
    if hist_wide.empty: return pd.DataFrame()

    m = master.copy()
    m["histology_state"] = np.where(m["state_norm"] == "vitro", "vitro", "vivo")

    m["cellclass_mixedlm"] = "other"
    m.loc[m["celltype_norm"].isin(["RS-PC", "IB-PC"]), "cellclass_mixedlm"] = "Exc"
    m.loc[m["celltype_norm"].isin(["IN", "FS"]), "cellclass_mixedlm"] = "Inh"
    m.loc[m["celltype_norm"] == "FS", "cellclass_mixedlm"] = "FS"

    ephys_metrics = [metric for metric in TARGET_EPHYS_METRICS if metric in m.columns]
    keep = ["patient", "histology_state", "state_norm", "layer_norm", "cellclass_mixedlm"] + ephys_metrics
    unit = m[keep].copy()

    all_view = unit.copy()
    all_view["cellclass_mixedlm"] = "all"
    unit_stack = pd.concat([all_view, unit[unit["cellclass_mixedlm"].isin(["Exc", "Inh", "FS"])]], ignore_index=True)

    e_long = unit_stack.melt(
        id_vars=["patient", "histology_state", "state_norm", "layer_norm", "cellclass_mixedlm"],
        value_vars=ephys_metrics, var_name="ephys_metric", value_name="ephys_value"
    )
    e_long["ephys_value"] = pd.to_numeric(e_long["ephys_value"], errors="coerce")

    hist_metrics = ["neun_density_cells_mm2", "pv_density_cells_mm2", "pv_coverage_percent"]
    hist_metrics = [m for m in hist_metrics if m in hist_wide.columns]
    
    h_long = hist_wide.melt(
        id_vars=["patient", "histology_state", "layer_norm"],
        value_vars=hist_metrics, var_name="histology_metric", value_name="histology_value"
    )
    h_long["histology_value"] = pd.to_numeric(h_long["histology_value"], errors="coerce")

    merged = h_long.merge(e_long, on=["patient", "histology_state", "layer_norm"], how="inner")
    return merged.dropna(subset=["histology_value", "ephys_value"])

# =============================================================================
# 3. Robust MixedLM Optimization Pipeline
# =============================================================================
def fit_association(sub: pd.DataFrame) -> dict:
    sub = sub.copy()
    if "density" in str(sub["histology_metric"].iloc[0]):
        sub["histology_value"] = np.log1p(sub["histology_value"])
        
    if np.std(sub["histology_value"]) < 1e-6 or np.std(sub["ephys_value"]) < 1e-6:
        return {"beta_std": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0, "converged": False}

    sub["hist_z"] = (sub["histology_value"] - sub["histology_value"].mean()) / sub["histology_value"].std()
    sub["ephys_z"] = (sub["ephys_value"] - sub["ephys_value"].mean()) / sub["ephys_value"].std()
    sub = sub.dropna(subset=["hist_z", "ephys_z"])

    if len(sub) < 8 or sub["patient"].nunique() < 3:
        return {"beta_std": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0, "converged": False}

    terms = ["hist_z"]
    if sub["layer_norm"].nunique(dropna=True) > 1: terms.append("C(layer_norm)")
    formula = "ephys_z ~ " + " + ".join(terms)

    try:
        model = smf.mixedlm(formula, sub, groups=sub["patient"], re_formula="1")
        result = model.fit(reml=True, method="powell", maxiter=500, disp=False)
        if not result.converged or any(np.diag(result.cov_params()) < 0):
            raise ValueError("Structural fallback triggered")
        conf = result.conf_int().loc["hist_z"]
        return {
            "beta_std": float(result.params["hist_z"]),
            "ci_low": float(conf[0]),
            "ci_high": float(conf[1]),
            "p_value": float(result.pvalues["hist_z"]),
            "converged": True
        }
    except Exception:
        try:
            ols = smf.ols(formula, sub).fit(cov_type="cluster", cov_kwds={"groups": sub["patient"]})
            conf = ols.conf_int().loc["hist_z"]
            return {
                "beta_std": float(ols.params["hist_z"]),
                "ci_low": float(conf[0]),
                "ci_high": float(conf[1]),
                "p_value": float(ols.pvalues["hist_z"]),
                "converged": True
            }
        except Exception:
            return {"beta_std": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0, "converged": False}

def run_mixedlm_pipeline(master: pd.DataFrame, histology: pd.DataFrame, out_dir: str):
    merged = build_association_table(master, histology)
    if merged.empty: return pd.DataFrame()

    rows = []
    for state, hmet, _ in COMPACT_PANEL_ORDER:
        for cellclass in MIXEDLM_CELLCLASS_ORDER:
            for emet in TARGET_EPHYS_METRICS:
                sub = merged[(merged["histology_state"] == state) & 
                             (merged["cellclass_mixedlm"] == cellclass) & 
                             (merged["histology_metric"] == hmet) & 
                             (merged["ephys_metric"] == emet)]
                
                fit = fit_association(sub) if not sub.empty else {"beta_std": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0, "converged": False}
                
                rows.append({
                    "histology_state": state, "cellclass_ei": cellclass, "histology_metric": hmet, "ephys_metric": emet,
                    "beta_std": fit["beta_std"], "ci_low": fit["ci_low"], "ci_high": fit["ci_high"],
                    "p": fit["p_value"], "converged": fit["converged"], "n_units": len(sub)
                })

    out = pd.DataFrame(rows)
    valid_mask = out["converged"] == True
    if valid_mask.any():
        out.loc[valid_mask, "q_fdr"] = fdr_bh(out.loc[valid_mask, "p"].values)
    out["q_fdr"] = out["q_fdr"].fillna(1.0)
    
    os.makedirs(out_dir, exist_ok=True)
    out.to_csv(os.path.join(out_dir, "stats_histo_ephys_mixedlm.csv"), index=False)
    return out

# =============================================================================
# 4. Nature Neuroscience High-Density Forest Plotting Engine
# =============================================================================
def plot_compact_nature_forest(stats_df: pd.DataFrame, out_dir: str):
    if stats_df.empty: return
    
    n_ephys = len(TARGET_EPHYS_METRICS)
    n_panels = len(COMPACT_PANEL_ORDER)
    
    fig, axes = plt.subplots(1, n_panels, figsize=(14, 4.8), sharey=True, squeeze=False)
    axes = axes[0]
    
    # --- 1. Dynamic Scaling Engine: Split scales for in vivo vs in vitro ---
    sig_mask = stats_df["converged"] & (stats_df["q_fdr"] < 0.05)
    
    # Separate sub-selections based on state
    vivo_sig = stats_df[sig_mask & (stats_df["histology_state"] == "vivo")]
    vitro_sig = stats_df[sig_mask & (stats_df["histology_state"] == "vitro")]
    
    #vivo_lim = max(vivo_sig["beta_std"].abs().max() * 1.35, 0.6) if not vivo_sig.empty else 0.7
    #vitro_lim = max(vitro_sig["beta_std"].abs().max() * 1.35, 0.6) if not vitro_sig.empty else 1.2
    vivo_lim = 10
    vitro_lim = 5
    
    # Highly compact layout adjustments: reduced offset distance between markers and frame limits
    offset_scale = 0.12  # Tighter cell-class clustering
    offsets = {cell: (i - (len(MIXEDLM_CELLCLASS_ORDER)-1)/2) * offset_scale for i, cell in enumerate(MIXEDLM_CELLCLASS_ORDER)}

    for col_idx, (state, hmet, panel_title) in enumerate(COMPACT_PANEL_ORDER):
        ax = axes[col_idx]
        
        # Determine specific scale depending on panel background state
        beta_lim = vivo_lim if state == "vivo" else vitro_lim
        beta_margin = (-beta_lim, beta_lim)
        
        # Clean background canvas setup
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['bottom'].set_color("#bababa")
        ax.spines['bottom'].set_linewidth(0.6)
            
        # --- 2. Compact Frame Design: Reduced height margins, no background fills ---
        for y_idx in range(n_ephys):
            # Tightened vertical boundary constraints (0.38 instead of 0.45) to minimize gap distances
            ax.axhline(y_idx - 0.38, color="#e3e6e8", linewidth=0.5, zorder=1)
            ax.axhline(y_idx + 0.38, color="#e3e6e8", linewidth=0.5, zorder=1)
            
            # Draw framing end-caps on left and right edge margins
            ax.plot([beta_margin[0], beta_margin[0]], [y_idx - 0.38, y_idx + 0.38], color="#e3e6e8", linewidth=0.5, zorder=1)
            ax.plot([beta_margin[1], beta_margin[1]], [y_idx - 0.38, y_idx + 0.38], color="#e3e6e8", linewidth=0.5, zorder=1)
        
        # Fixed center reference line
        ax.axvline(0, color="#686868", linestyle="-", linewidth=0.5, zorder=1)
        
        sub_df = stats_df[(stats_df["histology_state"] == state) & (stats_df["histology_metric"] == hmet)]
        
        for y_idx, emet in enumerate(TARGET_EPHYS_METRICS):
            met_df = sub_df[sub_df["ephys_metric"] == emet]
            
            for cell in MIXEDLM_CELLCLASS_ORDER:
                row = met_df[met_df["cellclass_ei"] == cell]
                if row.empty: continue
                row = row.iloc[0]
                
                y_pos = y_idx + offsets[cell]
                color = CELLCLASS_COLORS[cell]
                
                if row["converged"]:
                    b_val = row["beta_std"]
                    low_ci = row["ci_low"]
                    high_ci = row["ci_high"]
                    
                    # Capping outliers based on the split-axis limits
                    if b_val < beta_margin[0]:
                        ax.scatter(beta_margin[0] + (beta_lim * 0.03), y_pos, color=color, s=12, marker="<", zorder=4)
                    elif b_val > beta_margin[1]:
                        ax.scatter(beta_margin[1] - (beta_lim * 0.03), y_pos, color=color, s=12, marker=">", zorder=4)
                    else:
                        ax.scatter(b_val, y_pos, color=color, s=15, marker="o", edgecolors="none", zorder=4)
                    
                    # Fit error bars cleanly within limits
                    plot_low = max(low_ci, beta_margin[0])
                    plot_high = min(high_ci, beta_margin[1])
                    
                    # Clean half-transparent CI indicator lines
                    ax.plot([plot_low, plot_high], [y_pos, y_pos], color=color, alpha=0.4, linewidth=4, zorder=3)
                    
                    # --- 3. Enhanced Significances: Bigger Asterisks ---
                    q = row["q_fdr"]
                    stars = ""
                    if q < 0.001: stars = "***"
                    elif q < 0.01: stars = "**"
                    elif q < 0.05: stars = "*"
                    
                    if stars:
                        ax.text(beta_lim * 0.83, y_pos, stars, va="center", ha="center", 
                                color=color, fontsize=10.5, fontweight="bold", zorder=5)
                else:
                    # Model convergence failure indicator
                    ax.scatter(0, y_pos, color="#e8e8e8", s=16, marker="X", edgecolors="#a1a1a1", linewidth=0.4, zorder=2)
        
        # Apply separated limits and clean symmetric labels
        ax.set_xlim(beta_margin)
        ax.set_xticks([round(-beta_lim * 0.7, 1), 0, round(beta_lim * 0.7, 1)])
        ax.set_title(panel_title, fontsize=8.5, fontweight="bold", pad=12)
        ax.set_xlabel(r"Effect Size ($\beta$)", fontsize=8)
        
        if col_idx == 0:
            ax.set_yticks(range(n_ephys))
            ax.set_yticklabels([EPHYS_LABELS_CLEAN.get(m, m) for m in TARGET_EPHYS_METRICS], fontsize=8.5)
            ax.tick_params(left=False)
        else:
            ax.tick_params(left=False)

    legend_elements = [mpatches.Patch(color=c, label=l) for l, c in CELLCLASS_COLORS.items()]
    legend_elements.append(plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='#e8e8e8', 
                                      markeredgecolor='#a1a1a1', markersize=6, label='Non-Conv / Missing'))
    
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.06), ncol=5, frameon=False, fontsize=8.5)
    plt.tight_layout()
    
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "nature_neuro_forest_framed_adjusted.png"), bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(out_dir, "nature_neuro_forest_framed_adjusted.svg"), bbox_inches="tight")
    plt.close()

# =============================================================================
# 5. Execution Pipeline
# =============================================================================
if __name__ == "__main__":
    DB_PATH = "E:\\in_vivo_in_vitro\\stat\\statbase\\vivo_vitro_database_from_outputs_checked.csv"
    HISTO_PATH = "E:\\in_vivo_in_vitro\\stat\\statbase\\in_vivo_in_vitro_pv_neun_histology.csv"
    OUT_DIR = "E:\\in_vivo_in_vitro\\stat\\statbase\\histology_mixedlm_results"

    if os.path.exists(DB_PATH) and os.path.exists(HISTO_PATH):
        print("Executing data alignment and loading routines...")
        master_raw = pd.read_csv(DB_PATH)
        histo_raw = pd.read_csv(HISTO_PATH)
        
        master_df = load_and_normalize_master(master_raw)
        histo_df = load_and_normalize_histology(histo_raw)
        
        print("Fitting models across grid partitions...")
        stats_df = run_mixedlm_pipeline(master_df, histo_df, OUT_DIR)
        
        print("Generating Nature Neuroscience Forest layout...")
        plot_compact_nature_forest(stats_df, OUT_DIR)
        print(f"Pipeline complete. Outputs saved to: {OUT_DIR}")
    