import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from scipy.stats import mannwhitneyu, gaussian_kde
from pathlib import Path

warnings.filterwarnings("ignore", ".*swarmplot.*")
warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# CONFIGURATION & PUBLICATION PALETTES
# =============================================================================
sns.set_theme(style="white")
plt.rcParams.update({
    'axes.grid': False,
    'figure.dpi': 300,
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.family': 'sans-serif',
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9
})

STATE_ORDER = ["awake", "sleep", "vitro"]
CELLTYPE_ORDER = ["RS-PC", "IB-PC", "IN", "FS"]
LAYER_ORDER = ["supra", "gran", "infra"]
BANDS = ["delta", "theta", "low_gamma", "high_gamma"]

# Original palette used for standard plots
COLORS_CELLTYPE = {
    "RS-PC": "#EA5189",
    "IB-PC": "#FFB74D",  # Faint yellow-orange
    "IN": "#3D5AFE",
    "FS": "#00BFA5",
}

# Alternative 2: High-contrast/darkened palette for white backgrounds
COLORS_CELLTYPE_HIGH_CONTRAST = {
    "RS-PC": "#C2185B",  # Deep Rose
    "IB-PC": "#E65100",  # Dark Burnt Orange (Fixes the pale yellow visibility)
    "IN": "#1A237E",     # Deep Royal Blue
    "FS": "#004D40",     # Dark Pine Green
}

COLORS_LAYER = {
    "supra": "#FF5E00",
    "gran":  "#9C27B0",
    "infra": "#2196F3",
}

TARGET_METRICS = [
    ("firing_rate_Hz", "Firing rate (Hz)"),
    ("max_firing_rate_10s_Hz", "Max firing 10s (Hz)"),
    ("max_firing_rate_1s_Hz", "Max firing 1s (Hz)"), 
    ("burstiness_percent", "Burstiness (%)"),
    ("bursti_2", "Burst index (B2)"),
    ("halfwidth1_ms", "Main halfwidth (ms)"),
    ("halfwidth2_ms", "Repolarization slope (ms)"),
    ("population_coupling", "Population coupling"),
    ("mean_plv_delta", "Mean PLV (Delta)"),
    ("mean_plv_theta", "Mean PLV (Theta)"),
    ("mean_plv_low_gamma", "Mean PLV (Low $\gamma$)"),
    ("mean_plv_high_gamma", "Mean PLV (High $\gamma$)"),
    ("mean_plv_sig_low_gamma", "Sig. PLV (Low $\gamma$)"),
    ("mean_plv_sig_high_gamma", "Sig. PLV (High $\gamma$)"),
    ("peak_trough_ratio", "Peak-Trough Ratio"),
    ("peak_to_trough_delay_ms", "Peak-Trough Delay (ms)"),
    ("symmetry", "Symmetry"),
    ("max_slope", "Max Slope"),
    ("repol_slope", "Repolarization Slope"),
    ("isi_cv", "ISI CV"),
    ("plv_low_gamma_strength", "PLV Low $\gamma$ Strength"),
    ("plv_high_gamma_strength", "PLV High $\gamma$ Strength"),
    ("plv_low_gamma_clustering", "PLV Low $\gamma$ Clustering"),
    ("plv_high_gamma_clustering", "PLV High $\gamma$ Clustering"),
    ("plv_low_gamma_pagerank", "PLV Low $\gamma$ PageRank"),
    ("plv_high_gamma_pagerank", "PLV High $\gamma$ PageRank"), 
    ("plv_low_gamma_betweenness", "PLV Low $\gamma$ Betweenness"),
    ("plv_high_gamma_betweenness", "PLV High $\gamma$ Betweenness"),
    ("plv_low_gamma_closeness", "PLV Low $\gamma$ Closeness"),
    ("plv_high_gamma_closeness", "PLV High $\gamma$ Closeness"),
    ("plv_low_gamma_local_efficiency", "PLV Low $\gamma$ Local Eff."),
    ("plv_high_gamma_local_efficiency", "PLV High $\gamma$ Local Eff."),
    ("plv_low_gamma_hubness_z", "PLV Low $\gamma$ Hubness (z)"),
    ("plv_high_gamma_hubness_z", "PLV High $\gamma$ Hubness (z)")
]

# =============================================================================
# STATS ENGINE
# =============================================================================
def native_fdr_correction(p_values):
    p_arr = np.asfarray(p_values)
    by_descend = p_arr.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p_arr)) / np.arange(len(p_arr), 0, -1)
    q_arr = np.minimum.accumulate(p_arr[by_descend] * steps)[by_orig]
    return np.minimum(q_arr, 1.0)

def draw_flat_sig_bar(ax, x1, x2, y, p, color, step_size):
    if p >= 0.05:
        return y
    ax.plot([x1, x1, x2, x2], [y, y + step_size * 0.2, y + step_size * 0.2, y], 
            color=color, lw=0.9, clip_on=False)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*"
    ax.text((x1 + x2) / 2, y + step_size * 0.25, sig, 
            ha='center', va='bottom', color=color, fontweight='bold', fontsize=7)
    return y + step_size

def compute_all_stats(df, metric):
    rows = []
    state_pairs = [("awake", "sleep"), ("sleep", "vitro"), ("awake", "vitro")]
    
    def _add_row(scope, group, s1, s2, g1, g2):
        if len(g1) > 2 and len(g2) > 2:
            _, p = mannwhitneyu(g1, g2, alternative='two-sided')
            rows.append({
                "scope": scope, "group": group, "s1": s1, "s2": s2, 
                "p": p, "mean_s1": np.mean(g1), "mean_s2": np.mean(g2)
            })

    for s1, s2 in state_pairs:
        _add_row("global", "all", s1, s2, 
                 df[df["state_norm"] == s1][metric].dropna().values,
                 df[df["state_norm"] == s2][metric].dropna().values)
            
    for group_col, order in [("celltype_norm", CELLTYPE_ORDER), ("layer_norm", LAYER_ORDER)]:
        if group_col not in df.columns: continue
        for grp in order:
            sub = df[df[group_col] == grp]
            for s1, s2 in state_pairs:
                _add_row(group_col, grp, s1, s2, 
                         sub[sub["state_norm"] == s1][metric].dropna().values,
                         sub[sub["state_norm"] == s2][metric].dropna().values)
    
    if not rows: return pd.DataFrame(), {}
    stats_df = pd.DataFrame(rows)
    stats_df["q"] = native_fdr_correction(stats_df["p"].values)
    
    lookup = {}
    for _, r in stats_df.iterrows():
        lookup[(r["scope"], r["group"], r["s1"], r["s2"])] = r["q"]
    return stats_df, lookup

# =============================================================================
# PLOTTING ENGINES
# =============================================================================
def save_dual_format(fig, base_path: str):
    p = Path(base_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p.with_suffix('.png'), dpi=300, transparent=True)
    fig.savefig(p.with_suffix('.svg'), transparent=True)

def plot_variant_a_generic(df, metric, ylabel, stats_lookup, out_png, group_col, order_list, color_dict):
    if df[metric].dropna().empty: return
    upper = df[metric].quantile(0.99)
    df = df.copy()
    df[metric] = df[metric].clip(upper=upper)
    
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    spacing = 0.18
    state_offsets = {"awake": 0.0, "sleep": 0.9, "vitro": 1.8}
    
    for state in STATE_ORDER:
        state_data = df[df["state_norm"] == state][metric].dropna()
        if not state_data.empty:
            base = state_offsets[state]
            box_center = base + ((len(order_list) - 1) * spacing) / 2.0
            box_width = (len(order_list) * spacing) + 0.08
            
            ax.boxplot(state_data.values, positions=[box_center], widths=box_width,
                       patch_artist=True, showfliers=False, zorder=0,
                       boxprops=dict(facecolor="#F6F6F6", edgecolor="#E2E2E2", lw=0.7),
                       whiskerprops=dict(color="#E2E2E2", lw=0.5),
                       capprops=dict(color="#E2E2E2", lw=0.5),
                       medianprops=dict(color="#999999", lw=1.0))

    for state in STATE_ORDER:
        base = state_offsets[state]
        for c_idx, group in enumerate(order_list):
            pos = base + (c_idx * spacing)
            sub = df[(df["state_norm"] == state) & (df[group_col] == group)][metric].dropna()
            if not sub.empty:
                ax.scatter(np.random.normal(pos, 0.04, len(sub)), sub, 
                           facecolors='none', edgecolors=color_dict.get(group, "#333333"), 
                           s=12, alpha=0.6, linewidth=0.5, zorder=2)
                ax.errorbar(pos, sub.mean(), yerr=sub.std(), fmt='_', color='#222222', lw=0.9, capsize=0, zorder=4)

    y_min, y_max = df[metric].min(), df[metric].max()
    data_range = y_max - (y_min if y_min > 0 else 0)
    step_size = data_range * 0.07
    y_bar = y_max + step_size * 0.2
    state_pairs = [("awake", "sleep"), ("sleep", "vitro"), ("awake", "vitro")]
    
    for c_idx, group in enumerate(order_list):
        color = color_dict.get(group, "#333333")
        for s1, s2 in state_pairs:
            q_val = stats_lookup.get((group_col, group, s1, s2), 1.0)
            if q_val < 0.05:
                x1 = state_offsets[s1] + (c_idx * spacing)
                x2 = state_offsets[s2] + (c_idx * spacing)
                y_bar = draw_flat_sig_bar(ax, x1, x2, y_bar, q_val, color, step_size)
                
    for s1, s2 in state_pairs:
        q_val = stats_lookup.get(("global", "all", s1, s2), 1.0)
        if q_val < 0.05:
            mid_offset = ((len(order_list) - 1) * spacing) / 2.0
            x1 = state_offsets[s1] + mid_offset
            x2 = state_offsets[s2] + mid_offset
            y_bar = draw_flat_sig_bar(ax, x1, x2, y_bar, q_val, "black", step_size)

    ax.set_ylim(y_min if y_min > 0 else 0, y_bar + step_size * 0.2)
    ax.set_xticks([state_offsets[s] + ((len(order_list) - 1) * spacing) / 2.0 for s in STATE_ORDER])
    ax.set_xticklabels([s.upper() for s in STATE_ORDER], fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.spines[['right', 'top']].set_visible(False)
    
    save_dual_format(fig, out_png)
    plt.close(fig)

def plot_variant_b_unified_with_stats(df, metric, ylabel, stats_lookup, out_png):
    upper = df[metric].quantile(0.99)
    df = df.copy()
    df[metric] = df[metric].clip(upper=upper)
    
    fig, ax = plt.subplots(figsize=(3.2, 3.5))
    
    for s_idx, state in enumerate(STATE_ORDER):
        state_data = df[df["state_norm"] == state][metric].dropna()
        if not state_data.empty:
            ax.boxplot(state_data.values, positions=[s_idx], widths=0.35,
                       patch_artist=True, showfliers=False, zorder=0,
                       boxprops=dict(facecolor="#F6F6F6", edgecolor="#E2E2E2", lw=0.7),
                       whiskerprops=dict(color="#E2E2E2", lw=0.5),
                       capprops=dict(color="#E2E2E2", lw=0.5),
                       medianprops=dict(color="#999999", lw=1.0))
            
    for s_idx, state in enumerate(STATE_ORDER):
        sub = df[df["state_norm"] == state]
        for ct in CELLTYPE_ORDER:
            ct_sub = sub[sub["celltype_norm"] == ct][metric].dropna()
            if not ct_sub.empty:
                ax.scatter(np.random.normal(s_idx, 0.05, len(ct_sub)), ct_sub,
                           facecolors='none', edgecolors=COLORS_CELLTYPE.get(ct, "black"),
                           s=12, alpha=0.5, linewidth=0.5, zorder=2)
        
        state_data = sub[metric].dropna()
        if not state_data.empty:
            ax.errorbar(s_idx, state_data.mean(), yerr=state_data.std(), fmt='_', 
                        color='#111111', lw=1.2, ms=6, zorder=5)

    y_min, y_max = df[metric].min(), df[metric].max()
    data_range = y_max - (y_min if y_min > 0 else 0)
    step_size = data_range * 0.07
    y_bar = y_max + step_size * 0.2
    state_pairs = [("awake", "sleep"), ("sleep", "vitro"), ("awake", "vitro")]
    
    for ct in CELLTYPE_ORDER:
        color = COLORS_CELLTYPE.get(ct, "black")
        for s1, s2 in state_pairs:
            q_val = stats_lookup.get(("celltype_norm", ct, s1, s2), 1.0)
            if q_val < 0.05:
                x1, x2 = STATE_ORDER.index(s1), STATE_ORDER.index(s2)
                y_bar = draw_flat_sig_bar(ax, x1, x2, y_bar, q_val, color, step_size)
                
    for s1, s2 in state_pairs:
        q_val = stats_lookup.get(("global", "all", s1, s2), 1.0)
        if q_val < 0.05:
            x1, x2 = STATE_ORDER.index(s1), STATE_ORDER.index(s2)
            y_bar = draw_flat_sig_bar(ax, x1, x2, y_bar, q_val, "black", step_size)

    ax.set_ylim(y_min if y_min > 0 else 0, y_bar + step_size * 0.2)
    ax.set_xticks(range(len(STATE_ORDER)))
    ax.set_xticklabels([s.upper() for s in STATE_ORDER], fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.spines[['right', 'top']].set_visible(False)
    
    save_dual_format(fig, out_png)
    plt.close(fig)


# =============================================================================
# REFACTORED PHASE RASTER WITH ENGINE VARIANTS
# =============================================================================
def plot_compact_phase_raster_with_sine(df, band, out_png, variant="sqrt_saturated"):
    """
    Refactored sinusoid plotter with state-specific bolding, coloring, and vertical separation.
    """
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    
    fig, axs = plt.subplots(4, 1, figsize=(3.2, 5.5), 
                            gridspec_kw={'height_ratios': [0.6, 1, 1, 1]}, 
                            constrained_layout=False)
    
    y_levels = {ct: 0.8 - (i * 0.2) for i, ct in enumerate(CELLTYPE_ORDER)}
    x_sine = np.linspace(0, 360, 361)
    y_base_sine = np.sin(np.deg2rad(x_sine))
    
    # --- Step 1: State-specific density analysis for bolding ---
    state_colors = {'awake': '#D3D3D3', 'sleep': '#808080', 'vitro': '#36454F'}
    state_offsets = {'awake': 0.3, 'sleep': 0.0, 'vitro': -0.3}
    bold_masks = {}
    
    deg_col = f"preferred_phase_all_deg_{band}"
    rad_col_sig = f"preferred_phase_sig_rad_{band}"
    rad_col_all = f"preferred_phase_all_rad_{band}"
    
    for state in STATE_ORDER:
        state_df = df[df["state_norm"] == state]
        phases = []
        for ct in CELLTYPE_ORDER:
            ct_df = state_df[state_df["celltype_norm"] == ct]
            if ct_df.empty: continue
            if deg_col in ct_df.columns: phases.extend(ct_df[deg_col].dropna().values % 360)
            elif rad_col_sig in ct_df.columns: phases.extend(np.degrees(ct_df[rad_col_sig].dropna().values) % 360)
            elif rad_col_all in ct_df.columns: phases.extend(np.degrees(ct_df[rad_col_all].dropna().values) % 360)
        
        bold_masks[state] = np.zeros_like(x_sine, dtype=bool)
        if len(phases) > 5:
            try:
                extended = np.concatenate([phases, np.array(phases)-360, np.array(phases)+360])
                kde = gaussian_kde(extended, bw_method=0.15)
                density = kde(x_sine)
                bold_masks[state] = density >= np.percentile(density, 70)
            except: pass

    # --- Step 2: Plot Separated State Sine Waves ---
    ax_top = axs[0]
    for state in STATE_ORDER:
        offset = state_offsets[state]
        color = state_colors[state]
        y_shifted = y_base_sine + offset
        
        # Plot full faint line
        ax_top.plot(x_sine, y_shifted, color=color, lw=1.0, alpha=0.4, zorder=1)
        
        # Plot bold segments
        mask = bold_masks[state]
        for i in range(len(x_sine)-1):
            if mask[i] and mask[i+1]:
                ax_top.plot(x_sine[i:i+2], y_shifted[i:i+2], color=color, lw=2.5, zorder=2)
            
    ax_top.set_xlim(0, 360)
    ax_top.set_ylim(-1.6, 1.6)
    ax_top.axis('off')
    ax_top.set_title(f"{band.upper()} PHASE REFERENCE", fontsize=8, fontweight="bold", pad=2)

    # --- Step 3: Populate Track Rasters (Unchanged) ---
    palette = COLORS_CELLTYPE_HIGH_CONTRAST if variant == "high_contrast" else COLORS_CELLTYPE
    
    for s_idx, state in enumerate(STATE_ORDER):
        ax = axs[s_idx + 1]
        if band == 'delta' and state != 'sleep':
            ax.text(180, 0.5, "N/A", ha='center', va='center', color="#BBBBBB", fontsize=8, fontstyle='italic')
            ax.set_ylabel(state.upper(), fontsize=8, fontweight="bold", rotation=0, labelpad=15, va="center")
            ax.set_xlim(0, 360); ax.set_yticks([]); ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
            continue
            
        state_df = df[df["state_norm"] == state]
        for ct in CELLTYPE_ORDER:
            ct_df = state_df[state_df["celltype_norm"] == ct]
            if ct_df.empty: continue
            if deg_col in ct_df.columns: phases = ct_df[deg_col].fillna(0) % 360
            elif rad_col_sig in ct_df.columns: phases = np.degrees(ct_df[rad_col_sig].fillna(0)) % 360
            elif rad_col_all in ct_df.columns: phases = np.degrees(ct_df[rad_col_all].fillna(0)) % 360
            else: continue
            
            plv_cols = [f"mean_plv_sig_{band}", f"mean_plv_all_{band}", f"mean_plv_{band}"]
            plv_col = next((c for c in plv_cols if c in ct_df.columns), None)
            plvs = ct_df[plv_col].fillna(0) if plv_col else np.ones(len(phases)) * 0.3
            
            y_base = y_levels[ct]
            for deg, plv in zip(phases, plvs):
                if variant in ["sqrt_saturated", "high_contrast"]:
                    vmax_plv = 0.35; alpha_min = 0.30; alpha = np.clip(np.sqrt(plv / vmax_plv), alpha_min, 1.0); lw = 1.6
                elif variant == "binary_mask":
                    threshold = 0.15; alpha = 0.85 if plv >= threshold else 0.20; lw = 1.6 if plv >= threshold else 1.1
                else:
                    alpha = np.clip(plv / 0.6, 0.15, 1.0); lw = 1.1
                ax.vlines(deg, y_base - 0.07, y_base + 0.07, color=palette.get(ct, "#333333"), alpha=alpha, lw=lw)

        ax.set_ylim(0, 1.0); ax.set_ylabel(state.upper(), fontsize=8, fontweight="bold", rotation=0, labelpad=15, va="center")
        ax.set_xlim(0, 360); ax.spines[['right', 'top', 'left']].set_visible(False); ax.set_yticks([])
        if s_idx < len(STATE_ORDER) - 1: ax.spines['bottom'].set_visible(False); ax.set_xticks([])
        else: ax.set_xticks([0, 180, 360]); ax.set_xlabel("Phase (°)", fontsize=8)

    plt.subplots_adjust(bottom=0.18, hspace=0.25)
    cbar_ax = fig.add_axes([0.25, 0.06, 0.5, 0.02])
    vmax_cbar = 0.35 if variant in ["sqrt_saturated", "high_contrast"] else 0.50
    norm = mcolors.Normalize(vmin=0.0, vmax=vmax_cbar)
    sm = ScalarMappable(norm=norm, cmap=plt.cm.binary)
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label("Coupling Strength (PLV Scale)", fontsize=7, labelpad=3)
    cbar.ax.tick_params(labelsize=6)

    fig.savefig(Path(out_png).with_suffix('.png'), dpi=300, transparent=True)
    fig.savefig(Path(out_png).with_suffix('.svg'), transparent=True)
    plt.close(fig)

# =============================================================================
# MAIN PIPELINE EXECUTION
# =============================================================================
def run_refined_pipeline(db_csv: str, output_root: str):
    print(f"Reading Database: {db_csv}")
    master_df = pd.read_csv(db_csv)
    
    if "state_norm" not in master_df.columns and "state" in master_df.columns:
        master_df["state_norm"] = master_df["state"].astype(str).str.lower().str.strip()
    if "celltype_norm" not in master_df.columns and "celltype" in master_df.columns:
        master_df["celltype_norm"] = master_df["celltype"]

    master_df = master_df[master_df["state_norm"].isin(STATE_ORDER)]
    out_base = Path(output_root)
    
    # Updated to automatically generate all three proposed variant plots sequentially
    raster_variants = ["sqrt_saturated", "binary_mask", "high_contrast"]
    print("Generating Phase Rasters across alternative visibility strategies...")
    for variant in raster_variants:
        print(f" -> Processing variant template: '{variant}'")
        for band in BANDS:
            plot_compact_phase_raster_with_sine(
                master_df, band, str(out_base / "phase_rasters" / f"raster_{band}_{variant}"), variant=variant
            )

    print("Generating Compressed Celltype & Layer Variant Panels...")
    beeswarm_dir = out_base / "refined_beeswarms"
    
    for m_col, m_label in TARGET_METRICS:
        if m_col in master_df.columns:
            master_df[m_col] = pd.to_numeric(master_df[m_col], errors="coerce")
            _, stats_lookup = compute_all_stats(master_df, m_col)
            
            if "celltype_norm" in master_df.columns:
                ct_df = master_df.dropna(subset=[m_col, "state_norm", "celltype_norm"])
                if not ct_df.empty:
                    plot_variant_a_generic(
                        df=ct_df, metric=m_col, ylabel=m_label, stats_lookup=stats_lookup,
                        out_png=str(beeswarm_dir / f"varA_celltype_{m_col}"),
                        group_col="celltype_norm", order_list=CELLTYPE_ORDER, color_dict=COLORS_CELLTYPE
                    )
                    plot_variant_b_unified_with_stats(
                        df=ct_df, metric=m_col, ylabel=m_label, stats_lookup=stats_lookup,
                        out_png=str(beeswarm_dir / f"varB_unified_{m_col}")
                    )

            if "layer_norm" in master_df.columns:
                ly_df = master_df.dropna(subset=[m_col, "state_norm", "layer_norm"])
                if not ly_df.empty:
                    plot_variant_a_generic(
                        df=ly_df, metric=m_col, ylabel=m_label, stats_lookup=stats_lookup,
                        out_png=str(beeswarm_dir / f"varA_layer_{m_col}"),
                        group_col="layer_norm", order_list=LAYER_ORDER, color_dict=COLORS_LAYER
                    )

    print(f"Pipeline Execution Complete. Files saved as SVG/PNG in: {output_root}")

if __name__ == "__main__":
    INPUT_DATABASE = r"E:/in_vivo_in_vitro/stat/statbase/vivo_vitro_database_from_outputs_checked.csv"
    ROOT = r"E:/in_vivo_in_vitro/stat/refactored_publication_panels"
    run_refined_pipeline(INPUT_DATABASE, ROOT)