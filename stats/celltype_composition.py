"""
Cell-type composition analysis for human cortical microcircuits.

Purpose
-------
Analyze cell-type composition across cortical layers (supragranular, granular,
infragranular) and recording states (awake, sleep, vitro). Perform Fisher's
exact tests to assess significant differences in cell-type proportions.

Outputs
-------
- Cell-type composition plots per layer and state
- Fisher's exact test p-values for pairwise comparisons

Usage
-----
    python -m stats.celltype_composition

Dependencies
------------
    pandas, numpy, matplotlib, scipy.stats.fisher_exact
"""
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import fisher_exact
from itertools import combinations

# =============================================================================
# Configuration
# =============================================================================
DB_PATH = r"E:/in_vivo_in_vitro/stat/statbase/vivo_vitro_database_from_outputs_checked.csv"
OUTPUT_DIR = r"E:/in_vivo_in_vitro/stat/statbase/celltype_composition_new"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Reversed order as requested
CELLTYPE_ORDER = ["FS", "IN", "IB-PC", "RS-PC"]
LAYER_ORDER = ["supra", "gran", "infra"]
STATE_ORDER = ["awake", "sleep", "vitro"]

COLORS_CELLTYPE = {
    "RS-PC": "#D81B60",
    "IB-PC": "#FB8C00",
    "IN": "#1E88E5",
    "FS": "#00897B",
}

def calculate_significance(df, group_col, group_a, group_b, celltype_norm):
    a_ct = len(df[(df[group_col] == group_a) & (df["celltype_norm"] == celltype_norm)])
    a_total = len(df[df[group_col] == group_a])
    b_ct = len(df[(df[group_col] == group_b) & (df["celltype_norm"] == celltype_norm)])
    b_total = len(df[df[group_col] == group_b])
    
    a_other = a_total - a_ct
    b_other = b_total - b_ct
    
    table = [[a_ct, a_other], [b_ct, b_other]]
    _, p = fisher_exact(table)
    return p, a_ct, b_ct, a_total, b_total

def plot_stacked_composition(df, group_col, group_order, filename):
    # Filter and setup categorical order
    df_f = df[df["celltype_norm"].isin(CELLTYPE_ORDER)].copy()
    df_f["celltype_norm"] = pd.Categorical(df_f["celltype_norm"], categories=CELLTYPE_ORDER, ordered=True)
    df_f[group_col] = pd.Categorical(df_f[group_col], categories=group_order, ordered=True)
    
    # Calculate counts and proportions
    counts = pd.crosstab(df_f[group_col], df_f["celltype_norm"])
    proportions = counts.div(counts.sum(axis=1), axis=0)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot
    colors = [COLORS_CELLTYPE[ct] for ct in proportions.columns]
    proportions.plot(kind="bar", stacked=True, ax=ax, color=colors)
    
    # Add Total Counts at top of each bar
    for i, (idx, row) in enumerate(counts.iterrows()):
        total = row.sum()
        ax.text(i, 1.01, f"n={total}", ha='center', fontsize=9, fontweight='bold')
    
    ax.set_title(f"Celltype Composition by {group_col}")
    ax.set_ylabel("Proportion")
    ax.legend(title="Cell Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Significance and count labeling
    groups = [g for g in group_order if g in proportions.index]
    y_max = 1.10 # Lift brackets higher to avoid hitting n= counts
    
    for (g_a, g_b) in combinations(groups, 2):
        sig_info = []
        for ct in CELLTYPE_ORDER:
            if ct in proportions.columns:
                p, ca, cb, ta, tb = calculate_significance(df_f, group_col, g_a, g_b, ct)
                if p < 0.05:
                    sig_info.append((ct, ca, cb))
        
        for k, (ct, ca, cb) in enumerate(sig_info):
            bracket_y = y_max + (k * 0.08)
            color = COLORS_CELLTYPE[ct]
            ax.plot([0, 1], [bracket_y, bracket_y], color=color, lw=2)
            label = f"{ct} (n={ca} vs {cb}) *"
            ax.text(0.5, bracket_y + 0.01, label, color=color, ha="center", fontsize=9, fontweight='bold')
            
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(f"{path}.png", dpi=300)
    plt.savefig(f"{path}.svg")
    plt.close()

if __name__ == "__main__":
    df = pd.read_csv(DB_PATH)
    plot_stacked_composition(df, "layer_norm", LAYER_ORDER, "celltype_proportion_by_layer")
    plot_stacked_composition(df, "state_norm", STATE_ORDER, "celltype_composition_by_state")
    print(f"Plots saved to: {OUTPUT_DIR}")