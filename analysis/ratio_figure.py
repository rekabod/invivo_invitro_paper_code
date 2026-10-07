import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")

# =============================================================================
# DESIGN SYSTEM & CONFIGURATION
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

COLORS_CELLTYPE = {
    "RS-PC": "#EA5189",
    "IB-PC": "#FFB74D",
    "IN": "#3D5AFE",
    "FS": "#00BFA5",
}

# Critical Quality Filter: Exclude recordings with sparse yields to avoid ratio inflation
MIN_CELLS_PER_REC = 3 

# =============================================================================
# DATA PROCESSING PIPELINE
# =============================================================================
def prepare_composition_dataframe(df):
    """Computes mathematically valid cell-type proportions per recording session."""
    # Standardize naming keys
    if "state_norm" not in df.columns and "state" in df.columns:
        df["state_norm"] = df["state"].astype(str).str.lower().str.strip()
    if "celltype_norm" not in df.columns and "celltype" in df.columns:
        df["celltype_norm"] = df["celltype"]
        
    df = df[df["state_norm"].isin(STATE_ORDER) & df["celltype_norm"].isin(CELLTYPE_ORDER)]
    
    # 1. Compute global cell counts for right-hand reporting panels
    global_counts = df.groupby(["state_norm", "celltype_norm"]).size().to_dict()
    
    # 2. Compute recording-level cell type matrices
    counts_matrix = df.groupby(["state_norm", "recording_id", "celltype_norm"]).size().unstack(fill_value=0)
    total_cells_per_rec = counts_matrix.sum(axis=1)
    
    # Filter out statistically noisy, low-yield recordings
    valid_recordings = total_cells_per_rec[total_cells_per_rec >= MIN_CELLS_PER_REC].index
    counts_matrix = counts_matrix.loc[valid_recordings]
    total_cells_per_rec = total_cells_per_rec.loc[valid_recordings]
    
    # Compute true localized ratios
    ratios_matrix = counts_matrix.div(total_cells_per_rec, axis=0).reset_index()
    
    # Melt back to long-form format for clean visualization loops
    melted_ratios = ratios_matrix.melt(
        id_vars=["state_norm", "recording_id"], 
        value_vars=CELLTYPE_ORDER, 
        var_name="celltype_norm", 
        value_name="ratio"
    )
    
    return melted_ratios, global_counts

# =============================================================================
# HORIZONTAL PLOTTING ENGINE
# =============================================================================
def plot_horizontal_composition(melted_df, global_counts, out_path):
    """Generates a highly-compact horizontal box-scatter hybrid with right-anchored counts."""
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    
    # Custom tight vertical coordinate positioning framework
    state_offsets = {"awake": 8.0, "sleep": 4.5, "vitro": 1.0}
    spacing = 0.35  # Compressed inner-cell spacing
    
    y_ticks_positions = []
    y_ticks_labels = []
    
    for state in STATE_ORDER:
        base_y = state_offsets[state]
        state_data = melted_df[melted_df["state_norm"] == state]
        
        # Calculate group center for clean state labeling
        state_center_y = base_y - ((len(CELLTYPE_ORDER) - 1) * spacing) / 2.0
        y_ticks_positions.append(state_center_y)
        y_ticks_labels.append(state.upper())
        
        # Draw clean visual bounding separators between behavioral/experimental states
        ax.axhspan(base_y + 0.25, base_y - (len(CELLTYPE_ORDER) * spacing) + 0.1, 
                   color="#F9F9F9", alpha=0.5, zorder=-5)
        
        for c_idx, ct in enumerate(CELLTYPE_ORDER):
            y_pos = base_y - (c_idx * spacing)
            ratios = state_data[state_data["celltype_norm"] == ct]["ratio"].values
            
            if len(ratios) > 0:
                # Horizontal low-contrast box distribution
                ax.boxplot(ratios, positions=[y_pos], vert=False, widths=0.22,
                           patch_artist=True, showfliers=False, zorder=1,
                           boxprops=dict(facecolor="#FFFFFF", edgecolor="#DCDCDC", lw=0.8),
                           whiskerprops=dict(color="#DCDCDC", lw=0.6),
                           capprops=dict(color="#DCDCDC", lw=0.6),
                           medianprops=dict(color="#444444", lw=1.2))
                
                # Jittered scatter overlay mapping individual recording ratios
                ax.scatter(ratios, np.random.normal(y_pos, 0.03, len(ratios)),
                           facecolors='none', edgecolors=COLORS_CELLTYPE[ct],
                           s=14, alpha=0.7, linewidth=0.5, zorder=2)
            
            # Anchor absolute cell count metadata (N) on the right margin
            cell_pool_count = global_counts.get((state, ct), 0)
            ax.text(1.04, y_pos, f"{ct} (n={cell_pool_count})", 
                    va='center', ha='left', fontsize=7.5, 
                    color=COLORS_CELLTYPE[ct], fontweight='bold', clip_on=False)

    # Layout tuning and clean publication polishing
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Cell Type Proportion per Recording", fontweight="bold", labelpad=8)
    
    ax.set_yticks(y_ticks_positions)
    ax.set_yticklabels(y_ticks_labels, fontweight="bold")
    ax.tick_params(axis='y', which='both', length=0)  # Remove y-ticks lines to preserve clean structure
    
    ax.spines[['right', 'top', 'left']].set_visible(False)
    ax.spines['bottom'].set_color('#CCCCCC')
    
    plt.tight_layout()
    
    # Export dual vector/raster targets
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p.with_suffix('.png'), dpi=300, transparent=True)
    fig.savefig(p.with_suffix('.svg'), transparent=True)
    plt.close(fig)

# =============================================================================
# MAIN RUNTIME EXECUTION
# =============================================================================
if __name__ == "__main__":
    DATABASE_PATH = r"E:\in_vivo_in_vitro\stat\statbase\vivo_vitro_database_from_outputs_checked.csv"
    OUTPUT_FILE = r"E:\in_vivo_in_vitro\stat\refactored_publication_panels\network_dimensionality\network_celltype_composition"
    
    print(f"Reading master database: {DATABASE_PATH}")
    master_df = pd.read_csv(DATABASE_PATH)
    
    print("Extracting recording compositions...")
    ratios_df, total_counts = prepare_composition_dataframe(master_df)
    
    print("Generating refined horizontal publication panel...")
    plot_horizontal_composition(ratios_df, total_counts, OUTPUT_FILE)
    print(f"Success. Network composition panels exported to: {OUTPUT_FILE}.png/.svg")