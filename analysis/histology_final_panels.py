import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from scipy.stats import gaussian_kde, mannwhitneyu
from matplotlib.patches import Polygon

# --- CONFIGURATION ---
INPUT_PATH = r"C:\Users\bodreka\Downloads\table_histology.csv"
OUTPUT_DIR = Path(r"C:\Users\bodreka\Downloads\plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STATE_PALETTE = {'vivo': '#E0E0E0', 'vitro': '#757575'}
STATE_ORDER = ['vivo', 'vitro']

def load_and_prep(path):
    df = pd.read_csv(path)
    return df.rename(columns={'density (cell/mm^2)': 'density', 'coverage (%)': 'coverage'})

def full_violin(ax, values, pos, color='#999999', width=0.35, bw_method='scott', zorder=2):
    """
    Draw a symmetrical full violin with rounded ends using KDE.
    """
    values = pd.Series(values).dropna().values
    if len(values) < 2:
        return

    kde = gaussian_kde(values, bw_method=bw_method)
    y_min, y_max = values.min(), values.max()
    y_pad = (y_max - y_min) * 0.15 if y_max > y_min else 1.0
    y = np.linspace(y_min - y_pad, y_max + y_pad, 200)
    dens = kde(y)
    dens = dens / dens.max() * width

    # Symmetrical left and right boundaries centered on 'pos'
    x_left = pos - dens
    x_right = pos + dens

    verts = np.concatenate([
        np.column_stack([x_left, y]),
        np.column_stack([x_right[::-1], y[::-1]])
    ])

    poly = Polygon(verts, closed=True, facecolor=color, edgecolor='black', linewidth=1, alpha=0.85, zorder=zorder)
    ax.add_patch(poly)

def add_boxquartiles(ax, values, pos, color='black', width=0.08):
    """
    Draws median and quartile markers centered inside the full violin.
    """
    values = pd.Series(values).dropna().values
    if len(values) == 0:
        return
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    
    # Median line
    ax.plot([pos - width, pos + width], [med, med], color=color, lw=1.5, zorder=4)
    # Q1 and Q3 lines
    ax.plot([pos - width*0.6, pos + width*0.6], [q1, q1], color=color, lw=1.0, alpha=0.8, zorder=4)
    ax.plot([pos - width*0.6, pos + width*0.6], [q3, q3], color=color, lw=1.0, alpha=0.8, zorder=4)
    # Vertical connector bridging Q1 and Q3
    ax.plot([pos, pos], [q1, q3], color=color, lw=1.0, alpha=0.8, zorder=4)

def plot_metric(df, staining, metric, layer, y_max):
    """
    Plots a specific staining + metric combination, forcing a unified Y max,
    and testing for significance (p < 0.04).
    """
    sub = df[(df['staining'] == staining) & (df['layer'] == layer)].copy()
    sub = sub.dropna(subset=[metric])
    
    fig, ax = plt.subplots(figsize=(4.5, 5))
    positions = np.arange(len(STATE_ORDER))
    
    # Store arrays for statistical testing later
    state_data = []
    
    for i, state in enumerate(STATE_ORDER):
        vals = sub.loc[sub['state'] == state, metric].dropna().values
        state_data.append(vals)
        
        # Full KDE Violin 
        full_violin(ax, vals, pos=i, color=STATE_PALETTE[state], width=0.35)
        add_boxquartiles(ax, vals, pos=i, color='black')
        
        # (Scatter point drawing logic removed)
            
    # Add Significance lines if p < 0.04
    if len(state_data) == 2 and len(state_data[0]) >= 2 and len(state_data[1]) >= 2:
        stat, p_val = mannwhitneyu(state_data[0], state_data[1], alternative='two-sided')
        
        if p_val < 0.04:
            # Format significance asterisks
            sig_stars = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
            sig_text = f"{sig_stars}\n(p={p_val:.3f})"
            
            # Position bracket near the very top of the plot inside the unified y_max
            x1, x2 = 0, 1
            y_bar = y_max * 0.88
            y_tick = y_max * 0.02
            
            # Draw horizontal and vertical bracket lines
            ax.plot([x1, x1, x2, x2], [y_bar, y_bar+y_tick, y_bar+y_tick, y_bar], lw=1.2, color='black', zorder=5)
            # Annotate text right above the center of the bracket
            ax.text((x1+x2)*0.5, y_bar+y_tick+ (y_max*0.01), sig_text, 
                    ha='center', va='bottom', color='black', fontsize=9, zorder=5)

    # X-Axis formatting
    ax.set_xticks(positions)
    ax.set_xticklabels(STATE_ORDER)
    ax.set_xlabel("")
    ax.set_xlim(-0.5, len(STATE_ORDER) - 0.5)
    
    # Y-Axis formatting (Unified maximum bounds applied here)
    unit = " (cell/mm²)" if metric == "density" else " (%)"
    ax.set_ylabel(f"{staining} {metric}{unit}")
    ax.set_ylim(0, y_max)
    
    ax.set_title(f"{staining} {metric.capitalize()} - {layer.capitalize()}")
    sns.despine(ax=ax)
    plt.tight_layout()
    
    name = f"{staining}_{metric}_{layer}_full_violin_unified_axis"
    plt.savefig(OUTPUT_DIR / f"{name}.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / f"{name}.svg", bbox_inches='tight')
    plt.close()

# --- EXECUTION ---
df = load_and_prep(INPUT_PATH)

# 1. Calculate global max values across layers for uniform y-axes
max_vals = {}
for staining in ['NEUN', 'PV']:
    max_vals[staining] = {}
    for metric in ['density', 'coverage']:
        vals = df[df['staining'] == staining][metric].dropna()
        if len(vals) > 0:
            # 25% padding added to ensure there is plenty of room at the top for the significance brackets
            max_vals[staining][metric] = vals.max() * 1.25

# 2. Generate plots explicitly for the 3 conditions
for layer in ['supra', 'infra']:
    
    # NeuN Density Plot
    if 'density' in max_vals['NEUN']:
        plot_metric(df, 'NEUN', 'density', layer, max_vals['NEUN']['density'])
        
    # PV Density Plot
    if 'density' in max_vals['PV']:
        plot_metric(df, 'PV', 'density', layer, max_vals['PV']['density'])
        
    # PV Coverage Plot
    if 'coverage' in max_vals['PV']:
        plot_metric(df, 'PV', 'coverage', layer, max_vals['PV']['coverage'])

print("All synchronized plots generated successfully with p < 0.04 significance evaluation.")