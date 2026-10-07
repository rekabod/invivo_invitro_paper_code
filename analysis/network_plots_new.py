import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from itertools import combinations
from pathlib import Path

# =============================================================================
# DESIGN & PUBLICATION STYLES (Nature Neuroscience Standard)
# =============================================================================
sns.set_theme(style="white")
plt.rcParams.update({
    'axes.grid': False,
    'figure.dpi': 300,
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.family': 'sans-serif',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'svg.fonttype': 'none'
})

STATE_ORDER = ["awake", "sleep", "vitro"]
STATE_COLORS = {"awake": "#8D8D8DF4", "sleep": "#4C4C4C", "vitro": "#000000"}

# =============================================================================
# DATA PREPARATION & STATS
# =============================================================================
file_path = Path('E:/in_vivo_in_vitro/stat/statbase/vivo_vitro_database_from_outputs_checked.csv')
df = pd.read_csv(file_path)

METRICS = [
    'dimensionality_pc1_variance',
    'dimensionality_participation_ratio',
    'dimensionality_eigenspectrum_entropy'
]

# ---------------------------------------------------------------------------
# STEP 1: Aggregate to recording-level statistics (mean per recording_id/state)
# ---------------------------------------------------------------------------
df_recording = (
    df.groupby(['recording_id', 'state'])[METRICS]
      .mean()
      .reset_index()
)

# ---------------------------------------------------------------------------
# STEP 2: Compute state-level mean + SEM from the recording-level means
# ---------------------------------------------------------------------------
mean_df = df_recording.groupby('state')[METRICS].mean().reset_index()
sem_df  = df_recording.groupby('state')[METRICS].sem().reset_index()
# Rename metric columns, keep 'state' as-is
for c in METRICS:
    mean_df = mean_df.rename(columns={c: f'{c}_mean'})
    sem_df  = sem_df.rename(columns={c: f'{c}_sem'})
df_state_stats = pd.merge(mean_df, sem_df, on='state')

# Print recording-level summary for verification
print("=" * 70)
print("RECORDING-LEVEL STATISTICS (mean of measurements per recording)")
print("=" * 70)
for metric in METRICS:
    print(f"\n--- {metric} ---")
    # Count recordings per state
    n_per_state = df_recording.groupby('state')['recording_id'].nunique()
    print(f"  Recordings per state: {n_per_state.to_dict()}")
    print(f"  State-level mean ± SEM:")
    for state in STATE_ORDER:
        row = df_state_stats[df_state_stats['state'] == state]
        if not row.empty:
            m = row[f'{metric}_mean'].values[0]
            s = row[f'{metric}_sem'].values[0]
            n = int(n_per_state.get(state, 0))
            print(f"    {state:8s}: mean={m:10.4f}, sem={s:10.4f}, n={n}")
print("\n" + "=" * 70)

# Execute Pipeline
output_path = Path("E:/in_vivo_in_vitro/stat/statbase/plots")
output_path.mkdir(exist_ok=True)

def run_stats_and_plot(data, metric, output_dir):
    """
    Plot recording-level mean ± SEM (not raw boxplots).
    Each point = one recording's mean; error bars = SEM across recordings.
    """
    # Setup Plot
    fig, ax = plt.subplots(figsize=(3.5, 4.5))

    # --- 1. Compute stats for error bars ---
    means = data.groupby('state')[metric].mean()
    se    = data.groupby('state')[metric].sem()

    # --- 2. Bar plot: recording-level mean ± SEM ---
    x_pos = [STATE_ORDER.index(s) for s in STATE_ORDER]
    bars = ax.bar(
        x_pos,
        [means[s] for s in STATE_ORDER],
        yerr=[[se[s] for s in STATE_ORDER], [se[s] for s in STATE_ORDER]],
        capsize=4,
        width=0.6,
        color=[STATE_COLORS[s] for s in STATE_ORDER],
        edgecolor='black',
        linewidth=1.2,
        alpha=0.85,
        zorder=3
    )

    # --- 3. Overlay individual recording-level means as points ---
    jitter_vals = [-0.15, 0.0, 0.15]
    for i, state in enumerate(STATE_ORDER):
        subset = data[data['state'] == state]['recording_id'].nunique()
        if subset > 0:
            ax.scatter(
                [x_pos[i]] * subset,
                data[data['state'] == state][metric].values,
                s=30,
                facecolors='none',
                edgecolors=STATE_COLORS[state],
                linewidths=1.0,
                alpha=0.7,
                zorder=4
            )

    # --- 4. Stats: Pairwise Mann-Whitney with FDR (on recording-level means) ---
    groups = STATE_ORDER
    comparisons = list(combinations(groups, 2))
    p_values = []

    for g1, g2 in comparisons:
        vals1 = data[data['state'] == g1][metric].dropna()
        vals2 = data[data['state'] == g2][metric].dropna()
        n1, n2 = len(vals1), len(vals2)
        _, p_val = mannwhitneyu(vals1, vals2, alternative='two-sided')
        p_values.append(p_val)

    reject, q_values, _, _ = multipletests(p_values, method='fdr_bh')

    # --- 5. Add significance brackets ---
    y_max = data[metric].max()
    y_range = data[metric].max() - data[metric].min()
    y_offset = y_range * 0.05

    for i, (g1, g2) in enumerate(comparisons):
        if q_values[i] < 0.05:
            x1, x2 = groups.index(g1), groups.index(g2)
            y = y_max + (i + 1) * y_offset
            ax.plot([x1, x2], [y, y], lw=1.5, c='k')
            ax.plot([x1, x1], [y_max + i * y_offset, y], lw=1.5, c='k')
            ax.plot([x2, x2], [y_max + i * y_offset, y], lw=1.5, c='k')

            sig = "***" if q_values[i] < 0.001 else "**" if q_values[i] < 0.01 else "*"
            ax.text(
                (x1 + x2) * .5, y, sig,
                ha='center', va='bottom',
                fontsize=14, fontweight='bold'
            )

    # --- 6. Axis labels & styling ---
    ax.set_title(metric.replace("dimensionality_", "").replace("_", " ").title())
    ax.set_xlabel("")
    ax.set_ylabel("Value")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(STATE_ORDER)

    # Ensure error bars are visible (adjust ylim if needed)
    max_err = max(se)
    ylim_top = max(y_max, max(means) + max_err) * 1.1
    ax.set_ylim(bottom=min(data[metric].min(), min(means) - max_err) * 0.95, top=ylim_top)

    sns.despine()
    plt.tight_layout()

    # Save as SVG/PNG
    fig.savefig(output_dir / f"{metric}_updated.svg")
    fig.savefig(output_dir / f"{metric}_updated.png", dpi=300)
    plt.close()
    print(f"  Generated plot for {metric}")

# Execute Pipeline
for metric in METRICS:
    run_stats_and_plot(df_recording, metric, output_path)

print("\nAll plots generated successfully.")
