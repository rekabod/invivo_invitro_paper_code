import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set Nature Neuroscience design theme
sns.set_theme(style="white")
plt.rcParams.update({
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.grid': False,
    'figure.dpi': 300,
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.family': 'sans-serif',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'svg.fonttype': 'none'
})

# Color palette definition
BAND_COLORS = {
    'delta': '#2196F3',
    'theta': '#4CAF50',
    'lowgamma': '#FF9800',
    'highgamma': '#E91E63'
}

BAND_ORDER = ['delta', 'theta', 'lowgamma', 'highgamma']
STATE_ORDER = ['awake', 'sleep', 'vitro']

# 1. Load Data
df = pd.read_csv(r"E:\in_vivo_in_vitro\stat\epoch_plots\lfp_band_epoch_summary_table.csv")

# -------------------------------------------------------------
# FIGURE 1: State x Band Grouped Hollow Boxplot with Stripplot
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

sns.boxplot(
    data=df,
    x='state',
    y='proportion',
    hue='band',
    order=STATE_ORDER,
    hue_order=BAND_ORDER,
    palette=BAND_COLORS,
    boxprops={'facecolor': 'none', 'linewidth': 1.2},
    whiskerprops={'linewidth': 1.2},
    capprops={'linewidth': 1.2},
    medianprops={'linewidth': 1.5},
    showfliers=False,
    ax=ax
)

# Overlay individual recordings as stripplot
sns.stripplot(
    data=df,
    x='state',
    y='proportion',
    hue='band',
    order=STATE_ORDER,
    hue_order=BAND_ORDER,
    palette=BAND_COLORS,
    dodge=True,
    alpha=0.6,
    size=5,
    jitter=0.2,
    rasterized=True,
    ax=ax
)

# Clean duplicate legend entries from stripplot
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:4], labels[:4], title="LFP Band", frameon=False, loc="upper left")

ax.set_ylim(-0.05, 1.05)
ax.set_xlabel("Operating State", fontweight='bold')
ax.set_ylabel("Proportion of Significant Duration", fontweight='bold')
ax.set_title("LFP Band Epoch Proportions across States", pad=12, fontweight='bold')

plt.tight_layout()
fig.savefig("E:/in_vivo_in_vitro/stat/epoch_plots/fig1_lfp_proportions_state_band.png", dpi=300)
fig.savefig("E:/in_vivo_in_vitro/stat/epoch_plots/fig1_lfp_proportions_state_band.svg")
plt.close()

# -------------------------------------------------------------
# FIGURE 2: Patientwise Trajectories across States per Band
# -------------------------------------------------------------
patient_means = df.groupby(['patient', 'state', 'band'])['proportion'].mean().reset_index()

fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharey=True)

for i, band in enumerate(BAND_ORDER):
    ax = axes[i]
    band_df = patient_means[patient_means['band'] == band]
    
    # Plot individual patient lines
    sns.lineplot(
        data=band_df,
        x='state',
        y='proportion',
        units='patient',
        estimator=None,
        color=BAND_COLORS[band],
        alpha=0.4,
        marker='o',
        linewidth=1,
        ax=ax
    )
    
    # Plot population mean line
    pop_mean = band_df.groupby('state')['proportion'].mean().reindex(STATE_ORDER)
    ax.plot(STATE_ORDER, pop_mean, color=BAND_COLORS[band], linewidth=2.5, marker='s', markersize=6, label='Population Mean')
    
    ax.set_title(f"{band.capitalize()} Band", color=BAND_COLORS[band], fontweight='bold')
    ax.set_xlabel("State")
    if i == 0:
        ax.set_ylabel("Mean Proportion")
    ax.set_ylim(-0.05, 1.05)

plt.suptitle("Patient-Level Trajectories Across Operating States", y=1.02, fontweight='bold')
plt.tight_layout()
fig.savefig("E:/in_vivo_in_vitro/stat/epoch_plots/fig2_patient_trajectories_by_band.png", dpi=300)
fig.savefig("E:/in_vivo_in_vitro/stat/epoch_plots/fig2_patient_trajectories_by_band.svg")
plt.close()