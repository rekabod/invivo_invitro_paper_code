import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load and prepare data
df = pd.read_csv("C:/Users/bodreka/Desktop/vitro_units.csv")
df_clean = df.dropna(subset=['halfwidth1_ms', 'halfwidth2_ms', 'Celltype']).copy()

# 2. Define NNS style and colors
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

colors = {
    "RS-PC": "#EA5189",
    "IB-PC": "#FFB74D",
    "IN": "#3D5AFE",
    "FS": "#00BFA5",
    "unclassified": "grey"
}

# 3. Plotting
fig, ax = plt.subplots(figsize=(6, 5))

# Plot defined types
for ctype, color in colors.items():
    if ctype != "unclassified":
        subset = df_clean[df_clean['Celltype'] == ctype]
        ax.scatter(subset['halfwidth2_ms'], subset['halfwidth1_ms'], 
                   label=ctype, color=color, alpha=0.7, edgecolors='white', s=40)

# Plot unclassified/others in grey
unclassified = df_clean[~df_clean['Celltype'].isin([k for k in colors.keys() if k != "unclassified"])]
ax.scatter(unclassified['halfwidth2_ms'], unclassified['halfwidth1_ms'], 
           label='unclassified', color='grey', alpha=0.7, edgecolors='white', s=40)

# Formatting
ax.set_xlabel('Halfwidth 2 (ms) - Repolarization slope')
ax.set_ylabel('Halfwidth 1 (ms) - Main halfwidth')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(frameon=False)

plt.tight_layout()

# Save outputs
fig.savefig('C:/Users/bodreka/Desktop/halfwidth_scatter_vitro.png')
fig.savefig('C:/Users/bodreka/Desktop/halfwidth_scatter_vitro.svg')
plt.close()