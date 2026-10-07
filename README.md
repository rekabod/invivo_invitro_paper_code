# Cortical Isolation Separates Rhythmic Synchrony from Network Integration in the Human Neocortex

> **Code & data repository** for the paper: *"Cortical isolation separates rhythmic synchrony from network integration in the human neocortex"*

## Overview

This repository contains the complete code, computational models, data processing pipelines, figures, and statistical analyses for the study of cortical state transitions (in vivo awake → in vivo sleep → in vitro slice).

### Scientific Goal

We investigate how **cortical isolation** — the disconnection of long-range inputs in brain slice preparations — separates two distinct phenomena:
- **Rhythmic synchrony** (gamma oscillations) — which *persists* in vitro
- **Network integration** (population coupling, graph connectivity, dimensionality) — which *collapses* in vitro

## Repository Structure

```
├── preprocessing/       # Data preprocessing scripts (EEG/ECoG)
├── analysis/            # Core analysis scripts
│   ├── cluster_plots.py            # Cell-type cluster scatter plots
│   ├── histology_final_panels.py   # Histology violin panels (NEUN/PV)
│   ├── network_plots_new.py        # Network dimensionality plots
│   ├── ratio_figure.py             # Cell-type composition ratios
│   ├── epoch_ratio_plots.py        # LFP band epoch proportions
│   ├── statsplots_single_values.py # Phase rasters & cell-type panels
│   ├── epoch_detection_calibration_10min.py  # LFP epoch detection (Youden's J)
│   ├── gamma_csd_laminar_analysis.py  # Laminar CSD analysis
│   ├── cell_metric_extractor.py    # Single-cell ephys metrics
├── models/              # Computational models
│   ├── refine_mesoscale.py         # Analytical mesoscopic model (final)
│   ├── figures_final/              # Publication figures
│   ├── saved_final/                # Optimized parameters & metrics
├── stats/               # Statistical analysis scripts
│   ├── all_basic_stat_fdr.py       # Basic stats with FDR correction
│   ├── celltype_composition.py     # Cell-type composition analysis
│   ├── ep_stat_new.py              # Comprehensive ephys statistics
│   ├── histo_stat.py               # Histology curve statistics (ANOVA)
│   ├── histology_ephys_analysis.py # Histology–ephys correlation pipeline
│   ├── mixedlm_histo_compact.py    # Linear mixed-effects models
│   ├── plv_classifier.py           # PLV circular statistics classifier
├── figures/             # Generated figures and visualizations
├── notebooks/           # Jupyter notebooks for exploratory analysis
├── data/                # Raw and processed data (gitignored)
├── requirements.txt     # Python dependencies
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/rekabod/invivo_invitro_paper_code.git
cd invivo_invitro_paper_code
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Computational Model: Analytical Mesoscopic Framework

The **analytical mesoscopic model** (`models/refine_mesoscale.py`) implements an analytical framework that optimizes 6 latent parameters to match empirical neural metrics across three brain states (Awake, Sleep, In Vitro).

### Key Findings

1. **Independent Control Axes**: External drive (η_ext) and local coupling are independent control axes — the same structural connectivity matrix W can produce entirely different functional states via independent parameter tuning.

2. **Critical Transition Manifold**: At deafferentation index ~0.5, the network abruptly shifts from integration-dominant (high PR) to resonance-dominant (high PLV).

3. **In Vitro State Uniqueness**: The in vitro state occupies a narrow, well-defined region in latent parameter space.

4. **Millisecond Evaluation**: Analytical Lyapunov + saturating PLV → ~1-5 ms per evaluation vs. ~30 s for time-domain simulation (6000× speedup).

### Latent Parameters (Optimized)

| Parameter | Awake | Sleep | In Vitro | Interpretation |
|-----------|-------|-------|----------|----------------|
| η_ext | External drive | External drive | External drive | Weakest in vitro |
| W_E | Excitation strength | Excitation strength | Strongest in vitro | |
| W_I | Inhibition strength | Inhibition strength | Strongest in vitro | |
| PV_surv | PV axonal survival | PV axonal survival | Lowest in vitro | |
| G_res | Resonance gain | Resonance gain | Highest in vitro | |
| σ_obs | Observation noise | Observation noise | Lowest in vitro | |

### Running the Model

```bash
cd models
python refine_mesoscale.py
```

**Outputs:**
- `figures_final/` — Publication-ready figures (Fig1a–e)
- `saved_final/` — Optimized parameters, metrics, deafferentation sweep

### Figures

| Figure | Description |
|--------|-------------|
| Fig1a_metrics.pdf | Empirical vs. model bars (PopC, PLV low-γ, PLV high-γ) |
| Fig1b_parameters.pdf | Latent parameter heatmap |
| Fig1c_Sankey_{state}.pdf | Spacious coloured Sankey diagrams (one per state) |
| Fig1d_deafferentation.pdf | Continuous deafferentation trajectory |
| Fig1e_sobol.pdf | Sobol sensitivity indices |

## Analysis Scripts

### LFP Epoch Detection (`analysis/epoch_detection_calibration_10min.py`)

Batch LFP epoch detection using sliding Welch PSD with Youden's J threshold calibration from paired sleep + awake reference recordings. Supports per-recording gamma threshold rescaling and Hilbert-based boundary refinement.

### Laminar CSD Analysis (`analysis/gamma_csd_laminar_analysis.py`)

Laminar Current Source Density analysis across cortical layers (supragranular, granular, infragranular) to identify depth-specific current flow during gamma oscillations.

### Single-Cell Metrics (`analysis/cell_metric_extractor.py`)

Extract and compute cell-level metrics from raw electrophysiology recordings including firing rates, burstiness, inter-spike interval statistics, phase-locking values, population coupling, and dimensionality measures.

### Statistical Analysis (`stats/`)

| Script | Purpose |
|--------|---------|
| `all_basic_stat_fdr.py` | Basic stats with Benjamini-Hochberg FDR correction |
| `celltype_composition.py` | Cell-type composition analysis with Fisher's exact tests |
| `ep_stat_new.py` | Comprehensive ephys statistics with publication-ready Excel export |
| `histo_stat.py` | Histology curve statistics (ANOVA, Kruskal-Wallis, post-hoc) |
| `histology_ephys_analysis.py` | Histology–ephys correlation pipeline with mixed-effects models |
| `mixedlm_histo_compact.py` | Linear mixed-effects models for histology data |
| `plv_classifier.py` | Advanced circular statistics for spike-phase relationships |

### Figure Generation (`analysis/`)

| Script | Purpose |
|--------|---------|
| `cluster_plots.py` | Cell-type cluster scatter plots (halfwidth vs. repolarization) |
| `histology_final_panels.py` | Full violin panels for NEUN/PV density & coverage |
| `network_plots_new.py` | Network dimensionality plots (PC1 variance, PR, entropy) |
| `ratio_figure.py` | Horizontal cell-type composition box-scatter hybrid |
| `epoch_ratio_plots.py` | LFP band epoch proportions across states |
| `statsplots_single_values.py` | Phase rasters & compressed cell-type panels |

## Reproducing Results

### Figures

Run the appropriate analysis or model script to regenerate figures. Most scripts output both PNG (300 dpi) and SVG formats.

### Statistical Analysis

The `stats/` scripts produce Excel workbooks and CSV outputs with FDR-corrected p-values, effect sizes, and publication-ready formatting.

## Citation

```bibtex
@article {Bod2026.09.02.748777,
	author = {Bod, R{\'e}ka and T{\'o}th, Kinga and Farkas, Orsolya and Michaeli, Yossef and T{\'o}th, Katalin Zs{\'o}fia and Kandr{\'a}cs, {\'A}gnes and Hofer, Katharina T. and Fab{\'o}, D{\'a}niel and Hajnal, Bogl{\'a}rka and Szab{\'o}, Johanna Petra and Er{\H o}ss, Lor{\'a}nd and Entz, L{\'a}szl{\'o} and Ulbert, Istv{\'a}n and Wittner, Lucia},
	title = {Cortical isolation separates rhythmic synchrony from network integration in the human neocortex},
	elocation-id = {2026.09.02.748777},
	year = {2026},
	doi = {10.64898/2026.09.02.748777},
	URL = {https://www.biorxiv.org/content/early/2026/09/06/2026.09.02.748777},
	eprint = {https://www.biorxiv.org/content/early/2026/09/06/2026.09.02.748777.full.pdf},
	journal = {bioRxiv}
}
```

## License

MIT license

## Contact

For questions about the code or the paper, please open an issue on GitHub or contact the authors.
