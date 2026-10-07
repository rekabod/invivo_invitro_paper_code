# Analytical Mesoscopic Framework

> **Structure-Function Paradox**: Independent control of cortical network states via Lyapunov steady-state analysis.

## What This Model Learned

The `meso_model.py` implements an **analytical mesoscopic framework** that optimizes 7 latent parameters to match empirical neural metrics across three brain states (Awake, Sleep, In Vitro). Key findings:

### Core Improvement: Independent Control Axes

The optimizer discovered that **external drive (η<sub>ext</sub>)** and **local coupling (K<sub>local</sub>)** are **independent control axes** — tuning one does not affect the other's contribution to network metrics. This is the "Structure-Function Paradox": the same structural connectivity matrix W can produce entirely different functional states via independent parameter tuning.

### State-Specific Parameter Solutions

| Parameter | Awake | Sleep | In Vitro | Interpretation |
|-----------|-------|-------|----------|----------------|
| λ<sub>exc</sub> | 3.0 mm | 3.0 mm | 3.0 mm | Short axons in all states |
| λ<sub>inh</sub> | 1.65 mm | 1.65 mm | 1.65 mm | Fixed inhibition range |
| W<sub>exc</sub> | 0.81 | 0.82 | 1.03 | Strongest excitation in vitro |
| W<sub>inh</sub> | 0.12 | 0.12 | 0.20 | Strongest inhibition in vitro |
| η<sub>ext</sub> | 0.10 | 0.12 | 0.06 | Weakest external drive in vitro |
| K<sub>local</sub> | 0.05 | 0.08 | 0.25 | **Strongest local coupling in vitro** |
| phase_noise | 0.90 | 0.82 | 0.45 | **Lowest noise in vitro** |

### Key Insights

1. **Millisecond evaluation**: Analytical Lyapunov + saturating PLV → ~1-5 ms per evaluation vs. ~30 s for time-domain simulation (6000× speedup)
2. **Critical transition manifold**: At deafferentation index ~0.5, the network abruptly shifts from integration-dominant (high PR) to resonance-dominant (high PLV)
3. **In vitro state uniqueness**: The in vitro state occupies a narrow, well-defined region in latent parameter space — not a general transition but a specific point
4. **PR overestimation**: The 80-node network simplification causes Participation Ratio to be overestimated (~13-16 vs. empirical ~3-6). Future work should increase network size

## Repository Structure

```
final_model/
├── meso_model.py          # Main model code (v2.8)
├── generate_report.py     # Figure generation script
├── report.html            # 4-panel HTML report
├── empirical_targets.csv  # Empirical neural metrics (Awake/Sleep/Vitro)
├── histology_targets.csv  # Histology-derived targets
├── analytical_state_transition_manifold.csv  # Deafferentation sweep data
├── metrics_v28.csv        # Optimized model metrics
├── figures/
│   ├── panel_a.png        # Framework schematic
│   ├── panel_b.png        # Model fitting accuracy
│   ├── panel_c.png        # Transition manifold
│   └── panel_d.png        # Latent parameter space
└── saved/
    └── opt_v28.json       # Final optimized parameters
```

## Quick Start

```bash
# Run the model
python meso_model.py

# Generate the report figures
python generate_report.py

# Open the HTML report
open report.html   # macOS
xdg-open report.html  # Linux
start report.html  # Windows
```

## Scientific Context

This model addresses the **mesoscopic inverse problem**: given empirical neural metrics (PR, PC1 variance, Population Coupling, Gamma PLV), find the latent structural and dynamical parameters that produce them.

The analytical approach solves the continuous Lyapunov equation:

**AΣ + ΣAᵀ + Q = 0**

where:
- **A** = -I + W<sub>eff</sub> (effective connectivity)
- **Σ** = covariance matrix (output)
- **Q** = η<sub>ext</sub>·uuᵀ + σ<sub>ind</sub>·I (drive matrix)

The saturating PLV formula:

**PLV = (K·scale) / (K·scale + φ + ε)**

avoids the sharp Kuramoto threshold while preserving the phase-locking dynamics.

## References

- meso_model.py: — Local inhibition + saturating gain
- analytical_state_transition_manifold.csv: Deafferentation sweep (0–100%)
- empirical_targets.csv: Empirical metrics from IBL data (RS-PC, IB-PC, FS, IN cell types)

## License

Proprietary — Réka Bod
