#!/usr/bin/env python3
"""
analytical_mesoscopic_minimal_FINAL.py
Minimal mechanistic analytical model of the Structure–Function Paradox

Final integrated version
------------------------
• Experimental-style population coupling (corr with population mean)
• Explicit gamma-resonance module → low-γ & high-γ PLV
• Six biologically interpretable latent parameters
• Plotly Sankey (spacious, coloured ribbons) – one per state
• Continuous de-afferentation trajectory
• Sobol sensitivity
• Publication-ready metric & parameter figures
"""

import numpy as np
import pandas as pd
import json, time, os, warnings
from pathlib import Path
from scipy.linalg import solve_continuous_lyapunov
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist

# Sobol (optional)
try:
    from SALib.sample.sobol import sample as sobol_sample
    from SALib.analyze import sobol
    HAS_SALIB = True
except ImportError:
    HAS_SALIB = False
    print("SALib not installed – Sobol analysis will be skipped.")

import plotly.graph_objects as go

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Style
# ------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# User-specified Sankey colours
PARAM_COLORS = ["#0cb2af", "#a1c65d", "#fac723", "#f29222", "#e95e50", "#936fac"]
TARGET_COLORS = ["#A3A3A3", "#7B7878", "#575756"]

OUT = Path(os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else ".")
FIG = OUT / "figures_final"
SAV = OUT / "saved_final"
FIG.mkdir(exist_ok=True)
SAV.mkdir(exist_ok=True)

np.random.seed(42)
N_NODES   = 80
SIGMA_IND = 0.05
STATES    = ["awake", "sleep", "vitro"]
STATE_LAB = {"awake": "Awake", "sleep": "Sleep", "vitro": "In Vitro"}

# Empirical targets (primary: PopC + two PLVs)
EMP = {
    "awake": dict(plv_low_gamma=0.0295, plv_high_gamma=0.0173,
                  population_coupling=0.1560, pc1_variance=0.3540,
                  participation_ratio=5.6593),
    "sleep": dict(plv_low_gamma=0.0830, plv_high_gamma=0.0915,
                  population_coupling=0.2063, pc1_variance=0.4017,
                  participation_ratio=4.8130),
    "vitro": dict(plv_low_gamma=0.2328, plv_high_gamma=0.2125,
                  population_coupling=0.0777, pc1_variance=0.4688,
                  participation_ratio=3.6305),
}

# =====================================================================
# 1. Geometry & connectivity (PV axonal truncation)
# =====================================================================
def generate_nodes(n, rng):
    supra = int(n * 0.645)
    nodes = []
    for i in range(n):
        layer = "supra" if i < supra else "infra"
        side  = 0.36 if layer == "supra" else 0.46
        nodes.append(dict(
            x=rng.uniform(-side/2, side/2),
            y=rng.uniform(-side/2, side/2),
            z=rng.uniform(0, 1) if layer == "supra" else rng.uniform(-1, 0),
            layer=layer
        ))
    return nodes

def build_connectivity(nodes, W_E, W_I, PV_surv, lam_E=50.0, lam_I0=27.0):
    pos = np.array([[nd["x"], nd["y"], nd["z"]] for nd in nodes])
    D   = cdist(pos, pos)
    Wexc = W_E * np.exp(-D / max(lam_E, 1e-3))
    np.fill_diagonal(Wexc, 0.0)
    # PV axonal survival shortens and weakens inhibition
    lam_I = lam_I0 * (0.30 + 0.70 * PV_surv)
    Winh  = W_I * np.exp(-D / max(lam_I, 1e-3)) * (0.35 + 0.65 * PV_surv)
    np.fill_diagonal(Winh, 0.0)
    W = 2.0 * np.tanh((Wexc - Winh) / 2.0)
    return W, Wexc, Winh

# =====================================================================
# 2. Lyapunov + experimental-style PopC
# =====================================================================
def lyapunov_metrics(W, eta_ext, drive_mask, sigma_ind=SIGMA_IND):
    n = W.shape[0]
    rho = np.linalg.norm(W, 2)
    Wef = W * (0.975 / rho) if rho > 0.975 else W
    A   = -np.eye(n) + Wef

    u = np.zeros(n)
    u[drive_mask] = 1.0
    u /= (np.linalg.norm(u) + 1e-12)

    Q = eta_ext * np.outer(u, u) + sigma_ind * np.eye(n)
    try:
        Sig = solve_continuous_lyapunov(A, -Q)
        Sig = 0.5 * (Sig + Sig.T)
        ev, evecs = np.linalg.eigh(Sig)
        ev = np.maximum(ev, 0.0)
        Sig = evecs @ np.diag(ev) @ evecs.T
    except Exception:
        Sig = sigma_ind * np.eye(n)
        ev  = np.full(n, sigma_ind)

    # Experimental-style population coupling = mean |corr| with population mean
    ones = np.ones(n)
    var_mean = ones @ Sig @ ones / n**2
    cov_i_mean = (Sig @ ones) / n
    var_i = np.maximum(np.diag(Sig), 1e-15)
    corr_with_mean = cov_i_mean / np.sqrt(var_i * max(var_mean, 1e-15))
    popc = float(np.mean(np.abs(corr_with_mean)))

    # Secondary decomposition (for information only)
    Pu = np.outer(u, u)
    Sig_shared = Pu @ Sig @ Pu
    var_mean_s = ones @ Sig_shared @ ones / n**2
    cov_s = (Sig_shared @ ones) / n
    popc_shared = float(np.mean(np.abs(
        cov_s / np.sqrt(var_i * max(var_mean_s, 1e-15))))) if eta_ext > 1e-8 else 0.0
    popc_local = max(0.0, popc - 0.5 * popc_shared)

    tr  = ev.sum()
    pr  = (tr**2 / np.sum(ev**2)) if tr > 0 else 0.0
    pc1 = ev[-1] / (tr + 1e-15)
    return dict(popc=popc, popc_shared=popc_shared, popc_local=popc_local,
                pr=pr, pc1=pc1)

# =====================================================================
# 3. Explicit gamma resonance module
# =====================================================================
def gamma_resonance(Wexc, Winh, eta_ext, G_res, PV_surv, sigma_obs):
    n = Wexc.shape[0]
    mask = ~np.eye(n, dtype=bool)
    E_loc = np.mean(Wexc[mask])
    I_loc = np.mean(Winh[mask])
    ei = E_loc / (I_loc + 0.06)

    pv_lg = 0.50 + 0.50 * PV_surv
    pv_hg = 0.22 + 0.78 * PV_surv
    drive_mod = 1.0 / (1.0 + 1.5 * eta_ext)

    A_lg = G_res * ei * pv_lg * drive_mod
    A_hg = G_res * ei * pv_hg * drive_mod * 0.87

    plv_lg = A_lg / (A_lg + sigma_obs + 1e-8)
    plv_hg = A_hg / (A_hg + sigma_obs * 0.80 + 1e-8)
    return (float(np.clip(plv_lg, 0, 0.99)),
            float(np.clip(plv_hg, 0, 0.99)))

# =====================================================================
# 4. Forward model
# =====================================================================
def forward(nodes, params, drive_mask):
    W, Wexc, Winh = build_connectivity(
        nodes, params["W_E"], params["W_I"], params["PV_surv"])
    cov = lyapunov_metrics(W, params["eta_ext"], drive_mask)
    plv_lg, plv_hg = gamma_resonance(
        Wexc, Winh, params["eta_ext"], params["G_res"],
        params["PV_surv"], params["sigma_obs"])
    return dict(
        population_coupling = cov["popc"],
        popc_shared         = cov["popc_shared"],
        popc_local          = cov["popc_local"],
        plv_low_gamma       = plv_lg,
        plv_high_gamma      = plv_hg,
        participation_ratio = cov["pr"],
        pc1_variance        = cov["pc1"],
    )

# =====================================================================
# 5. Loss (only PopC + PLVs)
# =====================================================================
FIT = ["population_coupling", "plv_low_gamma", "plv_high_gamma"]
WGT = dict(population_coupling=2.0, plv_low_gamma=1.3, plv_high_gamma=1.1)

def loss_fn(x, state, nodes, mask):
    p = dict(
        eta_ext   = np.clip(x[0], 0.0, 0.35),
        W_E       = np.clip(x[1], 0.15, 2.2),
        W_I       = np.clip(x[2], 0.10, 2.0),
        PV_surv   = np.clip(x[3], 0.05, 1.0),
        G_res     = np.clip(x[4], 0.25, 3.5),
        sigma_obs = np.clip(x[5], 0.04, 1.2),
    )
    m = forward(nodes, p, mask)
    L = 0.0
    for k in FIT:
        emp = EMP[state][k]
        L += WGT[k] * ((m[k] - emp) / (abs(emp) + 0.01))**2
    # soft biological priors
    if state == "vitro":
        L += 0.8 * (p["PV_surv"] - 0.18)**2
    elif state == "awake":
        L += 0.3 * (p["PV_surv"] - 0.85)**2
    return L / len(FIT)

# =====================================================================
# 6. Optimisation
# =====================================================================
def run_optimisation():
    print("=" * 64)
    print("MINIMAL MECHANISTIC MODEL – FINAL")
    print("=" * 64)

    node_pop, masks = {}, {}
    for i, st in enumerate(STATES):
        rng = np.random.default_rng(42 + i * 29)
        node_pop[st] = generate_nodes(N_NODES, rng)
        mask = np.array([nd["layer"] == "supra" and rng.random() < 0.28
                         for nd in node_pop[st]])
        if mask.sum() < 9:
            mask[:11] = True
        masks[st] = mask

    BOUNDS = {
        "awake": [(0.05, 0.28), (0.4, 1.8), (0.2, 1.4), (0.65, 1.0), (0.3, 2.0), (0.2, 1.0)],
        "sleep": [(0.08, 0.32), (0.4, 1.8), (0.2, 1.5), (0.50, 0.95), (0.4, 2.4), (0.12, 0.9)],
        "vitro": [(0.00, 0.12), (0.5, 2.0), (0.3, 1.8), (0.08, 0.40), (1.1, 3.4), (0.06, 0.5)],
    }

    opt, final = {}, {}
    for st in STATES:
        print(f"\nOptimising {st.upper()} ...")
        t0 = time.perf_counter()
        res = differential_evolution(
            loss_fn, BOUNDS[st], args=(st, node_pop[st], masks[st]),
            seed=42, maxiter=120, popsize=16, mutation=(0.5, 1.0),
            recombination=0.85, tol=1e-8, polish=True, workers=1)
        print(f"  loss = {res.fun:.4f}   ({time.perf_counter()-t0:.1f}s)")
        x = res.x
        p = dict(eta_ext=x[0], W_E=x[1], W_I=x[2],
                 PV_surv=x[3], G_res=x[4], sigma_obs=x[5])
        opt[st] = p
        final[st] = forward(node_pop[st], p, masks[st])
        for k, v in p.items():
            print(f"    {k:12s}: {v:.4f}")

    print("\nPRIMARY METRICS")
    for st in STATES:
        print(f"\n  {STATE_LAB[st]}")
        for k in FIT:
            sim, emp = final[st][k], EMP[st][k]
            pct = abs(sim - emp) / (abs(emp) + 1e-9) * 100
            flag = "OK" if pct < 30 else ("~" if pct < 50 else "…")
            print(f"    {k:<22s} {sim:7.4f}  {emp:7.4f}  {pct:5.1f}%  {flag}")

    with open(SAV / "opt_params_final.json", "w") as f:
        json.dump(opt, f, indent=2)
    pd.DataFrame([{"state": s, **final[s]} for s in STATES]).to_csv(
        SAV / "metrics_final.csv", index=False)
    return opt, final, node_pop, masks

# =====================================================================
# 7. Continuous de-afferentation sweep
# =====================================================================
def deafferentation_sweep(opt, node_pop, masks, n_points=51):
    aw, vt = opt["awake"], opt["vitro"]
    keys = ["eta_ext", "W_E", "W_I", "PV_surv", "G_res", "sigma_obs"]
    rows = []
    for a in np.linspace(0, 1, n_points):
        p = {k: (1 - a) * aw[k] + a * vt[k] for k in keys}
        m = forward(node_pop["awake"], p, masks["awake"])
        rows.append({"alpha": a, **p, **m})
    df = pd.DataFrame(rows)
    df.to_csv(SAV / "deafferentation_sweep.csv", index=False)
    return df

# =====================================================================
# 8. Sobol sensitivity
# =====================================================================
def run_sobol(node_pop, masks, n_samples=256):
    if not HAS_SALIB:
        return None, None, None, None

    problem = {
        "num_vars": 6,
        "names": ["η_ext", "W_E", "W_I", "PV_surv", "G_res", "σ_obs"],
        "bounds": [[0.0, 0.30], [0.2, 2.0], [0.15, 1.8],
                   [0.1, 1.0], [0.3, 3.2], [0.05, 1.0]]
    }

    N = 2 ** int(np.round(np.log2(n_samples)))
    X = sobol_sample(problem, N, calc_second_order=False)

    nodes, mask = node_pop["awake"], masks["awake"]
    Y_popc = np.zeros(len(X))
    Y_plv_l  = np.zeros(len(X))
    Y_plv_h  = np.zeros(len(X))

    for i, x in enumerate(X):
        p = dict(eta_ext=x[0], W_E=x[1], W_I=x[2],
                 PV_surv=x[3], G_res=x[4], sigma_obs=x[5])
        m = forward(nodes, p, mask)
        Y_popc[i] = m["population_coupling"]
        Y_plv_l[i] = m["plv_low_gamma"]
        Y_plv_h[i] = m["plv_high_gamma"]

    Si_popc = sobol.analyze(problem, Y_popc, calc_second_order=False, print_to_console=False)
    Si_plv_l  = sobol.analyze(problem, Y_plv_l,  calc_second_order=False, print_to_console=False)
    Si_plv_h  = sobol.analyze(problem, Y_plv_h,  calc_second_order=False, print_to_console=False)

    return problem, Si_popc, Si_plv_l, Si_plv_h
# =====================================================================
# 9. Figures
# =====================================================================
def fig_metrics(final):
    """Panel a – empirical vs model bars"""
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.0))
    mets   = ["population_coupling", "plv_low_gamma", "plv_high_gamma"]
    titles = ["Population Coupling", "Low-γ PLV", "High-γ PLV"]
    x = np.arange(3)
    w = 0.35
    for ax, met, title in zip(axes, mets, titles):
        emp = [EMP[s][met] for s in STATES]
        mod = [final[s][met] for s in STATES]
        ax.bar(x - w/2, emp, w, color="#2C3E50", label="Empirical", alpha=0.9)
        ax.bar(x + w/2, mod, w, color="#E74C3C", label="Model", alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(["Awake", "Sleep", "In Vitro"], rotation=12)
        ax.set_title(title)
    axes[0].legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "Fig1a_metrics.pdf", bbox_inches="tight")
    fig.savefig(FIG / "Fig1a_metrics.svg", bbox_inches="tight")
    plt.close(fig)
    print("  Fig1a_metrics")

def fig_parameters(opt):
    """Panel b – latent parameter heatmap"""
    keys   = ["eta_ext", "W_E", "W_I", "PV_surv", "G_res", "sigma_obs"]
    labels = [r"$\eta_{\rm ext}$", r"$W_E$", r"$W_I$",
              r"PV$_{\rm surv}$", r"$G_{\rm res}$", r"$\sigma_{\rm obs}$"]
    data = np.array([[opt[s][k] for k in keys] for s in STATES])
    fig, ax = plt.subplots(figsize=(5.4, 2.7))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(6))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(3))
    ax.set_yticklabels([STATE_LAB[s] for s in STATES])
    for i in range(3):
        for j in range(6):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=7.5)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Value")
    ax.set_title("Inferred latent parameters")
    fig.tight_layout()
    fig.savefig(FIG / "Fig1b_parameters.pdf", bbox_inches="tight")
    fig.savefig(FIG / "Fig1b_parameters.svg", bbox_inches="tight")
    plt.close(fig)
    print("  Fig1b_parameters")

def fig_plotly_sankey(opt, final):
    """Panel c – spacious coloured Sankey (one per state)"""
    param_labels = ["ηext", "W E", "W I", "PV axonal survival",
                    "resonance gain", "observation noise"]
    target_labels = ["Population coupling", "Low-γ PLV", "High-γ PLV"]

    for st in STATES:
        p = opt[st]
        m = final[st]

        # mechanistic contributions → normalise per target
        raw = np.array([
            [0.60 * p["eta_ext"],   0.10 * p["eta_ext"],   0.08 * p["eta_ext"]],
            [0.15 * p["W_E"],       0.12 * p["W_E"],       0.10 * p["W_E"]],
            [0.10 * p["W_I"],       0.08 * p["W_I"],       0.07 * p["W_I"]],
            [0.05 * p["PV_surv"],   0.28 * p["PV_surv"],   0.38 * p["PV_surv"]],
            [0.00,                  0.50 * p["G_res"]/2.5, 0.45 * p["G_res"]/2.5],
            [0.00,                  0.20 * p["sigma_obs"], 0.18 * p["sigma_obs"]],
        ])
        for j in range(3):
            s = raw[:, j].sum()
            if s > 0:
                raw[:, j] /= s

        scale = np.maximum(
            [m["population_coupling"], m["plv_low_gamma"], m["plv_high_gamma"]],
            0.02)
        flows = raw * scale[np.newaxis, :] * 100

        sources, targets, values, link_cols = [], [], [], []
        for i in range(6):
            for j in range(3):
                if flows[i, j] < 0.25:
                    continue
                sources.append(i)
                targets.append(6 + j)
                values.append(float(flows[i, j]))
                hexcol = PARAM_COLORS[i].lstrip("#")
                r, g, b = int(hexcol[0:2], 16), int(hexcol[2:4], 16), int(hexcol[4:6], 16)
                link_cols.append(f"rgba({r},{g},{b},0.48)")

        node_labels = param_labels + [
            f"PopC  {m['population_coupling']:.3f}",
            f"Low-γ  {m['plv_low_gamma']:.3f}",
            f"High-γ  {m['plv_high_gamma']:.3f}"
        ]
        node_colors = PARAM_COLORS + TARGET_COLORS

        fig = go.Figure(data=[go.Sankey(
            arrangement="snap",
            node=dict(
                pad=32,
                thickness=20,
                line=dict(color="rgba(0,0,0,0.25)", width=0.5),
                label=node_labels,
                color=node_colors,
                x=[0.02]*6 + [0.98]*3,
                y=[0.07, 0.23, 0.39, 0.55, 0.71, 0.87] + [0.18, 0.50, 0.82],
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_cols,
            )
        )])
        fig.update_layout(
            title_text=f"{STATE_LAB[st]} – Latent-parameter contributions",
            font=dict(size=13, family="Arial"),
            height=460,
            width=720,
            margin=dict(l=30, r=30, t=55, b=25),
            paper_bgcolor="white",
            plot_bgcolor="white",
        )
        fig.write_image(FIG / f"Fig1c_Sankey_{st}.pdf")
        fig.write_image(FIG / f"Fig1c_Sankey_{st}.svg")
        fig.write_html(FIG / f"Fig1c_Sankey_{st}.html")
        print(f"  Fig1c_Sankey_{st}")

def fig_deafferentation(sweep):
    """Panel d – continuous trajectory"""
    fig, ax1 = plt.subplots(figsize=(5.0, 3.3))
    a = sweep["alpha"].values
    ax1.plot(a, sweep["population_coupling"], color="#27AE60", lw=2.3,
             label="Population coupling")
    ax1.set_ylabel("Population coupling", color="#27AE60")
    ax1.tick_params(axis="y", labelcolor="#27AE60")
    ax1.set_xlabel("De-afferentation index")
    ax1.set_xticks([0, 0.5, 1.0])
    ax1.set_xticklabels(["Awake", "Intermediate", "In Vitro"])

    ax2 = ax1.twinx()
    ax2.plot(a, sweep["plv_low_gamma"], color="#E74C3C", lw=2.0, label="Low-γ PLV")
    ax2.plot(a, sweep["plv_high_gamma"], color="#F39C12", lw=2.0, ls="--",
             label="High-γ PLV")
    ax2.set_ylabel("Phase-locking value")
    ax2.spines["right"].set_visible(True)

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="center right", frameon=False)
    ax1.set_title("Continuous de-afferentation trajectory")
    fig.tight_layout()
    fig.savefig(FIG / "Fig1d_deafferentation.pdf", bbox_inches="tight")
    fig.savefig(FIG / "Fig1d_deafferentation.svg", bbox_inches="tight")
    plt.close(fig)
    print("  Fig1d_deafferentation")

def fig_sobol(problem, Si_popc, Si_plv_l, Si_plv_h):
    """Panel e – Sobol indices (tight layout)"""
    if problem is None:
        return
    names = problem["names"]
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.6))
    for ax, Si, title in zip(axes,
                             [Si_popc, Si_plv_l, Si_plv_h],
                             ["Population coupling", "Low-γ PLV", "High-γ PLV"]):
        x = np.arange(len(names))
        ax.bar(x - 0.15, Si["S1"], 0.30, label="First-order",
               color="#0cb2af", alpha=0.9)
        ax.bar(x + 0.15, Si["ST"], 0.30, label="Total-order",
               color="#e95e50", alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=8.5)
        if ax is axes[0]:
            ax.set_ylabel("Sobol index", fontsize=8)
            ax.legend(frameon=False, fontsize=6.5)
    fig.suptitle("Parameter sensitivity", fontsize=9, y=1.01)
    fig.subplots_adjust(wspace=0.18, top=0.88, bottom=0.18)
    fig.savefig(FIG / "Fig1e_sobol.pdf", bbox_inches="tight")
    fig.savefig(FIG / "Fig1e_sobol.svg", bbox_inches="tight")
    plt.close(fig)
    print("  Fig1e_sobol")

# =====================================================================
# Main
# =====================================================================
if __name__ == "__main__":
    opt, final, node_pop, masks = run_optimisation()

    print("\nDe-afferentation sweep ...")
    sweep = deafferentation_sweep(opt, node_pop, masks)

    print("Sobol analysis ...")
    problem, Si_popc, Si_plv_l, Si_plv_h = run_sobol(node_pop, masks, n_samples=256)

    print("\nGenerating all figures ...")
    fig_metrics(final)
    fig_parameters(opt)
    fig_plotly_sankey(opt, final)
    fig_deafferentation(sweep)
    fig_sobol(problem, Si_popc, Si_plv_l, Si_plv_h)

    print("\n" + "=" * 64)
    print("FINAL MODEL COMPLETE")
    print("Figures →", FIG)
    print("Data    →", SAV)
    print("=" * 64)