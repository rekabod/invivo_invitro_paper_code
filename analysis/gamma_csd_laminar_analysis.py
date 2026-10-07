"""
Laminar Current Source Density (CSD) analysis across cortical layers.

Purpose
-------
Analyze CSD patterns across cortical layers (supragranular, granular,
infragranular) to identify depth-specific current flow during gamma
oscillations.

Key features
------------
- Layer-specific CSD analysis
- Depth-resolved current sink/source mapping
- Comparison across recording states (awake, sleep, vitro)
- Publication-ready laminar CSD plots

Usage
-----
    python -m gamma_csd.gamma_csd_laminar_analysis

Dependencies
------------
    numpy, pandas, scipy, matplotlib, seaborn
"""
import os
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from itertools import combinations

import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats
import scipy.ndimage
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 130
plt.rcParams["axes.grid"] = False

ADC_TO_UV = 0.30518

# -----------------------------------------------------------------------------
# Path & Structure Handlers
# -----------------------------------------------------------------------------
def _to_windows_abs_path(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return None
    return str(p).strip().replace("/", "\\")

def infer_recording_paths_from_base_prefix(base_prefix):
    base_prefix = _to_windows_abs_path(base_prefix)
    if base_prefix is None or base_prefix == "":
        raise ValueError("Empty base_prefix")
    return dict(
        base_prefix=base_prefix,
        bin_path=base_prefix + ".bin"
    )

def safe_makedirs(p):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def save_fig_png_svg(fig, out_basepath):
    fig.savefig(out_basepath + ".png", dpi=220, facecolor="white")
    fig.savefig(out_basepath + ".svg", facecolor="white")

def load_epochs_from_csvs_seconds(base_prefix, bands=("low_gamma", "high_gamma")):
    token_map = {"low_gamma": "lowgamma", "high_gamma": "highgamma"}
    out = {b: [] for b in bands}
    for band in bands:
        csv_path = f"{base_prefix}_{token_map[band]}_epochs.csv"
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        if not {"start_s", "end_s"}.issubset(df.columns):
            continue
        out[band] = [(float(r.start_s), float(r.end_s)) for r in df.itertuples(index=False)]
    return out

# -----------------------------------------------------------------------------
# Robust Signal Processing & Filtering Engine
# -----------------------------------------------------------------------------
def spatial_hamming_filter(data, iterations=2):
    """
    Applies spatial smoothing: 54% center, 23% neighbor above, 23% below.
    Applied iteratively to rows 1 to N-2, perfectly preserving edge values.
    """
    out = data.copy()
    for _ in range(iterations):
        temp = out.copy()
        # Apply to internal channels only (indices 1 to N-2)
        for i in range(1, out.shape[0] - 1):
            temp[i] = (0.23 * out[i-1]) + (0.54 * out[i]) + (0.23 * out[i+1])
        out = temp
    return out

def temporal_bandpass_filter(x, fs, band=(1.0, 500.0), order=4):
    """
    Butterworth zero-phase bandpass filter. 
    Protects against Nyquist and short-epoch errors that crash FIR filters.
    """
    nyq = fs / 2.0
    lo, hi = band
    hi = min(hi, nyq * 0.95)
    b, a = scipy.signal.butter(order, [lo / nyq, hi / nyq], btype="band")
    return scipy.signal.filtfilt(b, a, x, axis=-1)

def bandpass(x, fs, band, order=4):
    lo, hi = band
    nyq = fs / 2.0
    padlen = 3 * order
    if x.shape[-1] <= padlen:
        order = max(1, x.shape[-1] // 4)
        padlen = 3 * order
    b, a = scipy.signal.butter(order, [lo / nyq, hi / nyq], btype="band")
    if x.shape[-1] <= padlen:
        return scipy.signal.filtfilt(b, a, x, axis=-1, padtype='odd', padlen=x.shape[-1] - 1)
    return scipy.signal.filtfilt(b, a, x, axis=-1)

def notch_filter(x, fs, freqs=(50.0, 100.0), q=30.0):
    y = np.asarray(x, dtype=float)
    for f0 in freqs:
        if f0 <= 0 or f0 >= (fs / 2.0):
            continue
        b, a = scipy.signal.iirnotch(w0=f0, Q=q, fs=fs)
        y = scipy.signal.filtfilt(b, a, y, axis=-1)
    return y

def compute_csd_from_gradient_referenced_lfp(lfp_gradref, spacing_um=150.0):
    depth = np.arange(lfp_gradref.shape[0], dtype=float) * float(spacing_um)
    return -np.gradient(lfp_gradref, depth, axis=0, edge_order=2)

# -----------------------------------------------------------------------------
# Laminar Analysis & ISS Calculation
# -----------------------------------------------------------------------------
def calculate_integrated_sink_strength(csd_matrix, fs, supra_idx, gran_idx, infra_idx) -> tuple[float, float, float]:
    sinks_only = np.maximum(csd_matrix, 0.0)
    integral_per_channel = np.sum(sinks_only, axis=1) / float(fs)
    
    iss_supra = float(np.mean(integral_per_channel[supra_idx])) if len(supra_idx) > 0 else 0.0
    iss_gran = float(np.mean(integral_per_channel[gran_idx])) if len(gran_idx) > 0 else 0.0
    iss_infra = float(np.mean(integral_per_channel[infra_idx])) if len(infra_idx) > 0 else 0.0
    
    return iss_supra, iss_gran, iss_infra

# -----------------------------------------------------------------------------
# Publication-Ready Visualization
# -----------------------------------------------------------------------------
def plot_true_gamma_csd_with_traces(
    matrix_data, title, out_basepath, time_axis, ytick_labels,
    bad_channel_mask=None, overlay_traces=None, trace_color="black", trace_alpha=0.22,
    trace_type="CSD"
):
    # Strip artificial edges immediately before spatial processing
    display = matrix_data[1:-1, :]
    clean_labels = ytick_labels[1:-1]
    
    # Process overlay traces with spatial Hamming if present
    if overlay_traces is not None:
        # Filter applied to full matrix to maintain spatial context for the internal channels
        processed_traces = spatial_hamming_filter(overlay_traces, iterations=2)
        processed_traces = processed_traces[1:-1, :]
    
    clean_bad_mask = None
    if bad_channel_mask is not None:
        clean_bad_mask = bad_channel_mask[1:-1]

    # Flip channel order for anatomical logic (top channel = surface)
    display = display[::-1, :]
    clean_labels = clean_labels[::-1]
    if overlay_traces is not None:
        processed_traces = processed_traces[::-1, :]
    if clean_bad_mask is not None:
        clean_bad_mask = clean_bad_mask[::-1]

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Gaussian smoothing only on valid core heatmap data
    display = scipy.ndimage.gaussian_filter(display, sigma=(0.2, 0.6), mode="nearest")
    
    # Unified Static Limits
    v_limit = 0.005
    norm = TwoSlopeNorm(vmin=-v_limit, vcenter=0.0, vmax=v_limit)
    extent = [float(time_axis[0]), float(time_axis[-1]), -0.5, display.shape[0] - 0.5]
    
    im = ax.imshow(
        display, aspect="auto", origin="upper", interpolation="bicubic",
        cmap="RdBu", norm=norm, extent=extent
    )

    if overlay_traces is not None:
        n_channels = display.shape[0]
        time_axis_raw = np.linspace(extent[0], extent[1], processed_traces.shape[1])
        for ch_idx in range(n_channels):
            trace = processed_traces[ch_idx, :]
            y_center = ch_idx 
            
            t_min, t_max = np.percentile(trace, [1, 99])
            t_range = (t_max - t_min) if (t_max - t_min) > 1e-6 else 1.0
            
            # Trace amplitude normalization to prevent overlap
            norm_trace = ((trace - np.median(trace)) / t_range) * 0.85
            ax.plot(time_axis_raw, y_center + norm_trace, color=trace_color, alpha=trace_alpha, linewidth=0.55)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Physical Channel")
    
    ticks = np.arange(0, display.shape[0], 1)
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(label) for label in clean_labels])
    ax.set_xlim(extent[0], extent[1])

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Gamma CSD Map (Sink: >0, Source: <0)")

    if clean_bad_mask is not None:
        for i, is_bad in enumerate(clean_bad_mask):
            if is_bad:
                ax.axhspan(i - 0.5, i + 0.5, color="black", alpha=0.12)

    save_fig_png_svg(fig, out_basepath)
    plt.close(fig)

# -----------------------------------------------------------------------------
# Aggregate Statistics Module
# -----------------------------------------------------------------------------
def save_and_plot_aggregate_stats(df_results, out_root):
    if df_results.empty:
        return
        
    compare_rows = []
    metrics = ["mean_abs_csd", "peak_abs_csd", "iss_supra", "iss_gran", "iss_infra"]
    states = sorted(df_results["state"].unique().tolist())
    bands = sorted(df_results["band"].unique().tolist())
    
    for band in bands:
        sub = df_results[df_results["band"] == band]
        for m in metrics:
            for s1, s2 in combinations(states, 2):
                a = sub[sub["state"] == s1][m].dropna().to_numpy()
                b = sub[sub["state"] == s2][m].dropna().to_numpy()
                if len(a) < 2 or len(b) < 2:
                    continue
                stat = scipy.stats.mannwhitneyu(a, b, alternative="two-sided")
                compare_rows.append(dict(
                    band=band, metric=m, state1=s1, state2=s2,
                    n1=len(a), n2=len(b), u_stat=float(stat.statistic), p_value=float(stat.pvalue)
                ))
    
    pd.DataFrame(compare_rows).to_csv(os.path.join(out_root, "gamma_csd_comparison_stats.csv"), index=False)
    
    state_order = [s for s in ["awake", "sleep", "vitro"] if s in set(df_results["state"])]
    if not state_order:
        state_order = states

    for band in bands:
        sub = df_results[df_results["band"] == band]
        for m in metrics:
            fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
            sns.violinplot(data=sub, x="state", y=m, order=state_order, ax=ax, inner="quartile", palette="muted")
            sns.stripplot(data=sub, x="state", y=m, order=state_order, ax=ax, color="#333333", alpha=0.4, size=3, jitter=0.15)
            ax.set_title(f"{band.upper()} - {m.replace('_', ' ').title()} by State", fontsize=12, fontweight="bold")
            save_fig_png_svg(fig, os.path.join(out_root, f"{band}_{m}_violin"))
            plt.close(fig)

# -----------------------------------------------------------------------------
# Main Pipeline Engine
# -----------------------------------------------------------------------------
def main(metadata_csv, out_root=None, lfp_fs=1000):
    metadata_csv = _to_windows_abs_path(metadata_csv)
    if out_root is None:
        out_root = os.path.join(os.path.dirname(metadata_csv), "gamma_csd_laminar_output")
    safe_makedirs(out_root)

    df_meta = pd.read_csv(metadata_csv)
    rows = []
    pad_s = 0.6  

    for _, row in df_meta.iterrows():
        paths = infer_recording_paths_from_base_prefix(row["base_prefix"])
        if not os.path.exists(paths["bin_path"]):
            print(f"⚠️ [MISSING] File not found: {paths['bin_path']}")
            continue

        recording_id = str(row["recording_id"])
        state = str(row["state_norm"]).lower() 
        fs_row = int(row["fs"])
        raw_num_channels = int(row["raw_num_channels"])

        print(f" PROCESSING: {recording_id} ({state.upper()}) | Fs: {fs_row} Hz | Ch: {raw_num_channels}")

        rec_out = os.path.join(out_root, recording_id)
        snippet_dir = os.path.join(rec_out, "snippets")
        safe_makedirs(snippet_dir)

        file_bytes = os.path.getsize(paths["bin_path"])
        total_samples = file_bytes // (raw_num_channels * np.dtype(np.int16).itemsize)

        epochs_by_band = load_epochs_from_csvs_seconds(paths["base_prefix"])
        band_defs = {"low_gamma": (30, 50), "high_gamma": (50, 80)}
        noisy_mask = np.zeros(23, dtype=bool)

        for band_name, band in band_defs.items():
            epochs = sorted([(float(s), float(e)) for s, e in epochs_by_band.get(band_name, [])], 
                            key=lambda se: (se[1] - se[0]), reverse=True)
            
            selected_epochs = []
            for s, e in epochs:
                if len(selected_epochs) >= 5: 
                    break
                if (e - s) >= 3.0:
                    mid = (s + e) / 2.0
                    selected_epochs.append((max(s, mid - 2.0), min(e, mid + 2.0)))

            for i, (start_s, end_s) in enumerate(selected_epochs, start=1):
                padded_start_s = max(0.0, start_s - pad_s)
                padded_end_s = min(float(total_samples / fs_row), end_s + pad_s)

                idx_start = int(math.floor(padded_start_s * fs_row))
                idx_end = int(math.ceil(padded_end_s * fs_row))
                
                with open(paths["bin_path"], "rb") as f:
                    f.seek(idx_start * raw_num_channels * np.dtype(np.int16).itemsize)
                    raw_data = np.fromfile(f, dtype=np.int16, count=(idx_end - idx_start) * raw_num_channels)
                
                raw_data = raw_data.reshape((-1, raw_num_channels)).T
                
                if raw_num_channels == 26:
                    raw_data = raw_data[3:26, :]  
                elif raw_num_channels == 24:
                    raw_data = raw_data[0:23, :]  
                else:
                    raw_data = raw_data[:23, :]   
                
                raw_data = raw_data[::-1, :].astype(float) * ADC_TO_UV

                gcd_fs = math.gcd(fs_row, lfp_fs)
                snippet_ds = scipy.signal.resample_poly(raw_data, up=lfp_fs // gcd_fs, down=fs_row // gcd_fs, axis=1)
                
                ds_pad_samples = int(round(pad_s * lfp_fs))
                if snippet_ds.shape[1] <= (2 * ds_pad_samples + 15):
                    continue
                    
                actual_snippet_lfp = snippet_ds[:, ds_pad_samples : snippet_ds.shape[1] - ds_pad_samples]
                
                # --- SIGNAL PROCESSING PIPELINE ---
                
                # 1. Base Filters & Gamma Isolation (For Matrix Generation)
                lfp_notch = notch_filter(actual_snippet_lfp, fs=lfp_fs)
                csd_broadband = compute_csd_from_gradient_referenced_lfp(lfp_notch, spacing_um=150.0)
                
                lfp_gamma = bandpass(lfp_notch, lfp_fs, band)
                csd_gamma = bandpass(csd_broadband, lfp_fs, band)

                # 2. Overlay-Specific Filtering (Applied strictly to the traces shown over the heatmap)
                # Apply 1-500Hz Butterworth bandpass on top of the notched LFP
                lfp_overlay_clean = temporal_bandpass_filter(lfp_notch, fs=lfp_fs, band=(1.0, 500.0))
                
                # Compute CSD from this cleanly bandpassed LFP for the 3rd plot
                csd_overlay_clean = compute_csd_from_gradient_referenced_lfp(lfp_overlay_clean, spacing_um=150.0)

                # --- LAMINAR LAYER INDICES ---
                supra_idx = np.arange(1, 8)    
                gran_idx = np.arange(8, 13)   
                infra_idx = np.arange(13, 22)  

                iss_supra, iss_gran, iss_infra = calculate_integrated_sink_strength(
                    csd_gamma, lfp_fs, supra_idx, gran_idx, infra_idx
                )

                core_channels = np.arange(1, 22)
                mean_abs_csd_core = float(np.nanmean(np.abs(csd_gamma[core_channels, :])))
                peak_abs_csd_core = float(np.nanmax(np.abs(csd_gamma[core_channels, :])))

                rows.append(dict(
                    recording_id=recording_id, state=state, band=band_name, epoch_idx=i,
                    start_s=start_s, end_s=end_s,
                    mean_abs_csd=mean_abs_csd_core, peak_abs_csd=peak_abs_csd_core,
                    iss_supra=iss_supra, iss_gran=iss_gran, iss_infra=iss_infra
                ))

                # Heatmap temporal binning
                bin_samples = max(1, int(round(0.01 * lfp_fs)))
                n_bins = int(np.ceil(csd_gamma.shape[1] / bin_samples))
                csd_tc = np.zeros((csd_gamma.shape[0], n_bins), dtype=float)
                for b_idx in range(n_bins):
                    a = b_idx * bin_samples
                    b = min(csd_gamma.shape[1], (b_idx + 1) * bin_samples)
                    csd_tc[:, b_idx] = np.nanmean(csd_gamma[:, a:b], axis=1)

                duration_s = float(end_s - start_s)
                time_axis_binned = np.linspace(0.0, duration_s, csd_tc.shape[1])
                labels = list(range(1, actual_snippet_lfp.shape[0] + 1))
                base_fn = f"{recording_id}_{band_name}_snippet{i}"

                # --- VISUALIZATION BLOCK ---
                
                # Plot 1: 1-500Hz + Notch LFP traces on Gamma CSD Heatmap
                plot_true_gamma_csd_with_traces(
                    csd_tc, 
                    title=f"{recording_id} ({band_name}) Snippet {i}: 1-500Hz Cleaned LFP over Gamma CSD",
                    out_basepath=os.path.join(snippet_dir, f"{base_fn}_csd_with_unfiltered_lfp"), 
                    time_axis=time_axis_binned, 
                    ytick_labels=labels,
                    bad_channel_mask=noisy_mask, 
                    overlay_traces=lfp_overlay_clean, 
                    trace_color="black", 
                    trace_alpha=0.28,
                    trace_type="LFP"
                )

                # Plot 2: Gamma CSD traces projected over Gamma CSD heatmap
                plot_true_gamma_csd_with_traces(
                    csd_tc, 
                    title=f"{recording_id} ({band_name}) Snippet {i}: Gamma CSD Traces over Gamma CSD Map",
                    out_basepath=os.path.join(snippet_dir, f"{base_fn}_csd_with_gamma_traces"), 
                    time_axis=time_axis_binned, 
                    ytick_labels=labels,
                    bad_channel_mask=noisy_mask, 
                    overlay_traces=csd_gamma,  
                    trace_color="darkred", 
                    trace_alpha=0.30,
                    trace_type="CSD"
                )

                # Plot 3: 1-500Hz + Notch CSD traces on Gamma CSD Heatmap
                plot_true_gamma_csd_with_traces(
                    csd_tc, 
                    title=f"{recording_id} ({band_name}) Snippet {i}: 1-500Hz Cleaned CSD Traces",
                    out_basepath=os.path.join(snippet_dir, f"{base_fn}_csd_with_broadband_csd_traces"), 
                    time_axis=time_axis_binned, 
                    ytick_labels=labels,
                    bad_channel_mask=noisy_mask, 
                    overlay_traces=csd_overlay_clean, 
                    trace_color="navy", 
                    trace_alpha=0.25,
                    trace_type="CSD"
                )

    df_results = pd.DataFrame(rows)
    if not df_results.empty:
        df_results.to_csv(os.path.join(out_root, "gamma_csd_laminar_results.csv"), index=False)
        save_and_plot_aggregate_stats(df_results, out_root)
        print(f"\n🚀 [SUCCESS] Pipeline execution complete. Aggregated data, statistics, and Violin plots saved to: {out_root}")
    else:
        print("\n❌ [ERROR] Failed to generate data from any snippet. Verify input paths.")
        
    return df_results

if __name__ == "__main__":
    main(
        metadata_csv=r"E:\in_vivo_in_vitro\stat\statbase\table_recordings_main.csv",
        out_root=r"E:\in_vivo_in_vitro\stat\gamma_csd_outputs_06_10"
    )