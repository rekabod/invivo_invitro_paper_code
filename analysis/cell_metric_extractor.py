"""
Cell metric extraction for single-cell electrophysiology data.

Purpose
-------
Extract and compute cell-level metrics from raw electrophysiology recordings
including firing rates, burstiness, inter-spike interval statistics,
phase-locking values, population coupling, and dimensionality measures.

Key features
------------
- Peak detection and spike sorting
- Circular statistics for phase-locking analysis
- Graph-theoretic metrics (strength, clustering, closeness, hubness)
- Population coupling computation
- Dimensionality analysis (participation ratio, PC1 variance, eigenspectrum entropy)

Usage
-----
    python -m stats.cell_metric_extractor

Dependencies
------------
    numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, networkx
"""
import os

import numpy as np
import pandas as pd
import scipy.signal
from scipy.signal import find_peaks

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import List
from typing import Optional, Dict

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sns.set(style="whitegrid")
plt.rcParams["figure.dpi"] = 120


# -----------------------------
# Visual palette (matches your request)
# -----------------------------
CELLTYPE_COLORS = {
    "RS-PC": "#D81B60",  # magenta
    "IB-PC": "#FB8C00",  # orange
    "IN": "#1E88E5",     # blue
    "FS": "#00897B",     # teal
    "unclassified": "#9E9E9E",
}


def normalize_celltype(celltype_raw: str) -> str:
    if celltype_raw is None or (isinstance(celltype_raw, float) and np.isnan(celltype_raw)):
        return "unclassified"
    s = str(celltype_raw).strip().lower()
    if s in ("regularly_spiking", "rs", "rs-pc", "pc_rs", "pyramidal_rs"):
        return "RS-PC"
    if s in ("bursting", "ib", "ib-pc", "pc_ib", "pyramidal_ib"):
        return "IB-PC"
    if s in ("interneuron", "in", "int"):
        return "IN"
    if s in ("fs", "fast_spiking"):
        return "FS"
    return "unclassified"


def _to_windows_abs_path(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return None
    return str(p).strip().replace("/", "\\")

def state_to_letter(state):
    if state is None or (isinstance(state, float) and np.isnan(state)):
        return "u"
    s = str(state).strip().lower()
    if s in ("v", "vitro", "in_vitro", "invitro"):
        return "v"
    if s in ("s", "sleep"):
        return "s"
    if s in ("a", "awake", "wake"):
        return "a"
    return s[0] if s else "u"

def make_cell_uid(recording_id, state, unit_id):
    """
    Format: {recording_id}_{state_letter}{unit_id}
    Example: epi55_4_s4 (patient_recording_sleep_unit)
    Matches vivo_sleep_awake_matched_units.csv format for lookup.
    """
    return f"{recording_id}_{state_to_letter(state)}{int(unit_id)}"

def zscore(x):
    x = np.asarray(x, dtype=float)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if not np.isfinite(s) or s < 1e-12:
        return np.full_like(x, np.nan, dtype=float)
    return (x - m) / s

def safe_makedirs(p):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def percentile_threshold_from_weights(W, pct=75, positive_only=True):
    W = np.asarray(W, dtype=float)
    if W.size == 0:
        return 0.0
    triu = W[np.triu_indices_from(W, k=1)]
    triu = triu[np.isfinite(triu)]
    if positive_only:
        triu = triu[triu > 0]
    if triu.size == 0:
        return 0.0
    return float(np.percentile(triu, pct))


def infer_recording_paths_from_unit_stats_csv(path_in_file_column):
    p = _to_windows_abs_path(path_in_file_column)
    if p is None or p == "":
        raise ValueError("Empty path in in_vitro_cells.csv 'file' column")

    p_low = p.lower()

    if p_low.endswith("_unit_stats.csv"):
        base_prefix = p[:-len("_unit_stats.csv")]
        bin_path = base_prefix + ".bin"
        ks_dir = base_prefix
    elif p_low.endswith(".bin"):
        base_prefix = p[:-4]
        bin_path = p
        ks_dir = base_prefix
    else:
        raise ValueError(
            "Expected either *_unit_stats.csv or *.bin in in_vitro_cells.csv 'file' column, got: "
            f"{p}"
        )

    return dict(
        base_prefix=base_prefix,
        bin_path=bin_path,
        ks_dir=ks_dir,
        cluster_info_tsv=os.path.join(ks_dir, "cluster_info.tsv"),
        spike_times_npy=os.path.join(ks_dir, "spike_times.npy"),
        spike_clusters_npy=os.path.join(ks_dir, "spike_clusters.npy"),
    )

def load_bin_lfp(bin_path, num_channels, dtype=np.int16):
    with open(bin_path, "rb") as f:
        data = np.fromfile(f, dtype=dtype)
    if data.size % num_channels != 0:
        raise ValueError(f"File size not divisible by num_channels={num_channels}: {bin_path}")
    return data.reshape((-1, num_channels)).T  # (channels, samples)

def alignment_to_mode(alignment_str):
    if alignment_str is None:
        return "none"
    s = str(alignment_str).strip().lower()
    if s in ("none", ""):
        return "none"
    if s in ("peak", "positive", "pos"):
        return "positive"
    if s in ("trough", "negative", "neg"):
        return "negative"
    raise ValueError(f"Unknown alignment: {alignment_str}")

def biphasic_to_bool(val):
    if val is None:
        return False
    s = str(val).strip().lower()
    if s in ("none", "", "0", "false", "nan"):
        return False
    return True


_BAND_FILETOKENS = {
    "delta": "delta",
    "theta": "theta",
    "low_gamma": "lowgamma",
    "high_gamma": "highgamma",
}

def load_epochs_from_csvs_seconds(base_prefix, bands=("delta", "theta", "low_gamma", "high_gamma")):
    out = {b: [] for b in bands}
    for band in bands:
        token = _BAND_FILETOKENS[band]
        csv_path = f"{base_prefix}_{token}_epochs.csv"
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        if not {"start_s", "end_s"}.issubset(df.columns):
            raise ValueError(f"{csv_path} must contain columns start_s,end_s")
        out[band] = [(float(r.start_s), float(r.end_s)) for r in df.itertuples(index=False)]
    return out


def downsample_lfp_channel(x, fs_in, fs_out=1000):
    x = np.asarray(x)
    if fs_in % fs_out == 0:
        decim = fs_in // fs_out
        x_ds = scipy.signal.resample_poly(x.astype(float), up=1, down=decim)
        return x_ds, int(decim)
    x_ds = scipy.signal.resample_poly(x.astype(float), up=int(fs_out), down=int(fs_in))
    return x_ds, None

def spikes_in_epoch_seconds(spike_idx_20k, fs_spike, start_s, end_s):
    t = np.asarray(spike_idx_20k, dtype=np.int64) / float(fs_spike)
    mask = (t >= float(start_s)) & (t <= float(end_s))
    return np.asarray(spike_idx_20k, dtype=np.int64)[mask]

def phase_at_spikes_from_downsampled(phase_ds, spike_idx_20k, fs_spike, fs_lfp):
    decim = int(round(fs_spike / fs_lfp))
    if abs(fs_spike - fs_lfp * decim) > 1e-9:
        raise ValueError("fs_spike must be divisible by fs_lfp for integer mapping.")
    idx_lfp = (np.asarray(spike_idx_20k, dtype=np.int64) // decim).astype(int)
    idx_lfp = idx_lfp[(idx_lfp >= 0) & (idx_lfp < len(phase_ds))]
    return phase_ds[idx_lfp], idx_lfp


def deduplicate_spikes(spike_times, min_isi_ms=1, fs=20000):
    spike_times = np.asarray(spike_times, dtype=np.int64)
    min_isi = int(min_isi_ms * fs / 1000.0)
    spike_times = np.sort(spike_times)
    if spike_times.size == 0:
        return spike_times
    keep = [0]
    for i in range(1, len(spike_times)):
        if spike_times[i] - spike_times[keep[-1]] >= min_isi:
            keep.append(i)
    return spike_times[keep]

def find_bursts_silence_consecutive_isi(spike_times, fs=20000, burst_n=3, max_isi_ms=10, prepost_quiet_ms=20):
    st = np.asarray(spike_times, dtype=np.int64)
    n = st.size
    if n < burst_n:
        return np.zeros(n, dtype=bool)

    st = np.sort(st)
    max_isi = (max_isi_ms / 1000.0) * fs
    quiet = (prepost_quiet_ms / 1000.0) * fs

    isis = np.diff(st)
    link = isis < max_isi

    mask = np.zeros(n, dtype=bool)

    i = 0
    while i < link.size:
        if not link[i]:
            i += 1
            continue
        j = i
        while j < link.size and link[j]:
            j += 1
        run_len_spikes = (j - i) + 1
        if run_len_spikes >= burst_n:
            start_idx = i
            end_idx = j
            pre_ok = (start_idx == 0) or ((st[start_idx] - st[start_idx - 1]) >= quiet)
            post_ok = (end_idx == n - 1) or ((st[end_idx + 1] - st[end_idx]) >= quiet)
            if pre_ok and post_ok:
                mask[start_idx:end_idx + 1] = True
        i = j

    return mask

def compute_max_firing_rate_10s(spike_times, fs=20000, window_sec=10):
    return compute_max_firing_rate_window(spike_times, fs=fs, window_sec=window_sec, step_sec=5.0)


def compute_max_firing_rate_window(spike_times, fs=20000, window_sec=1.0, step_sec=0.01):
    spike_times = np.asarray(spike_times, dtype=np.int64)
    if spike_times.size < 2:
        return np.nan, np.nan, np.nan

    spike_times_sec = np.sort(spike_times) / float(fs)
    recording_duration = float(spike_times_sec[-1]) if spike_times_sec.size else 0.0
    if recording_duration <= 0:
        return np.nan, np.nan, np.nan
    if recording_duration < window_sec:
        return float(spike_times_sec.size / recording_duration), 0.0, float(recording_duration)

    start_times = np.arange(0.0, recording_duration - window_sec + 1e-12, float(step_sec), dtype=float)
    if start_times.size == 0:
        start_times = np.array([0.0], dtype=float)
    left_idx = np.searchsorted(spike_times_sec, start_times, side="left")
    right_idx = np.searchsorted(spike_times_sec, start_times + window_sec, side="left")
    counts = right_idx - left_idx
    best_idx = int(np.argmax(counts))
    best_start = float(start_times[best_idx])
    best_end = float(best_start + window_sec)
    return float(counts[best_idx] / window_sec), best_start, best_end

def compute_autocorrelogram(spike_times, fs=20000, bin_size_ms=1, window_ms=40):
    spike_times = np.asarray(spike_times, dtype=np.int64)
    if spike_times.size < 2:
        return np.array([]), np.array([])
    spike_times = np.sort(spike_times)
    bin_size = int(bin_size_ms * fs / 1000.0)
    window = int(window_ms * fs / 1000.0)
    bins = np.arange(-window, window + bin_size, bin_size)
    acorr = np.zeros(len(bins) - 1, dtype=float)
    for t in spike_times:
        diffs = spike_times - t
        diffs = diffs[(diffs >= -window) & (diffs <= window) & (diffs != 0)]
        hist, _ = np.histogram(diffs, bins=bins)
        acorr += hist
    centers = (bins[:-1] + bins[1:]) / 2 / fs * 1000.0
    return centers, acorr

def autocorr_evenness(ac_x, ac_y):
    if len(ac_x) == 0 or len(ac_y) == 0:
        return np.nan, False
    ac_x = np.asarray(ac_x, dtype=float)
    ac_y = np.asarray(ac_y, dtype=float)
    center_mask = (ac_x >= -10) & (ac_x <= 10)
    off_center_mask = (np.abs(ac_x) >= 20) & (np.abs(ac_x) <= 40)
    center_mean = ac_y[center_mask].mean() if center_mask.any() else np.nan
    off_mean = ac_y[off_center_mask].mean() if off_center_mask.any() else np.nan
    evenness = float(center_mean / (off_mean + 1e-12)) if np.isfinite(center_mean) and np.isfinite(off_mean) else np.nan
    peaks, _ = find_peaks(ac_y)
    peak_lags = ac_x[peaks] if len(peaks) > 0 else np.array([])
    has_short = bool(np.any(np.abs(peak_lags) <= 10))
    return evenness, has_short


def align_waveforms(waveforms, align_mode="negative"):
    waveforms = np.asarray(waveforms, dtype=float)
    if waveforms.size == 0:
        return waveforms, np.nan
    peak_indices = []
    for wf in waveforms:
        if align_mode == "positive":
            peak_indices.append(int(np.nanargmax(wf)))
        else:
            peak_indices.append(int(np.nanargmin(wf)))
    target_idx = int(np.median(peak_indices))
    L = waveforms.shape[1]
    aligned = []
    for wf, idx in zip(waveforms, peak_indices):
        shift = target_idx - idx
        if shift > 0:
            aligned.append(np.pad(wf, (shift, 0), constant_values=np.nan)[:L])
        elif shift < 0:
            aligned.append(np.pad(wf, (0, -shift), constant_values=np.nan)[-shift:])
        else:
            aligned.append(wf)
    return np.asarray(aligned), target_idx

def halfwidth_interp(wf, peak_idx, fs):
    wf = np.asarray(wf, dtype=float)
    if not (0 <= peak_idx < wf.size):
        return np.nan
    amp = np.abs(wf[peak_idx])
    if not np.isfinite(amp) or amp <= 0:
        return np.nan
    half_amp = amp / 2.0

    left = peak_idx
    while left > 0 and np.abs(wf[left]) > half_amp:
        left -= 1
    if left < peak_idx and np.abs(wf[left + 1]) != np.abs(wf[left]):
        l_frac = (half_amp - np.abs(wf[left])) / (np.abs(wf[left + 1]) - np.abs(wf[left]) + 1e-18)
        left_time = left + l_frac
    else:
        left_time = float(left)

    right = peak_idx
    while right < wf.size - 1 and np.abs(wf[right]) > half_amp:
        right += 1
    if right > peak_idx and np.abs(wf[right - 1]) != np.abs(wf[right]):
        r_frac = (half_amp - np.abs(wf[right])) / (np.abs(wf[right - 1]) - np.abs(wf[right]) + 1e-18)
        right_time = right - r_frac
    else:
        right_time = float(right)

    return float((right_time - left_time) / fs * 1000.0)


def _baseline_center_waveform(waveform, fs=20000, pre_window_ms=(-2.0, -0.5)):
    waveform = np.asarray(waveform, dtype=float)
    if waveform.size == 0:
        return waveform, np.nan
    center_idx = waveform.size // 2
    start_idx = int(np.floor(center_idx + pre_window_ms[0] * fs / 1000.0))
    end_idx = int(np.ceil(center_idx + pre_window_ms[1] * fs / 1000.0))
    start_idx = max(0, start_idx)
    end_idx = min(waveform.size, max(start_idx + 1, end_idx))
    baseline = float(np.nanmedian(waveform[start_idx:end_idx])) if end_idx > start_idx else np.nan
    return waveform - baseline, baseline


def _select_peak_pair(centered_waveform):
    centered_waveform = np.asarray(centered_waveform, dtype=float)
    if centered_waveform.size < 3 or not np.any(np.isfinite(centered_waveform)):
        return np.nan, np.nan

    pos_peaks, _ = find_peaks(centered_waveform)
    neg_peaks, _ = find_peaks(-centered_waveform)
    all_peaks = np.unique(np.concatenate([pos_peaks, neg_peaks])) if (pos_peaks.size or neg_peaks.size) else np.array([], dtype=int)
    if all_peaks.size == 0:
        return np.nan, np.nan

    peak_vals = centered_waveform[all_peaks]
    main_idx = int(all_peaks[np.argmax(np.abs(peak_vals))])
    main_sign = np.sign(centered_waveform[main_idx])

    remaining = all_peaks[all_peaks != main_idx]
    if remaining.size == 0:
        return main_idx, np.nan

    if main_sign > 0:
        opp = remaining[centered_waveform[remaining] < 0]
    elif main_sign < 0:
        opp = remaining[centered_waveform[remaining] > 0]
    else:
        opp = np.array([], dtype=int)

    candidates = opp if opp.size else remaining
    sec_idx = int(candidates[np.argmax(np.abs(centered_waveform[candidates]))])
    return main_idx, sec_idx


def _waveform_metric_block(waveform, fs=20000):
    waveform = np.asarray(waveform, dtype=float)
    centered_waveform, baseline = _baseline_center_waveform(waveform, fs=fs)
    main_idx, sec_idx = _select_peak_pair(centered_waveform)

    if np.isnan(main_idx):
        return dict(
            halfwidth1_ms=np.nan,
            halfwidth2_ms=np.nan,
            peak_trough_ratio=np.nan,
            peak_distance_samples=np.nan,
            peak_to_trough_delay_ms=np.nan,
            symmetry=np.nan,
            max_slope=np.nan,
            repol_slope=np.nan,
            main_peak_idx=np.nan,
            secondary_peak_idx=np.nan,
            centered_baseline=np.nan,
            centered_waveform=centered_waveform,
        )

    hw1 = halfwidth_interp(centered_waveform, int(main_idx), fs)
    hw2 = halfwidth_interp(centered_waveform, int(sec_idx), fs) if np.isfinite(sec_idx) else np.nan
    pt_ratio = float(centered_waveform[int(main_idx)] / centered_waveform[int(sec_idx)]) if np.isfinite(sec_idx) and np.abs(centered_waveform[int(sec_idx)]) > 0 else np.nan
    peak_dist = int(main_idx - sec_idx) if np.isfinite(sec_idx) else np.nan

    pos_peaks, _ = find_peaks(centered_waveform)
    neg_peaks, _ = find_peaks(-centered_waveform)
    if len(pos_peaks) == 0 or len(neg_peaks) == 0:
        peak_to_trough_delay = np.nan
        symmetry = np.nan
        repol_slope = np.nan
    else:
        trough = int(neg_peaks[np.argmin(centered_waveform[neg_peaks])])
        peak = int(pos_peaks[np.argmax(centered_waveform[pos_peaks])])
        peak_to_trough_delay = float(np.abs(peak - trough) / fs * 1000.0)
        left_hw = halfwidth_interp(centered_waveform, trough, fs) if trough < peak else halfwidth_interp(centered_waveform, peak, fs)
        right_hw = halfwidth_interp(centered_waveform, peak, fs) if trough < peak else halfwidth_interp(centered_waveform, trough, fs)
        symmetry = float(left_hw / right_hw) if np.isfinite(left_hw) and np.isfinite(right_hw) and right_hw != 0 else np.nan
        repol_slope = float((centered_waveform[trough + 2] - centered_waveform[trough]) / ((2.0) / fs)) if trough < centered_waveform.size - 2 else np.nan

    max_slope = float(np.nanmax(np.abs(np.diff(centered_waveform))) * fs) if centered_waveform.size > 1 else np.nan

    return dict(
        halfwidth1_ms=hw1,
        halfwidth2_ms=hw2,
        peak_trough_ratio=pt_ratio,
        peak_distance_samples=peak_dist,
        peak_to_trough_delay_ms=peak_to_trough_delay,
        symmetry=symmetry,
        max_slope=max_slope,
        repol_slope=repol_slope,
        main_peak_idx=int(main_idx),
        secondary_peak_idx=int(sec_idx) if np.isfinite(sec_idx) else np.nan,
        centered_baseline=baseline,
        centered_waveform=centered_waveform,
    )


def compute_unit_waveform_metrics(aligned_waveforms, raw_waveforms=None, fs=20000):
    """
    Compute a unit's waveform metrics once and reuse them for exports and plots.

    Pipeline:
    1. Align the per-spike snippets by their negative or positive peak before averaging.
    2. Average the aligned snippets to get the mean waveform.
    3. Subtract the median of the pre-spike baseline window (-2.0 ms to -0.5 ms by default).
    4. Find the absolute main peak on the centered mean waveform and compute hf1 with the standard halfwidth interpolation.
    5. Find all positive and negative peaks, choose the strongest non-main peak (preferring opposite polarity), and compute hf2.
    6. Repeat the same centered-waveform metric block for raw snippets when provided.
    7. Return all waveform and firing metrics from one helper so tables and plots consume the same values.
    """
    aligned_waveforms = np.asarray(aligned_waveforms, dtype=float)
    if aligned_waveforms.size:
        mean_waveform = np.nanmean(aligned_waveforms, axis=0)
        filtered_metrics = _waveform_metric_block(mean_waveform, fs=fs)
    else:
        filtered_metrics = _waveform_metric_block(np.array([], dtype=float), fs=fs)

    raw_source = raw_waveforms if raw_waveforms is not None else aligned_waveforms
    raw_source = np.asarray(raw_source, dtype=float)
    if raw_source.size:
        raw_mean_waveform = np.nanmean(raw_source, axis=0)
        raw_metrics = _waveform_metric_block(raw_mean_waveform, fs=fs)
    else:
        raw_metrics = _waveform_metric_block(np.array([], dtype=float), fs=fs)

    return dict(
        hf1_ms=filtered_metrics["halfwidth1_ms"],
        hf2_ms=filtered_metrics["halfwidth2_ms"],
        halfwidth1_ms=filtered_metrics["halfwidth1_ms"],
        halfwidth2_ms=filtered_metrics["halfwidth2_ms"],
        peak_trough_ratio=filtered_metrics["peak_trough_ratio"],
        peak_distance_samples=filtered_metrics["peak_distance_samples"],
        peak_to_trough_delay_ms=filtered_metrics["peak_to_trough_delay_ms"],
        symmetry=filtered_metrics["symmetry"],
        max_slope=filtered_metrics["max_slope"],
        repol_slope=filtered_metrics["repol_slope"],
        centered_baseline=filtered_metrics["centered_baseline"],
        raw_hf_ms=raw_metrics["halfwidth1_ms"],
        raw_halfwidth1_ms=raw_metrics["halfwidth1_ms"],
        raw_halfwidth2_ms=raw_metrics["halfwidth2_ms"],
        raw_peak_trough_ratio=raw_metrics["peak_trough_ratio"],
        raw_peak_distance_samples=raw_metrics["peak_distance_samples"],
        raw_peak_to_trough_delay_ms=raw_metrics["peak_to_trough_delay_ms"],
        raw_symmetry=raw_metrics["symmetry"],
        raw_max_slope=raw_metrics["max_slope"],
        raw_repol_slope=raw_metrics["repol_slope"],
        raw_centered_baseline=raw_metrics["centered_baseline"],
    )

def waveform_shape_metrics(waveform, fs=20000):
    waveform = np.asarray(waveform, dtype=float)
    if waveform.size < 3 or not np.any(np.isfinite(waveform)):
        return np.nan, np.nan, np.nan, np.nan
    pos_peaks, _ = find_peaks(waveform)
    neg_peaks, _ = find_peaks(-waveform)
    all_peaks = np.unique(np.concatenate([pos_peaks, neg_peaks]))
    if all_peaks.size < 2:
        return np.nan, np.nan, np.nan, np.nan
    peak_vals = waveform[all_peaks]
    idx_order = np.argsort(-np.abs(peak_vals))
    main_idx = int(all_peaks[idx_order[0]])
    sec_idx = int(all_peaks[idx_order[1]])
    hw1 = halfwidth_interp(waveform, main_idx, fs)
    hw2 = halfwidth_interp(waveform, sec_idx, fs)
    pt_ratio = float(waveform[main_idx] / waveform[sec_idx]) if np.abs(waveform[sec_idx]) > 0 else np.nan
    peak_dist = int(main_idx - sec_idx)
    return hw1, hw2, pt_ratio, peak_dist

def waveform_extra_metrics(waveform, fs=20000):
    waveform = np.asarray(waveform, dtype=float)
    pos_peaks, _ = find_peaks(waveform)
    neg_peaks, _ = find_peaks(-waveform)
    if len(pos_peaks) == 0 or len(neg_peaks) == 0:
        return np.nan, np.nan, np.nan, np.nan
    trough = int(neg_peaks[np.argmin(waveform[neg_peaks])])
    peak = int(pos_peaks[np.argmax(waveform[pos_peaks])])
    peak_to_trough_delay = float(np.abs(peak - trough) / fs * 1000.0)
    left_hw = halfwidth_interp(waveform, trough, fs) if trough < peak else halfwidth_interp(waveform, peak, fs)
    right_hw = halfwidth_interp(waveform, peak, fs) if trough < peak else halfwidth_interp(waveform, trough, fs)
    symmetry = float(left_hw / right_hw) if np.isfinite(left_hw) and np.isfinite(right_hw) and right_hw != 0 else np.nan
    slope = float(np.nanmax(np.abs(np.diff(waveform))) * fs) if waveform.size > 1 else np.nan
    repol_slope = float((waveform[trough + 2] - waveform[trough]) / ((2.0) / fs)) if trough < waveform.size - 2 else np.nan
    return peak_to_trough_delay, symmetry, slope, repol_slope

def firing_stats(spikes, fs=20000):
    spikes = np.asarray(spikes, dtype=np.int64)
    if spikes.size < 2:
        return np.nan, np.nan
    isis = np.diff(np.sort(spikes)) / fs * 1000.0
    isi_cv = float(np.std(isis) / np.mean(isis)) if isis.size > 1 and np.mean(isis) != 0 else np.nan
    ref_viol = float(np.mean(isis < 2.0)) if isis.size > 1 else np.nan
    return isi_cv, ref_viol

def all_spike_unit_metrics(waveforms, spikes, fs=20000):
    waveforms = np.asarray(waveforms, dtype=float)
    spikes = np.asarray(spikes, dtype=np.int64)
    isi_cv, ref_viol = firing_stats(spikes, fs)
    metrics = compute_unit_waveform_metrics(waveforms, raw_waveforms=waveforms, fs=fs)
    metrics.update(dict(isi_cv=isi_cv, ref_viol=ref_viol))
    return metrics


def bandpass(lfp, fs, band):
    b, a = scipy.signal.butter(2, np.array(band) / (fs / 2.0), btype="band")
    return scipy.signal.filtfilt(b, a, np.asarray(lfp, dtype=float))

def compute_phase(lfp_band):
    analytic = scipy.signal.hilbert(lfp_band)
    return np.angle(analytic)

def compute_rayleigh_p(phases):
    phases = np.asarray(phases, dtype=float)
    n = phases.size
    if n == 0:
        return np.nan
    R = np.abs(np.sum(np.exp(1j * phases))) / n
    Z = n * R**2
    return float(np.exp(-Z))


def bin_spike_train(spike_idx, n_samples, fs, bin_size_s=1.0):
    spike_idx = np.asarray(spike_idx, dtype=np.int64)
    bin_size_samples = int(round(bin_size_s * fs))
    if bin_size_samples <= 0:
        raise ValueError("bin_size_s too small")
    n_bins = int(np.ceil(n_samples / bin_size_samples))
    edges = np.arange(0, n_bins + 1) * bin_size_samples
    counts, _ = np.histogram(spike_idx, bins=edges)
    fr = counts.astype(float) / bin_size_s
    edges_s = edges / fs
    return fr, edges_s

def population_coupling_from_fr_matrix(fr_matrix, method="pearson"):
    from scipy.stats import pearsonr, spearmanr
    fr_matrix = np.asarray(fr_matrix, dtype=float)
    n_units, _ = fr_matrix.shape
    pop_rate = np.nansum(fr_matrix, axis=0)
    coupling = np.full(n_units, np.nan, dtype=float)
    for i in range(n_units):
        fr_i = fr_matrix[i]
        pop_excl = pop_rate - fr_i
        mask = np.isfinite(fr_i) & np.isfinite(pop_excl)
        if mask.sum() < 10:
            continue
        if method == "spearman":
            r, _ = spearmanr(fr_i[mask], pop_excl[mask])
        else:
            r, _ = pearsonr(fr_i[mask], pop_excl[mask])
        coupling[i] = float(r)
    return coupling, pop_rate


def build_weighted_graph_from_adjacency(units, W, weight_threshold=0.0):
    units = list(units)
    W = np.asarray(W, dtype=float)
    n = len(units)
    G = nx.Graph()
    for u in units:
        G.add_node(u)
    for i in range(n):
        for j in range(i + 1, n):
            w = float(W[i, j])
            if np.isfinite(w) and w > weight_threshold:
                G.add_edge(units[i], units[j], weight=w)
    return G


def _graph_with_distance_weights(G: nx.Graph, eps: float = 1e-12) -> nx.Graph:
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    for u, v, d in G.edges(data=True):
        w = float(d.get("weight", np.nan))
        if np.isfinite(w) and w > 0:
            attrs = dict(d)
            attrs["distance"] = 1.0 / (w + eps)
            H.add_edge(u, v, **attrs)
    return H


def _weighted_global_efficiency(G: nx.Graph) -> float:
    n = G.number_of_nodes()
    if n < 2:
        return np.nan
    denom = float(n * (n - 1))
    inv_dist_sum = 0.0
    for src, lengths in nx.all_pairs_dijkstra_path_length(G, weight="distance"):
        for dst, dist in lengths.items():
            if src == dst:
                continue
            if np.isfinite(dist) and dist > 0:
                inv_dist_sum += 1.0 / dist
    return float(inv_dist_sum / denom)


def _weighted_local_efficiency_by_node(G: nx.Graph) -> Dict[int, float]:
    out = {}
    for node in G.nodes():
        nbrs = list(G.neighbors(node))
        if len(nbrs) < 2:
            out[node] = np.nan
            continue
        sub = G.subgraph(nbrs).copy()
        out[node] = _weighted_global_efficiency(sub)
    return out


def graph_node_metrics(G, prefix):
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["unit_id"])
    H = _graph_with_distance_weights(G)
    strength = pd.Series(dict(G.degree(weight="weight")), name=f"{prefix}_strength").astype(float)
    clustering = pd.Series(nx.clustering(G, weight="weight"), name=f"{prefix}_clustering").astype(float)
    degree = pd.Series(dict(G.degree()), name=f"{prefix}_degree").astype(int)
    try:
        pagerank = pd.Series(nx.pagerank(G, weight="weight"), name=f"{prefix}_pagerank").astype(float)
    except Exception:
        pagerank = pd.Series({n: np.nan for n in G.nodes()}, name=f"{prefix}_pagerank").astype(float)
    try:
        betweenness = pd.Series(nx.betweenness_centrality(H, weight="distance", normalized=True), name=f"{prefix}_betweenness").astype(float)
    except Exception:
        betweenness = pd.Series({n: np.nan for n in G.nodes()}, name=f"{prefix}_betweenness").astype(float)
    try:
        closeness = pd.Series(nx.closeness_centrality(H, distance="distance"), name=f"{prefix}_closeness").astype(float)
    except Exception:
        closeness = pd.Series({n: np.nan for n in G.nodes()}, name=f"{prefix}_closeness").astype(float)
    local_eff = pd.Series(_weighted_local_efficiency_by_node(H), name=f"{prefix}_local_efficiency").astype(float)
    hubness_z = pd.Series(zscore(strength.values), index=strength.index, name=f"{prefix}_hubness_z")
    df = pd.concat([strength, clustering, pagerank, betweenness, closeness, local_eff, degree, hubness_z], axis=1).reset_index().rename(columns={"index": "unit_id"})
    df["unit_id"] = df["unit_id"].astype(int)
    return df


def graph_global_metrics(
    G: nx.Graph,
    graph_name: str,
    node_celltype: Optional[Dict[int, str]] = None,
    node_layer: Optional[Dict[int, str]] = None,
) -> Dict[str, float]:
    H = _graph_with_distance_weights(G)
    if node_celltype:
        nx.set_node_attributes(H, {int(k): str(v) for k, v in node_celltype.items() if k in H.nodes()}, "celltype_norm")
    if node_layer:
        nx.set_node_attributes(H, {int(k): str(v) for k, v in node_layer.items() if k in H.nodes()}, "layer_norm")

    n_nodes = H.number_of_nodes()
    n_edges = H.number_of_edges()
    density = float(nx.density(H)) if n_nodes > 1 else np.nan
    n_components = int(nx.number_connected_components(H)) if n_nodes > 0 else 0

    if n_nodes > 0:
        lcc_nodes = max(nx.connected_components(H), key=len)
        H_lcc = H.subgraph(lcc_nodes).copy()
        n_nodes_lcc = int(H_lcc.number_of_nodes())
    else:
        H_lcc = nx.Graph()
        n_nodes_lcc = 0

    if n_nodes_lcc >= 2:
        try:
            avg_shortest_path_lcc = float(nx.average_shortest_path_length(H_lcc, weight="distance"))
        except Exception:
            avg_shortest_path_lcc = np.nan
    else:
        avg_shortest_path_lcc = np.nan

    global_efficiency = _weighted_global_efficiency(H)
    local_eff_by_node = _weighted_local_efficiency_by_node(H)
    local_vals = np.asarray(list(local_eff_by_node.values()), dtype=float) if local_eff_by_node else np.array([], dtype=float)
    local_efficiency_mean = float(np.nanmean(local_vals)) if local_vals.size and np.any(np.isfinite(local_vals)) else np.nan

    clust_vals = np.asarray(list(nx.clustering(H, weight="weight").values()), dtype=float) if n_nodes else np.array([], dtype=float)
    mean_clustering = float(np.nanmean(clust_vals)) if clust_vals.size and np.any(np.isfinite(clust_vals)) else np.nan

    if n_edges > 0:
        try:
            comms = list(nx.algorithms.community.greedy_modularity_communities(H, weight="weight"))
            modularity = float(nx.algorithms.community.quality.modularity(H, comms, weight="weight")) if comms else np.nan
            n_communities = int(len(comms))
        except Exception:
            modularity = np.nan
            n_communities = np.nan
    else:
        modularity = np.nan
        n_communities = np.nan

    try:
        assortativity_celltype = float(nx.attribute_assortativity_coefficient(H, "celltype_norm")) if n_edges > 0 else np.nan
    except Exception:
        assortativity_celltype = np.nan
    try:
        assortativity_layer = float(nx.attribute_assortativity_coefficient(H, "layer_norm")) if n_edges > 0 else np.nan
    except Exception:
        assortativity_layer = np.nan

    edge_weights = np.asarray([d.get("weight", np.nan) for _, _, d in H.edges(data=True)], dtype=float) if n_edges else np.array([], dtype=float)
    mean_edge_weight = float(np.nanmean(edge_weights)) if edge_weights.size and np.any(np.isfinite(edge_weights)) else np.nan

    return dict(
        graph_name=str(graph_name),
        n_nodes=int(n_nodes),
        n_edges=int(n_edges),
        density=density,
        n_components=n_components,
        n_nodes_lcc=n_nodes_lcc,
        avg_shortest_path_lcc=avg_shortest_path_lcc,
        global_efficiency=global_efficiency,
        local_efficiency_mean=local_efficiency_mean,
        mean_clustering=mean_clustering,
        modularity=modularity,
        n_communities=n_communities,
        assortativity_celltype=assortativity_celltype,
        assortativity_layer=assortativity_layer,
        mean_edge_weight=mean_edge_weight,
    )


def _edge_widths_from_weights(weights: np.ndarray, w_min: float = 0.5, w_max: float = 6.0) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if weights.size == 0:
        return weights
    w = weights.copy()
    w = w[np.isfinite(w)]
    if w.size == 0:
        return np.full_like(weights, w_min, dtype=float)
    mn, mx = float(np.min(w)), float(np.max(w))
    if mx - mn < 1e-12:
        return np.full_like(weights, (w_min + w_max) * 0.5, dtype=float)
    scaled = (weights - mn) / (mx - mn + 1e-12)
    return w_min + scaled * (w_max - w_min)


def _draw_electrode_margin(
    ax,
    num_channels: int,
    unit_channels_1based: List[int],
    title: str = "Electrode",
    unit_color_by_unit: Optional[Dict[int, str]] = None,
    unit_label_by_unit: Optional[Dict[int, str]] = None,
):
    """
    Simple schematic column of sites; highlight channels that have units.

    Not a real geometric shank model, but gives the "margin electrode" visual you requested.
    """
    ax.set_title(title, fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, num_channels + 1])
    ax.axis("off")

    unit_color_by_unit = unit_color_by_unit or {}
    unit_label_by_unit = unit_label_by_unit or {}

    # draw sites
    for ch in range(1, num_channels + 1):
        y = ch
        ax.plot([0.45], [y], marker="o", ms=4, color="#DDDDDD", mec="#AAAAAA", mew=0.5)

    channel_to_units = {}
    for unit_id, ch in unit_channels_1based:
        if np.isfinite(ch):
            channel_to_units.setdefault(int(ch), []).append(int(unit_id))

    for ch, units in sorted(channel_to_units.items()):
        if not (1 <= ch <= num_channels):
            continue
        x_offsets = np.linspace(-0.07, 0.07, num=max(1, len(units))) if len(units) > 1 else np.array([0.0])
        for x_off, unit_id in zip(x_offsets, units):
            x = 0.45 + float(x_off)
            color = unit_color_by_unit.get(int(unit_id), CELLTYPE_COLORS["unclassified"])
            label = unit_label_by_unit.get(int(unit_id), str(unit_id))
            ax.plot([x], [ch], marker="o", ms=7, color=color, mec="k", mew=0.6)
            ax.text(x + 0.02, ch, label, fontsize=7, va="center", ha="left")

    ax.text(0.1, num_channels + 0.5, f"n_ch={num_channels}", fontsize=8, va="bottom")


def plot_graph(
    G: nx.Graph,
    out_png: str,
    out_svg: str,
    title: str,
    node_color_by_unit: Optional[Dict[int, str]] = None,
    unit_channels_1based: Optional[Dict[int, int]] = None,
    num_channels: Optional[int] = None,
):
    """
    Improved graph plotting:
      - node colors by celltype
      - edge widths proportional to weight
      - electrode margin schematic
    """
    if node_color_by_unit is None:
        node_color_by_unit = {}
    if unit_channels_1based is None:
        unit_channels_1based = {}

    # Figure with margin panel
    fig = plt.figure(figsize=(12.5, 8))
    gs = fig.add_gridspec(1, 5, width_ratios=[4.3, 0.15, 0.55, 0.05, 0.0])

    ax = fig.add_subplot(gs[0, 0])
    ax_m = fig.add_subplot(gs[0, 2])

    pos = nx.spring_layout(G, seed=0, weight="weight") if G.number_of_nodes() else {}

    # Node colors
    node_colors = []
    for u in G.nodes():
        node_colors.append(node_color_by_unit.get(int(u), "#9E9E9E"))

    # Edge widths by weight
    weights = np.array([G[u][v].get("weight", 0.0) for u, v in G.edges()], dtype=float)
    widths = _edge_widths_from_weights(weights, w_min=0.7, w_max=6.0) if weights.size else np.array([])

    nx.draw_networkx_nodes(G, pos, node_size=320, node_color=node_colors, edgecolors="k", linewidths=0.6, ax=ax)
    if G.number_of_edges() > 0:
        nx.draw_networkx_edges(G, pos, width=widths, alpha=0.65, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    ax.set_title(title)
    if G.number_of_edges() == 0:
        ax.text(0.02, 0.02, "no edges above threshold", transform=ax.transAxes, fontsize=8, va="bottom")
    ax.axis("off")

    # Electrode margin
    if num_channels is not None and num_channels > 0:
        _draw_electrode_margin(
            ax_m,
            num_channels=int(num_channels),
            unit_channels_1based=[(int(u), unit_channels_1based.get(int(u), np.nan)) for u in G.nodes()],
            title="Electrode (unit channels)",
            unit_color_by_unit=node_color_by_unit,
            unit_label_by_unit={int(u): str(int(u)) for u in G.nodes()},
        )
    else:
        ax_m.axis("off")

    # Legend
    handles = []
    for lab, col in CELLTYPE_COLORS.items():
        handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=col, markeredgecolor="k", markersize=8, label=lab))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_svg)
    plt.close(fig)


def build_plv_graph_option_a(units, coupling_strength_by_unit, weight_threshold=0.0):
    units = list(units)
    S = np.array([float(coupling_strength_by_unit.get(u, 0.0) or 0.0) for u in units], dtype=float)
    W = np.outer(S, S)
    np.fill_diagonal(W, 0.0)
    G = build_weighted_graph_from_adjacency(units, W, weight_threshold=weight_threshold)
    return W, G


# -----------------------------
# CCG FIXES
# -----------------------------
def compute_ccg_directional(
    spike_times_ref,     # reference spikes (events at t=0)
    spike_times_target,  # target spikes to count around ref spikes
    fs,
    window_ms=50,
    bin_ms=1,
    normalize=False,
    exclude_zero=False,
):
    """
    Directional CCG:
      counts of (target - ref) in a lag window.

    This fixes the "same both directions" issue caused by the previous implementation
    selecting short/long and inverting symmetry.
    """
    window = int(window_ms * fs / 1000)
    bin_size = int(bin_ms * fs / 1000)
    if bin_size < 1:
        bin_size = 1

    bins = np.arange(-window, window + bin_size, bin_size)
    counts = np.zeros(len(bins) - 1, dtype=float)

    ref = np.asarray(spike_times_ref, dtype=np.int64)
    tgt = np.asarray(spike_times_target, dtype=np.int64)
    if ref.size == 0 or tgt.size == 0:
        centers = (bins[:-1] + bins[1:]) / 2 / fs * 1000
        return centers, counts

    ref = np.sort(ref)
    tgt = np.sort(tgt)

    # For each reference spike, count target spikes in [t-window, t+window]
    # Use searchsorted for efficiency
    for t in ref:
        lo = np.searchsorted(tgt, t - window, side="left")
        hi = np.searchsorted(tgt, t + window, side="right")
        if hi <= lo:
            continue
        diffs = tgt[lo:hi] - t
        if exclude_zero:
            diffs = diffs[diffs != 0]
        if diffs.size:
            hist, _ = np.histogram(diffs, bins=bins)
            counts += hist

    if normalize:
        mx = counts.max()
        counts = counts / mx if mx > 0 else counts

    centers = (bins[:-1] + bins[1:]) / 2 / fs * 1000
    return centers, counts


def save_corrmat_2x2_plot(
    unit_i,
    unit_j,
    si,
    sj,
    fs,
    out_png,
    out_svg,
    window_ms=50,
    bin_ms=1,
    title=None,
    unit_to_channel_1based: Optional[Dict[int, int]] = None,
):
    """
    2x2 plot:
      - Auto i (i relative to i)
      - i -> j (j relative to i)   [directional]
      - j -> i (i relative to j)   [directional]
      - Auto j
    """
    if unit_to_channel_1based is None:
        unit_to_channel_1based = {}

    # Autos (exclude zero)
    ac_x_i, ac_y_i = compute_ccg_directional(si, si, fs, window_ms=window_ms, bin_ms=bin_ms, exclude_zero=True)
    ac_x_j, ac_y_j = compute_ccg_directional(sj, sj, fs, window_ms=window_ms, bin_ms=bin_ms, exclude_zero=True)

    # Cross (directional)
    x_ij, y_ij = compute_ccg_directional(si, sj, fs, window_ms=window_ms, bin_ms=bin_ms, exclude_zero=False)
    x_ji, y_ji = compute_ccg_directional(sj, si, fs, window_ms=window_ms, bin_ms=bin_ms, exclude_zero=False)

    chi = unit_to_channel_1based.get(int(unit_i), None)
    chj = unit_to_channel_1based.get(int(unit_j), None)

    def fmt_ch(ch):
        return f"ch{int(ch)}" if ch is not None and np.isfinite(ch) else "ch?"

    fig, axs = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    axs[0, 0].bar(ac_x_i, ac_y_i, width=bin_ms, color="black")
    axs[0, 0].set_title(f"Auto: u{unit_i} ({fmt_ch(chi)})")

    axs[0, 1].bar(x_ij, y_ij, width=bin_ms, color="blue")
    axs[0, 1].set_title(f"Cross: u{unit_j} rel u{unit_i}\n(u{unit_i} → u{unit_j})")

    axs[1, 0].bar(x_ji, y_ji, width=bin_ms, color="red")
    axs[1, 0].set_title(f"Cross: u{unit_i} rel u{unit_j}\n(u{unit_j} → u{unit_i})")

    axs[1, 1].bar(ac_x_j, ac_y_j, width=bin_ms, color="black")
    axs[1, 1].set_title(f"Auto: u{unit_j} ({fmt_ch(chj)})")

    for ax in axs.ravel():
        ax.axvline(0, color="gray", lw=1, alpha=0.6)
        ax.set_xlabel("Lag (ms)")
        ax.set_ylabel("Count")
        ax.set_xlim([-window_ms, window_ms])

    if title is None:
        title = f"Correlogram matrix: u{unit_i} ({fmt_ch(chi)}) & u{unit_j} ({fmt_ch(chj)})"
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fig.savefig(out_png, dpi=220)
    fig.savefig(out_svg)
    plt.close(fig)

    return (x_ij, y_ij), (x_ji, y_ji)


def infer_putative_connection_from_ccg(centers_ms, counts,
                                      mono_window_ms=(1, 5),
                                      baseline_window_ms=(20, 50),
                                      z_thresh=4.0):
    centers_ms = np.asarray(centers_ms, dtype=float)
    counts = np.asarray(counts, dtype=float)

    base_mask = (np.abs(centers_ms) >= baseline_window_ms[0]) & (np.abs(centers_ms) <= baseline_window_ms[1])
    mono_mask = (centers_ms >= mono_window_ms[0]) & (centers_ms <= mono_window_ms[1])

    if not base_mask.any() or not mono_mask.any():
        return dict(label="none", peak_z=np.nan, trough_z=np.nan, baseline=np.nan, baseline_std=np.nan,
                    peak_latency_ms=np.nan, peak_value=np.nan, trough_value=np.nan)

    base = counts[base_mask]
    baseline = float(np.mean(base))
    baseline_std = float(np.std(base, ddof=1)) if base.size > 1 else float(np.std(base))

    mono = counts[mono_mask]
    mono_centers = centers_ms[mono_mask]

    peak_idx = int(np.argmax(mono))
    trough_idx = int(np.argmin(mono))
    peak_val = float(mono[peak_idx])
    trough_val = float(mono[trough_idx])
    peak_lat = float(mono_centers[peak_idx])

    peak_z = (peak_val - baseline) / (baseline_std + 1e-12) if baseline_std > 0 else np.nan
    trough_z = (baseline - trough_val) / (baseline_std + 1e-12) if baseline_std > 0 else np.nan

    label = "none"
    if np.isfinite(peak_z) and peak_z >= z_thresh:
        label = "putative_monosynaptic_exc"
    elif np.isfinite(trough_z) and trough_z >= z_thresh:
        label = "putative_monosynaptic_inh"

    return dict(
        label=label,
        peak_z=float(peak_z) if np.isfinite(peak_z) else np.nan,
        trough_z=float(trough_z) if np.isfinite(trough_z) else np.nan,
        baseline=baseline,
        baseline_std=baseline_std,
        peak_latency_ms=peak_lat,
        peak_value=peak_val,
        trough_value=trough_val,
    )


def compute_dimensionality_metrics(fr_matrix, method="covariance"):
    """
    Compute dimensionality metrics from firing rate matrix.
    
    Parameters:
    - fr_matrix: np.ndarray, shape (n_units, n_time_bins)
    - method: str, "covariance" for PCA on covariance matrix
    
    Returns:
    - dict with pc1_variance, participation_ratio, eigenspectrum_entropy, eigenspectrum_entropy_normalized
    """
    fr_matrix = np.asarray(fr_matrix, dtype=float)
    if fr_matrix.shape[0] < 2 or fr_matrix.shape[1] < 2:
        return {
            "pc1_variance": np.nan,
            "participation_ratio": np.nan,
            "eigenspectrum_entropy": np.nan,
            "eigenspectrum_entropy_normalized": np.nan
        }
    
    # Compute covariance matrix
    cov_mat = np.cov(fr_matrix)
    
    # Eigenvalue decomposition
    eigenvalues, _ = np.linalg.eigh(cov_mat)
    eigenvalues = eigenvalues[::-1]  # descending order
    
    # Ensure positive eigenvalues
    eigenvalues = np.maximum(eigenvalues, 0)
    
    total_var = np.sum(eigenvalues)
    if total_var == 0:
        return {
            "pc1_variance": np.nan,
            "participation_ratio": np.nan,
            "eigenspectrum_entropy": np.nan,
            "eigenspectrum_entropy_normalized": np.nan
        }
    
    # PC1 variance explained
    pc1_variance = eigenvalues[0] / total_var
    
    # Participation ratio: (sum λ_i)^2 / sum λ_i^2
    participation_ratio = total_var**2 / np.sum(eigenvalues**2)
    
    # Eigenspectrum entropy: -sum (p_i * log p_i) where p_i = λ_i / sum λ_j
    p = eigenvalues / total_var
    p = p[p > 0]  # avoid log(0)
    eigenspectrum_entropy = -np.sum(p * np.log(p))
    
    # Normalized entropy (divided by log(n) for max entropy in uniform case)
    n = len(eigenvalues)
    max_entropy = np.log(n) if n > 1 else 1
    eigenspectrum_entropy_normalized = eigenspectrum_entropy / max_entropy
    
    return {
        "pc1_variance": float(pc1_variance),
        "participation_ratio": float(participation_ratio),
        "eigenspectrum_entropy": float(eigenspectrum_entropy),
        "eigenspectrum_entropy_normalized": float(eigenspectrum_entropy_normalized)
    }

def analyze_recording(
    base_prefix,
    recording_id,
    num_channels,
    in_vitro_cells_df_for_recording,
    fs=20000,
    lowcut=300.0,
    highcut=3000.0,
    uV_per_bit=0.30518,
    spike_snip_ms=2.0,
    plv_n_surr=500,
    coupling_n_spikes_min=10,
    coupling_alpha=0.05,
    bin_size_s_pop=1.0,
    pop_coupling_method="pearson",
    graph_weight_threshold=0.0,
    ccg_window_ms=50,
    ccg_bin_ms=1,
    out_dir=None,
    max_corrmat_pairs=400,
    frsim_percentile=75,
    frsim_positive_only=True,
    lfp_fs=1000,
    filter_snippets=False,
):
    if out_dir is None:
        out_dir = os.path.dirname(base_prefix)
    rec_out = os.path.join(out_dir, recording_id)
    safe_makedirs(rec_out)

    paths = infer_recording_paths_from_unit_stats_csv(base_prefix + ".bin")
    for k in ("bin_path", "cluster_info_tsv", "spike_times_npy", "spike_clusters_npy"):
        if not os.path.exists(paths[k]):
            raise FileNotFoundError(paths[k])

    loaded_data = load_bin_lfp(paths["bin_path"], num_channels=num_channels, dtype=np.int16)

    spike_times = np.load(paths["spike_times_npy"])
    spike_clusters = np.load(paths["spike_clusters_npy"])
    cluster_info = pd.read_csv(paths["cluster_info_tsv"], sep="\t")

    present_units = set(np.unique(spike_clusters).astype(int).tolist())
    if "group" not in cluster_info.columns:
        raise ValueError(f"{paths['cluster_info_tsv']} must contain column: group")
    good_units = cluster_info[(cluster_info["group"] == "good") & (cluster_info["cluster_id"].isin(present_units))]["cluster_id"].astype(int).tolist()

    requested_units = in_vitro_cells_df_for_recording["unit_id"].astype(int).tolist()
    good_units = [u for u in good_units if u in requested_units]

    channel_mapping = {int(r["cluster_id"]): int(r["ch"]) for _, r in cluster_info.iterrows()}
    epochs_by_band = load_epochs_from_csvs_seconds(base_prefix, bands=("delta", "theta", "low_gamma", "high_gamma"))

    band_freqs = {
        "delta": (0.5, 3.5),
        "theta": (4, 8),
        "low_gamma": (30, 50),
        "high_gamma": (50, 80),
    }

    n_samples = loaded_data.shape[1]
    duration_s = n_samples / fs if n_samples > 0 else np.nan
    spike_window = int(round((spike_snip_ms / 1000.0) * fs))

    lfp_ds_cache = {}
    phase_ds_cache = {}
    filt_ds_cache = {}

    unit_rows = []
    plv_epoch_rows = []

    fr_matrix = []
    unit_order = []
    unit_spikes = {}

    for unit_id in good_units:
        meta_row = in_vitro_cells_df_for_recording[in_vitro_cells_df_for_recording["unit_id"].astype(int) == int(unit_id)].iloc[0]
        celltype = str(meta_row.get("label", ""))
        celltype_norm = normalize_celltype(celltype)
        align_mode = alignment_to_mode(meta_row.get("alignment", "none"))
        state = meta_row.get("state", "")
        cell_uid = make_cell_uid(recording_id, state, unit_id)

        channel0 = int(channel_mapping.get(unit_id, -1))
        if channel0 < 0:
            continue
        channel_1based = int(channel0 + 1)

        spikes = spike_times[spike_clusters == unit_id]
        spikes = deduplicate_spikes(spikes, min_isi_ms=1, fs=fs)
        spike_idx = np.asarray(spikes, dtype=np.int64)
        unit_spikes[int(unit_id)] = spike_idx

        fr = float(spike_idx.size / duration_s) if np.isfinite(duration_s) and duration_s > 0 else np.nan
        max_fr_10s, _, _ = compute_max_firing_rate_10s(spike_idx, fs=fs)
        max_fr_1s, _, _ = compute_max_firing_rate_window(spike_idx, fs=fs, window_sec=1.0, step_sec=0.01)

        burst_mask = find_bursts_silence_consecutive_isi(spike_idx, fs=fs, burst_n=3, max_isi_ms=10, prepost_quiet_ms=20)
        burstiness_percent = 100.0 * (burst_mask.sum() / max(1, spike_idx.size))
        isis_ms = np.diff(np.sort(spike_idx)) / fs * 1000.0
        bursti_2 = 100.0 * float(np.mean(isis_ms < 20.0)) if isis_ms.size else 0.0

        ac_x, ac_y = compute_autocorrelogram(spike_idx, fs=fs, bin_size_ms=1, window_ms=40)
        evenness, has_short_peak = autocorr_evenness(ac_x, ac_y)

        raw_snippets = []
        for t in spike_idx:
            t = int(t)
            if t - spike_window >= 0 and t + spike_window < n_samples:
                raw_snippets.append(loaded_data[channel0, t - spike_window:t + spike_window].astype(float))
        raw_snippets = np.asarray(raw_snippets, dtype=float)

        snippets = raw_snippets.copy()

        if filter_snippets and snippets.size:
            b, a = scipy.signal.butter(2, [lowcut / (fs / 2.0), highcut / (fs / 2.0)], btype="band")
            snippets = scipy.signal.filtfilt(b, a, snippets, axis=1)

        if snippets.shape[0] > 0:
            if align_mode != "none":
                aligned, _ = align_waveforms(snippets, align_mode=("positive" if align_mode == "positive" else "negative"))
                raw_aligned, _ = align_waveforms(raw_snippets, align_mode=("positive" if align_mode == "positive" else "negative")) if raw_snippets.size else (raw_snippets, np.nan)
            else:
                aligned = snippets
                raw_aligned = raw_snippets
            avg_wf = np.nanmean(aligned, axis=0)
            wf_metrics = compute_unit_waveform_metrics(aligned, raw_waveforms=raw_aligned, fs=fs)
            isi_cv, ref_viol = firing_stats(spike_idx, fs=fs)
            wf_metrics.update(dict(isi_cv=isi_cv, ref_viol=ref_viol))
        else:
            avg_wf = np.full(spike_window * 2, np.nan)
            wf_metrics = {k: np.nan for k in [
                "hf1_ms", "hf2_ms", "halfwidth1_ms", "halfwidth2_ms", "peak_trough_ratio", "peak_distance_samples",
                "peak_to_trough_delay_ms", "symmetry", "max_slope", "repol_slope", "centered_baseline",
                "raw_hf_ms", "raw_halfwidth1_ms", "raw_halfwidth2_ms", "raw_peak_trough_ratio",
                "raw_peak_distance_samples", "raw_peak_to_trough_delay_ms", "raw_symmetry", "raw_max_slope",
                "raw_repol_slope", "raw_centered_baseline", "isi_cv", "ref_viol"
            ]}

        amplitude_uV = (np.nanmax(np.abs(avg_wf)) * uV_per_bit) if np.any(np.isfinite(avg_wf)) else np.nan

        fr_ts, _edges_s = bin_spike_train(spike_idx, n_samples=n_samples, fs=fs, bin_size_s=bin_size_s_pop)
        fr_matrix.append(fr_ts)
        unit_order.append(int(unit_id))

        coupling_strength = {"low_gamma": 0.0, "high_gamma": 0.0}
        plv_summary = {}

        lfp_raw = loaded_data[channel0].astype(float)
        if channel0 not in lfp_ds_cache:
            lfp_ds, decim = downsample_lfp_channel(lfp_raw, fs_in=fs, fs_out=lfp_fs)
            if decim is None:
                raise ValueError("Expected integer decimation for fs=20000 -> lfp_fs=1000.")
            lfp_ds_cache[channel0] = lfp_ds
        lfp_ds = lfp_ds_cache[channel0]

        for band_name, band in band_freqs.items():
            key = (channel0, band_name)
            if key not in phase_ds_cache:
                filt_ds = bandpass(lfp_ds, lfp_fs, band)
                filt_ds_cache[key] = filt_ds
                phase_ds_cache[key] = compute_phase(filt_ds)
            phase_ds = phase_ds_cache[key]

            epochs = epochs_by_band.get(band_name, [])
            plvs_all = []
            plvs_sig = []
            best_rayleigh_p_sig = np.nan
            best_p_perm_sig = np.nan
            n_sig_epochs = 0

            for (start_s, end_s) in epochs:
                spikes_epoch_20k = spikes_in_epoch_seconds(spike_idx, fs_spike=fs, start_s=start_s, end_s=end_s)
                nsp = int(spikes_epoch_20k.size)
                if nsp < 1:
                    continue

                phases_spikes, spike_idx_lfp = phase_at_spikes_from_downsampled(phase_ds, spikes_epoch_20k, fs_spike=fs, fs_lfp=lfp_fs)
                plv_val = float(np.abs(np.mean(np.exp(1j * phases_spikes)))) if phases_spikes.size else np.nan
                ray_p = compute_rayleigh_p(phases_spikes) if phases_spikes.size else np.nan
                preferred_phase_rad = float(np.angle(np.mean(np.exp(1j * phases_spikes)))) if phases_spikes.size and np.isfinite(plv_val) and plv_val > 0 else np.nan
                preferred_phase_deg = float(np.degrees(preferred_phase_rad)) if np.isfinite(preferred_phase_rad) else np.nan

                if (nsp >= coupling_n_spikes_min) and np.isfinite(ray_p) and (ray_p <= coupling_alpha) and (spike_idx_lfp.size >= 5):
                    L = len(phase_ds)
                    rng = np.random.RandomState(None)
                    plv_surr = np.zeros(int(plv_n_surr), dtype=float)
                    for i_s in range(int(plv_n_surr)):
                        shift = rng.randint(0, L)
                        ph_s = phase_ds[(spike_idx_lfp + shift) % L]
                        plv_surr[i_s] = np.abs(np.mean(np.exp(1j * ph_s)))
                    p_perm = (np.sum(plv_surr >= plv_val) + 1) / (int(plv_n_surr) + 1)
                else:
                    p_perm = np.nan

                coupling_sig = bool(
                    (nsp >= coupling_n_spikes_min)
                    and np.isfinite(ray_p) and (ray_p <= coupling_alpha)
                    and np.isfinite(p_perm) and (p_perm <= coupling_alpha)
                )

                plv_epoch_rows.append(dict(
                    recording_id=recording_id,
                    base_prefix=base_prefix,
                    state=str(state),
                    cell_uid=cell_uid,
                    unit_id=int(unit_id),
                    channel=int(channel_1based),
                    celltype=celltype,
                    celltype_norm=celltype_norm,
                    band=band_name,
                    start_s=float(start_s),
                    end_s=float(end_s),
                    n_spikes=nsp,
                    plv=float(plv_val) if np.isfinite(plv_val) else np.nan,
                    rayleigh_p=float(ray_p) if np.isfinite(ray_p) else np.nan,
                    p_perm=float(p_perm) if np.isfinite(p_perm) else np.nan,
                    coupling_significant=coupling_sig,
                    preferred_phase_rad=preferred_phase_rad,
                    preferred_phase_deg=preferred_phase_deg
                ))

                if np.isfinite(plv_val):
                    plvs_all.append(float(plv_val))

                if coupling_sig and np.isfinite(plv_val):
                    plvs_sig.append(float(plv_val))
                    n_sig_epochs += 1
                    best_rayleigh_p_sig = float(ray_p) if np.isnan(best_rayleigh_p_sig) else min(best_rayleigh_p_sig, float(ray_p))
                    best_p_perm_sig = float(p_perm) if np.isnan(best_p_perm_sig) else min(best_p_perm_sig, float(p_perm))

            plv_summary[band_name] = dict(
                n_epochs=int(len(epochs)),
                mean_plv=float(np.mean(plvs_all)) if plvs_all else np.nan,
                max_plv=float(np.max(plvs_all)) if plvs_all else np.nan,
                mean_plv_sig=float(np.mean(plvs_sig)) if plvs_sig else np.nan,
                max_plv_sig=float(np.max(plvs_sig)) if plvs_sig else np.nan,
                n_coupling_sig_epochs=int(n_sig_epochs),
                best_rayleigh_p_sig=best_rayleigh_p_sig,
                best_p_perm_sig=best_p_perm_sig,
            )

            if band_name in ("low_gamma", "high_gamma"):
                coupling_strength[band_name] = float(np.max(plvs_sig)) if plvs_sig else 0.0

        
        unit_row = dict(
            recording_id=recording_id,
            base_prefix=base_prefix,
            state=str(state),
            state_letter=state_to_letter(state),
            cell_uid=cell_uid,
            unit_id=int(unit_id),
            channel=int(channel_1based),
            celltype=celltype,
            celltype_norm=celltype_norm,
            alignment=str(meta_row.get("alignment", "")),
            biphasic=bool(biphasic_to_bool(meta_row.get("biphasic", "none"))),
            num_channels=int(num_channels),
            duration_s=float(duration_s),
            firing_rate_Hz=float(fr) if np.isfinite(fr) else np.nan,
            max_firing_rate_10s_Hz=float(max_fr_10s) if np.isfinite(max_fr_10s) else np.nan,
            max_firing_rate_1s_Hz=float(max_fr_1s) if np.isfinite(max_fr_1s) else np.nan,
            burstiness_percent=float(burstiness_percent),
            bursti_2=float(bursti_2),
            autocorr_evenness=float(evenness) if np.isfinite(evenness) else np.nan,
            autocorr_short_lag_peak=bool(has_short_peak),
            amplitude_uV=float(amplitude_uV) if np.isfinite(amplitude_uV) else np.nan,
            coupling_strength_low_gamma=float(coupling_strength["low_gamma"]),
            coupling_strength_high_gamma=float(coupling_strength["high_gamma"]),
        )
        unit_row.update(wf_metrics)

        for bn, d in plv_summary.items():
            unit_row[f"n_epochs_{bn}"] = d["n_epochs"]
            unit_row[f"mean_plv_{bn}"] = d["mean_plv"]
            unit_row[f"max_plv_{bn}"] = d["max_plv"]
            unit_row[f"mean_plv_sig_{bn}"] = d["mean_plv_sig"]
            unit_row[f"max_plv_sig_{bn}"] = d["max_plv_sig"]
            unit_row[f"n_coupling_sig_epochs_{bn}"] = d["n_coupling_sig_epochs"]
            unit_row[f"best_rayleigh_p_sig_{bn}"] = d["best_rayleigh_p_sig"]
            unit_row[f"best_p_perm_sig_{bn}"] = d["best_p_perm_sig"]
            unit_row[f"coupling_significant_{bn}"] = bool(d["n_coupling_sig_epochs"] > 0)
            unit_row[f"plv_band_significant_{bn}"] = bool(d["n_coupling_sig_epochs"] > 0)

        unit_rows.append(unit_row)

    # population coupling
    fr_matrix = np.vstack(fr_matrix) if len(fr_matrix) else np.empty((0, 0))
    dim_metrics = compute_dimensionality_metrics(fr_matrix)
    if fr_matrix.size > 0:
        coupling_vals, _pop_rate = population_coupling_from_fr_matrix(fr_matrix, method=pop_coupling_method)
        pc_df = pd.DataFrame({
            "recording_id": recording_id,
            "unit_id": unit_order,
            "population_coupling": coupling_vals,
            "mean_fr_binned_Hz": np.nanmean(fr_matrix, axis=1),
            "bin_size_s": float(bin_size_s_pop),
        })
    else:
        pc_df = pd.DataFrame(columns=["recording_id", "unit_id", "population_coupling", "mean_fr_binned_Hz", "bin_size_s"])

    df_units = pd.DataFrame(unit_rows)
    df_units['dimensionality_pc1_variance'] = dim_metrics['pc1_variance']
    df_units['dimensionality_participation_ratio'] = dim_metrics['participation_ratio']
    df_units['dimensionality_eigenspectrum_entropy'] = dim_metrics['eigenspectrum_entropy']
    df_units['dimensionality_eigenspectrum_entropy_normalized'] = dim_metrics['eigenspectrum_entropy_normalized']
    df_plv_epochs = pd.DataFrame(plv_epoch_rows)

    df_units = df_units.merge(pc_df[["unit_id", "population_coupling"]], on="unit_id", how="left")

    units_list = df_units["unit_id"].astype(int).tolist()

    # mapping for plotting (channels + node colors)
    unit_to_channel_1based = {int(r["unit_id"]): int(r["channel"]) for _, r in df_units.iterrows() if np.isfinite(r.get("channel", np.nan))}
    unit_to_color = {}
    unit_to_celltype_norm = {}
    unit_to_layer_norm = {}
    for _, r in df_units.iterrows():
        u = int(r["unit_id"])
        ct = str(r.get("celltype_norm", "unclassified"))
        unit_to_color[u] = CELLTYPE_COLORS.get(ct, CELLTYPE_COLORS["unclassified"])
        unit_to_celltype_norm[u] = ct
        unit_to_layer_norm[u] = str(r.get("layer_norm", "other"))

    global_graph_rows = []

    # Graphs: PLV option A low/high
    W_low, G_low = build_plv_graph_option_a(
        units_list,
        {int(r["unit_id"]): float(r.get("coupling_strength_low_gamma", 0.0) or 0.0) for _, r in df_units.iterrows()},
        weight_threshold=graph_weight_threshold,
    )
    low_adj = pd.DataFrame(W_low, index=units_list, columns=units_list)
    low_nodes = graph_node_metrics(G_low, prefix="plv_low_gamma")

    W_high, G_high = build_plv_graph_option_a(
        units_list,
        {int(r["unit_id"]): float(r.get("coupling_strength_high_gamma", 0.0) or 0.0) for _, r in df_units.iterrows()},
        weight_threshold=graph_weight_threshold,
    )
    high_adj = pd.DataFrame(W_high, index=units_list, columns=units_list)
    high_nodes = graph_node_metrics(G_high, prefix="plv_high_gamma")

    df_units = df_units.merge(low_nodes, on="unit_id", how="left")
    df_units = df_units.merge(high_nodes, on="unit_id", how="left")

    low_adj.to_csv(os.path.join(rec_out, f"{recording_id}_plv_graph_low_gamma_adj.csv"))
    high_adj.to_csv(os.path.join(rec_out, f"{recording_id}_plv_graph_high_gamma_adj.csv"))
    low_nodes.to_csv(os.path.join(rec_out, f"{recording_id}_plv_graph_low_gamma_nodes.csv"), index=False)
    high_nodes.to_csv(os.path.join(rec_out, f"{recording_id}_plv_graph_high_gamma_nodes.csv"), index=False)

    plot_graph(
        G_low,
        os.path.join(rec_out, f"{recording_id}_plv_graph_low_gamma.png"),
        os.path.join(rec_out, f"{recording_id}_plv_graph_low_gamma.svg"),
        f"{recording_id} low_gamma PLV co-coupling graph (Option A)",
        node_color_by_unit=unit_to_color,
        unit_channels_1based=unit_to_channel_1based,
        num_channels=int(num_channels),
    )
    global_graph_rows.append(graph_global_metrics(G_low, "plv_low_gamma", unit_to_celltype_norm, unit_to_layer_norm))
    plot_graph(
        G_high,
        os.path.join(rec_out, f"{recording_id}_plv_graph_high_gamma.png"),
        os.path.join(rec_out, f"{recording_id}_plv_graph_high_gamma.svg"),
        f"{recording_id} high_gamma PLV co-coupling graph (Option A)",
        node_color_by_unit=unit_to_color,
        unit_channels_1based=unit_to_channel_1based,
        num_channels=int(num_channels),
    )
    global_graph_rows.append(graph_global_metrics(G_high, "plv_high_gamma", unit_to_celltype_norm, unit_to_layer_norm))

    # pop-coupling scalar graph
    popc = df_units.set_index("unit_id")["population_coupling"].to_dict()
    pop_strength = {int(u): float(max(popc.get(int(u), 0.0) if np.isfinite(popc.get(int(u), np.nan)) else 0.0, 0.0)) for u in units_list}
    W_pop_scalar = np.outer([pop_strength[u] for u in units_list], [pop_strength[u] for u in units_list]).astype(float)
    np.fill_diagonal(W_pop_scalar, 0.0)
    G_pop_scalar = build_weighted_graph_from_adjacency(units_list, W_pop_scalar, weight_threshold=graph_weight_threshold)
    pop_scalar_adj = pd.DataFrame(W_pop_scalar, index=units_list, columns=units_list)
    pop_scalar_nodes = graph_node_metrics(G_pop_scalar, prefix="popc_scalar")

    pop_scalar_adj.to_csv(os.path.join(rec_out, f"{recording_id}_popcoupling_graph_scalar_adj.csv"))
    pop_scalar_nodes.to_csv(os.path.join(rec_out, f"{recording_id}_popcoupling_graph_scalar_nodes.csv"), index=False)
    plot_graph(
        G_pop_scalar,
        os.path.join(rec_out, f"{recording_id}_popcoupling_graph_scalar.png"),
        os.path.join(rec_out, f"{recording_id}_popcoupling_graph_scalar.svg"),
        f"{recording_id} population-coupling scalar graph",
        node_color_by_unit=unit_to_color,
        unit_channels_1based=unit_to_channel_1based,
        num_channels=int(num_channels),
    )
    global_graph_rows.append(graph_global_metrics(G_pop_scalar, "popc_scalar", unit_to_celltype_norm, unit_to_layer_norm))
    df_units = df_units.merge(pop_scalar_nodes, on="unit_id", how="left")

    # FR similarity graph
    if fr_matrix.size > 0 and fr_matrix.shape[0] >= 2:
        C = np.corrcoef(fr_matrix)
        C = np.asarray(C, dtype=float)
        C[np.isnan(C)] = 0.0
        if frsim_positive_only:
            C = np.where(C > 0, C, 0.0)
        np.fill_diagonal(C, 0.0)
        thr_frsim = percentile_threshold_from_weights(C, pct=frsim_percentile, positive_only=frsim_positive_only)
        G_frsim = build_weighted_graph_from_adjacency(units_list, C, weight_threshold=thr_frsim)
        frsim_adj = pd.DataFrame(C, index=units_list, columns=units_list)
        frsim_nodes = graph_node_metrics(G_frsim, prefix="frsim")
    else:
        thr_frsim = 0.0
        frsim_adj = pd.DataFrame(np.zeros((len(units_list), len(units_list))), index=units_list, columns=units_list)
        frsim_nodes = pd.DataFrame({"unit_id": units_list})
        G_frsim = nx.Graph()

    frsim_adj.to_csv(os.path.join(rec_out, f"{recording_id}_popcoupling_graph_frsim_adj.csv"))
    frsim_nodes.to_csv(os.path.join(rec_out, f"{recording_id}_popcoupling_graph_frsim_nodes.csv"), index=False)
    plot_graph(
        G_frsim,
        os.path.join(rec_out, f"{recording_id}_popcoupling_graph_frsim.png"),
        os.path.join(rec_out, f"{recording_id}_popcoupling_graph_frsim.svg"),
        f"{recording_id} FR similarity graph (thr={thr_frsim:.3f})",
        node_color_by_unit=unit_to_color,
        unit_channels_1based=unit_to_channel_1based,
        num_channels=int(num_channels),
    )
    global_graph_rows.append(graph_global_metrics(G_frsim, "frsim", unit_to_celltype_norm, unit_to_layer_norm))
    df_units = df_units.merge(frsim_nodes, on="unit_id", how="left")

    df_graph_global = pd.DataFrame(global_graph_rows)
    df_graph_global.to_csv(os.path.join(rec_out, f"{recording_id}_graph_global_metrics.csv"), index=False)

    # Correlation matrices (2x2)
    corrmat_dir = os.path.join(rec_out, "corrmats_2x2")
    safe_makedirs(corrmat_dir)

    pairs = [(units_list[i], units_list[j]) for i in range(len(units_list)) for j in range(i + 1, len(units_list))]
    if len(pairs) > max_corrmat_pairs:
        pairs = pairs[:max_corrmat_pairs]

    conn_rows = []
    for (ui, uj) in pairs:
        si = unit_spikes.get(int(ui), np.array([], dtype=int))
        sj = unit_spikes.get(int(uj), np.array([], dtype=int))
        if si.size == 0 or sj.size == 0:
            continue

        out_png = os.path.join(corrmat_dir, f"{recording_id}_corrmat_u{ui}_u{uj}.png")
        out_svg = os.path.join(corrmat_dir, f"{recording_id}_corrmat_u{ui}_u{uj}.svg")

        (x_ij, y_ij), (x_ji, y_ji) = save_corrmat_2x2_plot(
            ui, uj, si, sj, fs,
            out_png, out_svg,
            window_ms=ccg_window_ms, bin_ms=ccg_bin_ms,
            title=f"{recording_id}: units {ui} & {uj}",
            unit_to_channel_1based=unit_to_channel_1based,
        )

        inf_ij = infer_putative_connection_from_ccg(x_ij, y_ij)
        inf_ji = infer_putative_connection_from_ccg(x_ji, y_ji)

        def score_inf(d):
            return max(d.get("peak_z", np.nan) if np.isfinite(d.get("peak_z", np.nan)) else -np.inf,
                       d.get("trough_z", np.nan) if np.isfinite(d.get("trough_z", np.nan)) else -np.inf)

        best_dir = "none"
        best_label = "none"
        best = inf_ij
        if inf_ij["label"] != "none" or inf_ji["label"] != "none":
            if score_inf(inf_ij) >= score_inf(inf_ji):
                best_dir = f"{ui}->{uj}"
                best_label = inf_ij["label"]
                best = inf_ij
            else:
                best_dir = f"{uj}->{ui}"
                best_label = inf_ji["label"]
                best = inf_ji

        conn_rows.append(dict(
            recording_id=recording_id,
            unit_i=int(ui),
            unit_j=int(uj),
            unit_i_channel=int(unit_to_channel_1based.get(int(ui), -1)),
            unit_j_channel=int(unit_to_channel_1based.get(int(uj), -1)),
            inferred_direction=best_dir,
            inferred_type=best_label,
            peak_latency_ms=best.get("peak_latency_ms", np.nan),
            peak_z=best.get("peak_z", np.nan),
            trough_z=best.get("trough_z", np.nan),
            baseline=best.get("baseline", np.nan),
            baseline_std=best.get("baseline_std", np.nan),
            peak_value=best.get("peak_value", np.nan),
            trough_value=best.get("trough_value", np.nan),
            corrmat_png=os.path.basename(out_png),
            corrmat_svg=os.path.basename(out_svg),
        ))

    conn_df = pd.DataFrame(conn_rows)
    conn_df.to_csv(os.path.join(rec_out, f"{recording_id}_connections_putative.csv"), index=False)

    # Save outputs (PCA tables removed)
    df_units.to_csv(os.path.join(rec_out, f"{recording_id}_unit_stats.csv"), index=False)
    df_plv_epochs.to_csv(os.path.join(rec_out, f"{recording_id}_plv_epochs.csv"), index=False)
    pc_df.to_csv(os.path.join(rec_out, f"{recording_id}_population_coupling.csv"), index=False)

    # Excel output: remove PCA sheets
    def write_excel_openpyxl(path, sheets):
        from openpyxl import Workbook
        from openpyxl.utils.dataframe import dataframe_to_rows
        wb = Workbook()
        default = wb.active
        wb.remove(default)
        for name, df in sheets.items():
            ws = wb.create_sheet(title=str(name)[:31])
            if df is None:
                ws.append(["(None)"])
                continue
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
            for r in dataframe_to_rows(df, index=False, header=True):
                ws.append(r)
        wb.save(path)

    excel_path = os.path.join(rec_out, f"{recording_id}_analysis.xlsx")
    sheets = {
        "units": df_units,
        "plv_epochs": df_plv_epochs,
        "population_coupling": pc_df,
        "putative_connections": conn_df,
        "plv_low_gamma_nodes": low_nodes,
        "plv_high_gamma_nodes": high_nodes,
        "plv_low_gamma_adj": low_adj.reset_index().rename(columns={"index": "unit_id"}),
        "plv_high_gamma_adj": high_adj.reset_index().rename(columns={"index": "unit_id"}),
        "popc_scalar_nodes": pop_scalar_nodes,
        "popc_scalar_adj": pop_scalar_adj.reset_index().rename(columns={"index": "unit_id"}),
        "frsim_nodes": frsim_nodes,
        "frsim_adj": frsim_adj.reset_index().rename(columns={"index": "unit_id"}),
        "graph_global_metrics": df_graph_global,
    }
    write_excel_openpyxl(excel_path, sheets)

    print(f"[OK] {recording_id}: outputs in {rec_out}")
    return {"recording_id": recording_id, "out_dir": rec_out, "excel_path": excel_path}


def run_all_recordings_from_in_vitro_cells(in_vitro_cells_csv, fs=20000, lowcut=300.0, highcut=3000.0, out_dir=None):
    in_vitro_cells_csv = _to_windows_abs_path(in_vitro_cells_csv)
    df = pd.read_csv(in_vitro_cells_csv)

    required = {"file", "unit_id", "label", "alignment", "biphasic", "num_channels", "state"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"in_vitro_cells.csv missing columns: {missing}")

    df["file"] = df["file"].apply(_to_windows_abs_path)
    df["base_prefix"] = df["file"].apply(lambda p: infer_recording_paths_from_unit_stats_csv(p)["base_prefix"])
    df["recording_id"] = df["base_prefix"].apply(lambda p: os.path.basename(str(p)))

    if out_dir is None:
        out_dir = os.path.dirname(in_vitro_cells_csv)
    out_dir = _to_windows_abs_path(out_dir)
    safe_makedirs(out_dir)

    results = []
    for (base_prefix, recording_id), grp in df.groupby(["base_prefix", "recording_id"], sort=False):
        chans = grp["num_channels"].dropna().unique().tolist()
        if len(chans) != 1:
            raise ValueError(f"Recording {recording_id}: expected exactly one num_channels, got: {chans}")
        num_channels = int(chans[0])

        print(f"=== Analyzing {recording_id} (num_channels={num_channels}) ===")
        results.append(
            analyze_recording(
                base_prefix=base_prefix,
                recording_id=recording_id,
                num_channels=num_channels,
                in_vitro_cells_df_for_recording=grp.copy(),
                fs=fs,
                lowcut=lowcut,
                highcut=highcut,
                out_dir=out_dir,
            )
        )
    return results


if __name__ == "__main__":
    IN_VITRO_CELLS_CSV = r"E:\in_vivo_in_vitro\stat\vivo_vitro_units_s2.csv"
    run_all_recordings_from_in_vitro_cells(
        in_vitro_cells_csv=IN_VITRO_CELLS_CSV,
        fs=20000,
        lowcut=300.0,
        highcut=3000.0,
        out_dir=os.path.dirname(IN_VITRO_CELLS_CSV),
    )