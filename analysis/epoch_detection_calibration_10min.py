"""
epoch_detection.py
==================
Batch LFP epoch detection using sliding Welch PSD with
Youden's-J threshold calibration from paired sleep + awake
reference recordings, compound band-specific state-specificity
criteria, 3-window running median smoothing, Hilbert-based fine
boundary refinement on transition windows only, per-recording
gamma threshold rescaling, and ratio-based theta separation.

Threshold calibration
---------------------
  Youden's J on paired sleep vs awake metric distributions.
  Fallback to percentile method if AUC < AUC_FALLBACK_THRESHOLD.
  Ref: Youden WJ. Cancer 1950; Zweig & Campbell. Clin Chem 1993.

Compound criteria
-----------------
    delta     : delta_frac > thresh  AND  delta_frac > theta_frac_raw
    theta     : theta_delta_ratio > thresh  only  (no secondary)
  lowgamma  : lowgamma_frac > thresh  only
  highgamma : highgamma_frac > thresh  only

Mutual exclusivity
------------------
    Theta is separated by the theta/delta criterion.

SPA recordings
--------------
  lfp_directory.csv column 'spa' (TRUE/FALSE).
  When spa=TRUE, theta detection is skipped entirely.

Per-recording gamma rescaling
------------------------------
  For each recording, gamma thresholds are compared against the
  recording's own gamma fractional power distribution.
  If recording median / awake reference median < GAMMA_RESCALE_MIN
    → flagged 'undetectable', gamma CSVs written empty.
  Elif ratio < GAMMA_RESCALE_MAX
    → threshold rescaled multiplicatively (Option A).
  Else → global Youden threshold used unchanged.
  Rescaling applies to lowgamma and highgamma independently.
  detection_mode column written to every epoch CSV.

References
----------
  Delta   : Franken et al. J Neurosci 2001;
            Rechtschaffen & Kales 1968
  Theta   : Vanderwolf Electroencephalogr Clin Neurophysiol 1969;
            Buzsáki Neuron 2002
  LGamma  : Colgin et al. Nature 2009; Lisman & Jensen Neuron 2013
  HGamma  : Ray & Maunsell PLoS Biol 2011;
            Scheffer-Teixeira & Tort Sci Rep 2016
  Smoothing: Helfrich & Knight Trends Cogn Sci 2016

Pipeline per recording
-----------------------
  1.  Read spa flag from lfp_directory.csv
  2.  Sliding Welch PSD  (4 s windows, 2 s step)
  3.  Fractional power + diagnostic metrics per window
  4.  3-window running median smoothing
  5.  Per-recording gamma threshold rescaling
  6.  Compound threshold per band
  7.  Consensus >= MIN_CHANNELS_REQUIRED channels per window
  8.  IF coarse epochs exist → fine Hilbert boundary refinement
  9.  Merge gaps < MERGE_GAP_SEC, drop < MIN_EPOCH_SEC
    10. Theta retained by theta/delta criterion  (skipped if spa=TRUE)
  11. Save CSVs  (start_sample, end_sample, start_s, end_s,
                   duration_samples, duration_s, detection_mode)
  12. Save overview PNG per recording
    13. Save combined reference overview PNG on first calibration

Usage in JupyterLab
-------------------
    %run epoch_detection.py
or
    from epoch_detection import (run_epoch_detection_batch,
                                  check_epoch_coverage,
                                  load_or_compute_reference_stats,
                                  inspect_gamma_distributions)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.signal
from scipy.signal import butter, filtfilt

# ============================================================
#  CONSTANTS
# ============================================================

CHANNELS_23 = [3, 5, 7, 15, 17, 19, 21, 23]     # 1-indexed
CHANNELS_26 = [6, 8, 10, 18, 20, 22, 24, 26]     # 1-indexed

BANDS = {
    'delta':     (0.5,   3.5),
    'theta':     (4.0,   8.0),
    'lowgamma':  (30.0,  50.0),
    'highgamma': (50.0,  80.0),
}

TOTAL_POWER_BAND = (0.5, 200.0)

# Welch
WELCH_WIN_SEC  = 4.0
WELCH_OVERLAP  = 0.5       # → 2 s step

# Smoothing (set 1 to disable)
SMOOTH_WINDOWS = 3

# Fallback percentile calibration when AUC < AUC_FALLBACK_THRESHOLD
COVERAGE_TARGET        = 0.75
AUC_FALLBACK_THRESHOLD = 0.60

# Consensus
MIN_CHANNELS_REQUIRED = 6

# Post-processing
MIN_EPOCH_SEC  = 30.0
MERGE_GAP_SEC  = 15.0

# Positive class per band
BAND_POSITIVE_CLASS = {
    'delta':     'sleep',
    'theta':     'awake',
    'lowgamma':  'awake',
    'highgamma': 'awake',
}

# Primary detection metric per band  (fractional power)
BAND_METRIC = {
    'delta':     'delta_frac',
    'theta':     'theta_delta_ratio',
    'lowgamma':  'lowgamma_frac',
    'highgamma': 'highgamma_frac',
}

BAND_REFERENCE_SOURCE = {
    'delta':     'sleep',
    'theta':     'awake',
    'lowgamma':  'awake',
    'highgamma': 'awake',
}

# Delta secondary criterion only
DELTA_THETA_DOMINANCE = True

# Per-recording gamma rescaling
GAMMA_RESCALE_ENABLED = True    # set False to disable entirely
GAMMA_RESCALE_MIN     = 0.10    # below this ratio → undetectable
GAMMA_RESCALE_MAX     = 0.80    # below this ratio → rescale threshold
# Number of seconds sampled from start of recording to estimate
# the recording's gamma distribution (keeps it fast)
GAMMA_SAMPLE_SEC      = 600.0

BAND_COLORS = {
    'delta':     '#2196F3',
    'theta':     '#4CAF50',
    'lowgamma':  '#FF9800',
    'highgamma': '#E91E63',
}

# ============================================================
#  CHANNEL HELPERS
# ============================================================

def channels_for_recording(n_channels):
    if n_channels == 26:
        return [c - 1 for c in CHANNELS_26]
    return [c - 1 for c in CHANNELS_23]


def epochs_csv_paths(bin_path):
    base = bin_path[:-4]
    return {b: f"{base}_{b}_epochs.csv" for b in BANDS}


def all_epoch_csvs_exist(bin_path):
    return all(os.path.exists(p) for p in epochs_csv_paths(bin_path).values())


# ============================================================
#  I/O
# ============================================================

def load_bin(bin_path, n_channels):
    with open(bin_path, 'rb') as f:
        raw = np.fromfile(f, dtype=np.int16)
    return raw.reshape((-1, n_channels)).T.astype(np.float32)


def load_bin_partial(bin_path, n_channels, n_samples):
    """Load only the first n_samples from a .bin file."""
    with open(bin_path, 'rb') as f:
        raw = np.fromfile(f, dtype=np.int16,
                          count=n_samples * n_channels)
    actual = raw.size // n_channels
    return raw[:actual * n_channels].reshape(
        (actual, n_channels)).T.astype(np.float32)


def read_spa_flag(file_row):
    """
    Read the 'spa' column from a CSV row.
    Returns True if spa is TRUE/True/1/yes (case-insensitive).
    Returns False if column is absent or any other value.
    """
    val = file_row.get('spa', False)
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    return str(val).strip().lower() in ('true', '1', 'yes')


# ============================================================
#  EMPTY EPOCH CSV  (with detection_mode column)
# ============================================================

_EPOCH_COLUMNS = [
    'start_sample', 'end_sample',
    'start_s', 'end_s',
    'duration_samples', 'duration_s',
    'detection_mode',
]


def _empty_epoch_df():
    return pd.DataFrame(columns=_EPOCH_COLUMNS)


def _records_to_df(records, detection_mode='global'):
    """Convert list of epoch record dicts to DataFrame with detection_mode."""
    if not records:
        return _empty_epoch_df()
    df = pd.DataFrame(records)
    df['detection_mode'] = detection_mode
    return df[_EPOCH_COLUMNS]


# ============================================================
#  WELCH PSD
# ============================================================

def compute_sliding_welch(signal, fs,
                           win_sec=WELCH_WIN_SEC,
                           overlap=WELCH_OVERLAP):
    n_win     = int(win_sec * fs)
    n_step    = max(1, int(n_win * (1.0 - overlap)))
    nperseg   = min(n_win, len(signal))
    starts    = np.arange(0, len(signal) - n_win + 1,
                          n_step, dtype=np.int64)
    if len(starts) == 0:
        freqs, _ = scipy.signal.welch(signal, fs=fs, nperseg=nperseg)
        return (freqs,
                np.zeros((0, len(freqs)), dtype=np.float32),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.int64))
    freqs, _ = scipy.signal.welch(signal[:nperseg], fs=fs, nperseg=nperseg)
    psd_mat  = np.zeros((len(starts), len(freqs)), dtype=np.float32)
    for i, s in enumerate(starts):
        _, pxx     = scipy.signal.welch(signal[s: s + n_win],
                                         fs=fs, nperseg=nperseg)
        psd_mat[i] = pxx.astype(np.float32)
    t_centers = (starts + n_win / 2.0) / fs
    return freqs, psd_mat, t_centers, starts


def _band_power(freqs, psd_row, band):
    mask = (freqs >= band[0]) & (freqs <= band[1])
    return float(np.trapz(psd_row[mask], freqs[mask])) if mask.any() else 0.0


def compute_band_powers(freqs, psd_matrix):
    if psd_matrix.shape[0] == 0:
        empty = np.zeros(0, dtype=np.float32)
        out   = {'total': empty}
        for b in BANDS:
            out[b] = empty.copy()
        return out
    total = np.array([_band_power(freqs, r, TOTAL_POWER_BAND)
                      for r in psd_matrix], dtype=np.float32)
    out   = {'total': total}
    for bname, band in BANDS.items():
        out[bname] = np.array([_band_power(freqs, r, band)
                                for r in psd_matrix], dtype=np.float32)
    return out


def compute_metrics(bp):
    eps = np.float32(1e-12)
    t   = bp['total']    + eps
    d   = bp['delta']    + eps
    th  = bp['theta']    + eps
    lg  = bp['lowgamma'] + eps
    return {
        'delta_frac':           bp['delta']     / t,
        'theta_frac':           bp['theta']     / t,
        'lowgamma_frac':        bp['lowgamma']  / t,
        'highgamma_frac':       bp['highgamma'] / t,
        'theta_delta_ratio':    bp['theta']     / d,
        'lowgamma_theta':       bp['lowgamma']  / th,
        'highgamma_theta':      bp['highgamma'] / th,
        'highgamma_lowgamma':   bp['highgamma'] / lg,
        'theta_frac_raw':       bp['theta']     / t,
    }


# ============================================================
#  ROC / YOUDEN CALIBRATION
# ============================================================

def _compute_roc(pos_scores, neg_scores):
    all_scores = np.concatenate([pos_scores, neg_scores])
    thresholds = np.sort(np.unique(all_scores))[::-1]
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    tpr   = np.zeros(len(thresholds))
    fpr   = np.zeros(len(thresholds))
    for i, t in enumerate(thresholds):
        tpr[i] = np.sum(pos_scores >= t) / (n_pos + 1e-12)
        fpr[i] = np.sum(neg_scores >= t) / (n_neg + 1e-12)
    order = np.argsort(fpr)
    auc   = float(np.trapz(tpr[order], fpr[order]))
    return thresholds, tpr, fpr, auc


def _youden_threshold(pos_scores, neg_scores):
    thresholds, tpr, fpr, auc = _compute_roc(pos_scores, neg_scores)
    j        = tpr + (1.0 - fpr) - 1.0
    best_idx = int(np.argmax(j))
    return (float(thresholds[best_idx]), auc,
            float(tpr[best_idx]), float(1.0 - fpr[best_idx]),
            thresholds, tpr, fpr, j)


def calibrate_thresholds(sleep_met, awake_met,
                          coverage_target=COVERAGE_TARGET,
                          auc_fallback=AUC_FALLBACK_THRESHOLD):
    secondary_descriptions = {
        'delta':     (f"delta_frac > theta_frac_raw  "
                      f"({'ON' if DELTA_THETA_DOMINANCE else 'OFF'})"),
        'theta':     "theta/delta ratio",
        'lowgamma':  "none (primary only)",
        'highgamma': "none (primary only)",
    }
    calib   = {}
    pct_cut = (1.0 - coverage_target) * 100.0
    print("\n  YOUDEN'S J THRESHOLD CALIBRATION")
    print("  " + "-" * 62)

    for band in BANDS:
        metric  = BAND_METRIC[band]
        pos_cls = BAND_POSITIVE_CLASS[band]
        neg_cls = 'awake' if pos_cls == 'sleep' else 'sleep'
        pos_arr = sleep_met[metric] if pos_cls == 'sleep' else awake_met[metric]
        neg_arr = awake_met[metric] if pos_cls == 'sleep' else sleep_met[metric]
        lo, hi  = BANDS[band]

        if len(pos_arr) == 0 or len(neg_arr) == 0:
            print(f"\n  [{band:>10s}]  WARNING: empty — threshold = 0")
            calib[band] = _empty_calib(band, secondary_descriptions)
            continue

        (thresh, auc, sens, spec,
         roc_t, roc_tpr, roc_fpr,
         youden_j) = _youden_threshold(pos_arr, neg_arr)

        fallback = False
        fallback_reason = ''
        if auc < auc_fallback:
            fallback        = True
            if band == 'theta':
                thresh = 0.2
                fallback_reason = (f"AUC={auc:.3f} < {auc_fallback} "
                                   f"→ fixed theta/delta=0.2")
            else:
                fallback_reason = (f"AUC={auc:.3f} < {auc_fallback} "
                                   f"→ {pct_cut:.0f}th pct of {pos_cls}")
                thresh = float(np.percentile(pos_arr, pct_cut))
            sens   = float(np.mean(pos_arr >= thresh))
            spec   = float(np.mean(neg_arr <  thresh))

        comp_cov = _estimate_compound_coverage_arrays(
            pos_arr, pos_cls, band, thresh, sleep_met, awake_met)

        calib[band] = {
            'metric':              metric,
            'source':              BAND_REFERENCE_SOURCE[band],
            'positive_class':      pos_cls,
            'threshold':           thresh,
            # store awake reference median for gamma rescaling
            'awake_median':        float(np.median(awake_met[metric]))
                                   if len(awake_met.get(metric, [])) else np.nan,
            'auc':                 auc,
            'sensitivity':         sens,
            'specificity':         spec,
            'youden_j_max':        float(np.max(youden_j)),
            'fallback_used':       fallback,
            'fallback_reason':     fallback_reason,
            'sleep_mean':          float(np.mean(sleep_met[metric])),
            'sleep_std':           float(np.std( sleep_met[metric])),
            'awake_mean':          float(np.mean(awake_met[metric])),
            'awake_std':           float(np.std( awake_met[metric])),
            'sleep_p25':           float(np.percentile(sleep_met[metric], 25)),
            'sleep_p75':           float(np.percentile(sleep_met[metric], 75)),
            'awake_p25':           float(np.percentile(awake_met[metric], 25)),
            'awake_p75':           float(np.percentile(awake_met[metric], 75)),
            'compound_coverage':   comp_cov,
            'secondary_criterion': secondary_descriptions[band],
            '_roc_thresholds': roc_t,
            '_roc_tpr':        roc_tpr,
            '_roc_fpr':        roc_fpr,
            '_roc_youden_j':   youden_j,
        }

        print(f"\n  [{band:>10s}]  ({lo}–{hi} Hz)  "
              f"metric={metric}  pos={pos_cls}")
        print(f"    sleep  : mean={calib[band]['sleep_mean']:.4f}  "
              f"std={calib[band]['sleep_std']:.4f}  "
              f"[{calib[band]['sleep_p25']:.4f}–"
              f"{calib[band]['sleep_p75']:.4f}]")
        print(f"    awake  : mean={calib[band]['awake_mean']:.4f}  "
              f"std={calib[band]['awake_std']:.4f}  "
              f"[{calib[band]['awake_p25']:.4f}–"
              f"{calib[band]['awake_p75']:.4f}]")
        print(f"    AUC    : {auc:.4f}"
              + ("  ✓" if not fallback else f"  ✗  {fallback_reason}"))
        print(f"    thresh : {thresh:.4f}  J={calib[band]['youden_j_max']:.3f}"
              + (" [FALLBACK]" if fallback else ""))
        print(f"    sens   : {sens*100:.1f}%  spec : {spec*100:.1f}%")
        print(f"    2nd    : {secondary_descriptions[band]}")
        print(f"    comp_cov: {comp_cov*100:.1f}%  in {pos_cls} ref")

    print("\n  " + "-" * 62)
    return calib


def _empty_calib(band, secondary_descriptions):
    return {
        'metric':              BAND_METRIC[band],
        'source':              BAND_REFERENCE_SOURCE[band],
        'positive_class':      BAND_POSITIVE_CLASS[band],
        'threshold':           0.0,
        'awake_median':        np.nan,
        'auc':                 np.nan,
        'sensitivity':         np.nan,
        'specificity':         np.nan,
        'youden_j_max':        np.nan,
        'fallback_used':       True,
        'fallback_reason':     'empty arrays',
        'sleep_mean': np.nan, 'sleep_std': np.nan,
        'awake_mean': np.nan, 'awake_std': np.nan,
        'sleep_p25':  np.nan, 'sleep_p75': np.nan,
        'awake_p25':  np.nan, 'awake_p75': np.nan,
        'compound_coverage':   np.nan,
        'secondary_criterion': secondary_descriptions.get(band, ''),
        '_roc_thresholds': np.array([]),
        '_roc_tpr':        np.array([]),
        '_roc_fpr':        np.array([]),
        '_roc_youden_j':   np.array([]),
    }


def _estimate_compound_coverage_arrays(pos_arr, pos_cls, band, threshold,
                                        sleep_met, awake_met):
    met_src      = sleep_met if pos_cls == 'sleep' else awake_met
    primary_pass = pos_arr > threshold
    n            = len(primary_pass)
    if band == 'delta' and DELTA_THETA_DOMINANCE:
        df  = met_src.get('delta_frac',    np.zeros(n))
        tf  = met_src.get('theta_frac_raw', np.zeros(n))
        sec = df > tf
        k   = min(n, len(sec))
        return float(np.mean(primary_pass[:k] & sec[:k]))
    return float(np.mean(primary_pass))


# ============================================================
#  PER-RECORDING GAMMA RESCALING
# ============================================================

def _recording_gamma_medians(bin_path, n_channels, fs,
                              sample_sec=GAMMA_SAMPLE_SEC):
    """
    Compute median gamma fractional power for this recording using
    the first sample_sec seconds only (fast diagnostic sample).

    Returns dict  band -> median_value  for gamma bands only.
    """
    n_samp  = int(sample_sec * fs * n_channels)
    raw     = load_bin_partial(bin_path, n_channels, int(sample_sec * fs))
    ch_0idx = channels_for_recording(n_channels)
    medians = {b: [] for b in ('lowgamma', 'highgamma')}

    for ch in ch_0idx:
        freqs, psd_mat, _, _ = compute_sliding_welch(raw[ch], fs)
        if psd_mat.shape[0] == 0:
            continue
        bp  = compute_band_powers(freqs, psd_mat)
        met = compute_metrics(bp)
        for b in ('lowgamma', 'highgamma'):
            arr = met[BAND_METRIC[b]]
            arr = arr[np.isfinite(arr)]
            if len(arr):
                medians[b].append(float(np.median(arr)))

    del raw
    return {b: float(np.median(medians[b])) if medians[b] else 0.0
            for b in ('lowgamma', 'highgamma')}


def _rescale_gamma_threshold(band, global_thresh, rec_median,
                              awake_median):
    """
    Decide detection mode and effective threshold for a gamma band
    in a specific recording.

    Parameters
    ----------
    band          : 'lowgamma' or 'highgamma'
    global_thresh : float  Youden-calibrated threshold
    rec_median    : float  recording's median gamma_frac
    awake_median  : float  awake reference median gamma_frac

    Returns
    -------
    mode      : str   'global' | 'rescaled' | 'undetectable'
    threshold : float  effective threshold to use
    ratio     : float  rec_median / awake_median
    """
    if awake_median <= 0 or not GAMMA_RESCALE_ENABLED:
        return 'global', global_thresh, np.nan

    ratio = rec_median / (awake_median + 1e-12)

    if ratio < GAMMA_RESCALE_MIN:
        return 'undetectable', global_thresh, ratio

    if ratio < GAMMA_RESCALE_MAX:
        rescaled = global_thresh * ratio
        return 'rescaled', rescaled, ratio

    return 'global', global_thresh, ratio


def _get_effective_calib(calib, rec_gamma_medians):
    """
    Return a shallow copy of calib with gamma thresholds adjusted
    for this recording.  Also returns detection_modes dict.

    Parameters
    ----------
    calib             : global calib dict
    rec_gamma_medians : dict  band -> median  (from _recording_gamma_medians)

    Returns
    -------
    eff_calib        : dict  (copy with adjusted thresholds)
    detection_modes  : dict  band -> mode string
    gamma_ratios     : dict  band -> ratio float
    """
    import copy
    eff_calib       = copy.deepcopy(calib)
    detection_modes = {b: 'global' for b in BANDS}
    gamma_ratios    = {}

    for band in ('lowgamma', 'highgamma'):
        rec_med   = rec_gamma_medians.get(band, 0.0)
        awk_med   = calib[band].get('awake_median', np.nan)
        g_thresh  = calib[band]['threshold']

        mode, eff_thresh, ratio = _rescale_gamma_threshold(
            band, g_thresh, rec_med, awk_med)

        eff_calib[band]['threshold'] = eff_thresh
        detection_modes[band]        = mode
        gamma_ratios[band]           = ratio

    return eff_calib, detection_modes, gamma_ratios


# ============================================================
#  INSPECT GAMMA DISTRIBUTIONS  (diagnostic, no files written)
# ============================================================

def inspect_gamma_distributions(csv_path, calib, fs=20000,
                                  save_png=False, png_dir=None):
    """
    For every recording in csv_path, compute the recording's median
    gamma fractional power and compare it against the awake reference.

    Prints a table:
        file | dur | lg_median | lg_awake_med | lg_ratio | lg_mode |
             | hg_median | hg_awake_med | hg_ratio | hg_mode

    Parameters
    ----------
    csv_path  : str
    calib     : dict  (from load_or_compute_reference_stats)
    fs        : float
    save_png  : bool   if True, save a distribution comparison PNG
    png_dir   : str    directory for PNG output (defaults to csv dir)

    Returns
    -------
    df_report : pd.DataFrame  one row per recording
    """
    df_dir = pd.read_csv(csv_path)
    lg_awk = calib['lowgamma'].get('awake_median', np.nan)
    hg_awk = calib['highgamma'].get('awake_median', np.nan)

    print("\n" + "=" * 88)
    print("GAMMA DISTRIBUTION INSPECTION")
    print(f"  GAMMA_RESCALE_ENABLED : {GAMMA_RESCALE_ENABLED}")
    print(f"  GAMMA_RESCALE_MIN     : {GAMMA_RESCALE_MIN}  "
          f"(below → undetectable)")
    print(f"  GAMMA_RESCALE_MAX     : {GAMMA_RESCALE_MAX}  "
          f"(below → rescaled)")
    print(f"  Awake ref median  lowgamma : {lg_awk:.5f}")
    print(f"  Awake ref median highgamma : {hg_awk:.5f}")
    print(f"  Sample duration per recording : {GAMMA_SAMPLE_SEC} s")
    print("=" * 88)
    hdr = (f"  {'File':<32s}"
           f"{'lg_med':>9s}{'lg_ratio':>10s}{'lg_mode':>14s}"
           f"{'hg_med':>9s}{'hg_ratio':>10s}{'hg_mode':>14s}")
    print(hdr)
    print("  " + "-" * 86)

    rows = []
    for _, file_row in df_dir.iterrows():
        bin_path   = str(file_row['filename']).strip()
        n_channels = int(file_row['n_channels'])
        spa        = read_spa_flag(file_row)
        name       = os.path.basename(bin_path)

        if not os.path.exists(bin_path):
            print(f"  [SKIP]  {name}")
            continue
        try:
            gm = _recording_gamma_medians(bin_path, n_channels, fs)

            lg_mode, _, lg_ratio = _rescale_gamma_threshold(
                'lowgamma',  calib['lowgamma']['threshold'],
                gm['lowgamma'],  lg_awk)
            hg_mode, _, hg_ratio = _rescale_gamma_threshold(
                'highgamma', calib['highgamma']['threshold'],
                gm['highgamma'], hg_awk)

            # colour-code mode in terminal
            _MODE_MARK = {
                'global':        '✓ global',
                'rescaled':      '~ rescaled',
                'undetectable':  '✗ undetect',
            }
            print(f"  {name:<32s}"
                  f"{gm['lowgamma']:>9.5f}"
                  f"{lg_ratio:>10.3f}"
                  f"{_MODE_MARK[lg_mode]:>14s}"
                  f"{gm['highgamma']:>9.5f}"
                  f"{hg_ratio:>10.3f}"
                  f"{_MODE_MARK[hg_mode]:>14s}"
                  + (" [SPA]" if spa else ""))

            rows.append({
                'filename':       bin_path,
                'spa':            spa,
                'lg_median':      gm['lowgamma'],
                'lg_awake_med':   lg_awk,
                'lg_ratio':       lg_ratio,
                'lg_mode':        lg_mode,
                'hg_median':      gm['highgamma'],
                'hg_awake_med':   hg_awk,
                'hg_ratio':       hg_ratio,
                'hg_mode':        hg_mode,
            })
        except Exception as e:
            print(f"  [ERROR]  {name}: {e}")

    print("  " + "-" * 86)
    print("=" * 88)

    df_report = pd.DataFrame(rows)

    if save_png and len(df_report) > 0:
        _save_gamma_inspection_png(df_report, calib, png_dir, csv_path)

    return df_report


def _save_gamma_inspection_png(df_report, calib, png_dir, csv_path):
    """
    Two-panel PNG:
    Left  : lowgamma  median per recording vs awake reference median + zones
    Right : highgamma median per recording vs awake reference median + zones
    """
    if png_dir is None:
        png_dir = os.path.dirname(os.path.abspath(csv_path))

    # Explicit prefix map — avoids fragile band[:2] slicing
    band_prefix = {
        'lowgamma':  'lg',
        'highgamma': 'hg',
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Gamma Distribution Inspection\n"
                 "Recording medians vs awake reference",
                 fontsize=12, fontweight='bold')

    for ax, band, col in zip(
            axes,
            ['lowgamma', 'highgamma'],
            [BAND_COLORS['lowgamma'], BAND_COLORS['highgamma']]):

        pfx       = band_prefix[band]
        med_col   = f"{pfx}_median"
        mode_col  = f"{pfx}_mode"
        awk_med   = calib[band].get('awake_median', np.nan)
        thresh    = calib[band]['threshold']

        names   = [os.path.basename(r) for r in df_report['filename']]
        medians = df_report[med_col].values
        modes   = df_report[mode_col].values
        xs      = np.arange(len(names))

        mode_colors = {
            'global':       col,
            'rescaled':     'orange',
            'undetectable': 'red',
        }
        bar_cols = [mode_colors.get(m, col) for m in modes]

        ax.bar(xs, medians, color=bar_cols, alpha=0.75, edgecolor='none')
        ax.axhline(awk_med, color='steelblue', lw=1.5, linestyle='--',
                   label=f'Awake ref median = {awk_med:.5f}')
        ax.axhline(awk_med * GAMMA_RESCALE_MIN, color='red', lw=1.2,
                   linestyle=':',
                   label=f'Undetectable < ×{GAMMA_RESCALE_MIN}')
        ax.axhline(awk_med * GAMMA_RESCALE_MAX, color='orange', lw=1.2,
                   linestyle=':',
                   label=f'Rescale < ×{GAMMA_RESCALE_MAX}')
        ax.axhline(thresh, color='black', lw=1.2, linestyle='--',
                   alpha=0.7, label=f'Youden thresh = {thresh:.5f}')

        ax.set_xticks(xs)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel(f'{band}_frac  median', fontsize=9)
        lo, hi = BANDS[band]
        ax.set_title(f"{band.capitalize()}  ({lo}–{hi} Hz)", fontsize=10)

        from matplotlib.patches import Patch
        bar_legend = [
            Patch(facecolor=col,      label='global'),
            Patch(facecolor='orange', label='rescaled'),
            Patch(facecolor='red',    label='undetectable'),
        ]
        ax.legend(handles=bar_legend, fontsize=7, loc='upper right')

    plt.tight_layout()
    out_png = os.path.join(png_dir, "gamma_inspection.png")
    plt.savefig(out_png, dpi=100)
    plt.close(fig)
    print(f"  Gamma inspection PNG  →  {out_png}")


# ============================================================
#  LEGACY THETA/DELTA EXCLUSION HELPER
# ============================================================

def _apply_delta_exclusion(theta_records, delta_records, fs,
                            min_epoch_sec=MIN_EPOCH_SEC):
    """
    Legacy helper retained for reference; current pipeline no longer
    trims theta epochs against delta after detection.
    """
    if not theta_records or not delta_records:
        return theta_records
    min_samp = int(min_epoch_sec * fs)
    delta_intervals = sorted(
        [(r['start_sample'], r['end_sample']) for r in delta_records],
        key=lambda x: x[0])
    trimmed = []
    for rec in theta_records:
        for frag_s, frag_e in _subtract_intervals(
                rec['start_sample'], rec['end_sample'], delta_intervals):
            dur = frag_e - frag_s
            if dur >= min_samp:
                trimmed.append({
                    'start_sample':    int(frag_s),
                    'end_sample':      int(frag_e),
                    'start_s':         frag_s / fs,
                    'end_s':           frag_e / fs,
                    'duration_samples': int(dur),
                    'duration_s':      dur / fs,
                })
    return trimmed


def _subtract_intervals(start, end, sorted_exclusions):
    fragments = []
    cursor    = start
    for exc_s, exc_e in sorted_exclusions:
        if exc_e <= cursor: continue
        if exc_s >= end:    break
        if exc_s > cursor:
            fragments.append((cursor, exc_s))
        cursor = max(cursor, exc_e)
    if cursor < end:
        fragments.append((cursor, end))
    return fragments


# ============================================================
#  COMPOUND BOOLEAN MASK
# ============================================================

def _compound_mask(met, band, threshold):
    primary = met[BAND_METRIC[band]] > threshold
    if band == 'delta' and DELTA_THETA_DOMINANCE:
        return primary & (met['delta_frac'] > met['theta_frac_raw'])
    return primary


# ============================================================
#  RUNNING MEDIAN SMOOTHING
# ============================================================

def running_median(arr, k=SMOOTH_WINDOWS):
    if k <= 1 or len(arr) == 0:
        return arr.copy()
    if k % 2 == 0:
        k += 1
    half   = k // 2
    padded = np.pad(arr, half, mode='reflect')
    out    = np.empty_like(arr)
    for i in range(len(arr)):
        out[i] = np.median(padded[i: i + k])
    return out


# ============================================================
#  FINE BOUNDARY REFINEMENT
# ============================================================

def _bandpass_butter(signal, fs, band, order=3):
    nyq  = fs / 2.0
    low  = max(band[0] / nyq, 1e-4)
    high = min(band[1] / nyq, 1.0 - 1e-4)
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal.astype(np.float64)).astype(np.float32)


def _hilbert_envelope(signal):
    return np.abs(
        scipy.signal.hilbert(signal.astype(np.float64))
    ).astype(np.float32)


def _refine_one_boundary(raw_ch, fs, band,
                          bnd_s, bnd_e, ref_s, ref_e, direction):
    n = len(raw_ch)
    bnd_s = max(0, bnd_s);  bnd_e = min(n, bnd_e)
    ref_s = max(0, ref_s);  ref_e = min(n, ref_e)
    if bnd_e <= bnd_s or ref_e <= ref_s:
        return bnd_s if direction == 'start' else bnd_e
    env_bnd   = _hilbert_envelope(
        _bandpass_butter(raw_ch[bnd_s:bnd_e], fs, band))
    env_ref   = _hilbert_envelope(
        _bandpass_butter(raw_ch[ref_s:ref_e], fs, band))
    threshold = float(np.median(env_ref))
    if direction == 'start':
        for i, v in enumerate(env_bnd):
            if v >= threshold: return bnd_s + i
        return bnd_e
    else:
        for i in range(len(env_bnd) - 1, -1, -1):
            if env_bnd[i] >= threshold: return bnd_s + i
        return bnd_s


def refine_epoch_boundaries(raw, fs, ch_0idx, band,
                             coarse_start, coarse_end,
                             win_samples, n_total):
    s_cands, e_cands = [], []
    for ch in ch_0idx:
        sig = raw[ch]
        rs  = _refine_one_boundary(
            sig, fs, band,
            max(0, coarse_start - win_samples), coarse_start,
            coarse_start, min(n_total, coarse_start + win_samples),
            'start')
        re  = _refine_one_boundary(
            sig, fs, band,
            coarse_end, min(n_total, coarse_end + win_samples),
            max(0, coarse_end - win_samples), coarse_end,
            'end')
        s_cands.append(rs)
        e_cands.append(re)
    rs_final = max(0, min(int(np.median(s_cands)), n_total - 1))
    re_final = max(rs_final + 1, min(int(np.median(e_cands)), n_total))
    return rs_final, re_final


# ============================================================
#  EPOCH POST-PROCESSING
# ============================================================

def postprocess_epoch_records(records, fs,
                               min_epoch_sec=MIN_EPOCH_SEC,
                               merge_gap_sec=MERGE_GAP_SEC):
    if not records:
        return []
    merge_samp = int(merge_gap_sec * fs)
    min_samp   = int(min_epoch_sec * fs)
    records    = sorted(records, key=lambda r: r['start_sample'])
    merged = []
    cs, ce = records[0]['start_sample'], records[0]['end_sample']
    for r in records[1:]:
        if r['start_sample'] - ce < merge_samp:
            ce = max(ce, r['end_sample'])
        else:
            merged.append({'start_sample': cs, 'end_sample': ce})
            cs, ce = r['start_sample'], r['end_sample']
    merged.append({'start_sample': cs, 'end_sample': ce})
    kept = []
    for r in merged:
        dur = r['end_sample'] - r['start_sample']
        if dur >= min_samp:
            kept.append({
                'start_sample':    int(r['start_sample']),
                'end_sample':      int(r['end_sample']),
                'start_s':         r['start_sample'] / fs,
                'end_s':           r['end_sample']   / fs,
                'duration_samples': int(dur),
                'duration_s':      dur / fs,
            })
    return kept


# ============================================================
#  REFERENCE STATS
# ============================================================

def _metric_distributions_for_file(bin_path, n_channels, fs, label):
    print(f"\n  [{label}]  {os.path.basename(bin_path)}  "
          f"({n_channels} ch)", flush=True)
    raw     = load_bin_partial(bin_path, n_channels, int(600 * fs))
    ch_0idx = channels_for_recording(n_channels)
    dur_s   = raw.shape[1] / fs
    print(f"    shape {raw.shape}  ({dur_s:.1f} s)  "
          f"channels (0-idx) {ch_0idx}")
    all_metrics = {}
    for ch in ch_0idx:
        freqs, psd_mat, _, _ = compute_sliding_welch(raw[ch], fs)
        if psd_mat.shape[0] == 0: continue
        bp  = compute_band_powers(freqs, psd_mat)
        met = compute_metrics(bp)
        for m, arr in met.items():
            all_metrics.setdefault(m, []).append(arr)
    del raw
    combined = {}
    for m, arrays in all_metrics.items():
        if arrays:
            c = np.concatenate(arrays)
            combined[m] = c[np.isfinite(c)]
        else:
            combined[m] = np.array([])
    return combined


_CALIB_CSV_COLS = [
    'band', 'metric', 'source', 'positive_class',
    'threshold', 'awake_median',
    'auc', 'sensitivity', 'specificity',
    'youden_j_max', 'fallback_used', 'fallback_reason',
    'sleep_mean', 'sleep_std', 'sleep_p25', 'sleep_p75',
    'awake_mean', 'awake_std', 'awake_p25', 'awake_p75',
    'compound_coverage', 'secondary_criterion',
]


def compute_reference_stats(sleep_bin_path, sleep_n_channels,
                             awake_bin_path,  awake_n_channels,
                             fs=20000,
                             coverage_target=COVERAGE_TARGET,
                             auc_fallback=AUC_FALLBACK_THRESHOLD,
                             save_path=None,
                             save_png_dir=None):
    print("\n" + "=" * 64)
    print("REFERENCE STATISTICS — YOUDEN'S J CALIBRATION")
    print(f"  Primary metric  : delta-referenced theta ratio for theta")
    print(f"  AUC fallback at : {auc_fallback}")
    print(f"  Smoothing       : {SMOOTH_WINDOWS}-window running median")
    print(f"  Delta secondary : delta_frac > theta_frac_raw  "
          f"({'ON' if DELTA_THETA_DOMINANCE else 'OFF'})")
    print(f"  Gamma rescaling : enabled={GAMMA_RESCALE_ENABLED}  "
          f"min={GAMMA_RESCALE_MIN}  max={GAMMA_RESCALE_MAX}")
    print(f"  Theta separation: theta/delta ratio only  (post-detection)")
    print(f"  SPA recordings  : theta skipped when spa=TRUE")
    print("=" * 64)

    sleep_met = _metric_distributions_for_file(
        sleep_bin_path, sleep_n_channels, fs, 'SLEEP')
    awake_met = _metric_distributions_for_file(
        awake_bin_path, awake_n_channels, fs, 'AWAKE')

    print(f"\n  DIAGNOSTIC CROSS-BAND RATIOS")
    print(f"  " + "-" * 44)
    for src_label, src_met in [('SLEEP', sleep_met), ('AWAKE', awake_met)]:
        print(f"  [{src_label}]")
        for diag, lbl in [
            ('theta_delta_ratio',  'Theta/Delta'),
            ('lowgamma_theta',     'LowGamma/Theta'),
            ('highgamma_theta',    'HighGamma/Theta'),
            ('highgamma_lowgamma', 'HighGamma/LowGamma'),
        ]:
            arr = src_met.get(diag, np.array([]))
            if len(arr):
                print(f"    {lbl:>22s} :  "
                      f"mean={np.mean(arr):.3f}  "
                      f"median={np.median(arr):.3f}  "
                      f"p25={np.percentile(arr,25):.3f}  "
                      f"p75={np.percentile(arr,75):.3f}")

    calib = calibrate_thresholds(
        sleep_met, awake_met,
        coverage_target=coverage_target,
        auc_fallback=auc_fallback)

    if save_path:
        rows = []
        for band, d in calib.items():
            row = {'band': band}
            for k in _CALIB_CSV_COLS:
                if k != 'band': row[k] = d.get(k, np.nan)
            rows.append(row)
        pd.DataFrame(rows).to_csv(save_path, index=False)
        print(f"  Saved stats CSV  →  {save_path}")

    if save_png_dir is None:
        save_png_dir = os.path.dirname(os.path.abspath(sleep_bin_path))

    _save_reference_overview_png(
        sleep_bin_path=sleep_bin_path, sleep_n_channels=sleep_n_channels,
        awake_bin_path=awake_bin_path, awake_n_channels=awake_n_channels,
        fs=fs, sleep_met=sleep_met, awake_met=awake_met,
        calib=calib, save_dir=save_png_dir)

    return calib


def load_or_compute_reference_stats(sleep_bin_path,  sleep_n_channels,
                                     awake_bin_path,  awake_n_channels,
                                     fs=20000,
                                     coverage_target=COVERAGE_TARGET,
                                     auc_fallback=AUC_FALLBACK_THRESHOLD,
                                     stats_csv_path=None,
                                     save_png_dir=None,
                                     force=False):
    if stats_csv_path is None:
        stats_csv_path = sleep_bin_path[:-4] + "_reference_stats.csv"

    if not force and os.path.exists(stats_csv_path):
        print(f"\n  Loading reference stats from:\n    {stats_csv_path}")
        df    = pd.read_csv(stats_csv_path)
        calib = {}
        for _, row in df.iterrows():
            b = row['band']
            calib[b] = {k: row.get(k, np.nan) for k in _CALIB_CSV_COLS
                        if k != 'band'}
            calib[b]['threshold']         = float(calib[b]['threshold'])
            calib[b]['awake_median']      = float(calib[b].get(
                                                   'awake_median', np.nan))
            calib[b]['auc']               = float(calib[b]['auc'])
            calib[b]['sensitivity']       = float(calib[b]['sensitivity'])
            calib[b]['specificity']       = float(calib[b]['specificity'])
            calib[b]['compound_coverage'] = float(calib[b]['compound_coverage'])
            calib[b]['fallback_used']     = bool(calib[b]['fallback_used'])
            calib[b]['_roc_thresholds']   = np.array([])
            calib[b]['_roc_tpr']          = np.array([])
            calib[b]['_roc_fpr']          = np.array([])
            calib[b]['_roc_youden_j']     = np.array([])
            fb = ' (fallback)' if calib[b]['fallback_used'] else ''
            print(f"  [{b:>10s}]  threshold={calib[b]['threshold']:.4f}  "
                  f"AUC={calib[b]['auc']:.3f}  "
                  f"sens={calib[b]['sensitivity']*100:.1f}%  "
                  f"spec={calib[b]['specificity']*100:.1f}%{fb}")
        return calib

    return compute_reference_stats(
        sleep_bin_path, sleep_n_channels,
        awake_bin_path, awake_n_channels,
        fs=fs, coverage_target=coverage_target,
        auc_fallback=auc_fallback,
        save_path=stats_csv_path,
        save_png_dir=save_png_dir)


# ============================================================
#  DETECTION CORE
# ============================================================

def _detect_consensus(raw, fs, ch_0idx, calib,
                       min_channels_required,
                       skip_bands=None):
    """
    Coarse Welch + smoothing + compound mask + consensus.

    Parameters
    ----------
    skip_bands : set of band names to skip entirely  (e.g. {'theta'})
    """
    if skip_bands is None:
        skip_bands = set()

    ch_band_masks    = {b: [] for b in BANDS}
    metrics_per_ch   = {b: [] for b in BANDS}
    secondary_per_ch = {b: [] for b in BANDS}
    t_centers = None
    win_starts = None

    for ch in ch_0idx:
        freqs, psd_mat, tc, ws = compute_sliding_welch(raw[ch], fs)
        if psd_mat.shape[0] == 0: continue
        if t_centers is None:
            t_centers  = tc
            win_starts = ws
        bp      = compute_band_powers(freqs, psd_mat)
        met_raw = compute_metrics(bp)
        met     = {k: running_median(v, SMOOTH_WINDOWS)
                   for k, v in met_raw.items()}
        for band in BANDS:
            if band in skip_bands:
                n = len(tc)
                ch_band_masks[band].append(np.zeros(n, dtype=bool))
                metrics_per_ch[band].append(np.zeros(n, dtype=np.float32))
                secondary_per_ch[band].append(np.zeros(n, dtype=np.float32))
                continue
            thresh = calib[band]['threshold']
            mask   = _compound_mask(met, band, thresh)
            ch_band_masks[band].append(mask)
            metrics_per_ch[band].append(met[BAND_METRIC[band]])
            if band == 'delta' and DELTA_THETA_DOMINANCE:
                sec = met['delta_frac'] > met['theta_frac_raw']
            else:
                sec = np.ones(len(mask), dtype=bool)
            secondary_per_ch[band].append(sec.astype(np.float32))

    if t_centers is None:
        t_centers  = np.array([], dtype=np.float64)
        win_starts = np.array([], dtype=np.int64)

    n_win       = int(WELCH_WIN_SEC * fs)
    n_step      = max(1, int(n_win * (1.0 - WELCH_OVERLAP)))
    win_samples = n_step

    consensus_masks = {}
    for band in BANDS:
        if ch_band_masks[band]:
            stack     = np.array(ch_band_masks[band], dtype=bool)
            consensus = np.sum(stack, axis=0) >= min_channels_required
        else:
            consensus = np.zeros(len(t_centers), dtype=bool)
        if band in skip_bands:
            consensus[:] = False
        consensus_masks[band]  = consensus
        metrics_per_ch[band]   = (
            np.array(metrics_per_ch[band], dtype=np.float32)
            if metrics_per_ch[band]
            else np.zeros((0, len(t_centers)), dtype=np.float32))
        secondary_per_ch[band] = (
            np.array(secondary_per_ch[band], dtype=np.float32)
            if secondary_per_ch[band]
            else np.zeros((0, len(t_centers)), dtype=np.float32))

    return (consensus_masks, t_centers, win_starts,
            metrics_per_ch, secondary_per_ch, win_samples)


def _coarse_mask_to_records(consensus_mask, win_starts,
                             win_samples, n_total):
    if not consensus_mask.any(): return []
    records, in_ep, ep_start_win = [], False, None
    for i, v in enumerate(consensus_mask):
        if v and not in_ep:
            ep_start_win = i;  in_ep = True
        elif not v and in_ep:
            records.append({
                'start_sample': int(win_starts[ep_start_win]),
                'end_sample':   int(min(win_starts[i-1] + win_samples,
                                        n_total)),
            })
            in_ep = False
    if in_ep:
        records.append({
            'start_sample': int(win_starts[ep_start_win]),
            'end_sample':   int(min(win_starts[-1] + win_samples, n_total)),
        })
    return records


def _apply_fine_refinement(raw, fs, ch_0idx, band,
                            coarse_records, win_samples):
    n_total = raw.shape[1]
    return [
        dict(zip(['start_sample', 'end_sample'],
                 refine_epoch_boundaries(
                     raw, fs, ch_0idx, band,
                     rec['start_sample'], rec['end_sample'],
                     win_samples, n_total)))
        for rec in coarse_records
    ]


# ============================================================
#  REFERENCE OVERVIEW PNG
# ============================================================

def _save_reference_overview_png(sleep_bin_path, sleep_n_channels,
                                  awake_bin_path, awake_n_channels,
                                  fs, sleep_met, awake_met,
                                  calib, save_dir):
    print("  Saving reference PNG ...", end=' ', flush=True)

    def _collect_reference_rows(bin_path, n_channels):
        raw     = load_bin_partial(bin_path, n_channels, int(600 * fs))
        ch_0idx = channels_for_recording(n_channels)
        dur_s   = raw.shape[1] / fs
        ch_metrics = {b: [] for b in BANDS}
        t_centers  = None
        for ch in ch_0idx:
            freqs, psd_mat, tc, _ = compute_sliding_welch(raw[ch], fs)
            if psd_mat.shape[0] == 0:
                continue
            if t_centers is None:
                t_centers = tc
            bp      = compute_band_powers(freqs, psd_mat)
            met_raw = compute_metrics(bp)
            met     = {k: running_median(v, SMOOTH_WINDOWS)
                       for k, v in met_raw.items()}
            for band in BANDS:
                ch_metrics[band].append(met[BAND_METRIC[band]])
        del raw
        return t_centers, ch_metrics, dur_s

    sleep_tc, sleep_rows, sleep_dur = _collect_reference_rows(
        sleep_bin_path, sleep_n_channels)
    awake_tc, awake_rows, awake_dur = _collect_reference_rows(
        awake_bin_path, awake_n_channels)

    t_centers = sleep_tc if sleep_tc is not None else awake_tc
    if t_centers is None or len(t_centers) == 0:
        print("skipped (no windows)"); return

    fig = plt.figure(figsize=(24, 15))
    gs = gridspec.GridSpec(len(BANDS), 4, figure=fig,
                           hspace=0.55, wspace=0.28)
    fig.suptitle(
        "Reference Calibration Overview\n"
        f"SLEEP: {os.path.basename(sleep_bin_path)}  |  "
        f"AWAKE: {os.path.basename(awake_bin_path)}",
        fontsize=20, fontweight='bold')

    col_titles = ["Distribution", "Sleep time series", "Awake time series", "ROC"]
    for col_idx, title in enumerate(col_titles):
        fig.text(0.10 + 0.23 * col_idx, 0.965, title,
                 ha='center', va='top', fontsize=16,
                 fontweight='bold')

    for row_idx, band in enumerate(BANDS):
        col    = BAND_COLORS[band]
        metric = BAND_METRIC[band]
        thresh = calib[band]['threshold']
        auc    = calib[band]['auc']
        lo, hi = BANDS[band]

        ax_h = fig.add_subplot(gs[row_idx, 0])
        ax_s = fig.add_subplot(gs[row_idx, 1])
        ax_a = fig.add_subplot(gs[row_idx, 2])
        ax_r = fig.add_subplot(gs[row_idx, 3])

        sleep_arr = sleep_met.get(metric, np.array([]))
        awake_arr = awake_met.get(metric, np.array([]))
        all_v = np.concatenate([sleep_arr, awake_arr]) \
                if (len(sleep_arr) and len(awake_arr)) else np.array([0, 1])
        bins = np.linspace(float(np.percentile(all_v, 0.5)),
                           float(np.percentile(all_v, 99.5)), 60)
        if len(sleep_arr):
            ax_h.hist(sleep_arr, bins=bins, color='steelblue',
                      alpha=0.55, density=True, label='Sleep',
                      edgecolor='none')
        if len(awake_arr):
            ax_h.hist(awake_arr, bins=bins, color='tomato',
                      alpha=0.55, density=True, label='Awake',
                      edgecolor='none')
        ax_h.axvline(thresh, color='black', lw=2.4, linestyle='--',
                     label=f'thresh={thresh:.4f}'
                           + (' [FB]' if calib[band]['fallback_used'] else ''))
        ax_h.set_xlabel(metric, fontsize=13)
        ax_h.set_ylabel('Density', fontsize=13)
        ax_h.set_title(f"{band.capitalize()}  ({lo}–{hi} Hz)",
                       fontsize=16, fontweight='bold')
        ax_h.legend(fontsize=10, frameon=True)

        for ax_src, rows, plot_col, lab in [
            (ax_s, sleep_rows.get(band, []), 'steelblue', 'Sleep'),
            (ax_a, awake_rows.get(band, []), 'tomato', 'Awake'),
        ]:
            mat = np.array(rows) if rows else np.zeros((0, len(t_centers)))
            if mat.shape[0] > 0:
                for ch_row in mat:
                    ax_src.plot(t_centers, ch_row, color=plot_col,
                                lw=0.7, alpha=0.30, zorder=1)
                mean_met = mat.mean(axis=0)
                ax_src.plot(t_centers, mean_met, color=plot_col, lw=2.6,
                            alpha=1.0, zorder=3, label=f'{lab} mean')
                above = mean_met > thresh
                shading, sh_s = False, None
                for tv, av in zip(t_centers, above):
                    if av and not shading:
                        sh_s = tv; shading = True
                    elif not av and shading:
                        ax_src.axvspan(sh_s, tv, color=plot_col,
                                       alpha=0.10, lw=0)
                        shading = False
                if shading and sh_s is not None:
                    ax_src.axvspan(sh_s, t_centers[-1], color=plot_col,
                                   alpha=0.10, lw=0)

            ax_src.axhline(thresh, color='black', lw=1.8, linestyle='--',
                           alpha=0.95, label=f'thresh={thresh:.4f}')
            ax_src.set_xlabel('Time (s)', fontsize=13)
            ax_src.set_ylabel('theta/delta' if band == 'theta' else metric,
                           fontsize=13)
            ax_src.legend(fontsize=9, frameon=True, loc='upper right')
            ax_src.text(
                0.01, 0.92, f"{lab}",
                transform=ax_src.transAxes, ha='left', va='top',
                fontsize=14, fontweight='bold', color=plot_col)
            if band == 'theta' and lab == 'Sleep':
                ax_src.text(
                    0.98, 0.96,
                    f"Youden\nthr={thresh:.4f}\nAUC={auc:.3f}",
                    transform=ax_src.transAxes, ha='right', va='top',
                    fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.25',
                              facecolor='white', alpha=0.80,
                              edgecolor='black', linewidth=0.8))

        if len(sleep_arr) and len(awake_arr):
            if band == 'theta':
                ax_s.set_ylim(0, 1)  # Change 1 to a higher value (e.g., 4 or 5) to avoid clipping peaks
                ax_a.set_ylim(0, 1)
            else:
                ymin = min(np.nanmin(sleep_arr), np.nanmin(awake_arr))
                ymax = max(np.nanmax(sleep_arr), np.nanmax(awake_arr))
                pad = (ymax - ymin) * 0.08 if ymax > ymin else 0.05
                ax_s.set_ylim(ymin - pad, ymax + pad)
                ax_a.set_ylim(ymin - pad, ymax + pad)

        for ax in (ax_h, ax_s, ax_a, ax_r):
            ax.grid(False)
            ax.set_frame_on(True)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.1)
            ax.tick_params(labelsize=12)

        ax_r.plot([], [])  # ensure frames stay initialized before content
        roc_t    = calib[band]['_roc_thresholds']
        roc_tpr  = calib[band]['_roc_tpr']
        roc_fpr  = calib[band]['_roc_fpr']
        roc_j    = calib[band]['_roc_youden_j']
        if len(roc_fpr) > 1:
            ax_r.plot(roc_fpr, roc_tpr, color=col, lw=2.2,
                      label=f'ROC  AUC={auc:.3f}')
            ax_r.plot([0, 1], [0, 1], 'k--', lw=1.0, alpha=0.5)
            if len(roc_j) > 0:
                best_idx = int(np.argmax(roc_j))
                ax_r.scatter(roc_fpr[best_idx], roc_tpr[best_idx],
                             color='black', s=70, zorder=5,
                             label=f'J={roc_j[best_idx]:.3f}  '
                                   f'thresh={thresh:.4f}')
            ax_r.set_xlabel('1 - Specificity (FPR)', fontsize=13)
            ax_r.set_ylabel('Sensitivity (TPR)', fontsize=13)
            ax_r.set_xlim(0, 1); ax_r.set_ylim(0, 1)
            ax_r.set_title("ROC", fontsize=16, fontweight='bold')
            ax_r.legend(fontsize=10, frameon=True)
            ax_inset = ax_r.inset_axes([0.54, 0.08, 0.42, 0.34])
            ax_inset.plot(roc_t, roc_j, color=col, lw=1.6)
            if len(roc_j) > 0:
                ax_inset.axvline(thresh, color='black', lw=1.2, linestyle='--')
                ax_inset.scatter([thresh], [roc_j[best_idx]],
                                 color='black', s=35, zorder=5)
            ax_inset.set_xlabel('threshold', fontsize=8)
            ax_inset.set_ylabel("Youden's J", fontsize=8)
            ax_inset.tick_params(labelsize=7)
            ax_inset.grid(False)
            for spine in ax_inset.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.9)
        else:
            ax_r.text(0.5, 0.5, 'ROC not available\n(loaded from CSV)',
                      transform=ax_r.transAxes, ha='center', va='center',
                      fontsize=11, color='gray')
            ax_r.set_title("ROC", fontsize=16, fontweight='bold')

        ax_r.tick_params(labelsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_png = os.path.join(save_dir, "reference_overview_combined.svg")
    plt.savefig(out_png)
    plt.close(fig)
    print(f"→  {os.path.basename(out_png)}")


# ============================================================
#  RECORDING OVERVIEW PNG
# ============================================================

def _save_epoch_overview_png(bin_path, t_centers, consensus_masks,
                              metrics_per_ch, secondary_per_ch,
                              calib, min_channels_required,
                              epoch_records_dict, fs,
                              detection_modes=None, spa=False):
    if detection_modes is None:
        detection_modes = {b: 'global' for b in BANDS}

    n_rows = 5
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(22, 17),
        gridspec_kw={'height_ratios': [3, 3, 3, 3, 1.3]},
        sharex=True)
    spa_tag = "  [SPA — theta skipped]" if spa else ""
    fig.suptitle(
        f"Epoch Overview\n{os.path.basename(bin_path)}{spa_tag}",
        fontsize=20, fontweight='bold')

    for row_idx, band in enumerate(BANDS):
        ax       = axes[row_idx]
        col      = BAND_COLORS[band]
        thresh   = calib[band]['threshold']
        metric   = calib[band]['metric']
        auc      = calib[band]['auc']
        sens     = calib[band]['sensitivity']
        spec     = calib[band]['specificity']
        fallback = calib[band]['fallback_used']
        mat      = metrics_per_ch[band]
        cmask    = consensus_masks[band]
        records  = epoch_records_dict[band]
        lo, hi   = BANDS[band]
        dmode    = detection_modes.get(band, 'global')

        if mat.shape[0] > 0 and len(t_centers) > 0:
            for ch_row in mat:
                ax.plot(t_centers, ch_row, color=col,
                        lw=0.6, alpha=0.45, zorder=1)
            mean_met = mat.mean(axis=0)
            ax.plot(t_centers, mean_met, color=col, lw=1.8,
                    alpha=1.0, zorder=3, label=f'Mean {metric}')

        # threshold line colour by detection mode
        tc_col = {'global': 'black', 'rescaled': 'orange',
                  'undetectable': 'red'}.get(dmode, 'black')
        ax.axhline(thresh, color=tc_col, lw=1.2, linestyle='--',
                   alpha=0.85, zorder=4,
                   label=f'thresh={thresh:.4f} [{dmode}]'
                         + (' [FB]' if fallback else ''))
        ax.set_ylabel('Value', fontsize=13)

        shading, sh_s = False, None
        for tv, mv in zip(t_centers, cmask):
            if mv and not shading:
                sh_s = tv; shading = True
            elif not mv and shading:
                ax.axvspan(sh_s, tv, color=col, alpha=0.10, lw=0, zorder=0)
                shading = False
        if shading and sh_s is not None:
            ax.axvspan(sh_s, t_centers[-1], color=col, alpha=0.10, lw=0)

        for rec in records:
            ax.axvspan(rec['start_s'], rec['end_s'],
                       ymin=0.0, ymax=0.07,
                       color=col, alpha=0.92, lw=0, zorder=5)
        ax.set_title(f"{band.capitalize()}", fontsize=15, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
        ax.grid(False)
        ax.set_frame_on(True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.1)
        ax.tick_params(labelsize=12)
        if len(t_centers) > 0:
            ax.set_ylim(0, 1.05)

    ax_bot = axes[4]
    ax_bot.set_yticks([])
    ax_bot.set_ylabel('Epochs', fontsize=13)
    ax_bot.set_xlabel('Time (s)', fontsize=13)
    ax_bot.set_title('Final epochs', fontsize=14, fontweight='bold')
    for yi, band in enumerate(BANDS):
        col     = BAND_COLORS[band]
        records = epoch_records_dict[band]
        y_lo    = yi / 4;  y_hi = (yi + 0.85) / 4
        for rec in records:
            ax_bot.axvspan(rec['start_s'], rec['end_s'],
                           ymin=y_lo, ymax=y_hi,
                           color=col, alpha=0.78, lw=0)
        ax_bot.text(-0.005, (y_lo + y_hi) / 2, band,
                    transform=ax_bot.get_yaxis_transform(),
                    ha='right', va='center',
                    fontsize=12, color=col, fontweight='bold')
    if len(t_centers) > 0:
        ax_bot.set_xlim(axes[0].get_xlim())

    ax_bot.grid(False)
    ax_bot.set_frame_on(True)
    for spine in ax_bot.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.1)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = bin_path[:-4] + "_epoch_overview.svg"
    plt.savefig(out_png); plt.close(fig)
    print(f"         Overview PNG  →  {os.path.basename(out_png)}")


# ============================================================
#  COVERAGE CHECK
# ============================================================

def check_epoch_coverage(csv_path, calib, fs=20000,
                          min_epoch_sec=MIN_EPOCH_SEC,
                          merge_gap_sec=MERGE_GAP_SEC,
                          min_channels_required=MIN_CHANNELS_REQUIRED):
    df_dir = pd.read_csv(csv_path)
    print("\n" + "=" * 76)
    print("EPOCH COVERAGE CHECK  (no files written)")
    print(f"  min_epoch={min_epoch_sec}s  merge={merge_gap_sec}s  "
          f"min_ch={min_channels_required}  smooth={SMOOTH_WINDOWS}")
    print(f"  gamma rescaling: enabled={GAMMA_RESCALE_ENABLED}  "
          f"min={GAMMA_RESCALE_MIN}  max={GAMMA_RESCALE_MAX}")
    print(f"  theta calibrated as theta/delta  |  spa=TRUE → theta skipped")
    print("=" * 76)
    print(f"\n  {'File':<32s}"
          f"{'spa':>5s}"
          f"{'dur(m)':>7s}"
          f"{'delta%':>8s}"
          f"{'theta%':>8s}"
          f"{'lgam%':>8s}"
          f"{'hgam%':>8s}"
          f"{'lg_mode':>12s}"
          f"{'hg_mode':>12s}")
    print("  " + "-" * 84)

    summary = {b: [] for b in BANDS}
    for _, file_row in df_dir.iterrows():
        bin_path   = str(file_row['filename']).strip()
        n_channels = int(file_row['n_channels'])
        spa        = read_spa_flag(file_row)
        if not os.path.exists(bin_path):
            print(f"  [SKIP]  {os.path.basename(bin_path)}"); continue
        try:
            raw      = load_bin_partial(bin_path, n_channels, int(600*fs))
            ch_0idx  = channels_for_recording(n_channels)
            dur_s    = raw.shape[1] / fs
            skip_bands = {'theta'} if spa else set()

            # gamma rescaling
            gm = _recording_gamma_medians(bin_path, n_channels, fs)
            eff_calib, det_modes, _ = _get_effective_calib(calib, gm)

            # zero out undetectable gamma
            for band in ('lowgamma', 'highgamma'):
                if det_modes[band] == 'undetectable':
                    skip_bands.add(band)

            (consensus_masks, _, win_starts,
             _, _, win_samples) = _detect_consensus(
                raw, fs, ch_0idx, eff_calib,
                min_channels_required, skip_bands=skip_bands)

            epoch_records = {}
            for band in BANDS:
                coarse    = _coarse_mask_to_records(
                    consensus_masks[band], win_starts,
                    win_samples, raw.shape[1])
                processed = postprocess_epoch_records(
                    coarse, fs, min_epoch_sec, merge_gap_sec)
                epoch_records[band] = processed

            if not spa:
                epoch_records['theta'] = _apply_delta_exclusion(
                    epoch_records['theta'], epoch_records['delta'],
                    fs, min_epoch_sec)

            coverages = {}
            for band in BANDS:
                ep_sec = sum(r['duration_s']
                             for r in epoch_records[band])
                coverages[band] = ep_sec / dur_s * 100.0
                summary[band].append(coverages[band])
            del raw
            name = os.path.basename(bin_path)[:30]
            print(f"  {name:<32s}"
                  f"{'T' if spa else 'F':>5s}"
                  f"{dur_s/60:>6.1f}m"
                  f"{coverages['delta']:>7.1f}%"
                  f"{coverages['theta']:>7.1f}%"
                  f"{coverages['lowgamma']:>7.1f}%"
                  f"{coverages['highgamma']:>7.1f}%"
                  f"{det_modes['lowgamma']:>12s}"
                  f"{det_modes['highgamma']:>12s}")
        except Exception as e:
            print(f"  [ERROR]  {os.path.basename(bin_path)}: {e}")

    print("  " + "-" * 84)
    for lbl, fn in [('MEAN', np.mean), ('MIN', np.min), ('MAX', np.max)]:
        vals = [f"{fn(summary[b]):>7.1f}%" if summary[b] else f"{'N/A':>8s}"
                for b in BANDS]
        print(f"  {lbl:>37s}{'':>7s}" + "".join(vals))
    print("=" * 76)
    return summary


# ============================================================
#  MAIN BATCH RUNNER
# ============================================================

def run_epoch_detection_batch(csv_path,
                               sleep_reference_path,
                               sleep_n_channels,
                               awake_reference_path,
                               awake_n_channels,
                               fs=20000,
                               coverage_target=COVERAGE_TARGET,
                               auc_fallback=AUC_FALLBACK_THRESHOLD,
                               min_channels_required=MIN_CHANNELS_REQUIRED,
                               min_epoch_sec=MIN_EPOCH_SEC,
                               merge_gap_sec=MERGE_GAP_SEC,
                               stats_csv_path=None,
                               save_png_dir=None,
                               force_stats=False,
                               force_epochs=False):
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV not found: {csv_path}"); return

    calib = load_or_compute_reference_stats(
        sleep_reference_path, sleep_n_channels,
        awake_reference_path, awake_n_channels,
        fs=fs, coverage_target=coverage_target,
        auc_fallback=auc_fallback,
        stats_csv_path=stats_csv_path,
        save_png_dir=save_png_dir, force=force_stats)

    df_dir  = pd.read_csv(csv_path)
    n_files = len(df_dir)
    print(f"\n{'='*64}")
    print(f"EPOCH DETECTION  —  {n_files} recording(s)")
    print(f"  fs={fs}  min_ch={min_channels_required}  "
          f"smooth={SMOOTH_WINDOWS}  "
          f"min_epoch={min_epoch_sec}s  merge={merge_gap_sec}s")
    print(f"  gamma rescaling: enabled={GAMMA_RESCALE_ENABLED}  "
          f"min={GAMMA_RESCALE_MIN}  max={GAMMA_RESCALE_MAX}")
    print(f"  theta calibrated as theta/delta  |  spa=TRUE → theta skipped")
    print(f"{'='*64}\n")

    for file_num, (_, file_row) in enumerate(df_dir.iterrows(), start=1):
        bin_path   = str(file_row['filename']).strip()
        n_channels = int(file_row['n_channels'])
        spa        = read_spa_flag(file_row)

        print(f"[{file_num}/{n_files}]  {os.path.basename(bin_path)}"
              + ("  [SPA]" if spa else ""))

        if not os.path.exists(bin_path):
            print(f"  [SKIP]  File not found.\n"); continue
        if not force_epochs and all_epoch_csvs_exist(bin_path):
            print(f"  [SKIP]  All 4 epoch CSVs exist.\n"); continue

        print(f"  Loading ({n_channels} ch) ...", end=' ', flush=True)
        try:
            raw = load_bin_partial(bin_path, n_channels, int(600 * fs))
        except Exception as e:
            print(f"\n  [ERROR loading]  {e}\n"); continue
        dur_s = raw.shape[1] / fs
        print(f"shape {raw.shape}  ({dur_s:.1f} s)")
        ch_0idx = channels_for_recording(n_channels)
        print(f"  Channels (0-idx): {ch_0idx}")

        # ---- gamma rescaling ----
        print(f"  Gamma inspection ...", end=' ', flush=True)
        try:
            gm = _recording_gamma_medians(bin_path, n_channels, fs)
            eff_calib, det_modes, g_ratios = _get_effective_calib(calib, gm)
        except Exception as e:
            print(f"[error: {e}] using global thresholds  ")
            eff_calib = calib
            det_modes = {b: 'global' for b in BANDS}
            g_ratios  = {}

        skip_bands = {'theta'} if spa else set()
        for band in ('lowgamma', 'highgamma'):
            if det_modes[band] == 'undetectable':
                skip_bands.add(band)

        for band in ('lowgamma', 'highgamma'):
            ratio = g_ratios.get(band, float('nan'))
            print(f"\n    [{band}]  "
                  f"rec_median={gm.get(band, float('nan')):.5f}  "
                  f"ratio={ratio:.3f}  "
                  f"mode={det_modes[band]}  "
                  f"eff_thresh={eff_calib[band]['threshold']:.5f}")

        if spa:
            print(f"  [SPA] theta detection skipped.")

        # ---- coarse detection ----
        print(f"  Coarse detection ...", end=' ', flush=True)
        try:
            (consensus_masks, t_centers, win_starts,
             metrics_per_ch, secondary_per_ch,
             win_samples) = _detect_consensus(
                raw, fs, ch_0idx, eff_calib,
                min_channels_required, skip_bands=skip_bands)
        except Exception as e:
            print(f"\n  [ERROR]  {e}\n"); del raw; continue
        print("done")

        csv_paths          = epochs_csv_paths(bin_path)
        epoch_records_dict = {}

        for band in BANDS:
            print(f"  [{band:>10s}] ", end='', flush=True)

            if band in skip_bands:
                reason = ('SPA' if band == 'theta' and spa
                          else det_modes.get(band, ''))
                print(f"skipped [{reason}]  →  0 epochs")
                epoch_records_dict[band] = []
                _empty_epoch_df().to_csv(csv_paths[band], index=False)
                continue

            coarse = _coarse_mask_to_records(
                consensus_masks[band], win_starts,
                win_samples, raw.shape[1])

            if not coarse:
                print("no coarse epochs  →  0 epochs")
                epoch_records_dict[band] = []
                _empty_epoch_df().to_csv(csv_paths[band], index=False)
                continue

            print(f"{len(coarse)} coarse  →  refinement ...",
                  end=' ', flush=True)
            try:
                refined = _apply_fine_refinement(
                    raw, fs, ch_0idx, BANDS[band], coarse, win_samples)
            except Exception as e:
                print(f"[error: {e}] using coarse  ", end='')
                refined = coarse

            processed = postprocess_epoch_records(
                refined, fs, min_epoch_sec, merge_gap_sec)
            epoch_records_dict[band] = processed
            total_min = sum(r['duration_s'] for r in processed) / 60.0
            print(f"→  {len(processed)} epochs  ({total_min:.2f} min)")

        # ---- save CSVs ----
        for band in BANDS:
            records = epoch_records_dict.get(band, [])
            dmode   = det_modes.get(band, 'global')
            df      = _records_to_df(records, detection_mode=dmode)
            try:
                df.to_csv(csv_paths[band], index=False)
                print(f"  [{band:>10s}]  saved  →  "
                      f"{os.path.basename(csv_paths[band])}")
            except Exception as e:
                print(f"  [{band}]  [ERROR saving CSV]  {e}")

        # ---- overview PNG ----
        print(f"  Saving overview PNG ...", end=' ', flush=True)
        try:
            _save_epoch_overview_png(
                bin_path, t_centers, consensus_masks,
                metrics_per_ch, secondary_per_ch,
                eff_calib, min_channels_required,
                epoch_records_dict, fs,
                detection_modes=det_modes, spa=spa)
        except Exception as e:
            print(f"\n  [ERROR saving PNG]  {e}")

        del raw
        print(f"  [DONE]\n")

    print("=" * 64)
    print("Batch epoch detection complete.")


# ============================================================
#  INTERACTIVE ENTRY POINT
# ============================================================

def _prompt(text, default):
    try:
        v = input(f"  {text} [{default}]: ").strip()
    except EOFError:
        return default
    return v if v != '' else default


def _interactive_run():
    print("=" * 64)
    print("  LFP Epoch Detection")
    print(f"  Delta   : frac > thresh  AND  delta_frac > theta_frac_raw")
    print(f"  Theta   : theta/delta ratio > thresh  (skipped if spa=TRUE)")
    print(f"  LGamma  : frac > thresh  (rescaled if rec median low)")
    print(f"  HGamma  : frac > thresh  (rescaled if rec median low)")
    print("=" * 64)

    csv_path   = _prompt("Path to lfp_directory.csv",     "lfp_directory.csv")
    sleep_path = _prompt("Path to SLEEP reference .bin",  "")
    sleep_nch  = _prompt("n_channels in sleep file",      "23")
    awake_path = _prompt("Path to AWAKE reference .bin",  "")
    awake_nch  = _prompt("n_channels in awake file",      "23")
    fs_s       = _prompt("Sampling rate (Hz)",            "20000")
    cov_s      = _prompt("Fallback coverage target 0–1",  "0.75")
    auc_fb_s   = _prompt("AUC fallback threshold 0–1",    "0.60")
    min_ch_s   = _prompt("Min channels for consensus",    "6")
    min_ep_s   = _prompt("Min epoch duration (s)",        "30")
    merge_s    = _prompt("Merge gap (s)",                 "15")
    stats_csv  = _prompt("Reference stats CSV (Enter=auto)", "")
    png_dir    = _prompt("Reference PNG dir  (Enter=auto)",  "")
    f_st_s     = _prompt("Force recompute ref stats? (y/n)", "n")
    f_ep_s     = _prompt("Force recompute epochs? (y/n)",    "n")

    def _i(s, d):
        try: return int(s)
        except: return d
    def _f(s, d):
        try: return float(s)
        except: return d

    sleep_nch  = _i(sleep_nch,  23)
    awake_nch  = _i(awake_nch,  23)
    fs         = _f(fs_s,       20000.0)
    cov        = _f(cov_s,      0.75)
    auc_fb     = _f(auc_fb_s,   0.60)
    min_ch     = _i(min_ch_s,   6)
    min_ep     = _f(min_ep_s,   30.0)
    merge      = _f(merge_s,    15.0)
    stats_path = stats_csv if stats_csv != '' else None
    png_path   = png_dir   if png_dir   != '' else None
    force_st   = f_st_s.lower() in ('y', 'yes')
    force_ep   = f_ep_s.lower() in ('y', 'yes')

    errors = []
    if not os.path.exists(csv_path):
        errors.append(f"lfp_directory.csv not found: {csv_path}")
    for lbl, p in [('Sleep', sleep_path), ('Awake', awake_path)]:
        if p == '': errors.append(f"{lbl} reference path required.")
        elif not os.path.exists(p): errors.append(f"{lbl} not found: {p}")
    if not (0 < cov < 1):    errors.append("Coverage target must be 0–1.")
    if not (0 < auc_fb < 1): errors.append("AUC fallback must be 0–1.")
    if errors:
        print("\nErrors:")
        for e in errors: print(f"  • {e}")
        return

    print("\n" + "-" * 64)
    print(f"  CSV              : {csv_path}")
    print(f"  Sleep reference  : {sleep_path}  ({sleep_nch} ch)")
    print(f"  Awake reference  : {awake_path}  ({awake_nch} ch)")
    print(f"  fs               : {fs} Hz")
    print(f"  Fallback cov     : {cov*100:.0f}%")
    print(f"  AUC fallback     : {auc_fb}")
    print(f"  Min channels     : {min_ch}")
    print(f"  Smoothing        : {SMOOTH_WINDOWS}-window median")
    print(f"  Min epoch        : {min_ep} s")
    print(f"  Merge gap        : {merge} s")
    print(f"  Gamma rescale    : min={GAMMA_RESCALE_MIN}  "
          f"max={GAMMA_RESCALE_MAX}  "
          f"sample={GAMMA_SAMPLE_SEC}s")
    print(f"  Stats CSV        : {stats_path or 'auto'}")
    print(f"  PNG dir          : {png_path or 'auto'}")
    print(f"  Force stats      : {force_st}")
    print(f"  Force epochs     : {force_ep}")
    print("-" * 64)

    print("\n  What would you like to do?")
    print("    1  Coverage check only    (no files written)")
    print("    2  Gamma inspection only  (no files written)")
    print("    3  Full detection         (write CSVs + PNGs)")
    print("    4  All (inspect → check → detect)")
    action = _prompt("Choice", "4")
    if action not in ('1', '2', '3', '4'):
        print("Aborted."); return

    calib = load_or_compute_reference_stats(
        sleep_bin_path=sleep_path, sleep_n_channels=sleep_nch,
        awake_bin_path=awake_path, awake_n_channels=awake_nch,
        fs=fs, coverage_target=cov, auc_fallback=auc_fb,
        stats_csv_path=stats_path, save_png_dir=png_path,
        force=force_st)

    if action in ('2', '4'):
        save_g_png = _prompt(
            "Save gamma inspection PNG? (y/n)", "y").lower() in ('y', 'yes')
        inspect_gamma_distributions(
            csv_path, calib, fs=fs,
            save_png=save_g_png, png_dir=png_path)

    if action in ('1', '4'):
        check_epoch_coverage(
            csv_path, calib, fs=fs,
            min_epoch_sec=min_ep, merge_gap_sec=merge,
            min_channels_required=min_ch)

    if action in ('3', '4'):
        if action == '4':
            go = _prompt("\nProceed with full detection? (y/n)", "y")
            if go.lower() not in ('y', 'yes'):
                print("Detection skipped."); return
        run_epoch_detection_batch(
            csv_path=csv_path,
            sleep_reference_path=sleep_path,
            sleep_n_channels=sleep_nch,
            awake_reference_path=awake_path,
            awake_n_channels=awake_nch,
            fs=fs, coverage_target=cov, auc_fallback=auc_fb,
            min_channels_required=min_ch,
            min_epoch_sec=min_ep, merge_gap_sec=merge,
            stats_csv_path=stats_path, save_png_dir=png_path,
            force_stats=False, force_epochs=force_ep)


if __name__ == '__main__':
    _interactive_run()