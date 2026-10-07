"""
Histology statistics for NeuN and PV immunohistochemistry data.

Purpose
-------
Analyze cat autofluorescence data from histology experiments. Compute
group-level statistics, ANOVA, normality tests, and post-hoc comparisons
for histology metrics.

Key features
------------
- Group-level curve statistics
- One-way ANOVA and Kruskal-Wallis tests
- Normality tests (Shapiro-Wilk)
- Post-hoc pairwise comparisons
- Publication-ready curve plots

Usage
-----
    python -m stats.histo_stat

Dependencies
------------
    numpy, pandas, scipy.stats, matplotlib, seaborn
"""


import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import shapiro, levene, kruskal, f_oneway, mannwhitneyu
from itertools import combinations


# -----------------------------
# CONFIG
# -----------------------------
ALL_CATS_CSV = "all_cats.csv"
DESCRIPTIONS_CSV = "descriptions.csv"

OUTPUT_GROUPS_CSV = "group_dictionary.csv"
OUTPUT_SUMMARY_CSV = "curve_statistics.csv"
OUTPUT_ANOVA_CSV = "anova_results.csv"
OUTPUT_NORMALITY_CSV = "normality_results.csv"
OUTPUT_POSTHOC_CSV = "posthoc_results.csv"
OUTPUT_PLOT_PNG = "group_curves.png"

ALPHA = 0.05


# -----------------------------
# UTILITIES
# -----------------------------
def safe_name(x):
    x = str(x).strip()
    x = x.replace(" ", "_")
    x = re.sub(r'[^\w\-]+', "_", x, flags=re.UNICODE)
    x = re.sub(r"_+", "_", x).strip("_")
    return x


def parse_int_list(value):
    """
    Parse values like:
      - "875, 826, 13"
      - "14"
      - 14
      - NaN
    Returns list[int]
    """
    if pd.isna(value):
        return []
    s = str(value).strip()
    if s in {"", "-", "N/A", "nan"}:
        return []
    parts = re.split(r"[,\s]+", s)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        try:
            out.append(int(float(p)))
        except ValueError:
            pass
    return out


def parse_cat_code_list(value):
    """
    For fields that may contain multiple codes in a comma-separated string.
    Example: "875, 826, 13" -> [875, 826, 13]
    Example: "GCaMP6s" stays as string if not numeric parsing is desired elsewhere.
    """
    return parse_int_list(value)


def intensity_axis(n_bins):
    return np.arange(n_bins)


# -----------------------------
# LOAD DATA
# -----------------------------
def load_data(all_cats_csv=ALL_CATS_CSV, descriptions_csv=DESCRIPTIONS_CSV):
    cats_df = pd.read_csv(all_cats_csv)
    desc_df = pd.read_csv(descriptions_csv)
    return cats_df, desc_df


def build_cat_lookup(cats_df, desc_df):
    """
    Merge/align by cat name.
    all_cats.csv:
      first column = intensity label (pixel_intensity_values)
      remaining columns = cat names
    descriptions.csv:
      first column = cat names
      code column = numeric code / ROI index
    """
    # Intensities in rows, cats in columns -> transpose so rows become cats
    intensity_col = cats_df.columns[0]
    cat_names = list(cats_df.columns[1:])

    cat_matrix = cats_df.iloc[:, 1:].to_numpy().T  # shape: cats x intensities

    # description table indexed by cat name
    desc_df = desc_df.copy()
    desc_df["cat"] = desc_df["cat"].astype(str)
    desc_df = desc_df.set_index("cat")

    # build aligned dataframe of cat metadata
    rows = []
    for i, cat in enumerate(cat_names):
        row = {"cat": cat, "roi_index_in_matrix": i}
        if cat in desc_df.index:
            meta = desc_df.loc[cat].to_dict()
            row.update(meta)
        rows.append(row)

    meta_df = pd.DataFrame(rows)

    # add matrix as separate mapping
    cat_to_idx = {cat: i for i, cat in enumerate(cat_names)}

    return cat_matrix, meta_df, cat_to_idx


# -----------------------------
# GROUPING BASED ON descriptions.csv
# -----------------------------
def build_groups_from_descriptions(meta_df, cat_to_idx):
    """
    Grouping logic based on descriptions.csv.

    Edit these rules to match your desired analysis.
    Here we provide useful groups derived from the CSV columns.
    """
    groups = {}

    # Convenience selectors
    def idx_where(mask):
        cats = meta_df.loc[mask, "cat"].tolist()
        return [cat_to_idx[c] for c in cats if c in cat_to_idx]

    # Example groups based on your metadata:
    groups["Hemosiderin_0"] = idx_where(meta_df["Hemosiderin"] == 0)
    groups["Hemosiderin_1"] = idx_where(meta_df["Hemosiderin"] == 1)
    groups["Hemosiderin_2"] = idx_where(meta_df["Hemosiderin"] == 2)

    groups["Lipofuscin_0"] = idx_where(meta_df["Lipofuscin"] == 0)
    groups["Lipofuscin_1"] = idx_where(meta_df["Lipofuscin"] == 1)
    groups["Lipofuscin_2"] = idx_where(meta_df["Lipofuscin"] == 2)

    groups["Background_0"] = idx_where(meta_df["Background"] == 0)
    groups["Background_1"] = idx_where(meta_df["Background"] == 1)
    groups["Background_2"] = idx_where(meta_df["Background"] == 2)
    groups["Background_3"] = idx_where(meta_df["Background"] == 3)

    groups["Vessels_0"] = idx_where(meta_df["Vessels"] == 0)
    groups["Vessels_1"] = idx_where(meta_df["Vessels"] == 1)

    groups["Labelled_cells_0"] = idx_where(meta_df['"""Labelled"" cells in the autofluor sample'] == 0)
    groups["Labelled_cells_1"] = idx_where(meta_df['"""Labelled"" cells in the autofluor sample'] == 1)

    groups["Layer4_bright_0"] = idx_where(meta_df["Layer 4 bright"] == 0)
    groups["Layer4_bright_1"] = idx_where(meta_df["Layer 4 bright"] == 1)

    groups["Total_score_4"] = idx_where(meta_df["Total score for autofluorescence"] == 4)
    groups["Total_score_5"] = idx_where(meta_df["Total score for autofluorescence"] == 5)
    groups["Total_score_6"] = idx_where(meta_df["Total score for autofluorescence"] == 6)
    groups["Total_score_7"] = idx_where(meta_df["Total score for autofluorescence"] == 7)

    groups["Perfusion_planned"] = idx_where(meta_df["Perfusion"].astype(str).str.contains("planned", case=False, na=False))
    groups["Perfusion_postmortem"] = idx_where(meta_df["Perfusion"].astype(str).str.contains("post-mortem", case=False, na=False))
    groups["Perfusion_alive"] = idx_where(meta_df["Perfusion"].astype(str).str.contains("alive", case=False, na=False))
    groups["Perfusion_immersion"] = idx_where(meta_df["Perfusion"].astype(str).str.contains("immersion", case=False, na=False))

    groups["Healthy_yes"] = idx_where(meta_df["Healthy?"].astype(str).str.contains("healthy", case=False, na=False))
    groups["Healthy_no"] = idx_where(meta_df["Healthy?"].astype(str).str.contains("sick", case=False, na=False))

    groups["Tissue_quality_low"] = idx_where(meta_df["Tissue quality after perfusion"].astype(str).str.contains("low", case=False, na=False))
    groups["Tissue_quality_medium"] = idx_where(meta_df["Tissue quality after perfusion"].astype(str).str.contains("medium", case=False, na=False))
    groups["Tissue_quality_high"] = idx_where(meta_df["Tissue quality after perfusion"].astype(str).str.contains("high", case=False, na=False))

    groups["Postfixation_overnight"] = idx_where(meta_df["Postfixation"].astype(str).str.contains("overnight", case=False, na=False))
    groups["Postfixation_days"] = idx_where(meta_df["Postfixation"].astype(str).str.contains("day", case=False, na=False))

    groups["Cutting_matrix_yes"] = idx_where(meta_df["Use of cutting matrix"].astype(str).str.contains("yes", case=False, na=False))
    groups["Cutting_matrix_no"] = idx_where(meta_df["Use of cutting matrix"].astype(str).str.contains("no", case=False, na=False))

    groups["Postfix_quality_0_3"] = idx_where(meta_df["Postfix quality"] <= 3)
    groups["Postfix_quality_4_5"] = idx_where(meta_df["Postfix quality"].between(4, 5))
    groups["Postfix_quality_6_7"] = idx_where(meta_df["Postfix quality"].between(6, 7))

    groups["Bad_storage_months_ge_12"] = idx_where(meta_df["Bad storage at -20C (months)"] >= 12)
    groups["Bad_storage_months_lt_12"] = idx_where(meta_df["Bad storage at -20C (months)"] < 12)

    groups["Injection_eyes"] = idx_where(meta_df["Injection"].astype(str).str.contains("eyes", case=False, na=False))
    groups["Injection_right_eye"] = idx_where(meta_df["Injection"].astype(str).str.contains("right eye", case=False, na=False))
    groups["Injection_left_eye"] = idx_where(meta_df["Injection"].astype(str).str.contains("left eye", case=False, na=False))
    groups["Injection_IV"] = idx_where(meta_df["Injection"].astype(str).str.contains("IV", case=False, na=False))
    groups["Injection_IC"] = idx_where(meta_df["Injection"].astype(str).str.contains("IC", case=False, na=False))
    groups["Injection_ICM"] = idx_where(meta_df["Injection"].astype(str).str.contains("ICM", case=False, na=False))

    groups["Age_lt_100"] = idx_where(meta_df["Age of the animal (days)"] < 100)
    groups["Age_100_500"] = idx_where(meta_df["Age of the animal (days)"].between(100, 500, inclusive="both"))
    groups["Age_gt_500"] = idx_where(meta_df["Age of the animal (days)"] > 500)

    groups["Expression_lt_100"] = idx_where(meta_df["Time of expression (days)"] < 100)
    groups["Expression_100_500"] = idx_where(meta_df["Time of expression (days)"].between(100, 500, inclusive="both"))
    groups["Expression_gt_500"] = idx_where(meta_df["Time of expression (days)"] > 500)

    # Biological score categories from text fields
    groups["Native_autofluorescent_neurons"] = idx_where(meta_df["Native"].astype(str).str.contains("autofluorescent neurons", case=False, na=False))
    groups["Native_no_neurons"] = idx_where(meta_df["Native"].astype(str).str.contains("no neurons", case=False, na=False))
    groups["IF_neurons"] = idx_where(meta_df["Immunofluorescence"].astype(str).str.contains("neurons", case=False, na=False))
    groups["Peroxidase_neurons"] = idx_where(meta_df["Peroxidase"].astype(str).str.contains("neurons", case=False, na=False))

    # Drop empty groups
    groups = {k: v for k, v in groups.items() if len(v) > 0}
    return groups


# -----------------------------
# CURVE STATS
# -----------------------------
def summarize_group_curves(cat_matrix, groups):
    """
    Compute group-mean curve, SD, SEM, AUC, peak, peak intensity, etc.
    """
    records = []
    for group_name, idxs in groups.items():
        sub = cat_matrix[idxs, :]  # rows = cats, cols = intensities
        mean_curve = sub.mean(axis=0)
        sd_curve = sub.std(axis=0, ddof=1) if sub.shape[0] > 1 else np.zeros(sub.shape[1])
        sem_curve = sd_curve / np.sqrt(sub.shape[0]) if sub.shape[0] > 0 else np.nan

        auc = np.trapz(mean_curve)
        peak_intensity = int(np.argmax(mean_curve))
        peak_value = float(np.max(mean_curve))
        total_signal = float(mean_curve.sum())

        records.append({
            "group": group_name,
            "n_cats": sub.shape[0],
            "auc": auc,
            "peak_intensity": peak_intensity,
            "peak_value": peak_value,
            "total_signal": total_signal,
            "mean_of_mean_curve": float(np.mean(mean_curve)),
            "sd_across_curve": float(np.mean(sd_curve)),
        })

    return pd.DataFrame(records)


# -----------------------------
# NORMALITY / ANOVA / POST-HOCS
# -----------------------------
def flatten_group_values(cat_matrix, idxs):
    """
    For a group, flatten all pixel-intensity counts into one long vector.
    Useful for normality / omnibus tests.
    """
    if len(idxs) == 0:
        return np.array([])
    return cat_matrix[idxs, :].flatten()


def run_normality_tests(cat_matrix, groups):
    """
    Shapiro-Wilk per group on flattened values.
    """
    rows = []
    for group_name, idxs in groups.items():
        vals = flatten_group_values(cat_matrix, idxs)
        if len(vals) < 3:
            rows.append({
                "group": group_name,
                "n": len(vals),
                "test": "Shapiro-Wilk",
                "statistic": np.nan,
                "pvalue": np.nan,
                "note": "too few values",
            })
            continue

        # Shapiro is limited for very large N; sample if needed
        sample = vals
        if len(vals) > 5000:
            rng = np.random.default_rng(0)
            sample = rng.choice(vals, size=5000, replace=False)

        stat_val, p_val = shapiro(sample)
        rows.append({
            "group": group_name,
            "n": len(vals),
            "test": "Shapiro-Wilk",
            "statistic": stat_val,
            "pvalue": p_val,
            "note": "sampled 5000" if len(vals) > 5000 else "",
        })
    return pd.DataFrame(rows)


def run_variance_test(cat_matrix, groups):
    """
    Levene test across groups.
    """
    samples = []
    labels = []
    for group_name, idxs in groups.items():
        vals = flatten_group_values(cat_matrix, idxs)
        if len(vals) > 0:
            samples.append(vals)
            labels.append(group_name)

    if len(samples) < 2:
        return pd.DataFrame([{
            "test": "Levene",
            "statistic": np.nan,
            "pvalue": np.nan,
            "note": "not enough groups"
        }])

    stat_val, p_val = levene(*samples, center="median")
    return pd.DataFrame([{
        "test": "Levene",
        "statistic": stat_val,
        "pvalue": p_val,
        "note": ""
    }])


def run_omnibus_tests(cat_matrix, groups):
    """
    Run one-way ANOVA and Kruskal-Wallis on flattened group values.
    """
    samples = []
    labels = []
    for group_name, idxs in groups.items():
        vals = flatten_group_values(cat_matrix, idxs)
        if len(vals) > 0:
            samples.append(vals)
            labels.append(group_name)

    results = []

    if len(samples) >= 2:
        f_stat, f_p = f_oneway(*samples)
        results.append({
            "test": "one_way_ANOVA",
            "statistic": f_stat,
            "pvalue": f_p,
            "n_groups": len(samples),
        })

        h_stat, h_p = kruskal(*samples)
        results.append({
            "test": "Kruskal_Wallis",
            "statistic": h_stat,
            "pvalue": h_p,
            "n_groups": len(samples),
        })

    return pd.DataFrame(results)


def pairwise_posthoc_mannwhitney(cat_matrix, groups, p_adjust="bonferroni"):
    """
    Pairwise post-hoc tests between all groups using Mann-Whitney U.
    Bonferroni correction by default.
    """
    valid = []
    for group_name, idxs in groups.items():
        vals = flatten_group_values(cat_matrix, idxs)
        if len(vals) > 0:
            valid.append((group_name, vals))

    rows = []
    pairs = list(combinations(valid, 2))
    m = len(pairs)

    for (g1, v1), (g2, v2) in pairs:
        if len(v1) == 0 or len(v2) == 0:
            continue
        u_stat, p_val = mannwhitneyu(v1, v2, alternative="two-sided")
        p_adj = min(p_val * m, 1.0) if p_adjust == "bonferroni" else p_val
        rows.append({
            "group_1": g1,
            "group_2": g2,
            "test": "Mann-Whitney U",
            "statistic": u_stat,
            "pvalue": p_val,
            "pvalue_adjusted": p_adj,
            "significant_alpha_0.05": p_adj < ALPHA,
        })

    return pd.DataFrame(rows)


# -----------------------------
# GROUP DICTIONARY OUTPUT
# -----------------------------
def save_group_dictionary(groups, meta_df, output_csv=OUTPUT_GROUPS_CSV):
    """
    Save a long-form CSV listing each group and the cats in it.
    """
    rows = []
    for group_name, idxs in groups.items():
        cats = meta_df.loc[meta_df["roi_index_in_matrix"].isin(idxs), "cat"].tolist()
        for cat in cats:
            rows.append({
                "group": group_name,
                "cat": cat,
                "roi_index_in_matrix": int(meta_df.loc[meta_df["cat"] == cat, "roi_index_in_matrix"].iloc[0]),
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    return df


# -----------------------------
# PLOTTING
# -----------------------------
def plot_group_mean_curves(cat_matrix, groups, output_png=OUTPUT_PLOT_PNG):
    fig, ax = plt.subplots(figsize=(15, 9))
    cmap = plt.cm.get_cmap("tab20", max(len(groups), 1))
    x = intensity_axis(cat_matrix.shape[1])

    for i, (group_name, idxs) in enumerate(groups.items()):
        sub = cat_matrix[idxs, :]
        if sub.shape[0] == 0:
            continue
        mean_curve = sub.mean(axis=0)
        sd_curve = sub.std(axis=0, ddof=1) if sub.shape[0] > 1 else np.zeros(sub.shape[1])

        color = cmap(i)
        ax.plot(x, mean_curve, label=f"{group_name} (n={sub.shape[0]})", color=color, linewidth=2.2)
        ax.fill_between(x, mean_curve - sd_curve, mean_curve + sd_curve, color=color, alpha=0.15)

    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Pixel count")
    ax.set_title("Autofluorescence curves by group")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.show()


# -----------------------------
# MAIN
# -----------------------------
def main():
    cats_df, desc_df = load_data(ALL_CATS_CSV, DESCRIPTIONS_CSV)
    cat_matrix, meta_df, cat_to_idx = build_cat_lookup(cats_df, desc_df)

    groups = build_groups_from_descriptions(meta_df, cat_to_idx)

    # Save group dictionary
    group_dict_df = save_group_dictionary(groups, meta_df, OUTPUT_GROUPS_CSV)

    # Curve summaries
    summary_df = summarize_group_curves(cat_matrix, groups)
    summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False)

    # Stats
    normality_df = run_normality_tests(cat_matrix, groups)
    normality_df.to_csv(OUTPUT_NORMALITY_CSV, index=False)

    anova_df = run_omnibus_tests(cat_matrix, groups)
    anova_df.to_csv(OUTPUT_ANOVA_CSV, index=False)

    posthoc_df = pairwise_posthoc_mannwhitney(cat_matrix, groups, p_adjust="bonferroni")
    posthoc_df.to_csv(OUTPUT_POSTHOC_CSV, index=False)

    # Plot
    plot_group_mean_curves(cat_matrix, groups, OUTPUT_PLOT_PNG)

    print("Done.")
    print(f"Saved: {OUTPUT_GROUPS_CSV}")
    print(f"Saved: {OUTPUT_SUMMARY_CSV}")
    print(f"Saved: {OUTPUT_NORMALITY_CSV}")
    print(f"Saved: {OUTPUT_ANOVA_CSV}")
    print(f"Saved: {OUTPUT_POSTHOC_CSV}")
    print(f"Saved: {OUTPUT_PLOT_PNG}")


if __name__ == "__main__":
    main()