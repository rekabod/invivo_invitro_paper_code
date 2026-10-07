"""
Basic statistical tests with FDR (Benjamini-Hochberg) correction.

Purpose
-------
Compute global and group-level summary statistics for electrophysiology
metrics across states (awake, sleep, vitro), cell types, and layers.
Perform Mann-Whitney U tests with FDR correction for pairwise comparisons.

Outputs
-------
Excel workbook with sheets:
  - Global_Stats: overall metric summaries
  - State_Stats: per-state summaries
  - State_Celltype_Stats: per-state × cell-type summaries
  - State_Layer_Stats: per-state × layer summaries
  - Recording_Dim_Stats: per-recording dimensionality metrics
  - Stats_*: pairwise comparison results with FDR-corrected p-values

Usage
-----
    python -m stats.all_basic_stat_fdr

Dependencies
------------
    pandas, numpy, scipy, statsmodels, openpyxl
"""
import pandas as pd

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import warnings
import openpyxl
from openpyxl.styles import Font, PatternFill, Border, Side, Alignment
from openpyxl.formatting.rule import CellIsRule
import itertools

warnings.filterwarnings('ignore')

# --- Helper Functions for Stats ---
def get_sem(x): return stats.sem(x, nan_policy='omit', ddof=1)

def get_ci_low(x):
    data = x.dropna()
    if len(data) < 2: return np.nan
    m, se = np.mean(data), stats.sem(data, ddof=1)
    h = se * stats.t.ppf(0.975, len(data) - 1)
    return m - h

def get_ci_high(x):
    data = x.dropna()
    if len(data) < 2: return np.nan
    m, se = np.mean(data), stats.sem(data, ddof=1)
    h = se * stats.t.ppf(0.975, len(data) - 1)
    return m + h

# --- Load the CSV file ---
try:
    df = pd.read_csv("E:/in_vivo_in_vitro/stat/statbase/vivo_vitro_database_from_outputs_checked.csv")
except Exception as e:
    print(f"Error loading file: {e}")
    df = pd.DataFrame()

if not df.empty:
    # Defining metrics
    basic_metrics = ['firing_rate_Hz', 'max_firing_rate_10s_Hz', 'max_firing_rate_1s_Hz', 'burstiness_percent', 'bursti_2', 'hf1_ms', 'isi_cv']
    osc_metrics = ['mean_plv_delta', 'mean_plv_theta', 'mean_plv_low_gamma', 'mean_plv_high_gamma', 'population_coupling',
                   'plv_low_gamma_strength', 'plv_low_gamma_clustering', 'plv_low_gamma_closeness', 'plv_low_gamma_local_efficiency', 'plv_low_gamma_hubness_z',
                   'plv_high_gamma_strength', 'plv_high_gamma_clustering', 'plv_high_gamma_closeness', 'plv_high_gamma_local_efficiency', 'plv_high_gamma_hubness_z',
                   'frsim_strength', 'frsim_clustering', 'frsim_closeness', 'frsim_local_efficiency', 'frsim_hubness_z']
    
    # New dimensionality metrics
    dim_metrics = ['dimensionality_pc1_variance', 'dimensionality_participation_ratio', 'dimensionality_eigenspectrum_entropy']
    
    all_metrics = [c for c in (basic_metrics + osc_metrics) if c in df.columns]
    valid_dim = [c for c in dim_metrics if c in df.columns]

    state_col = 'state' if 'state' in df.columns else None
    celltype_col = 'celltype_norm' if 'celltype_norm' in df.columns else ('celltype' if 'celltype' in df.columns else None)
    layer_col = 'layer_norm' if 'layer_norm' in df.columns else ('layer' if 'layer' in df.columns else None)

    # Global and Group Stats logic (as before)
    def calc_stats(data, group_cols, metrics):
        if not group_cols or not all(c in data.columns for c in group_cols): return pd.DataFrame()
        stats_df = data.groupby(group_cols)[metrics].agg(['mean', 'median', 'count', get_sem, get_ci_low, get_ci_high])
        stats_df.columns = ['_'.join(col).strip() for col in stats_df.columns.values]
        return stats_df.reset_index()
    
    recording_stats = df.groupby(['recording_id', state_col])[valid_dim].mean().reset_index()

    metrics_data = df[all_metrics]

    # Standard aggregators
    stats_base = metrics_data.agg(['mean', 'median', 'count'])

    # Custom aggregators using apply
    sem_vals = metrics_data.apply(get_sem)
    ci_low_vals = metrics_data.apply(get_ci_low)
    ci_high_vals = metrics_data.apply(get_ci_high)

    # Combine into a single DataFrame
    global_stats = pd.concat([stats_base, 
                            pd.DataFrame({'get_sem': sem_vals, 
                                            'get_ci_low': ci_low_vals, 
                                            'get_ci_high': ci_high_vals}).T])

    # Transpose so metrics are rows, stats are columns
    global_stats = global_stats.T
    global_stats.index.name = 'Metric'
    global_stats.reset_index(inplace=True)

    state_stats = calc_stats(df, [state_col] if state_col else [], all_metrics)
    state_cell_stats = calc_stats(df, [state_col, celltype_col] if state_col and celltype_col else [], all_metrics)
    state_layer_stats = calc_stats(df, [state_col, layer_col] if state_col and layer_col else [], all_metrics)

    # Statistical comparisons (Mann-Whitney test FDR corrections)
    def calc_comparisons(data, group_col, target_col, metrics):
        if not group_col or not target_col or group_col not in data.columns or target_col not in data.columns: 
            return pd.DataFrame()
        results = []
        
        targets = data[target_col].dropna().unique()
        
        for t in targets:
            sub_df = data[data[target_col] == t]
            groups = sub_df[group_col].dropna().unique()
            valid_groups = [g for g in groups if len(sub_df[sub_df[group_col] == g]) > 3] 
            
            for g1, g2 in itertools.combinations(valid_groups, 2):
                for m in metrics:
                    d1 = sub_df[sub_df[group_col] == g1][m].dropna()
                    d2 = sub_df[sub_df[group_col] == g2][m].dropna()
                    if len(d1) > 3 and len(d2) > 3:
                        try:
                            stat, p = stats.mannwhitneyu(d1, d2, alternative='two-sided')
                            results.append({
                                'Condition': t, 
                                'Comparison': f"{g1} vs {g2}", 
                                'Metric': m,
                                'N1': len(d1), 'N2': len(d2),
                                'Mean_1': d1.mean(), 'Median_1': d1.median(), 'SEM_1': get_sem(d1), 'CI_Low_1': get_ci_low(d1), 'CI_High_1': get_ci_high(d1),
                                'Mean_2': d2.mean(), 'Median_2': d2.median(), 'SEM_2': get_sem(d2), 'CI_Low_2': get_ci_low(d2), 'CI_High_2': get_ci_high(d2),
                                'p_uncorr': p
                            })
                        except Exception:
                            pass
        
        if results:
            res_df = pd.DataFrame(results)
            valid_p = res_df['p_uncorr'].notna()
            if valid_p.any():
                _, pvals_corrected, _, _ = multipletests(res_df.loc[valid_p, 'p_uncorr'], alpha=0.05, method='fdr_bh')
                res_df.loc[valid_p, 'p_FDR'] = pvals_corrected
                res_df['Significant'] = res_df['p_FDR'] < 0.05
            return res_df
        return pd.DataFrame()

    # Run comparisons
    comp_state_within_cell = calc_comparisons(df, state_col, celltype_col, all_metrics)
    comp_cell_within_state = calc_comparisons(df, celltype_col, state_col, all_metrics)
    comp_layer_within_state = calc_comparisons(df, layer_col, state_col, exist_osc)
    comp_dim_state = calc_comparisons(recording_stats, 'state', 'state', valid_dim)
    

    # Export to .xlsx
    out_file = 'E:/in_vivo_in_vitro/stat/statbase/stat_analysis_report.xlsx'
    with pd.ExcelWriter(out_file, engine='openpyxl') as writer:
        global_stats.to_excel(writer, sheet_name='Global_Stats', index=False)
        if not state_stats.empty: state_stats.to_excel(writer, sheet_name='State_Stats', index=False)
        if not state_cell_stats.empty: state_cell_stats.to_excel(writer, sheet_name='State_Celltype_Stats', index=False)
        if not state_layer_stats.empty: state_layer_stats.to_excel(writer, sheet_name='State_Layer_Stats', index=False)
        if not comp_state_within_cell.empty: comp_state_within_cell.to_excel(writer, sheet_name='Stats_State_in_Cell', index=False)
        if not comp_cell_within_state.empty: comp_cell_within_state.to_excel(writer, sheet_name='Stats_Cell_in_State', index=False)
        if not comp_layer_within_state.empty: comp_layer_within_state.to_excel(writer, sheet_name='Stats_Layer_in_State', index=False)
        recording_stats.to_excel(writer, sheet_name='Recording_Dim_Stats', index=False)
        if not comp_dim_state.empty: comp_dim_state.to_excel(writer, sheet_name='Stats_Dim_State', index=False)

    # Formatting
    wb = openpyxl.load_workbook(out_file)
    header_fill = PatternFill(start_color="2A3439", end_color="2A3439", fill_type="solid")
    header_font = Font(color="FFFFFF", bold=True)
    border = Border(left=Side(style='thin', color='E0E0E0'), right=Side(style='thin', color='E0E0E0'), 
                    top=Side(style='thin', color='E0E0E0'), bottom=Side(style='thin', color='E0E0E0'))

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        if ws.dimensions != 'A1:A1': ws.auto_filter.ref = ws.dimensions
        ws.freeze_panes = "A2"
        for cell in ws[1]:
            cell.fill = header_fill; cell.font = header_font; cell.alignment = Alignment(horizontal="center", vertical="center")
            ws.column_dimensions[cell.column_letter].width = max(len(str(cell.value)) + 5, 12)
        for row in ws.iter_rows(min_row=2, max_row=ws.max_row, min_col=1, max_col=ws.max_column):
            for cell in row:
                cell.border = border
                if isinstance(cell.value, float):
                    header_val = ws.cell(row=1, column=cell.column).value
                    if header_val and 'p_' in header_val: cell.number_format = '0.000E+00'
                    else: cell.number_format = '0.00'
        
        # FDR highlighting
        headers = [c.value for c in ws[1]]
        if 'p_FDR' in headers:
            fdr_col_idx = headers.index('p_FDR') + 1
            fdr_col_letter = openpyxl.utils.get_column_letter(fdr_col_idx)
            ws.conditional_formatting.add(f"{fdr_col_letter}2:{fdr_col_letter}{ws.max_row}",
                                          CellIsRule(operator='lessThan', formula=['0.05'], fill=PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid"), font=Font(color="006100")))
            
    wb.save(out_file)
    print(f"File generated: {out_file}")
else:
    print("DataFrame is empty.")