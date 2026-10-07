"""
PLV (Phase-Locking Value) classifier.

Purpose
-------
Compute advanced circular statistics for spike-phase relationships including
circular harmonics (unimodal, bimodal, trimodal, quadmodal), temporal phase
shift, and temporal stability metrics.

Key features
------------
- Circular harmonic analysis (R1-R4)
- Temporal phase shift and stability
- Spike count thresholding (<15 spikes: Low_Spike_Count)
- Shape indicators for phase distribution

Usage
-----
    python -m stats.plv_classifier

Dependencies
------------
    numpy, pandas, scipy
"""
import os
import glob
import numpy as np
import pandas as pd

def calculate_circular_stats(phases, spike_times):
    """
    Calculates advanced circular harmonics, shape indicators, 
    and temporal stability metrics for a set of spike phases.
    """
    phases = np.asarray(phases, dtype=float)
    spike_times = np.asarray(spike_times, dtype=float)
    
    # Remove NaNs, infinities, or invalid values
    valid_mask = np.isfinite(phases) & np.isfinite(spike_times)
    phases = phases[valid_mask]
    spike_times = spike_times[valid_mask]
    
    n_spikes = phases.size
    if n_spikes < 15:  # Minimum spike count threshold for reliable shape profiling
        return {
            "n_spikes": n_spikes, "R_1_unimodal": np.nan, "circ_std_deg": np.nan,
            "R_2_bimodal": np.nan, "R_3_trimodal": np.nan, "R_4_quadmodal": np.nan,
            "temporal_phase_shift_deg": np.nan, "temporal_R_ratio": np.nan, 
            "category": "Low_Spike_Count"
        }
    
    # 1. Unimodal Vector Length (R1) and Preferred Phase
    vec_unimodal = np.mean(np.exp(1j * phases))
    r1 = float(np.abs(vec_unimodal))
    
    # 2. Circular Standard Deviation (in degrees)
    if r1 > 0:
        circ_std_rad = np.sqrt(-2 * np.log(r1))
        circ_std_deg = np.degrees(circ_std_rad)
    else:
        circ_std_deg = np.inf

    # 3. Higher-order Circular Harmonics to detect symmetric multi-peaked star/spiky configurations
    r2 = float(np.abs(np.mean(np.exp(1j * 2 * phases))))  # Bimodal / Opposing peaks
    r3 = float(np.abs(np.mean(np.exp(1j * 3 * phases))))  # Trimodal / 3-peak star
    r4 = float(np.abs(np.mean(np.exp(1j * 4 * phases))))  # Quadmodal / 4-peak star
    
    max_star_harmonic = max(r3, r4)
    
    # 4. Temporal Stability (Split-half analysis based on chronology of spike times)
    mid_time = np.median(spike_times)
    early_mask = spike_times <= mid_time
    late_mask = spike_times > mid_time
    
    phases_early = phases[early_mask]
    phases_late = phases[late_mask]
    
    if phases_early.size >= 5 and phases_late.size >= 5:
        vec_early = np.mean(np.exp(1j * phases_early))
        vec_late = np.mean(np.exp(1j * phases_late))
        
        r_early = np.abs(vec_early)
        r_late = np.abs(vec_late)
        
        pref_early = np.angle(vec_early)
        pref_late = np.angle(vec_late)
        
        # Wrapped angular distance between early and late blocks
        phase_shift_rad = np.angle(np.exp(1j * (pref_early - pref_late)))
        phase_shift_deg = np.abs(np.degrees(phase_shift_rad))
        
        # Stability of phase locking magnitude over time
        r_ratio = r_late / r_early if r_early > 0 else np.nan
    else:
        phase_shift_deg = np.nan
        r_ratio = np.nan

    # 5. Advanced Category Classification Logic
    if r1 >= 0.55 and circ_std_deg < 100:
        # Extremely high concentration, narrow single slice
        category = "Peacock_Tail_Narrow"
    elif r1 >= 0.25 and r1 > r2 and r1 > max_star_harmonic:
        # Classic single-cluster but broader distribution
        category = "Polarized_Pie_Wide"
    elif r2 > r1 and r2 > max_star_harmonic and r2 >= 0.25:
        # Symmetric 2-peaked configuration (standard PLV drops, but highly locked)
        category = "Bimodal_Symmetric"
    elif max_star_harmonic > r1 and max_star_harmonic > r2 and max_star_harmonic >= 0.20:
        # 3 or 4 distinct geometric spokes ("Star" profile)
        category = f"Star_Multimodal_R{3 if r3 > r4 else 4}"
    else:
        # Significant coupling exists, but spikes form an asymmetric, uneven or noisy cluster
        category = "Spiky_Asymmetric_Random"
        
    # Append temporal drift suffix if preferred phase rotated significantly during recording
    if np.isfinite(phase_shift_deg) and phase_shift_deg > 60:
        category += "_Phase_Drifting"

    return {
        "n_spikes": n_spikes,
        "R_1_unimodal": r1,
        "circ_std_deg": circ_std_deg,
        "R_2_bimodal": r2,
        "R_3_trimodal": r3,
        "R_4_quadmodal": r4,
        "temporal_phase_shift_deg": phase_shift_deg,
        "temporal_R_ratio": r_ratio,
        "category": category
    }

def process_all_detailed_data(root_dir, output_csv):
    """
    Recursively scans root_dir and its subdirectories for *_detailed_data.csv files,
    parses phases for all matching frequency bands, and aggregates profiling metrics.
    """
    aggregated_results = []
    
    # Discover all detailed data files recursively across all subdirectories
    search_pattern = os.path.join(root_dir, "**", "*_detailed_data.csv")
    file_list = glob.glob(search_pattern, recursive=True)
    
    if not file_list:
        print(f"No matching files found in {root_dir}. Check paths.")
        return
        
    print(f"Found {len(file_list)} target data files to process.")
    
    for file_path in file_list:
        file_name = os.path.basename(file_path)
        print(f"Processing: {file_name}")
        
        try:
            df = pd.read_csv(file_path)
            
            # Validate crucial target columns exist in file
            required_cols = ['cell_uid', 'unit_id', 'band', 'spike_phase_rad_in_epochs', 'spike_time_s_in_epochs']
            if not all(col in df.columns for col in required_cols):
                print(f"  --> Skipping {file_name}: Missing necessary data columns.")
                continue
                
            # Isolate and group populations by unique cluster and specific oscillatory band
            grouped = df.groupby(['cell_uid', 'unit_id', 'band'])
            
            for (cell_uid, unit_id, band), group in grouped:
                phases = group['spike_phase_rad_in_epochs'].values
                spike_times = group['spike_time_s_in_epochs'].values
                
                # Execute statistical feature mapping
                stats = calculate_circular_stats(phases, spike_times)
                
                # Attach file mapping identities
                stats['cell_uid'] = cell_uid
                stats['unit_id'] = unit_id
                stats['band'] = band
                stats['source_file'] = file_name
                
                aggregated_results.append(stats)
                
        except Exception as e:
            print(f"  --> Failed to process file {file_name} due to error: {e}")
            
    # Save compilation out to target destination
    if aggregated_results:
        summary_df = pd.DataFrame(aggregated_results)
        
        # Enforce column structure layout
        col_order = [
            'source_file', 'cell_uid', 'unit_id', 'band', 'category', 
            'n_spikes', 'R_1_unimodal', 'circ_std_deg', 'R_2_bimodal', 
            'R_3_trimodal', 'R_4_quadmodal', 'temporal_phase_shift_deg', 'temporal_R_ratio'
        ]
        summary_df = summary_df[[c for c in col_order if c in summary_df.columns]]
        
        # Ensure target destination path directory tree exists
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        summary_df.to_csv(output_csv, index=False)
        
        print(f"\nProcessing finalized. Global summary exported to: {output_csv}")
        print("\n=== Distribution Profile of Identified Phase Locking Shapes ===")
        print(summary_df['category'].value_counts())
    else:
        print("No units matched parsing/filtering conditions successfully.")

# ==========================================
# Execution Entry Point Example:
# ==========================================
if __name__ == "__main__":
    ROOT_DATA_DIR = "E:/in_vivo_in_vitro/analysis_out_prev"
    OUTPUT_SUMMARY_FILE = "E:/in_vivo_in_vitro/analysis_out_prev/all_bands_phase_locking_summary.csv"
    
    process_all_detailed_data(ROOT_DATA_DIR, OUTPUT_SUMMARY_FILE)
    pass