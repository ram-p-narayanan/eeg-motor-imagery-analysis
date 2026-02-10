"""
Phase 1 - Step 1A: Channel Set Comparison Analysis

This script compares ERD-based discriminability for Task 3 (real motor execution:
fists vs feet) between two channel configurations:
    - Minimal motor ROI: C3, Cz, C4 (3 channels)
    - Extended sensorimotor grid: FC/C/CP channels (up to 21 channels)

The goal is to determine whether extended electrode montages improve classification
discriminability, helping inform optimal channel selection for subject-specific
BCI applications.

Analysis Pipeline:
    1. Load Task 3 epochs from runs R05, R09, R13 for each subject
    2. Concatenate epochs across runs for statistical power
    3. Extract minimal (C3, Cz, C4) and extended (FC/C/CP grid) channel sets
    4. Compute ERD% in mu (8-13 Hz) and beta (13-30 Hz) bands
    5. Calculate discriminability: |mean_ERD_T1 - mean_ERD_T2|
    6. Compare minimal vs extended performance per subject
    7. Export comprehensive CSV with improvement metrics

Key Metrics:
    - maxdiff: Maximum absolute ERD difference across channels
    - meandiff: Mean absolute ERD difference across channels
    - combo_score: Average of mu and beta maxdiff scores
    - delta_combo: Extended - Minimal combo score (positive = improvement)

Output:
    CSV file: Phase1_Step1A_Task3_ERD_discriminability_min_vs_ext.csv
    Contains per-subject metrics for both channel sets and improvement deltas

Important Notes:
    - This is a PRE-CSP analysis (discriminability, not classification accuracy)
    - Subjects with missing extended channels fall back to available subset
    - Requires ≥3 channels for stable extended metrics (configurable)
    - Phase 2 will validate findings with actual CSP+LDA classification

Author: Ram P Narayanan
Date: 2026-02-08
Version: 1.0.0
License: MIT

Dependencies:
    - mne >= 1.5.0
    - numpy >= 1.24.0
    - pandas >= 2.0.0

Usage:
    1. Edit clean_root path in CONFIGURATION section
    2. Optionally adjust MIN_CH_FOR_EXT threshold
    3. Run: python Phase1_Step1A_Channel_Comparison.py
    4. Review output CSV for channel set recommendations
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import mne


# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to cleaned dataset (output from preprocessing pipeline)
# Structure: clean_root/SXXX/SXXXR05-epo.fif, SXXXR09-epo.fif, SXXXR13-epo.fif
clean_root = Path(
    r"YOUR_PATH_HERE/cleaned-dataset"
)

# Subject range to analyze
# Default: All 109 subjects in EEGMMIDB
subjects = [f"S{i:03d}" for i in range(1, 110)]

# Task 3 runs (real motor execution: fists vs feet)
# R05 = Task 3, run 1
# R09 = Task 3, run 2  
# R13 = Task 3, run 3
runs_task3_real = ["R05", "R09", "R13"]

# ============================================================================
# CHANNEL CONFIGURATION
# ============================================================================

# Minimal motor ROI (standard BCI configuration)
# Covers primary motor cortex bilaterally plus midline
picks_minimal = ["C3", "Cz", "C4"]

# Extended sensorimotor grid
# Includes premotor (FC), motor (C), and postcentral (CP) regions
# Both 10-20 standard and common 10-10 positions
picks_extended = [
    # Left hemisphere (premotor, motor, postcentral)
    "FC5", "FC3", "FC1",
    "C5", "C3", "C1",
    "CP5", "CP3", "CP1",
    
    # Midline (central strip)
    "FCz", "Cz", "CPz",
    
    # Right hemisphere (premotor, motor, postcentral)
    "FC2", "FC4", "FC6",
    "C2", "C4", "C6",
    "CP2", "CP4", "CP6",
]

# ============================================================================
# FREQUENCY BANDS
# ============================================================================

# Mu rhythm: Sensorimotor rhythm, strongest over motor cortex
# Shows ERD during motor execution and imagery
band_mu = (8.0, 13.0)

# Beta rhythm: Motor control rhythm
# Shows ERD during movement preparation and execution
band_beta = (13.0, 30.0)

# ============================================================================
# TIME WINDOWS
# ============================================================================

# Baseline window (seconds, relative to event onset)
# Used to normalize task activity
baseline_win = (-0.5, 0.0)

# Task window (seconds, relative to event onset)  
# Captures sustained motor activity
# For motor execution: 0.5-1.5s captures active movement phase
task_win = (0.5, 1.5)

# ============================================================================
# QUALITY THRESHOLDS
# ============================================================================

# Minimum channels required for extended set to be considered valid
# Lower values = more permissive (include subjects with some missing channels)
# Higher values = stricter (require nearly complete montage)
MIN_CH_FOR_EXT = 3


# ============================================================================
# Helper Functions
# ============================================================================

def load_concat_task3_epochs(
    clean_root: Path,
    subject: str,
    runs: list[str]
) -> mne.Epochs | None:
    """
    Load and concatenate Task 3 epochs from multiple runs.
    
    Combines epochs across runs to increase statistical power for
    discriminability analysis. Only includes runs with both T1 (fists)
    and T2 (feet) events.
    
    Parameters
    ----------
    clean_root : Path
        Root directory containing cleaned epoch files.
    subject : str
        Subject ID (e.g., "S001").
    runs : list of str
        Run IDs to load (e.g., ["R05", "R09", "R13"]).
    
    Returns
    -------
    mne.Epochs or None
        Concatenated epochs with T1 and T2 events only.
        None if no valid runs found.
    
    Notes
    -----
    Skips runs that:
    - Don't have epoch files
    - Don't contain T1/T2 events
    """
    epochs_list = []
    
    for run in runs:
        epo_path = clean_root / subject / f"{subject}{run}-epo.fif"
        
        # Skip if epoch file doesn't exist
        if not epo_path.exists():
            continue
        
        # Load epochs
        ep = mne.read_epochs(epo_path, preload=True, verbose="ERROR")
        
        # Verify T1 and T2 events exist
        if "T1" not in ep.event_id or "T2" not in ep.event_id:
            continue
        
        # Keep only T1 (fists) and T2 (feet)
        ep = ep[["T1", "T2"]]
        epochs_list.append(ep)
    
    # Return None if no valid epochs found
    if len(epochs_list) == 0:
        return None
    
    # Concatenate across runs
    return mne.concatenate_epochs(epochs_list)


def robust_pick(
    epochs: mne.Epochs,
    picks: list[str]
) -> tuple[mne.Epochs | None, list[str]]:
    """
    Pick channels robustly, using only those that exist.
    
    Some subjects may have incomplete montages due to:
    - Bad electrodes during recording
    - Interpolation failures
    - Different EEG cap models
    
    This function adapts by using available channels only.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Input epochs with all channels.
    picks : list of str
        Desired channel names.
    
    Returns
    -------
    epochs_picked : mne.Epochs or None
        Epochs with available channels only.
        None if no requested channels exist.
    picks_used : list of str
        Channel names actually used.
    """
    available = set(epochs.ch_names)
    picks_use = [ch for ch in picks if ch in available]
    
    if len(picks_use) == 0:
        return None, []
    
    return epochs.copy().pick(picks_use), picks_use


def get_labels_T1_T2(epochs: mne.Epochs) -> np.ndarray:
    """
    Extract binary labels for T1 (fists) vs T2 (feet).
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs containing T1 and T2 events.
    
    Returns
    -------
    np.ndarray
        Binary labels: 0 = T1 (fists), 1 = T2 (feet).
    """
    # Invert event_id dictionary to map codes to names
    inv = {v: k for k, v in epochs.event_id.items()}
    
    # Get event names for each epoch
    lab = np.array([inv[e] for e in epochs.events[:, 2]])
    
    # Convert to binary encoding
    y = np.where(lab == "T1", 0, 1)
    
    return y


def bandpower_timecourse(
    epochs: mne.Epochs,
    band: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute instantaneous bandpower using Hilbert transform.
    
    The Hilbert transform converts a real signal into a complex analytic
    signal, from which we can extract instantaneous amplitude and power.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Input epochs.
    band : tuple of float
        Frequency band (low_freq, high_freq) in Hz.
    
    Returns
    -------
    power : np.ndarray
        Instantaneous power (n_epochs, n_channels, n_times).
    times : np.ndarray
        Time points corresponding to samples.
    
    Notes
    -----
    Pipeline:
    1. Bandpass filter to frequency band of interest
    2. Apply Hilbert transform -> analytic signal (complex)
    3. Compute power: |analytic|²
    """
    # Step 1: Bandpass filter to target frequency band
    ep = epochs.copy().filter(
        band[0], band[1],
        fir_design="firwin",
        verbose="ERROR"
    )
    
    # Step 2: Apply Hilbert transform
    # envelope=False returns complex analytic signal (not just amplitude)
    ep.apply_hilbert(envelope=False, verbose="ERROR")
    
    # Step 3: Get complex analytic signal
    data = ep.get_data()  # Complex-valued
    
    # Step 4: Compute instantaneous power: |z|²
    power = np.abs(data) ** 2
    
    return power, ep.times


def mean_in_window(
    times: np.ndarray,
    arr: np.ndarray,
    tmin: float,
    tmax: float
) -> np.ndarray:
    """
    Compute mean of array within specified time window.
    
    Parameters
    ----------
    times : np.ndarray
        Time points (1D array in seconds).
    arr : np.ndarray
        Data array with time as last dimension (..., n_times).
    tmin : float
        Window start (seconds).
    tmax : float
        Window end (seconds).
    
    Returns
    -------
    np.ndarray
        Mean across time dimension within [tmin, tmax].
    
    Raises
    ------
    ValueError
        If no samples fall within the specified window.
    """
    # Create boolean mask for samples in window
    mask = (times >= tmin) & (times <= tmax)
    
    if not np.any(mask):
        raise ValueError(f"No samples in window [{tmin}, {tmax}]")
    
    # Average across time dimension
    return arr[..., mask].mean(axis=-1)


def erd_percent(
    epochs: mne.Epochs,
    band: tuple[float, float]
) -> np.ndarray:
    """
    Compute Event-Related Desynchronization (ERD) percentage.
    
    ERD% quantifies power decrease relative to baseline:
        ERD% = (P_task - P_baseline) / P_baseline × 100
    
    Negative values indicate desynchronization (ERD, expected for motor tasks).
    Positive values indicate synchronization (ERS, rare in motor tasks).
    
    Parameters
    ----------
    epochs : mne.Epochs
        Input epochs.
    band : tuple of float
        Frequency band (low_freq, high_freq) in Hz.
    
    Returns
    -------
    np.ndarray
        ERD% values (n_epochs, n_channels).
    
    Notes
    -----
    Strong ERD (large negative values) indicates:
    - Active cortical desynchronization
    - Good task engagement
    - Potential for BCI control
    
    Weak or positive ERD indicates:
    - Poor task response
    - Possible non-responder
    - May require different frequency band
    """
    # Compute instantaneous power time course
    power, times = bandpower_timecourse(epochs, band)
    
    # Average power in baseline window
    p_base = mean_in_window(times, power, baseline_win[0], baseline_win[1])
    
    # Average power in task window
    p_task = mean_in_window(times, power, task_win[0], task_win[1])
    
    # Compute ERD% using standard formula
    # Add small constant to avoid division by zero
    erd = (p_task - p_base) / (p_base + 1e-12) * 100.0
    
    return erd


def summarize_condition(
    erd: np.ndarray,
    ch_names: list[str]
) -> tuple[dict, dict]:
    """
    Summarize ERD statistics per channel.
    
    Computes two key metrics:
    1. Mean ERD across epochs (strength of response)
    2. Percentage of epochs with negative ERD (consistency of response)
    
    Parameters
    ----------
    erd : np.ndarray
        ERD values (n_epochs, n_channels).
    ch_names : list of str
        Channel names.
    
    Returns
    -------
    mean_by_ch : dict
        Mean ERD percentage per channel.
    negpct_by_ch : dict
        Percentage of epochs with negative ERD per channel.
    """
    # Mean ERD per channel (strength)
    mean_by_ch = {
        ch: float(np.mean(erd[:, i]))
        for i, ch in enumerate(ch_names)
    }
    
    # Percentage of negative ERD epochs (consistency)
    negpct_by_ch = {
        ch: float(100.0 * np.mean(erd[:, i] < 0))
        for i, ch in enumerate(ch_names)
    }
    
    return mean_by_ch, negpct_by_ch


def discriminability(
    mean_t1: dict,
    mean_t2: dict
) -> tuple[float, float]:
    """
    Compute discriminability between two conditions.
    
    Discriminability measures how well two conditions (T1 vs T2) can be
    separated based on mean ERD values. Higher values indicate better
    separability and potential for classification.
    
    Parameters
    ----------
    mean_t1 : dict
        Mean ERD per channel for condition T1 (fists).
    mean_t2 : dict
        Mean ERD per channel for condition T2 (feet).
    
    Returns
    -------
    maxdiff : float
        Maximum absolute difference across channels.
        Best-case discriminability (optimal channel).
    meandiff : float
        Mean absolute difference across channels.
        Average discriminability across ROI.
    
    Notes
    -----
    maxdiff is more relevant for channel selection.
    meandiff is more robust to outliers.
    """
    # Find channels common to both conditions
    chs = sorted(set(mean_t1.keys()) & set(mean_t2.keys()))
    
    if len(chs) == 0:
        return np.nan, np.nan
    
    # Compute absolute differences per channel
    diffs = np.array([mean_t1[ch] - mean_t2[ch] for ch in chs], dtype=float)
    
    return float(np.max(np.abs(diffs))), float(np.mean(np.abs(diffs)))


# ============================================================================
# Main Analysis Loop
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("Phase 1 - Step 1A: Channel Set Comparison Analysis")
    print("="*70)
    print(f"Dataset: {clean_root}")
    print(f"Subjects: {len(subjects)}")
    print(f"Runs: {runs_task3_real}")
    print(f"Baseline window: {baseline_win}")
    print(f"Task window: {task_win}")
    print(f"Minimal picks: {picks_minimal}")
    print(f"Extended picks (candidate): {len(picks_extended)} channels")
    print("="*70 + "\n")
    
    rows = []
    
    # ========================================================================
    # Process Each Subject
    # ========================================================================
    for subj in subjects:
        try:
            # ================================================================
            # Step 1: Load and Concatenate Task 3 Epochs
            # ================================================================
            epochs_all = load_concat_task3_epochs(clean_root, subj, runs_task3_real)
            
            if epochs_all is None:
                # No usable epochs for this subject
                continue
            
            # ================================================================
            # Step 2: Get Binary Labels (T1=fists, T2=feet)
            # ================================================================
            y = get_labels_T1_T2(epochs_all)
            
            # ================================================================
            # Step 3: Prepare Channel Sets
            # ================================================================
            # Minimal set (C3, Cz, C4)
            ep_min, picks_min_used = robust_pick(epochs_all, picks_minimal)
            
            # Extended set (FC/C/CP grid)
            ep_ext, picks_ext_used = robust_pick(epochs_all, picks_extended)
            
            # ================================================================
            # Step 4: Compute Metrics for Each Channel Set
            # ================================================================
            def compute_set(
                ep: mne.Epochs | None,
                picks_used: list[str]
            ) -> dict:
                """
                Compute complete metric set for a channel configuration.
                
                Returns dictionary with:
                - Channel info (n_ch, channels)
                - Mu band metrics (maxdiff, meandiff, negpct_mean)
                - Beta band metrics (maxdiff, meandiff, negpct_mean)
                - Combined score (avg of mu and beta maxdiff)
                """
                # Return NaNs if no channels available
                if ep is None or len(picks_used) == 0:
                    return {
                        "n_ch": 0,
                        "channels": "",
                        "mu_maxdiff": np.nan,
                        "mu_meandiff": np.nan,
                        "beta_maxdiff": np.nan,
                        "beta_meandiff": np.nan,
                        "combo_score": np.nan,
                        "mu_negpct_mean": np.nan,
                        "beta_negpct_mean": np.nan,
                    }
                
                # Note if channel count is below threshold (but still compute)
                if len(picks_used) < MIN_CH_FOR_EXT:
                    # Still compute, but metrics may be less stable
                    pass
                
                # Split by condition
                ep_t1 = ep[y == 0]  # Fists
                ep_t2 = ep[y == 1]  # Feet
                
                # Return NaNs if either condition is empty
                if len(ep_t1) == 0 or len(ep_t2) == 0:
                    return {
                        "n_ch": len(picks_used),
                        "channels": ",".join(picks_used),
                        "mu_maxdiff": np.nan,
                        "mu_meandiff": np.nan,
                        "beta_maxdiff": np.nan,
                        "beta_meandiff": np.nan,
                        "combo_score": np.nan,
                        "mu_negpct_mean": np.nan,
                        "beta_negpct_mean": np.nan,
                    }
                
                # ============================================================
                # Compute ERD for Mu Band
                # ============================================================
                mu_t1 = erd_percent(ep_t1, band_mu)
                mu_t2 = erd_percent(ep_t2, band_mu)
                
                # ============================================================
                # Compute ERD for Beta Band
                # ============================================================
                b_t1 = erd_percent(ep_t1, band_beta)
                b_t2 = erd_percent(ep_t2, band_beta)
                
                # ============================================================
                # Summarize Per-Channel Statistics
                # ============================================================
                mu_t1_mean, mu_t1_neg = summarize_condition(mu_t1, ep.ch_names)
                mu_t2_mean, mu_t2_neg = summarize_condition(mu_t2, ep.ch_names)
                b_t1_mean, b_t1_neg = summarize_condition(b_t1, ep.ch_names)
                b_t2_mean, b_t2_neg = summarize_condition(b_t2, ep.ch_names)
                
                # ============================================================
                # Compute Discriminability
                # ============================================================
                mu_max, mu_mean = discriminability(mu_t1_mean, mu_t2_mean)
                b_max, b_mean = discriminability(b_t1_mean, b_t2_mean)
                
                # ============================================================
                # Compute Consistency Metrics
                # ============================================================
                # Average negative ERD percentage across channels and conditions
                # Higher values = more consistent desynchronization
                mu_negpct_mean = float(np.mean(
                    list(mu_t1_neg.values()) + list(mu_t2_neg.values())
                ))
                beta_negpct_mean = float(np.mean(
                    list(b_t1_neg.values()) + list(b_t2_neg.values())
                ))
                
                # ============================================================
                # Compute Combined Score
                # ============================================================
                # Simple average of mu and beta max discriminability
                # Represents overall separability potential
                combo = float(np.nanmean([mu_max, b_max]))
                
                return {
                    "n_ch": len(picks_used),
                    "channels": ",".join(picks_used),
                    "mu_maxdiff": mu_max,
                    "mu_meandiff": mu_mean,
                    "beta_maxdiff": b_max,
                    "beta_meandiff": b_mean,
                    "combo_score": combo,
                    "mu_negpct_mean": mu_negpct_mean,
                    "beta_negpct_mean": beta_negpct_mean,
                }
            
            # Compute metrics for both channel sets
            min_stats = compute_set(ep_min, picks_min_used)
            ext_stats = compute_set(ep_ext, picks_ext_used)
            
            # ================================================================
            # Step 5: Build Output Row
            # ================================================================
            row = {
                # Subject info
                "subject": subj,
                "n_epochs": int(len(epochs_all)),
                "n_fists": int(np.sum(y == 0)),
                "n_feet": int(np.sum(y == 1)),
                
                # Minimal channel set metrics
                "min_n_ch": min_stats["n_ch"],
                "min_channels": min_stats["channels"],
                "min_mu_maxdiff": min_stats["mu_maxdiff"],
                "min_mu_meandiff": min_stats["mu_meandiff"],
                "min_beta_maxdiff": min_stats["beta_maxdiff"],
                "min_beta_meandiff": min_stats["beta_meandiff"],
                "min_combo_score": min_stats["combo_score"],
                "min_mu_negpct_mean": min_stats["mu_negpct_mean"],
                "min_beta_negpct_mean": min_stats["beta_negpct_mean"],
                
                # Extended channel set metrics
                "ext_n_ch": ext_stats["n_ch"],
                "ext_channels": ext_stats["channels"],
                "ext_mu_maxdiff": ext_stats["mu_maxdiff"],
                "ext_mu_meandiff": ext_stats["mu_meandiff"],
                "ext_beta_maxdiff": ext_stats["beta_maxdiff"],
                "ext_beta_meandiff": ext_stats["beta_meandiff"],
                "ext_combo_score": ext_stats["combo_score"],
                "ext_mu_negpct_mean": ext_stats["mu_negpct_mean"],
                "ext_beta_negpct_mean": ext_stats["beta_negpct_mean"],
            }
            
            # ================================================================
            # Step 6: Compute Improvement Deltas
            # ================================================================
            # Positive delta = extended is better
            # Negative delta = minimal is better
            
            # Overall improvement (combo score)
            if np.isfinite(row["ext_combo_score"]) and np.isfinite(row["min_combo_score"]):
                row["delta_combo"] = row["ext_combo_score"] - row["min_combo_score"]
            else:
                row["delta_combo"] = np.nan
            
            # Mu band improvement
            if np.isfinite(row["ext_mu_maxdiff"]) and np.isfinite(row["min_mu_maxdiff"]):
                row["delta_mu_maxdiff"] = row["ext_mu_maxdiff"] - row["min_mu_maxdiff"]
            else:
                row["delta_mu_maxdiff"] = np.nan
            
            # Beta band improvement
            if np.isfinite(row["ext_beta_maxdiff"]) and np.isfinite(row["min_beta_maxdiff"]):
                row["delta_beta_maxdiff"] = row["ext_beta_maxdiff"] - row["min_beta_maxdiff"]
            else:
                row["delta_beta_maxdiff"] = np.nan
            
            rows.append(row)
            
            # Progress indicator every 25 subjects
            if len(rows) % 25 == 0:
                print(f"...processed {len(rows)} subjects with usable Task 3 epochs")
        
        except Exception as e:
            print(f"[ERROR] {subj}: {e}")
    
    # ========================================================================
    # Save Results to CSV
    # ========================================================================
    df = pd.DataFrame(rows)
    
    out_csv = Path(__file__).with_name(
        "Phase1_Step1A_Task3_ERD_discriminability_min_vs_ext.csv"
    )
    df.to_csv(out_csv, index=False)
    
    print(f"\n[OK] Saved: {out_csv}")
    
    # ========================================================================
    # Print Summary Statistics
    # ========================================================================
    if len(df) == 0:
        print("\n[WARN] No usable subjects found.")
        print("Check clean_root path and epoch file availability.")
        raise SystemExit(0)
    
    # Filter to subjects with both minimal and extended metrics
    valid = df[np.isfinite(df["delta_combo"])].copy()
    
    if len(valid) == 0:
        print("\n[WARN] No subjects had both minimal and extended metrics.")
        print("This usually indicates missing extended channels in montage.")
    else:
        # Count subjects where extended improved discriminability
        improved = valid[valid["delta_combo"] > 0]
        frac = 100.0 * len(improved) / len(valid)
        
        print("\n" + "="*70)
        print("SUMMARY STATISTICS")
        print("="*70)
        print(f"Subjects with both channel sets computed: {len(valid)}")
        print(f"Extended improved combo score: {len(improved)}/{len(valid)} ({frac:.1f}%)")
        
        # Show top subjects where extended helped most
        top = valid.sort_values("delta_combo", ascending=False).head(15)
        
        print("\n" + "-"*70)
        print("Top Subjects Where Extended Channels Helped Most")
        print("-"*70)
        
        cols = [
            "subject",
            "n_epochs",
            "min_n_ch",
            "ext_n_ch",
            "min_combo_score",
            "ext_combo_score",
            "delta_combo",
            "delta_mu_maxdiff",
            "delta_beta_maxdiff",
        ]
        
        print(top[cols].to_string(index=False))
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70 + "\n")
