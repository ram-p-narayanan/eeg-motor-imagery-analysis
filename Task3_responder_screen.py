"""
Task 3 Responder Screening: Dual-Band ERD Analysis

This script identifies "dual responders" - subjects showing strong Event-Related
Desynchronization (ERD) in both mu (8-13 Hz) and beta (13-30 Hz) frequency bands
during motor execution.

Task 3 (Real Movement):
- T1: Both fists clenching
- T2: Both feet movement

Dual Responder Criteria:
- At least one channel shows mean ERD ≤ -10% in mu band
- At least one channel shows ≥60% negative ERD epochs in mu band
- Same criteria met for beta band

Output:
- CSV file with subject rankings and metrics
- Dual responder identification
- Per-channel ERD statistics

Author: Ram P Narayanan
Date: 2026-02-06
Version: 1.0.0
License: MIT

Dependencies:
    - mne >= 1.5.0
    - numpy >= 1.24.0
    - pandas >= 2.0.0

Usage:
    1. Edit clean_root path below
    2. Optionally adjust thresholds (TH_MEAN, TH_NEGPCT)
    3. Run: python Task3_responder_screen.py
    4. Review output CSV: task3_responder_screen_S001_S109.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import mne


# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to cleaned dataset (output from preprocessing pipeline)
# Structure: clean_root/S001/S001R05-epo.fif, S001R09-epo.fif, etc.
clean_root = Path(
    r"YOUR_PATH_HERE/cleaned-dataset"
)

# Subject range to screen
# Default: S001-S109 (all subjects in EEGMMIDB)
subjects = [f"S{i:03d}" for i in range(1, 110)]

# Task 3 runs (real motor execution)
# R05 = Task 3, run 1
# R09 = Task 3, run 2
# R13 = Task 3, run 3
runs_task3_real = ["R05", "R09", "R13"]

# Channel selection (motor cortex region)
# Start with central channels (C3, Cz, C4)
# Optionally add: C1, C2, C5, C6, T7, T8
picks = ["C3", "Cz", "C4", "C1", "C2"]

# Frequency bands for ERD analysis
band_mu = (8.0, 13.0)       # Mu rhythm (sensorimotor)
band_beta = (13.0, 30.0)    # Beta rhythm (motor control)

# Time windows (in seconds)
baseline_win = (-0.5, 0.0)  # Pre-stimulus baseline
task_win = (0.5, 1.5)       # Active movement window

# ============================================================================
# RESPONDER THRESHOLDS
# ============================================================================
# These determine what constitutes a "strong responder"
# Adjust based on your specific requirements

TH_MEAN = -10.0     # Mean ERD must be ≤ -10% (more negative = stronger ERD)
TH_NEGPCT = 60.0    # At least 60% of epochs must show negative ERD (desynchronization)

# Note: More stringent thresholds (e.g., TH_MEAN = -15%, TH_NEGPCT = 70%)
# will identify fewer but more reliable responders


# ============================================================================
# Helper Functions
# ============================================================================

def load_concat_epochs_task3(
    clean_root: Path,
    subject: str,
    runs: list
) -> mne.Epochs | None:
    """
    Load and concatenate Task 3 epochs from multiple runs.
    
    Parameters
    ----------
    clean_root : Path
        Root directory containing cleaned epochs.
    subject : str
        Subject ID (e.g., "S001").
    runs : list of str
        Run IDs to concatenate (e.g., ["R05", "R09", "R13"]).
    
    Returns
    -------
    mne.Epochs or None
        Concatenated epochs with T1 and T2 events, or None if no valid runs.
    
    Notes
    -----
    Only includes runs that:
    1. Have existing epoch files
    2. Contain both T1 and T2 events
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
        
        # Keep only T1 (fists) and T2 (feet) epochs
        ep = ep[["T1", "T2"]]
        epochs_list.append(ep)
    
    # Return None if no valid epochs found
    if len(epochs_list) == 0:
        return None
    
    # Concatenate across runs
    return mne.concatenate_epochs(epochs_list)


def robust_pick(epochs: mne.Epochs, picks: list) -> mne.Epochs:
    """
    Pick channels robustly, using only those that exist.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Input epochs.
    picks : list of str
        Desired channel names.
    
    Returns
    -------
    mne.Epochs
        Epochs with available channels only.
    
    Raises
    ------
    RuntimeError
        If none of the requested channels exist.
    """
    available = set(epochs.ch_names)
    picks_use = [ch for ch in picks if ch in available]
    
    if len(picks_use) == 0:
        raise RuntimeError("None of the requested channels are present.")
    
    return epochs.copy().pick(picks_use)


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
    # Invert event_id mapping to get event names from codes
    inv = {v: k for k, v in epochs.event_id.items()}
    
    # Get event names for each epoch
    lab = np.array([inv[e] for e in epochs.events[:, 2]])
    
    # Convert to binary: 0=fists, 1=feet
    y = np.where(lab == "T1", 0, 1)
    
    return y


def bandpower_timecourse(
    epochs: mne.Epochs,
    band: tuple
) -> tuple:
    """
    Compute bandpower time course using Hilbert transform.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Input epochs.
    band : tuple of float
        Frequency band (low, high) in Hz.
    
    Returns
    -------
    power : np.ndarray
        Instantaneous power (n_epochs, n_channels, n_times).
    times : np.ndarray
        Time points corresponding to samples.
    
    Notes
    -----
    Uses Hilbert transform to compute analytic signal, then |analytic|² for power.
    """
    # Bandpass filter
    ep = epochs.copy().filter(band[0], band[1], fir_design="firwin", verbose="ERROR")
    
    # Apply Hilbert transform to get analytic signal
    ep.apply_hilbert(envelope=False, verbose="ERROR")
    
    # Get complex analytic signal
    data = ep.get_data()
    
    # Compute instantaneous power: |analytic|²
    power = np.abs(data) ** 2
    
    return power, ep.times


def mean_in_window(
    times: np.ndarray,
    arr: np.ndarray,
    tmin: float,
    tmax: float
) -> np.ndarray:
    """
    Compute mean of array within time window.
    
    Parameters
    ----------
    times : np.ndarray
        Time points (1D array).
    arr : np.ndarray
        Data array (..., n_times).
    tmin : float
        Window start (seconds).
    tmax : float
        Window end (seconds).
    
    Returns
    -------
    np.ndarray
        Mean across time dimension within window.
    
    Raises
    ------
    ValueError
        If no samples fall within the window.
    """
    mask = (times >= tmin) & (times <= tmax)
    
    if not np.any(mask):
        raise ValueError(f"No samples in window [{tmin}, {tmax}]")
    
    return arr[..., mask].mean(axis=-1)


def erd_epoch_channel(
    epochs: mne.Epochs,
    band: tuple,
    baseline_win: tuple,
    task_win: tuple
) -> np.ndarray:
    """
    Compute ERD% for each epoch and channel.
    
    ERD% = (P_task - P_baseline) / P_baseline × 100
    
    Negative values indicate Event-Related Desynchronization (ERD).
    Positive values indicate Event-Related Synchronization (ERS).
    
    Parameters
    ----------
    epochs : mne.Epochs
        Input epochs.
    band : tuple of float
        Frequency band (low, high) in Hz.
    baseline_win : tuple of float
        Baseline window (tmin, tmax) in seconds.
    task_win : tuple of float
        Task window (tmin, tmax) in seconds.
    
    Returns
    -------
    np.ndarray
        ERD% values (n_epochs, n_channels).
    """
    # Compute instantaneous power
    power, times = bandpower_timecourse(epochs, band)
    
    # Average power in baseline window
    p_base = mean_in_window(times, power, baseline_win[0], baseline_win[1])
    
    # Average power in task window
    p_task = mean_in_window(times, power, task_win[0], task_win[1])
    
    # Compute ERD%
    erd = (p_task - p_base) / (p_base + 1e-12) * 100.0
    
    return erd


def summarize_band(erd: np.ndarray, ch_names: list) -> tuple:
    """
    Summarize ERD statistics per channel.
    
    Parameters
    ----------
    erd : np.ndarray
        ERD values (n_epochs, n_channels).
    ch_names : list of str
        Channel names.
    
    Returns
    -------
    mean_by_ch : dict
        Mean ERD per channel.
    negpct_by_ch : dict
        Percentage of epochs with negative ERD per channel.
    """
    mean_by_ch = {
        ch: float(np.mean(erd[:, i]))
        for i, ch in enumerate(ch_names)
    }
    
    negpct_by_ch = {
        ch: float(100.0 * np.mean(erd[:, i] < 0))
        for i, ch in enumerate(ch_names)
    }
    
    return mean_by_ch, negpct_by_ch


def best_responder_score(mean_by_ch: dict, negpct_by_ch: dict) -> tuple:
    """
    Compute best responder score across channels.
    
    Score = max(0, -mean_ERD) × (negpct / 100)
    
    Higher scores indicate stronger, more consistent ERD.
    
    Parameters
    ----------
    mean_by_ch : dict
        Mean ERD per channel.
    negpct_by_ch : dict
        Negative ERD percentage per channel.
    
    Returns
    -------
    score : float
        Best responder score.
    channel : str
        Channel with best score.
    """
    best = (-np.inf, None)
    
    for ch in mean_by_ch.keys():
        mean_erd = mean_by_ch[ch]
        negpct = negpct_by_ch[ch]
        
        # Score: magnitude of negative ERD × consistency
        # max(0, -mean) ensures only ERD (not ERS) contributes
        score = max(0.0, -mean_erd) * (negpct / 100.0)
        
        if score > best[0]:
            best = (score, ch)
    
    return float(best[0]), best[1]


def meets_threshold(mean_by_ch: dict, negpct_by_ch: dict) -> tuple:
    """
    Check if any channel meets responder thresholds.
    
    Parameters
    ----------
    mean_by_ch : dict
        Mean ERD per channel.
    negpct_by_ch : dict
        Negative ERD percentage per channel.
    
    Returns
    -------
    meets : bool
        True if any channel meets both thresholds.
    channel : str or None
        First channel meeting thresholds, or None.
    """
    for ch in mean_by_ch.keys():
        if mean_by_ch[ch] <= TH_MEAN and negpct_by_ch[ch] >= TH_NEGPCT:
            return True, ch
    
    return False, None


# ============================================================================
# Main Screening Loop
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("Task 3 Responder Screening: Dual-Band ERD Analysis")
    print("="*70)
    print(f"Dataset: {clean_root}")
    print(f"Subjects: {len(subjects)}")
    print(f"Runs: {runs_task3_real}")
    print(f"Channels: {picks}")
    print(f"Mu band: {band_mu[0]}-{band_mu[1]} Hz")
    print(f"Beta band: {band_beta[0]}-{band_beta[1]} Hz")
    print(f"Baseline window: {baseline_win}")
    print(f"Task window: {task_win}")
    print(f"Thresholds: mean ERD ≤ {TH_MEAN}%, negative % ≥ {TH_NEGPCT}%")
    print("="*70 + "\n")
    
    rows = []
    
    # ========================================================================
    # Process Each Subject
    # ========================================================================
    for subj in subjects:
        try:
            # ================================================================
            # Load and Concatenate Task 3 Epochs
            # ================================================================
            epochs = load_concat_epochs_task3(clean_root, subj, runs_task3_real)
            
            if epochs is None:
                print(f"[SKIP] {subj}: No usable epochs found.")
                continue
            
            # ================================================================
            # Pick Motor Channels
            # ================================================================
            epochs = robust_pick(epochs, picks)
            ch_names = epochs.ch_names
            
            # ================================================================
            # Get Binary Labels (T1=fists, T2=feet)
            # ================================================================
            y = get_labels_T1_T2(epochs)
            
            ep_fists = epochs[y == 0]  # T1
            ep_feet = epochs[y == 1]   # T2
            
            # ================================================================
            # Compute ERD% for Each Band and Condition
            # ================================================================
            # Mu band
            mu_f = erd_epoch_channel(ep_fists, band_mu, baseline_win, task_win)
            mu_t = erd_epoch_channel(ep_feet, band_mu, baseline_win, task_win)
            
            # Beta band
            b_f = erd_epoch_channel(ep_fists, band_beta, baseline_win, task_win)
            b_t = erd_epoch_channel(ep_feet, band_beta, baseline_win, task_win)
            
            # ================================================================
            # Summarize Per-Channel Statistics
            # ================================================================
            mu_f_mean, mu_f_neg = summarize_band(mu_f, ch_names)
            mu_t_mean, mu_t_neg = summarize_band(mu_t, ch_names)
            b_f_mean, b_f_neg = summarize_band(b_f, ch_names)
            b_t_mean, b_t_neg = summarize_band(b_t, ch_names)
            
            # ================================================================
            # Combine Across Conditions (Best Evidence)
            # ================================================================
            # For each channel, take the more negative mean (stronger ERD)
            mu_mean_comb = {ch: min(mu_f_mean[ch], mu_t_mean[ch]) for ch in ch_names}
            b_mean_comb = {ch: min(b_f_mean[ch], b_t_mean[ch]) for ch in ch_names}
            
            # For each channel, take the higher negative percentage (more consistent)
            mu_neg_comb = {ch: max(mu_f_neg[ch], mu_t_neg[ch]) for ch in ch_names}
            b_neg_comb = {ch: max(b_f_neg[ch], b_t_neg[ch]) for ch in ch_names}
            
            # ================================================================
            # Check Thresholds
            # ================================================================
            mu_ok, mu_ch = meets_threshold(mu_mean_comb, mu_neg_comb)
            b_ok, b_ch = meets_threshold(b_mean_comb, b_neg_comb)
            
            # ================================================================
            # Compute Best Responder Scores
            # ================================================================
            mu_score, mu_best_ch = best_responder_score(mu_mean_comb, mu_neg_comb)
            b_score, b_best_ch = best_responder_score(b_mean_comb, b_neg_comb)
            
            # ================================================================
            # Determine Dual Responder Status
            # ================================================================
            dual = bool(mu_ok and b_ok)
            total_score = mu_score + b_score
            
            # ================================================================
            # Build Output Row
            # ================================================================
            row = {
                "subject": subj,
                "n_epochs_total": int(len(epochs)),
                "n_fists": int(len(ep_fists)),
                "n_feet": int(len(ep_feet)),
                "channels_used": ",".join(ch_names),
                
                # Mu band results
                "mu_best_channel": mu_best_ch,
                "mu_best_score": mu_score,
                "mu_meets_threshold": mu_ok,
                "mu_threshold_channel": mu_ch if mu_ok else "",
                
                # Beta band results
                "beta_best_channel": b_best_ch,
                "beta_best_score": b_score,
                "beta_meets_threshold": b_ok,
                "beta_threshold_channel": b_ch if b_ok else "",
                
                # Dual responder status
                "dual_responder": dual,
                "dual_total_score": total_score,
            }
            
            # ================================================================
            # Add Per-Channel Metrics for Standard Channels
            # ================================================================
            for ch in ["C3", "Cz", "C4"]:
                if ch in ch_names:
                    row[f"mu_mean_best_{ch}"] = mu_mean_comb[ch]
                    row[f"mu_negpct_best_{ch}"] = mu_neg_comb[ch]
                    row[f"beta_mean_best_{ch}"] = b_mean_comb[ch]
                    row[f"beta_negpct_best_{ch}"] = b_neg_comb[ch]
                else:
                    row[f"mu_mean_best_{ch}"] = np.nan
                    row[f"mu_negpct_best_{ch}"] = np.nan
                    row[f"beta_mean_best_{ch}"] = np.nan
                    row[f"beta_negpct_best_{ch}"] = np.nan
            
            rows.append(row)
            
            print(f"[OK] {subj}: epochs={len(epochs)}, "
                  f"fists={len(ep_fists)}, feet={len(ep_feet)} | "
                  f"dual={dual} | score={total_score:.3f}")
        
        except Exception as e:
            print(f"[ERROR] {subj}: {e}")
    
    # ========================================================================
    # Save Results to CSV
    # ========================================================================
    df = pd.DataFrame(rows)
    out_csv = Path(__file__).with_name("task3_responder_screen_S001_S109.csv")
    df.to_csv(out_csv, index=False)
    
    print(f"\n[OK] Saved: {out_csv}")
    
    # ========================================================================
    # Print Summary
    # ========================================================================
    if len(df) == 0:
        print("\n[WARN] No subjects processed. Check paths and epoch files.")
    else:
        # Rank by dual responder status and total score
        df_rank = df.sort_values(
            by=["dual_responder", "dual_total_score"],
            ascending=[False, False]
        ).reset_index(drop=True)
        
        print("\n" + "="*70)
        print("SUMMARY STATISTICS")
        print("="*70)
        print(f"Total subjects processed: {len(df)}")
        
        dual_df = df_rank[df_rank["dual_responder"] == True]
        print(f"Dual responders: {len(dual_df)} ({100*len(dual_df)/len(df):.1f}%)")
        
        print("\n" + "-"*70)
        print("Top 10 Subjects (Ranked by Dual Responder Score)")
        print("-"*70)
        
        show_cols = [
            "subject",
            "dual_responder",
            "dual_total_score",
            "mu_best_channel",
            "mu_best_score",
            "beta_best_channel",
            "beta_best_score",
            "n_epochs_total",
        ]
        
        print(df_rank[show_cols].head(10).to_string(index=False))
        
        if len(dual_df) > 0:
            print("\n" + "-"*70)
            print("Dual Responders (meeting thresholds in both mu and beta):")
            print("-"*70)
            print(", ".join(dual_df["subject"].head(20).tolist()))
        else:
            print("\n[INFO] No dual responders found with current thresholds.")
            print(f"       Consider relaxing: TH_MEAN (currently {TH_MEAN}%)")
            print(f"                          TH_NEGPCT (currently {TH_NEGPCT}%)")
        
        print("\n" + "="*70)
        print("Screening complete!")
        print("="*70 + "\n")
