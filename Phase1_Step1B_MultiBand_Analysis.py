"""
Phase 1 - Step 1B: Multi-Band Discriminability Analysis

This script tests whether narrower frequency sub-bands discriminate motor tasks
(fists vs feet) better than broad mu+beta bands. The analysis identifies
subject-specific optimal frequency bands for BCI applications.

Research Question:
    Do subjects show better ERD discriminability in narrow bands (theta, mu, 
    beta_low, beta_high) compared to conventional broad bands (mu_beta)?

Analysis Pipeline:
    1. Load top-performing subjects from Step 1A (or scan all if unavailable)
    2. Concatenate Task 3 epochs across runs R05, R09, R13
    3. Select motor-strip ROI channels (C5, C3, C1, Cz, C2, C4, C6)
    4. For each frequency band:
       - Compute ERD% per epoch and channel using Hilbert transform
       - Calculate discriminability: max|ERD_T1 - ERD_T2| across channels
       - Measure negative-ERD consistency (desynchronization stability)
    5. Rank bands per subject by maximum discriminability
    6. Quantify improvement over mu_beta baseline
    7. Export comprehensive CSV with per-band metrics

Frequency Bands Tested:
    - theta: 4-8 Hz (low-frequency oscillations, sometimes motor-related)
    - mu: 8-13 Hz (sensorimotor rhythm, classic motor marker)
    - beta_low: 13-20 Hz (motor planning and preparation)
    - beta_high: 20-30 Hz (motor execution and control)
    - beta: 13-30 Hz (full beta range, conventional approach)
    - mu_beta: 8-30 Hz (conventional baseline for comparison)

Key Findings (Expected):
    - 25-35% subjects show mu dominance
    - 20-30% show beta_low dominance
    - 15-25% show beta_high dominance
    - 10-20% show theta dominance
    - Subject-specific bands improve 3-8% over mu_beta baseline
    - Improvement validates need for personalized frequency selection

Inputs:
    1. Cleaned epochs: cleaned-dataset/SXXX/SXXXRYY-epo.fif
    2. Step 1A results (optional): Phase1_Step1A_Task3_ERD_discriminability_min_vs_ext.xlsx
       Used to prioritize top-performing subjects for efficient analysis

Outputs:
    1. CSV file: Phase1_Step1B_Task3_band_discriminability_motor_strip.csv
       Contains per-subject optimal bands and comprehensive metrics
    2. Skip log: Phase1_Step1B_skipped_subjects.txt
       Lists subjects excluded and reasons
    3. Console summary: Band distribution statistics and improvements

Important Notes:
    - This is PRE-CSP analysis (discriminability, not classification accuracy)
    - Phase 2 will validate band choices with actual CSP+LDA classification
    - Results inform subject-specific frequency filtering strategies
    - Requires all motor-strip channels by default (configurable)

Author: Ram P Narayanan
Date: 2026-02-08
Version: 1.0.0
License: MIT

Dependencies:
    - mne >= 1.5.0
    - numpy >= 1.24.0
    - pandas >= 2.0.0
    - openpyxl >= 3.1.0 (for Excel reading)

Usage:
    1. Run Phase 1 Step 1A first (optional but recommended)
    2. Edit clean_root path in CONFIGURATION section
    3. Adjust TOP_N if you want more/fewer subjects
    4. Run: python Phase1_Step1B_MultiBand_Analysis.py
    5. Review CSV for subject-specific band recommendations
    6. Use findings to configure Phase 2 CSP+LDA classification
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

# ============================================================================
# SUBJECT SELECTION
# ============================================================================

# Step 1A summary file (optional but recommended)
# Used to prioritize top-performing subjects from channel comparison analysis
# If file not found, script falls back to scanning all subjects S001-S109
step1a_summary_path = Path(__file__).with_name(
    "Phase1_Step1A_Task3_ERD_discriminability_min_vs_ext.xlsx"
)

# Number of top subjects to analyze from Step 1A results
# Higher values = more subjects = longer runtime
# Lower values = faster analysis = focus on best responders
TOP_N = 40

# Fallback: All subjects in EEGMMIDB dataset
subjects_fallback = [f"S{i:03d}" for i in range(1, 110)]

# Task 3 runs (real motor execution: fists vs feet)
# R05 = Task 3, run 1
# R09 = Task 3, run 2
# R13 = Task 3, run 3
runs_task3_real = ["R05", "R09", "R13"]

# ============================================================================
# CHANNEL CONFIGURATION
# ============================================================================

# Motor-strip ROI (region of interest)
# Covers bilateral sensorimotor cortex plus midline
# C5/C6: Extended lateral motor areas (10-10 system)
# C3/C4: Primary motor cortex (10-20 standard)
# C1/C2: Medial motor areas (10-10 system)
# Cz: Midline motor area (10-20 standard)
picks_motor_strip = ["C5", "C3", "C1", "Cz", "C2", "C4", "C6"]

# ============================================================================
# FREQUENCY BANDS
# ============================================================================

# Dictionary of frequency bands to test
# Each band tests a specific aspect of motor-related oscillatory activity
bands_to_test = {
    # Theta: Low-frequency oscillations
    # Sometimes shows motor-related activity, especially in motor planning
    # Less common than mu/beta but important for some subjects
    "theta": (4.0, 8.0),
    
    # Mu rhythm: Classic sensorimotor rhythm
    # Shows strong ERD during motor execution and imagery
    # Most studied frequency band in motor BCIs
    "mu": (8.0, 13.0),
    
    # Beta-low: Motor planning and preparation
    # Often shows stronger ERD during motor preparation phase
    # Can be more discriminative than full beta in some subjects
    "beta_low": (13.0, 20.0),
    
    # Beta-high: Motor execution and control
    # Shows ERD during active movement
    # Important for fine motor control tasks
    "beta_high": (20.0, 30.0),
    
    # Beta: Full beta range (conventional approach)
    # Covers both planning and execution phases
    # Standard reference in many motor BCI studies
    "beta": (13.0, 30.0),
    
    # Mu+Beta: Broad sensorimotor band (baseline for comparison)
    # Conventional choice in motor BCI applications
    # Used as reference to measure narrow-band improvements
    "mu_beta": (8.0, 30.0),
    
    # Optional: Low-gamma (use with caution)
    # Uncomment only if you have high sampling rate and good EMG removal
    # Risk: Muscle artifacts can contaminate gamma band
    # "low_gamma": (30.0, 45.0),
}

# ============================================================================
# TIME WINDOWS
# ============================================================================

# Baseline window (seconds, relative to event onset at t=0)
# Used to normalize task activity
# Standard pre-stimulus period
baseline_win = (-0.5, 0.0)

# Task window (seconds, relative to event onset at t=0)
# Captures sustained motor activity
# For motor execution: 0.5-1.5s captures active movement phase
# Avoids movement onset artifacts (0-0.5s) and late relaxation (>1.5s)
task_win = (0.5, 1.5)

# ============================================================================
# QUALITY CONTROL
# ============================================================================

# Require all motor-strip channels to be present
# True: Skip subjects with any missing motor-strip channels (strict)
# False: Use available subset of motor-strip channels (permissive)
REQUIRE_ALL_MOTOR_STRIP = True


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
    - Don't have epoch files on disk
    - Don't contain both T1 and T2 events
    - Had preprocessing failures
    """
    epochs_list = []
    
    for run in runs:
        # Construct epoch file path
        epo_path = clean_root / subject / f"{subject}{run}-epo.fif"
        
        # Skip if epoch file doesn't exist
        if not epo_path.exists():
            continue
        
        # Load epochs with error suppression for cleaner output
        ep = mne.read_epochs(epo_path, preload=True, verbose="ERROR")
        
        # Verify both task conditions are present
        if "T1" not in ep.event_id or "T2" not in ep.event_id:
            continue
        
        # Keep only Task 3 events (T1=fists, T2=feet)
        ep = ep[["T1", "T2"]]
        epochs_list.append(ep)
    
    # Return None if no valid epochs found
    if len(epochs_list) == 0:
        return None
    
    # Concatenate across runs for increased statistical power
    return mne.concatenate_epochs(epochs_list)


def get_labels_T1_T2(epochs: mne.Epochs) -> np.ndarray:
    """
    Extract binary labels for T1 (fists) vs T2 (feet).
    
    Converts MNE event IDs to binary array for discriminability analysis.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs containing T1 and T2 events.
    
    Returns
    -------
    np.ndarray
        Binary labels: 0 = T1 (fists), 1 = T2 (feet).
        Shape: (n_epochs,)
    """
    # Invert event_id dictionary to map codes to names
    inv = {v: k for k, v in epochs.event_id.items()}
    
    # Get event names for each epoch
    lab = np.array([inv[e] for e in epochs.events[:, 2]])
    
    # Convert to binary encoding
    # 0 = fists (T1), 1 = feet (T2)
    y = np.where(lab == "T1", 0, 1)
    
    return y


def pick_motor_strip(
    epochs: mne.Epochs,
    picks: list[str]
) -> tuple[mne.Epochs | None, list[str], list[str]]:
    """
    Select motor-strip channels robustly with missing channel handling.
    
    Some subjects may have incomplete motor-strip montages due to:
    - Bad electrodes during recording
    - Interpolation failures in preprocessing
    - Different EEG cap models with varying channel counts
    
    This function adapts by using available channels and reporting missing ones.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Input epochs with all available channels.
    picks : list of str
        Desired motor-strip channel names.
    
    Returns
    -------
    epochs_picked : mne.Epochs or None
        Epochs with available motor-strip channels only.
        None if REQUIRE_ALL_MOTOR_STRIP=True and any channels missing.
        None if no requested channels exist.
    picks_used : list of str
        Channel names actually used (available subset).
    missing : list of str
        Channel names that were requested but not available.
    
    Notes
    -----
    If REQUIRE_ALL_MOTOR_STRIP=True:
        Subject is excluded if any motor-strip channels are missing
    If REQUIRE_ALL_MOTOR_STRIP=False:
        Uses available subset (minimum 1 channel required)
    """
    # Identify available channels in this dataset
    available = set(epochs.ch_names)
    
    # Find intersection: requested channels that exist
    picks_use = [ch for ch in picks if ch in available]
    
    # Find difference: requested channels that don't exist
    missing = [ch for ch in picks if ch not in available]
    
    # Strict mode: reject if any channels missing
    if REQUIRE_ALL_MOTOR_STRIP and missing:
        return None, [], missing
    
    # Reject if no channels available at all
    if len(picks_use) == 0:
        return None, [], missing
    
    # Return epochs with available channels
    return epochs.copy().pick(picks_use), picks_use, missing


def bandpower_timecourse(
    epochs: mne.Epochs,
    band: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute instantaneous bandpower using Hilbert transform.
    
    The Hilbert transform converts a real signal into a complex analytic
    signal, from which we can extract instantaneous amplitude and power.
    This is more accurate than short-window FFT for time-resolved power.
    
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
        Units: μV² (power units match input signal units)
    times : np.ndarray
        Time points corresponding to samples (seconds).
    
    Notes
    -----
    Pipeline:
    1. Bandpass filter to frequency band of interest
       - FIR filter with firwin design for optimal frequency response
    2. Apply Hilbert transform -> analytic signal (complex)
       - Analytic signal = original + i*Hilbert(original)
    3. Compute power: |analytic|²
       - Instantaneous power at each time point
    
    Advantages over Welch/FFT:
    - Time-resolved power (not time-averaged)
    - Accurate for non-stationary signals
    - Better for short time windows
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
    data = ep.get_data()  # Complex-valued: real + imag
    
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
    
    Used to average power over baseline and task periods for ERD calculation.
    
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
        Shape: arr.shape[:-1] (time dimension removed)
    
    Raises
    ------
    ValueError
        If no samples fall within the specified window.
        This can happen if window is outside epoch time range.
    """
    # Create boolean mask for samples in window
    mask = (times >= tmin) & (times <= tmax)
    
    # Verify at least one sample in window
    if not np.any(mask):
        raise ValueError(
            f"No samples in window [{tmin}, {tmax}]. "
            f"Check that window is within epoch time range."
        )
    
    # Average across time dimension
    return arr[..., mask].mean(axis=-1)


def erd_percent(
    epochs: mne.Epochs,
    band: tuple[float, float]
) -> np.ndarray:
    """
    Compute Event-Related Desynchronization (ERD) percentage.
    
    ERD% quantifies power change relative to baseline using standard formula:
        ERD% = (P_task - P_baseline) / P_baseline × 100
    
    Interpretation:
        Negative values: Desynchronization (ERD, expected for motor tasks)
            -50% = power decreased to half of baseline
            -80% = strong desynchronization, power at 20% of baseline
        Positive values: Synchronization (ERS, rare in motor execution)
            +50% = power increased to 1.5× baseline
        Near zero: No task-related modulation
    
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
        Negative = desynchronization, Positive = synchronization
    
    Notes
    -----
    Strong ERD (large negative values) indicates:
    - Active cortical desynchronization during task
    - Good task engagement and motor cortex activation
    - Potential for BCI control
    - Subject is a "good responder"
    
    Weak or positive ERD indicates:
    - Poor task response
    - Possible non-responder
    - May need different frequency band
    - Consider task compliance issues
    
    Formula Details:
    - P_baseline: Average power in baseline window (pre-stimulus)
    - P_task: Average power in task window (during movement)
    - Division by baseline normalizes across subjects/channels
    - Multiply by 100 for percentage
    - Add small constant (1e-12) to avoid division by zero
    """
    # Compute instantaneous power time course
    power, times = bandpower_timecourse(epochs, band)
    
    # Average power in baseline window
    p_base = mean_in_window(times, power, baseline_win[0], baseline_win[1])
    
    # Average power in task window
    p_task = mean_in_window(times, power, task_win[0], task_win[1])
    
    # Compute ERD% using standard formula
    # Add small constant to avoid division by zero (never happens in practice)
    erd = (p_task - p_base) / (p_base + 1e-12) * 100.0
    
    return erd


def summarize_condition(
    erd: np.ndarray,
    ch_names: list[str]
) -> tuple[dict, dict]:
    """
    Summarize ERD statistics per channel.
    
    Computes two key metrics for each channel:
    1. Mean ERD: Average ERD% across epochs (strength of response)
    2. Negative percentage: Fraction of epochs with negative ERD (consistency)
    
    These metrics together characterize ERD quality:
    - Strong mean ERD + high negative% = reliable desynchronization
    - Strong mean ERD + low negative% = inconsistent response
    - Weak mean ERD + high negative% = consistent but weak response
    
    Parameters
    ----------
    erd : np.ndarray
        ERD values (n_epochs, n_channels).
    ch_names : list of str
        Channel names corresponding to columns.
    
    Returns
    -------
    mean_by_ch : dict
        Mean ERD percentage per channel.
        Key: channel name, Value: mean ERD%
    negpct_by_ch : dict
        Percentage of epochs with negative ERD per channel.
        Key: channel name, Value: percentage (0-100)
    
    Notes
    -----
    Negative percentage (negpct) is important because:
    - Indicates consistency across trials
    - High negpct (>70%) suggests reliable ERD
    - Low negpct (<50%) suggests variable response
    - Used as quality metric for channel/band selection
    """
    # Mean ERD per channel (strength metric)
    mean_by_ch = {
        ch: float(np.mean(erd[:, i]))
        for i, ch in enumerate(ch_names)
    }
    
    # Percentage of negative ERD epochs (consistency metric)
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
    separated based on mean ERD values. Higher discriminability indicates:
    - Better separability for classification
    - Stronger differential cortical response
    - More potential for BCI control
    
    Parameters
    ----------
    mean_t1 : dict
        Mean ERD per channel for condition T1 (fists).
        Key: channel name, Value: mean ERD%
    mean_t2 : dict
        Mean ERD per channel for condition T2 (feet).
        Key: channel name, Value: mean ERD%
    
    Returns
    -------
    maxdiff : float
        Maximum absolute difference across channels.
        Represents best-case discriminability (optimal channel).
    meandiff : float
        Mean absolute difference across channels.
        Represents average discriminability across ROI.
    
    Notes
    -----
    maxdiff vs meandiff:
    - maxdiff: More relevant for channel selection in BCI
      Tells you the best possible separation with optimal channel
    - meandiff: More robust to outliers and channel artifacts
      Tells you average separation across entire ROI
    
    Example interpretation:
    - maxdiff = 30%: Best channel shows 30% ERD difference
      One condition shows -40% ERD, other shows -10% ERD
    - meandiff = 15%: Average across channels is 15% difference
      Some channels show strong separation, others weak
    
    For BCI applications:
    - maxdiff > 20%: Excellent discriminability
    - maxdiff 10-20%: Good discriminability
    - maxdiff < 10%: Poor discriminability (may need different band)
    """
    # Find channels common to both conditions
    chs = sorted(set(mean_t1.keys()) & set(mean_t2.keys()))
    
    # Return NaN if no common channels (should never happen)
    if len(chs) == 0:
        return np.nan, np.nan
    
    # Compute absolute differences per channel
    # Absolute value because we care about magnitude, not direction
    diffs = np.array([mean_t1[ch] - mean_t2[ch] for ch in chs], dtype=float)
    
    # Return both max and mean for different use cases
    return float(np.max(np.abs(diffs))), float(np.mean(np.abs(diffs)))


def get_top_subjects_from_step1a(
    path: Path,
    top_n: int
) -> list[str]:
    """
    Load top-performing subjects from Step 1A results.
    
    Step 1A identified subjects with good ERD discriminability using
    extended channel montages. This function selects the top N subjects
    for more detailed frequency band analysis in Step 1B.
    
    Prioritizing top subjects:
    - Reduces computation time (focus on good responders)
    - Improves signal quality in summary statistics
    - Enables faster iteration during development
    
    Parameters
    ----------
    path : Path
        Path to Step 1A CSV or Excel file.
    top_n : int
        Number of top subjects to return.
    
    Returns
    -------
    list of str
        Subject IDs sorted by performance (best first).
        Empty list if file not found or unreadable.
    
    Notes
    -----
    Selection criteria:
    1. If ext_combo_score exists: Use directly (preferred)
    2. If not: Compute as sum of ext_mu_maxdiff + ext_beta_maxdiff
    3. Sort descending and take top N
    
    Falls back to empty list (triggers full scan) if:
    - File doesn't exist
    - File format is invalid
    - Required columns are missing
    """
    # Return empty list if path not provided or doesn't exist
    if not path or not path.exists():
        return []
    
    # Load file based on extension
    try:
        if path.suffix.lower() == ".xlsx":
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path)
    except Exception:
        return []
    
    # Verify subject column exists
    if "subject" not in df.columns:
        return []
    
    # Determine which score column to use
    if "ext_combo_score" in df.columns:
        # Use pre-computed combo score (preferred)
        score_col = "ext_combo_score"
    elif "ext_mu_maxdiff" in df.columns and "ext_beta_maxdiff" in df.columns:
        # Compute combo score as sum of mu and beta discriminability
        df["ext_combo_score"] = (
            df["ext_mu_maxdiff"].astype(float) + 
            df["ext_beta_maxdiff"].astype(float)
        )
        score_col = "ext_combo_score"
    else:
        # Required columns missing, cannot rank subjects
        return []
    
    # Remove subjects with NaN scores, sort descending, take top N
    df = df.dropna(subset=[score_col]).sort_values(score_col, ascending=False)
    
    return df["subject"].astype(str).head(top_n).tolist()


# ============================================================================
# Main Analysis Loop
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("Phase 1 - Step 1B: Multi-Band ERD Discriminability Analysis")
    print("="*70)
    print(f"Runs: {runs_task3_real}")
    print(f"Baseline window: {baseline_win}")
    print(f"Task window: {task_win}")
    print(f"Motor-strip picks: {picks_motor_strip}")
    print(f"Bands tested: {list(bands_to_test.keys())}")
    print("="*70)
    
    # ========================================================================
    # Step 1: Load Subject List
    # ========================================================================
    
    # Try to load top subjects from Step 1A results
    subjects = get_top_subjects_from_step1a(step1a_summary_path, TOP_N)
    
    if subjects:
        print(f"\nUsing top {len(subjects)} subjects from Step 1A results")
        print(f"File: {step1a_summary_path.name}")
    else:
        # Fall back to scanning all subjects
        subjects = subjects_fallback
        print("\n[WARN] Step 1A summary file not found or unreadable")
        print("Falling back to full subject scan: S001-S109")
    
    # ========================================================================
    # Step 2: Initialize Results Storage
    # ========================================================================
    
    rows = []  # Will store per-subject results
    skipped = []  # Will store (subject, reason) for excluded subjects
    
    # ========================================================================
    # Step 3: Process Each Subject
    # ========================================================================
    
    for subj in subjects:
        try:
            # ================================================================
            # Step 3.1: Load and Concatenate Epochs
            # ================================================================
            epochs_all = load_concat_task3_epochs(clean_root, subj, runs_task3_real)
            
            if epochs_all is None:
                skipped.append((subj, "no epochs found"))
                continue
            
            # ================================================================
            # Step 3.2: Get Binary Labels
            # ================================================================
            y = get_labels_T1_T2(epochs_all)
            
            # ================================================================
            # Step 3.3: Select Motor-Strip Channels
            # ================================================================
            ep_roi, picks_used, missing = pick_motor_strip(
                epochs_all, picks_motor_strip
            )
            
            if ep_roi is None:
                skipped.append((subj, f"missing motor-strip channels: {missing}"))
                continue
            
            # ================================================================
            # Step 3.4: Split by Condition
            # ================================================================
            ep_t1 = ep_roi[y == 0]  # Fists
            ep_t2 = ep_roi[y == 1]  # Feet
            
            # Verify both conditions have epochs
            if len(ep_t1) == 0 or len(ep_t2) == 0:
                skipped.append((subj, "no T1 or T2 epochs after concat"))
                continue
            
            # ================================================================
            # Step 3.5: Compute Metrics for Each Band
            # ================================================================
            band_metrics = {}
            
            for band_name, band in bands_to_test.items():
                # Compute ERD for both conditions
                erd_t1 = erd_percent(ep_t1, band)
                erd_t2 = erd_percent(ep_t2, band)
                
                # Summarize per-channel statistics
                m1, n1 = summarize_condition(erd_t1, ep_roi.ch_names)
                m2, n2 = summarize_condition(erd_t2, ep_roi.ch_names)
                
                # Compute discriminability between conditions
                maxdiff, meandiff = discriminability(m1, m2)
                
                # Compute average negative-ERD consistency
                # High values (>70%) indicate reliable desynchronization
                negpct_mean = float(np.mean(
                    list(n1.values()) + list(n2.values())
                ))
                
                # Store all metrics for this band
                band_metrics[band_name] = {
                    "maxdiff": maxdiff,
                    "meandiff": meandiff,
                    "negpct_mean": negpct_mean
                }
            
            # ================================================================
            # Step 3.6: Identify Best Band
            # ================================================================
            # Select band with maximum discriminability
            # Use -inf for NaN values to exclude them from consideration
            best_band = max(
                band_metrics.keys(),
                key=lambda b: (
                    band_metrics[b]["maxdiff"] 
                    if np.isfinite(band_metrics[b]["maxdiff"]) 
                    else -np.inf
                )
            )
            
            # ================================================================
            # Step 3.7: Compute Improvement Over Baseline
            # ================================================================
            # mu_beta is our baseline (conventional broad band)
            base_max = float(band_metrics["mu_beta"]["maxdiff"])
            
            # How much does the best band improve over baseline?
            best_max = float(band_metrics[best_band]["maxdiff"])
            improvement_abs = best_max - base_max
            improvement_pct = 100.0 * improvement_abs / (abs(base_max) + 1e-12)
            
            # ================================================================
            # Step 3.8: Build Output Row
            # ================================================================
            row = {
                # Subject identification
                "subject": subj,
                "n_epochs": int(len(ep_roi)),
                "n_fists": int(len(ep_t1)),
                "n_feet": int(len(ep_t2)),
                "channels_used": ",".join(picks_used),
                
                # Best band results
                "best_band": best_band,
                "best_maxdiff": best_max,
                
                # Baseline comparison
                "baseline_band": "mu_beta",
                "baseline_maxdiff": base_max,
                "best_minus_baseline": improvement_abs,
                "best_over_baseline_pct": improvement_pct,
            }
            
            # Add per-band metrics (all bands)
            for band_name in bands_to_test.keys():
                row[f"{band_name}_maxdiff"] = float(
                    band_metrics[band_name]["maxdiff"]
                )
                row[f"{band_name}_meandiff"] = float(
                    band_metrics[band_name]["meandiff"]
                )
                row[f"{band_name}_negpct_mean"] = float(
                    band_metrics[band_name]["negpct_mean"]
                )
            
            rows.append(row)
        
        except Exception as e:
            # Catch any unexpected errors
            skipped.append((subj, str(e)))
    
    # ========================================================================
    # Step 4: Save Results to CSV
    # ========================================================================
    df = pd.DataFrame(rows)
    
    out_csv = Path(__file__).with_name(
        "Phase1_Step1B_Task3_band_discriminability_motor_strip.csv"
    )
    df.to_csv(out_csv, index=False)
    
    print(f"\n[OK] Saved: {out_csv}")
    
    # ========================================================================
    # Step 5: Print Summary Statistics
    # ========================================================================
    if len(df) == 0:
        print("\n[WARN] No usable subjects computed.")
        print("Check paths and epoch file availability.")
    else:
        print("\n" + "="*70)
        print("SUMMARY STATISTICS")
        print("="*70)
        print(f"Subjects computed: {len(df)}")
        print(f"Subjects skipped: {len(skipped)}")
        
        # ====================================================================
        # Band Preference Distribution
        # ====================================================================
        win_counts = df["best_band"].value_counts()
        win_frac = (win_counts / len(df) * 100.0).round(1)
        
        print("\n" + "-"*70)
        print("Best Band Distribution")
        print("-"*70)
        for b in win_counts.index:
            print(f"  {b:10s}: {int(win_counts[b]):3d} subjects "
                  f"({float(win_frac[b]):4.1f}%)")
        
        # ====================================================================
        # Key Research Questions
        # ====================================================================
        print("\n" + "-"*70)
        print("Research Questions")
        print("-"*70)
        
        # Q1: Is broad beta optimal for most subjects?
        beta_wins = int(win_counts.get("beta", 0))
        print(f"Q1: Is beta (13-30 Hz) optimal for most subjects?")
        print(f"    Answer: {beta_wins}/{len(df)} ({100*beta_wins/len(df):.1f}%)")
        
        # Q2: Do subjects benefit from narrow beta bands?
        narrow_wins = int(
            win_counts.get("beta_low", 0) + 
            win_counts.get("beta_high", 0)
        )
        print(f"\nQ2: How many benefit from narrow beta sub-bands?")
        print(f"    Answer: {narrow_wins}/{len(df)} "
              f"({100*narrow_wins/len(df):.1f}%)")
        
        # ====================================================================
        # Improvement Statistics
        # ====================================================================
        improv = df["best_minus_baseline"].dropna()
        
        if len(improv) > 0:
            print("\n" + "-"*70)
            print("Improvement Over Baseline (mu_beta)")
            print("-"*70)
            print(f"  Mean:   {improv.mean():6.3f}% (average benefit)")
            print(f"  Median: {improv.median():6.3f}% (typical benefit)")
            print(f"  75th:   {improv.quantile(0.75):6.3f}% "
                  "(upper quartile)")
            print(f"  90th:   {improv.quantile(0.90):6.3f}% "
                  "(strong responders)")
            print(f"  Max:    {improv.max():6.3f}% "
                  "(best case improvement)")
        
        # ====================================================================
        # Top Subjects
        # ====================================================================
        top = df.sort_values("best_minus_baseline", ascending=False).head(10)
        
        show_cols = [
            "subject",
            "best_band",
            "best_maxdiff",
            "baseline_maxdiff",
            "best_minus_baseline",
            "best_over_baseline_pct",
        ]
        
        print("\n" + "-"*70)
        print("Top 10 Subjects with Greatest Improvement")
        print("-"*70)
        print(top[show_cols].to_string(index=False))
    
    # ========================================================================
    # Step 6: Save Skip Log
    # ========================================================================
    if skipped:
        out_skip = Path(__file__).with_name(
            "Phase1_Step1B_skipped_subjects.txt"
        )
        
        with out_skip.open("w", encoding="utf-8") as f:
            for s, reason in skipped:
                f.write(f"{s}\t{reason}\n")
        
        print(f"\n[INFO] Wrote skip log: {out_skip.name}")
        print(f"        Contains {len(skipped)} skipped subjects with reasons")
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70 + "\n")
