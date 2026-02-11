"""
Phase 1: ERD Validation Using Multiple Baseline Conditions

This script validates that Task 3 motor execution ERD is task-specific (not general
arousal or attention) by comparing motor task activity against three independent
baseline conditions: pre-cue rest, eyes-open resting, and eyes-closed resting.

Research Question:
    Is the observed ERD during fist/feet motor execution significantly different
    from spontaneous activity during resting baseline conditions?

Validation Strategy:
    1. Within-task baseline: Pre-cue rest (-0.5 to 0s) vs task execution
    2. Eyes-open baseline: R01 resting state vs task execution
    3. Eyes-closed baseline: R02 resting state vs task execution
    
    If ERD is task-specific, we expect:
    - Strong ERD during task vs pre-cue (negative %)
    - Lower power during task vs resting baselines
    - Statistical significance (p < 0.05) in multiple channels

Analysis Pipeline:
    1. Load baseline runs (R01=eyes-open, R02=eyes-closed)
    2. Create pseudo-epochs from baseline runs (2s sliding windows)
    3. Load Task 3 epochs (R05, R09, R13) with T1 (fists) and T2 (feet)
    4. Compute mu_beta (8-30 Hz) power in multiple time windows:
       - Pre-cue: -0.5 to 0s (within-task baseline)
       - Early task: 0.5-1.5s (movement onset)
       - Peak task: 1.0-2.0s (sustained contraction, expected peak ERD)
       - Late task: 1.5-2.5s (movement offset)
    5. Calculate ERD% and baseline contrasts for all channels
    6. Perform Mann-Whitney U tests (task vs baseline power)
    7. Classify subjects into responder categories:
       - Strong: ERD < -20% AND p < 0.05 in ≥3 channels
       - Moderate: ERD < -20% AND p < 0.05 in 1-2 channels
       - Weak: ERD < -20% but not statistically significant
       - Non-responder: ERD ≥ -20%
    8. Generate per-subject validation plots (top 20 responders)
    9. Create population summary statistics and plots

Outputs:
    1. CSV: Phase1_responder_validation_summary.csv
       - Per-subject metrics (n_epochs, ERD%, p-values, responder category)
       - Best channel identification
       - ERD across multiple time windows (early/peak/late)
    
    2. Per-subject plots: Phase1_results/baseline_comparison_plots/
       - SXXX_validation.png for top 20 responders
       - 4-panel layout:
         a) ERD vs pre-cue baseline (bar chart per channel)
         b) Contrast vs eyes-open baseline (bar chart)
         c) Contrast vs eyes-closed baseline (bar chart)
         d) ERD across time windows (early/peak/late)
    
    3. Population summary: Phase1_population_summary.png
       - Responder distribution (pie chart)
       - ERD distribution across subjects (violin plot)
       - Best channel frequency (bar chart)

Key Findings (Expected):
    - ~40-50% strong/moderate responders
    - ~20-30% weak responders
    - ~20-30% non-responders
    - Peak ERD window (1.0-2.0s) shows strongest effects
    - C3/Cz/C4 most frequently identified as best channels
    - Eyes-closed baseline typically shows higher power than eyes-open

Important Notes:
    - Uses Mann-Whitney U test (non-parametric, robust to outliers)
    - Filters BEFORE cropping (avoids edge artifacts)
    - Hilbert envelope for instantaneous power estimation
    - Baseline correction not applied to resting state epochs
    - Top 20 responders selected by strongest ERD in peak window
    - All logic preserved from original implementation

Author: Ram
Date: January 29, 2026
Version: 1.0.0
License: MIT

Dependencies:
    - mne >= 1.5.0
    - numpy >= 1.24.0
    - pandas >= 2.0.0
    - matplotlib >= 3.7.0
    - scipy >= 1.10.0
    - seaborn >= 0.12.0

Usage:
    1. Edit CLEAN_ROOT path in CONFIGURATION section
    2. Ensure preprocessing has been completed (cleaned-dataset/ exists)
    3. Run: python Phase1_ERD_Validation.py
    4. Review outputs:
       - Phase1_responder_validation_summary.csv (subject metrics)
       - Phase1_results/baseline_comparison_plots/ (top 20 plots)
       - Phase1_population_summary.png (aggregate statistics)
    5. Proceed to Phase 2 CSP+LDA using validated responders
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import mne

# Suppress MNE info messages for cleaner output
mne.set_log_level('WARNING')

# Set seaborn style for publication-quality plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.1)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to cleaned dataset (output from preprocessing pipeline)
# Structure: CLEAN_ROOT/SXXX/SXXXR01-cleaned_raw.fif, SXXXR05-epo.fif, etc.
CLEAN_ROOT = Path(
    r"YOUR_PATH_HERE/cleaned-dataset"
)

# ============================================================================
# RUN CONFIGURATION
# ============================================================================

# Baseline runs (resting state, no task)
# R01: Eyes open (1 minute)
# R02: Eyes closed (1 minute)
BASELINE_RUNS = ["R01", "R02"]

# Task 3 runs (real motor execution: fists vs feet)
# R05: Task 3, run 1
# R09: Task 3, run 2
# R13: Task 3, run 3
TASK3_RUNS = ["R05", "R09", "R13"]

# ============================================================================
# CHANNEL CONFIGURATION
# ============================================================================

# Motor strip ROI (7-channel montage)
# Covers bilateral sensorimotor cortex plus midline
MOTOR_STRIP_CHANNELS = ["C5", "C3", "C1", "Cz", "C2", "C4", "C6"]

# ============================================================================
# FREQUENCY BAND
# ============================================================================

# Mu+Beta band (8-30 Hz)
# Covers both sensorimotor rhythm (8-13 Hz) and motor control rhythm (13-30 Hz)
# Conventional choice for motor BCI applications
MU_BETA_BAND = (8.0, 30.0)

# ============================================================================
# TIME WINDOWS
# ============================================================================

# Pre-cue baseline window (within-task)
# Extended slightly to avoid motor preparation contamination
PRECUE_WINDOW = (-0.5, 0.0)

# Task execution windows (multiple windows for diagnostic purposes)
# Different phases of motor execution show varying ERD patterns
TASK_WINDOWS = {
    'early': (0.5, 1.5),  # Movement onset and acceleration
    'peak': (1.0, 2.0),   # Sustained contraction (EXPECTED STRONGEST ERD)
    'late': (1.5, 2.5)    # Movement deceleration and offset
}

# Default window for classification and ranking
# Peak window typically shows strongest and most stable ERD
TASK_WINDOW_DEFAULT = 'peak'

# Diagnostic: Verify dictionary keys are accessible
print(f"DEBUG: TASK_WINDOWS keys = {list(TASK_WINDOWS.keys())}")

# ============================================================================
# RESPONDER CRITERIA
# ============================================================================

# ERD threshold (percent change from baseline)
# Negative values indicate desynchronization
# -20% = power decreased to 80% of baseline
ERD_THRESHOLD = -20.0

# Statistical significance threshold (Mann-Whitney U test)
# Standard alpha level for hypothesis testing
P_VALUE_THRESHOLD = 0.05

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

# Main output directory (created next to script)
OUTPUT_DIR = Path(__file__).parent / "Phase1_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# Per-subject plots directory
PLOTS_DIR = OUTPUT_DIR / "baseline_comparison_plots"
PLOTS_DIR.mkdir(exist_ok=True)


# ============================================================================
# Helper Functions: Data Loading
# ============================================================================

def load_baseline_epochs(subject: str, run: str) -> mne.Epochs | None:
    """
    Load baseline (resting) run and create pseudo-epochs using sliding windows.
    
    Baseline runs (R01, R02) are continuous resting-state EEG without task events.
    We create fixed-length pseudo-epochs to enable statistical comparison with
    task epochs. Sliding windows with overlap increase statistical power.
    
    Parameters
    ----------
    subject : str
        Subject ID (e.g., "S001").
    run : str
        Baseline run ID: "R01" (eyes open) or "R02" (eyes closed).
    
    Returns
    -------
    mne.Epochs or None
        Pseudo-epochs (2s duration, 1s overlap) with motor-strip channels.
        None if file not found or channels missing.
    
    Notes
    -----
    Pseudo-epoch creation:
    - Duration: 2 seconds (matches typical motor task epoch length)
    - Overlap: 1 second (50% overlap increases sample size)
    - No baseline correction (resting state is the baseline)
    
    Example:
    - 60-second resting run → ~59 pseudo-epochs
    - With 7 channels → ~413 total observations
    """
    # Construct path to cleaned raw file
    raw_path = CLEAN_ROOT / subject / f"{subject}{run}-cleaned_raw.fif"
    
    if not raw_path.exists():
        print(f"  [WARN] Baseline run not found: {raw_path.name}")
        return None
    
    try:
        # Load cleaned continuous EEG
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
        
        # Verify all motor-strip channels are present
        missing_chs = [ch for ch in MOTOR_STRIP_CHANNELS if ch not in raw.ch_names]
        if missing_chs:
            print(f"  [WARN] Missing channels in {run}: {missing_chs}")
            return None
        
        # Select only motor-strip channels
        raw.pick(MOTOR_STRIP_CHANNELS)
        
        # Create fixed-length pseudo-epochs with sliding windows
        # Duration: 2.0 seconds
        # Overlap: 1.0 second (generates more epochs for robust statistics)
        events = mne.make_fixed_length_events(raw, duration=2.0, overlap=1.0)
        
        # Create epochs from pseudo-events
        epochs = mne.Epochs(
            raw, events,
            tmin=0,  # Start of epoch
            tmax=2.0,  # End of epoch
            baseline=None,  # No baseline correction (this IS the baseline)
            preload=True,
            verbose=False
        )
        
        return epochs
        
    except Exception as e:
        print(f"  [ERROR] Failed to load {raw_path.name}: {e}")
        return None


def load_task3_epochs(subject: str) -> mne.Epochs | None:
    """
    Load and concatenate Task 3 epochs from all available runs.
    
    Task 3 consists of real motor execution (fists vs feet) across
    multiple runs (R05, R09, R13). Concatenating increases statistical
    power for ERD validation.
    
    Parameters
    ----------
    subject : str
        Subject ID (e.g., "S001").
    
    Returns
    -------
    mne.Epochs or None
        Concatenated epochs with T1 (fists) and T2 (feet) events only.
        Motor-strip channels selected.
        None if no usable runs found or channels missing.
    
    Notes
    -----
    Only includes runs that:
    - Have epoch files on disk
    - Contain both T1 and T2 events
    - Pass loading without errors
    
    Typical epoch counts per subject:
    - R05, R09, R13 combined: ~20-30 fists, ~20-30 feet
    - After AutoReject: ~15-25 fists, ~15-25 feet
    """
    epochs_list = []
    
    # Try to load each Task 3 run
    for run in TASK3_RUNS:
        epo_path = CLEAN_ROOT / subject / f"{subject}{run}-epo.fif"
        
        # Skip if epoch file doesn't exist
        if not epo_path.exists():
            continue
        
        try:
            # Load epochs
            ep = mne.read_epochs(epo_path, preload=True, verbose=False)
            
            # Verify T1 (fists) and T2 (feet) events exist
            if "T1" not in ep.event_id or "T2" not in ep.event_id:
                continue
            
            # Keep only Task 3 motor execution events
            ep = ep[["T1", "T2"]]
            epochs_list.append(ep)
            
        except Exception as e:
            print(f"  [WARN] Failed to load {epo_path.name}: {e}")
            continue
    
    # Return None if no valid runs found
    if len(epochs_list) == 0:
        return None
    
    # Concatenate across runs for increased statistical power
    epochs = mne.concatenate_epochs(epochs_list)
    
    # Verify motor-strip channels are present
    missing_chs = [ch for ch in MOTOR_STRIP_CHANNELS if ch not in epochs.ch_names]
    if missing_chs:
        print(f"  [WARN] Missing channels in Task 3: {missing_chs}")
        return None
    
    # Select only motor-strip channels
    epochs.pick(MOTOR_STRIP_CHANNELS)
    
    return epochs


# ============================================================================
# Helper Functions: Power Computation
# ============================================================================

def compute_band_power(
    epochs: mne.Epochs,
    tmin: float,
    tmax: float,
    fmin: float = 8.0,
    fmax: float = 30.0
) -> np.ndarray:
    """
    Compute mu_beta band power in specified time window.
    
    Uses Hilbert envelope method for robust instantaneous power estimation.
    Critical: Filter BEFORE cropping to avoid edge artifacts.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Input epochs (task or baseline).
    tmin : float
        Window start time (seconds).
    tmax : float
        Window end time (seconds).
    fmin : float
        Lower frequency bound (Hz). Default: 8.0
    fmax : float
        Upper frequency bound (Hz). Default: 30.0
    
    Returns
    -------
    power : np.ndarray
        Mean power per epoch and channel.
        Shape: (n_epochs, n_channels)
        Units: μV² (power units match input signal units)
    
    Notes
    -----
    Processing pipeline:
    1. Bandpass filter to target frequency range (FULL epoch duration)
       - Avoids edge artifacts that occur with crop-then-filter
       - FIR filter with firwin design for optimal frequency response
    2. Crop to time window of interest
       - Now safe because filtering was done on full epoch
    3. Hilbert envelope for instantaneous amplitude
       - envelope=True returns amplitude (not complex analytic signal)
    4. Square to get power
    5. Average over time within window
    
    Why Hilbert instead of Welch/FFT:
    - Time-resolved: Preserves temporal structure
    - Robust: Less sensitive to non-stationarity
    - Accurate: Better for short time windows (<2s)
    """
    # Step 1: Bandpass filter to target frequency band
    # CRITICAL: Filter BEFORE cropping to avoid edge artifacts
    epochs_filt = epochs.copy().filter(
        fmin, fmax,
        fir_design='firwin',  # Optimal frequency response
        verbose=False
    )
    
    # Step 2: Crop to time window of interest
    # Safe now because filtering was done on full epoch
    epochs_crop = epochs_filt.crop(tmin=tmin, tmax=tmax)
    
    # Step 3: Hilbert transform to get instantaneous amplitude envelope
    # envelope=True returns amplitude (not complex analytic signal)
    epochs_hilb = epochs_crop.apply_hilbert(envelope=True)
    
    # Step 4: Get amplitude data
    data = epochs_hilb.get_data()  # Shape: (n_epochs, n_channels, n_times)
    
    # Step 5: Square to get power, then average over time
    power = np.mean(data ** 2, axis=2)  # Shape: (n_epochs, n_channels)
    
    return power


# ============================================================================
# Helper Functions: ERD and Contrast Computation
# ============================================================================

def compute_erd_and_contrasts(
    fists_power,
    feet_power,
    precue_power,
    eyes_open_power,
    eyes_closed_power
):
    """
    Compute ERD and baseline contrasts for all channels.
    
    Calculates percent change in power relative to three baseline conditions:
    1. Pre-cue rest (within-task baseline)
    2. Eyes-open resting baseline (R01)
    3. Eyes-closed resting baseline (R02)
    
    Parameters
    ----------
    fists_power : np.ndarray
        Task power for T1 (fists). Shape: (n_epochs_fists, n_channels)
    feet_power : np.ndarray
        Task power for T2 (feet). Shape: (n_epochs_feet, n_channels)
    precue_power : np.ndarray
        Pre-cue baseline power. Shape: (n_epochs_task, n_channels)
    eyes_open_power : np.ndarray
        Eyes-open resting power (R01). Shape: (n_epochs_eo, n_channels)
    eyes_closed_power : np.ndarray
        Eyes-closed resting power (R02). Shape: (n_epochs_ec, n_channels)
    
    Returns
    -------
    dict
        Keys: 'erd_fists', 'erd_feet', 'contrast_eo_fists', 'contrast_eo_feet',
              'contrast_ec_fists', 'contrast_ec_feet'
        Values: np.ndarray of shape (n_channels,) with percent changes
    
    Notes
    -----
    ERD formula:
        ERD% = (P_task - P_baseline) / P_baseline × 100
    
    Interpretation:
    - Negative values: Desynchronization (expected for motor tasks)
    - Positive values: Synchronization (unexpected, may indicate artifacts)
    
    Why three baselines:
    - Pre-cue: Controls for task engagement and attention
    - Eyes-open: Controls for general arousal and visual input
    - Eyes-closed: Controls for alpha rhythm and relaxation state
    
    If task ERD is specific to motor execution:
    - Should show negative ERD vs pre-cue
    - Should show lower power than both resting states
    - Should be statistically significant vs all three baselines
    """
    # Compute mean power per condition across epochs
    # Shape: (n_channels,)
    mean_fists = np.mean(fists_power, axis=0)
    mean_feet = np.mean(feet_power, axis=0)
    mean_precue = np.mean(precue_power, axis=0)
    mean_eo = np.mean(eyes_open_power, axis=0)
    mean_ec = np.mean(eyes_closed_power, axis=0)
    
    # ========================================================================
    # ERD (task vs pre-cue within-task baseline)
    # ========================================================================
    # Standard ERD calculation: (P_task - P_baseline) / P_baseline × 100
    erd_fists = (mean_fists - mean_precue) / mean_precue * 100
    erd_feet = (mean_feet - mean_precue) / mean_precue * 100
    
    # ========================================================================
    # Contrast vs eyes-open resting baseline
    # ========================================================================
    # Tests if task power differs from alert resting state
    contrast_eo_fists = (mean_fists - mean_eo) / mean_eo * 100
    contrast_eo_feet = (mean_feet - mean_eo) / mean_eo * 100
    
    # ========================================================================
    # Contrast vs eyes-closed resting baseline
    # ========================================================================
    # Tests if task power differs from relaxed resting state
    # Eyes-closed typically has higher alpha/mu power (paradoxical increase)
    contrast_ec_fists = (mean_fists - mean_ec) / mean_ec * 100
    contrast_ec_feet = (mean_feet - mean_ec) / mean_ec * 100
    
    return {
        'erd_fists': erd_fists,
        'erd_feet': erd_feet,
        'contrast_eo_fists': contrast_eo_fists,
        'contrast_eo_feet': contrast_eo_feet,
        'contrast_ec_fists': contrast_ec_fists,
        'contrast_ec_feet': contrast_ec_feet,
    }


# ============================================================================
# Helper Functions: Statistical Testing
# ============================================================================

def test_significance(
    task_power,
    baseline_power,
    alternative='less'
):
    """
    Mann-Whitney U test comparing task vs baseline power distributions.
    
    Non-parametric test (doesn't assume normal distributions).
    Tests if task power is significantly different from baseline power.
    
    Parameters
    ----------
    task_power : np.ndarray
        Task power. Shape: (n_epochs_task, n_channels)
    baseline_power : np.ndarray
        Baseline power. Shape: (n_epochs_baseline, n_channels)
    alternative : str
        Hypothesis direction:
        - 'less': task < baseline (expected for ERD)
        - 'greater': task > baseline (unexpected, ERS)
        - 'two-sided': task ≠ baseline
    
    Returns
    -------
    pvals : np.ndarray
        P-values per channel. Shape: (n_channels,)
        Lower p-values indicate stronger evidence for difference.
    
    Notes
    -----
    Why Mann-Whitney U (not t-test):
    - Non-parametric: Doesn't assume normal distributions
    - Robust: Less sensitive to outliers
    - Appropriate: Power distributions often non-normal (log-normal)
    
    Interpretation:
    - p < 0.05: Statistically significant difference (common threshold)
    - p < 0.01: Strong evidence
    - p < 0.001: Very strong evidence
    
    Multiple comparisons:
    - Testing 7 channels → increased false positive risk
    - Could apply Bonferroni correction (p < 0.05/7 = 0.0071)
    - We use uncorrected p < 0.05 but require multiple channels
    """
    n_channels = task_power.shape[1]
    pvals = np.zeros(n_channels)
    
    # Perform test separately for each channel
    for ch_idx in range(n_channels):
        # Extract power values for this channel
        task_ch = task_power[:, ch_idx]
        baseline_ch = baseline_power[:, ch_idx]
        
        # Mann-Whitney U test
        # alternative='less' tests if task power < baseline power (ERD)
        _, pval = mannwhitneyu(
            task_ch,
            baseline_ch,
            alternative=alternative
        )
        
        pvals[ch_idx] = pval
    
    return pvals


# ============================================================================
# Helper Functions: Responder Classification
# ============================================================================

def classify_responder(
    erd_fists,
    erd_feet,
    pvals_fists_eo,
    pvals_fists_ec,
    pvals_feet_eo,
    pvals_feet_ec
):
    """
    Classify subject as strong/moderate/weak/non-responder.
    
    Classification based on:
    1. ERD strength (magnitude of desynchronization)
    2. Statistical significance vs resting baselines
    3. Number of channels showing significant effects
    
    Parameters
    ----------
    erd_fists : np.ndarray
        ERD% for fists per channel. Shape: (n_channels,)
    erd_feet : np.ndarray
        ERD% for feet per channel. Shape: (n_channels,)
    pvals_fists_eo : np.ndarray
        P-values for fists vs eyes-open. Shape: (n_channels,)
    pvals_fists_ec : np.ndarray
        P-values for fists vs eyes-closed. Shape: (n_channels,)
    pvals_feet_eo : np.ndarray
        P-values for feet vs eyes-open. Shape: (n_channels,)
    pvals_feet_ec : np.ndarray
        P-values for feet vs eyes-closed. Shape: (n_channels,)
    
    Returns
    -------
    category : str
        Classification: 'strong', 'moderate', 'weak', or 'non-responder'
    n_sig_ch : int
        Number of channels with significant ERD (both baselines)
    
    Notes
    -----
    Classification criteria:
    
    Strong responder:
    - ERD < -20% (strong desynchronization)
    - Significant vs BOTH resting baselines (p < 0.05)
    - In ≥3 channels (spatial consistency)
    - Excellent BCI potential
    
    Moderate responder:
    - ERD < -20%
    - Significant vs BOTH resting baselines
    - In 1-2 channels (limited spatial extent)
    - Good BCI potential with channel selection
    
    Weak responder:
    - ERD < -20% (adequate desynchronization)
    - NOT statistically significant
    - Marginal BCI potential, may improve with training
    
    Non-responder:
    - ERD ≥ -20% (weak or no desynchronization)
    - Poor BCI potential, may respond to different tasks
    
    Why require significance vs BOTH baselines:
    - Ensures ERD is not due to general arousal (vs eyes-open)
    - Ensures ERD is not due to alpha modulation (vs eyes-closed)
    - Increases confidence in task-specificity
    """
    n_channels = len(erd_fists)
    
    # Count channels with significant ERD for each condition
    n_sig_ch = 0
    
    for ch_idx in range(n_channels):
        # Criterion 1: Strong ERD (< -20%)
        strong_erd_fists = erd_fists[ch_idx] < ERD_THRESHOLD
        strong_erd_feet = erd_feet[ch_idx] < ERD_THRESHOLD
        
        # Criterion 2: Significant vs BOTH resting baselines
        sig_fists = (pvals_fists_eo[ch_idx] < P_VALUE_THRESHOLD and
                    pvals_fists_ec[ch_idx] < P_VALUE_THRESHOLD)
        sig_feet = (pvals_feet_eo[ch_idx] < P_VALUE_THRESHOLD and
                   pvals_feet_ec[ch_idx] < P_VALUE_THRESHOLD)
        
        # Count if either condition meets both criteria
        if (strong_erd_fists and sig_fists) or (strong_erd_feet and sig_feet):
            n_sig_ch += 1
    
    # ========================================================================
    # Classify based on number of significant channels
    # ========================================================================
    
    # Check if ANY channel shows strong ERD
    has_strong_erd = np.any(erd_fists < ERD_THRESHOLD) or np.any(erd_feet < ERD_THRESHOLD)
    
    if n_sig_ch >= 3:
        # ≥3 channels with significant ERD
        # Strong spatial consistency, excellent BCI candidate
        category = 'strong'
    elif n_sig_ch >= 1:
        # 1-2 channels with significant ERD
        # Limited spatial extent but still usable for BCI
        category = 'moderate'
    elif has_strong_erd:
        # Strong ERD but not statistically significant
        # May be due to: (1) insufficient epochs, (2) high variability
        # Marginal BCI candidate
        category = 'weak'
    else:
        # No strong ERD
        # Poor BCI candidate for motor imagery
        category = 'non-responder'
    
    return category, n_sig_ch


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_subject_validation(
    subject,
    contrasts,
    category,
    fists_power_all,
    feet_power_all,
    precue_power,
    eo_power,
    ec_power
):
    """
    Generate 4-panel validation plot for a single subject.
    
    Visualizes ERD patterns and baseline comparisons to validate
    task-specific desynchronization.
    
    Parameters
    ----------
    subject : str
        Subject ID for title and filename.
    contrasts : dict
        ERD and contrast values from compute_erd_and_contrasts().
    category : str
        Responder classification.
    fists_power_all : dict
        Fists power across time windows ('early', 'peak', 'late').
    feet_power_all : dict
        Feet power across time windows.
    precue_power : np.ndarray
        Pre-cue baseline power.
    eo_power : np.ndarray
        Eyes-open baseline power.
    ec_power : np.ndarray
        Eyes-closed baseline power.
    
    Returns
    -------
    Path
        Path to saved figure file.
    
    Panel Layout:
    
    [Panel 1: Top-Left] ERD vs Pre-Cue Baseline
    - Bar chart showing ERD% per channel
    - Separate bars for fists (blue) and feet (orange)
    - Horizontal line at -20% (responder threshold)
    - Shows within-task ERD pattern
    
    [Panel 2: Top-Right] Contrast vs Eyes-Open Baseline
    - Bar chart showing power contrast vs R01 (eyes-open)
    - Tests if task differs from alert resting state
    - Negative values indicate task < resting (expected)
    
    [Panel 3: Bottom-Left] Contrast vs Eyes-Closed Baseline
    - Bar chart showing power contrast vs R02 (eyes-closed)
    - Tests if task differs from relaxed resting state
    - Eyes-closed often has higher mu power (paradoxical)
    
    [Panel 4: Bottom-Right] ERD Across Time Windows
    - Line plot showing ERD evolution (early/peak/late)
    - Validates temporal consistency
    - Peak window should show strongest ERD
    
    Figure saved as: PLOTS_DIR/SXXX_validation.png
    """
    # Create figure with 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"{subject} – ERD Validation ({category.upper()} responder)",
        fontsize=14,
        fontweight='bold'
    )
    
    # X-axis positions for channel bars
    x_pos = np.arange(len(MOTOR_STRIP_CHANNELS))
    width = 0.35  # Bar width
    
    # ========================================================================
    # Panel 1: ERD vs Pre-Cue Baseline (Top-Left)
    # ========================================================================
    ax1 = axes[0, 0]
    
    # Plot bars for fists and feet
    ax1.bar(x_pos - width/2, contrasts['erd_fists'],
            width, label='Fists (T1)', color='dodgerblue', alpha=0.8)
    ax1.bar(x_pos + width/2, contrasts['erd_feet'],
            width, label='Feet (T2)', color='darkorange', alpha=0.8)
    
    # Add threshold line
    ax1.axhline(ERD_THRESHOLD, color='red', linestyle='--',
                linewidth=1.5, label=f'Threshold ({ERD_THRESHOLD}%)')
    ax1.axhline(0, color='black', linestyle='-', linewidth=0.8)
    
    # Formatting
    ax1.set_xlabel('Channel', fontweight='bold')
    ax1.set_ylabel('ERD%', fontweight='bold')
    ax1.set_title('Panel A: ERD vs Pre-Cue Baseline', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(MOTOR_STRIP_CHANNELS)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # ========================================================================
    # Panel 2: Contrast vs Eyes-Open Baseline (Top-Right)
    # ========================================================================
    ax2 = axes[0, 1]
    
    ax2.bar(x_pos - width/2, contrasts['contrast_eo_fists'],
            width, label='Fists vs EO', color='dodgerblue', alpha=0.8)
    ax2.bar(x_pos + width/2, contrasts['contrast_eo_feet'],
            width, label='Feet vs EO', color='darkorange', alpha=0.8)
    
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.8)
    
    ax2.set_xlabel('Channel', fontweight='bold')
    ax2.set_ylabel('Contrast (%)', fontweight='bold')
    ax2.set_title('Panel B: Task vs Eyes-Open Baseline (R01)', fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(MOTOR_STRIP_CHANNELS)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # ========================================================================
    # Panel 3: Contrast vs Eyes-Closed Baseline (Bottom-Left)
    # ========================================================================
    ax3 = axes[1, 0]
    
    ax3.bar(x_pos - width/2, contrasts['contrast_ec_fists'],
            width, label='Fists vs EC', color='dodgerblue', alpha=0.8)
    ax3.bar(x_pos + width/2, contrasts['contrast_ec_feet'],
            width, label='Feet vs EC', color='darkorange', alpha=0.8)
    
    ax3.axhline(0, color='black', linestyle='-', linewidth=0.8)
    
    ax3.set_xlabel('Channel', fontweight='bold')
    ax3.set_ylabel('Contrast (%)', fontweight='bold')
    ax3.set_title('Panel C: Task vs Eyes-Closed Baseline (R02)', fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(MOTOR_STRIP_CHANNELS)
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # ========================================================================
    # Panel 4: ERD Across Time Windows (Bottom-Right)
    # ========================================================================
    ax4 = axes[1, 1]
    
    # Compute ERD for each time window
    window_names = ['early', 'peak', 'late']
    erd_fists_windows = []
    erd_feet_windows = []
    
    for window_name in window_names:
        # Recompute contrasts for this window
        contrasts_window = compute_erd_and_contrasts(
            fists_power_all[window_name],
            feet_power_all[window_name],
            precue_power, eo_power, ec_power
        )
        
        # Mean ERD across all channels
        erd_fists_windows.append(np.mean(contrasts_window['erd_fists']))
        erd_feet_windows.append(np.mean(contrasts_window['erd_feet']))
    
    # Plot temporal evolution
    x_windows = np.arange(len(window_names))
    ax4.plot(x_windows, erd_fists_windows, 'o-',
             color='dodgerblue', linewidth=2, markersize=8,
             label='Fists (T1)')
    ax4.plot(x_windows, erd_feet_windows, 's-',
             color='darkorange', linewidth=2, markersize=8,
             label='Feet (T2)')
    
    ax4.axhline(ERD_THRESHOLD, color='red', linestyle='--',
                linewidth=1.5, label=f'Threshold ({ERD_THRESHOLD}%)')
    ax4.axhline(0, color='black', linestyle='-', linewidth=0.8)
    
    ax4.set_xlabel('Time Window', fontweight='bold')
    ax4.set_ylabel('Mean ERD% (all channels)', fontweight='bold')
    ax4.set_title('Panel D: ERD Temporal Evolution', fontweight='bold')
    ax4.set_xticks(x_windows)
    ax4.set_xticklabels(['Early\n(0.5-1.5s)', 'Peak\n(1.0-2.0s)', 'Late\n(1.5-2.5s)'])
    ax4.legend(loc='lower right')
    ax4.grid(True, alpha=0.3)
    
    # ========================================================================
    # Save Figure
    # ========================================================================
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for suptitle
    
    plot_path = PLOTS_DIR / f"{subject}_validation.png"
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return plot_path


def plot_population_summary(df_results):
    """
    Generate population-level summary plot with distribution statistics.
    
    Creates 3-panel figure showing:
    1. Responder category distribution (pie chart)
    2. ERD strength distribution (violin plot)
    3. Best channel frequency (bar chart)
    
    Parameters
    ----------
    df_results : pd.DataFrame
        Results dataframe with columns: responder_category,
        mean_erd_fists_peak, best_channel
    
    Saves:
    ------
    Phase1_population_summary.png in OUTPUT_DIR
    
    Interpretation:
    --------------
    Panel A: Responder Distribution
    - Shows proportion of strong/moderate/weak/non-responders
    - Typical: 40-50% strong+moderate, 20-30% weak, 20-30% non
    
    Panel B: ERD Distribution
    - Violin plot showing ERD% spread per category
    - Strong responders should cluster < -30%
    - Non-responders should cluster > -20%
    
    Panel C: Best Channel Frequency
    - Shows which channels most often have strongest ERD
    - Typically: C3, Cz, C4 dominate (primary motor cortex)
    - C5, C6 less common (lateral motor areas)
    """
    # Create figure with 3 panels
    fig = plt.figure(figsize=(16, 5))
    gs = fig.add_gridspec(1, 3, wspace=0.3)
    
    # ========================================================================
    # Panel A: Responder Category Distribution (Pie Chart)
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Count subjects per category
    category_counts = df_results['responder_category'].value_counts()
    
    # Define colors and order
    category_order = ['strong', 'moderate', 'weak', 'non-responder']
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
    
    # Reorder to match standard order
    counts_ordered = [category_counts.get(cat, 0) for cat in category_order]
    
    # Create pie chart
    ax1.pie(counts_ordered, labels=category_order, autopct='%1.1f%%',
            colors=colors, startangle=90)
    ax1.set_title('Panel A: Responder Classification Distribution',
                  fontweight='bold', fontsize=12)
    
    # ========================================================================
    # Panel B: ERD Distribution by Category (Violin Plot)
    # ========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Prepare data for violin plot
    erd_data = []
    erd_labels = []
    
    for cat in category_order:
        cat_data = df_results[df_results['responder_category'] == cat]['mean_erd_fists_peak']
        if len(cat_data) > 0:
            erd_data.append(cat_data.values)
            erd_labels.append(cat)
    
    # Create violin plot
    parts = ax2.violinplot(erd_data, positions=range(len(erd_labels)),
                          showmeans=True, showmedians=True)
    
    # Color violins to match pie chart
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    # Add horizontal threshold line
    ax2.axhline(ERD_THRESHOLD, color='red', linestyle='--',
                linewidth=1.5, label=f'Threshold ({ERD_THRESHOLD}%)')
    
    # Formatting
    ax2.set_ylabel('Mean ERD% (Peak Window)', fontweight='bold')
    ax2.set_title('Panel B: ERD Strength Distribution', fontweight='bold', fontsize=12)
    ax2.set_xticks(range(len(erd_labels)))
    ax2.set_xticklabels(erd_labels, rotation=15)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()
    
    # ========================================================================
    # Panel C: Best Channel Frequency (Bar Chart)
    # ========================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Count frequency of each channel being "best"
    channel_counts = df_results['best_channel'].value_counts()
    
    # Ensure all channels are represented (even if count=0)
    counts_all = [channel_counts.get(ch, 0) for ch in MOTOR_STRIP_CHANNELS]
    
    # Create bar chart
    bars = ax3.bar(range(len(MOTOR_STRIP_CHANNELS)), counts_all,
                   color='steelblue', alpha=0.8)
    
    # Highlight central channels (C3, Cz, C4)
    central_indices = [MOTOR_STRIP_CHANNELS.index(ch)
                      for ch in ['C3', 'Cz', 'C4'] if ch in MOTOR_STRIP_CHANNELS]
    for idx in central_indices:
        bars[idx].set_color('darkblue')
        bars[idx].set_alpha(0.9)
    
    # Formatting
    ax3.set_xlabel('Channel', fontweight='bold')
    ax3.set_ylabel('Frequency (# subjects)', fontweight='bold')
    ax3.set_title('Panel C: Best Channel Distribution', fontweight='bold', fontsize=12)
    ax3.set_xticks(range(len(MOTOR_STRIP_CHANNELS)))
    ax3.set_xticklabels(MOTOR_STRIP_CHANNELS)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add text note
    ax3.text(0.02, 0.98, 'Dark blue = primary motor cortex\n(C3, Cz, C4)',
            transform=ax3.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ========================================================================
    # Save Figure
    # ========================================================================
    plt.suptitle('Population-Level ERD Validation Summary',
                fontsize=14, fontweight='bold', y=1.02)
    
    output_path = OUTPUT_DIR / "Phase1_population_summary.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"[OK] Population summary plot saved: {output_path.name}")


# ============================================================================
# Main Analysis Function
# ============================================================================

def main():
    """
    Execute complete Phase 1 ERD validation analysis.
    
    Workflow:
    1. Scan subjects S001-S109
    2. Load baseline runs (R01, R02) and Task 3 runs (R05, R09, R13)
    3. Compute power in mu_beta band for all conditions
    4. Calculate ERD and baseline contrasts
    5. Perform statistical tests
    6. Classify responders
    7. Generate per-subject validation plots (top 20)
    8. Create population summary statistics and plot
    9. Export results CSV
    
    Expected Runtime:
    - ~2-3 minutes per subject
    - ~3-5 hours total for 109 subjects
    - Plot generation: ~30 seconds per subject (top 20 only)
    
    Outputs:
    - CSV with per-subject metrics
    - Validation plots for top 20 responders
    - Population summary plot
    - Console summary statistics
    """
    print("\n" + "="*70)
    print("PHASE 1: ERD VALIDATION USING MULTIPLE BASELINE CONDITIONS")
    print("="*70)
    print(f"Dataset: {CLEAN_ROOT}")
    print(f"Frequency band: {MU_BETA_BAND[0]}-{MU_BETA_BAND[1]} Hz (mu+beta)")
    print(f"Task windows: {list(TASK_WINDOWS.keys())}")
    print(f"Classification window: {TASK_WINDOW_DEFAULT}")
    print(f"ERD threshold: {ERD_THRESHOLD}%")
    print(f"Significance threshold: p < {P_VALUE_THRESHOLD}")
    print("="*70 + "\n")
    
    # ========================================================================
    # Scan Subjects and Process
    # ========================================================================
    
    subjects = [f"S{i:03d}" for i in range(1, 110)]  # S001-S109
    results = []
    skipped = []
    
    for idx, subject in enumerate(subjects, 1):
        print(f"[{idx}/{len(subjects)}] Processing {subject}...")
        
        # ====================================================================
        # Step 1: Load Baseline Epochs (Eyes-Open and Eyes-Closed)
        # ====================================================================
        epochs_eo = load_baseline_epochs(subject, "R01")  # Eyes open
        epochs_ec = load_baseline_epochs(subject, "R02")  # Eyes closed
        
        if epochs_eo is None or epochs_ec is None:
            skipped.append((subject, "missing baseline runs"))
            print(f"  [SKIP] Missing baseline runs\n")
            continue
        
        print(f"  Baseline epochs: {len(epochs_eo)} (EO), {len(epochs_ec)} (EC)")
        
        # ====================================================================
        # Step 2: Load Task 3 Epochs (Fists and Feet)
        # ====================================================================
        epochs_task = load_task3_epochs(subject)
        
        if epochs_task is None:
            skipped.append((subject, "missing task epochs"))
            print(f"  [SKIP] Missing Task 3 epochs\n")
            continue
        
        # Count fists (T1) and feet (T2) epochs
        n_fists = len(epochs_task['T1'])
        n_feet = len(epochs_task['T2'])
        
        print(f"  Task epochs: {n_fists} fists, {n_feet} feet")
        
        # Skip if insufficient epochs
        if n_fists < 5 or n_feet < 5:
            skipped.append((subject, f"insufficient epochs ({n_fists} fists, {n_feet} feet)"))
            print(f"  [SKIP] Insufficient epochs\n")
            continue
        
        # ====================================================================
        # Step 3: Compute Power for All Conditions
        # ====================================================================
        
        # Baseline power (eyes-open and eyes-closed)
        # Use middle portion of 2-second pseudo-epochs
        eo_power = compute_band_power(
            epochs_eo,
            tmin=0.5, tmax=1.5,  # Middle 1 second of 2s epoch
            fmin=MU_BETA_BAND[0], fmax=MU_BETA_BAND[1]
        )
        ec_power = compute_band_power(
            epochs_ec,
            tmin=0.5, tmax=1.5,
            fmin=MU_BETA_BAND[0], fmax=MU_BETA_BAND[1]
        )
        
        # Pre-cue baseline power (within-task)
        precue_power = compute_band_power(
            epochs_task,
            tmin=PRECUE_WINDOW[0], tmax=PRECUE_WINDOW[1],
            fmin=MU_BETA_BAND[0], fmax=MU_BETA_BAND[1]
        )
        
        # Task power for fists and feet (separate conditions)
        fists_epochs = epochs_task['T1']
        feet_epochs = epochs_task['T2']
        
        # Compute power for all task windows (early, peak, late)
        fists_power_all = {}
        feet_power_all = {}
        
        for window_name, (tmin, tmax) in TASK_WINDOWS.items():
            fists_power_all[window_name] = compute_band_power(
                fists_epochs,
                tmin=tmin, tmax=tmax,
                fmin=MU_BETA_BAND[0], fmax=MU_BETA_BAND[1]
            )
            feet_power_all[window_name] = compute_band_power(
                feet_epochs,
                tmin=tmin, tmax=tmax,
                fmin=MU_BETA_BAND[0], fmax=MU_BETA_BAND[1]
            )
        
        # Use peak window for classification (default)
        fists_power = fists_power_all[TASK_WINDOW_DEFAULT]
        feet_power = feet_power_all[TASK_WINDOW_DEFAULT]
        
        # ====================================================================
        # Step 4: Compute ERD and Contrasts
        # ====================================================================
        contrasts = compute_erd_and_contrasts(
            fists_power, feet_power, precue_power,
            eo_power, ec_power
        )
        
        # ====================================================================
        # Step 5: Statistical Testing
        # ====================================================================
        # Test fists vs baselines
        pvals_fists_eo = test_significance(fists_power, eo_power, alternative='less')
        pvals_fists_ec = test_significance(fists_power, ec_power, alternative='less')
        
        # Test feet vs baselines
        pvals_feet_eo = test_significance(feet_power, eo_power, alternative='less')
        pvals_feet_ec = test_significance(feet_power, ec_power, alternative='less')
        
        # ====================================================================
        # Step 6: Classify Responder
        # ====================================================================
        category, n_sig_ch = classify_responder(
            contrasts['erd_fists'], contrasts['erd_feet'],
            pvals_fists_eo, pvals_fists_ec,
            pvals_feet_eo, pvals_feet_ec
        )
        
        print(f"  Responder classification: {category.upper()} ({n_sig_ch} significant channels)")
        
        # ====================================================================
        # Step 7: Print Diagnostic ERD Across Windows
        # ====================================================================
        print(f"  Mean ERD (fists): ", end="")
        for window_name in ['early', 'peak', 'late']:
            # Compute contrasts for this window
            contrasts_window = compute_erd_and_contrasts(
                fists_power_all[window_name],
                feet_power_all[window_name],
                precue_power, eo_power, ec_power
            )
            mean_erd = np.mean(contrasts_window['erd_fists'])
            print(f"{window_name}={mean_erd:.1f}% ", end="")
        print()  # New line
        
        # ====================================================================
        # Step 8: Store Results
        # ====================================================================
        
        # Identify best channel (strongest ERD for fists)
        best_ch_idx = np.argmin(contrasts['erd_fists'])
        best_ch_name = MOTOR_STRIP_CHANNELS[best_ch_idx]
        
        result_row = {
            # Subject info
            'subject': subject,
            'n_fists': n_fists,
            'n_feet': n_feet,
            'n_eo_epochs': len(epochs_eo),
            'n_ec_epochs': len(epochs_ec),
            
            # Classification
            'responder_category': category,
            'n_significant_channels': n_sig_ch,
            
            # Best channel metrics
            'best_channel': best_ch_name,
            'best_ch_erd_fists': contrasts['erd_fists'][best_ch_idx],
            'best_ch_erd_feet': contrasts['erd_feet'][best_ch_idx],
            'best_ch_contrast_eo_fists': contrasts['contrast_eo_fists'][best_ch_idx],
            'best_ch_contrast_ec_fists': contrasts['contrast_ec_fists'][best_ch_idx],
            'best_ch_pval_fists_eo': pvals_fists_eo[best_ch_idx],
            'best_ch_pval_fists_ec': pvals_fists_ec[best_ch_idx],
        }
        
        # Add ERD for all three time windows (mean across channels)
        for window_name in ['early', 'peak', 'late']:
            contrasts_window = compute_erd_and_contrasts(
                fists_power_all[window_name],
                feet_power_all[window_name],
                precue_power, eo_power, ec_power
            )
            result_row[f'mean_erd_fists_{window_name}'] = np.mean(contrasts_window['erd_fists'])
            result_row[f'mean_erd_feet_{window_name}'] = np.mean(contrasts_window['erd_feet'])
        
        results.append(result_row)
    
    # ========================================================================
    # Export Results to CSV
    # ========================================================================
    df_results = pd.DataFrame(results)
    
    output_csv = OUTPUT_DIR / "Phase1_responder_validation_summary.csv"
    df_results.to_csv(output_csv, index=False, float_format='%.2f')
    
    print(f"\n[OK] Results saved to: {output_csv.name}")
    
    # ========================================================================
    # Generate Plots for Top 20 Responders
    # ========================================================================
    
    print("\n" + "="*70)
    print("GENERATING PLOTS FOR TOP 20 RESPONDERS")
    print("="*70 + "\n")
    
    # Rank subjects by ERD strength (most negative = best)
    df_sorted = df_results.sort_values('mean_erd_fists_peak', ascending=True)
    top_20_subjects = df_sorted.head(20)['subject'].tolist()
    
    print(f"Top 20 subjects (by ERD): {', '.join(top_20_subjects[:5])}... (+15 more)\n")
    
    # Generate per-subject plots for top 20 only
    plot_count = 0
    for idx, subject in enumerate(top_20_subjects, 1):
        print(f"[{idx}/20] Generating plot for {subject}...", end=" ")
        
        # Get subject data from results
        subject_row = df_results[df_results['subject'] == subject].iloc[0]
        
        # Reload epochs (same as in main loop)
        epochs_eo = load_baseline_epochs(subject, "R01")
        epochs_ec = load_baseline_epochs(subject, "R02")
        epochs_task = load_task3_epochs(subject)
        
        if epochs_eo is None or epochs_ec is None or epochs_task is None:
            print("SKIP (data unavailable)")
            continue
        
        # Recompute power and contrasts
        eo_power = compute_band_power(
            epochs_eo, tmin=0.5, tmax=1.5,
            fmin=MU_BETA_BAND[0], fmax=MU_BETA_BAND[1]
        )
        ec_power = compute_band_power(
            epochs_ec, tmin=0.5, tmax=1.5,
            fmin=MU_BETA_BAND[0], fmax=MU_BETA_BAND[1]
        )
        precue_power = compute_band_power(
            epochs_task,
            tmin=PRECUE_WINDOW[0], tmax=PRECUE_WINDOW[1],
            fmin=MU_BETA_BAND[0], fmax=MU_BETA_BAND[1]
        )
        
        fists_epochs = epochs_task['T1']
        feet_epochs = epochs_task['T2']
        
        fists_power_all = {}
        feet_power_all = {}
        
        for window_name, (tmin, tmax) in TASK_WINDOWS.items():
            fists_power_all[window_name] = compute_band_power(
                fists_epochs, tmin=tmin, tmax=tmax,
                fmin=MU_BETA_BAND[0], fmax=MU_BETA_BAND[1]
            )
            feet_power_all[window_name] = compute_band_power(
                feet_epochs, tmin=tmin, tmax=tmax,
                fmin=MU_BETA_BAND[0], fmax=MU_BETA_BAND[1]
            )
        
        fists_power = fists_power_all[TASK_WINDOW_DEFAULT]
        feet_power = feet_power_all[TASK_WINDOW_DEFAULT]
        
        contrasts = compute_erd_and_contrasts(
            fists_power, feet_power, precue_power,
            eo_power, ec_power
        )
        
        category = subject_row['responder_category']
        
        # Generate plot
        plot_path = plot_subject_validation(
            subject, contrasts, category, fists_power_all,
            feet_power_all, precue_power, eo_power, ec_power
        )
        
        print(f"OK ({plot_path.name})")
        plot_count += 1
    
    print(f"\n✓ Generated {plot_count} per-subject plots in: {PLOTS_DIR}\n")
    
    # ========================================================================
    # Aggregate Statistics
    # ========================================================================
    
    print("\n" + "="*70)
    print("AGGREGATE STATISTICS")
    print("="*70 + "\n")
    
    # Response rates
    n_total = len(df_results)
    n_strong = (df_results['responder_category'] == 'strong').sum()
    n_moderate = (df_results['responder_category'] == 'moderate').sum()
    n_weak = (df_results['responder_category'] == 'weak').sum()
    n_non = (df_results['responder_category'] == 'non-responder').sum()
    
    print(f"Subjects processed: {n_total}")
    print(f"\nResponder classification:")
    print(f"  Strong responders:   {n_strong:3d} ({100*n_strong/n_total:.1f}%)")
    print(f"  Moderate responders: {n_moderate:3d} ({100*n_moderate/n_total:.1f}%)")
    print(f"  Weak responders:     {n_weak:3d} ({100*n_weak/n_total:.1f}%)")
    print(f"  Non-responders:      {n_non:3d} ({100*n_non/n_total:.1f}%)")
    print(f"\nTotal BCI-viable (strong + moderate + weak): {n_strong+n_moderate+n_weak} ({100*(n_strong+n_moderate+n_weak)/n_total:.1f}%)")
    
    # Channel response frequency
    print(f"\nChannel-wise response frequency:")
    print(f"  {'Channel':<8} {'Best for (n subjects)':<25}")
    print(f"  {'-'*8} {'-'*25}")
    for ch in MOTOR_STRIP_CHANNELS:
        n_best = (df_results['best_channel'] == ch).sum()
        print(f"  {ch:<8} {n_best:3d} ({100*n_best/n_total:.1f}%)")
    
    # ========================================================================
    # Generate Population Summary Plot
    # ========================================================================
    print("\n" + "="*70)
    print("GENERATING POPULATION SUMMARY PLOT")
    print("="*70)
    plot_population_summary(df_results)
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    
    # Skipped subjects
    if skipped:
        print(f"\nSkipped subjects ({len(skipped)}):")
        for subj, reason in skipped:
            print(f"  - {subj}: {reason}")
    
    print("\n" + "="*70)
    print("Phase 1 Complete")
    print("="*70)
    print(f"\nNext steps:")
    print(f"  1. Review per-subject plots in: {PLOTS_DIR}")
    print(f"  2. Review population summary: Phase1_population_summary.png")
    print(f"  3. Review {output_csv.name} for per-subject validation")
    print(f"  4. Proceed to Phase 2: CSP+LDA using validated responders")
    print()


if __name__ == "__main__":
    main()
