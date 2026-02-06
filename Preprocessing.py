"""
Batch Preprocessing Pipeline for EEGMMIDB Dataset

This script performs automated preprocessing and epoching for the PhysioNet
EEG Motor Movement/Imagery Database (EEGMMIDB). It processes all subjects
and runs, creating cleaned raw files, epochs, and proxy ECG when available.

Pipeline per subject/run:
1. Load EDF file
2. Detect bad channels and log them
3. Apply ICA-based artifact removal
4. Save cleaned continuous EEG
5. Save proxy ECG (if heartbeat ICs detected)
6. Create epochs for task runs (skip baseline R01/R02)
7. Apply AutoReject if available
8. Save epochs

Output Structure:
    cleaned-dataset/
    └── S001/
        ├── S001_bad_channels.txt        # Bad channel log
        ├── S001R01-cleaned_raw.fif      # Cleaned continuous EEG
        ├── S001R01_proxy_ecg_raw.fif   # Proxy ECG (if available)
        ├── S001R03-epo.fif              # Task epochs (R03-R14)
        └── ...

Author: Ram P Narayanan
Date: 2026-02-06
Version: 1.0.1
License: MIT

Dependencies:
    - mne >= 1.5.0
    - numpy >= 1.24.0
    - preprocess_eegmmidb (local module)
    - autoreject >= 0.4.0 (optional but recommended)

Usage:
    1. Edit paths in CONFIGURATION section below
    2. Set subject range (e.g., range(1, 110) for all subjects)
    3. Run: python EDIH_Preprocessing.py
    
    For testing, start with a small range:
        subject_ids = range(1, 4)  # S001-S003
"""

from pathlib import Path
from preprocess_eegmmidb import (
    preprocess_eegmmidb_edf_with_proxy_ecg,
    detect_bad_channels_simple,
)
import mne
import numpy as np

# ============================================================================
# Optional Dependency: AutoReject
# ============================================================================
try:
    from autoreject import AutoReject
    HAVE_AUTOREJECT = True
    
    # Configure AutoReject with conservative settings
    # n_interpolate: try interpolating 1, 2, 4, or 8 bad channels per epoch
    ar = AutoReject(
        n_interpolate=[1, 2, 4, 8],
        random_state=97,
        n_jobs=1  # Use 1 job to avoid memory issues; increase for speed
    )
except ImportError:
    HAVE_AUTOREJECT = False
    ar = None
    print("[WARN] AutoReject not installed. Epochs will be saved without automated rejection.")


# ============================================================================
# CONFIGURATION - EDIT THESE PATHS
# ============================================================================

# Path to raw EEGMMIDB dataset
# Example structure: data_root/S001/S001R01.edf, S001R02.edf, ...
data_root = Path(
    r"YOUR_PATH_HERE/eeg-motor-movementimagery-dataset-1.0.0/files"
)

# Output directory for cleaned data
# Will create: clean_root/S001/, clean_root/S002/, etc.
clean_root = Path(
    r"YOUR_PATH_HERE/cleaned-dataset"
)

# ============================================================================
# PREPROCESSING PARAMETERS
# ============================================================================

# Filtering settings (in Hz)
L_FREQ_FINAL = 0.5      # High-pass: removes slow drifts
H_FREQ_FINAL = 40.0     # Low-pass: removes muscle artifacts and high-freq noise
L_FREQ_ICA = 1.0        # High-pass for ICA (more aggressive for better decomposition)
NOTCH_FREQ = 50.0       # Powerline frequency (50 Hz for EU/Asia, 60 Hz for US)

# ICA settings
ICA_METHOD = "extended-infomax"  # 'infomax' or 'extended-infomax' (recommended)
N_COMPONENTS = 0.99              # Fraction of variance or exact number (0-1 = variance)
RANDOM_STATE = 97                # Seed for reproducibility

# Bad channel detection thresholds
BAD_Z_THRESH = 6.0      # Z-score threshold (lower = stricter detection)
BAD_FLAT_UV = 0.2       # Flatness threshold in µV

# Epoching parameters (in seconds)
EPOCH_TMIN = -0.5       # Epoch start (relative to event)
EPOCH_TMAX = 4.0        # Epoch end
BASELINE_TMIN = -0.5    # Baseline correction window start
BASELINE_TMAX = 0.0     # Baseline correction window end


# ============================================================================
# Helper Functions
# ============================================================================

def standardize_edf_channel_name(ch: str) -> str:
    """
    Standardize EDF channel names to match standard_1020 montage.
    
    Removes common EDF artifacts like 'EEG ', '-REF', trailing dots.
    
    Parameters
    ----------
    ch : str
        Raw channel name from EDF file.
    
    Returns
    -------
    str
        Cleaned channel name.
    
    Examples
    --------
    >>> standardize_edf_channel_name("EEG Fp1-REF")
    'Fp1'
    """
    ch = ch.strip()
    ch = ch.replace("EEG ", "")
    ch = ch.replace("EEG", "")
    ch = ch.replace("-REF", "")
    ch = ch.replace("REF", "")
    ch = ch.replace(".", "")
    return ch.strip()


def has_eeg_dig_points(info: mne.Info) -> bool:
    """
    Check if EEG digitization points are present.
    
    Required for AutoReject and some visualization functions.
    
    Parameters
    ----------
    info : mne.Info
        MNE info structure.
    
    Returns
    -------
    bool
        True if dig points exist, False otherwise.
    """
    dig = info.get("dig", None)
    if not dig:
        return False
    
    # kind=3 corresponds to FIFFV_POINT_EEG in MNE
    return any(d.get("kind", None) == 3 for d in dig)


def channels_have_positions(inst) -> bool:
    """
    Verify that EEG channels have non-zero 3D positions.
    
    Checks the 'loc' field (first 3 entries are X, Y, Z coordinates).
    
    Parameters
    ----------
    inst : mne.io.BaseRaw or mne.Epochs
        MNE data instance.
    
    Returns
    -------
    bool
        True if at least one EEG channel has non-zero position.
    """
    eeg_picks = mne.pick_types(inst.info, eeg=True, exclude=[])
    if len(eeg_picks) == 0:
        return False
    
    # Extract XYZ coordinates from channel locations
    xyz = np.array([inst.info["chs"][p]["loc"][:3] for p in eeg_picks])
    
    # Check if any channel has non-zero position
    return np.any(np.linalg.norm(xyz, axis=1) > 0)


def detect_bads_pre_ica(
    edf_path: Path,
    l_freq_ica: float = 1.0,
    h_freq: float = 40.0,
    notch_freq: float = 50.0,
    bad_z_thresh: float = 6.0,
    bad_flat_uv: float = 0.2
) -> list:
    """
    Lightweight bad channel detection before running full preprocessing.
    
    This allows logging bad channels before ICA, which is useful for
    quality control and debugging.
    
    Parameters
    ----------
    edf_path : Path
        Path to EDF file.
    l_freq_ica : float, default=1.0
        High-pass filter for ICA preparation.
    h_freq : float, default=40.0
        Low-pass filter.
    notch_freq : float, default=50.0
        Powerline frequency.
    bad_z_thresh : float, default=6.0
        Z-score threshold for bad channel detection.
    bad_flat_uv : float, default=0.2
        Flatness threshold in µV.
    
    Returns
    -------
    list of str
        Bad channel names.
    """
    # Load EDF
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    raw.pick("eeg")
    
    # Standardize and set montage
    raw.rename_channels(standardize_edf_channel_name)
    raw.set_montage("standard_1020", match_case=False, on_missing="ignore")
    
    # Apply filters (same as ICA preparation)
    raw.notch_filter(freqs=[notch_freq], verbose="ERROR")
    raw_ica = raw.filter(l_freq=l_freq_ica, h_freq=h_freq, fir_design="firwin", verbose="ERROR")
    raw_ica.set_eeg_reference("average", projection=False, verbose="ERROR")
    
    # Detect bad channels
    bads = detect_bad_channels_simple(raw_ica, z_thresh=bad_z_thresh, flat_uv=bad_flat_uv)
    
    return bads


# ============================================================================
# Main Processing Loop
# ============================================================================

if __name__ == "__main__":
    
    # ========================================================================
    # Subject Selection
    # ========================================================================
    # EDIT THIS LINE to process different subjects
    # Examples:
    #   range(1, 4)      → S001, S002, S003 (testing)
    #   range(1, 110)    → All subjects S001-S109
    #   range(50, 60)    → S050-S059 (subset)
    subject_ids = range(1, 4)  # START WITH SMALL RANGE FOR TESTING
    
    print("\n" + "="*70)
    print("EEG Motor Movement/Imagery Database - Batch Preprocessing")
    print("="*70)
    print(f"Data root: {data_root}")
    print(f"Output root: {clean_root}")
    print(f"Subjects to process: {len(list(subject_ids))}")
    print(f"ICA method: {ICA_METHOD}")
    print(f"AutoReject available: {HAVE_AUTOREJECT}")
    print("="*70 + "\n")
    
    # ========================================================================
    # Process Each Subject
    # ========================================================================
    for sid in subject_ids:
        subject = f"S{sid:03d}"
        
        print(f"\n{'#'*70}")
        print(f"### Starting SUBJECT {subject}")
        print(f"{'#'*70}\n")
        
        # ====================================================================
        # Create Output Directory
        # ====================================================================
        out_dir = clean_root / subject
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Per-subject bad channel log
        badlog_path = out_dir / f"{subject}_bad_channels.txt"
        
        # ====================================================================
        # Open Bad Channel Log File
        # ====================================================================
        with open(badlog_path, "w", encoding="utf-8") as fbad:
            fbad.write(f"Bad Channel Log for {subject}\n")
            fbad.write("Format: SXXXRYY --> Bad channels: ch1, ch2, ...\n\n")
            
            # ================================================================
            # Process Each Run (R01-R14)
            # ================================================================
            for run_idx in range(1, 15):
                run = f"R{run_idx:02d}"
                edf_path = data_root / subject / f"{subject}{run}.edf"
                
                print(f"\n=== Processing {subject} {run} ===")
                
                # ============================================================
                # Check if EDF File Exists
                # ============================================================
                if not edf_path.exists():
                    print(f"[SKIP] File not found: {edf_path}")
                    fbad.write(f"{subject}{run} --> Missing EDF file\n")
                    continue
                
                try:
                    # ========================================================
                    # Step 0: Detect and Log Bad Channels (Pre-ICA)
                    # ========================================================
                    bads = detect_bads_pre_ica(
                        edf_path,
                        l_freq_ica=L_FREQ_ICA,
                        h_freq=H_FREQ_FINAL,
                        notch_freq=NOTCH_FREQ,
                        bad_z_thresh=BAD_Z_THRESH,
                        bad_flat_uv=BAD_FLAT_UV
                    )
                    fbad.write(f"{subject}{run} --> Bad channels: {', '.join(bads) if bads else 'None'}\n")
                    
                    # ========================================================
                    # Step 1: Preprocess Raw (Filter + ICA)
                    # ========================================================
                    print("[1/5] Running preprocessing pipeline...")
                    raw_clean, ica, exclude, raw_proxy_ecg = preprocess_eegmmidb_edf_with_proxy_ecg(
                        edf_path,
                        l_freq_final=L_FREQ_FINAL,
                        h_freq_final=H_FREQ_FINAL,
                        notch_freqs=(NOTCH_FREQ,),
                        l_freq_ica=L_FREQ_ICA,
                        n_components=N_COMPONENTS,
                        random_state=RANDOM_STATE,
                        ica_method=ICA_METHOD,
                        bad_z_thresh=BAD_Z_THRESH,
                        bad_flat_uv=BAD_FLAT_UV
                    )
                    
                    # ========================================================
                    # Step 2: Save Cleaned Raw
                    # ========================================================
                    print("[2/5] Saving cleaned raw...")
                    out_path = out_dir / f"{subject}{run}-cleaned_raw.fif"
                    raw_clean.save(out_path, overwrite=True)
                    print(f"[OK] Saved: {out_path.name}")
                    
                    # ========================================================
                    # Step 3: Save Proxy ECG (if available)
                    # ========================================================
                    print("[3/5] Checking for proxy ECG...")
                    if raw_proxy_ecg is not None:
                        ecg_out_path = out_dir / f"{subject}{run}_proxy_ecg_raw.fif"
                        raw_proxy_ecg.save(ecg_out_path, overwrite=True)
                        print(f"[OK] Saved proxy ECG: {ecg_out_path.name}")
                    else:
                        print("[INFO] No heartbeat ICs detected; proxy ECG not available.")
                    
                    # ========================================================
                    # Step 4: Skip Epoching for Baseline Runs (R01, R02)
                    # ========================================================
                    # R01 = eyes open baseline, R02 = eyes closed baseline
                    # These typically only have T0 (rest) events, not useful for MI analysis
                    if run in ("R01", "R02"):
                        print(f"[INFO] {run} is baseline run. Skipping epoching.")
                        print(f"[COMPLETE] {subject}{run} - Raw saved only")
                        continue
                    
                    # ========================================================
                    # Step 5: Create Epochs (Task Runs R03-R14)
                    # ========================================================
                    print("[4/5] Creating epochs...")
                    
                    # Standardize channel names (needed for montage consistency)
                    raw_clean.rename_channels(standardize_edf_channel_name)
                    raw_clean.set_montage("standard_1020", match_case=False, on_missing="ignore")
                    
                    # Extract events from annotations
                    events, event_id = mne.events_from_annotations(raw_clean)
                    
                    # Create epochs with baseline correction
                    epochs = mne.Epochs(
                        raw_clean,
                        events,
                        event_id=event_id,
                        tmin=EPOCH_TMIN,
                        tmax=EPOCH_TMAX,
                        baseline=(BASELINE_TMIN, BASELINE_TMAX),  # Apply baseline correction
                        preload=True,
                        reject_by_annotation=True,  # Respect bad segments
                        verbose="ERROR",
                    )
                    
                    # Check if epoching produced any epochs
                    if len(epochs) == 0:
                        print("[WARN] 0 epochs created (likely edge effects or missing events). Skipping epoch save.")
                        print(f"[COMPLETE] {subject}{run} - Raw saved, no epochs")
                        continue
                    
                    epo_out_path = out_dir / f"{subject}{run}-epo.fif"
                    
                    # ========================================================
                    # Step 6: Apply AutoReject (if available and applicable)
                    # ========================================================
                    print("[5/5] Applying AutoReject (if available)...")
                    
                    if HAVE_AUTOREJECT and has_eeg_dig_points(epochs.info):
                        # AutoReject: Automatically interpolate/reject bad epochs
                        epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)
                        epochs_clean.save(epo_out_path, overwrite=True)
                        
                        n_bad = reject_log.bad_epochs.sum()
                        n_total = len(reject_log.bad_epochs)
                        print(f"[OK] Saved clean epochs (AutoReject): {epo_out_path.name}")
                        print(f"[OK] Dropped epochs: {n_bad}/{n_total} ({100*n_bad/n_total:.1f}%)")
                    
                    else:
                        # Save epochs without AutoReject
                        epochs.save(epo_out_path, overwrite=True)
                        
                        if not HAVE_AUTOREJECT:
                            print("[WARN] AutoReject not installed. Saved epochs without automated rejection.")
                        else:
                            print("[WARN] No EEG dig points. Saved epochs without AutoReject.")
                        
                        print(f"[OK] Saved epochs: {epo_out_path.name}")
                    
                    print(f"[COMPLETE] {subject}{run} - All outputs saved successfully")
                
                except Exception as e:
                    # ========================================================
                    # Error Handling: Log and Continue
                    # ========================================================
                    print(f"[ERROR] Failed processing {subject}{run}")
                    print(f"[ERROR] Exception: {e}")
                    fbad.write(f"{subject}{run} --> ERROR during processing: {repr(e)}\n")
        
        # ====================================================================
        # Subject Complete
        # ====================================================================
        print(f"\n[OK] Completed subject {subject}")
        print(f"[OK] Bad channel log: {badlog_path}")
    
    # ========================================================================
    # All Subjects Complete
    # ========================================================================
    print("\n" + "="*70)
    print("Batch preprocessing complete!")
    print(f"Output directory: {clean_root}")
    print("="*70 + "\n")
