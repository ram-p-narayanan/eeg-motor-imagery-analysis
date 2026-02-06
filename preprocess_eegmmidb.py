"""
Preprocessing Utilities for EEG Motor Movement/Imagery Database (EEGMMIDB)

This module provides core preprocessing functions for EEG data, including:
- Channel name standardization
- Bad channel detection and interpolation
- ICA-based artifact removal (eye, muscle, heart, line noise)
- ICLabel-based component classification
- Proxy ECG extraction from ICA sources

Author: Ram P Narayanan
Date: 2026-02-06
Version: 1.0.0
License: MIT

Dependencies:
    - mne >= 1.5.0
    - numpy >= 1.24.0
    - mne-icalabel >= 0.4.0 (optional but recommended)

Usage:
    from preprocess_eegmmidb import preprocess_eegmmidb_edf_with_proxy_ecg
    
    raw_clean, ica, exclude, proxy_ecg = preprocess_eegmmidb_edf_with_proxy_ecg(
        edf_path="data/S001/S001R01.edf",
        ica_method="extended-infomax"
    )
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import mne


# ============================================================================
# Optional Dependencies
# ============================================================================
try:
    from mne_icalabel import label_components
    HAVE_ICALABEL = True
except ImportError:
    HAVE_ICALABEL = False
    label_components = None


# ============================================================================
# Private Helper Functions
# ============================================================================

def _standardize_name(ch: str) -> str:
    """
    Standardize EDF channel names to match MNE montage conventions.
    
    Removes common EDF-specific prefixes and suffixes that prevent
    automatic montage matching.
    
    Parameters
    ----------
    ch : str
        Raw channel name from EDF file (e.g., "EEG Fp1-REF").
    
    Returns
    -------
    str
        Standardized channel name (e.g., "Fp1").
    
    Examples
    --------
    >>> _standardize_name("EEG Fp1-REF")
    'Fp1'
    >>> _standardize_name("EEG C3.")
    'C3'
    """
    ch = ch.strip()
    
    # Remove common EDF prefixes
    ch = ch.replace("EEG ", "").replace("EEG", "")
    
    # Remove reference notation
    ch = ch.replace("-REF", "").replace("REF", "")
    
    # Remove trailing dots (common in some EDF files)
    ch = ch.replace(".", "")
    
    return ch.strip()


def _has_eeg_dig_points(info: mne.Info) -> bool:
    """
    Check if EEG digitization points are present in the info structure.
    
    Digitization points are required for certain operations like ICLabel
    and AutoReject. This function checks if they exist.
    
    Parameters
    ----------
    info : mne.Info
        MNE info structure.
    
    Returns
    -------
    bool
        True if EEG dig points exist, False otherwise.
    
    Notes
    -----
    FIFFV_POINT_EEG = 3 in MNE constants.
    """
    dig = info.get("dig", None)
    if not dig:
        return False
    
    # Check if any digitization point is of type EEG (kind=3)
    return any(d.get("kind", None) == 3 for d in dig)


# ============================================================================
# ICA Configuration
# ============================================================================

def set_ica_method_params(ica_method: str) -> Tuple[str, dict]:
    """
    Map ICA method string to MNE ICA parameters.
    
    Parameters
    ----------
    ica_method : str
        Method name: 'infomax', 'extended-infomax', or 'extended_infomax'.
    
    Returns
    -------
    method : str
        MNE-compatible method name.
    fit_params : dict
        Additional parameters for ICA.fit().
    
    Examples
    --------
    >>> method, params = set_ica_method_params("extended-infomax")
    >>> print(method, params)
    'infomax' {'extended': True}
    """
    ica_method = ica_method.lower().strip()
    
    if ica_method in ["infomax", "extended-infomax", "extended_infomax"]:
        method = "infomax"
        fit_params = {}
        
        # Enable extended mode for sub-Gaussian and super-Gaussian sources
        if ica_method != "infomax":
            fit_params["extended"] = True
        
        return method, fit_params
    
    # Fallback: return as-is (for other MNE-supported methods like 'fastica', 'picard')
    return ica_method, {}


# ============================================================================
# Bad Channel Detection
# ============================================================================

def detect_bad_channels_simple(
    raw: mne.io.BaseRaw,
    z_thresh: float = 6.0,
    flat_uv: float = 0.2
) -> List[str]:
    """
    Detect bad EEG channels using robust statistics.
    
    Identifies two types of bad channels:
    1. Flat channels: Very low standard deviation (< flat_uv µV)
    2. Noisy channels: Extremely high standard deviation (Z-score > z_thresh)
    
    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw EEG data (should be filtered for ICA, e.g., 1-40 Hz).
    z_thresh : float, default=6.0
        Z-score threshold for noisy channel detection.
        Higher values = more conservative (fewer channels marked bad).
    flat_uv : float, default=0.2
        Flatness threshold in microvolts.
        Channels with std < flat_uv are marked as flat.
    
    Returns
    -------
    bad_channels : list of str
        Channel names identified as bad.
    
    Notes
    -----
    Uses median absolute deviation (MAD) for robustness to outliers.
    Formula: Z = (x - median) / (1.4826 * MAD)
    
    Examples
    --------
    >>> raw = mne.io.read_raw_fif("data.fif", preload=True)
    >>> bads = detect_bad_channels_simple(raw, z_thresh=6.0, flat_uv=0.2)
    >>> print(f"Bad channels: {bads}")
    Bad channels: ['T8', 'Fp2']
    """
    # Convert to microvolts for interpretable threshold
    data = raw.get_data() * 1e6
    ch_names = raw.ch_names
    
    # Compute standard deviation per channel
    std = np.std(data, axis=1)
    
    # Robust statistics using MAD
    med = np.median(std)
    mad = np.median(np.abs(std - med)) + 1e-12  # Avoid division by zero
    
    # Compute z-scores (robust to outliers)
    z = (std - med) / (1.4826 * mad)
    
    # Identify flat channels (disconnected or bridged electrodes)
    bad_flat = [ch_names[i] for i, s in enumerate(std) if s < flat_uv]
    
    # Identify noisy channels (high impedance or artifacts)
    bad_noisy = [ch_names[i] for i, zi in enumerate(z) if zi > z_thresh]
    
    # Combine and remove duplicates
    bads = sorted(set(bad_flat + bad_noisy))
    
    return bads


# ============================================================================
# Proxy ECG Utilities
# ============================================================================

def _make_proxy_ecg_raw(
    proxy_ecg: np.ndarray,
    sfreq: float,
    first_samp: int
) -> mne.io.Raw:
    """
    Create a 1-channel Raw object for proxy ECG.
    
    Parameters
    ----------
    proxy_ecg : np.ndarray
        1D array of proxy ECG time series.
    sfreq : float
        Sampling frequency in Hz.
    first_samp : int
        First sample index (for time alignment with EEG).
    
    Returns
    -------
    mne.io.RawArray
        Single-channel Raw with 'ECG' channel type.
    """
    # Ensure 1D array and remove DC offset
    proxy_ecg = np.asarray(proxy_ecg, dtype=float)
    proxy_ecg = proxy_ecg - np.mean(proxy_ecg)
    
    # Create MNE info for ECG channel
    info = mne.create_info(ch_names=["ECG"], sfreq=sfreq, ch_types=["ecg"])
    
    # Create Raw object (data must be 2D: n_channels x n_times)
    raw_ecg = mne.io.RawArray(proxy_ecg[np.newaxis, :], info)
    
    # Align time base with reference raw
    raw_ecg._first_samps = np.array([int(first_samp)], dtype=int)
    
    return raw_ecg


def extract_proxy_ecg_from_ica(
    raw_ref: mne.io.BaseRaw,
    ica: mne.preprocessing.ICA,
    heartbeat_ic_inds: List[int]
) -> Optional[mne.io.Raw]:
    """
    Reconstruct proxy ECG from ICA heartbeat components.
    
    Extracts ICA sources corresponding to heartbeat activity and
    combines them into a single pseudo-ECG time series.
    
    Parameters
    ----------
    raw_ref : mne.io.BaseRaw
        Reference raw data (used to get ICA sources).
    ica : mne.preprocessing.ICA
        Fitted ICA object.
    heartbeat_ic_inds : list of int
        Indices of ICA components labeled as "heart beat" or "heart".
    
    Returns
    -------
    mne.io.Raw or None
        1-channel ECG raw if heartbeat ICs exist, None otherwise.
    
    Notes
    -----
    - Uses robust median aggregation across multiple heartbeat components
    - Standardizes output (z-score normalization)
    - Useful for heart rate variability analysis or cardiac artifact validation
    """
    if heartbeat_ic_inds is None or len(heartbeat_ic_inds) == 0:
        return None
    
    # Extract ICA sources (n_components x n_times)
    sources = ica.get_sources(raw_ref).get_data()
    
    # Select heartbeat components
    hb = sources[heartbeat_ic_inds, :]
    
    # Aggregate using median (robust to outliers)
    proxy = np.median(hb, axis=0)
    
    # Standardize for comparability across runs
    proxy = (proxy - np.mean(proxy)) / (np.std(proxy) + 1e-12)
    
    return _make_proxy_ecg_raw(
        proxy,
        sfreq=raw_ref.info["sfreq"],
        first_samp=raw_ref.first_samp
    )


# ============================================================================
# Main Preprocessing Functions
# ============================================================================

def preprocess_eegmmidb_edf(
    edf_path: Path,
    l_freq_final: float = 0.5,
    h_freq_final: float = 40.0,
    notch_freqs: Tuple[float, ...] = (50.0,),
    l_freq_ica: float = 1.0,
    n_components: float = 0.99,
    random_state: int = 97,
    ica_method: str = "infomax",
    bad_z_thresh: float = 6.0,
    bad_flat_uv: float = 0.2,
) -> Tuple[mne.io.BaseRaw, mne.preprocessing.ICA, List[int]]:
    """
    Preprocess EEGMMIDB EDF file (backward-compatible version).
    
    This is a wrapper around preprocess_eegmmidb_edf_with_proxy_ecg that
    maintains backward compatibility by not returning proxy ECG.
    
    Parameters
    ----------
    edf_path : Path
        Path to EDF file.
    l_freq_final : float, default=0.5
        Final high-pass filter cutoff (Hz).
    h_freq_final : float, default=40.0
        Final low-pass filter cutoff (Hz).
    notch_freqs : tuple of float, default=(50.0,)
        Powerline frequencies to notch filter.
    l_freq_ica : float, default=1.0
        High-pass filter for ICA (more aggressive than final).
    n_components : float, default=0.99
        Number of ICA components (0-1 = variance explained, >1 = exact number).
    random_state : int, default=97
        Random seed for reproducibility.
    ica_method : str, default='infomax'
        ICA algorithm: 'infomax' or 'extended-infomax'.
    bad_z_thresh : float, default=6.0
        Z-score threshold for bad channel detection.
    bad_flat_uv : float, default=0.2
        Flatness threshold for bad channels (µV).
    
    Returns
    -------
    raw_clean : mne.io.BaseRaw
        Cleaned and filtered raw data.
    ica : mne.preprocessing.ICA
        Fitted ICA object.
    exclude : list of int
        Indices of excluded ICA components.
    
    See Also
    --------
    preprocess_eegmmidb_edf_with_proxy_ecg : Extended version with proxy ECG
    """
    raw_clean, ica, exclude, _ = preprocess_eegmmidb_edf_with_proxy_ecg(
        edf_path=edf_path,
        l_freq_final=l_freq_final,
        h_freq_final=h_freq_final,
        notch_freqs=notch_freqs,
        l_freq_ica=l_freq_ica,
        n_components=n_components,
        random_state=random_state,
        ica_method=ica_method,
        bad_z_thresh=bad_z_thresh,
        bad_flat_uv=bad_flat_uv,
    )
    return raw_clean, ica, exclude


def preprocess_eegmmidb_edf_with_proxy_ecg(
    edf_path: Path,
    l_freq_final: float = 0.5,
    h_freq_final: float = 40.0,
    notch_freqs: Tuple[float, ...] = (50.0,),
    l_freq_ica: float = 1.0,
    n_components: float = 0.99,
    random_state: int = 97,
    ica_method: str = "infomax",
    bad_z_thresh: float = 6.0,
    bad_flat_uv: float = 0.2,
) -> Tuple[mne.io.BaseRaw, mne.preprocessing.ICA, List[int], Optional[mne.io.Raw]]:
    """
    Complete preprocessing pipeline for EEGMMIDB EDF files.
    
    Pipeline steps:
    1. Load EDF and keep EEG channels only
    2. Standardize channel names
    3. Set montage (standard_1020)
    4. Notch filter (powerline noise)
    5. Detect and interpolate bad channels
    6. High-pass filter for ICA (1 Hz)
    7. Average reference
    8. Fit ICA (Extended Infomax recommended)
    9. Label components with ICLabel (if available)
    10. Exclude artifact components (eye, muscle, heart, line noise)
    11. Extract proxy ECG from heartbeat components
    12. Apply final bandpass filter (0.5-40 Hz)
    13. Apply ICA cleaning
    
    Parameters
    ----------
    edf_path : Path
        Path to EDF file (e.g., "data/S001/S001R01.edf").
    l_freq_final : float, default=0.5
        Final high-pass cutoff (Hz). Removes slow drifts.
    h_freq_final : float, default=40.0
        Final low-pass cutoff (Hz). Removes high-frequency noise and muscle.
    notch_freqs : tuple of float, default=(50.0,)
        Powerline frequencies to remove. Use (60.0,) for North America.
        Note: For EEGMMIDB (@160 Hz), harmonics (100, 150 Hz) exceed Nyquist.
    l_freq_ica : float, default=1.0
        High-pass for ICA (more aggressive than final to improve decomposition).
    n_components : float, default=0.99
        ICA components: 0-1 = fraction of variance, >1 = exact number.
    random_state : int, default=97
        Random seed for ICA reproducibility.
    ica_method : str, default='infomax'
        ICA algorithm: 'infomax' (super-Gaussian) or 'extended-infomax'
        (super- and sub-Gaussian, recommended).
    bad_z_thresh : float, default=6.0
        Z-score threshold for noisy channel detection. Lower = stricter.
    bad_flat_uv : float, default=0.2
        Flatness threshold in µV. Channels with std < this are marked bad.
    
    Returns
    -------
    raw_clean : mne.io.BaseRaw
        Cleaned, filtered, ICA-corrected raw data with average reference.
    ica : mne.preprocessing.ICA
        Fitted ICA object with excluded components.
    exclude : list of int
        Indices of excluded ICA components (artifacts).
    raw_proxy_ecg : mne.io.Raw or None
        1-channel proxy ECG reconstructed from heartbeat ICs.
        None if ICLabel unavailable, no dig points, or no heartbeat ICs detected.
    
    Notes
    -----
    - Proxy ECG requires: ICLabel installed, EEG dig points, heartbeat ICs detected
    - Fallback without ICLabel: Heuristic EOG detection using frontal channels
    - Bad channels are interpolated before ICA to avoid rank deficiency
    
    Examples
    --------
    >>> from pathlib import Path
    >>> edf_path = Path("data/S001/S001R01.edf")
    >>> raw_clean, ica, exclude, proxy_ecg = preprocess_eegmmidb_edf_with_proxy_ecg(
    ...     edf_path,
    ...     ica_method="extended-infomax"
    ... )
    >>> raw_clean.save("S001R01-cleaned_raw.fif", overwrite=True)
    >>> if proxy_ecg is not None:
    ...     proxy_ecg.save("S001R01_proxy_ecg_raw.fif", overwrite=True)
    """
    edf_path = Path(edf_path)
    
    # ========================================================================
    # Step 1: Load EDF and keep EEG channels only
    # ========================================================================
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    raw.pick("eeg")  # Ignore EDF annotation channel and non-EEG
    
    # ========================================================================
    # Step 2: Standardize channel names early (enables montage matching)
    # ========================================================================
    raw.rename_channels(_standardize_name)
    
    # ========================================================================
    # Step 3: Set montage (approximate 3D locations for topography plots)
    # ========================================================================
    raw.set_montage("standard_1020", match_case=False, on_missing="warn")
    
    # ========================================================================
    # Step 4: Notch filter to remove powerline noise
    # ========================================================================
    # Note: For EEGMMIDB (@160 Hz), only 50 Hz fits within Nyquist (80 Hz)
    # If using other datasets with higher sampling rates, include harmonics
    raw.notch_filter(freqs=list(notch_freqs), verbose="ERROR")
    
    # Keep a copy for final filtering after ICA
    raw_for_final = raw.copy()
    
    # ========================================================================
    # Step 5: Prepare data for ICA (higher high-pass improves decomposition)
    # ========================================================================
    raw_ica = raw.copy().filter(
        l_freq=l_freq_ica,
        h_freq=h_freq_final,
        fir_design="firwin",
        verbose="ERROR"
    )
    
    # ========================================================================
    # Step 6: Average reference (ICLabel expects average reference)
    # ========================================================================
    raw_ica.set_eeg_reference("average", projection=False, verbose="ERROR")
    
    # ========================================================================
    # Step 7: Detect and interpolate bad channels BEFORE ICA
    # ========================================================================
    # Bad channels can distort ICA decomposition, so interpolate them first
    bads = detect_bad_channels_simple(
        raw_ica,
        z_thresh=bad_z_thresh,
        flat_uv=bad_flat_uv
    )
    
    if bads:
        print(f"[QC] Bad channels detected (pre-ICA): {bads}")
        raw_ica.info["bads"] = bads
        raw_ica.interpolate_bads(reset_bads=True, verbose="ERROR")
        
        # Re-apply montage to ensure consistency after interpolation
        raw_ica.set_montage("standard_1020", match_case=False, on_missing="warn")
    
    # ========================================================================
    # Step 8: Fit ICA
    # ========================================================================
    method, fit_params = set_ica_method_params(ica_method)
    
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method=method,
        fit_params=fit_params if fit_params else None,
        random_state=random_state,
        max_iter="auto",
    )
    ica.fit(raw_ica, verbose="ERROR")
    
    # ========================================================================
    # Step 9: Identify artifact ICA components
    # ========================================================================
    exclude: List[int] = []
    heartbeat_ic_inds: List[int] = []
    
    if HAVE_ICALABEL and _has_eeg_dig_points(raw_ica.info):
        # ====================================================================
        # ICLabel-based artifact detection (recommended)
        # ====================================================================
        labels = label_components(raw_ica, ica, method="iclabel")
        lab = np.array(labels["labels"])
        
        # Identify heartbeat components for proxy ECG reconstruction
        heartbeat_ic_inds = np.where(
            np.isin(lab, ["heart beat", "heart"])
        )[0].tolist()
        
        # Exclude artifact classes (keep brain and other)
        artifact_classes = {
            "eye blink",
            "muscle artifact",
            "muscle",
            "heart beat",
            "heart",
            "line noise",
            "channel noise"
        }
        exclude = np.where(np.isin(lab, list(artifact_classes)))[0].tolist()
        
        print(f"[ICA] ICLabel exclude: {exclude} | Heartbeat ICs: {heartbeat_ic_inds}")
    
    else:
        # ====================================================================
        # Fallback: Heuristic artifact detection (without ICLabel)
        # ====================================================================
        if HAVE_ICALABEL and not _has_eeg_dig_points(raw_ica.info):
            print("[WARN] No EEG dig points; skipping ICLabel. Using heuristic EOG detection.")
            print("[WARN] Proxy ECG will not be available without ICLabel.")
        
        # Try to detect EOG-like components using frontal channels as proxies
        frontal = [
            ch for ch in ["Fp1", "Fp2", "AF7", "AF8", "F7", "F8"]
            if ch in raw_ica.ch_names
        ]
        
        if frontal:
            try:
                eog_inds, _ = ica.find_bads_eog(
                    raw_ica,
                    ch_name=frontal,
                    verbose="ERROR"
                )
                exclude += eog_inds
                print(f"[ICA] Heuristic EOG exclude: {eog_inds}")
            except Exception as e:
                print(f"[ICA] EOG detection failed: {e}")
        
        # Note: ECG detection without ECG channel is unreliable, so we skip it
        exclude = sorted(set(exclude))
    
    ica.exclude = exclude
    
    # ========================================================================
    # Step 10: Apply final bandpass filter for analysis
    # ========================================================================
    raw_final_ref = raw_for_final.filter(
        l_freq=l_freq_final,
        h_freq=h_freq_final,
        fir_design="firwin",
        verbose="ERROR"
    )
    raw_final_ref.set_eeg_reference("average", projection=False, verbose="ERROR")
    raw_final_ref.set_montage("standard_1020", match_case=False, on_missing="warn")
    
    # ========================================================================
    # Step 11: Extract proxy ECG from heartbeat ICs (if available)
    # ========================================================================
    raw_proxy_ecg = extract_proxy_ecg_from_ica(
        raw_final_ref,
        ica,
        heartbeat_ic_inds
    )
    
    # ========================================================================
    # Step 12: Apply ICA to remove artifact components
    # ========================================================================
    raw_clean = ica.apply(raw_final_ref.copy(), verbose="ERROR")
    
    return raw_clean, ica, exclude, raw_proxy_ecg
