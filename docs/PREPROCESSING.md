# Preprocessing Documentation

## Overview

The preprocessing pipeline consists of two main components:

1. **`preprocess_eegmmidb.py`**: Core preprocessing utilities
2. **`EDIH_Preprocessing_v0_1.py`**: Batch processing script

## Usage

### Basic Usage

```python
from pathlib import Path
from preprocess_eegmmidb import preprocess_eegmmidb_edf_with_proxy_ecg

# Preprocess a single EDF file
edf_path = Path("data/S001/S001R01.edf")
raw_clean, ica, exclude, raw_proxy_ecg = preprocess_eegmmidb_edf_with_proxy_ecg(
    edf_path,
    ica_method="extended-infomax"
)

# Save outputs
raw_clean.save("S001R01-cleaned_raw.fif", overwrite=True)
if raw_proxy_ecg is not None:
    raw_proxy_ecg.save("S001R01_proxy_ecg_raw.fif", overwrite=True)
```

### Batch Processing

```python
# Edit paths in EDIH_Preprocessing_v0_1.py
data_root = Path("path/to/raw/data")
clean_root = Path("path/to/output")

# Configure subject range (line ~180)
subject_ids = range(1, 110)  # All subjects S001-S109

# Run
python EDIH_Preprocessing_v0_1.py
```

## Configuration Parameters

### Core Preprocessing Function

```python
preprocess_eegmmidb_edf_with_proxy_ecg(
    edf_path: Path,
    l_freq_final: float = 0.5,        # Low-pass for final output
    h_freq_final: float = 40.0,       # High-pass for final output
    notch_freqs: tuple = (50.0,),     # Powerline frequency
    l_freq_ica: float = 1.0,          # High-pass for ICA (more aggressive)
    n_components: float = 0.99,       # ICA components (variance retained)
    random_state: int = 97,           # Reproducibility seed
    ica_method: str = "extended-infomax",  # ICA algorithm
    bad_z_thresh: float = 6.0,        # Z-score threshold for bad channels
    bad_flat_uv: float = 0.2,         # Flatness threshold (µV)
)
```

### Parameter Guidelines

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `l_freq_final` | 0.5 Hz | Removes slow drifts while preserving ERD |
| `h_freq_final` | 40 Hz | Avoids muscle artifacts, sufficient for mu/beta |
| `notch_freqs` | (50.0,) for EU, (60.0,) for US | Powerline noise |
| `l_freq_ica` | 1.0 Hz | Higher than final to improve ICA decomposition |
| `ica_method` | "extended-infomax" | Better for super-/sub-Gaussian sources |
| `bad_z_thresh` | 6.0 | Conservative; lower = more channels removed |

## Output Structure

### Per Subject (e.g., S001)

```
cleaned-dataset/
└── S001/
    ├── S001_bad_channels.txt           # Bad channel log
    ├── S001R01-cleaned_raw.fif         # Baseline eyes open
    ├── S001R02-cleaned_raw.fif         # Baseline eyes closed
    ├── S001R03-cleaned_raw.fif         # Task 1 (left/right fist)
    ├── S001R03-epo.fif                 # Task 1 epochs
    ├── S001R03_proxy_ecg_raw.fif       # Reconstructed ECG (if available)
    ├── ...
    ├── S001R05-cleaned_raw.fif         # Task 3 run 1 (fists/feet)
    ├── S001R05-epo.fif
    └── ...
```

### File Types

1. **`SXXXRYY-cleaned_raw.fif`**: 
   - Cleaned continuous EEG
   - Filtered (0.5-40 Hz), ICA-cleaned
   - Average referenced
   - All runs (R01-R14)

2. **`SXXXRYY-epo.fif`**:
   - Epoched data (-0.5 to 4.0 s)
   - Baseline corrected (-0.5 to 0 s)
   - Only task runs (R03-R14)
   - AutoReject applied if available

3. **`SXXXRYY_proxy_ecg_raw.fif`**:
   - Reconstructed ECG from ICA heartbeat components
   - 1 channel, same sampling rate as EEG
   - Only available if ICLabel detects heartbeat ICs

4. **`SXXX_bad_channels.txt`**:
   - Log of bad channels detected per run
   - Format: `SXXXRYY --> Bad channels: ch1, ch2, ...`

## Preprocessing Steps

### Step-by-Step Pipeline

```
1. Load EDF
   ↓
2. Keep EEG channels only
   ↓
3. Standardize channel names (remove 'EEG ', '-REF', '.')
   ↓
4. Set montage (standard_1020)
   ↓
5. Notch filter (50 Hz)
   ↓
6. Create ICA copy with higher high-pass (1 Hz)
   ↓
7. Average reference
   ↓
8. Detect bad channels (Z-score + flatness)
   ↓
9. Interpolate bad channels
   ↓
10. Fit ICA (Extended Infomax)
   ↓
11. Label components (ICLabel if available)
   ↓
12. Identify heartbeat ICs for proxy ECG
   ↓
13. Exclude artifact ICs (eye, muscle, heart, line noise)
   ↓
14. Apply final filter (0.5-40 Hz) to original data
   ↓
15. Apply ICA cleaning
   ↓
16. Create epochs (for task runs only)
   ↓
17. Apply AutoReject (optional)
   ↓
18. Save outputs
```

## Bad Channel Detection

### Algorithm

```python
def detect_bad_channels_simple(raw, z_thresh=6.0, flat_uv=0.2):
    """
    1. Flat channels: std < 0.2 µV
    2. Noisy channels: robust z-score > 6.0
    
    Uses median absolute deviation (MAD) for robustness.
    """
```

### Interpretation

- **Flat channels**: Often disconnected or bridged electrodes
- **Noisy channels**: High impedance, movement artifacts, or external interference
- **Interpolation**: Spherical spline interpolation using neighboring channels

## ICA Configuration

### Supported Methods

1. **Infomax**: 
   ```python
   ica_method="infomax"
   ```
   - Classic Bell-Sejnowski algorithm
   - Assumes super-Gaussian sources

2. **Extended Infomax** (Recommended):
   ```python
   ica_method="extended-infomax"
   ```
   - Handles both super- and sub-Gaussian sources
   - Better for mixed artifact types

### Component Exclusion

With **ICLabel** (recommended):
- Eye blink
- Muscle artifact
- Heart beat
- Line noise
- Channel noise

Without ICLabel (fallback):
- Heuristic EOG detection using frontal channels (Fp1, Fp2, AF7, AF8, F7, F8)
- No automatic ECG/muscle removal

### Proxy ECG Extraction

**When available**:
- ICLabel installed
- EEG dig points present
- At least one IC labeled as "heart beat"

**Method**:
1. Extract ICA sources for all heartbeat ICs
2. Compute robust median across heartbeat sources
3. Standardize (z-score)
4. Create 1-channel Raw with 'ECG' type

**Use cases**:
- Heart rate variability analysis
- Cardiac artifact validation
- Physiological state monitoring

## Epoching Parameters

### Default Settings

```python
epochs = mne.Epochs(
    raw_clean,
    events,
    event_id=event_id,
    tmin=-0.5,                      # 500 ms pre-stimulus baseline
    tmax=4.0,                       # 4 s post-stimulus
    baseline=(-0.5, 0.0),           # Baseline correction window
    preload=True,
    reject_by_annotation=True,      # Respect bad segments
    verbose="ERROR"
)
```

### Event IDs (Task 3 Example)

- **T0**: Baseline (rest)
- **T1**: Real movement - both fists
- **T2**: Real movement - both feet

## AutoReject

### When Applied

- `autoreject` package installed
- EEG digitization points available
- Epochs successfully created

### Configuration

```python
ar = AutoReject(
    n_interpolate=[1, 2, 4, 8],  # Interpolation options
    random_state=97,
    n_jobs=1
)
epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)
```

### Outputs

- **Cleaned epochs**: Bad epochs removed, bad channels interpolated
- **Reject log**: Per-epoch decisions (good/bad/interpolated)

## Troubleshooting

### Common Issues

1. **"No EEG dig points for AutoReject"**
   - Expected behavior: Epochs saved without AutoReject
   - Solution: Not critical; manual rejection possible later

2. **"Missing channels" warning**
   - Some subjects may lack certain 10-20 positions
   - Pipeline continues with available channels
   - Bad channels are logged

3. **ICLabel not working**
   - Requires EEG dig points (set by `set_montage`)
   - Falls back to heuristic EOG detection
   - Proxy ECG will not be available

4. **Long processing time**
   - ICA fitting is computationally expensive (~2-5 min/run)
   - Consider parallel processing for batch jobs
   - GPU acceleration not supported by MNE

### Recommendations

- **Start small**: Test on 1-2 subjects first
- **Monitor logs**: Check bad channel logs for patterns
- **Validate outputs**: Use QC scripts before full batch processing
- **Disk space**: ~500 MB per subject for all outputs

## Advanced Configuration

### Custom Filtering

```python
# For gamma band analysis (use with caution)
raw_clean, ica, exclude, _ = preprocess_eegmmidb_edf_with_proxy_ecg(
    edf_path,
    l_freq_final=0.5,
    h_freq_final=80.0,  # Extended to gamma
    notch_freqs=(50.0, 100.0),  # Include harmonics if <Nyquist
)
```

### Different ICA Seeds

```python
# For reproducibility testing
for seed in [42, 97, 123]:
    raw_clean, ica, exclude, _ = preprocess_eegmmidb_edf_with_proxy_ecg(
        edf_path,
        random_state=seed
    )
```

## Quality Control

After preprocessing, use:

```bash
python Compare_S049_vs_S016_vs_S058_QCPlots.py
```

This generates:
- Raw signal overlays
- Montage plots
- PSD comparisons
- Evoked responses (T1 vs T2)

## References

- **MNE-Python**: [mne.tools](https://mne.tools/)
- **ICLabel**: [mne-icalabel](https://mne.tools/mne-icalabel/)
- **AutoReject**: [autoreject](http://autoreject.github.io/)
- **EEGMMIDB**: [PhysioNet](https://physionet.org/content/eegmmidb/)

---

**Last updated**: February 2026
