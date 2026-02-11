# EEG Motor Imagery Analysis Pipeline

A comprehensive Python pipeline for preprocessing, analyzing, and validating Event-Related Desynchronization (ERD) patterns in the PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB).

## Overview

This repository contains production-ready tools for analyzing motor imagery EEG data, focusing on identifying subject-specific optimal frequency bands for brain-computer interface (BCI) applications. The pipeline performs automated preprocessing with ICA-based artifact removal and multi-band ERD discriminability analysis across 109 subjects.

## Key Features

- **Automated Preprocessing**: Batch processing with ICA-based artifact removal and ICLabel classification
- **Multi-Band ERD Analysis**: Subject-specific frequency optimization (theta, mu, beta_low, beta_high)
- **ERD Validation**: Statistical validation against multiple baseline conditions (pre-cue, eyes-open, eyes-closed)
- **Responder Screening**: Automated dual-responder classification based on ERD thresholds
- **Channel Set Comparison**: Minimal vs. extended sensorimotor montage analysis
- **Publication-Quality Plots**: TFR contrast maps, discriminability charts, PSD overlays, and validation plots
- **Proxy ECG Extraction**: Heartbeat component reconstruction from ICA sources
- **Comprehensive Documentation**: Detailed docstrings and inline comments in all scripts

## Dataset

This pipeline is designed for the **PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB)**:
- **Subjects**: 109 (S001-S109)
- **Channels**: 64-channel EEG (10-20 system)
- **Runs**: 14 per subject (R01-R14)
- **Sampling rate**: 160 Hz
- **Tasks**: Baseline (eyes open/closed), motor execution, and motor imagery

**Task 3 Focus** (Real motor execution):
- **T1**: Both fists clenching
- **T2**: Both feet movement

[Download dataset from PhysioNet](https://physionet.org/content/eegmmidb/1.0.0/)

## Installation

### Requirements

```bash
# Core dependencies (required)
mne >= 1.5.0
numpy >= 1.24.0
pandas >= 2.0.0
matplotlib >= 3.7.0
scipy >= 1.10.0
seaborn >= 0.12.0

# Recommended for full functionality
mne-icalabel >= 0.4.0  # ICLabel component classification
autoreject >= 0.4.0    # Automated epoch rejection
openpyxl >= 3.1.0      # Excel file I/O
```

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/yourusername/eeg-motor-imagery-analysis.git
cd eeg-motor-imagery-analysis

# Create conda environment (recommended)
conda create -n eeg_analysis python=3.10
conda activate eeg_analysis

# Install dependencies
pip install mne numpy pandas matplotlib scipy seaborn openpyxl
pip install mne-icalabel autoreject
```

## Project Structure

```
eeg-motor-imagery-analysis/
│
├── README.md                                    # Project overview (this file)
├── LICENSE                                      # MIT License
├── CHANGELOG.md                                 # Version history
├── CONTRIBUTING.md                              # Contribution guidelines
├── .gitignore                                   # Git ignore patterns
│
├── preprocess_eegmmidb.py                      # Core preprocessing utilities
├── EDIH_Preprocessing.py                        # Batch preprocessing pipeline
│
├── Task3_responder_screen.py                   # Dual-responder screening
├── Phase1_Step1A_Channel_Comparison.py         # Minimal vs Extended channels
├── Phase1_Step1B_MultiBand_Analysis.py         # Multi-band discriminability
├── Phase1_Step1B_Representative_Plots.py       # Visualization (TFR, PSD, etc.)
├── Phase1_ERD_Validation.py                    # ERD validation vs baselines
│
└── docs/
    ├── PREPROCESSING.md                         # Preprocessing documentation
    ├── ANALYSIS.md                              # Analysis pipeline guide
    ├── QUICKSTART.md                            # 5-minute quick start
    └── EXAMPLE_WORKFLOW.md                      # Complete workflow example
```

## Quick Start

### 1. Download and Organize Dataset

Download the EEGMMIDB dataset from PhysioNet and organize as follows:

```
YOUR_PATH_HERE/
└── eeg-motor-movementimagery-dataset-1.0.0/
    └── files/
        ├── S001/
        │   ├── S001R01.edf
        │   ├── S001R02.edf
        │   └── ... (R01-R14)
        ├── S002/
        └── ... (S001-S109)
```

### 2. Preprocessing (One-Time Setup)

```python
# Edit path in EDIH_Preprocessing.py
data_root = Path("YOUR_PATH_HERE/eeg-motor-movementimagery-dataset-1.0.0/files")
clean_root = Path("YOUR_PATH_HERE/cleaned-dataset")

# Run batch preprocessing (~8 hours for all 109 subjects)
python EDIH_Preprocessing.py
```

**Output**: Creates `cleaned-dataset/` with ~55 GB total

### 3. Responder Screening

```python
# Edit clean_root path in Task3_responder_screen.py
python Task3_responder_screen.py
```

**Output**: `task3_responder_screen.csv` with dual-responder rankings

### 4. Channel Set Comparison (Step 1A)

```python
# Edit clean_root path in Phase1_Step1A_Channel_Comparison.py
python Phase1_Step1A_Channel_Comparison.py
```

**Output**: `Phase1_Step1A_Task3_ERD_discriminability_min_vs_ext.csv`

### 5. Multi-Band Analysis (Step 1B)

```python
# Edit paths in Phase1_Step1B_MultiBand_Analysis.py
python Phase1_Step1B_MultiBand_Analysis.py
```

**Output**: `Phase1_Step1B_Task3_band_discriminability_motor_strip.csv`

### 6. Generate Representative Plots

```python
# Edit paths in Phase1_Step1B_Representative_Plots.py
python Phase1_Step1B_Representative_Plots.py
```

**Output**: Publication-quality figures in `Phase1_Step1B_Representative_Plots_figs/`

### 7. ERD Validation (Baseline Comparison)

```python
# Edit CLEAN_ROOT path in Phase1_ERD_Validation.py
python Phase1_ERD_Validation.py
```

**Output**: 
- `Phase1_results/Phase1_responder_validation_summary.csv`
- `Phase1_results/baseline_comparison_plots/` (top 20 subjects)
- `Phase1_results/Phase1_population_summary.png`

---

## Analysis Scripts Detailed

### Task3_responder_screen.py

**Purpose**: Identify dual responders with strong ERD in both mu and beta bands

**Key Features**:
- Loads Task 3 epochs (runs R05, R09, R13)
- Computes ERD% for mu (8-13 Hz) and beta (13-30 Hz)
- Applies thresholds: Mean ERD ≤ -10%, Negative ERD ≥ 60%
- Scores each subject on best channel performance

**Thresholds**:
```python
TH_MEAN = -10.0       # Mean ERD threshold (%)
TH_NEGPCT = 60.0      # Negative ERD consistency (%)
```

**Usage**:
```python
import pandas as pd
df = pd.read_csv("task3_responder_screen.csv")
top10 = df.sort_values("dual_total_score", ascending=False).head(10)
```

---

### Phase1_Step1A_Channel_Comparison.py

**Purpose**: Compare minimal (C3, Cz, C4) vs. extended (21 channels) sensorimotor montages

**Key Features**:
- **Minimal**: 3 channels (C3, Cz, C4)
- **Extended**: Up to 21 channels (FC/C/CP grid)
- Computes discriminability for mu (8-13 Hz) and beta (13-30 Hz)
- Calculates improvement: `delta_combo = ext_combo_score - min_combo_score`

**Discriminability Metrics**:
- `maxdiff`: max(|mean_ERD_T1 - mean_ERD_T2|) across channels
- `meandiff`: mean(|diff|) across channels
- `combo_score`: Average of mu_maxdiff + beta_maxdiff

**Expected Results**:
- Extended improves discriminability in ~60-70% of subjects
- Mean improvement: 5-8% increase in combo score

**Analysis Windows**:
```python
baseline_win = (-0.5, 0.0)   # Pre-stimulus baseline
task_win = (0.5, 1.5)        # Active movement
```

**Interpretation**:
```python
# Subjects where extended improved
improved = df[df["delta_combo"] > 0]
print(f"Extended improved: {len(improved)}/{len(df)} ({100*len(improved)/len(df):.1f}%)")

# Top improvers
top = df.sort_values("delta_combo", ascending=False).head(15)
```

---

### Phase1_Step1B_MultiBand_Analysis.py

**Purpose**: Identify subject-specific optimal frequency bands

**Key Features**:
- Tests 6 frequency bands against mu_beta baseline
- Uses motor-strip ROI (C5, C3, C1, Cz, C2, C4, C6)
- Selects top N subjects from Step 1A (default: 40)
- Ranks bands by discriminability per subject

**Frequency Bands**:
```python
"theta": (4.0, 8.0)         # Low-frequency oscillations
"mu": (8.0, 13.0)           # Sensorimotor rhythm
"beta_low": (13.0, 20.0)    # Motor planning
"beta_high": (20.0, 30.0)   # Motor execution
"beta": (13.0, 30.0)        # Full beta range
"mu_beta": (8.0, 30.0)      # Baseline reference
```

**Output Columns**:
- `best_band`: Optimal band for this subject
- `best_maxdiff`: Discriminability of best band
- `best_minus_baseline`: Improvement over mu_beta
- Per-band metrics: `theta_maxdiff`, `mu_maxdiff`, etc.

**Expected Findings**:
- ~30% subjects: mu dominance
- ~25% subjects: beta_low dominance  
- ~20% subjects: beta_high dominance
- ~15% subjects: theta dominance
- Mean improvement: 3-8% over baseline

**Interpretation**:
```python
# Band distribution
win_counts = df["best_band"].value_counts()

# Narrow beta benefit
narrow_beta = df[df["best_band"].isin(["beta_low", "beta_high"])]
print(f"Narrow beta optimal: {len(narrow_beta)}/{len(df)} subjects")

# Improvement statistics
print(f"Mean improvement: {df['best_minus_baseline'].mean():.3f}%")
print(f"Median: {df['best_minus_baseline'].median():.3f}%")
```

---

### Phase1_Step1B_Representative_Plots.py

**Purpose**: Generate publication-quality figures for representative subjects

**Key Features**:
- Auto-selects one subject per dominance category (theta, mu, beta_low, beta_high)
- Generates 3 plots per subject + optional montage
- 200 DPI output (configurable to 300 for publication)
- Organized output structure by category

**Selection Criteria**:
1. Filter: `best_band == category`
2. Sort: By `best_minus_baseline` (descending)
3. Select: Top subject

**Generated Plots**:

**Plot 1: Time-Frequency ERD Contrast Map**
- Shows (T1 - T2) power difference
- 4-30 Hz, baseline-corrected (%)
- Averaged across motor-strip
- **Reveals**: When/where discriminability is strongest

**Plot 2: Band Discriminability Bar Chart**
- Mean |ERD_T1 - ERD_T2| per band
- Across motor-strip channels
- **Shows**: Which band gives best separation

**Plot 3: Power Spectral Density Overlay**
- T1 (blue) vs T2 (red)
- Welch method, shaded band regions
- **Identifies**: Frequency-specific modulation

**Output Structure**:
```
Phase1_Step1B_Representative_Plots_figs/
├── S023_theta/
│   ├── S023_Plot1_TFR_Contrast_T1minusT2.png
│   ├── S023_Plot2_BandDiscriminability.png
│   ├── S023_Plot3_PSD_T1vsT2.png
│   └── S023_Montage_motor_strip.png
├── S045_mu/
├── S067_beta_low/
└── S089_beta_high/
```

**Configuration**:
```python
# TFR settings
tfr_freqs = np.arange(4.0, 31.0, 1.0)
tfr_n_cycles = tfr_freqs / 2.0

# Figure quality
dpi = 200  # Change to 300 for publication
```

---

### Phase1_ERD_Validation.py

**Purpose**: Validate that Task 3 ERD is task-specific by comparing against multiple baseline conditions

**Research Question**:
Is the observed ERD during motor execution significantly different from spontaneous activity during resting baseline conditions (not just general arousal)?

**Key Features**:
- Compares task execution against **three independent baselines**:
  1. **Pre-cue rest** (-0.5 to 0s): Within-task baseline
  2. **Eyes-open resting** (R01): Alert, relaxed state
  3. **Eyes-closed resting** (R02): Relaxed, high alpha state
- Tests statistical significance using Mann-Whitney U test
- Classifies subjects into responder categories (strong/moderate/weak/non-responder)
- Analyzes ERD across multiple time windows (early/peak/late)
- Generates validation plots for top 20 responders

**Validation Strategy**:
If ERD is truly task-specific (not just arousal or attention):
- Should show strong ERD vs pre-cue baseline
- Should show significantly lower power than both resting states
- Should be statistically significant (p < 0.05) in multiple channels

**Time Windows**:
```python
PRECUE_WINDOW = (-0.5, 0.0)          # Within-task baseline
TASK_WINDOWS = {
    'early': (0.5, 1.5),             # Movement onset
    'peak': (1.0, 2.0),              # Sustained contraction (EXPECTED PEAK)
    'late': (1.5, 2.5)               # Movement offset
}
```

**Classification Criteria**:

**Strong Responder**:
- ERD < -20% (strong desynchronization)
- Significant vs BOTH resting baselines (p < 0.05)
- In ≥3 channels (spatial consistency)
- **BCI Potential**: Excellent

**Moderate Responder**:
- ERD < -20%
- Significant vs BOTH resting baselines
- In 1-2 channels (limited spatial extent)
- **BCI Potential**: Good with channel selection

**Weak Responder**:
- ERD < -20% (adequate desynchronization)
- NOT statistically significant
- **BCI Potential**: Marginal, may improve with training

**Non-Responder**:
- ERD ≥ -20% (weak or no desynchronization)
- **BCI Potential**: Poor, try different tasks

**Outputs**:

**1. CSV Summary**: `Phase1_responder_validation_summary.csv`
```
Columns:
- subject, n_fists, n_feet, n_eo_epochs, n_ec_epochs
- responder_category (strong/moderate/weak/non-responder)
- n_significant_channels (how many channels meet criteria)
- best_channel (strongest ERD channel)
- best_ch_erd_fists, best_ch_erd_feet (ERD% for best channel)
- best_ch_contrast_eo_fists, best_ch_contrast_ec_fists (vs baselines)
- best_ch_pval_fists_eo, best_ch_pval_fists_ec (statistical tests)
- mean_erd_fists_early, mean_erd_fists_peak, mean_erd_fists_late
- mean_erd_feet_early, mean_erd_feet_peak, mean_erd_feet_late
```

**2. Per-Subject Plots**: `Phase1_results/baseline_comparison_plots/`
- Generated for top 20 responders (ranked by peak ERD)
- 4-panel validation layout per subject:

**Panel A**: ERD vs Pre-Cue Baseline
- Bar chart showing ERD% per channel
- Separate bars for fists (blue) and feet (orange)
- Threshold line at -20%
- Shows within-task ERD pattern

**Panel B**: Contrast vs Eyes-Open Baseline
- Tests if task differs from alert resting state
- Negative values = task < resting (expected)

**Panel C**: Contrast vs Eyes-Closed Baseline
- Tests if task differs from relaxed resting state
- Eyes-closed often has higher mu power (paradoxical)

**Panel D**: ERD Temporal Evolution
- Line plot showing ERD across early/peak/late windows
- Validates temporal consistency
- Peak window should show strongest ERD

**3. Population Summary**: `Phase1_population_summary.png`
- 3-panel aggregate statistics:
  * Responder distribution (pie chart)
  * ERD strength distribution (violin plot by category)
  * Best channel frequency (bar chart)

**Expected Results** (typical for EEGMMIDB):
- Strong responders: ~20-30%
- Moderate responders: ~20-25%
- Weak responders: ~15-20%
- Non-responders: ~25-35%
- Total BCI-viable: ~60-75%
- Best channels: C3, Cz, C4 most common

**Statistical Method**:
```python
# Mann-Whitney U test (non-parametric)
# Tests if task power < baseline power
# Robust to outliers, no normality assumption
pval = mannwhitneyu(task_power, baseline_power, alternative='less')

# Significance criterion
significant = (pval < 0.05)  # Standard alpha level
```

**Why Three Baselines?**:
1. **Pre-cue**: Controls for task engagement and attention
2. **Eyes-open**: Controls for general arousal and visual input
3. **Eyes-closed**: Controls for alpha rhythm and relaxation
   - Eyes-closed typically has higher mu/alpha power
   - Task should still show lower power (validates motor-specific ERD)

**Interpretation Example**:
```python
import pandas as pd
df = pd.read_csv("Phase1_results/Phase1_responder_validation_summary.csv")

# Strong responders for BCI applications
strong = df[df['responder_category'] == 'strong']
print(f"Strong responders: {len(strong)} subjects")
print(f"Mean peak ERD: {strong['mean_erd_fists_peak'].mean():.1f}%")

# Top 5 subjects by ERD strength
top5 = df.nsmallest(5, 'mean_erd_fists_peak')
print(top5[['subject', 'responder_category', 'mean_erd_fists_peak', 'best_channel']])

# Channel distribution
print(df['best_channel'].value_counts())
```

**Usage**:
```bash
# Edit CLEAN_ROOT path in script
python Phase1_ERD_Validation.py

# Outputs created:
# - Phase1_results/Phase1_responder_validation_summary.csv
# - Phase1_results/baseline_comparison_plots/SXXX_validation.png (top 20)
# - Phase1_results/Phase1_population_summary.png
```

**Runtime**: ~3-5 hours for 109 subjects (~2-3 min per subject + 30s per plot × 20)

---

## Preprocessing Pipeline

The `EDIH_Preprocessing.py` script performs 16 steps:

1. Load EDF (keep EEG only)
2. Standardize channel names
3. Set standard_1020 montage
4. Notch filter (50 Hz)
5. Detect bad channels (Z-score + flatness)
6. Interpolate bad channels
7. Bandpass for ICA (1-40 Hz)
8. Average reference
9. Fit ICA (Extended Infomax)
10. ICLabel component classification
11. Exclude artifacts (eye, muscle, heart, line noise)
12. Extract proxy ECG from cardiac components
13. Final bandpass (0.5-40 Hz)
14. Apply ICA cleaning
15. Create epochs (-0.5 to 4.0 s)
16. AutoReject epoch rejection

**Output per subject** (~500 MB):
- `*-cleaned_raw.fif`: Cleaned continuous EEG
- `*-epo.fif`: Epoched data
- `*_proxy_ecg_raw.fif`: Heartbeat signal
- `*_bad_channels.txt`: Bad channel log

---

## ERD Computation Method

All scripts use **Hilbert transform** for ERD:

```python
def erd_percent(epochs, band):
    # 1. Bandpass filter
    ep = epochs.filter(band[0], band[1])
    
    # 2. Hilbert transform
    ep.apply_hilbert(envelope=False)
    
    # 3. Instantaneous power
    power = np.abs(ep.get_data()) ** 2
    
    # 4. Average in windows
    p_base = power[..., baseline_mask].mean(axis=-1)
    p_task = power[..., task_mask].mean(axis=-1)
    
    # 5. ERD% = (P_task - P_base) / P_base × 100
    erd = (p_task - p_base) / (p_base + 1e-12) * 100.0
    
    return erd
```

**Interpretation**:
- ERD < -30%: Strong desynchronization
- -30% to -10%: Moderate desynchronization
- -10% to 0%: Weak desynchronization  
- ERD > 0%: Synchronization (unexpected)

---

## Troubleshooting

**"No usable epochs"**
- Check R05/R09/R13 files exist
- Verify T1/T2 events in epochs

**"Missing channels"**
- Set `REQUIRE_ALL_MOTOR_STRIP = False`
- Pipeline adapts to available channels

**"NaN in results"**
- Insufficient epochs or bad quality
- Review preprocessing logs

**"Low discriminability"**
- Subject may be non-responder (~25%)
- Verify with QC plots

**"All subjects classified as non-responder" (ERD Validation)**
- Check ERD_THRESHOLD setting (default: -20%)
- Verify time windows are correct
- Review baseline run quality (R01, R02)
- May need to adjust thresholds for specific dataset

**"Missing baseline runs R01/R02"**
- Ensure preprocessing included baseline runs
- Check `cleaned-dataset/SXXX/SXXXR01-cleaned_raw.fif` exists
- R01 = eyes open, R02 = eyes closed (both required)

---

## Citation

```bibtex
@misc{eeg_motor_imagery_pipeline,
  author = {Ram P Narayanan},
  title = {EEG Motor Imagery Analysis Pipeline},
  year = {2025-2026},
  publisher = {GitHub},
  url = {https://github.com/yourusername/eeg-motor-imagery-analysis}
}

@article{schalk2004bci2000,
  title={BCI2000: a general-purpose brain-computer interface (BCI) system},
  author={Schalk, Gerwin and others},
  journal={IEEE Trans. Biomed. Eng.},
  volume={51},
  number={6},
  pages={1034--1043},
  year={2004}
}
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - See [LICENSE](LICENSE) for details.

## Acknowledgments

- PhysioNet for EEGMMIDB dataset
- MNE-Python development team
- ICLabel and AutoReject developers

---

## Documentation

- [PREPROCESSING.md](docs/PREPROCESSING.md) - Preprocessing guide
- [ANALYSIS.md](docs/ANALYSIS.md) - Analysis documentation
- [QUICKSTART.md](docs/QUICKSTART.md) - Quick start
- [EXAMPLE_WORKFLOW.md](docs/EXAMPLE_WORKFLOW.md) - Complete workflow

---

**Last Updated**: February 2026 | **Version**: 1.0.1
